# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-operation L1 breakdown derived from a ttnn graph capture.

WHAT GRAPH CAPTURE IS
---------------------
``ttnn.graph.begin_graph_capture()`` pushes a ``GraphProcessor`` (an ``IGraphProcessor``) onto the
process-wide ``GraphTracker``.  From then on, ttnn/metal call the tracker at every interesting
moment, and the processor appends a *node* to a flat list.  ``end_graph_capture()`` pops the
processor and hands that list back as plain JSON (list of dicts).

The hook sites that matter for L1:

  ``Buffer::allocate_impl``            -> ``track_allocate``            -> ``buffer_allocate`` node
  ``Buffer::deallocate``               -> ``track_deallocate``          -> ``buffer_deallocate`` node
  ``mesh_device_operation_utils.hpp``  -> ``track_program``             -> ``circular_buffer_deallocate_all``
    (called per program in the workload, right before it is enqueued)      + one ``circular_buffer_allocate``
                                                                           per CB in the program
  Metal 2.0 program-scope L1                                            -> ``dataflow_buffer_allocate``
    (``track_program_l1`` in graph_tracking.cpp)                           + ``scratchpad_allocate``
  ttnn op entry/exit (the C++ op decorator)                             -> ``function_start`` / ``function_end``

Two run modes, both of which produce the same node stream:

  ``RunMode.NORMAL``       ops really execute; buffer addresses are real.
  ``RunMode.NO_DISPATCH``  ``ProcessorHooks`` returns true from ``hook_allocate`` / ``hook_program``,
                           so ``Buffer::allocate_impl`` sets ``address_ = 0`` without touching the
                           allocator and programs are never enqueued.  Sizes are still exact,
                           because they come from the TensorSpec / CircularBufferConfig, not from
                           the allocator.  This is what lets you price an op that would not fit.

THE NODE SCHEMA
---------------
Every node is ``{counter, node_type, params, arguments, connections, input_tensors,
stacking_level}``.  ``counter`` is the index in the list and is strictly increasing (the C++
extractor asserts this), so the list *is* the execution order.  ``connections`` make it a graph;
for memory accounting you only need the linear order.  ``stacking_level`` is the op nesting depth,
which is how a composite op (conv2d) is separated from its children (Halo, matmul, ...).

``params`` per node type (only the fields this test uses):

  ``circular_buffer_allocate``  size, address, core_range_set, globally_allocated, device_id
  ``dataflow_buffer_allocate``  size, address, core_range_set, borrows_memory, device_id
  ``scratchpad_allocate``       size, address, core_range_set, device_id
  ``buffer_allocate`` / ``buffer_deallocate``
                                size, max_size_per_bank, address, type (DRAM|L1), buffer_type,
                                layout (INTERLEAVED|HEIGHT_SHARDED|WIDTH_SHARDED|BLOCK_SHARDED),
                                page_size, num_cores (0 for interleaved), device_id
  ``function_start``            name, inputs        ``function_end``  name (+ duration_ns)
  ``tensor``                    tensor_id, shape, dtype, layout, memory_config, address, ...

Note which number is which.  ``size`` on a buffer node is the WHOLE buffer across all banks;
``max_size_per_bank`` is the per-core footprint, and that is the one the peak-L1 accounting uses.
``size`` on a CB / dataflow-buffer / scratchpad node is already per core.

WHAT "COUNTED" MEANS
--------------------
``ttnn.graph.extract_resource_usage_per_core`` walks the same list and keeps four running totals
(CB, L1 buffers, dataflow buffers, scratchpads) plus their sum, reporting the high-water mark of
each.  Two classes of allocation are deliberately NOT added:

  * a CB with ``globally_allocated == 1`` -- a tensor-backed CB, i.e. the CB *is* the tensor's
    L1 shard (``CircularBufferConfig::set_globally_allocated_address``).  Its bytes are already
    counted as the tensor under ``peak_l1``; counting the CB too would double-count.
  * a dataflow buffer with ``borrows_memory == 1`` -- same argument, Metal 2.0 flavour.

``circular_buffer_deallocate_all`` drops the CB + dataflow-buffer + scratchpad totals back to zero:
program-scope L1 lives exactly as long as the program.  ``buffer_deallocate`` subtracts one tensor.

TWO THINGS THAT SURPRISE PEOPLE
-------------------------------
1. The capture is a WINDOW.  A tensor allocated before ``begin_graph_capture`` has no
   ``buffer_allocate`` node, so it contributes nothing to ``peak_l1`` -- even though it is
   physically occupying L1 the whole time.  See ``test_capture_window_bounds_peak_l1``.
2. ``extract_resource_usage_per_core`` sums CB sizes IGNORING ``core_range_set``.  Two CBs on
   disjoint core sets are added as if they shared a core.  That is a deliberate worst case; when
   you need the truth, aggregate per core yourself -- ``Breakdown.peak_per_core`` below does.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

import pytest
import torch
import ttnn

# ----------------------------------------------------------------------------------------------
# Node / param names, mirroring ttnn/api/ttnn/graph/graph_consts.hpp
# ----------------------------------------------------------------------------------------------
FN_START = "function_start"
FN_END = "function_end"
CB_ALLOC = "circular_buffer_allocate"
DFB_ALLOC = "dataflow_buffer_allocate"
SCRATCHPAD_ALLOC = "scratchpad_allocate"
PROGRAM_L1_FREE = "circular_buffer_deallocate_all"
BUF_ALLOC = "buffer_allocate"
BUF_FREE = "buffer_deallocate"

# node_type -> the running total it moves
PROGRAM_SCOPE_KINDS = {CB_ALLOC: "cb", DFB_ALLOC: "dfb", SCRATCHPAD_ALLOC: "scratchpad"}

# CoreRangeSet::str() renders as "{[x-y - x-y], [x-y - x-y]}"
_CORE_RANGE_RE = re.compile(r"\[(\d+)-(\d+) - (\d+)-(\d+)\]")


def cores_in(core_range_set: str):
    """Expand a CoreRangeSet string into logical (x, y) coords."""
    out = []
    for x0, y0, x1, y1 in _CORE_RANGE_RE.findall(core_range_set or ""):
        for x in range(int(x0), int(x1) + 1):
            for y in range(int(y0), int(y1) + 1):
                out.append((x, y))
    return out


@dataclass
class Event:
    """One L1-moving node, with the enclosing op resolved."""

    counter: int
    op: str  # innermost enclosing function_start
    depth: int  # op nesting depth
    kind: str  # cb | dfb | scratchpad | l1_buffer | dram_buffer
    event: str  # alloc | free
    per_core_bytes: int  # signed contribution to the per-core running total
    counted: bool  # does extract_resource_usage_per_core add this?
    note: str  # why not, when counted is False
    cores: str  # core_range_set, when the node carries one
    params: dict


@dataclass
class OpStat:
    name: str
    counter: int
    depth: int
    cb: int = 0
    dfb: int = 0
    scratchpad: int = 0
    l1_alloc: int = 0
    l1_free: int = 0
    peak_total: int = 0  # worst running total seen while this op was on the stack

    @property
    def program_l1(self):
        return self.cb + self.dfb + self.scratchpad


@dataclass
class Breakdown:
    events: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    peak_cb: int = 0
    peak_l1: int = 0
    peak_dfb: int = 0
    peak_scratchpad: int = 0
    peak_total: int = 0
    # (x, y) -> peak program-scope L1 on that specific core.  Only program-scope allocations
    # carry a core_range_set, so tensor buffers are not in here.
    peak_per_core: dict = field(default_factory=dict)

    def ops_named(self, needle):
        return [o for o in self.ops if needle in o.name]

    def globally_allocated_cbs(self):
        return [e for e in self.events if e.kind == "cb" and not e.counted]


def analyze(trace):
    """Re-implement extract_resource_usage_per_core in Python, keeping the per-event detail."""
    b = Breakdown()
    stack = []  # indices into b.ops
    cur = {"cb": 0, "dfb": 0, "scratchpad": 0, "l1": 0}
    total = 0
    live_per_core = defaultdict(int)
    peak_per_core = defaultdict(int)

    def move(kind, delta, cores=None):
        """Apply delta to one running total and to every peak that watches it."""
        nonlocal total
        cur[kind] += delta
        total += delta
        if kind == "cb":
            b.peak_cb = max(b.peak_cb, cur["cb"])
        elif kind == "dfb":
            b.peak_dfb = max(b.peak_dfb, cur["dfb"])
        elif kind == "scratchpad":
            b.peak_scratchpad = max(b.peak_scratchpad, cur["scratchpad"])
        elif kind == "l1":
            b.peak_l1 = max(b.peak_l1, cur["l1"])
        b.peak_total = max(b.peak_total, total)
        for i in stack:
            b.ops[i].peak_total = max(b.ops[i].peak_total, total)
        if cores is not None:
            for c in cores:
                live_per_core[c] += delta
                peak_per_core[c] = max(peak_per_core[c], live_per_core[c])

    expected_counter = 0
    for node in trace:
        assert node["counter"] == expected_counter, "trace must be in execution order"
        expected_counter += 1
        nt = node["node_type"]
        p = node.get("params", {})

        if nt == FN_START:
            b.ops.append(OpStat(name=p.get("name", "?"), counter=node["counter"], depth=len(stack)))
            stack.append(len(b.ops) - 1)
            continue
        if nt == FN_END:
            if stack:
                stack.pop()
            continue

        op = b.ops[stack[-1]].name if stack else "<capture>"
        depth = len(stack)
        top = b.ops[stack[-1]] if stack else None

        if nt in PROGRAM_SCOPE_KINDS:
            kind = PROGRAM_SCOPE_KINDS[nt]
            size = int(p["size"])
            cores = cores_in(p.get("core_range_set", ""))
            if kind == "cb" and int(p.get("globally_allocated", 0)):
                counted, note = False, "tensor-backed CB: these bytes are the tensor, already in peak_l1"
            elif kind == "dfb" and int(p.get("borrows_memory", 0)):
                counted, note = False, "borrowed view of a tensor buffer, already in peak_l1"
            else:
                counted, note = True, ""
            if counted:
                move(kind, size, cores)
                if top is not None:
                    setattr(top, kind, getattr(top, kind) + size)
            b.events.append(
                Event(
                    node["counter"],
                    op,
                    depth,
                    kind,
                    "alloc",
                    size if counted else 0,
                    counted,
                    note,
                    p.get("core_range_set", ""),
                    p,
                )
            )
            continue

        if nt == PROGRAM_L1_FREE:
            freed = cur["cb"] + cur["dfb"] + cur["scratchpad"]
            for kind in ("cb", "dfb", "scratchpad"):
                if cur[kind]:
                    move(kind, -cur[kind])
            live_per_core.clear()
            b.events.append(
                Event(
                    node["counter"],
                    op,
                    depth,
                    "program_l1",
                    "free",
                    -freed,
                    True,
                    "program-scope L1 released: CBs + dataflow buffers + scratchpads",
                    "",
                    p,
                )
            )
            continue

        if nt in (BUF_ALLOC, BUF_FREE):
            is_l1 = p.get("type") == "L1"
            per_bank = int(p["max_size_per_bank"])
            sign = 1 if nt == BUF_ALLOC else -1
            if is_l1:
                move("l1", sign * per_bank)
                if top is not None:
                    if sign > 0:
                        top.l1_alloc += per_bank
                    else:
                        top.l1_free += per_bank
                counted, note = True, ""
            else:
                counted, note = False, "DRAM buffer: not L1, ignored by the per-core L1 peaks"
            b.events.append(
                Event(
                    node["counter"],
                    op,
                    depth,
                    "l1_buffer" if is_l1 else "dram_buffer",
                    "alloc" if sign > 0 else "free",
                    sign * per_bank if is_l1 else 0,
                    counted,
                    note,
                    "",
                    p,
                )
            )
            continue

    b.peak_per_core = dict(peak_per_core)
    return b


def kb(n):
    return f"{n / 1024:8.2f} KB"


def format_breakdown(b, title=""):
    lines = []
    if title:
        lines += [f"=== {title} " + "=" * max(0, 74 - len(title))]
    lines += [
        "",
        "-- allocation events, in execution order --",
        f"{'#':>4}  {'op':38} {'kind':11} {'ev':5} {'per-core':>12}  {'counted':7} detail",
    ]
    for e in b.events:
        detail = ""
        if e.kind in ("l1_buffer", "dram_buffer"):
            detail = (
                f"{e.params.get('layout', '')} size={e.params.get('size')} "
                f"num_cores={e.params.get('num_cores')} page={e.params.get('page_size')} "
                f"addr={e.params.get('address')}"
            )
        elif e.kind in ("cb", "dfb", "scratchpad"):
            detail = f"cores={e.cores}"
        if e.note:
            detail += ("  <- " + e.note) if detail else ("<- " + e.note)
        lines.append(
            f"{e.counter:>4}  {'  ' * e.depth + e.op:38.38} {e.kind:11} {e.event:5} "
            f"{kb(e.per_core_bytes)}  {str(e.counted):7} {detail}"
        )

    lines += [
        "",
        "-- per operation (nesting preserved) --",
        f"{'op':44} {'CB':>12} {'DFB':>12} {'scratch':>12} {'L1 alloc':>12} {'L1 free':>12} {'peak while live':>16}",
    ]
    for o in b.ops:
        lines.append(
            f"{'  ' * o.depth + o.name:44.44} {kb(o.cb)} {kb(o.dfb)} {kb(o.scratchpad)} "
            f"{kb(o.l1_alloc)} {kb(o.l1_free)} {kb(o.peak_total)}"
        )

    lines += [
        "",
        "-- peaks per core (what extract_resource_usage_per_core returns) --",
        f"  peak_cb          {kb(b.peak_cb)}",
        f"  peak_l1          {kb(b.peak_l1)}",
        f"  peak_dfb         {kb(b.peak_dfb)}",
        f"  peak_scratchpad  {kb(b.peak_scratchpad)}",
        f"  peak_total       {kb(b.peak_total)}",
    ]
    if b.peak_per_core:
        by_val = defaultdict(list)
        for core, v in b.peak_per_core.items():
            by_val[v].append(core)
        lines += ["", "-- true per-core program-scope L1 (core_range_set aware) --"]
        for v in sorted(by_val, reverse=True):
            cores = sorted(by_val[v])
            lines.append(f"  {kb(v)} on {len(cores):>4} cores  e.g. {cores[:4]}")
    return "\n".join(lines)


def assert_matches_cpp(trace, b):
    """The Python walk above must agree with the C++ extractor, field for field."""
    u = ttnn.graph.extract_resource_usage_per_core(trace)
    assert b.peak_cb == u.peak_cb, f"peak_cb {b.peak_cb} != {u.peak_cb}"
    assert b.peak_l1 == u.peak_l1, f"peak_l1 {b.peak_l1} != {u.peak_l1}"
    assert b.peak_dfb == u.peak_dataflow_buffer
    assert b.peak_scratchpad == u.peak_scratchpad
    assert b.peak_total == u.peak_total
    return u


@pytest.fixture
def traced():
    """Capture a callable and return (trace, result, breakdown).

    Fast runtime mode is off so the op decorator emits the full function_start/function_end
    nesting; with it on you still get the device-op nodes but not the composite ttnn-level frames.
    """

    def _run(fn, mode=ttnn.graph.RunMode.NO_DISPATCH):
        with ttnn.manage_config("enable_fast_runtime_mode", False):
            ttnn.graph.begin_graph_capture(mode)
            try:
                result = fn()
            finally:
                trace = ttnn.graph.end_graph_capture()
        return trace, result, analyze(trace)

    return _run


def dram_tiled(device, shape):
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def height_sharded_cfg(device, height, width, grid=(8, 8)):
    """Height-shard [height, width] over grid, tile-aligned."""
    num_cores = grid[0] * grid[1]
    assert height % (32 * num_cores) == 0, "keep the shard tile-aligned"
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))]),
            [height // num_cores, width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# ==============================================================================================
# 1. the node schema we depend on
# ==============================================================================================
def test_node_schema(device, traced):
    """Pin the shape of the JSON: keys, ordering, and which params each L1 node carries."""
    a = dram_tiled(device, (1, 1, 512, 512))
    trace, _, _ = traced(lambda: ttnn.matmul(a, a, memory_config=ttnn.L1_MEMORY_CONFIG))

    assert trace[0]["node_type"] == "capture_start"
    assert trace[-1]["node_type"] == "capture_end"
    for i, node in enumerate(trace):
        assert node["counter"] == i
        assert {"counter", "node_type", "params", "connections", "stacking_level"} <= set(node)

    by_type = defaultdict(list)
    for node in trace:
        by_type[node["node_type"]].append(node)

    assert by_type[CB_ALLOC], "a matmul program must declare circular buffers"
    for node in by_type[CB_ALLOC]:
        assert {"size", "address", "core_range_set", "globally_allocated", "device_id"} <= set(node["params"])

    l1_bufs = [n for n in by_type[BUF_ALLOC] if n["params"]["type"] == "L1"]
    assert l1_bufs, "the L1 output buffer must show up as a buffer_allocate"
    for node in l1_bufs:
        p = node["params"]
        assert {"size", "max_size_per_bank", "address", "type", "layout", "page_size", "num_cores"} <= set(p)
        # size is the whole buffer, max_size_per_bank is the per-core slice
        assert p["size"] >= p["max_size_per_bank"]

    # track_program fires before each program is enqueued, so a program's CBs are always
    # preceded by the release of the previous program's.
    first_cb = min(n["counter"] for n in by_type[CB_ALLOC])
    assert any(n["counter"] < first_cb for n in by_type[PROGRAM_L1_FREE])


# ==============================================================================================
# 2. matmul: the plain case
# ==============================================================================================
@pytest.mark.parametrize("out_mem", ["dram", "l1"], ids=["dram_out", "l1_out"])
def test_matmul_breakdown(device, traced, out_mem):
    a = dram_tiled(device, (1, 1, 1024, 1024))
    b_t = dram_tiled(device, (1, 1, 1024, 1024))
    mcfg = ttnn.DRAM_MEMORY_CONFIG if out_mem == "dram" else ttnn.L1_MEMORY_CONFIG

    trace, out, br = traced(lambda: ttnn.matmul(a, b_t, memory_config=mcfg))
    print("\n" + format_breakdown(br, f"matmul 1024x1024 @ 1024x1024, {out_mem} output"))
    assert_matches_cpp(trace, br)

    # One device op, one program, so every CB is charged to it and nothing is released mid-op.
    mm = br.ops_named("Matmul")
    assert len(mm) == 1, [o.name for o in br.ops]
    assert mm[0].cb == br.peak_cb > 0
    assert br.peak_scratchpad == 0 and br.peak_dfb == 0, "classic (non-Metal-2.0) program"

    if out_mem == "dram":
        # The output lands in DRAM, so no L1 tensor bytes at all: peak_total is pure CB.
        assert br.peak_l1 == 0
        assert br.peak_total == br.peak_cb
    else:
        # The L1 output is charged at its per-bank size, and it is still live when the CBs peak.
        assert br.peak_l1 > 0
        assert br.peak_total == br.peak_cb + br.peak_l1
        allocs = [e for e in br.events if e.kind == "l1_buffer" and e.event == "alloc"]
        assert len(allocs) == 1
        assert allocs[0].per_core_bytes == br.peak_l1
        assert int(allocs[0].params["num_cores"]) == 0, "interleaved buffers report num_cores=0"


# ==============================================================================================
# 3. the capture is a window, not a snapshot of the device
# ==============================================================================================
def test_capture_window_bounds_peak_l1(device, traced):
    """A tensor allocated before begin_graph_capture contributes nothing to peak_l1."""
    shape = (1, 1, 2048, 512)
    cfg = height_sharded_cfg(device, 2048, 512)

    # inputs allocated OUTSIDE the window
    a = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    b = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    _, _, outside = traced(lambda: ttnn.add(a, b, memory_config=cfg))

    # same op, inputs allocated INSIDE the window
    def inside_fn():
        x = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )
        y = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )
        return ttnn.add(x, y, memory_config=cfg)

    _, _, inside = traced(inside_fn)

    print("\n" + format_breakdown(outside, "sharded add, inputs allocated OUTSIDE the capture"))
    print("\n" + format_breakdown(inside, "sharded add, inputs allocated INSIDE the capture"))

    shard_bytes = outside.peak_l1  # the output shard
    assert shard_bytes > 0
    # Outside: only the output is charged.  Inside: the two inputs are charged as well.
    assert inside.peak_l1 == 3 * shard_bytes, (inside.peak_l1, shard_bytes)
    assert inside.peak_total > outside.peak_total


# ==============================================================================================
# 4. already-sharded tensors: tensor-backed CBs are not double-counted
# ==============================================================================================
def test_sharded_inputs_use_tensor_backed_cbs(device, traced):
    """A sharded op builds CBs directly over the shards; those CBs are excluded from peak_cb."""
    shape = (1, 1, 2048, 512)
    cfg = height_sharded_cfg(device, 2048, 512)
    a = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    b = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )

    trace, _, sharded = traced(lambda: ttnn.add(a, b, memory_config=cfg))
    print("\n" + format_breakdown(sharded, "sharded add (tensor-backed CBs)"))
    assert_matches_cpp(trace, sharded)

    tensor_backed = sharded.globally_allocated_cbs()
    assert tensor_backed, "a sharded binary op should bind its CBs to the shards"
    for e in tensor_backed:
        assert e.params["globally_allocated"] in ("1", 1)
        assert e.per_core_bytes == 0, "excluded from every running total"

    # The excluded bytes are real L1 -- they are the shards.  Prove the arithmetic:
    # what peak_cb would have been if we (wrongly) counted them.
    naive = sharded.peak_cb + sum(int(e.params["size"]) for e in tensor_backed)
    assert naive > sharded.peak_cb
    shard_bytes = sharded.peak_l1
    assert any(int(e.params["size"]) == shard_bytes for e in tensor_backed), (
        [int(e.params["size"]) for e in tensor_backed],
        shard_bytes,
    )

    # A tensor-backed CB's core_range_set is the PROGRAM's core range, not the tensor's shard
    # grid -- the op declares the CB over every core it may run on and repoints it at the
    # shard.  So you cannot read the shard grid off the CB node.
    cb_cores = cores_in(tensor_backed[0].cores)
    shard_cores = int([e for e in sharded.events if e.kind == "l1_buffer"][0].params["num_cores"])
    assert len(cb_cores) >= shard_cores

    # Contrast: the same op on interleaved DRAM inputs has no tensor-backed CBs at all.
    a_d = dram_tiled(device, shape)
    trace_i, _, interleaved = traced(lambda: ttnn.add(a_d, a_d))
    print("\n" + format_breakdown(interleaved, "interleaved add (ordinary CBs)"))
    assert_matches_cpp(trace_i, interleaved)
    assert not interleaved.globally_allocated_cbs()


def test_per_core_program_l1_is_bounded_by_the_flat_peak(device, traced):
    """peak_cb adds CB sizes ignoring core_range_set; the per-core map respects it.

    The flat sum is exact when every CB shares one core range (the usual case) and is a
    worst-case over-count when they do not.  It can never under-report a core.
    """
    a = dram_tiled(device, (1, 1, 1024, 1024))
    _, _, mm = traced(lambda: ttnn.matmul(a, a, memory_config=ttnn.L1_MEMORY_CONFIG))

    assert mm.peak_per_core, "an interleaved matmul allocates ordinary, non-tensor-backed CBs"
    worst = max(mm.peak_per_core.values())
    assert worst <= mm.peak_cb, "the flat sum can never under-report a core"
    assert worst == mm.peak_cb, "matmul puts every CB on one core range, so the flat sum is exact"
    print(
        f"\nmatmul: flat peak_cb = {mm.peak_cb} B, worst single core = {worst} B " f"over {len(mm.peak_per_core)} cores"
    )

    # A fully tensor-backed sharded op is the opposite extreme: nothing is counted as CB at
    # all, so there is no program-scope L1 to attribute to any core.
    cfg = height_sharded_cfg(device, 2048, 512)
    s = ttnn.from_torch(
        torch.rand((1, 1, 2048, 512), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=cfg,
    )
    _, _, sharded = traced(lambda: ttnn.add(s, s, memory_config=cfg))
    assert sharded.peak_cb == 0 and sharded.peak_per_core == {}
    print(f"sharded add: peak_cb = {sharded.peak_cb} B (all CBs tensor-backed), " f"peak_l1 = {sharded.peak_l1} B")


# ==============================================================================================
# 5. in-place: writing into an existing buffer allocates nothing
# ==============================================================================================
def test_in_place_output_allocates_no_buffer(device, traced):
    shape = (1, 1, 2048, 512)
    cfg = height_sharded_cfg(device, 2048, 512)

    def mk():
        return ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )

    a, b = mk(), mk()
    trace_ip, out_ip, in_place = traced(lambda: ttnn.add(a, b, output_tensor=a, memory_config=cfg))
    print("\n" + format_breakdown(in_place, "sharded add, in place into input a"))
    assert_matches_cpp(trace_ip, in_place)

    a2, b2 = mk(), mk()
    trace_oop, _, out_of_place = traced(lambda: ttnn.add(a2, b2, memory_config=cfg))

    # In place: no buffer_allocate inside the window at all, so peak_l1 is zero even though the
    # op is reading and writing three shards' worth of live L1.
    assert not [e for e in in_place.events if e.kind == "l1_buffer" and e.event == "alloc"]
    assert in_place.peak_l1 == 0
    # Out of place: exactly one new L1 buffer, the output.
    assert len([e for e in out_of_place.events if e.kind == "l1_buffer" and e.event == "alloc"]) == 1
    assert out_of_place.peak_l1 > 0
    assert in_place.peak_total < out_of_place.peak_total

    # The in-place output aliases the input, and the op's tensor-backed CBs still describe
    # all three operands -- two of which are the same buffer.
    assert out_ip.buffer_address() == a.buffer_address()
    assert len(in_place.globally_allocated_cbs()) >= 2


def test_deallocate_lowers_the_running_total(device, traced):
    """peak is a high-water mark, not a sum: freeing an intermediate makes room."""
    shape = (1, 1, 2048, 512)
    cfg = height_sharded_cfg(device, 2048, 512)

    def keep_both():
        x = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )
        first = ttnn.add(x, x, memory_config=cfg)
        return ttnn.add(first, first, memory_config=cfg)

    def free_first():
        x = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=cfg,
        )
        first = ttnn.add(x, x, memory_config=cfg)
        ttnn.deallocate(x)
        return ttnn.add(first, first, memory_config=cfg)

    _, _, both = traced(keep_both)
    _, _, freed = traced(free_first)
    print("\n" + format_breakdown(freed, "two chained sharded adds, input deallocated between"))

    assert [e for e in freed.events if e.kind == "l1_buffer" and e.event == "free"]
    assert freed.peak_l1 < both.peak_l1


# ==============================================================================================
# 6. conv2d: a composite op, with HaloDeviceOperation ahead of the matmul
# ==============================================================================================
@pytest.mark.skipif(not hasattr(ttnn, "conv2d"), reason="conv2d not available in this build")
def test_conv2d_composite_breakdown(device, traced):
    from ttnn.operations.conv2d import Conv2dConfig

    batch, in_c, out_c, h, w = 1, 32, 64, 128, 128
    torch_input = torch.randn([1, 1, batch * h * w, in_c], dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_weight = ttnn.from_torch(torch.randn([out_c, in_c, 3, 3], dtype=torch.bfloat16), dtype=ttnn.bfloat16)
    ttnn_bias = ttnn.from_torch(torch.randn([1, 1, 1, out_c], dtype=torch.bfloat16), dtype=ttnn.bfloat16)

    conv_config = Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    def run_conv():
        return ttnn.conv2d(
            input_tensor=ttnn_input,
            weight_tensor=ttnn_weight,
            bias_tensor=ttnn_bias,
            in_channels=in_c,
            out_channels=out_c,
            batch_size=batch,
            input_height=h,
            input_width=w,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            dtype=ttnn.bfloat16,
        )

    trace, _, br = traced(run_conv)
    print("\n" + format_breakdown(br, f"conv2d {in_c}->{out_c} 3x3 on {h}x{w}"))
    assert_matches_cpp(trace, br)

    names = [o.name for o in br.ops]
    print("\nop nesting:")
    for o in br.ops:
        print(f"  {'  ' * o.depth}{o.name}  (depth={o.depth}, program L1 {o.program_l1} B)")

    # The composite frame wraps at least a halo and a matmul-family device op.
    halo = [o for o in br.ops if "Halo" in o.name]
    assert halo, names
    assert [o for o in br.ops if "Matmul" in o.name or "Conv" in o.name], names

    # Every child device op gets its own program, so program-scope L1 is released between them:
    # there is one circular_buffer_deallocate_all per program.
    releases = [e for e in br.events if e.kind == "program_l1"]
    assert len(releases) >= 2, "each child op's program releases the previous program's L1"

    # No child's own program L1 can exceed the whole-capture CB peak, and the peak is reached
    # inside one specific child -- that child is the one to attack if the conv does not fit.
    children = [o for o in br.ops if o.depth > 0 and o.program_l1 > 0]
    assert children
    worst = max(children, key=lambda o: o.program_l1)
    print(f"\nlargest program-scope L1 among children: {worst.name} at {worst.program_l1} B")
    assert worst.program_l1 <= br.peak_cb + br.peak_dfb + br.peak_scratchpad

    # Halo's output is a sharded L1 intermediate: it is allocated inside the capture and is
    # still live while the conv's own program runs, so it lands in peak_l1.
    assert br.peak_l1 > 0
    assert [e for e in br.events if e.kind == "l1_buffer" and e.event == "alloc"]
