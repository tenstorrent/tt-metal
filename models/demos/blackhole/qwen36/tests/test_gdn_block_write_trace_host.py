# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the captured GDN block writes of gdn_block_state.GdnBlockState (record_traces,
_replay_write) and the order of the prefill rows' logits reads, block writes and wait in
Qwen36ForCausalLM.prefill_forward.

A recording stand-in for ttnn keeps every paged_fill_cache as (destination view, source, page table). An eager write
resolves the page table to the page ids staged in it at that moment; a captured write keeps the device table and a
replay resolves it to the page ids staged before the replay, as the device reads them. So the two paths can be
compared write by write: the replay must issue the eager path's writes, from the same sources into the same pages.
"""

import re
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt.kv_cache_spec import GdnDims, plan_gdn_block_layout

QWEN_27B_TP8 = GdnDims(nk=2, nv=6, dk=128, dv=128, kconv=4)
ATTENTION = [i % 4 == 3 for i in range(16)]  # 4 attention layers (4 layer slots), 12 GDN layers in 3 groups
NUM_BLOCKS = 40
BMAX = 4


class _Dev:
    """A device tensor stand-in that only has a name."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class _Gdn:
    """TPGatedDeltaNet's prefill checkpoint bookkeeping (take / release / note_traced_prefill) over named tensors."""

    def __init__(self, i):
        self.name = i
        self.rec_state, self.conv_carry = _Dev(f"rec{i}"), _Dev(f"carry{i}")
        self.prefill_checkpoint_buffers = (_Dev(f"ck_rec{i}"), _Dev(f"ck_carry{i}"))
        self.prefill_checkpoint = None
        self.state_blocks = SimpleNamespace(rec_k=f"rec_k{i % 4}", rec_v=f"rec_v{i % 4}", hist=f"hist{i % 4}")
        self.freed = []

    def take_prefill_checkpoint(self):
        ck, self.prefill_checkpoint = self.prefill_checkpoint, None
        assert ck is not None, "no checkpoint to take"
        return ck

    def release_prefill_checkpoint(self, rec, conv):
        if rec is not self.prefill_checkpoint_buffers[0]:
            self.freed += [rec, conv]

    def note_traced_prefill(self):
        self.prefill_checkpoint = self.prefill_checkpoint_buffers


@pytest.fixture
def device(monkeypatch):
    """Record the device work GdnBlockState issues: the effective writes, table copies, captures and waits."""
    dev = SimpleNamespace(writes=[], copies=[], syncs=0, traces={}, capturing=None, replays=[], released=[], staged={})

    def from_torch(t, **kw):
        return SimpleNamespace(host=t.clone())

    def copy(host, table):
        dev.copies.append(table)
        dev.staged[table] = host.host.reshape(-1).tolist()

    def fill(cache, src, table, batch_idx=0):
        if dev.capturing is not None:
            dev.traces[dev.capturing].append((cache, src, table))
        else:
            dev.writes.append((cache, src, list(dev.staged.get(table, [])), "eager", table))

    def begin(mesh, cq_id=0):
        tid = len(dev.traces) + 1
        dev.traces[tid] = []
        dev.capturing = tid
        return tid

    def end(mesh, tid, cq_id=0):
        assert dev.capturing == tid
        dev.capturing = None

    def execute(mesh, tid, cq_id=0, blocking=True):
        assert not blocking, "a block write replays without waiting"
        dev.replays.append(tid)
        dev.writes += [
            (cache, src, list(dev.staged.get(table, [])), "trace", table) for cache, src, table in dev.traces[tid]
        ]

    def sync(mesh):
        dev.syncs += 1

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy)
    monkeypatch.setattr(ttnn.experimental, "paged_fill_cache", fill, raising=False)
    monkeypatch.setattr(ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(ttnn, "end_trace_capture", end)
    monkeypatch.setattr(ttnn, "execute_trace", execute)
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: dev.released.append(tid))
    monkeypatch.setattr(ttnn, "synchronize_device", sync)
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape, **kw: ("reshape", t))
    monkeypatch.setattr(ttnn, "slice", lambda t, a, b, **kw: ("slice", t, tuple(a), tuple(b)))
    monkeypatch.setattr(ttnn, "to_layout", lambda t, layout, **kw: ("row major", t))
    monkeypatch.setattr(ttnn, "embedding", lambda table, rows, **kw: ("packed", rows))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(gbs, "_free_copy", lambda view, source: None)
    return dev


def _state(block_size=1024):
    """A GdnBlockState over 12 stand-in GDN layers in 3 MambaSpec groups, with its write and trace page tables, built
    as the constructor builds them but without a device. bfloat8_b 1024-token blocks put 4 value heads in K and 2 in V,
    so a write slices the recurrent state into both pools."""
    layout = plan_gdn_block_layout(block_size=block_size, kv_heads=1, head_dim=256, kv_bf8=True, dims=QWEN_27B_TP8)
    plugin = PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, True, num_blocks=NUM_BLOCKS, block_size=block_size)
    st = object.__new__(gbs.GdnBlockState)
    st.mesh, st.bmax, st.num_blocks, st.dims, st.layout = None, BMAX, NUM_BLOCKS, QWEN_27B_TP8, layout
    st.layers = {i: _Gdn(i) for i, a in enumerate(ATTENTION) if not a}
    st._group_names = st._group_layers = None
    st._decode_blocks, st._index_refs = {}, {}
    st._pack_table = "pack table"
    parts = len(layout.write_tables(0))
    st._write_tables = [_Dev(f"write table {p}") for p in range(parts)]
    st._write_refs = []
    st._write_group = gbs.table_groups(plugin.per_layer_specs)
    st._trace_tables = {
        v: [[_Dev(f"trace table {int(v)}.{k}.{p}") for p in range(parts)] for k in range(plugin.num_groups)]
        for v in (False, True)
    }
    st._trace_staged = {}
    st._hold_tables = False
    st._held_tables = []
    st._write_traces = {}
    st._trace_sources = {}
    st._warmed_variants = ()
    st._write_order = None
    st._zero_trace = None
    st._warmed = False
    st.kv_caches = []
    return st, plugin


def _capture(st, dev):
    """Compile (warm) and capture the writes as Qwen36ForCausalLM.warmup_model_prefill does; returns the writes the
    warm pass issued."""
    st._warmed_variants = st._write_trace_variants()
    for checkpoint in st._warmed_variants:
        st._write_trace_body(checkpoint)
    warm = list(dev.writes)
    dev.writes.clear()
    st.record_traces()
    return warm


def _strip(writes):
    return [w[:3] for w in writes]


def _eager_writes(dev, checkpoint, dests):
    """The writes of the eager path for ``dests`` (one block per group)."""
    st, plugin = _state()
    state = plugin.step_state([dests], BMAX)
    if checkpoint:
        for dn in st.layers.values():
            dn.note_traced_prefill()
        st.write_checkpoint(state, dests)
    else:
        st.write_state(state, dests)
    assert {w[3] for w in dev.writes} == {"eager"}
    return _strip(dev.writes)


def _leaves(x):
    if isinstance(x, tuple):
        for y in x:
            yield from _leaves(y)
    elif isinstance(x, _Dev):
        yield x


def _layer_of(src):
    """The GDN layer whose tensor a write's source is computed from."""
    (layer,) = {int(re.sub(r"\D", "", repr(t))) for t in _leaves(src)}
    return layer


@pytest.mark.parametrize("checkpoint", [False, True])
def test_a_replay_issues_the_eager_writes_into_each_groups_block(device, checkpoint):
    """The captured write of every GDN layer reads the same source (the workspace, or the checkpoint buffers of a
    captured prefill), through the same operations, into the same pages as the eager write: each MambaSpec group's
    page tables are staged for its own block before the replay. So the replay writes the same bytes."""
    st, plugin = _state()
    _capture(st, device)
    assert device.writes == [] and len(device.traces) == 2, "a capture writes nothing"
    dests = (11, 12, 13)
    state = plugin.step_state([dests], BMAX)
    syncs = device.syncs
    if checkpoint:
        for dn in st.layers.values():
            dn.note_traced_prefill()
        st.write_checkpoint(state, dests)
    else:
        st.write_state(state, dests)
    assert device.syncs == syncs, "a replayed write does not wait for the device"
    assert {w[3] for w in device.writes} == {"trace"}, "no eager write"
    assert len(device.replays) == 1
    replayed = _strip(device.writes)
    device.writes.clear()
    # The trace writes layer by layer, the eager path group by group; no two writes share a page.
    assert sorted(map(repr, replayed)) == sorted(map(repr, _eager_writes(device, checkpoint, dests)))
    # 12 layers x (2 recurrent pools + the history), each into the pages of its own group's block.
    assert len(replayed) == 3 * len(st.layers)
    group_of = {i: g for g, layers in enumerate(plugin.gdn_groups) for i in layers}
    for i in st.layers:
        got = [ids for _, src, ids in replayed if _layer_of(src) == i]
        assert got == st.layout.write_tables(dests[group_of[i]])
    if checkpoint:
        assert all(dn.prefill_checkpoint is None and dn.freed == [] for dn in st.layers.values())


def test_the_trace_reads_the_workspace_or_the_checkpoint_buffers(device):
    """The workspace trace reads each GDN layer's batch-1 workspace, the checkpoint trace the buffers a captured
    prefill copies its checkpoint into; each layer writes through the tables of its table group."""
    st, plugin = _state()
    _capture(st, device)
    work, ck = device.traces[1], device.traces[2]
    assert {repr(t) for _, src, _ in work for t in _leaves(src)} == {
        f"{n}{i}" for i in st.layers for n in ("rec", "carry")
    }
    assert {repr(t) for _, src, _ in ck for t in _leaves(src)} == {
        f"{n}{i}" for i in st.layers for n in ("ck_rec", "ck_carry")
    }
    for trace, variant in ((work, False), (ck, True)):
        for g, layers in enumerate(plugin.gdn_groups):
            for i in layers:
                assert [t for _, src, t in trace if _layer_of(src) == i] == st._trace_tables[variant][g]


def test_unchanged_page_tables_are_not_staged_again(device):
    """A write into the blocks of the previous write of the same variant stages nothing; the host tensor of each
    staged table is held until that table is staged again. The two variants have their own tables."""
    st, plugin = _state()
    _capture(st, device)
    state = plugin.step_state([(11, 12, 13)], BMAX)
    st.write_state(state, (11, 12, 13))
    parts = len(st.layout.write_tables(0))
    assert len(device.copies) == 3 * parts
    held = {key: host for key, (_, host) in st._trace_staged.items()}
    device.copies.clear()
    st.write_state(state, (11, 12, 13))
    assert device.copies == [] and len(device.replays) == 2
    st.write_state(state, (11, 12, 14))
    assert device.copies == st._trace_tables[False][2]
    assert all(st._trace_staged[key][1] is host for key, host in held.items() if key[1] != 2)
    device.copies.clear()
    for dn in st.layers.values():
        dn.note_traced_prefill()
    st.write_checkpoint(state, (11, 12, 14))
    assert set(device.copies) == {t for tables in st._trace_tables[True] for t in tables}


def test_tables_replaced_while_holding_stay_alive_until_released(device):
    """While hold_replaced_tables holds, a write that restages a table keeps the host tensor of the table it replaces
    until release_held_tables: the rows of a prefill step enqueue their writes without waiting for the device, so the
    copy of the replaced table may not have run. An unchanged table is not restaged and keeps nothing; a table replaced
    after the hold ends is dropped as before. Holding never waits for the device."""
    st, plugin = _state()
    _capture(st, device)
    state = plugin.step_state([(11, 12, 13)], BMAX)
    st.write_state(state, (11, 12, 13))
    first = [host for key, (_, host) in st._trace_staged.items() if key[1] == 2]
    syncs = device.syncs
    st.hold_replaced_tables(True)
    st.write_state(state, (11, 12, 13))
    assert st._held_tables == []
    st.write_state(state, (11, 12, 14))
    assert len(first) == len(st.layout.write_tables(0))
    assert len(st._held_tables) == len(first) and all(a is b for a, b in zip(st._held_tables, first))
    st.hold_replaced_tables(False)
    st.write_state(state, (11, 12, 15))
    assert len(st._held_tables) == len(first), "a table replaced after the hold ends is not kept"
    st.release_held_tables()
    assert st._held_tables == [] and device.syncs == syncs


def test_writes_without_a_serving_trace_run_eagerly(device, expect_error):
    """The eager path writes when no trace serves the write: before a capture, for a checkpoint that an eager forward
    computed (new tensors, freed after the write), with another GDN workspace bound, and for MambaSpec groups that do
    not hold the k-th GDN layer of every layer slot. A null block is refused before anything is written."""
    st, plugin = _state()
    dests = (11, 12, 13)
    state = plugin.step_state([dests], BMAX)
    st.write_state(state, dests)
    assert {w[3] for w in device.writes} == {"eager"} and device.replays == []
    _capture(st, device)
    device.writes.clear()
    for i, dn in st.layers.items():
        dn.prefill_checkpoint = (_Dev(f"eager_rec{i}"), _Dev(f"eager_carry{i}"))
    st.write_checkpoint(state, dests)
    assert {w[3] for w in device.writes} == {"eager"} and device.replays == []
    assert all(len(dn.freed) == 2 and dn.prefill_checkpoint is None for dn in st.layers.values())
    device.writes.clear()
    saved = st.layers[0].rec_state
    st.layers[0].rec_state = _Dev("scratch")
    st.write_state(state, dests)
    assert {w[3] for w in device.writes} == {"eager"} and device.replays == []
    st.layers[0].rec_state = saved
    device.writes.clear()
    groups = list(state.groups)
    a, b = groups[0], groups[1]
    groups[0] = a.__class__(
        layer_names=(b.layer_names[0],) + a.layer_names[1:], windows=a.windows, start_blocks=a.start_blocks
    )
    groups[1] = b.__class__(
        layer_names=(a.layer_names[0],) + b.layer_names[1:], windows=b.windows, start_blocks=b.start_blocks
    )
    st.write_state(state.__class__(groups=tuple(groups)), dests)
    assert {w[3] for w in device.writes} == {"eager"} and device.replays == []
    device.writes.clear()
    # Group 0 split in two: each half holds k = 0 layers only, but not all of them.
    split = (
        a.__class__(layer_names=a.layer_names[:2], windows=a.windows, start_blocks=a.start_blocks),
        a.__class__(layer_names=a.layer_names[2:], windows=a.windows, start_blocks=a.start_blocks),
        *state.groups[1:],
    )
    st.write_state(state.__class__(groups=split), (10, *dests))
    assert {w[3] for w in device.writes} == {"eager"} and device.replays == []
    device.writes.clear()
    st.write_state(state, dests)
    assert device.replays == [1]
    writes = len(device.writes)
    with expect_error(ValueError, "has the null block 0"):
        st.write_state(state, (11, 0, 13))
    assert len(device.writes) == writes and device.replays == [1]


def test_warmup_runs_the_captured_writes_before_any_capture(device, monkeypatch):
    """GdnBlockState.warmup runs both captured writes once, on the trace sources and page tables (the null block 0),
    before it zeroes block 0 and before any capture, so a capture finds every program compiled."""
    st, plugin = _state()
    st._warmed = False
    monkeypatch.setattr(st, "_read_state", lambda dn, block: None)
    monkeypatch.setattr(st, "_copy_state", lambda dn, a, b: None)
    monkeypatch.setattr(st, "zero_blocks", lambda ids: device.writes.append(("zero", tuple(ids))))
    st.warmup()
    assert st._warmed_variants == (False, True) and device.traces == {}
    assert device.writes[-1] == ("zero", (0,))
    for variant, names in ((False, ("rec", "carry")), (True, ("ck_rec", "ck_carry"))):
        tables = {t for g in st._trace_tables[variant] for t in g}
        warm = [w for w in device.writes[:-1] if w[4] in tables]
        assert {repr(t) for w in warm for t in _leaves(w[1])} == {f"{n}{i}" for i in st.layers for n in names}
        assert {w[4] for w in warm} == tables
    st.record_traces()
    assert len(st._write_traces) == 2


def test_capture_needs_warmup_and_is_idempotent(device):
    """record_traces captures only after warmup compiled the writes, each variant once; without checkpoint buffers
    only the workspace write. release() releases the traces and later writes run eagerly."""
    st, plugin = _state()
    st.record_traces()
    assert device.traces == {}
    _capture(st, device)
    st.record_traces()
    assert sorted(st._write_traces) == [False, True] and len(device.traces) == 2
    st.release()
    assert sorted(device.released) == [1, 2] and st._write_traces == {} and st.layers == {}
    st, plugin = _state()
    for dn in st.layers.values():
        dn.prefill_checkpoint_buffers = None
    _capture(st, device)
    assert list(st._write_traces) == [False]


# ------------------------------------------------------------------------------------- prefill row order


@pytest.fixture
def adapter(monkeypatch):
    """Qwen36ForCausalLM.prefill_forward over a stand-in model that records the prefills, the logits reads and frees,
    the block writes, the hold and release of the host inputs, the event and the wait."""
    from models.common.tests.vllm_stubs import stub_vllm_if_missing

    stub_vllm_if_missing(monkeypatch)
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    events = []
    vocab = 8
    logits = _Dev("logits")

    def prefill(tokens, pt, actual_len, vision_tokens=None, start_pos=0, rope_token_ids=None, checkpoint_pos=0):
        events.append(("prefill", start_pos, actual_len, checkpoint_pos))
        return logits

    blocks = SimpleNamespace(
        layout=SimpleNamespace(block_size=1024),
        restore=lambda state, b: events.append(("restore", tuple(b))),
        write_state=lambda state, b: events.append(("write", tuple(b))),
        write_checkpoint=lambda state, b: events.append(("checkpoint", tuple(b))),
        hold_replaced_tables=lambda hold: events.append(("hold tables", hold)),
        release_held_tables=lambda: events.append(("release tables",)),
    )
    model = SimpleNamespace(
        args=SimpleNamespace(max_batch_size=4, vocab_size=vocab),
        mesh_device="mesh",
        gdn_blocks=blocks,
        prefill_traced_chunked=prefill,
        prefill_chunk_size=2048,
        hold_masked_trace_inputs=lambda hold: events.append(("hold inputs", hold)),
        release_masked_trace_inputs=lambda: events.append(("release inputs",)),
    )

    def from_device(t, blocking=True):
        assert t is logits and not blocking, "a row enqueues its logits read without waiting"
        events.append(("read logits",))
        return "host logits"

    def to_torch(t, **kw):
        assert t == "host logits"
        events.append(("logits to torch",))
        return torch.zeros(1, 1, vocab)

    def record_event(mesh, cq_id):
        assert mesh == "mesh" and cq_id == 0
        events.append(("event",))
        return "event"

    monkeypatch.setattr(ttnn, "from_device", from_device)
    monkeypatch.setattr(ttnn, "to_torch", to_torch)
    monkeypatch.setattr(ttnn, "record_event", record_event)
    monkeypatch.setattr(ttnn, "event_synchronize", lambda event: events.append(("wait", event)))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("sync",)))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t])
    monkeypatch.setattr(ttnn, "ConcatMeshToTensor", lambda *a, **kw: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: events.append(("free logits",)) if t is logits else None)
    wrapper = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    wrapper.model = [model]
    wrapper.data_parallel = 1
    plugin = PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, True, num_blocks=64, block_size=1024)

    def run_rows(rows):
        """rows: (start, end, dest, start_blocks, checkpoint) per prefill row of one step."""
        events.clear()
        state = plugin.step_state(
            [r[2] for r in rows], BMAX, start_blocks=[r[3] for r in rows], checkpoints=[r[4] for r in rows]
        )
        ends = [r[1] for r in rows]
        tokens, page_table = torch.zeros(len(rows), max(ends), dtype=torch.int64), torch.zeros(len(rows), 4)
        out, _ = wrapper.prefill_forward(
            tokens, page_table.to(torch.int32), None, ends, start_pos=[r[0] for r in rows], mamba_state=state
        )
        assert tuple(out.shape) == (len(rows), 1, vocab)
        return list(events)

    def run(start, end, dest, start_blocks=None, checkpoint=None):
        return run_rows([(start, end, dest, start_blocks, checkpoint)])

    run.rows = run_rows
    run.events = events
    return run


HOLD = [("hold inputs", True), ("hold tables", True)]
UNHOLD = [("hold inputs", False), ("hold tables", False)]
RELEASE = [("wait", "event"), ("release inputs",), ("release tables",)]


def test_a_prefill_row_reads_and_frees_its_logits_before_its_block_writes(adapter):
    """The logits copy of a prefill trace replay is allocated after the captures, and a write trace's replay can
    overwrite such a buffer: the row enqueues the read of its logits and frees them first, then writes its checkpoint
    and its end state. Between the two parts of a two-part prefill the first part's logits are freed before the
    checkpoint write. The host waits once, after the row's writes are enqueued, for an event recorded right after the
    read, then releases the held host inputs and converts the logits."""
    assert adapter(0, 2000, (4, 5, 6), checkpoint=(1024, (1, 2, 3))) == HOLD + [
        ("prefill", 0, 2000, 1024),
        ("read logits",),
        ("free logits",),
        ("event",),
        ("checkpoint", (1, 2, 3)),
        ("write", (4, 5, 6)),
        *UNHOLD,
        *RELEASE,
        ("logits to torch",),
    ]
    assert adapter(0, 2500, (4, 5, 6), checkpoint=(2048, (1, 2, 3))) == HOLD + [
        ("prefill", 0, 2048, 0),
        ("free logits",),
        ("write", (1, 2, 3)),
        ("prefill", 2048, 2500, 0),
        ("read logits",),
        ("free logits",),
        ("event",),
        ("write", (4, 5, 6)),
        *UNHOLD,
        *RELEASE,
        ("logits to torch",),
    ]
    assert adapter(2048, 2500, (7, 8, 9), start_blocks=(1, 2, 3)) == HOLD + [
        ("restore", (1, 2, 3)),
        ("prefill", 2048, 2500, 0),
        ("read logits",),
        ("free logits",),
        ("event",),
        ("write", (7, 8, 9)),
        *UNHOLD,
        *RELEASE,
        ("logits to torch",),
    ]


def test_the_rows_of_a_prefill_step_enqueue_back_to_back(adapter):
    """Three rows in one step: each row's forward, logits read, free and block writes are enqueued in the order of a
    step that waits per row, with no wait between rows. The one event follows the last row's read, so the host does not
    wait for that row's block writes; the held host inputs are released only after the wait, and every row's logits
    are converted after it."""
    events = adapter.rows(
        [
            (0, 2000, (4, 5, 6), None, (1024, (1, 2, 3))),
            (0, 700, (7, 8, 9), None, None),
            (2048, 2500, (10, 11, 12), (13, 14, 15), None),
        ]
    )
    assert events == HOLD + [
        ("prefill", 0, 2000, 1024),
        ("read logits",),
        ("free logits",),
        ("checkpoint", (1, 2, 3)),
        ("write", (4, 5, 6)),
        ("prefill", 0, 700, 0),
        ("read logits",),
        ("free logits",),
        ("write", (7, 8, 9)),
        ("restore", (13, 14, 15)),
        ("prefill", 2048, 2500, 0),
        ("read logits",),
        ("free logits",),
        ("event",),
        ("write", (10, 11, 12)),
        *UNHOLD,
        *RELEASE,
        *[("logits to torch",)] * 3,
    ]


def test_a_failed_row_waits_for_the_enqueued_reads_and_ends_the_hold(adapter, expect_error):
    """A row that raises after earlier rows enqueued their reads: the host waits for the device before the read host
    tensors go away, ends the hold and releases nothing (a later replay waits before it restages)."""
    with expect_error(ValueError, "has the start blocks"):
        adapter.rows([(0, 700, (7, 8, 9), None, None), (2048, 2500, (10, 11, 12), None, None)])
    assert adapter.events == HOLD + [
        ("prefill", 0, 700, 0),
        ("read logits",),
        ("free logits",),
        ("write", (7, 8, 9)),
        ("sync",),
        *UNHOLD,
    ]
