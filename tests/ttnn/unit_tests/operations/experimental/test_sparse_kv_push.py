# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.sparse_push_plan + sparse_kv_push, bit-exact on every chip of an 8x4 torus, and on the other 2D
fabrics a Galaxy can run (no wraparound, one axis wrapped, and submeshes with a linear or a 2-chip axis).

Each cache row encodes its own global row and slot, so reading the compact buffer through the plan's rewritten indices
must return exactly the rows the top-k names. Index patterns are synthetic edge cases for the plan's per-owner runs,
packet packing, masking, and the prefix/remainder split; real GLM-5.3 top-k runs in the nightly sparse SDPA push test.
"""

import zlib

import numpy as np
import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart as Part

# The ops' compact layout (pages_per_packet, banks), exported for tests to lay out reference buffers.
packet_shape = ttnn._ttnn.operations.experimental._sparse_kv_push_packet_shape

SENTINEL = 0xFFFFFFFF
FILL = 0x5A5A5A5A
TOPK = 2048


def _device_params(fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY):
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 6144
    return {
        "fabric_config": fabric_config,
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 2048,
        "trace_region_size": 16 * 1024 * 1024,
    }


GALAXY = [
    pytest.mark.parametrize("device_params", [_device_params()], indirect=True),
    pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True),
]


def galaxy(fn):
    for mark in reversed(GALAXY):
        fn = mark(fn)
    return fn


# The other 2D fabrics, as (id, fabric config, submesh shape and offset or None for the whole 8x4). The push sends only
# on the links each chip has, so a chip at the end of a line has fewer channels than one in a ring. Submeshes are cut
# from the open 8x4, as the model splits a Galaxy: a fabric opened on fewer chips than its torus spans fails router sync.
FABRICS = [
    ("2d-8x4", ttnn.FabricConfig.FABRIC_2D, None),  # no wraparound
    ("torus_x-8x4", ttnn.FabricConfig.FABRIC_2D_TORUS_X, None),  # rows of 4 wrap, columns of 8 are lines
    ("torus_y-8x4", ttnn.FabricConfig.FABRIC_2D_TORUS_Y, None),  # columns of 8 wrap, rows of 4 are lines
    ("torus_xy-4x4", ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ((4, 4), (4, 0))),  # half a Galaxy: [LINE, RING]
    ("torus_xy-2x4", ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ((2, 4), (0, 0))),  # a 2-chip axis never wraps
]


def other_fabrics(fn):
    return pytest.mark.parametrize(
        "device_params, mesh_device, submesh",
        [(_device_params(config), (8, 4), submesh) for _, config, submesh in FABRICS],
        ids=[name for name, _, _ in FABRICS],
        indirect=["device_params", "mesh_device"],
    )(fn)


def _mesh(mesh_device, submesh):
    """The whole mesh, or its (shape, offset) submesh."""
    if submesh is None:
        return mesh_device
    shape, offset = submesh
    return mesh_device.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(*offset))


def _num_devices(mesh_device):
    return mesh_device.get_num_devices()


# ----------------------------------------------------------------------------------------------------------------------
# Cache, buffers and the plan + push
# ----------------------------------------------------------------------------------------------------------------------
def _row_words(rows, slot, width):
    """Contents of global cache rows: [row, slot, hash...], so a misplaced row names where it came from."""
    rows = np.asarray(rows, dtype=np.int64)[:, None]
    words = (rows * 1000003 + slot * 7919 + np.arange(width) * 31) & 0x7FFFFFFF
    words[:, 0], words[:, 1] = rows[:, 0], slot
    return words.astype(np.int32)


def _cache(mesh_device, slots, max_rows, sc, width):
    """Block-cyclic cache: global row n lives on chip (n / sc) % N at local row (n / sc / N) * sc + n % sc."""
    num_chips = _num_devices(mesh_device)
    per_chip = max_rows // num_chips
    p = np.arange(max_rows)
    chip, local = p // per_chip, p % per_chip
    global_row = ((local // sc) * num_chips + chip) * sc + local % sc
    host = np.stack([_row_words(global_row, s, width) for s in range(slots)])[:, None]
    return ttnn.from_torch(
        torch.from_numpy(host),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )


def _replicated(mesh_device, host):
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(host).astype(np.uint32).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _scalar(mesh_device, value):
    return _replicated(mesh_device, np.array([[[[value]]]]))


def _filled_buffer(mesh_device, rows, width):
    return _replicated(mesh_device, np.full((1, 1, rows, width), FILL, dtype=np.uint32))


def _semaphores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return [ttnn.create_global_semaphore(mesh_device, cores, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]


def _required_rows(idx, sc, unit):
    """Compact rows the plan lays out: every owner's run padded to whole units; at least the op's lower bound."""
    n, need = idx.shape[0], 0
    for d in range(n):
        rows = np.unique(idx[d][idx[d] != SENTINEL])
        counts = np.bincount((rows // sc) % n, minlength=n)
        need = max(need, int((-(-counts // unit) * unit).sum()))
    return max(need, n * unit)


class Push:
    """One plan for `idx`, its all-gathered tables, and pushes from `cache` into compact buffers."""

    def __init__(self, mesh_device, cache, idx, *, max_rows, sc, page_bytes, chunk_start, links=2, start_tensor=True):
        self.mesh_device, self.cache, self.idx = mesh_device, cache, idx
        self.n = _num_devices(mesh_device)
        self.max_rows, self.sc, self.links = max_rows, sc, links
        self.width = page_bytes // 4
        p, banks = packet_shape(mesh_device, page_bytes)
        self.rows = _required_rows(idx, sc, p * banks)
        self.sems = _semaphores(mesh_device)
        self.indices = ttnn.from_torch(
            torch.from_numpy(idx.reshape(1, 1, -1, idx.shape[-1]).astype(np.uint32).view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )
        # The chunk start as the traced path passes it (a device tensor) or as the eager path does (a runtime arg).
        self.start = _scalar(mesh_device, chunk_start) if start_tensor else None
        self.chunk_start = chunk_start
        self.tables = None
        self.page_bytes = page_bytes

    def plan(self):
        table, self.compact_indices = ttnn.experimental.sparse_push_plan(
            self.indices,
            stripe_chunk=self.sc,
            max_rows=self.max_rows,
            page_bytes=self.page_bytes,
            chunk_start=None if self.start is not None else self.chunk_start,
            chunk_start_tensor=self.start,
        )
        # The plan op labels each chip's table as its share of the rows, for the full-mesh gather.
        assert all(isinstance(p, ttnn.PlacementShard) and p.dim == 2 for p in table.tensor_topology().placements())
        if self.tables is None:
            self.tables = _replicated(self.mesh_device, np.zeros((1, 1, table.shape[2] * self.n, table.shape[3])))
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self.tables, cluster_axis=None, num_links=self.links
        )

    def push(self, out, part, **slot):
        pushed = ttnn.experimental.sparse_kv_push(
            self.cache,
            output_tensor=out,
            table=self.tables,
            stripe_chunk=self.sc,
            max_rows=self.max_rows,
            part=part,
            num_links=self.links,
            ready_semaphore=self.sems[0],
            done_semaphore=self.sems[1],
            **slot,
        )
        # Each chip's compact buffer holds different rows: labelled like the cache, sharded over its rows.
        assert all(isinstance(p, ttnn.PlacementShard) and p.dim == 2 for p in pushed.tensor_topology().placements())
        return pushed


def _check(push, out, *, slot, part, chunk_start, tag):
    """Every referenced row sits at its rewritten slot, distinct rows take distinct slots, and masked indices stay
    masked. PREFIX pushes the blocks before the one holding chunk_start and REST the others, so each leaves the
    other's rows untouched."""
    mesh = push.mesh_device
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).numpy().reshape(push.n, -1, push.width)
    compact = ttnn.to_torch(push.compact_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
    compact = (compact.to(torch.int64).numpy() & 0xFFFFFFFF).reshape(push.idx.shape)
    for d in range(push.n):
        idx, cmp = push.idx[d], compact[d]
        valid = idx != SENTINEL
        assert np.all(cmp[~valid] == SENTINEL), f"{tag}: chip {d} rewrote a masked index"
        rows, slots = np.unique(np.stack([idx[valid], cmp[valid]]), axis=1)
        assert len(np.unique(rows)) == len(rows) == len(np.unique(slots)), f"{tag}: chip {d} slots not one-to-one"
        assert np.all(slots < got.shape[1]), f"{tag}: chip {d} slot past the buffer"
        split = chunk_start // push.sc * push.sc
        pushed = {Part.ALL: np.ones_like(rows, dtype=bool), Part.PREFIX: rows < split, Part.REST: rows >= split}[part]
        want = _row_words(rows[pushed], slot, push.width)
        bad = np.flatnonzero(np.any(got[d, slots[pushed]] != want, axis=1))
        assert not bad.size, (
            f"{tag}: chip {d} row {rows[pushed][bad[0]]} at slot {slots[pushed][bad[0]]} "
            f"holds row {got[d, slots[pushed][bad[0]], 0]} slot {got[d, slots[pushed][bad[0]], 1]}"
        )
        assert np.all(got[d, slots[~pushed]] == np.int32(FILL)), f"{tag}: chip {d} part {part} wrote another part's row"


# ----------------------------------------------------------------------------------------------------------------------
# Index patterns: per chip, the set of rows its top-k references, laid out over its queries
# ----------------------------------------------------------------------------------------------------------------------
def _owner_rows(owner, sc, end, n):
    """Every row below `end` that `owner` of `n` chips holds."""
    blocks = np.arange(owner, -(-end // sc), n)
    rows = (blocks[:, None] * sc + np.arange(sc)).reshape(-1)
    return rows[rows < end]


def _lay_out(row_sets, queries, rng):
    """[N, queries, TOPK]: each query takes a window of its chip's rows (so rows repeat across queries, never within
    one), in shuffled order, padded with masked entries."""
    idx = np.full((len(row_sets), queries, TOPK), SENTINEL, dtype=np.int64)
    for d, rows in enumerate(row_sets):
        rows = rng.permutation(np.asarray(rows, dtype=np.int64))
        n = min(len(rows), TOPK)
        for q in range(queries):
            if len(rows):
                idx[d, q, :n] = np.roll(rows, -q * 97)[:n]
            idx[d, q, :n] = rng.permutation(idx[d, q, :n])
    return idx


def _pattern(name, rng, *, sc, end, start, shape=(8, 4)):
    """Row sets per chip of a `shape` mesh for the synthetic pattern `name`; rows are below `end` (the populated
    cache)."""
    pick = lambda pool, k: rng.choice(pool, size=min(k, len(pool)), replace=False)
    mesh_rows, mesh_cols = shape
    n = mesh_rows * mesh_cols
    owner_rows = lambda owner: _owner_rows(owner, sc, end, n)
    if name == "random":
        return [pick(np.arange(end), 3000) for _ in range(n)]
    if name == "clustered":
        centers = rng.integers(0, end, size=(n, 8))
        return [np.unique((c[:, None] + np.arange(400)) % end) for c in centers]
    if name == "one_owner":
        return [pick(owner_rows((d + 1) % n), 3000) for d in range(n)]
    if name == "local_only":
        return [pick(owner_rows(d), 3000) for d in range(n)]
    if name == "routing_ties":
        # The chip opposite on both axes: on a torus both directions are equally short on each axis.
        opposite = (
            lambda d: ((d // mesh_cols + mesh_rows // 2) % mesh_rows) * mesh_cols
            + (d % mesh_cols + mesh_cols // 2) % mesh_cols
        )
        return [pick(owner_rows(opposite(d)), 3000) for d in range(n)]
    if name == "run_lengths":
        # Per-owner counts around one packet (8 rows) and one bank-lane unit (64 rows).
        counts = [1, 7, 8, 9, 63, 64, 65]
        return [
            np.concatenate([pick(owner_rows(o), counts[(o + d) % len(counts)]) for o in range(n)]) for d in range(n)
        ]
    if name == "edges":
        # Block and bitmap-word edges, the reader's 16-block window edge, and the first and last row.
        blocks = np.array([o + n * b for o in range(n) for b in (0, 15, 16)])
        rows = (blocks[:, None] * sc + np.array([0, 31, 32, sc - 1])).reshape(-1)
        rows = np.unique(np.concatenate([rows[rows < end], [0, end - 1]]))
        return [rows for _ in range(n)]
    if name == "duplicates":
        hot = pick(np.arange(end), 50)
        return [hot for _ in range(n)]
    if name == "remainder_only":
        return [pick(np.arange(start, end), 3000) for _ in range(n)]
    if name == "prefix_only":
        return [pick(np.arange(start), 3000) for _ in range(n)]
    if name == "everything":
        return [np.arange(end) for _ in range(n)]
    if name == "tiny_prefix":
        return [np.arange(min(end, 3 * sc)) for _ in range(n)]
    raise ValueError(name)


def _mask(idx, rng):
    """Masked tails of varying length, fully masked queries, and one fully masked chip."""
    q, k = idx.shape[1], idx.shape[2]
    for d in range(idx.shape[0]):
        for j in range(q):
            idx[d, j, k - (j * 131) % k :] = SENTINEL
        idx[d, ::7] = SENTINEL
    idx[5] = SENTINEL
    return idx


# ----------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------
# (pattern, cache rows, chunk start offset from the last chunk); 92160 rows give every owner more than 16 blocks.
PATTERNS = [
    ("random", 92160, 0),
    ("one_owner", 92160, 0),
    ("local_only", 92160, 0),
    ("routing_ties", 92160, 0),
    ("run_lengths", 92160, 0),
    ("edges", 92160, 0),
    ("duplicates", 92160, 0),
    ("remainder_only", 92160, 0),
    ("prefix_only", 92160, 0),
    ("unaligned_start", 92160, 2592),
    ("everything", 10240, 0),
    ("tiny_prefix", 10240, 0),
]


def _run_parts(mesh_device, push, *, slot, chunk_start, tag, **slot_args):
    push.plan()
    full = push.push(_filled_buffer(mesh_device, push.rows, push.width), Part.ALL, **slot_args)
    _check(push, full, slot=slot, part=Part.ALL, chunk_start=chunk_start, tag=f"{tag} all")
    rest = push.push(_filled_buffer(mesh_device, push.rows, push.width), Part.REST, **slot_args)
    _check(push, rest, slot=slot, part=Part.REST, chunk_start=chunk_start, tag=f"{tag} rest")
    prefix = push.push(_filled_buffer(mesh_device, push.rows, push.width), Part.PREFIX, **slot_args)
    _check(push, prefix, slot=slot, part=Part.PREFIX, chunk_start=chunk_start, tag=f"{tag} prefix")
    both = push.push(prefix, Part.REST, **slot_args)
    _check(push, both, slot=slot, part=Part.ALL, chunk_start=chunk_start, tag=f"{tag} prefix + rest")


def _run_pattern(mesh_device, pattern, max_rows, offset):
    sc, page_bytes = 160, 704
    rng = np.random.default_rng(zlib.crc32(pattern.encode()))
    start = max_rows - sc * _num_devices(mesh_device) + offset
    rows = _pattern(
        "random" if pattern == "unaligned_start" else pattern,
        rng,
        sc=sc,
        end=max_rows,
        start=start,
        shape=tuple(mesh_device.shape),
    )
    # Masked entries in every pattern; query 0 and the unmasked heads of the others still reference rows.
    idx = _mask(_lay_out(rows, sc, rng), rng)
    cache = _cache(mesh_device, 2, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start)
    _run_parts(mesh_device, push, slot=1, chunk_start=start, tag=pattern, batch_index=1)


@galaxy
@pytest.mark.parametrize("pattern, max_rows, offset", PATTERNS, ids=[p[0] for p in PATTERNS])
def test_sparse_kv_push_patterns(mesh_device, pattern, max_rows, offset):
    _run_pattern(mesh_device, pattern, max_rows, offset)


# Every pair of values of any two settings appears in some case; the first case is production.
CONFIGS = [
    (704, 160, 2, True),
    (704, 64, 1, False),
    (1152, 160, 1, False),
    (1152, 64, 2, False),
    (1152, 64, 1, True),
]


def _config_id(page_bytes, sc, links, slot_tensor):
    return f"{page_bytes}B-stripe{sc}-{links}link-{'slot_tensor' if slot_tensor else 'slot_int'}"


def _run_config(mesh_device, page_bytes, sc, links, slot_tensor):
    max_rows = 92160
    rng = np.random.default_rng(sc * 7 + links)
    start = max_rows - sc * _num_devices(mesh_device)
    rows = _pattern("clustered", rng, sc=sc, end=max_rows, start=start, shape=tuple(mesh_device.shape))
    idx = _mask(_lay_out(rows, sc, rng), rng)
    cache = _cache(mesh_device, 4, max_rows, sc, page_bytes // 4)
    # The slot and the chunk start both as tensors (the traced path) or both as ints (the eager path).
    push = Push(
        mesh_device,
        cache,
        idx,
        max_rows=max_rows,
        sc=sc,
        page_bytes=page_bytes,
        chunk_start=start,
        links=links,
        start_tensor=slot_tensor,
    )
    # Slot 3 either way: user 1 of 2 layers, layer 1.
    slot_args = (
        dict(batch_index_tensor=_scalar(mesh_device, 1), batch_slot_num_layers=2, batch_slot_layer_idx=1)
        if slot_tensor
        else dict(batch_index=3)
    )
    _run_parts(mesh_device, push, slot=3, chunk_start=start, tag=f"{page_bytes}B sc{sc} {links}L", **slot_args)


@galaxy
@pytest.mark.parametrize("page_bytes, sc, links, slot_tensor", CONFIGS, ids=[_config_id(*c) for c in CONFIGS])
def test_sparse_kv_push_configs(mesh_device, page_bytes, sc, links, slot_tensor):
    _run_config(mesh_device, page_bytes, sc, links, slot_tensor)


@galaxy
def test_sparse_kv_push_worst_case_union(mesh_device):
    """Every query names TOPK rows no other query names, the most a chip can need: the buffer the op sizes for
    queries x top-k distinct rows (that many plus one unit per owner) must hold it. 32 queries keep it cheap; the bound
    is the same for the model's 160."""
    sc, page_bytes, queries = 160, 64, 32
    n = _num_devices(mesh_device)
    distinct = queries * TOPK
    max_rows = -(-distinct // (sc * n)) * sc * n
    idx = np.stack([np.random.default_rng(d).permutation(distinct).reshape(queries, TOPK) for d in range(n)])
    start = max_rows - sc * n
    cache = _cache(mesh_device, 1, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start)
    cap = ttnn.experimental.sparse_kv_push_output_spec(
        mesh_device, distinct_rows=distinct, max_rows=max_rows, stripe_chunk=sc, row_width=push.width, dtype=ttnn.uint32
    ).shape[2]
    assert push.rows <= cap, f"{push.rows} rows needed, over the op's buffer of {cap}"
    _run_parts(mesh_device, push, slot=0, chunk_start=start, tag="worst case", batch_index=0)


@galaxy
def test_sparse_kv_push_traced(mesh_device):
    """Plan + table gather + push captured once, replayed with new indices, chunk start and user slot in place."""
    sc, page_bytes, max_rows = 160, 704, 92160
    n = _num_devices(mesh_device)
    rng = np.random.default_rng(3)
    first, second = max_rows - sc * n, max_rows - 2 * sc * n
    idx_a = _lay_out(_pattern("clustered", rng, sc=sc, end=max_rows, start=first), sc, rng)
    idx_b = _lay_out(_pattern("random", rng, sc=sc, end=second + sc * n, start=second), sc, rng)
    cache = _cache(mesh_device, 4, max_rows, sc, page_bytes // 4)
    # Size the buffer for either set of indices: the capture fixes its address.
    unit = np.prod(packet_shape(mesh_device, page_bytes))
    push = Push(mesh_device, cache, idx_a, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=first)
    rows = max(_required_rows(idx_a, sc, unit), _required_rows(idx_b, sc, unit))
    user = _scalar(mesh_device, 0)
    out = _filled_buffer(mesh_device, rows, push.width)
    slot_args = dict(batch_index_tensor=user, batch_slot_num_layers=2, batch_slot_layer_idx=1)

    def issue():
        push.plan()
        push.push(out, Part.ALL, **slot_args)

    issue()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    issue()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for u, idx, start in [(1, idx_b, second), (0, idx_a, first)]:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.from_numpy(idx.reshape(1, 1, -1, TOPK).astype(np.uint32).view(np.int32)),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
                ),
                push.indices,
            )
            for value, tensor in ((u, user), (start, push.start)):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.tensor([[[[value]]]], dtype=torch.int32),
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    ),
                    tensor,
                )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.from_numpy(np.full((1, 1, rows, push.width), FILL, dtype=np.uint32).view(np.int32)),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                ),
                out,
            )
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            push.idx = idx
            _check(push, out, slot=u * 2 + 1, part=Part.ALL, chunk_start=start, tag=f"replay user {u} start {start}")
    finally:
        ttnn.release_trace(mesh_device, tid)


# ----------------------------------------------------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------------------------------------------------
def _mismatch_marker(reference, actual, mask=None):
    """Per chip, non-zero when `actual` differs from `reference` anywhere `mask` (broadcast over the row) is set."""
    marker = ttnn.ne(reference, actual, dtype=ttnn.bfloat16)
    if mask is not None:
        marker = ttnn.multiply(marker, mask)
    return ttnn.max(marker)


def _merge_markers(current, new):
    return new if current is None else ttnn.maximum(current, new)


def _marker_set(marker):
    host = ttnn.from_device(marker)
    return any(float(ttnn.to_torch(chip).item()) != 0.0 for chip in ttnn.get_device_tensors(host))


def _run_determinism(mesh_device):
    """The prefix push then the rest, repeated on one plan, fill every referenced slot bit-identically. Compared on
    device, one buffer reused across runs; only referenced slots count, since the rest of a packet is undefined."""
    sc, page_bytes, max_rows, runs = 160, 704, 92160, 5
    n = _num_devices(mesh_device)
    rng = np.random.default_rng(11)
    start = max_rows - sc * n
    rows = _pattern("clustered", rng, sc=sc, end=max_rows, start=start, shape=tuple(mesh_device.shape))
    idx = _mask(_lay_out(rows, sc, rng), rng)
    cache = _cache(mesh_device, 2, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start)
    push.plan()

    compact = ttnn.to_torch(push.compact_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
    compact = (compact.to(torch.int64).numpy() & 0xFFFFFFFF).reshape(n, -1)
    referenced = np.zeros((n, push.rows), dtype=np.float32)
    for d in range(n):
        referenced[d, compact[d][compact[d] != SENTINEL]] = 1
    mask = ttnn.from_torch(
        torch.from_numpy(referenced).reshape(1, 1, n * push.rows, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    fill = ttnn.from_torch(
        torch.from_numpy(np.full((1, 1, push.rows, push.width), FILL, dtype=np.uint32).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run(out):
        ttnn.copy_host_to_device_tensor(fill, out)
        push.push(out, Part.PREFIX, batch_index=1)
        push.push(out, Part.REST, batch_index=1)

    reference = _filled_buffer(mesh_device, push.rows, push.width)
    run(reference)
    out = _filled_buffer(mesh_device, push.rows, push.width)
    marker = None
    for _ in range(runs - 1):
        run(out)
        marker = _merge_markers(marker, _mismatch_marker(reference, out, mask))
    assert not _marker_set(marker), f"a push differs from the first of {runs} in a referenced slot"


@galaxy
def test_sparse_kv_push_determinism(mesh_device):
    _run_determinism(mesh_device)


# ----------------------------------------------------------------------------------------------------------------------
# Other 2D fabrics: a few patterns (incl. the axis-opposite chips), one non-production config, and determinism
# ----------------------------------------------------------------------------------------------------------------------
FABRIC_PATTERNS = [p for p in PATTERNS if p[0] in ("random", "routing_ties", "run_lengths")]
FABRIC_CONFIGS = [(1152, 64, 1, True)]


@other_fabrics
@pytest.mark.parametrize("pattern, max_rows, offset", FABRIC_PATTERNS, ids=[p[0] for p in FABRIC_PATTERNS])
def test_sparse_kv_push_fabric_patterns(mesh_device, submesh, pattern, max_rows, offset):
    mesh = _mesh(mesh_device, submesh)
    assert ttnn.experimental.is_sparse_kv_push_supported(mesh, 160, max_rows)
    _run_pattern(mesh, pattern, max_rows, offset)


@other_fabrics
@pytest.mark.parametrize(
    "page_bytes, sc, links, slot_tensor", FABRIC_CONFIGS, ids=[_config_id(*c) for c in FABRIC_CONFIGS]
)
def test_sparse_kv_push_fabric_configs(mesh_device, submesh, page_bytes, sc, links, slot_tensor):
    _run_config(_mesh(mesh_device, submesh), page_bytes, sc, links, slot_tensor)


@other_fabrics
def test_sparse_kv_push_fabric_determinism(mesh_device, submesh):
    _run_determinism(_mesh(mesh_device, submesh))
