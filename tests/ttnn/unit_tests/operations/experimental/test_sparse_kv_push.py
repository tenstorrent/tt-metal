# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.sparse_push_plan + sparse_kv_push, bit-exact on every chip of an 8x4 torus.

Each cache row encodes its own global row and slot, so reading the compact buffer through the plan's rewritten indices
must return exactly the rows the top-k names. Index patterns: real GLM-5.3 top-k plus synthetic edge cases for the
plan's per-owner runs, packet packing, masking, and the prefix/remainder split.
"""

import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SparseKvPushPart as Part
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import packet_shape

SENTINEL = 0xFFFFFFFF
FILL = 0x5A5A5A5A
N_DEV = 32
TOPK = 2048
TRACE_1M = Path("/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120")
TRACE_1M_ROWS = 1044480


def _device_params():
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 6144
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
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
    per_chip = max_rows // N_DEV
    p = np.arange(max_rows)
    chip, local = p // per_chip, p % per_chip
    global_row = ((local // sc) * N_DEV + chip) * sc + local % sc
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
    need = 0
    for d in range(N_DEV):
        rows = np.unique(idx[d][idx[d] != SENTINEL])
        counts = np.bincount((rows // sc) % N_DEV, minlength=N_DEV)
        need = max(need, int((-(-counts // unit) * unit).sum()))
    return max(need, N_DEV * unit)


class Push:
    """One plan for `idx`, its all-gathered tables, and pushes from `cache` into compact buffers."""

    def __init__(self, mesh_device, cache, idx, *, max_rows, sc, page_bytes, chunk_start, links=2):
        self.mesh_device, self.cache, self.idx = mesh_device, cache, idx
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
        self.start = _scalar(mesh_device, chunk_start)
        self.tables = None
        self.page_bytes = page_bytes

    def plan(self):
        table, self.compact_indices = ttnn.experimental.sparse_push_plan(
            self.indices,
            num_devices=N_DEV,
            stripe_chunk=self.sc,
            max_rows=self.max_rows,
            page_bytes=self.page_bytes,
            chunk_start=self.start,
        )
        _shard_over_mesh(table, self.mesh_device)
        if self.tables is None:
            self.tables = _replicated(self.mesh_device, np.zeros((1, 1, table.shape[2] * N_DEV, table.shape[3])))
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self.tables, cluster_axis=None, num_links=self.links
        )

    def push(self, out, part, **slot):
        return ttnn.experimental.sparse_kv_push(
            self.cache,
            output_tensor=out,
            table=self.tables,
            stripe_chunk=self.sc,
            max_rows=self.max_rows,
            part=part,
            num_links=self.links,
            ready_semaphore=self.sems[0],
            data_valid_semaphore=self.sems[1],
            **slot,
        )


def _shard_over_mesh(tensor, mesh_device):
    """Mark `tensor` as sharded on dim 2 over every chip in row-major order, as the full-mesh gather expects."""
    dist = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    tensor.update_tensor_topology(ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords))


def _check(push, out, *, slot, part, chunk_start, tag):
    """Every referenced row sits at its rewritten slot, distinct rows take distinct slots, and masked indices stay
    masked. PREFIX pushes the blocks before the one holding chunk_start and REST the others, so each leaves the
    other's rows untouched."""
    mesh = push.mesh_device
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).numpy().reshape(N_DEV, -1, push.width)
    compact = ttnn.to_torch(push.compact_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
    compact = (compact.to(torch.int64).numpy() & 0xFFFFFFFF).reshape(push.idx.shape)
    for d in range(N_DEV):
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
def _owner_rows(owner, sc, end):
    """Every row below `end` that `owner` holds."""
    blocks = np.arange(owner, -(-end // sc), N_DEV)
    rows = (blocks[:, None] * sc + np.arange(sc)).reshape(-1)
    return rows[rows < end]


def _lay_out(row_sets, queries, rng):
    """[N, queries, TOPK]: each query takes a window of its chip's rows (so rows repeat across queries, never within
    one), in shuffled order, padded with masked entries."""
    idx = np.full((N_DEV, queries, TOPK), SENTINEL, dtype=np.int64)
    for d, rows in enumerate(row_sets):
        rows = rng.permutation(np.asarray(rows, dtype=np.int64))
        n = min(len(rows), TOPK)
        for q in range(queries):
            if len(rows):
                idx[d, q, :n] = np.roll(rows, -q * 97)[:n]
            idx[d, q, :n] = rng.permutation(idx[d, q, :n])
    return idx


def _pattern(name, rng, *, sc, end, start):
    """Row sets per chip for the synthetic pattern `name`; rows are below `end` (the populated cache)."""
    pick = lambda pool, k: rng.choice(pool, size=min(k, len(pool)), replace=False)
    if name == "random":
        return [pick(np.arange(end), 3000) for _ in range(N_DEV)]
    if name == "clustered":
        centers = rng.integers(0, end, size=(N_DEV, 8))
        return [np.unique((c[:, None] + np.arange(400)) % end) for c in centers]
    if name == "one_owner":
        return [pick(_owner_rows((d + 1) % N_DEV, sc, end), 3000) for d in range(N_DEV)]
    if name == "local_only":
        return [pick(_owner_rows(d, sc, end), 3000) for d in range(N_DEV)]
    if name == "routing_ties":
        # The chip opposite on both rings of the 8x4 torus: both directions are equally short on each axis.
        opposite = lambda d: ((d // 4 + 4) % 8) * 4 + (d % 4 + 2) % 4
        return [pick(_owner_rows(opposite(d), sc, end), 3000) for d in range(N_DEV)]
    if name == "run_lengths":
        # Per-owner counts around one packet (8 rows) and one bank-lane unit (64 rows).
        counts = [1, 7, 8, 9, 63, 64, 65]
        return [
            np.concatenate([pick(_owner_rows(o, sc, end), counts[(o + d) % len(counts)]) for o in range(N_DEV)])
            for d in range(N_DEV)
        ]
    if name == "edges":
        # Block and bitmap-word edges, the reader's 16-block window edge, and the first and last row.
        blocks = np.array([o + N_DEV * b for o in range(N_DEV) for b in (0, 15, 16)])
        rows = (blocks[:, None] * sc + np.array([0, 31, 32, sc - 1])).reshape(-1)
        rows = np.unique(np.concatenate([rows[rows < end], [0, end - 1]]))
        return [rows for _ in range(N_DEV)]
    if name == "duplicates":
        hot = pick(np.arange(end), 50)
        return [hot for _ in range(N_DEV)]
    if name == "remainder_only":
        return [pick(np.arange(start, end), 3000) for _ in range(N_DEV)]
    if name == "prefix_only":
        return [pick(np.arange(start), 3000) for _ in range(N_DEV)]
    if name == "everything":
        return [np.arange(end) for _ in range(N_DEV)]
    if name == "tiny_prefix":
        return [np.arange(min(end, 3 * sc)) for _ in range(N_DEV)]
    raise ValueError(name)


def _mask(idx, rng):
    """Masked tails of varying length, fully masked queries, and one fully masked chip."""
    q, k = idx.shape[1], idx.shape[2]
    for d in range(N_DEV):
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


@galaxy
@pytest.mark.parametrize("pattern, max_rows, offset", PATTERNS, ids=[p[0] for p in PATTERNS])
def test_sparse_kv_push_patterns(mesh_device, pattern, max_rows, offset):
    sc, page_bytes = 160, 704
    rng = np.random.default_rng(zlib.crc32(pattern.encode()))
    start = max_rows - sc * N_DEV + offset
    rows = _pattern("random" if pattern == "unaligned_start" else pattern, rng, sc=sc, end=max_rows, start=start)
    # Masked entries in every pattern; query 0 and the unmasked heads of the others still reference rows.
    idx = _mask(_lay_out(rows, sc, rng), rng)
    cache = _cache(mesh_device, 2, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start)
    _run_parts(mesh_device, push, slot=1, chunk_start=start, tag=pattern, batch_index=1)


# Every pair of values of any two settings appears in some case; the first case is production.
CONFIGS = [
    (704, 160, 2, True),
    (704, 64, 1, False),
    (1152, 160, 1, False),
    (1152, 64, 2, False),
    (1152, 64, 1, True),
]


@galaxy
@pytest.mark.parametrize(
    "page_bytes, sc, links, slot_tensor",
    CONFIGS,
    ids=[f"{p}B-stripe{sc}-{l}link-{'slot_tensor' if t else 'slot_int'}" for p, sc, l, t in CONFIGS],
)
def test_sparse_kv_push_configs(mesh_device, page_bytes, sc, links, slot_tensor):
    max_rows = 92160
    rng = np.random.default_rng(sc * 7 + links)
    start = max_rows - sc * N_DEV
    idx = _mask(_lay_out(_pattern("clustered", rng, sc=sc, end=max_rows, start=start), sc, rng), rng)
    cache = _cache(mesh_device, 4, max_rows, sc, page_bytes // 4)
    push = Push(
        mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start, links=links
    )
    # Slot 3 either way: user 1 of 2 layers, layer 1.
    slot_args = (
        dict(batch_index_tensor=_scalar(mesh_device, 1), batch_slot_num_layers=2, batch_slot_layer_idx=1)
        if slot_tensor
        else dict(batch_index=3)
    )
    _run_parts(mesh_device, push, slot=3, chunk_start=start, tag=f"{page_bytes}B sc{sc} {links}L", **slot_args)


# Large caches use 64 B rows to keep host memory and readback small; production page sizes are covered above.
@galaxy
@pytest.mark.parametrize("layer", [14, 62], ids=["L14", "L62"])
def test_sparse_kv_push_real_topk(mesh_device, layer):
    from safetensors.numpy import load_file

    f = TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
    if not f.exists():
        pytest.skip(f"GLM-5.3 1M top-k not staged: {f}")
    sc, page_bytes = 160, 64
    idx = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(N_DEV, sc, TOPK)
    idx[idx < 0] = SENTINEL
    start = TRACE_1M_ROWS - sc * N_DEV
    cache = _cache(mesh_device, 2, TRACE_1M_ROWS, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=TRACE_1M_ROWS, sc=sc, page_bytes=page_bytes, chunk_start=start)
    _run_parts(mesh_device, push, slot=1, chunk_start=start, tag=f"L{layer}", batch_index=1)


@galaxy
def test_sparse_kv_push_worst_case_union(mesh_device):
    """Every query names 2048 rows no other query names, the most a chip can need: the model's buffer cap
    (query rows x top-k + one unit per owner) must hold it."""
    sc, page_bytes, max_rows = 160, 64, 337920
    distinct = sc * TOPK
    idx = np.stack([np.random.default_rng(d).permutation(distinct).reshape(sc, TOPK) for d in range(N_DEV)])
    start = max_rows - sc * N_DEV
    cache = _cache(mesh_device, 1, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=start)
    unit = np.prod(packet_shape(mesh_device, page_bytes))
    assert push.rows <= distinct + N_DEV * unit, f"{push.rows} rows needed, over the cap {distinct + N_DEV * unit}"
    _run_parts(mesh_device, push, slot=0, chunk_start=start, tag="worst case", batch_index=0)


@galaxy
def test_sparse_kv_push_traced(mesh_device):
    """Plan + table gather + push captured once, replayed with new indices, chunk start and user slot in place."""
    sc, page_bytes, max_rows = 160, 704, 92160
    rng = np.random.default_rng(3)
    first, second = max_rows - sc * N_DEV, max_rows - 2 * sc * N_DEV
    idx_a = _lay_out(_pattern("clustered", rng, sc=sc, end=max_rows, start=first), sc, rng)
    idx_b = _lay_out(_pattern("random", rng, sc=sc, end=second + sc * N_DEV, start=second), sc, rng)
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


@galaxy
@pytest.mark.parametrize(
    "case",
    ["stripe_0", "stripe_48", "stripe_1024", "tile_input", "same_semaphore", "output_below_minimum"],
)
def test_sparse_kv_push_rejects(mesh_device, case, expect_error):
    sc, page_bytes, max_rows = 160, 704, 10240
    rng = np.random.default_rng(0)
    idx = _lay_out(_pattern("random", rng, sc=sc, end=max_rows, start=max_rows - sc * N_DEV), sc, rng)
    cache = _cache(mesh_device, 1, max_rows, sc, page_bytes // 4)
    push = Push(mesh_device, cache, idx, max_rows=max_rows, sc=sc, page_bytes=page_bytes, chunk_start=0)
    if case.startswith("stripe_"):
        push.sc = int(case.split("_")[1])
        with expect_error(RuntimeError, "must be a positive multiple of 32"):
            push.plan()
        return
    push.plan()
    out = _filled_buffer(mesh_device, push.rows, push.width)
    if case == "tile_input":
        push.cache = ttnn.from_torch(
            torch.zeros(1, 1, max_rows, 352, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )
    elif case == "same_semaphore":
        push.sems = [push.sems[0], push.sems[0]]
    elif case == "output_below_minimum":
        out = _filled_buffer(mesh_device, 8, push.width)
    message = {
        "tile_input": "must be a ROW_MAJOR DRAM tensor",
        "same_semaphore": "two semaphores must differ",
        "output_below_minimum": "output needs at least",
    }[case]
    with expect_error(RuntimeError, message):
        push.push(out, Part.ALL, batch_index=0)
