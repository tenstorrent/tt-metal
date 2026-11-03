# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt/verify_kv_write.py on hardware: the DFlash verify's K / V rows written into bf8 paged caches (Laguna-S shapes
per chip: 2 KV heads, head_dim 128, block 32; a shuffled page table) on one chip and on a 1 x 4 mesh (each chip its
own cache contents and rows).

Each case writes B rows at consecutive positions (start 30: the first two rows end tile 0 and the rest go to tile 1;
start 64: the rows start a tile), then a second step of B rows after them (a program-cache hit with new positions), and
compares the caches with torch.equal against
  per_row:      main's ttnn.experimental.paged_update_cache, one call per row in row order (no two rows in flight);
  branch_group: this branch's ttnn.experimental.paged_fused_update_cache(..., share_cache=True) (its group mode);
                skipped, with the op's error, when the op refuses share_cache with a paged cache (main, after the
                paged_cache revert).
Both also check that every cache row the step did not write (including the other rows of the written tiles) kept its
bits, and that the written rows hold the inputs to bf8 precision. test_verify_kv_write_trace replays one captured
write for new positions and rows.

Run: pytest models/demos/laguna/tests/test_verify_kv_write.py"""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.laguna.tt.verify_kv_write import verify_kv_write

NKV, HD, BLOCK, TILE_ROWS = 2, 128, 32, 32  # Laguna-S per chip on p150x4 (8 KV heads / 4 chips)
BLOCKS_PER_USER = 6  # positions 0..191
NUM_USERS = 2
NUM_BLOCKS = NUM_USERS * BLOCKS_PER_USER + 4  # 4 spare blocks no row maps to
P150_GRAPH = (
    Path(__file__).resolve().parents[4] / "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
)


def _open_mesh(cols, trace_region_size=0):
    n = ttnn.get_num_devices()
    if n < cols:
        pytest.skip(f"needs {cols} devices, {n} present")
    if n == 1 and P150_GRAPH.is_file():  # a lone P150 opens through its singleton mesh graph (laguna_test_utils)
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", str(P150_GRAPH))
    return ttnn.open_mesh_device(ttnn.MeshShape(1, cols), trace_region_size=trace_region_size)


def _shard_mem(B, y0):
    """The decoder's K / V row layout (OptimizedDecoder._kv_shard_memcfg): one row per core from core row y0."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, y0), ttnn.CoreCoord((B - 1) % 8, y0 + (B - 1) // 8))})
    return ttnn.create_sharded_memory_config(
        shape=(TILE_ROWS, HD),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class _Case:
    """Device tensors on one mesh: chip d gets cache t[d] and rows t[d]; positions / page tables are replicated."""

    def __init__(self, mesh):
        self.mesh, self.D = mesh, mesh.get_num_devices()
        self.shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)

    def cache(self, t):  # t [D, NUM_BLOCKS, NKV, BLOCK, HD] -> per chip [NUM_BLOCKS, NKV, BLOCK, HD] bf8 DRAM
        return ttnn.from_torch(t.reshape(self.D * NUM_BLOCKS, NKV, BLOCK, HD), dtype=ttnn.bfloat8_b,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=self.shard)  # fmt: skip

    def rows(self, t, mem):  # t [D, B, NKV, HD] -> per chip [1, B, NKV, HD] bf16 in mem
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.shard)  # fmt: skip
        return x if mem == ttnn.DRAM_MEMORY_CONFIG else ttnn.to_memory_config(x, mem)

    def int32(self, t):
        return ttnn.from_torch(t.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.rep)  # fmt: skip

    def host(self, cache):
        t = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        return t.reshape(self.D, NUM_BLOCKS, NKV, BLOCK, HD).float()


def _page_table():
    """[NUM_USERS, BLOCKS_PER_USER]: user u's virtual block j -> a shuffled physical block (spares unused)."""
    g = torch.Generator().manual_seed(7)
    return torch.randperm(NUM_BLOCKS, generator=g)[: NUM_USERS * BLOCKS_PER_USER].reshape(NUM_USERS, -1)


def _rows_layout(B, start, layout, step):
    """(users [B], positions [B]) of verify step `step` (each user's rows follow its previous step's); -1 = skip."""
    users = torch.zeros(B, dtype=torch.int64)
    if layout == "two_users":  # the batched verify: rows of two users, user 1's from position 64
        h = B // 2
        users[h:] = 1
        pos = torch.cat([torch.arange(start + step * h, start + step * h + h),
                         torch.arange(64 + step * (B - h), 64 + step * (B - h) + B - h)])  # fmt: skip
    else:
        pos = torch.arange(start + step * B, start + step * B + B)
        if layout == "skip":  # one row skipped
            pos[1] = -1
    return users, pos


def _reference_write(case, cache, rows_t, users, pos, pt):
    """main's paged_update_cache, one call per row in row order."""
    for b in range(len(pos)):
        if int(pos[b]) < 0:
            continue
        u = int(users[b])
        row = ttnn.to_memory_config(case.rows(rows_t[:, b : b + 1], ttnn.DRAM_MEMORY_CONFIG), _shard_mem(1, 0))
        p = case.int32(pos[b : b + 1])
        pt_b = case.int32(pt[u : u + 1])
        ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=p, page_table=pt_b)
        for t in (row, p, pt_b):
            ttnn.deallocate(t)


def _check_written(new, before, rows_t, users, pos, pt, what):
    """Rows the step did not write keep their bits; written rows hold the inputs to bf8 precision."""
    mask = torch.zeros(NUM_BLOCKS, NKV, BLOCK, HD, dtype=torch.bool)
    for b in range(len(pos)):
        p = int(pos[b])
        if p >= 0:
            mask[int(pt[int(users[b]), p // BLOCK]), :, p % BLOCK, :] = True
    for d in range(new.shape[0]):
        assert torch.equal(new[d][~mask], before[d][~mask]), f"{what}: chip {d} changed rows it was not given"
        for b in range(len(pos)):
            p = int(pos[b])
            if p >= 0:
                got = new[d, int(pt[int(users[b]), p // BLOCK]), :, p % BLOCK, :]
                assert torch.allclose(got, rows_t[d, b].float(), atol=0.1, rtol=0), f"{what}: chip {d} row {b} ({p})"


@pytest.mark.parametrize("against", ["per_row", "branch_group"])
@pytest.mark.parametrize("mesh_cols", [1, 4], ids=["1chip", "1x4"])
@pytest.mark.parametrize(
    "B, start, layout",
    [
        (2, 30, "one_user"),
        (2, 64, "one_user"),
        (6, 30, "one_user"),
        (6, 64, "one_user"),
        (16, 30, "one_user"),
        (16, 64, "one_user"),
        (32, 30, "one_user"),
        (32, 64, "one_user"),
        (6, 30, "skip"),
        (32, 30, "two_users"),
    ],
)
def test_verify_kv_write(against, mesh_cols, B, start, layout):
    mesh = _open_mesh(mesh_cols)
    try:
        case = _Case(mesh)
        D = case.D
        torch.manual_seed(B * 1000 + start)
        pt_users = _page_table()
        init = {kv: torch.randn(D, NUM_BLOCKS, NKV, BLOCK, HD) for kv in "kv"}
        new = {kv: case.cache(init[kv]) for kv in "kv"}
        ref = {kv: case.cache(init[kv]) for kv in "kv"}
        before = {kv: case.host(new[kv]) for kv in "kv"}  # the bf8-quantized initial caches
        for step in range(2):  # step 1: the next positions, a program-cache hit
            users, pos = _rows_layout(B, start, layout, step)
            rows = {kv: torch.randn(D, B, NKV, HD).to(torch.bfloat16) for kv in "kv"}
            cur = case.int32(pos)
            pt_dev = case.int32(pt_users[users])  # [B, BLOCKS_PER_USER]: row b's page-table row
            k_sh, v_sh = case.rows(rows["k"], _shard_mem(B, 0)), case.rows(rows["v"], _shard_mem(B, 4))

            verify_kv_write(new["k"], k_sh, new["v"], v_sh, cur, pt_dev)
            if against == "per_row":
                for kv in "kv":
                    _reference_write(case, ref[kv], rows[kv], users, pos, pt_users)
            else:
                try:
                    ttnn.experimental.paged_fused_update_cache(
                        ref["k"], k_sh, ref["v"], v_sh, update_idxs_tensor=cur, page_table=pt_dev, share_cache=True
                    )
                except RuntimeError as e:
                    if "share_cache" not in str(e):
                        raise
                    pytest.skip(f"paged_fused_update_cache refuses share_cache with a paged cache: {e}".splitlines()[0])

            for kv in "kv":
                got = case.host(new[kv])
                what = f"{kv.upper()} step {step}"
                assert torch.equal(got, case.host(ref[kv])), f"{what}: differs from the {against} reference"
                _check_written(got, before[kv], rows[kv], users, pos, pt_users, what)
                before[kv] = got
            for t in (cur, pt_dev, k_sh, v_sh):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("mesh_cols", [1, 4], ids=["1chip", "1x4"])
def test_verify_kv_write_one_cache_and_shared_cores(mesh_cols):
    """The decoder's chained path (TT_LAGUNA_SEQ_KV_FUSED=0: K and V rows on the same cores, one call per cache), and
    both pairs given on overlapping cores (verify_kv_write then runs one program per cache)."""
    mesh = _open_mesh(mesh_cols)
    try:
        case = _Case(mesh)
        torch.manual_seed(5)
        B, pt_users = 6, _page_table()
        users, pos = _rows_layout(B, 30, "one_user", 0)
        init = {kv: torch.randn(case.D, NUM_BLOCKS, NKV, BLOCK, HD) for kv in "kv"}
        rows = {kv: torch.randn(case.D, B, NKV, HD).to(torch.bfloat16) for kv in "kv"}
        cur, pt_dev = case.int32(pos), case.int32(pt_users[users])
        k_sh, v_sh = case.rows(rows["k"], _shard_mem(B, 0)), case.rows(rows["v"], _shard_mem(B, 0))
        one = {kv: case.cache(init[kv]) for kv in "kv"}
        both = {kv: case.cache(init[kv]) for kv in "kv"}
        ref = {kv: case.cache(init[kv]) for kv in "kv"}
        verify_kv_write(one["k"], k_sh, None, None, cur, pt_dev)
        verify_kv_write(one["v"], v_sh, None, None, cur, pt_dev)
        verify_kv_write(both["k"], k_sh, both["v"], v_sh, cur, pt_dev)
        for kv in "kv":
            _reference_write(case, ref[kv], rows[kv], users, pos, pt_users)
            want = case.host(ref[kv])
            assert torch.equal(case.host(one[kv]), want), f"{kv}: one call per cache"
            assert torch.equal(case.host(both[kv]), want), f"{kv}: K and V on shared cores"
    finally:
        ttnn.close_mesh_device(mesh)


def test_verify_kv_write_trace():
    """One captured verify_kv_write replayed for new positions and rows (copied into the same device tensors): the
    positions and the page table are read on device, so each replay writes its rows where they belong."""
    mesh = _open_mesh(1, trace_region_size=10_000_000)
    try:
        case = _Case(mesh)
        torch.manual_seed(11)
        B, pt_users = 16, _page_table()
        users = torch.zeros(B, dtype=torch.int64)
        init = {kv: torch.randn(1, NUM_BLOCKS, NKV, BLOCK, HD) for kv in "kv"}
        new = {kv: case.cache(init[kv]) for kv in "kv"}
        ref = {kv: case.cache(init[kv]) for kv in "kv"}
        pt_dev = case.int32(pt_users[users])
        steps = [torch.arange(30, 30 + B), torch.arange(46, 46 + B), torch.arange(100, 100 + B)]
        rows = [{kv: torch.randn(1, B, NKV, HD).to(torch.bfloat16) for kv in "kv"} for _ in steps]
        # persistent inputs: the rows in DRAM (resharded inside the trace) and the positions
        k_dram = case.rows(rows[0]["k"], ttnn.DRAM_MEMORY_CONFIG)
        v_dram = case.rows(rows[0]["v"], ttnn.DRAM_MEMORY_CONFIG)
        cur = case.int32(steps[0])

        def body():
            k_sh = ttnn.to_memory_config(k_dram, _shard_mem(B, 0))
            v_sh = ttnn.to_memory_config(v_dram, _shard_mem(B, 4))
            verify_kv_write(new["k"], k_sh, new["v"], v_sh, cur, pt_dev)
            return k_sh, v_sh

        eager = body()  # step 0, eagerly (compiles the programs)
        for kv in "kv":
            _reference_write(case, ref[kv], rows[0][kv], users, steps[0], pt_users)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = body()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for s in (1, 2):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(rows[s]["k"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=case.shard),
                k_dram,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(rows[s]["v"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=case.shard),
                v_dram,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(steps[s].to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=case.rep),
                cur,
            )  # fmt: skip
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            for kv in "kv":
                _reference_write(case, ref[kv], rows[s][kv], users, steps[s], pt_users)
                assert torch.equal(case.host(new[kv]), case.host(ref[kv])), f"{kv}: trace replay {s}"
        ttnn.release_trace(mesh, tid)
        del eager, traced
    finally:
        ttnn.close_mesh_device(mesh)
