# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DS41-216 stress: gathers of a window ring over the whole mesh, every chip's replica checked (device;
run_safe_pytest.sh, the spec's mesh).

Each iteration copies one of K per-chip sources into a persistent ring ([1, 1, R, 512] bf16, R = 192 ROW_MAJOR: the
snapshot's ring; R = 128 TILE: _ring_carry's carry) and gathers it over the whole mesh (chip-linear r TP + c, chip q's
rows at [q R, (q + 1) R)) with the variant's gathers. Every element is an exact small integer: column 0 the source
chip, 1 the row, 2 the version (iteration mod K), so a wrong row says where it came from. Right after the gather (no
sync) a consumer slices columns [0, 32) of the output (what the snapshot's slices do); after every batch of B
iterations the mesh is synchronized and the same slice is taken again ("late"). Per iteration and chip: the consumer's
copy and the late read vs the expected rows. consumer wrong + late right = the gather returned before its data landed;
both wrong = the data never landed (or was overwritten).

``even``: no busy work; ``skew``: the chips reach the gather at different times (_Skew). Variants (each gather into a
fresh output unless noted; the last three are known-bad repros, xfail):
  hb_shared   ttnn.experimental.high_bw_all_gather TP then SP, ONE semaphore pair for both axes (the model's tt/ccl.py
              before the DS41-216 fix: one pair a mesh)
  hb_axis     the same with one pair per axis (tt/ccl.py after the fix)
  model       tt/ccl.all_gather TP then SP (whatever the model ships; CCL_IMPL of the settings)
  stock_axis  ttnn.all_gather TP then SP
  stock_persist  ttnn.all_gather TP then SP into persistent outputs (P.42's _snapshot_rows; no init barrier)
  hb_mesh     one high_bw_all_gather over the whole mesh (cluster_axis None), its own semaphore pair
  stock_mesh  one ttnn.all_gather over the whole mesh (cluster_axis None); stock_mesh_persist: into a persistent output

8x4, 512 iterations, skew (2026-11-07): model / hb_axis / hb_mesh / stock_axis / stock_persist 0 wrong replicas;
hb_shared 2048 (rm192) and 504 (tile128) wrong, also after the sync, whole mesh columns; stock_mesh rm192 401 wrong at
the consumer, right after the sync (tile128 0). ~5 min the whole file.
"""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize

DRAM = ttnn.DRAM_MEMORY_CONFIG
WIDTH = 512
K = 8  # source versions
ITERS = 512  # iterations a variant
SP_AXIS, TP_AXIS = 0, 1


def _pattern(sp, tp, rows, v):
    """[sp, tp, rows, 512] bf16: chip q = r tp + c, row i: col 0 q, col 1 i, col 2 v, the rest (q + i + v + j) % 128."""
    q = torch.arange(sp * tp).view(sp, tp, 1, 1).float()
    i = torch.arange(rows).view(1, 1, rows, 1).float()
    j = torch.arange(WIDTH).view(1, 1, 1, WIDTH).float()
    t = (q + i + v + j) % 128
    t = t.expand(sp, tp, rows, WIDTH).clone()
    t[..., 0] = q[..., 0]
    t[..., 1] = i[..., 0]
    t[..., 2] = float(v)
    return t.to(torch.bfloat16)


class _Gather:
    def __init__(self, mesh, variant, rows, layout):
        self.mesh, self.variant, self.rows, self.layout = mesh, variant, rows, layout
        self.sp, self.tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        sem = lambda: ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)  # noqa: E731
        if variant == "hb_shared":
            pair = (sem(), sem())
            self.sems = {TP_AXIS: pair, SP_AXIS: pair}
        elif variant == "hb_mesh":
            self.sems = {None: (sem(), sem())}
        elif variant == "hb_axis":
            self.sems = {TP_AXIS: (sem(), sem()), SP_AXIS: (sem(), sem())}
        self.persist = None
        if variant == "stock_mesh_persist":
            self.persist = {None: self._alloc(self.sp * self.tp * rows)}
        if variant == "stock_persist":
            self.persist = {
                TP_AXIS: self._alloc(self.tp * rows),
                SP_AXIS: self._alloc(self.sp * self.tp * rows),
            }

    def _alloc(self, rows):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, WIDTH]), ttnn.bfloat16, self.layout, self.mesh, DRAM
        )

    def __call__(self, src):
        """-> (the whole-mesh gather, [tensors to free after the check])"""
        v, out, tmp = self.variant, src, []
        if v in ("stock_mesh", "stock_mesh_persist", "hb_mesh"):
            rows = int(src.shape[-2]) * self.sp * self.tp
            if v == "stock_mesh":
                dst = ttnn.all_gather(src, dim=2, memory_config=DRAM)
                return dst, [dst]
            if v == "hb_mesh":
                s0, s1 = self.sems[None]
                dst = self._alloc(rows)
                ttnn.experimental.high_bw_all_gather(
                    src, 2, dst, cluster_axis=None, ready_semaphore=s0, data_valid_semaphore=s1
                )
                return dst, [dst]
            dst = self.persist[None]
            ttnn.all_gather(src, dim=2, memory_config=DRAM, output_tensor=dst)
            return dst, []
        for axis in (TP_AXIS, SP_AXIS):
            g = self.mesh.shape[axis]
            if g == 1:
                continue
            rows = int(out.shape[-2]) * g
            if v in ("hb_shared", "hb_axis"):
                s0, s1 = self.sems[axis]
                dst = self._alloc(rows)
                ttnn.experimental.high_bw_all_gather(
                    out, 2, dst, cluster_axis=axis, ready_semaphore=s0, data_valid_semaphore=s1
                )
            elif v == "model":
                from models.demos.deepseek_v41_flash_d_p.tt import ccl

                dst = ccl.all_gather(out, 2, axis)
            elif v == "stock_axis":
                dst = ttnn.all_gather(out, dim=2, cluster_axis=axis, memory_config=DRAM)
            elif v == "stock_persist":
                dst = self.persist[axis]
                ttnn.all_gather(out, dim=2, cluster_axis=axis, memory_config=DRAM, output_tensor=dst)
            else:
                raise ValueError(v)
            if out is not src and self.persist is None:
                tmp.append(out)
            out = dst
        if self.persist is None:
            tmp.append(out)
        return out, tmp


def _cols(t, rows):
    """columns [0, 32) of every row, RM (the consumer)."""
    if t.layout == ttnn.TILE_LAYOUT:
        s = ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, 32], memory_config=DRAM)
        r = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(s)
        return r
    return ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, 32], memory_config=DRAM)


def _host(t, rows):
    return [ttnn.to_torch(d).reshape(rows, 32)[:, :3].to(torch.int32) for d in ttnn.get_device_tensors(t)]


class _Skew:
    """Per-chip busy work before a gather, so the chips reach it at different times, as in the model (SP rows hold
    different tokens: data-dependent runtimes). One ttnn.sparse_matmul over the whole mesh whose per-chip sparsity
    (the experts it computes) differs: one random mesh row, one random mesh column or a random fifth of the chips
    dense (16 experts of [32, 2048] x [2048, 2048]), the rest one expert. S precomputed patterns, cycled at random. (No submesh: a parent mesh's and a submesh's
    workloads must not interleave.)"""

    E, S = 16, 24

    def __init__(self, mesh, seed=7):
        sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
        gen = torch.Generator().manual_seed(seed)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.a = ttnn.from_torch(
            torch.randn(1, 1, 32, 2048, generator=gen),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        self.b = ttnn.from_torch(
            torch.randn(1, self.E, 2048, 2048, generator=gen),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1))
        self.pats = []
        for k in range(self.S):
            sp_mask = torch.zeros(sp, tp, 1, self.E)
            sp_mask[..., 0] = 1.0
            if k % 3 == 0:
                sp_mask[int(torch.randint(0, sp, (1,), generator=gen))] = 1.0
            elif k % 3 == 1:
                sp_mask[:, int(torch.randint(0, tp, (1,), generator=gen))] = 1.0
            else:  # random single chips: a ring waits on a late member while its neighbours' rings are done
                pick = torch.rand(sp, tp, generator=gen) < 0.2
                sp_mask[pick] = 1.0
            self.pats.append(
                ttnn.from_torch(
                    sp_mask.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=mapper,
                )
            )
        self.pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=2048 // 32 // 16,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        self.gen = gen

    def __call__(self):
        k = int(torch.randint(0, self.S, (1,), generator=self.gen))
        out = ttnn.sparse_matmul(
            self.a,
            self.b,
            sparsity=self.pats[k],
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=DRAM,
            program_config=self.pc,
        )
        ttnn.deallocate(out)


# The gathers that must be exact; and the known-bad ones, kept as repros (expected to show wrong replicas: a pass means
# the stress no longer sees the failure, or upstream fixed it)
_REPRO = {
    "hb_shared": "DS41-216: one high_bw semaphore pair for both mesh axes (the model's tt/ccl.py before the fix)",
    "stock_mesh": "DS41-216: stock ttnn.all_gather over the whole mesh (cluster_axis None), ROW_MAJOR: returns before "
    "the rows of the chips 4 mesh rows away land",
    "stock_mesh_persist": "DS41-216: as stock_mesh, into a persistent output",
}
_VARIANTS = [
    pytest.param(
        v,
        marks=[pytest.mark.xfail(reason=_REPRO[v], strict=False)] if v in _REPRO else [],
    )
    for v in (
        "model",
        "hb_axis",
        "hb_mesh",
        "stock_axis",
        "stock_persist",
        "hb_shared",
        "stock_mesh",
        "stock_mesh_persist",
    )
]


@pytest.mark.parametrize("skew", ["even", "skew"])
@pytest.mark.parametrize("layout", ["rm192", "tile128"])
@pytest.mark.parametrize("variant", _VARIANTS)
@mesh_parametrize(timeout_s=0)
def test_gather_stress(mesh_device, variant, layout, skew):
    if mesh_device is None:
        pytest.skip("device test")
    mesh = mesh_device
    busy = _Skew(mesh) if skew == "skew" else None
    iters = ITERS
    batch = 16
    rows, lay = (192, ttnn.ROW_MAJOR_LAYOUT) if layout == "rm192" else (128, ttnn.TILE_LAYOUT)
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    n = sp * tp
    if variant == "hb_mesh":  # (a whole-mesh gather takes a tensor sharded over the gather dim across all chips)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=2)
        host = lambda v: _pattern(sp, tp, rows, v).reshape(1, 1, n * rows, WIDTH)  # noqa: E731
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1))
        host = lambda v: _pattern(sp, tp, rows, v)  # noqa: E731
    up = lambda v: ttnn.from_torch(  # noqa: E731
        host(v), dtype=ttnn.bfloat16, layout=lay, device=mesh, memory_config=DRAM, mesh_mapper=mapper
    )
    srcs = [up(v) for v in range(K)]
    ring = up(0)
    gather = _Gather(mesh, variant, rows, lay)
    want = {}
    for v in range(K):
        q = torch.arange(n).repeat_interleave(rows)
        i = torch.arange(rows).repeat(n)
        want[v] = torch.stack([q, i, torch.full_like(q, v)], dim=1).to(torch.int32)
    total = n * rows
    bad_consumer, bad_late, detail = 0, 0, []
    per_chip = torch.zeros(n, dtype=torch.int64)
    it = 0
    while it < iters:
        kept = []
        for _ in range(min(batch, iters - it)):
            v = it % K
            ttnn.copy(srcs[v], ring)
            if busy is not None:
                busy()
            out, tmp = gather(ring)
            kept.append((it, v, _cols(out, total), out, tmp))
            it += 1
        ttnn.synchronize_device(mesh)
        for k, v, cons, out, tmp in kept:
            # (a persistent output holds the batch's last iteration only: the late read checks that one)
            last = k == kept[-1][0] or variant not in ("stock_persist", "stock_mesh_persist")
            late = _cols(out, total)
            gc, gl = _host(cons, total), _host(late, total)
            for q in range(n):
                ec = (gc[q] != want[v]).any(dim=1)
                el = (gl[q] != want[v]).any(dim=1) if last else torch.zeros(1, dtype=torch.bool)
                if ec.any():
                    bad_consumer += 1
                    per_chip[q] += 1
                    rws = torch.nonzero(ec).flatten()
                    srcq = sorted(set((rws // rows).tolist()))
                    found = gc[q][rws[0]].tolist()
                    if len(detail) < 40:
                        detail.append(
                            f"iter {k} v {v} chip {q} (r {q // tp} c {q % tp}): consumer {int(ec.sum())} rows wrong, "
                            f"from source chips {srcq}, first row {int(rws[0])} holds (chip, row, v) {found}; "
                            f"late read {'right' if not el.any() else f'{int(el.sum())} rows wrong'}"
                        )
                if el.any():
                    bad_late += 1
            ttnn.deallocate(cons)
            ttnn.deallocate(late)
            for t in tmp:
                ttnn.deallocate(t)
    print(
        f"[gather stress] {variant} {layout} {skew}: {iters} iterations x {n} chips: consumer wrong {bad_consumer}, "
        f"late (after sync) wrong {bad_late}; per chip {per_chip.tolist()}"
    )
    for d in detail:
        print("  " + d)
    assert (
        bad_consumer == 0 and bad_late == 0
    ), f"{variant} {layout}: {bad_consumer} consumer / {bad_late} late replicas wrong"
