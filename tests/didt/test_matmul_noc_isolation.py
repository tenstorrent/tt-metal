# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""NoC-isolation probe for the marginal-die matmul drift on b06u02.

Settles one question: is the run-to-run drift on device 14 / core (6,8) a NoC
race, or in-core compute drooping under di/dt? Three dataflows run the SAME
per-core math -- one output tile [32,32] = sum over K=7168 of [32,K]x[K,32] --
on the same co-active-core windows, differing only in how operands move:

  C1  2D mcast, in0_block_w=8  -> num_blocks=28: operands stream from DRAM and
      multicast core-to-core over NoC; K accumulates across 28 packer-L1-acc
      blocks. This is the failing MLA-matmul dataflow -- the repro.
  C2  2D mcast, in0_block_w=Kt -> num_blocks=1: still NoC/mcast, but all of K
      arrives in one block so there is no cross-block packer-L1 accumulate.
      Holds NoC, drops num_blocks. Expected to OOM at K=7168 (in0 CB = 224
      tiles x double-buffer); best-effort, skipped if it will not fit.
  C3  L1-sharded reuse (MatmulMultiCoreReuseProgramConfig): in0/in1/output all
      L1-resident per core, no mcast, no DRAM round-trip -- ZERO NoC during the
      timed compute. Each core owns one independent [32,K]x[K,32]; there is no
      inter-core data at all, so no race is possible. Sharding forces
      in0_block_w=Kt -> num_blocks=1 (same confound as C2).

Prior work removed DRAM traffic (L1-resident operands) and still drifted, but
that path kept inter-core mcast. C3 is the first config with no NoC of any kind.

Decode, reading drift on core (6,8):
  C3 drifts           -> in-core FPU accumulate droops under di/dt; NoC is not
                         necessary for the fault. Not a race.
  C3 clean, C1 drifts -> NoC or num_blocks matters. C2 disambiguates:
     C2 clean too     -> the multi-block packer-L1-acc path is the trigger
                         (num_blocks), not NoC per se.
     C2 drifts        -> the mcast burst is the trigger (NoC).

The C2/C3 confound (both collapse to num_blocks=1) is why C1 -> C2 -> C3 are
read as a chain, not C1 vs C3 alone.

Diagnostic probe, not a gate: it reports per-cell drift and does not assert on
it. If C1 is itself bit-exact the box is healthy this session and the whole
comparison is moot -- that is logged, not failed.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0

EMB_DIM = 7168  # reduction depth K; the accumulate axis where a marginal FPU bit shows
KT = EMB_DIM // 32  # 224 tiles; the sharded path forces in0_block_w == KT (num_blocks=1)
SUSPECT_CORE = (6, 8)  # marginal logical core on b06u02
MAX_GRID = (11, 10)  # cap co-active cores at 110; the device reports 12x10 but 120 is off-limits here

MESH_DEVICE_PARAMS = [pytest.param((8, 4), id="galaxy")]


def _shards(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _crs(ox, oy, gx, gy):
    """Rectangular CoreRangeSet of size gx*gy anchored at (ox,oy) -- the co-active-core window."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + gx - 1, oy + gy - 1))})


@skip_for_wormhole_b0("Grid position (6,8) and the fault under test are Blackhole-only")
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_matmul_noc_isolation(mesh_device):
    iters = int(os.environ.get("TT_DET_NOC_ITERS", "10"))
    tx, ty = SUSPECT_CORE
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    gx_max, gy_max = min(grid.x, MAX_GRID[0]), min(grid.y, MAX_GRID[1])  # clamp to the 110-core budget
    ids = list(mesh_device.get_device_ids())  # shard index -> physical device id (not identity)

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    # Match the repro: LoFi, dest not fp32 -- the low-precision accumulate is where the marginal bit
    # shows. Fidelity scales the drift magnitude, it does not select the die, so this is faithfulness
    # to the failing config, not a knob under test. Held identical across all three cells.
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def _dram(t, dtype):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def build_mcast(ox, oy, gx, gy, in0_block_w):
        """C1/C2: one [gy*32, K] x [K, gx*32] matmul partitioned 1 tile/core, operands mcast from DRAM."""
        x = _dram(torch.randn(1, 1, gy * 32, EMB_DIM) * 0.02, ttnn.bfloat16)
        w = _dram(torch.randn(EMB_DIM, gx * 32) * 0.02, ttnn.bfloat8_b)  # bf16 x bf8 == the real MLA matmul
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
            allowed_worker_cores=_crs(ox, oy, gx, gy),
        )
        return x, w, pc, None  # None -> default DRAM output (a NoC write, fine: these are the NoC cells)

    def build_sharded(ox, oy, gx, gy):
        """C3: G=gx*gy independent [32,K]x[K,32], all operands + output L1-sharded on the window. Zero NoC."""
        g = gx * gy
        crs = _crs(ox, oy, gx, gy)

        def l1_sharded(shard_h, shard_w):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
            )

        x = ttnn.from_torch(
            torch.randn(g, 1, 32, EMB_DIM) * 0.02,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=l1_sharded(32, EMB_DIM),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        w = ttnn.from_torch(
            torch.randn(g, 1, EMB_DIM, 32) * 0.02,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,  # bf16 x bf8 == the real MLA matmul; also keeps the shard inside L1 at K=7168
            layout=ttnn.TILE_LAYOUT,
            memory_config=l1_sharded(EMB_DIM, 32),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),  # size only; placement is the shard grid
            in0_block_w=KT,  # forced: in0 shard holds the full K, so one block (no cross-block L1-acc)
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )
        return x, w, pc, l1_sharded(32, 32)

    def blame_mcast(ox, oy, gx, a, b):
        """transpose_mcast=False, 1 tile/core: N-block index is grid x, M-block index is grid y."""
        return {(ox + int(t[-1]) // 32, oy + int(t[-2]) // 32) for t in (a != b).nonzero()}

    def blame_sharded(ox, oy, gx, a, b):
        """Output is [G,1,32,32]; batch index g is the g-th core of the window, row-major."""
        return {(ox + int(t[0]) % gx, oy + int(t[0]) // gx) for t in (a != b).nonzero()}

    def probe(build, blame, ox, oy, gx, gy):
        """Run `iters` matmuls; return run-to-run drift {shard: {core: ndiff}} and vs-shard0 offsets."""
        x, w, pc, out_mc = build()
        r2r, vs0, baseline = {}, {}, None
        for i in range(iters):
            out = ttnn.matmul(x, w, program_config=pc, compute_kernel_config=compute_config, memory_config=out_mc)
            ttnn.synchronize_device(mesh_device)
            cur = _shards(out)
            ttnn.deallocate(out)
            for c in range(1, len(cur)):
                if not torch.equal(cur[0], cur[c]):
                    n = int((cur[0] != cur[c]).sum())
                    per = vs0.setdefault(c, {})
                    for core in blame(ox, oy, gx, cur[0], cur[c]):
                        per[core] = per.get(core, 0) + n
            if i == 0:
                baseline = cur
            else:
                for c in range(len(cur)):
                    if not torch.equal(baseline[c], cur[c]):
                        n = int((baseline[c] != cur[c]).sum())
                        per = r2r.setdefault(c, {})
                        for core in blame(ox, oy, gx, baseline[c], cur[c]):
                            per[core] = per.get(core, 0) + n
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return r2r, vs0

    # (2,2) at the suspect isolates the "4 cores is too little for di/dt" claim; the full 110-core
    # window is the strongest di/dt stress the box budget allows. Override with TT_DET_NOC_WINDOWS.
    sx = min(max(tx - 1, 0), gx_max - 2)
    sy = min(max(ty - 1, 0), gy_max - 2)
    windows = [(sx, sy, 2, 2), (0, 0, gx_max, gy_max)]

    cells = [
        ("C1 mcast   nb=28", lambda ox, oy, gx, gy: build_mcast(ox, oy, gx, gy, 8), blame_mcast, True),
        ("C2 mcast   nb=1 ", lambda ox, oy, gx, gy: build_mcast(ox, oy, gx, gy, KT), blame_mcast, False),
        ("C3 sharded nb=1 ", build_sharded, blame_sharded, True),
    ]

    logger.info(
        f"NoC-isolation probe: device grid {grid.x}x{grid.y}, capped {gx_max}x{gy_max}; "
        f"K={EMB_DIM}, 1 tile/core, {iters} iters/cell; suspect logical {SUSPECT_CORE}. "
        f"C1=mcast/DRAM/28-block, C2=mcast/DRAM/1-block, C3=L1-sharded/zero-NoC/1-block"
    )

    for ox, oy, gx, gy in windows:
        logger.info(f"window {gx}x{gy} at ({ox},{oy}) = {gx * gy} co-active cores")
        for name, build, blame, required in cells:
            try:
                r2r, vs0 = probe(lambda: build(ox, oy, gx, gy), blame, ox, oy, gx, gy)
            except Exception as e:  # noqa: BLE001 -- C2 is expected not to fit L1 at K=7168; skip, don't fail
                if required:
                    raise
                logger.warning(f"  [{name}] skipped: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
                continue

            if not r2r:
                logger.success(f"  [{name}] all 32 shards bit-exact run-to-run")
            for c, per_core in sorted(r2r.items()):
                for core, ndiff in sorted(per_core.items()):
                    phys = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
                    suspect = " <-- SUSPECT" if core == SUSPECT_CORE else ""
                    logger.warning(
                        f"  [{name}] DRIFT shard {c} (device id {ids[c]}), logical {core} -> "
                        f"physical ({phys.x},{phys.y}): {ndiff} elements{suspect}"
                    )
            if vs0:
                logger.info(f"  [{name}] stable vs-shard0 offsets on shards {sorted(vs0)} (fixed inter-chip, not ND)")

    logger.info(
        "read the decode in the module docstring: C3 drift => not a NoC race (in-core di/dt); "
        "C3 clean while C1 drifts => NoC/num_blocks, disambiguated by C2"
    )
