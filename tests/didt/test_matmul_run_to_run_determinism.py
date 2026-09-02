# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Run-to-run determinism gate for the failing 2D-mcast MLA matmul.

The largest window carries the failing MLA matmul's config -- per_core_M x per_core_N =
10x12 over the full K -- but is capped at MAX_GRID = 11x10 = 110 co-active cores, the box's
validated safe worker budget. That is 3200x4224, one column short of the true 3200x4608
(which needs the 12th col = 120); the missing column costs a little di/dt swing, not the
fault, which onsets at ~4 co-active cores. Each shard's iteration-0 output is its own
baseline; every later iteration is compared run-to-run with torch.equal on host copies of
all 32 shards. On a box with a marginal die the matmul is not bit-exact run-to-run and this
fails, naming the drifting physical device, its logical/physical core, and the diff count.

The die that drifts is selected per device-open, not by this test. On b06u02 device 14
(shard 21, core (6,8)) and its diagonal-torus neighbour device 23 (shard 18) are both
marginal at a shared corner, and which one drifts varies session to session -- the stock
OpTestBase port (test_core_count_sweep_matmul) and this host-shard port both tip device 14,
yet earlier runs of the same recipe tipped device 23. A single run therefore names only
one die; run repeatedly across fresh opens and treat any named die as a fault. Do not
close the box on a single-die replacement while this gate can still go red.

Only run-to-run drift is nondeterminism and only it asserts. A shard whose stable output
merely differs from shard 0 is a fixed inter-chip offset, not ND -- it is logged, not
failed.

di/dt control: a lone core draws too little current to droop, so the matmul is bit-exact
at a 1x1 window and drifts only once a grid of cores switches beside the marginal one
(dose-response onset ~4 co-active cores). The ramp starts at 1x1 to show that threshold;
test_single_core_matmul is the standalone form of that negative control.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0

EMB_DIM = 7168  # reduction depth K; the accumulate axis where a marginal FPU bit shows
SUSPECT_CORE = (6, 8)  # marginal logical core on b06u02
PER_CORE_M, PER_CORE_N = 10, 12  # output-tile block one core owns in the 3200x4608 MLA matmul
IN0_BLOCK_W = 8  # must divide Kt = EMB_DIM/32 = 224
MAX_GRID = (11, 10)  # cap co-active cores at 110; the device reports 12x10 but 120 is off-limits here

MESH_DEVICE_PARAMS = [pytest.param((8, 4), id="galaxy")]


def _shards(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _plan(grid, suspect):
    """Square windows centred+clamped on the suspect, growing to the full grid.

    Clamping keeps the suspect inside every window, so the same core is exercised at each
    size and only the surrounding co-active-core count changes.
    """
    gx, gy = grid
    tx, ty = suspect
    sides = [s for s in (1, 2, 3, 4, 5, 6, 8) if s <= min(gx, gy)]
    plan = [(min(max(tx - (s - 1) // 2, 0), gx - s), min(max(ty - (s - 1) // 2, 0), gy - s), s, s) for s in sides]
    plan.append((0, 0, gx, gy))
    return plan


@skip_for_wormhole_b0("Grid position (6,8) and the fault under test are Blackhole-only")
@pytest.mark.parametrize("mesh_device", MESH_DEVICE_PARAMS, indirect=["mesh_device"])
def test_matmul_run_to_run_determinism(mesh_device):
    iters = int(os.environ.get("TT_DET_SWEEP_ITERS", "10"))
    tx, ty = SUSPECT_CORE
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    gx_max, gy_max = min(grid.x, MAX_GRID[0]), min(grid.y, MAX_GRID[1])  # clamp to the 110-core budget
    ids = list(mesh_device.get_device_ids())  # shard index -> physical device id (not identity)

    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    # A program_config forces the matmul to LoFi internally; pin LoFi so the recipe carries its own
    # fidelity rather than depending on that default. The drift is di/dt and shows at HiFi2 too --
    # fidelity scales the magnitude, it does not select the die -- so LoFi here is only faithfulness.
    compute_config = ComputeConfigClass(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def window(ox, oy, gx, gy):
        """Run `iters` matmuls on the gx x gy window at (ox,oy); return run-to-run and vs-shard0 hits."""
        seq, hidden = gy * PER_CORE_M * 32, gx * PER_CORE_N * 32
        x, w = (
            ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for t in (torch.randn(1, 1, seq, EMB_DIM) * 0.02, torch.randn(EMB_DIM, hidden) * 0.02)
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=IN0_BLOCK_W,
            out_subblock_h=2,  # h*w must fit the 8-tile dest register
            out_subblock_w=4,
            per_core_M=PER_CORE_M,
            per_core_N=PER_CORE_N,
            transpose_mcast=False,
            fused_activation=None,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + gx - 1, oy + gy - 1))}
            ),
        )

        def blame(a, b):
            """Logical cores of this window owning the differing elements. transpose_mcast=False, so
            the M block index is the grid y and the N block index is the grid x."""
            return {
                (ox + int(t[-1]) // 32 // PER_CORE_N, oy + int(t[-2]) // 32 // PER_CORE_M) for t in (a != b).nonzero()
            }

        # A shard drifting run-to-run (its own output changes between iters) is a marginal die; a shard
        # whose stable output merely differs from shard 0 is not. Keep the two counts apart -- merging
        # them lets a fixed inter-chip offset masquerade as nondeterminism.
        r2r, vs0, baseline = {}, {}, None
        for i in range(iters):
            out = ttnn.matmul(x, w, program_config=program_config, compute_kernel_config=compute_config)
            ttnn.synchronize_device(mesh_device)
            cur = _shards(out)
            ttnn.deallocate(out)
            for c in range(1, len(cur)):
                if not torch.equal(cur[0], cur[c]):
                    n = int((cur[0] != cur[c]).sum())
                    per = vs0.setdefault(c, {})
                    for core in blame(cur[0], cur[c]):
                        per[core] = per.get(core, 0) + n
            if i == 0:
                baseline = cur
            else:
                for c in range(len(cur)):
                    if not torch.equal(baseline[c], cur[c]):
                        n = int((baseline[c] != cur[c]).sum())
                        per = r2r.setdefault(c, {})
                        for core in blame(baseline[c], cur[c]):
                            per[core] = per.get(core, 0) + n
        ttnn.deallocate(x)
        ttnn.deallocate(w)
        return r2r, vs0

    logger.info(
        f"device grid {grid.x}x{grid.y}, capped to {gx_max}x{gy_max}={gx_max * gy_max} cores; "
        f"{PER_CORE_M}x{PER_CORE_N} tiles/core over K={EMB_DIM}, {iters} iters/window; suspect logical {SUSPECT_CORE}"
    )

    failures = {}  # (shard, logical core) -> total run-to-run differing elements
    for ox, oy, gx, gy in _plan((gx_max, gy_max), SUSPECT_CORE):
        r2r, vs0 = window(ox, oy, gx, gy)
        for c, per_core in r2r.items():
            for core, ndiff in per_core.items():
                failures[(c, core)] = failures.get((c, core), 0) + ndiff
                phys = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
                logger.warning(
                    f"  cores={gx * gy:>3} ({gx}x{gy} at {ox},{oy}): shard {c} (device id {ids[c]}), "
                    f"logical core {core} -> physical ({phys.x},{phys.y}): run-to-run NOT deterministic, {ndiff} elements"
                )
        if vs0:
            logger.info(f"  cores={gx * gy:>3}: stable vs-shard0 offsets on shards {sorted(vs0)} (not ND)")
        if not r2r:
            logger.info(f"  cores={gx * gy:>3} ({gx}x{gy} at {ox},{oy}): all 32 shards bit-exact run-to-run")

    assert not failures, "matmul is nondeterministic run-to-run (di/dt marginal die): " + "; ".join(
        f"shard {c} device id {ids[c]} core {core}: {ndiff} elements" for (c, core), ndiff in sorted(failures.items())
    )
    logger.success(f"all 32 shards bit-exact run-to-run across every window, {iters} iters each")
