# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# COPY THIS FILE TO (relative to the tt-metal repo root):
#     tests/nightly/tg/ccl/test_rms_fused_vs_decompose_pcc.py
#
# In this tt-mlir checkout that is:
#     third_party/tt-metal/src/tt-metal/tests/nightly/tg/ccl/test_rms_fused_vs_decompose_pcc.py
#
# It must live under the tt-metal test tree so its imports
# (models.common.*, tests.tt_eager.*) resolve. It will NOT run from the repo
# root. Run on a galaxy (TG / 8x4):
#     source env/activate
#     export ARCH_NAME=wormhole_b0
#     export PYTHONPATH="${TT_METAL_HOME}/tools:${PYTHONPATH}"
#     pytest -svv tests/nightly/tg/ccl/test_rms_fused_vs_decompose_pcc.py
# =============================================================================

"""
Head-to-head PCC comparison: fused distributed RMSNorm (`ttnn.fused_rms_minimal`,
the rms_allgather kernel) vs the decomposed primitives (`ttnn.rms_norm_pre_all_gather`
+ `ttnn.experimental.all_gather_async` + `ttnn.rms_norm_post_all_gather`).

Purpose: prove, with a pure-ttnn repro (no tt-mlir), that on IDENTICAL inputs and
config the fused kernel is less accurate than the decomposed primitives — measured
against the same fp32 torch golden. Config is copied verbatim from
test_distributed_rms_norm_decode_configs.py (DeepSeek-V3 / Kimi decode:
8-device cluster axis, hidden 7168 -> 896/device = 28 width tiles on a 4x7 shard,
bf16, eps=1e-6).

Two measurements:
  (1) SINGLE-OP head-to-head: for each of N random inputs, PCC(fused, golden) vs
      PCC(decompose, golden). This is the direct proof.
  (2) DEPTH-COMPOUNDING loop: apply each method M times with a residual add between
      steps (mimicking the ~122 norm applications in a 61-layer decode), showing the
      per-op gap compounding into the full-model gap (~0.996 fused vs ~0.9998 decompose).

Run on a galaxy (TG / 8x4) exactly like the reference test:
  source env/activate
  export ARCH_NAME=wormhole_b0
  export PYTHONPATH="${TT_METAL_HOME}/tools:${PYTHONPATH}"
  pytest -svv third_party/tt-metal/src/tt-metal/tests/nightly/tg/ccl/test_rms_fused_vs_decompose_pcc.py
"""

import torch
import pytest
from loguru import logger
import ttnn

from models.common.utility_functions import skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def get_torch_rms(x, gamma, eps):
    """fp32 RMSNorm golden over the last dim (matches the reference test)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()


def _pcc_value(tt_torch, golden):
    """Pearson correlation (PCC) computed directly — robust, no string parsing.

    (comp_pcc's message is 'Max ATOL Delta: X, Max RTOL Delta: Y, PCC: Z'; naive
    first-float parsing grabs the ATOL delta, not the PCC. Compute it ourselves.)
    """
    a = tt_torch.flatten().to(torch.float32)
    b = golden.flatten().to(torch.float32)
    if a.numel() != b.numel():
        return float("nan")
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    if a.numel() < 2:
        return float("nan")
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    if denom == 0:
        return float("nan")
    return float((va @ vb) / denom)


def _build_common(mesh_device, num_devices, batch_size_per_row, seq_len, hidden_size, input_shard_grid):
    """Replicates the sharding/config setup from the reference test."""
    ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 7))})
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    num_cores = input_shard_grid.num_cores()
    total_cores = num_cores * num_devices
    shard_width_per_core = ttnn.core.roundup(hidden_size // total_cores, ttnn.TILE_SIZE)
    shard_height = ttnn.core.roundup(batch_size_per_row, ttnn.TILE_SIZE)

    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width_per_core),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        subblock_w=1,
        block_h=1,
        block_w=shard_width_per_core // ttnn.TILE_SIZE,
        inplace=False,
    )
    # Persistent stats buffer for the FUSED op (single-core width shard -> padded[-1] = num_devices*32).
    ag_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_stats = ttnn.from_torch(
        torch.zeros([1, 1, 32, num_devices], dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ag_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
    )
    return {
        "ccl_sub_device_crs": ccl_sub_device_crs,
        "worker_sub_device_id": worker_sub_device_id,
        "input_memory_config": input_memory_config,
        "layer_norm_config": layer_norm_config,
        "tt_stats": tt_stats,
        "shard_width_per_core": shard_width_per_core,
    }


def _make_stats(mesh_device, num_devices, dtype):
    """Persistent stats scratch for the fused op. Its dtype MUST track fp32_dest_acc_en:
    with fp32 acc the kernel CB is fp32-width (4096B); a bf16 stats bank (2048B) overflows.
    tt-mlir sizes this scratch the same way (TTNNOps.cpp derives it from fp32_dest_acc_en)."""
    ag_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(
        torch.zeros([1, 1, 32, num_devices], dtype=torch_dtype),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ag_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
    )


def _make_inputs(mesh_device, num_devices, input_shape, hidden_size, input_memory_config):
    x_torch = torch.randn(input_shape)
    g_torch = torch.randn((1, 1, 1, hidden_size))
    x_tt = ttnn.as_tensor(
        x_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )
    g_tt = ttnn.as_tensor(
        g_torch.reshape([1, 1, hidden_size // 32, 32]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(2, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x_torch, g_torch, x_tt, g_tt


def _to_torch(mesh_device, tt_out, num_devices):
    return ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(num_devices, 1))
    )[0].unsqueeze(0)


def _run_fused(mesh_device, x_tt, g_tt, cfg, sem, epsilon, compute_kernel_config=None, stats=None):
    return ttnn.fused_rms_minimal(
        x_tt,
        cfg["layer_norm_config"],
        0,  # cluster_axis
        mesh_device,
        sem,
        topology=ttnn.Topology.Linear,
        memory_config=cfg["input_memory_config"],
        epsilon=epsilon,
        dtype=ttnn.bfloat16,
        weight=g_tt,
        residual_input_tensor=None,
        stats=stats if stats is not None else cfg["tt_stats"],
        use_noc1_only=False,
        compute_kernel_config=compute_kernel_config,  # None => device default (approx=T, fp32=F, l1=F)
    )


def _run_decompose(mesh_device, x_tt, g_tt, cfg, sems, epsilon):
    # The decompose primitives run on an INTERLEAVED input. The width-sharded L1 layout
    # is a fused-kernel requirement, not a decompose one (tt-mlir's decompose path does
    # not width-shard). Same VALUES as the fused input, just interleaved -> DRAM.
    x_il = ttnn.to_memory_config(x_tt, ttnn.DRAM_MEMORY_CONFIG)
    # Same device-default compute config as fused (pass None) so the ONLY variable is the kernel.
    stats = ttnn.rms_norm_pre_all_gather(x_il, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(mesh_device)
    gathered = ttnn.experimental.all_gather_async(
        stats,
        dim=3,
        cluster_axis=0,  # hidden is on mesh axis 0 (dims=(3,None), mesh_shape=(num_devices,1))
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=sems,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.rms_norm_post_all_gather(
        x_il,
        gathered,
        epsilon=epsilon,
        weight=g_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return out


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_rms_fused_vs_decompose_single_op(mesh_device, function_level_defaults):
    """Direct per-op PCC comparison against a shared fp32 golden."""
    num_devices = 8
    hidden_size = 896 * 8  # 7168
    seq_len = 32
    batch_size_per_row = 8
    epsilon = 1e-6
    num_iters = 10
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    cfg = _build_common(mesh_device, num_devices, batch_size_per_row, seq_len, hidden_size, input_shard_grid)
    fused_sems = [ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0) for _ in range(num_iters)]
    decompose_sems = [ttnn.create_global_semaphore(mesh_device, cfg["ccl_sub_device_crs"], 0) for _ in range(2)]

    pcc_fused, pcc_decompose = [], []
    for i in range(num_iters):
        x_torch, g_torch, x_tt, g_tt = _make_inputs(
            mesh_device, num_devices, (1, 1, seq_len, hidden_size), hidden_size, cfg["input_memory_config"]
        )
        golden = get_torch_rms(x_torch, g_torch, epsilon)

        f = _to_torch(mesh_device, _run_fused(mesh_device, x_tt, g_tt, cfg, fused_sems[i], epsilon), num_devices)
        pf = _pcc_value(f, golden)
        pcc_fused.append(pf)

        try:
            d = _to_torch(
                mesh_device, _run_decompose(mesh_device, x_tt, g_tt, cfg, decompose_sems, epsilon), num_devices
            )
            pd = _pcc_value(d, golden)
        except Exception as e:
            logger.error(f"[iter {i}] decompose path raised (tune API args here): {e!r}")
            pd = float("nan")
        pcc_decompose.append(pd)

        logger.info(f"[iter {i}]  FUSED PCC={pf:.6f}   DECOMPOSE PCC={pd:.6f}   delta={pd - pf:+.6f}")

    n = sum(1 for a, b in zip(pcc_fused, pcc_decompose) if a == a and b == b)
    mf = sum(a for a in pcc_fused if a == a) / max(1, sum(1 for a in pcc_fused if a == a))
    md = sum(b for b in pcc_decompose if b == b) / max(1, sum(1 for b in pcc_decompose if b == b))
    logger.info("=" * 70)
    logger.info(f"MEAN over {n} iters:   FUSED PCC={mf:.6f}   DECOMPOSE PCC={md:.6f}   delta={md - mf:+.6f}")
    logger.info("=" * 70)
    mesh_device.reset_sub_device_stall_group()


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("depth", [122])  # ~61 layers x 2 norms
def test_rms_fused_vs_decompose_compounding(mesh_device, depth, function_level_defaults):
    """
    Apply each method `depth` times with a residual add between steps, mimicking the
    decode residual stream. Reports PCC vs depth so the per-op gap is shown compounding
    into the full-model gap. SECONDARY: the residual re-shard between steps may need a
    small tweak on your setup; the single-op test above is the robust primary proof.
    """
    num_devices = 8
    hidden_size = 896 * 8
    seq_len = 32
    batch_size_per_row = 8
    epsilon = 1e-6
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    cfg = _build_common(mesh_device, num_devices, batch_size_per_row, seq_len, hidden_size, input_shard_grid)
    fused_sem = ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0)
    decompose_sems = [ttnn.create_global_semaphore(mesh_device, cfg["ccl_sub_device_crs"], 0) for _ in range(2)]

    for method_name in ("fused", "decompose"):
        x_torch, g_torch, x_tt, g_tt = _make_inputs(
            mesh_device, num_devices, (1, 1, seq_len, hidden_size), hidden_size, cfg["input_memory_config"]
        )
        acc_torch = x_torch.float()
        acc_tt = x_tt
        for k in range(depth):
            # golden: residual + norm(residual)
            acc_torch = acc_torch + get_torch_rms(acc_torch, g_torch, epsilon)
            # device
            if method_name == "fused":
                y = _run_fused(mesh_device, acc_tt, g_tt, cfg, fused_sem, epsilon)
            else:
                y = _run_decompose(mesh_device, acc_tt, g_tt, cfg, decompose_sems, epsilon)
                # bring decompose output back to the width-sharded input layout for the residual add
                y = ttnn.to_memory_config(y, cfg["input_memory_config"])
            acc_tt = ttnn.add(acc_tt, y, memory_config=cfg["input_memory_config"])
            if (k + 1) % 20 == 0 or k == depth - 1:
                cur = _to_torch(mesh_device, acc_tt, num_devices)
                logger.info(f"[{method_name}] depth={k + 1:3d}  PCC={_pcc_value(cur, acc_torch):.6f}")
    mesh_device.reset_sub_device_stall_group()


# Compute-kernel-config sweep on the FUSED op. HiFi4 fidelity fixed; the three flags
# swept are the ones tt-mlir originally hardcoded ("old_mlir") vs what we changed them
# to so they match the tt-metal device default ("metal_match"), plus one-flag-at-a-time
# isolations from the metal_match baseline. HiFi4/approx=T/fp32=F/l1=F IS the device
# default, so "device_default(None)" and "metal_match" should agree (sanity check).
#   name -> None (device default) or dict of the three flags (math_fidelity=HiFi4 always)
_CONFIG_SWEEP = [
    ("device_default(None)", None),
    ("metal_match  approx=T fp32=F l1=F", dict(math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False)),
    ("old_mlir     approx=F fp32=T l1=T", dict(math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)),
    ("isolate fp32=T (else metal_match)", dict(math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False)),
    ("isolate approx=F (else metal_match)", dict(math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)),
    ("isolate l1=T (else metal_match)", dict(math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)),
]


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_rms_fused_config_sweep(mesh_device, function_level_defaults):
    """Fused-op PCC vs compute-kernel-config, to see which of the three flags moves accuracy.

    All configs run on the SAME inputs and the SAME fp32 golden, so differences are purely
    the compute config.
    """
    num_devices = 8
    hidden_size = 896 * 8  # 7168
    seq_len = 32
    batch_size_per_row = 8
    epsilon = 1e-6
    num_iters = 5
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))})

    cfg = _build_common(mesh_device, num_devices, batch_size_per_row, seq_len, hidden_size, input_shard_grid)

    # Build the input set ONCE so every config is scored on identical data.
    inputs = [
        _make_inputs(mesh_device, num_devices, (1, 1, seq_len, hidden_size), hidden_size, cfg["input_memory_config"])
        for _ in range(num_iters)
    ]
    goldens = [get_torch_rms(x_torch, g_torch, epsilon) for (x_torch, g_torch, _, _) in inputs]

    results = {}
    for name, flags in _CONFIG_SWEEP:
        ckc = (
            None
            if flags is None
            else ttnn.init_device_compute_kernel_config(
                mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, **flags
            )
        )
        # Stats scratch dtype must track fp32_dest_acc_en (else the fp32 CB overflows the
        # bf16 L1 bank). This mirrors tt-mlir's own buffer allocation.
        fp32 = bool(flags and flags.get("fp32_dest_acc_en"))
        stats = _make_stats(mesh_device, num_devices, ttnn.float32 if fp32 else ttnn.bfloat16)
        pccs = []
        try:
            for i in range(num_iters):
                _, _, x_tt, g_tt = inputs[i]
                sem = ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0)
                out = _run_fused(mesh_device, x_tt, g_tt, cfg, sem, epsilon, compute_kernel_config=ckc, stats=stats)
                pccs.append(_pcc_value(_to_torch(mesh_device, out, num_devices), goldens[i]))
            results[name] = sum(pccs) / len(pccs)
        except Exception as e:
            results[name] = float("nan")
            logger.error(f"{name:38s}  CRASHED: {e!r}")
        logger.info(f"{name:38s}  mean PCC over {num_iters} = {results[name]:.6f}")

    logger.info("=" * 72)
    logger.info("FUSED compute-config sweep (higher = better):")
    for name, _ in _CONFIG_SWEEP:
        logger.info(f"  {name:38s}  {results[name]:.6f}")
    logger.info("=" * 72)
    mesh_device.reset_sub_device_stall_group()
