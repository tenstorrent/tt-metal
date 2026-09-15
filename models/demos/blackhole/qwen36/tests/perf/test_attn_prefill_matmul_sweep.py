# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-measure the attention PREFILL one-K-pass progcfgs at 27B / TP=8 shapes.

WHY
---
``model_config.attn_qkv_fused_prefill_progcfg`` and ``attn_wo_prefill_progcfg`` swap the shared
halved-block prefill factory for ``create_prefill_kpass1_matmul_program_config``, paired with a
compute config that turns fp32 dest accumulation OFF (and drops to LoFi). They are gated
``None if is_blackhole() else <kpass1>``, so every Wormhole config gets them -- but the numbers in
those comments were measured on the **9B at N300 (TP=2)**:

    QKV  M=2048 K=4096 N=5120   1683.7 -> 1011.5 us  (-39.9%)   pcc 0.99997 -> 0.99992
    wo   M=2048 K=2048 N=4096     558.6 ->  517.2 us  (-7.4%)    pcc 0.99997 -> 0.99993

The 27B at TP=8 has DIFFERENT shapes, and the pass-count argument those comments rest on does not
survive the change:

    QKV  M=2048 K=dim=5120            N=attn_qkv_fused_dim_tp=2048   (64 tiles, per_core_N=8)
    wo   M=2048 K=attn_out_dim_tp=768 N=dim=5120                     (160 tiles, per_core_N=20)

At the 9B the QKV win came from 5 K passes collapsing to 1 (per_core_N=20 vs an fp32-acc subblock
cap of 4). At the 27B per_core_N is 8, which the cap of 4 already divides, so the baseline is at
most 2 passes -- the headline -39.9% cannot transfer. Worth knowing whether what remains is a win,
a wash, or a regression, because the change is not free: it costs PCC, and it is a COMPUTE-CONFIG
change, which is the axis this codebase has already found isolated sweeps least trustworthy on
(mlp.py: packer_l1_acc under DRAM saturation). So this file reports isolated numbers and the real
layer is the tiebreak.

Also worth recording: the sweep those comments cite,
``tests/perf/test_attn_qkv_inproj_sweep.py``, is NOT in the tree (nor is
``test_oproj_dtype_isl`` cited elsewhere), so neither result can be re-derived -- only re-measured.

DTYPES mirror the real call sites: the fused QKV weight and the wo weight are bf8; QKV's in0 is the
gathered attention_norm output (bf16 on a full-attention layer -- layer.py only narrows GDN layers),
and wo's in0 is the post-SDPA activation, which is bf8.

Run (needs a device; skipped unless the env var is set)::

    QWEN_ATTN_SWEEP=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/tests/perf/test_attn_prefill_matmul_sweep.py -v -s

Under tracy each (case, variant) is signposted; read DEVICE KERNEL DURATION.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole

M = 2048
ITERS = 3
_SKIP = os.environ.get("QWEN_ATTN_SWEEP") != "1"

try:
    from tracy import signpost as _SP
except ImportError:  # pragma: no cover
    _SP = None


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]


@pytest.mark.skipif(_SKIP, reason="set QWEN_ATTN_SWEEP=1 to run the attention prefill matmul sweep")
@pytest.mark.timeout(2400)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_attn_prefill_matmul_sweep(mesh_device, device_params):
    """baseline (shared halved-block + fp32 acc) vs kpass1 (+no-fp32-acc/LoFi), at this config's shapes."""
    del device_params
    from models.demos.blackhole.qwen36.tests.test_factory import model_path
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    mesh_device.enable_program_cache()
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=32, max_seq_len=4096)
    grid = args._prefill_grid
    gw = getattr(args, "decode_grid_w", 8)

    # (label, K, N, in0_dtype, weight_dtype, baseline-progcfg-builder, kpass1-compute-cfg)
    cases = [
        (
            "qkv",
            args.dim,
            args.attn_qkv_fused_dim_tp,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            # _col_proj's baseline is the SHARED halved-block prefill_progcfg with self.compute_cfg.
            lambda m, k, n: args.prefill_progcfg(m, k, n),
            tpc.COMPUTE_LOFI_NO_FP32_ACC,  # _col_proj pairs LoFi with the kpass1 override
        ),
        (
            "wo",
            args.attn_out_dim_tp,
            args.dim,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            # _wo_proj's baseline is create_prefill_mlp_matmul_program_config(max_cols=decode_grid_w).
            lambda m, k, n: tpc.create_prefill_mlp_matmul_program_config(m, k, n, max_cols=gw),
            tpc.COMPUTE_LOFI_NO_FP32_ACC,
        ),
    ]

    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    logger.info(
        f"attn prefill sweep: M={M} dim={args.dim} qkv_fused_tp={args.attn_qkv_fused_dim_tp} "
        f"attn_out_tp={args.attn_out_dim_tp} grid={grid} mesh={MESH_SHAPE}"
    )

    summary = []
    for label, K, N, in0_dt, w_dt, base_pc, kpass_ck in cases:
        x = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        w = torch.randn(K, N, dtype=torch.bfloat16) * (K**-0.5)
        ref = x.float()[0, 0] @ w.float()

        def _mk(t, dt):
            return ttnn.from_torch(
                t,
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **({"mesh_mapper": rep} if rep else {}),
            )

        x_tt, w_tt = _mk(x, in0_dt), _mk(w, w_dt)
        n_tiles, k_tiles = N // 32, K // 32
        logger.info(f"  [{label}] M={M} K={K} N={N} ({n_tiles} n-tiles, {k_tiles} k-tiles) in0={in0_dt} w={w_dt}")

        variants = {
            "baseline": (base_pc(M, K, N), tpc.COMPUTE_HIFI2),
            "kpass1": (tpc.create_prefill_kpass1_matmul_program_config(M, K, N, grid_size=grid), kpass_ck),
        }
        for vname, (pc, ck) in variants.items():
            try:
                o = ttnn.linear(
                    x_tt, w_tt, compute_kernel_config=ck, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.synchronize_device(mesh_device)
                host = ttnn.to_torch(ttnn.get_device_tensors(o)[0] if _MULTI else o).float()[0, 0]
                pcc = float(comp_pcc(ref, host, 0.0)[1])
                ttnn.deallocate(o)
            except Exception as e:  # an illegal blocking at this shape is a RESULT
                logger.warning(f"    {vname:9} UNSUPPORTED: {type(e).__name__}: {str(e)[:170]}")
                summary.append((label, vname, None, None))
                continue

            if _SP is not None:
                _SP(f"{label}_{vname}_start")
            for _ in range(ITERS):
                o = ttnn.linear(
                    x_tt, w_tt, compute_kernel_config=ck, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            if _SP is not None:
                _SP(f"{label}_{vname}_stop")

            sb = f"{pc.out_subblock_h}x{pc.out_subblock_w}"
            obw = getattr(pc, "out_block_w", None)
            passes = "?" if not obw else f"{-(-pc.per_core_N // obw)}"
            logger.info(
                f"    {vname:9} pcc={pcc:.7f}  per_core_N={pc.per_core_N} in0_block_w={pc.in0_block_w} "
                f"sub={sb} out_block_w={obw} Kpasses={passes}"
            )
            summary.append((label, vname, pcc, passes))

        ttnn.deallocate(x_tt)
        ttnn.deallocate(w_tt)

    logger.info("=== attn prefill sweep summary (PCC vs fp32; device time from tracy) ===")
    for label, vname, pcc, passes in summary:
        logger.info(f"  {label:4} {vname:9} " + ("UNSUPPORTED" if pcc is None else f"pcc={pcc:.7f} Kpasses={passes}"))
