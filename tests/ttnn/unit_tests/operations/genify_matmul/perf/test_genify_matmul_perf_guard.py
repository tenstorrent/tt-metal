# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in device-ns guard set for genify_matmul (bf16 x bf16, interleaved DRAM, epilogue none).

Skipped unless GENIFY_MATMUL_PERF=1. Run through the eval runner so device_kernel_ns is recorded:
    GENIFY_MATMUL_PERF=1 eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/genify_matmul/perf <out_dir>
Shapes span the R1 work-split regimes: 2-D rectangle, 1-D N split (M <= 32), small-M / deep-K
DRAM-bound DiT shapes, and the tall-M shapes where the factorization trades W against re-reads.
"""
import math
import os
import pytest
import torch
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf guard: set GENIFY_MATMUL_PERF=1"),
]

SHAPES = [
    (9472, 5120, 3456),
    (9472, 3456, 5120),
    (4096, 4096, 4096),
    (2048, 9216, 6144),
    (2048, 5120, 8704),
    (256, 6144, 9216),
    (256, 9216, 6144),
    (256, 15360, 3072),
    (1, 6144, 6144),
    (384, 1536, 1024),
    (2048, 128, 3072),
    (2048, 3072, 128),
    (32, 1024, 2048),
    (2048, 256, 512),
]


@pytest.mark.parametrize("fid", ["HiFi4", "HiFi2"])
@pytest.mark.parametrize("mkn", SHAPES, ids=[f"{m}-{k}-{n}" for m, k, n in SHAPES])
def test_perf(mkn, fid, device):
    m, k, n = mkn
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)


# Refinement 1: one representative per new numeric path (activation / weight dtype, DEST width),
# at the model's own fidelity. (m, k, n, A dtype, B dtype, out dtype, fidelity, fp32_dest_acc_en)
FORMAT_CASES = [
    (2048, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", True),  # Qwen prefill, bf4b weight
    (512, 5120, 3072, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", True),  # FLUX text encoder, bf8b weight
    (1, 6144, 6144, "float32", "bfloat16", "float32", "HiFi2", True),  # FLUX timestep linear_2, fp32 A
    (2048, 256, 512, "bfloat8_b", "bfloat8_b", "bfloat8_b", "LoFi", False),  # bf8b x bf8b, 16-bit DEST
    (4096, 256, 1024, "bfloat8_b", "bfloat8_b", "bfloat8_b", "LoFi", False),  # PDL upsample W pass
    (64, 16384, 256, "bfloat16", "bfloat16", "bfloat16", "HiFi4", False),  # 16-bit DEST, capped K-walk
]


@pytest.mark.parametrize(
    "case", FORMAT_CASES, ids=[f"{m}-{k}-{n}-{a}-{w}-{f}-acc{int(acc)}" for m, k, n, a, w, _, f, acc in FORMAT_CASES]
)
def test_perf_formats(case, device):
    m, k, n, a_dt, w_dt, o_dt, fid, acc = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, dtype=getattr(ttnn, o_dt), compute_kernel_config=cfg)
    ttnn.synchronize_device(device)


# Refinement 2: one representative per fused-epilogue kernel path at the model's own config, each
# next to its epilogue-none twin (same shape, same formats) so the epilogue's cost stays visible.
# (m, k, n, A dtype, B dtype, fidelity, fp32_dest_acc_en, epilogue)
EPILOGUE_CASES = [
    (2048, 5120, 8704, "bfloat16", "bfloat4_b", "LoFi", True, "swiglu"),  # PERF FOCUS 1
    (2048, 6144, 3072, "bfloat16", "bfloat16", "HiFi2", True, "addcmul"),  # FLUX attn out + gated residual
    (2048, 256, 512, "bfloat8_b", "bfloat8_b", "LoFi", False, "bias_relu"),  # PDL 1x1 conv
    (384, 1536, 1024, "bfloat16", "bfloat16", "HiFi4", True, "bias"),
]


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "none"])
@pytest.mark.parametrize(
    "case",
    EPILOGUE_CASES,
    ids=[f"{m}-{k}-{n}-{a}-{w}-{f}-acc{int(acc)}-{e}" for m, k, n, a, w, f, acc, e in EPILOGUE_CASES],
)
def test_perf_epilogues(case, fused, device):
    m, k, n, a_dt, w_dt, fid, acc, epilogue = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    a_dtype, w_dtype = getattr(ttnn, a_dt), getattr(ttnn, w_dt)
    ta = create_ttnn_input_tensor(A, device, dtype=a_dtype, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=w_dtype, layout=ttnn.TILE_LAYOUT)
    kwargs = {}
    if fused and epilogue in ("bias", "bias_relu"):
        bias_dtype = ttnn.bfloat8_b if w_dtype == ttnn.bfloat4_b else w_dtype
        kwargs["bias"] = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=bias_dtype, layout=ttnn.TILE_LAYOUT)
        if epilogue == "bias_relu":
            kwargs["activation"] = "relu"
    elif fused and epilogue == "swiglu":
        kwargs["activation"] = "swiglu"
    elif fused and epilogue == "addcmul":
        kwargs["residual"] = create_ttnn_input_tensor(
            torch.randn(1, m, n), device, dtype=a_dtype, layout=ttnn.TILE_LAYOUT
        )
        kwargs["gate"] = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=a_dtype, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=acc, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, **kwargs)
    ttnn.synchronize_device(device)


# Refinement 3: L1-resident A and output on the R1a (2-D operand multicast) path, so the operand
# multicast is measured where A no longer comes from DRAM. (m, k, n, B dtype, fidelity, epilogue)
L1_CASES = [
    (128, 5120, 8704, "bfloat4_b", "LoFi", "swiglu"),  # Qwen prefill-128 gate_up (act_buffer=l1)
    (384, 1536, 1024, "bfloat16", "HiFi4", "none"),
]


@pytest.mark.parametrize("case", L1_CASES, ids=[f"{m}-{k}-{n}-{w}-{f}-{e}" for m, k, n, w, f, e in L1_CASES])
def test_perf_l1_resident(case, device):
    m, k, n, w_dt, fid, epilogue = case
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(
        A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG, **kwargs)
    ttnn.synchronize_device(device)


# Refinement 4: regime R4 `dram_sharded` (B DRAM width-sharded over the banks, A L1 width-sharded over
# K on `cores` cores, output in the op-built L1 width shard). Qwen3.8 decode at its own config (bf16 x
# bf4b, LoFi, fp32 DEST), plus a multi-row bf16 shape. (m, k, n, A cores, B dtype, fidelity, epilogue)
DRAM_SHARDED_CASES = [
    (1, 4352, 5120, 8, "bfloat4_b", "LoFi", "none"),  # PERF FOCUS 2: decode MLP down
    (1, 5120, 16384, 10, "bfloat4_b", "LoFi", "none"),  # LM head vocab chunk (K split per A shard)
    (32, 5120, 9216, 10, "bfloat4_b", "LoFi", "swiglu"),  # decode gate_up, batch 32
    (384, 1536, 1024, 48, "bfloat16", "HiFi4", "none"),  # cartesian multi-row, 1-tile A shards
]


@pytest.mark.parametrize(
    "case", DRAM_SHARDED_CASES, ids=[f"{m}-{k}-{n}-c{c}-{w}-{f}-{e}" for m, k, n, c, w, f, e in DRAM_SHARDED_CASES]
)
def test_perf_dram_sharded(case, device):
    m, k, n, cores, w_dt, fid, epilogue = case
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    m_pad = -(-m // 32) * 32
    grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True)
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [m_pad, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    banks = device.dram_grid_size().x
    granule = 64 if epilogue == "swiglu" else 32
    width = -(-n // (granule * banks)) * granule
    b_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT, memory_config=b_mc)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, **kwargs)
    ttnn.synchronize_device(device)


# Refinement 5: regime R3 `block_sharded` (A L1 block-sharded: rows over grid rows, K over grid
# columns; output L1 block-sharded on A's grid). PDL's 1x1 convs at their own config (bf8b x bf8b,
# LoFi, fp32 DEST, bf8b bias). `exchange`: the peer K-slice exchange (BS_ROW_MCAST True = rotating row
# multicast, the default; False = the unicast stepping stone). (m, k, n, shard, grid (x, y), epilogue)
BLOCK_SHARDED_CASES = [
    (2048, 1024, 2048, (224, 128), (8, 10), "none"),  # res5 shortcut
    (8192, 512, 256, (1024, 64), (8, 8), "bias_relu"),  # res4 reduce
    (2048, 512, 2048, (224, 64), (8, 10), "none"),  # res5 expand
    (2048, 2048, 256, (256, 256), (8, 8), "bias_relu"),  # ASPP 1x1
]


@pytest.mark.parametrize("exchange", ["mcast", "unicast"])
@pytest.mark.parametrize(
    "case",
    BLOCK_SHARDED_CASES,
    ids=[f"{m}-{k}-{n}-{gx}x{gy}-{e}" for m, k, n, _, (gx, gy), e in BLOCK_SHARDED_CASES],
)
def test_perf_block_sharded(case, exchange, device, monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    monkeypatch.setattr(pd, "BS_ROW_MCAST", exchange == "mcast")
    m, k, n, shard, (gx, gy), epilogue = case
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, list(shard), ttnn.ShardOrientation.ROW_MAJOR),
    )
    o_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard[0], -(-n // (32 * gx)) * 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    kwargs = {}
    if epilogue == "bias_relu":
        kwargs["bias"] = create_ttnn_input_tensor(
            torch.randn(1, n), device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        kwargs["activation"] = "relu"
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=o_mc, **kwargs)
    ttnn.synchronize_device(device)


# Refinement 7: regime R2 `height_sharded` (A L1 height-sharded on `cores` cores laid out row-major,
# consumed in place; B from DRAM per core; output in place in its L1 height shard) at PDL's own config
# (bf8b x bf8b, LoFi, fp32 DEST, bf8b bias). (m, k, n, shard height, cores, epilogue)
HEIGHT_SHARDED_CASES = [
    (32768, 128, 256, 320, 103, "none"),  # res2 shortcut
    (8192, 128, 512, 96, 86, "none"),  # res3 expand
    (8192, 512, 128, 96, 86, "bias_relu"),  # res3 reduce
    (32768, 256, 64, 320, 103, "bias_relu"),  # res2 reduce
]


@pytest.mark.parametrize(
    "case", HEIGHT_SHARDED_CASES, ids=[f"{m}-{k}-{n}-c{c}-{e}" for m, k, n, _, c, e in HEIGHT_SHARDED_CASES]
)
def test_perf_height_sharded(case, device):
    m, k, n, shard_h, cores, epilogue = case
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True)
    a_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, k], ttnn.ShardOrientation.ROW_MAJOR),
    )
    o_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, n], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    kwargs = {}
    if epilogue == "bias_relu":
        kwargs["bias"] = create_ttnn_input_tensor(
            torch.randn(1, n), device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        kwargs["activation"] = "relu"
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=o_mc, **kwargs)
    ttnn.synchronize_device(device)


# Refinement 7: batched weight (R1d), PERF FOCUS 3 at its own config: PDL bilinear upsample H pass,
# A (1, 32, 512, 128) bf8b DRAM x B (1, 32, 128, 1024) bf8b in L1, LoFi, 16-bit DEST.
def test_perf_batched(device):
    torch.manual_seed(0)
    A = torch.randn(1, 32, 512, 128)
    B = torch.randn(1, 32, 128, 1024) / math.sqrt(128)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(
        B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)


# Refinement 8: operand residency on a one-K-block R1a walk (B resident across M-blocks): the PDL
# bilinear upsample W pass, A (1, 32, 128, 256) and the output in L1 interleaved, B DRAM, bf8b, LoFi,
# 16-bit DEST. (PERF FOCUS 3 above is the A-resident twin on the L1-discounted 32x3x1 split.)
def test_perf_upsample_w_pass(device):
    torch.manual_seed(0)
    A = torch.randn(1, 32, 128, 256)
    B = torch.randn(256, 1024) / math.sqrt(256)
    ta = create_ttnn_input_tensor(
        A, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
