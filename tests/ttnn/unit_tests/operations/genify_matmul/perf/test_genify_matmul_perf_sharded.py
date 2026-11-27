# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in device-ns bench: every PDL 1x1-conv model case on the sharded regimes (Refinement 10).

R2 `height_sharded` / R3 `block_sharded` at PDL's own config (bf8b weight + bias, LoFi, fp32 DEST, bf8b
output), with the weight multicast (`SHARDED_B_MCAST`, R2a / R3a) on and off. Skipped unless
GENIFY_MATMUL_PERF=1; profile with `scripts/run_safe_pytest.sh --profile <this file>`.
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
    pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench: set GENIFY_MATMUL_PERF=1"),
]

# (name, m, k, n, placement, shard, cores | (gx, gy), epilogue, A dtype, achievable_ns) -- feature_spec LOOSE_CASES
PDL_CASES = [
    ("res2_shortcut", 32768, 128, 256, "hs", (320, 128), 103, "bias", "bf8b", 9800),
    ("res2_reduce", 32768, 128, 64, "hs", (320, 128), 103, "bias_relu", "bf8b", 3980),
    ("res2_expand", 32768, 64, 256, "hs", (320, 64), 103, "bias", "bf8b", 8720),
    ("res2_reduce2", 32768, 256, 64, "hs", (320, 256), 103, "bias_relu", "bf8b", 5010),
    ("res3_reduce_a", 32768, 256, 128, "hs", (384, 256), 86, "bias_relu", "bf8b", 9360),
    ("res3_expand", 8192, 128, 512, "hs", (96, 128), 86, "bias", "bf8b", 8170),
    ("res3_reduce", 8192, 512, 128, "hs", (96, 512), 86, "bias_relu", "bf8b", 6900),
    ("dec_skip_a", 8192, 512, 64, "hs", (128, 512), 64, "bias_relu", "bf8b", 4650),
    ("dec_skip_b", 32768, 256, 32, "hs", (320, 256), 103, "bias_relu", "bf8b", 3600),
    ("sem_logits", 32768, 256, 32, "hs", (320, 256), 103, "bias", "bf8b", 3600),
    ("heads", 32768, 32, 32, "hs", (320, 32), 103, "bias", "bf8b", 3460),
    ("res3_shortcut", 8192, 256, 512, "hs", (96, 256), 86, "bias", "bf16", 13050),
    ("res4_reduce", 8192, 512, 256, "bs", (1024, 64), (8, 8), "bias_relu", "bf8b", 16560),
    ("res4_expand", 2048, 256, 1024, "bs", (256, 32), (8, 8), "bias", "bf8b", 13640),
    ("res4_reduce5", 2048, 1024, 256, "bs", (256, 128), (8, 8), "bias_relu", "bf8b", 9420),
    ("res5_shortcut", 2048, 1024, 2048, "bs", (224, 128), (8, 10), "bias", "bf8b", 72350),
    ("res5_reduce", 2048, 1024, 512, "bs", (224, 128), (8, 10), "bias_relu", "bf8b", 10610),
    ("res5_expand", 2048, 512, 2048, "bs", (224, 64), (8, 10), "bias", "bf8b", 24160),
    ("res5_reduce2", 2048, 2048, 512, "bs", (224, 256), (8, 10), "bias_relu", "bf8b", 16100),
    ("aspp_1x1", 2048, 2048, 256, "bs", (256, 256), (8, 8), "bias_relu", "bf8b", 13780),
    ("aspp_pool", 1, 2048, 256, "bs", (32, 256), (8, 1), "bias_relu", "bf16", 6945),
    ("aspp_proj", 2048, 1280, 256, "bs", (512, 160), (8, 4), "bias_relu", "bf8b", 16830),
    ("res4_shortcut", 2048, 512, 1024, "bs", (256, 64), (8, 8), "bias", "bf16", 16210),
]

_DT = {"bf8b": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def run_pdl_case(case, device):
    _, m, k, n, placement, shard, cores, epilogue, a_dt, _ = case
    torch.manual_seed(0)
    A = torch.randn(1, 1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    if placement == "hs":
        grid = ttnn.num_cores_to_corerangeset(cores, device.compute_with_storage_grid_size(), True)
        layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        out_shard = [shard[0], n]
    else:
        gx, gy = cores
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        out_shard = [shard[0], -(-n // (32 * gx)) * 32]
    orient = ttnn.ShardOrientation.ROW_MAJOR
    a_mc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(shard), orient))
    o_mc = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, out_shard, orient))
    ta = create_ttnn_input_tensor(A, device, dtype=_DT[a_dt], layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    bias = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    kwargs = {"bias": bias}
    if epilogue == "bias_relu":
        kwargs["activation"] = "relu"
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(ta, tb, compute_kernel_config=cfg, memory_config=o_mc, dtype=ttnn.bfloat8_b, **kwargs)
    ttnn.synchronize_device(device)
    return A, B, bias, out


@pytest.mark.parametrize("b_mcast", ["on", "off"], ids=["mc1", "mc0"])
@pytest.mark.parametrize("case", PDL_CASES, ids=[f"{c[4]}_{c[0]}" for c in PDL_CASES])
def test_perf_pdl_sharded(case, b_mcast, device, monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

    monkeypatch.setattr(pd, "SHARDED_B_MCAST", b_mcast == "on", raising=False)
    # GM_KNOBS="NAME=value,..." overrides program-descriptor knobs for a sweep (ints only).
    for kv in filter(None, os.environ.get("GM_KNOBS", "").split(",")):
        name, value = kv.split("=")
        monkeypatch.setattr(pd, name, int(value))
    run_pdl_case(case, device)
