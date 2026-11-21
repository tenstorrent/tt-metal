# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of avg_pool2d (verifier, Phase 0).

Shapes: small (7x7x64, one cluster pair), the model's map (7x7x2048, width-sharded on two cores in L1),
many-tile column (14x14x32: seven position tiles, the last one with 4 valid rows) and a three-tile-wide one
(3x5x96: 15 positions, one partial tile). Configs: the op's default (HiFi4, 32-bit DEST), HiFi4 with a 16-bit
DEST, and the model's (LoFi, approximation on, 16-bit DEST).

Measured per case: PCC (comp_pcc), max / mean abs error (comp_allclose and torch), relative RMS (RMS error over
the reference's std, as eval/metrics.py check_output), ULP distance at bfloat16 against the reference rounded to
bfloat16 (p50 / p99 / max), and the got/true ratio spread (median, p5, p95) over elements whose reference is
away from zero -- a tight cluster around a constant other than 1.0 would be a scale bug, not rounding.

Gates: PCC and relative RMS at NUMERIC["legacy_tolerances"] of the specification for (bfloat16,
fp32_dest_acc_en); the model's config on uniform [0, 1) inputs is graded by PCC alone (>= 0.99), as the
specification's loose case is. Results are written to $AVG_POOL2D_BASELINE_DIR (default
/tmp/avg_pool2d_precision_baseline), one JSON per case: test output is not shown under the simulator's workers.
"""

import json
import os
import re

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.avg_pool2d import avg_pool2d

TILE = 32
OUT_DIR = os.environ.get("AVG_POOL2D_BASELINE_DIR", "/tmp/avg_pool2d_precision_baseline")

# (pcc >=, relative rms <=): NUMERIC["legacy_tolerances"] of eval/specs/avg_pool2d/spec.py.
TOLERANCES = {True: (0.995, 0.04), False: (0.99, 0.13)}

SHAPES = {  # id -> (H, W, C, placement)
    "7x7x64_dram": (7, 7, 64, "dram"),
    "7x7x2048_width_l1": (7, 7, 2048, "width"),
    "14x14x32_l1": (14, 14, 32, "l1"),
    "3x5x96_dram": (3, 5, 96, "dram"),
}

CONFIGS = {  # id -> (math_fidelity, math_approx_mode, fp32_dest_acc_en, distribution)
    "hifi4_fp32dest": (ttnn.MathFidelity.HiFi4, False, True, "normal"),
    "hifi4_fp16dest": (ttnn.MathFidelity.HiFi4, False, False, "normal"),
    "lofi_approx_fp16dest_uniform": (ttnn.MathFidelity.LoFi, True, False, "uniform"),
}


def _memory_config(device, placement, rows, C):
    if placement == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if placement == "l1":
        return ttnn.L1_MEMORY_CONFIG
    padded_rows = -(-rows // TILE) * TILE
    return ttnn.create_sharded_memory_config(
        shape=(padded_rows, C // 2),
        core_grid=ttnn.num_cores_to_corerangeset(2, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _ulp_bf16(got, ref):
    """Per-element ULP distance at bfloat16 between `got` and the reference rounded to bfloat16."""

    def ordered(x):
        bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int64)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(got) - ordered(ref)).abs().double()


@pytest.mark.parametrize("config_id", list(CONFIGS))
@pytest.mark.parametrize("shape_id", list(SHAPES))
def test_precision_baseline(device, shape_id, config_id, record_property):
    H, W, C, placement = SHAPES[shape_id]
    fidelity, approx, fp32, distribution = CONFIGS[config_id]
    torch.manual_seed(1234)
    shape = (1, 1, H * W, C)
    x = torch.rand(shape) if distribution == "uniform" else torch.randn(shape)
    x = x.to(torch.bfloat16)

    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=_memory_config(device, placement, H * W, C))
    config = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32)
    y = avg_pool2d(x_dev, 1, H, W, C, [H, W], [1, 1], [0, 0, 0, 0],
                   compute_kernel_config=config, dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT)
    got = ttnn.to_torch(y).double().reshape(1, C)
    ref = x.double().reshape(H * W, C).mean(0, keepdim=True)

    _, pcc = comp_pcc(ref, got)
    _, allclose_msg = comp_allclose(ref, got)
    diff = got - ref
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_rms = diff.pow(2).mean().sqrt().item() / ref.std().item()
    ulp = _ulp_bf16(got, ref)

    away = ref.abs() >= ref.abs().median()  # ratios of near-zero means are noise, not scale
    ratio = (got[away] / ref[away]).flatten()
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    result = {
        "shape": shape_id, "config": config_id, "pcc": float(pcc), "max_abs": max_abs, "mean_abs": mean_abs,
        "rel_rms": rel_rms, "ulp_p50": torch.quantile(ulp, 0.5).item(), "ulp_p99": torch.quantile(ulp, 0.99).item(),
        "ulp_max": ulp.max().item(), "ratio_p5": q[0], "ratio_median": q[1], "ratio_p95": q[2],
        "comp_allclose": allclose_msg,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, re.sub(r"[^\w.-]", "_", f"{shape_id}__{config_id}") + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    for k, v in result.items():
        record_property(k, v)

    # Scale-bug detector: the median ratio of the well-conditioned means stays at 1.
    assert abs(q[1] - 1.0) < 0.02, result
    pcc_min, rms_max = TOLERANCES[fp32]
    if distribution == "uniform":
        assert pcc >= 0.99, result
    else:
        assert pcc >= pcc_min and rel_rms <= rms_max, result
