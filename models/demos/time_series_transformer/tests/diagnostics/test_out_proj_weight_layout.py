# tests/test_out_proj_weight_layout.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC -- not part of the real bounty test suite.

Context: test_concatenate_heads_layout.py PROVED concatenate_heads' output
layout is correct (bit-identical to manual per-head reassembly, max_abs_diff
0.0). So the [4] final-output PCC failure (0.8194) must be in out_proj_weight
/ out_proj_bias themselves, built by _build_out_proj() in tst_model.py:

    def _build_out_proj(state, prefix, device):
        W = state[f"{prefix}.out_proj.weight"].float()   # HF: [26, 26] (out, in)
        b = state[f"{prefix}.out_proj.bias"].float()
        W_t = W.T                                          # [26, 26] (in, out) -- ttnn.linear convention
        W_padded_out = F.pad(W_t, (0, PADDED_WIDTH - D_MODEL))      # pad OUTPUT dim: 26->64, tail-padded
        W_padded = F.pad(W_padded_out, (0, 0, 0, PADDED_WIDTH - D_MODEL))  # pad INPUT dim: 26->64, tail-padded
        b_padded = F.pad(b, (0, PADDED_WIDTH - D_MODEL))
        return _to_ttnn(W_padded, device), _to_ttnn(b_padded, device)

HYPOTHESIS: the INPUT-dimension padding (the second F.pad, padding ROWS
0..63 where only rows 0..25 hold real HF weight data) assumes the actual
input activation has its real data CONTIGUOUSLY at columns [0:26], with
padding at [26:64]. But concatenate_heads' actual output (verified in the
prior test) has real data PER-HEAD: columns [0:13] real (head 0),
[13:32] padding, [32:45] real (head 1), [45:64] padding.

If true: out_proj_weight's rows [13:32] and [45:64] are ZERO (no real HF
weight there, by construction), but the input activation's REAL data sits
exactly inside [32:45] -- which out_proj_weight treats as padding (zero
rows). That means head 1's entire real contribution gets multiplied by
ZERO weight rows and silently dropped from the output. Only head 0's data
(rows [0:13], which DO hold real weight) would actually contribute.

THIS TEST: directly inspect out_proj_weight's row-wise structure (which
rows are nonzero) and compare against the known real-data column layout
of concatenate_heads' output, to confirm or refute this precisely --
no guessing from the construction code alone.
"""

from pathlib import Path

import pytest
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_PADDED, HEAD_DIM_TRUE
from tt.tst_model import load_weights

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    yield device, hf_model, weights
    ttnn.close_device(device)


def test_out_proj_weight_row_layout(setup):
    device, hf_model, weights = setup

    w = weights["decoder.layers.0"]["self_attn"]
    out_proj_weight = ttnn.to_torch(w["out_proj_weight"]).float()  # expect [64, 64] (in, out)
    out_proj_bias = ttnn.to_torch(w["out_proj_bias"]).float()  # expect [64]

    logger.info(f"out_proj_weight shape: {tuple(out_proj_weight.shape)}")
    logger.info(f"out_proj_bias shape: {tuple(out_proj_bias.shape)}")

    # Per-row max abs value tells us which INPUT rows (0..63) actually hold
    # nonzero weight data vs. which are pure construction-padding.
    row_max_abs = out_proj_weight.abs().max(dim=1).values  # [64]
    nonzero_rows = (row_max_abs > 1e-6).nonzero(as_tuple=True)[0].tolist()
    logger.info(f"out_proj_weight nonzero INPUT rows (threshold 1e-6): {nonzero_rows}")
    logger.info(f"  -> first 30 row max-abs values: {row_max_abs[:30].tolist()}")
    logger.info(
        f"  -> rows 32-46 max-abs values (where head-1's REAL data lives "
        f"in concatenate_heads' output): {row_max_abs[32:46].tolist()}"
    )

    # Known-good fact from prior test: concatenate_heads' REAL (non-padding)
    # data sits at columns [0:13] (head 0) and [32:45] (head 1).
    head0_real_cols = list(range(0, HEAD_DIM_TRUE))  # [0:13]
    head1_real_cols = list(range(HEAD_DIM_PADDED, HEAD_DIM_PADDED + HEAD_DIM_TRUE))  # [32:45]
    all_real_cols = head0_real_cols + head1_real_cols
    all_padding_cols = [c for c in range(2 * HEAD_DIM_PADDED) if c not in all_real_cols]

    real_rows_nonzero = row_max_abs[all_real_cols]
    padding_rows_nonzero = row_max_abs[all_padding_cols]

    logger.info(
        f"out_proj_weight rows at REAL data columns {all_real_cols[:3]}...{all_real_cols[-3:]}: "
        f"mean max-abs = {real_rows_nonzero.mean().item():.6f}, "
        f"min = {real_rows_nonzero.min().item():.6f}"
    )
    logger.info(
        f"out_proj_weight rows at PADDING columns (sample): "
        f"mean max-abs = {padding_rows_nonzero.mean().item():.6f}, "
        f"max = {padding_rows_nonzero.max().item():.6f}"
    )

    # The critical check: is head 1's real-data region [32:45] actually
    # ZERO in out_proj_weight (meaning it gets silently dropped)?
    head1_region_weight = out_proj_weight[HEAD_DIM_PADDED : HEAD_DIM_PADDED + HEAD_DIM_TRUE, :]
    head1_region_all_zero = head1_region_weight.abs().max().item() < 1e-6
    logger.info(
        f"Is out_proj_weight ZERO at rows [32:45] (head 1's real-data columns)? "
        f"{head1_region_all_zero} (max abs value there: {head1_region_weight.abs().max().item():.6f})"
    )

    # And for contrast: is out_proj_weight nonzero at rows [13:26] (where
    # the CONTIGUOUS-padding construction assumption thinks head 1's real
    # data should be, i.e. directly after head 0's 13 real columns)?
    contiguous_assumption_region = out_proj_weight[HEAD_DIM_TRUE : 2 * HEAD_DIM_TRUE, :]
    contiguous_region_nonzero = contiguous_assumption_region.abs().max().item()
    logger.info(
        f"out_proj_weight rows [13:26] (where a CONTIGUOUS 26-wide-input "
        f"assumption would expect head 1's real data): max abs value = "
        f"{contiguous_region_nonzero:.6f} (nonzero here + zero at [32:45] "
        f"would CONFIRM the contiguous-vs-per-head layout mismatch)"
    )

    logger.info(
        "=== Verdict: if rows [13:26] are nonzero AND rows [32:45] are zero, "
        "out_proj_weight was built assuming a CONTIGUOUS 26-wide input, but "
        "the real input (concatenate_heads output) is PER-HEAD padded -- "
        "confirming the layout mismatch hypothesis precisely. ==="
    )
