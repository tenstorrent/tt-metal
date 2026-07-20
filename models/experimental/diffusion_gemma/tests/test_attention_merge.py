# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flash-attention partial merge (design task T7).

Two layers:

1. ``test_merge_formula_matches_full_softmax`` — a **hostless** torch-only check
   of the merge MATH. It builds random scores, splits the keys into a prefix
   group (A) and a canvas group (B), computes each group's softmax-normalized
   output and log-sum-exp, applies the merge formula, and asserts the result
   equals the single full-softmax attention over the concatenation of both
   groups. Runs with no device.

2. ``test_device_merge_partials_matches_torch`` — a **device-gated** check of the
   real ttnn ``merge_attention_partials`` against the same torch reference, at a
   bf16-appropriate PCC. Gated exactly like the sibling on-device tests
   (``DG_RUN_DEVICE=1`` + shared module device) so the default CPU suite never
   opens a device.

The merge is the Phase-2 companion to the ``return_lse`` SDPA extension; see
``doc/optimize_perf/paged_prefix_denoise_design.md`` §1a and task T7.
"""

import os

import pytest
import torch


# --- shared torch reference ------------------------------------------------


def _group_softmax(scores_g, values_g):
    """Softmax-normalized output + flash log-sum-exp for one key group.

    Args:
        scores_g: ``[H, C, Kg]`` fp32 attention scores for this group.
        values_g: ``[H, Kg, vhd]`` fp32 values for this group.

    Returns:
        ``(out, lse)`` with ``out`` ``[H, C, vhd]`` = softmax(scores_g) @ values_g
        and ``lse`` ``[H, C, 1]`` = logsumexp(scores_g) — the exact statistic the
        ``return_lse=True`` SDPA kernel emits (``m + log(l)``).
    """
    probs = torch.softmax(scores_g, dim=-1)
    out = probs @ values_g
    lse = torch.logsumexp(scores_g, dim=-1, keepdim=True)
    return out, lse


def _torch_merge(out_a, lse_a, out_b, lse_b):
    """Torch mirror of ``merge_attention_partials`` (the exact merge formula)."""
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    denom = wa + wb
    return (out_a * wa + out_b * wb) / denom


def _build_reference(*, num_heads, canvas, prefix_len, canvas_keys, vhd, seed):
    """Build a random two-group attention problem and its exact merged golden.

    Returns ``(out_a, lse_a, out_b, lse_b, full_out)`` all fp32 torch tensors:
    per-group partials for the merge inputs plus ``full_out`` ``[H, C, vhd]`` =
    the single full-softmax attention over ``concat(group_a, group_b)``.
    """
    torch.manual_seed(seed)
    total_keys = prefix_len + canvas_keys
    scores = torch.randn(num_heads, canvas, total_keys, dtype=torch.float32)
    values = torch.randn(num_heads, total_keys, vhd, dtype=torch.float32)

    # Ground truth: one softmax over all keys.
    full_out = torch.softmax(scores, dim=-1) @ values

    # Split keys into group A (prefix) and group B (canvas), each self-normalized.
    out_a, lse_a = _group_softmax(scores[..., :prefix_len], values[:, :prefix_len, :])
    out_b, lse_b = _group_softmax(scores[..., prefix_len:], values[:, prefix_len:, :])
    return out_a, lse_a, out_b, lse_b, full_out


# --- 1. hostless math check ------------------------------------------------


def test_merge_formula_matches_full_softmax():
    out_a, lse_a, out_b, lse_b, full_out = _build_reference(
        num_heads=4, canvas=16, prefix_len=48, canvas_keys=16, vhd=32, seed=47470
    )
    merged = _torch_merge(out_a, lse_a, out_b, lse_b)
    torch.testing.assert_close(merged, full_out, atol=1e-5, rtol=1e-5)


def test_merge_reduces_to_dominant_group():
    """When one group's lse dwarfs the other, the merge returns that group's output.

    Guards the numeric-stability branch: the max-shift makes the losing weight
    underflow to ~0 instead of ``exp`` overflowing.
    """
    out_a, lse_a, out_b, lse_b, _ = _build_reference(
        num_heads=2, canvas=8, prefix_len=32, canvas_keys=8, vhd=16, seed=991
    )
    # Force group A to dominate by a wide margin.
    merged = _torch_merge(out_a, lse_a + 50.0, out_b, lse_b)
    torch.testing.assert_close(merged, out_a, atol=1e-5, rtol=1e-5)


# --- 2. device-gated check -------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
def test_device_merge_partials_matches_torch(device):
    """Real ttnn merge vs the torch golden at a bf16-appropriate PCC.

    bf16 partial outputs (activation dtype) + fp32 lse, so the merged result
    carries only bf16 rescale drift; PCC 0.99 mirrors the sibling on-device SDPA
    tests. Bitwise agreement is NOT expected (gated on decision-agreement per the
    design doc)."""
    import ttnn
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.diffusion_gemma.tt.attention_merge import merge_attention_partials

    out_a, lse_a, out_b, lse_b, full_out = _build_reference(
        num_heads=8, canvas=64, prefix_len=256, canvas_keys=64, vhd=128, seed=47471
    )

    # Merge inputs: [1, H, C, vhd] bf16 outputs + [1, H, C, 1] fp32 lse.
    tt_out_a = ttnn.from_torch(out_a.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out_b = ttnn.from_torch(out_b.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lse_a = ttnn.from_torch(lse_a.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lse_b = ttnn.from_torch(lse_b.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    tt_merged = merge_attention_partials(tt_out_a, tt_lse_a, tt_out_b, tt_lse_b)
    merged = ttnn.to_torch(tt_merged)[0]  # drop the batch dim -> [H, C, vhd]

    assert_with_pcc(full_out, merged, 0.99)
    tt_merged.deallocate(True)
