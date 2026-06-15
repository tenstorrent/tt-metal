# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for the ViT-Base prefill SDPA PCC regression introduced by #46281
("Support streaming SDPA sliding-window prefill").

ViT-Base attention lowers to:
    ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=0.125)
with q == k == v of shape [1, 12, 197, 64] (bf16) and no attn_mask.

Because the op is non-causal and Sk == 197 is not a multiple of TILE_HEIGHT (32), the host
sets use_padded_mask=True (sdpa_program_factory.cpp) and, with the default compute config
(fp32_dest_acc_en=False), routes to the streaming compute v2 path reworked by #46281. The
non-causal partial-last-K-tile case had no test coverage in that PR (its parametrizations are
all seq % 32 == 0) and regressed numerically.

The regression is data-dependent: random / fa_rand inputs stay at PCC ~0.999, so it is only
exposed by the real ViT activation distribution (near-uniform attention with large V outliers).
The q/k/v here are therefore captured from a CPU forward of google/vit-base-patch16-224 (the
two worst layers) rather than synthesized. Capture is done on the fly so no binary fixture is
committed; weights are pulled from the HF hub and cached.

Expected:
  * streaming path (default, fp32_dest_acc_en=False) -> FAILS (PCC ~0.95-0.99)
  * legacy path    (fp32_dest_acc_en=True)           -> PASSES (PCC ~0.9999)  [control]
The legacy comparison proves the inputs are well-conditioned and that only the streaming
compute path is wrong.
"""

import math
import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

MODEL_NAME = "google/vit-base-patch16-224"
WORST_LAYERS = [10, 11]
PCC_THRESHOLD = 0.99


def _capture_vit_sdpa_inputs():
    """Run a CPU ViT-Base forward and intercept every scaled_dot_product_attention call,
    returning the (q, k, v, scale) tuples it was invoked with (one per encoder layer)."""
    transformers = pytest.importorskip("transformers")

    captured = []
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        eff_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        captured.append(
            (query.detach().clone(), key.detach().clone(), value.detach().clone(), bool(is_causal), eff_scale)
        )
        return orig_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    model = transformers.ViTModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa").eval()
    torch.nn.functional.scaled_dot_product_attention = spy
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            model(pixel_values=torch.randn(1, 3, 224, 224))
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig_sdpa

    assert captured, "No SDPA calls were intercepted; check attn_implementation='sdpa'"
    return captured


@pytest.fixture(scope="module")
def vit_sdpa_inputs():
    return _capture_vit_sdpa_inputs()


def _reference(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=False, scale=scale
    )


def _run_tt(device, q, k, v, scale, fp32_dest_acc_en):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=False, scale=scale, compute_kernel_config=compute_kernel_config
    )
    return ttnn.to_torch(out)[:, :, : q.shape[2], :].float()


@pytest.mark.parametrize("layer", WORST_LAYERS)
def test_vit_base_sdpa_prefill_streaming_pcc(device, vit_sdpa_inputs, layer):
    """Real ViT-Base SDPA inputs on the default (streaming) compute path. Fails at 64891006."""
    q, k, v, is_causal, scale = vit_sdpa_inputs[layer]
    assert not is_causal and tuple(q.shape) == (1, 12, 197, 64)

    gt = _reference(q, k, v, scale)
    tt = _run_tt(device, q, k, v, scale, fp32_dest_acc_en=False)

    passed, pcc = comp_pcc(gt, tt, PCC_THRESHOLD)
    logger.info(f"[layer {layer}] streaming-path PCC = {pcc}")
    assert passed, f"layer {layer}: streaming SDPA PCC {pcc} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("layer", WORST_LAYERS)
def test_vit_base_sdpa_prefill_legacy_pcc_control(device, vit_sdpa_inputs, layer):
    """Control: same inputs on the legacy path (fp32_dest_acc_en=True). Passes everywhere."""
    q, k, v, is_causal, scale = vit_sdpa_inputs[layer]

    gt = _reference(q, k, v, scale)
    tt = _run_tt(device, q, k, v, scale, fp32_dest_acc_en=True)

    passed, pcc = comp_pcc(gt, tt, PCC_THRESHOLD)
    logger.info(f"[layer {layer}] legacy-path PCC = {pcc}")
    assert passed, f"layer {layer}: legacy SDPA PCC {pcc} < {PCC_THRESHOLD}"
