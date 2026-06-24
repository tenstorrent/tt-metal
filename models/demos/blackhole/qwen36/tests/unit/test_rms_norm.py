# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3.5-9B zero-centered RMSNorm (``tt/rms_norm.py``).

Qwen3.5 uses zero-centered RMSNorm for *every* norm:

    out = x * rsqrt(mean(x^2) + eps) * (1 + raw_weight)

The ``+1`` is folded into the stored weight at load time (the "+1 landmine"),
so ``rms_norm_ttnn`` itself runs the *standard* fused op against a pre-offset
weight. We validate it against the torch reference across the shapes the module
docstring guarantees: the hidden-dim norm ``[1, 1, 4096]`` and the per-head
q/k-norm shapes ``[1, 1, 16, 256]`` and ``[1, 1, 32, 128]`` (rms_norm reduces
over the last dim).

Requires a Blackhole P150 device.
Run: pytest models/demos/blackhole/qwen36/tests/unit/test_rms_norm.py -v
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc
from models.demos.blackhole.qwen36.tt.rms_norm import rms_norm_ttnn

pytestmark = run_for_blackhole()
EPS = 1e-6
PCC_THRESHOLD = 0.999  # measured ~0.99999 across all shapes on Blackhole


def torch_rms_norm(x: torch.Tensor, stored_weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Reference fused RMSNorm: reduce over the last dim, scale by stored_weight."""
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps) * stored_weight.float()).to(torch.bfloat16)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def _run_rms_norm(device, x, stored_weight, eps=EPS):
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w_t = ttnn.from_torch(stored_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm_ttnn(x_t, w_t, eps=eps)
    return ttnn.to_torch(out).reshape(x.shape)


class TestRMSNorm:
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 4096),  # hidden-dim norm (input_layernorm / post_attention_layernorm)
            (1, 4, 4096),  # hidden-dim norm, short prefill
            (1, 1, 16, 256),  # q_norm: 16 heads x head_dim 256
            (1, 1, 32, 128),  # k_norm style: 32 heads x 128
        ],
        ids=["hidden_1tok", "hidden_4tok", "qnorm_16x256", "knorm_32x128"],
    )
    def test_pcc_vs_torch(self, device, shape):
        """RMSNorm matches the torch reference across hidden- and head-dim shapes."""
        torch.manual_seed(0)
        last_dim = shape[-1]
        x = torch.randn(*shape, dtype=torch.bfloat16)
        # Stored weight is already +1-offset: centered around 1.0 with small spread.
        stored_weight = (1.0 + 0.1 * torch.randn(last_dim)).to(torch.bfloat16)

        ref = torch_rms_norm(x, stored_weight)
        out = _run_rms_norm(device, x, stored_weight)

        assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
        pcc = compute_pcc(ref, out)
        logger.info(f"RMSNorm {shape}: PCC={pcc:.6f}")
        assert pcc > PCC_THRESHOLD, f"RMSNorm PCC too low for {shape}: {pcc}"

    def test_zero_centered_plus_one_semantics(self, device):
        """The stored weight carries the +1 offset; confirm scaling uses (1 + raw)
        and that dropping the +1 would give a materially different result."""
        torch.manual_seed(1)
        x = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
        raw_weight = (0.1 * torch.randn(4096)).to(torch.bfloat16)
        stored_weight = (1.0 + raw_weight).to(torch.bfloat16)

        out = _run_rms_norm(device, x, stored_weight)
        ref_with_plus1 = torch_rms_norm(x, stored_weight)
        ref_without_plus1 = torch_rms_norm(x, raw_weight)

        pcc_correct = compute_pcc(ref_with_plus1, out)
        pcc_wrong = compute_pcc(ref_without_plus1, out)
        logger.info(f"+1 semantics: PCC(stored)={pcc_correct:.6f}, PCC(raw-only)={pcc_wrong:.6f}")

        assert pcc_correct > PCC_THRESHOLD, f"matched (1+raw) reference poorly: {pcc_correct}"
        # The raw-only (missing +1) reference must NOT match — the offset is real.
        assert pcc_wrong < pcc_correct - 0.05, "raw-only weight matched too well; +1 offset not exercised"

    def test_identity_weight_normalizes(self, device):
        """With a stored weight of all-ones, output rows should have ~unit RMS."""
        torch.manual_seed(2)
        x = torch.randn(1, 8, 4096, dtype=torch.bfloat16)
        stored_weight = torch.ones(4096, dtype=torch.bfloat16)

        out = _run_rms_norm(device, x, stored_weight).float()
        rms = out.pow(2).mean(-1).sqrt()  # per-row RMS over the hidden dim
        logger.info(f"identity-weight per-row RMS range: [{rms.min():.4f}, {rms.max():.4f}]")
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05), f"rows not unit-RMS: {rms}"
