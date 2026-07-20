# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the ``return_lse`` SDPA kernel extension (design task T6).

``ttnn.transformer.scaled_dot_product_attention(..., return_lse=True)`` returns
``(output, lse)`` where ``lse`` is the per-row log-sum-exp ``scale*m_raw + log(l)``
== ``logsumexp(scale * Q @ Kᵀ [+ mask], dim=-1)`` as an fp32 tensor
``[b, nqh, s, 1]``. This is the statistic the Phase-2 paged-prefix + new-chunk
merge (task T7) consumes.

Two device-gated checks (gated exactly like ``tests/test_attention_merge.py``:
``DG_RUN_DEVICE=1`` + the shared module device, so the default CPU suite never
opens a device):

1. ``test_return_lse_matches_logsumexp`` — the emitted LSE matches the torch
   reference ``logsumexp(scale * Q @ Kᵀ [+ causal mask])`` at an fp32-appropriate
   tolerance (bf16 inputs, fp32 accumulation).
2. ``test_return_lse_false_is_bit_identical`` — the ``return_lse=True`` output is
   **bit-identical** to the ``return_lse=False`` output. The LSE emit must not
   perturb the attention output for any existing caller; the emit is a
   compile-time-guarded add-on that only writes a second tensor.

The kernel emit lives on the STREAMING compute path only, so the tests pin
``fp32_dest_acc_en=False`` (the config the program factory routes to streaming).
"""

import os

import pytest
import torch


# --- shared torch reference ------------------------------------------------


def _reference_lse(q, k, scale, is_causal):
    """torch fp32 reference for the emitted LSE.

    Args:
        q: ``[B, H, S, D]`` fp32.
        k: ``[B, H, S, D]`` fp32.
        scale: attention scale folded into the softmax exponent.
        is_causal: apply the lower-triangular causal mask before logsumexp.

    Returns:
        ``[B, H, S, 1]`` fp32 = ``logsumexp(scale * Q @ Kᵀ [+ causal mask], dim=-1)``.
    """
    scores = scale * torch.matmul(q, k.transpose(-1, -2))  # [B, H, Sq, Sk]
    if is_causal:
        sq, sk = scores.shape[-2], scores.shape[-1]
        causal = torch.tril(torch.ones(sq, sk, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.logsumexp(scores, dim=-1, keepdim=True)


# --- device-gated checks ---------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("head_dim", [128, 512], ids=["hd128", "hd512"])
def test_return_lse_matches_logsumexp(device, is_causal, head_dim):
    """Emitted LSE ≈ torch logsumexp(scale * Q @ Kᵀ [+ mask])."""
    import ttnn
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(47460 + head_dim + int(is_causal))

    b, num_heads, seq_len = 1, 2, 512
    scale = 1.0 / (head_dim**0.5)

    q = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=32,
        k_chunk_size=32,
    )
    # fp32_dest_acc_en=False routes to the streaming compute path, where return_lse is implemented.
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    out_tt, lse_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        return_lse=True,
    )

    lse = ttnn.to_torch(lse_tt).to(torch.float32)  # [B, H, S, 1]
    assert lse.shape[-1] == 1, f"LSE last dim must be 1, got {tuple(lse.shape)}"

    lse_ref = _reference_lse(q.to(torch.float32), k.to(torch.float32), scale, is_causal)

    # LSE feeds exp() weight differences in merge_attention_partials, so ABSOLUTE error is the
    # meaningful gate (a 0.06 LSE error → a ~6% softmax-weight error, within the bf16 #48291 floor).
    # PCC is a poor metric here: the noncausal LSE clusters tightly near log(#keys) (low variance),
    # so bf16 noise deflates PCC even when abs error is small — keep it only as a loose sanity floor.
    max_abs = (lse - lse_ref).abs().max().item()
    passing, pcc = comp_pcc(lse_ref, lse, 0.98)
    torch.testing.assert_close(lse, lse_ref, atol=6e-2, rtol=6e-2)
    assert passing, f"LSE PCC sanity floor 0.98 not met: pcc={pcc}, max_abs={max_abs}"

    out_tt.deallocate(True)
    lse_tt.deallocate(True)


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_return_lse_false_is_bit_identical(device, is_causal):
    """return_lse=True output is bit-identical to the return_lse=False output.

    Guards the top correctness requirement: the LSE emit must not change the
    attention output for any existing caller.
    """
    import ttnn

    torch.manual_seed(47461 + int(is_causal))

    b, num_heads, seq_len, head_dim = 1, 2, 512, 128
    scale = 1.0 / (head_dim**0.5)

    q = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=32,
        k_chunk_size=32,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    kwargs = dict(
        is_causal=is_causal,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    # return_lse=False must be byte-identical to today (the "current call").
    out_false_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, return_lse=False, **kwargs)
    out_true_tt, lse_tt = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, return_lse=True, **kwargs)

    out_false = ttnn.to_torch(out_false_tt)
    out_true = ttnn.to_torch(out_true_tt)

    assert torch.equal(out_false, out_true), "return_lse=True perturbed the attention output (must be bit-identical)"

    out_false_tt.deallocate(True)
    out_true_tt.deallocate(True)
    lse_tt.deallocate(True)
