# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_compute_axial_cis_real():
    """Verify real-valued decomposition matches complex version."""
    from sam3.model.vitdet import compute_axial_cis as ref_compute
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_rope import compute_axial_cis_real

    dim, end_x, end_y = 64, 24, 24  # head_dim=64, window_size=24

    ref_cis = ref_compute(dim, end_x, end_y)  # complex tensor
    ref_cos = ref_cis.real.float()
    ref_sin = ref_cis.imag.float()

    our_cos, our_sin = compute_axial_cis_real(dim, end_x, end_y)

    assert torch.allclose(our_cos, ref_cos, atol=1e-6)
    assert torch.allclose(our_sin, ref_sin, atol=1e-6)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_apply_rotary_enc_tt(device, reset_seeds):
    """Verify ttnn RoPE matches PyTorch reference."""
    from sam3.model.vitdet import compute_axial_cis, apply_rotary_enc as ref_apply
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_rope import (
        compute_axial_cis_real,
        apply_rotary_enc_tt,
    )

    B, num_heads, seq_len, head_dim = 1, 16, 576, 64  # 24*24=576 for window attention

    torch.manual_seed(42)
    xq = torch.randn(B, num_heads, seq_len, head_dim)
    xk = torch.randn(B, num_heads, seq_len, head_dim)

    # Reference
    freqs_cis = compute_axial_cis(head_dim, 24, 24)
    ref_q, ref_k = ref_apply(xq, xk, freqs_cis)

    # Our implementation
    freqs_cos, freqs_sin = compute_axial_cis_real(head_dim, 24, 24)

    tt_xq = ttnn.from_torch(xq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_xk = ttnn.from_torch(xk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_cos = ttnn.from_torch(
        freqs_cos.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sin = ttnn.from_torch(
        freqs_sin.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_q_out, tt_k_out = apply_rotary_enc_tt(tt_xq, tt_xk, tt_cos, tt_sin)

    q_out = ttnn.to_torch(tt_q_out)
    k_out = ttnn.to_torch(tt_k_out)

    assert_with_pcc(ref_q.float(), q_out.float(), 0.99)
    assert_with_pcc(ref_k.float(), k_out.float(), 0.99)
