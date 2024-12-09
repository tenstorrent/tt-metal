import pytest
import torch
import ttnn
from models.experimental.mochi.common import compute_metrics

from genmo.mochi_preview.dit.joint_model.temporal_rope import apply_rotary_emb_qk_real
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3.tt.llama_common import get_rot_transformation_mat


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


@pytest.mark.parametrize("batch, seq_len", [(1, 256), (1, 45056)])
@pytest.mark.parametrize("num_heads, head_dim", [(3, 128), (24, 128)])
def test_apply_rotary_emb_qk_real(device, batch, seq_len, num_heads, head_dim):
    # Create input tensors
    xqk = torch.randn(batch, seq_len, num_heads, head_dim)
    freqs_cos = torch.randn(seq_len, num_heads, head_dim // 2)
    freqs_sin = torch.randn(seq_len, num_heads, head_dim // 2)

    # Run reference implementation
    gt = apply_rotary_emb_qk_real(xqk, freqs_cos, freqs_sin)

    # Run ttnn implementation

    # Get inputs to have NH in -3 dimension
    xqk_reshape = xqk.permute(0, 2, 1, 3)
    freqs_cos_reshape = freqs_cos.permute(1, 0, 2).unsqueeze(0)
    freqs_sin_reshape = freqs_sin.permute(1, 0, 2).unsqueeze(0)
    cos_stacked, sin_stacked = stack_cos_sin(freqs_cos_reshape, freqs_sin_reshape)

    trans_mat = get_rot_transformation_mat(None)

    xqk_tt = ttnn.from_torch(xqk_reshape, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    freqs_cos_tt = ttnn.from_torch(cos_stacked, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    freqs_sin_tt = ttnn.from_torch(sin_stacked, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    out_tt = ttnn.experimental.rotary_embedding_llama(xqk_tt, freqs_cos_tt, freqs_sin_tt, trans_mat_tt)
    out = ttnn.to_torch(out_tt)

    # Reshape back to match shape of gt
    out = out.permute(0, 2, 1, 3)

    # Compute accuracy metrics
    pcc, mse, mae = compute_metrics(gt, out)

    # Check if model meets requirements
    print(f"PCC={pcc}, MSE={mse}, MAE={mae}")
    pcc_required = 0.99
    assert pcc >= pcc_required, f"Output does not meet PCC requirement {pcc_required}: PCC={pcc}, MSE={mse}, MAE={mae}"
