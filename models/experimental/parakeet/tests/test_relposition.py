"""
PCC test: PyTorch NeMo RelPositionMultiHeadAttention
vs TTNN RelPositionMultiHeadAttentionTTNN
"""

import pytest
import torch
import ttnn
import nemo.collections.asr as nemo_asr

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.parakeet.tt.tt_relposition import RelPositionMultiHeadAttentionTTNN


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_relpos_mha_pcc(device):
    torch.manual_seed(0)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2",
        map_location="cpu",
    )

    encoder = asr_model.encoder
    layer = encoder.layers[0]
    torch_mha = layer.self_attn

    torch_mha = torch_mha.to(torch.bfloat16)
    torch_mha.eval()

    d_model = torch_mha.linear_q.weight.shape[1]
    num_heads = torch_mha.h
    dim_head = d_model // num_heads

    class Config:
        def __init__(self):
            self.num_heads = num_heads
            self.dim_head = dim_head
            self.context_size = 128

    config = Config()

    tt_mha = RelPositionMultiHeadAttentionTTNN(device, config)

    tt_mha.prepare_weights(
        torch_mha.linear_q.weight.data.clone(),
        torch_mha.linear_k.weight.data.clone(),
        torch_mha.linear_v.weight.data.clone(),
        torch_mha.linear_q.bias.data.clone()
        if torch_mha.linear_q.bias is not None
        else torch.zeros(d_model, dtype=torch.bfloat16),
        torch_mha.linear_k.bias.data.clone()
        if torch_mha.linear_k.bias is not None
        else torch.zeros(d_model, dtype=torch.bfloat16),
        torch_mha.linear_v.bias.data.clone()
        if torch_mha.linear_v.bias is not None
        else torch.zeros(d_model, dtype=torch.bfloat16),
        torch_mha.linear_out.weight.data.clone(),
        torch_mha.linear_out.bias.data.clone()
        if torch_mha.linear_out.bias is not None
        else torch.zeros(d_model, dtype=torch.bfloat16),
        torch_mha.linear_pos.weight.data.clone(),
        torch_mha.pos_bias_u.data.clone(),
        torch_mha.pos_bias_v.data.clone(),
    )

    hidden_states = torch.randn(1, 32, d_model, dtype=torch.bfloat16)

    with torch.no_grad():
        _, pos_emb = encoder.pos_enc(hidden_states)
        pos_emb = pos_emb.to(torch.bfloat16)
        ref_out = torch_mha(hidden_states, hidden_states, hidden_states, None, pos_emb)

    tt_hidden = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device)
    tt_pos = ttnn.from_torch(pos_emb, dtype=ttnn.bfloat16, device=device)

    tt_out = tt_mha.forward(tt_hidden, tt_hidden, tt_hidden, None, tt_pos)

    tt_out_torch = ttnn.to_torch(tt_out)
    if tt_out_torch.shape != ref_out.shape:
        tt_out_torch = tt_out_torch[:, : ref_out.size(1), :]
    passed, msg = check_with_pcc(ref_out.float(), tt_out_torch.float(), pcc=0.95)
    print(f"RelPosition MHA PCC: {msg}")

    passed, msg = check_with_pcc(ref_out.float(), tt_out_torch.float(), pcc=0.95)

    assert passed, f"PCC failed: {msg}"
