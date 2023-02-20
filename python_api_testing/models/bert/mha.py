import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import numpy as np

import ll_buda_bindings.ll_buda_bindings._C as _C
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax
from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax

def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device):
    assert isinstance(num_heads, int) and num_heads > 0

    QProjection = TtLinear(
        hidden_dim, hidden_dim, qw, qb, device
    )
    KProjection = TtLinear(
        hidden_dim, hidden_dim, kw, kb, device
    )
    VProjection = TtLinear(
        hidden_dim, hidden_dim, vw, vb, device
    )

    # Used to scale down the input to the softmax 
    reciprocal_of_sqrt_hidden_dim_tensor = _C.tensor.Tensor(
        [1 / math.sqrt(hidden_dim)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )

    def make_attention_heads(x, num_heads):
        if num_heads == 1:
            return x
        else:
            assert False, "num_heads > 1 not implemented yet"

    def mha_(activation):
        Q = QProjection(activation)
        K = KProjection(activation)
        V = VProjection(activation)
        K_T = _C.tensor.transpose(K)

        Q_heads = make_attention_heads(Q, num_heads)
        K_heads = make_attention_heads(K_T, num_heads)
        V_heads = make_attention_heads(V, num_heads)

        # Attention scores computation
        attention_score_input = _C.tensor.bcast(
            _C.tensor.matmul(Q_heads, K_heads),
            reciprocal_of_sqrt_hidden_dim_tensor,
            _C.tensor.BcastOpMath.MUL,
            _C.tensor.BcastOpDim.HW
        )
        attention_scores = softmax(attention_score_input)

        # Apply attention to value matrix
        weighted_activation = _C.tensor.matmul(attention_scores, V_heads)
        return weighted_activation

    return mha_

class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        qw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.query.weight"])
        qb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.query.bias"])
        kw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.key.weight"])
        kb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.key.bias"])
        vw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.value.weight"])
        vb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.value.bias"])

        # Hidden dim
        hidden_dim = qw.shape[-1]

        # Tilized
        parameters = [
            tilize_to_list(qw),
            tilize_to_list(qb),
            tilize_to_list(kw),
            tilize_to_list(kb),
            tilize_to_list(vw),
            tilize_to_list(vb)
        ]

        self.mha = mha(*parameters, hidden_dim, 1, device)
    
    def forward(self, activation):
        return self.mha(activation)

class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()
    
    def forward(self, x):
        
        return self.mha(x)[0]

    
def run_mha_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(hugging_face_reference_model.state_dict(), device)
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(1, 1, 128, 128) * 2) - 1

    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    tt_mha_input = tilize_to_list(pad_activation(mha_input))
    tt_mha_input = _C.tensor.Tensor(tt_mha_input, mha_input.shape, _C.tensor.DataFormat.FLOAT32,  _C.tensor.Layout.TILE, device)

    tt_out = tt_mha_model(tt_mha_input).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))
    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

if __name__ == "__main__":
    # Initialize the device
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_mha_inference()
    _C.device.CloseDevice(device)