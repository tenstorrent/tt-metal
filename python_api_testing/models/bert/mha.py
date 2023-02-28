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
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
# from utility_functions import get_FR, set_FR
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

    def make_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            # ref code from modeling_bert.py:
            #    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            #        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            #        x = x.view(new_x_shape)
            #        return x.permute(0, 2, 1, 3)

            untilized_x = _C.tensor.untilize(x)
            reshaped_unt = _C.tensor.reshape(untilized_x, x.shape()[0], x.shape()[2], num_heads, x.shape()[3] // num_heads)

            # N, 128, 2, 64
            transposed = _C.tensor.transpose_hc_rm(reshaped_unt)
            # N, 2, 128, 64
            retilized = _C.tensor.tilize(transposed)
            return retilized

    def unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            """
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            debug_state["context_reshaped"] = context_layer.clone()

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            """
            untilized_x = _C.tensor.untilize(x)
            ctx = _C.tensor.transpose_hc_rm(untilized_x)
            ushape = ctx.shape()
            reshaped = _C.tensor.reshape(ctx, ushape[0], 1, ushape[1], ushape[2]*ushape[3])
            retval = _C.tensor.tilize(reshaped)
            return retval

    def multiply_by_sqrt_hidden_dim(x):
        return _C.tensor.bcast(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            _C.tensor.BcastOpMath.MUL,
            _C.tensor.BcastOpDim.HW
        )

    def mha_(activation):
        Q = QProjection(activation)
        K = KProjection(activation)
        V = VProjection(activation)

        Q_heads = make_attention_heads(Q)
        K_heads = make_attention_heads(K)
        V_heads = make_attention_heads(V)
        K_T_heads = _C.tensor.transpose(K_heads)

        qkt = _C.tensor.bmm(Q_heads, K_T_heads)

        # Attention scores computation
        N, C, H, W = qkt.shape() # Need to reshape right now since multi-C not supported for broadcast yet
        new_shape = [N, 1, C*H, W]
        _C.tensor.reshape(qkt, *new_shape)
        attention_score_input = multiply_by_sqrt_hidden_dim(qkt)
        attention_scores = softmax(attention_score_input)
        _C.tensor.reshape(attention_scores, N, C, H, W) # Reshape back to original shape

        # Apply attention to value matrix
        weighted_activation = _C.tensor.bmm(attention_scores, V_heads)
        return unmake_attention_heads(weighted_activation) # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]

    return mha_

class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, num_heads, encoder_idx, state_dict, device):
        super().__init__()
        qw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"])
        qb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"])
        kw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"])
        kb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"])
        vw = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"])
        vb = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"])

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

        self.mha = mha(*parameters, hidden_dim, num_heads, device)

    def forward(self, activation):
        result = self.mha(activation)
        return result

class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x):
        result = self.mha(x)[0]
        return result


def run_mha_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(2, 0, hugging_face_reference_model.state_dict(), device)
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(2, 1, 128, 128) * 2) - 1

    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    tt_mha_input = tilize_to_list(pad_activation(mha_input))
    tt_mha_input = _C.tensor.Tensor(tt_mha_input, mha_input.shape, _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)

    tt_out = tt_mha_model(tt_mha_input).to(host)
    tt_out1 = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    print_diff_argmax(pytorch_out, tt_out1)
    assert np.allclose(pytorch_out.detach().numpy(), tt_out1, 1e-5, 0.17)

if __name__ == "__main__":
    # Initialize the device
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_mha_inference()
    _C.device.CloseDevice(device)
