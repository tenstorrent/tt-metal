import torch
from transformers import BertForQuestionAnswering

from gpai import gpai
from python_api_testing.models.bert.mha import TtMultiHeadAttentionModel
from python_api_testing.models.bert.ffn import TtFeedForwardModel
from python_api_testing.fused_ops.layernorm import Layernorm
from python_api_testing.fused_ops.add_and_norm import AddAndNorm
from python_api_testing.fused_ops.linear import Linear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax


class TtBertEncoder(torch.nn.Module):
    def __init__(self, num_heads, encoder_idx, state_dict, device):
        super().__init__()
        hidden_dim = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"]).shape[-1]

        # MHA part
        self.mha = TtMultiHeadAttentionModel(num_heads, encoder_idx, state_dict, device)
        attention_output_weight = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.weight"]))
        attention_output_bias = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.bias"]))
        self.attention_output = Linear(hidden_dim, hidden_dim, attention_output_weight, attention_output_bias, device)

        # MHA layernorm part
        mha_gamma = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.weight"]))
        mha_beta = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.bias"]))
        self.mha_add_and_norm = AddAndNorm(mha_gamma, mha_beta, 1e-12, 128, 128, device)

        # FFN part
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device)

        # FFN layernorm part
        ffn_gamma = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.weight"]))
        ffn_beta = tilize_to_list(pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.bias"]))
        self.ffn_add_and_norm = AddAndNorm(ffn_gamma, ffn_beta, 1e-12, 128, 128, device)

    def forward(self, activation):
        mha_out = self.attention_output(self.mha(activation))
        mha_out_add_and_norm = self.mha_add_and_norm(activation, mha_out)
        ffn_out = self.ffn(mha_out_add_and_norm)
        ffn_out_add_and_norm = self.ffn_add_and_norm(mha_out_add_and_norm, ffn_out)
        return ffn_out_add_and_norm

class PytorchBertEncoder(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.bert_encoder = hugging_face_reference_model.bert.encoder.layer[0]

    def forward(self, x):
        return self.bert_encoder(x)[0]


def run_bert_encoder_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    tt_bert_encoder_model = TtBertEncoder(2, 0, hugging_face_reference_model.state_dict(), device)
    pytorch_bert_model = PytorchBertEncoder(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    bert_encoder_input = (torch.rand(1, 1, 128, 128) * 2) - 1

    pytorch_out = pytorch_bert_model(bert_encoder_input.squeeze(1)).unsqueeze(1)

    tt_bert_encoder_input = tilize_to_list(pad_activation(bert_encoder_input))
    tt_bert_encoder_input = gpai.tensor.Tensor(tt_bert_encoder_input, bert_encoder_input.shape, gpai.tensor.DataFormat.FLOAT32,  gpai.tensor.Layout.TILE, device)

    tt_out = tt_bert_encoder_model(tt_bert_encoder_input).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))
    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

if __name__ == "__main__":
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    run_bert_encoder_inference()
    gpai.device.CloseDevice(device)
