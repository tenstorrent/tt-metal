from abc import abstractmethod
import torch
from transformers import BertForQuestionAnswering

from gpai import gpai
from python_api_testing.models.bert.embeddings import PytorchEmbeddings
from python_api_testing.models.bert.mha import TtMultiHeadAttentionModel
from python_api_testing.models.bert.ffn import TtFeedForwardModel
from python_api_testing.models.bert.bert_encoder import TtBertEncoder
from python_api_testing.fused_ops.layernorm import Layernorm
from python_api_testing.fused_ops.add_and_norm import AddAndNorm
from python_api_testing.fused_ops.linear import Linear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax

class TtBertShared(torch.nn.Module):
    @abstractmethod
    def __init__(self, num_heads, num_encoders, hugging_face_reference_model, device):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        # So far on CPU until we add embeddings support on device
        self.embeddings = PytorchEmbeddings(hugging_face_reference_model)

        self.encoders = torch.nn.Sequential(*[TtBertEncoder(num_heads, encoder_idx, state_dict, device) for encoder_idx in range(num_encoders)])
        self.device = device

    @abstractmethod
    def forward(self, x):
        embeddings = self.embeddings(x)
        # Convert to ll buda tensor
        tt_embeddings = tilize_to_list(pad_activation(embeddings))
        tt_embeddings = gpai.tensor.Tensor(tt_embeddings, (embeddings.shape[0], 1, embeddings.shape[-2], embeddings.shape[-1]), gpai.tensor.DataFormat.FLOAT32,  gpai.tensor.Layout.TILE, self.device)

        encoder_output = self.encoders(tt_embeddings)
        return encoder_output

class TtBertForQuestionAnswering(TtBertShared):
    def __init__(self, num_heads, num_encoders, hugging_face_reference_model, device):
        super().__init__(num_heads, num_encoders, hugging_face_reference_model, device)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        state_dict = hugging_face_reference_model.state_dict()

        num_classes, hidden_size = state_dict["qa_outputs.weight"].shape

        weight = tilize_to_list(pad_weight(state_dict["qa_outputs.weight"]))
        bias   = tilize_to_list(pad_weight(state_dict["qa_outputs.bias"]))

        # QA linear
        self.qa_linear = Linear(32, hidden_size, weight, bias, device)

    def forward(self, x):
        encoder_output = super().forward(x)
        return self.qa_linear(encoder_output)

def run_bert_question_and_answering_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    hugging_face_reference_model.eval()
    tt_bert_model = TtBertForQuestionAnswering(2, 2, hugging_face_reference_model, device)

    batch = 5
    seq_len = 128
    bert_input = torch.arange(seq_len*batch).reshape(batch, seq_len)

    # tt_bert_input = tilize_to_list(pad_activation(bert_input))

    pytorch_out = hugging_face_reference_model(bert_input)
    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    tt_out = tt_bert_model(bert_input).to(host)
    tt_untilized_output = untilize(torch.Tensor(tt_out.data()).reshape(batch, 1, seq_len, -1))

    tt_start_logits = tt_untilized_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_untilized_output[..., :, 1].squeeze(1)

    pytorch_start_logits = pytorch_out.start_logits
    pytorch_end_logits = pytorch_out.end_logits

    start_logit_match = (abs(tt_start_logits - pytorch_start_logits) < 0.1).all().item()
    if not start_logit_match:
        print("Start logits don't match")

    end_logit_match = (abs(tt_end_logits - pytorch_end_logits) < 0.1).all().item()
    if not end_logit_match:
        print("End logits don't match")

    assert start_logit_match and end_logit_match, "At least one of start or end logits don't match to an absolute difference of 0.1"

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    run_bert_question_and_answering_inference()
    gpai.device.CloseDevice(device)
