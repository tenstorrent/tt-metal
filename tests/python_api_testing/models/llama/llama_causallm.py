from abc import abstractmethod
import torch
from transformers import BertForQuestionAnswering

import sys
from pathlib import Path
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from libs import tt_lib as ttl
from python_api_testing.models.llama.llama_utils import *
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP
from python_api_testing.models.llama.llama_attention import TtLlamaAttention
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama.llama_embeddings import PytorchEmbeddings
from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel

from python_api_testing.fused_ops.linear import Linear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax
from utility_functions import enable_binary_cache, enable_compile_cache, get_compile_cache_enabled, get_binary_cache_enabled


class LlamaForCausalLMShared(torch.nn.Module):
    @abstractmethod
    def __init__(self, config, num_decoders, state_dict, device):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict

        # So far on CPU until we add embeddings support on device
        # self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(state_dict[f"model.embed_tokens.weight"])

        # stack all decoders
        self.decoders = torch.nn.Sequential(*[TtLlamaDecoderLayer(device, self.state_dict, decoder_idx, config) for decoder_idx in range(num_decoders)])

        # add final normalization layer
        self.layer_num = None
        self.layer_position = 'norm'
        self.final_layernorm = TtLlamaRMSNorm(
            device,
            state_dict=self.state_dict,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )

        self.device = device

    @abstractmethod
    def forward(self, x):
        embeddings = self.embeddings(x)
        # Convert to ll buda tensor
        tt_embeddings = tilize_to_list(pad_activation(embeddings))
        tt_embeddings = ttl.tensor.Tensor(tt_embeddings, (embeddings.shape[0], 1, embeddings.shape[-2], embeddings.shape[-1]), ttl.tensor.DataType.BFLOAT16,  ttl.tensor.Layout.TILE, self.device)
        # apply decoders
        encoder_output = self.decoders(tt_embeddings)
        # apply final norm layer
        encoder_output = self.final_layernorm(encoder_output)
        return encoder_output


class LlamaForCausalLM(LlamaForCausalLMShared):
    def __init__(self, config, num_decoders, state_dict, device):
        super().__init__(config, num_decoders, state_dict, device)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict # hugging_face_reference_model.state_dict()

        num_classes, hidden_size = state_dict["lm_head.weight"].shape

        weight = tilize_to_list(pad_weight(state_dict["lm_head.weight"]))
        bias = None

        # CausalLM linear
        self.CausalLM_linear = Linear(hidden_size, config.vocab_size, weight, bias, device)

    def forward(self, x):
        encoder_output = super().forward(x)
        return self.CausalLM_linear(encoder_output)


def run_llama_inference():

    tokenizer = AutoTokenizer.from_pretrained("aleksickx/llama-7b-hf")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("aleksickx/llama-7b-hf")
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    batch = 4
    seq_len = 32
    if 1:
        llama_input = torch.arange(seq_len*batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)]*batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    pytorch_out = hugging_face_reference_model(llama_input)
    pytorch_out = pytorch_out.logits
    print("PyTorch model finished")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    num_decoders = 32
    tt_llama_model = LlamaForCausalLM(configuration, num_decoders, state_dict, device)
    print("Tenstorrent llama model finished")

    tt_out = tt_llama_model(llama_input).to(host)
    tt_untilized_output = untilize(torch.Tensor(tt_out.data()).reshape(batch, 1, seq_len, -1)).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    # check correlation ceofficient
    print_corr_coef(pytorch_out, tt_untilized_output)

    logits_diff = (abs(pytorch_out - tt_untilized_output) < 0.1).all().item()

    if not logits_diff:
        print("logits don't match")
    else:
        print("logits match")

    assert logits_diff, "At least one of logits don't match to an absolute difference of 0.1"


if __name__ == "__main__":
    # TODO(AP): currently necessary, otherwise get bit discrepancies
    torch.manual_seed(1234)
    # Initialize the device
    #enable_binary_cache()
    #enable_compile_cache()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_llama_inference()
    ttl.device.CloseDevice(device)
