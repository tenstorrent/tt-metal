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


class LlamaShared(torch.nn.Module):
    @abstractmethod
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    def __init__(self, device, state_dict, base_url, max_position_embeddings, config, num_decoders):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings

        print(f"Decoder: {num_decoders}")

        # So far on CPU until we add embeddings support on device
        # self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(state_dict[f"model.embed_tokens.weight"])

        # stack all decoders
        TtLlamaDecoderLayer(self.device, self.state_dict, self.base_url, 0, self.max_position_embeddings, config)
        print("Test 2a")
        self.decoders = torch.nn.Sequential(*[TtLlamaDecoderLayer(self.device, self.state_dict, self.base_url, decoder_idx, self.max_position_embeddings, config) for decoder_idx in range(num_decoders)])
        print("Test 3a")

        # add final normalization layer
        self.layer_num = None
        self.layer_position = 'norm'
        self.final_layernorm = TtLlamaRMSNorm(
            device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )

        self.device = device

    @abstractmethod
    def forward(self, x):
        embeddings = self.embeddings(x)
        print("2 =====================================")
        # Convert to ll buda tensor
        pad_embeddings = pad_activation(embeddings)
        tt_embeddings = ttl.tensor.Tensor(pad_embeddings.reshape(-1).tolist(), (pad_embeddings.shape[0], 1, pad_embeddings.shape[-2], pad_embeddings.shape[-1]), ttl.tensor.DataType.BFLOAT16,  ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE)
        tt_embeddings = tt_embeddings.to(self.device)

        # tt_embeddings = tilize_to_list(pad_activation(embeddings))
        # tt_embeddings = ttl.tensor.Tensor(tt_embeddings, (embeddings.shape[0], 1, embeddings.shape[-2], embeddings.shape[-1]), ttl.tensor.DataType.BFLOAT16,  ttl.tensor.Layout.TILE, self.device)
        # apply decoders
        print("3 =====================================")
        past_key_values_length = 0
        print(f"SHAPE: {tt_embeddings.shape()}")
        # sys.exit(0)
        seq_length = tt_embeddings.shape()[2]

        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=None
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        # =================================================

        encoder_output = self.decoders(tt_embeddings, position_ids)
        # apply final norm layer
        encoder_output = self.final_layernorm(encoder_output)
        return encoder_output


class TtLlamaModel(LlamaShared):
    def __init__(self, device, state_dict, base_url, max_position_embeddings, config, num_decoders):
        print("before =====================================")
        # config, num_decoders, state_dict, device
        super().__init__(device, state_dict, base_url, max_position_embeddings, config, num_decoders)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict
        print("Prvo =====================================")

        num_classes, hidden_size = state_dict["lm_head.weight"].shape

        weight = tilize_to_list(pad_weight(state_dict["lm_head.weight"]))
        bias = None

    def forward(self, x):
        encoder_output = super().forward(x)
        return encoder_output


def run_llama_inference():

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # get only llama model (no linear layer at the end)
    llama_model = hugging_face_reference_model.get_decoder()

    batch = 4
    seq_len = 32
    if 1:
        llama_input = torch.arange(seq_len*batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)]*batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    pytorch_out = llama_model(llama_input)
    pytorch_out = pytorch_out.last_hidden_state
    print("PyTorch model finished")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    num_decoders = 16
    base_url = "model.layers"
    max_position_embeddings = 2048

    tt_llama_model = TtLlamaModel(device, state_dict, base_url, max_position_embeddings, configuration, num_decoders)
    print("TT llama model finished")

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
