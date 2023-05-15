from abc import abstractmethod
import torch

import sys
from pathlib import Path
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
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
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_causallm import TtLlamaForCausalLM


def run_test_llamaCausallm_inference(device, host, model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka, pcc):

    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    batch = batch
    seq_len = seq_len
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
    num_decoders = num_decoders
    base_url = "model.layers"
    max_position_embeddings = max_position_embeddings

    tt_llama_model = TtLlamaForCausalLM(device, state_dict, base_url, max_position_embeddings, configuration, num_decoders)
    print("Tenstorrent llama model finished")

    tt_out = tt_llama_model(llama_input).to(host)
    tt_untilized_output = untilize(torch.Tensor(tt_out.data()).reshape(batch, 1, seq_len, -1)).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_untilized_output))
    print(comp_pcc(pytorch_out, tt_untilized_output))

    passing_pcc, output_pcc = comp_pcc(pytorch_out, tt_untilized_output, pcc)

    assert passing_pcc, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka, pcc",
    (
        ("decapoda-research/llama-7b-hf", "hf-internal-testing/llama-tokenizer", 4, 128, 32, 2048, False, 0.98),
    ),
)
def test_llamaCausallm_inference(model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka, pcc):
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_llamaCausallm_inference(
        device,
        host,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        num_decoders,
        max_position_embeddings,
        on_weka,
        pcc
    )
    ttl.device.CloseDevice(device)
