# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest


from models.experimental.llama_old.llama_utils import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from models.utility_functions import (
    comp_allclose,
    comp_pcc,
    tt_to_torch_tensor,
)
from models.experimental.llama_old.tt.llama_causallm import TtLlamaForCausalLM


def run_test_llamaCausallm_inference(
    device,
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    num_decoders,
    max_position_embeddings,
    on_weka,
    pcc,
):
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
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    pytorch_out = hugging_face_reference_model(llama_input)
    pytorch_out = pytorch_out.logits

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    num_decoders = num_decoders
    base_url = "model.layers"
    max_position_embeddings = max_position_embeddings

    tt_causal_llama_model = TtLlamaForCausalLM(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders,
    )

    tt_out = tt_causal_llama_model(llama_input).cpu()
    tt_out = tt_to_torch_tensor(tt_out)
    tt_out = tt_out.squeeze(1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama CausalLm Model Passed!")
    else:
        logger.warning("Llama CausalLm Model Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka, pcc",
    (
        (
            "baffo32/decapoda-research-llama-7B-hf",
            "hf-internal-testing/llama-tokenizer",
            4,
            128,
            32,
            2048,
            False,
            0.98,
        ),
    ),
)
def test_llamaCausallm_inference(
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    num_decoders,
    max_position_embeddings,
    on_weka,
    pcc,
    device,
):
    run_test_llamaCausallm_inference(
        device,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        num_decoders,
        max_position_embeddings,
        on_weka,
        pcc,
    )
