# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import (
    BloomForQuestionAnswering,
    AutoTokenizer,
    BloomTokenizerFast,
    pipeline,
)
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import models.experimental.bloom_old.tt.bloom_qa as bloom_qa


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_bloom_qa_inference(device):
    torch.manual_seed(0)

    model_name = "bigscience/bloom-560m"
    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_qa = bloom_qa.TtBloomForQuestionAnswering(config, state_dict, device)
    pt_bloom_qa = hugging_bloom_reference_model

    # Prepare input
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    nlp = pipeline("question-answering", model=hugging_bloom_reference_model, tokenizer=tokenizer)
    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()

    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")

    input_ids = pad_input_32(tokenized.input_ids, config.pad_token_id)
    attention_mask = pad_input_32(tokenized.attention_mask, 0)

    pt_out = pt_bloom_qa.forward(input_ids=input_ids)  # , attention_mask=attention_mask)
    print("PT finished")

    tt_out = tt_bloom_qa.forward(device, input_ids=input_ids)  # , attention_mask=attention_mask)
    print("TT finished")

    pt_start_logits = pt_out[0]  # start_logits
    pt_end_logits = pt_out[1]

    tt_start_logits = tt_out[0]
    tt_start_logits = tt_start_logits.squeeze(0)

    tt_end_logits = tt_out[1]
    tt_end_logits = tt_end_logits.squeeze(0)

    # tt_res = {
    #    "start": tt_start_logits,
    #    "end": tt_end_logits,
    #    "example": single_input["example"],
    #    **single_input["inputs"],
    # }

    # tt_answer = nlp.postprocess([tt_res], **postprocess_params)['answer']

    print_diff_argmax(pt_start_logits, tt_start_logits)
    does_pass, pcc_message = comp_pcc(pt_start_logits, tt_start_logits, 0.91)

    print(comp_allclose(pt_start_logits, tt_start_logits))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_qa: Passed!")
    else:
        logger.warning("bloom_qa: Failed!")

    assert does_pass


def test_bloom_qa(device):
    run_bloom_qa_inference(device)
