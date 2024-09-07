# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import networkx as nx
import torch
import transformers

import ttnn
from ttnn.tracer import trace, visualize, get_graph

from models.utility_functions import is_wormhole_b0, is_blackhole

from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
def test_exp():
    with trace():
        tensor = torch.randint(0, 100, (1, 64))
        tensor = torch.exp(tensor)

    visualize(tensor)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
def test_reshape():
    with trace():
        tensor = torch.randint(0, 100, (4, 64))
        tensor = ttnn.from_torch(tensor)
        tensor = ttnn.reshape(tensor, (2, 4, 32))
        tensor = ttnn.to_torch(tensor)

    assert len(get_graph(tensor)) == 4
    visualize(tensor)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("show_modules", [True, False])
def test_torch_bert(show_modules):
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()

    with trace():
        input_tensor = torch.randint(0, 100, (1, 64))
        output = model(input_tensor)

    last_hidden_state = output.last_hidden_state
    visualize(last_hidden_state, show_modules=show_modules)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("show_modules", [True, False])
def test_bloom(show_modules):
    model_name = "bigscience/bloom-560m"
    config = transformers.BloomConfig.from_pretrained(model_name)
    config.use_cache = False
    model = transformers.BloomModel.from_pretrained(model_name, config=config).eval()

    with trace():
        input_tensor = torch.randint(0, 100, (1, 384))
        output = model(input_tensor)

    last_hidden_state = output.last_hidden_state
    graph = last_hidden_state.graph
    assert not list(nx.simple_cycles(graph))
    if show_modules:
        visualize(last_hidden_state, show_modules=show_modules)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("bert", [ttnn_bert, ttnn_optimized_bert])
def test_ttnn_bert(device, use_program_cache, model_name, batch_size, sequence_size, bert):
    config = transformers.BertConfig.from_pretrained(model_name)

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    with trace():
        input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
        torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
        torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
        torch_attention_mask = torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None

        ttnn_bert_inputs = bert.preprocess_inputs(
            input_ids,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            device=device,
        )

        output = bert.bert_for_question_answering(
            config,
            *ttnn_bert_inputs,
            parameters=parameters,
        )
        output = ttnn.from_device(output)

    visualize(output)


@pytest.mark.requires_fast_runtime_mode_off
def test_falcon7b_instruct():
    from functools import partial
    from loguru import logger
    from transformers import FalconConfig, FalconForCausalLM

    model_version = "tiiuae/falcon-7b-instruct"

    logger.info("Initializing tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_version)

    logger.info("Initializing CausalLM Model")
    config = FalconConfig.from_pretrained(model_version)
    config.num_hidden_layers = 2
    model = FalconForCausalLM.from_pretrained(model_version, config=config, device_map="auto").eval()

    def post_process(logits):
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        ids = next_tokens[:, None]
        return ids

    def generate_next_id(model, post_processor, input_ids, kv_cache=None, use_cache=None):
        outputs = model(input_ids, past_key_values=kv_cache, use_cache=use_cache)
        return (
            post_processor(logits=outputs.logits),
            outputs.past_key_values,
        )

    post_processor = partial(post_process)

    batch_size = 1
    num_tokens = 3

    logger.info("Creating inputs")
    prompt_text = ["Write a poem about Valencia"] * batch_size

    logger.info("Tokenizing inputs")
    tokenized_inputs = tokenizer(prompt_text, padding=False, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"]
    generator = partial(generate_next_id, model=model, post_processor=post_processor)

    with trace():
        logger.info("Generating new ids")
        ids = input_ids
        for i in range(num_tokens):
            logger.info(f"generating token {i}")
            ids, kv_cache = generator(input_ids=ids)

    ttnn.tracer.codegen(ids)
