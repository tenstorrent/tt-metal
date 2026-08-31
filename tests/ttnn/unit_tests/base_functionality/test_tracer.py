# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys

import pytest

import networkx as nx
import torch
import transformers

import ttnn
from ttnn.tracer import trace, visualize, get_graph

from models.common.utility_functions import is_wormhole_b0, is_blackhole

from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert
from ttnn.model_preprocessing import preprocess_model_parameters


def read_tracer_startup_config(overrides):
    env = os.environ.copy()
    env.pop("TTNN_CONFIG_PATH", None)
    if overrides is None:
        env.pop("TTNN_CONFIG_OVERRIDES", None)
    else:
        env["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)

    script = """
import json
import ttnn

print(json.dumps({
    "enable_graph_report": ttnn.CONFIG.enable_graph_report,
    "enable_detailed_buffer_report": ttnn.CONFIG.enable_detailed_buffer_report,
    "enable_torch_tracer": ttnn._PYTHON_CONFIG["enable_torch_tracer"],
    "is_tracing_enabled": ttnn.tracer.is_tracing_enabled(),
}))
"""
    return subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True)


def parse_tracer_startup_config(result):
    result.check_returncode()
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.requires_fast_runtime_mode_off
def test_exp():
    with trace():
        tensor = torch.randint(0, 100, (1, 64))
        tensor = torch.exp(tensor)

    visualize(tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_nn_parameter_in_module_init_under_tracer():
    # Regression: under the tracer, nn.Parameter(torch.empty(...)) inside Module.__init__
    # used to fail with "Cannot assign non-leaf Tensor to parameter 'weight'" because
    # TracedTorchTensor.__new__ wrapped the result of requires_grad_() as an alias view
    # (grad_fn=AliasBackward0, is_leaf=False), which register_parameter rejects.
    with trace():

        class Gate(torch.nn.Module):
            def __init__(self, n_routed_experts: int, gating_dim: int):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((n_routed_experts, gating_dim)))

        gate = Gate(n_routed_experts=8, gating_dim=16)
        # Verify both the tensor autograd properties and that Module.__setattr__
        # registered the attribute as a real nn.Parameter.
        assert isinstance(gate.weight, torch.nn.Parameter)
        assert "weight" in gate._parameters
        assert gate._parameters["weight"] is gate.weight
        assert gate.weight.is_leaf
        assert gate.weight.grad_fn is None


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


@pytest.mark.requires_fast_runtime_mode_off
def test_config_owned_tracing_is_disabled_when_config_turns_off():
    """Config-owned tracing stops when its opt-in flag is restored."""
    with ttnn.manage_config("enable_graph_report", False), ttnn.manage_config("enable_torch_tracer", False):
        with ttnn.manage_config("enable_torch_tracer", True):
            assert ttnn.tracer.is_tracing_enabled()

        assert not ttnn.tracer.is_tracing_enabled()
        assert not ttnn.torch_tracer.is_tracing_enabled()


@pytest.mark.requires_fast_runtime_mode_off
def test_config_changes_do_not_disable_explicit_trace_session():
    """Config restoration leaves an explicit trace session active."""
    with ttnn.manage_config("enable_graph_report", False), ttnn.manage_config("enable_torch_tracer", False):
        with trace():
            with ttnn.manage_config("enable_torch_tracer", True):
                assert ttnn.tracer.is_tracing_enabled()

            assert ttnn.tracer.is_tracing_enabled()


@pytest.mark.requires_fast_runtime_mode_off
def test_config_owned_tracing_is_disabled_when_context_raises(expect_error):
    with ttnn.manage_config("enable_graph_report", False), ttnn.manage_config("enable_torch_tracer", False):
        with expect_error(RuntimeError, "expected failure"):
            with ttnn.manage_config("enable_torch_tracer", True):
                assert ttnn.tracer.is_tracing_enabled()
                raise RuntimeError("expected failure")

        assert not ttnn.tracer.is_tracing_enabled()
        assert not ttnn.torch_tracer.is_tracing_enabled()


@pytest.mark.requires_fast_runtime_mode_off
def test_explicit_tracing_is_disabled_when_context_raises(expect_error):
    with ttnn.manage_config("enable_torch_tracer", False):
        with expect_error(RuntimeError, "expected failure"):
            with trace():
                assert ttnn.tracer.is_tracing_enabled()
                raise RuntimeError("expected failure")

        assert not ttnn.tracer.is_tracing_enabled()
        assert not ttnn.torch_tracer.is_tracing_enabled()


@pytest.mark.requires_fast_runtime_mode_off
def test_explicit_torch_tracing_with_graph_report_is_rejected(expect_error):
    with ttnn.manage_config("enable_torch_tracer", False), ttnn.manage_config("enable_graph_report", True):
        with expect_error(ValueError, "Torch tracing is not supported while enable_graph_report is enabled"):
            with trace():
                pass


@pytest.mark.requires_fast_runtime_mode_off
def test_graph_report_does_not_enable_torch_tracer_during_dtype_conversion():
    with (
        ttnn.manage_config("enable_torch_tracer", False),
        ttnn.manage_config("enable_logging", True),
        ttnn.manage_config("enable_graph_report", True),
        ttnn.manage_config("enable_detailed_buffer_report", False),
    ):
        output = ttnn.from_torch(torch.ones((1, 1), dtype=torch.float32), dtype=ttnn.bfloat16)

        assert output.dtype == ttnn.bfloat16
        assert not ttnn.tracer.is_tracing_enabled()
        assert not ttnn.torch_tracer.is_tracing_enabled()


def test_torch_tracer_is_disabled_by_default_at_startup():
    config = parse_tracer_startup_config(read_tracer_startup_config(None))

    assert not config["enable_torch_tracer"]
    assert not config["is_tracing_enabled"]


def test_graph_report_startup_config_does_not_enable_torch_tracer():
    config = parse_tracer_startup_config(
        read_tracer_startup_config(
            {
                "enable_logging": True,
                "enable_fast_runtime_mode": False,
                "enable_graph_report": True,
                "enable_detailed_buffer_report": False,
                "report_name": "tracer_startup_test",
            }
        )
    )

    assert config["enable_graph_report"]
    assert not config["enable_detailed_buffer_report"]
    assert not config["enable_torch_tracer"]
    assert not config["is_tracing_enabled"]


def test_torch_tracer_can_be_enabled_independently_at_startup():
    config = parse_tracer_startup_config(
        read_tracer_startup_config(
            {
                "enable_fast_runtime_mode": False,
                "enable_graph_report": False,
                "enable_torch_tracer": True,
            }
        )
    )

    assert not config["enable_graph_report"]
    assert config["enable_torch_tracer"]
    assert config["is_tracing_enabled"]


def test_torch_and_graph_tracer_startup_config_is_rejected():
    result = read_tracer_startup_config(
        {
            "enable_fast_runtime_mode": False,
            "enable_graph_report": True,
            "enable_torch_tracer": True,
        }
    )

    assert result.returncode != 0
    assert "enable_torch_tracer and enable_graph_report cannot both be enabled" in result.stderr


def test_torch_tracer_in_fast_runtime_mode_is_rejected():
    result = read_tracer_startup_config(
        {
            "enable_fast_runtime_mode": True,
            "enable_graph_report": False,
            "enable_torch_tracer": True,
        }
    )

    assert result.returncode != 0
    assert "enable_torch_tracer requires enable_fast_runtime_mode=false" in result.stderr


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
def test_ttnn_bert(device, model_name, batch_size, sequence_size, bert):
    config = transformers.BertConfig.from_pretrained(model_name)

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(model_name).eval(),
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
