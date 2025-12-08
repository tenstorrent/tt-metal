# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import os

# Set environment first

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import ttml
from ttml.common.config import load_config, get_device_config
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import round_up_to_tile, initialize_device

from huggingface_hub import hf_hub_download


@pytest.fixture(scope="module")
def configs():
    yaml_config = load_config(
        "training_shakespeare_tinyllama.yaml", f"{os.getcwd()}/configs/training_configs"
    )
    model_config = load_config(
        yaml_config["training_config"]["model_config"], f"{os.getcwd()}"
    )
    device_config = get_device_config("training_shakespeare_tinyllama.yaml")

    if device_config.total_devices() > 1:
        initialize_device(yaml_config)

    return yaml_config, model_config, device_config


@pytest.fixture(scope="module")
def tinyllama_model(tokenizer, configs):
    safetensors_path = hf_hub_download(
        repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        filename="model.safetensors",
    )
    safetensors_path = safetensors_path.replace("model.safetensors", "")

    model_config = configs[1]

    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    tt_model_factory.transformer_config.max_sequence_length = 512
    model = tt_model_factory.create_model()
    model.load_from_safetensors(safetensors_path)

    return model, tt_model_factory


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )


@pytest.fixture(scope="module")
def causal_mask(tinyllama_model):
    # [1,1,T,T] float32 with 1s for allowed positions (i >= j), else 0\n",
    T = tinyllama_model[1].transformer_config.max_sequence_length
    m = np.tril(np.ones((T, T), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(
        m.reshape(1, 1, T, T), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )


@pytest.fixture(scope="module")
def logits_mask_tensor(tokenizer):
    orig_vocab_size = tokenizer.vocab_size
    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, orig_vocab_size:] = 1e4
    return ttml.autograd.Tensor.from_numpy(
        logits_mask, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )  # [1,1,1,T], bfloat16


@pytest.fixture(scope="module")
def output_size_128():
    """
    Expected output tokens for input string
    'A dog is:'
    with max_sequence_length=128"
    """

    return "\na) a member of the family of the human race\nb) a member of the family of the animal race\nc) a member of the family of the plant race\nd) a member of the family of the mineral race\ne) a member of the family of the inorganic race\nf) a member of the family of the organic race\ng) a member of the family of the living race\nh) a member of the family of the non-living race\ni) a member of the family of the inanimate race\nj) a member of the family of the inorganic non-living race\nk) a member of the family of the inorganic living race\nl) a member of the family of the inorganic non-living race\nm) a member of the family of the inorganic living race\nn) a member of the family of the inorganic non-living race\no) a member of the family of the inorganic living race\np) a member of the family of the inorganic non-living race\nq) a member of the family of the inorganic living race\nr)"


@pytest.fixture(scope="module")
def output_size_256():
    """
    Expected output tokens for input string
    'The three main differences between cats and dogs are:'
    with max_sequence_length=256"
    """

    return "1. Dogs are more social than cats. 2. Dogs are more intelligent than cats. 3. Dogs are more affectionate than cats.\nDogs are more social than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats.\nDogs are more social than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats.\nDogs are more social than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate than cats. Dogs are more intelligent than cats. Dogs are more affectionate"


@pytest.fixture(scope="module")
def output_size_512():
    """
    Expected output tokens for input string
    'The sky outside is blue, and the grass is green, then the weather is:'
    with max_sequence_length=512"
    """

    return r"\n\n\n*\n\n*Fine\n\n*Cloudy\n\n*Rainy\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are no clouds, then the weather is:\n\n\n*\n\n*Fine\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Cloudy\n\n*Rainy\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Rainy\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Cloudy\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Cloudy\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Sunny\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Cloudy\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n*\n\n*Rainy\n\n\nIf the sky outside is blue, the grass is green, and there are clouds, then the weather is:\n\n\n"


def generate_text_tt(
    model,
    tokenizer,
    prompt,
    causal_mask,
    logits_mask_tensor,
    max_gen_tokens=576,
    pad_token_id=None,
):
    """
    Greedy/temperature=0 generation that prints the *full* text once at the end.
    Uses a sliding window if prompt exceeds max_sequence_length.
    """
    model.eval()
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(
        ttml.autograd.GradMode.DISABLED
    )

    # --- Tokenize once ---
    prompt_tokens = tokenizer.encode(prompt)
    if pad_token_id is None:
        # Try tokenizer.pad_token_id, else fall back to 0
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate once
    padded_prompt_tokens = np.full((1, 1, 1, 512), pad_token_id, dtype=np.uint32)
    for _ in range(max_gen_tokens):
        # Sliding window for long prompts
        if len(prompt_tokens) > 512:
            start_idx = len(prompt_tokens) - 512
            window = prompt_tokens[start_idx:]
        else:
            start_idx = 0
            window = prompt_tokens

        # Refill buffer (fully) to avoid stale ids
        padded_prompt_tokens[...] = pad_token_id
        padded_prompt_tokens[0, 0, 0, : len(window)] = np.asarray(
            window, dtype=np.uint32
        )

        # [1,1,1,T] -> TT tensor
        padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
            padded_prompt_tokens, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )

        # Forward: logits [1,1,T,V]
        logits = model(padded_prompt_tensor, causal_mask)

        # Sample: next tokens for all positions [1,1,T,1]
        # With temperature=0.0 this behaves like argmax/greedy.
        next_token_tensor = ttml.ops.sample.sample_op(
            logits, 0.0, np.random.randint(low=1e7), logits_mask_tensor
        )

        # Take the token at the last active position in the current window
        next_token_idx = 512 - 1 if len(prompt_tokens) > 512 else len(window) - 1
        next_token = int(
            next_token_tensor.to_numpy(composer=composer).reshape(-1, 1)[
                next_token_idx
            ][0]
        )

        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        prompt_tokens.append(next_token)

    # Decode once at the end
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(
        ttml.autograd.GradMode.ENABLED
    )
    return tokenizer.decode(generated_tokens)


def test_tinyllama_grad_sync(tinyllama_model, causal_mask, logits_mask_tensor, configs):
    mapper = None
    if configs[2].enable_ddp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

    X = ttml.autograd.Tensor.from_numpy(
        np.random.randint(0, 100, size=(32, 1, 1, 512)).astype(np.uint32),
        ttml.Layout.ROW_MAJOR,
        ttml.autograd.DataType.UINT32,
        mapper,
    )

    y = ttml.autograd.Tensor.from_numpy(
        np.random.randint(0, 100, size=(32, 512)).astype(np.uint32),
        ttml.Layout.ROW_MAJOR,
        ttml.autograd.DataType.UINT32,
        mapper,
    )

    logits = tinyllama_model[0](X, causal_mask)
    loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.NONE)
    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()

    ttml.core.distributed.synchronize_parameters(tinyllama_model[0].parameters())


def test_tinyllama_inference_128(
    tinyllama_model, tokenizer, causal_mask, logits_mask_tensor, output_size_128
):
    input_text = "A dog is:"

    generated_out = generate_text_tt(
        tinyllama_model[0],
        tokenizer,
        input_text,
        causal_mask,
        logits_mask_tensor,
        max_gen_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )

    assert generated_out == output_size_128


def test_tinyllama_inference_256(
    tinyllama_model, tokenizer, causal_mask, logits_mask_tensor, output_size_256
):
    input_text = "The three main differences between cats and dogs are:"

    generated_out = generate_text_tt(
        tinyllama_model[0],
        tokenizer,
        input_text,
        causal_mask,
        logits_mask_tensor,
        max_gen_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )

    assert generated_out == output_size_256


def test_tinyllama_inference_512(
    tinyllama_model, tokenizer, causal_mask, logits_mask_tensor, output_size_512
):
    input_text = "If the sky outside is blue, the grass is green, and there are no clouds, then the weather is:"

    generated_out = generate_text_tt(
        tinyllama_model[0],
        tokenizer,
        input_text,
        causal_mask,
        logits_mask_tensor,
        max_gen_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )

    assert generated_out == output_size_512
