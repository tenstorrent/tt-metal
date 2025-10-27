import numpy as np
import sys
import os
import pytest

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import ttml
from ttml.common.config import *
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.data import build_causal_mask
from ttml.common.utils import round_up_to_tile


@pytest.fixture
def tinyllama_model(tokenizer):
    safetensors_path = hf_hub_download(
        repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", filename="model.safetensors"
    )
    safetensors_path = safetensors_path.replace("model.safetensors", "")

    yaml_config = get_config(f'{os.environ["TT_METAL_HOME"]}/tt-train/configs/training_shakespeare_tinyllama.yaml')

    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(yaml_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    tt_model_factory.transformer_config.max_sequence_length = 128

    model = tt_model_factory.create_model()
    model.load_from_safetensors(safetensors_path)

    return model


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")


@pytest.fixture
def causal_mask(tinyllama_model):
    max_sequence_length = tinyllama_model.config.max_sequence_length
    causal_mask = build_causal_mask(max_sequence_length)
    causal_mask = ttml.autograd.Tensor.from_numpy(causal_mask, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16)
    return causal_mask


@pytest.fixture
def logits_mask_tensor(tinyllama_model, tokenizer):
    orig_vocab_size = tokenizer.vocab_size
    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    return logits_mask_tensor


def expected_out():
    tinyllama_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", torch_dtype=torch.bfloat16
    )
    tinyllama_model.eval()

    input_text = "The difference between dogs and cats is:"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        generated_ids = tinyllama_model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (excluding input)

    return generated_ids


def generate_text_tt(
    model,
    tokenizer,
    question,
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
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    # --- Tokenize once ---
    prompt_tokens = tokenizer.encode(question)
    if pad_token_id is None:
        # Try tokenizer.pad_token_id, else fall back to 0
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate once
    padded_prompt_tokens = np.full((1, 1, 1, 128), pad_token_id, dtype=np.uint32)
    for _ in range(max_gen_tokens):
        # Sliding window for long prompts
        if len(prompt_tokens) > 128:
            start_idx = len(prompt_tokens) - 128
            window = prompt_tokens[start_idx:]
        else:
            start_idx = 0
            window = prompt_tokens

        # Refill buffer (fully) to avoid stale ids
        padded_prompt_tokens[...] = pad_token_id
        padded_prompt_tokens[0, 0, 0, : len(window)] = np.asarray(window, dtype=np.uint32)

        # [1,1,1,T] -> TT tensor
        padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
            padded_prompt_tokens, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )

        # Forward: logits [1,1,T,V]
        logits = model(padded_prompt_tensor, causal_mask)

        # Sample: next tokens for all positions [1,1,T,1]
        # With temperature=0.0 this behaves like argmax/greedy.
        next_token_tensor = ttml.ops.sample.sample_op(logits, 0.0, np.random.randint(low=1e7), logits_mask_tensor)

        # Take the token at the last active position in the current window
        next_token_idx = 128 - 1 if len(prompt_tokens) > 128 else len(window) - 1
        next_token = int(next_token_tensor.to_numpy(composer=composer).reshape(-1, 1)[next_token_idx][0])

        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        prompt_tokens.append(next_token)

    # Decode once at the end
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    return generated_tokens


def test_tinyllama_inference(tinyllama_model, tokenizer, causal_mask, expected_out):
    input_text = "The difference between dogs and cats is:"
    inputs = tokenizer(input_text, return_tensors="np")

    input_ids = inputs["input_ids"]
