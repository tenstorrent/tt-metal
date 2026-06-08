# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer-related tests split from test_prefill_transformer.py.

Tests tokenize_prompt_to_isl, tokenize_prompt_to_chat_template,
and first-token generation from saved reference outputs.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    get_4d_causal_mask,
    tokenize_prompt_to_chat_template,
    tokenize_prompt_to_isl,
)


@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_tokenize_prompt_to_isl(tokenizer):
    max_isl = 10
    input_ids, attention_mask, tokens = tokenize_prompt_to_isl(
        tokenizer, max_isl=max_isl, prompt_text="This is a test prompt.", debug=True
    )

    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Attention Mask: {attention_mask}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"

    torch.set_printoptions(threshold=float("inf"), edgeitems=3, precision=2, linewidth=200)
    logger.debug(f"4D Causal Attention Mask shape:\n{get_4d_causal_mask(attention_mask, causal_only=True)}")
    logger.debug(f"4D Causal Attention Mask Paddshape:\n{get_4d_causal_mask(attention_mask, causal_only=False)}")


@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_tokenize_prompt_to_chat_template(tokenizer):
    max_isl = 64
    input_ids, tokens = tokenize_prompt_to_chat_template(
        tokenizer,
        max_isl=max_isl,
        user_prompt="What is the capital of Serbia?",
        system_prompt="You are a helpful assistant.",
        debug=True,
    )
    logger.debug(f"Input IDs: {input_ids}")
    logger.debug(f"Tokens: {tokens}")

    assert input_ids.shape == (1, max_isl), f"Expected input_ids shape (1, {max_isl}), got {input_ids.shape}"


@pytest.mark.parametrize(
    "json_path",
    [
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_ABC_short.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_64tok.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_prompt_960tok.json"),
        Path("models/demos/deepseek_v3_d_p/demo/test_pie_960tok.json"),
    ],
    ids=["short", "64tok", "960tok", "pie"],
)
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_token_count(tokenizer, json_path):
    """Tokenize a prompt JSON without padding and report token count."""
    from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

    logger.info(f"{json_path=}")
    prompts = load_prompts_from_json(str(json_path))
    assert prompts, f"No prompts found in {json_path}"
    prompt_text = prompts[0]

    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    logger.info(f"{len(tokens)=}")
    logger.debug(f"Prompt: {repr(prompt_text)}")
    logger.debug(f"Token IDs: {tokens}")

    input_ids, attention_mask, tokens_padded = tokenize_prompt_to_isl(
        tokenizer, max_isl=1024, prompt_text=prompts, debug=True
    )
    number_of_non_padded_tokens = attention_mask.sum().item()  # should be returned by tokenize..
    logger.info(f"{number_of_non_padded_tokens=}")
    logger.debug(f"Prompt: {repr(tokens_padded)}")
    logger.debug(f"Token IDs: {input_ids}")

    # note number_of_non_padded_tokens is len(tokens) + 1 for the added BOS token


@pytest.mark.parametrize(
    "input_path",
    [
        Path("/tmp/r1/pretrained_abc_1k_isl1024_layers61_experts256.pt"),
        Path(
            "/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Reference-prefill/pretrained_abc_1k_isl1024_layers61_experts256.pt"
        ),
        Path(
            "/workspace/ds_output_all_sources_new_/norm_20260415_114721_mesh8x4_isl1024_L61_e256_cf32_gatehost_all_pretrained_abc_1k.pt"
        ),
    ],
)
@pytest.mark.parametrize("tokenizer", ["right", "left"], indirect=True, ids=["right_pad", "left_pad"])
def test_first_token_from_reference(input_path, model_path, config_only, tokenizer):
    # Use weights_only=False since this is a trusted local file with custom objects
    logger.info(f"{input_path=}")
    if not input_path.exists():
        pytest.skip(f"Reference artifact not found: {input_path}")
    data = torch.load(input_path, weights_only=False)

    if "norm_output" in data:  # this is ttnn output; norm only
        norm_output = data["norm_output"]
        logger.info(f"{norm_output.shape=}")
    elif "ref_snapshots" in data:  # this is torch output; all emb + layers + norm
        logger.info(f"Number of reference snapshots: {len(data['ref_snapshots'])}")
        assert len(data["ref_snapshots"]) == DeepSeekV3Config.NUM_LAYERS + 2  # token embedding + final RMS norm
        norm_output = data["ref_snapshots"][-1]  # Last snapshot's tensor
        logger.warning("Loaded data does not have 'norm_output' key, assuming last snapshot is final rms norm output")
    else:
        logger.warning(data)
        raise ValueError("Loaded data format is unexpected and does not contain 'norm_output' or expected snapshots.")

    #  Remove batch dimension if present
    if norm_output.shape[0] == 1:
        norm_output = norm_output.squeeze(0)
    logger.success("Data loaded successfully.")

    # LM HEAD loading
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    lazy_sd = LazyStateDict(Path(model_path))
    # lm_head_sd = sub_state_dict(lazy_sd, "model.embed_tokens.")
    lm_head_sd = sub_state_dict(lazy_sd, "lm_head.")
    lm_head_dequant = dequantize_state_dict(lm_head_sd, config_only)
    lm_head_dequant_weight = lm_head_dequant.get("weight")
    assert (
        lm_head_dequant_weight.shape[0] == config_only.vocab_size
    ), f"Expected lm_head_dequant_weight shape[0] to be {config_only.vocab_size}, got {lm_head_dequant_weight.shape[0]}"
    logger.success("LM head weight loaded and dequantized successfully.")

    # Apply LM HEAD
    norm_output = norm_output.to(lm_head_dequant_weight.dtype)
    # norm_output = ref_snapshots[-1][0,:,:].to(lm_head_dequant_weight.dtype)
    logger.debug("Computing logits...")
    with torch.no_grad():
        # lm_head is just a linear layer: logits = hidden @ lm_head_weight.T
        logits = torch.matmul(norm_output, lm_head_dequant_weight.T)  # Shape: [1, 1, vocab_size]
    logger.debug(f"Logits shape: {logits.shape}")

    # Apply sampling
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/generate.py
    def sample(logits, temperature: float = 1.0):
        """
        Samples a token from the logits using temperature scaling.

        Args:
            logits (torch.Tensor): The logits tensor for token predictions.
            temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

        Returns:
            torch.Tensor: The sampled token.
        """
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    index = -1  # Last token
    last_logit = logits[index, :].clone()
    top_k = 5

    for temperature in [0.0, 0.5, 1.0]:
        # Sample token
        first_token = sample(logits=last_logit.clone(), temperature=temperature)
        token_text = tokenizer.decode([first_token.item()])
        logger.info(f"First token (temp={temperature}): ID={first_token.item()} [{repr(token_text)}]")

        # Compute temperature-scaled probabilities
        scaled_logits = last_logit / max(temperature, 1e-5)
        probs = torch.softmax(scaled_logits, dim=-1)

        # Get top-5 by probability (after temperature scaling)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        for i, (token_id, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
            token_text = tokenizer.decode([token_id])
            logger.info(f"  top{i+1}: ID={token_id:6d} | Prob: {prob*100:6.2f}% | Text: {repr(token_text)}")
