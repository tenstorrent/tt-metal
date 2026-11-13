# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters between ttml's fused layers and PEFT's per-projection ones, and the HF row layout of
fused parameters. NumPy only; device-free."""

import numpy as np
import pytest

from ttml.common.utils import permute_proj_rows, unpermute_proj_rows
from ttml.models.llama.safetensors_loader import HfTensor, split_hf_tensors
from ttml.unified_trainer import UnsupportedTrainingConfig
from ttml.models.peft import (
    PEFT_PREFIX,
    check_peft_keys,
    lora_to_peft,
    peft_adapter_config,
    peft_keys,
    peft_to_lora,
    with_peft_prefix,
)

HEADS, KV_HEADS, HEAD_DIM, HIDDEN, RANK = 4, 2, 8, 32, 4
Q_ROWS, KV_ROWS = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM


def qkv_blocks(layer=0):
    """What ``hf_layout`` says Llama's ``qkv_linear`` holds: q and k are RoPE-permuted, v is not."""
    attention = f"model.layers.{layer}.self_attn"
    return (
        HfTensor(f"{attention}.q_proj.weight", (Q_ROWS, HIDDEN), HEADS),
        HfTensor(f"{attention}.k_proj.weight", (KV_ROWS, HIDDEN), KV_HEADS),
        HfTensor(f"{attention}.v_proj.weight", (KV_ROWS, HIDDEN), None),
    )


def random_adapter(rng, rows=Q_ROWS + 2 * KV_ROWS):
    return rng.standard_normal((RANK, HIDDEN)).astype(np.float32), rng.standard_normal((rows, RANK)).astype(np.float32)


@pytest.mark.parametrize("shape", [(16, 3), (16,), (16, 2, 5)])
def test_permute_inverts_unpermute(shape):
    rows = np.random.default_rng(0).standard_normal(shape)
    np.testing.assert_array_equal(permute_proj_rows(unpermute_proj_rows(rows, 2), 2), rows)
    np.testing.assert_array_equal(unpermute_proj_rows(permute_proj_rows(rows, 2), 2), rows)


def test_permute_rejects_rows_that_do_not_split_into_heads(expect_error):
    with expect_error(RuntimeError, "not divisible"):
        permute_proj_rows(np.zeros((6, 2)), 4)
    with expect_error(RuntimeError, "must be even"):
        permute_proj_rows(np.zeros((6, 2)), 2)


def test_peft_names():
    assert peft_keys(qkv_blocks()[0]) == (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    )


def test_the_peft_adapters_make_the_same_weight_update():
    """ttml's fused update, ``B @ A`` on the permuted rows, is PEFT's per-projection update permuted the same way."""
    rng = np.random.default_rng(1)
    a, b = random_adapter(rng)
    tensors = lora_to_peft(a, b, qkv_blocks())

    fused_update = b @ a
    hf_updates = []
    for block in qkv_blocks():
        key_a, key_b = peft_keys(block)
        np.testing.assert_array_equal(tensors[key_a], a)
        update = tensors[key_b] @ tensors[key_a]
        hf_updates.append(update if block.rope_heads is None else unpermute_proj_rows(update, block.rope_heads))
    np.testing.assert_allclose(np.concatenate(hf_updates), fused_update, rtol=1e-6, atol=1e-6)


def test_round_trip():
    rng = np.random.default_rng(2)
    a, b = random_adapter(rng)
    a_back, b_back = peft_to_lora(lora_to_peft(a, b, qkv_blocks()), qkv_blocks())
    np.testing.assert_array_equal(a_back, a)
    np.testing.assert_array_equal(b_back, b)


def test_a_single_projection_is_passed_through():
    rng = np.random.default_rng(3)
    a, b = random_adapter(rng, rows=HIDDEN)
    block = HfTensor("model.layers.0.self_attn.o_proj.weight", (HIDDEN, HIDDEN))
    tensors = lora_to_peft(a, b, (block,))
    np.testing.assert_array_equal(tensors[peft_keys(block)[1]], b)


def test_fused_projections_must_share_lora_a(expect_error):
    rng = np.random.default_rng(4)
    a, b = random_adapter(rng)
    tensors = lora_to_peft(a, b, qkv_blocks())
    tensors[peft_keys(qkv_blocks()[2])[0]] = a + 1.0
    with expect_error(UnsupportedTrainingConfig, "v_proj.weight has another lora_A"):
        peft_to_lora(tensors, qkv_blocks())


def test_lora_b_must_cover_the_blocks(expect_error):
    rng = np.random.default_rng(5)
    a, b = random_adapter(rng, rows=Q_ROWS)
    with expect_error(ValueError, "lora_B has 32 rows"):
        lora_to_peft(a, b, qkv_blocks())


def test_the_adapter_must_have_exactly_the_model_s_layers(expect_error):
    rng = np.random.default_rng(6)
    layers = [qkv_blocks(0), qkv_blocks(1)]
    tensors = {}
    for blocks in layers:
        tensors.update(lora_to_peft(*random_adapter(rng), blocks))
    check_peft_keys(tensors, layers)

    with expect_error(UnsupportedTrainingConfig, "missing .*layers.1.self_attn.q_proj"):
        check_peft_keys({k: v for k, v in tensors.items() if "layers.1.self_attn.q_proj" not in k}, layers)
    with expect_error(UnsupportedTrainingConfig, "unexpected .*gate_proj"):
        check_peft_keys({**tensors, f"{PEFT_PREFIX}model.layers.0.mlp.gate_proj.lora_A.weight": np.zeros(1)}, layers)


def test_bare_peft_names_get_the_prefix():
    named = with_peft_prefix({"model.layers.0.self_attn.o_proj.lora_A.weight": 1, f"{PEFT_PREFIX}x": 2})
    assert named == {f"{PEFT_PREFIX}model.layers.0.self_attn.o_proj.lora_A.weight": 1, f"{PEFT_PREFIX}x": 2}


def test_adapter_config():
    config = peft_adapter_config("meta-llama/Llama-3.2-1B", 8, 16, 0.05, ["v_proj", "q_proj", "k_proj"], "CAUSAL_LM")
    assert config["peft_type"] == "LORA" and config["task_type"] == "CAUSAL_LM"
    assert (config["r"], config["lora_alpha"], config["lora_dropout"]) == (8, 16, 0.05)
    assert config["target_modules"] == ["k_proj", "q_proj", "v_proj"]
    assert config["base_model_name_or_path"] == "meta-llama/Llama-3.2-1B"
    assert not config["use_rslora"] and not config["use_dora"] and config["bias"] == "none"


def test_split_hf_tensors_inverts_how_the_loader_assembles_a_fused_parameter():
    rng = np.random.default_rng(7)
    q, k, v = (rng.standard_normal(block.shape).astype(np.float32) for block in qkv_blocks())
    fused = np.concatenate([unpermute_proj_rows(q, HEADS), unpermute_proj_rows(k, KV_HEADS), v])
    split = split_hf_tensors(fused, qkv_blocks())
    for block, expected in zip(qkv_blocks(), (q, k, v)):
        np.testing.assert_array_equal(split[block.name], expected)


def test_split_hf_tensors_drops_padding_rows_and_reads_norms_as_1d():
    embedding = np.arange(12, dtype=np.float32).reshape(6, 2)
    split = split_hf_tensors(embedding, (HfTensor("model.embed_tokens.weight", (5, 2)),))
    np.testing.assert_array_equal(split["model.embed_tokens.weight"], embedding[:5])

    gamma = np.arange(4, dtype=np.float32).reshape(1, 4)
    split = split_hf_tensors(gamma, (HfTensor("model.norm.weight", (4,)),))
    assert split["model.norm.weight"].shape == (4,)


def test_split_hf_tensors_copies_out_of_the_parameter():
    values = np.zeros((6, 2), dtype=np.float32)
    split = split_hf_tensors(values, (HfTensor("model.embed_tokens.weight", (5, 2)),))
    values[:] = 1.0
    assert not split["model.embed_tokens.weight"].any()


def test_split_hf_tensors_needs_enough_rows(expect_error):
    with expect_error(RuntimeError, "at least 64 rows"):
        split_hf_tensors(np.zeros((40, HIDDEN)), qkv_blocks())
