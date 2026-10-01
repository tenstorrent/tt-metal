# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint contract: keys, shapes, dtypes, and what must not be present.

The contract is generated from the config by expected_checkpoint_keys, not stored as a
captured list, so these tests exercise the generator's logic against the real checkpoint.

Needs the checkpoint.
"""

import json

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import CHECKPOINT, resolve_config
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_hf_config
from models.experimental.nomic_embed_text_v2_moe.reference.loader import (
    ABSENT_KEY_SUBSTRINGS,
    expected_checkpoint_keys,
    load_pretrained_reference_model,
    load_reference_model,
)
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertModel

pytestmark = pytest.mark.needs_weights


def layer_indices_with(state_dict, marker: str) -> list[int]:
    return sorted({int(key.split(".")[2]) for key in state_dict if marker in key})


def test_tensor_count_and_parameter_total(state_dict):
    assert len(state_dict) == CHECKPOINT.n_tensors
    assert sum(tensor.numel() for tensor in state_dict.values()) == CHECKPOINT.n_parameters


def test_all_tensors_are_float32(state_dict):
    dtypes = {tensor.dtype for tensor in state_dict.values()}
    assert dtypes == {torch.float32}, f"expected a pure fp32 checkpoint, got {dtypes}"


def test_generated_contract_matches_checkpoint_keys(config, state_dict):
    expected = expected_checkpoint_keys(config)

    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    assert not missing, f"generator produced keys absent from the checkpoint: {missing[:10]}"
    assert not unexpected, f"checkpoint has keys the generator did not produce: {unexpected[:10]}"


def test_generated_contract_matches_checkpoint_shapes(config, state_dict):
    expected = expected_checkpoint_keys(config)
    mismatched = {
        key: (tuple(state_dict[key].shape), shape)
        for key, shape in expected.items()
        if tuple(state_dict[key].shape) != shape
    }
    assert not mismatched, f"shape mismatches (actual, expected): {mismatched}"


def test_moe_layers_are_the_odd_ones(config, state_dict):
    moe = layer_indices_with(state_dict, ".mlp.router.")
    dense = layer_indices_with(state_dict, ".mlp.fc1.")

    assert tuple(moe) == config.moe_layers
    assert tuple(dense) == config.dense_layers
    # The `i % every_n == 0` variant would put layer 0 in the MoE set.
    assert 0 not in moe


def test_expert_packing_is_expert_outer(config, state_dict):
    experts, ffn, hidden = config.num_experts, config.intermediate_size, config.hidden_size

    for layer_idx in config.moe_layers:
        prefix = f"encoder.layers.{layer_idx}.mlp."
        assert tuple(state_dict[prefix + "experts.mlp.w1"].shape) == (experts * ffn, hidden)
        assert tuple(state_dict[prefix + "experts.mlp.w2"].shape) == (experts * ffn, hidden)
        # One shared bias of width hidden, not a per-expert (num_experts, hidden) block.
        assert tuple(state_dict[prefix + "experts.bias"].shape) == (hidden,)
        assert tuple(state_dict[prefix + "router.layer.weight"].shape) == (experts, hidden)
        assert prefix + "router.layer.bias" not in state_dict


def test_w2_ambiguous_view_is_not_disambiguated_by_size(config):
    """Why the expert orientation needs a numerical test rather than a shape check: the element
    count is symmetric, so the wrong view succeeds and nothing raises."""
    experts, ffn, hidden = config.num_experts, config.intermediate_size, config.hidden_size
    assert (experts * ffn) * hidden == experts * hidden * ffn


@pytest.mark.parametrize("substring", ABSENT_KEY_SUBSTRINGS)
def test_absent_keys(state_dict, substring):
    hits = sorted(key for key in state_dict if substring in key)
    assert not hits, f"unexpected {substring!r} keys present: {hits[:5]}"


def test_strict_load_is_clean(config, state_dict):
    """Name isomorphism: zero missing, zero unexpected, no remapping layer."""
    model = load_reference_model(config, state_dict)
    assert model.config.num_hidden_layers == config.num_hidden_layers


def test_load_pretrained_reference_model_helper(config, state_dict, checkpoint_path):
    """The one-call helper must return the real weights, not a random init.

    Comparing one parameter against the checkpoint is what separates the two: a randomly
    initialised model of the same shape would pass every structural assertion here.
    """
    model = load_pretrained_reference_model(allow_download=False)

    assert isinstance(model, NomicBertModel)
    assert not model.training
    assert len(model.encoder.layers) == config.num_hidden_layers

    key = "encoder.layers.0.attn.Wqkv.weight"
    torch.testing.assert_close(dict(model.named_parameters())[key], state_dict[key])


def test_pad_embedding_row_is_not_zero(config, state_dict):
    """nn.Embedding(padding_idx=...) zeroes at init, but the trained row survives loading, so
    ttnn.embedding must not be given padding_idx."""
    pad_row = state_dict["embeddings.word_embeddings.weight"][config.pad_token_id]
    assert pad_row.abs().max() > 0


def test_vocab_size_is_padded_to_the_configured_multiple(config):
    """Upstream would grow the embedding table if this did not divide evenly, which would make
    the generated shape contract wrong."""
    assert config.vocab_size % config.pad_vocab_size_multiple == 0


def test_vendored_config_matches_the_pinned_revision(checkpoint_path):
    # Depends on checkpoint_path for its skip-when-uncached guard: the config ships in the same
    # snapshot, so without it this fails instead of skipping on a cold cache.
    live = json.load(open(resolve_config(allow_download=False)))
    assert load_vendored_hf_config() == live
