# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint contract: keys, shapes, dtypes, and what must NOT be there.

The contract is *generated* from the config by `expected_checkpoint_keys`, not stored as a
captured list. Comparing a frozen list to a frozen list proves nothing; comparing a
generator to the real checkpoint puts the generator's own logic -- the `i % 2 == 1` MoE
placement predicate, the `E * ffn_hidden` expert packing, the bias-free router -- under
test. Every one of those is something the TTNN port has to get right independently.
"""

import json

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import (
    MODEL_REVISION,
    N_CHECKPOINT_TENSORS,
    N_PARAMETERS,
    resolve_config,
)
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import (
    ConfigAssumptionError,
    from_hf_config,
    load_vendored_hf_config,
)
from models.experimental.nomic_embed_text_v2_moe.reference.loader import (
    ABSENT_KEY_SUBSTRINGS,
    expected_checkpoint_keys,
    load_reference_model,
)

pytestmark = pytest.mark.needs_weights


def test_tensor_count_and_parameter_total(state_dict):
    assert len(state_dict) == N_CHECKPOINT_TENSORS
    assert sum(t.numel() for t in state_dict.values()) == N_PARAMETERS


def test_all_tensors_are_float32(state_dict):
    dtypes = {t.dtype for t in state_dict.values()}
    assert dtypes == {torch.float32}, f"expected a pure fp32 checkpoint, got {dtypes}"


def test_generated_contract_matches_checkpoint_keys(config, state_dict):
    """The generator, not a captured list, is what is being tested here."""
    expected = expected_checkpoint_keys(config)

    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    assert not missing, f"generator produced keys absent from the checkpoint: {missing[:10]}"
    assert not unexpected, f"checkpoint has keys the generator did not produce: {unexpected[:10]}"


def test_generated_contract_matches_checkpoint_shapes(config, state_dict):
    expected = expected_checkpoint_keys(config)
    mismatched = {k: (tuple(state_dict[k].shape), v) for k, v in expected.items() if tuple(state_dict[k].shape) != v}
    assert not mismatched, f"shape mismatches (actual, expected): {mismatched}"


def test_moe_layers_are_the_odd_ones(config, state_dict):
    """`i % moe_every_n_layers == 1`, so layer 0 is dense and layer 1 is the first MoE."""
    moe_from_checkpoint = sorted({int(k.split(".")[2]) for k in state_dict if ".mlp.router." in k})
    dense_from_checkpoint = sorted({int(k.split(".")[2]) for k in state_dict if ".mlp.fc1." in k})

    assert moe_from_checkpoint == [1, 3, 5, 7, 9, 11]
    assert dense_from_checkpoint == [0, 2, 4, 6, 8, 10]
    assert tuple(moe_from_checkpoint) == config.moe_layers
    assert tuple(dense_from_checkpoint) == config.dense_layers
    # The off-by-one that `== 0` would produce must not be what the checkpoint shows.
    assert 0 not in moe_from_checkpoint


def test_expert_packing_is_expert_outer(config, state_dict):
    """`w1`/`w2` are `[E * ffn_hidden, hidden]` -- expert axis OUTER, both stored [F, H]."""
    E, F, H = config.num_experts, config.intermediate_size, config.hidden_size
    for i in config.moe_layers:
        assert tuple(state_dict[f"encoder.layers.{i}.mlp.experts.mlp.w1"].shape) == (E * F, H)
        assert tuple(state_dict[f"encoder.layers.{i}.mlp.experts.mlp.w2"].shape) == (E * F, H)
        # ONE shared bias of width `hidden`, not a per-expert `[E, H]` block.
        assert tuple(state_dict[f"encoder.layers.{i}.mlp.experts.bias"].shape) == (H,)
        # The router is bias-free.
        assert tuple(state_dict[f"encoder.layers.{i}.mlp.router.layer.weight"].shape) == (E, H)
        assert f"encoder.layers.{i}.mlp.router.layer.bias" not in state_dict


def test_w2_ambiguous_view_is_not_disambiguated_by_size(config):
    """Why the expert orientation needs a test rather than a shape check.

    `E * F * H` is symmetric in F and H, so viewing `w2` as `(E, H, F)` instead of
    `(E, F, H)` succeeds and the downstream matmul typechecks. Nothing raises. The only
    signal is the numerical result, which `test_reference_modules.py` pins.
    """
    E, F, H = config.num_experts, config.intermediate_size, config.hidden_size
    assert (E * F) * H == E * H * F


@pytest.mark.parametrize("substring", ABSENT_KEY_SUBSTRINGS)
def test_absent_keys(state_dict, substring):
    """Features this checkpoint does not have. A hit means the reference drops a real weight."""
    hits = sorted(k for k in state_dict if substring in k)
    assert not hits, f"unexpected {substring!r} keys present: {hits[:5]}"


def test_strict_load_is_clean(config, state_dict):
    """Name isomorphism, proven: zero missing, zero unexpected, no remapping layer."""
    model = load_reference_model(config, state_dict)  # strict=True inside
    assert model.config.num_hidden_layers == config.num_hidden_layers


def test_pad_embedding_row_is_not_zero(config, state_dict):
    """`nn.Embedding(padding_idx=...)` zeroes at init; the trained row survives loading.

    This matters for the TTNN port: `ttnn.embedding` takes an optional `padding_idx` that
    WOULD zero it. Passing it would diverge from upstream on every padded sequence.
    """
    table = state_dict["embeddings.word_embeddings.weight"]
    pad_row = table[config.pad_token_id]
    assert pad_row.abs().max() > 0, "the <pad> row is zero; do not pass padding_idx on device"


def test_vendored_config_matches_the_pinned_revision():
    """The committed snapshot must be the pinned revision's config, byte-for-byte in content."""
    live = json.load(open(resolve_config(revision=MODEL_REVISION, allow_download=False)))
    assert load_vendored_hf_config() == live


def test_config_validation_rejects_a_violated_assumption(expect_error):
    """`from_hf_config` must fail loudly, not silently accept a divergent config."""
    hf_config = dict(load_vendored_hf_config())
    hf_config["moe_normalize_expert_weights"] = True
    with expect_error(ConfigAssumptionError, "moe_normalize_expert_weights"):
        from_hf_config(hf_config)


def test_config_validation_rejects_a_missing_field(expect_error):
    hf_config = dict(load_vendored_hf_config())
    del hf_config["prenorm"]
    with expect_error(ConfigAssumptionError, "prenorm"):
        from_hf_config(hf_config)
