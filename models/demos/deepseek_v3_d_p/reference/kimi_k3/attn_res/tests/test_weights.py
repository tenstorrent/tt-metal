# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint-to-query conversion, checked on CPU. No device, no checkpoint.

`fold_queries` is the only thing standing between a state dict and the queries every walk
issues, and nothing downstream can tell a mis-ordered fold from a correct one: swap the two
read sites of a layer, or the two factors of a query, and every later gate still compares a
device against an oracle built from the same wrong queries. So the ordering is checked here
against weights chosen to make each site distinguishable.

The names are equally load-bearing and equally invisible — a wrong prefix or a wrong factor
spelling raises `missing AttnRes weight` at load time on a real checkpoint, months after the
suite went green on random weights.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import HIDDEN_SIZE
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import (
    CHECKPOINT_PREFIX,
    fold_queries,
    layer_query_names,
    output_query_names,
    query_weight_names,
    validate_query_weights,
)

NUM_LAYERS = 3
SMALL_HIDDEN_SIZE = 8


def _state_dict(num_layers=NUM_LAYERS, hidden_size=SMALL_HIDDEN_SIZE, prefix=CHECKPOINT_PREFIX):
    """One state dict holding every AttnRes weight, each with a distinguishable value.

    Norm factors count up from one and projection factors are constant, so a folded query
    equals its norm factor scaled — a swapped site or factor moves the value, not just its
    position.
    """
    weights = {}
    for index, name in enumerate(query_weight_names(num_layers, prefix), start=1):
        is_proj = name.endswith("_proj.weight")
        value = torch.full((hidden_size,), float(index), dtype=torch.float32)
        weights[name] = value.reshape(1, -1) if is_proj else value
    return weights


def test_query_weight_names_covers_every_site_once():
    names = query_weight_names(NUM_LAYERS, prefix="")

    # Two factors at each of two read sites per layer, plus the model-level read's two.
    assert len(names) == 4 * NUM_LAYERS + 2
    assert len(set(names)) == len(names)
    assert set(names) == {name for idx in range(NUM_LAYERS) for name in layer_query_names(idx, "")} | set(
        output_query_names("")
    )


def test_query_weight_names_spells_the_published_checkpoint():
    """Kimi K3 nests the decoder under a multimodal wrapper, so the published keys carry it."""
    names = query_weight_names(1)

    assert set(names) == {
        "language_model.model.layers.0.self_attention_res_norm.weight",
        "language_model.model.layers.0.self_attention_res_proj.weight",
        "language_model.model.layers.0.mlp_res_norm.weight",
        "language_model.model.layers.0.mlp_res_proj.weight",
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
    }


def test_query_weight_names_follows_the_prefix():
    """A state dict taken from an instantiated model is already rooted at the decoder."""
    assert all(not name.startswith(CHECKPOINT_PREFIX) for name in query_weight_names(1, prefix=""))


def test_validate_accepts_a_complete_state_dict():
    validate_query_weights(_state_dict(), NUM_LAYERS, SMALL_HIDDEN_SIZE)


@pytest.mark.parametrize("dropped", ["layers.1.mlp_res_proj.weight", "output_attn_res_norm.weight"])
def test_validate_rejects_a_missing_weight(dropped, expect_error):
    weights = _state_dict()
    del weights[f"{CHECKPOINT_PREFIX}{dropped}"]

    with expect_error(ValueError, "missing AttnRes weight"):
        validate_query_weights(weights, NUM_LAYERS, SMALL_HIDDEN_SIZE)


def test_validate_rejects_a_transposed_projection(expect_error):
    """`fold_query` flattens, so `[d, 1]` folds to the same numbers and passes a bare count."""
    weights = _state_dict()
    name = f"{CHECKPOINT_PREFIX}layers.0.mlp_res_proj.weight"
    weights[name] = weights[name].reshape(SMALL_HIDDEN_SIZE, 1)

    with expect_error(ValueError, "expected"):
        validate_query_weights(weights, NUM_LAYERS, SMALL_HIDDEN_SIZE)


def test_validate_rejects_an_unsqueezed_norm(expect_error):
    weights = _state_dict()
    name = f"{CHECKPOINT_PREFIX}layers.0.mlp_res_norm.weight"
    weights[name] = weights[name].reshape(1, -1)

    with expect_error(ValueError, "expected"):
        validate_query_weights(weights, NUM_LAYERS, SMALL_HIDDEN_SIZE)


def test_validate_rejects_the_wrong_hidden_size(expect_error):
    with expect_error(ValueError, "expected"):
        validate_query_weights(_state_dict(), NUM_LAYERS, SMALL_HIDDEN_SIZE + 1)


def test_fold_queries_returns_the_walk_order():
    """`attn_res_stack` takes `(q_pre, q_post, q_out)`, one query per site per layer.

    Validation runs at the production width, so this is the one case that has to be built
    there; the schedule and the fold are both independent of `d`.
    """
    weights = _state_dict(hidden_size=HIDDEN_SIZE)
    q_pre, q_post, q_out = fold_queries(weights, NUM_LAYERS)

    assert len(q_pre) == len(q_post) == NUM_LAYERS
    assert all(q.shape == (HIDDEN_SIZE,) for q in (*q_pre, *q_post, q_out))


def test_fold_queries_pairs_each_site_with_its_own_factors():
    """The failure this catches is a fold that lands on the wrong site, not one that errors.

    `_state_dict` numbers the weights in `query_weight_names` order, four per layer as
    `(self_attention norm, self_attention proj, mlp norm, mlp proj)`, so layer `i`'s
    pre-attention query folds to `(4i+1)(4i+2)` and its pre-MLP query to `(4i+3)(4i+4)`.
    """
    q_pre, q_post, q_out = fold_queries(_state_dict(hidden_size=HIDDEN_SIZE), NUM_LAYERS)

    for layer_idx in range(NUM_LAYERS):
        base = 4 * layer_idx
        assert torch.equal(q_pre[layer_idx], torch.full((HIDDEN_SIZE,), float((base + 1) * (base + 2))))
        assert torch.equal(q_post[layer_idx], torch.full((HIDDEN_SIZE,), float((base + 3) * (base + 4))))

    last = 4 * NUM_LAYERS
    assert torch.equal(q_out, torch.full((HIDDEN_SIZE,), float((last + 1) * (last + 2))))


def test_fold_queries_validates_before_folding(expect_error):
    """A missing weight has to name itself, not surface as a `KeyError` from the fold."""
    weights = _state_dict(hidden_size=HIDDEN_SIZE)
    del weights[f"{CHECKPOINT_PREFIX}layers.0.self_attention_res_norm.weight"]

    with expect_error(ValueError, "missing AttnRes weight"):
        fold_queries(weights, NUM_LAYERS)
