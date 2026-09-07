# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compares the vendored reference against the HF implementation layer by layer using PCC.

We expect a near perfect match (PCC ~ 1.0, max-abs 0.0). Missing the bar is a regression.
The thresholds are a sanity check and are not meant to be adjusted.

Needs the checkpoint and a warm HF cache or network.
"""

import pytest
import torch

from models.common.metrics import compute_max_abs_error, compute_pcc
from models.experimental.nomic_embed_text_v2_moe.common import (
    PARITY,
    capture_hidden_states,
    layer_ladder_paths,
    random_input_ids,
)
from models.experimental.nomic_embed_text_v2_moe.reference.hf_reference import (
    RemoteCodeResolutionError,
    assert_resolved_from_remote_code,
    hf_layer_ladder,
    hf_last_hidden_state,
)

pytestmark = pytest.mark.needs_weights


def reference_ladder(model, config, input_ids, attention_mask) -> dict[str, torch.Tensor]:
    paths = layer_ladder_paths(config.num_hidden_layers)
    captures, handles = capture_hidden_states(model, paths)
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captures


def test_hf_model_came_from_remote_code(hf_model):
    """A bare AutoModel.from_pretrained resolves to the native v1.5 class, discards the expert
    weights and does not raise. If this fires, the golden reference has been downgraded."""
    assert type(hf_model).__module__.startswith("transformers_modules")
    assert [key for key in hf_model.state_dict() if "experts" in key]


def test_remote_code_guard_rejects_a_native_class(expect_error):
    class Impostor:
        pass

    with expect_error(RemoteCodeResolutionError, "native transformers"):
        assert_resolved_from_remote_code(Impostor(), "AutoModel")


@pytest.mark.parametrize(
    "batch,seqlen,pad_lengths",
    [
        (1, 8, None),
        (2, 24, [0, 7]),
        (3, 17, [0, 3, 11]),
    ],
)
def test_end_to_end_parity(config, reference_model, hf_model, batch, seqlen, pad_lengths):
    input_ids, attention_mask = random_input_ids(batch, seqlen, config, seed=seqlen, pad_lengths=pad_lengths)

    with torch.no_grad():
        ours = reference_model(input_ids, attention_mask=attention_mask)
    theirs = hf_last_hidden_state(hf_model, input_ids, attention_mask)

    assert ours.shape == theirs.shape
    assert compute_pcc(ours, theirs) > PARITY.pcc
    assert compute_max_abs_error(ours, theirs) < PARITY.max_abs


def test_layer_ladder_parity(config, reference_model, hf_model):
    """Per-layer comparison localises the first divergence to a single block."""
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0, pad_lengths=[0, 7])
    paths = layer_ladder_paths(config.num_hidden_layers)

    ours = reference_ladder(reference_model, config, input_ids, attention_mask)
    theirs = hf_layer_ladder(hf_model, input_ids, attention_mask)

    assert len(paths) == config.num_ladder_points

    failures = [
        f"{path}: pcc={compute_pcc(ours[path], theirs[path]):.9f} max_abs={compute_max_abs_error(ours[path], theirs[path]):.3e}"
        for path in paths
        if compute_pcc(ours[path], theirs[path]) <= PARITY.pcc
        or compute_max_abs_error(ours[path], theirs[path]) >= PARITY.max_abs
    ]
    assert not failures, "first divergence at " + failures[0]


def test_moe_layers_carry_the_largest_activations(config, reference_model):
    """MoE blocks produce larger activations than the dense ones, so a relative tolerance
    calibrated on layer 0 would be far too loose from layer 1 onwards."""
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0)
    captures = reference_ladder(reference_model, config, input_ids, attention_mask)

    absmax = {path: float(tensor.abs().max()) for path, tensor in captures.items()}
    moe_peak = max(absmax[f"encoder.layers.{i}"] for i in config.moe_layers)

    assert moe_peak > absmax["encoder.layers.0"]
    assert all(torch.isfinite(tensor).all() for tensor in captures.values())


def test_upstream_requires_an_attention_mask_but_the_reference_does_not(config, reference_model, hf_model):
    """Upstream calls get_extended_attention_mask unconditionally and raises on None."""
    input_ids, attention_mask = random_input_ids(1, 8, config, seed=3)

    with torch.no_grad():
        ours = reference_model(input_ids)
    assert ours.shape == (1, 8, config.hidden_size)

    raised = None
    try:
        with torch.no_grad():
            hf_model(input_ids=input_ids, attention_mask=None)
    except Exception as exc:  # noqa: BLE001 - the exception type is what is under observation
        raised = exc
    assert raised is not None, "upstream unexpectedly accepted attention_mask=None"

    with torch.no_grad():
        ours_masked = reference_model(input_ids, attention_mask=attention_mask)
    assert compute_pcc(ours_masked, hf_last_hidden_state(hf_model, input_ids, attention_mask)) > PARITY.pcc


def test_upstream_matryoshka_dim_slices_the_sequence_axis(config, hf_model):
    """Upstream slices sequence_output[:, :matryoshka_dim], dropping tokens rather than
    features. pipeline.py truncates after pooling instead."""
    seqlen = 10
    input_ids, attention_mask = random_input_ids(2, seqlen, config, seed=5)
    with torch.no_grad():
        out = hf_model(input_ids=input_ids, attention_mask=attention_mask, matryoshka_dim=256).last_hidden_state

    assert out.shape == (2, seqlen, config.hidden_size)
