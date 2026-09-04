# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity between the vendored reference and the upstream HF model, layer by layer.

Requires the network (or a warm HF cache) and the 1.8 GB checkpoint.

The ladder matters more than the end-to-end number. A single end-to-end PCC can hide
compensating errors -- two layers wrong in opposite directions still land near the right
answer -- whereas a 13-point ladder localises the first divergence to one block. When the
TTNN port starts diverging in Phase 1, this is the harness that says where.

Measured on the pinned revisions: the vendored reference is BIT-EXACT with upstream at
every rung (max-abs 0.0), so the thresholds here are deliberately near-machine-tight.
Relaxing one should be treated as a regression, not a tolerance adjustment.
"""

import pytest
import torch

from models.experimental.nomic_embed_text_v2_moe.common import (
    capture_hidden_states,
    layer_ladder_paths,
    max_abs_diff,
    pcc,
    random_input_ids,
)
from models.experimental.nomic_embed_text_v2_moe.reference.hf_reference import (
    RemoteCodeResolutionError,
    assert_resolved_from_remote_code,
    hf_layer_ladder,
    hf_last_hidden_state,
)

pytestmark = pytest.mark.needs_weights

# Bit-exact in practice; this leaves room only for fp32 non-determinism.
LADDER_PCC = 0.9999999
LADDER_MAX_ABS = 1e-4


def test_hf_model_came_from_remote_code(hf_model):
    """The canary for the native-vs-remote collision.

    `transformers` registers a native `nomic_bert` for this exact `model_type`, targeting
    nomic-embed-text-v1.5. A bare `AutoModel.from_pretrained` resolves to it, silently
    discards every MoE tensor as UNEXPECTED, randomly initialises `gate_proj`/`up_proj`,
    and returns a working model that computes the wrong thing. If this assertion ever
    fires, the golden reference has been silently downgraded.
    """
    assert type(hf_model).__module__.startswith("transformers_modules")
    assert hasattr(hf_model, "encoder")
    # The MoE really is present in what we loaded.
    moe_keys = [k for k in hf_model.state_dict() if "experts" in k]
    assert moe_keys, "the loaded HF model has no expert weights -- native class resolved"


def test_remote_code_guard_rejects_a_native_class(expect_error):
    """The guard itself must fail on a native class rather than wave it through."""

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
    assert pcc(ours, theirs) > LADDER_PCC
    assert max_abs_diff(ours, theirs) < LADDER_MAX_ABS


def test_thirteen_point_layer_ladder(config, reference_model, hf_model):
    """`emb_ln` plus each of the 12 blocks, compared at the same input."""
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0, pad_lengths=[0, 7])
    paths = layer_ladder_paths(config.num_hidden_layers)

    ours, handles = capture_hidden_states(reference_model, paths)
    try:
        with torch.no_grad():
            reference_model(input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    theirs = hf_layer_ladder(hf_model, input_ids, attention_mask)

    assert len(paths) == 13
    failures = []
    for path in paths:
        layer_pcc = pcc(ours[path], theirs[path])
        layer_max = max_abs_diff(ours[path], theirs[path])
        if layer_pcc <= LADDER_PCC or layer_max >= LADDER_MAX_ABS:
            failures.append(f"{path}: pcc={layer_pcc:.9f} max_abs={layer_max:.3e}")
    assert not failures, "first divergence at " + failures[0] + f" (all: {failures})"


def test_moe_layers_carry_the_largest_activations(config, reference_model):
    """Documented so Phase 1 tolerances are set against the right magnitudes.

    The MoE blocks produce noticeably larger activations than the dense ones. A relative
    tolerance calibrated on layer 0 would be far too loose at layer 1.
    """
    input_ids, attention_mask = random_input_ids(2, 24, config, seed=0)
    paths = layer_ladder_paths(config.num_hidden_layers)
    captures, handles = capture_hidden_states(reference_model, paths)
    try:
        with torch.no_grad():
            reference_model(input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    absmax = {p: float(t.abs().max()) for p, t in captures.items()}
    moe_peak = max(absmax[f"encoder.layers.{i}"] for i in config.moe_layers)
    dense_first = absmax["encoder.layers.0"]
    assert moe_peak > dense_first
    for value in absmax.values():
        assert value == value and value < 1e4  # finite and not exploding


def test_upstream_requires_an_attention_mask_but_the_reference_does_not(config, reference_model, hf_model):
    """A documented upstream quirk the vendored reference deliberately does not inherit.

    Upstream calls `get_extended_attention_mask(attention_mask, ...)` unconditionally, so
    `attention_mask=None` raises inside transformers. Ours defaults to all-ones.
    """
    input_ids, attention_mask = random_input_ids(1, 8, config, seed=3)

    with torch.no_grad():
        ours = reference_model(input_ids)  # no mask: must work
    assert ours.shape == (1, 8, config.hidden_size)

    raised = None
    try:
        with torch.no_grad():
            hf_model(input_ids=input_ids, attention_mask=None)
    except Exception as exc:  # noqa: BLE001 - the type is the thing under observation
        raised = exc
    assert raised is not None, "upstream unexpectedly accepted attention_mask=None"

    # And with a mask, the two agree.
    theirs = hf_last_hidden_state(hf_model, input_ids, attention_mask)
    with torch.no_grad():
        ours_masked = reference_model(input_ids, attention_mask=attention_mask)
    assert pcc(ours_masked, theirs) > LADDER_PCC


def test_upstream_matryoshka_dim_slices_the_sequence_axis(config, hf_model):
    """Records the upstream bug that motivates doing truncation in `pipeline.py`.

    `NomicBertModel.forward(matryoshka_dim=256)` does `sequence_output[:, :256]`. That is
    the SEQUENCE axis: the returned tensor keeps its full 768-wide features and instead
    drops tokens. Truncation belongs after pooling, on the feature axis.
    """
    input_ids, attention_mask = random_input_ids(2, 10, config, seed=5)
    with torch.no_grad():
        out = hf_model(input_ids=input_ids, attention_mask=attention_mask, matryoshka_dim=256).last_hidden_state

    # If it sliced features, the last axis would be 256. It is not.
    assert out.shape == (2, 10, config.hidden_size)
