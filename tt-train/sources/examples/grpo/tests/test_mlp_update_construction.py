# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``MLP.update`` (real weights, HF auth).

Round-trip per layer: generate -> snapshot w1/w2/w3 -> overwrite with a constant
(must change output) -> restore via update -> generate must match the original.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer, to_torch_2d

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy decoding -> deterministic, byte-comparable
OVERWRITE_VALUE = 0.0


@pytest.fixture(scope="module")
def completer():
    with open_completer(dummy_weights=False) as c:
        yield c


def _snapshot_mlp_hf(mlp):
    """Read ``w1`` / ``w2`` / ``w3`` back as HF-shape torch tensors (internal
    storage is HF transposed, so squeeze to 2D and transpose)."""
    gate = to_torch_2d(mlp.w1).transpose(0, 1).contiguous()  # (H, I) -> (I, H)
    up = to_torch_2d(mlp.w3).transpose(0, 1).contiguous()
    down = to_torch_2d(mlp.w2).transpose(0, 1).contiguous()  # (I, H) -> (H, I)
    return {"gate_proj": gate, "up_proj": up, "down_proj": down}


def _restore_mlp(mlp, snap):
    gate_in = as_update_input(snap["gate_proj"], mlp.mesh_device)
    up_in = as_update_input(snap["up_proj"], mlp.mesh_device)
    down_in = as_update_input(snap["down_proj"], mlp.mesh_device)
    mlp.update(gate_proj=gate_in, up_proj=up_in, down_proj=down_in)


def _overwrite_mlp(mlp, value):
    H = mlp.args.dim
    I = mlp.args.hidden_dim
    gate_hf = torch.full((I, H), value, dtype=torch.bfloat16)
    up_hf = torch.full((I, H), value, dtype=torch.bfloat16)
    down_hf = torch.full((H, I), value, dtype=torch.bfloat16)

    mlp.update(
        gate_proj=as_update_input(gate_hf, mlp.mesh_device),
        up_proj=as_update_input(up_hf, mlp.mesh_device),
        down_proj=as_update_input(down_hf, mlp.mesh_device),
    )


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_mlp_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    model = completer.models[0]
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)

    snapshots = [_snapshot_mlp_hf(layer.feed_forward) for layer in model.layers]

    for layer in model.layers:
        _overwrite_mlp(layer.feed_forward, OVERWRITE_VALUE)
    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting gate/up/down_proj with {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    for layer, snap in zip(model.layers, snapshots):
        _restore_mlp(layer.feed_forward, snap)
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "MLP.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
