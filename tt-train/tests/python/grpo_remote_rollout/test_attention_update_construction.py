# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``Attention.update`` (real weights, HF auth).

Round-trip per layer: generate -> snapshot wqkv/wo -> overwrite with a constant
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


def _snapshot_attn_hf(a):
    """Invert the constructor to read ``wqkv`` / ``wo`` back as HF-shape torch
    tensors (q/k/v/o_proj), ready for ``as_update_input``."""
    n_q = a.n_heads * a.head_dim
    n_kv = a.n_kv_heads * a.head_dim

    wqkv_t = to_torch_2d(a.wqkv).transpose(0, 1)  # (qkv, H)
    q, k, v = torch.split(wqkv_t, [n_q, n_kv, n_kv], dim=0)
    o = to_torch_2d(a.wo).transpose(0, 1)  # (H, n_q) -- HF o_proj shape

    return {
        "q_proj": q.contiguous(),
        "k_proj": k.contiguous(),
        "v_proj": v.contiguous(),
        "o_proj": o.contiguous(),
    }


def _restore_attn(a, snap):
    q_in = as_update_input(snap["q_proj"], a.mesh_device)
    k_in = as_update_input(snap["k_proj"], a.mesh_device)
    v_in = as_update_input(snap["v_proj"], a.mesh_device)
    o_in = as_update_input(snap["o_proj"], a.mesh_device)
    a.update(q_proj=q_in, k_proj=k_in, v_proj=v_in, o_proj=o_in)


def _overwrite_attn(a, value):
    H = a.hidden_size
    D = a.head_dim
    n_q = a.n_heads * D
    n_kv = a.n_kv_heads * D

    q_hf = torch.full((n_q, H), value, dtype=torch.bfloat16)
    k_hf = torch.full((n_kv, H), value, dtype=torch.bfloat16)
    v_hf = torch.full((n_kv, H), value, dtype=torch.bfloat16)
    o_hf = torch.full((H, n_q), value, dtype=torch.bfloat16)

    a.update(
        q_proj=as_update_input(q_hf, a.mesh_device),
        k_proj=as_update_input(k_hf, a.mesh_device),
        v_proj=as_update_input(v_hf, a.mesh_device),
        o_proj=as_update_input(o_hf, a.mesh_device),
    )


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_attention_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    model = completer.models[0]
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)

    snapshots = [_snapshot_attn_hf(layer.attention) for layer in model.layers]

    for layer in model.layers:
        _overwrite_attn(layer.attention, OVERWRITE_VALUE)
    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting q/k/v/o with {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    for layer, snap in zip(model.layers, snapshots):
        _restore_attn(layer.attention, snap)
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "Attention.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
