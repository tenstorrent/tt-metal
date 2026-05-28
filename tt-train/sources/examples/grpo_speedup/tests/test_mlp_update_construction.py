# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence pytest test for ``MLP.update``.

Real Llama-3.2-1B-Instruct weights (HF auth required). Round-trip on
every MLP layer:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot every layer's ``w1`` / ``w2`` / ``w3`` to torch.
    3.  Overwrite every layer with a constant (deliberately break the model).
    4.  Generate again -> ``tokens_broken`` (sanity: must differ from
        ``tokens_A``, otherwise the overwrite was a no-op).
    5.  Restore every layer via ``MLP.update(w1=..., w2=..., w3=...)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

Mirrors ``test_attention_update_construction.py``'s structure.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import build_completer, teardown_completer

PROMPT = "Explain a tensor in a paragraph."
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy decoding -> deterministic, byte-comparable
OVERWRITE_VALUE = 0.0


@pytest.fixture(scope="module")
def completer():
    c = build_completer(dummy_weights=False)
    try:
        yield c
    finally:
        teardown_completer(c)


def _w1_w3_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w1`` and ``self.w3``."""
    import ttnn

    dims = (-1, -2) if mlp.args.is_galaxy else (-2, -1)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _w2_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w2``."""
    import ttnn

    dims = (-2, -1) if mlp.args.is_galaxy else (-1, -2)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _ttnn_like(template, torch_t, device, mapper):
    """Push a torch tensor onto device with the same dtype/layout/memcfg as ``template``."""
    import ttnn

    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=device,
        memory_config=template.memory_config(),
        mesh_mapper=mapper,
    )


def _const_like(mlp, template, value, mapper):
    return _ttnn_like(
        template,
        torch.full(tuple(template.shape), float(value), dtype=torch.bfloat16),
        mlp.mesh_device,
        mapper,
    )


def _snapshot(mlp):
    import ttnn

    return {"w1": ttnn.to_torch(mlp.w1), "w2": ttnn.to_torch(mlp.w2), "w3": ttnn.to_torch(mlp.w3)}


def _restore(mlp, snap):
    mlp.update(
        w1=_ttnn_like(mlp.w1, snap["w1"], mlp.mesh_device, _w1_w3_mesh_mapper(mlp)),
        w2=_ttnn_like(mlp.w2, snap["w2"], mlp.mesh_device, _w2_mesh_mapper(mlp)),
        w3=_ttnn_like(mlp.w3, snap["w3"], mlp.mesh_device, _w1_w3_mesh_mapper(mlp)),
    )


def _overwrite(mlp, value):
    mlp.update(
        w1=_const_like(mlp, mlp.w1, value, _w1_w3_mesh_mapper(mlp)),
        w2=_const_like(mlp, mlp.w2, value, _w2_mesh_mapper(mlp)),
        w3=_const_like(mlp, mlp.w3, value, _w1_w3_mesh_mapper(mlp)),
    )


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_mlp_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)

    snapshots = [_snapshot(layer.feed_forward) for layer in completer.model.layers]

    for layer in completer.model.layers:
        _overwrite(layer.feed_forward, OVERWRITE_VALUE)
    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting w1/w2/w3 with {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    for layer, snap in zip(completer.model.layers, snapshots):
        _restore(layer.feed_forward, snap)
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "MLP.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
