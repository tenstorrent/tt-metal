# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence pytest test for ``Attention.update``.

Real Llama-3.2-1B-Instruct weights (HF auth required). Round-trip on
every attention layer:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot every layer's ``wqkv`` / ``wo`` to torch.
    3.  Overwrite every layer with a constant (deliberately break the model).
    4.  Generate again -> ``tokens_broken`` (sanity: must differ from
        ``tokens_A``, otherwise the overwrite was a no-op and the rest
        of the test is meaningless).
    5.  Restore every layer via ``Attention.update(snapshot)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

Greedy generation through the full stack is the cheapest way to exercise
every consumer of the buffers (prefill matmul, decode matmul, fused
all-gather matmul if used, KV cache attention, captured traces).
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


def _wqkv_mesh_mapper(a):
    """Mirror the mesh mapping used in ``Attention.__init__`` for ``self.wqkv``."""
    import ttnn

    return ttnn.ShardTensor2dMesh(
        a.mesh_device,
        dims=(3, 2) if a.TG else (2, 3),
        mesh_shape=a.args.cluster_shape,
    )


def _wo_mesh_mapper(a):
    """Mirror the mesh mapping used in ``Attention.__init__`` for ``self.wo``."""
    import ttnn

    if a.use_fused_all_gather_matmul or a.TG:
        return ttnn.ShardTensor2dMesh(a.mesh_device, dims=(2, 3), mesh_shape=a.args.cluster_shape)
    return ttnn.ShardTensorToMesh(a.mesh_device, dim=2)


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


def _const_like(a, template, value, mapper):
    return _ttnn_like(
        template,
        torch.full(tuple(template.shape), float(value), dtype=torch.bfloat16),
        a.mesh_device,
        mapper,
    )


def _snapshot(a):
    """Read ``wqkv`` and ``wo`` back to torch.

    We only snapshot what ``Attention.update`` is responsible for. The
    prefetcher mirror ``wo_sharded_ring`` is rederived from ``wo`` inside
    ``_update_wo``, so it doesn't need to be snapshotted independently.
    """
    import ttnn

    return {"wqkv": ttnn.to_torch(a.wqkv), "wo": ttnn.to_torch(a.wo)}


def _restore(a, snap):
    a.update(
        wqkv=_ttnn_like(a.wqkv, snap["wqkv"], a.mesh_device, _wqkv_mesh_mapper(a)),
        wo=_ttnn_like(a.wo, snap["wo"], a.mesh_device, _wo_mesh_mapper(a)),
    )


def _overwrite(a, value):
    a.update(
        wqkv=_const_like(a, a.wqkv, value, _wqkv_mesh_mapper(a)),
        wo=_const_like(a, a.wo, value, _wo_mesh_mapper(a)),
    )


def _generate(completer, prompt_ids):
    return completer.generate([prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)[0]


def test_attention_update_round_trip(completer):
    """Snapshot -> overwrite -> restore must reproduce the original tokens."""
    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_A = _generate(completer, prompt_ids)

    snapshots = [_snapshot(layer.attention) for layer in completer.model.layers]

    for layer in completer.model.layers:
        _overwrite(layer.attention, OVERWRITE_VALUE)
    tokens_broken = _generate(completer, prompt_ids)
    assert tokens_broken != tokens_A, (
        f"overwriting wqkv/wo with {OVERWRITE_VALUE} did not change generation; "
        "the overwrite step was a no-op, so the rest of the test is meaningless"
    )

    for layer, snap in zip(completer.model.layers, snapshots):
        _restore(layer.attention, snap)
    tokens_B = _generate(completer, prompt_ids)
    assert tokens_B == tokens_A, (
        "Attention.update did not reproduce __init__-equivalent state: " f"tokens_A={tokens_A}, tokens_B={tokens_B}"
    )
