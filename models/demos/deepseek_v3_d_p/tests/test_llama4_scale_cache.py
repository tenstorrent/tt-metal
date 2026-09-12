# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only coverage for the sharing of Mistral's llama4 query-scale cache.

One entry is a [1, heads_local, chunk, width] bf16 device tensor (3.28 MB/device at 8x4 / chunk 5120)
and an offset is visited once per request, so a per-ttMLA cache multiplied residency by the layer
count: 24 GB/device at 1,048,576 tokens over 36 layers. TtPrefillTransformer builds ONE dict and
threads it through every TtPrefillBlock into ttMLA.

What must hold is that every layer reaches the SAME dict and that the cache key stays
layer-independent (nothing the tensor derives from varies by layer -- see ttMLA._llama4_scale). Both
are checked here without a device; a memory benchmark would need the full model and prove less."""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.tt import tt_prefill_block, tt_prefill_transformer
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

NUM_LAYERS = 4  # enough to distinguish "shared" from "per layer"; the real model has 36


class _FakeMeshDevice:
    """Only the attributes the two constructors read on the paths under test."""

    shape = [8, 4]

    def get_num_devices(self):
        return 32


def _config():
    """Mistral-Small-4's MLA-relevant fields. has_indexer=False keeps the DSA paths out."""
    return SimpleNamespace(
        hidden_size=5120,
        num_attention_heads=32,
        kv_lora_rank=256,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
        vocab_size=131072,
        has_indexer=False,
        rope_scaling={"llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 5120},
    )


class _Recorder:
    """Stands in for a constructed submodule, keeping the kwargs it was handed."""

    seen = []

    def __init__(self, *args, **kwargs):
        type(self).seen.append(kwargs)


def _recorder(monkeypatch, module, name):
    cls = type(f"Recorded{name}", (_Recorder,), {"seen": []})
    monkeypatch.setattr(module, name, cls)
    return cls


def test_transformer_threads_one_cache_into_every_block(monkeypatch):
    """All NUM_LAYERS blocks get the transformer's single dict -- the same object, not equal copies."""
    blocks = _recorder(monkeypatch, tt_prefill_transformer, "TtPrefillBlock")
    _recorder(monkeypatch, tt_prefill_transformer, "RotarySetup")

    model = TtPrefillTransformer(
        mesh_device=_FakeMeshDevice(),
        config=_config(),
        model_cfg=SimpleNamespace(NUM_DENSE_LAYERS=1),
        state_dict={"layers": [{} for _ in range(NUM_LAYERS)]},
        num_layers=NUM_LAYERS,
        seq_len=5120,
        is_first_rank=False,  # skip the embedding / norm / LM head, which need a real device
        is_last_rank=False,
    )

    caches = [kwargs["llama4_scale_cache"] for kwargs in blocks.seen]
    assert len(caches) == NUM_LAYERS
    assert all(c is model._llama4_scale_cache for c in caches)


def _build_block(llama4_scale_cache=...):
    """A kv_only block: attn_norm + MLA only, so no FFN/MoE to stub out."""
    kwargs = {} if llama4_scale_cache is ... else {"llama4_scale_cache": llama4_scale_cache}
    return TtPrefillBlock(
        mesh_device=_FakeMeshDevice(),
        config=_config(),
        model_cfg=SimpleNamespace(NUM_DENSE_LAYERS=1),
        state_dict={},
        layer_idx=0,
        seq_len=5120,
        kv_only=True,
        **kwargs,
    )


def test_block_forwards_cache_to_mla(monkeypatch):
    """The block passes the dict straight through to its ttMLA."""
    _recorder(monkeypatch, tt_prefill_block, "TtDistributedRmsNorm")
    mlas = _recorder(monkeypatch, tt_prefill_block, "ttMLA")

    shared = {}
    _build_block(shared)
    assert mlas.seen[-1]["llama4_scale_cache"] is shared


def test_cache_is_optional(monkeypatch):
    """Omitting the kwarg reaches ttMLA as None, which is what keeps every existing caller
    (7 constructing ttMLA directly, 3 TtPrefillBlock, 3 TtPrefillTransformer) unchanged."""
    _recorder(monkeypatch, tt_prefill_block, "TtDistributedRmsNorm")
    mlas = _recorder(monkeypatch, tt_prefill_block, "ttMLA")

    _build_block()
    assert mlas.seen[-1]["llama4_scale_cache"] is None


@pytest.mark.parametrize("start", [0, 5120])
def test_scale_lookup_is_layer_independent(start):
    """Two layers sharing one dict hit the same entry: the key is (offset, seq_len_local) with no
    layer term, so a layer's lookup is another layer's hit. Guards against a layer_idx creeping into
    the key, which would silently restore the per-layer copies."""
    shared = {}
    layers = [SimpleNamespace(layer_idx=i, _llama4_cache=shared) for i in range(NUM_LAYERS)]
    sentinel = object()
    shared[(start, 640)] = sentinel

    for layer in layers:
        assert ttMLA._llama4_scale(layer, start, 640, None) is sentinel
    assert len(shared) == 1
