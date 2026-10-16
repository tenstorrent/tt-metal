# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for bounded sliding attention (GEMMA4_D_P_BOUNDED_SWA)."""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gemma4_d_p.tt import common
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata


@pytest.mark.parametrize(("value", "enabled"), [(None, False), ("0", False), ("1", True), ("true", True)])
def test_toggle_defaults_to_unbounded(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv("GEMMA4_D_P_BOUNDED_SWA", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_D_P_BOUNDED_SWA", value)
    assert common.bounded_swa_enabled() is enabled


def test_metadata_wraps_only_the_bounded_write_offset(monkeypatch):
    monkeypatch.setattr(PrefillMetadata, "_host_scalar", lambda self, value: value)
    monkeypatch.setattr(PrefillMetadata, "_device_scalar", lambda self: SimpleNamespace(value=None))
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda value, tensor: setattr(tensor, "value", value))
    mesh_config = SimpleNamespace(device=object())

    metadata = PrefillMetadata(mesh_config, circular_capacity=16384)
    metadata.update(slot_idx=3, kv_actual_global=5 * 8192)

    assert metadata.slot_idx.value == 3
    assert metadata.kv_actual_global.value == 40960
    assert metadata.wrapped_kv_actual_global.value == 8192
    assert PrefillMetadata(mesh_config).wrapped_kv_actual_global is None


@pytest.mark.parametrize("circular", [False, True])
def test_sliding_write_wraps_only_a_circular_cache(monkeypatch, circular):
    writes = []
    monkeypatch.setattr(
        ttnn.experimental.deepseek_prefill,
        "update_padded_kv_cache",
        lambda **kwargs: writes.append(kwargs["kv_actual_global"]),
    )
    # 2048 local rows on each of 8 CP ranks: a 16384-token cache.
    cache = SimpleNamespace(dtype=ttnn.bfloat8_b, shape=(1, 4, 2048, 256))
    chunk = SimpleNamespace(dtype=ttnn.bfloat8_b)
    mesh_config = SimpleNamespace(cp_degree=8, cp_axis=0)
    metadata = SimpleNamespace(slot_idx="slot", kv_actual_global="absolute", wrapped_kv_actual_global="wrapped")

    for prefill_metadata in (metadata, None):
        ring_prefill.write_chunk_to_sliding_ring_cache(
            cache,
            cache,
            chunk,
            chunk,
            mesh_config,
            kv_actual_global=3 * 8192,
            prefill_metadata=prefill_metadata,
            circular_kv_cache=circular,
        )

    # K then V, from the metadata tensors and then from the host scalar.
    assert writes == (["wrapped"] * 2 + [8192] * 2 if circular else ["absolute"] * 2 + [24576] * 2)


def test_external_circular_cache_must_hold_exactly_the_bounded_capacity(monkeypatch, expect_error):
    from models.demos.gemma4_d_p.config import MeshConfig
    from models.demos.gemma4_d_p.tt import attention
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import SlidingRingKVCache

    monkeypatch.setattr(attention, "load_attention_weights", lambda **_: SimpleNamespace(is_global=False))

    def build(seq_local):
        tensor = SimpleNamespace(shape=(1, 4, seq_local, 256))
        return attention.Gemma4Attention(
            config=SimpleNamespace(is_sliding=True, sliding_window_size=1024),
            state_dict={},
            ccl_manager=object(),
            mesh_config=MeshConfig(SimpleNamespace(shape=(8, 4))),
            layer_idx=0,
            max_seq_len=16384,
            ring_kv_cache=SlidingRingKVCache(tensor, tensor),
            circular_kv_cache=True,
        )

    assert build(2048).circular_kv_cache
    with expect_error(ValueError, "holds 262144 tokens"):
        build(32768)


@pytest.mark.parametrize("bounded_sliding_window_attention", [False, True])
def test_model_bounds_only_sliding_layers(monkeypatch, bounded_sliding_window_attention):
    from models.demos.gemma4_d_p.tt import model as model_module

    layers = []
    monkeypatch.setattr(
        model_module,
        "Gemma4DecoderLayer",
        lambda **kwargs: layers.append(kwargs) or SimpleNamespace(self_attn=SimpleNamespace(ring_kv_cache=None)),
    )
    monkeypatch.setattr(
        model_module,
        "PrefillMetadata",
        lambda _mesh_config, circular_capacity=None: SimpleNamespace(circular_capacity=circular_capacity),
    )
    monkeypatch.setattr(ttnn, "from_torch", lambda *_, **__: None)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda _: None)
    hf_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"], hidden_size=64, vocab_size=8, num_hidden_layers=2
    )

    model = model_module.Gemma4Model(
        mesh_config=SimpleNamespace(device=object(), cp_degree=8, tp_degree=4),
        hf_config=hf_config,
        state_dict={"model.language_model.norm.weight": None},
        ccl_manager=None,
        prefill_chunk_size=8192,
        precision={},
        max_seq_len=262144,
        bounded_sliding_window_attention=bounded_sliding_window_attention,
    )

    sliding = (16384, True) if bounded_sliding_window_attention else (262144, False)
    assert [(layer["max_seq_len"], layer["circular_kv_cache"]) for layer in layers] == [sliding, (262144, False)]
    assert model.prefill_metadata.circular_capacity == (16384 if bounded_sliding_window_attention else None)


@pytest.mark.parametrize("mesh_shape", [(8, 4), (4, 8)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("bounded_sliding_window_attention", [False, True])
def test_sliding_allocation_is_constant_in_max_seq_len(monkeypatch, mesh_shape, bounded_sliding_window_attention):
    from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches

    monkeypatch.setattr(
        ring_prefill, "_allocate_migration_ring_cache", lambda _device, shape, *_: SimpleNamespace(shape=tuple(shape))
    )
    hf_config = SimpleNamespace(
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        num_hidden_layers=6,
        hidden_size=5376,
        num_attention_heads=32,
        rms_norm_eps=1e-6,
        attention_k_eq_v=True,
        num_key_value_heads=16,
        head_dim=256,
        sliding_window=1024,
        rope_theta=10000.0,
        num_global_key_value_heads=4,
        global_head_dim=512,
        global_rope_theta=1000000.0,
        partial_rotary_factor=0.25,
    )
    cp, tp = mesh_shape
    mesh_config = SimpleNamespace(device=None, cp_degree=cp, tp_degree=tp)
    chunk_size = 8192

    def per_device_shapes(max_seq_len):
        caches = allocate_ring_kv_caches(
            mesh_config,
            hf_config,
            num_users=6,
            max_seq_len=max_seq_len,
            prefill_chunk_size=chunk_size,
            bounded_sliding_window_attention=bounded_sliding_window_attention,
        )
        sliding = [(cache.k.shape, cache.v.shape) for cache in caches.layers[:5]]
        return sliding, caches.layers[5].kv.shape

    shapes = {max_seq_len: per_device_shapes(max_seq_len) for max_seq_len in (32768, 131072, 262144)}
    for max_seq_len, (sliding, global_shape) in shapes.items():
        sliding_rows = (2 * chunk_size if bounded_sliding_window_attention else max_seq_len) // cp
        assert sliding == [((6, 16 // tp, sliding_rows, 256),) * 2] * 5
        assert global_shape[-2] == max_seq_len // cp
