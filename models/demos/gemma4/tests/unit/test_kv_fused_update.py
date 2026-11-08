# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GEMMA4_KV_FUSED_UPDATE (item 11 of the gemma4_12b_qb2 port): the fused K+V paged cache write, one
``paged_fused_update_cache`` program per layer, equals the two ``paged_update_cache`` programs bit for bit.

One decode step of a ``Gemma4Attention`` layer on a fresh paged cache, knob off and knob on from the same seed: every
shard of the K and V caches and the attention output must be identical. Covers the composite head path (V resharded
onto the disjoint grid by the reshard it pays anyway) and the ``QkvHeadsNormRope`` path (the op emits V on the next
``batch`` cores), the sliding and the full layers, batch 1 and 32, bf16 and bfp8 caches.

    pytest -k "1x4" models/demos/gemma4/tests/unit/test_kv_fused_update.py
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.decode import (
    KV_FUSED_STATS,
    KV_FUSED_UPDATE_ENV,
    kv_fused_update_default,
    kv_fused_update_reason,
)
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.ccl import CCLManager
from models.tt_transformers.tt.common import PagedAttentionConfig

from ...tests.test_factory import TestFactory, mesh_tp_and_axis, parametrize_mesh_with_fabric
from .test_attention import _from_device, _kv_fill_to_tt, _skip_if_l1_overflow

CACHE_LEN = 512
QKV_PATHS = {
    "composite": {"GEMMA4_QKV_HEADS_FUSED": "0", "GEMMA4_QKV_NORM_ROPE_FUSED": "0", "GEMMA4_QKV_L1_OUT": "0"},
    "heads_fused": {"GEMMA4_QKV_HEADS_FUSED": "1", "GEMMA4_QKV_NORM_ROPE_FUSED": "0", "GEMMA4_QKV_L1_OUT": "1"},
}


@contextmanager
def _env(values):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _shards(t):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None


def _decode_step(mesh_device, layer_idx, batch, cache_dtype, qkv_path, knob, seed):
    """One decode step from a fresh paged cache: (output rows, K shards, V shards, fused-program calls, reason)."""
    torch.manual_seed(seed)
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx)
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    state_dict = {k: v.clone() for k, v in hf_layer.self_attn.state_dict().items() if not k.startswith("v_norm")}
    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    num_devices = mesh_device.get_num_devices()
    replicate = _replicate(mesh_device)
    positions = [CACHE_LEN - b for b in range(batch)]
    block_size = 64
    blocks_per_user = (CACHE_LEN + block_size) // block_size + 1
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=blocks_per_user * batch)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=tp_axis)
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    kv_cache = init_kv_cache(
        mesh_device=mesh_device, config=config, paged_attention_config=paged_attention_config, cache_dtype=cache_dtype
    )
    tt_attn = Gemma4Attention(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        layer_idx=layer_idx,
    )
    tt_attn.kv_cache = kv_cache
    page_table = torch.zeros(batch, blocks_per_user * batch, dtype=torch.int32)
    for b in range(batch):
        page_table[b, :blocks_per_user] = torch.arange(
            b * blocks_per_user, (b + 1) * blocks_per_user, dtype=torch.int32
        )
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )
    k_cache_tt, v_cache_tt = kv_cache
    for b in range(batch):
        for cache in (k_cache_tt, v_cache_tt):
            data = torch.randn(1, config.num_key_value_heads, CACHE_LEN, config.head_dim)
            fill = _kv_fill_to_tt(
                data,
                mesh_device,
                num_kv_heads=config.num_key_value_heads,
                num_attention_heads=config.num_attention_heads,
                tp=tp,
                num_devices=num_devices,
            )
            if fill.dtype != cache_dtype:
                fill = ttnn.typecast(fill, cache_dtype)  # paged_fill_cache wants the cache dtype (bfp8 caches)
            ttnn.experimental.paged_fill_cache(cache, fill, page_table_tt, batch_idx=b)
    x_tt = ttnn.from_torch(
        torch.randn(1, 1, batch, config.hidden_size).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    cos_tt, sin_tt = TestFactory.create_tt_rope_cache_2d(mesh_device, hf_text_config, CACHE_LEN + 128, layer_idx)
    pos_padded = torch.zeros(1, 32, dtype=torch.int32)
    pos_padded[0, :batch] = torch.tensor(positions, dtype=torch.int32)
    position_idx_tt = ttnn.from_torch(
        pos_padded, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=replicate
    )
    position_idx_cache_tt = ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )
    nkv_local = 1 if tt_attn.weights.kv_replicated else config.num_key_value_heads // tp
    reason = kv_fused_update_reason(
        k_cache_tt,
        v_cache_tt,
        config,
        nkv_local,
        batch=batch,
        page_table=page_table_tt,
        sequential=False,
        qkv_path=qkv_path,
    )
    with _env({**QKV_PATHS[qkv_path], KV_FUSED_UPDATE_ENV: "1" if knob else "0"}):
        before = KV_FUSED_STATS["fused"]
        if qkv_path == "heads_fused":
            # the model's per-step gather (tt/model.py): row u of the [1, 1, 32, head_dim] table = user u's position
            cos_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_tt, cos_tt, layout=ttnn.TILE_LAYOUT))
            sin_pos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx_tt, sin_tt, layout=ttnn.TILE_LAYOUT))
            out = tt_attn(
                x_tt,
                rope_mats=(cos_pos, sin_pos),
                position_idx=position_idx_tt,
                is_decode=True,
                page_table=page_table_tt,
                position_idx_cache=position_idx_cache_tt,
                rope_presliced=True,
            )
        else:
            out = tt_attn(
                x_tt,
                rope_mats=(cos_tt, sin_tt),
                position_idx=position_idx_tt,
                is_decode=True,
                page_table=page_table_tt,
                position_idx_cache=position_idx_cache_tt,
            )
        fused_calls = KV_FUSED_STATS["fused"] - before
    rows = _from_device(out, mesh_device)[:, :, :batch, :].float()
    k_shards, v_shards = _shards(k_cache_tt), _shards(v_cache_tt)
    for t in (k_cache_tt, v_cache_tt, out, x_tt, page_table_tt, position_idx_tt, position_idx_cache_tt):
        ttnn.deallocate(t)
    return rows, k_shards, v_shards, fused_calls, reason


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding", "global"])
@pytest.mark.parametrize("batch", [1, 32], ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
@pytest.mark.parametrize("qkv_path", list(QKV_PATHS))
def test_kv_fused_update_bit_exact(mesh_device, device_params, layer_idx, batch, cache_dtype, qkv_path, reset_seeds):
    """Knob off vs on from one seed: identical K / V caches (every shard) and attention output."""
    config = Gemma4AttentionConfig(TestFactory.create_hf_config(), layer_idx)
    _skip_if_l1_overflow(config, mesh_device)
    off = _decode_step(mesh_device, layer_idx, batch, cache_dtype, qkv_path, False, seed=1234)
    on = _decode_step(mesh_device, layer_idx, batch, cache_dtype, qkv_path, True, seed=1234)
    assert off[3] == 0, "the fused write ran with the knob off"
    assert on[4] is None, f"the fused write refused this layer: {on[4]}"
    assert on[3] > 0, "the fused write did not run with the knob on"
    for name, a, b in (("K", off[1], on[1]), ("V", off[2], on[2])):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(
                x, y
            ), f"{name} cache shard {i} differs (max |diff| {(x.float() - y.float()).abs().max()})"
    assert torch.equal(off[0], on[0]), f"attention output differs (max |diff| {(off[0] - on[0]).abs().max()})"


def test_kv_fused_update_reason_gates():
    """Host: every condition the fused program cannot take keeps the two-program write."""
    own = SimpleNamespace(padded_shape=(64, 2, 64, 256), dtype="bf16")
    cfg = SimpleNamespace(head_dim=256, cache_position_modulo=None)
    ok = dict(batch=1, page_table=object(), sequential=False, qkv_path="composite")
    assert kv_fused_update_reason(own, own, cfg, 2, **ok) is None
    assert kv_fused_update_reason(own, own, cfg, 2, **{**ok, "qkv_path": "heads_fused"}) is None
    assert kv_fused_update_reason(own, own, cfg, 2, **{**ok, "page_table": None})
    assert kv_fused_update_reason(own, own, cfg, 2, **{**ok, "sequential": True, "batch": 4})
    assert kv_fused_update_reason(own, own, cfg, 2, **{**ok, "sequential": True, "batch": 1}) is None
    assert kv_fused_update_reason(own, own, cfg, 2, **{**ok, "qkv_path": "qkv_norm_rope"})
    assert kv_fused_update_reason(own, own, SimpleNamespace(head_dim=256, cache_position_modulo=1024), 2, **ok)
    # the full layer's view (1 head x 512) of a buffer allocated for the sliding layers (2 heads x 256): same bytes
    # per block, a different cache shape than the op would read
    shared = SimpleNamespace(padded_shape=(64, 2, 64, 256), dtype="bf16")
    assert kv_fused_update_reason(shared, shared, SimpleNamespace(head_dim=512, cache_position_modulo=None), 1, **ok)
    other = SimpleNamespace(padded_shape=(64, 2, 64, 256), dtype="bfp8")
    assert kv_fused_update_reason(own, other, cfg, 2, **ok)


def test_kv_fused_update_default_policy(monkeypatch):
    """Host: the env var overrides, else the measured-shape policy."""
    mesh = SimpleNamespace(shape=(1, 4))
    monkeypatch.setenv(KV_FUSED_UPDATE_ENV, "1")
    assert kv_fused_update_default(mesh) is True
    monkeypatch.setenv(KV_FUSED_UPDATE_ENV, "0")
    assert kv_fused_update_default(mesh) is False
    monkeypatch.delenv(KV_FUSED_UPDATE_ENV)
    from models.demos.gemma4.tt.attention import decode as decode_module

    monkeypatch.setattr(decode_module, "KV_FUSED_UPDATE_DEFAULT_SHAPES", ((1, 4), (1, 8)))
    monkeypatch.setattr(decode_module, "is_blackhole", lambda: True)
    assert kv_fused_update_default(mesh) is True
    assert kv_fused_update_default(SimpleNamespace(shape=(1, 8))) is True
    assert kv_fused_update_default(SimpleNamespace(shape=(8, 4))) is False
    assert kv_fused_update_default(None) is False
    monkeypatch.setattr(decode_module, "is_blackhole", lambda: False)  # Wormhole (T3K is (1,8) too)
    assert kv_fused_update_default(mesh) is False
    assert kv_fused_update_default(SimpleNamespace(shape=(1, 8))) is False
