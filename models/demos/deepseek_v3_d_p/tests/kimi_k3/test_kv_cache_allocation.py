# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K3's shared ``allocate_kv_cache`` serves Kimi-K3 (dense MLA) and Kimi-K3.1 (Gated DSA).

K3 keeps its bf8 TILE kvpe cache and gets no index cache. K3.1 gets a scaled-FP8 kvpe cache and a
bf8 TILE indexer key cache with the kvpe cache's slabs and SP/TP split, on its full-attention layers
only. Both get the same KDA state slabs. The device test allocates every stage at K3's ``max_seq_len``, on
the PP4 8x4 stages (SP on axis 0) and on the eight 4x4 stages of the manifest's layout (SP on axis 1, TP-deduped
over 16 chips), and asserts each model's cache layouts, that K3's caches are unchanged, and that one more K3.1
user costs exactly its kvpe and index rows more per chip than one more K3 user.
"""

from __future__ import annotations

import pytest
from ttnn.device import is_blackhole

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_stage import (
    CHUNK,
    CHUNK_4X4,
    MAX_SEQ_LEN,
    MESH,
    MESH_4X4,
    SP_AXIS_4X4,
    STAGES,
    STAGES_4X4,
    TP_AXIS_4X4,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, allocate_mla_kvpe_cache

SP_AXIS, TP_AXIS = 0, 1
# Per token per full-attention layer, in DRAM: bf8 TILE row of 512+64 = 18 tiles of 1088 B per 32
# tokens; the scaled-FP8 row is 512 fp8 + 4 fp32 scales + 64 bf16 = 656 B, padded to 704; the index
# key is 4 bf8 tiles (128 wide) per 32 tokens.
K3_KVPE_BYTES_PER_TOKEN = 18 * 1088 // 32  # 612
K31_KVPE_BYTES_PER_TOKEN = 704
K31_INDEX_BYTES_PER_TOKEN = 4 * 1088 // 32  # 136


def _params(first, count, num_users=1, sparse_kv_cache_format=None, tp_shard_kv=True, placement=None):
    mesh, sp_axis, tp_axis, chunk = placement or (MESH, SP_AXIS, TP_AXIS, CHUNK)
    return PrefillRunParams(
        mesh_shape=mesh,
        num_layers=count,
        first_layer_idx=first,
        is_first_rank=first == 0,
        is_last_rank=first + count == KimiK3Config.NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=chunk,
        num_users=num_users,
        capacity_factor=2,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        tp_shard_kv=tp_shard_kv,
        sparse_kv_cache_format=sparse_kv_cache_format,
    )


def _allocate(adapter, mesh_device, params):
    return adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=adapter.load_hf_config(), params=params)


def _free(caches):
    for tensor in (getattr(caches.kvpe, "storage", None), caches.index):
        if tensor is not None:
            ttnn.deallocate(tensor)
    if caches.kda_states is not None:
        ttnn.deallocate(caches.kda_states.recurrent)
        ttnn.deallocate(caches.kda_states.convolution)


@pytest.mark.parametrize(
    "model, requested",
    [
        ("kimi_k3", MlaKvCacheFormat.SCALED_FP8),  # dense MLA cannot use a sparse kvpe
        ("kimi_k3_1", MlaKvCacheFormat.BFP8_TILE),  # sparse_sdpa cannot read bf8
    ],
)
def test_kvpe_format_must_match_the_attention(model, requested, expect_error):
    """Rejected from the config and format alone, before anything is allocated (hence no device)."""
    with expect_error(ValueError, "cannot use"):
        _allocate(get_adapter(model), None, _params(0, 24, sparse_kv_cache_format=requested))


def test_kimi_k3_1_needs_tp_deduped_caches(expect_error):
    """Sparse ttMLA only reads the TP-deduped layout, so a replicated K3.1 allocation is refused (no device)."""
    with expect_error(ValueError, "TP-deduped"):
        _allocate(get_adapter("kimi_k3_1"), None, _params(0, 24, tp_shard_kv=False))


def test_kimi_k3_1_defaults_to_scaled_fp8_and_k3_stays_dense():
    assert get_adapter("kimi_k3_1").default_sparse_kv_cache_format == MlaKvCacheFormat.SCALED_FP8
    assert get_adapter("kimi_k3").default_sparse_kv_cache_format is None


def _dram_bytes(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _layout(tensor):
    return (tuple(tensor.shape), tensor.dtype, tensor.layout, tensor.memory_config(), tensor.tensor_topology())


# (mesh, sp_axis, tp_axis, chunk), the stages, and the full-attention layers (slabs) each stage holds.
_PLACEMENTS = [
    pytest.param(
        MESH,
        (MESH, SP_AXIS, TP_AXIS, CHUNK),
        STAGES,
        [6, 6, 6, 6],
        marks=pytest.mark.requires_mesh_topology(mesh_shape=MESH, topology="mesh-8x4"),
        id="8x4-sp0",
    ),
    pytest.param(
        MESH_4X4,
        (MESH_4X4, SP_AXIS_4X4, TP_AXIS_4X4, CHUNK_4X4),
        STAGES_4X4,
        [3] * 7 + [2],
        marks=pytest.mark.requires_mesh_topology(mesh_shape=MESH_4X4, topology="mesh-4x4"),
        id="4x4-sp1",
    ),
]


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 runs on Blackhole only")
@pytest.mark.parametrize("mesh_device, placement, stages, stage_slabs", _PLACEMENTS, indirect=["mesh_device"])
def test_caches_per_stage_and_per_chip_footprint(mesh_device, placement, stages, stage_slabs):
    k3, k31 = get_adapter("kimi_k3"), get_adapter("kimi_k3_1")
    mesh, sp_axis, tp_axis, _ = placement
    # TP-deduped: every chip of the stage holds its own 1/(sp*tp) of the sequence.
    tokens_per_chip = MAX_SEQ_LEN // (mesh[0] * mesh[1])
    per_user = {}
    assert len(stages) == len(stage_slabs)
    for (first, count), expected_slabs in zip(stages, stage_slabs):
        schedule = KimiK3LayerSchedule.build(KimiK3Config, first, count)
        slabs = schedule.num_mla_layers
        assert slabs == expected_slabs, (first, count)
        for adapter in (k3, k31):
            # K3.1 as the runner builds it: the adapter's default format.
            fmt = adapter.default_sparse_kv_cache_format
            footprint = []
            for num_users in (1, 2):
                before = _dram_bytes(mesh_device)
                caches = _allocate(adapter, mesh_device, _params(first, count, num_users, fmt, placement=placement))
                footprint.append(_dram_bytes(mesh_device) - before)

                kvpe = caches.kvpe
                assert kvpe.storage.shape[0] == num_users * slabs
                assert kvpe.storage.shape[2] == tokens_per_chip
                assert caches.kda_states.layer_ids == tuple(schedule.rank_kda_layer_ids())
                assert caches.kda_states.num_slots == num_users
                if adapter is k3:
                    assert kvpe.format == MlaKvCacheFormat.BFP8_TILE and caches.index is None
                    # K3 unchanged: exactly what its allocator called before K3.1 shared it.
                    reference = allocate_mla_kvpe_cache(
                        mesh_device=mesh_device,
                        hf_config=adapter.load_hf_config(),
                        max_seq_len=MAX_SEQ_LEN,
                        mesh_shape=mesh,
                        sp_axis=sp_axis,
                        num_layers=slabs,
                        num_users=num_users,
                        tp_axis=tp_axis,
                        bank_aligned=True,
                    )
                    assert _layout(kvpe.storage) == _layout(reference.storage)
                    assert kvpe.geometry == reference.geometry
                    ttnn.deallocate(reference.storage)
                else:
                    assert kvpe.format == MlaKvCacheFormat.SCALED_FP8
                    index = caches.index
                    assert tuple(index.shape) == (num_users * slabs, 1, tokens_per_chip, 128)
                    assert (index.dtype, index.layout) == (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
                    # Same SP/TP split and placement as kvpe: one token-addressed table entry per chunk.
                    assert index.tensor_topology() == kvpe.storage.tensor_topology()
                    assert index.memory_config().nd_shard_spec.grid == kvpe.storage.memory_config().nd_shard_spec.grid
                _free(caches)
            # The marginal user: allocation granularity leaves a few KiB that do not scale with users.
            per_user[(adapter.name, first)] = footprint[1] - footprint[0]

        extra = per_user[("kimi_k3_1", first)] - per_user[("kimi_k3", first)]
        expected = (
            slabs * tokens_per_chip * (K31_KVPE_BYTES_PER_TOKEN + K31_INDEX_BYTES_PER_TOKEN - K3_KVPE_BYTES_PER_TOKEN)
        )
        assert extra == expected, (extra, expected)
