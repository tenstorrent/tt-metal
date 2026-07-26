# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-layer GLM-5.2 paged-cache prefill integration proxy.

This is the smallest full ``ttMLA`` integration test for the SP4xTP1 paged path: the
module runs a 5,120-token indexed-RoPE chunk against a one-bundle paged pool.
It deliberately does not exercise the scheduler, migration, decode, or more than
one model layer.

Run only on an exclusively assigned four-device Blackhole QuietBox::

    TT_RUN_GLM52_PAGED_MLA_PARITY=1 \
    TT_GLM52_PAGED_PREFILL=1 \
      scripts/run_safe_pytest.sh \
      models/demos/deepseek_v3_d_p/tests/paged_prefill/test_glm52_paged_mla_qb.py -s

Paged sparse attention uses UDM to read each selected row from its SP owner.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.glm52_paged_kv_cache import allocate_glm52_paged_kv_cache_pool
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

CHUNK = 5 * 1024
SP_AXIS, TP_AXIS = 0, 1
MESH_SHAPE = (4, 1)
PRIMARY_LAYERS = GLM52Config.NUM_LAYERS
SENTINEL = 0xFFFFFFFF

OPT_IN = os.environ.get("TT_RUN_GLM52_PAGED_MLA_PARITY") == "1"
PAGED_PREFILL = os.environ.get("TT_GLM52_PAGED_PREFILL") == "1"

QB_SP4_TP1_PAGED = pytest.param(
    MESH_SHAPE,
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_tensix_config": ttnn.FabricTensixConfig.UDM,
        "fabric_udm_mode": ttnn.FabricUDMMode.ENABLED,
    },
    id="glm52-sp4xtp1-udm",
)


def _mesh_composer(mesh_device, dims):
    return ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)


def _collect_output(tensor, mesh_device):
    return ttnn.to_torch(
        tensor,
        mesh_composer=_mesh_composer(mesh_device, dims=(-1, -2)),
    ).to(torch.bfloat16)


def _collect_topk(tensor, mesh_device):
    # Sequence is SP-sharded and TP is replicated.  Keep the first TP replica.
    host = ttnn.to_torch(
        tensor,
        mesh_composer=_mesh_composer(mesh_device, dims=(2, 1)),
    )
    return host[0, 0].to(torch.int64)


def _collect_cache_layer(storage, mesh_device, flat_layer):
    """Read one folded layer without copying all 78 physical layer slots."""

    layer = ttnn.slice(
        storage,
        [flat_layer, 0, 0, 0],
        [flat_layer + 1, 1, storage.shape[2], storage.shape[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    host = ttnn.to_torch(
        layer,
        mesh_composer=_mesh_composer(mesh_device, dims=(2, 1)),
    ).to(torch.bfloat16)
    ttnn.deallocate(layer)

    # Drop the TP replica, then invert the block-cyclic SP layout.  For the
    # aligned first bundle this is currently identity, but retaining the
    # explicit mapping protects the assertion if the cache layout changes.
    physical = host[0, 0]
    positions = blockcyclic_positions(MESH_SHAPE[SP_AXIS], CHUNK, CHUNK)
    natural = torch.empty_like(physical)
    natural[positions] = physical
    return natural


def _assert_topk_causal(topk):
    assert topk.shape == (CHUNK, GLM52Config.INDEX_TOPK)
    valid = topk != SENTINEL
    positions = torch.arange(CHUNK, dtype=torch.int64)[:, None]
    assert torch.all(topk[valid] >= 0)
    assert torch.all(topk[valid] <= positions.expand_as(topk)[valid])
    expected_count = torch.minimum(
        torch.arange(1, CHUNK + 1, dtype=torch.int64),
        torch.tensor(GLM52Config.INDEX_TOPK, dtype=torch.int64),
    )
    torch.testing.assert_close(valid.sum(dim=-1), expected_count, rtol=0, atol=0)


def _run_mla(mla, hidden_host, rope_tensors, caches, mesh_device, *, paged_cache=None):
    shard_dims = [None, None]
    shard_dims[TP_AXIS], shard_dims[SP_AXIS] = -1, -2
    hidden = ttnn.from_torch(
        hidden_host.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=shard_dims),
    )
    output, topk = mla.forward(
        hidden_states=hidden,
        rope_tensors=rope_tensors,
        kvpe_cache=caches.kvpe,
        actual_start=0,
        cache_user_id=0,
        cache_layer_idx=0,
        index_kv_cache=caches.index,
        return_indexer_indices=True,
        paged_cache=paged_cache,
    )
    ttnn.synchronize_device(mesh_device)
    output_host = _collect_output(output, mesh_device)
    topk_host = _collect_topk(topk, mesh_device)
    ttnn.deallocate(output)
    ttnn.deallocate(topk)
    ttnn.deallocate(hidden)
    return output_host, topk_host


@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True)
@pytest.mark.parametrize("mesh_device,device_params", [QB_SP4_TP1_PAGED], indirect=True)
@pytest.mark.skipif(not OPT_IN, reason="set TT_RUN_GLM52_PAGED_MLA_PARITY=1 on an exclusively assigned QB")
@pytest.mark.skipif(
    not PAGED_PREFILL,
    reason="set TT_GLM52_PAGED_PREFILL=1",
)
@pytest.mark.skipif(not is_blackhole(), reason="GLM-5.2 sparse prefill is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm52_one_layer_paged_mla(mesh_device, device_params, variant, config_only):
    del device_params, variant
    assert tuple(mesh_device.shape) == MESH_SHAPE

    config = config_only
    config.max_seq_len = CHUNK
    weights = random_mla_weights(config, seed=42)
    index_layers = num_full_indexer_layers(config)
    assert PRIMARY_LAYERS == 78
    assert index_layers > 0

    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=CHUNK,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        topology=ttnn.Topology.Linear,
        is_chunked=True,
        slot_num=1,
        layer_num=PRIMARY_LAYERS,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
    )
    # Galaxy exposes two forwarding links for the production collectives; the
    # four-device QuietBox proxy has one link between adjacent devices.
    mla.ccl_num_links = 1
    mla._indexer.ccl_num_links = 1
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=CHUNK,
        chunk_size_global=CHUNK,
    )

    torch.manual_seed(7)
    hidden_host = torch.randn(1, CHUNK, config.hidden_size, dtype=torch.bfloat16)

    # Paged allocation: one physical bundle, with all primary layers folded
    # bundle-major/layer-inner.  The page table has one entry and valid_end is
    # tracked independently even though compute covers the full 5,120 rows.
    paged = allocate_glm52_paged_kv_cache_pool(
        mesh_device=mesh_device,
        hf_config=config,
        mesh_shape=MESH_SHAPE,
        sp_axis=SP_AXIS,
        num_primary_layers=PRIMARY_LAYERS,
        num_logical_slots=1,
        max_sequence_length=CHUNK,
        capacity_bundles=1,
    )
    allocation = paged.allocate_chunk(logical_slot=0, start_token=0, actual_end=CHUNK)[0]
    assert allocation.physical_bundle == 0
    assert paged.valid_end(0) == CHUNK
    assert paged.kvpe.storage.shape[0] == PRIMARY_LAYERS
    assert paged.index.shape[0] == index_layers

    print("GLM52 MLA proxy: running paged-cache path", flush=True)
    paged_output, paged_topk = _run_mla(
        mla,
        hidden_host,
        rope_tensors,
        paged,
        mesh_device,
        paged_cache=paged,
    )
    print("GLM52 MLA proxy: paged-cache path complete", flush=True)
    paged_kv = _collect_cache_layer(
        paged.kvpe.storage,
        mesh_device,
        flat_layer=allocation.physical_bundle * PRIMARY_LAYERS,
    )
    paged_index_k = _collect_cache_layer(
        paged.index,
        mesh_device,
        flat_layer=allocation.physical_bundle * index_layers,
    )

    assert torch.isfinite(paged_output).all()
    assert torch.count_nonzero(paged_kv) > 0
    assert torch.count_nonzero(paged_index_k) > 0
    _assert_topk_causal(paged_topk)
