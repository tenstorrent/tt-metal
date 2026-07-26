# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in four-device tests for the existing cache writer as a paged primitive.

Enable only on an exclusively assigned Blackhole QB:

    TT_RUN_PAGED_PREFILL_QB_TESTS=1 \
      scripts/run_safe_pytest.sh --dev \
      models/demos/deepseek_v3_d_p/tests/paged_prefill/test_paged_writer_qb.py -s
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.paged_prefill.support import (
    INDEX_HEAD_DIM,
    KVPE_HEAD_DIM,
    PREFILL_PAGE_TOKENS,
    sample_dram_memory,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

QB_DEVICE_CASES = [
    pytest.param(
        (4, 1),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
        id="qb-sp4xtp1-ring",
    ),
]

CACHE_CASES = [
    pytest.param(KVPE_HEAD_DIM, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="kvpe-bf16-rm"),
    pytest.param(INDEX_HEAD_DIM, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="index-bfp8-tile"),
]

QB_OPT_IN = os.environ.get("TT_RUN_PAGED_PREFILL_QB_TESTS") == "1"


def _mapped_local_scratch(pool, physical_pages, num_layers, layer_idx):
    pieces = [
        ttnn.slice(
            pool,
            [physical * num_layers + layer_idx, 0, 0, 0],
            [physical * num_layers + layer_idx + 1, 1, pool.shape[2], pool.shape[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for physical in physical_pages
    ]
    scratch = ttnn.concat(pieces, dim=2)
    for piece in pieces:
        ttnn.deallocate(piece)
    return scratch


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_DEVICE_CASES,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("head_dim,dtype,layout", CACHE_CASES)
@pytest.mark.skipif(not QB_OPT_IN, reason="set TT_RUN_PAGED_PREFILL_QB_TESTS=1 on an exclusively assigned QB")
@pytest.mark.skipif(not is_blackhole(), reason="GLM-5.2 sparse prefill is Blackhole-only")
@pytest.mark.timeout(0)
def test_noncontiguous_page_writes_reconstruct_logical_order(
    mesh_device,
    device_params,
    head_dim,
    dtype,
    layout,
):
    del device_params  # Indirect fixture configures the requested linear/ring fabric.
    sp, tp = tuple(mesh_device.shape)
    sp_axis = 0
    num_layers = 2
    layer_idx = 1
    pool_pages = 5

    # This is the mapping produced by interleaving two logical requests and
    # reusing the second request's released physical bundle.
    physical_pages = (0, 2, 1)

    before = sample_dram_memory(mesh_device, "before-pool")
    pool = init_kvpe_cache(
        kvpe_cache_head_dim=head_dim,
        mesh_device=mesh_device,
        seq_len=PREFILL_PAGE_TOKENS,
        mesh_shape=[sp, tp],
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=pool_pages,
        dtype=dtype,
        layout=layout,
    )
    after_pool = sample_dram_memory(mesh_device, "after-pool")
    assert after_pool.allocated_bytes > before.allocated_bytes
    assert after_pool.num_banks > 0

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    local_tokens = PREFILL_PAGE_TOKENS // sp

    torch.manual_seed(20260725)
    encoded_pages = []
    for physical_page in physical_pages:
        source = torch.randn(1, 1, PREFILL_PAGE_TOKENS, head_dim, dtype=torch.bfloat16)
        tt_source = ttnn.from_torch(
            source,
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tt_source_sharded = ttnn.to_memory_config(tt_source, pool.memory_config())
        ttnn.deallocate(tt_source)
        tt_source = tt_source_sharded
        encoded_pages.append(ttnn.to_torch(tt_source, mesh_composer=composer).to(torch.bfloat16)[0, 0])
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            pool,
            tt_source,
            slot_idx=physical_page,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=0,
            cluster_axis=sp_axis,
        )
        ttnn.deallocate(tt_source)

    scratch = _mapped_local_scratch(pool, physical_pages, num_layers, layer_idx)
    ttnn.synchronize_device(mesh_device)
    actual = ttnn.to_torch(scratch, mesh_composer=composer).to(torch.bfloat16)[0, 0]
    expected = torch.cat(
        [
            encoded_pages[logical][chip * local_tokens : (chip + 1) * local_tokens]
            for chip in range(sp)
            for logical in range(len(physical_pages))
        ],
        dim=0,
    )
    assert torch.equal(actual, expected), (
        f"paged writer mismatch for SP={sp}, TP={tp}, dtype={dtype}; "
        f"max diff={(actual.float() - expected.float()).abs().max().item()}"
    )

    # A page/layer never leased to request 0 must retain the zero-fill sentinel.
    untouched = ttnn.slice(
        pool,
        [3 * num_layers + layer_idx, 0, 0, 0],
        [3 * num_layers + layer_idx + 1, 1, pool.shape[2], pool.shape[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    untouched_host = ttnn.to_torch(untouched, mesh_composer=composer).float()
    assert untouched_host.abs().max().item() == 0.0

    ttnn.deallocate(untouched)
    ttnn.deallocate(scratch)
    ttnn.deallocate(pool)
