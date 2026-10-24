# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-sharded absorbed Q (ttMLA seq_sharded_q / MLA_SEQ_SHARDED_Q) against the head-sharded path.

GLM-5.3 on the production 8x4 torus-XY mesh at the production 5120-token chunk (640 rows per chip, 160 per TP
quarter), random weights. One full-indexer layer feeds its top-k to one shared layer, over two chunks (so the
second chunk ropes and attends at a non-zero start). Each layer is built twice from the same weights, once per
mode; the shared layer gets the SAME indices in both modes (the head-sharded run's) so its comparison isolates
the attention path. Prints "SEQQ_MLA" lines: per-chunk output PCC, index agreement, DRAM cost of the mode.
"""

import copy
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from tests.ttnn.utils_for_testing import comp_pcc

SP_AXIS, TP_AXIS = 0, 1
CHUNK = 5120
N_CHUNKS = int(os.environ.get("SEQQ_MLA_CHUNKS", "2"))
FULL_LAYER, SHARED_LAYER = 0, 3


def log(*a):
    print("SEQQ_MLA", *a, flush=True)


def dram_alloc_mib(mesh):
    return ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 2**20


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=1536,
                trace_region_size=64 * 1024 * 1024,
            ),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm_5_3"])
@pytest.mark.parametrize("cache_format", [MlaKvCacheFormat.SCALED_FP8], ids=["kv_scaled_fp8"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_seqq_mla_matches_head_sharded(mesh_device, device_params, variant, cache_format, config_only):
    mesh = mesh_device
    config = copy.deepcopy(config_only)
    seq_len = CHUNK * N_CHUNKS
    config.max_seq_len = seq_len
    assert not indexer_layer_is_reused(config, FULL_LAYER) and indexer_layer_is_reused(config, SHARED_LAYER)
    weights = {layer: random_mla_weights(config, seed=100 + layer) for layer in (FULL_LAYER, SHARED_LAYER)}
    rope_tensors = RotarySetup(config, mesh, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=CHUNK
    )
    torch.manual_seed(7)
    hidden = torch.randn(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[TP_AXIS], shard_dims[SP_AXIS] = -1, -2
    composer = ttnn.ConcatMesh2dToTensor(mesh, dims=shard_dims, mesh_shape=mesh.shape)
    index_cache_layers = sum(kind == "full" for kind in config.indexer_types)

    def build(layer, seq_q):
        before = dram_alloc_mib(mesh)
        mla = ttMLA(
            config,
            weights[layer],
            mesh,
            layer_idx=layer,
            seq_len=seq_len,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
            weight_cache_path=None,
            sparse_kv_cache_format=cache_format,
            is_chunked=True,
            active_seq_len=CHUNK,
            layer_num=1,
            seq_sharded_q=seq_q,
        )
        assert mla._seq_q == seq_q, f"layer {layer}: seq_sharded_q={seq_q} did not resolve (got {mla._seq_q})"
        log(f"built layer {layer} seq_q={seq_q}: DRAM +{dram_alloc_mib(mesh) - before:.1f} MiB/bank")
        return mla

    def run(full, shared, ref_indices=None):
        kvpe_full = init_mla_kv_cache(
            cache_format=cache_format,
            hf_config=config,
            mesh_device=mesh,
            seq_len=seq_len,
            mesh_shape=list(mesh.shape),
            sp_axis=SP_AXIS,
            num_kvpe_cache_layers=1,
            tp_axis=TP_AXIS,
        )
        kvpe_shared = init_mla_kv_cache(
            cache_format=cache_format,
            hf_config=config,
            mesh_device=mesh,
            seq_len=seq_len,
            mesh_shape=list(mesh.shape),
            sp_axis=SP_AXIS,
            num_kvpe_cache_layers=1,
            tp_axis=TP_AXIS,
        )
        index_kv = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh,
            seq_len=seq_len,
            mesh_shape=list(mesh.shape),
            sp_axis=SP_AXIS,
            num_kvpe_cache_layers=index_cache_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
            tp_axis=TP_AXIS,
        )
        outs_full, outs_shared, indices = [], [], []
        for c in range(N_CHUNKS):
            start = c * CHUNK
            tt_hidden = ttnn.from_torch(
                hidden[:, start : start + CHUNK].unsqueeze(0),
                device=mesh,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=shard_dims),
            )
            out_f, idx = full.forward(
                tt_hidden,
                rope_tensors,
                kvpe_full,
                actual_start=start,
                index_kv_cache=index_kv,
                return_indexer_indices=True,
            )
            out_s = shared.forward(
                tt_hidden,
                rope_tensors,
                kvpe_shared,
                actual_start=start,
                indexer_indices=idx if ref_indices is None else ref_indices[c],
            )
            ttnn.synchronize_device(mesh)
            outs_full.append(ttnn.to_torch(out_f, mesh_composer=composer).float())
            outs_shared.append(ttnn.to_torch(out_s, mesh_composer=composer).float())
            indices.append(idx)
            ttnn.deallocate(out_f)
            ttnn.deallocate(out_s)
            ttnn.deallocate(tt_hidden)
        return outs_full, outs_shared, indices

    full_ref, shared_ref = build(FULL_LAYER, False), build(SHARED_LAYER, False)
    full_new, shared_new = build(FULL_LAYER, True), build(SHARED_LAYER, True)
    try:
        ref_f, ref_s, ref_idx = run(full_ref, shared_ref)
        new_f, new_s, new_idx = run(full_new, shared_new, ref_indices=ref_idx)
    finally:
        for m in (full_ref, full_new):
            m.release_sparse_mla_overlap_manager()

    ok = True
    for c in range(N_CHUNKS):
        a = ttnn.to_torch(ttnn.get_device_tensors(ref_idx[c])[0]).to(torch.int64)
        b = ttnn.to_torch(ttnn.get_device_tensors(new_idx[c])[0]).to(torch.int64)
        same_rows = (
            sum(
                len(set(a[0, 0, r].tolist()) & set(b[0, 0, r].tolist())) / max(1, a.shape[-1])
                for r in range(a.shape[2])
            )
            / a.shape[2]
        )
        log(f"chunk {c}: indexer top-k set overlap (dev0, mean over rows) = {same_rows:.5f}; shape {tuple(a.shape)}")
        for name, x, y in (("full", ref_f[c], new_f[c]), ("shared", ref_s[c], new_s[c])):
            passed, p = comp_pcc(x, y, 0.999)
            md = (x - y).abs().max().item()
            log(
                f"chunk {c} {name} layer: PCC seq-Q vs head-sharded = {p} max|diff| = {md:.4g} (ref max {x.abs().max().item():.4g})"
            )
            ok = ok and passed
    assert ok, "sequence-sharded Q diverged from the head-sharded path"
