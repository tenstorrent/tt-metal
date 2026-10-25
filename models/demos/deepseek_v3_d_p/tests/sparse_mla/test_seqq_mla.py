# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-sharded absorbed Q (ttMLA seq_sharded_q / MLA_SEQ_SHARDED_Q) against the head-sharded path and CPU truth.

GLM-5.3 on the production 8x4 torus-XY mesh at the production 5120-token chunk (640 rows per chip, 160 per TP
quarter), random weights, two chunks (the second ropes and attends at a non-zero start). Each layer is built once per
mode from the same weights:
  head    the production head-sharded path
  seq_ar  sequence-sharded Q but the head-sharded path's q_a stem (all-reduce + norm, then partition): isolates the
          q_a reduce-scatter summation order, everything after it should be bit-exact with head
  seq     the candidate (q_a reduce-scattered over rows)
Full layer 0 runs its own indexer; shared layer 3 gets the CPU full layer's top-k injected (identical index sets on
CPU and every device mode). With SEQQ_CPU_REF=<pt> (written by a pure-torch SparseMLAReference run on the same
seeds) every mode is also compared with CPU truth. Prints "SEQQ_MLA" lines.
"""

import copy
import os

import pytest
import torch

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
N_CHUNKS = 2
FULL_LAYER, SHARED_LAYER = 0, 3
SENTINEL = 0xFFFFFFFF
MODES = ("head", "seq_ar", "seq")


def log(*a):
    print("SEQQ_MLA", *a, flush=True)


def dram_alloc_mib(mesh):
    return ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 2**20


def pcc(x, y):
    return float(comp_pcc(x.float(), y.float(), 0.0)[1])


def device_indices(topk, start, mesh):
    """CPU top-k [1, CHUNK, k] (absolute positions, future entries where p+1 < k) -> the device's TP-sequence-sharded
    [1, 1, 160, k] uint32 shards (chip (s, t) holds chunk rows s*640 + t*160 ...), future -> sentinel, sorted."""
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    rows = CHUNK // (sp * tp)
    t = topk[0].to(torch.int64).clone()  # [CHUNK, k]
    pos = torch.arange(start, start + CHUNK, dtype=torch.int64).unsqueeze(1)
    t[t > pos] = SENTINEL
    t = t.sort(dim=-1).values
    return ttnn.from_torch(
        t.reshape(sp, tp, rows, t.shape[-1]),
        device=mesh,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
    )


def index_sets(idx_tensor, dev, start, row0):
    """Valid (causal, non-sentinel) index sets of one device's [1,1,rows,k] indices; rows start at position row0."""
    a = ttnn.to_torch(ttnn.get_device_tensors(idx_tensor)[dev]).to(torch.int64)[0, 0]
    return [set(x for x in a[r].tolist() if x != SENTINEL and x <= row0 + r) for r in range(a.shape[0])]


def jaccard(sa, sb):
    return sum(len(a & b) / max(1, len(a | b)) for a, b in zip(sa, sb)) / len(sa)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                # Production uses 1536 B; three modes' CCL programs in one process (every all-to-all program keeps its
                # carried counters in L1_SMALL) overflow it on best. A larger L1_SMALL only shrinks regular L1, so the
                # CB-clash check of MLA_SEQQ_L1 stays conservative.
                l1_small_size=8192,
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
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
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
    cpu_path = os.environ.get("SEQQ_CPU_REF")
    cpu = torch.load(cpu_path, weights_only=True) if cpu_path else None
    log(f"CPU reference: {cpu_path if cpu else 'none'}")

    def build(layer, mode):
        before = dram_alloc_mib(mesh)
        old = os.environ.get("MLA_SEQQ_QA")
        os.environ["MLA_SEQQ_QA"] = "allreduce" if mode == "seq_ar" else "rows"
        try:
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
                seq_sharded_q=mode != "head",
            )
        finally:
            if old is None:
                os.environ.pop("MLA_SEQQ_QA")
            else:
                os.environ["MLA_SEQQ_QA"] = old
        assert mla._seq_q == (mode != "head"), f"layer {layer} mode {mode}: seq_sharded_q did not resolve"
        log(f"built layer {layer} mode={mode}: DRAM +{dram_alloc_mib(mesh) - before:.1f} MiB/bank")
        return mla

    def caches():
        kv = [
            init_mla_kv_cache(
                cache_format=cache_format,
                hf_config=config,
                mesh_device=mesh,
                seq_len=seq_len,
                mesh_shape=list(mesh.shape),
                sp_axis=SP_AXIS,
                num_kvpe_cache_layers=1,
                tp_axis=TP_AXIS,
            )
            for _ in range(2)
        ]
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
        return kv[0], kv[1], index_kv

    def run(mode):
        full, shared = layers[mode]
        kvpe_full, kvpe_shared, index_kv = caches()
        res = {"full": [], "shared": [], "idx": []}
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
            inj = device_indices(cpu["topk"][c], start, mesh) if cpu else idx
            out_s = shared.forward(tt_hidden, rope_tensors, kvpe_shared, actual_start=start, indexer_indices=inj)
            ttnn.synchronize_device(mesh)
            res["full"].append(ttnn.to_torch(out_f, mesh_composer=composer).float())
            res["shared"].append(ttnn.to_torch(out_s, mesh_composer=composer).float())
            # dev 0 = (s 0, t 0) and dev 6 = (s 1, t 2): rows start at c*CHUNK + s*640 + t*160
            res["idx"].append(
                {
                    d: index_sets(idx, d, start, start + (d // tp) * (CHUNK // sp) + (d % tp) * (CHUNK // (sp * tp)))
                    for d in (0, 6)
                }
            )
            for t in (out_f, out_s, tt_hidden, idx) + ((inj,) if cpu else ()):
                ttnn.deallocate(t)
        return res

    # Build every instance first: the full layers share ONE set of TT_CCL sparse-MLA overlap resources (its global
    # semaphores live in the small L1_SMALL region, which does not survive repeated release/re-create).
    layers = {mode: (build(FULL_LAYER, mode), build(SHARED_LAYER, mode)) for mode in MODES}
    try:
        results = {mode: run(mode) for mode in MODES}
    finally:
        layers["head"][0].release_sparse_mla_overlap_manager()

    ok = True
    for c in range(N_CHUNKS):
        start = c * CHUNK
        for d in (0, 6):
            row0 = start + (d // tp) * (CHUNK // sp) + (d % tp) * (CHUNK // (sp * tp))
            msg = f"chunk {c} dev{d}: top-k Jaccard vs head: " + " ".join(
                f"{m}={jaccard(results[m]['idx'][c][d], results['head']['idx'][c][d]):.5f}" for m in MODES[1:]
            )
            if cpu:
                t = cpu["topk"][c][0, (row0 - start) : (row0 - start) + CHUNK // (sp * tp)].tolist()
                cpu_sets = [set(x for x in row if x <= row0 + r) for r, row in enumerate(t)]
                msg += " | vs CPU: " + " ".join(f"{m}={jaccard(results[m]['idx'][c][d], cpu_sets):.5f}" for m in MODES)
            log(msg)
        for layer in ("full", "shared"):
            ref = results["head"][layer][c]
            line = f"chunk {c} {layer:6s}: vs head " + " ".join(
                f"{m}={pcc(results[m][layer][c], ref):.6f}" for m in MODES[1:]
            )
            line += f" bitexact(seq_ar)={torch.equal(results['seq_ar'][layer][c], ref)}"
            if cpu:
                truth = cpu[f"{layer}_out"][c].float().reshape(ref.shape)
                vs = {m: pcc(results[m][layer][c], truth) for m in MODES}
                line += " | vs CPU " + " ".join(f"{m}={v:.6f}" for m, v in vs.items())
                # The candidate must be as accurate as production (same CPU bar the sparse MLA tests use, 0.98),
                # within run-to-run index-flip noise.
                # No accuracy loss vs production, within run-to-run index-flip noise. (No absolute bar: with random
                # weights production itself is 0.975 at chunk 1, below the 0.98 the small-shape sparse MLA tests use.)
                ok = ok and vs["seq"] >= vs["head"] - 0.002 and vs["seq_ar"] >= vs["head"] - 0.002
            log(line)
            if layer == "shared":
                ok = ok and pcc(results["seq_ar"][layer][c], ref) >= 0.99999
    assert ok, "sequence-sharded Q is less accurate than the head-sharded path (see SEQQ_MLA lines)"
