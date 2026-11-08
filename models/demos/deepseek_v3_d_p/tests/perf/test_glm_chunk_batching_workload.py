# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Packed forwards vs one chunk per forward, on one multi-user workload: real tokens per second.

Packing F users' chunks into one forward runs attention per chunk and the MoE once on all F chunks' rows, so
the expert weights are read once per forward instead of once per chunk.

Workload: USERS users, each with history tokens of KV already prefilled (filled untimed, through the
same runtime), then one new request each (PRODUCTION_REQUEST_LENGTHS). The timed phase prefills every
new request, traced, in the configuration's forwards (cK-fF: K-token chunks, F chunks per forward).

  galaxy 8x4   c5120-f1 (today's production shape), c5120-f2 / f4 (same chunks, 2 or 4 per forward)

Model: 4 layers starting at global layer 6 (one full-indexer MoE layer, three that reuse its top-k,
GLM-5.3's 1:3 ratio among MoE layers), synthetic weights with the experts shared across layers (each layer
has its own gate). Record only, no gate: logs GLM_WORKLOAD lines.

Control: every config saves each user's KVPE and indexer-K rows (natural order: the last history chunk and
the new request) to GLM_WORKLOAD_KV_DIR, and test_glm_chunk_batching_workload_kv_agree (host only, run it
after the configs) requires the packed configs to match c5120-f1 closely.
"""

import copy
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.prefill_chunk import PrefillChunk
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural, unrotate_cache_layer
from tests.ttnn.utils_for_testing import comp_pcc

# Sweep knobs (GLM_WORKLOAD_*): the layer window, the history every user has already prefilled, and the new
# requests (comma-separated lengths, one user each; a request may span several chunks).
FIRST_LAYER = int(os.environ.get("GLM_WORKLOAD_FIRST_LAYER", "6"))
NUM_LAYERS = int(os.environ.get("GLM_WORKLOAD_NUM_LAYERS", "4"))
PRODUCTION_HISTORY = int(os.environ.get("GLM_WORKLOAD_HISTORY", 40 * 1024))
# Production-like new-request lengths (median ~1.9K, against 5K chunks), one per user.
PRODUCTION_REQUEST_LENGTHS = [
    int(n) for n in os.environ.get("GLM_WORKLOAD_REQUESTS", "1600,2200,896,2496,1792,2048,1280,2400").split(",")
]
USERS = len(PRODUCTION_REQUEST_LENGTHS)
REPEATS = int(os.environ.get("GLM_WORKLOAD_REPEATS", "3"))
KV_DIR = os.environ.get("GLM_WORKLOAD_KV_DIR", "generated/glm_workload_kv")
# Positions saved per user: the last 1,280 history tokens and the whole new request.
KV_TAIL = 1280
_TRACE_REGION_SIZE = 512 * 1024 * 1024


def _workload(scale):
    """(history, request lengths) at ``scale`` (1 = production; larger divides history and requests)."""
    return PRODUCTION_HISTORY // scale, [n // scale for n in PRODUCTION_REQUEST_LENGTHS]


_MESHES = {
    "galaxy-8x4": (
        (8, 4),
        torus_xy_device_params,
        "mesh-8x4",
        2,
        1,
        [(5120, 1), (5120, 2), (5120, 4)],
    ),
}


def _kv_path(mesh_shape, chunk, forward_chunks):
    return os.path.join(KV_DIR, f"{mesh_shape[0]}x{mesh_shape[1]}-c{chunk}-f{forward_chunks}.pt")


def _metadata_msg(mesh_device, words):
    return ttnn.from_torch(
        torch.tensor(list(words), dtype=torch.int64).reshape(1, 1, 1, len(words)),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _chunks(start, length, chunk, slot):
    """One request's chunks: [start, start + length) cut at chunk boundaries; the last may be partial."""
    out, pos = [], start
    while pos < start + length:
        end = min(pos + chunk, start + length)
        out.append(PrefillChunk(slot, pos, end))
        pos = pos + chunk
    return out


def _forwards(chunk_lists, forward_chunks):
    """Pack users' chunk lists, in user order, into forwards of forward_chunks chunks; empties at the tail.
    A user's chunks stay consecutive and in order, and a partial chunk is always its user's last."""
    flat = [c for cl in chunk_lists for c in cl]
    empty = PrefillChunk(0, 0, 0)
    out = []
    for i in range(0, len(flat), forward_chunks):
        group = flat[i : i + forward_chunks]
        out.append(group + [empty] * (forward_chunks - len(group)))
    return out


@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, scale, chunk, forward_chunks",
    [
        pytest.param(
            mesh_shape,
            make_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM53Adapter.l1_small_size,
                trace_region_size=_TRACE_REGION_SIZE,
            ),
            num_links,
            scale,
            chunk,
            forward_chunks,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=topology),
            id=f"{mesh_id}-c{chunk}-f{forward_chunks}",
        )
        for mesh_id, (mesh_shape, make_device_params, topology, num_links, scale, configs) in _MESHES.items()
        for chunk, forward_chunks in configs
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
def test_glm_chunk_batching_workload(
    variant, config_only, mesh_device, device_params, num_links, scale, chunk, forward_chunks
):
    history_len, request_lengths = _workload(scale)
    big_chunk = 5120 // scale  # the largest chunk any config of this scale uses
    assert history_len % big_chunk == 0, f"history {history_len} must be whole {big_chunk}-token chunks"
    assert history_len >= KV_TAIL, f"history {history_len} is shorter than the saved KV tail {KV_TAIL}"
    # Room for the longest request, in whole 2 x 5,120-token blocks.
    seq_cache = history_len + 2 * big_chunk * -(-max(request_lengths) // (2 * big_chunk))
    config = copy.deepcopy(config_only)
    config.max_seq_len = seq_cache
    rt_config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        first_layer_idx=FIRST_LAYER,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=chunk,
        forward_tokens_budget=forward_chunks * chunk,
        num_users=USERS,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        model_cfg=GLM53Config,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
        use_trace=True,
        tp_shard_kv=True,
    )
    mesh_device.enable_program_cache()
    state_dict = glm_random_transformer_state_dict(
        config, GLM53Config, NUM_LAYERS, first_layer_idx=FIRST_LAYER, share_experts=True
    )
    runtime = TtPrefillRuntime(mesh_device, config, state_dict, rt_config)
    del state_dict
    slots = rt_config.cache_num_users
    mesh_shape = list(mesh_device.shape)
    index_layers = full_indexer_rank(config, FIRST_LAYER + NUM_LAYERS) - full_indexer_rank(config, FIRST_LAYER)
    caches = MlaKvCaches(
        kvpe=init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=NUM_LAYERS,
            num_users=slots,
        ),
        index=init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=0,
            tp_axis=1,
            num_kvpe_cache_layers=index_layers,
            num_users=slots,
            dtype=ttnn.bfloat8_b,
        ),
    )
    g = torch.Generator().manual_seed(11)
    ids = [torch.randint(0, config.vocab_size, (seq_cache,), generator=g) for _ in range(USERS)]

    def run_forward(chunks):
        tokens = []
        for c in chunks:
            real = [] if c.is_empty else ids[c.slot_id][c.actual_start : c.actual_end].tolist()
            tokens.append(real + [0] * (chunk - len(real)))
        if forward_chunks == 1:
            c = chunks[0]
            msg = _metadata_msg(mesh_device, (c.slot_id, c.actual_start, c.actual_end))
            runtime.prefill_chunk(
                runtime.make_chunk_input(tokens[0]),
                caches,
                slot_id=c.slot_id,
                actual_start=c.actual_start,
                actual_end=c.actual_end,
                metadata_msg=msg,
            )
            ttnn.deallocate(msg)
        else:
            runtime.prefill_forward(
                runtime.make_forward_input(tokens, [c.actual_start for c in chunks]), caches, chunks
            )

    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)

        # History, untimed: every user's first history_len tokens, in the same forwards the config uses.
        history = _forwards([_chunks(0, history_len, chunk, u) for u in range(USERS)], forward_chunks)
        t0 = time.perf_counter()
        for chunks in history:
            run_forward(chunks)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"history: {len(history)} forwards in {time.perf_counter() - t0:.1f} s")

        # Timed: every user's new request on top of its history.
        requests = _forwards([_chunks(history_len, request_lengths[u], chunk, u) for u in range(USERS)], forward_chunks)
        times = []
        for _ in range(max(1, REPEATS)):
            t0 = time.perf_counter()
            for chunks in requests:
                run_forward(chunks)
            ttnn.synchronize_device(mesh_device)
            times.append(time.perf_counter() - t0)

        # Control: each user's KV in natural order, for the cross-config comparison.
        saved = {}
        for name, cache, layers in (("kvpe", caches.kvpe, NUM_LAYERS), ("indexer_k", caches.index, index_layers)):
            full, stripes = gather_cache_natural(cache, mesh_device, tp_shard_kv=True)  # [slots*layers, S, D]
            pos = blockcyclic_positions(stripes, chunk, seq_cache)
            saved[name] = [
                torch.stack(
                    [
                        unrotate_cache_layer(full[u * layers + layer], pos, history_len + request_lengths[u])[
                            history_len - KV_TAIL :
                        ]
                        for layer in range(layers)
                    ]
                ).to(torch.bfloat16)
                for u in range(USERS)
            ]
        os.makedirs(KV_DIR, exist_ok=True)
        torch.save(saved, _kv_path(mesh_shape, chunk, forward_chunks))
    finally:
        runtime.release_trace()

    real = sum(request_lengths)
    computed = len(requests) * forward_chunks * chunk
    median_s = statistics.median(times)
    logger.warning(
        f"GLM_WORKLOAD mesh={mesh_shape[0]}x{mesh_shape[1]} config=c{chunk}-f{forward_chunks} history={history_len} "
        f"layers={FIRST_LAYER}+{NUM_LAYERS} users={USERS} real_tokens={real} "
        f"forwards={len(requests)} computed_tokens={computed} padding={1 - real / computed:.1%} "
        f"time_ms={median_s * 1000:.1f} real_tok_per_s={real / median_s:.0f} "
        f"runs_ms={' '.join(f'{t * 1000:.1f}' for t in times)}"
    )


# Host only: compares what the configs above saved, each against its mesh's one-chunk-per-forward config.
# Packing changes only the MoE's batch (and with it which routed-expert kernel runs); on the LoudBox the KV
# drifted from layer 7 on, to 0.985 at layer 9, the same for every user and position, as flipped top-8
# choices under the synthetic router would.
_KV_PCC = 0.98
_KV_REFERENCE = {"galaxy-8x4": (5120, 1)}


@pytest.mark.parametrize(
    "mesh, other",
    [
        pytest.param(mesh_id, config, id=f"{mesh_id}-c{config[0]}-f{config[1]}")
        for mesh_id, (_, _, _, _, _, configs) in _MESHES.items()
        for config in configs
        if config != _KV_REFERENCE[mesh_id]
    ],
)
def test_glm_chunk_batching_workload_kv_agree(mesh, other):
    mesh_shape = _MESHES[mesh][0]
    reference = _KV_REFERENCE[mesh]
    paths = [_kv_path(mesh_shape, *config) for config in (reference, other)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"run test_glm_chunk_batching_workload first; missing {missing}")
    ref, got = (torch.load(p) for p in paths)
    failures = []
    for name in ("kvpe", "indexer_k"):
        worst = 1.0
        for u in range(len(ref[name])):
            for layer in range(ref[name][u].shape[0]):
                _, pcc = comp_pcc(ref[name][u][layer].float(), got[name][u][layer].float())
                worst = min(worst, float(pcc))
                if pcc < _KV_PCC:
                    failures.append(f"{name} user {u} layer {layer}: PCC {float(pcc):.6f} < {_KV_PCC}")
        logger.warning(
            f"GLM_WORKLOAD_KV mesh={mesh_shape[0]}x{mesh_shape[1]} c{other[0]}-f{other[1]} vs "
            f"c{reference[0]}-f{reference[1]} {name}: min PCC {worst:.6f}"
        )
    assert not failures, "\n".join(failures)
