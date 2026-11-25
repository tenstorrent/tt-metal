# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of Kimi-K3.1's production DSA layer 3 (indexer top-2048 + sparse MLA), teacher-fed from the
FP64 chain golden ``k31-l0-3-fp64-1m``, per context rung.

The production module (``ttMLA`` for layer 3 from the 4x4 stage's TTNN weight cache, with the adapter's scaled-FP8
kvpe and bf8 index-key caches, eager) prefills every 2560-row chunk of the prompt in order, reading the keys its
earlier chunks wrote. Each chunk's input is the golden's ``input_layernorm(dsa_input_3)`` (FP64, rounded once to
bf16), so only the layer's own error enters its caches and output. The golden is opened by id
(``k31_chain_golden``); a missing, incomplete or other-weights golden fails the test.

Scoring (``dsa_long_context.py``) runs chunk by chunk on the host. Per 16K window of rows, on 256 evenly spread rows
past top-k plus the golden's sampled rows: the device output's absolute and relative error and the selection-only
error against FP64 attention recomputed from the golden's keys and top-k. That recompute must match the golden's own
stored attention output on its sampled rows first. Every row: top-k recall, miss ranks, future keys. At the end the
caches against the golden's FP64 keys. Gated per window: ``sel_nrmse_median``, ``attn_nrmse``, ``attn_rms_err``
(``BARS``) and no future key; the rest is reported in the ``DSA_LONG_CONTEXT_ACCURACY`` line.

Needs the K3.1 checkpoint (``$KIMI_K31_HF_MODEL``, layer 3, checked against the golden's digest) and the 4x4 TTNN
weight cache (``$TT_KIMI_K31_PREFILL_TTNN_CACHE``); runs on one Blackhole Galaxy tray half (4x4, SP on axis 1).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import check
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_long_context as dsa
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.k31_chain_golden import PRODUCER, PRODUCER_VERSION, open_k31_chain
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import k31_topk_rows

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]

# Per-window maxima of the gated metrics, per rung (rows; each a multiple of the 2560-row chunk, so no rung ends in a
# zero-padded chunk). PROPOSED, pending the owner's decision (epic m06n): 1.25x the worst window measured on main's
# production DSA (this test, 4x4-sp1), rounded up at two significant figures. Measured worst windows:
#   5k:   sel_nrmse_median 0.00518, attn_nrmse 0.0280, attn_rms_err 5.68e-4 (one window, 49 scored rows)
#   128k: sel_nrmse_median 0.0299,  attn_nrmse 0.0476, attn_rms_err 9.69e-4 (rows [98304, 114688) / [114688, 128000))
#   1M:   sel_nrmse_median 0.0299,  attn_nrmse 0.0461, attn_rms_err 1.20e-3 (rows [98304, ...), [114688, ...),
#         [999424, ...)); the absolute error grows 1.8x over 1M (reference RMS 1.33x): DSA's own selection-driven
#         growth, accepted by these ceilings, not gated by a growth bar.
# Reruns are bit-identical on full windows; choosing other scored rows moved a window's metrics by 3-4%.
BARS: dict[int, dict[str, float]] = {
    5120: {"sel_nrmse_median": 0.0065, "attn_nrmse": 0.036, "attn_rms_err": 7.1e-4},
    128000: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.060, "attn_rms_err": 1.3e-3},
    1044480: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.058, "attn_rms_err": 1.6e-3},
}


def checkpoint() -> Path:
    adapter = k31_stage.ADAPTER
    return Path(os.environ.get(adapter.env_var) or adapter.hf_model_path)


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(1800)  # 1M: ~13 min measured (device ~3 min, host golden read and scoring ~10 min)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
@pytest.mark.parametrize("rows", list(BARS), ids=["5k", "128k", "1M"])
def test_k31_dsa_long_context(mesh_device, device_params, rows):
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    began = time.perf_counter()
    golden = open_k31_chain()
    w, digest = dsa.load_weights(checkpoint())
    golden.require(producer=PRODUCER, producer_version=PRODUCER_VERSION, weights_sha256=digest)
    scorer = dsa.RunScorer(golden, rows, w)

    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    max_seq_len = -(-rows // CACHE_LEN) * CACHE_LEN
    hf_config, params = k31_stage.window_params(
        mesh_shape, dsa.LAYER, 1, max_seq_len, is_first_rank=False, is_last_rank=True
    )
    adapter = k31_stage.ADAPTER
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    mla = ttMLA(
        hf_config,
        {},
        mesh_device,
        layer_idx=dsa.LAYER,
        seq_len=max_seq_len,
        sp_axis=layout.sp_axis,
        tp_axis=layout.tp_axis,
        topology=per_axis_topology(mesh_device=mesh_device),
        is_balanced=False,  # chunked prefill is block-cyclic, as the runtime builds it
        weight_cache_path=params.weight_cache_path,
        is_chunked=True,
        active_seq_len=chunk,
        slot_num=adapter.kv_cache_slots(params),
        layer_num=1,
        first_layer_idx=dsa.LAYER,
        tp_shard_kv=True,
        sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
    )
    shard_dims = [None, None]
    shard_dims[layout.sp_axis], shard_dims[layout.tp_axis] = -2, -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(shard_dims))

    seconds = {"setup": time.perf_counter() - began, "golden": 0.0, "device": 0.0, "score": 0.0}
    num_chunks = -(-rows // chunk)
    try:
        for c in range(num_chunks):
            lo, hi = c * chunk, min((c + 1) * chunk, rows)
            t = time.perf_counter()
            x = torch.zeros(chunk, hf_config.hidden_size, dtype=torch.bfloat16)
            x[: hi - lo] = scorer.chunk_input(lo, hi).to(torch.bfloat16)  # FP64 input_layernorm, one bf16 rounding
            seconds["golden"] += time.perf_counter() - t

            t = time.perf_counter()
            x_tt = ttnn.from_torch(
                x.reshape(1, 1, chunk, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            attn, indices = mla.forward(
                x_tt,
                {},  # K3.1's MLA is NoPE: no rotary tables
                caches.kvpe,
                index_kv_cache=caches.index,
                cache_layer_idx=0,
                actual_start=lo,
                actual_end=hi,
                cache_user_id=0,
                return_indexer_indices=True,
            )
            ttnn.deallocate(x_tt)
            topk = k31_topk_rows(mesh_device, indices, chunk, layout.sp_axis, layout.tp_axis)[: hi - lo]
            out = compose(mesh_device, attn, hi - lo)
            ttnn.deallocate(attn)
            ttnn.deallocate(indices)
            seconds["device"] += time.perf_counter() - t

            t = time.perf_counter()
            scorer.add(lo, out, dsa.device_selection(topk))
            seconds["score"] += time.perf_counter() - t
            if c % 20 == 0 or c == num_chunks - 1:
                done = {k: round(v, 1) for k, v in seconds.items()}
                logger.info(f"DSA_LC chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) seconds {json.dumps(done)}")
        t = time.perf_counter()
        kvpe = cache_rows(mesh_device, caches.kvpe, chunk, rows)[0]
        index = cache_rows(mesh_device, caches.index, chunk, rows)[0]
        seconds["device"] += time.perf_counter() - t
    finally:
        mla.release_sparse_mla_overlap_manager()
        ttnn.deallocate(caches.kvpe.storage)
        if caches.index is not None:
            ttnn.deallocate(caches.index)

    t = time.perf_counter()
    windows, summary = scorer.finish(kvpe, index)
    seconds["score"] += time.perf_counter() - t
    seconds["total"] = time.perf_counter() - began
    try:
        failures = check(windows, dsa.bars(BARS[rows]))
    except ValueError as error:  # a rung without bar values fails after reporting its measurements
        failures = [str(error)]
    result = {
        "golden_id": golden.golden_id,
        "rows": rows,
        "window": dsa.WINDOW,
        **summary,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "bars": BARS[rows],
        "bar_failures": failures,
        "windows": windows,
    }
    print("DSA_LONG_CONTEXT_ACCURACY=" + json.dumps(result, sort_keys=True))
    assert not failures, failures
