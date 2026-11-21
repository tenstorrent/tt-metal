# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A Kimi-K3.1 DSA layer teacher-fed on one 4x4 stage (SP on axis 1), chunk by chunk: layer 92 from a vLLM trace, or
layer 3 from the FP64 chain (``dsa_teacher_feed.py input --chain``); the layer is the input's.

The device half of ``dsa_teacher_feed.py`` (read its docstring for the three steps; its CPU half is tested in
``test_teacher_feed_contract.py``). The production DSA module (``ttMLA`` for the input's layer from the stage's TTNN
weight cache, scaled-FP8 kvpe and bf8 index-key caches allocated by the adapter as the runner does) runs every
2560-token chunk of the prefix in order, eager, reading the KV and index keys its earlier chunks wrote. Its input is
the trace-rebuilt ``input_layernorm`` output, so each chunk is teacher-fed and only the attention module's own error
accumulates in its caches. Each chunk's attention output and top-k indices, then both caches, are saved to
``$TT_DSA_TF_OUT``; nothing is scored while the device is held (``dsa_teacher_feed.py score`` does it). Skips
unless ``$TT_DSA_TF_INPUT`` and ``$TT_DSA_TF_OUT`` are set.

    TT_DSA_TF_INPUT=/mnt/.../k31-l92-input-128k-v2 TT_DSA_TF_OUT=/mnt/.../k31-l92-dev-128k \\
    models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_dsa_teacher_feed.py::test_k31_dsa_layer_teacher_fed[blackhole-4x4-sp1]"
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
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, provenance
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path, finish_run, load_input, start_run
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import k31_topk_rows

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]
SOURCE, OUT = os.environ.get("TT_DSA_TF_INPUT"), os.environ.get("TT_DSA_TF_OUT")


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.skipif(
    not (SOURCE and OUT),
    reason="set TT_DSA_TF_INPUT (a dsa_teacher_feed input) and TT_DSA_TF_OUT (the run's output dir)",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
def test_k31_dsa_layer_teacher_fed(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    source, out = Path(SOURCE), Path(OUT)
    built = json.loads((source / MANIFEST).read_text())
    layer = built["layer"]
    assert layer + 1 in KimiK31Config.DSA_LAYERS_1BASED, f"layer {layer} is not a DSA layer"
    count = int(os.environ.get("TT_DSA_TF_COUNT", built["segments"][-1]["end"]))
    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    max_seq_len = -(-count // CACHE_LEN) * CACHE_LEN
    hf_config, params = k31_stage.window_params(
        mesh_shape, layer, 1, max_seq_len, is_first_rank=False, is_last_rank=True
    )
    adapter = k31_stage.ADAPTER
    manifest = {
        "layer": layer,
        "input": str(source),
        "input_manifest": {k: built[k] for k in ("trace", "trace_commit", "input", "checkpoint")},
        "count": count,
        "chunk": chunk,
        "mesh": list(mesh_shape),
        "sp_axis": layout.sp_axis,
        "max_seq_len": max_seq_len,
        "weight_cache": str(params.weight_cache_path),
        "kvpe_format": str(adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format)),
        **provenance(),
    }
    start_run(out, manifest)

    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    mla = ttMLA(
        hf_config,
        {},
        mesh_device,
        layer_idx=layer,
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
        first_layer_idx=layer,
        tp_shard_kv=True,
        sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
    )
    dims = [None, None]
    dims[layout.sp_axis], dims[layout.tp_axis] = -2, -1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=tuple(dims))
    num_chunks = -(-count // chunk)
    began = time.perf_counter()
    try:
        for c in range(num_chunks):
            lo, hi = c * chunk, min((c + 1) * chunk, count)
            rows = torch.zeros(chunk, hf_config.hidden_size, dtype=torch.bfloat16)
            rows[: hi - lo] = load_input(source, lo, hi)
            normed = ttnn.from_torch(
                rows.reshape(1, 1, chunk, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            attn, indices = mla.forward(
                normed,
                {},  # K3.1's MLA is NoPE: no rotary tables
                caches.kvpe,
                index_kv_cache=caches.index,
                cache_layer_idx=0,
                actual_start=lo,
                actual_end=hi,
                cache_user_id=0,
                return_indexer_indices=True,
            )
            ttnn.deallocate(normed)
            topk = k31_topk_rows(mesh_device, indices, chunk, layout.sp_axis, layout.tp_axis)[: hi - lo]
            torch.save(
                {
                    "start": lo,
                    "end": hi,
                    "attn_out": compose(mesh_device, attn, hi - lo).to(torch.bfloat16),
                    "topk": torch.where(topk >= 2**31, -1, topk).to(torch.int32),
                },
                chunk_path(out, c),
            )
            ttnn.deallocate(attn)
            ttnn.deallocate(indices)
            logger.info(
                f"DSA_TF chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) done, {time.perf_counter() - began:.1f} s"
            )
        kvpe = cache_rows(mesh_device, caches.kvpe, chunk, count)[0]
        index = cache_rows(mesh_device, caches.index, chunk, count)[0]
        torch.save({"kvpe": kvpe.to(torch.bfloat16), "index": index.to(torch.bfloat16)}, out / "caches.pt")
    finally:
        mla.release_sparse_mla_overlap_manager()
        ttnn.deallocate(caches.kvpe.storage)
        if caches.index is not None:
            ttnn.deallocate(caches.index)
    for name, values in (("kvpe", kvpe), ("index", index)):
        assert torch.isfinite(values).all(), f"non-finite rows in the {name} cache"
    seconds = time.perf_counter() - began
    finish_run(out, manifest, seconds)
    logger.info(f"DSA_TF done: {num_chunks} chunks, {count} rows, {seconds:.1f} s -> {out}")
