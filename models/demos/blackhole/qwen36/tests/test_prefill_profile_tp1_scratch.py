# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH profiling harness for Qwen3.8-27B TP=1 (single-die) prefill on one Blackhole P150 die.

TP=1 fork of test_prefill_profile_scratch.py (P150x4): mesh (1,1), the TP code path forced on one die
(QWEN36_FORCE_TP_PATH=1), bf8 paged KV (QWEN_SDPA_BF8=1), traced masked buckets
(QWEN36_PREFILL_BUCKET_TRACE=1) -- i.e. exactly the serving prefill path of the p1d1 prefill node.
Runs the traced chunk-outer replay (+ masked-bucket tail) between tracy signposts "start"/"stop".

    W=/home/ttuser/experiments/qwen36_27b/tt-metal-p1
    TT_METAL_HOME=$W PYTHONPATH=$W LD_LIBRARY_PATH=$W/build/lib MESH_DEVICE=P150 TT_VISIBLE_DEVICES=1 \
    TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=.../weights_qwen38 TT_CACHE_PATH=.../tt_cache_lane_a QWEN36_FORCE_TP_PATH=1 QWEN_SDPA_BF8=1 \
    QWEN36_PREFILL_BUCKET_TRACE=1 QWEN36_MTP=0 QWEN36_SKIP_VISION=1 TT_QWEN35_TEXT_VER=qwen36_blackhole \
    python -m tracy -r -v --op-support-count 20000 -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_profile_tp1_scratch.py -k tp1_2048_L4

layers="0,1,2,3" -> 3 GDN layers + 1 full-attention layer (real checkpoint layers, real types).
Env knobs: PROFILE_REPEATS (default 1), PROFILE_DUMP_LOGITS=<tag> (save the last logits row to
profiles/p1d1_opt/logits_<tag>.pt for before/after numerics checks), PROFILE_REAL_PROMPT=1.
"""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

_MESH_SHAPE = (1, 1)
# Same device params as the served node (--additional-config tt.l1_small_size / trace_region_size).
DEVICE_PARAMS = [{"l1_small_size": 24576, "trace_region_size": 512 * 1024 * 1024}]
BLOCK_SIZE = 64
CHUNK = 2048
_OUT_DIR = "/home/ttuser/experiments/qwen36_27b/profiles/p1d1_opt"


def _blocks_for(isl):
    bucket = ((isl + CHUNK - 1) // CHUNK) * CHUNK
    blocks = max(64, (bucket + 256 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return ((blocks + 31) // 32) * 32


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "isl, layers, repeats",
    [
        pytest.param(2048, "0,1,2,3", 1, id="tp1_2048_L4"),  # one traced chunk, no tail
        pytest.param(2047, "0,1,2,3", 1, id="tp1_2047_L4"),  # traced 2048 masked bucket
        pytest.param(128, "0,1,2,3", 1, id="tp1_128_L4"),  # traced 128 masked bucket
        pytest.param(3072, "0,1,2,3", 1, id="tp1_3072_L4"),  # chunk + traced 1024 bucket tail
        pytest.param(8192, "0,1,2,3", 1, id="tp1_8192_L4"),
        pytest.param(16384, "0,1,2,3", 1, id="tp1_16384_L4"),
        pytest.param(32768, "0,1,2,3", 1, id="tp1_32768_L4"),
        pytest.param(2048, "all", 1, id="tp1_2048_all"),
        pytest.param(4096, "all", 1, id="tp1_4096_all"),
        pytest.param(8192, "all", 1, id="tp1_8192_all"),
        pytest.param(32768, "all", 1, id="tp1_32768_all"),
        pytest.param(2048, "3", 1, id="tp1_2048_A3"),  # single full-attention layer
        pytest.param(2048, "0,1,2", 1, id="tp1_2048_G012"),  # GDN-only layers
        pytest.param(8192, "3", 1, id="tp1_8192_A3"),
        pytest.param(32768, "3", 1, id="tp1_32768_A3"),
    ],
)
def test_prefill_profile_tp1(mesh_device, isl, layers, repeats):
    device = mesh_device
    device.enable_program_cache()
    assert device.get_num_devices() == 1, "TP=1 harness: one die"
    layer_indices = None if layers == "all" else [int(x) for x in layers.split(",")]
    num_blocks = _blocks_for(isl)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len, layer_indices=layer_indices)
    assert model.use_tp, "expected the TP code path on one die (QWEN36_FORCE_TP_PATH=1)"
    logger.info(
        f"[PROFILE] model load {time.time() - t0:.1f}s  layers={model.layer_indices} "
        f"grid={device.compute_with_storage_grid_size()} tuning={model.args.prefill_tuning}"
    )

    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat8_b, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    t0 = time.time()
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
    logger.info(f"[PROFILE] trace capture + warmup {time.time() - t0:.1f}s  bucket traces={sorted(model._mb_traces)}")

    torch.manual_seed(0)
    token_ids = torch.randint(1000, 100000, (1, isl), dtype=torch.int64)
    if os.environ.get("PROFILE_REAL_PROMPT") == "1":
        import json

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
        with open("models/demos/blackhole/qwen36/demo/sample_prompts/input_data_long_4k.json") as f:
            text = json.load(f)[0]["prompt"]
        ids = tok(text, return_tensors="pt")["input_ids"]
        while ids.shape[1] < isl:
            ids = torch.cat([ids, ids], dim=1)
        token_ids = ids[:, :isl].to(torch.int64)

    # Untimed warm request (anything lazily allocated/compiled happens here, not in the profiled region).
    _ = model.prefill_traced_chunked(token_ids, page_table, actual_len=isl)
    ttnn.synchronize_device(device)

    ttfts = []
    repeats = int(os.environ.get("PROFILE_REPEATS", repeats))
    signpost("start")
    for _ in range(repeats):
        t0 = time.time()
        logits = model.prefill_traced_chunked(token_ids, page_table, actual_len=isl)
        ttnn.synchronize_device(device)
        ttfts.append(time.time() - t0)
    signpost("stop")
    n_layers = len(model.layers)
    lt = ttnn.to_torch(logits).reshape(-1, model.args.vocab_size)[0].float().clone()
    dump = os.environ.get("PROFILE_DUMP_LOGITS")
    if dump:
        torch.save(lt, f"{_OUT_DIR}/logits_{dump}.pt")
        logger.info(f"[PROFILE] dumped logits to {_OUT_DIR}/logits_{dump}.pt")
    top = torch.topk(lt, 2)
    logger.info(
        f"[PROFILE] isl={isl} layers={n_layers} TTFT(s)={['%.3f' % t for t in ttfts]} "
        f"-> per-layer {min(ttfts) / n_layers * 1000:.2f} ms, extrapolated 64-layer {min(ttfts) / n_layers * 64:.3f} s "
        f"argmax={int(top.indices[0])} top2gap={float(top.values[0] - top.values[1]):.4f}"
    )
    print(
        f"PROFILE_RESULT isl={isl} layers={n_layers} ttft_s={min(ttfts):.4f} "
        f"argmax={int(top.indices[0])} top2gap={float(top.values[0] - top.values[1]):.4f}"
    )
