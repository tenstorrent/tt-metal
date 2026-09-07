# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (untracked) profiling harness for Qwen3.6-27B TP prefill on P150x4.

Runs the exact serving prefill path (traced chunk-outer replay + masked-bucket tail) between
tracy signposts "start"/"stop" so the device profiler CSV can be sliced to the prefill region.

    export HF_MODEL=/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B
    export MESH_DEVICE=P150x4 TT_CACHE_PATH=/home/ttuser/experiments/qwen36_27b/tt_cache
    python -m tracy -r -v -m pytest models/demos/blackhole/qwen36/tests/test_prefill_profile_scratch.py -k traced_4096_L4

layers="0,1,2,3" -> 3 GDN layers + 1 full-attention layer (real checkpoint layers, real types).
"""
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

_MESH_SHAPE = (1, 4)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 1024 * 1024 * 1024,
    }
]
BLOCK_SIZE = 64
CHUNK = 2048


def _blocks_for(isl):
    bucket = ((isl + CHUNK - 1) // CHUNK) * CHUNK
    blocks = max(64, (bucket + 256 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return ((blocks + 31) // 32) * 32


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "isl, layers, repeats",
    [
        pytest.param(4096, "0,1,2,3", 1, id="traced_4096_L4"),
        pytest.param(128, "0,1,2,3", 1, id="eager_128_L4"),
        pytest.param(3072, "0,1,2,3", 1, id="traced_3072_tail1024_L4"),
        pytest.param(4096, "all", 1, id="traced_4096_all"),
        pytest.param(4096, "0,1,2,3,4,5,6,7", 1, id="traced_4096_L8"),
        pytest.param(4096, "3", 1, id="traced_4096_A3"),  # single full-attention layer (determinism bisect)
        pytest.param(4096, "0,1,2", 1, id="traced_4096_G012"),  # GDN-only layers (determinism bisect)
        pytest.param(4096, "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", 1, id="traced_4096_L16"),
        pytest.param(128, "all", 1, id="eager_128_all"),
        pytest.param(16384, "all", 1, id="traced_16k_all"),
        pytest.param(32768, "all", 1, id="traced_32k_all"),
        pytest.param(16384, "0,1,2,3", 1, id="traced_16k_L4"),
        pytest.param(16384, "0,1,2,3,4,5,6,7", 1, id="traced_16k_L8"),
        pytest.param(32768, "0,1,2,3", 1, id="traced_32k_L4"),
        pytest.param(32768, "0,1,2,3,4,5,6,7", 1, id="traced_32k_L8"),
        pytest.param(65536, "0,1,2,3", 1, id="traced_64k_L4"),
        pytest.param(65536, "all", 1, id="traced_64k_all"),
    ],
)
def test_prefill_profile(mesh_device, isl, layers, repeats):
    device = mesh_device
    device.enable_program_cache()
    layer_indices = None if layers == "all" else [int(x) for x in layers.split(",")]
    num_blocks = _blocks_for(isl)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len, layer_indices=layer_indices)
    logger.info(f"[PROFILE] model load {time.time() - t0:.1f}s  layers={model.layer_indices}")

    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    t0 = time.time()
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
    logger.info(f"[PROFILE] trace capture + warmup {time.time() - t0:.1f}s")

    torch.manual_seed(0)
    token_ids = torch.randint(1000, 100000, (1, isl), dtype=torch.int64)
    if os.environ.get("PROFILE_REAL_PROMPT") == "1":
        # Real text (repeated 4k sample prompt) so PCC comparisons between configs see realistic activations.
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
    dump = os.environ.get("PROFILE_DUMP_LOGITS")
    if dump:
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        lt = lt.reshape(-1, model.args.vocab_size)[0].float().clone()
        torch.save(lt, f"/home/ttuser/experiments/qwen36_27b/profiles/logits_{dump}.pt")
        logger.info(f"[PROFILE] dumped logits to logits_{dump}.pt  argmax={int(lt.argmax())}")
    logger.info(
        f"[PROFILE] isl={isl} layers={n_layers} TTFT(s)={['%.3f' % t for t in ttfts]} "
        f"-> per-layer {min(ttfts) / n_layers * 1000:.2f} ms, extrapolated 64-layer {min(ttfts) / n_layers * 64:.3f} s"
    )
    print(f"PROFILE_RESULT isl={isl} layers={n_layers} ttft_s={min(ttfts):.4f}")
