# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH profiling harness for Qwen3.8-27B TP=2 prefill on a (1,2) Blackhole mesh (the p150x2 `tp2` profile).

TP=2 fork of test_prefill_profile_tp1_scratch.py (lane Q, opt_round4): mesh (1,2), FABRIC_1D, trace region 1 GiB
(as tp2_serve.sh). Everything below about the TP=1 path applies with TP=2 (no QWEN36_FORCE_TP_PATH needed).

(original TP=1 docstring:)

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
PROFILE_EAGER=1 (TP=2 fork): run the EAGER prompt prefill instead of the traced one -- prompts < one chunk take the
eager masked bucket (prefill_masked_bucket; set QWEN36_PREFILL_BUCKET_TRACE=0 so no bucket trace is replayed), longer
prompts the eager chunk loop _prefill_chunked_eager_tp (the forward the tp2-dflash2 spec prompt prefill runs, minus the
drafter taps); used to gate QWEN36_EAGER_FULL_CHUNK_UNMASKED.
"""

import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

_MESH_SHAPE = (1, 2)
# Same device params as the served node (--additional-config tt.l1_small_size / trace_region_size).
DEVICE_PARAMS = [
    {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1024 * 1024 * 1024}
]
BLOCK_SIZE = 64
CHUNK = 2048
_OUT_DIR = os.environ.get("PROFILE_OUT_DIR", "/home/ttuser/experiments/qwen36_27b/profiles/opt_round4")


def _blocks_for(isl):
    bucket = ((isl + CHUNK - 1) // CHUNK) * CHUNK
    blocks = max(64, (bucket + 256 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return ((blocks + 31) // 32) * 32


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "isl, layers, repeats",
    [
        pytest.param(2048, "0,1,2,3", 1, id="tp2_2048_L4"),  # one traced chunk, no tail
        pytest.param(2047, "0,1,2,3", 1, id="tp2_2047_L4"),  # traced 2048 masked bucket
        pytest.param(128, "0,1,2,3", 1, id="tp2_128_L4"),  # traced 128 masked bucket
        pytest.param(512, "0,1,2,3", 1, id="tp2_512_L4"),  # traced 512 masked bucket
        pytest.param(3072, "0,1,2,3", 1, id="tp2_3072_L4"),  # chunk + traced 1024 bucket tail
        pytest.param(8192, "0,1,2,3", 1, id="tp2_8192_L4"),
        pytest.param(16384, "0,1,2,3", 1, id="tp2_16384_L4"),
        pytest.param(32768, "0,1,2,3", 1, id="tp2_32768_L4"),
        pytest.param(8192, ",".join(str(i) for i in range(16)), 1, id="tp2_8192_L16"),  # lane L: layer-count scaling
        pytest.param(32768, ",".join(str(i) for i in range(16)), 1, id="tp2_32768_L16"),
        pytest.param(2048, "all", 1, id="tp2_2048_all"),
        pytest.param(8192, "all", 1, id="tp2_8192_all"),
        pytest.param(32768, "all", 1, id="tp2_32768_all"),
        pytest.param(2048, "3", 1, id="tp2_2048_A3"),  # single full-attention layer
        pytest.param(32768, "3", 1, id="tp2_32768_A3"),
    ],
)
def test_prefill_profile_tp2(mesh_device, isl, layers, repeats):
    device = mesh_device
    device.enable_program_cache()
    assert device.get_num_devices() == 2, "TP=2 harness: (1,2) mesh"
    layer_indices = None if layers == "all" else [int(x) for x in layers.split(",")]
    num_blocks = _blocks_for(isl)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len, layer_indices=layer_indices)
    assert model.use_tp, "expected the TP code path"
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

    isls = [isl]
    if os.environ.get("PROFILE_ISLS"):
        # Several prompt lengths in one process (wall TTFT / logits dumps); the model is sized for `isl` (the param).
        isls = [int(v) for v in os.environ["PROFILE_ISLS"].split(",")]
        assert max(isls) <= isl, "PROFILE_ISLS must not exceed the param isl (KV / max_seq_len sizing)"
    for cur in isls:
        _run_one(model, device, page_table, cur, repeats)


def _prompt_ids(model, isl):
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
    return token_ids


def _prefill(model, token_ids, page_table, isl):
    if os.environ.get("PROFILE_EAGER") != "1":
        return model.prefill_traced_chunked(token_ids, page_table, actual_len=isl)
    num_full, tail = isl // CHUNK, isl % CHUNK
    model._build_request_rope(token_ids[:, :isl], None)
    if num_full == 0:
        return model.prefill_masked_bucket(token_ids[:, :isl], page_table, actual_len=isl, chunk_start=0)
    return model._prefill_chunked_eager_tp(token_ids, page_table, isl, num_full, CHUNK, tail)


def _run_one(model, device, page_table, isl, repeats):
    token_ids = _prompt_ids(model, isl)

    # Untimed warm request (anything lazily allocated/compiled happens here, not in the profiled region).
    _ = _prefill(model, token_ids, page_table, isl)
    ttnn.synchronize_device(device)

    ttfts = []
    repeats = int(os.environ.get("PROFILE_REPEATS", repeats))
    # PROFILE_CHUNK_TIMING=1 (lane L): host timestamps of every execute_trace / synchronize_device inside the timed
    # requests (pair with QWEN36_PREFILL_OVERLAP=0 for a sync after every chunk replay = per-chunk wall).
    _ev = []
    _orig = (ttnn.execute_trace, ttnn.synchronize_device)
    if os.environ.get("PROFILE_CHUNK_TIMING") == "1":

        def _ex(*a, **k):
            _ev.append(("x", time.perf_counter()))
            return _orig[0](*a, **k)

        def _sy(*a, **k):
            r = _orig[1](*a, **k)
            _ev.append(("s", time.perf_counter()))
            return r

        ttnn.execute_trace, ttnn.synchronize_device = _ex, _sy
    signpost("start")
    for _ in range(repeats):
        t0 = time.time()
        _ev.append(("0", time.perf_counter()))
        logits = _prefill(model, token_ids, page_table, isl)
        ttnn.synchronize_device(device)
        ttfts.append(time.time() - t0)
    signpost("stop")
    ttnn.execute_trace, ttnn.synchronize_device = _orig
    if _ev:
        t_prev, line = None, []
        for tag, t in _ev:
            if tag == "0":
                if line:
                    print("PROFILE_CHUNKS " + " ".join(line))
                line, t_prev = [], t
                continue
            line.append(f"{tag}{(t - t_prev) * 1e3:.2f}")
            t_prev = t
        print("PROFILE_CHUNKS " + " ".join(line))
    n_layers = len(model.layers)
    lg_all = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    lg_all = lg_all.reshape(-1, model.args.vocab_size).float()
    lt = lg_all[0].clone()
    if not torch.equal(lg_all[0], lg_all[-1]):
        logger.warning("[PROFILE] device replicas of the logits differ")
    dump = os.environ.get("PROFILE_DUMP_LOGITS")
    if dump:
        torch.save(lt, f"{_OUT_DIR}/logits_{dump}_{isl}.pt")
        logger.info(f"[PROFILE] dumped logits to {_OUT_DIR}/logits_{dump}_{isl}.pt")
    top = torch.topk(lt, 2)
    logger.info(
        f"[PROFILE] isl={isl} layers={n_layers} TTFT(s)={['%.3f' % t for t in ttfts]} "
        f"-> per-layer {min(ttfts) / n_layers * 1000:.2f} ms, extrapolated 64-layer {min(ttfts) / n_layers * 64:.3f} s "
        f"argmax={int(top.indices[0])} top2gap={float(top.values[0] - top.values[1]):.4f}"
    )
    print(
        f"PROFILE_RESULT isl={isl} layers={n_layers} ttft_s={min(ttfts):.4f} "
        f"mean_s={sum(ttfts) / len(ttfts):.4f} "
        f"argmax={int(top.indices[0])} top2gap={float(top.values[0] - top.values[1]):.4f}"
    )
