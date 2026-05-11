# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run video tests back-to-back on the TTNN Molmo2 model.

Warm-up (JIT compile + decode-trace capture) happens automatically on the
first generate() call.  All subsequent tests reuse the captured decode trace.

Usage:
    cd /home/ttuser/ssinghal/PR-fix/molmo2/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    # Run all 105 tests:
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_10_videos.py -v -s
    # Run first N tests:
    MESH_DEVICE=T3K pytest models/demos/molmo2/tests/test_10_videos.py -v -s --max_tests 10

Environment:
    MESH_DEVICE       T3K (default), N150, N300
    MOLMO2_VERIF_DIR  directory containing test.jsonl, test_results.jsonl, videos/
                      default: /home/ttuser/ssinghal/PR-fix/tt-metal/models/demos/molmo2/verification
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HF_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    )
)

VERIF_DIR = Path(
    os.environ.get(
        "MOLMO2_VERIF_DIR",
        "/home/ttuser/ssinghal/PR-fix/tt-metal/models/demos/molmo2/verification",
    )
)
VIDEO_DIR = VERIF_DIR / "videos"
TEST_JSONL = VERIF_DIR / "test.jsonl"
RESULTS_JSONL = VERIF_DIR / "test_results.jsonl"
WEIGHT_CACHE = Path(os.environ.get("MOLMO2_WEIGHT_CACHE", f"/tmp/molmo2_weight_cache_u{os.getuid()}"))

_MESH_SHAPE = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), (1, 8))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def url_to_local(url: str) -> Path:
    return VIDEO_DIR / url.split("/")[-1]


def extract_video_info(test: dict):
    """Return (prompt_text, video_url, max_tokens) from a test entry."""
    content = test["messages"][0]["content"]
    prompt, video_url = "", None
    for item in content:
        if item.get("type") == "text":
            prompt = item["text"]
        elif item.get("type") == "video_url":
            video_url = item["video_url"]["url"]
    return prompt, video_url, test.get("max_tokens", 16)


def build_video_inputs(processor, video_path: Path, prompt: str):
    """Run HF processor and return (input_ids, pixel_values, pool_idx, token_type_ids).

    Follows the demo.py pattern exactly: apply_chat_template(tokenize=False) to
    get raw text, then processor(text=..., videos=path, ...) to get tensors.
    """
    # Single conversation, wrapped in a list for apply_chat_template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        }
    ]
    # tokenize=False → returns text string (not token IDs)
    formatted = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
    if isinstance(formatted, list):
        formatted = formatted[0]

    inputs = processor(text=formatted, videos=str(video_path), return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"]  # [1, S]
    token_type_ids = inputs.get("token_type_ids")

    pv = inputs.get("pixel_values_videos")
    pool_idx = inputs.get("video_token_pooling")

    if pv is not None:
        # pv: [n_frames, n_patches, px_dim] → [1, n_frames, n_patches, px_dim]
        pv = pv.float().unsqueeze(0)
        pool_idx = pool_idx.unsqueeze(0)

    return input_ids, pv, pool_idx, token_type_ids


# ---------------------------------------------------------------------------
# Mesh fixture (module-scoped — one device for all 10 tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH_SHAPE
    is_single = rows * cols == 1
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    logger.info(f"Opened {rows}×{cols} mesh: {device.get_num_devices()} devices")
    yield device
    ttnn.close_mesh_device(device)
    if not is_single:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def tt_model(mesh_device):
    """Load TTNN Molmo2 model (once per module)."""
    from transformers import AutoModelForImageTextToText

    from models.demos.molmo2.tt.model import TtMolmo2Model
    from models.demos.molmo2.tt.model_config import Molmo2Config
    from models.tt_transformers.tt.ccl import TT_CCL

    sys.path.insert(0, str(HF_PATH))
    logger.info("Loading HF weights (bfloat16)...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        str(HF_PATH), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    state_dict = hf_model.state_dict()
    del hf_model

    cfg = Molmo2Config(mesh_device=mesh_device)
    ccl = TT_CCL(mesh_device)
    WEIGHT_CACHE.mkdir(parents=True, exist_ok=True)
    model = TtMolmo2Model(
        mesh_device=mesh_device,
        tt_ccl=ccl,
        state_dict=state_dict,
        weight_cache_path=WEIGHT_CACHE,
        dtype=ttnn.bfloat16,
        configuration=cfg,
    )
    del state_dict
    from models.demos.molmo2.tt.model import PREFILL_BUCKETS

    # Step 1: JIT-compile all text decoder buckets (128..32768) without trace
    logger.info(f"[warmup] JIT compile text decoder buckets {PREFILL_BUCKETS}...")
    model.warmup_all_buckets(bucket_sizes=PREFILL_BUCKETS, use_trace=False)

    # Step 2: Capture prefill traces for buckets ≤ 4096 (larger buckets OOM trace DRAM)
    logger.info("[warmup] Capturing prefill traces (buckets ≤ 4096)...")
    model.warmup_all_buckets(use_trace=True)

    # Step 3: JIT-compile vision ops (ViT + pooling + projector)
    logger.info("[warmup] vision (ViT + pooling + projector)...")
    model.warmup_vision_compile()

    # Step 4: Capture decode trace so first inference has no trace-capture stall
    logger.info("[warmup] Capturing decode trace...")
    model.warmup_decode_trace()

    logger.info("Model fully warmed up (JIT + traces + vision + decode trace)")
    return model, cfg


@pytest.fixture(scope="module")
def processor():
    sys.path.insert(0, str(HF_PATH))
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(HF_PATH), trust_remote_code=True)


# ---------------------------------------------------------------------------
# Data fixture — collect the first 10 tests that have local video files
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--max_tests", default=None, type=int, help="Max number of tests to run (default: all)")


@pytest.fixture(scope="module")
def video_tests(request):
    try:
        _MAX_TESTS = request.config.getoption("--max_tests") or 10**6
    except ValueError:
        _MAX_TESTS = int(os.environ.get("MAX_TESTS", 10**6))
    all_tests = load_jsonl(TEST_JSONL)
    results_map = {r["test_index"]: r for r in load_jsonl(RESULTS_JSONL)}

    collected = []
    for idx, test in enumerate(all_tests):
        _, video_url, max_tokens = extract_video_info(test)
        if video_url is None:
            continue
        vpath = url_to_local(video_url)
        if not vpath.exists():
            logger.warning(f"  test {idx}: video file missing — {vpath.name}")
            continue
        ref = results_map.get(idx, {}).get("response", "")
        collected.append(
            {
                "idx": idx,
                "test": test,
                "vpath": vpath,
                "max_tokens": max_tokens,
                "ref_response": (ref or "").strip(),
            }
        )
        if len(collected) >= _MAX_TESTS:
            break

    assert collected, f"No video tests found. Check VIDEO_DIR={VIDEO_DIR}"
    logger.info(f"Loaded {len(collected)} video tests (indices {collected[0]['idx']}–{collected[-1]['idx']})")
    return collected


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(0)  # disable global 300s pytest-timeout; this test runs 105 videos
def test_10_videos_back_to_back(mesh_device, tt_model, processor, video_tests):
    """Run video tests back-to-back with a single warm-up and one decode trace."""
    model, cfg = tt_model
    eos = processor.tokenizer.eos_token_id

    pass_count = fail_count = 0
    results = []
    results_path = Path(os.environ.get("MOLMO2_RESULTS_PATH", f"/tmp/molmo2_ttnn_results_u{os.getuid()}.jsonl"))
    if results_path.exists() and os.access(results_path, os.W_OK):
        results_path.unlink(missing_ok=True)  # fresh file for this run

    logger.info(f"\n{'='*70}")
    logger.info(f"Running {len(video_tests)} video tests back-to-back")
    logger.info(f"  Decode trace: captured once after first prefill, reused for all")
    logger.info(f"  KV cache: reset before each test")
    logger.info(f"  Results → {results_path}")
    logger.info(f"{'='*70}")

    for rank, entry in enumerate(video_tests):
        idx = entry["idx"]
        prompt, video_url, max_tokens = extract_video_info(entry["test"])
        vpath = entry["vpath"]
        ref = entry["ref_response"]

        logger.info(f"\n[{rank+1}/{len(video_tests)}] test_index={idx}  {vpath.name}")
        logger.info(f"  prompt:   {prompt[:80]!r}")
        logger.info(f"  expected: {ref!r}")

        # ---- Preprocess ----
        t0 = time.time()
        try:
            input_ids, pv, pool_idx, token_type_ids = build_video_inputs(processor, vpath, prompt)
        except Exception as e:
            # decord / ffmpeg errors on corrupt/unsupported video files
            t_prep = time.time() - t0
            logger.error(f"  PREPROCESS ERROR: {e}  ({t_prep:.1f}s) — skipping")
            fail_count += 1
            results.append(
                {
                    "test_index": idx,
                    "expected": ref,
                    "response": f"PREPROCESS_ERROR: {e}",
                    "status": "ERROR",
                    "seq_len": -1,
                    "prep_s": round(t_prep, 3),
                    "gen_s": None,
                    "total_s": round(time.time() - t0, 3),
                }
            )
            with open(results_path, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")
            continue
        S = input_ids.shape[1]
        t_prep = time.time() - t0
        logger.info(f"  preprocess: {t_prep:.1f}s  seq_len={S}")

        # ---- TTNN generate ----
        # First test: triggers JIT compilation for prefill + captures decode trace.
        # Subsequent tests: reuse cached JIT kernels and the same decode trace.
        t_gen_start = time.time()
        t_gen = t_prefill = t_decode = None
        try:
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pv,
                pooled_patches_idx=pool_idx,
                token_type_ids=token_type_ids,
                max_new_tokens=max_tokens,
                eos_token_id=eos,
                temperature=0.0,
                user_id=0,
            )
            t_gen = time.time() - t_gen_start
            n_new = len(generated_ids)

            response = processor.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
            ok = response == ref
            status = "PASS" if ok else "FAIL"
            if ok:
                pass_count += 1
            else:
                fail_count += 1

            tps = n_new / t_gen if t_gen > 0 else 0
            logger.info(
                f"  response: {response!r}  [{status}]  " f"generate={t_gen:.2f}s  tokens={n_new}  {tps:.1f} tok/s"
            )

        except Exception as e:
            t_gen = time.time() - t_gen_start
            response = f"ERROR: {e}"
            status = "ERROR"
            fail_count += 1
            logger.error(f"  EXCEPTION: {e}  ({t_gen:.2f}s)")

        record = {
            "test_index": idx,
            "expected": ref,
            "response": response,
            "status": status,
            "seq_len": S,
            "prep_s": round(t_prep, 3),
            "gen_s": round(t_gen, 3) if t_gen is not None else None,
            "total_s": round(time.time() - t0, 3),
        }
        results.append(record)
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---- Summary ----
    n = len(video_tests)
    gen_times = [r["gen_s"] for r in results if r["gen_s"] is not None and r["status"] != "ERROR"]
    avg_gen = sum(gen_times) / len(gen_times) if gen_times else 0

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: {pass_count}/{n} PASS,  {fail_count}/{n} FAIL")
    logger.info(f"Avg generate time (PASS tests): {avg_gen:.2f}s")
    logger.info(f"{'='*70}")
    for r in results:
        sym = "✓" if r["status"] == "PASS" else "✗"
        logger.info(
            f"  [{sym}] idx={r['test_index']:3d}  S={r['seq_len']:5d}  "
            f"prep={r['prep_s']:5.1f}s  gen={r['gen_s'] or 'ERR':>6}s  "
            f"got={r['response'][:20]!r}  exp={r['expected']!r}"
        )
    logger.info(f"\nFull results written to {results_path}")

    assert fail_count == 0, f"{fail_count}/{n} tests failed — see log above"
