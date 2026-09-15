# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-profiler session for JUST the "first part" of a DFlash session: the
target model's REAL prefill (with dFlash tap capture armed, matching the real
demo's setup) -- no drafter, no verify, no steady-state decode loop.

Sibling of ``test_dflash_drafter_tracy.py`` (drafter-only) and
``test_dflash_verify_tracy.py`` (verify-only): together the three cover
DFlash's three model-invocation phases (prefill once, then draft/verify
repeated). See ``demo/dflash_fused_decoder_demo.py`` for the full traced
session these three decompose.

**Do NOT use ``python -m tracy`` for a full-checkpoint capture by default.**
Host Tracy has a hard 32K distinct-source-location cap; loading all 60 target
layers overflows it. Device-only profiling avoids that (same recipe as
``test_dflash_drafter_tracy.py``):

    rm -rf generated/profiler/.logs generated/profiler/reports
    export HF_MODEL=google/gemma-4-31B-it
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

    pytest models/demos/gemma4/tests/dflash/test_dflash_prefill_tracy.py \\
        -k 1x8 -s --timeout 1800

    python tools/tracy/process_ops_logs.py --date --device-only

``prefill_start``/``prefill_stop`` signposts mark the measured window -- names
distinct from ``test_dflash_drafter_tracy.py``'s ``start``/``stop`` and
``test_decode_forward_tracy.py``'s ``decode_forward_start``/``_stop`` so a
combined capture (if ever run in one Tracy session) can't collide.

Or: models/demos/gemma4/scripts/run_dflash_prefill_profile.sh

Env knobs:
    GEMMA4_DFLASH_PREFILL_TRACY_NUM_LAYERS   [8]     truncate the target (Tracy's
                                                       32K-source-location cap);
                                                       op mix / timing only, NOT a
                                                       correctness profile at this
                                                       count.
    GEMMA4_DFLASH_PREFILL_TRACY_ISL           [4096]  real prompt length to prefill
                                                       (a Gutenberg-style long
                                                       context is synthesized to
                                                       reach it).
    GEMMA4_DFLASH_PREFILL_TRACY_ITERS         [1]     measured prefill calls inside
                                                       start/stop. >1 simply repeats
                                                       prefill_forward_text on the
                                                       same kv_cache state -- only
                                                       ITERS=1 has been validated;
                                                       treat >1 as exploratory.
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch
from loguru import logger

from ...tests.test_factory import parametrize_mesh_with_fabric

try:
    from tracy import signpost

    _HAS_SIGNPOST = True
except ModuleNotFoundError:
    _HAS_SIGNPOST = False

TRACY_NUM_LAYERS = int(os.environ.get("GEMMA4_DFLASH_PREFILL_TRACY_NUM_LAYERS", 8))
ISL = int(os.environ.get("GEMMA4_DFLASH_PREFILL_TRACY_ISL", 4096))
ITERS = int(os.environ.get("GEMMA4_DFLASH_PREFILL_TRACY_ITERS", 1))
# MUST be a multiple of the target's max_prefill_chunk_size (2048 on WH T3K):
# the eager per-chunk prefill loop rounds its LAST chunk up to a full chunk
# width regardless of how much real content remains, and that rounded-up
# position can exceed a non-chunk-aligned MAX_SEQ_LEN -- hits a real TT_FATAL
# this way (RoPE table sliced past the width it was built to). Same
# constraint dflash_fused_decoder_demo.py documents for GEMMA4_DFLASH_MAX_SEQ_LEN.
_PREFILL_CHUNK = 2048
MAX_SEQ_LEN = max(((ISL + 256 + _PREFILL_CHUNK - 1) // _PREFILL_CHUNK) * _PREFILL_CHUNK, 4096)
PAGE_BLOCK_SIZE = 64


def _flush_profiler(mesh_device):
    import ttnn

    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)


def _synthetic_long_prompt(tokenizer, isl):
    """Build a real (tokenizer-verified) prompt of ~isl tokens without a network
    fetch -- repeats a short paragraph until encoding reaches the target length,
    then trims. Only the SHAPE (real ISL, real tokens) matters for a device op
    mix / timing profile, not the content."""
    paragraph = (
        "The history of computing is a long chain of small, deliberate ideas "
        "compounding into machines that reshape how people work and think. "
    )
    text = paragraph
    while len(tokenizer.encode(text)) < isl + 64:
        text += paragraph
    ids = tokenizer.encode(text)[:isl]
    return tokenizer.decode(ids)


@parametrize_mesh_with_fabric([(1, 8)], device_params_extra={"trace_region_size": 256_000_000})
def test_dflash_prefill_tracy(mesh_device, device_params, reset_seeds):
    """Measured window = repeated real target prefill_forward_text only (signposted)."""
    import ttnn
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        logger.warning("[dflash-prefill-tracy] TT_METAL_DEVICE_PROFILER is not 1 -- device op CSV will be empty.")
    if not _HAS_SIGNPOST:
        logger.warning("tracy.signpost not importable -- CSV won't have prefill_start/prefill_stop markers")

    # Tap-arming needs real target_layer_ids -- reuse DFlashDrafter's own
    # config-driven resolution (mirrors test_dflash_fused_decoder_tracy.py) so
    # the prefill's tap-hook overhead matches a real DFlash session, even
    # though this test never uses the taps for anything past capture.
    snap = os.environ.get("GEMMA4_DFLASH_DRAFTER")
    if not snap:
        import glob

        hits = glob.glob(
            os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/*/")
        )
        snap = hits[0] if hits else None
    if not snap:
        pytest.skip("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")

    logger.warning(
        f"[dflash-prefill-tracy] truncating to {TRACY_NUM_LAYERS} layers (Tracy's 32K-source-location "
        "cap); this profile is for device op mix / timing only, not correctness."
    )
    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=TRACY_NUM_LAYERS,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    model0 = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)

    # target_layer_ids: same remap-to-truncated-range logic as
    # test_dflash_fused_decoder_tracy.py (WHICH layers get tapped doesn't
    # matter for a timing profile, only the tap COUNT, to keep fc's input
    # width consistent with the real checkpoint's tap count).
    weights_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if not weights_dir:
        import glob

        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*--{model_path.split('/')[-1]}/snapshots/*/")
        )
        weights_dir = hits[0].rstrip("/") if hits else None

    def _embed_loader():
        import json

        from safetensors import safe_open

        idx = json.load(open(f"{weights_dir}/model.safetensors.index.json"))
        key = next(
            k
            for k in idx["weight_map"]
            if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
        )
        with safe_open(f"{weights_dir}/{idx['weight_map'][key]}", framework="pt") as f:
            return f.get_tensor(key)

    drafter = DFlashDrafter(
        mesh_device=mesh_device,
        drafter_path=snap,
        target_embed_weight_loader=_embed_loader,
        mesh_config=model0.mesh_config,
        ccl_manager=model0.ccl_manager,
        tensor_cache_path=None,
    )
    orig_taps = list(drafter.target_layer_ids)
    n = len(orig_taps)
    new_taps = sorted(set(round(i * (TRACY_NUM_LAYERS - 1) / (n - 1)) for i in range(n))) if n > 1 else [0]
    while len(new_taps) < n:
        for cand in range(TRACY_NUM_LAYERS):
            if cand not in new_taps:
                new_taps.append(cand)
                break
        new_taps.sort()
    target_layer_ids = new_taps[:n]

    prompt = _synthetic_long_prompt(tokenizer, ISL)
    in_pt, encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=MAX_SEQ_LEN
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    real_isl = int(decoding_pos[0])

    _flush_profiler(mesh_device)
    logger.info(
        f"[dflash-prefill-tracy] setup: layers={TRACY_NUM_LAYERS} target_layer_ids={target_layer_ids} "
        f"real_isl={real_isl} (requested {ISL}) iters={ITERS}"
    )

    def _run_once():
        model0.dflash_capture_taps(target_layer_ids, keep_last=12)
        try:
            generator.prefill_forward_text(
                in_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                enable_trace=False,
                warmup_prefill=False,
            )
        finally:
            model0.pop_dflash_taps()
            model0.dflash_capture_taps(None)

    if _HAS_SIGNPOST:
        signpost("prefill_start")
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _run_once()
    ttnn.synchronize_device(mesh_device)
    wall = time.perf_counter() - t0
    if _HAS_SIGNPOST:
        signpost("prefill_stop")
    _flush_profiler(mesh_device)

    per_call_s = wall / ITERS if ITERS else float("nan")
    tps = real_isl / per_call_s if per_call_s else float("nan")
    logger.info(
        f"[dflash-prefill-tracy] measured region: {ITERS} prefill call(s) of {real_isl} tokens in "
        f"{wall:.3f}s -> {per_call_s:.3f}s/call ({tps:.1f} tok/s prefill throughput)"
    )
    logger.info(
        "[dflash-prefill-tracy] next: `python tools/tracy/process_ops_logs.py --date --device-only`, "
        'then filter ops_perf_results_*.csv to rows between signposts "prefill_start" and "prefill_stop".'
    )
