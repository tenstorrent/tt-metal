# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-profiler session for JUST the "part after drafter" of a DFlash
session: the target model's packed-verify forward (``ttnn_packed_verify_forward``,
tt/model.py) -- the step that checks the drafter's candidate block against the
real 31B target, right after the drafter proposes it.

Sibling of ``test_dflash_drafter_tracy.py`` (drafter-only) and
``test_dflash_prefill_tracy.py`` (prefill-only): together the three cover
DFlash's three model-invocation phases (prefill once, then draft/verify
repeated in the steady-state loop). See ``demo/dflash_fused_decoder_demo.py``
for the full traced session these three decompose.

Unlike the drafter/decode_forward siblings (synthetic weights, no real
checkpoint), packed-verify's inputs are TIGHTLY internally coupled to
``DFlashFusedDecoder``'s own persistent device buffers (built by its
``_pv_setup``/``_pv_upload``, tt/dflash_drafter.py). Rather than hand-build
synthetic packed-verify tensors (real risk of a subtly wrong shape/mask giving
a misleading profile), this test runs the REAL prefill + drafter + capture
setup (identical to test_dflash_fused_decoder_tracy.py) and then reuses
``dec``'s own already-correct buffers, calling ``ttnn_packed_verify_forward``
directly in an EAGER (non fused-trace) loop -- the exact same code the fused
``_body()`` runs for its verify sub-step, just isolated and repeated outside
the trace so Tracy/the device profiler can attribute time to it alone.

The candidate token ids fed to verify are a shape-only placeholder (the
anchor token repeated across the block) -- this measures verify's REAL device
op cost at DFlash's REAL packed shape, but the logits/output are NOT
meaningful text. For end-to-end correctness+numbers, use
``demo/dflash_fused_decoder_demo.py`` instead.

    rm -rf generated/profiler/.logs generated/profiler/reports
    export HF_MODEL=google/gemma-4-31B-it
    export GEMMA4_DFLASH_SHARD_ARGMAX=1
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000

    pytest models/demos/gemma4/tests/dflash/test_dflash_verify_tracy.py \\
        -k 1x8 -s --timeout 1800

    python tools/tracy/process_ops_logs.py --date --device-only

``verify_start``/``verify_stop`` signposts mark the measured window.

Or: models/demos/gemma4/scripts/run_dflash_verify_profile.sh

Env knobs:
    GEMMA4_DFLASH_VERIFY_TRACY_NUM_LAYERS   [8]   truncate the target (Tracy's
                                                    32K-source-location cap).
    GEMMA4_DFLASH_VERIFY_TRACY_ITERS        [8]   measured verify calls inside
                                                    start/stop.
    GEMMA4_DFLASH_VERIFY_TRACY_WARMUP       [1]   compile warmups OUTSIDE
                                                    start/stop.
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

PROMPT = (
    "Write a Python function that checks whether a given string is a "
    "palindrome, ignoring spaces, punctuation and case. Include a short "
    "docstring and two example calls."
)
MAX_SEQ_LEN = 2048
PAGE_BLOCK_SIZE = 64
TRACY_NUM_LAYERS = int(os.environ.get("GEMMA4_DFLASH_VERIFY_TRACY_NUM_LAYERS", 8))
ITERS = int(os.environ.get("GEMMA4_DFLASH_VERIFY_TRACY_ITERS", 8))
WARMUP = int(os.environ.get("GEMMA4_DFLASH_VERIFY_TRACY_WARMUP", 1))


def _unwrap_kv_layers(kv_cache):
    kv_layers = kv_cache
    if (
        isinstance(kv_layers, (list, tuple))
        and kv_layers
        and isinstance(kv_layers[0], (list, tuple))
        and kv_layers[0]
        and isinstance(kv_layers[0][0], (list, tuple))
    ):
        kv_layers = kv_layers[0]
    return kv_layers


def _flush_profiler(mesh_device):
    import ttnn

    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)


def _verify_once(dec, vxp, ttnn):
    """Exact replica of DFlashFusedDecoder._body()'s packed-verify sub-step
    (dflash_drafter.py, the ``if self.use_packed:`` branch around the
    ``ttnn_packed_verify_forward`` call) -- reusing dec's own real, persistent
    buffers rather than hand-built ones. See that method for the authoritative
    version; keep this in sync if it changes."""
    H_l = dec.target.layers[0].self_attn.config.num_attention_heads // dec.drafter.tp
    pv_pos = ttnn.slice(dec.blk_pos, [0, 0], [1, dec.P_v]) if dec.P_v < dec.K + 1 else dec.blk_pos
    widx = [ttnn.slice(dec.pv_widx_all, [p], [p + 1]) for p in range(dec.P_v)]
    posf = ttnn.typecast(ttnn.to_layout(ttnn.reshape(pv_pos, (1, 1, dec.P_v, 1)), ttnn.TILE_LAYOUT), ttnn.float32)
    diff = ttnn.sub(dec.pv_iota, posf)
    gt = ttnn.gtz(diff)
    mf_p = ttnn.typecast(ttnn.multiply(gt, -1e9), ttnn.bfloat16)
    m_full = ttnn.repeat(mf_p, ttnn.Shape([1, 1, H_l, 1]))
    if dec.pv_slide_ring:
        m_slide = ttnn.repeat(dec.pv_mask_slide, ttnn.Shape([1, 1, H_l, 1]))
    else:
        W_t = float(dec.target.hf_config.sliding_window)
        far = ttnn.gez(ttnn.multiply(ttnn.add(diff, W_t), -1.0))
        ind = ttnn.add(gt, far)
        ms_p = ttnn.typecast(ttnn.multiply(ind, -1e9), ttnn.bfloat16)
        m_slide = ttnn.repeat(ms_p, ttnn.Shape([1, 1, H_l, 1]))
    logits, hidden = dec.target.ttnn_packed_verify_forward(
        x=vxp,
        position_idx=pv_pos,
        attn_mask_full=m_full,
        attn_mask_sliding=m_slide,
        packed_p=dec.P_v,
        page_table=dec.v_pt,
        kv_cache=dec.kv_layers,
        kv_write_idxs=widx,
        page_tables_per_layer=dec.pv_tables,
    )
    m_full.deallocate(True)
    m_slide.deallocate(True)
    logits.deallocate(True)
    hidden.deallocate(True)


@parametrize_mesh_with_fabric([(1, 8)], device_params_extra={"trace_region_size": 256_000_000})
def test_dflash_verify_tracy(mesh_device, device_params, reset_seeds):
    """Measured window = repeated packed-verify forward only (signposted)."""
    import ttnn
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter, DFlashFusedDecoder
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    weights_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if not weights_dir:
        import glob

        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*--{model_path.split('/')[-1]}/snapshots/*/")
        )
        weights_dir = hits[0].rstrip("/") if hits else None
    if not weights_dir or not os.path.isdir(weights_dir):
        pytest.skip(f"MODEL_WEIGHTS_DIR not found/set (tried {weights_dir!r}); set it explicitly")
    snap = os.environ.get("GEMMA4_DFLASH_DRAFTER")
    if not snap:
        import glob

        hits = glob.glob(
            os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/*/")
        )
        snap = hits[0] if hits else None
    if not snap:
        pytest.skip("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")
    if os.environ.get("GEMMA4_DFLASH_SHARD_ARGMAX") != "1":
        pytest.skip("set GEMMA4_DFLASH_SHARD_ARGMAX=1 -- required, see dflash_fused_decoder_demo.py's docstring")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        logger.warning("[dflash-verify-tracy] TT_METAL_DEVICE_PROFILER is not 1 -- device op CSV will be empty.")
    if not _HAS_SIGNPOST:
        logger.warning("tracy.signpost not importable -- CSV won't have verify_start/verify_stop markers")

    logger.warning(
        f"[dflash-verify-tracy] truncating to {TRACY_NUM_LAYERS} layers (Tracy's 32K-source-location "
        "cap); candidate token ids are a shape-only placeholder -- op mix / timing only, NOT correctness."
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
    drafter.target_layer_ids = new_taps[:n]

    in_pt, encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        [PROMPT], tokenizer, generator.model_args, True, 32 + 32, max_prefill_len=MAX_SEQ_LEN
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    n_tok = int(decoding_pos[0])
    anchor_token = int(encoded[0][n_tok - 1])

    model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
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
        taps = model0.pop_dflash_taps()
        model0.dflash_capture_taps(None)

    kv_layers = _unwrap_kv_layers(tt_kv_cache)
    dec = DFlashFusedDecoder(model0, drafter, kv_layers, page_table[:1])
    dec.prefill_ingest(taps, n_tok)
    # capture() runs _pv_setup/_pv_upload (builds the real persistent verify
    # buffers this test reuses) plus one real eager verify call as part of its
    # own compile pass -- exactly the setup this isolated loop needs.
    dec.capture(anchor_token, n_tok, max_new=32)

    # Shape-only candidate ids: anchor token repeated across the packed-verify
    # width. Real DFlash uses concat(anchor_tok, draft_ids) here -- this test
    # skips running the drafter per-iteration since only verify's device time
    # is being measured; token VALUES don't affect op timing, only SHAPES do.
    vxp_torch = torch.full((1, dec.P_v), anchor_token, dtype=torch.int64)
    vxp = ttnn.from_torch(vxp_torch, device=mesh_device, dtype=ttnn.uint32, mesh_mapper=dec._mapper)

    _flush_profiler(mesh_device)
    logger.info(
        f"[dflash-verify-tracy] setup: layers={TRACY_NUM_LAYERS} P_v={dec.P_v} K={dec.K} "
        f"warmup={WARMUP} iters={ITERS}"
    )

    for _ in range(WARMUP):
        _verify_once(dec, vxp, ttnn)
    _flush_profiler(mesh_device)

    if _HAS_SIGNPOST:
        signpost("verify_start")
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _verify_once(dec, vxp, ttnn)
    ttnn.synchronize_device(mesh_device)
    wall = time.perf_counter() - t0
    if _HAS_SIGNPOST:
        signpost("verify_stop")
    _flush_profiler(mesh_device)

    per_call_ms = (wall * 1000.0 / ITERS) if ITERS else float("nan")
    logger.info(
        f"[dflash-verify-tracy] measured region: {ITERS} packed-verify calls in {wall:.3f}s "
        f"-> {per_call_ms:.2f} ms/call"
    )
    logger.info(
        "[dflash-verify-tracy] next: `python tools/tracy/process_ops_logs.py --date --device-only`, "
        'then filter ops_perf_results_*.csv to rows between signposts "verify_start" and "verify_stop".'
    )
