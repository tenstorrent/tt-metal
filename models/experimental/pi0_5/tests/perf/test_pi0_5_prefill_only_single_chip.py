# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 VLM-prefill-only performance — single chip, no cross-chip D2D.

Isolates the VLM (Gemma-2B) prefill on ONE Blackhole chip: the 18-block
backbone forward over the prefix that produces the prefix KV cache. The prefix
embeddings (SigLIP + language embed + concat) are built ONCE outside the timed
region; only `backbone.forward_vlm(prefix_embs, use_cache=True)` is the measured
stage. Single device => NO SendDirectAsync / RecvDirectAsync (D2D); this is the
pure on-chip VLM-prefill compute baseline.

Mirrors test_pi0_5_denoise_only_single_chip.py: a default TRACED wall-clock
path plus an EAGER=1 Tracy-profiling path with PHASE_prefill/PHASE_end
signposts (Tracy cannot attribute device ops inside a traced replay).

Prefill cost scales with prefix length = num_patches*cameras + lang_seq_len.
Use PI05_LANG_SEQ_LEN to vary the language input (e.g. 32 for short-prompt).

Run (wall-clock):
  PI05_CHECKPOINT_DIR=/local/.../pi05_base \\
    python_env/bin/python -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_prefill_only_single_chip.py

Run (Tracy device-op profile, ISL 32):
  EAGER=1 PI05_LANG_SEQ_LEN=32 PI05_CHECKPOINT_DIR=/local/.../pi05_base \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \\
    -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_prefill_only_single_chip.py
  # then: tt-perf-report <csv> --start-signpost PHASE_prefill --end-signpost PHASE_end

Skipped if the checkpoint isn't present locally.
"""

import os
import re
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

NUM_WARMUP = int(os.environ.get("PI05_TRACE_NUM_WARMUP", "2"))
NUM_ITERS = int(os.environ.get("PI05_TRACE_NUM_ITERS", "20"))
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
LANG_SEQ_LEN = int(os.environ.get("PI05_LANG_SEQ_LEN", "256"))  # input language seq len (try 32 for short-prompt)
# PI05_CHUNK_TOKENS: replicate one pipeline-parallel CHUNK on a single chip.
# 0 (default) = run the real full prefix (768 image + lang). >0 = feed an
# N-token synthetic hidden sequence straight through the 18 VLM blocks,
# modeling the per-stage workload of a PP chunk (e.g. 768 / 24 = 32).
# Capped at 768 (the full image-token budget a chunk can be sliced from).
CHUNK_TOKENS = int(os.environ.get("PI05_CHUNK_TOKENS", "0"))
assert CHUNK_TOKENS <= 768, f"PI05_CHUNK_TOKENS={CHUNK_TOKENS} exceeds the 768 image-token budget"
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _apply_production_env_defaults() -> None:
    root = os.environ.get("TT_METAL_HOME") or str(Path(__file__).resolve().parents[5])
    envf = Path(root) / "_bench_runs" / "pi05_production.env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
        if not m or m.group(1) == "PI05_CHECKPOINT_DIR":
            continue
        os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()


def _build_inputs(device, num_cameras: int):
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, 224, 224, dtype=torch.float32) for _ in range(num_cameras)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    use_fold = os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
    if use_fold:
        _PATCH = 14
        stacked = torch.cat([im.permute(0, 2, 3, 1).contiguous() for im in images], dim=0)
        n, h, w, c = stacked.shape
        stacked = stacked.reshape(n, h, w // _PATCH, c * _PATCH).contiguous()
        images_ttnn = [
            ttnn.from_torch(
                stacked,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]
    else:
        images_ttnn = [
            ttnn.from_torch(
                im,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for im in images
        ]
    img_masks_ttnn = [
        ttnn.from_torch(
            m.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for m in img_masks
    ]
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return images_ttnn, img_masks_ttnn, lang_tokens_ttnn, lang_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_prefill_only_single_chip(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN, use_upstream_masks

    # GUARD: single chip so the op graph emits no cross-chip D2D. The `device`
    # fixture hands back a single-device handle; assert its span is exactly 1.
    n_chips = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    assert n_chips == 1, (
        f"prefill-only single-chip test requires a 1-chip device handle, got {n_chips}. "
        f"A multi-chip mesh would introduce D2D and invalidate the no-D2D baseline."
    )
    print(f"\n🔒 single-chip guard OK: device spans {n_chips} chip")

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    model = Pi0_5ModelTTNN(cfg, loader, device)

    # CHUNK MODE: feed an N-token synthetic hidden sequence through the 18 VLM
    # blocks directly (the per-PP-chunk transformer workload). Skips SigLIP /
    # lang-embed / concat since those are not part of the repeated per-chunk
    # backbone cost. RoPE is built internally by each block (overrides=None).
    if CHUNK_TOKENS > 0:
        vlm_hidden = cfg.vlm_config.width
        torch.manual_seed(SEED)
        chunk_host = torch.randn(1, CHUNK_TOKENS, vlm_hidden, dtype=torch.float32) * 0.02
        prefix_embs = ttnn.from_torch(
            chunk_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn = cos = sin = None
        ttnn.synchronize_device(device)
        print(f"\n📦 CHUNK mode: {CHUNK_TOKENS}-token synthetic hidden through 18 VLM blocks")
    else:
        imgs, im_masks, lt, lm = _build_inputs(device, NUM_CAMERAS)

        if use_upstream_masks():
            prefix_len = cfg.siglip_config.num_patches * len(im_masks) + LANG_SEQ_LEN
            model.prepare_upstream_artifacts(im_masks, lm, prefix_len=prefix_len)

        # ---- Build prefix embeddings ONCE (SigLIP + lang embed + concat),
        #      OUTSIDE the timed region. Only the VLM backbone is measured.
        prefix_embs, _, _ = model.embed_prefix(imgs, im_masks, lt, lm)
        if prefix_embs.layout != ttnn.TILE_LAYOUT:
            prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)

        upstream = getattr(model, "_cached_upstream_artifacts", None) if use_upstream_masks() else None
        attn = upstream["prefix_attn_mask"] if upstream else None
        cos = upstream["prefix_cos"] if upstream else None
        sin = upstream["prefix_sin"] if upstream else None
        ttnn.synchronize_device(device)
        print(f"\n📦 prefix embeds built once (len={int(prefix_embs.shape[1])}); VLM prefill is the measured region")

    def _prefill():
        # forward_vlm = 18-block Gemma-2B backbone over the prefix, producing
        # the prefix KV cache (use_cache=True). This is the VLM prefill stage.
        _out, _kv = model.backbone.forward_vlm(
            prefix_embs,
            attention_mask=attn,
            cos_override=cos,
            sin_override=sin,
            use_cache=True,
        )
        return _out

    # ---- EAGER mode (for Tracy device-op profiling) ----
    if os.environ.get("EAGER", "").lower() in ("1", "true", "yes", "on"):
        from tracy import signpost

        print("\n🔬 EAGER device-profile mode: 1 warmup + 1 signposted prefill-only iter")
        out = _prefill()
        ttnn.synchronize_device(device)

        signpost("PHASE_prefill")
        t0 = time.perf_counter()
        out = _prefill()
        ttnn.synchronize_device(device)
        signpost("PHASE_end")
        eager_ms = (time.perf_counter() - t0) * 1000.0
        host = ttnn.to_torch(out).float()
        assert torch.isfinite(host).all(), "eager VLM prefill produced NaN/Inf"

        print("\n" + "=" * 72)
        print(f"  PI0.5 VLM-PREFILL-ONLY EAGER (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
        print("=" * 72)
        print(f"   Config:            chunk_tokens={CHUNK_TOKENS or 'full'}, prefix_len={int(prefix_embs.shape[1])}")
        print(f"   Eager signposted:  {eager_ms:7.2f} ms")
        print(f"   Signposts:         PHASE_prefill → PHASE_end")
        print("=" * 72)
        return

    # ---- Eager reference (non-traced) ----
    ref = _prefill()
    ttnn.synchronize_device(device)
    ref_host = ttnn.to_torch(ref).float()
    assert torch.isfinite(ref_host).all(), "eager VLM prefill produced NaN/Inf"

    # ---- Warmup the trace path (JIT) ----
    for _ in range(NUM_WARMUP):
        _ = _prefill()
        ttnn.synchronize_device(device)

    # ---- Capture prefill-only trace ----
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    _ = _prefill()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    # ---- Time steady-state replay ----
    times_ms: List[float] = []
    for _ in range(NUM_ITERS):
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    ttnn.release_trace(device, tid)

    avg = statistics.mean(times_ms)
    mn, mx = min(times_ms), max(times_ms)
    sd = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print("\n" + "=" * 72)
    print(f"  PI0.5 VLM-PREFILL-ONLY (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
    print("=" * 72)
    print(f"   Config:            chunk_tokens={CHUNK_TOKENS or 'full'}, prefix_len={int(prefix_embs.shape[1])}")
    print(f"   Trace capture:     {capture_ms:7.2f} ms (one-time)")
    print(f"   Prefill-only avg:  {avg:7.2f} ms")
    print(f"   Per-call min/max:  {mn:7.2f} / {mx:7.2f} ms   stddev {sd:.2f}")
    print("=" * 72)

    assert avg > 0
