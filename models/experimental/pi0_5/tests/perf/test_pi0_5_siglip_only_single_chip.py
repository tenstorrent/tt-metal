# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 SigLIP-only performance — single chip, no cross-chip D2D.

Isolates the SigLIP vision tower + multimodal projector on ONE Blackhole chip.
Inputs are uploaded once OUTSIDE the timed region; only
`PrefixEmbeddingTTNN.embed_images` (the SigLIP encoder + projector) is the
measured stage. Single device => NO SendDirectAsync / RecvDirectAsync (D2D);
this is the pure on-chip SigLIP compute baseline for the team to optimize.

Mirrors test_pi0_5_denoise_only_single_chip.py: a default TRACED wall-clock
path plus an EAGER=1 Tracy-profiling path with PHASE_siglip/PHASE_end
signposts (Tracy cannot attribute device ops inside a traced replay).

Run (wall-clock):
  PI05_CHECKPOINT_DIR=/local/.../pi05_base \\
    python_env/bin/python -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py

Run (Tracy device-op profile, ISL 32):
  EAGER=1 PI05_LANG_SEQ_LEN=32 PI05_CHECKPOINT_DIR=/local/.../pi05_base \\
    python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \\
    -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py
  # then: tt-perf-report <csv> --start-signpost PHASE_siglip --end-signpost PHASE_end

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
# 0 (default) = run the real full SigLIP path (256 patches/image). >0 = feed an
# N-token synthetic hidden sequence straight through the 27 encoder blocks,
# modeling the per-stage workload of a PP chunk (e.g. 768 patches / 24 = 32).
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
    return images_ttnn, img_masks_ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_siglip_only_single_chip(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    # GUARD: single chip so the op graph emits no cross-chip D2D. The `device`
    # fixture hands back a single-device handle; assert its span is exactly 1.
    n_chips = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    assert n_chips == 1, (
        f"siglip-only single-chip test requires a 1-chip device handle, got {n_chips}. "
        f"A multi-chip mesh would introduce D2D and invalidate the no-D2D baseline."
    )
    print(f"\n🔒 single-chip guard OK: device spans {n_chips} chip")

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    model = Pi0_5ModelTTNN(cfg, loader, device)

    # CHUNK MODE: feed an N-token synthetic hidden sequence through the 27
    # SigLIP encoder blocks directly (the per-PP-chunk transformer workload).
    # Skips patch_embed/pos_emb/projector since those are fixed to 256 patches
    # and are not part of the repeated per-chunk encoder cost.
    if CHUNK_TOKENS > 0:
        tower = model.backbone.vision_tower
        assert tower is not None, "vision_tower is None (PI0_SIGLIP_HF host path?); chunk mode needs the TTNN tower"
        hidden_size = cfg.siglip_config.hidden_size
        torch.manual_seed(SEED)
        chunk_host = torch.randn(1, CHUNK_TOKENS, hidden_size, dtype=torch.float32) * 0.02
        chunk = ttnn.from_torch(
            chunk_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"\n📦 CHUNK mode: {CHUNK_TOKENS}-token synthetic hidden through 27 SigLIP encoder blocks")

        def _siglip():
            h = chunk
            for block in tower.blocks:
                h = block.forward(h)
            return h

    else:
        imgs, im_masks = _build_inputs(device, NUM_CAMERAS)
        print(f"\n📦 inputs uploaded once; SigLIP encoder + projector is the measured region")

        def _siglip():
            # embed_images = SigLIP vision tower (27 blocks) + multimodal projector.
            # Returns (list[image_embs], list[img_pad_masks]).
            return model.prefix_embedding.embed_images(imgs, im_masks)

    # ---- EAGER mode (for Tracy device-op profiling) ----
    if os.environ.get("EAGER", "").lower() in ("1", "true", "yes", "on"):
        from tracy import signpost

        print("\n🔬 EAGER device-profile mode: 1 warmup + 1 signposted SigLIP-only iter")
        out = _siglip()
        ttnn.synchronize_device(device)

        signpost("PHASE_siglip")
        t0 = time.perf_counter()
        out = _siglip()
        ttnn.synchronize_device(device)
        signpost("PHASE_end")
        eager_ms = (time.perf_counter() - t0) * 1000.0
        embs = out[0] if isinstance(out, tuple) else out
        first = embs[0] if isinstance(embs, (list, tuple)) else embs
        host = ttnn.to_torch(first).float()
        assert torch.isfinite(host).all(), "eager SigLIP produced NaN/Inf"

        print("\n" + "=" * 72)
        print(f"  PI0.5 SIGLIP-ONLY EAGER (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
        print("=" * 72)
        print(f"   Config:            chunk_tokens={CHUNK_TOKENS or 'full'}, cameras={NUM_CAMERAS}")
        print(f"   Eager signposted:  {eager_ms:7.2f} ms")
        print(f"   Signposts:         PHASE_siglip → PHASE_end")
        print("=" * 72)
        return

    # ---- Eager reference (non-traced) ----
    ref_out = _siglip()
    ttnn.synchronize_device(device)
    ref_embs = ref_out[0] if isinstance(ref_out, tuple) else ref_out
    ref_first = ttnn.to_torch(ref_embs[0] if isinstance(ref_embs, (list, tuple)) else ref_embs).float()
    assert torch.isfinite(ref_first).all(), "eager SigLIP produced NaN/Inf"

    # ---- Warmup the trace path (JIT) ----
    for _ in range(NUM_WARMUP):
        _ = _siglip()
        ttnn.synchronize_device(device)

    # ---- Capture SigLIP-only trace ----
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    _ = _siglip()
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
    print(f"  PI0.5 SIGLIP-ONLY (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
    print("=" * 72)
    print(f"   Config:            chunk_tokens={CHUNK_TOKENS or 'full'}, cameras={NUM_CAMERAS}")
    print(f"   Trace capture:     {capture_ms:7.2f} ms (one-time)")
    print(f"   SigLIP-only avg:   {avg:7.2f} ms")
    print(f"   Per-call min/max:  {mn:7.2f} / {mx:7.2f} ms   stddev {sd:.2f}")
    print("=" * 72)

    assert avg > 0
