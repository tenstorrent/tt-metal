# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PI0.5 denoise-only performance — single chip, no cross-chip D2D.

Isolates the flow-matching Euler denoise loop on ONE Blackhole chip. The
prefix (SigLIP + VLM prefill) is built once OUTSIDE the timed region; only
`Pi0_5ModelTTNN._run_denoise_loop` is captured as a TTNN trace and replayed.
Because everything runs on a single device, there are NO SendDirectAsync /
RecvDirectAsync (D2D) ops in this measurement — it is the pure on-chip
denoise compute baseline.

Contrast with the GLX socket-trace denoise-only experiment
(tt/tt_bh_glx/socket_trace_experiment/run_denoise_only.py) which spreads the
6 expert chips across the Galaxy mesh and therefore DOES include D2D in the
loop. Comparing the two isolates the D2D contribution (~0.54 ms device time,
~8% of denoise device kernel time per the Tracy breakdown).

What it reports:
  - denoise-only steady-state replay latency (ms/loop) for N steps
  - PCC of the traced denoise output vs the eager (non-traced) denoise output
    on identical noise — guards correctness of the trace path

Run:
  PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \\
    python_env/bin/python -m pytest -svq \\
    models/experimental/pi0_5/tests/perf/test_pi0_5_denoise_only_single_chip.py

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
NUM_DENOISE_STEPS = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
NUM_CAMERAS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
LANG_SEQ_LEN = int(os.environ.get("PI05_LANG_SEQ_LEN", "256"))  # input language seq len (try 32 for short-prompt)
SEED = 0
TRACE_REGION_SIZE = 134_217_728  # 128 MiB
PCC_THRESHOLD = float(os.environ.get("PI05_DENOISE_PCC", "0.99"))

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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


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


def _set_fixed_noise(model):
    """Pin deterministic noise so eager and traced denoise are comparable."""
    cfg = model.config
    ah = cfg.action_horizon
    ahp = model._action_horizon_padded
    torch.manual_seed(SEED + 1)
    noise = torch.zeros(1, ahp, cfg.action_dim, dtype=torch.float32)
    noise[:, :ah, :] = torch.randn(1, ah, cfg.action_dim)
    model.x_t_ttnn = ttnn.from_torch(
        noise,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model.device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    model.resample_noise = False


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
def test_pi0_5_denoise_only_single_chip(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN, use_upstream_masks

    # GUARD: this test must run on a SINGLE chip so the denoise op graph emits
    # no cross-chip D2D (SendDirectAsync/RecvDirectAsync). The `device` fixture
    # hands back a single-device handle; assert its span is exactly 1 before we
    # build/run anything. (A multi-chip MeshDevice — e.g. the GLX galaxy mesh —
    # would report 28-32 here.) Note: the cluster driver still powers all
    # physical chips at init; what matters is that THIS handle spans one.
    n_chips = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    assert n_chips == 1, (
        f"denoise-only single-chip test requires a 1-chip device handle, "
        f"got {n_chips}. A multi-chip mesh would introduce D2D ops and "
        f"invalidate the no-D2D baseline."
    )
    print(f"\n🔒 single-chip guard OK: device spans {n_chips} chip")

    action_horizon = action_horizon_from_checkpoint(CHECKPOINT_DIR)
    cfg = Pi0_5ModelConfig(action_horizon=action_horizon, num_denoising_steps=NUM_DENOISE_STEPS)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    model = Pi0_5ModelTTNN(cfg, loader, device)
    ah = cfg.action_horizon

    imgs, im_masks, lt, lm = _build_inputs(device, NUM_CAMERAS)
    batch_size = lt.shape[0]

    if use_upstream_masks():
        prefix_len = cfg.siglip_config.num_patches * len(im_masks) + LANG_SEQ_LEN
        model.prepare_upstream_artifacts(im_masks, lm, prefix_len=prefix_len)

    # ---- Build the prefix ONCE (outside the timed region). No D2D, single chip.
    prefix_kv_cache, upstream_artifacts, keep_padded = model._prepare_prefix(imgs, im_masks, lt, lm, batch_size)
    ttnn.synchronize_device(device)
    print(f"\n📦 prefix KV built once; denoise-only loop is the measured region")

    def _denoise():
        return model._run_denoise_loop(
            prefix_kv_cache,
            upstream_artifacts,
            keep_padded_expert=keep_padded,
            batch_size=batch_size,
            state=None,
        )

    # ---- EAGER mode (for Tracy device-op profiling) ----
    # Tracy cannot attribute device ops inside a traced replay, so the
    # profiling path runs the denoise loop EAGERLY (non-traced) wrapped in
    # PHASE_denoise/PHASE_end signposts. Run under the tracy profiler:
    #   EAGER=1 python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \
    #     -m pytest -svq <this file>
    # then filter the CSV by --start-signpost PHASE_denoise --end-signpost PHASE_end.
    # 1 warmup iter (JIT) + 1 signposted iter; no trace capture, no D2D.
    if os.environ.get("EAGER", "").lower() in ("1", "true", "yes", "on"):
        from tracy import signpost

        print("\n🔬 EAGER device-profile mode: 1 warmup + 1 signposted denoise-only iter")
        _set_fixed_noise(model)
        out = _denoise()
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)

        _set_fixed_noise(model)
        signpost("PHASE_denoise")
        t0 = time.perf_counter()
        out = _denoise()
        ttnn.synchronize_device(device)
        signpost("PHASE_end")
        eager_ms = (time.perf_counter() - t0) * 1000.0
        eager_actions = ttnn.to_torch(out)[:, :ah, : cfg.action_dim].float()
        assert torch.isfinite(eager_actions).all(), "eager denoise produced NaN/Inf"

        print("\n" + "=" * 72)
        print(f"  PI0.5 DENOISE-ONLY EAGER (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
        print("=" * 72)
        print(f"   Config:            cameras={NUM_CAMERAS}, denoise_steps={NUM_DENOISE_STEPS}")
        print(f"   Eager signposted:  {eager_ms:7.2f} ms/loop   ({eager_ms / NUM_DENOISE_STEPS:.2f} ms/step)")
        print(f"   Signposts:         PHASE_denoise → PHASE_end")
        print("=" * 72)
        return

    # ---- Eager reference (non-traced) on fixed noise ----
    _set_fixed_noise(model)
    eager_out = _denoise()
    ttnn.synchronize_device(device)
    eager_actions = ttnn.to_torch(eager_out)[:, :ah, : cfg.action_dim].float()
    assert torch.isfinite(eager_actions).all(), "eager denoise produced NaN/Inf"

    # ---- Warmup the trace path (JIT) ----
    for _ in range(NUM_WARMUP):
        _set_fixed_noise(model)
        out = _denoise()
        ttnn.synchronize_device(device)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)

    # ---- Capture denoise-only trace ----
    _set_fixed_noise(model)
    capture_start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_trace = _denoise()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    capture_ms = (time.perf_counter() - capture_start) * 1000.0

    traced_actions = ttnn.to_torch(out_trace)[:, :ah, : cfg.action_dim].float()
    assert torch.isfinite(traced_actions).all(), "traced denoise produced NaN/Inf"
    pcc = _pcc(traced_actions, eager_actions)

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
    per_step = avg / NUM_DENOISE_STEPS

    print("\n" + "=" * 72)
    print(f"  PI0.5 DENOISE-ONLY (SINGLE CHIP, NO D2D) — {CHECKPOINT_DIR.name}")
    print("=" * 72)
    print(f"   Config:            cameras={NUM_CAMERAS}, denoise_steps={NUM_DENOISE_STEPS}")
    print(f"   Trace capture:     {capture_ms:7.2f} ms (one-time)")
    print(f"   Denoise-only avg:  {avg:7.2f} ms/loop   ({per_step:.2f} ms/step)")
    print(f"   Per-call min/max:  {mn:7.2f} / {mx:7.2f} ms   stddev {sd:.2f}")
    print(f"   PCC traced vs eager: {pcc:.6f}  (threshold {PCC_THRESHOLD})")
    print("=" * 72)

    assert pcc >= PCC_THRESHOLD, f"denoise trace PCC {pcc:.6f} < {PCC_THRESHOLD}"
    assert avg > 0
