# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SwinV2-style end-to-end performant test for ACE-Step v1.5.

Mirrors ``models/experimental/swin_v2/tests/perf/test_e2e_performant.py``: the
:class:`AceStepPerformantRunner` constructor auto-captures the DiT body trace
(``begin_trace_capture`` / ``end_trace_capture``) via one warmup ``generate`` call,
and the test then times N back-to-back ``run(prompt)`` calls on the steady-state
trace+2CQ path.

Differences from :mod:`models.demos.ace_step_v1_5.perf.test_perf_e2e_wall_time`:

- The wall-time test exposes the eager-vs-trace toggle via ``ACE_STEP_USE_TRACE``
  and reports compile/warmup/perf separately. Here tracing is **always on** (the
  runner force-enables it) and we report a single average + FPS, matching SwinV2.
- The runner amortizes the trace capture into ``__init__``, so the timed loop has
  no compile-pass tail blending into the average.

Run::

    pytest models/demos/ace_step_v1_5/perf/test_perf_e2e_performant.py -v -s

Useful env (mostly shared with the other perf tests):

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_PERF_VARIANT``: checkpoint location + DiT
  bundle name (default variant: ``acestep-v15-turbo`` so the warmup capture stays
  short and the timed loop covers the steady-state DiT body).
- ``ACE_STEP_PERF_DURATION_SEC`` (default ``1.0``): latent duration; frames = 25 * secs.
- ``ACE_STEP_PERF_INFER_STEPS`` (default ``8``): Euler steps per generate.
- ``ACE_STEP_PERF_GUIDANCE_SCALE`` / ``ACE_STEP_PERF_USE_ADG``: override CFG.
- ``ACE_STEP_PERF_PROMPT``: caption text (default: short electronic prompt).
- ``ACE_STEP_PERF_ITERS`` (default ``10``): number of timed ``run`` iterations.
- ``ACE_STEP_PERF_MAX_SECONDS``: optional per-iteration budget — fails the test
  when the average exceeds it (unset to disable).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD=0``: skip Hugging Face
  fetch (same semantics as the wall-time test).

Device fixture: uses ``perf/conftest.py``'s session-scoped ``device`` (2 CQs + 128 MB
trace region by default — see ``ACE_STEP_NUM_CQS`` to override). No additional
``device_params`` decorator is needed; the runner asserts the TTNN build has
``begin_trace_capture`` / ``execute_trace`` at construction time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.runner.performant_runner import AceStepPerformantRunner
from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import E2EConfig

# ---------------------------------------------------------------------------------------------
# Checkpoint resolution / Hugging Face fallback. Duplicated (intentionally) from
# ``test_perf_e2e_wall_time.py`` so this file is self-contained and can be run / picked up
# independently. If you change the layout, change both.
# ---------------------------------------------------------------------------------------------


def _resolve_checkpoint_layout(
    ckpt_dir: Path,
    variant: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return ``(model_dir, dit_safetensors, silence_latent, vae_dir, qwen_dir)``."""
    model_dir = ckpt_dir / variant
    dit_st = model_dir / "model.safetensors"
    if not dit_st.is_file():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if shards:
            dit_st = shards[0]
    silence = model_dir / "silence_latent.pt"
    vae_dir = ckpt_dir / "vae"
    text_dir = ckpt_dir / "Qwen3-Embedding-0.6B"
    return model_dir, dit_st, silence, vae_dir, text_dir


def _layout_complete(layout: tuple[Path, Path, Path, Path, Path]) -> bool:
    _model_dir, dit_st, silence_pt, vae_dir, text_dir = layout
    qwen_st = text_dir / "model.safetensors"
    return dit_st.is_file() and silence_pt.is_file() and (vae_dir / "config.json").is_file() and qwen_st.is_file()


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_or_fetch_checkpoints(ckpt_root: Path, variant: str) -> tuple[Path, Path, Path, Path]:
    """Return ``(dit_safetensors, silence_pt, vae_dir, qwen_safetensors)`` or skip the test."""
    layout = _resolve_checkpoint_layout(ckpt_root, variant)
    if not _layout_complete(layout):
        if _perf_download_disabled():
            pytest.skip(
                "Missing ACE-Step checkpoints under "
                f"{ckpt_root} (variant {variant!r}). "
                "Allow automatic download (default): unset ACE_STEP_PERF_NO_DOWNLOAD and "
                "ACE_STEP_PERF_DOWNLOAD must not be 0/false/no; or install weights per "
                "models/demos/ace_step_v1_5/README.md."
            )
        pytest.importorskip("huggingface_hub")
        ckpt_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ACE-Step performant perf: checkpoints incomplete under {}; fetching {} + vae + Qwen via Hugging Face …",
            ckpt_root,
            variant,
        )
        try:
            _ensure_variant(variant, ckpt_root)
            _ensure_variant("vae", ckpt_root)
            _ensure_variant("Qwen3-Embedding-0.6B", ckpt_root)
        except Exception as exc:
            pytest.skip(
                f"Hugging Face download failed ({type(exc).__name__}: {exc}). "
                "Check network / HF_TOKEN, or place checkpoints manually under "
                f"{ckpt_root} (see README)."
            )

        layout = _resolve_checkpoint_layout(ckpt_root, variant)
        if not _layout_complete(layout):
            pytest.fail(
                "Download finished but expected layout is still incomplete "
                f"(DiT+VAE+Qwen under {ckpt_root}). Inspect hub cache paths."
            )

    _model_dir, dit_st, silence_pt, vae_dir, text_dir = layout
    qwen_st = text_dir / "model.safetensors"
    return dit_st, silence_pt, vae_dir, qwen_st


def _make_config(*, dit_st: Path, silence_pt: Path, vae_dir: Path, qwen_st: Path, variant: str) -> E2EConfig:
    """Pick a sensible E2EConfig from env (matches the wall-time / Tracy test defaults)."""
    duration = float(os.environ.get("ACE_STEP_PERF_DURATION_SEC", "1.0"))
    infer_steps = int(os.environ.get("ACE_STEP_PERF_INFER_STEPS", "8"))

    is_turbo = "turbo" in variant.lower()
    gs = float(os.environ.get("ACE_STEP_PERF_GUIDANCE_SCALE", "1.0" if is_turbo else "7.0"))
    use_adg_env = os.environ.get("ACE_STEP_PERF_USE_ADG", "").lower()
    if use_adg_env == "":
        use_adg_bool = not is_turbo
    else:
        use_adg_bool = use_adg_env in ("1", "true", "yes")

    return E2EConfig(
        checkpoint_safetensors_path=str(dit_st),
        vae_dir=str(vae_dir),
        text_model_dir=str(qwen_st.parent),
        silence_latent_path=str(silence_pt),
        duration_sec=duration,
        infer_steps=infer_steps,
        guidance_scale=gs,
        use_adg=use_adg_bool,
        qwen_safetensors_path=str(qwen_st),
    )


# ---------------------------------------------------------------------------------------------
# Test entry point (SwinV2-style: 10 timed runs through the always-on runner).
# ---------------------------------------------------------------------------------------------


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_e2e_performant(device):
    """End-to-end perf with the always-on trace+2CQ runner.

    Stages (only the timed loop is reported into ``inference_time_avg``):

        1. ``AceStepPerformantRunner(device, config)``:
              - builds :class:`AceStepE2EModel` (weight uploads, conv prep,
                per-step temb precompute, silence-context cache)
              - drives one warmup ``generate(...)`` so the DiT trace is captured
                via ``begin_trace_capture`` / ``end_trace_capture``
              - logged separately as ``warmup_seconds``
        2. Timed loop: ``ACE_STEP_PERF_ITERS`` (default 10) calls of
           ``runner.run(prompt)`` back-to-back. Each call hits the prompt cache
           after the first iteration, so steady-state cost reduces to: SDPA mask
           build (cached after iter 1) + ``prime_per_prompt`` + ``recapture`` +
           N traced DiT replays + VAE decode.

    The reported ``inference_time_avg`` is comparable to SwinV2's ``inference_time_avg``
    in :mod:`models.experimental.swin_v2.tests.perf.test_e2e_performant`.
    """
    pytest.importorskip("transformers")

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_PERF_VARIANT", "acestep-v15-base")
    dit_st, silence_pt, vae_dir, qwen_st = _resolve_or_fetch_checkpoints(ckpt_root, variant)

    cfg = _make_config(dit_st=dit_st, silence_pt=silence_pt, vae_dir=vae_dir, qwen_st=qwen_st, variant=variant)
    prompt = os.environ.get(
        "ACE_STEP_PERF_PROMPT",
        "Electronic dance track with deep bass, punchy kick drum, instrumental.",
    )
    iters = max(1, int(os.environ.get("ACE_STEP_PERF_ITERS", "10")))

    # --- Build runner + auto-capture trace via warmup generate ----------------------------------
    runner = AceStepPerformantRunner(device, cfg)
    try:
        # --- Timed loop --------------------------------------------------------------------------
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = runner.run(prompt)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()

        total = float(t1 - t0)
        inference_time_avg = total / float(iters)
        # Frames per *audio second*: total audio seconds produced / wall-clock seconds.
        audio_seconds_per_iter = float(cfg.duration_sec)
        rt_factor = audio_seconds_per_iter / inference_time_avg if inference_time_avg > 0 else float("inf")

        print(
            f"[ace_step_v1_5][performant] variant={variant} "
            f"duration={cfg.duration_sec}s frames={int(round(cfg.duration_sec * 25.0))} "
            f"steps={cfg.infer_steps} gs={cfg.guidance_scale} adg={cfg.use_adg}",
            flush=True,
        )
        print(
            f"[ace_step_v1_5][performant] warmup_capture={runner.warmup_seconds:.3f}s "
            f"inference_time_avg({iters}x)={inference_time_avg:.3f}s "
            f"audio_realtime_factor={rt_factor:.2f}x",
            flush=True,
        )
        logger.info(
            "ACE-Step performant runner: warmup_capture={:.3f}s, inference_time_avg({}x)={:.6f}s, "
            "audio_realtime_factor={:.2f}x",
            runner.warmup_seconds,
            iters,
            inference_time_avg,
            rt_factor,
        )

        budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
        if budget:
            budget_s = float(budget)
            assert inference_time_avg <= budget_s, (
                f"inference_time_avg={inference_time_avg:.3f}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget_s}s. "
                "Unset ACE_STEP_PERF_MAX_SECONDS to disable this assert."
            )
    finally:
        runner.release()
