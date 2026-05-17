# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall-time perf test for ACE-Step v1.5 — Tracy-free.

This is the integration counterpart to the five per-module trace+2CQ tests
(``tests/test_*_trace_2cq.py``). It runs the real ``AceStepE2EModel.generate(prompt)``
against the production checkpoints and reports wall-clock time for:

- compile pass (first generate, populates program cache + per-layer mask caches)
- warmup pass (second generate, settles steady-state allocations)
- timed perf pass (third generate, the number you compare against budgets)

Why not just use ``test_perf_e2e_model_tt_tracy.py``? That test was authored to be
invoked under ``python -m tracy -p -r -v -m pytest …``, which adds Tracy's device-merge
step (``cpp_device_perf_report.csv`` join). At ACE-Step's op-count the device merge
asserts ``Device data missing: Op N not present`` once N crosses ~1 M, so the Tracy
invocation reliably fails on production schedules. Plain pytest works on the same
test, but the docstring strongly encourages Tracy.

This file is the explicit "I just want wall-time, no Tracy" entrypoint:

    pytest models/demos/ace_step_v1_5/perf/test_perf_e2e_wall_time.py -v -s

Useful env (shared with the Tracy test):

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_PERF_VARIANT``: checkpoint location + DiT bundle.
- ``ACE_STEP_PERF_DURATION_SEC`` (default 1.0): latent duration → frames = 25 × secs.
- ``ACE_STEP_PERF_INFER_STEPS`` (default 8): Euler steps in the denoise loop.
- ``ACE_STEP_PERF_WARMUP`` (default 1): number of warmup generates between compile and perf.
- ``ACE_STEP_PERF_PROMPT`` (default: a short electronic prompt): caption text.
- ``ACE_STEP_PERF_GUIDANCE_SCALE`` / ``ACE_STEP_PERF_USE_ADG``: override CFG behavior.
- ``ACE_STEP_PERF_MAX_SECONDS``: assert the timed generate finishes within this budget.
- ``ACE_STEP_PERF_TIMED_RUNS`` (default 1): number of timed generates to average for the report.
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD=0``: skip Hugging Face fetch.
- ``ACE_STEP_USE_TRACE=1``: wrap the per-step DiT body (``pipe.forward_with_temb_tp``) in a
  captured TTNN trace + 2CQ replay. The trace covers patch_embed + DiT core + output_head;
  the pre/post-DiT glue (BF16 row build, CFG split, APG/ADG guidance, Euler step) remains
  eager because it depends on per-step Python scalars. The trace is captured lazily after the
  first two eager Euler steps of the compile pass, then replayed for every remaining step in
  that generate and every step of subsequent generates. Expect ``compile_and_first_generate``
  to grow slightly (warmup + capture overhead), ``warmup_generate_total`` to shrink, and
  ``perf_generate_avg`` to shrink the most.

The test uses the **shared session ``device``** from ``perf/conftest.py`` (which now
defaults to 2 CQs + 128 MB trace region), so it runs cleanly alongside the sibling
``test_perf_e2e_model_tt_tracy_profile`` in the same pytest session.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.ace_step_v1_5.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import AceStepE2EModel, E2EConfig


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
            "ACE-Step wall-time perf: checkpoints incomplete under {}; fetching {} + vae + Qwen via Hugging Face …",
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
    """Pick a sensible E2EConfig from env (matches the Tracy test defaults)."""
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


def _timed_generate(model: AceStepE2EModel, prompt: str, *, device) -> float:
    """Run one ``generate`` blocking on device sync, return wall-clock seconds."""
    t0 = time.perf_counter()
    _ = model.generate(prompt)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return float(t1 - t0)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_e2e_wall_time(device):
    """End-to-end wall-time perf without Tracy.

    Uses the ``device`` fixture from ``perf/conftest.py`` (session-scoped, 2 CQs by default)
    so this test shares the same device with ``test_perf_e2e_model_tt_tracy_profile`` when
    both are collected in the same pytest invocation.

    Stages (each timed separately, all printed on stdout):

        1. ``compile_and_first_generate``: model construction + first ``generate``. Includes
           weight loads, program-cache fills, conv ``prepare_conv_weights``, the per-step
           ``compute_temb_tp`` precompute, and the lyric/timbre encoder one-time cache. This is
           always the slowest pass.
        2. ``warmup_generate_total``: ``ACE_STEP_PERF_WARMUP`` (default 1) extra ``generate``
           calls back-to-back. Steady-state cache hits + reused trace region.
        3. ``perf_generate_avg``: mean wall time over ``ACE_STEP_PERF_TIMED_RUNS`` (default 1)
           timed generates. If ``ACE_STEP_PERF_MAX_SECONDS`` is set, the test asserts the
           per-generate average stays under that budget.
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
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    timed_runs = max(1, int(os.environ.get("ACE_STEP_PERF_TIMED_RUNS", "1")))

    # --- COMPILE + FIRST GENERATE ----------------------------------------------------------------
    t0 = time.perf_counter()
    model = AceStepE2EModel(cfg, device)
    _ = model.generate(prompt)
    ttnn.synchronize_device(device)
    compile_and_first = float(time.perf_counter() - t0)

    # --- WARMUP -----------------------------------------------------------------------------------
    warmup_total = 0.0
    for _ in range(warmup):
        warmup_total += _timed_generate(model, prompt, device=device)

    # --- TIMED PERF PASS(ES) ----------------------------------------------------------------------
    perf_runs = [_timed_generate(model, prompt, device=device) for _ in range(timed_runs)]
    perf_avg = sum(perf_runs) / len(perf_runs)
    perf_best = min(perf_runs)
    perf_worst = max(perf_runs)

    # --- REPORT -----------------------------------------------------------------------------------
    print(
        f"[ace_step_v1_5][wall_time] variant={variant} "
        f"duration={cfg.duration_sec}s frames={int(round(cfg.duration_sec * 25.0))} "
        f"steps={cfg.infer_steps} gs={cfg.guidance_scale} adg={cfg.use_adg}",
        flush=True,
    )
    print(
        f"[ace_step_v1_5][wall_time] compile_and_first_generate={compile_and_first:.3f}s "
        f"warmup_total({warmup}x)={warmup_total:.3f}s "
        f"perf_generate_avg({timed_runs}x)={perf_avg:.3f}s "
        f"(best={perf_best:.3f}s worst={perf_worst:.3f}s)",
        flush=True,
    )
    logger.info(
        "ACE-Step wall-time perf: compile+first_generate=%.3fs warmup_total(%dx)=%.3fs perf_generate_avg(%dx)=%.3fs",
        compile_and_first,
        warmup,
        warmup_total,
        timed_runs,
        perf_avg,
    )

    # --- OPTIONAL BUDGET ASSERTION ----------------------------------------------------------------
    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        budget_s = float(budget)
        assert perf_avg <= budget_s, (
            f"perf_generate_avg={perf_avg:.3f}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget_s}s "
            f"(per-run: {[f'{r:.3f}s' for r in perf_runs]}). "
            "Unset ACE_STEP_PERF_MAX_SECONDS to disable this assert."
        )
