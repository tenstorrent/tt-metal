# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy E2E perf for ``AceStepE2EModel`` (trace on by default).

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_e2e_model_tt_tracy.py -v -s

CSV / Tracy artifacts are written under ``generated/profiler/reports/<timestamp>/`` as documented in
``docs/source/tt-metalium/tools/tracy_profiler.rst``.

Optional environment variables:

- ``ACE_STEP_CKPT_DIR``: checkpoint root (default: ``~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints``).
- ``ACE_STEP_PERF_VARIANT``: DiT bundle folder name (default: ``acestep-v15-turbo``).
- ``ACE_STEP_PERF_NO_DOWNLOAD``: set to ``1`` to skip Hugging Face fetch when weights are missing (offline / CI without cache).
- ``ACE_STEP_PERF_DOWNLOAD``: ``0`` / ``false`` / ``no`` also disables automatic download (same as ``NO_DOWNLOAD``).
- ``ACE_STEP_PERF_DURATION_SEC``, ``ACE_STEP_PERF_INFER_STEPS``: shorten the run while iterating on kernels
  (default duration ``1.0`` s → 25 latent frames, single-shot VAE decode with default chunk size).
- ``ACE_STEP_PERF_WARMUP``: number of ``generate`` warmup iterations after the compile pass (default: ``1``).
- ``ACE_STEP_PERF_MAX_SECONDS``: optional upper bound on the timed perf ``generate`` (fails the test if exceeded).
- ``ACE_STEP_TRACY_EACH_DENOISE_STEP``: set to ``1`` for one Tracy signpost per Euler step inside the DiT loop.

When weights are absent, this test pulls the same bundles as ``run_prompt_to_wav.py`` via ``huggingface_hub``
(requires network on first run).

**Important:** do **not** enable TTNN graph trace here (``ACE_STEP_USE_TRACE=1`` / default
:class:`AceStepE2EModel` trace). Device Tracy + ``TT_METAL_DEVICE_PROFILER`` conflict with
``begin_trace_capture`` (no host read/write or event sync during capture). This harness uses
``use_trace=False`` on :class:`AceStepE2EModel` — same constraint as the DiT/cond/VAE Tracy tests.

If Tracy's report step fails with ``Device data missing`` / CSV mismatch (known limitation when too many
device ops are captured), prefer ``TT_METAL_DEVICE_PROFILER=1 pytest …`` followed by
``python tools/tracy/process_ops_logs.py --date``, or drop Tracy's ``-p`` merge flags for host-only timelines.

``tools/tracy/process_ops_logs.py`` keeps Tracy host-side rows when ``cpp_device_perf_report.csv`` omits an
op (with a loguru warning). Set ``TRACY_STRICT_DEVICE_PERF_CSV=1`` to restore the old hard failure for debugging.
Hosts-only capture: ``python -m tracy … --no-device`` (no ``TT_METAL_DEVICE_PROFILER``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import Profiler
from models.demos.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import AceStepE2EModel, E2EConfig


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


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


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_e2e_model_tt_tracy_profile(device):
    """Fill program caches, warm up, then run one timed ``generate`` with Tracy signposts."""
    pytest.importorskip("transformers")

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_PERF_VARIANT", "acestep-v15-turbo")

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
            "ACE-Step perf test: checkpoints incomplete under {}; fetching {} + vae + Qwen via Hugging Face …",
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

    duration = float(os.environ.get("ACE_STEP_PERF_DURATION_SEC", "1.0"))
    infer_steps = int(os.environ.get("ACE_STEP_PERF_INFER_STEPS", "8"))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    prompt = os.environ.get(
        "ACE_STEP_PERF_PROMPT",
        "Electronic dance track with deep bass, punchy kick drum, instrumental.",
    )

    is_turbo = "turbo" in variant.lower()
    gs = float(os.environ.get("ACE_STEP_PERF_GUIDANCE_SCALE", "1.0" if is_turbo else "7.0"))
    use_adg_env = os.environ.get("ACE_STEP_PERF_USE_ADG", "").lower()
    if use_adg_env == "":
        use_adg_bool = not is_turbo
    else:
        use_adg_bool = use_adg_env in ("1", "true", "yes")

    cfg = E2EConfig(
        checkpoint_safetensors_path=str(dit_st),
        vae_dir=str(vae_dir),
        text_model_dir=str(text_dir),
        silence_latent_path=str(silence_pt),
        duration_sec=duration,
        infer_steps=infer_steps,
        guidance_scale=gs,
        use_adg=use_adg_bool,
        qwen_safetensors_path=str(qwen_st),
    )

    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()

    profiler.disable()
    profiler.start("ace_step_e2e_init_and_compile_pass", force_enable=True)
    if not is_ci:
        try:
            from tracy import signpost

            signpost("COMPILE_RUN")
        except ImportError:
            pass

    # Eager E2E only — TTNN trace + Tracy/device profiler cannot share the command queue safely.
    model = AceStepE2EModel(cfg, device, use_trace=False)
    _ = model.generate(prompt)

    profiler.end("ace_step_e2e_init_and_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)

    profiler.start("ace_step_e2e_warmup", force_enable=True)
    if not is_ci:
        try:
            from tracy import signpost

            signpost("WARMUP_RUN")
        except ImportError:
            pass
    for _ in range(warmup):
        _ = model.generate(prompt)
    profiler.end("ace_step_e2e_warmup", force_enable=True)
    ttnn.synchronize_device(device)

    profiler.enable()
    profiler.start("ace_step_e2e_perf_pass")
    if not is_ci:
        try:
            from tracy import signpost

            signpost("PERF_RUN")
        except ImportError:
            pass

    _ = model.generate(prompt)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_e2e_perf_pass")

    profiler.print()
    compile_wall = profiler.get("ace_step_e2e_init_and_compile_pass")
    infer_wall = profiler.get("ace_step_e2e_perf_pass")
    warmup_wall = profiler.get("ace_step_e2e_warmup")
    logger.info(
        f"AceStep E2E Tracy harness: compile+first_generate={compile_wall:.3f}s, "
        f"warmup_total({warmup}x)={warmup_wall:.3f}s, perf_generate={infer_wall:.3f}s"
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert infer_wall <= float(budget), (
            f"perf_generate {infer_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s "
            "(unset variable to disable this assert)."
        )
