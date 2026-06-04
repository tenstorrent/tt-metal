# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy performance harness for the ACE-Step **DiT denoise loop** (default: 1 Euler step).

Profiles the production path used by ``run_prompt_to_wav.py`` after conditioning:

    ``run_ttnn_denoise_loop`` (``ttnn_impl/e2e_model_tt.py``)
    → ``AceStepV15TTNNPipeline.forward_with_temb_tp`` × ``infer_steps``
    → ``dit_sampling_ttnn`` Euler / APG (or ADG) glue

Defaults: ``ACE_STEP_PERF_INFER_STEPS=1`` (one Euler update), ``ACE_STEP_PERF_DURATION_SEC=1.0``
(25 latent frames). DiT body trace is **off** (``trace_state=None``) so device profiler + Tracy CSV
merge work reliably.

Capture (from repository root):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/experimental/ace_step_v1_5/perf/test_perf_dit_denoise_loop_tracy.py::test_perf_ace_step_dit_denoise_loop_tracy_profile \\
        -v -s

Tracy writes ``generated/profiler/reports/<timestamp>/`` including ``cpp_device_perf_report.csv``.
Analyze with `tt-perf-report <https://github.com/tenstorrent/tt-perf-report>`_:

    pip install tt-perf-report
    tt-perf-report generated/profiler/reports/<timestamp>/cpp_device_perf_report.csv

If Tracy merge reports ``Device data missing``, post-process without ``-p`` or run:

    python tools/tracy/process_ops_logs.py --date generated/profiler/reports/<timestamp>

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` (profiler flush and TTNN trace capture conflict).

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_PERF_VARIANT`` (default ``acestep-v15-base``).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control.
- ``ACE_STEP_DIT_PERF_TINY=1``: random 2-layer weights (smoke / CI without hub checkpoints).
- ``ACE_STEP_PERF_DURATION_SEC`` (default ``1.0``): audio duration → ``frames = 25 × sec``.
- ``ACE_STEP_PERF_INFER_STEPS`` (default ``1``): Euler steps inside each timed ``run_ttnn_denoise_loop``.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): full denoise-loop warmups before the timed pass.
- ``ACE_STEP_PERF_GUIDANCE_SCALE``: CFG scale (default ``7.0`` for base, ``1.0`` for turbo variants).
- ``ACE_STEP_DIT_PERF_ENC_SEQ`` (default ``128``): synthetic encoder sequence length.
- ``ACE_STEP_PERF_SEED`` (default ``0``): latent noise seed.
- ``ACE_STEP_TRACY_EACH_DENOISE_STEP``: set to ``1`` for one Tracy signpost per Euler step.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf iterations (default ``1``).
- ``ACE_STEP_PERF_MAX_SECONDS``: optional wall-time budget on the timed perf pass.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import Profiler
from models.experimental.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.experimental.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
    bf16_tile_l1_from_numpy_bc,
    concat_duplicate_batch,
    dit_init_latents_fp32_tile,
    fp32_tile_to_bf16_tile_l1,
)
from models.experimental.ace_step_v1_5.ttnn_impl.e2e_model_tt import run_ttnn_denoise_loop
from models.experimental.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_enable_tracy_profiler_env,
    ace_step_flush_device_profiler,
)


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_dit_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    model_dir = ckpt_dir / variant
    dit_st = model_dir / "model.safetensors"
    if not dit_st.is_file():
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if shards:
            dit_st = shards[0]
    return dit_st


def _dit_checkpoint_ready(ckpt_dir: Path, variant: str) -> bool:
    return _resolve_dit_checkpoint(ckpt_dir, variant).is_file()


def _build_t_schedule(*, shift: float, infer_steps: int) -> list[float]:
    return [float(shift) * (float(i) / float(infer_steps)) for i in range(infer_steps, -1, -1)]


def _tracy_signpost(label: str) -> None:
    if _is_ci():
        return
    try:
        from tracy import signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        signpost(label)
    except Exception:
        pass


def _make_tiny_checkpoint(tmp_dir: Path) -> Path:
    from models.experimental.ace_step_v1_5.tests.test_e2e_model import _make_random_safetensors

    ckpt_path, _dims = _make_random_safetensors(tmp_dir, num_layers=2)
    return ckpt_path


def _prepare_dit_inputs(
    *,
    pipe: AceStepV15TTNNPipeline,
    device: ttnn.Device,
    mem: ttnn.MemoryConfig,
    frames: int,
    batch: int,
    enc_seq: int,
    seed: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor | None, np.ndarray]:
    cond_dim = int(pipe.cond_dim)
    enc_hs = torch.randn(batch, enc_seq, cond_dim, dtype=torch.bfloat16)
    ctx = torch.randn(batch, frames, 128, dtype=torch.bfloat16)

    enc_tt = bf16_tile_l1_from_numpy_bc(enc_hs.contiguous().float().numpy(), device=device, dram=mem)
    ctx_tt = bf16_tile_l1_from_numpy_bc(ctx.contiguous().float().numpy(), device=device, dram=mem)

    xt_tt = dit_init_latents_fp32_tile(
        batch=1,
        frames=int(frames),
        channels=64,
        device=device,
        dram=mem,
        seed=int(seed),
    )
    xt_row = fp32_tile_to_bf16_tile_l1(xt_tt, dram=mem)
    try:
        ttnn.deallocate(xt_tt)
    except Exception:
        pass

    if batch > 1:
        xt_pipe = concat_duplicate_batch(xt_row)
        try:
            ttnn.deallocate(xt_row)
        except Exception:
            pass
    else:
        xt_pipe = xt_row

    enc_mask_np = np.ones((batch, enc_seq), dtype=np.bool_)
    enc_mask_b1qk = pipe.build_encoder_attention_mask_b1qk_optional(
        xt_bt64=xt_pipe,
        context_latents_bt128=ctx_tt,
        encoder_hidden_states_btd=enc_tt,
        encoder_attention_mask_1d_bk=enc_mask_np,
    )
    try:
        ttnn.deallocate(xt_pipe)
    except Exception:
        pass
    return enc_tt, ctx_tt, enc_mask_b1qk, enc_mask_np


def _run_denoise_loop_tracy_harness(
    device: ttnn.Device,
    *,
    checkpoint_path: Path,
    variant_label: str,
) -> None:
    duration = float(os.environ.get("ACE_STEP_PERF_DURATION_SEC", "1.0"))
    infer_steps = max(1, int(os.environ.get("ACE_STEP_PERF_INFER_STEPS", "1")))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    enc_seq = int(os.environ.get("ACE_STEP_DIT_PERF_ENC_SEQ", "128"))
    shift = float(os.environ.get("ACE_STEP_PERF_SHIFT", "1.0"))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))

    is_turbo = "turbo" in variant_label.lower()
    gs = float(os.environ.get("ACE_STEP_PERF_GUIDANCE_SCALE", "1.0" if is_turbo else "7.0"))
    do_cfg = gs > 1.0 + 1e-6
    batch = 2 if do_cfg else 1

    frames = max(1, int(round(duration * 25.0)))
    t_schedule = _build_t_schedule(shift=shift, infer_steps=infer_steps)
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")
    profiler = Profiler()
    profiler.clear()
    is_ci = _is_ci()
    trace_each_step = os.environ.get("ACE_STEP_TRACY_EACH_DENOISE_STEP", "").lower() in ("1", "true", "yes")
    try:
        flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        flush_every = 1

    # --- INIT -------------------------------------------------------------------------
    profiler.disable()
    profiler.start("ace_step_dit_denoise_init", force_enable=True)
    _tracy_signpost("DIT_DENOISE_INIT")

    pipe = AceStepV15TTNNPipeline(
        device=device,
        checkpoint_safetensors_path=str(checkpoint_path),
        timesteps_host=timesteps_host,
        expected_input_length=frames,
    )

    profiler.end("ace_step_dit_denoise_init", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    def _denoise_once() -> ttnn.Tensor | None:
        if trace_each_step and not is_ci:
            for step_idx in range(infer_steps):
                _tracy_signpost(f"DIT_DENOISE_STEP_{step_idx}")
        enc_tt, ctx_tt, enc_mask_b1qk, enc_mask_np = _prepare_dit_inputs(
            pipe=pipe,
            device=device,
            mem=mem,
            frames=frames,
            batch=batch,
            enc_seq=enc_seq,
            seed=seed,
        )
        try:
            return run_ttnn_denoise_loop(
                pipe=pipe,
                device=device,
                act_dtype=ttnn.bfloat16,
                mem=mem,
                t_schedule=t_schedule,
                frames=frames,
                enc_tt_pipe=enc_tt,
                ctx_tt_pipe=ctx_tt,
                do_cfg=do_cfg,
                seed=seed,
                guidance_scale=gs,
                use_adg=False,
                enc_mask=None if enc_mask_b1qk is not None else (enc_mask_np[0:1] if do_cfg else enc_mask_np),
                encoder_attention_mask_b1qk=enc_mask_b1qk,
                deallocate_ctx_latents=False,
                deallocate_encoder_mask=False,
                trace_state=None,
                return_device_latents=True,
            )
        finally:
            if ctx_tt is not None:
                try:
                    ttnn.deallocate(ctx_tt)
                except Exception:
                    pass

    # --- COMPILE PASS -----------------------------------------------------------------
    profiler.start("ace_step_dit_denoise_compile_pass", force_enable=True)
    _tracy_signpost("DIT_DENOISE_COMPILE_PASS")
    lat_tt = _denoise_once()
    if lat_tt is not None:
        try:
            ttnn.deallocate(lat_tt)
        except Exception:
            pass
    profiler.end("ace_step_dit_denoise_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # --- WARMUP -----------------------------------------------------------------------
    profiler.start("ace_step_dit_denoise_warmup", force_enable=True)
    _tracy_signpost("DIT_DENOISE_WARMUP")
    for _ in range(warmup):
        lat_tt = _denoise_once()
        if lat_tt is not None:
            try:
                ttnn.deallocate(lat_tt)
            except Exception:
                pass
    profiler.end("ace_step_dit_denoise_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    ace_step_flush_device_profiler(device)

    # --- PERF PASS (one full denoise loop = infer_steps Euler steps) ------------------
    profiler.enable()
    profiler.start("ace_step_dit_denoise_perf_pass")
    _tracy_signpost("DIT_DENOISE_PERF_PASS")

    lat_tt = _denoise_once()
    if lat_tt is not None:
        try:
            ttnn.deallocate(lat_tt)
        except Exception:
            pass

    ttnn.synchronize_device(device)
    profiler.end("ace_step_dit_denoise_perf_pass")
    ace_step_flush_device_profiler(device)

    profiler.print()
    init_wall = profiler.get("ace_step_dit_denoise_init")
    compile_wall = profiler.get("ace_step_dit_denoise_compile_pass")
    warmup_wall = profiler.get("ace_step_dit_denoise_warmup")
    perf_wall = profiler.get("ace_step_dit_denoise_perf_pass")
    per_step_ms = (perf_wall * 1000.0 / max(1, infer_steps)) if infer_steps else 0.0

    logger.info(
        "AceStep DiT denoise-loop Tracy (variant={}, frames={}, euler_steps={}, do_cfg={}, batch={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/euler_step)",
        variant_label,
        frames,
        infer_steps,
        do_cfg,
        batch,
        init_wall,
        compile_wall,
        warmup,
        warmup_wall,
        perf_wall,
        per_step_ms,
    )
    logger.info(
        "Tracy CSV: generated/profiler/reports/<latest>/cpp_device_perf_report.csv — "
        "analyze with: tt-perf-report <path>/cpp_device_perf_report.csv"
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_dit_denoise_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_dit_denoise_loop_tracy_profile(device, request):
    """Profile ``run_ttnn_denoise_loop`` with Tracy (default 1 Euler step) and device CSV export."""
    ace_step_enable_tracy_profiler_env()

    def _final_profiler_flush() -> None:
        for _ in range(2):
            ace_step_flush_device_profiler(device)

    request.addfinalizer(_final_profiler_flush)

    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_PERF_VARIANT", "acestep-v15-base")
    use_tiny = os.environ.get("ACE_STEP_DIT_PERF_TINY", "").lower() in ("1", "true", "yes")

    if use_tiny:
        with tempfile.TemporaryDirectory(prefix="ace_step_dit_denoise_perf_tiny_") as tmp:
            dit_st = _make_tiny_checkpoint(Path(tmp))
            ace_step_flush_device_profiler(device)
            _run_denoise_loop_tracy_harness(device, checkpoint_path=dit_st, variant_label="tiny")
        return

    if not _dit_checkpoint_ready(ckpt_root, variant):
        if _perf_download_disabled():
            pytest.skip(
                f"Missing DiT checkpoint under {ckpt_root / variant}. "
                "Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch, or set ACE_STEP_DIT_PERF_TINY=1."
            )
        pytest.importorskip("huggingface_hub")
        ckpt_root.mkdir(parents=True, exist_ok=True)
        logger.info("ACE-Step DiT denoise perf: fetching {} …", variant)
        try:
            _ensure_variant(variant, ckpt_root)
        except Exception as exc:
            pytest.skip(f"Hugging Face download failed: {exc}")
        if not _dit_checkpoint_ready(ckpt_root, variant):
            pytest.fail(f"Download finished but DiT weights still missing under {ckpt_root / variant}")

    dit_st = _resolve_dit_checkpoint(ckpt_root, variant)
    ace_step_flush_device_profiler(device)
    _run_denoise_loop_tracy_harness(device, checkpoint_path=dit_st, variant_label=variant)
