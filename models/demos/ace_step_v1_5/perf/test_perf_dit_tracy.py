# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy-oriented performance harness for the ACE-Step v1.5 DiT stack.

Profiles the production DiT path wired in ``ttnn_impl/full_pipeline.py``:

    patch_embed (``patchify.py``) → ``TtAceStepDiTCore`` (``dit_decoder_core.py``)
    → ``TtAceStepDiTOutputHead`` (``output_head.py``)

via :meth:`AceStepV15TTNNPipeline.forward_with_temb_tp` inside a multi-step Euler schedule
(``temb`` / ``timestep_proj`` precomputed once per step, matching ``AceStepE2EModel``).

Run from the repository root (example):

    TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r -v -m pytest \\
        models/demos/ace_step_v1_5/perf/test_perf_dit_tracy.py::test_perf_ace_step_dit_tracy_profile \\
        -v -s

CSV / Tracy artifacts land under ``generated/profiler/reports/<timestamp>/`` — see
``docs/source/tt-metalium/tools/tracy_profiler.rst``.

**Important:** do **not** set ``ACE_STEP_USE_TRACE=1`` for this test. Device Tracy profiling
and TTNN trace capture are mutually exclusive (``ttnn.synchronize_device`` is illegal during
trace capture, and trace replay does not re-execute Python layer code for per-layer profiler
flush). Use trace for steady-state wall time (``test_perf_e2e_wall_time.py`` /
``perf/module_trace/test_pipe_body_trace_2cq.py``); use this test for kernel-level Tracy reports.

Optional environment variables:

- ``ACE_STEP_CKPT_DIR`` / ``ACE_STEP_PERF_VARIANT``: DiT checkpoint root + bundle folder
  (default variant: ``acestep-v15-turbo``).
- ``ACE_STEP_PERF_NO_DOWNLOAD`` / ``ACE_STEP_PERF_DOWNLOAD``: Hugging Face fetch control (same
  semantics as ``test_perf_e2e_model_tt_tracy.py``).
- ``ACE_STEP_DIT_PERF_TINY=1``: use random tiny weights (2 layers) — fast CI smoke without
  hub checkpoints; not representative of production layer count.
- ``ACE_STEP_PERF_DURATION_SEC`` (default ``1.0``): audio duration → ``frames = 25 × sec``.
- ``ACE_STEP_PERF_INFER_STEPS`` (default ``8``): Euler steps timed in the perf pass.
- ``ACE_STEP_PERF_WARMUP`` (default ``1``): extra ``forward_with_temb_tp`` warmup iterations.
- ``ACE_STEP_PERF_GUIDANCE_SCALE``: when ``> 1``, DiT batch is 2 (CFG cond+uncond), like E2E.
- ``ACE_STEP_DIT_PERF_ENC_SEQ`` (default ``128``): encoder sequence length for synthetic inputs.
- ``ACE_STEP_TRACY_EACH_DENOISE_STEP``: set to ``1`` for one Tracy signpost per Euler step.
- ``ACE_STEP_PROFILER_FLUSH_EVERY``: flush device profiler every N perf steps (default ``1``) to
  avoid Tracy ``Device data missing`` on long schedules — same as E2E denoise loop.
- ``ACE_STEP_PERF_MAX_SECONDS``: optional wall-time budget on the timed perf pass.
DiT uses production defaults from ``math_perf_env`` (LoFi + ``bfloat8_b`` weights + L1 activations);
same path as PCC tests under ``tests/``.

Alternative mode — full denoise loop (includes ``dit_sampling_ttnn`` Euler/CFG glue):

- ``ACE_STEP_DIT_PERF_MODE=denoise``: times ``run_ttnn_denoise_loop`` instead of raw
  ``forward_with_temb_tp``. Still Tracy-friendly when ``ACE_STEP_USE_TRACE`` is unset.

If Tracy's merge step fails with ``Device data missing``, run without ``-p`` for host-only
timelines, or post-process with ``python tools/tracy/process_ops_logs.py --date``.
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
from models.demos.ace_step_v1_5.demo.run_prompt_to_wav import _DEFAULT_CKPT_DIR, _ensure_variant
from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
    bf16_tile_l1_from_numpy_bc,
    concat_duplicate_batch,
    dit_init_latents_fp32_tile,
    fp32_tile_to_bf16_tile_l1,
)
from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import run_ttnn_denoise_loop
from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline


def _is_ci() -> bool:
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _perf_download_disabled() -> bool:
    if os.environ.get("ACE_STEP_PERF_NO_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return True
    dl = os.environ.get("ACE_STEP_PERF_DOWNLOAD", "").lower()
    return dl in ("0", "false", "no")


def _resolve_dit_checkpoint(ckpt_dir: Path, variant: str) -> Path:
    """Return ``model.safetensors`` for the DiT decoder bundle."""
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
    ts = [float(shift) * (float(i) / float(infer_steps)) for i in range(infer_steps, -1, -1)]
    return ts


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


def _ace_step_flush_device_profiler(device) -> None:
    if os.environ.get("TTNN_OP_PROFILER") != "1" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        return
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def _make_tiny_checkpoint(tmp_dir: Path) -> Path:
    """Minimal random DiT weights (2 layers) for offline / CI runs."""
    from models.demos.ace_step_v1_5.tests.test_e2e_model import _make_random_safetensors

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
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor | None, np.ndarray]:
    """Build device tensors for one DiT forward (CFG batch already applied when ``batch > 1``)."""
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
    return xt_pipe, ctx_tt, enc_tt, enc_mask_b1qk, enc_mask_np


def _run_forward_with_temb_tp(
    *,
    pipe: AceStepV15TTNNPipeline,
    xt_pipe: ttnn.Tensor,
    ctx_tt: ttnn.Tensor,
    enc_tt: ttnn.Tensor,
    enc_mask_b1qk: ttnn.Tensor | None,
    enc_mask_np: np.ndarray,
    temb: ttnn.Tensor,
    tp: ttnn.Tensor,
) -> ttnn.Tensor:
    return pipe.forward_with_temb_tp(
        xt_bt64=xt_pipe,
        context_latents_bt128=ctx_tt,
        encoder_hidden_states_btd=enc_tt,
        temb_bd=temb,
        timestep_proj_b6d=tp,
        attention_mask_1d_bt=None,
        encoder_attention_mask_1d_bk=None if enc_mask_b1qk is not None else enc_mask_np,
        encoder_attention_mask_b1qk=enc_mask_b1qk,
    )


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_perf_ace_step_dit_tracy_profile(device):
    """Profile DiT patch_embed + core + output_head with Tracy signposts and host Profiler."""
    ckpt_root = Path(os.environ.get("ACE_STEP_CKPT_DIR", str(_DEFAULT_CKPT_DIR))).expanduser().resolve()
    variant = os.environ.get("ACE_STEP_PERF_VARIANT", "acestep-v15-turbo")
    use_tiny = os.environ.get("ACE_STEP_DIT_PERF_TINY", "").lower() in ("1", "true", "yes")

    if use_tiny:
        with tempfile.TemporaryDirectory(prefix="ace_step_dit_perf_tiny_") as tmp:
            dit_st = _make_tiny_checkpoint(Path(tmp))
            _run_dit_tracy_harness(device, checkpoint_path=dit_st, variant_label="tiny")
    else:
        if not _dit_checkpoint_ready(ckpt_root, variant):
            if _perf_download_disabled():
                pytest.skip(
                    f"Missing DiT checkpoint under {ckpt_root / variant}. "
                    "Unset ACE_STEP_PERF_NO_DOWNLOAD to fetch, or set ACE_STEP_DIT_PERF_TINY=1."
                )
            pytest.importorskip("huggingface_hub")
            ckpt_root.mkdir(parents=True, exist_ok=True)
            logger.info("ACE-Step DiT perf: fetching {} …", variant)
            try:
                _ensure_variant(variant, ckpt_root)
            except Exception as exc:
                pytest.skip(f"Hugging Face download failed: {exc}")
            if not _dit_checkpoint_ready(ckpt_root, variant):
                pytest.fail(f"Download finished but DiT weights still missing under {ckpt_root / variant}")

        dit_st = _resolve_dit_checkpoint(ckpt_root, variant)
        _run_dit_tracy_harness(device, checkpoint_path=dit_st, variant_label=variant)


def _run_dit_tracy_harness(device, *, checkpoint_path: Path, variant_label: str) -> None:
    duration = float(os.environ.get("ACE_STEP_PERF_DURATION_SEC", "1.0"))
    infer_steps = int(os.environ.get("ACE_STEP_PERF_INFER_STEPS", "8"))
    warmup = max(0, int(os.environ.get("ACE_STEP_PERF_WARMUP", "1")))
    enc_seq = int(os.environ.get("ACE_STEP_DIT_PERF_ENC_SEQ", "128"))
    shift = float(os.environ.get("ACE_STEP_PERF_SHIFT", "1.0"))
    seed = int(os.environ.get("ACE_STEP_PERF_SEED", "0"))
    perf_mode = os.environ.get("ACE_STEP_DIT_PERF_MODE", "body").strip().lower()

    is_turbo = "turbo" in variant_label.lower()
    gs = float(os.environ.get("ACE_STEP_PERF_GUIDANCE_SCALE", "1.0" if is_turbo else "7.0"))
    do_cfg = gs > 1.0 + 1e-6
    pipe_batch = 2 if do_cfg else 1

    frames = int(round(duration * 25.0))
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

    # --- INIT (weight upload + program cache) -----------------------------------------
    profiler.disable()
    profiler.start("ace_step_dit_init", force_enable=True)
    _tracy_signpost("DIT_PIPELINE_INIT")

    pipe = AceStepV15TTNNPipeline(
        device=device,
        checkpoint_safetensors_path=str(checkpoint_path),
        timesteps_host=timesteps_host,
        expected_input_length=frames,
    )

    profiler.end("ace_step_dit_init", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    if perf_mode == "denoise":
        _run_denoise_tracy_perf(
            device=device,
            pipe=pipe,
            mem=mem,
            profiler=profiler,
            is_ci=is_ci,
            frames=frames,
            t_schedule=t_schedule,
            infer_steps=infer_steps,
            warmup=warmup,
            do_cfg=do_cfg,
            gs=gs,
            seed=seed,
            enc_seq=enc_seq,
            flush_every=flush_every,
            trace_each_step=trace_each_step,
        )
        return

    if perf_mode not in ("body", ""):
        pytest.fail(f"Unknown ACE_STEP_DIT_PERF_MODE={perf_mode!r}; use 'body' or 'denoise'.")

    xt_pipe, ctx_tt, enc_tt, enc_mask_b1qk, enc_mask_np = _prepare_dit_inputs(
        pipe=pipe,
        device=device,
        mem=mem,
        frames=frames,
        batch=pipe_batch,
        enc_seq=enc_seq,
        seed=seed,
    )

    temb_per_step: list[ttnn.Tensor] = []
    tp_per_step: list[ttnn.Tensor] = []
    for idx in range(infer_steps):
        temb, tp = pipe.compute_temb_tp(int(idx), target_batch=pipe_batch)
        temb_per_step.append(temb)
        tp_per_step.append(tp)

    # --- COMPILE PASS (first forward_with_temb_tp) ------------------------------------
    profiler.start("ace_step_dit_compile_pass", force_enable=True)
    _tracy_signpost("DIT_COMPILE_PASS")

    acoustic = _run_forward_with_temb_tp(
        pipe=pipe,
        xt_pipe=xt_pipe,
        ctx_tt=ctx_tt,
        enc_tt=enc_tt,
        enc_mask_b1qk=enc_mask_b1qk,
        enc_mask_np=enc_mask_np,
        temb=temb_per_step[0],
        tp=tp_per_step[0],
    )
    try:
        ttnn.deallocate(acoustic)
    except Exception:
        pass

    profiler.end("ace_step_dit_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    # --- WARMUP -----------------------------------------------------------------------
    profiler.start("ace_step_dit_warmup", force_enable=True)
    _tracy_signpost("DIT_WARMUP")

    for _ in range(warmup):
        step_idx = int(_ % max(1, infer_steps))
        out = _run_forward_with_temb_tp(
            pipe=pipe,
            xt_pipe=xt_pipe,
            ctx_tt=ctx_tt,
            enc_tt=enc_tt,
            enc_mask_b1qk=enc_mask_b1qk,
            enc_mask_np=enc_mask_np,
            temb=temb_per_step[step_idx],
            tp=tp_per_step[step_idx],
        )
        try:
            ttnn.deallocate(out)
        except Exception:
            pass

    profiler.end("ace_step_dit_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    # --- PERF PASS (steady-state DiT body × infer_steps) ------------------------------
    profiler.enable()
    profiler.start("ace_step_dit_perf_pass")
    _tracy_signpost("DIT_PERF_PASS")

    for step_idx in range(infer_steps):
        if trace_each_step and not is_ci:
            _tracy_signpost(f"DIT_STEP_{step_idx}")
        out = _run_forward_with_temb_tp(
            pipe=pipe,
            xt_pipe=xt_pipe,
            ctx_tt=ctx_tt,
            enc_tt=enc_tt,
            enc_mask_b1qk=enc_mask_b1qk,
            enc_mask_np=enc_mask_np,
            temb=temb_per_step[step_idx],
            tp=tp_per_step[step_idx],
        )
        try:
            ttnn.deallocate(out)
        except Exception:
            pass
        if flush_every > 0 and (step_idx + 1) % flush_every == 0:
            _ace_step_flush_device_profiler(device)

    ttnn.synchronize_device(device)
    profiler.end("ace_step_dit_perf_pass")
    _ace_step_flush_device_profiler(device)

    profiler.print()
    init_wall = profiler.get("ace_step_dit_init")
    compile_wall = profiler.get("ace_step_dit_compile_pass")
    warmup_wall = profiler.get("ace_step_dit_warmup")
    perf_wall = profiler.get("ace_step_dit_perf_pass")
    per_step_ms = (perf_wall * 1000.0 / max(1, infer_steps)) if infer_steps else 0.0

    logger.info(
        "AceStep DiT Tracy harness (mode=body, variant={}, frames={}, steps={}, batch={}): "
        "init={:.3f}s compile={:.3f}s warmup({}x)={:.3f}s perf_pass={:.3f}s (~{:.1f}ms/step)",
        variant_label,
        frames,
        infer_steps,
        pipe_batch,
        init_wall,
        compile_wall,
        warmup,
        warmup_wall,
        perf_wall,
        per_step_ms,
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert perf_wall <= float(
            budget
        ), f"ace_step_dit_perf_pass {perf_wall}s exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"

    for t in (*temb_per_step, *tp_per_step):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    for t in (xt_pipe, ctx_tt, enc_tt):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def _run_denoise_tracy_perf(
    *,
    device: ttnn.Device,
    pipe: AceStepV15TTNNPipeline,
    mem: ttnn.MemoryConfig,
    profiler: Profiler,
    is_ci: bool,
    frames: int,
    t_schedule: list[float],
    infer_steps: int,
    warmup: int,
    do_cfg: bool,
    gs: float,
    seed: int,
    enc_seq: int,
    flush_every: int,
    trace_each_step: bool,
) -> None:
    """Time ``run_ttnn_denoise_loop`` (DiT body + ``dit_sampling_ttnn`` Euler/CFG glue).

    Each call runs the full ``t_schedule`` (``infer_steps`` Euler steps) internally.
    """
    batch = 2 if do_cfg else 1
    xt_pipe, ctx_tt, enc_tt, enc_mask_b1qk, enc_mask_np = _prepare_dit_inputs(
        pipe=pipe,
        device=device,
        mem=mem,
        frames=frames,
        batch=batch,
        enc_seq=enc_seq,
        seed=seed,
    )
    try:
        ttnn.deallocate(xt_pipe)
    except Exception:
        pass
    _ = enc_mask_np

    def _denoise_once() -> None:
        _ = run_ttnn_denoise_loop(
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
            encoder_attention_mask_b1qk=enc_mask_b1qk,
            deallocate_ctx_latents=False,
            deallocate_encoder_mask=False,
            trace_state=None,
        )

    profiler.start("ace_step_dit_compile_pass", force_enable=True)
    _tracy_signpost("DIT_COMPILE_PASS")
    _denoise_once()
    profiler.end("ace_step_dit_compile_pass", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    profiler.start("ace_step_dit_warmup", force_enable=True)
    _tracy_signpost("DIT_WARMUP")
    for _ in range(warmup):
        _denoise_once()
    profiler.end("ace_step_dit_warmup", force_enable=True)
    ttnn.synchronize_device(device)
    _ace_step_flush_device_profiler(device)

    profiler.enable()
    profiler.start("ace_step_dit_perf_pass")
    _tracy_signpost("DIT_PERF_PASS")
    if trace_each_step and not is_ci:
        _tracy_signpost("DIT_DENOISE_LOOP")
    _denoise_once()
    ttnn.synchronize_device(device)
    profiler.end("ace_step_dit_perf_pass")
    if flush_every > 0:
        _ace_step_flush_device_profiler(device)
    _ace_step_flush_device_profiler(device)

    profiler.print()
    logger.info(
        "AceStep DiT Tracy harness (mode=denoise, steps={}, do_cfg={}): perf_pass={:.3f}s",
        infer_steps,
        do_cfg,
        profiler.get("ace_step_dit_perf_pass"),
    )

    budget = os.environ.get("ACE_STEP_PERF_MAX_SECONDS")
    if budget:
        assert profiler.get("ace_step_dit_perf_pass") <= float(
            budget
        ), f"ace_step_dit_perf_pass exceeds ACE_STEP_PERF_MAX_SECONDS={budget}s"
