# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Always-on trace+2CQ performant runner for ACE-Step v1.5.

Mirrors ``models/experimental/swin_v2/runner/performant_runner.py``:

- Construction takes a device handle + model config, builds the TTNN model, and
  immediately runs a warmup ``generate`` so the DiT body trace is captured before
  the first user-visible ``run(prompt)`` call.
- ``run(prompt)`` replays the captured trace via 2 command queues (host->device
  copies on CQ 1, ``execute_trace`` on CQ 0). Steady-state cost is text encoder +
  condition encoder + SDPA mask build + N traced DiT replays + VAE decode.
- ``release()`` releases the captured trace id + persistent DiT buffers.

Unlike :mod:`models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt` with ``use_trace=False``,
this runner always constructs :class:`AceStepE2EModel` with trace enabled (the default).

Device requirements (same as the SwinV2 runner):

- 2 command queues (``ttnn.open_device(num_command_queues=2, ...)``)
- ``trace_region_size`` large enough for the DiT body capture
  (the demo conftest under ``perf/`` defaults to 128 MB which is sufficient)

The ``perf/conftest.py`` ``device`` fixture already satisfies both.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import torch

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import AceStepE2EModel, E2EConfig


class AceStepPerformantRunner:
    """SwinV2-style perf runner wrapping :class:`AceStepE2EModel`.

    Construction:
        1. Builds :class:`AceStepE2EModel` with ``use_trace=True`` — uploads weights, prepares convs,
           precomputes per-step time embeddings, caches the silence-context.
        3. Runs one warmup ``generate(warmup_prompt)`` to:
              - drive the first two eager Euler steps (program-cache fill)
              - capture the DiT body trace (``begin_trace_capture`` /
                ``end_trace_capture``)
              - leave the trace id released at end-of-loop so the next
                ``run(prompt)`` re-arms it via :meth:`_E2EDenoiseTrace.recapture`
                against the same persistent buffers.

    Steady-state ``run(prompt)``:
        - text encoder (skipped on cache hit for the same prompt + duration)
        - condition encoder (same caching)
        - SDPA mask build (cached per-prompt inside the pipeline)
        - prime persistent trace buffers with this prompt's enc/ctx/mask
        - recapture trace id, run N traced DiT replays
        - VAE decode (tiled)

    Cleanup via :meth:`release` frees the captured trace id and every persistent
    buffer on ``_trace_state``. Device lifecycle remains owned by the caller (the
    runner never calls ``ttnn.close_device``).

    Example
    -------

    ::

        config = E2EConfig(
            checkpoint_safetensors_path=str(dit_safetensors),
            vae_dir=str(vae_dir),
            text_model_dir=str(qwen_dir),
            silence_latent_path=str(silence_pt),
            duration_sec=1.0,
            infer_steps=8,
            guidance_scale=1.0,
            use_adg=False,
        )
        runner = AceStepPerformantRunner(device, config)
        try:
            for prompt in prompts:
                wav = runner.run(prompt)
                ...
        finally:
            runner.release()
    """

    _DEFAULT_WARMUP_PROMPT = "Instrumental warmup prompt: deep bass kick drum, bright synth lead, electronic dance."

    def __init__(
        self,
        device,
        config: E2EConfig,
        *,
        warmup_prompt: Optional[str] = None,
    ) -> None:
        if not hasattr(ttnn, "begin_trace_capture") or not hasattr(ttnn, "execute_trace"):
            raise RuntimeError(
                "AceStepPerformantRunner requires a TTNN build with begin_trace_capture / "
                "execute_trace. Rebuild TTNN with trace support enabled."
            )
        self.device = device
        self.config = config
        self._warmup_prompt = warmup_prompt or self._DEFAULT_WARMUP_PROMPT

        # Build the model (weight upload + conv prep + per-step temb precompute) and
        # immediately capture the DiT trace via a warmup generate. The capture happens
        # lazily inside ``run_ttnn_denoise_loop`` after two eager Euler steps; by the
        # time ``__init__`` returns, ``self.model._trace_state.has_buffers()`` is True
        # and subsequent ``run(prompt)`` calls only pay for ``recapture`` (no fresh
        # buffer clones, no two-step warmup).
        self.model: Optional[AceStepE2EModel] = AceStepE2EModel(config, device, use_trace=True)
        self._warmup_seconds: float = 0.0
        self._capture_dit_trace()

    def _capture_dit_trace(self) -> None:
        """Run one warmup ``generate`` so the DiT body trace + persistent buffers exist.

        Mirrors :meth:`SwinV2PerformantRunner._capture_swinv2_trace_2cqs` — the SwinV2
        runner does its own warmup forwards before ``begin_trace_capture``; for
        ACE-Step that bookkeeping lives inside :func:`run_ttnn_denoise_loop`, so we
        just need to drive one full ``generate`` to make sure the capture has run.
        """
        assert self.model is not None
        t0 = time.perf_counter()
        _ = self.model.generate(self._warmup_prompt)
        ttnn.synchronize_device(self.device)
        self._warmup_seconds = float(time.perf_counter() - t0)

    @property
    def warmup_seconds(self) -> float:
        """Wall-clock seconds spent on the warmup ``generate`` (compile + capture)."""
        return self._warmup_seconds

    def run(
        self,
        prompt: str,
        *,
        return_waveform_ttnn: bool = False,
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        """Generate one waveform for *prompt*, reusing the captured DiT trace.

        Args:
            prompt: caption to condition the DiT on. Repeated prompts hit the
                model's per-prompt (text encoder + condition encoder) cache.
            return_waveform_ttnn: when True, return the raw TTNN waveform tensor
                from the VAE without host peak-normalization (saves one device->host
                copy and the host RMS pass).

        Returns:
            ``[1, channels, samples]`` ``torch.Tensor`` normalized to ``[-1, 1]`` by
            default, or an on-device ``ttnn.Tensor`` when *return_waveform_ttnn* is
            True.
        """
        if self.model is None:
            raise RuntimeError("AceStepPerformantRunner.run called after release().")
        return self.model.generate(prompt, return_waveform_ttnn=return_waveform_ttnn)

    def release(self) -> None:
        """Free the captured trace id + persistent DiT buffers; safe to call repeatedly.

        Does NOT close the device — the runner's caller (typically a pytest fixture
        or the demo script) owns device lifecycle. After ``release()`` the runner
        is unusable; further ``run()`` calls raise ``RuntimeError``.
        """
        if self.model is None:
            return
        ts = getattr(self.model, "_trace_state", None)
        if ts is not None:
            try:
                ts.release(self.device)
            except Exception:
                # Best-effort cleanup; device fixture teardown will still close cleanly.
                pass
        # Drop the model reference so its TTNN tensors can be garbage-collected before
        # the device fixture closes the device.
        self.model = None


__all__ = ["AceStepPerformantRunner"]
