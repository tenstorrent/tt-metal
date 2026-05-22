# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-pass demo session for ``run_prompt_to_wav.py``.

Supports ``--warmup`` (compile-cache pass + timed pass in one process),
``--repeat N``, and ``--serve`` without reloading handlers or DiT/VAE each time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List

_DEFAULT_WARMUP_PROMPT = "Instrumental warmup: deep bass kick drum, bright synth lead, electronic dance music."


@dataclass
class CachedPreprocess:
    """Host-side condition tensors reused when prompt/duration/seed match."""

    prompt: str
    duration_sec: float
    seed: int
    frames: int
    enc_hs: Any
    enc_mask: Any
    ctx_lat: Any
    null_emb: Any

    def matches(self, *, prompt: str, duration_sec: float, seed: int) -> bool:
        return (
            self.prompt == str(prompt)
            and float(self.duration_sec) == float(duration_sec)
            and int(self.seed) == int(seed)
        )


@dataclass
class DemoRunSpec:
    """One generation pass inside a :class:`AceStepDemoSession`."""

    prompt: str
    out_path: Path | None
    is_warmup: bool = False
    record_perf: bool = True
    summary_label: str = "demo_total"


@dataclass
class AceStepDemoSession:
    """Persistent TTNN objects reused across passes in the same process."""

    preprocess_dev: Any = None
    dit_dev: Any = None
    dit_handler: Any = None
    llm_handler: Any = None
    qwen_tt_encoder: Any = None
    audio_code_detokenizer: Any = None
    condition_encoder: Any = None
    pipe: Any = None
    tt_vae: Any = None
    trace_state: Any = None
    dit_frames: int | None = None
    run_specs: list = field(default_factory=list)
    cached_preprocess: CachedPreprocess | None = None
    session_perf: Any = None
    _pass_count: int = 0

    def __post_init__(self) -> None:
        if self.session_perf is None:
            from models.demos.ace_step_v1_5.ace_step_perf_log import SessionPerfState

            self.session_perf = SessionPerfState()

    @property
    def multi_pass(self) -> bool:
        return self._pass_count > 0

    def mark_pass_started(self) -> None:
        self._pass_count += 1

    def can_reuse_preprocess(self, *, prompt: str, duration_sec: float, seed: int) -> bool:
        cached = self.cached_preprocess
        return cached is not None and cached.matches(prompt=prompt, duration_sec=duration_sec, seed=seed)

    def should_keep_dit_mesh_open(
        self,
        *,
        run_specs: list,
        session_pass: int,
        duration_sec: float,
        seed: int,
        force_close: bool = False,
    ) -> bool:
        """Keep the DiT mesh open between passes when the next pass can skip preprocess.

        Preprocess needs a separate 1×1 device; the DiT 2×2 mesh must be closed before opening it.
        When the next pass reuses cached host condition tensors, we can keep pipe/VAE/trace alive
        and avoid ~2 s ``dit_pipeline_init`` plus trace recapture on the next pass.
        """
        if force_close or self.dit_dev is None:
            return False
        next_idx = int(session_pass) + 1
        if next_idx >= len(run_specs):
            return False
        next_spec = run_specs[next_idx]
        return self.can_reuse_preprocess(
            prompt=str(next_spec.prompt),
            duration_sec=float(duration_sec),
            seed=int(seed),
        )

    def store_preprocess(
        self,
        *,
        prompt: str,
        duration_sec: float,
        seed: int,
        frames: int,
        enc_hs: Any,
        enc_mask: Any,
        ctx_lat: Any,
        null_emb: Any,
    ) -> None:
        self.cached_preprocess = CachedPreprocess(
            prompt=str(prompt),
            duration_sec=float(duration_sec),
            seed=int(seed),
            frames=int(frames),
            enc_hs=enc_hs,
            enc_mask=enc_mask,
            ctx_lat=ctx_lat,
            null_emb=null_emb,
        )

    def close_dit_device(self, ttnn_mod: Any) -> None:
        """Close the DiT mesh between session passes (preprocess uses a separate 1×1 device)."""
        from models.demos.ace_step_v1_5.tt_device import close_ace_step_device

        if self.trace_state is not None and self.dit_dev is not None:
            try:
                self.trace_state.release(self.dit_dev)
            except Exception:
                pass
        self.trace_state = None
        self.pipe = None
        self.tt_vae = None
        if self.dit_dev is not None:
            try:
                close_ace_step_device(ttnn_mod, self.dit_dev)
            except Exception:
                pass
        self.dit_dev = None

    def clear_preprocess_device(self, ttnn_mod: Any) -> None:
        from models.demos.ace_step_v1_5.tt_device import close_ace_step_device

        if self.preprocess_dev is not None:
            try:
                close_ace_step_device(ttnn_mod, self.preprocess_dev)
            except Exception:
                pass
        self.preprocess_dev = None
        self.qwen_tt_encoder = None
        self.audio_code_detokenizer = None
        self.condition_encoder = None

    def release(self, ttnn_mod: Any) -> None:
        from models.demos.ace_step_v1_5.tt_device import close_ace_step_device

        if self.trace_state is not None:
            dev = self.dit_dev or self.preprocess_dev
            if dev is not None:
                try:
                    self.trace_state.release(dev)
                except Exception:
                    pass
            self.trace_state = None

        for dev in (self.dit_dev, self.preprocess_dev):
            if dev is not None:
                try:
                    close_ace_step_device(ttnn_mod, dev)
                except Exception:
                    pass
        self.dit_dev = None
        self.preprocess_dev = None
        self.pipe = None
        self.tt_vae = None
        self.qwen_tt_encoder = None
        self.audio_code_detokenizer = None
        self.condition_encoder = None


def build_demo_run_specs(args) -> List[DemoRunSpec]:
    """Build ordered passes for a single-process demo run."""
    specs: List[DemoRunSpec] = []
    main_prompt = str(args.prompt or "").strip()
    if not main_prompt and not bool(getattr(args, "serve", False)):
        raise ValueError("--prompt is required unless --serve is set")

    if bool(getattr(args, "warmup", False)):
        warmup_prompt = getattr(args, "warmup_prompt", None) or main_prompt or _DEFAULT_WARMUP_PROMPT
        specs.append(
            DemoRunSpec(
                prompt=str(warmup_prompt),
                out_path=None,
                is_warmup=True,
                record_perf=bool(getattr(args, "warmup_perf", False)),
                summary_label="warmup_total",
            )
        )

    if not bool(getattr(args, "serve", False)):
        if not main_prompt:
            raise ValueError("--prompt is required for non-serve runs")
        out = Path(args.out)
        specs.append(
            DemoRunSpec(
                prompt=main_prompt,
                out_path=out,
                is_warmup=False,
                record_perf=True,
                summary_label="demo_total",
            )
        )
        repeat = max(1, int(getattr(args, "repeat", 1)))
        stem = out.stem
        suffix = out.suffix or ".wav"
        for idx in range(1, repeat):
            repeat_out = out.with_name(f"{stem}_{idx + 1}{suffix}")
            specs.append(
                DemoRunSpec(
                    prompt=main_prompt,
                    out_path=repeat_out,
                    is_warmup=False,
                    record_perf=True,
                    summary_label=f"demo_total_{idx + 1}",
                )
            )
    return specs


def serve_prompt_loop(*, default_prompt: str | None = None) -> Iterator[DemoRunSpec]:
    """Yield :class:`DemoRunSpec` instances from stdin until EOF or empty line."""
    idx = 0
    while True:
        try:
            line = input("[ace_step_v1_5] prompt> ").strip()
        except EOFError:
            break
        if not line:
            break
        idx += 1
        yield DemoRunSpec(
            prompt=line,
            out_path=None,
            is_warmup=False,
            record_perf=True,
            summary_label=f"serve_{idx}",
        )


__all__ = [
    "AceStepDemoSession",
    "CachedPreprocess",
    "DemoRunSpec",
    "build_demo_run_specs",
    "serve_prompt_loop",
]
