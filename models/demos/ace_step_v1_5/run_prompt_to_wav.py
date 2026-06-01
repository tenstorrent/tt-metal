"""
ACE-Step v1.5 demo: official-style host preprocessing + TTNN DiT sampler + VAE decode.

Latent diffusion runs on TTNN; VAE decode uses the TTNN Oobleck port from ``ign/ACE_perf`` by default
(HF-style ``ckpt_dir/vae/`` with ``config.json``). Long latent sequences are decoded in overlapping
time tiles so ``ttnn.conv1d`` stays within L1 limits. Pass ``--torch-vae`` for PyTorch ``AutoencoderOobleck``.

By default this matches ``torch_ref/run_prompt_to_wav.py --use-official-acestep`` for **Phase 1**
(5 Hz LM / CoT, audio codes, handler ``preprocess_batch``, TTNN Qwen3 caption encoder via
``infer_text_embeddings``, and TTNN ``prepare_condition`` replacement with **precomputed LM hints**),
emitting the same style of **loguru** / model logs as the official CLI. DiT sampling runs on TTNN.

With the default ``--use-trace`` flag, traceable stages use TTNN trace + 2CQ replay: Qwen3
caption, lyric embed, audio-code detokenizer, 5 Hz LM prefill + decode,
handler ``forward_payload_traced`` (lyric 8L + timbre 4L + text + concat), and DiT
``_E2EDenoiseTrace`` body per step (APG/ADG + Euler stay eager after ``release_trace_only``).
DiT CFG enc/ctx concat and VAE decode stay eager (trace replay was not bit-accurate → noise).
Pass ``--no-use-trace`` to disable all traces (single CQ, fully eager).

- **5 Hz LM (`acestep-5Hz-lm-1.7B` by default)**: TTNN causal LM
  (``ttnn_impl/five_hz_causal_lm_experimental.py`` →
  ``ttnn_impl/qwen_tt_transformers_lm.QwenModelTtTransformers``).
  The whole decoder (Embedding, ``Attention`` with fused QKV / paged SDPA, ``MLP``,
  ``DistributedNorm(RMSNorm)``, ``HfRotarySetup``, ``LMHead``) is built from the stock
  ``models/tt_transformers`` graph via
  :func:`models.tt_transformers.tt.common.create_tt_model`. Use ``--pytorch-lm`` to fall back
  to host PyTorch HF Qwen 1.7B forward instead.


``--use-official-lm`` runs full ``acestep.inference.generate_music`` (PyTorch DiT on host) with no TTNN.

Requires **torchaudio** (for ``AceStepHandler`` / 5 Hz LM preprocessing). After the TTNN device is opened for the Qwen3 caption
encoder, the same device is attached to the 5 Hz LM handler so the **CFG logit combine**
(``uncond + cfg_scale * (cond - uncond)``) runs on TTNN with strict fallbacks disabled inside that op:
valid-audio **slice** in codes phase when a mask is applied, otherwise **full vocabulary**; see
``ttnn_impl/lm_logits_ttnn.py``. The 5 Hz causal LM forward uses TTNN by default
(``ttnn_impl/five_hz_causal_lm_experimental.py``). Pass ``--pytorch-lm`` for host HF only.
"""


from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

# Turbo discrete timesteps (aligned with ACE-Step turbo modeling).
_VALID_TIMESTEPS = [
    1.0,
    0.9545454545454546,
    0.9333333333333333,
    0.9,
    0.875,
    0.8571428571428571,
    0.8333333333333334,
    0.7692307692307693,
    0.75,
    0.6666666666666666,
    0.6428571428571429,
    0.625,
    0.5454545454545454,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.2222222222222222,
    0.125,
]

_SHIFT_TIMESTEPS: dict[float, list[float]] = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [
        1.0,
        0.9333333333333333,
        0.8571428571428571,
        0.7692307692307693,
        0.6666666666666666,
        0.5454545454545454,
        0.4,
        0.2222222222222222,
    ],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
}


def _prepend_ttnn_pkg_to_syspath() -> None:
    tt_metal_root = str(Path(__file__).resolve().parents[3])
    ttnn_pkg_root = str(Path(tt_metal_root) / "ttnn")
    for p in (tt_metal_root, ttnn_pkg_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _configure_ttnn_runtime() -> None:
    """Insert tt-metal roots and strict TTNN fallback (throw on silent CPU fallback)."""
    _prepend_ttnn_pkg_to_syspath()
    os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'


def _require_torchaudio() -> None:
    """ACE-Step handler preprocessing needs torchaudio; fail fast with install instructions."""
    import importlib.util

    if importlib.util.find_spec("torchaudio") is not None:
        return
    raise RuntimeError(
        "torchaudio is required for ACE-Step v1.5 demo preprocessing (5 Hz LM + AceStepHandler).\n"
        "Install a build that matches your PyTorch/CUDA version, for example:\n"
        "  pip install torchaudio\n"
        "See https://pytorch.org/get-started/locally/ for the correct wheel index."
    )


def _open_tt_device(ttnn: Any, *, device_id: int, num_command_queues: int = 1) -> Any:
    """Open device with L1 small arena sized for conv/VAE (same default as ``tests/conftest.py`` / ign/ACE_perf).

    Pass ``num_command_queues=2`` to enable the host->device copy on CQ 1 / ``execute_trace``
    on CQ 0 pattern required by ``_E2EDenoiseTrace.replay`` (``--use-trace``).
    """
    open_kwargs = dict(
        device_id=int(device_id),
        l1_small_size=int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304")),
        trace_region_size=128 << 20,
    )
    if int(num_command_queues) > 1:
        open_kwargs["num_command_queues"] = int(num_command_queues)
    return ttnn.open_device(**open_kwargs)


def _finalize_condition_trace_tensors(
    enc_hs_tt: Any,
    condition_encoder: Any | None,
    device: Any,
    *,
    use_trace: bool,
    ctx_tt: Any | None = None,
) -> tuple[Any, Any | None]:
    """Clone traced ``enc`` / ``ctx`` and release condition traces before DiT."""
    if not use_trace or condition_encoder is None:
        return enc_hs_tt, ctx_tt
    import ttnn

    ttnn.synchronize_device(device)
    if not hasattr(ttnn, "clone"):
        raise RuntimeError("ttnn.clone is required when --use-trace is on (condition encoder trace).")
    enc_owned = ttnn.clone(enc_hs_tt)
    ctx_owned = ttnn.clone(ctx_tt) if ctx_tt is not None else None
    if hasattr(condition_encoder, "release_trace"):
        condition_encoder.release_trace()
    return enc_owned, ctx_owned


def _build_t_schedule(*, infer_steps: int, variant: str) -> list[float]:
    """Build diffusion timestep schedule (shift=1.0; turbo uses discrete tables)."""
    variant_l = (variant or "").lower()
    is_turbo = "turbo" in variant_l
    shift = 1.0

    infer_steps = int(infer_steps)
    if infer_steps <= 1:
        raise ValueError("--infer_steps must be >= 2")

    if is_turbo:
        s = min(_SHIFT_TIMESTEPS.keys(), key=lambda v: abs(v - float(shift)))
        if infer_steps == 8:
            return list(_SHIFT_TIMESTEPS[float(s)])
        lin = [1.0 - (i / float(infer_steps - 1)) for i in range(infer_steps)]
        mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in lin]
        out = []
        for t in mapped:
            if not out or out[-1] != t:
                out.append(t)
        return sorted(out, reverse=True)

    t = [1.0 - (i / float(infer_steps)) for i in range(infer_steps)]
    if float(shift) != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


# Vendored ACE-Step copy bundled with the demo so it runs without an external clone of
# https://github.com/ace-step/ACE-Step-1.5. ``_resolve_ace_step_repo_root`` prefers this
# path (after explicit CLI / env overrides) so the demo is repo-independent by default.
_VENDORED_ACESTEP_ROOT = Path(__file__).resolve().parent / "torch_ref" / "_vendored_acestep"


_WELL_KNOWN_REPO_ROOTS = [
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
    Path.home() / "ACE-Step-1.5",
    Path("/opt") / "ACE-Step-1.5",
]


def _resolve_ace_step_repo_root(*, ckpt_dir: str | None, ace_step_repo_root: str | None) -> Path | None:
    """Return a directory containing an ``acestep/`` package.

    Search order:

    1. ``ACE_STEP_REPO_ROOT`` env var (explicit override).
    3. **Vendored copy** at ``models/demos/ace_step_v1_5/torch_ref/_vendored_acestep/`` —
       default; lets the demo run with no external clone of ACE-Step-1.5.
    4. Walk up from ``ckpt_dir`` (looks for an ``acestep/`` sibling).
    5. Well-known external paths (``~/proj_sdk/ACE-Step-1.5``, ``~/ACE-Step-1.5``,
       ``/opt/ACE-Step-1.5``).
    """

    candidates: list[Path] = []
    if ace_step_repo_root:
        candidates.append(Path(ace_step_repo_root).expanduser().resolve())
    env = os.environ.get("ACE_STEP_REPO_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(_VENDORED_ACESTEP_ROOT)

    if ckpt_dir:
        cur = Path(ckpt_dir).expanduser().resolve()
        for _ in range(8):
            candidates.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    candidates.extend(_WELL_KNOWN_REPO_ROOTS)
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if (c / "acestep" / "__init__.py").is_file():
            return c
    return None


def _normalize_wav_for_save(wav: "torch.Tensor") -> "torch.Tensor":
    """Peak-normalize without boosting a weak/noisy signal floor."""
    import torch

    peak = wav.abs().amax(dim=(1, 2), keepdim=True)
    if float(peak.max()) < 0.02:
        return wav.clamp(-1.0, 1.0)
    rms = wav.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
    denom = torch.maximum(peak, rms * 4.0).clamp(min=1e-2)
    return (wav / denom).clamp(-1.0, 1.0)


def _log_duration_preprocess_health(
    payload: dict[str, Any],
    *,
    duration_sec: float,
    frames: int,
    audio_code_string: str = "",
) -> None:
    """Log LM code count vs duration — short code streams pad with silence and sound noisy."""
    from models.demos.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import parse_audio_code_string

    expected_codes = max(1, int(round(float(duration_sec) * 5.0)))
    n_codes = len(parse_audio_code_string(str(audio_code_string or "")))
    hints = payload.get("precomputed_lm_hints_25Hz")
    hint_t = int(hints.shape[1]) if hints is not None and hasattr(hints, "shape") else 0
    print(
        f"[ace_step_v1_5] duration health: {float(duration_sec):g}s latent_frames={int(frames)} "
        f"lm_audio_codes={n_codes} (expect~{expected_codes}) lm_hint_T={hint_t}",
        flush=True,
    )
    if n_codes > 0 and n_codes + 2 < expected_codes:
        print(
            f"[ace_step_v1_5] WARNING: LM produced only {n_codes}/{expected_codes} audio codes; "
            "hint tail is silence-padded → expect muddy/noisy audio after "
            f"~{n_codes / 5.0:.1f}s",
            flush=True,
        )


def _configure_vae_quality(*, frames: int, mesh_sku: str | None, duration_sec: float, clarity: bool = False) -> None:
    """Use BF16 VAE compute/weights for long mesh clips (≥30 s with ``--clarity``, else ≥40 s)."""
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_vae_quality_decode_enabled

    os.environ["ACE_STEP_VAE_LATENT_FRAMES"] = str(int(frames))
    os.environ["ACE_STEP_VAE_DURATION_SEC"] = str(float(duration_sec))
    if bool(clarity):
        os.environ["ACE_STEP_VAE_CLARITY"] = "1"
    else:
        os.environ.pop("ACE_STEP_VAE_CLARITY", None)
    if mesh_sku is not None:
        os.environ["ACE_STEP_VAE_MESH_SKU"] = str(mesh_sku)
    if not ace_step_vae_quality_decode_enabled(
        latent_frames=int(frames),
        mesh_sku=mesh_sku,
        duration_sec=float(duration_sec),
        clarity_mode=bool(clarity),
    ):
        return
    # Force (not setdefault): BFP8 conv weights are chosen at VAE module init time.
    os.environ["ACE_STEP_VAE_BFLOAT8_ACTIVATIONS"] = "0"
    os.environ["ACE_STEP_VAE_BF16_CONV_WEIGHTS"] = "1"
    print(
        "[ace_step_v1_5] VAE quality mode (long mesh clip): BF16 activation compute + BF16 k>1 conv weights",
        flush=True,
    )


def _audio_to_numpy_frames_channels(wav: "torch.Tensor") -> np.ndarray:
    """Return ``[num_frames, num_channels]`` float32 numpy from ``[C,T]`` or ``[T,C]``."""
    if wav.ndim != 2:
        raise ValueError(f"Expected rank-2 audio, got shape {tuple(wav.shape)}")
    a, b = int(wav.shape[0]), int(wav.shape[1])
    if a <= 16 and b > a:
        arr = wav.transpose(0, 1).contiguous()
    elif b <= 16 and a > b:
        arr = wav.contiguous()
    elif a <= b and a <= 16:
        arr = wav.transpose(0, 1).contiguous()
    elif b <= a and b <= 16:
        arr = wav.contiguous()
    else:
        raise ValueError(f"Ambiguous audio layout {tuple(wav.shape)}; expected [C,T] or [T,C] with C<=16 and T>>C")
    audio = np.ascontiguousarray(arr.numpy(), dtype=np.float32)
    if int(audio.shape[1]) > 16:
        raise ValueError(f"Refusing to write audio with {int(audio.shape[1])} channels (layout {tuple(wav.shape)})")
    return audio


def _save_wav_fallback(wav: Any, out_path: Path, sample_rate: int = 48000) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.is_dir():
        raise ValueError(f"Output path is a directory, not a file: {out_path}")

    import torch

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu()
    else:
        wav = torch.as_tensor(wav, dtype=torch.float32).cpu()

    if wav.ndim == 1:
        audio = wav.numpy()
    elif wav.ndim == 2:
        audio = _audio_to_numpy_frames_channels(wav)
    else:
        raise ValueError(f"Expected wav rank 1 or 2, got shape {tuple(wav.shape)}")

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Refusing to write empty audio to {out_path}")
    if not np.isfinite(audio).all():
        bad = int((~np.isfinite(audio)).sum())
        print(f"[ace_step_v1_5] warning: replacing {bad} non-finite audio samples before write", flush=True)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if audio.ndim == 2 and int(audio.shape[1]) == 1:
        audio = audio[:, 0]

    if out_path.exists():
        try:
            out_path.unlink()
        except OSError:
            pass

    try:
        import soundfile as sf  # type: ignore

        sf.write(str(out_path), audio, samplerate=int(sample_rate), subtype="PCM_16")
        return
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        print(f"[ace_step_v1_5] soundfile write failed ({exc!r}); falling back to scipy", flush=True)

    from scipy.io import wavfile  # type: ignore

    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), int(sample_rate), audio_i16)


_DEFAULT_CKPT_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints"

_HF_REPO_MAP = {
    "acestep-v15-base": ("ACE-Step/acestep-v15-base", False),
    "acestep-v15-sft": ("ACE-Step/acestep-v15-sft", False),
    "acestep-v15-turbo": ("ACE-Step/Ace-Step1.5", True),
    "acestep-5Hz-lm-0.6B": ("ACE-Step/acestep-5Hz-lm-0.6B", False),
    "acestep-5Hz-lm-1.7B": ("ACE-Step/Ace-Step1.5", True),
    "acestep-5Hz-lm-4B": ("ACE-Step/acestep-5Hz-lm-4B", False),
    "vae": ("ACE-Step/Ace-Step1.5", True),
    "Qwen3-Embedding-0.6B": ("ACE-Step/Ace-Step1.5", True),
}


def _ensure_variant(name: str, ckpt_dir: Path) -> Path:
    """Return the local path for *name* under *ckpt_dir*, downloading from
    HuggingFace on first use.  Files are stored under *ckpt_dir/<name>/*."""
    local = ckpt_dir / name
    has_weights = any(local.glob("*.safetensors")) or any(local.glob("*.pt"))
    if has_weights:
        return local

    entry = _HF_REPO_MAP.get(name)
    if entry is None:
        raise FileNotFoundError(
            f"No HuggingFace repo mapping for variant '{name}'. " f"Known variants: {list(_HF_REPO_MAP.keys())}"
        )
    repo_id, is_subfolder = entry
    from huggingface_hub import snapshot_download

    print(f"[ace_step_v1_5] Downloading {name} from {repo_id} ...", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if is_subfolder:
        snapshot_download(
            repo_id,
            allow_patterns=f"{name}/*",
            local_dir=str(ckpt_dir),
        )
    else:
        snapshot_download(repo_id, local_dir=str(local))
    if not any(local.glob("*.safetensors")) and not any(local.glob("*.pt")):
        raise FileNotFoundError(f"Download succeeded but no weights found in {local}")
    print(f"[ace_step_v1_5] {name} ready at {local}", flush=True)
    return local


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ACE-Step v1.5: HF preprocessing + TTNN DiT + TTNN or PyTorch VAE decode.",
    )
    ap.add_argument("--prompt", type=str, required=True, help="Caption / text prompt.")
    ap.add_argument(
        "--variant",
        type=str,
        default="acestep-v15-base",
        choices=["acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo"],
        help="DiT model variant (default: acestep-v15-base).",
    )
    ap.add_argument(
        "--lm_variant",
        type=str,
        default="acestep-5Hz-lm-1.7B",
        choices=["acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"],
        help="5 Hz LM variant (default: acestep-5Hz-lm-1.7B).",
    )
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument(
        "--mesh-device",
        type=str,
        default=None,
        help=(
            "Target mesh SKU for DiT/VAE (e.g. P150, BH_QB, BH_LB). "
            "Also read from ACE_STEP_MESH_DEVICE or MESH_DEVICE. "
            "Multi-device SKUs run preprocess on host CPU and DiT/VAE on the full mesh."
        ),
    )
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--infer_steps", type=int, default=None, help="Default: 8 turbo, 50 base.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help=(
            "DiT CFG strength. Default: 1 for turbo variants, 7 for base/sft. "
            "Independent of the 5 Hz LM backend (TTNN LM forces lm_cfg_scale=1). "
            "Set 1 to disable DiT CFG."
        ),
    )
    ap.add_argument("--out", type=str, default="ttnn_out.wav")
    ap.add_argument(
        "--use-official-lm",
        action="store_true",
        help="Run full official generate_music (LLM+handlers, CPU). Does not use TTNN; writes --out for A/B.",
    )
    ap.add_argument(
        "--pytorch-lm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load the 5 Hz LM with host PyTorch ``AutoModelForCausalLM`` instead of the default "
            "TTNN causal stack (``ttnn_impl/five_hz_causal_lm_experimental.py`` -> "
            "``ttnn_impl/qwen_tt_transformers_lm.QwenModelTtTransformers``). Default: TTNN. "
            "TTNN LM opens the device before LM init and forces ``lm_cfg_scale=1`` "
            "(``max_batch_size=1``). DiT ``--guidance_scale`` is unchanged."
        ),
    )
    ap.add_argument(
        "--torch-vae",
        action="store_true",
        help=(
            "Decode latents with PyTorch Diffusers AutoencoderOobleck on GPU/CPU. "
            "Default: TTNN Oobleck decoder (requires ~/.cache/.../ACE-Step-1.5-checkpoints/vae/config.json + weights)."
        ),
    )
    ap.add_argument(
        "--use-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap the Qwen3 caption encoder (``forward_traced``), 5 Hz LM prefill+decode "
            "(when not using ``--pytorch-lm``), the instrumental condition "
            "encoder (``forward_payload_traced`` on the handler path), and the per-step "
            "DiT body (patch_embed + DiT core + output_head) in captured TTNN trace + 2CQ replay. "
            "DiT tracing delegates to ``run_ttnn_denoise_loop`` with ``_E2EDenoiseTrace`` "
            "(``ttnn_impl/e2e_model_tt.py``). Default: on. Opens the TTNN device with "
            "``num_command_queues=2`` so host->device copies on CQ 1 can overlap "
            "``execute_trace`` on CQ 0. DiT trace is captured lazily after two eager Euler "
            "steps; Qwen3 and condition traces are captured on the first encode. Use "
            "``--no-use-trace`` "
            "for the legacy single-CQ eager path (A/B vs. trace, or Tracy device profiling, "
            "which is mutually exclusive with trace)."
        ),
    )
    ap.add_argument(
        "--clarity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "BH mesh quality preset: ADG guidance (default on mesh when set), wider TTNN VAE "
            "tile overlap for long clips. Single-chip runs ignore mesh-specific parts."
        ),
    )
    ap.add_argument(
        "--use-adg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "DiT CFG guidance: ADG (default on single-chip base/sft) vs APG (default on mesh). "
            "``--clarity`` on mesh defaults to ADG unless overridden."
        ),
    )
    ap.add_argument(
        "--llm-debug",
        action="store_true",
        help=(
            "Deep LLM perf logging: show Tokens/sec in KEY METRICS and enable constrained-decoding debug logs "
            "(ACE-Step BENCHMARK.md --llm-debug)."
        ),
    )
    args = ap.parse_args()

    from models.demos.ace_step_v1_5.official_lm_preprocess import configure_acestep_logging

    configure_acestep_logging(level="INFO")

    _require_torchaudio()
    use_ttnn_5hz_lm = not bool(args.pytorch_lm)

    import torch

    ckpt_dir = _DEFAULT_CKPT_DIR.expanduser().resolve()
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt_dir)

    model_dir = _ensure_variant(args.variant, ckpt_dir)
    _ensure_variant("vae", ckpt_dir)
    _ensure_variant("Qwen3-Embedding-0.6B", ckpt_dir)
    _ensure_variant(args.lm_variant, ckpt_dir)

    safetensors_path = model_dir / "model.safetensors"
    silence_latent_path = model_dir / "silence_latent.pt"
    vae_dir = ckpt_dir / "vae"
    text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"

    if not safetensors_path.is_file():
        safetensors_shards = sorted(model_dir.glob("model-*.safetensors"))
        if not safetensors_shards:
            raise FileNotFoundError(f"Missing checkpoint: {safetensors_path}")
    if not silence_latent_path.is_file():
        raise FileNotFoundError(f"Missing silence_latent: {silence_latent_path}")

    infer_steps = args.infer_steps
    if infer_steps is None:
        infer_steps = 8 if "turbo" in str(args.variant).lower() else 50

    from models.demos.ace_step_v1_5.tt_device import (
        ace_step_device_num_chips,
        ace_step_dit_pipe_batch_size,
        ace_step_log_mesh_quality_hints,
        ace_step_mesh_perf_log_default,
        ace_step_mesh_use_adg,
        ace_step_mesh_use_host_latent_sampler,
        ace_step_mesh_use_host_temb_precompute,
        ace_step_mesh_use_sequential_cfg,
        ace_step_mesh_use_split_ttnn_preprocess,
        ace_step_needs_split_device,
        ace_step_preprocess_num_command_queues,
        ace_step_resolve_vae_tiling,
        ace_step_synchronize_device,
        ace_step_ttnn_to_torch,
        open_dit_device,
        open_preprocess_device,
        resolve_ace_step_mesh_sku,
        run_mesh_sequential_cfg_forwards,
        transition_preprocess_to_dit_device,
    )

    mesh_sku = resolve_ace_step_mesh_sku(cli_value=args.mesh_device)
    split_device = ace_step_needs_split_device(mesh_sku)

    _predicted_latent_frames = max(1, int(round(float(args.duration_sec) * 25.0)))
    from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_configure_dit_long_clip_quality

    if ace_step_configure_dit_long_clip_quality(
        latent_frames=_predicted_latent_frames,
        duration_sec=float(args.duration_sec),
        mesh_sku=mesh_sku,
    ) and bool(args.use_trace):
        print(
            "[ace_step_v1_5] long clip (>=30s on mesh): forcing --no-use-trace "
            "(DiT body trace is not bit-accurate at patch_seq>=375)",
            flush=True,
        )
        args.use_trace = False

    perf_log_enabled = ace_step_mesh_perf_log_default(mesh_sku=mesh_sku)
    vae_chunk_latents = 32
    vae_overlap_latents = 4

    cfg_interval_start = 0.0
    cfg_interval_end = 1.0
    if bool(args.clarity) and split_device:
        vae_overlap_latents = 12
        print(
            "[ace_step_v1_5] --clarity: mesh ADG (default), VAE overlap≥12, BF16 VAE at ≥30 s",
            flush=True,
        )
    host_only_preprocess = bool(split_device)
    mesh_ttnn_preprocess = host_only_preprocess and ace_step_mesh_use_split_ttnn_preprocess(mesh_sku)
    dev: Any = None
    dev_opened_for_ttnn_text_encoder = False

    if mesh_sku is not None:
        print(f"[ace_step_v1_5] mesh SKU={mesh_sku} split_preprocess={split_device}", flush=True)
    if host_only_preprocess:
        if mesh_ttnn_preprocess:
            print(
                "[ace_step_v1_5] multi-device mesh: split TTNN preprocess on 1×1 "
                "(5 Hz LM + Qwen + condition encoder), then DiT/VAE on full mesh.",
                flush=True,
            )
        else:
            print(
                "[ace_step_v1_5] multi-device mesh: host PyTorch preprocess (DiT/VAE on full mesh after preprocess).",
                flush=True,
            )
        if use_ttnn_5hz_lm and not mesh_ttnn_preprocess:
            print(
                "[ace_step_v1_5] multi-device mesh: forcing --pytorch-lm (host 5 Hz LM on preprocess CPU).",
                flush=True,
            )
            args.pytorch_lm = True
            use_ttnn_5hz_lm = False
    gs = args.guidance_scale
    if gs is None:
        # NOTE: TTNN 5 Hz LM does not force gs=1.0. The LM batch=1 constraint only affects
        # lm_cfg_scale (forced to 1 below); DiT CFG is independent
        # and base/sft are trained for gs=7. Resolve purely from the variant.
        if "turbo" in str(args.variant).lower():
            gs = 1.0
        else:
            gs = 7.0
    gs = float(gs)

    cli_use_adg = args.use_adg
    if bool(args.clarity) and split_device and cli_use_adg is None:
        cli_use_adg = True
    use_adg = ace_step_mesh_use_adg(
        mesh_sku=mesh_sku,
        variant=str(args.variant),
        cli_use_adg=cli_use_adg,
    )

    if mesh_sku is not None:
        ace_step_log_mesh_quality_hints(
            mesh_sku=mesh_sku,
            variant=str(args.variant),
            infer_steps=int(infer_steps),
            guidance_scale=float(gs),
            use_trace=bool(args.use_trace),
            torch_vae=bool(args.torch_vae),
            use_adg=bool(use_adg),
        )

    from models.demos.ace_step_v1_5.ace_step_perf_log import (
        AceStepPerfRecorder,
        emit_session_summary,
        make_denoise_progress_fn,
    )

    perf = AceStepPerfRecorder(
        enabled=perf_log_enabled,
        params={
            "variant": str(args.variant),
            "lm_variant": str(args.lm_variant),
            "duration_sec": float(args.duration_sec),
            "infer_steps": int(infer_steps),
            "guidance_scale": float(gs),
            "use_adg": bool(use_adg),
            "seed": int(args.seed),
            "ttnn_condition": True,
            "torch_vae": bool(args.torch_vae),
            "use_trace": bool(args.use_trace),
            "ttnn_5hz_lm": bool(use_ttnn_5hz_lm),
            "ttnn_5hz_lm_prefill_trace": bool(args.use_trace) and bool(use_ttnn_5hz_lm),
            "ttnn_5hz_lm_decode_trace": bool(args.use_trace) and bool(use_ttnn_5hz_lm),
            "mesh_sku": mesh_sku,
            "split_device": bool(split_device),
            "mesh_ttnn_preprocess": bool(mesh_ttnn_preprocess),
            "session_pass": 0,
            "llm_debug": bool(args.llm_debug),
        },
    )

    from models.demos.ace_step_v1_5.demo_session import AceStepDemoSession

    demo_session = AceStepDemoSession()
    if demo_session.session_perf.session_t0 is None:
        demo_session.session_perf.session_t0 = time.perf_counter()

    perf.begin_run(summary_label="demo_total", record=True)
    run_prompt = str(args.prompt)
    run_out_path = Path(args.out)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_acestep_on_path() -> Path:
        root = _resolve_ace_step_repo_root(ckpt_dir=str(ckpt_dir), ace_step_repo_root=None)
        if root is None:
            raise RuntimeError(
                "Could not find ACE-Step-1.5 'acestep' package. The vendored copy at "
                "models/demos/ace_step_v1_5/torch_ref/_vendored_acestep/acestep/ should normally "
                "be used automatically; if it is missing, set "
                "ACE_STEP_REPO_ROOT to an external checkout."
            )

        from models.demos.ace_step_v1_5.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(root)
        return root

    # --- Optional: full official path (LLM), no TTNN ---
    if args.use_official_lm:
        from models.demos.ace_step_v1_5.torch_ref.transformers_cache_compat import apply_transformers_cache_compat

        apply_transformers_cache_compat()
        ref_root = _ensure_acestep_on_path()
        try:
            from acestep.handler import AceStepHandler
            from acestep.inference import GenerationConfig, GenerationParams, generate_music

            from models.demos.ace_step_v1_5.torch_ref.five_hz_lm import LocalFiveHzLMHandler
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "--use-official-lm requires the upstream ACE-Step ``acestep`` package "
                "(``acestep.handler.AceStepHandler`` + ``acestep.inference.generate_music``).\n"
                f"  Missing module: {e.name!r}.\n"
                "Fix one of:\n"
                "  1. Keep the vendored copy at "
                "models/demos/ace_step_v1_5/torch_ref/_vendored_acestep/ in place "
                "(it ships with this demo by default).\n"
                "  2. Set ACE_STEP_REPO_ROOT to an external clone "
                "of https://github.com/ace-step/ACE-Step-1.5.\n"
                "  3. Run without --use-official-lm (the default TTNN path doesn't need "
                "acestep.inference)."
            ) from e

        # Note: weights are already downloaded by ``_ensure_variant`` earlier in main()
        # via ``huggingface_hub.snapshot_download``; the ACE-Step handler's own download
        # paths (``acestep.model_downloader``) are no-ops by the time they run because
        # every file exists on disk. The historical ``_mdl.MAIN_MODEL_COMPONENTS = [...]``
        # mutation here was informational only (declaring which sub-components live in
        # the main repo); it has been removed because the handler's defaults already
        # cover the same set.

        dit_handler = AceStepHandler()
        llm_handler = LocalFiveHzLMHandler()
        device = "cpu"

        status, ok = dit_handler.initialize_service(
            project_root=str(ref_root),
            config_path=args.variant,
            device=device,
            use_flash_attention=False,
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("AceStepHandler.initialize_service failed")
        _ensure_variant(args.lm_variant, ckpt_dir)
        status, ok = llm_handler.initialize(
            checkpoint_dir=str(ckpt_dir),
            lm_model_path=args.lm_variant,
            backend="pt",
            device=device,
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("5 Hz LM (local HF) initialize failed")
        params = GenerationParams(
            task_type="text2music",
            caption=args.prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            reference_audio=None,
            duration=float(args.duration_sec),
            inference_steps=int(infer_steps),
            thinking=True,
            use_constrained_decoding=True,
            use_adg=use_adg,
            guidance_scale=gs,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=1.0,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(args.seed)], audio_format="wav")
        out_dir = Path(args.out).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=str(out_dir))
        if not result.success:
            raise RuntimeError(result.error or "generate_music failed")
        first = Path(result.audios[0]["path"]).resolve()
        dst = Path(args.out).resolve()
        if first != dst:
            dst.write_bytes(first.read_bytes())
        print(f"Wrote (official LM, not TTNN): {dst}", flush=True)
        return

    condition_tensors_on_device = False
    condition_encoder = None
    enc_hs_tt_one = None
    ctx_tt_one = None
    null_emb_tt = None

    # --- 5 Hz LM + AceStepHandler batching + prepare_condition (precomputed LM hints) ---
    ref_root = _ensure_acestep_on_path()

    from models.demos.ace_step_v1_5.official_lm_preprocess import (
        attach_payload_preprocess_ttnn,
        build_filtered_dit_kwargs_for_handler,
        configure_acestep_logging,
        handler_prepare_condition_payload,
    )

    configure_acestep_logging()
    try:
        from acestep.handler import AceStepHandler

        from models.demos.ace_step_v1_5.ttnn_impl.five_hz_lm import LocalFiveHzLMHandler

    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The default preprocessing path needs the upstream ACE-Step ``acestep`` package "
            "(``acestep.handler.AceStepHandler`` owns ``preprocess_batch`` and "
            "``prepare_condition``).\n"
            f"  Missing module: {e.name!r}.\n"
            "Fix one of:\n"
            "  1. Keep the vendored copy at "
            "models/demos/ace_step_v1_5/torch_ref/_vendored_acestep/ in place "
            "(it ships with this demo by default).\n"
            "  2. Set ACE_STEP_REPO_ROOT to an external clone "
            "of https://github.com/ace-step/ACE-Step-1.5.\n"
            "  3. If e.name == 'torchaudio': pip install torchaudio (match your torch/CUDA "
            "build from pytorch.org; required for handler preprocessing)."
        ) from e

    from models.demos.ace_step_v1_5.acestep_preprocess_shim import GenerationConfig, GenerationParams

    # Note: weights are already downloaded by ``_ensure_variant`` earlier in main()
    # via ``huggingface_hub.snapshot_download``; the historical
    # ``import acestep.model_downloader as _mdl; _mdl.MAIN_MODEL_COMPONENTS = [args.variant,
    # "vae", "Qwen3-Embedding-0.6B", args.lm_variant]`` mutation here was informational
    # only and has been removed. The handler's defaults already cover the same set, and
    # by the time the handler tries to download anything every file exists on disk.

    tt_dev_early = None
    need_preprocess_ttnn_dev = bool(use_ttnn_5hz_lm) or bool(mesh_ttnn_preprocess)
    if need_preprocess_ttnn_dev:
        # TTNN 5 Hz LM uses ``max_batch_size=1`` (see ``QwenModelTtTransformers``). That
        # applies to the LM only (``lm_cfg_scale=1`` below), not DiT CFG.
        _configure_ttnn_runtime()
        import ttnn as _ttnn_pre_lm

        if hasattr(_ttnn_pre_lm, "CONFIG") and hasattr(_ttnn_pre_lm.CONFIG, "throw_exception_on_fallback"):
            _ttnn_pre_lm.CONFIG.throw_exception_on_fallback = True
        if mesh_ttnn_preprocess:
            tt_dev_early = open_preprocess_device(
                _ttnn_pre_lm,
                device_id=int(args.device_id),
                num_command_queues=ace_step_preprocess_num_command_queues(use_trace=bool(args.use_trace)),
            )
        else:
            tt_dev_early = _open_tt_device(
                _ttnn_pre_lm,
                device_id=int(args.device_id),
                num_command_queues=(2 if bool(args.use_trace) else 1),
            )
        if hasattr(tt_dev_early, "enable_program_cache"):
            tt_dev_early.enable_program_cache()
        demo_session.preprocess_dev = tt_dev_early

    dit_handler = AceStepHandler()
    llm_handler = LocalFiveHzLMHandler()

    device = "cpu"
    _init_t0 = time.perf_counter() if perf.enabled else None
    with perf.timed("handler_init"):
        status, ok = dit_handler.initialize_service(
            project_root=str(ref_root),
            config_path=args.variant,
            device=device,
            use_flash_attention=False,
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("AceStepHandler.initialize_service failed")
        status, ok = llm_handler.initialize(
            checkpoint_dir=str(ckpt_dir),
            lm_model_path=args.lm_variant,
            backend="pt",
            device=device,
            ttnn_causal_device=tt_dev_early,
            use_ttnn_causal_lm=bool(use_ttnn_5hz_lm),
            ttnn_lm_prefill_trace=bool(args.use_trace) and bool(use_ttnn_5hz_lm),
            ttnn_lm_decode_trace=bool(args.use_trace) and bool(use_ttnn_5hz_lm),
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("5 Hz LM (local HF) initialize failed")
    if _init_t0 is not None:
        _init_ms = (time.perf_counter() - _init_t0) * 1000.0
        perf.record_init_once("handler_init", _init_ms)
        demo_session.session_perf.note_init("handler_init", _init_ms)
    demo_session.dit_handler = dit_handler
    demo_session.llm_handler = llm_handler

    reuse_preprocess = demo_session.can_reuse_preprocess(
        prompt=run_prompt,
        duration_sec=float(args.duration_sec),
        seed=int(args.seed),
    )
    if reuse_preprocess:
        cached = demo_session.cached_preprocess
        assert cached is not None
        enc_hs = cached.enc_hs
        enc_mask = cached.enc_mask
        ctx_lat = cached.ctx_lat
        null_emb = cached.null_emb
        frames = int(cached.frames)
        perf.set_params(frames=frames)
        _configure_vae_quality(
            frames=frames,
            mesh_sku=mesh_sku,
            duration_sec=float(args.duration_sec),
            clarity=bool(args.clarity),
        )
        condition_tensors_on_device = False
        enc_hs_tt_one = None
        ctx_tt_one = None
        null_emb_tt = None
        dev_opened_for_ttnn_text_encoder = False
        print(
            "[ace_step_v1_5] reusing cached preprocess tensors (skip 5 Hz LM + condition encode on this pass)",
            flush=True,
        )
    else:
        params = GenerationParams(
            task_type="text2music",
            caption=run_prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            reference_audio=None,
            duration=float(args.duration_sec),
            inference_steps=int(infer_steps),
            guidance_scale=gs,
            lm_cfg_scale=1.0 if use_ttnn_5hz_lm else 2.0,
            use_adg=use_adg,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=1.0,
            thinking=True,
            use_constrained_decoding=True,
            timesteps=None,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[int(args.seed)],
            audio_format="wav",
            constrained_decoding_debug=bool(args.llm_debug),
        )
        with perf.timed("five_hz_lm_generate"):
            filtered = build_filtered_dit_kwargs_for_handler(dit_handler, llm_handler, params, config, progress=None)
        lm_perf = getattr(llm_handler, "last_lm_perf", None)
        if isinstance(lm_perf, dict):
            perf.set_params(**lm_perf)
            if bool(args.llm_debug) and lm_perf.get("lm_num_tokens") and lm_perf.get("lm_gen_time_s"):
                n_tok = int(lm_perf["lm_num_tokens"])
                gen_s = float(lm_perf["lm_gen_time_s"])
                if gen_s > 0.0:
                    print(
                        f"[ace_step_v1_5][perf][llm-debug] LM generated {n_tok} tokens in {gen_s:.2f}s "
                        f"({n_tok / gen_s:.1f} tok/s)",
                        flush=True,
                    )
        qwen_safetensors = text_model_dir / "model.safetensors"
        if not qwen_safetensors.is_file():
            raise FileNotFoundError(f"Missing Qwen embedding weights at {qwen_safetensors}")

        _configure_ttnn_runtime()
        import ttnn
        from models.demos.ace_step_v1_5.ttnn_impl.audio_code_detokenizer import TtAceStepAudioCodeDetokenizer
        from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step import (
            AceStepQwen3Encoder as TtQwen3EmbeddingEncoder,
        )

        if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
            ttnn.CONFIG.throw_exception_on_fallback = True

        if demo_session.preprocess_dev is not None and demo_session.dit_dev is None:
            dev = demo_session.preprocess_dev
            tt_dev_early = dev
            dev_opened_for_ttnn_text_encoder = True
        elif tt_dev_early is not None:
            dev = tt_dev_early
            dev_opened_for_ttnn_text_encoder = True
        else:
            if mesh_ttnn_preprocess:
                dev = open_preprocess_device(
                    ttnn,
                    device_id=int(args.device_id),
                    num_command_queues=ace_step_preprocess_num_command_queues(use_trace=bool(args.use_trace)),
                )
            else:
                dev = _open_tt_device(
                    ttnn,
                    device_id=int(args.device_id),
                    num_command_queues=(2 if bool(args.use_trace) else 1),
                )
            if hasattr(dev, "enable_program_cache"):
                dev.enable_program_cache()
            dev_opened_for_ttnn_text_encoder = True
            demo_session.preprocess_dev = dev
            tt_dev_early = dev
        llm_handler.set_ttnn_logits_device(dev)

        if demo_session.qwen_tt_encoder is None:
            with perf.timed("qwen_encoder_init", device=dev):
                demo_session.qwen_tt_encoder = TtQwen3EmbeddingEncoder(
                    device=dev, hf_model_dir=str(text_model_dir), qwen_safetensors_path=str(qwen_safetensors)
                )
                demo_session.audio_code_detokenizer = TtAceStepAudioCodeDetokenizer(
                    device=dev,
                    checkpoint_safetensors_path=str(safetensors_path),
                    dtype=getattr(ttnn, "bfloat16", None),
                )
        qwen_tt_encoder = demo_session.qwen_tt_encoder
        audio_code_detokenizer = demo_session.audio_code_detokenizer
        _restore_infer_txt = attach_payload_preprocess_ttnn(
            dit_handler,
            tt_qwen_encoder=qwen_tt_encoder,
            tt_audio_detokenizer=audio_code_detokenizer,
            max_seq_len=256,
            use_trace=bool(args.use_trace),
        )
        if bool(args.use_trace):
            print("[qwen3] backend=ttnn trace+2cq caption encode (handler infer_text_embeddings)", flush=True)
        try:
            from models.demos.ace_step_v1_5.official_lm_preprocess import (
                condition_encode_payload_tt,
                release_preprocess_device_traces,
            )
            from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

            use_trace = bool(args.use_trace)
            with perf.timed("handler_preprocess", device=dev):
                payload, frames = handler_prepare_condition_payload(dit_handler, filtered)
            _log_duration_preprocess_health(
                payload,
                duration_sec=float(args.duration_sec),
                frames=int(frames),
                audio_code_string=str(filtered.get("audio_code_string") or ""),
            )
            perf.set_params(frames=int(frames))
            _configure_vae_quality(
                frames=int(frames),
                mesh_sku=mesh_sku,
                duration_sec=float(args.duration_sec),
                clarity=bool(args.clarity),
            )
            if use_trace:
                release_preprocess_device_traces(
                    device=dev,
                    tt_qwen_encoder=qwen_tt_encoder,
                    tt_audio_detokenizer=audio_code_detokenizer,
                )
            condition_encoder = TtAceStepInstrumentalConditionEncoder(
                device=dev,
                checkpoint_safetensors_path=str(safetensors_path),
                dtype=getattr(ttnn, "bfloat16", None),
            )
            with perf.timed("condition_encoder", device=dev):
                enc_hs_tt_one, enc_mask_np, ctx_tt_one, null_emb_tt = condition_encode_payload_tt(
                    condition_encoder,
                    payload,
                    use_trace=use_trace,
                )
            enc_hs_tt_one, ctx_tt_one = _finalize_condition_trace_tensors(
                enc_hs_tt_one,
                condition_encoder,
                dev,
                use_trace=use_trace,
                ctx_tt=ctx_tt_one,
            )
            if mesh_ttnn_preprocess:
                with perf.timed("preprocess_readback", device=dev):
                    enc_hs = ace_step_ttnn_to_torch(enc_hs_tt_one, mesh_device=dev, dtype=torch.float32).cpu()
                    ctx_lat = ace_step_ttnn_to_torch(ctx_tt_one, mesh_device=dev, dtype=torch.float32).cpu()
                    null_emb = ace_step_ttnn_to_torch(null_emb_tt, mesh_device=dev, dtype=torch.float32).cpu()
                enc_mask = torch.from_numpy(np.asarray(enc_mask_np, dtype=np.float32))
                for _maybe_tt in (enc_hs_tt_one, ctx_tt_one, null_emb_tt):
                    if _maybe_tt is not None:
                        try:
                            ttnn.deallocate(_maybe_tt)
                        except Exception:
                            pass
                enc_hs_tt_one = None
                ctx_tt_one = None
                null_emb_tt = None
                condition_tensors_on_device = False
            else:
                enc_mask = torch.from_numpy(enc_mask_np).to(dtype=torch.float32)
                condition_tensors_on_device = True
            if use_trace:
                print(
                    "[condition] backend=ttnn trace+2cq official "
                    "(lyric 8L + timbre 4L + text+concat + ctx concat trace)",
                    flush=True,
                )
                print(
                    "[preprocess] qwen caption + lyric embed + audio detokenizer + "
                    "5Hz LM prefill+decode trace enabled",
                    flush=True,
                )
            else:
                print("[condition] backend=ttnn official lyric+timbre+text+context (eager)", flush=True)
        finally:
            _restore_infer_txt()
            if hasattr(qwen_tt_encoder, "release_trace"):
                qwen_tt_encoder.release_trace()
            if hasattr(audio_code_detokenizer, "release_trace"):
                audio_code_detokenizer.release_trace()
        if not condition_tensors_on_device:
            demo_session.store_preprocess(
                prompt=run_prompt,
                duration_sec=float(args.duration_sec),
                seed=int(args.seed),
                frames=int(frames),
                enc_hs=enc_hs,
                enc_mask=enc_mask,
                ctx_lat=ctx_lat,
                null_emb=null_emb,
            )
    do_cfg = gs > 1.0 + 1e-6
    perf.set_params(do_cfg=bool(do_cfg))

    t_schedule = _build_t_schedule(
        infer_steps=int(infer_steps),
        variant=str(args.variant),
    )
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

    # --- TTNN (DiT): device may already be open after TTNN Qwen3 caption embedding (handler path) ---
    _configure_ttnn_runtime()
    import ttnn
    from models.demos.ace_step_v1_5.torch_ref._vendored_acestep.acestep.models.common.apg_guidance import MomentumBuffer
    from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
        TtnnMomentumBufferApg,
        adg_guidance_velocity_host,
        adg_guidance_velocity_ttnn,
        apg_guidance_velocity_host,
        apg_guidance_velocity_ttnn,
        bf16_row_from_numpy_bc,
        concat_duplicate_batch,
        dit_init_latents_fp32_tile,
        dit_init_latents_host_f32,
        euler_subtract_v_dt,
        euler_subtract_v_dt_host,
        fp32_tile_to_bf16_tile_l1,
        precompute_dit_temb_steps,
        prepare_latents_for_ttnn_vae,
        refresh_fp32_tile_from_host,
        slice_batch_btc,
        stage_host_temb_steps_to_device,
        stage_host_temb_tp_row,
        tile_fp32_from_numpy_bc,
        typecast_bf16_any_to_fp32_tile,
    )
    from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline
    from models.demos.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

    if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    dit_num_cqs = 2 if bool(args.use_trace) else 1
    if demo_session.dit_dev is not None:
        dev = demo_session.dit_dev
        dev_opened_for_ttnn_text_encoder = True
    elif split_device and dev_opened_for_ttnn_text_encoder and dev is not None:
        dev = transition_preprocess_to_dit_device(
            ttnn,
            dev,
            mesh_sku=mesh_sku,
            device_id=int(args.device_id),
            num_command_queues=dit_num_cqs,
        )
        demo_session.dit_dev = dev
        demo_session.clear_preprocess_device(ttnn)
        print(f"[ace_step_v1_5] opened DiT mesh for SKU={mesh_sku}", flush=True)
    elif not dev_opened_for_ttnn_text_encoder:
        dev = open_dit_device(
            ttnn,
            mesh_sku=mesh_sku,
            device_id=int(args.device_id),
            num_command_queues=dit_num_cqs,
        )
        dev_opened_for_ttnn_text_encoder = True
        demo_session.dit_dev = dev
        if split_device:
            print(f"[ace_step_v1_5] opened DiT mesh for SKU={mesh_sku}", flush=True)

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mem is None:
        raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG.")

    act_dtype = getattr(ttnn, "bfloat16", None)
    if act_dtype is None:
        raise RuntimeError("TTNN DiT needs ttnn.bfloat16; build may be incomplete.")

    def _as_host_numpy_f32(t: torch.Tensor) -> np.ndarray:
        """TTNN staging: never call ``.numpy()`` on tensors that may still require grad."""
        return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()

    def _readback_ttnn(t: Any) -> torch.Tensor:
        return ace_step_ttnn_to_torch(t, mesh_device=dev, dtype=torch.float32)

    pred_latents: Any = None
    wav_bct_cpu: Any = None

    try:
        if demo_session.pipe is None or demo_session.dit_frames != int(frames):
            with perf.timed("dit_pipeline_init", device=dev):
                demo_session.pipe = AceStepV15TTNNPipeline(
                    device=dev,
                    checkpoint_safetensors_path=str(safetensors_path),
                    timesteps_host=timesteps_host,
                    expected_input_length=int(frames),
                )
            demo_session.dit_frames = int(frames)
        pipe = demo_session.pipe
        if ace_step_device_num_chips(dev) > 1:
            ace_step_synchronize_device(ttnn, dev)
        print("[ace_step_v1_5] DiT pipeline init complete", flush=True)

        defer_vae_init = ace_step_device_num_chips(dev) > 1
        from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import ace_step_vae_quality_decode_enabled

        vae_quality_on = ace_step_vae_quality_decode_enabled(
            latent_frames=int(frames),
            mesh_sku=mesh_sku,
            duration_sec=float(args.duration_sec),
        )
        vae_init_key = (int(frames), bool(vae_quality_on))
        if demo_session.tt_vae is not None and demo_session.vae_init_key != vae_init_key:
            demo_session.tt_vae = None
        tt_vae: TtOobleckVaeDecoder | None = demo_session.tt_vae
        if not bool(args.torch_vae):
            vae_cfg = vae_dir / "config.json"
            if not vae_cfg.is_file():
                raise FileNotFoundError(
                    f"TTNN VAE expects a Hugging Face-style folder at {vae_dir} (config.json). "
                    "Install the VAE checkpoint there or pass --torch-vae for PyTorch decode."
                )
            if defer_vae_init:
                print(
                    "[ace_step_v1_5] deferring TTNN VAE init until after denoise "
                    "(multi-device mesh must finish DiT work first)",
                    flush=True,
                )
            else:
                from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
                    ace_step_vae_activation_storage_dtype,
                    ace_step_vae_bfloat8_activations_enabled,
                    ace_step_vae_host_weight_staging_dtype,
                )

                act_dtype_vae = ace_step_vae_activation_storage_dtype(ttnn)
                w_dtype_vae = ace_step_vae_host_weight_staging_dtype(ttnn)
                if ace_step_vae_bfloat8_activations_enabled(
                    latent_frames=int(frames),
                    mesh_sku=mesh_sku,
                    duration_sec=float(args.duration_sec),
                ):
                    print(
                        "[ace_step_v1_5] VAE: bfloat8 activation compute enabled "
                        "(set ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=0 to disable; inter-op buffers stay BF16 ROW_MAJOR)",
                        flush=True,
                    )
                if tt_vae is None:
                    with perf.timed("vae_init", device=dev):
                        tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
                            str(vae_dir),
                            device=dev,
                            latent_frames=int(frames),
                            batch_size=1,
                            activation_dtype=act_dtype_vae,
                            weights_dtype=w_dtype_vae,
                        )
                    demo_session.tt_vae = tt_vae
                    demo_session.vae_init_key = vae_init_key

        _ensure_acestep_on_path()

        num_steps = len(t_schedule)
        cfg_lo = cfg_interval_start
        cfg_hi = cfg_interval_end

        frames_i = int(frames)
        c_lat = 64
        use_host_latent_sampler = ace_step_mesh_use_host_latent_sampler(dev, use_trace=bool(args.use_trace))
        use_seq_cfg = ace_step_mesh_use_sequential_cfg(dev, do_cfg=do_cfg)
        if use_host_latent_sampler:
            print(
                "[ace_step_v1_5] multi-device eager: latents + Euler/CFG on host CPU; DiT forward on mesh",
                flush=True,
            )

        xt_tt: Any = None
        xt_host: torch.Tensor | None = None
        encoder_attn_1d_bk_np: Any = None
        if not bool(args.use_trace):
            if use_host_latent_sampler:
                print("[ace_step_v1_5] initializing latent noise on host CPU …", flush=True)
                xt_host = dit_init_latents_host_f32(
                    batch=1,
                    frames=frames_i,
                    channels=c_lat,
                    seed=int(args.seed),
                )
                print("[ace_step_v1_5] latent noise ready (host CPU)", flush=True)
            else:
                xt_tt = dit_init_latents_fp32_tile(
                    batch=1,
                    frames=frames_i,
                    channels=c_lat,
                    device=dev,
                    dram=mem,
                    seed=int(args.seed),
                )
                print("[ace_step_v1_5] latent noise ready", flush=True)

            encoder_keep_np_single = np.asarray(enc_mask.detach().cpu().numpy(), dtype=np.float32)
            if encoder_keep_np_single.ndim != 2:
                raise ValueError(f"encoder_attention_mask must be rank-2 [B,S], got {encoder_keep_np_single.shape}")
            encoder_keep_np_single = (encoder_keep_np_single > np.float32(0.0)).astype(np.bool_)
            encoder_attn_1d_bk_np = (
                np.concatenate([encoder_keep_np_single, encoder_keep_np_single], axis=0)
                if do_cfg
                else encoder_keep_np_single
            )

        if ace_step_device_num_chips(dev) > 1:
            ace_step_synchronize_device(ttnn, dev)
        print("[ace_step_v1_5] staging encoder/context tensors on device …", flush=True)

        if condition_tensors_on_device:
            if enc_hs_tt_one is None or ctx_tt_one is None:
                raise RuntimeError("Internal error: TTNN condition path did not produce device tensors.")
            if do_cfg:
                if null_emb_tt is None:
                    raise RuntimeError("Internal error: TTNN condition path missing null condition embedding.")
                s_enc = int(enc_hs_tt_one.shape[1])
                d_enc = int(enc_hs_tt_one.shape[-1])
                null_4d = ttnn.reshape(null_emb_tt, (1, 1, 1, d_enc))
                null_rep_4d = ttnn.repeat(null_4d, (1, 1, s_enc, 1))
                null_rep = ttnn.reshape(null_rep_4d, (1, s_enc, d_enc))

                # Eager CFG batch setup: traced replay was not bit-accurate vs eager (audible noise).
                enc_tt_pipe = ttnn.concat([enc_hs_tt_one, null_rep], dim=0)
                ctx_tt_pipe = concat_duplicate_batch(ctx_tt_one)
                try:
                    ttnn.deallocate(enc_hs_tt_one)
                    ttnn.deallocate(null_4d)
                    ttnn.deallocate(null_rep_4d)
                    ttnn.deallocate(null_rep)
                    ttnn.deallocate(ctx_tt_one)
                    ttnn.deallocate(null_emb_tt)
                except Exception:
                    pass
                enc_hs_tt_one = None
                ctx_tt_one = None
                null_emb_tt = None
            else:
                if bool(args.use_trace) and hasattr(ttnn, "clone"):
                    enc_tt_pipe = ttnn.clone(enc_hs_tt_one)
                else:
                    enc_tt_pipe = enc_hs_tt_one
                ctx_tt_pipe = ctx_tt_one
                enc_hs_tt_one = None
                ctx_tt_one = None
                if null_emb_tt is not None:
                    try:
                        ttnn.deallocate(null_emb_tt)
                    except Exception:
                        pass
                    null_emb_tt = None
        elif do_cfg:
            print("[ace_step_v1_5] staging encoder hidden states …", flush=True)
            enc_tt_pipe = bf16_row_from_numpy_bc(
                np.concatenate(
                    [_as_host_numpy_f32(enc_hs), _as_host_numpy_f32(null_emb.expand_as(enc_hs))],
                    axis=0,
                ),
                device=dev,
                dram=mem,
            )
            print("[ace_step_v1_5] staging context latents …", flush=True)
            ctx_row_one = bf16_row_from_numpy_bc(_as_host_numpy_f32(ctx_lat), device=dev, dram=mem)
            ctx_tt_pipe = concat_duplicate_batch(ctx_row_one)
            try:
                ttnn.deallocate(ctx_row_one)
            except Exception:
                pass
        else:
            enc_tt_pipe = bf16_row_from_numpy_bc(_as_host_numpy_f32(enc_hs), device=dev, dram=mem)
            ctx_tt_pipe = bf16_row_from_numpy_bc(_as_host_numpy_f32(ctx_lat), device=dev, dram=mem)

        print("[ace_step_v1_5] condition tensors ready; starting denoise loop …", flush=True)

        temb_per_step: list[Any] | None = None
        tp_per_step: list[Any] | None = None
        temb_on_host = False
        if use_host_latent_sampler or use_seq_cfg or ace_step_device_num_chips(dev) > 1:
            _temb_batch = 1 if use_seq_cfg or use_host_latent_sampler else (2 if do_cfg else 1)
            if not ace_step_mesh_use_host_temb_precompute(dev):
                print(
                    f"[ace_step_v1_5] precomputing timestep embeddings ({num_steps} steps, B={_temb_batch}) …",
                    flush=True,
                )
            temb_per_step, tp_per_step, temb_on_host = precompute_dit_temb_steps(
                pipe,
                num_steps=num_steps,
                target_batch=int(_temb_batch),
                device=dev,
                checkpoint_safetensors_path=str(safetensors_path),
                timesteps_host=timesteps_host,
            )
            if not temb_on_host:
                print("[ace_step_v1_5] timestep embeddings ready", flush=True)

        if bool(args.use_trace):
            # --- Trace + 2CQ path (DiT body trace only when fused_M <= 16) ----------------------
            from models.demos.ace_step_v1_5.ttnn_impl.e2e_model_tt import _E2EDenoiseTrace, run_ttnn_denoise_loop
            from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
                ace_step_dit_body_trace_safe,
                ace_step_dit_fused_m_tiles,
            )

            def _trace_progress(step_idx: int, n_steps: int, t_curr: float, euler_dt: float) -> None:
                if step_idx >= n_steps - 1:
                    print(f"[ttnn] final t={t_curr:.5f}", flush=True)
                else:
                    print(
                        f"[ttnn] step {step_idx + 1}/{n_steps - 1} t={t_curr:.5f} dt={euler_dt:.5f}",
                        flush=True,
                    )

            patch_sz = int(pipe.patch_embed.config.patch_size)
            patch_seq = (int(frames_i) + patch_sz - 1) // patch_sz
            pipe_batch = ace_step_dit_pipe_batch_size(dev, do_cfg=do_cfg)
            if demo_session.trace_state is not None and demo_session.trace_state.has_buffers():
                if not demo_session.trace_state.matches_shape(
                    frames=int(frames_i),
                    pipe_batch=int(pipe_batch),
                    c_lat=64,
                    do_cfg=bool(do_cfg),
                ):
                    demo_session.trace_state.release(dev)
                    demo_session.trace_state = None
            if demo_session.trace_state is None:
                if ace_step_dit_body_trace_safe(batch_size=pipe_batch, patch_seq_len=patch_seq):
                    demo_session.trace_state = _E2EDenoiseTrace(use_full_step=False)
                else:
                    fused_m = ace_step_dit_fused_m_tiles(batch_size=pipe_batch, seq_len=patch_seq)
                    print(
                        f"[ttnn] DiT body trace disabled (fused_M={fused_m}>16): "
                        "eager denoise + DRAM activations for clean audio on long clips",
                        flush=True,
                    )
                    demo_session.trace_state = None
            trace_state = demo_session.trace_state
            _step_prog = make_denoise_progress_fn(perf, num_steps=len(t_schedule))
            _temb_dev = temb_per_step
            _tp_dev = tp_per_step
            if temb_on_host and temb_per_step is not None and tp_per_step is not None:
                print("[ace_step_v1_5] uploading host temb to device for trace denoise …", flush=True)
                _temb_dev, _tp_dev = stage_host_temb_steps_to_device(
                    temb_per_step,
                    tp_per_step,
                    device=dev,
                    dram=mem,
                )

            enc_row = np.asarray(enc_mask.detach().cpu().numpy(), dtype=np.float32).reshape(1, -1)
            encoder_keep_np_single = (enc_row > np.float32(0.0)).astype(np.bool_)
            trace_enc_attn_1d_bk = (
                np.concatenate([encoder_keep_np_single, encoder_keep_np_single], axis=0)
                if do_cfg
                else encoder_keep_np_single
            )
            mask_tt = None
            b_mask = 2 if do_cfg else 1
            xt_dummy = ttnn.zeros(
                (b_mask, int(frames_i), 64),
                device=dev,
                dtype=act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
            )
            with perf.timed("dit_mask_prep", device=dev):
                mask_tt = pipe.build_encoder_attention_mask_b1qk_optional(
                    xt_bt64=xt_dummy,
                    context_latents_bt128=ctx_tt_pipe,
                    encoder_hidden_states_btd=enc_tt_pipe,
                    encoder_attention_mask_1d_bk=trace_enc_attn_1d_bk,
                )
            try:
                ttnn.deallocate(xt_dummy)
            except Exception:
                pass

            try:
                with perf.timed("dit_denoise_loop", device=dev):
                    _trace_loop_kw = dict(
                        pipe=pipe,
                        device=dev,
                        act_dtype=act_dtype,
                        mem=mem,
                        t_schedule=t_schedule,
                        frames=int(frames),
                        do_cfg=do_cfg,
                        seed=int(args.seed),
                        use_adg=use_adg,
                        guidance_scale=float(gs),
                        cfg_interval_start=cfg_interval_start,
                        cfg_interval_end=cfg_interval_end,
                        enc_tt_pipe=enc_tt_pipe,
                        ctx_tt_pipe=ctx_tt_pipe,
                        return_device_latents=(tt_vae is not None or not bool(args.torch_vae)),
                        progress_fn=_step_prog if _step_prog is not None else _trace_progress,
                        trace_state=trace_state,
                        temb_per_step=_temb_dev,
                        tp_per_step=_tp_dev,
                        deallocate_encoder_mask=False,
                    )
                    if mask_tt is None:
                        _trace_loop_kw["enc_mask"] = enc_mask
                    else:
                        _trace_loop_kw["enc_mask"] = None
                        _trace_loop_kw["encoder_attention_mask_b1qk"] = mask_tt
                    _trace_result = run_ttnn_denoise_loop(**_trace_loop_kw)
            finally:
                if trace_state is not None:
                    trace_state.release(dev)

            # ``run_ttnn_denoise_loop`` deallocated ``enc_tt_pipe`` / ``ctx_tt_pipe`` on exit;
            # mark them consumed so the bottom-of-block cleanup does not double-free.
            enc_tt_pipe = None
            ctx_tt_pipe = None

            if not bool(args.torch_vae):
                # TTNN VAE path: device TILE latents (VAE init may still be deferred on mesh).
                xt_tt = _trace_result
            else:
                xt_tt = None
                pred_latents = _trace_result
        else:
            # --- Legacy eager DiT loop (single CQ, no trace) -----------------------------------
            if use_host_latent_sampler:
                if xt_host is None:
                    raise RuntimeError("Internal error: host latent sampler enabled without xt_host.")
                if temb_per_step is None or tp_per_step is None:
                    raise RuntimeError("Internal error: temb precompute missing for host latent sampler.")
                momentum_host = MomentumBuffer() if do_cfg and not use_adg else None
                _xt_tile_buf: Any = None

                def _diffusion_iterate_host(*, step_idx: int, t_curr_f: float, euler_dt: float, log_line: str) -> None:
                    nonlocal xt_host, _xt_tile_buf
                    if step_idx == 0:
                        print(
                            "[ace_step_v1_5] denoise step 1: staging xt (first step may compile kernels) …", flush=True
                        )
                    xt_f32_tile, _xt_tile_buf = refresh_fp32_tile_from_host(
                        xt_host, device=dev, dram=mem, buf=_xt_tile_buf
                    )
                    xt_row = fp32_tile_to_bf16_tile_l1(xt_f32_tile, dram=mem)
                    if xt_f32_tile is not _xt_tile_buf:
                        try:
                            ttnn.deallocate(xt_f32_tile)
                        except Exception:
                            pass

                    if temb_on_host:
                        if step_idx == 0:
                            print("[ace_step_v1_5] denoise step 1: staging temb/tp …", flush=True)
                        temb_tt, tp_tt = stage_host_temb_tp_row(
                            temb_per_step[int(step_idx)],
                            tp_per_step[int(step_idx)],
                            device=dev,
                            dram=mem,
                        )
                    else:
                        temb_tt = temb_per_step[int(step_idx)]
                        tp_tt = tp_per_step[int(step_idx)]

                    if use_seq_cfg:
                        if step_idx == 0:
                            print("[ace_step_v1_5] denoise step 1: sequential B=1 CFG DiT forwards …", flush=True)
                        vpc_rm, vpu_rm = run_mesh_sequential_cfg_forwards(
                            pipe=pipe,
                            xt_b1=xt_row,
                            enc_tt_pipe=enc_tt_pipe,
                            ctx_tt_pipe=ctx_tt_pipe,
                            temb_bd=temb_tt,
                            timestep_proj_b6d=tp_tt,
                            encoder_attention_mask_1d_bk=encoder_attn_1d_bk_np,
                            device=dev,
                        )
                        try:
                            ttnn.deallocate(xt_row)
                        except Exception:
                            pass
                    elif do_cfg:
                        xt_pipe_in = concat_duplicate_batch(xt_row)
                        try:
                            ttnn.deallocate(xt_row)
                        except Exception:
                            pass
                        acoustic = pipe.forward_with_temb_tp(
                            xt_bt64=xt_pipe_in,
                            context_latents_bt128=ctx_tt_pipe,
                            encoder_hidden_states_btd=enc_tt_pipe,
                            temb_bd=temb_tt,
                            timestep_proj_b6d=tp_tt,
                            attention_mask_1d_bt=None,
                            encoder_attention_mask_1d_bk=encoder_attn_1d_bk_np,
                        )
                        apply_cfg_now = cfg_lo <= t_curr_f <= cfg_hi
                        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
                        vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
                        try:
                            ttnn.deallocate(acoustic)
                            ttnn.deallocate(xt_pipe_in)
                        except Exception:
                            pass
                    else:
                        acoustic = pipe.forward_with_temb_tp(
                            xt_bt64=xt_row,
                            context_latents_bt128=ctx_tt_pipe,
                            encoder_hidden_states_btd=enc_tt_pipe,
                            temb_bd=temb_tt,
                            timestep_proj_b6d=tp_tt,
                            attention_mask_1d_bt=None,
                            encoder_attention_mask_1d_bk=encoder_attn_1d_bk_np,
                        )
                        try:
                            ttnn.deallocate(xt_row)
                        except Exception:
                            pass
                        vpc_rm = acoustic
                        vpu_rm = None

                    if temb_on_host:
                        try:
                            ttnn.deallocate(temb_tt)
                            ttnn.deallocate(tp_tt)
                        except Exception:
                            pass

                    if do_cfg:
                        apply_cfg_now = cfg_lo <= t_curr_f <= cfg_hi
                        vpc = _readback_ttnn(vpc_rm)
                        vpu = _readback_ttnn(vpu_rm) if vpu_rm is not None else None
                        try:
                            ttnn.deallocate(vpc_rm)
                            if vpu_rm is not None:
                                ttnn.deallocate(vpu_rm)
                        except Exception:
                            pass
                        if apply_cfg_now:
                            if use_adg:
                                vt = adg_guidance_velocity_host(
                                    xt_host,
                                    vpc,
                                    vpu,
                                    float(t_curr_f),
                                    float(gs),
                                )
                            else:
                                vt = apg_guidance_velocity_host(
                                    vpc,
                                    vpu,
                                    float(gs),
                                    momentum_buffer=momentum_host,
                                    dims=[1],
                                )
                        else:
                            vt = vpc
                    else:
                        vt = _readback_ttnn(vpc_rm)
                        try:
                            ttnn.deallocate(vpc_rm)
                        except Exception:
                            pass

                    xt_host = euler_subtract_v_dt_host(xt=xt_host, vt=vt, dt=float(euler_dt))
                    print(log_line, flush=True)

                with perf.timed("dit_denoise_loop", device=dev):
                    for step_idx in range(num_steps - 1):
                        t_curr_f = float(t_schedule[step_idx])
                        t_next_f = float(t_schedule[step_idx + 1])
                        dt = t_curr_f - t_next_f
                        _diffusion_iterate_host(
                            step_idx=step_idx,
                            t_curr_f=t_curr_f,
                            euler_dt=dt,
                            log_line=f"[ttnn] step {step_idx+1}/{num_steps-1} t={t_curr_f:.5f} dt={dt:.5f}",
                        )

                    t_curr_final = float(t_schedule[-1])
                    _diffusion_iterate_host(
                        step_idx=num_steps - 1,
                        t_curr_f=t_curr_final,
                        euler_dt=t_curr_final,
                        log_line=f"[ttnn] final t={t_curr_final:.5f}",
                    )
                pred_latents = xt_host
            else:
                momentum_ttnn = TtnnMomentumBufferApg() if do_cfg and not use_adg else None

                def _diffusion_iterate(*, step_idx: int, t_curr_f: float, euler_dt: float, log_line: str) -> None:
                    nonlocal xt_tt
                    xt_row = fp32_tile_to_bf16_tile_l1(xt_tt, dram=mem)
                    if do_cfg:
                        xt_pipe_in = concat_duplicate_batch(xt_row)
                        try:
                            ttnn.deallocate(xt_row)
                        except Exception:
                            pass
                    else:
                        xt_pipe_in = xt_row

                    acoustic = pipe.forward(
                        xt_bt64=xt_pipe_in,
                        context_latents_bt128=ctx_tt_pipe,
                        timestep_index=int(step_idx),
                        encoder_hidden_states_btd=enc_tt_pipe,
                        attention_mask_1d_bt=None,
                        encoder_attention_mask_1d_bk=encoder_attn_1d_bk_np,
                    )

                    if do_cfg:
                        apply_cfg_now = cfg_lo <= t_curr_f <= cfg_hi
                        vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
                        vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
                        if apply_cfg_now:
                            if use_adg:
                                vt_tt = adg_guidance_velocity_ttnn(
                                    xt_tt,
                                    vpc_rm,
                                    vpu_rm,
                                    float(t_curr_f),
                                    float(gs),
                                    device=dev,
                                    dram=mem,
                                )
                            else:
                                vt_tt = apg_guidance_velocity_ttnn(
                                    vpc_rm,
                                    vpu_rm,
                                    float(gs),
                                    momentum_buffer=momentum_ttnn,
                                    dims=[1],
                                    dram=mem,
                                )
                        else:
                            try:
                                ttnn.deallocate(vpu_rm)
                            except Exception:
                                pass
                            vt_tt = typecast_bf16_any_to_fp32_tile(vpc_rm, dram=mem)
                    else:
                        vt_tt = typecast_bf16_any_to_fp32_tile(acoustic, dram=mem)

                    try:
                        ttnn.deallocate(xt_pipe_in)
                    except Exception:
                        pass
                    try:
                        ttnn.deallocate(acoustic)
                    except Exception:
                        pass

                    xt_old = xt_tt
                    xt_tt = euler_subtract_v_dt(xt=xt_tt, vt=vt_tt, dt=float(euler_dt), dram=mem)
                    try:
                        ttnn.deallocate(vt_tt)
                    except Exception:
                        pass
                    try:
                        ttnn.deallocate(xt_old)
                    except Exception:
                        pass
                    print(log_line, flush=True)

                with perf.timed("dit_denoise_loop", device=dev):
                    for step_idx in range(num_steps - 1):
                        t_curr_f = float(t_schedule[step_idx])
                        t_next_f = float(t_schedule[step_idx + 1])
                        dt = t_curr_f - t_next_f
                        _diffusion_iterate(
                            step_idx=step_idx,
                            t_curr_f=t_curr_f,
                            euler_dt=dt,
                            log_line=f"[ttnn] step {step_idx+1}/{num_steps-1} t={t_curr_f:.5f} dt={dt:.5f}",
                        )

                    t_curr_final = float(t_schedule[-1])
                    _diffusion_iterate(
                        step_idx=num_steps - 1,
                        t_curr_f=t_curr_final,
                        euler_dt=t_curr_final,
                        log_line=f"[ttnn] final t={t_curr_final:.5f}",
                    )
            if not use_host_latent_sampler and momentum_ttnn is not None:
                momentum_ttnn.reset()

        if tt_vae is None and not bool(args.torch_vae):
            from models.demos.ace_step_v1_5.ttnn_impl.math_perf_env import (
                ace_step_vae_activation_storage_dtype,
                ace_step_vae_bfloat8_activations_enabled,
                ace_step_vae_host_weight_staging_dtype,
            )

            act_dtype_vae = ace_step_vae_activation_storage_dtype(ttnn)
            w_dtype_vae = ace_step_vae_host_weight_staging_dtype(ttnn)
            if ace_step_vae_bfloat8_activations_enabled(
                latent_frames=int(frames),
                mesh_sku=mesh_sku,
                duration_sec=float(args.duration_sec),
            ):
                print(
                    "[ace_step_v1_5] VAE: bfloat8 activation compute enabled "
                    "(set ACE_STEP_VAE_BFLOAT8_ACTIVATIONS=0 to disable; inter-op buffers stay BF16 ROW_MAJOR)",
                    flush=True,
                )
            if ace_step_device_num_chips(dev) > 1:
                ace_step_synchronize_device(ttnn, dev)
            print("[ace_step_v1_5] VAE init starting …", flush=True)
            with perf.timed("vae_init", device=dev):
                tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
                    str(vae_dir),
                    device=dev,
                    latent_frames=int(frames),
                    batch_size=1,
                    activation_dtype=act_dtype_vae,
                    weights_dtype=w_dtype_vae,
                )
            demo_session.tt_vae = tt_vae
            demo_session.vae_init_key = vae_init_key
            print("[ace_step_v1_5] VAE init done", flush=True)

        if tt_vae is not None:
            if xt_tt is None:
                if pred_latents is None:
                    raise RuntimeError("Internal error: TTNN VAE branch reached without latents.")
                if isinstance(pred_latents, ttnn.Tensor):
                    xt_tt = pred_latents
                    pred_latents = None
                else:
                    print("[ace_step_v1_5] uploading denoised latents for TTNN VAE …", flush=True)
                    xt_tt = tile_fp32_from_numpy_bc(_as_host_numpy_f32(pred_latents), device=dev, dram=mem)
            vae_cs, vae_ov = ace_step_resolve_vae_tiling(
                frames=int(frames),
                mesh_sku=mesh_sku,
                chunk_cli=vae_chunk_latents,
                overlap_cli=vae_overlap_latents,
            )
            if vae_ov > vae_overlap_latents:
                print(
                    f"[ace_step_v1_5] VAE tiling: chunk={vae_cs} overlap={vae_ov} "
                    f"(auto widened for {int(frames)} latent frames on mesh)",
                    flush=True,
                )
            ace_step_synchronize_device(ttnn, dev)
            xt_vae_in = prepare_latents_for_ttnn_vae(xt_tt, dram=mem)
            if xt_vae_in is not xt_tt:
                try:
                    ttnn.deallocate(xt_tt)
                except Exception:
                    pass
                xt_tt = xt_vae_in
            use_vae_trace = bool(args.use_trace)
            if use_vae_trace:
                print(
                    "[vae] backend=ttnn decode_tiled (trace if single window; overlap-add eager)",
                    flush=True,
                )
            else:
                print("[vae] backend=ttnn eager decode_tiled", flush=True)
            with perf.timed("vae_decode", device=dev):
                if use_vae_trace:
                    ttnn.synchronize_device(dev)
                wav_tt = tt_vae.decode_tiled(
                    xt_tt,
                    chunk_size=vae_cs,
                    overlap=vae_ov,
                    use_trace=use_vae_trace,
                )
                if use_vae_trace:
                    ttnn.synchronize_device(dev)
                    if hasattr(tt_vae, "release_trace"):
                        tt_vae.release_trace()
            ace_step_synchronize_device(ttnn, dev)
            wav_ntc = _readback_ttnn(wav_tt)
            try:
                ttnn.deallocate(wav_tt)
            except Exception:
                pass
            # VAE decode returns ``[B, T_audio, C]``; keep NTC (do not permute to BCT).
            wav_bct_cpu = wav_ntc.detach().cpu()
        elif pred_latents is None:
            # Non-trace + torch-VAE: bring DiT latents to host. (Trace + torch-VAE path already
            # populated ``pred_latents`` via ``run_ttnn_denoise_loop(return_device_latents=False)``.)
            pred_latents = _readback_ttnn(xt_tt)

        try:
            if enc_tt_pipe is not None:
                ttnn.deallocate(enc_tt_pipe)
            if ctx_tt_pipe is not None:
                ttnn.deallocate(ctx_tt_pipe)
            if xt_tt is not None:
                ttnn.deallocate(xt_tt)
        except Exception:
            pass
        # ``momentum_ttnn.reset()`` is already invoked inside the non-trace branch above;
        # the trace branch does not allocate a momentum buffer.
    finally:
        if condition_encoder is not None and hasattr(condition_encoder, "release_trace"):
            condition_encoder.release_trace()
        if tt_vae is not None and hasattr(tt_vae, "release_trace"):
            tt_vae.release_trace()
        for _maybe_tt in (enc_hs_tt_one, ctx_tt_one, null_emb_tt):
            if _maybe_tt is not None:
                try:
                    ttnn.deallocate(_maybe_tt)
                except Exception:
                    pass
        demo_session.release(ttnn)

    if wav_bct_cpu is not None:
        wav = _normalize_wav_for_save(wav_bct_cpu.float())
        wav_to_save = wav[0]
        out_path = run_out_path if run_out_path is not None else Path(args.out)
        print(
            f"[ace_step_v1_5] writing wav shape={tuple(wav_to_save.shape)} -> {out_path}",
            flush=True,
        )
        with perf.timed("audio_save"):
            _save_wav_fallback(wav_to_save, out_path, sample_rate=48000)
        print(f"Wrote: {out_path}", flush=True)
    else:
        if pred_latents is None:
            raise RuntimeError("Internal error: latent decode path neither TTNN nor PyTorch.")

        from diffusers.models import AutoencoderOobleck

        vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(torch_dev)
        with perf.timed("vae_decode_torch"):
            with torch.inference_mode():
                lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
                wav = vae.decode(lat).sample.float().cpu()
        wav = _normalize_wav_for_save(wav)
        wav_to_save = wav[0]
        out_path = run_out_path if run_out_path is not None else Path(args.out)
        print(
            f"[ace_step_v1_5] writing wav shape={tuple(wav_to_save.shape)} -> {out_path}",
            flush=True,
        )
        with perf.timed("audio_save"):
            _save_wav_fallback(wav_to_save, out_path, sample_rate=48000)
        print(f"Wrote: {out_path}", flush=True)

    if perf.enabled:
        demo_session.session_perf.add_pass_snapshot(
            perf.export_pass_snapshot(
                label="demo_total",
                session_pass=0,
                is_warmup=False,
            )
        )
    perf.emit_summary(label="demo_total")
    emit_session_summary(demo_session.session_perf, params=perf.params)
    from loguru import logger

    logger.success("Wrote: {}", run_out_path.resolve())


if __name__ == "__main__":
    main()
