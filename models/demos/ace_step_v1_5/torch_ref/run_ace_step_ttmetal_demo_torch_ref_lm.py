"""
ACE-Step v1.5 demo (tt-metal pipeline) using **PyTorch-only** LM from ``torch_ref.five_hz_lm``.

This file is a copy of ``../run_prompt_to_wav.py`` with ``LocalFiveHzLMHandler`` imported from
``models.demos.ace_step_v1_5.torch_ref.five_hz_lm`` for PCC / parity against the production
``five_hz_lm`` handler. See ``torch_ref/README_LM_PCC_TORCH_REF.md`` and ``torch_ref/five_hz_lm/README.md``.

Latent diffusion runs on TTNN; VAE decode uses the TTNN Oobleck port by default unless ``--torch-vae``.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def _require_torchaudio() -> None:
    import importlib.util

    if importlib.util.find_spec("torchaudio") is not None:
        return
    raise RuntimeError(
        "torchaudio is required for ACE-Step v1.5 demo preprocessing (5 Hz LM + AceStepHandler).\n"
        "Install a build that matches your PyTorch/CUDA version, for example:\n"
        "  pip install torchaudio\n"
        "See https://pytorch.org/get-started/locally/ for the correct wheel index."
    )


def _configure_ttnn_runtime(*, no_ttnn_strict: bool) -> None:
    """Insert tt-metal roots and optional strict fallback before ``import ttnn``."""
    _prepend_ttnn_pkg_to_syspath()
    if not no_ttnn_strict:
        os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'


def _open_tt_device(ttnn: Any, *, device_id: int) -> Any:
    """Open device with L1 small arena sized for conv/VAE (same default as ``tests/conftest.py`` / ign/ACE_perf)."""
    return ttnn.open_device(
        device_id=int(device_id),
        l1_small_size=int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304")),
        trace_region_size=128 << 20,
    )


def _build_t_schedule(*, shift: float, infer_steps: int, timesteps: str | None, variant: str) -> list[float]:
    variant_l = (variant or "").lower()
    is_turbo = "turbo" in variant_l

    if timesteps:
        raw = [float(x.strip()) for x in timesteps.split(",") if x.strip()]
        while raw and raw[-1] == 0.0:
            raw.pop()
        if not raw:
            raise ValueError("--timesteps provided but empty after removing zeros")
        if is_turbo:
            mapped = [min(_VALID_TIMESTEPS, key=lambda v, t=t: abs(v - t)) for t in raw]
            out: list[float] = []
            for t in mapped:
                if not out or out[-1] != t:
                    out.append(t)
            return out
        return raw

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


_VENDORED_ACESTEP_ROOT = Path(__file__).resolve().parent / "_vendored_acestep"

_WELL_KNOWN_REPO_ROOTS = [
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
    Path.home() / "ACE-Step-1.5",
    Path("/opt") / "ACE-Step-1.5",
]


def _resolve_ace_step_repo_root(*, ckpt_dir: str | None, ace_step_repo_root: str | None) -> Path | None:
    """Locate a directory containing an ``acestep/`` package.

    Prefers the **vendored copy** under
    ``models/demos/ace_step_v1_5/torch_ref/_vendored_acestep/`` so the demo runs without an
    external clone. ``--ace-step-repo-root`` / ``ACE_STEP_REPO_ROOT`` still take precedence as
    explicit overrides.
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


def _save_wav_fallback(wav: Any, out_path: Path, sample_rate: int = 48000) -> None:
    pass

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().float().cpu()
    if wav.ndim == 1:
        audio = wav.numpy()
    elif wav.ndim == 2:
        if wav.shape[0] in (1, 2):
            audio = wav.transpose(0, 1).contiguous().numpy()
        else:
            audio = wav.numpy()
    else:
        raise ValueError(f"Expected wav rank 1 or 2, got shape {tuple(wav.shape)}")
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(out_path), audio, samplerate=sample_rate)
        return
    except ModuleNotFoundError:
        pass
    from scipy.io import wavfile  # type: ignore

    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), sample_rate, audio_i16)


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
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=str(_DEFAULT_CKPT_DIR),
        help="Checkpoint root dir (default: ~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints).",
    )
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
    ap.add_argument("--duration_sec", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--infer_steps", type=int, default=None, help="Default: 8 turbo, 50 base.")
    ap.add_argument("--timesteps", type=str, default=None, help="Comma-separated t schedule (optional).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG strength (default: 7 base, 1 turbo). Set 1 to disable CFG.",
    )
    ap.add_argument("--cfg_interval_start", type=float, default=0.0)
    ap.add_argument("--cfg_interval_end", type=float, default=1.0)
    ap.add_argument(
        "--use_adg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Angular CFG (ADG apply_norm=False) on device after B=2 TTNN forward (default: base on, turbo off).",
    )
    ap.add_argument("--out", type=str, default="ttnn_out.wav")
    ap.add_argument(
        "--ace-step-repo-root",
        type=str,
        default=None,
        help="ACE-Step-1.5 repo (contains acestep/). Defaults to env ACE_STEP_REPO_ROOT or walk from ckpt_dir.",
    )
    ap.add_argument(
        "--use-official-lm",
        action="store_true",
        help="Run full official generate_music (LLM+handlers, CPU). Does not use TTNN; writes --out for A/B.",
    )
    ap.add_argument(
        "--ttnn-condition-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run ACE condition embedding in TTNN (handler path). Default: on. "
            "Use --no-ttnn-condition-embedding for Torch prepare_condition."
        ),
    )
    ap.add_argument(
        "--no-ttnn-strict",
        action="store_true",
        help="Do not set throw_exception_on_fallback (may hide TTNN fallbacks).",
    )
    ap.add_argument(
        "--pytorch-lm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use host PyTorch HF for the 5 Hz LM. Default: TTNN causal stack "
            "(ttnn_impl/qwen_tt_transformers_lm.py). TTNN LM forces lm_cfg_scale=1."
        ),
    )
    ap.add_argument(
        "--torch-vae",
        action="store_true",
        help=(
            "Decode latents with PyTorch Diffusers AutoencoderOobleck on GPU/CPU. "
            "Default: TTNN Oobleck decoder (requires ckpt_dir/vae/config.json + weights)."
        ),
    )
    ap.add_argument(
        "--vae-chunk-latents",
        type=int,
        default=32,
        help=(
            "TTNN VAE only: maximum latent time length per decode tile (overlap-add for longer clips). "
            "If decode still overflows L1, lower this value. "
            "Override with env ACE_STEP_VAE_CHUNK_LATENTS."
        ),
    )
    ap.add_argument(
        "--vae-overlap-latents",
        type=int,
        default=4,
        help=(
            "TTNN VAE only: latent-frame overlap between tiles (min 4 internally when possible). "
            "Override with env ACE_STEP_VAE_OVERLAP_LATENTS."
        ),
    )
    args = ap.parse_args()

    _require_torchaudio()

    import torch

    ckpt_dir = Path(args.ckpt_dir)
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

    dev_opened_for_ttnn_text_encoder = False

    gs = args.guidance_scale
    if gs is None:
        gs = 1.0 if "turbo" in str(args.variant).lower() else 7.0
    gs = float(gs)

    use_adg = args.use_adg
    if use_adg is None:
        use_adg = "base" in str(args.variant).lower() and "turbo" not in str(args.variant).lower()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_acestep_on_path() -> Path:
        root = _resolve_ace_step_repo_root(ckpt_dir=str(args.ckpt_dir), ace_step_repo_root=args.ace_step_repo_root)
        if root is None:
            raise RuntimeError(
                "Could not find ACE-Step-1.5 repo (needed for acestep imports). "
                "Pass --ace-step-repo-root or set ACE_STEP_REPO_ROOT."
            )
        from models.demos.ace_step_v1_5.demo.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(root)
        return root

    # --- Optional: full official path (LLM), no TTNN ---
    if args.use_official_lm:
        from models.demos.ace_step_v1_5.official_lm_preprocess import configure_acestep_logging
        from models.demos.ace_step_v1_5.torch_ref.transformers_cache_compat import apply_transformers_cache_compat

        apply_transformers_cache_compat()
        configure_acestep_logging()
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
                "  2. Point --ace-step-repo-root / $ACE_STEP_REPO_ROOT at an external clone "
                "of https://github.com/ace-step/ACE-Step-1.5.\n"
                "  3. Run without --use-official-lm (the default TTNN path doesn't need "
                "acestep.inference)."
            ) from e

        # Weights are pre-downloaded by ``_ensure_variant`` earlier in main(); the
        # historical ``_mdl.MAIN_MODEL_COMPONENTS = [...]`` mutation here was informational
        # only (telling the vendored downloader which sub-components live in the main repo).
        # Removed because the handler's defaults already cover the same set and every file
        # exists on disk by the time the handler runs.

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
            cfg_interval_start=float(args.cfg_interval_start),
            cfg_interval_end=float(args.cfg_interval_end),
            shift=float(args.shift),
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
    enc_hs_tt_one = None
    ctx_tt_one = None
    null_emb_tt = None

    # --- 5 Hz LM + AceStepHandler batching + prepare_condition (precomputed LM hints) ---
    ref_root = _ensure_acestep_on_path()

    from models.demos.ace_step_v1_5.official_lm_preprocess import (
        attach_infer_text_embeddings_ttnn,
        build_filtered_dit_kwargs_for_handler,
        configure_acestep_logging,
        handler_prepare_condition_payload,
        handler_prepare_condition_tensors,
    )

    configure_acestep_logging()
    try:
        from acestep.handler import AceStepHandler

        from models.demos.ace_step_v1_5.torch_ref.five_hz_lm import LocalFiveHzLMHandler
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
            "  2. Point --ace-step-repo-root / $ACE_STEP_REPO_ROOT at an external clone "
            "of https://github.com/ace-step/ACE-Step-1.5.\n"
            "  3. If e.name == 'torchaudio': pip install torchaudio (match your torch/CUDA "
            "build from pytorch.org; required for handler preprocessing)."
        ) from e

    from models.demos.ace_step_v1_5.acestep_preprocess_shim import GenerationConfig, GenerationParams

    # Weights are pre-downloaded by ``_ensure_variant`` earlier in main(); the historical
    # ``import acestep.model_downloader as _mdl; _mdl.MAIN_MODEL_COMPONENTS = [...]``
    # mutation here was informational only and has been removed.
    # torch_ref.five_hz_lm is PyTorch-only (see torch_ref/five_hz_lm/README.md).
    # --pytorch-lm is accepted for CLI parity with run_prompt_to_wav.py but has no effect here.
    use_ttnn_5hz_lm = False

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
    status, ok = llm_handler.initialize(
        checkpoint_dir=str(ckpt_dir),
        lm_model_path=args.lm_variant,
        backend="pt",
        device=device,
        experimental_ttnn_causal_lm=False,
    )
    print(status, flush=True)
    if not ok:
        raise RuntimeError("5 Hz LM (local HF) initialize failed")

    ts_list = None
    if args.timesteps:
        raw_ts = [float(x.strip()) for x in args.timesteps.split(",") if x.strip()]
        while raw_ts and raw_ts[-1] == 0.0:
            raw_ts.pop()
        ts_list = raw_ts or None

    params = GenerationParams(
        task_type="text2music",
        caption=args.prompt,
        lyrics="[Instrumental]",
        instrumental=True,
        reference_audio=None,
        duration=float(args.duration_sec),
        inference_steps=int(infer_steps),
        guidance_scale=gs,
        lm_cfg_scale=1.0 if use_ttnn_5hz_lm else 2.0,
        use_adg=use_adg,
        cfg_interval_start=float(args.cfg_interval_start),
        cfg_interval_end=float(args.cfg_interval_end),
        shift=float(args.shift),
        thinking=True,
        use_constrained_decoding=True,
        timesteps=ts_list,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[int(args.seed)],
        audio_format="wav",
        constrained_decoding_debug=True,
    )
    filtered = build_filtered_dit_kwargs_for_handler(dit_handler, llm_handler, params, config, progress=None)
    qwen_safetensors = text_model_dir / "model.safetensors"
    if not qwen_safetensors.is_file():
        raise FileNotFoundError(f"Missing Qwen embedding weights at {qwen_safetensors}")

    _configure_ttnn_runtime(no_ttnn_strict=args.no_ttnn_strict)
    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_ace_step import (
        AceStepQwen3Encoder as TtQwen3EmbeddingEncoder,
    )

    if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    dev = _open_tt_device(ttnn, device_id=int(args.device_id))
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    dev_opened_for_ttnn_text_encoder = True
    # No-op in torch_ref.five_hz_lm (PyTorch logits path only).
    llm_handler.set_ttnn_logits_device(dev)

    qwen_tt_encoder = TtQwen3EmbeddingEncoder(
        device=dev, hf_model_dir=str(text_model_dir), qwen_safetensors_path=str(qwen_safetensors)
    )
    _restore_infer_txt = attach_infer_text_embeddings_ttnn(
        dit_handler, tt_qwen_encoder=qwen_tt_encoder, max_seq_len=256
    )
    try:
        if args.ttnn_condition_embedding:
            from models.demos.ace_step_v1_5.ttnn_impl.condition_encoder import TtAceStepInstrumentalConditionEncoder

            payload, frames = handler_prepare_condition_payload(dit_handler, filtered)
            condition_encoder = TtAceStepInstrumentalConditionEncoder(
                device=dev,
                checkpoint_safetensors_path=str(safetensors_path),
                dtype=getattr(ttnn, "bfloat16", None),
            )
            enc_hs_tt_one, enc_mask_np, ctx_tt_one, null_emb_tt = condition_encoder.forward_payload(payload)
            enc_mask = torch.from_numpy(enc_mask_np).to(dtype=torch.float32)
            condition_tensors_on_device = True
            print("[condition] backend=ttnn official lyric+timbre+text+context", flush=True)
        else:
            enc_hs, enc_mask, ctx_lat, frames, null_emb = handler_prepare_condition_tensors(dit_handler, filtered)
    finally:
        _restore_infer_txt()
        del qwen_tt_encoder
    do_cfg = gs > 1.0 + 1e-6

    t_schedule = _build_t_schedule(
        shift=float(args.shift),
        infer_steps=int(infer_steps),
        timesteps=args.timesteps,
        variant=str(args.variant),
    )
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

    # --- TTNN (DiT): device may already be open after TTNN Qwen3 caption embedding (handler path) ---
    _configure_ttnn_runtime(no_ttnn_strict=args.no_ttnn_strict)
    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import (
        TtnnMomentumBufferApg,
        adg_guidance_velocity_ttnn,
        apg_guidance_velocity_ttnn,
        bf16_row_from_numpy_bc,
        concat_duplicate_batch,
        euler_subtract_v_dt,
        fp32_tile_to_row_bf16,
        slice_batch_btc,
        typecast_bf16_any_to_fp32_tile,
    )
    from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline
    from models.demos.ace_step_v1_5.ttnn_impl.oobleck_vae_decoder import TtOobleckVaeDecoder

    if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    if not dev_opened_for_ttnn_text_encoder:
        dev = _open_tt_device(ttnn, device_id=int(args.device_id))
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mem is None:
        raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG.")

    act_dtype = getattr(ttnn, "bfloat16", None)
    if act_dtype is None:
        raise RuntimeError("TTNN DiT needs ttnn.bfloat16; build may be incomplete.")

    def _as_host_numpy_f32(t: torch.Tensor) -> np.ndarray:
        """TTNN staging: never call ``.numpy()`` on tensors that may still require grad."""
        return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()

    pred_latents: Any = None
    wav_bct_cpu: Any = None

    try:
        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=int(frames),
        )

        tt_vae: TtOobleckVaeDecoder | None = None
        if not bool(args.torch_vae):
            vae_cfg = vae_dir / "config.json"
            if not vae_cfg.is_file():
                raise FileNotFoundError(
                    f"TTNN VAE expects a Hugging Face-style folder at {vae_dir} (config.json). "
                    "Install the VAE checkpoint there or pass --torch-vae for PyTorch decode."
                )
            act_dtype_vae = getattr(ttnn, "bfloat16", None)
            if act_dtype_vae is None:
                raise RuntimeError("TTNN VAE needs ttnn.bfloat16; build may be incomplete.")
            tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
                str(vae_dir),
                device=dev,
                latent_frames=int(frames),
                batch_size=1,
                activation_dtype=act_dtype_vae,
                weights_dtype=act_dtype_vae,
            )

        _ensure_acestep_on_path()

        num_steps = len(t_schedule)
        cfg_lo = float(args.cfg_interval_start)
        cfg_hi = float(args.cfg_interval_end)

        frames_i = int(frames)
        c_lat = 64

        # ``ttnn.rand`` is uniform in [from,to] (default [0,1]); ACE-Step uses standard-normal latents.
        # Use ``ttnn.randn`` for parity with ``torch.randn`` / prior NumPy Gaussian noise.
        if not hasattr(ttnn, "randn"):
            raise RuntimeError(
                "This demo needs ``ttnn.randn`` (Gaussian) for latent init; ``ttnn.rand`` is uniform-only."
            )
        xt_tt = ttnn.randn(
            (1, frames_i, c_lat),
            dev,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            seed=int(np.uint32(int(args.seed))),
        )

        encoder_keep_np_single = np.asarray(enc_mask.detach().cpu().numpy(), dtype=np.float32)
        if encoder_keep_np_single.ndim != 2:
            raise ValueError(f"encoder_attention_mask must be rank-2 [B,S], got {encoder_keep_np_single.shape}")
        encoder_keep_np_single = (encoder_keep_np_single > np.float32(0.0)).astype(np.bool_)
        encoder_attn_1d_bk_np = (
            np.concatenate([encoder_keep_np_single, encoder_keep_np_single], axis=0)
            if do_cfg
            else encoder_keep_np_single
        )

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
            enc_tt_pipe = bf16_row_from_numpy_bc(
                np.concatenate(
                    [_as_host_numpy_f32(enc_hs), _as_host_numpy_f32(null_emb.expand_as(enc_hs))],
                    axis=0,
                ),
                device=dev,
                dram=mem,
            )
            ctx_row_one = bf16_row_from_numpy_bc(_as_host_numpy_f32(ctx_lat), device=dev, dram=mem)
            ctx_tt_pipe = concat_duplicate_batch(ctx_row_one)
            try:
                ttnn.deallocate(ctx_row_one)
            except Exception:
                pass
        else:
            enc_tt_pipe = bf16_row_from_numpy_bc(_as_host_numpy_f32(enc_hs), device=dev, dram=mem)
            ctx_tt_pipe = bf16_row_from_numpy_bc(_as_host_numpy_f32(ctx_lat), device=dev, dram=mem)

        momentum_ttnn = TtnnMomentumBufferApg() if do_cfg and not use_adg else None

        def _diffusion_iterate(*, step_idx: int, t_curr_f: float, euler_dt: float, log_line: str) -> None:
            nonlocal xt_tt
            xt_row = fp32_tile_to_row_bf16(xt_tt, dram=mem)
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

        if tt_vae is not None:
            vae_cs = int(os.environ.get("ACE_STEP_VAE_CHUNK_LATENTS", str(int(args.vae_chunk_latents))))
            vae_ov = int(os.environ.get("ACE_STEP_VAE_OVERLAP_LATENTS", str(int(args.vae_overlap_latents))))
            wav_tt = tt_vae.decode_tiled(xt_tt, chunk_size=vae_cs, overlap=vae_ov)
            wav_ntc = ttnn.to_torch(wav_tt, dtype=torch.float32).contiguous()
            try:
                ttnn.deallocate(wav_tt)
            except Exception:
                pass
            wav_bct_cpu = wav_ntc.permute(0, 2, 1).detach().cpu()
        else:
            pred_latents = ttnn.to_torch(xt_tt, dtype=torch.float32).contiguous()

        try:
            ttnn.deallocate(enc_tt_pipe)
            ttnn.deallocate(ctx_tt_pipe)
            ttnn.deallocate(xt_tt)
        except Exception:
            pass
        if momentum_ttnn is not None:
            momentum_ttnn.reset()
    finally:
        for _maybe_tt in (enc_hs_tt_one, ctx_tt_one, null_emb_tt):
            if _maybe_tt is not None:
                try:
                    ttnn.deallocate(_maybe_tt)
                except Exception:
                    pass
        ttnn.close_device(dev)

    if wav_bct_cpu is not None:
        wav = wav_bct_cpu.float()
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)
    else:
        if pred_latents is None:
            raise RuntimeError("Internal error: latent decode path neither TTNN nor PyTorch.")

        from diffusers.models import AutoencoderOobleck

        vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(torch_dev)
        with torch.inference_mode():
            lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
            wav = vae.decode(lat).sample.float().cpu()
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)

    out_path = Path(args.out)
    _save_wav_fallback(wav[0], out_path, sample_rate=48000)
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
