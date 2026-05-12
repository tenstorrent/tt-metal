"""
ACE-Step v1.5 demo: official-style host preprocessing + TTNN DiT sampler + host VAE.

By default this matches ``torch_ref/run_prompt_to_wav.py --use-official-acestep`` for **Phase 1**
(5 Hz LM / CoT, audio codes, handler ``preprocess_batch``, TTNN Qwen3 caption encoder via
``infer_text_embeddings``, and HF ``prepare_condition`` with **precomputed LM hints**), emitting the
same style of **loguru** / model logs as the official CLI. DiT sampling runs on TTNN.

Use ``--fast-preprocess`` to skip the LM and use the lightweight path (tokenizer + TTNN ``Qwen3Model``
embedding encoder + ``precomputed_lm_hints_25Hz=None``), avoiding a PyTorch Qwen forward while still
using HF ``prepare_condition`` on the host.

``--use-official-lm`` runs full ``acestep.inference.generate_music`` (PyTorch DiT on host) with no TTNN.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

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


def _configure_ttnn_runtime(*, no_ttnn_strict: bool) -> None:
    """Insert tt-metal roots and optional strict fallback before ``import ttnn``."""
    _prepend_ttnn_pkg_to_syspath()
    if not no_ttnn_strict:
        os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'


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


_WELL_KNOWN_REPO_ROOTS = [
    Path.home() / "proj_sdk" / "ACE-Step-1.5",
    Path.home() / "ACE-Step-1.5",
    Path("/opt") / "ACE-Step-1.5",
]


def _resolve_ace_step_repo_root(*, ckpt_dir: str | None, ace_step_repo_root: str | None) -> Path | None:
    candidates: list[Path] = []
    if ace_step_repo_root:
        candidates.append(Path(ace_step_repo_root).expanduser().resolve())
    env = os.environ.get("ACE_STEP_REPO_ROOT")
    if env:
        candidates.append(Path(env).expanduser().resolve())
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


def _null_condition_emb(ace: Any) -> torch.Tensor:
    pass

    nc = getattr(ace, "null_condition_emb", None)
    if nc is None:
        inner = getattr(ace, "model", None)
        if inner is not None:
            nc = getattr(inner, "null_condition_emb", None)
    if nc is None:
        raise RuntimeError("Could not find null_condition_emb on ACE-Step model (needed for CFG).")
    return nc


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
        description="ACE-Step v1.5: HF-style preprocessing + TTNN DiT + host VAE.",
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
        "--fast-preprocess",
        action="store_true",
        help=(
            "Skip 5 Hz LM + handler batching; use tokenizer + TTNN Qwen3 embedding encoder and HF prepare_condition "
            "(no precomputed LM hints). Avoids importing AceStepHandler (no torchaudio / full ACE-Step training stack). "
            "If omitted and torchaudio is not installed, this path is selected automatically."
        ),
    )
    ap.add_argument(
        "--no-ttnn-strict",
        action="store_true",
        help="Do not set throw_exception_on_fallback (may hide TTNN fallbacks).",
    )
    args = ap.parse_args()

    fast_preprocess = bool(args.fast_preprocess)
    if not fast_preprocess:
        import importlib.util

        if importlib.util.find_spec("torchaudio") is None:
            print(
                "[ace_step_v1_5] torchaudio not found; using --fast-preprocess "
                "(install torchaudio for the 5 Hz LM / AceStepHandler path).",
                file=sys.stderr,
                flush=True,
            )
            fast_preprocess = True

    import torch
    from transformers import AutoModel, AutoTokenizer  # tokenizer always; AceStep ``prepare_condition`` via AutoModel

    ckpt_dir = Path(args.ckpt_dir)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(ckpt_dir)

    model_dir = _ensure_variant(args.variant, ckpt_dir)
    _ensure_variant("vae", ckpt_dir)
    _ensure_variant("Qwen3-Embedding-0.6B", ckpt_dir)
    if not fast_preprocess:
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
        from models.demos.ace_step_v1_5.ref_decoder_compare import ensure_acestep_repo_on_path

        ensure_acestep_repo_on_path(root)
        return root

    # --- Optional: full official path (LLM), no TTNN ---
    if args.use_official_lm:
        from models.demos.ace_step_v1_5.official_lm_preprocess import configure_acestep_logging

        configure_acestep_logging()
        ref_root = _ensure_acestep_on_path()
        try:
            from acestep.handler import AceStepHandler
            from acestep.inference import GenerationConfig, GenerationParams, generate_music
            from acestep.llm_inference import LLMHandler
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "--use-official-lm requires AceStepHandler and its deps "
                f"(missing {e.name!r}). pip install torchaudio (match your PyTorch build), "
                "or run without --use-official-lm and use --fast-preprocess for TTNN demos."
            ) from e

        import acestep.model_downloader as _mdl

        _mdl.MAIN_MODEL_COMPONENTS = [args.variant, "vae", "Qwen3-Embedding-0.6B", args.lm_variant]

        dit_handler = AceStepHandler()
        llm_handler = LLMHandler()
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
            raise RuntimeError("LLMHandler.initialize failed")
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

    if fast_preprocess:
        # --- Lightweight: TTNN Qwen3 embedding encoder + HF prepare_condition (no 5 Hz LM / no precomputed hints) ---
        tok = AutoTokenizer.from_pretrained(str(text_model_dir))
        dit_instruction = "Fill the audio semantic mask based on the given conditions:"
        metas = {"caption": args.prompt, "duration": float(args.duration_sec), "language": "en"}
        text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{args.prompt}

# Metas
{metas}<|endoftext|>
"""
        tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        attn_mask_bool = tokens["attention_mask"].to(torch_dev).to(torch.bool)

        frames = int(round(float(args.duration_sec) * 25.0))
        if frames <= 0:
            raise ValueError("duration_sec must be > 0")

        silence = torch.load(str(silence_latent_path), map_location="cpu").to(torch.float32)
        if silence.ndim != 3:
            raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
        if int(silence.shape[-1]) == 64:
            pass
        elif int(silence.shape[1]) == 64:
            silence = silence.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")
        src_latents = silence[:, :frames, :].contiguous()
        if src_latents.shape[1] < frames:
            rep = (frames + src_latents.shape[1] - 1) // src_latents.shape[1]
            src_latents = src_latents.repeat(1, rep, 1)[:, :frames, :].contiguous()

        chunk_masks = torch.ones((1, frames, 64), dtype=torch.float32)

        _configure_ttnn_runtime(no_ttnn_strict=args.no_ttnn_strict)
        import ttnn
        from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder

        if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
            ttnn.CONFIG.throw_exception_on_fallback = True

        qwen_safetensors = text_model_dir / "model.safetensors"
        if not qwen_safetensors.is_file():
            raise FileNotFoundError(f"Missing Qwen embedding weights at {qwen_safetensors}")

        dev = ttnn.open_device(device_id=int(args.device_id))
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        dev_opened_for_ttnn_text_encoder = True

        input_ids_np = tokens["input_ids"].numpy().astype(np.uint32)
        attn_mask_np = tokens["attention_mask"].numpy().astype(np.float32)

        qwen_enc = TtQwen3EmbeddingEncoder(
            device=dev,
            hf_model_dir=str(text_model_dir),
            qwen_safetensors_path=str(qwen_safetensors),
        )
        try:
            text_hs_tt = qwen_enc.forward(input_ids_np, attn_mask_np)
            text_hidden_states = ttnn.to_torch(text_hs_tt).float()
        finally:
            del qwen_enc

        if text_hidden_states.ndim == 4:
            text_hidden_states = text_hidden_states.squeeze(1).contiguous()
        text_hidden_states = text_hidden_states.to(torch_dev)

        ace = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).eval().to(torch_dev)
        B = 1
        lyric_dim = int(text_hidden_states.shape[-1])
        lyric_hidden_states = torch.zeros((B, 1, lyric_dim), dtype=torch.float32, device=torch_dev)
        lyric_attention_mask = torch.ones((B, 1), dtype=torch.bool, device=torch_dev)
        refer_audio_acoustic_hidden_states_packed = torch.zeros((B, 1, 64), dtype=torch.float32, device=torch_dev)
        refer_audio_order_mask = torch.zeros((B,), dtype=torch.long, device=torch_dev)
        latent_attention_mask = torch.ones((B, frames), dtype=torch.float32, device=torch_dev)

        with torch.inference_mode():
            enc_hs, enc_mask, ctx_lat = ace.prepare_condition(
                text_hidden_states=text_hidden_states.to(dtype=torch.float32),
                text_attention_mask=attn_mask_bool,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                hidden_states=src_latents.to(device=torch_dev, dtype=torch.float32),
                attention_mask=latent_attention_mask,
                silence_latent=silence.to(device=torch_dev, dtype=torch.float32),
                src_latents=src_latents.to(device=torch_dev, dtype=torch.float32),
                chunk_masks=chunk_masks.to(device=torch_dev, dtype=torch.float32),
                is_covers=torch.zeros((B,), dtype=torch.bool, device=torch_dev),
                precomputed_lm_hints_25Hz=None,
            )

        enc_hs = enc_hs.float().cpu()
        enc_mask = enc_mask.float().cpu()
        ctx_lat = ctx_lat.float().cpu()

        null_emb = _null_condition_emb(ace).float().cpu()
    else:
        # --- Official: 5 Hz LM + AceStepHandler batching + prepare_condition (precomputed LM hints) ---
        ref_root = _ensure_acestep_on_path()

        from models.demos.ace_step_v1_5.official_lm_preprocess import (
            attach_infer_text_embeddings_ttnn,
            build_filtered_dit_kwargs_for_handler,
            configure_acestep_logging,
            handler_prepare_condition_tensors,
        )

        configure_acestep_logging()
        try:
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Default preprocessing imports AceStepHandler, which pulls ACE-Step training code "
                f"(e.g. torchaudio). Missing module: {e.name!r}. "
                "Fix: pip install torchaudio (match your torch/CUDA build from pytorch.org), "
                "or rerun with --fast-preprocess to skip the 5 Hz LM + handler path."
            ) from e

        import acestep.model_downloader as _mdl

        from models.demos.ace_step_v1_5.acestep_preprocess_shim import GenerationConfig, GenerationParams

        _mdl.MAIN_MODEL_COMPONENTS = [args.variant, "vae", "Qwen3-Embedding-0.6B", args.lm_variant]

        dit_handler = AceStepHandler()
        llm_handler = LLMHandler()
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
        )
        print(status, flush=True)
        if not ok:
            raise RuntimeError("LLMHandler.initialize failed")

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
        from models.demos.ace_step_v1_5.ttnn_impl.qwen3_embedding_encoder import TtQwen3EmbeddingEncoder

        if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
            ttnn.CONFIG.throw_exception_on_fallback = True

        dev = ttnn.open_device(device_id=int(args.device_id))
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        dev_opened_for_ttnn_text_encoder = True

        qwen_tt_encoder = TtQwen3EmbeddingEncoder(
            device=dev,
            hf_model_dir=str(text_model_dir),
            qwen_safetensors_path=str(qwen_safetensors),
        )
        _restore_infer_txt = attach_infer_text_embeddings_ttnn(
            dit_handler, tt_qwen_encoder=qwen_tt_encoder, max_seq_len=256
        )
        try:
            enc_hs, enc_mask, ctx_lat, frames, null_emb = handler_prepare_condition_tensors(dit_handler, filtered)
        finally:
            _restore_infer_txt()
            del qwen_tt_encoder
    do_cfg = gs > 1.0 + 1e-6

    noise = torch.randn((1, frames, 64), dtype=torch.float32)

    t_schedule = _build_t_schedule(
        shift=float(args.shift),
        infer_steps=int(infer_steps),
        timesteps=args.timesteps,
        variant=str(args.variant),
    )
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

    # --- TTNN (DiT): device may already be open after TTNN Qwen3 caption embedding (fast or handler path) ---
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
        tile_fp32_from_numpy_bc,
        typecast_bf16_any_to_fp32_tile,
    )
    from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline

    if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    if not dev_opened_for_ttnn_text_encoder:
        dev = ttnn.open_device(device_id=int(args.device_id))
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mem is None:
        raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG.")

    def _as_host_numpy_f32(t: torch.Tensor) -> np.ndarray:
        """TTNN staging: never call ``.numpy()`` on tensors that may still require grad."""
        return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()

    try:
        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=int(frames),
        )

        _ensure_acestep_on_path()

        num_steps = len(t_schedule)
        cfg_lo = float(args.cfg_interval_start)
        cfg_hi = float(args.cfg_interval_end)

        latent_keep = torch.ones((1, frames), dtype=torch.bool)
        frames_i = int(frames)
        c_lat = int(noise.shape[2])

        mask2_torch = torch.cat([enc_mask, enc_mask], dim=0) if do_cfg else enc_mask
        keep2_torch = torch.cat([latent_keep, latent_keep], dim=0) if do_cfg else latent_keep

        if do_cfg:
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

        xt_tt = tile_fp32_from_numpy_bc(_as_host_numpy_f32(noise), device=dev, dram=mem)
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
                enc_attn = mask2_torch
                attn_1d = keep2_torch
            else:
                xt_pipe_in = xt_row
                enc_attn = enc_mask
                attn_1d = latent_keep

            acoustic = pipe.forward(
                xt_bt64=xt_pipe_in,
                context_latents_bt128=ctx_tt_pipe,
                timestep_index=int(step_idx),
                encoder_hidden_states_btd=enc_tt_pipe,
                attention_mask_1d_bt=attn_1d,
                encoder_attention_mask_1d_bk=enc_attn,
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

        pred_latents = ttnn.to_torch(xt_tt).to(torch.float32).contiguous()

        try:
            ttnn.deallocate(enc_tt_pipe)
            ttnn.deallocate(ctx_tt_pipe)
            ttnn.deallocate(xt_tt)
        except Exception:
            pass
        if momentum_ttnn is not None:
            momentum_ttnn.reset()
    finally:
        ttnn.close_device(dev)

    from diffusers.models import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(str(vae_dir)).eval().to(torch_dev)
    with torch.inference_mode():
        lat = pred_latents.transpose(1, 2).contiguous().to(device=torch_dev, dtype=next(vae.parameters()).dtype)
        wav = vae.decode(lat).sample.float().cpu()
        # Full peak normalize so quiet VAE outputs (|x| << 1) are still audible in WAV players.
        # The old rule only scaled when peak > 1.0, which often leaves TTNN latents' decode near-silent.
        peak = wav.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
        wav = (wav / peak).clamp(-1.0, 1.0)

    out_path = Path(args.out)
    _save_wav_fallback(wav[0], out_path, sample_rate=48000)
    print(f"Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
