"""
Compare one DiT-decoder forward: HuggingFace ``AceStepDiTModel`` (PyTorch) vs ``AceStepV15TTNNPipeline`` (TTNN).

Uses the same conditioning path as ``run_prompt_to_wav.py`` (official LM + handler, or fast Qwen-only),
then runs HF and TTNN on identical ``xt`` and timestep schedule entries, reporting PCC per step.

Run from ``tt-metal`` root with ``PYTHONPATH=.`` and the ACE-Step repo on disk (checkpoints + optional 5 Hz LM).

Example::

    PYTHONPATH=. python3 models/demos/ace_step_v1_5/pcc_compare_torch_ttnn.py \\
      --prompt \"Lo-fi hip hop, warm pads, mellow drums\" \\
      --ckpt_dir /path/to/ACE-Step-1.5/checkpoints \\
      --variant acestep-v15-base \\
      --ace-step-repo-root /path/to/ACE-Step-1.5 \\
      --duration_sec 5 --infer_steps 4 --seed 0 --shift 1.0 --guidance-scale 1 \\
      --device_id 0 --precomputed-hints src
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure tt-metal and bundled ttnn are importable before any module that pulls in ``ttnn``.
_tt_metal_root = Path(__file__).resolve().parents[3]
for _p in (str(_tt_metal_root), str(_tt_metal_root / "ttnn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from models.demos.ace_step_v1_5.ref_decoder_compare import (
    ensure_acestep_repo_on_path,
    hf_decoder_velocity,
    load_hf_decoder_from_checkpoint_dir,
)
from models.demos.ace_step_v1_5.run_prompt_to_wav import _build_t_schedule, _resolve_ace_step_repo_root


def _comp_pcc(golden: torch.Tensor, calculated: torch.Tensor) -> tuple[bool, float]:
    """Pearson PCC over flattened finite values (same spirit as ``comp_pcc`` in utility_functions)."""
    g = golden.detach().float().cpu().reshape(-1).numpy()
    c = calculated.detach().float().cpu().reshape(-1).numpy()
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    if np.array_equal(g, c):
        return True, 1.0
    cal_pcc = float(np.ma.corrcoef(np.ma.masked_invalid(g), np.ma.masked_invalid(c))[0, 1])
    if isinstance(cal_pcc, np.ma.core.MaskedConstant) or not np.isfinite(cal_pcc):
        return True, 1.0
    return cal_pcc >= 0.99, cal_pcc


def _null_condition_emb(ace: torch.nn.Module) -> torch.Tensor:
    nc = getattr(ace, "null_condition_emb", None)
    if nc is None:
        inner = getattr(ace, "model", None)
        if inner is not None:
            nc = getattr(inner, "null_condition_emb", None)
    if nc is None:
        raise RuntimeError("Could not find null_condition_emb on ACE-Step model (needed for CFG tooling).")
    return nc


def _as_host_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()


def _prepare_condition_fast(
    *,
    prompt: str,
    duration_sec: float,
    ckpt_dir: Path,
    model_dir: Path,
    silence_latent_path: Path,
    text_model_dir: Path,
    torch_dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    tok = AutoTokenizer.from_pretrained(str(text_model_dir))
    txt_model = AutoModel.from_pretrained(str(text_model_dir)).eval().to(torch_dev)
    dit_instruction = "Fill the audio semantic mask based on the given conditions:"
    metas = {"caption": prompt, "duration": float(duration_sec), "language": "en"}
    text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""
    tokens = tok(text_prompt, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = tokens["input_ids"].to(torch_dev)
    attn_mask = tokens["attention_mask"].to(torch_dev).to(torch.bool)
    with torch.inference_mode():
        text_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        text_hidden_states = text_out.last_hidden_state

    frames = int(round(float(duration_sec) * 25.0))
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
            text_attention_mask=attn_mask,
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
    return enc_hs, enc_mask, ctx_lat, frames, null_emb


def _prepare_condition_official(
    *,
    prompt: str,
    duration_sec: float,
    infer_steps: int,
    guidance_scale: float,
    use_adg: bool,
    cfg_interval_start: float,
    cfg_interval_end: float,
    shift: float,
    timesteps: list[float] | None,
    seed: int,
    ckpt_dir: Path,
    ref_root: Path,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    from models.demos.ace_step_v1_5.acestep_preprocess_shim import GenerationConfig, GenerationParams
    from models.demos.ace_step_v1_5.official_lm_preprocess import (
        build_filtered_dit_kwargs_for_handler,
        configure_acestep_logging,
        handler_prepare_condition_tensors,
    )

    configure_acestep_logging()
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    device = "cpu"
    status, ok = dit_handler.initialize_service(
        project_root=str(ref_root),
        config_path=variant,
        device=device,
        use_flash_attention=False,
    )
    print(status, flush=True)
    if not ok:
        raise RuntimeError("AceStepHandler.initialize_service failed")
    lm_model = "acestep-5Hz-lm-0.6B"
    status, ok = llm_handler.initialize(
        checkpoint_dir=str(ckpt_dir),
        lm_model_path=lm_model,
        backend="pt",
        device=device,
    )
    print(status, flush=True)
    if not ok:
        raise RuntimeError("LLMHandler.initialize failed")

    params = GenerationParams(
        task_type="text2music",
        caption=prompt,
        lyrics="[Instrumental]",
        instrumental=True,
        reference_audio=None,
        duration=float(duration_sec),
        inference_steps=int(infer_steps),
        guidance_scale=float(guidance_scale),
        use_adg=use_adg,
        cfg_interval_start=float(cfg_interval_start),
        cfg_interval_end=float(cfg_interval_end),
        shift=float(shift),
        thinking=True,
        use_constrained_decoding=True,
        timesteps=timesteps,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[int(seed)],
        audio_format="wav",
        constrained_decoding_debug=True,
    )
    filtered = build_filtered_dit_kwargs_for_handler(dit_handler, llm_handler, params, config, progress=None)
    return handler_prepare_condition_tensors(dit_handler, filtered)


def _align_outputs(ref: torch.Tensor, tt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim to common [B, T, C] in case TTNN exposes tile padding on host."""
    b0 = min(int(ref.shape[0]), int(tt.shape[0]))
    t0 = min(int(ref.shape[1]), int(tt.shape[1]))
    c0 = min(int(ref.shape[2]), int(tt.shape[2]))
    return ref[:b0, :t0, :c0], tt[:b0, :t0, :c0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PCC: HF AceStepDiT decoder vs TTNN pipeline (single forward per timestep index)."
    )
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--variant", type=str, default="acestep-v15-base")
    ap.add_argument("--ace-step-repo-root", type=str, default=None)
    ap.add_argument("--duration_sec", type=float, default=5.0)
    ap.add_argument("--infer_steps", "--infer-steps", type=int, default=None, help="Default: 8 turbo else 50 base.")
    ap.add_argument("--timesteps", type=str, default=None, help="Comma-separated schedule (optional).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--guidance_scale", "--guidance-scale", type=float, default=None)
    ap.add_argument("--cfg_interval_start", type=float, default=0.0)
    ap.add_argument("--cfg_interval_end", type=float, default=1.0)
    ap.add_argument("--use_adg", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument(
        "--precomputed-hints",
        type=str,
        choices=("src", "none"),
        default="src",
        help="src: 5 Hz LM + handler prepare_condition (hints from batch). none: Qwen-only, precomputed_lm_hints_25Hz=None.",
    )
    ap.add_argument(
        "--min-pcc", type=float, default=None, help="If set, exit 1 when any step PCC is below this threshold."
    )
    ap.add_argument("--no-ttnn-strict", action="store_true")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    model_dir = ckpt_dir / args.variant
    safetensors_path = model_dir / "model.safetensors"
    silence_latent_path = model_dir / "silence_latent.pt"
    text_model_dir = ckpt_dir / "Qwen3-Embedding-0.6B"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {safetensors_path}")
    if not silence_latent_path.is_file():
        raise FileNotFoundError(f"Missing silence_latent: {silence_latent_path}")

    infer_steps = args.infer_steps
    if infer_steps is None:
        infer_steps = 8 if "turbo" in str(args.variant).lower() else 50

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

    ref_root = _resolve_ace_step_repo_root(ckpt_dir=str(ckpt_dir), ace_step_repo_root=args.ace_step_repo_root)
    if ref_root is None:
        raise RuntimeError("Could not find ACE-Step-1.5 repo. Pass --ace-step-repo-root or set ACE_STEP_REPO_ROOT.")
    ensure_acestep_repo_on_path(ref_root)

    ts_list: list[float] | None = None
    if args.timesteps:
        raw_ts = [float(x.strip()) for x in args.timesteps.split(",") if x.strip()]
        while raw_ts and raw_ts[-1] == 0.0:
            raw_ts.pop()
        ts_list = raw_ts or None

    if args.precomputed_hints == "none":
        enc_hs, enc_mask, ctx_lat, frames, _null = _prepare_condition_fast(
            prompt=args.prompt,
            duration_sec=float(args.duration_sec),
            ckpt_dir=ckpt_dir,
            model_dir=model_dir,
            silence_latent_path=silence_latent_path,
            text_model_dir=text_model_dir,
            torch_dev=torch_dev,
        )
    else:
        enc_hs, enc_mask, ctx_lat, frames, _null = _prepare_condition_official(
            prompt=args.prompt,
            duration_sec=float(args.duration_sec),
            infer_steps=int(infer_steps),
            guidance_scale=gs,
            use_adg=bool(use_adg),
            cfg_interval_start=float(args.cfg_interval_start),
            cfg_interval_end=float(args.cfg_interval_end),
            shift=float(args.shift),
            timesteps=ts_list,
            seed=int(args.seed),
            ckpt_dir=ckpt_dir,
            ref_root=ref_root,
            variant=str(args.variant),
        )

    t_schedule = _build_t_schedule(
        shift=float(args.shift),
        infer_steps=int(infer_steps),
        timesteps=args.timesteps,
        variant=str(args.variant),
    )
    timesteps_host = np.asarray(t_schedule + [0.0], dtype=np.float32)

    xt = torch.randn((1, frames, 64), dtype=torch.float32)
    latent_keep = torch.ones((1, frames), dtype=torch.bool)

    decoder = load_hf_decoder_from_checkpoint_dir(model_dir, ref_repo_root=ref_root, torch_dtype=torch.bfloat16)

    tt_metal_root = str(Path(__file__).resolve().parents[3])
    ttnn_pkg_root = str(Path(tt_metal_root) / "ttnn")
    for p in (tt_metal_root, ttnn_pkg_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    if not args.no_ttnn_strict:
        os.environ["TTNN_CONFIG_OVERRIDES"] = '{"throw_exception_on_fallback": true}'

    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl.full_pipeline import AceStepV15TTNNPipeline

    if not args.no_ttnn_strict and hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "throw_exception_on_fallback"):
        ttnn.CONFIG.throw_exception_on_fallback = True

    mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if mem is None:
        raise RuntimeError("TTNN build missing DRAM_MEMORY_CONFIG.")
    act_dtype = getattr(ttnn, "bfloat16", None)
    if act_dtype is None:
        raise RuntimeError("TTNN build missing bfloat16 dtype.")

    dev = ttnn.open_device(device_id=int(args.device_id))
    try:
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()

        pipe = AceStepV15TTNNPipeline(
            device=dev,
            checkpoint_safetensors_path=str(safetensors_path),
            timesteps_host=timesteps_host,
            expected_input_length=int(frames),
        )

        worst_pcc = 1.0
        for step_idx, t_curr in enumerate(t_schedule):
            ref = hf_decoder_velocity(
                decoder,
                xt=xt,
                context_latents=ctx_lat,
                encoder_hidden_states=enc_hs,
                t_curr=float(t_curr),
                encoder_attention_mask_1d=enc_mask,
            )

            xt_tt = ttnn.as_tensor(
                _as_host_numpy_f32(xt),
                device=dev,
                dtype=act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
            )
            ctx_tt = ttnn.as_tensor(
                _as_host_numpy_f32(ctx_lat),
                device=dev,
                dtype=act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
            )
            enc_tt = ttnn.as_tensor(
                _as_host_numpy_f32(enc_hs),
                device=dev,
                dtype=act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
            )
            acoustic = pipe.forward(
                xt_bt64=xt_tt,
                context_latents_bt128=ctx_tt,
                timestep_index=int(step_idx),
                encoder_hidden_states_btd=enc_tt,
                attention_mask_1d_bt=latent_keep,
                encoder_attention_mask_1d_bk=enc_mask,
            )
            tt_cpu = ttnn.to_torch(acoustic).to(torch.float32)

            ref_a, tt_a = _align_outputs(ref, tt_cpu)
            ok, pcc = _comp_pcc(ref_a, tt_a)
            worst_pcc = min(worst_pcc, float(pcc))
            print(
                f"step_idx={step_idx} t={t_curr:.6f} pcc={float(pcc):.6f} pass_vs_0.99={ok} "
                f"shape_ref={tuple(ref_a.shape)} shape_tt={tuple(tt_a.shape)}",
                flush=True,
            )

        print(f"worst_pcc={worst_pcc:.6f}", flush=True)
        if args.min_pcc is not None and worst_pcc < float(args.min_pcc):
            raise SystemExit(f"worst_pcc {worst_pcc} below --min-pcc {args.min_pcc}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
