#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 Audio+Video Generation Demo

Both video and audio denoising run on TT devices (2x4 WH LB mesh) using
the TTNN LTXTransformerModel. Text encoding and VAE/vocoder
decode use the reference LTX-2 pipeline (torch, CPU).

Usage:
    HF_TOKEN=... python models/tt_dit/demos/ltx/generate_audio_video.py \
        --prompt "A cat playing piano in a cozy room" \
        --output output.mp4 \
        --num_frames 33 --height 512 --width 768 --steps 30
"""

import argparse
import os
import sys
import time

import torch
from loguru import logger

# Monkey-patch CUDA for CPU-only text encoding
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **kw: None
torch.cuda.empty_cache = lambda: None

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT_DIR = os.environ.get("LTX_CHECKPOINT_DIR", os.path.expanduser("~/.cache/ltx-checkpoints"))


def resolve_gemma_path(gemma_path: str) -> str:
    """Resolve HF model ID to local cache directory."""
    if os.path.isdir(gemma_path):
        return gemma_path
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{gemma_path.replace('/', '--')}")
    if os.path.isdir(model_dir):
        snapshots = os.path.join(model_dir, "snapshots")
        if os.path.isdir(snapshots):
            versions = sorted(os.listdir(snapshots))
            if versions:
                return os.path.join(snapshots, versions[-1])
    return gemma_path


def encode_prompt(checkpoint_path: str, gemma_path: str, prompt: str):
    """Encode prompt using reference pipeline → (video_embeds, audio_embeds)."""
    from ltx_pipelines.utils.model_ledger import ModelLedger

    gemma_local = resolve_gemma_path(gemma_path)
    logger.info(f"Loading text encoder from {gemma_local}")
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_local,
    )

    text_encoder = ledger.text_encoder()
    hidden_states_tuple, attention_mask = text_encoder.encode(prompt)
    logger.info(f"Gemma: {len(hidden_states_tuple)} hidden layers")
    del text_encoder

    embeddings_processor = ledger.gemma_embeddings_processor()
    output = embeddings_processor.process_hidden_states(hidden_states_tuple, attention_mask)
    video_embeds = output.video_encoding.float()  # (1, L, 4096)
    audio_embeds = output.audio_encoding.float() if output.audio_encoding is not None else None  # (1, L, 2048)
    logger.info(f"Video embeds: {video_embeds.shape}")
    if audio_embeds is not None:
        logger.info(f"Audio embeds: {audio_embeds.shape}")
    del embeddings_processor

    return video_embeds, audio_embeds


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Audio+Video Generation (TTNN)")
    parser.add_argument("--prompt", type=str, default="A cat playing piano in a cozy room with warm lighting")
    parser.add_argument("--output", type=str, default="ltx_av_output.mp4")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    # Guidance parameters (defaults from LTX_2_3_PARAMS in official pipeline)
    parser.add_argument("--video-cfg-scale", type=float, default=3.0, help="Video CFG guidance scale")
    parser.add_argument("--audio-cfg-scale", type=float, default=7.0, help="Audio CFG guidance scale")
    parser.add_argument("--video-stg-scale", type=float, default=1.0, help="Video STG guidance scale")
    parser.add_argument("--audio-stg-scale", type=float, default=1.0, help="Audio STG guidance scale")
    parser.add_argument("--video-modality-scale", type=float, default=3.0, help="Video modality (A→V) guidance scale")
    parser.add_argument("--audio-modality-scale", type=float, default=3.0, help="Audio modality (V→A) guidance scale")
    parser.add_argument("--rescale-scale", type=float, default=0.7, help="CFG rescale scale (0=off)")
    parser.add_argument("--stg-block", type=int, default=28, help="Transformer block index for STG perturbation")
    parser.add_argument(
        "--guidance_scale", type=float, default=None, help="(deprecated) Use --video-cfg-scale and --audio-cfg-scale"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gemma_path", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--num_layers", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio_tokens", type=int, default=0, help="Audio tokens (0=auto from video shape)")
    parser.add_argument(
        "--no-cross-pe",
        action="store_true",
        help="Disable cross-modal positional embeddings (workaround for tile alignment)",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt (default: official LTX-2 negative prompt)"
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.path.join(CHECKPOINT_DIR, "ltx-2.3-22b-dev.safetensors")

    logger.info(f"Generating: '{args.prompt}'")
    logger.info(f"Output: {args.num_frames} frames @ {args.height}x{args.width}, {args.steps} steps")

    # Handle deprecated --guidance_scale
    if args.guidance_scale is not None:
        args.video_cfg_scale = args.guidance_scale
        args.audio_cfg_scale = args.guidance_scale

    # 1. Encode text (positive + negative prompts)
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger

    negative_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    t0 = time.time()
    gemma_local = resolve_gemma_path(args.gemma_path)
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        gemma_root_path=gemma_local,
    )
    results = encode_prompts([args.prompt, negative_prompt], ledger)
    video_embeds = results[0].video_encoding.float()
    audio_embeds = results[0].audio_encoding.float() if results[0].audio_encoding is not None else None
    neg_video_embeds = results[1].video_encoding.float()
    neg_audio_embeds = results[1].audio_encoding.float() if results[1].audio_encoding is not None else None
    del ledger
    logger.info(
        f"Text encoding: {time.time()-t0:.1f}s, v={video_embeds.shape}, a={audio_embeds.shape if audio_embeds is not None else None}"
    )

    if audio_embeds is None:
        audio_embeds = torch.zeros(1, video_embeds.shape[1], 2048)
        neg_audio_embeds = torch.zeros(1, neg_video_embeds.shape[1], 2048)
        logger.warning("No audio embeddings from encoder, using zeros")

    # 2. Load AV model on TT mesh
    import ttnn
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    logger.info(f"Opened 2x4 mesh ({mesh.get_num_devices()} devices)")

    sp_axis, tp_axis = 0, 1
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear)

    # Load checkpoint (all keys, no filtering)
    from safetensors.torch import load_file

    t0 = time.time()
    raw = load_file(checkpoint)
    prefix = "model.diffusion_model."
    state_dict = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    logger.info(f"Checkpoint: {len(state_dict)} DiT keys in {time.time()-t0:.1f}s")
    del raw

    t0 = time.time()
    model = LTXTransformerModel(
        num_layers=args.num_layers,
        mesh_device=mesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        has_audio=True,
    )
    model.load_torch_state_dict(state_dict)
    logger.info(f"AV transformer loaded in {time.time()-t0:.1f}s")
    del state_dict

    # 3. Prepare inputs
    latent_frames = (args.num_frames - 1) // 8 + 1
    latent_h = args.height // 32
    latent_w = args.width // 32
    video_N = latent_frames * latent_h * latent_w

    # Compute audio token count from video shape (matching official pipeline)
    if args.audio_tokens > 0:
        audio_N = args.audio_tokens
    else:
        from ltx_core.types import AudioLatentShape, VideoPixelShape

        vps = VideoPixelShape(batch=1, frames=args.num_frames, height=args.height, width=args.width, fps=args.fps)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N = als.frames  # Patchifier folds mel_bins into channels
        logger.info(
            f"Audio latent: {als.frames} frames x {als.mel_bins} mel x {als.channels} ch -> {audio_N} tokens x 128"
        )

    # Pad audio_N to be tile-aligned after SP sharding.
    # Track the real (unpadded) count separately — padded positions need masking.
    sp_factor = tuple(mesh.shape)[sp_axis]
    audio_N_real = audio_N  # Unpadded count from AudioLatentShape
    audio_N_local = audio_N // sp_factor if audio_N % sp_factor == 0 else audio_N
    if audio_N_local % 32 != 0:
        audio_N_padded = ((audio_N_local + 31) // 32 * 32) * sp_factor
        logger.info(f"Padding audio_N from {audio_N} to {audio_N_padded} for tile alignment (real={audio_N_real})")
        audio_N = audio_N_padded

    # Check tile alignment (warning only — TTNN handles non-aligned N transparently)
    for name, N in [("video", video_N), ("audio", audio_N)]:
        n_local = N // sp_factor
        if n_local % 32 != 0:
            logger.warning(f"{name} N_local={n_local} not tile-aligned (N={N}, SP={sp_factor}) — TTNN pads internally")

    logger.info(f"Video: {latent_frames}x{latent_h}x{latent_w} = {video_N} tokens")
    logger.info(f"Audio: {audio_N} tokens")

    # Video RoPE (SPLIT format with double-precision freq grid — matching official pipeline)
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.model.transformer.rope import LTXRopeType as RefLTXRopeType
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute
    from ltx_core.types import VideoLatentShape

    fps = args.fps
    v_patchifier = VideoLatentPatchifier(patch_size=1)
    v_latent_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=latent_h, width=latent_w)
    v_latent_coords = v_patchifier.get_patch_grid_bounds(output_shape=v_latent_shape, device="cpu")
    v_positions = get_pixel_coords(v_latent_coords, scale_factors=(8, 32, 32), causal_fix=True).float()
    v_positions[:, 0, ...] = v_positions[:, 0, ...] / fps

    v_cos, v_sin = ref_precompute(
        v_positions.bfloat16(),
        dim=4096,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    tt_v_cos = bf16_tensor_2dshard(v_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Audio RoPE (1D temporal, SPLIT format with double-precision)
    # Use official AudioPatchifier to compute time-in-seconds positions (not integer indices!)
    from ltx_core.components.patchifiers import AudioPatchifier as _AudioPatchifier

    _a_patchifier = _AudioPatchifier(patch_size=1)
    _a_latent_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N_real, mel_bins=16)
    a_positions = _a_patchifier.get_patch_grid_bounds(
        output_shape=_a_latent_shape, device="cpu"
    ).float()  # (1, 1, N, 2)
    a_cos, a_sin = ref_precompute(
        a_positions.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    # Pad audio RoPE to audio_N (padded) if needed — model processes padded tokens
    if audio_N > audio_N_real:
        a_cos_pad = torch.ones(1, 32, audio_N, a_cos.shape[-1])
        a_cos_pad[:, :, :audio_N_real, :] = a_cos
        a_sin_pad = torch.zeros(1, 32, audio_N, a_sin.shape[-1])
        a_sin_pad[:, :, :audio_N_real, :] = a_sin
        a_cos, a_sin = a_cos_pad, a_sin_pad
    tt_a_cos = bf16_tensor_2dshard(a_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Cross-modal positional embeddings for A↔V cross-attention.
    # Uses temporal positions only, inner_dim=2048 (audio_cross_attention_dim).
    # Reference: transformer_args.py MultiModalTransformerArgsPreprocessor.prepare()
    # NOTE: --no-cross-pe disables this due to TTNN subtile broadcast issue with D_half=32
    use_cross_pe = not getattr(args, "no_cross_pe", False)
    tt_v_cross_cos = tt_v_cross_sin = tt_a_cross_cos = tt_a_cross_sin = None
    tt_v_cross_cos_full = tt_v_cross_sin_full = tt_a_cross_cos_full = tt_a_cross_sin_full = None
    if use_cross_pe:
        cross_pe_max_pos = 20  # max(video_max_pos[0], audio_max_pos[0])
        # Video cross PE: temporal positions only from video coordinates
        v_cross_positions = v_positions[:, 0:1, :]  # (1, 1, video_N, 2) — temporal only
        v_cross_cos, v_cross_sin = ref_precompute(
            v_cross_positions.bfloat16(),
            dim=2048,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=RefLTXRopeType.SPLIT,
            freq_grid_generator=generate_freq_grid_np,
        )
        # Audio cross PE: temporal positions from audio
        a_cross_positions = a_positions  # (1, 1, audio_N, 2) — already temporal only
        a_cross_cos, a_cross_sin = ref_precompute(
            a_cross_positions.bfloat16(),
            dim=2048,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=32,
            rope_type=RefLTXRopeType.SPLIT,
            freq_grid_generator=generate_freq_grid_np,
        )
        # SP+TP sharded (for Q in cross-attention — Q is SP-sharded)
        tt_v_cross_cos = bf16_tensor_2dshard(v_cross_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_v_cross_sin = bf16_tensor_2dshard(v_cross_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_a_cross_cos = bf16_tensor_2dshard(a_cross_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
        tt_a_cross_sin = bf16_tensor_2dshard(a_cross_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
        # TP-only sharded (for K in cross-attention — context is full-sequence replicated)
        tt_v_cross_cos_full = bf16_tensor(v_cross_cos, device=mesh, mesh_axis=tp_axis, shard_dim=1)
        tt_v_cross_sin_full = bf16_tensor(v_cross_sin, device=mesh, mesh_axis=tp_axis, shard_dim=1)
        tt_a_cross_cos_full = bf16_tensor(a_cross_cos, device=mesh, mesh_axis=tp_axis, shard_dim=1)
        tt_a_cross_sin_full = bf16_tensor(a_cross_sin, device=mesh, mesh_axis=tp_axis, shard_dim=1)
        logger.info(f"Cross PE: video={v_cross_cos.shape}, audio={a_cross_cos.shape}")
    else:
        logger.info("Cross PE: disabled (--no-cross-pe)")

    # Push prompts to device
    tt_v_prompt = bf16_tensor(video_embeds.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(audio_embeds.unsqueeze(0), device=mesh)
    do_cfg = args.video_cfg_scale > 1.0 or args.audio_cfg_scale > 1.0
    if do_cfg:
        tt_neg_v_prompt = bf16_tensor(neg_video_embeds.unsqueeze(0), device=mesh)
        tt_neg_a_prompt = bf16_tensor(neg_audio_embeds.unsqueeze(0), device=mesh)

    # Sigma schedule and initial noise
    sigmas = compute_sigmas(steps=args.steps)  # Official uses default (no num_tokens)
    logger.info(f"Sigmas: {sigmas[0]:.4f} -> {sigmas[-1]:.4f}")

    torch.manual_seed(args.seed)
    video_latent = torch.randn(1, video_N, 128, dtype=torch.bfloat16).float() * sigmas[0]
    # Audio: only create noise for real tokens, zero-pad the rest
    audio_noise = torch.randn(1, audio_N_real, 128, dtype=torch.bfloat16).float() * sigmas[0]
    if audio_N > audio_N_real:
        audio_latent = torch.zeros(1, audio_N, 128)
        audio_latent[:, :audio_N_real, :] = audio_noise
    else:
        audio_latent = audio_noise

    # Create audio attention mask for SDPA: mask out padded K positions.
    # When SP > 1, ring attention does not support masks. Instead, K/V are gathered
    # across SP devices so the mask must cover the FULL K sequence (audio_N), not
    # just the local shard. Q dimension uses audio_N_local (local shard size) so
    # the mask shape is (1, 1, audio_N_local, audio_N).
    tt_audio_attn_mask = None
    if audio_N > audio_N_real:
        audio_N_local = audio_N // sp_factor
        mask = torch.zeros(1, 1, audio_N_local, audio_N)
        mask[:, :, :, audio_N_real:] = float("-inf")  # Mask ALL padded K positions
        tt_audio_attn_mask = bf16_tensor(mask, device=mesh)
        logger.info(f"Audio attn mask: {mask.shape}, masking K positions [{audio_N_real}:{audio_N}]")

    # Create audio padding mask for A-to-V cross-attention: zero out padded tokens
    # after SP-gather so they contribute nothing to video cross-attention context.
    # Shape: (1, 1, audio_N, 1) — 1.0 for real tokens, 0.0 for padded.
    tt_audio_padding_mask = None
    if audio_N > audio_N_real:
        pad_mask = torch.ones(1, 1, audio_N, 1)
        pad_mask[:, :, audio_N_real:, :] = 0.0
        tt_audio_padding_mask = bf16_tensor(pad_mask, device=mesh)
        logger.info(f"Audio padding mask: zeroing positions [{audio_N_real}:{audio_N}]")

    # 4. Joint denoising loop
    logger.info(f"Starting AV denoising: {args.steps} steps")
    denoise_start = time.time()

    for step_idx in range(args.steps):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()
        timestep = torch.tensor([sigma])

        # Helper: run model and convert velocity to denoised (X0)
        def velocity_to_denoised(v_out_tt, a_out_tt):
            vv = LTXTransformerModel.device_to_host(v_out_tt).squeeze(0)
            av = LTXTransformerModel.device_to_host(a_out_tt).squeeze(0)
            vd = (video_latent.bfloat16().float() - vv.float() * sigma).bfloat16()
            ad = (audio_latent.bfloat16().float() - av.float() * sigma).bfloat16()
            return vd, ad

        def run_model(v_prompt, a_prompt, skip_cross_attn=False, skip_self_attn_blocks=None):
            return model.inner_step(
                video_1BNI_torch=video_latent.unsqueeze(0),
                video_prompt_1BLP=v_prompt,
                video_rope_cos=tt_v_cos,
                video_rope_sin=tt_v_sin,
                video_N=video_N,
                audio_1BNI_torch=audio_latent.unsqueeze(0),
                audio_prompt_1BLP=a_prompt,
                audio_rope_cos=tt_a_cos,
                audio_rope_sin=tt_a_sin,
                audio_N=audio_N,
                trans_mat=None,
                timestep_torch=timestep,
                video_cross_pe_cos=tt_v_cross_cos,
                video_cross_pe_sin=tt_v_cross_sin,
                audio_cross_pe_cos=tt_a_cross_cos,
                audio_cross_pe_sin=tt_a_cross_sin,
                video_cross_pe_cos_full=tt_v_cross_cos_full,
                video_cross_pe_sin_full=tt_v_cross_sin_full,
                audio_cross_pe_cos_full=tt_a_cross_cos_full,
                audio_cross_pe_sin_full=tt_a_cross_sin_full,
                skip_cross_attn=skip_cross_attn,
                skip_self_attn_blocks=skip_self_attn_blocks,
                audio_attn_mask=tt_audio_attn_mask,
                audio_padding_mask=tt_audio_padding_mask,
            )

        # Pass 1: Conditional (positive prompts)
        v_out, a_out = run_model(tt_v_prompt, tt_a_prompt)
        v_denoised, a_denoised = velocity_to_denoised(v_out, a_out)

        # Multi-modal guidance (matching reference MultiModalGuider.calculate)
        # Each guidance type is independent — CFG, STG, and modality can be used separately.
        # Formula: pred = cond + (cfg-1)*(cond-uncond) + stg*(cond-perturbed) + (mod-1)*(cond-isolated)
        do_stg = args.video_stg_scale != 0.0 or args.audio_stg_scale != 0.0
        do_modality = args.video_modality_scale != 1.0 or args.audio_modality_scale != 1.0
        do_any_guidance = do_cfg or do_stg or do_modality

        # Pass 2: Unconditional (negative prompts) — for CFG
        v_uncond, a_uncond = 0.0, 0.0
        if do_cfg:
            neg_v_out, neg_a_out = run_model(tt_neg_v_prompt, tt_neg_a_prompt)
            v_uncond, a_uncond = velocity_to_denoised(neg_v_out, neg_a_out)

        # Pass 3: Perturbed (skip self-attention at stg_block) — for STG guidance
        v_perturbed, a_perturbed = 0.0, 0.0
        if do_stg:
            stg_v_out, stg_a_out = run_model(tt_v_prompt, tt_a_prompt, skip_self_attn_blocks=[args.stg_block])
            v_perturbed, a_perturbed = velocity_to_denoised(stg_v_out, stg_a_out)

        # Pass 4: Isolated modality (skip A↔V cross-attention) — for modality guidance
        v_isolated, a_isolated = 0.0, 0.0
        if do_modality:
            mod_v_out, mod_a_out = run_model(tt_v_prompt, tt_a_prompt, skip_cross_attn=True)
            v_isolated, a_isolated = velocity_to_denoised(mod_v_out, mod_a_out)

        # Apply full MultiModalGuider formula per modality
        if do_any_guidance:
            v_cond = v_denoised.float()
            v_pred = v_cond
            if do_cfg:
                v_pred = v_pred + (args.video_cfg_scale - 1) * (v_cond - v_uncond.float())
            if do_stg:
                v_pred = v_pred + args.video_stg_scale * (v_cond - v_perturbed.float())
            if do_modality:
                v_pred = v_pred + (args.video_modality_scale - 1) * (v_cond - v_isolated.float())
            if args.rescale_scale != 0:
                v_factor = args.rescale_scale * (v_cond.std() / v_pred.std()) + (1 - args.rescale_scale)
                v_pred = v_pred * v_factor
            v_denoised = v_pred.bfloat16()

            a_cond = a_denoised.float()
            a_pred = a_cond
            if do_cfg:
                a_pred = a_pred + (args.audio_cfg_scale - 1) * (a_cond - a_uncond.float())
            if do_stg:
                a_pred = a_pred + args.audio_stg_scale * (a_cond - a_perturbed.float())
            if do_modality:
                a_pred = a_pred + (args.audio_modality_scale - 1) * (a_cond - a_isolated.float())
            if args.rescale_scale != 0:
                a_factor = args.rescale_scale * (a_cond.std() / a_pred.std()) + (1 - args.rescale_scale)
                a_pred = a_pred * a_factor
            a_denoised = a_pred.bfloat16()

        video_latent = euler_step(video_latent, v_denoised.float(), sigma, sigma_next).bfloat16().float()
        audio_latent = euler_step(audio_latent, a_denoised.float(), sigma, sigma_next).bfloat16().float()
        # Zero out padded audio tokens to prevent drift
        if audio_N > audio_N_real:
            audio_latent[:, audio_N_real:, :] = 0.0

        if (step_idx + 1) % 5 == 0 or step_idx == 0 or step_idx == args.steps - 1:
            elapsed = time.time() - denoise_start
            logger.info(
                f"Step {step_idx+1}/{args.steps}: sigma {sigma:.4f}->{sigma_next:.4f}, "
                f"v=[{video_latent.min():.2f},{video_latent.max():.2f}], "
                f"a=[{audio_latent.min():.2f},{audio_latent.max():.2f}], {elapsed:.1f}s"
            )

    denoise_time = time.time() - denoise_start
    logger.info(f"AV denoising: {denoise_time:.1f}s ({denoise_time/args.steps:.2f}s/step)")
    # Save raw latent for debugging
    torch.save(video_latent, args.output.replace(".mp4", "_video_latent.pt"))
    torch.save(audio_latent, args.output.replace(".mp4", "_audio_latent.pt"))

    # 5. Video VAE decode (on TT device)
    logger.info("Decoding video with TTNN VAE...")
    from models.tt_dit.models.vae.ltx.vae_ltx import LTXVideoDecoder

    # 22B decoder config (from checkpoint metadata)
    decoder_blocks_22b = [
        ("res_x", {"num_layers": 4}),
        ("compress_space", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_time", {"multiplier": 2}),
        ("res_x", {"num_layers": 4}),
        ("compress_all", {"multiplier": 1}),
        ("res_x", {"num_layers": 2}),
        ("compress_all", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
    ]

    # Load VAE weights from checkpoint
    from safetensors.torch import load_file as load_safetensors

    raw_vae = load_safetensors(checkpoint)
    vae_state = {}
    for k, v in raw_vae.items():
        if k.startswith("vae.decoder."):
            vae_state[k[len("vae.decoder.") :]] = v
        elif k.startswith("vae.per_channel_statistics."):
            vae_state[k[len("vae.") :]] = v  # keep "per_channel_statistics.*"
    del raw_vae

    t0 = time.time()
    tt_vae = LTXVideoDecoder(
        decoder_blocks=decoder_blocks_22b,
        causal=False,  # 22B: causal_decoder=False
        mesh_device=mesh,
    )
    tt_vae.load_torch_state_dict(vae_state)
    logger.info(f"TTNN VAE decoder loaded in {time.time()-t0:.1f}s")
    del vae_state

    latent_spatial = video_latent.reshape(1, 128, latent_frames, latent_h, latent_w)
    with torch.no_grad():
        video_pixels = tt_vae(latent_spatial.bfloat16())
    logger.info(f"Video decoded: {video_pixels.shape}")
    del tt_vae

    # Close TT
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # 6. Audio VAE decode + vocoder (stays on CPU torch — different architecture)
    logger.info("Decoding audio...")
    audio_obj = None
    try:
        from ltx_core.model.audio_vae.audio_vae import decode_audio as vae_decode_audio
        from ltx_pipelines.utils.model_ledger import ModelLedger

        ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=checkpoint)
        audio_decoder = ledger.audio_decoder()
        vocoder = ledger.vocoder()

        # Unpatchify audio: (1, audio_N, 128) -> (1, 8, audio_N, 16) matching decoder's expected (B, C, F, mel_bins)
        audio_spatial = audio_latent.reshape(1, audio_N, 8, 16).permute(0, 2, 1, 3).bfloat16()  # (1, 8, F, 16)

        with torch.no_grad():
            audio_obj = vae_decode_audio(audio_spatial, audio_decoder, vocoder)

        # Trim to video duration if padded audio_N produced extra samples
        video_duration = args.num_frames / fps
        target_samples = int(video_duration * audio_obj.sampling_rate)
        if audio_obj.waveform.shape[-1] > target_samples:
            from ltx_core.types import Audio

            audio_obj = Audio(waveform=audio_obj.waveform[..., :target_samples], sampling_rate=audio_obj.sampling_rate)
        logger.info(
            f"Audio decoded: {audio_obj.waveform.shape} ({audio_obj.waveform.shape[-1]/audio_obj.sampling_rate:.2f}s @ {audio_obj.sampling_rate}Hz)"
        )
    except Exception as e:
        logger.warning(f"Audio decode failed ({e}), video only")
        audio_obj = None

    # 7. Export video + audio combined (matching reference: ltx_pipelines.utils.media_io.encode_video)
    video_pixels = video_pixels.float().clamp(-1, 1)
    video_pixels = (video_pixels + 1) / 2
    video_uint8 = (video_pixels[0].permute(1, 2, 3, 0) * 255).to(torch.uint8)  # (F, H, W, 3)

    if audio_obj is not None:
        try:
            from ltx_pipelines.utils.media_io import encode_video

            encode_video(
                video=video_uint8,
                fps=fps,
                audio=audio_obj,
                output_path=args.output,
                video_chunks_number=1,
            )
            logger.info(f"Video+audio saved to {os.path.abspath(args.output)}")
        except Exception as e:
            logger.warning(f"Combined export failed ({e}), saving separately")
            audio_obj = None

    if audio_obj is None:
        import imageio

        video_np = video_uint8.numpy()
        imageio.mimwrite(args.output, video_np, fps=fps, codec="libx264")
        logger.info(f"Video saved to {os.path.abspath(args.output)}")

    logger.info(f"Done! Denoise: {denoise_time:.1f}s")


if __name__ == "__main__":
    main()
