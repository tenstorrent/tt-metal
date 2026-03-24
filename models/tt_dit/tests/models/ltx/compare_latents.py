#!/usr/bin/env python3
"""
Step-by-step latent comparison between TTNN and CPU reference pipelines.

Runs both pipelines with identical inputs and compares the denoised latents
at each step to find where prompt adherence diverges.
"""
import os
import sys

import torch
from loguru import logger

torch.cuda.synchronize = lambda *a, **kw: None  # No CUDA

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT = os.environ.get("LTX_CHECKPOINT", "/localdev/kevinmi/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
GEMMA = os.environ.get(
    "GEMMA_PATH",
    "/localdev/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
)

PROMPT = "A cat playing piano in a cozy room with warm lighting"
SEED = 10
STEPS = 2
HEIGHT, WIDTH = 512, 768
NUM_FRAMES = 33


def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()
    num = (a_mean * b_mean).sum()
    den = a_mean.norm() * b_mean.norm()
    return (num / den).item() if den > 0 else float("nan")


def run_cpu_reference(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds):
    """Run CPU reference pipeline, return per-step denoised latents."""
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import Modality
    from ltx_core.types import AudioLatentShape, VideoPixelShape
    from ltx_pipelines.utils.model_ledger import ModelLedger

    latent_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_h = HEIGHT // 32
    latent_w = WIDTH // 32
    video_N = latent_frames * latent_h * latent_w

    vps = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N = als.frames

    # Sigma schedule
    scheduler = LTX2Scheduler()
    dummy_latent = torch.randn(1, 1, video_N)
    sigmas = scheduler.execute(steps=STEPS, latent=dummy_latent)
    logger.info(f"CPU sigmas: {sigmas}")

    # Initial noise (same seed)
    torch.manual_seed(SEED)
    video_latent = torch.randn(1, video_N, 128) * sigmas[0]
    audio_latent = torch.randn(1, audio_N, 128) * sigmas[0]

    # Load reference model (X0Model wraps LTXModel)
    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA
    )
    transformer = ledger.transformer()  # Returns X0Model

    # Positions
    t_ids = torch.arange(latent_frames)
    h_ids = torch.arange(latent_h)
    w_ids = torch.arange(latent_w)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_positions = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0).float()
    v_positions = torch.stack([v_positions, v_positions], dim=-1).unsqueeze(0)

    from ltx_core.components.patchifiers import AudioPatchifier as _AudioPatchifier

    _a_patchifier = _AudioPatchifier(patch_size=1)
    _a_latent_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N, mel_bins=16)
    a_positions_raw = _a_patchifier.get_patch_grid_bounds(output_shape=_a_latent_shape, device="cpu")

    perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])

    step_latents = []
    euler = EulerDiffusionStep()

    for step_idx in range(STEPS):
        sigma = sigmas[step_idx]

        video_mod = Modality(
            latent=video_latent.bfloat16(),
            sigma=sigma.unsqueeze(0),
            timesteps=torch.ones(1, video_N) * sigma,
            positions=v_positions,
            context=video_embeds.bfloat16(),
            context_mask=None,
            attention_mask=None,
        )
        audio_mod = Modality(
            latent=audio_latent.bfloat16(),
            sigma=sigma.unsqueeze(0),
            timesteps=torch.ones(1, audio_N) * sigma,
            positions=a_positions_raw,
            context=audio_embeds.bfloat16(),
            context_mask=None,
            attention_mask=None,
        )

        with torch.no_grad():
            v_denoised, a_denoised = transformer(video=video_mod, audio=audio_mod, perturbations=perturbations)

        step_latents.append(
            {
                "step": step_idx,
                "sigma": sigma.item(),
                "v_denoised": v_denoised.clone(),
                "a_denoised": a_denoised.clone(),
                "v_latent": video_latent.clone(),
                "a_latent": audio_latent.clone(),
            }
        )

        # Euler step
        video_latent = (
            euler.step(video_latent.bfloat16().float(), v_denoised.float(), sigmas, step_idx).bfloat16().float()
        )
        audio_latent = (
            euler.step(audio_latent.bfloat16().float(), a_denoised.float(), sigmas, step_idx).bfloat16().float()
        )

        logger.info(
            f"CPU step {step_idx}: sigma={sigma:.4f}, v=[{video_latent.min():.2f},{video_latent.max():.2f}], a=[{audio_latent.min():.2f},{audio_latent.max():.2f}]"
        )

    del transformer
    return step_latents, sigmas


def run_ttnn_pipeline(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds, sigmas_ref):
    """Run TTNN pipeline and return per-step denoised latents."""
    from safetensors.torch import load_file

    import ttnn
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step
    from models.tt_dit.utils.tensor import bf16_tensor

    latent_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_h = HEIGHT // 32
    latent_w = WIDTH // 32
    video_N = latent_frames * latent_h * latent_w

    from ltx_core.types import AudioLatentShape, VideoPixelShape

    vps = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N_real = als.frames
    audio_N = ((audio_N_real + 31) // 32) * 32

    # Use same sigmas as reference
    sigmas = compute_sigmas(steps=STEPS)
    logger.info(f"TTNN sigmas: {sigmas}")
    logger.info(f"Sigma diff: {(sigmas - sigmas_ref).abs().max():.6f}")

    # Same initial noise
    torch.manual_seed(SEED)
    video_latent = torch.randn(1, video_N, 128, dtype=torch.bfloat16).float() * sigmas[0]
    audio_latent_real = torch.randn(1, audio_N_real, 128, dtype=torch.bfloat16).float() * sigmas[0]
    # Pad audio
    audio_latent = torch.zeros(1, audio_N, 128)
    audio_latent[:, :audio_N_real, :] = audio_latent_real

    # Open mesh
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), l1_small_size=65536)
    sp_axis, tp_axis = 0, 1

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear)

    # Load model
    raw = load_file(CHECKPOINT)
    prefix = "model.diffusion_model."
    state_dict = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    del raw

    model = LTXTransformerModel(
        num_layers=48, mesh_device=mesh, ccl_manager=ccl_manager, parallel_config=parallel_config, has_audio=True
    )
    model.load_torch_state_dict(state_dict)
    del state_dict

    # Prepare embeddings
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    # Video RoPE
    t_ids = torch.arange(latent_frames)
    h_ids = torch.arange(latent_h)
    w_ids = torch.arange(latent_w)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    v_cos, v_sin = precompute_freqs_cis(
        v_grid, dim=4096, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=32
    )
    num_heads = 32
    head_dim = 128
    v_cos_h = v_cos.reshape(1, video_N, num_heads, head_dim).permute(0, 2, 1, 3)
    v_sin_h = v_sin.reshape(1, video_N, num_heads, head_dim).permute(0, 2, 1, 3)

    from models.tt_dit.utils.tensor import bf16_tensor_2dshard

    tt_v_cos = bf16_tensor_2dshard(v_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Audio RoPE
    from ltx_core.model.audio_vae.audio_vae import AudioPatchifier

    a_positions = AudioPatchifier.get_patch_grid_bounds(audio_N_real, 48000, 24)
    a_padded = torch.zeros(audio_N, 2)
    a_padded[:audio_N_real] = a_positions
    a_cos, a_sin = precompute_freqs_cis(
        a_padded.unsqueeze(0), dim=2048, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=32
    )
    a_heads = 32
    a_hdim = 64
    a_cos_h = a_cos.reshape(1, audio_N, a_heads, a_hdim).permute(0, 2, 1, 3)
    a_sin_h = a_sin.reshape(1, audio_N, a_heads, a_hdim).permute(0, 2, 1, 3)
    tt_a_cos = bf16_tensor_2dshard(a_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Prompts
    tt_v_prompt = bf16_tensor(video_embeds.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(audio_embeds.unsqueeze(0), device=mesh)

    sp_factor = tuple(mesh.shape)[sp_axis]

    # No guidance for clean comparison — just conditional pass
    step_latents = []
    for step_idx in range(STEPS):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()
        timestep = torch.tensor([sigma])

        v_out, a_out = model.inner_step(
            video_1BNI_torch=video_latent.unsqueeze(0),
            video_prompt_1BLP=tt_v_prompt,
            video_rope_cos=tt_v_cos,
            video_rope_sin=tt_v_sin,
            video_N=video_N,
            audio_1BNI_torch=audio_latent.unsqueeze(0),
            audio_prompt_1BLP=tt_a_prompt,
            audio_rope_cos=tt_a_cos,
            audio_rope_sin=tt_a_sin,
            audio_N=audio_N,
            trans_mat=None,
            timestep_torch=timestep,
        )

        vv = LTXTransformerModel.device_to_host(v_out).squeeze(0)
        av = LTXTransformerModel.device_to_host(a_out).squeeze(0)
        v_denoised = (video_latent.bfloat16().float() - vv.float() * sigma).bfloat16()
        a_denoised = (audio_latent.bfloat16().float() - av.float() * sigma).bfloat16()

        step_latents.append(
            {
                "step": step_idx,
                "sigma": sigma,
                "v_denoised": v_denoised.clone(),
                "a_denoised": a_denoised.clone()[:, :audio_N_real, :],
                "v_latent": video_latent.clone(),
                "a_latent": audio_latent.clone()[:, :audio_N_real, :],
            }
        )

        video_latent = euler_step(video_latent, v_denoised.float(), sigma, sigma_next).bfloat16().float()
        audio_latent_new = euler_step(audio_latent, a_denoised.float(), sigma, sigma_next).bfloat16().float()
        audio_latent = torch.zeros_like(audio_latent)
        audio_latent[:, :audio_N_real, :] = audio_latent_new[:, :audio_N_real, :]

        logger.info(f"TT step {step_idx}: sigma={sigma:.4f}, v=[{video_latent.min():.2f},{video_latent.max():.2f}]")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return step_latents


def main():
    # 1. Encode text (shared between both pipelines)
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger

    logger.info(f"Encoding prompts...")
    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA
    )
    results = encode_prompts([PROMPT, DEFAULT_NEGATIVE_PROMPT], ledger)
    video_embeds = results[0].video_encoding.float()
    audio_embeds = (
        results[0].audio_encoding.float()
        if results[0].audio_encoding is not None
        else torch.zeros(1, video_embeds.shape[1], 2048)
    )
    neg_video_embeds = results[1].video_encoding.float()
    neg_audio_embeds = (
        results[1].audio_encoding.float()
        if results[1].audio_encoding is not None
        else torch.zeros(1, neg_video_embeds.shape[1], 2048)
    )
    del ledger
    logger.info(f"Embeddings: v={video_embeds.shape}, a={audio_embeds.shape}")

    # 2. Run CPU reference (conditional only, no guidance, to isolate transformer comparison)
    logger.info("=== Running CPU reference ===")
    cpu_latents, sigmas_ref = run_cpu_reference(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds)

    # 3. Run TTNN
    logger.info("=== Running TTNN ===")
    tt_latents = run_ttnn_pipeline(video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds, sigmas_ref)

    # 4. Compare
    logger.info("=== Step-by-step comparison ===")
    for cpu_step, tt_step in zip(cpu_latents, tt_latents):
        step = cpu_step["step"]
        v_pcc = pcc(cpu_step["v_denoised"], tt_step["v_denoised"])
        a_pcc = pcc(cpu_step["a_denoised"], tt_step["a_denoised"])
        v_latent_pcc = pcc(cpu_step["v_latent"], tt_step["v_latent"])
        logger.info(
            f"Step {step} (sigma={cpu_step['sigma']:.4f}): video_denoised PCC={v_pcc:.6f}, audio_denoised PCC={a_pcc:.6f}, video_latent PCC={v_latent_pcc:.6f}"
        )

    # Save for further analysis
    torch.save({"cpu": cpu_latents, "tt": tt_latents}, "/tmp/latent_comparison.pt")
    logger.info("Saved comparison to /tmp/latent_comparison.pt")


if __name__ == "__main__":
    main()
