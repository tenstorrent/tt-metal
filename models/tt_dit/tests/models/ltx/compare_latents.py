#!/usr/bin/env python3
"""
Step-by-step latent comparison: TTNN vs CPU reference at 256x256.
Uses the OFFICIAL reference pipeline to avoid any manual setup divergence.
"""
import sys
import time

import torch
from loguru import logger

torch.cuda.synchronize = lambda *a, **kw: None

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT = "/localdev/kevinmi/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"
GEMMA = "/localdev/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
PROMPT = "A cat playing piano in a cozy room with warm lighting"
SEED = 10
STEPS = 2
HEIGHT, WIDTH = 256, 256
NUM_FRAMES = 33


def pcc(a, b):
    af, bf = a.flatten().float(), b.flatten().float()
    am, bm = af - af.mean(), bf - bf.mean()
    return ((am * bm).sum() / (am.norm() * bm.norm())).item()


def run_cpu_reference(v_ctx, a_ctx, neg_v_ctx, neg_a_ctx):
    """Run the official reference pipeline and capture per-step latents."""
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import create_multimodal_guider_factory
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.constants import LTX_2_3_PARAMS
    from ltx_pipelines.utils.helpers import denoise_audio_video, multi_modal_guider_factory_denoising_func
    from ltx_pipelines.utils.model_ledger import ModelLedger

    sigmas = LTX2Scheduler().execute(steps=STEPS).float()
    logger.info(f"CPU sigmas: {sigmas}")

    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA
    )
    transformer = ledger.transformer()
    components = ledger.pipeline_components()

    v_guider = create_multimodal_guider_factory(
        LTX_2_3_PARAMS.video_guider_params, negative_context=neg_v_ctx.bfloat16()
    )
    a_guider = create_multimodal_guider_factory(
        LTX_2_3_PARAMS.audio_guider_params, negative_context=neg_a_ctx.bfloat16()
    )

    denoise_fn = multi_modal_guider_factory_denoising_func(
        video_guider_factory=v_guider,
        audio_guider_factory=a_guider,
        v_context=v_ctx.bfloat16(),
        a_context=a_ctx.bfloat16(),
        transformer=transformer,
    )

    step_data = []

    def traced_loop(sigmas, vs, as_, stepper):
        from dataclasses import replace

        from ltx_pipelines.utils.helpers import post_process_latent

        for step_idx in range(len(sigmas) - 1):
            denoised_v, denoised_a = denoise_fn(vs, as_, sigmas, step_idx)
            denoised_v = post_process_latent(denoised_v, vs.denoise_mask, vs.clean_latent)
            denoised_a = post_process_latent(denoised_a, as_.denoise_mask, as_.clean_latent)

            step_data.append(
                {
                    "step": step_idx,
                    "sigma": sigmas[step_idx].item(),
                    "v_denoised": denoised_v.clone(),
                    "a_denoised": denoised_a.clone(),
                    "v_latent": vs.latent.clone(),
                    "a_latent": as_.latent.clone(),
                }
            )
            logger.info(
                f"CPU step {step_idx}: sigma={sigmas[step_idx]:.4f}, "
                f"v_den=[{denoised_v.min():.2f},{denoised_v.max():.2f}], "
                f"a_den=[{denoised_a.min():.2f},{denoised_a.max():.2f}]"
            )

            vs = replace(vs, latent=stepper.step(vs.latent, denoised_v, sigmas, step_idx))
            as_ = replace(as_, latent=stepper.step(as_.latent, denoised_a, sigmas, step_idx))
        return vs, as_

    output_shape = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    noiser = GaussianNoiser()
    stepper = EulerDiffusionStep()

    # Use the seed
    torch.manual_seed(SEED)
    with torch.no_grad():
        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=[],
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=traced_loop,
            components=components,
            dtype=torch.bfloat16,
        )

    del transformer
    return step_data, sigmas, video_state, audio_state


def run_ttnn_pipeline(v_ctx, a_ctx, neg_v_ctx, neg_a_ctx, ref_sigmas):
    """Run TTNN pipeline and capture per-step latents."""
    from safetensors.torch import load_file

    import ttnn
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import compute_sigmas, euler_step
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    latent_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_h, latent_w = HEIGHT // 32, WIDTH // 32
    video_N = latent_frames * latent_h * latent_w

    from ltx_core.model.audio_vae.audio_vae import AudioPatchifier
    from ltx_core.types import AudioLatentShape, VideoPixelShape

    vps = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N_real = als.frames
    sp_factor = 2
    audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

    sigmas = compute_sigmas(steps=STEPS)
    logger.info(f"TT sigmas: {sigmas}, diff from ref: {(sigmas - ref_sigmas).abs().max():.8f}")

    # Same noise
    torch.manual_seed(SEED)
    video_latent = torch.randn(1, video_N, 128, dtype=torch.bfloat16).float() * sigmas[0]
    audio_latent_real = torch.randn(1, audio_N_real, 128, dtype=torch.bfloat16).float() * sigmas[0]
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

    # RoPE
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    t_ids = torch.arange(latent_frames)
    h_ids = torch.arange(latent_h)
    w_ids = torch.arange(latent_w)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    v_cos, v_sin = precompute_freqs_cis(
        v_grid, dim=4096, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=32
    )
    v_cos_h = v_cos.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    v_sin_h = v_sin.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    tt_v_cos = bf16_tensor_2dshard(v_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    a_positions = AudioPatchifier.get_patch_grid_bounds(audio_N_real, 48000, 24)
    a_padded = torch.zeros(audio_N, 2)
    a_padded[:audio_N_real] = a_positions
    a_cos, a_sin = precompute_freqs_cis(
        a_padded.unsqueeze(0), dim=2048, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=32
    )
    a_cos_h = a_cos.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)
    a_sin_h = a_sin.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)
    tt_a_cos = bf16_tensor_2dshard(a_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Audio masks

    tt_audio_attn_mask = None
    tt_audio_padding_mask = None
    if audio_N > audio_N_real:
        audio_N_local = audio_N // sp_factor
        mask = torch.zeros(1, 1, audio_N_local, audio_N)
        mask[:, :, :, audio_N_real:] = float("-inf")
        tt_audio_attn_mask = bf16_tensor(mask, device=mesh)
        pad_mask = torch.ones(1, 1, audio_N, 1)
        pad_mask[:, :, audio_N_real:, :] = 0.0
        tt_audio_padding_mask = bf16_tensor(pad_mask, device=mesh)

    # Prompts
    tt_v_prompt = bf16_tensor(v_ctx.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(a_ctx.unsqueeze(0), device=mesh)
    tt_neg_v = bf16_tensor(neg_v_ctx.unsqueeze(0), device=mesh)
    tt_neg_a = bf16_tensor(neg_a_ctx.unsqueeze(0), device=mesh)

    def run_model(vp, ap, skip_cross_attn=False, skip_self_attn_blocks=None):
        return model.inner_step(
            video_1BNI_torch=video_latent.unsqueeze(0),
            video_prompt_1BLP=vp,
            video_rope_cos=tt_v_cos,
            video_rope_sin=tt_v_sin,
            video_N=video_N,
            audio_1BNI_torch=audio_latent.unsqueeze(0),
            audio_prompt_1BLP=ap,
            audio_rope_cos=tt_a_cos,
            audio_rope_sin=tt_a_sin,
            audio_N=audio_N,
            trans_mat=None,
            timestep_torch=torch.tensor([sigma]),
            skip_cross_attn=skip_cross_attn,
            skip_self_attn_blocks=skip_self_attn_blocks,
            audio_attn_mask=tt_audio_attn_mask,
            audio_padding_mask=tt_audio_padding_mask,
        )

    args_cfg_v, args_cfg_a = 3.0, 7.0
    args_stg_v, args_stg_a = 1.0, 1.0
    args_mod_v, args_mod_a = 3.0, 3.0
    args_rescale = 0.7

    step_data = []
    for step_idx in range(STEPS):
        sigma = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()

        # Conditional pass
        v_out, a_out = run_model(tt_v_prompt, tt_a_prompt)
        vv = LTXTransformerModel.device_to_host(v_out).squeeze(0)
        av = LTXTransformerModel.device_to_host(a_out).squeeze(0)
        v_denoised = (video_latent.bfloat16().float() - vv.float() * sigma).bfloat16()
        a_denoised = (audio_latent.bfloat16().float() - av.float() * sigma).bfloat16()

        # CFG
        neg_v_out, neg_a_out = run_model(tt_neg_v, tt_neg_a)
        nv = LTXTransformerModel.device_to_host(neg_v_out).squeeze(0)
        na = LTXTransformerModel.device_to_host(neg_a_out).squeeze(0)
        v_uncond = (video_latent.bfloat16().float() - nv.float() * sigma).bfloat16()
        a_uncond = (audio_latent.bfloat16().float() - na.float() * sigma).bfloat16()

        # STG
        stg_v_out, stg_a_out = run_model(tt_v_prompt, tt_a_prompt, skip_self_attn_blocks=[28])
        sv = LTXTransformerModel.device_to_host(stg_v_out).squeeze(0)
        sa = LTXTransformerModel.device_to_host(stg_a_out).squeeze(0)
        v_perturbed = (video_latent.bfloat16().float() - sv.float() * sigma).bfloat16()
        a_perturbed = (audio_latent.bfloat16().float() - sa.float() * sigma).bfloat16()

        # Modality
        mod_v_out, mod_a_out = run_model(tt_v_prompt, tt_a_prompt, skip_cross_attn=True)
        mv = LTXTransformerModel.device_to_host(mod_v_out).squeeze(0)
        ma = LTXTransformerModel.device_to_host(mod_a_out).squeeze(0)
        v_isolated = (video_latent.bfloat16().float() - mv.float() * sigma).bfloat16()
        a_isolated = (audio_latent.bfloat16().float() - ma.float() * sigma).bfloat16()

        # Apply guidance
        vc = v_denoised.float()
        v_pred = (
            vc
            + (args_cfg_v - 1) * (vc - v_uncond.float())
            + args_stg_v * (vc - v_perturbed.float())
            + (args_mod_v - 1) * (vc - v_isolated.float())
        )
        v_factor = args_rescale * (vc.std() / v_pred.std()) + (1 - args_rescale)
        v_pred = v_pred * v_factor

        ac = a_denoised.float()
        a_pred = (
            ac
            + (args_cfg_a - 1) * (ac - a_uncond.float())
            + args_stg_a * (ac - a_perturbed.float())
            + (args_mod_a - 1) * (ac - a_isolated.float())
        )
        a_factor = args_rescale * (ac.std() / a_pred.std()) + (1 - args_rescale)
        a_pred = a_pred * a_factor

        step_data.append(
            {
                "step": step_idx,
                "sigma": sigma,
                "v_denoised": v_pred.bfloat16().clone(),
                "a_denoised": a_pred.bfloat16().clone()[:, :audio_N_real, :],
                "v_latent": video_latent.clone(),
                "a_latent": audio_latent.clone()[:, :audio_N_real, :],
            }
        )

        video_latent = euler_step(video_latent, v_pred.bfloat16().float(), sigma, sigma_next).bfloat16().float()
        audio_latent_new = euler_step(audio_latent, a_pred.bfloat16().float(), sigma, sigma_next).bfloat16().float()
        audio_latent = torch.zeros_like(audio_latent)
        audio_latent[:, :audio_N_real, :] = audio_latent_new[:, :audio_N_real, :]

        logger.info(f"TT step {step_idx}: sigma={sigma:.4f}, v=[{video_latent.min():.2f},{video_latent.max():.2f}]")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return step_data


def main():
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger

    logger.info("Encoding prompts...")
    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA
    )
    results = encode_prompts([PROMPT, DEFAULT_NEGATIVE_PROMPT], ledger)
    v_ctx = results[0].video_encoding.float()
    a_ctx = (
        results[0].audio_encoding.float()
        if results[0].audio_encoding is not None
        else torch.zeros(1, v_ctx.shape[1], 2048)
    )
    neg_v = results[1].video_encoding.float()
    neg_a = (
        results[1].audio_encoding.float()
        if results[1].audio_encoding is not None
        else torch.zeros(1, neg_v.shape[1], 2048)
    )
    del ledger
    logger.info(f"Embeddings: v={v_ctx.shape}, a={a_ctx.shape}")

    # CPU reference
    logger.info("=== CPU reference (256x256, 48L, 2 steps, with guidance) ===")
    t0 = time.time()
    cpu_data, ref_sigmas, _, _ = run_cpu_reference(v_ctx, a_ctx, neg_v, neg_a)
    logger.info(f"CPU done in {time.time()-t0:.0f}s")

    # TTNN
    logger.info("=== TTNN (256x256, 48L, 2 steps, with guidance) ===")
    tt_data = run_ttnn_pipeline(v_ctx, a_ctx, neg_v, neg_a, ref_sigmas)

    # Compare
    logger.info("=== Comparison ===")
    for cpu_s, tt_s in zip(cpu_data, tt_data):
        step = cpu_s["step"]
        # Video denoised (after guidance)
        v_pcc_d = pcc(cpu_s["v_denoised"], tt_s["v_denoised"])
        a_pcc_d = pcc(cpu_s["a_denoised"], tt_s["a_denoised"])
        # Input latent
        v_pcc_l = pcc(cpu_s["v_latent"], tt_s["v_latent"])
        a_pcc_l = pcc(cpu_s["a_latent"], tt_s["a_latent"])
        logger.info(
            f"Step {step} (σ={cpu_s['sigma']:.4f}): "
            f"video_den PCC={v_pcc_d:.6f} audio_den PCC={a_pcc_d:.6f} | "
            f"video_lat PCC={v_pcc_l:.6f} audio_lat PCC={a_pcc_l:.6f}"
        )

    torch.save({"cpu": cpu_data, "tt": tt_data}, "/tmp/latent_comparison.pt")
    logger.info("Saved to /tmp/latent_comparison.pt")


if __name__ == "__main__":
    main()
