#!/usr/bin/env python3
"""
Single-step velocity comparison: TTNN vs CPU reference at 256x256.
Compares raw model velocity output (no guidance) to isolate transformer divergence.
"""
import sys
import time

import torch
from loguru import logger

torch.cuda.synchronize = lambda *a, **kw: None

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")

CHECKPOINT = "/localdev/kevinmi/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"
GEMMA = (
    "/localdev/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it/"
    "snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
)
PROMPT = "A cat playing piano in a cozy room with warm lighting"
SEED = 10
HEIGHT, WIDTH = 256, 256
NUM_FRAMES = 33


def pcc(a, b):
    af, bf = a.flatten().float(), b.flatten().float()
    am, bm = af - af.mean(), bf - bf.mean()
    d = am.norm() * bm.norm()
    return ((am * bm).sum() / d).item() if d > 0 else float("nan")


def main():
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger

    logger.info("Encoding prompts...")
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=CHECKPOINT,
        gemma_root_path=GEMMA,
    )
    results = encode_prompts([PROMPT, DEFAULT_NEGATIVE_PROMPT], ledger)
    v_ctx = results[0].video_encoding
    a_ctx = results[0].audio_encoding
    del results

    latent_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_h, latent_w = HEIGHT // 32, WIDTH // 32
    video_N = latent_frames * latent_h * latent_w

    from ltx_core.types import AudioLatentShape, VideoPixelShape

    vps = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N = als.frames

    sigma = 1.0
    torch.manual_seed(SEED)
    video_latent = torch.randn(1, video_N, 128, dtype=torch.bfloat16)
    audio_latent = torch.randn(1, audio_N, 128, dtype=torch.bfloat16)

    # ===== CPU Reference =====
    logger.info(f"CPU reference: {video_N} video tokens, {audio_N} audio tokens, 48 layers")
    # Use LTXModel directly (not X0Model which has broadcasting issues in to_denoised)
    transformer = ledger.transformer().velocity_model  # unwrap X0Model → LTXModel

    from ltx_core.components.patchifiers import AudioPatchifier as APatch
    from ltx_core.model.transformer.model import Modality

    a_patch = APatch(patch_size=1)
    a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N, mel_bins=16)
    a_coords = a_patch.get_patch_grid_bounds(output_shape=a_shape, device="cpu")

    # Video positions — use official patchifier for correct grid bounds
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.types import VideoLatentShape

    v_patch = VideoLatentPatchifier(patch_size=1)
    v_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=latent_h, width=latent_w)
    v_positions = v_patch.get_patch_grid_bounds(output_shape=v_shape, device="cpu")

    # LTXModel expects patchified latents (1, N, C)
    video_mod = Modality(
        latent=video_latent * sigma,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, video_N) * sigma,
        positions=v_positions,
        context=v_ctx,
        context_mask=None,
    )
    audio_mod = Modality(
        latent=audio_latent * sigma,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, audio_N) * sigma,
        positions=a_coords,
        context=a_ctx,
        context_mask=None,
    )

    t0 = time.time()
    with torch.no_grad():
        ref_v_vel, ref_a_vel = transformer(video=video_mod, audio=audio_mod, perturbations=None)
    cpu_time = time.time() - t0
    # Convert velocity to denoised: denoised = sample - velocity * sigma
    ref_v_denoised = (video_latent * sigma).float() - ref_v_vel.float() * sigma
    ref_a_denoised = (audio_latent * sigma).float() - ref_a_vel.float() * sigma
    logger.info(
        f"CPU done in {cpu_time:.0f}s. "
        f"v_vel=[{ref_v_vel.min():.3f},{ref_v_vel.max():.3f}], "
        f"a_vel=[{ref_a_vel.min():.3f},{ref_a_vel.max():.3f}]"
    )

    del transformer, video_mod, audio_mod, ledger

    # ===== TTNN =====
    logger.info("TTNN forward...")
    from safetensors.torch import load_file

    import ttnn
    from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    sp_factor = 2
    audio_N_padded = ((audio_N + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
    audio_latent_padded = torch.zeros(1, audio_N_padded, 128, dtype=torch.bfloat16)
    audio_latent_padded[:, :audio_N, :] = audio_latent

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), l1_small_size=65536)
    sp_axis, tp_axis = 0, 1

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear)

    raw = load_file(CHECKPOINT)
    prefix = "model.diffusion_model."
    state_dict = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    del raw

    model = LTXTransformerModel(
        num_layers=48,
        mesh_device=mesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        has_audio=True,
    )
    model.load_torch_state_dict(state_dict)
    del state_dict

    # Video RoPE
    t_ids, h_ids, w_ids = torch.arange(latent_frames), torch.arange(latent_h), torch.arange(latent_w)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_grid = torch.stack([gt.flatten(), gh.flatten(), gw.flatten()], dim=-1).float().unsqueeze(0)
    v_cos, v_sin = precompute_freqs_cis(
        v_grid, dim=4096, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=32
    )
    v_cos_h = v_cos.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    v_sin_h = v_sin.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    tt_v_cos = bf16_tensor_2dshard(v_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Audio RoPE
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute_freqs_cis

    _ap = AudioPatchifier(patch_size=1)
    _a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N, mel_bins=16)
    a_positions = _ap.get_patch_grid_bounds(output_shape=_a_shape, device="cpu").float()  # (1, 1, audio_N, 2)

    # Audio RoPE uses reference precompute with 1D max_pos (same as generate_audio_video.py)
    a_cos_raw, a_sin_raw = ref_precompute_freqs_cis(
        a_positions.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )  # (1, 32, audio_N, D_half)

    # Pad to audio_N_padded
    if audio_N_padded > audio_N:
        a_cos_h = torch.ones(1, 32, audio_N_padded, a_cos_raw.shape[-1])
        a_cos_h[:, :, :audio_N, :] = a_cos_raw
        a_sin_h = torch.zeros(1, 32, audio_N_padded, a_sin_raw.shape[-1])
        a_sin_h[:, :, :audio_N, :] = a_sin_raw
    else:
        a_cos_h = a_cos_raw
        a_sin_h = a_sin_raw
    tt_a_cos = bf16_tensor_2dshard(a_cos_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin_h, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    tt_v_prompt = bf16_tensor(v_ctx.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(a_ctx.unsqueeze(0), device=mesh)

    # Audio masks
    tt_attn_mask = tt_pad_mask = None
    if audio_N_padded > audio_N:
        audio_N_local = audio_N_padded // sp_factor
        m = torch.zeros(1, 1, audio_N_local, audio_N_padded)
        m[:, :, :, audio_N:] = float("-inf")
        tt_attn_mask = bf16_tensor(m, device=mesh)
        pm = torch.ones(1, 1, audio_N_padded, 1)
        pm[:, :, audio_N:, :] = 0.0
        tt_pad_mask = bf16_tensor(pm, device=mesh)

    t0 = time.time()
    v_out, a_out = model.inner_step(
        video_1BNI_torch=(video_latent * sigma).unsqueeze(0).float(),
        video_prompt_1BLP=tt_v_prompt,
        video_rope_cos=tt_v_cos,
        video_rope_sin=tt_v_sin,
        video_N=video_N,
        audio_1BNI_torch=(audio_latent_padded * sigma).unsqueeze(0).float(),
        audio_prompt_1BLP=tt_a_prompt,
        audio_rope_cos=tt_a_cos,
        audio_rope_sin=tt_a_sin,
        audio_N=audio_N_padded,
        trans_mat=None,
        timestep_torch=torch.tensor([sigma]),
        audio_attn_mask=tt_attn_mask,
        audio_padding_mask=tt_pad_mask,
    )
    tt_time = time.time() - t0

    vv = LTXTransformerModel.device_to_host(v_out).squeeze(0)
    av = LTXTransformerModel.device_to_host(a_out).squeeze(0)[:, :audio_N, :]

    # Model output is velocity. denoised = sample - velocity * sigma
    tt_v_denoised = (video_latent * sigma).float() - vv.float() * sigma
    tt_a_denoised = (audio_latent * sigma).float() - av.float() * sigma

    logger.info(
        f"TTNN done in {tt_time:.1f}s. "
        f"v_den=[{tt_v_denoised.min():.3f},{tt_v_denoised.max():.3f}], "
        f"a_den=[{tt_a_denoised.min():.3f},{tt_a_denoised.max():.3f}]"
    )

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # ===== Compare velocities =====
    v_vel_pcc = pcc(ref_v_vel, vv)
    a_vel_pcc = pcc(ref_a_vel, av)
    logger.info(f"=== 48-layer VELOCITY PCC (256x256, sigma=1.0) ===")
    logger.info(f"Video velocity PCC: {v_vel_pcc:.6f}")
    logger.info(f"Audio velocity PCC: {a_vel_pcc:.6f}")
    logger.info(
        f"Video vel range: ref=[{ref_v_vel.min():.3f},{ref_v_vel.max():.3f}], tt=[{vv.min():.3f},{vv.max():.3f}]"
    )
    logger.info(
        f"Audio vel range: ref=[{ref_a_vel.min():.3f},{ref_a_vel.max():.3f}], tt=[{av.min():.3f},{av.max():.3f}]"
    )

    torch.save(
        {
            "ref_v": ref_v_denoised,
            "ref_a": ref_a_denoised,
            "tt_v": tt_v_denoised,
            "tt_a": tt_a_denoised,
            "v_pcc": v_pcc,
            "a_pcc": a_pcc,
        },
        "/tmp/latent_comparison.pt",
    )
    logger.info("Saved to /tmp/latent_comparison.pt")


if __name__ == "__main__":
    main()
