#!/usr/bin/env python3
"""
48-layer velocity comparison: TTNN vs CPU reference at 256x256.
Uses IDENTICAL RoPE setup for both to ensure a fair comparison.
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
SEED = 10
HEIGHT, WIDTH = 256, 256
NUM_FRAMES = 33


def pcc(a, b):
    af, bf = a.flatten().float(), b.flatten().float()
    am, bm = af - af.mean(), bf - bf.mean()
    d = am.norm() * bm.norm()
    return ((am * bm).sum() / d).item() if d > 0 else float("nan")


def main():
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute
    from ltx_core.types import AudioLatentShape, VideoPixelShape
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
    from ltx_pipelines.utils.helpers import encode_prompts
    from ltx_pipelines.utils.model_ledger import ModelLedger
    from safetensors.torch import load_file

    lf, lh, lw = (NUM_FRAMES - 1) // 8 + 1, HEIGHT // 32, WIDTH // 32
    video_N = lf * lh * lw
    vps = VideoPixelShape(batch=1, frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    audio_N = als.frames

    # === Shared: text embeddings ===
    logger.info("Encoding prompts...")
    ledger = ModelLedger(
        dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA
    )
    results = encode_prompts(["A cat playing piano in a cozy room with warm lighting", DEFAULT_NEGATIVE_PROMPT], ledger)
    v_ctx, a_ctx = results[0].video_encoding, results[0].audio_encoding

    # === Shared: RoPE (use the reference precompute for BOTH CPU and TTNN) ===
    # Video: 3D grid with [idx, idx] pairs → middle = idx (matches generate_audio_video.py)
    t_ids, h_ids, w_ids = torch.arange(lf), torch.arange(lh), torch.arange(lw)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_indices = torch.stack([gt.flatten(), gh.flatten(), gw.flatten()], dim=0).float()
    v_positions = torch.stack([v_indices, v_indices], dim=-1).unsqueeze(0)  # (1, 3, N, 2)

    v_cos, v_sin = ref_precompute(
        v_positions.bfloat16(),
        dim=4096,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )  # (1, 32, N, 64)

    # Audio: 1D positions from AudioPatchifier
    from ltx_core.components.patchifiers import AudioPatchifier

    ap = AudioPatchifier(patch_size=1)
    a_shape = AudioLatentShape(batch=1, channels=8, frames=audio_N, mel_bins=16)
    a_positions = ap.get_patch_grid_bounds(output_shape=a_shape, device="cpu").float()
    a_cos, a_sin = ref_precompute(
        a_positions.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )  # (1, 32, audio_N, 32)

    # === Shared: initial noise ===
    torch.manual_seed(SEED)
    v_sample = torch.randn(1, video_N, 128, dtype=torch.bfloat16)
    a_sample = torch.randn(1, audio_N, 128, dtype=torch.bfloat16)

    sigma = 1.0
    pert = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])

    # === CPU Reference (48 layers) ===
    logger.info("Loading CPU reference model (48L)...")
    ref_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_layers=48,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        cross_attention_dim=4096,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    raw = load_file(CHECKPOINT)
    prefix = "model.diffusion_model."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    ref_model.load_state_dict(sd, strict=False)
    ref_model.eval().bfloat16()
    del raw, sd

    vm = Modality(
        latent=v_sample * sigma,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, video_N) * sigma,
        positions=v_positions,
        context=v_ctx,
        enabled=True,
    )
    am = Modality(
        latent=a_sample * sigma,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, audio_N) * sigma,
        positions=a_positions,
        context=a_ctx,
        enabled=True,
    )

    logger.info("CPU forward (48L)...")
    t0 = time.time()
    with torch.no_grad():
        ref_v_vel, ref_a_vel = ref_model(video=vm, audio=am, perturbations=pert)
    logger.info(f"CPU done in {time.time()-t0:.0f}s")
    logger.info(f"  v_vel: [{ref_v_vel.min():.3f}, {ref_v_vel.max():.3f}], std={ref_v_vel.std():.4f}")
    logger.info(f"  a_vel: [{ref_a_vel.min():.3f}, {ref_a_vel.max():.3f}], std={ref_a_vel.std():.4f}")
    del ref_model, vm, am

    # === TTNN (48 layers) ===
    logger.info("Loading TTNN model (48L)...")
    import ttnn
    from models.tt_dit.models.transformers.ltx.audio_ltx import LTXAudioVideoTransformerModel as LTXTransformerModel
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

    sp_factor = 2
    audio_N_padded = ((audio_N + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)

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
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    del raw
    model = LTXTransformerModel(
        num_layers=48,
        mesh_device=mesh,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    model.load_torch_state_dict(sd)
    del sd

    # Shard RoPE tensors for TTNN
    tt_v_cos = bf16_tensor_2dshard(v_cos, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_sin = bf16_tensor_2dshard(v_sin, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Pad audio RoPE
    if audio_N_padded > audio_N:
        a_cos_padded = torch.ones(1, 32, audio_N_padded, a_cos.shape[-1])
        a_cos_padded[:, :, :audio_N, :] = a_cos
        a_sin_padded = torch.zeros(1, 32, audio_N_padded, a_sin.shape[-1])
        a_sin_padded[:, :, :audio_N, :] = a_sin
    else:
        a_cos_padded, a_sin_padded = a_cos, a_sin
    tt_a_cos = bf16_tensor_2dshard(a_cos_padded, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_sin = bf16_tensor_2dshard(a_sin_padded, device=mesh, shard_mapping={sp_axis: 2, tp_axis: 1})

    tt_v_prompt = bf16_tensor(v_ctx.unsqueeze(0), device=mesh)
    tt_a_prompt = bf16_tensor(a_ctx.unsqueeze(0), device=mesh)

    # Audio padding
    a_padded = torch.zeros(1, audio_N_padded, 128, dtype=torch.bfloat16)
    a_padded[:, :audio_N, :] = a_sample

    tt_attn_mask = tt_pad_mask = None
    if audio_N_padded > audio_N:
        audio_N_local = audio_N_padded // sp_factor
        m = torch.zeros(1, 1, audio_N_local, audio_N_padded)
        m[:, :, :, audio_N:] = float("-inf")
        tt_attn_mask = bf16_tensor(m, device=mesh)
        pm = torch.ones(1, 1, audio_N_padded, 1)
        pm[:, :, audio_N:, :] = 0.0
        tt_pad_mask = bf16_tensor(pm, device=mesh)

    logger.info("TTNN forward (48L)...")
    t0 = time.time()
    v_out, a_out = model.inner_step(
        video_1BNI_torch=(v_sample * sigma).unsqueeze(0).float(),
        video_prompt_1BLP=tt_v_prompt,
        video_rope_cos=tt_v_cos,
        video_rope_sin=tt_v_sin,
        video_N=video_N,
        audio_1BNI_torch=(a_padded * sigma).unsqueeze(0).float(),
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

    tt_v_vel = LTXTransformerModel.device_to_host(v_out).squeeze(0)
    tt_a_vel = LTXTransformerModel.device_to_host(a_out).squeeze(0)[:, :audio_N, :]

    logger.info(f"TTNN done in {tt_time:.1f}s")
    logger.info(f"  v_vel: [{tt_v_vel.min():.3f}, {tt_v_vel.max():.3f}], std={tt_v_vel.std():.4f}")
    logger.info(f"  a_vel: [{tt_a_vel.min():.3f}, {tt_a_vel.max():.3f}], std={tt_a_vel.std():.4f}")

    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # === Compare ===
    v_pcc = pcc(ref_v_vel, tt_v_vel)
    a_pcc = pcc(ref_a_vel, tt_a_vel)
    logger.info(f"=== 48-LAYER VELOCITY PCC (256x256, sigma=1.0, same RoPE) ===")
    logger.info(f"Video velocity PCC: {v_pcc:.6f}")
    logger.info(f"Audio velocity PCC: {a_pcc:.6f}")

    torch.save(
        {"ref_v": ref_v_vel, "tt_v": tt_v_vel, "ref_a": ref_a_vel, "tt_a": tt_a_vel}, "/tmp/latent_comparison.pt"
    )


if __name__ == "__main__":
    main()
