# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC accuracy tests for the LongCat-Image TT transformer.

Compares the TT port (LongCatImageTransformer) against the reference
LongCatImageTransformer2DModel from diffusers using random inputs.

Run (single-node N300/T3000):
    pytest models/experimental/longcat_image/tests/test_transformer.py -v
    pytest models/experimental/longcat_image/tests/test_transformer.py::test_transformer_edit -v
"""

import pytest
import torch
from loguru import logger

import ttnn

from diffusers.models.transformers.transformer_longcat_image import LongCatImageTransformer2DModel
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

from models.experimental.longcat_image.tt.transformer import LongCatImageCheckpoint
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

# ── Checkpoint ────────────────────────────────────────────────────────────────
# Uses the HuggingFace cache if the model has already been downloaded;
# falls back to an HF download otherwise.
MODEL_NAME = "meituan-longcat/LongCat-Image"
EDIT_MODEL_NAME = "meituan-longcat/LongCat-Image-Edit"

# ── Mesh configurations ───────────────────────────────────────────────────────
# Each entry: (mesh_shape, sp_axis, tp_axis, num_links)
# mesh_shape is passed to the mesh_device fixture (indirect).
TEST_MESH_PARAMS = [
    # BH_QB (Blackhole Quad Board) — 4-chip 2×2 mesh
    pytest.param((2, 2), 0, 1, 2, id="2x2sp0tp1"),
    # BH_LB (Blackhole Long Board) / WH T3K — 8-chip 2×4 mesh
    pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
    # Single-node 1×8 (WH Galaxy sub-mesh or T3K)
    pytest.param((1, 8), 0, 1, 1, id="1x8sp0tp1"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Single dual-stream block test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    TEST_MESH_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    # Per-device spatial seq length must be divisible by TILE_HEIGHT (32) for the
    # ring joint SDPA op. 1344/2 = 672 = 21×32 (BH_QB, sp=2); 1344 = 42×32 (sp=1).
    ("batch_size", "spatial_seq_len", "prompt_seq_len"),
    [(1, 1344, 512)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    spatial_seq_len: int,
    prompt_seq_len: int,
) -> None:
    """Compare one dual-stream LongCatImageTransformerBlock (TT vs PyTorch).

    Uses random inputs at inner_dim so the block can be tested in isolation
    without a full weight load — only the single block's weights are copied.
    """
    torch.manual_seed(0)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # Load checkpoint to access config and one block's weights.
    torch_model = LongCatImageTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model.eval()

    inner_dim = torch_model.config.num_attention_heads * torch_model.config.attention_head_dim  # 3072
    num_heads = torch_model.config.num_attention_heads  # 24
    head_dim = torch_model.config.attention_head_dim  # 128

    torch_block = torch_model.transformer_blocks[0]

    # ── Build TT block ────────────────────────────────────────────────────────
    from models.experimental.longcat_image.tt.transformer import LongCatImageTransformerBlock
    from models.tt_dit.utils.padding import PaddingConfig

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    padding_config = (
        PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
        if num_heads % tp_factor != 0
        else None
    )

    tt_block = LongCatImageTransformerBlock(
        dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        context_pre_only=False,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    # ── Random inputs ─────────────────────────────────────────────────────────
    # Block operates at inner_dim (already projected). PyTorch reference uses full [B, seq, 3072];
    # TT block expects SP-sharded seq and TP-sharded features: [B, seq/sp, 3072/tp].
    spatial = torch.randn(batch_size, spatial_seq_len, inner_dim, dtype=torch.bfloat16)
    prompt = torch.randn(batch_size, prompt_seq_len, inner_dim, dtype=torch.bfloat16)
    temb = torch.randn(batch_size, inner_dim, dtype=torch.bfloat16)
    rope_cos = torch.randn(prompt_seq_len + spatial_seq_len, head_dim, dtype=torch.bfloat16)
    rope_sin = torch.randn(prompt_seq_len + spatial_seq_len, head_dim, dtype=torch.bfloat16)

    # ── PyTorch forward ───────────────────────────────────────────────────────
    # LongCatImageTransformerBlock.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
    with torch.no_grad():
        torch_prompt_out, torch_spatial_out = torch_block.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            temb=temb,
            image_rotary_emb=(rope_cos, rope_sin),
        )

    # ── TT forward ────────────────────────────────────────────────────────────
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 1, tp_axis: 2})
    tt_prompt = bf16_tensor(prompt, device=mesh_device, mesh_axis=tp_axis, shard_dim=2)
    tt_time_embed = bf16_tensor(temb.unsqueeze(1), device=mesh_device)
    tt_spatial_rope_cos = bf16_tensor(rope_cos[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_spatial_rope_sin = bf16_tensor(rope_sin[prompt_seq_len:], device=mesh_device, mesh_axis=sp_axis, shard_dim=0)
    tt_prompt_rope_cos = bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device)
    tt_prompt_rope_sin = bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device)

    tt_spatial_out, tt_prompt_out = tt_block.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        time_embed=tt_time_embed,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
    )

    # ── Compare ───────────────────────────────────────────────────────────────
    tt_spatial_torch = tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])[:batch_size]
    tt_prompt_torch = tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis])[:batch_size]

    logger.info("=== spatial stream ===")
    assert_quality(torch_spatial_out, tt_spatial_torch, pcc=0.99)
    logger.info("=== prompt stream ===")
    assert_quality(torch_prompt_out, tt_prompt_torch, pcc=0.99)


# ─────────────────────────────────────────────────────────────────────────────
# Full transformer test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    TEST_MESH_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    # latent_h × latent_w = number of packed image tokens (spatial_seq_len).
    # Using 32×42=1344 rather than the full 4032 (768×1344) for faster CI.
    # Per-device spatial seq (spatial_seq_len / sp_factor) must be divisible by
    # TILE_HEIGHT (32) for the ring joint SDPA op: 1344/2 = 672 = 21×32.
    # For the full-resolution run use latent_h=48, latent_w=84 (4032 tokens).
    ("batch_size", "latent_h", "latent_w", "prompt_seq_len"),
    [(1, 32, 42, 512)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31_000_000}],
    indirect=True,
)
def test_transformer(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    latent_h: int,
    latent_w: int,
    prompt_seq_len: int,
) -> None:
    """End-to-end PCC accuracy test: TT LongCatImageTransformer vs PyTorch reference."""
    torch.manual_seed(0)

    spatial_seq_len = latent_h * latent_w  # e.g. 24×42 = 1008

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    # ── 1. Load the reference model ───────────────────────────────────────────
    logger.info(f"loading reference model from {MODEL_NAME} ...")
    torch_model = LongCatImageTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model.eval()

    in_channels = torch_model.config.in_channels  # 64  (packed 2×2 latent patches)
    joint_attention_dim = torch_model.config.joint_attention_dim  # 3584 (Qwen2.5-VL hidden size)
    head_dim = torch_model.config.attention_head_dim  # 128
    num_heads = torch_model.config.num_attention_heads  # 24

    # ── 2. Build the TT model ─────────────────────────────────────────────────
    logger.info("building TT model ...")
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    # LongCatImageCheckpoint loads weights and shards them to the TT mesh.
    checkpoint = LongCatImageCheckpoint(MODEL_NAME)
    tt_model = checkpoint.build(ccl_manager=ccl_manager, parallel_config=parallel_config)

    # ── 3. Random inputs ──────────────────────────────────────────────────────
    # spatial: packed 2×2 latent patches, each 64-dim (16 ch × 4 spatial).
    # prompt:  Qwen2.5-VL last-hidden-state vectors, one per text token.
    spatial = torch.randn(batch_size, spatial_seq_len, in_channels, dtype=torch.bfloat16)
    prompt = torch.randn(batch_size, prompt_seq_len, joint_attention_dim, dtype=torch.bfloat16)
    # timestep in [0, 1000]; pipeline passes t/1000 to the transformer, which
    # multiplies by 1000 internally.  TT receives the pre-scaled value (500).
    timestep = torch.full([batch_size], fill_value=500, dtype=torch.float32)

    # ── 4. RoPE position ids and frequencies ──────────────────────────────────
    # Text tokens: modality 0, positions 0 … prompt_seq_len-1 (row == col == index).
    txt_ids = prepare_pos_ids(
        modality_id=0,
        type="text",
        start=(0, 0),
        num_token=prompt_seq_len,
    )  # [prompt_seq_len, 3]

    # Image tokens: modality 1, positions offset by prompt_seq_len so they don't
    # collide with text positions in the unified RoPE coordinate space.
    img_ids = prepare_pos_ids(
        modality_id=1,
        type="image",
        start=(prompt_seq_len, prompt_seq_len),
        height=latent_h,
        width=latent_w,
    )  # [spatial_seq_len, 3]

    # pos_embed converts (modality, row, col) triples → (cos, sin) rotary freqs.
    all_ids = torch.cat([txt_ids, img_ids], dim=0)  # [total_seq, 3]
    rope_cos, rope_sin = torch_model.pos_embed.forward(all_ids)
    # rope_cos / rope_sin: [prompt_seq + spatial_seq, head_dim]

    prompt_rope_cos = rope_cos[:prompt_seq_len]  # [prompt_seq,  head_dim]
    prompt_rope_sin = rope_sin[:prompt_seq_len]
    spatial_rope_cos = rope_cos[prompt_seq_len:]  # [spatial_seq, head_dim]
    spatial_rope_sin = rope_sin[prompt_seq_len:]

    # ── 5. PyTorch reference forward ──────────────────────────────────────────
    # The reference model's forward does: timestep × 1000 → sinusoidal embed.
    # So we pass timestep / 1000 (= 0.5) and it recovers 500 internally.
    logger.info("running torch model ...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,  # 0.5 → ×1000 = 500 inside forward
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[
            0
        ]  # [batch, spatial_seq_len, in_channels]

    # ── 6. Upload inputs to TT device ─────────────────────────────────────────
    # spatial: sequence sharded across sp_axis, features replicated across tp_axis.
    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    # prompt: replicated on all devices (short sequence, not sharded).
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    # timestep: float32 required by SD35CombinedTimestepTextProjEmbeddings.
    # Shape [batch, 1] after unsqueeze; TT receives 500 (pre-scaled).
    tt_timestep = tensor.from_torch(
        timestep.unsqueeze(-1),
        dtype=ttnn.float32,
        device=mesh_device,
    )
    # RoPE: spatial cos/sin sharded along sp_axis (sequence dim); prompt replicated.
    tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=mesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=mesh_device)

    # ── 7. TT forward ─────────────────────────────────────────────────────────
    logger.info("running TT model ...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    # ── 8. Gather output and compare ──────────────────────────────────────────
    # Output is [batch, spatial_seq / sp_factor, in_channels], sharded along sp_axis.
    # to_torch gathers along sp_axis and takes the first replica of tp replicas.
    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])[:batch_size]

    logger.info("=== PCC check (full transformer) ===")
    assert_quality(torch_output, tt_output_torch, pcc=0.99, relative_rmse=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Edit-shaped full transformer test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    TEST_MESH_PARAMS,
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    # Edit fuses noisy + reference latents along the sequence dim (2× image tokens).
    # 16×48=768 per stream → 1536 fused; per-device 1536/sp must be divisible by 32:
    #   sp=2 → 768, sp=4 → 384, sp=8 → 192.
    # prompt_seq_len simulates edit's multimodal encoder output: 2 prefix + N vision + 512 text.
    ("batch_size", "latent_h", "latent_w", "num_vision_tokens"),
    [(1, 16, 48, 64)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31_000_000}],
    indirect=True,
)
def test_transformer_edit(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    latent_h: int,
    latent_w: int,
    num_vision_tokens: int,
) -> None:
    """Edit-shaped PCC test: fused noisy+reference spatial stream, multimodal prompt, Edit weights."""
    torch.manual_seed(0)

    image_seq_len = latent_h * latent_w
    fused_spatial_seq_len = 2 * image_seq_len
    prompt_seq_len = 2 + num_vision_tokens + 512

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    per_device_spatial = fused_spatial_seq_len // sp_factor
    assert (
        per_device_spatial % 32 == 0
    ), f"per-device fused spatial seq {per_device_spatial} must be divisible by TILE_HEIGHT (32)"

    logger.info(f"loading reference model from {EDIT_MODEL_NAME} ...")
    torch_model = LongCatImageTransformer2DModel.from_pretrained(
        EDIT_MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model.eval()

    in_channels = torch_model.config.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim

    logger.info("building TT model ...")
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    checkpoint = LongCatImageCheckpoint(EDIT_MODEL_NAME)
    tt_model = checkpoint.build(ccl_manager=ccl_manager, parallel_config=parallel_config)

    noisy = torch.randn(batch_size, image_seq_len, in_channels, dtype=torch.bfloat16)
    reference = torch.randn(batch_size, image_seq_len, in_channels, dtype=torch.bfloat16)
    spatial = torch.cat([noisy, reference], dim=1)
    prompt = torch.randn(batch_size, prompt_seq_len, joint_attention_dim, dtype=torch.bfloat16)
    timestep = torch.full([batch_size], fill_value=500, dtype=torch.float32)

    # Edit RoPE: text modality 0; noisy modality 1; reference modality 2.
    # Noisy and reference share the same row/col grid; only modality_id differs.
    pos_start = (prompt_seq_len, prompt_seq_len)
    txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=prompt_seq_len)
    noisy_ids = prepare_pos_ids(modality_id=1, type="image", start=pos_start, height=latent_h, width=latent_w)
    reference_ids = prepare_pos_ids(modality_id=2, type="image", start=pos_start, height=latent_h, width=latent_w)
    img_ids = torch.cat([noisy_ids, reference_ids], dim=0)

    all_ids = torch.cat([txt_ids, img_ids], dim=0)
    rope_cos, rope_sin = torch_model.pos_embed.forward(all_ids)

    prompt_rope_cos = rope_cos[:prompt_seq_len]
    prompt_rope_sin = rope_sin[:prompt_seq_len]
    spatial_rope_cos = rope_cos[prompt_seq_len:]
    spatial_rope_sin = rope_sin[prompt_seq_len:]

    logger.info("running torch model ...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            timestep=timestep / 1000,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0][:, :image_seq_len]

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)
    tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=mesh_device, mesh_axes=[sp_axis, None])
    tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=mesh_device)
    tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=mesh_device)

    logger.info("running TT model ...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
        prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
        spatial_sequence_length=fused_spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])[:batch_size]
    tt_output_torch = tt_output_torch[:, :image_seq_len]

    logger.info("=== PCC check (edit-shaped transformer) ===")
    assert_quality(torch_output, tt_output_torch, pcc=0.99, relative_rmse=10.0)
