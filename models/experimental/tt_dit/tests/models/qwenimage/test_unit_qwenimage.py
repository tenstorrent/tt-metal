# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for QwenImage model components with T3K and Galaxy frequent testing.

This module provides accuracy tests for:
- Qwen25VL text encoder
- QwenImage VAE decoder
- QwenImage denoising transformer

All tests are designed for T3K (2x4) and Galaxy (4x8) mesh configurations.
"""

import diffusers.models.autoencoders.autoencoder_kl_qwenimage as vae_reference
import diffusers as transformer_reference
import pytest
import torch
import transformers
import ttnn
from loguru import logger

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ....encoders.qwen25vl.model_qwen25vl import (
    Qwen25VlTextEncoder,
)
from ....models.transformers.transformer_qwenimage import QwenImageTransformer
from ....models.vae.vae_qwenimage import (
    QwenImageVaeDecoder,
)
from ....parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VAEParallelConfig,
)
from ....parallel.manager import CCLManager
from ....utils import cache, tensor
from ....utils.check import assert_quality
from ....utils.padding import PaddingConfig
from ....utils.tracing import Tracer


# =============================================================================
# Qwen25VL Text Encoder Tests - T3K and Galaxy Frequent
# =============================================================================


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 4), (1, 4), id="t3k-1x4"),
        pytest.param((2, 4), (2, 4), id="t3k-2x4"),
        # pytest.param((4, 8), (1, 4), id="tg-1x4"),
        # pytest.param((4, 8), (4, 4), id="tg-4x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [
        (2, 512),
    ],
)
def test_qwenimage_encoder_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    batch_size: int,
    sequence_length: int,
    galaxy_type: str,
) -> None:
    """
    Accuracy test for QwenImage Qwen25VL text encoder on T3K and Galaxy.

    Tests the text encoder component accuracy by comparing TT output against
    PyTorch reference implementation.
    """
    # Skip if submesh is larger than parent mesh
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")

    # Skip 4U Galaxy configuration
    if galaxy_type == "4U":
        pytest.skip("4U Galaxy configuration not supported for this test")

    torch.manual_seed(0)

    encoder_submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    logger.info(f"Running on submesh {encoder_submesh.shape} of parent mesh {mesh_device.shape}")

    tp_axis = 1
    ccl_manager = CCLManager(encoder_submesh, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_submesh.shape[tp_axis], mesh_axis=tp_axis),
    )

    # Load reference PyTorch model
    torch_model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image", subfolder="text_encoder"
    )
    torch_text_model = torch_model.model.language_model

    # Create TT model
    model = Qwen25VlTextEncoder(
        vocab_size=torch_model.config.vocab_size,
        hidden_size=torch_model.config.hidden_size,
        intermediate_size=torch_model.config.intermediate_size,
        hidden_act=torch_model.config.hidden_act,
        num_hidden_layers=torch_model.config.num_hidden_layers,
        num_attention_heads=torch_model.config.num_attention_heads,
        num_key_value_heads=torch_model.config.num_key_value_heads,
        rms_norm_eps=torch_model.config.rms_norm_eps,
        rope_theta=torch_model.config.rope_theta,
        mrope_section=torch_model.config.rope_scaling["mrope_section"],
        device=encoder_submesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    model.load_torch_state_dict(torch_text_model.state_dict())

    # Prepare inputs
    tokens = torch.randint(0, torch_model.config.vocab_size, [batch_size, sequence_length])
    m = torch.randint(0, sequence_length + 1, [batch_size])
    attention_mask = torch.arange(sequence_length) < m.unsqueeze(1)
    cos, sin = model.create_rope_tensors(batch_size, sequence_length, attention_mask)

    tt_tokens = tensor.from_torch(tokens, device=encoder_submesh, dtype=ttnn.uint32)
    tt_attention_mask = tensor.from_torch(attention_mask, device=encoder_submesh)
    tt_pos_embeds_cos = tensor.from_torch(cos, device=encoder_submesh)
    tt_pos_embeds_sin = tensor.from_torch(sin, device=encoder_submesh)

    # Run TT model
    logger.info("Running TT model...")
    tt_hidden_states = model.forward(
        tt_tokens,
        attention_mask=tt_attention_mask,
        pos_embeds=(tt_pos_embeds_cos, tt_pos_embeds_sin),
    )
    tt_prompt_embeds = tt_hidden_states[-1]
    tt_prompt_embeds_torch = tensor.to_torch(tt_prompt_embeds)

    # Run reference PyTorch model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        out = torch_model.forward(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = out.hidden_states[-1]

    # Validate output dimensions
    assert len(out.hidden_states) == len(
        tt_hidden_states
    ), f"Hidden states count mismatch: expected {len(out.hidden_states)}, got {len(tt_hidden_states)}"

    # Check accuracy
    assert_quality(prompt_embeds, tt_prompt_embeds_torch, pcc=0.95, relative_rmse=0.35)

    logger.info("Encoder accuracy test passed!")


# =============================================================================
# QwenImage VAE Decoder Tests - T3K and Galaxy Frequent
# =============================================================================


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape"),
    [
        pytest.param((2, 4), (1, 4), id="t3k-1x4"),
        # pytest.param((4, 8), (1, 4), id="tg-1x4"),
        # pytest.param((4, 8), (1, 8), id="tg-1x8"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [
        (1, 128, 128),
    ],
)
def test_qwenimage_vae_decoder_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    batch_size: int,
    height: int,
    width: int,
    galaxy_type: str,
) -> None:
    """
    Accuracy test for QwenImage VAE decoder on T3K and Galaxy.

    Tests the VAE decoder component accuracy by comparing TT output against
    PyTorch reference implementation.
    """
    # Skip if submesh is larger than parent mesh
    parent_mesh_shape = tuple(mesh_device.shape)
    if any(x[0] < x[1] for x in zip(parent_mesh_shape, submesh_shape)):
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")

    # Skip 4U Galaxy configuration
    if galaxy_type == "4U":
        pytest.skip("4U Galaxy configuration not supported for this test")

    torch.manual_seed(0)

    vae_submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    logger.info(f"Running on submesh {vae_submesh.shape} of parent mesh {mesh_device.shape}")

    tp_axis = 1

    # Load reference PyTorch model
    torch_model = vae_reference.AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
    assert isinstance(torch_model, vae_reference.AutoencoderKLQwenImage)
    torch_model.eval()

    in_channels = torch_model.config["z_dim"]

    # Create TT model
    ccl_manager = CCLManager(vae_submesh, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(
        tensor_parallel=ParallelFactor(factor=vae_submesh.shape[tp_axis], mesh_axis=tp_axis)
    )

    tt_model = QwenImageVaeDecoder(
        base_dim=torch_model.config["base_dim"],
        z_dim=torch_model.config["z_dim"],
        dim_mult=torch_model.config["dim_mult"],
        num_res_blocks=torch_model.config["num_res_blocks"],
        temperal_downsample=torch_model.config["temperal_downsample"],
        device=vae_submesh,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Prepare input
    inp = torch.randn(batch_size, in_channels, height, width)
    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), device=vae_submesh)

    # Run reference PyTorch model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        torch_output = torch_model.decode(inp.unsqueeze(2)).sample.squeeze(2)

    # Run TT model
    logger.info("Running TT model...")
    tt_out = tt_model.forward(tt_inp)
    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)

    # Check accuracy
    assert_quality(torch_output, tt_out_torch, pcc=0.9997, relative_rmse=0.023)

    logger.info("VAE decoder accuracy test passed!")


# =============================================================================
# QwenImage Denoising Transformer Tests - T3K and Galaxy Frequent
# =============================================================================


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 4), 0, 1, 1, id="t3k-2x4sp0tp1"),
        # pytest.param((4, 8), 0, 1, 4, id="tg-4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "latents_height", "latents_width", "prompt_seq_len"),
    [
        (1, 128, 128, 512),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(False, id="not_traced"),
        pytest.param(True, id="traced"),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 34000000}],
    indirect=True,
)
def test_qwenimage_transformer_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    batch_size: int,
    latents_height: int,
    latents_width: int,
    prompt_seq_len: int,
    traced: bool,
    galaxy_type: str,
) -> None:
    """
    Accuracy test for QwenImage denoising transformer on T3K and Galaxy.

    Tests the transformer component accuracy by comparing TT output against
    PyTorch reference implementation.
    """
    # Skip 4U Galaxy configuration
    if galaxy_type == "4U":
        pytest.skip("4U Galaxy configuration not supported for this test")

    torch.manual_seed(0)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    logger.info(f"Running on mesh {mesh_device.shape} with sp={sp_factor}, tp={tp_factor}")

    # Load reference PyTorch model
    torch_model = transformer_reference.QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image", subfolder="transformer"
    )
    assert isinstance(torch_model, transformer_reference.QwenImageTransformer2DModel)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim
    patch_size = 2

    # Create TT model
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    if num_heads % tp_factor != 0:
        padding_config = PaddingConfig.from_tensor_parallel_factor(num_heads, head_dim, tp_factor)
    else:
        padding_config = None

    tt_model = QwenImageTransformer(
        patch_size=torch_model.config.patch_size,
        in_channels=in_channels,
        num_layers=torch_model.config.num_layers,
        attention_head_dim=head_dim,
        num_attention_heads=num_heads,
        joint_attention_dim=joint_attention_dim,
        out_channels=torch_model.out_channels,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=padding_config,
    )

    if not cache.initialize_from_cache(
        tt_model=tt_model,
        torch_state_dict=torch_model.state_dict(),
        model_name="qwen-image",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        dtype="bf16",
    ):
        logger.info(
            "Loading transformer weights from PyTorch state dict. "
            "To use cache, set TT_DIT_CACHE_DIR environment variable."
        )
        tt_model.load_torch_state_dict(torch_model.state_dict())

    spatial_seq_len = (latents_height // patch_size) * (latents_width // patch_size)

    tt_model_forward = Tracer(tt_model.forward, device=mesh_device) if traced else tt_model.forward

    # Run once for compilation or trace capture, and once for compiled run or trace execution
    for _ in range(2):
        spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
        prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
        timestep = torch.full([batch_size], fill_value=500)

        # Prepare ROPE embeddings
        img_shapes = [[(1, latents_height // patch_size, latents_width // patch_size)]] * batch_size
        txt_seq_lens = [prompt_seq_len] * batch_size
        spatial_rope, prompt_rope = torch_model.pos_embed.forward(img_shapes, txt_seq_lens, "cpu")

        spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
        spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
        prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
        prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

        tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
        tt_prompt = tensor.from_torch(prompt, device=mesh_device)
        tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)

        tt_spatial_rope_cos = tensor.from_torch(spatial_rope_cos, device=mesh_device, mesh_axes=[sp_axis, None])
        tt_spatial_rope_sin = tensor.from_torch(spatial_rope_sin, device=mesh_device, mesh_axes=[sp_axis, None])
        tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=mesh_device)
        tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=mesh_device)

        logger.info("Running TT model...")
        tt_output = tt_model_forward(
            spatial=tt_spatial,
            prompt=tt_prompt,
            timestep=tt_timestep,
            spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
            prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
            spatial_sequence_length=spatial_seq_len,
            prompt_sequence_length=prompt_seq_len,
        )

    # Run reference PyTorch model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=torch.tensor([]),  # empty tensor - value not used
            timestep=timestep / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
        ).sample

    # Convert TT output to torch tensor
    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])

    # Check accuracy
    assert_quality(torch_output, tt_output_torch, pcc=0.99914, relative_rmse=0.078)

    logger.info("Transformer accuracy test passed!")


# =============================================================================
# Encoder Pair Integration Test - T3K and Galaxy Frequent
# =============================================================================


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="t3k"),
        # pytest.param((4, 8), id="tg"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompts",
    [
        [
            "",
            "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
            'A coffee shop sign reading "Qwen Coffee $2" in neon lights',
        ],
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwenimage_encoder_pair_accuracy(
    *,
    mesh_device: ttnn.MeshDevice,
    prompts: list[str],
    galaxy_type: str,
) -> None:
    """
    Accuracy test for QwenImage encoder pair (tokenizer + encoder) on T3K and Galaxy.

    Tests the complete text encoding pipeline including tokenization and encoding.
    """
    # Skip 4U Galaxy configuration
    if galaxy_type == "4U":
        pytest.skip("4U Galaxy configuration not supported for this test")

    import diffusers.pipelines.qwenimage.pipeline_qwenimage

    # Note: There is a bug in HF implementation where prompt_embeds_mask is incorrectly repeated
    # if num_images_per_prompt != 1.
    num_images_per_prompt = 1

    checkpoint = "Qwen/Qwen-Image"

    torch_pipeline = diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline.from_pretrained(checkpoint)

    template = torch_pipeline.prompt_template_encode
    start_idx = torch_pipeline.prompt_template_encode_start_idx
    sequence_length = 512

    # Create TT encoder pair
    parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_encoder_pair = Qwen25VlTokenizerEncoderPair(
        checkpoint,
        tokenizer_subfolder="tokenizer",
        encoder_subfolder="text_encoder",
        use_torch=False,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )

    # Run reference PyTorch model
    logger.info("Running PyTorch reference model...")
    with torch.no_grad():
        embeds, mask = torch_pipeline.encode_prompt(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=sequence_length,
        )
        embeds = torch.nn.functional.pad(embeds, [0, 0, 0, sequence_length - embeds.shape[1]], value=0)
        mask = torch.nn.functional.pad(mask, [0, sequence_length - mask.shape[1]], value=0)

    # Run TT model
    logger.info("Running TT model...")
    formatted_prompts = [template.format(e) for e in prompts]
    tt_embeds, tt_mask = tt_encoder_pair.encode(
        formatted_prompts,
        num_images_per_prompt=num_images_per_prompt,
        sequence_length=sequence_length + start_idx,
    )
    tt_embeds = tt_embeds[:, start_idx:]
    tt_mask = tt_mask[:, start_idx:]
    tt_embeds *= tt_mask.unsqueeze(-1)

    # Check mask match
    assert torch.allclose(mask, tt_mask), "Attention masks do not match"

    # Check accuracy
    assert_quality(embeds, tt_embeds, pcc=0.98, relative_rmse=0.2)

    logger.info("Encoder pair accuracy test passed!")
