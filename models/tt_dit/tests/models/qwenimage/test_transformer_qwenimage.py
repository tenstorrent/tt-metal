# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import diffusers as reference
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageCheckpoint, QwenImageTransformer
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.padding import PaddingConfig, torch_pad
from models.tt_dit.utils.test import line_params_req_exact_devices
from models.tt_dit.utils.tracing import Tracer


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 2), 0, 1, 1, id="2x2sp0tp1"),
        pytest.param((1, 8), 0, 1, 1, id="1x8sp0tp1"),
        pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
        pytest.param((4, 8), 0, 1, 4, id="4x8sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("batch_size", "latents_height", "latents_width", "prompt_seq_len"),
    [
        (1, 128, 128, 512),  # TODO: set correct values
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
    # 60MB covers the 2x2 mesh, whose trace measured 47636480B in run 31911772448.
    # Larger meshes don't actually need this much memory but just keep the size the same for simplicity.
    [{**line_params_req_exact_devices, "trace_region_size": 60000000}],
    ids=["line"],
    indirect=True,
)
def test_transformer(
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
) -> None:
    torch.manual_seed(0)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    torch_model = reference.QwenImageTransformer2DModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer")
    assert isinstance(torch_model, reference.QwenImageTransformer2DModel)
    torch_model.eval()

    head_dim = torch_model.config.attention_head_dim
    num_heads = torch_model.config.num_attention_heads
    in_channels = torch_model.in_channels
    joint_attention_dim = torch_model.config.joint_attention_dim
    patch_size = 2

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

    cache.load_model(
        tt_model=tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="qwen-image",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
    )

    spatial_seq_len = (latents_height // patch_size) * (latents_width // patch_size)

    tt_model_forward = Tracer(tt_model.forward, device=mesh_device) if traced else tt_model.forward

    # run once for compilation or trace capture, and once for compiled run or trace execution
    for _ in range(2):
        spatial = torch.randn([batch_size, spatial_seq_len, in_channels])
        prompt = torch.randn([batch_size, prompt_seq_len, joint_attention_dim])
        timestep = torch.full([batch_size], fill_value=500)

        # prepare ROPE
        spatial_rope, prompt_rope = torch_model.pos_embed.forward(
            video_fhw=(1, latents_height // patch_size, latents_width // patch_size),
            device="cpu",
            max_txt_seq_len=prompt_seq_len,
        )

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

        logger.info("running TT model...")
        tt_output = tt_model_forward(
            spatial=tt_spatial,
            prompt=tt_prompt,
            timestep=tt_timestep,
            spatial_rope=(tt_spatial_rope_cos, tt_spatial_rope_sin),
            prompt_rope=(tt_prompt_rope_cos, tt_prompt_rope_sin),
            spatial_sequence_length=spatial_seq_len,
        )

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=None,  # None marks that this value is never used
            timestep=timestep / 1000,
            img_shapes=[[(1, latents_height // patch_size, latents_width // patch_size)]] * batch_size,
        ).sample

    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])
    assert_quality(torch_output, tt_output_torch, pcc=0.99914, relative_rmse=0.078)


# Edit-2511 is far more sensitive to rounding on random inputs: at 2048 spatial tokens, torch in
# bfloat16 reaches 99.47 % / 28.8 % against float32 (Edit 99.51 % / 12.9 %, Edit-2509 99.99 % /
# 2.7 %). The device measured 99.36 % / 30.3 % (1024) and 99.29 % / 30.6 % (mixed).
BOUNDS_2511 = (0.99, 0.35)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links"),
    [
        pytest.param((2, 2), 0, 1, 1, id="2x2sp0tp1"),
        pytest.param((2, 4), 0, 1, 1, id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("latents_size", "image_latents_size", "padded_spatial_seq_len", "pcc", "relative_rmse"),
    # Qwen-Image-Edit: the torch model itself in bfloat16 reaches a PCC of 99.89 % (1024) and
    # 99.62 % (mixed) against float32 on these inputs; the device 99.82 % and 99.61 %.
    [
        pytest.param((128, 128), (128, 128), 8192, 0.998, 0.07, id="1024"),
        # The input image is wider than the output, so it sets where the prompt positions start.
        pytest.param((96, 128), (80, 144), 6144, 0.995, 0.12, id="mixed"),
    ],
)
@pytest.mark.parametrize(
    "checkpoint_name",
    [
        pytest.param("Qwen/Qwen-Image-Edit", id="edit"),
        # Modulates the input image's rows as at timestep 0 (`zero_cond_t`).
        pytest.param("Qwen/Qwen-Image-Edit-2511", id="2511"),
    ],
)
@pytest.mark.parametrize("device_params", [line_params_req_exact_devices], indirect=True)
def test_transformer_edit(
    *,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    latents_size: tuple[int, int],
    image_latents_size: tuple[int, int],
    padded_spatial_seq_len: int,
    pcc: float,
    relative_rmse: float,
    checkpoint_name: str,
) -> None:
    """The edit models append the input image's latents to the spatial sequence."""
    if checkpoint_name.endswith("2511"):
        pcc, relative_rmse = BOUNDS_2511
    torch.manual_seed(0)
    prompt_seq_len = 512

    torch_model = reference.QwenImageTransformer2DModel.from_pretrained(checkpoint_name, subfolder="transformer")
    assert isinstance(torch_model, reference.QwenImageTransformer2DModel)
    torch_model.eval()

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    checkpoint = QwenImageCheckpoint(checkpoint_name)
    tt_model = checkpoint.build(ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=False)

    p = checkpoint.patch_size
    img_shapes = [(1, h // p, w // p) for h, w in (latents_size, image_latents_size)]
    noisy_seq_len = img_shapes[0][1] * img_shapes[0][2]
    spatial_seq_len = sum(f * h * w for f, h, w in img_shapes)

    spatial = torch.randn([1, spatial_seq_len, torch_model.in_channels])
    prompt = torch.randn([1, prompt_seq_len, torch_model.config.joint_attention_dim])
    timestep = torch.full([1], fill_value=500)

    spatial_rope, prompt_rope = checkpoint.rope_tables(
        latents_height=latents_size[0],
        latents_width=latents_size[1],
        image_latents_sizes=[image_latents_size],
        prompt_sequence_length=prompt_seq_len,
        padded_spatial_sequence_length=padded_spatial_seq_len,
        device=mesh_device,
        sp_axis=sp_axis,
    )
    tt_spatial = tensor.from_torch(
        torch_pad(spatial, padded_spatial_seq_len - spatial_seq_len, dim=1),
        device=mesh_device,
        mesh_axes=[None, sp_axis, None],
    )
    tt_prompt = tensor.from_torch(prompt, device=mesh_device)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)

    condition_mask = None
    if checkpoint.zero_cond_t:
        mask = torch.zeros([1, padded_spatial_seq_len, 1])
        mask[:, noisy_seq_len:spatial_seq_len] = 1
        condition_mask = tensor.from_torch(mask, device=mesh_device, mesh_axes=[None, sp_axis, None])

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        prompt=tt_prompt,
        timestep=tt_timestep,
        spatial_rope=spatial_rope,
        prompt_rope=prompt_rope,
        spatial_sequence_length=spatial_seq_len,
        condition_mask=condition_mask,
    )
    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None])[:, :spatial_seq_len]

    logger.info("running torch model...")
    with torch.no_grad():
        torch_output = torch_model.forward(
            hidden_states=spatial,
            encoder_hidden_states=prompt,
            encoder_hidden_states_mask=None,
            timestep=timestep / 1000,
            img_shapes=[img_shapes],
        ).sample

    # The pipeline keeps only the prediction for the generated image's tokens.
    assert_quality(
        torch_output[:, :noisy_seq_len], tt_output_torch[:, :noisy_seq_len], pcc=pcc, relative_rmse=relative_rmse
    )
