# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from ....utils.test import line_params, ring_params

# ---------------------------------------------------------------------------
# Wan2.2-T2V-14B model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 40
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _make_wan_transformer(*, mesh_device, ccl_manager, parallel_config, is_fsdp, num_layers=NUM_LAYERS):
    return WanTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=num_layers,
        cross_attn_norm=CROSS_ATTN_NORM,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
        # WH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
        # BH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 31, 40, 80, 118, id="5b-720p"),
        pytest.param(1, 21, 60, 104, 118, id="14b-480p"),
        pytest.param(1, 21, 90, 160, 118, id="14b-720p"),
    ],
)
def test_wan_transformer_block(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    MIN_PCC = 0.999_500
    MAX_RMSE = 0.032

    parent_mesh_device = mesh_device
    mesh_device = parent_mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    # Load Wan2.2-T2V-14B model from HuggingFace
    parent_torch_model = TorchWanTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
    )
    torch_model = parent_torch_model.blocks[0]
    torch_model.eval()

    # Create TT model
    tt_model = WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=NUM_HEADS,
        cross_attention_norm=CROSS_ATTN_NORM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Create input tensors
    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch.float32)
    temb_input = torch.randn((B, 6, DIM), dtype=torch.float32)

    # Create ROPE embeddings
    rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    # Create TT tensors
    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    tt_temb = from_torch(temb_input.unsqueeze(0), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis])
    tt_rope_cos = from_torch(rope_cos_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_rope_sin = from_torch(rope_sin_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {tt_spatial.shape}, prompt shape {tt_prompt.shape}, rope_cos shape {tt_rope_cos.shape}, rope_sin shape {tt_rope_sin.shape}"
    )
    tt_spatial_out = tt_model(
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        temb_1BTD=tt_temb,
        N=spatial_seq_len,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        trans_mat=tt_trans_mat,
    )

    spatial_concat_dims = [None, None]
    spatial_concat_dims[sp_axis] = 2
    spatial_concat_dims[tp_axis] = 3
    tt_spatial_out = ttnn.to_torch(
        tt_spatial_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=spatial_concat_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_spatial_out = tt_spatial_out[:, :, :spatial_seq_len, :]

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            temb=temb_input,
            rotary_emb=[torch_rope_cos, torch_rope_sin],
        )

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.parametrize(
    "dit_unit_test",
    [{"1": True, "0": False}.get(os.environ.get("DIT_UNIT_TEST"), False)],
)
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
        # WH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
        # BH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 8, 40, 50, 118, id="short_seq"),
        pytest.param(1, 31, 40, 80, 118, id="5b-720p"),
        pytest.param(1, 21, 60, 104, 118, id="14b-480p"),
        pytest.param(1, 21, 90, 160, 118, id="14b-720p"),
    ],
)
def test_wan_transformer_model(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    T: int,
    H: int,
    W: int,
    prompt_seq_len: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    dit_unit_test: bool,
    reset_seeds,
) -> None:
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    if dit_unit_test:
        num_layers = 1
        torch_model = TorchWanTransformer3DModel(num_layers=num_layers)
    else:
        torch_model = TorchWanTransformer3DModel.from_pretrained(
            MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
        )
        num_layers = NUM_LAYERS
    torch_model.eval()

    torch.manual_seed(0)
    spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )

    start = time.time()
    tt_model.load_torch_state_dict(torch_model.state_dict())
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}, timestep shape {timestep_input.shape}"
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
    )
    del tt_model

    # Run torch model
    logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            timestep=timestep_input,
            return_dict=False,
        )
    torch_spatial_out = torch_spatial_out[0]

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="2x2sp0tp1"),
        pytest.param((2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
        pytest.param((2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
        # WH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
        # BH (ring) on 4x8
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_wan_transformer_inner_step(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Test inner_step against the torch reference, mimicking the pipeline denoising loop."""
    B = 1
    T, H, W = 8, 40, 50
    prompt_seq_len = 118

    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    # Load pretrained torch model and truncate to 1 layer
    torch_model = TorchWanTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
    )
    torch_model.blocks = torch.nn.ModuleList([torch_model.blocks[0]])
    torch_model.eval()

    # Create 1-layer TT model with matching weights
    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=1,
    )
    start = time.time()
    tt_model.load_torch_state_dict(torch_model.state_dict(), on_host=True)
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Create inputs
    torch.manual_seed(0)
    spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

    # Prepare cached inputs on device (like the pipeline does once before the denoising loop)
    spatial_host, N = tt_model.preprocess_spatial_input_host(spatial_input)
    rope_cos_1HND, rope_sin_1HND, trans_mat = tt_model.prepare_rope_features(spatial_input)
    prompt_1BLP = tt_model.prepare_text_conditioning(prompt_input)

    # Run TT inner_step (returns on-device tensor)
    logger.info(f"Running TT inner_step with spatial_host shape {spatial_host.shape}, N={N}")
    tt_output_1BNI_tt = tt_model.inner_step(
        spatial_1BNI_torch=spatial_host,
        prompt_1BLP=prompt_1BLP,
        rope_cos_1HND=rope_cos_1HND,
        rope_sin_1HND=rope_sin_1HND,
        trans_mat=trans_mat,
        N=N,
        timestep_torch=timestep_input,
    )
    tt_output_1BNI = tt_model.device_to_host(tt_output_1BNI_tt)
    tt_output = tt_model.postprocess_spatial_output_host(tt_output_1BNI, T, H, W, N)
    del tt_model

    # Run torch reference
    logger.info(f"Running torch reference with spatial shape {spatial_input.shape}")
    with torch.no_grad():
        torch_output = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            timestep=timestep_input,
            return_dict=False,
        )
    torch_output = torch_output[0]

    assert_quality(torch_output, tt_output, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
