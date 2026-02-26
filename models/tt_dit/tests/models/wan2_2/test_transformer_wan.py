# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

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
from ....utils.golden import load_golden, save_golden
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.padding import pad_vision_seq_parallel
from ....utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch, local_device_to_torch
from ....utils.test import (
    line_params,
    line_params_req_exact_devices,
    ring_params,
    ring_params_req_exact_devices,
    skip_if_unsupported_num_links,
)

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
def _load_model_and_golden(test_name, param_id):
    """Resolve and log the model commit hash, retrieve cached golden data, and load the model.

    Returns ``(commit_hash, golden, torch_model)`` where ``golden`` is None if no cached
    golden data exists for this commit/test/param.
    """
    _, commit_hash = TorchWanTransformer3DModel.load_config(
        MODEL_NAME, subfolder="transformer", return_commit_hash=True
    )
    logger.info(f"Loaded {MODEL_NAME} (subfolder=transformer) at commit: {commit_hash}")

    golden = load_golden(commit_hash, test_name, param_id)

    torch_model = TorchWanTransformer3DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.float32, trust_remote_code=True
    )
    return commit_hash, golden, torch_model


def _register_block_hooks(model):
    """Register forward hooks on each transformer block to log when it is called."""
    hooks = []
    logger.info(type(model))
    for idx, block in enumerate(model.blocks):

        def _make_hook(block_idx):
            def hook(module, input, output):
                logger.info(f"Torch transformer block {block_idx} called")

            return hook

        hooks.append(block.register_forward_hook(_make_hook(idx)))
    return hooks


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
COMMON_MESH_PARAMS = [
    pytest.param(
        (2, 2), 0, 1, 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="2x2sp0tp1nl2_line_is_fsdp0"
    ),
    pytest.param(
        (2, 4), 0, 1, 1, line_params_req_exact_devices, ttnn.Topology.Linear, True, id="2x4sp0tp1nl1_line_is_fsdp1"
    ),
    pytest.param(
        (2, 4), 1, 0, 1, line_params_req_exact_devices, ttnn.Topology.Linear, True, id="2x4sp1tp0nl1_line_is_fsdp1"
    ),
    # WH (ring) on 4x8
    pytest.param(
        (4, 8), 1, 0, 4, ring_params_req_exact_devices, ttnn.Topology.Ring, True, id="4x8sp1tp0nl4_ring_is_fsdp1"
    ),
    # BH (ring) on 4x8
    pytest.param(
        (4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x8sp1tp0nl2_ring_is_fsdp0"
    ),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    COMMON_MESH_PARAMS
    + [
        pytest.param(
            (4, 8),
            1,
            0,
            2,
            line_params_req_exact_devices,
            ttnn.Topology.Linear,
            False,
            id="4x8sp1tp0nl2_line_is_fsdp0",
        ),
        pytest.param(
            (4, 32), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="4x32sp1tp0nl2_ring_is_fsdp0"
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 31, 40, 80, 512, id="5b-720p"),
        pytest.param(1, 21, 60, 104, 512, id="14b-480p"),
        pytest.param(1, 21, 90, 160, 512, id="14b-720p"),
    ],
)
def test_wan_transformer_block(
    mesh_device: ttnn.MeshDevice,
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
    reset_seeds,
    request,
) -> None:
    MIN_PCC = 0.999_500
    MAX_RMSE = 0.032

    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    param_id = request.node.name
    commit_hash, golden, parent_torch_model = _load_model_and_golden("wan_transformer_block", param_id)
    hooks = _register_block_hooks(parent_torch_model)
    torch_model = parent_torch_model.blocks[0]
    torch_model.eval()
    state_dict = torch_model.state_dict()

    if golden is not None:
        spatial_input = golden["spatial_input"]
        prompt_input = golden["prompt_input"]
        temb_input = golden["temb_input"]
        rope_cos = golden["rope_cos"]
        rope_sin = golden["rope_sin"]
        torch_rope_cos = golden["torch_rope_cos"]
        torch_rope_sin = golden["torch_rope_sin"]
        torch_spatial_out = golden["torch_spatial_out"]
    else:
        logger.warning("Computing reference results from scratch; this will be VERY SLOW.")
        # Generate inputs with fixed seed
        torch.manual_seed(0)
        spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
        prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch.float32)
        temb_input = torch.randn((B, 6, DIM), dtype=torch.float32)
        rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
        rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
        torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

        logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
        with torch.no_grad():
            torch_spatial_out = torch_model(
                hidden_states=spatial_input,
                encoder_hidden_states=prompt_input,
                temb=temb_input,
                rotary_emb=[torch_rope_cos, torch_rope_sin],
            )

        save_golden(
            commit_hash,
            "wan_transformer_block",
            param_id,
            {
                "spatial_input": spatial_input,
                "prompt_input": prompt_input,
                "temb_input": temb_input,
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
                "torch_rope_cos": torch_rope_cos,
                "torch_rope_sin": torch_rope_sin,
                "torch_spatial_out": torch_spatial_out,
            },
        )

    for h in hooks:
        h.remove()
    del parent_torch_model, torch_model

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
    logger.info(f"Loading TT model from torch state dict...")
    start = time.time()
    tt_model.load_torch_state_dict(state_dict)
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Prepare ROPE embeddings
    if golden is not None:
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

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.timeout(0) # Disable pytest timeout
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="ring_bh_4x8sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="line_bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 21, 90, 160, 512, id="14b-720p"),
    ],
)
@pytest.mark.parametrize("test_duration_s", [100, 200, 500, 1000], ids=["100s", "200s", "500s", "1000s"])
def test_wan_workload_power(
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
    test_duration_s: int,
    reset_seeds,
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

    # Load Wan2.2-T2V-14B transformer model from HuggingFace
    torch_model = TorchWanTransformer3DModel(num_layers=1).blocks[0]
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
    
    logger.info(f"Warmup iteration complete")
    
    start = time.time()
    
    itr = 0
    while time.time() - start < test_duration_s:
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Running iteration {itr}, time elapsed: {time.time() - start:.3f} seconds")
        for _ in range(40):
            tt_spatial_out = tt_model(
                spatial_1BND=tt_spatial,
                prompt_1BLP=tt_prompt,
                temb_1BTD=tt_temb,
                N=spatial_seq_len,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
                trans_mat=tt_trans_mat,
            )
        itr += 1
    
    end = time.time()
    logger.info(f"Completed test after: {end - start} seconds")
    



@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    COMMON_MESH_PARAMS
    + [
        pytest.param(
            (4, 8),
            1,
            0,
            2,
            line_params_req_exact_devices,
            ttnn.Topology.Linear,
            False,
            id="4x8sp1tp0nl2_line_is_fsdp0",
        ),
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
    reset_seeds,
    request,
) -> None:
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15
    num_layers = 1

    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    param_id = request.node.name
    commit_hash, golden, torch_model = _load_model_and_golden("wan_transformer_model", param_id)

    # Truncate to 1 layer
    torch_model.blocks = torch.nn.ModuleList([torch_model.blocks[0]])
    torch_model.eval()
    hooks = _register_block_hooks(torch_model)
    state_dict = torch_model.state_dict()

    if golden is not None:
        spatial_input = golden["spatial_input"]
        prompt_input = golden["prompt_input"]
        timestep_input = golden["timestep_input"]
        torch_spatial_out = golden["torch_spatial_out"]
    else:
        logger.warning("Computing reference results from scratch; this will be VERY SLOW.")
        # Generate inputs with fixed seed
        torch.manual_seed(0)
        spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
        prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
        timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

        logger.info(f"Running torch model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}")
        with torch.no_grad():
            torch_spatial_out = torch_model(
                hidden_states=spatial_input,
                encoder_hidden_states=prompt_input,
                timestep=timestep_input,
                return_dict=False,
            )
        torch_spatial_out = torch_spatial_out[0]
        for h in hooks:
            h.remove()
        del torch_model

        save_golden(
            commit_hash,
            "wan_transformer_model",
            param_id,
            {
                "spatial_input": spatial_input,
                "prompt_input": prompt_input,
                "timestep_input": timestep_input,
                "torch_spatial_out": torch_spatial_out,
            },
        )

    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)

    # TODO: This conversion looks sus
    tt_timestep = float32_tensor(timestep_input.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device)

    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )

    logger.info(f"Loading TT model from torch state dict...")
    start = time.time()
    tt_model.load_torch_state_dict(state_dict)
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Run TT model
    logger.info(
        f"Running TT model with spatial shape {spatial_input.shape}, prompt shape {prompt_input.shape}, timestep shape {timestep_input.shape}"
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=tt_prompt,
        timestep=tt_timestep,
    )
    del tt_model

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    COMMON_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_wan_transformer_inner_step(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    request,
) -> None:
    """Test inner_step against the torch reference, mimicking the pipeline denoising loop."""
    B = 1
    T, H, W = 8, 40, 50
    prompt_seq_len = 118
    num_layers = 1
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    param_id = request.node.name
    commit_hash, golden, torch_model = _load_model_and_golden("wan_transformer_inner_step", param_id)

    # Truncate to 1 layer
    torch_model.blocks = torch.nn.ModuleList([torch_model.blocks[0]])
    torch_model.eval()
    hooks = _register_block_hooks(torch_model)
    state_dict = torch_model.state_dict()

    if golden is not None:
        spatial_input = golden["spatial_input"]
        prompt_input = golden["prompt_input"]
        timestep_input = golden["timestep_input"]
        torch_output = golden["torch_output"]
    else:
        logger.warning("Computing reference results from scratch; this will be VERY SLOW.")
        # Generate inputs with fixed seed
        torch.manual_seed(0)
        spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
        prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
        timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

        logger.info(f"Running torch reference with spatial shape {spatial_input.shape}")
        with torch.no_grad():
            torch_output = torch_model(
                hidden_states=spatial_input,
                encoder_hidden_states=prompt_input,
                timestep=timestep_input,
                return_dict=False,
            )
        torch_output = torch_output[0]

        save_golden(
            commit_hash,
            "wan_transformer_inner_step",
            param_id,
            {
                "spatial_input": spatial_input,
                "prompt_input": prompt_input,
                "timestep_input": timestep_input,
                "torch_output": torch_output,
            },
        )

    for h in hooks:
        h.remove()
    del torch_model

    # Create 1-layer TT model with matching weights
    tt_model = _make_wan_transformer(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )

    logger.info(f"Loading TT model from torch state dict...")
    start = time.time()
    tt_model.load_torch_state_dict(state_dict)
    end = time.time()
    logger.info(f"Time taken to load state dict: {end - start} seconds")

    # Prepare cached inputs on device (like the pipeline does once before the denoising loop)
    spatial_host, N = tt_model.preprocess_spatial_input_host(spatial_input)
    rope_cos_1HND, rope_sin_1HND, trans_mat = tt_model.prepare_rope_features(spatial_input)
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    prompt_1BLP = tt_model.prepare_text_conditioning(tt_prompt)

    # TODO: This conversion looks sus
    tt_timestep = float32_tensor(timestep_input.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device)

    spatial_device = from_torch(
        spatial_host, device=mesh_device, mesh_axes=[None, None, parallel_config.sequence_parallel.mesh_axis, None]
    )

    # Run TT inner_step (returns on-device tensor)
    logger.info(f"Running TT inner_step with spatial_device shape {spatial_device.shape}, N={N}")
    tt_output_1BNI_tt = tt_model.inner_step(
        spatial_1BNI=spatial_device,
        prompt_1BLP=prompt_1BLP,
        rope_cos_1HND=rope_cos_1HND,
        rope_sin_1HND=rope_sin_1HND,
        trans_mat=trans_mat,
        N=N,
        timestep=tt_timestep,
    )
    tt_output_1BNI = local_device_to_torch(tt_output_1BNI_tt)
    tt_output = tt_model.postprocess_spatial_output_host(tt_output_1BNI, T, H, W, N)
    del tt_model

    assert_quality(torch_output, tt_output, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
