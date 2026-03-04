# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for WanTransformerBlock and WanTransformer3DModel with Lingbot-VA model parameters.

Uses the Lingbot-VA TT model from lingbot_va.tt (in_channels=48, action_dim=30) and compares
against the Lingbot-VA reference model with config: dim=3072, num_heads=24, ffn_dim=14336, num_layers=30, etc.
"""

import gc
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Ensure tt-metal root is on path when running from various working directories
_tt_metal_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_tt_metal_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_tt_metal_root))

from models.experimental.lingbot_va.reference.model import (
    WanTransformer3DModel as TorchWanTransformerBlock,
)
from models.experimental.lingbot_va.tt.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.cache import model_cache_dir
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat, stack_cos_sin
from models.tt_dit.utils.padding import pad_vision_seq_parallel
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from models.tt_dit.utils.test import line_params

# ---------------------------------------------------------------------------
# Lingbot-VA model configuration (from reference/model.py)
# ---------------------------------------------------------------------------
# num_attention_heads=24, attention_head_dim=128 -> inner_dim = 3072
DIM = 24 * 128  # 3072
FFN_DIM = 14336
NUM_HEADS = 24
HEAD_DIM = 128
IN_CHANNELS = 48
OUT_CHANNELS = 48
ACTION_DIM = 30
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 30
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024

# Checkpoint path for loading pretrained lingbot_va block (optional)
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", _tt_metal_root)
LINGBOT_VA_CHECKPOINT = Path(TT_METAL_HOME) / "models/experimental/lingbot_va/reference/checkpoints/transformer"


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
    """Lingbot-VA TT WanTransformer3DModel from lingbot_va.tt (in_channels=48, action_dim=30)."""
    return WanTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        action_dim=ACTION_DIM,
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


def _make_grid_id(B: int, F: int, H: int, W: int, patch_size: tuple, device: torch.device) -> torch.Tensor:
    """Build grid_id (B, 3, L) for reference model RoPE; L = (F//p0)*(H//p1)*(W//p2)."""
    p0, p1, p2 = patch_size
    f_p, h_p, w_p = F // p0, H // p1, W // p2
    f_idx = torch.arange(f_p, dtype=torch.float32, device=device)
    h_idx = torch.arange(h_p, dtype=torch.float32, device=device)
    w_idx = torch.arange(w_p, dtype=torch.float32, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    grid_id = torch.stack([ff.flatten(), hh.flatten(), ww.flatten()], dim=0)  # (3, L)
    grid_id = grid_id.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, L)
    return grid_id


def _make_action_grid_id(
    B: int,
    F_action: int,
    action_per_frame: int,
    device: torch.device,
    f_w: int = 1,
    f_shift: int = 0,
) -> torch.Tensor:
    """Build grid_id (B, 3, L) for action_mode RoPE; matches demo get_mesh_id(..., action=True). L = F_action * action_per_frame * 1."""
    f_idx = torch.arange(f_shift, F_action + f_shift, dtype=torch.float32, device=device) * f_w
    h_idx = torch.arange(action_per_frame, dtype=torch.float32, device=device)
    w_idx = torch.arange(1, dtype=torch.float32, device=device)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    # action=True semantics from demo get_mesh_id(..., action=True)
    ff_offset = (torch.ones([action_per_frame], device=device).cumsum(0) / (action_per_frame + 1)).view(1, -1, 1)
    ff = ff + ff_offset
    hh = torch.ones_like(hh, device=device) * -1
    ww = torch.ones_like(ww, device=device) * -1
    grid_id = torch.stack([ff.flatten(), hh.flatten(), ww.flatten()], dim=0)  # (3, L)
    grid_id = grid_id.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, L)
    return grid_id


def _ref_output_to_bcfhw(
    ref_out: torch.Tensor, B: int, F: int, H: int, W: int, patch_size: tuple, out_c: int
) -> torch.Tensor:
    """Reshape reference output (B, L*n, out_c) to (B, out_c, F, H, W) to match TT output."""
    p0, p1, p2 = patch_size
    patch_F, patch_H, patch_W = F // p0, H // p1, W // p2
    n = p0 * p1 * p2
    L = patch_F * patch_H * patch_W
    assert ref_out.shape == (B, L * n, out_c)
    x = ref_out.reshape(B, patch_F, patch_H, patch_W, p0, p1, p2, out_c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, out_c, F, H, W)
    return x


# ---------------------------------------------------------------------------
# Block test: single config matching demo/test_inference.py (B=1, F=8, H=W=24, patch (1,2,2))
# Always open full system mesh (1, 2), then create_submesh(1, 1) so we run on one device and
# avoid CCL (all_gather/ring). The 2-device submesh (1, 2) is skipped: it can hang in CCL then
# get killed (timeout/OOM).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_wan_transformer_block(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """PCC test: TT WanTransformerBlock vs Lingbot-VA reference block (first layer). Params match demo/test_inference.py."""
    MIN_PCC = 0.999_500
    MAX_RMSE = 0.032

    # Match demo/test_inference.py: B=1, F=8, H=W=24, patch_size=(1,2,2) -> spatial_seq = 8*12*12 = 1152
    B, T, H, W = 1, 8, 24, 24
    prompt_seq_len = 77

    # Full mesh is (1, 2); create submesh for this test (1 device or 2 devices)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    p_t, p_h, p_w = PATCH_SIZE
    spatial_seq_len = (T // p_t) * (H // p_h) * (W // p_w)

    # Load Lingbot-VA reference model and use first block
    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    torch_model = TorchWanTransformerBlock.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch.float32,
        attn_mode="torch",
    )
    torch_model = torch_model.blocks[0]
    torch_model.eval()

    # TT block with Lingbot-VA dimensions
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

    # Inputs
    torch.manual_seed(0)
    spatial_input = torch.randn((B, spatial_seq_len, DIM), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, DIM), dtype=torch.float32)
    # Reference block expects temb (B, L, 6, D); TT block expects (1, B, 6, D). Use one global temb broadcast for both.
    temb_single = torch.randn((B, 6, DIM), dtype=torch.float32)
    temb_input = temb_single.unsqueeze(1).expand(B, spatial_seq_len, 6, DIM)

    # RoPE embeddings: TT uses cos/sin separately; reference block expects a single complex tensor (freqs_cis)
    rope_cos = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    rope_sin = torch.randn(B, spatial_seq_len, 1, HEAD_DIM // 2)
    torch_rope_cos, torch_rope_sin = stack_cos_sin(rope_cos, rope_sin)

    # Reference WanAttention expects rotary_emb = single complex tensor (like rope(grid_id) -> torch.polar)
    rotary_emb_ref = torch.complex(
        torch_rope_cos[..., : HEAD_DIM // 2].float(),
        torch_rope_sin[..., : HEAD_DIM // 2].float(),
    )

    rope_cos_stack = torch_rope_cos.permute(0, 2, 1, 3)
    rope_sin_stack = torch_rope_sin.permute(0, 2, 1, 3)

    spatial_padded = pad_vision_seq_parallel(spatial_input.unsqueeze(0), num_devices=sp_factor)
    rope_cos_padded = pad_vision_seq_parallel(rope_cos_stack, num_devices=sp_factor)
    rope_sin_padded = pad_vision_seq_parallel(rope_sin_stack, num_devices=sp_factor)

    tt_spatial = bf16_tensor_2dshard(spatial_padded, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt_input.unsqueeze(0), device=mesh_device)
    # TT block expects temb_1BTD shape (1, B, 6, D), not per-token
    temb_for_tt = temb_single.unsqueeze(0)  # (1, B, 6, DIM)
    tt_temb = from_torch(temb_for_tt, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis])
    tt_rope_cos = from_torch(rope_cos_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_rope_sin = from_torch(rope_sin_padded, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    logger.info(
        f"--------------------------------\n"
        f"Running TT block (lingbot_va) spatial {tt_spatial.shape}, prompt {tt_prompt.shape}, "
        f"temb {tt_temb.shape}, spatial_seq_len {spatial_seq_len}, "
        f"rope_cos {tt_rope_cos.shape}, rope_sin {tt_rope_sin.shape}, trans_mat {tt_trans_mat.shape}"
        f"--------------------------------\n"
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
    # TT output is (1, 1, N, D) after concat; reference is (B, N, D). Squeeze to match.
    if tt_spatial_out.dim() == 4 and tt_spatial_out.shape[0] == 1:
        tt_spatial_out = tt_spatial_out.squeeze(0)
    logger.info(
        f"--------------------------------\n"
        f"torch model forward, "
        f"spatial_input: {spatial_input.shape}, prompt_input: {prompt_input.shape}, temb_input: {temb_input.shape}, rotary_emb_ref: {rotary_emb_ref.shape}"
        f"--------------------------------\n"
    )
    with torch.no_grad():
        torch_spatial_out = torch_model(
            hidden_states=spatial_input,
            encoder_hidden_states=prompt_input,
            temb=temb_input,
            rotary_emb=rotary_emb_ref,
        )

    assert_quality(torch_spatial_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


# Full model: use single device (1x1 submesh) to avoid CCL hang during 30-block forward.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
        # pytest.param((1, 2), (1, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="1x2sp0tp1"),
        # pytest.param((2, 2), (2, 2), 0, 1, 2, line_params, ttnn.Topology.Linear, False, id="2x2sp0tp1"),
        # pytest.param((2, 4), (2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
        # pytest.param((2, 4), (2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
        # pytest.param((4, 8), (4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
        # pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "T", "H", "W", "prompt_seq_len"),
    [
        pytest.param(1, 8, 24, 24, 77, id="lingbot_va_short"),
        # pytest.param(1, 8, 40, 50, 118, id="lingbot_va_medium"),
    ],
)
def test_wan_transformer_model(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
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
) -> None:
    """
    PCC test: Lingbot-VA TT WanTransformer3DModel (lingbot_va.tt) vs Lingbot-VA reference.

    Loads pretrained reference from checkpoint; TT model uses _prepare_torch_state to map
    patch_embedding_mlp -> patch_embedding. Compares video-path output (B, out_channels, F, H, W).

    To avoid OOM when loading the full state dict:
    - If TT_DIT_CACHE_DIR is set and cache exists for this config, loads TT model from cache.
    - Otherwise saves state_dict to a temp file and loads with mmap so only one copy is in RAM
      while filling TT parameters. Run test_wan_transformer_model_caching first to build cache.
    """
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    # Run on submesh (e.g. (1, 1)) to avoid CCL hang/kill with multi-device.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    torch_model = TorchWanTransformerBlock.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch.float32,
        attn_mode="torch",
    )
    torch_model.eval()

    torch.manual_seed(0)
    spatial_input = torch.randn((B, IN_CHANNELS, T, H, W), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)

    grid_id = _make_grid_id(B, T, H, W, PATCH_SIZE, spatial_input.device)
    # Reference: timesteps (B, F_patched), repeat_interleave(..., (H//p2)*(W//p3), dim=1) -> (B, L)
    F_patched = T // PATCH_SIZE[0]
    timesteps_ref = timestep_input.unsqueeze(1).expand(B, F_patched)
    input_dict = {
        "noisy_latents": spatial_input,
        "text_emb": prompt_input,
        "timesteps": timesteps_ref,
        "grid_id": grid_id,
    }

    print(
        {
            "noisy_latents": spatial_input.shape,
            "text_emb": prompt_input.shape,
            "timesteps": timesteps_ref.shape,
            "grid_id": grid_id.shape,
        }
    )

    # Run reference forward first, then free torch model to avoid OOM when loading TT model.
    with torch.no_grad():
        ref_out = torch_model(input_dict, action_mode=False, train_mode=False)
    ref_out_bcfhw = _ref_output_to_bcfhw(ref_out, B, T, H, W, PATCH_SIZE, OUT_CHANNELS)

    # Prefer loading from cache (like SD35/Mochi) to avoid holding state_dict + TT model in RAM.
    cache_dir = model_cache_dir(
        model_name="lingbot_va",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        required=False,
    )
    use_cache = cache_dir is not None and cache_dir.is_dir()

    if use_cache:
        del torch_model
        gc.collect()
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        start = time.time()
        tt_model.load(cache_dir)
        end = time.time()
        logger.info("Time taken to load from cache: %s seconds", end - start)
    else:
        # No cache: save state_dict to temp file and load with mmap (like Motif) so we don't
        # hold a full in-RAM copy while allocating TT tensors, which can cause OOM.
        state_dict = torch_model.state_dict()
        fd, state_path = tempfile.mkstemp(suffix=".pt")
        try:
            os.close(fd)
            torch.save(state_dict, state_path)
            del torch_model
            del state_dict
            gc.collect()
            state_dict = torch.load(state_path, map_location=torch.device("cpu"), mmap=True)
            tt_model = _make_wan_transformer(
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                num_layers=NUM_LAYERS,
            )
            start = time.time()
            tt_model.load_torch_state_dict(state_dict)
            end = time.time()
            logger.info("Time taken to load state dict (mmap): %s seconds", end - start)
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    logger.info(
        "Running TT model (lingbot_va.tt) spatial %s, prompt %s, timestep %s",
        spatial_input.shape,
        prompt_input.shape,
        timestep_input.shape,
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
        grid_id=grid_id,
        action_mode=False,
    )
    del tt_model
    gc.collect()

    print(ref_out_bcfhw.shape, tt_spatial_out.shape)
    print(ref_out_bcfhw[..., 0, 0, 0], tt_spatial_out[..., 0, 0, 0])

    ref_max = ref_out_bcfhw.max()
    tt_max = tt_spatial_out.max()
    ref_min = ref_out_bcfhw.min()
    tt_min = tt_spatial_out.min()

    ref_max_idx = ref_out_bcfhw.argmax()
    tt_max_idx = tt_spatial_out.argmax()
    ref_min_idx = ref_out_bcfhw.argmin()
    tt_min_idx = tt_spatial_out.argmin()

    import numpy as np

    # Convert flat indices to coordinates
    ref_max_coord = np.unravel_index(ref_max_idx, ref_out_bcfhw.shape)
    tt_max_coord = np.unravel_index(tt_max_idx, tt_spatial_out.shape)
    ref_min_coord = np.unravel_index(ref_min_idx, ref_out_bcfhw.shape)
    tt_min_coord = np.unravel_index(tt_min_idx, tt_spatial_out.shape)

    print("ref max:", ref_max, "at", ref_max_coord)
    print("tt max:", tt_max, "at", tt_max_coord)
    print("ref min:", ref_min, "at", ref_min_coord)
    print("tt min:", tt_min, "at", tt_min_coord)
    print()
    assert_quality(ref_out_bcfhw, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)


# Action mode: same device setup as full model test; inputs follow demo/test_inference.py action path.
ACTION_PER_FRAME = 16  # from demo
F_ACTION = 8  # number of action frames in demo


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((1, 2), (1, 1), 0, 1, 1, line_params, ttnn.Topology.Linear, False, id="1x1_single_device"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("B", "F_action", "action_per_frame", "prompt_seq_len"),
    [
        pytest.param(1, F_ACTION, ACTION_PER_FRAME, 77, id="lingbot_va_action_short"),
    ],
)
def test_wan_transformer_model_action_mode(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    B: int,
    F_action: int,
    action_per_frame: int,
    prompt_seq_len: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """
    PCC test: Lingbot-VA TT WanTransformer3DModel (lingbot_va.tt) vs reference with action_mode=True.

    Inputs match demo test_inference.py action path:
    - action_tokens (noisy_latents): (B, action_dim, F_action, action_per_frame, 1) e.g. (1, 30, 8, 16, 1)
    - grid_id: (B, 3, F_action*action_per_frame*1) via _make_action_grid_id (same as get_mesh_id(..., action=True))
    - timesteps: (B, F_action)
    Output: (B, N, action_dim) with N = F_action * action_per_frame * 1.
    """
    MIN_PCC = 0.992_000
    MAX_RMSE = 0.15

    if not LINGBOT_VA_CHECKPOINT.exists():
        pytest.skip(f"Lingbot-VA checkpoint not found: {LINGBOT_VA_CHECKPOINT}")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

    torch_model = TorchWanTransformerBlock.from_pretrained(
        str(LINGBOT_VA_CHECKPOINT),
        torch_dtype=torch.float32,
        attn_mode="torch",
    )
    torch_model.eval()

    torch.manual_seed(0)
    # Action path: match demo test_inference.py — (B, action_dim, F_action, action_per_frame, 1)
    spatial_input = torch.randn((B, ACTION_DIM, F_action, action_per_frame, 1), dtype=torch.float32)
    prompt_input = torch.randn((B, prompt_seq_len, TEXT_DIM), dtype=torch.float32)
    timestep_input = torch.randint(0, 1000, (B,), dtype=torch.float32)
    timesteps_ref = timestep_input.unsqueeze(1).expand(B, F_action)

    grid_id = _make_action_grid_id(B, F_action, action_per_frame, spatial_input.device)
    input_dict = {
        "noisy_latents": spatial_input,
        "text_emb": prompt_input,
        "timesteps": timesteps_ref,
        "grid_id": grid_id,
    }
    print(
        {
            "noisy_latents": spatial_input.shape,
            "text_emb": prompt_input.shape,
            "timesteps": timesteps_ref.shape,
            "grid_id": grid_id.shape,
        }
    )

    with torch.no_grad():
        ref_out = torch_model(input_dict, action_mode=True, train_mode=False)
    # ref_out shape: (B, N, action_dim), N = F_action * action_per_frame * 1

    cache_dir = model_cache_dir(
        model_name="lingbot_va",
        subfolder="transformer",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        required=False,
    )
    use_cache = cache_dir is not None and cache_dir.is_dir()

    if use_cache:
        del torch_model
        gc.collect()
        tt_model = _make_wan_transformer(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            num_layers=NUM_LAYERS,
        )
        start = time.time()
        tt_model.load(cache_dir)
        end = time.time()
        logger.info("Time taken to load from cache (action_mode test): %s seconds", end - start)
    else:
        state_dict = torch_model.state_dict()
        fd, state_path = tempfile.mkstemp(suffix=".pt")
        try:
            os.close(fd)
            torch.save(state_dict, state_path)
            del torch_model
            del state_dict
            gc.collect()
            state_dict = torch.load(state_path, map_location=torch.device("cpu"), mmap=True)
            tt_model = _make_wan_transformer(
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                num_layers=NUM_LAYERS,
            )
            start = time.time()
            tt_model.load_torch_state_dict(state_dict)
            end = time.time()
            logger.info("Time taken to load state dict (mmap, action_mode test): %s seconds", end - start)
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    logger.info(
        "Running TT model (lingbot_va.tt) action_mode=True spatial %s, prompt %s, timestep %s",
        spatial_input.shape,
        prompt_input.shape,
        timestep_input.shape,
    )
    tt_spatial_out = tt_model(
        spatial=spatial_input,
        prompt=prompt_input,
        timestep=timestep_input,
        grid_id=grid_id,
        action_mode=True,
    )
    del tt_model
    gc.collect()

    assert ref_out.shape == tt_spatial_out.shape, (ref_out.shape, tt_spatial_out.shape)
    assert_quality(ref_out, tt_spatial_out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
