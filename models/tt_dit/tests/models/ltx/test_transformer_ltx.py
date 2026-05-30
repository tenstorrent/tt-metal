# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 transformer unit tests.

Structure mirrors ``test_transformer_wan.py``:
  - module-level constants for the 22B distilled config
  - small ``_make_*`` helpers reused by all tests
  - three tests: block / model / inner_step
  - shared mesh × shape × modality × run_pcc parametrize axes

Modality axis (``video`` vs ``av``) selects which forward path runs. The block
test runs a single ``LTXTransformerBlock``; model/inner_step run the full
``LTXTransformerModel`` 1-layer slice. Set ``LTX_SKIP_PCC=1`` to skip the
torch reference and assert only shape + finiteness (useful at production
seq lengths where the host reference is heavy).

RoPE: the torch reference consumes SPLIT-rotation freqs; the TT runtime fuses
into ``rotary_embedding_llama`` (INTERLEAVED) with a load-time Q/K channel
permute, so every TT feed is INTERLEAVED + ``_interleaved_to_bhnd`` and a
``trans_mat`` is required (the runtime asserts it whenever rope is set).
"""

import os
import sys
import time

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock, LTXTransformerModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params, ring_params

sys.path.insert(0, "LTX-2/packages/ltx-core/src")

# ---------------------------------------------------------------------------
# LTX-2.3-22B distilled transformer configuration
# ---------------------------------------------------------------------------
DIM = 4096
NUM_HEADS = 32
HEAD_DIM = DIM // NUM_HEADS  # 128
IN_CHANNELS = 128
OUT_CHANNELS = 128
CTX_DIM = 4096
AUDIO_DIM = 2048
AUDIO_HEAD_DIM = AUDIO_DIM // NUM_HEADS  # 64
AUDIO_IN_CHANNELS = 128
AUDIO_CTX_DIM = 2048
EPS = 1e-6
PROMPT_LEN = 32

# RoPE
ROPE_THETA = 10000.0
VIDEO_ROPE_MAX_POS = [20, 2048, 2048]
AUDIO_ROPE_MAX_POS = [20]
CROSS_PE_MAX_POS = [20]

# AV-mode auxiliary
AUDIO_N = 256  # sp-aligned for sp ∈ {1, 2, 4, 8}
# 22B checkpoint variants (AV mode only — video mode uses random scaled weights).
#   "fast" = distilled-1.1, what the production Fast pipeline / stage-2 actually runs.
#   "dev"  = base 22B, the non-distilled checkpoint.
# Both share the same architecture/shapes, so the test config is identical; only the
# weight values differ. LTX_CHECKPOINT overrides whichever variant is selected.
_CHECKPOINT_22B_VARIANTS = {
    "fast": os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"),
    "dev": os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"),
}


def _resolve_checkpoint_22b(variant: str) -> str:
    """Path for the selected 22B variant; LTX_CHECKPOINT env wins if set."""
    return os.environ.get("LTX_CHECKPOINT", _CHECKPOINT_22B_VARIANTS[variant])


# Block-test torch reference uses small scaled random weights; default init blows
# up through `(1+scale)*x` adaln chains in fp32 with no training signal.
WEIGHT_SEED = 1234
INPUT_SEED = 42
TIMESTEP_VAL = 0.01

# Toggle PCC verification via env (mirrors Wan's `dit_unit_test`). Default ON.
_RUN_PCC_DEFAULT = {"1": False, "0": True}.get(os.environ.get("LTX_SKIP_PCC"), True)


# ---------------------------------------------------------------------------
# Parametrize lists
# ---------------------------------------------------------------------------
_LTX_TRANSFORMER_MESH_PARAMS = [
    pytest.param((1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, False, id="1x1sp0tp1"),
    # 2x4sp0tp1 keeps is_fsdp=True for FSDP-path coverage on a 2D mesh.
    pytest.param((2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    # 2x4sp1tp0 mirrors the production BH 2x4 config (from_pretrained): sp_axis=1/tp_axis=0,
    # num_links=2, is_fsdp=False. Production handles weight residency via dynamic_load (whole-model
    # load/evict), NOT per-layer FSDP weight all-gathers — so is_fsdp must stay False here or the
    # profile shows phantom weight-gather collectives that the real run never issues.
    pytest.param((2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="2x4sp1tp0"),
    pytest.param((4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="ring_bh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="line_bh_4x8sp1tp0"),
]

# 1080p fast-pipeline latent volumes, sp/tile-aligned (TILE * sp_factor = 128
# for sp=4). Production pads stage_1 9690→9728 and stage_2 38760→38784.
_LTX_TRANSFORMER_SHAPE_PARAMS = [
    pytest.param(19, 16, 32, id="stage_1"),  # F·H·W = 9728
    pytest.param(19, 32, 64, id="stage_2"),  # F·H·W = 38912
]

_LTX_TRANSFORMER_MODALITY_PARAMS = [
    pytest.param(False, id="video"),
    pytest.param(True, id="av"),
]

_LTX_TRANSFORMER_RUN_PCC_PARAMS = [pytest.param(_RUN_PCC_DEFAULT, id="pcc" if _RUN_PCC_DEFAULT else "nopcc")]

# Checkpoint variant for AV mode. "fast" (distilled) is the production stage-2 path and the
# default; select "dev" with `-k ckpt_dev` to exercise the base-22B weights. Ignored in video
# mode (random scaled weights). Override the resolved file with LTX_CHECKPOINT.
_LTX_TRANSFORMER_CKPT_PARAMS = [
    pytest.param("fast", id="ckpt_fast"),
    pytest.param("dev", id="ckpt_dev"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interleaved_to_bhnd(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, dim) INTERLEAVED freqs → (B, num_heads, N, head_dim) for rotary_embedding_llama."""
    B, N, dim = t.shape
    head_dim = dim // num_heads
    return t.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _video_grid(F, H, W):
    """Return (grid_t, grid_h, grid_w) flat tensors and stacked 3D positions (1, N, 3)."""
    t_ids, h_ids, w_ids = torch.arange(F), torch.arange(H), torch.arange(W)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    gt_f, gh_f, gw_f = gt.flatten(), gh.flatten(), gw.flatten()
    positions = torch.stack([gt_f, gh_f, gw_f], dim=-1).float().unsqueeze(0)
    return gt_f, gh_f, gw_f, positions


def _video_rope_freqs(F, H, W, rope_type=LTXRopeType.SPLIT):
    _, _, _, positions = _video_grid(F, H, W)
    return precompute_freqs_cis(
        positions,
        dim=DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=VIDEO_ROPE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _video_cross_pe_freqs(F, H, W, rope_type=LTXRopeType.SPLIT):
    """T-only cross-PE for V↔A, dim matches audio (AUDIO_DIM)."""
    _, _, _, positions = _video_grid(F, H, W)
    return precompute_freqs_cis(
        positions[..., 0:1],
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=CROSS_PE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _audio_rope_freqs(audio_N, rope_type=LTXRopeType.SPLIT):
    a_pos = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    return precompute_freqs_cis(
        a_pos,
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=AUDIO_ROPE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _audio_cross_pe_freqs(audio_N, rope_type=LTXRopeType.SPLIT):
    a_pos = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    return precompute_freqs_cis(
        a_pos,
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=CROSS_PE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _tt_rope(freqs_fn, *args, mesh_device, sp_axis, tp_axis):
    """Build INTERLEAVED freqs for the TT runtime: precompute → BHND reshape → 2D shard."""
    cos_i, sin_i = freqs_fn(*args, rope_type=LTXRopeType.INTERLEAVED)
    cos_i = _interleaved_to_bhnd(cos_i, NUM_HEADS)
    sin_i = _interleaved_to_bhnd(sin_i, NUM_HEADS)
    tt_cos = bf16_tensor_2dshard(cos_i, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_i, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    return tt_cos, tt_sin


def _tt_rope_full(freqs_fn, *args, mesh_device, tp_axis):
    """INTERLEAVED freqs replicated full-seq (cross-attn K side): BHND reshape, no SP shard."""
    cos_i, sin_i = freqs_fn(*args, rope_type=LTXRopeType.INTERLEAVED)
    cos_i = _interleaved_to_bhnd(cos_i, NUM_HEADS)
    sin_i = _interleaved_to_bhnd(sin_i, NUM_HEADS)
    tt_cos = bf16_tensor(cos_i, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_sin = bf16_tensor(sin_i, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    return tt_cos, tt_sin


def _make_tt_block(*, mesh_device, ccl_manager, parallel_config, is_fsdp, has_audio):
    return LTXTransformerBlock(
        video_dim=DIM,
        video_ffn_dim=DIM * 4,
        video_num_heads=NUM_HEADS,
        video_cross_attention_dim=CTX_DIM,
        audio_dim=AUDIO_DIM,
        audio_ffn_dim=AUDIO_DIM * 4,
        audio_num_heads=NUM_HEADS,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        apply_gated_attention=has_audio,
        cross_attention_adaln=True,
    )


def _make_tt_model(*, mesh_device, ccl_manager, parallel_config, is_fsdp, has_audio, num_layers):
    return LTXTransformerModel(
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_layers=num_layers,
        cross_attention_dim=CTX_DIM,
        audio_num_attention_heads=NUM_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        audio_out_channels=AUDIO_IN_CHANNELS,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        apply_gated_attention=has_audio,
        cross_attention_adaln=True,
    )


def _load_22b_state_dict(num_layers: int, checkpoint_path: str) -> dict | None:
    """Load LTX-2.3-22B weights (`checkpoint_path`) filtered to the first `num_layers` blocks.

    Returns None when the checkpoint is missing (caller should skip).
    """
    if not os.path.exists(checkpoint_path):
        return None
    raw = load_file(checkpoint_path)
    prefix = "model.diffusion_model."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    return {k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) < num_layers}


def _scale_init_(module: torch.nn.Module, seed: int = WEIGHT_SEED) -> None:
    """Deterministic N(0, 0.1²) reinit in-place. See block test comment for why."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, dtype=p.dtype) * 0.1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    F: int,
    H: int,
    W: int,
    has_audio: bool,
    run_pcc: bool,
    checkpoint_variant: str,
    reset_seeds,
) -> None:
    """Test LTXTransformerBlock: TT forward, with optional PCC vs ltx_core."""
    # Checkpoint variant only affects AV weight loading; skip the redundant copy in video mode.
    if not has_audio and checkpoint_variant != "fast":
        pytest.skip("checkpoint_variant only affects AV mode (video uses random scaled weights)")
    checkpoint_22b = _resolve_checkpoint_22b(checkpoint_variant)
    video_N = F * H * W
    audio_N = AUDIO_N
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    assert video_N % (32 * sp_factor) == 0, f"video_N={video_N} not sp/tile-aligned for sp={sp_factor}"
    assert audio_N % (32 * sp_factor) == 0, f"audio_N={audio_N} not sp/tile-aligned for sp={sp_factor}"

    # The AV-block torch reference requires LTXModel-level cross_scale_shift_timestep /
    # cross_gate_timestep / cross-PE plumbing not provided here, so AV mode asserts
    # shape + finiteness only; video mode runs the full PCC compare.
    do_pcc = run_pcc and not has_audio
    torch_block = None
    torch_out = None
    if do_pcc:
        from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
        from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

        video_cfg = TransformerConfig(
            dim=DIM, heads=NUM_HEADS, d_head=HEAD_DIM, context_dim=CTX_DIM, cross_attention_adaln=True
        )
        torch_block = BasicAVTransformerBlock(idx=0, video=video_cfg, audio=None, rope_type=RefRopeType.SPLIT)
        torch_block.eval()
        _scale_init_(torch_block)

    # TT block
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_block = _make_tt_block(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
    )

    if do_pcc:
        # Clone so TT-side `_prepare_torch_state` views can't alias torch params.
        tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in torch_block.state_dict().items()})
    elif has_audio:
        sd = _load_22b_state_dict(num_layers=1, checkpoint_path=checkpoint_22b)
        if sd is None:
            pytest.skip(f"22B checkpoint not found at {checkpoint_22b}")
        block_sd = {
            k[len("transformer_blocks.0.") :]: v for k, v in sd.items() if k.startswith("transformer_blocks.0.")
        }
        tt_block.load_torch_state_dict(block_sd, strict=False)
    else:
        # video mode, PCC disabled — harvest a shape-matching state dict from a throwaway block.
        from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
        from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

        video_cfg = TransformerConfig(
            dim=DIM, heads=NUM_HEADS, d_head=HEAD_DIM, context_dim=CTX_DIM, cross_attention_adaln=True
        )
        dummy = BasicAVTransformerBlock(idx=0, video=video_cfg, audio=None, rope_type=RefRopeType.SPLIT)
        _scale_init_(dummy)
        tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in dummy.state_dict().items()})
        del dummy

    # Inputs
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(1, video_N, DIM, dtype=torch.float32)
    context = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    temb = torch.randn(1, 1, 9 * DIM, dtype=torch.float32)  # 9 adaln params
    prompt_temb = torch.randn(1, 1, 2 * DIM, dtype=torch.float32)  # 2 adaln params for prompt
    embedded_timestep = torch.randn(1, DIM, dtype=torch.float32)

    # Torch forward (video-only branch): reference consumes SPLIT cos/sin.
    if do_pcc:
        from ltx_core.model.transformer.transformer import TransformerArgs

        cos_split, sin_split = _video_rope_freqs(F, H, W, rope_type=LTXRopeType.SPLIT)
        with torch.no_grad():
            args = TransformerArgs(
                x=x,
                context=context,
                context_mask=None,
                timesteps=temb,
                embedded_timestep=embedded_timestep,
                positional_embeddings=(cos_split, sin_split),
                cross_positional_embeddings=None,
                cross_scale_shift_timestep=None,
                cross_gate_timestep=None,
                enabled=True,
                prompt_timestep=prompt_temb,
            )
            out_args, _ = torch_block(video=args, audio=None)
            torch_out = out_args.x
        logger.info(f"torch block output {tuple(torch_out.shape)}")

    # TT video tensors. Runtime rope is INTERLEAVED + trans_mat.
    spatial = x.unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)
    tt_temb = bf16_tensor(temb.reshape(1, 9, DIM).unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
    tt_prompt_temb = bf16_tensor(prompt_temb.reshape(1, 2, DIM).unsqueeze(0), device=mesh_device)
    tt_cos, tt_sin = _tt_rope(_video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    forward_kwargs = dict(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=video_N,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_prompt_temb=tt_prompt_temb,
    )
    if has_audio:
        a_x = torch.randn(1, audio_N, AUDIO_DIM, dtype=torch.float32)
        a_ctx = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)
        a_temb = torch.randn(1, 1, 9 * AUDIO_DIM, dtype=torch.float32)
        a_prompt_temb = torch.randn(1, 1, 2 * AUDIO_DIM, dtype=torch.float32)
        av_ca_v = torch.randn(1, 1, 5 * DIM, dtype=torch.float32)  # 4 scale-shift + 1 gate
        av_ca_a = torch.randn(1, 1, 5 * AUDIO_DIM, dtype=torch.float32)

        a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        vx_cos, vx_sin = _tt_rope(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        ax_cos, ax_sin = _tt_rope(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        vx_cos_full, vx_sin_full = _tt_rope_full(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, tp_axis=tp_axis
        )
        ax_cos_full, ax_sin_full = _tt_rope_full(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis
        )

        forward_kwargs.update(
            audio_1BND=bf16_tensor_2dshard(
                a_x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            audio_prompt=bf16_tensor(a_ctx.unsqueeze(0), device=mesh_device),
            audio_temb=bf16_tensor(
                a_temb.reshape(1, 9, AUDIO_DIM).unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            audio_prompt_temb=bf16_tensor(a_prompt_temb.reshape(1, 2, AUDIO_DIM).unsqueeze(0), device=mesh_device),
            av_ca_temb=bf16_tensor(
                av_ca_v.reshape(1, 5, DIM).unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            av_ca_audio_temb=bf16_tensor(
                av_ca_a.reshape(1, 5, AUDIO_DIM).unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            audio_N=audio_N,
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            video_cross_pe_cos_full=vx_cos_full,
            video_cross_pe_sin_full=vx_sin_full,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
        )

    tt_out = tt_block(**forward_kwargs)
    if has_audio:
        tt_v, tt_a = tt_out
    else:
        tt_v = tt_out

    # Gather video output
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_v_torch = ttnn.to_torch(
        tt_v,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    assert tt_v_torch.shape == (1, video_N, DIM), f"video shape {tt_v_torch.shape}"
    assert torch.isfinite(tt_v_torch).all(), "video output NaN/Inf"

    if has_audio:
        tt_a_torch = ttnn.to_torch(
            tt_a,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        ).squeeze(0)
        assert tt_a_torch.shape == (1, audio_N, AUDIO_DIM), f"audio shape {tt_a_torch.shape}"
        assert torch.isfinite(tt_a_torch).all(), "audio output NaN/Inf"

    if do_pcc:
        # 4x8 mesh has 8-way SP ring all-gathers — looser tolerance.
        pcc = 0.988 if mesh_device.get_num_devices() > 8 else 0.999
        rmse = 0.10 if mesh_device.get_num_devices() > 8 else 0.032
        assert_quality(torch_out, tt_v_torch, pcc=pcc, relative_rmse=rmse)
        logger.info(f"PASSED block PCC: video {tuple(tt_v_torch.shape)}")
    else:
        logger.info(f"PASSED block (no PCC): video {tuple(tt_v_torch.shape)}")


def _run_inner_step(
    *,
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant: str,
    use_forward_alias: bool,
):
    """Shared body for model and inner_step tests.

    `use_forward_alias=True` invokes ``tt_model(...)`` (which delegates to
    inner_step); `False` calls ``tt_model.inner_step(...)`` explicitly. Used
    so the two tests have visibly different call sites.
    """
    # Checkpoint variant only affects AV weight loading; skip the redundant copy in video mode.
    if not has_audio and checkpoint_variant != "fast":
        pytest.skip("checkpoint_variant only affects AV mode (video uses random scaled weights)")
    checkpoint_22b = _resolve_checkpoint_22b(checkpoint_variant)
    video_N = F * H * W
    audio_N = AUDIO_N
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    assert video_N % (32 * sp_factor) == 0
    if has_audio:
        assert audio_N % (32 * sp_factor) == 0

    # === Reference model ===
    do_pcc = run_pcc
    ref_video = None
    ref_audio = None
    state_dict = None

    if has_audio:
        # AV PCC path: 22B with strict=False.
        state_dict = _load_22b_state_dict(num_layers=1, checkpoint_path=checkpoint_22b)
        if state_dict is None:
            pytest.skip(f"22B checkpoint not found at {checkpoint_22b}")
    else:
        # Video PCC path: random scaled weights (1 layer).
        from ltx_core.model.transformer.model import LTXModel, LTXModelType
        from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType

        torch_model = LTXModel(
            model_type=LTXModelType.VideoOnly,
            num_layers=1,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            cross_attention_dim=CTX_DIM,
            use_middle_indices_grid=True,
            cross_attention_adaln=True,
            rope_type=RefRopeType.SPLIT,
        )
        torch_model.eval()
        _scale_init_(torch_model)
        state_dict = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

    # === Inputs ===
    torch.manual_seed(INPUT_SEED)
    video_lat = torch.randn(1, video_N, IN_CHANNELS, dtype=torch.float32)
    video_prompt = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    sigma_val = TIMESTEP_VAL if not has_audio else 0.5  # AV path uses production sigma=0.5
    timestep_torch = torch.tensor([sigma_val])

    audio_lat = None
    audio_prompt = None
    if has_audio:
        audio_lat = torch.randn(1, audio_N, AUDIO_IN_CHANNELS, dtype=torch.float32)
        audio_prompt = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)

    # === Reference forward (before TT to avoid weight aliasing) ===
    # The torch LTXModel computes rope internally from `positions` (SPLIT), so the
    # reference needs no precomputed cos/sin — only the TT side builds INTERLEAVED freqs.
    if do_pcc and not has_audio:
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
        from ltx_core.model.transformer.model import Modality

        _, _, _, positions_3d = _video_grid(F, H, W)
        positions_for_ref = torch.stack([positions_3d[0, :, 0], positions_3d[0, :, 1], positions_3d[0, :, 2]], dim=0)
        positions_for_ref = torch.stack([positions_for_ref, positions_for_ref], dim=-1).unsqueeze(0)
        video_mod = Modality(
            latent=video_lat,
            sigma=torch.tensor([sigma_val]),
            timesteps=torch.ones(1, video_N) * sigma_val,
            positions=positions_for_ref,
            context=video_prompt,
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])
        with torch.no_grad():
            ref_video, _ = torch_model(video=video_mod, audio=None, perturbations=perturbations)
        logger.info(f"ref video {tuple(ref_video.shape)} range=[{ref_video.min():.3f}, {ref_video.max():.3f}]")
        del torch_model

    if do_pcc and has_audio:
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
        from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
        from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType

        ref_model = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_layers=1,
            num_attention_heads=NUM_HEADS,
            attention_head_dim=HEAD_DIM,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            cross_attention_dim=CTX_DIM,
            audio_num_attention_heads=NUM_HEADS,
            audio_attention_head_dim=AUDIO_HEAD_DIM,
            audio_in_channels=AUDIO_IN_CHANNELS,
            audio_out_channels=AUDIO_IN_CHANNELS,
            audio_cross_attention_dim=AUDIO_CTX_DIM,
            use_middle_indices_grid=True,
            apply_gated_attention=True,
            cross_attention_adaln=True,
            rope_type=RefRopeType.SPLIT,
        )
        ref_model.load_state_dict(state_dict, strict=False)
        ref_model.eval()

        gt_f, gh_f, gw_f, _ = _video_grid(F, H, W)
        v_pos_ref = torch.stack([gt_f, gh_f, gw_f], dim=0).float().unsqueeze(-1).repeat(1, 1, 2).unsqueeze(0)
        a_pos_ref = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 2)
        v_modality = Modality(
            latent=video_lat,
            sigma=torch.tensor([sigma_val]),
            timesteps=torch.ones(1, video_N) * sigma_val,
            positions=v_pos_ref,
            context=video_prompt,
            enabled=True,
        )
        a_modality = Modality(
            latent=audio_lat,
            sigma=torch.tensor([sigma_val]),
            timesteps=torch.ones(1, audio_N) * sigma_val,
            positions=a_pos_ref,
            context=audio_prompt,
            enabled=True,
        )
        perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])
        t0 = time.time()
        with torch.no_grad():
            ref_video, ref_audio = ref_model(video=v_modality, audio=a_modality, perturbations=perturbations)
        logger.info(
            f"AV ref forward: {time.time() - t0:.1f}s — video {tuple(ref_video.shape)} audio {tuple(ref_audio.shape)}"
        )
        del ref_model
        import gc

        gc.collect()

    # === TT model ===
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_model = _make_tt_model(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        num_layers=1,
    )
    t0 = time.time()
    tt_model.load_torch_state_dict(state_dict, strict=not has_audio)
    logger.info(f"state-dict load: {time.time() - t0:.1f}s")

    # === Build TT-side RoPE / cross-PE / prompt tensors (INTERLEAVED + trans_mat) ===
    tt_video_prompt = bf16_tensor(video_prompt.unsqueeze(0), device=mesh_device)
    tt_vc, tt_vs = _tt_rope(_video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    call_kwargs = dict(
        video_1BNI_torch=video_lat.unsqueeze(0),
        video_prompt_1BLP=tt_video_prompt,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        video_N=video_N,
        trans_mat=tt_trans_mat,
        timestep_torch=timestep_torch,
    )
    if has_audio:
        a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        vx_cos, vx_sin = _tt_rope(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        ax_cos, ax_sin = _tt_rope(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        vx_cos_full, vx_sin_full = _tt_rope_full(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, tp_axis=tp_axis
        )
        ax_cos_full, ax_sin_full = _tt_rope_full(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis
        )
        call_kwargs.update(
            audio_1BNI_torch=audio_lat.unsqueeze(0),
            audio_prompt_1BLP=bf16_tensor(audio_prompt.unsqueeze(0), device=mesh_device),
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            audio_N=audio_N,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            video_cross_pe_cos_full=vx_cos_full,
            video_cross_pe_sin_full=vx_sin_full,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
        )

    # === Forward ===
    t0 = time.time()
    result = tt_model(**call_kwargs) if use_forward_alias else tt_model.inner_step(**call_kwargs)
    logger.info(f"TT forward: {time.time() - t0:.1f}s")

    if has_audio:
        tt_v_dev, tt_a_dev = result
        tt_video = LTXTransformerModel.device_to_host(tt_v_dev).squeeze(0)
        tt_audio = LTXTransformerModel.device_to_host(tt_a_dev).squeeze(0)
        assert tt_video.shape == (1, video_N, OUT_CHANNELS), f"video shape {tt_video.shape}"
        assert tt_audio.shape == (1, audio_N, AUDIO_IN_CHANNELS), f"audio shape {tt_audio.shape}"
        assert torch.isfinite(tt_video).all() and torch.isfinite(tt_audio).all(), "NaN/Inf in TT output"
    else:
        tt_video = LTXTransformerModel.device_to_host(result).squeeze(0)
        assert tt_video.shape == (1, video_N, OUT_CHANNELS), f"video shape {tt_video.shape}"
        assert torch.isfinite(tt_video).all(), "NaN/Inf in TT output"

    del tt_model

    if do_pcc:
        assert_quality(ref_video, tt_video, pcc=0.992, relative_rmse=0.15)
        if has_audio:
            assert_quality(ref_audio, tt_audio, pcc=0.992, relative_rmse=0.15)
        logger.info("PASSED PCC")
    else:
        logger.info("PASSED (no PCC)")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_model(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """Full LTXTransformerModel forward (delegates to inner_step)."""
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=has_audio,
        run_pcc=run_pcc,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=True,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_inner_step(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """LTXTransformerModel.inner_step — denoising-loop path."""
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=has_audio,
        run_pcc=run_pcc,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=False,
    )
