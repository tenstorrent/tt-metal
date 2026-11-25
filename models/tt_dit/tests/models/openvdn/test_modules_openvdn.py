# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN modules on device vs the CPU reference (`reference_hybrid_openvdn.py`, copied from OpenVDN/vdn-minimax-h3;
the H3 parts are the pinned diffusers `transformer_minimax_h3`).

Components: the linear branch's features (K/V short conv, SiLU, L2 norm), its decay alpha, its text state, the
whole linear branch, the hybrid attention, and the VDN transformer block. Full model width (hidden 5376, heads of
128; 56 heads for the linear branch, 14 for the attention and block, see ATTN_HEADS), random weights rounded to bf16 (the checkpoint's dtype) and a small packed sequence: 64 text + 32 audio
rows, then 17 latent frames of a 6 x 16 grid (non-square, so a swapped conv axis shows). The reference runs in fp32.

Every check compares PCC and relative RMS error (RMSE / std of the reference): modules with bf16 activations must
reach PCC >= 0.9999 and RMSE/std <= 1 %, alpha (fp32 throughout) PCC >= 0.99999 and RMSE/std <= 0.5 %. The stock H3
attention and block are run on the same inputs as a baseline (recorded, not asserted).

Set OPENVDN_CHECKS=<file> to append one JSON line per check.

Run (one chip): pytest models/tt_dit/tests/models/openvdn/test_modules_openvdn.py -s --tb=short
Needs the pinned diffusers (models/tt_dit/models/MiniMaxH3.md, "Setup") for the attention and block tests.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from models.tt_dit.models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from models.tt_dit.models.transformers.openvdn import reference_hybrid_openvdn as ref_h
from models.tt_dit.models.transformers.openvdn.hybrid_attention_openvdn import HybridLayout, OpenVDNHybridAttention
from models.tt_dit.models.transformers.openvdn.linear_branch_openvdn import OpenVDNLinearBranch
from models.tt_dit.models.transformers.openvdn.reference_openvdn import window_bounds
from models.tt_dit.models.transformers.openvdn.transformer_block_openvdn import OpenVDNTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.minimax_h3.common import (
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    packed_layout,
    randomize_norm_weights,
    upload_rope,
)
from models.tt_dit.utils import tensor

FP32_PCC, FP32_RMSE = 0.99999, 0.005
BF16_PCC, BF16_RMSE = 0.9999, 0.01
# LOWERED, for the softmax-only (global) rows of the hybrid attention: each text/audio query attends densely over the
# whole sequence (1728 keys) with bf16 q/k/v, measured 1.3 % RMSE/std, and still 1.27 % with every H3 projection at
# HiFi4 (diagnostic below), so the floor is the bf16 dense softmax, not the projections. The stock H3 attention (dense
# for every row) is at 3.2 % on the same inputs. The video rows and the whole module are held to BF16_RMSE.
GLOBAL_ROWS_RMSE = 0.02

HIDDEN = REAL_BLOCK_CONFIG["hidden_size"]
HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
CHANNELS = HEADS * HEAD_DIM
# The attention and block tests run 14 heads, the per-chip head count of the 4x8 mesh (TP 4): H3's fused per-head QK
# norm (`dit_fused_distributed_rmsnorm`) does not fit 56 heads on one chip in L1 (1.89 MB of 1.57 MB), for the stock
# H3 attention as much as for the hybrid one.
ATTN_HEADS = HEADS // 4
NUM_TEXT, NUM_AUDIO = 64, 32
NUM_FRAMES = 17
GRID = (6, 16)
PER_FRAME = GRID[0] * GRID[1]
NUM_VIDEO = NUM_FRAMES * PER_FRAME
SEQ_LEN = NUM_TEXT + NUM_AUDIO + NUM_VIDEO
CHUNK, RADIUS = 5, 1  # released checkpoints: c1 window, anchors dense both ways

ONE_CHIP = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
DEVICE_PARAMS = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


def _metrics(out: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a = expected.detach().flatten().double()
    b = out.detach().flatten().double()
    cov = torch.cov(torch.stack([a, b]))
    pcc = (cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item()
    rmse = (torch.nn.functional.mse_loss(a, b).sqrt() / cov[0, 0].sqrt()).item()
    return pcc, rmse


def _write(records: list[dict]) -> None:
    path = os.environ.get("OPENVDN_CHECKS")
    if path:
        with open(path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


def _check(name: str, out: torch.Tensor, expected: torch.Tensor, *, pcc: float, rmse: float, **details) -> None:
    pcc_found, rmse_found = _metrics(out, expected)
    passed = not math.isnan(pcc_found) and pcc_found >= pcc and rmse_found <= rmse
    logger.info(f"{name}: PCC {pcc_found:.7f} (>= {pcc}), RMSE/std {rmse_found:.2e} (<= {rmse}) {details}")
    _write(
        [
            dict(name=name, metric=metric, value=value, threshold=threshold, passed=passed, **details)
            for metric, value, threshold in (("pcc", pcc_found, pcc), ("rel_rmse", rmse_found, rmse))
        ]
    )
    assert passed, f"{name}: PCC {pcc_found:.7f} < {pcc} or RMSE/std {rmse_found:.2e} > {rmse}"


def _record(name: str, out: torch.Tensor, expected: torch.Tensor, **details) -> None:
    """A baseline: logged and recorded, not asserted."""
    pcc_found, rmse_found = _metrics(out, expected)
    logger.info(f"{name} (baseline): PCC {pcc_found:.7f}, RMSE/std {rmse_found:.2e} {details}")
    _write(
        [
            dict(name=name, metric=metric, value=value, threshold=None, passed=None, baseline=True, **details)
            for metric, value in (("pcc", pcc_found), ("rel_rmse", rmse_found))
        ]
    )


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def _round_params(module: torch.nn.Module) -> torch.nn.Module:
    """Weights as the checkpoint stores them (bf16), evaluated in fp32."""
    for p in module.parameters():
        p.data = _bf16(p.data)
    return module


def _randomize_hybrid(model: ref_h.HybridAttention, gen: torch.Generator) -> None:
    """A live softmax gate (it initialises to a constant 0.99) and a non-trivial branch norm."""
    model.softmax_gate.up.weight.data = 0.02 * torch.randn(model.softmax_gate.up.weight.shape, generator=gen)
    norm = model.linear_attention.norm.weight
    norm.data = 1.0 + 0.5 * torch.randn(norm.shape, generator=gen)


def _from_torch(x: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return tensor.from_torch(x.reshape(1, 1, *x.shape[-2:]), device=device, dtype=dtype)


def _to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()


def _branch(gen_seed: int = 0) -> ref_h.BidirectionalLinearBranch:
    torch.manual_seed(gen_seed)
    branch = ref_h.BidirectionalLinearBranch(HIDDEN, HEADS, HEAD_DIM)
    branch.norm.weight.data = 1.0 + 0.5 * torch.randn(HEAD_DIM)
    return _round_params(branch).eval()


def _tt_branch(mesh_device, branch: ref_h.BidirectionalLinearBranch) -> OpenVDNLinearBranch:
    tt = OpenVDNLinearBranch(hidden_size=HIDDEN, num_heads=HEADS, head_dim=HEAD_DIM, mesh_device=mesh_device)
    tt.load_torch_state_dict(branch.state_dict())
    return tt


def _raw(rows: int, gen: torch.Generator, scale: float = 0.5) -> torch.Tensor:
    """bf16-valued raw projections [rows, H, d] at the scale of `to_q/k/v(x)`."""
    return _bf16(scale * torch.randn(rows, HEADS, HEAD_DIM, generator=gen))


def _hidden(rows: int, gen: torch.Generator) -> torch.Tensor:
    return _bf16(torch.randn(rows, HIDDEN, generator=gen))


def _parallel(mesh_device):
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=0, factor=1),
        sequence_parallel=ParallelFactor(mesh_axis=1, factor=1),
        cfg_parallel=None,
    )
    return ccl_manager, parallel_config


def _layouts():
    video_start = NUM_TEXT + NUM_AUDIO
    ref_layout = ref_h.SequenceLayout(
        seq_len=SEQ_LEN,
        video_start=video_start,
        num_frames=NUM_FRAMES,
        tokens_per_frame=PER_FRAME,
        frame_height=GRID[0],
        frame_width=GRID[1],
        text_start=0,
        text_len=NUM_TEXT,
    )
    tt_layout = HybridLayout(
        seq_len=SEQ_LEN,
        video_start=video_start,
        num_frames=NUM_FRAMES,
        frame_height=GRID[0],
        frame_width=GRID[1],
        text_start=0,
        text_len=NUM_TEXT,
    )
    return ref_layout, tt_layout


def _rope(mesh_device):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    position_ids, token_tags, timestep_indices = packed_layout(NUM_TEXT, NUM_AUDIO, NUM_VIDEO, grid_hw=GRID)
    cos, sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)(position_ids)
    tt_cos, tt_sin = upload_rope(*prepare_rope_tables(cos, sin, HEAD_DIM), mesh_device=mesh_device, sp_axis=1)
    return (cos, sin), (tt_cos, tt_sin), token_tags, timestep_indices


# ---- linear branch pieces ----


@ONE_CHIP
@pytest.mark.parametrize("proj", ["q", "k", "v"])
def test_linear_features(mesh_device: ttnn.MeshDevice, proj: str) -> None:
    """SiLU [+ L2 norm] of the raw projection; K and V through the 5x5 spatial + 5-tap temporal depthwise conv."""
    gen = torch.Generator().manual_seed(1)
    branch = _branch()
    tokens = _raw(NUM_VIDEO, gen)
    with torch.no_grad():
        expected = branch.feature(tokens, proj, NUM_FRAMES, GRID)
    expected = expected.view(NUM_FRAMES, PER_FRAME, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    tt = _tt_branch(mesh_device, branch)
    out = tt.features(_from_torch(tokens.reshape(NUM_VIDEO, CHANNELS), mesh_device), proj, NUM_FRAMES, GRID)
    _check(
        f"linear_features_{proj}",
        _to_torch(out),
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        details=("conv 5x5 + 5-tap, " if proj in "kv" else "") + f"F{NUM_FRAMES} grid {GRID[0]}x{GRID[1]}",
    )


@ONE_CHIP
def test_linear_alpha(mesh_device: ttnn.MeshDevice) -> None:
    """Per-frame decay `exp(-exp(A_log) * softplus(up(down(frame mean)) + dt_bias))`, fp32 as in the reference."""
    gen = torch.Generator().manual_seed(2)
    branch = _branch()
    xv = _hidden(NUM_VIDEO, gen)
    with torch.no_grad():
        expected = branch.frame_alpha(xv, NUM_FRAMES, PER_FRAME).reshape(NUM_FRAMES, CHANNELS)

    tt = _tt_branch(mesh_device, branch)
    out = _to_torch(tt.alpha(_from_torch(xv, mesh_device), NUM_FRAMES)).reshape(NUM_FRAMES, CHANNELS)
    _check("linear_alpha", out, expected, pcc=FP32_PCC, rmse=FP32_RMSE, details=f"alpha in [{expected.min():.3f}, 1)")
    # alpha compounds over the clip, so also compare the log decay the bridge sums.
    _check("linear_log_alpha", out.log(), expected.log(), pcc=FP32_PCC, rmse=FP32_RMSE)


@ONE_CHIP
def test_linear_text_state(mesh_device: ttnn.MeshDevice) -> None:
    """Half the prompt written into a zero state as one delta-rule chunk: `0.5 * B (I + A)^-1`, A and B over all
    prompt rows (no conv)."""
    gen = torch.Generator().manual_seed(3)
    branch = _branch()
    text_x, text_k, text_v = _hidden(NUM_TEXT, gen), _raw(NUM_TEXT, gen), _raw(NUM_TEXT, gen)
    with torch.no_grad():
        expected = branch.text_state(text_x, (None, text_k, text_v))

    tt = _tt_branch(mesh_device, branch)
    out = tt.text_state(
        _from_torch(text_x, mesh_device),
        _from_torch(text_k.reshape(NUM_TEXT, CHANNELS), mesh_device),
        _from_torch(text_v.reshape(NUM_TEXT, CHANNELS), mesh_device),
    )
    _check(
        "linear_text_state",
        _to_torch(out).reshape(expected.shape),
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        details=f"L{NUM_TEXT}",
    )


@ONE_CHIP
def test_linear_branch(mesh_device: ttnn.MeshDevice) -> None:
    """The whole branch, as the released checkpoints run it: anchors skipped, both scans from the text state."""
    gen = torch.Generator().manual_seed(4)
    branch = _branch()
    xv = _hidden(NUM_VIDEO, gen)
    qkv = tuple(_raw(NUM_VIDEO, gen) for _ in range(3))
    text_x, text_k, text_v = _hidden(NUM_TEXT, gen), _raw(NUM_TEXT, gen), _raw(NUM_TEXT, gen)
    bounds = window_bounds(NUM_FRAMES, RADIUS, CHUNK)
    with torch.no_grad():
        expected = branch(
            xv,
            NUM_FRAMES,
            PER_FRAME,
            bounds,
            qkv,
            GRID,
            skip_ends=True,
            text_x=text_x,
            text_qkv_raw=(None, text_k, text_v),
        )

    tt = _tt_branch(mesh_device, branch)
    flat = lambda t: _from_torch(t.reshape(t.shape[0], -1), mesh_device)  # noqa: E731
    out = tt(
        flat(xv),
        *(flat(t) for t in qkv),
        num_frames=NUM_FRAMES,
        frame_size=GRID,
        bounds=bounds,
        skip_ends=True,
        text=(flat(text_x), flat(text_k), flat(text_v)),
    )
    out = _to_torch(out).reshape(expected.shape)
    anchors = torch.cat([out[:PER_FRAME], out[-PER_FRAME:]])
    assert torch.count_nonzero(anchors) == 0, "anchor frames must read out exactly zero"
    _check(
        "linear_branch",
        out,
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        details=f"F{NUM_FRAMES} grid {GRID[0]}x{GRID[1]}, c{RADIUS} chunk {CHUNK}, anchors both, text L{NUM_TEXT}",
    )


# ---- hybrid attention and block ----


def _reference_attention(gen: torch.Generator):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention

    torch.manual_seed(5)
    orig = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN, heads=ATTN_HEADS, dim_head=HEAD_DIM, qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"]
    )
    randomize_norm_weights(orig)
    model = ref_h.HybridAttention(orig, HIDDEN, radius=RADIUS, chunk=CHUNK)
    _randomize_hybrid(model, gen)
    return _round_params(model).eval()


def _tt_attention_kwargs(mesh_device, rotary_dim: int) -> dict:
    ccl_manager, parallel_config = _parallel(mesh_device)
    return dict(
        hidden_size=HIDDEN,
        num_heads=ATTN_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=rotary_dim,
        qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"],
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )


@ONE_CHIP
@DEVICE_PARAMS
def test_hybrid_attention(mesh_device: ttnn.MeshDevice) -> None:
    """QKV + QK norm + RoPE (shared H3 code), window softmax with its per-head gate, `to_out`, plus the linear branch
    through `to_out_linear` on the video rows."""
    gen = torch.Generator().manual_seed(6)
    model = _reference_attention(gen)
    ref_layout, tt_layout = _layouts()
    model.layout = ref_layout
    rope, (tt_cos, tt_sin), _, _ = _rope(mesh_device)
    x = _bf16(torch.randn(1, SEQ_LEN, HIDDEN, generator=gen))
    with torch.no_grad():
        expected = model(x, rope)[0]
        saved = model.to_out_linear.weight.data.clone()
        model.to_out_linear.weight.data.zero_()
        softmax_only = model(x, rope)[0]
        model.to_out_linear.weight.data = saved
    linear_share = ((expected - softmax_only).norm() / expected.norm()).item()
    logger.info(f"linear branch share of the attention output: {linear_share:.3f}")

    tt_model = OpenVDNHybridAttention(radius=RADIUS, chunk=CHUNK, **_tt_attention_kwargs(mesh_device, rope[0].shape[-1]))
    tt_model.load_torch_state_dict(model.state_dict())
    tt_model.layout = tt_layout
    out = _to_torch(tt_model(_from_torch(x[0], mesh_device), rope_cos=tt_cos, rope_sin=tt_sin))
    out = out.reshape(SEQ_LEN, HIDDEN)

    vs, ve = ref_layout.video_start, ref_layout.video_end
    _check(
        "hybrid_attention",
        out,
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        details=f"N{SEQ_LEN} (text {NUM_TEXT}, audio {NUM_AUDIO}, video F{NUM_FRAMES} x {PER_FRAME}); "
        f"linear branch share {linear_share:.3f}",
    )
    _check("hybrid_attention_video_rows", out[vs:ve], expected[vs:ve], pcc=BF16_PCC, rmse=BF16_RMSE)
    _check(
        "hybrid_attention_global_rows",
        out[:vs],
        expected[:vs],
        pcc=BF16_PCC,
        rmse=GLOBAL_ROWS_RMSE,
        details="LOWERED RMSE threshold: bf16 dense softmax over all keys (see GLOBAL_ROWS_RMSE)",
    )

    # Diagnostic: the same module with every H3 projection at HiFi4, exact math (shows the projections are not the
    # floor of the global rows).
    tt_model.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = _to_torch(tt_model(_from_torch(x[0], mesh_device), rope_cos=tt_cos, rope_sin=tt_sin))
    out = out.reshape(SEQ_LEN, HIDDEN)
    _record("hybrid_attention_hifi4_projections", out, expected, details="whole module, H3 projections at HiFi4")
    _record("hybrid_attention_global_rows_hifi4_projections", out[:vs], expected[:vs])


@ONE_CHIP
@DEVICE_PARAMS
def test_h3_attention_baseline(mesh_device: ttnn.MeshDevice) -> None:
    """The stock H3 attention (dense softmax) on the same weights and inputs: the floor the hybrid inherits."""
    gen = torch.Generator().manual_seed(6)
    orig = _reference_attention(gen).orig
    rope, (tt_cos, tt_sin), _, _ = _rope(mesh_device)
    x = _bf16(torch.randn(1, SEQ_LEN, HIDDEN, generator=gen))
    with torch.no_grad():
        expected = orig(x, rope)[0]
    tt_model = MiniMaxH3Attention(**_tt_attention_kwargs(mesh_device, rope[0].shape[-1]))
    tt_model.load_torch_state_dict(orig.state_dict())
    out = _to_torch(tt_model(_from_torch(x[0], mesh_device), rope_cos=tt_cos, rope_sin=tt_sin)).reshape(SEQ_LEN, HIDDEN)
    _record("h3_attention_dense_baseline", out, expected, details=f"N{SEQ_LEN}, stock MiniMaxH3Attention")


def _block_inputs(mesh_device):
    """The same inputs for the VDN block and its H3 baseline."""
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    gen = torch.Generator().manual_seed(9)
    rope, (tt_cos, tt_sin), token_tags, timestep_indices = _rope(mesh_device)
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    x = _bf16(torch.randn(1, SEQ_LEN, HIDDEN, generator=gen))
    temb = torch.randn(num_timesteps, REAL_BLOCK_CONFIG["time_embed_dim"], generator=gen)
    tt_inputs = dict(
        spatial_1BND=_from_torch(x[0], mesh_device),
        logical_n=tensor.logical_length_tensor(SEQ_LEN, device=mesh_device),
        temb=tensor.from_torch(temb.reshape(1, 1, *temb.shape), device=mesh_device, dtype=ttnn.float32),
        adaln_indices=tensor.from_torch(
            adaln_indices.to(torch.int32).reshape(1, 1, 1, SEQ_LEN),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
        ),
        rope_cos=tt_cos,
        rope_sin=tt_sin,
    )
    return x, temb, adaln_indices, rope, tt_inputs


def _tt_block_kwargs(mesh_device, rotary_dim: int) -> dict:
    ccl_manager, parallel_config = _parallel(mesh_device)
    return dict(
        **{**TT_BLOCK_CONFIG, "num_heads": ATTN_HEADS},
        rotary_dim=rotary_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
    )


def _reference_block(gen: torch.Generator, hybrid: bool):
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block

    torch.manual_seed(7)
    block = TorchMiniMaxH3Block(**{**REAL_BLOCK_CONFIG, "num_attention_heads": ATTN_HEADS})
    randomize_norm_weights(block)
    if hybrid:
        block.attn = ref_h.HybridAttention(block.attn, HIDDEN, radius=RADIUS, chunk=CHUNK)
        _randomize_hybrid(block.attn, gen)
    return _round_params(block).eval()


@ONE_CHIP
@DEVICE_PARAMS
def test_transformer_block(mesh_device: ttnn.MeshDevice) -> None:
    """The VDN block: H3's AdaLN-modulated pre-norm attention and SwiGLU feed-forward, hybrid attention inside."""
    gen = torch.Generator().manual_seed(8)
    block = _reference_block(gen, hybrid=True)
    ref_layout, tt_layout = _layouts()
    block.attn.layout = ref_layout
    x, temb, adaln_indices, rope, tt_inputs = _block_inputs(mesh_device)
    with torch.no_grad():
        expected = block(x, temb, adaln_indices, rope)[0]

    tt_block = OpenVDNTransformerBlock(radius=RADIUS, chunk=CHUNK, **_tt_block_kwargs(mesh_device, rope[0].shape[-1]))
    tt_block.load_torch_state_dict(block.state_dict())
    tt_block.attn.layout = tt_layout
    out = _to_torch(tt_block(**tt_inputs)).reshape(SEQ_LEN, HIDDEN)
    _check("transformer_block", out, expected, pcc=BF16_PCC, rmse=BF16_RMSE, details=f"N{SEQ_LEN}, hybrid attention")


@ONE_CHIP
@DEVICE_PARAMS
def test_h3_block_baseline(mesh_device: ttnn.MeshDevice) -> None:
    """The stock H3 block (dense attention) on the same inputs: the floor the VDN block inherits."""
    gen = torch.Generator().manual_seed(8)
    block = _reference_block(gen, hybrid=False)
    x, temb, adaln_indices, rope, tt_inputs = _block_inputs(mesh_device)
    with torch.no_grad():
        expected = block(x, temb, adaln_indices, rope)[0]
    tt_block = MiniMaxH3TransformerBlock(**_tt_block_kwargs(mesh_device, rope[0].shape[-1]))
    tt_block.load_torch_state_dict(block.state_dict())
    out = _to_torch(tt_block(**tt_inputs)).reshape(SEQ_LEN, HIDDEN)
    _record("h3_block_dense_baseline", out, expected, details=f"N{SEQ_LEN}, stock MiniMaxH3TransformerBlock")
