# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The OpenVDN hybrid attention and VDN block on the Blackhole 4x8 mesh (TP 4 over heads, SP 8 over rows) vs the CPU
reference, at all 56 heads: the parallel path the pipeline runs (`hybrid_attention_openvdn.py`).

Same inputs and bars as the one-chip module tests (`test_modules_openvdn.py`), except the frame is 6 x 14 = 84 rows
(not a whole number of tiles, as the 768p frame of 24 x 42 = 1008 rows is not) and the packed length 1524 is padded
to the SP alignment (1536), so the pad tail and the unaligned frame boundaries are exercised.

Run: pytest models/tt_dit/tests/models/openvdn/test_mesh_openvdn.py -s --tb=short
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from models.tt_dit.models.transformers.openvdn import reference_hybrid_openvdn as ref_h
from models.tt_dit.models.transformers.openvdn.hybrid_attention_openvdn import HybridLayout, OpenVDNHybridAttention
from models.tt_dit.models.transformers.openvdn.transformer_block_openvdn import OpenVDNTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.minimax_h3.common import (
    MESH_4X8_RING,
    REAL_BLOCK_CONFIG,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TT_BLOCK_CONFIG,
    fabric_topology,
    packed_layout,
    randomize_norm_weights,
)
from models.tt_dit.tests.models.openvdn.test_modules_openvdn import (
    BF16_PCC,
    BF16_RMSE,
    GLOBAL_ROWS_RMSE,
    _bf16,
    _check,
    _randomize_hybrid,
    _round_params,
)
from models.tt_dit.utils import tensor

HIDDEN = REAL_BLOCK_CONFIG["hidden_size"]
HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
NUM_TEXT, NUM_AUDIO = 64, 32
NUM_FRAMES = 17
GRID = (6, 14)
PER_FRAME = GRID[0] * GRID[1]
NUM_VIDEO = NUM_FRAMES * PER_FRAME
SEQ_LEN = NUM_TEXT + NUM_AUDIO + NUM_VIDEO
CHUNK, RADIUS = 5, 1
TP_AXIS, SP_AXIS = 0, 1

MESH = pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=True)
#: Both packed orders the hybrid attention takes: logical (the SP pad tail only) and frame-aligned (stage 4).
LAYOUTS = pytest.mark.parametrize("aligned", [False, True], ids=["logical", "aligned"])


def _parallel(mesh_device):
    shape = tuple(mesh_device.shape)
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=fabric_topology(mesh_device))
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=TP_AXIS, factor=shape[TP_AXIS]),
        sequence_parallel=ParallelFactor(mesh_axis=SP_AXIS, factor=shape[SP_AXIS]),
        cfg_parallel=None,
    )
    return ccl_manager, parallel_config


def _layouts(aligned: bool = False):
    video_start = NUM_TEXT + NUM_AUDIO
    common = dict(
        seq_len=SEQ_LEN,
        video_start=video_start,
        num_frames=NUM_FRAMES,
        frame_height=GRID[0],
        frame_width=GRID[1],
        text_start=0,
        text_len=NUM_TEXT,
    )
    return ref_h.SequenceLayout(tokens_per_frame=PER_FRAME, **common), HybridLayout(**common, aligned=aligned)


class Packing:
    """Where the logical rows sit in the packed tensor the device takes: in order with an SP pad tail, or in the
    frame-aligned order (`HybridLayout.aligned_rows`)."""

    def __init__(self, mesh_device, layout: HybridLayout) -> None:
        sp = tuple(mesh_device.shape)[SP_AXIS]
        if layout.aligned:
            self.rows, self.padded_len = layout.aligned_rows(sp)
        else:
            align = sp * ttnn.TILE_SIZE
            self.padded_len = -(-layout.seq_len // align) * align
            self.rows = torch.arange(layout.seq_len)

    def pack(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, ...]` logical rows -> `[padded, ...]`, zeros elsewhere."""
        out = torch.zeros(self.padded_len, *x.shape[1:], dtype=x.dtype)
        out[self.rows] = x
        return out

    def unpack(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.rows]


def _rope(mesh_device, packing: Packing):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    position_ids, token_tags, timestep_indices = packed_layout(NUM_TEXT, NUM_AUDIO, NUM_VIDEO, grid_hw=GRID)
    cos, sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)(position_ids)
    tables = [packing.pack(t) for t in prepare_rope_tables(cos, sin, HEAD_DIM)]
    tt = [
        tensor.from_torch(
            t.reshape(1, 1, *t.shape), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., SP_AXIS, None]
        )
        for t in tables
    ]
    return (cos, sin), tt, token_tags, timestep_indices


def _upload_hidden(x: torch.Tensor, mesh_device, packing: Packing) -> ttnn.Tensor:
    """`[N, hidden]` -> `[1, 1, padded, hidden]` in the packed order, fractured rows on SP and hidden on TP."""
    x = packing.pack(x)
    return tensor.from_torch(
        x.reshape(1, 1, *x.shape), device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, None, SP_AXIS, TP_AXIS]
    )


def _download_hidden(x: ttnn.Tensor, packing: Packing) -> torch.Tensor:
    logger.info(f"reading back {tuple(x.shape)} per device")
    out = tensor.to_torch(x, mesh_axes=[None, None, SP_AXIS, TP_AXIS]).float()
    logger.info(f"read back {tuple(out.shape)}")
    return packing.unpack(out.reshape(out.shape[-2], out.shape[-1]))


def _attention_kwargs(mesh_device, rotary_dim: int) -> dict:
    ccl_manager, parallel_config = _parallel(mesh_device)
    return dict(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=rotary_dim,
        qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"],
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        use_persistent_ccl_buffers=False,
    )


@MESH
@LAYOUTS
def test_hybrid_attention_mesh(mesh_device: ttnn.MeshDevice, aligned: bool) -> None:
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention

    gen = torch.Generator().manual_seed(6)
    torch.manual_seed(5)
    orig = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN, heads=HEADS, dim_head=HEAD_DIM, qk_norm_eps=REAL_BLOCK_CONFIG["qk_norm_eps"]
    )
    randomize_norm_weights(orig)
    model = ref_h.HybridAttention(orig, HIDDEN, radius=RADIUS, chunk=CHUNK)
    _randomize_hybrid(model, gen)
    model = _round_params(model).eval()
    ref_layout, tt_layout = _layouts(aligned)
    model.layout = ref_layout

    packing = Packing(mesh_device, tt_layout)
    padded_len = packing.padded_len
    rope, (tt_cos, tt_sin), _, _ = _rope(mesh_device, packing)
    x = _bf16(torch.randn(1, SEQ_LEN, HIDDEN, generator=gen))
    with torch.no_grad():
        expected = model(x, rope)[0]

    tt_model = OpenVDNHybridAttention(radius=RADIUS, chunk=CHUNK, **_attention_kwargs(mesh_device, rope[0].shape[-1]))
    tt_model.load_torch_state_dict(model.state_dict())
    tt_model.layout = tt_layout
    out = tt_model(_upload_hidden(x[0], mesh_device, packing), rope_cos=tt_cos, rope_sin=tt_sin)
    out = _download_hidden(out, packing)
    logger.info(f"mesh {tuple(mesh_device.shape)}, N {SEQ_LEN} padded to {padded_len}, aligned {aligned}")

    vs, ve = ref_layout.video_start, ref_layout.video_end
    suffix = "_aligned" if aligned else ""
    details = f"4x8 TP4 x SP8, 56 heads, N{SEQ_LEN} padded {padded_len}, frame {GRID[0]}x{GRID[1]}, aligned {aligned}"
    _check("mesh_hybrid_attention" + suffix, out, expected, pcc=BF16_PCC, rmse=BF16_RMSE, details=details)
    _check("mesh_hybrid_attention_video_rows" + suffix, out[vs:ve], expected[vs:ve], pcc=BF16_PCC, rmse=BF16_RMSE)
    _check(
        "mesh_hybrid_attention_global_rows" + suffix,
        out[:vs],
        expected[:vs],
        pcc=BF16_PCC,
        rmse=GLOBAL_ROWS_RMSE,
        details="LOWERED RMSE threshold, as in 1c: bf16 dense softmax over all keys",
    )


@MESH
@LAYOUTS
def test_transformer_block_mesh(mesh_device: ttnn.MeshDevice, aligned: bool) -> None:
    pytest.importorskip("diffusers", reason="pinned diffusers reference not installed")
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block

    gen = torch.Generator().manual_seed(8)
    torch.manual_seed(7)
    block = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG)
    randomize_norm_weights(block)
    block.attn = ref_h.HybridAttention(block.attn, HIDDEN, radius=RADIUS, chunk=CHUNK)
    _randomize_hybrid(block.attn, gen)
    block = _round_params(block).eval()
    ref_layout, tt_layout = _layouts(aligned)
    block.attn.layout = ref_layout

    packing = Packing(mesh_device, tt_layout)
    padded_len = packing.padded_len
    rope, (tt_cos, tt_sin), token_tags, timestep_indices = _rope(mesh_device, packing)
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    x = _bf16(torch.randn(1, SEQ_LEN, HIDDEN, generator=gen))
    temb = torch.randn(num_timesteps, REAL_BLOCK_CONFIG["time_embed_dim"], generator=gen)
    with torch.no_grad():
        expected = block(x, temb, adaln_indices, rope)[0]

    ccl_manager, parallel_config = _parallel(mesh_device)
    tt_block = OpenVDNTransformerBlock(
        radius=RADIUS,
        chunk=CHUNK,
        **TT_BLOCK_CONFIG,
        rotary_dim=rope[0].shape[-1],
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
        use_persistent_ccl_buffers=False,
    )
    tt_block.load_torch_state_dict(block.state_dict())
    tt_block.attn.layout = tt_layout
    indices = packing.pack(adaln_indices.to(torch.int32))
    out = tt_block(
        spatial_1BND=_upload_hidden(x[0], mesh_device, packing),
        logical_n=tensor.logical_length_tensor(SEQ_LEN, device=mesh_device),
        temb=tensor.from_torch(temb.reshape(1, 1, *temb.shape), device=mesh_device, dtype=ttnn.float32),
        adaln_indices=tensor.from_torch(
            indices.reshape(1, 1, 1, padded_len),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, SP_AXIS],
        ),
        rope_cos=tt_cos,
        rope_sin=tt_sin,
    )
    out = _download_hidden(out, packing)
    _check(
        "mesh_transformer_block" + ("_aligned" if aligned else ""),
        out,
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        details=f"4x8 TP4 x SP8, 56 heads, N{SEQ_LEN} padded {padded_len}, aligned {aligned}",
    )
