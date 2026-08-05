# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The precomputed-AdaLN block must match the same torch reference the projected path matches.

`test_adaln_precompute_minimax_h3.py` gates the host table against a *recompute* of itself. That is
necessary but not sufficient: a table can be internally consistent and still be wired into the block
wrongly -- wrong row order, wrong parameter slice, a missing `1 +` on a scale, or a TP shard that
takes the wrong hidden columns. Those bugs are invisible to a self-consistency check and invisible to
the shapes.

So this compares against **torch**, and against the projected path's own PCC, using one block and one
step. The two TT paths are also compared to each other directly, which is the tightest of the three:
they consume the same weights by construction, so anything but near-equality is a wiring error rather
than precision.

The threshold reasoning from `test_transformer_block_minimax_h3.py` carries over verbatim -- the
residual stream dominates the block output, so every plausible gather bug still scores >= 0.9959 and
only a tight bar catches it.
"""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.adaln_cache_minimax_h3 import MiniMaxH3AdalnCache
from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import (
    MODALITY_NUM,
    NUM_MODULATION_PARAMS,
    MiniMaxH3TransformerBlock,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3 import adaln_precompute as ap
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import randomize_norm_weights
from .test_transformer_block_minimax_h3 import (
    FFN_DIM,
    HEAD_DIM,
    HIDDEN_SIZE,
    NORM_EPS,
    NUM_HEADS,
    QK_NORM_EPS,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TIME_EMBED_DIM,
    _packed_layout,
)

MIN_PCC = 0.9995
# The two TT paths read the same weights, so they should agree far more tightly than either agrees
# with torch. Anything looser than this is a wiring bug, not arithmetic.
MIN_PCC_PATHS_AGREE = 0.9999


class _SingleLayerTable:
    """One layer of the host table's surface, built directly from a block's own AdaLN weights.

    `MiniMaxH3AdalnCache` consumes the table structurally rather than importing its type (see the
    layering note there), which is what lets this stand in for `precompute_adaln_table` without the
    26 GB checkpoint. The projection itself is the real `adaln_precompute` code, so the numerical
    conventions under test -- fp32 SiLU before the bf16 cast, per-step `temb` -- are the shipping ones.
    """

    def __init__(self, temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, hidden_size: int) -> None:
        projected = ap.project_block_adaln(temb, weight, bias, hidden_size)
        self.block_params = projected.unsqueeze(0)  # [1 layer, rows * MODALITY_NUM, params, hidden]
        rows = temb.shape[0]
        self.final_shift = torch.zeros(rows, hidden_size, dtype=torch.bfloat16)
        self.final_scale = torch.zeros(rows, hidden_size, dtype=torch.bfloat16)
        self.step_offsets = torch.tensor([0, rows])
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.num_steps = 1


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "is_fsdp", "topology"),
    [
        pytest.param((4, 8), 1, 0, 2, False, ttnn.Topology.Ring, id="4x8sp1tp0nl2_ring_is_fsdp0"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [ring_params_req_exact_devices], indirect=True)
@pytest.mark.parametrize(("num_text", "num_audio", "num_video"), [pytest.param(512, 256, 1280, id="small_s2048")])
def test_precomputed_adaln_matches_projected_path(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    num_text: int,
    num_audio: int,
    num_video: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    position_ids, token_tags, timestep_indices = _packed_layout(num_text, num_audio, num_video)
    seq_len = position_ids.shape[0]
    num_timesteps = int(timestep_indices.max().item()) + 1
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

    torch_model = TorchMiniMaxH3Block(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(position_ids)
    rotary_dim = rope_cos.shape[-1]
    tt_rope_cos_t, tt_rope_sin_t = prepare_rope_tables(rope_cos, rope_sin, HEAD_DIM)

    spatial_input = torch.randn((1, seq_len, HIDDEN_SIZE), dtype=torch.float32)
    temb_input = torch.randn((num_timesteps, TIME_EMBED_DIM), dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(
            hidden_states=spatial_input,
            temb=temb_input,
            adaln_indices=adaln_indices,
            rotary_emb=(rope_cos, rope_sin),
            attention_mask=None,
        )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    common = dict(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=rotary_dim,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )

    tt_spatial = bf16_tensor_2dshard(
        spatial_input.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_temb = from_torch(
        temb_input.reshape(1, 1, num_timesteps, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32
    )
    tt_adaln_indices = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, seq_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos = from_torch(
        tt_rope_cos_t.reshape(1, 1, seq_len, HEAD_DIM),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )
    tt_rope_sin = from_torch(
        tt_rope_sin_t.reshape(1, 1, seq_len, HEAD_DIM),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3

    def run(block, **extra):
        out = block(
            tt_spatial,
            N=seq_len,
            temb=extra.pop("temb", tt_temb),
            adaln_indices=tt_adaln_indices,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            **extra,
        )
        out = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        return out[:, :, :seq_len, :]

    # 1. The shipping projected path, as the reference gate runs it.
    projected_block = MiniMaxH3TransformerBlock(**common)
    projected_block.load_torch_state_dict(torch_model.state_dict())
    projected_out = run(projected_block)

    # 2. The precomputed path: same weights, projected on host into a table instead.
    state = torch_model.state_dict()
    table = _SingleLayerTable(
        temb_input,
        state["adaln_proj.linear.weight"].bfloat16(),
        state["adaln_proj.linear.bias"].bfloat16(),
        HIDDEN_SIZE,
    )
    cache = MiniMaxH3AdalnCache(
        table,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
    )
    precomputed_block = MiniMaxH3TransformerBlock(**common, precomputed_adaln=True)
    precomputed_block.load_torch_state_dict(torch_model.state_dict())
    tables = cache.block_tables(0)
    assert len(tables) == NUM_MODULATION_PARAMS
    assert tuple(tables[0].shape)[-2:] == (num_timesteps * MODALITY_NUM, HIDDEN_SIZE // tp_factor)
    precomputed_out = run(precomputed_block, temb=None, modulation_tables=tables)

    logger.info("projected path vs torch")
    assert_quality(torch_out, projected_out, pcc=MIN_PCC)
    logger.info("precomputed path vs torch")
    assert_quality(torch_out, precomputed_out, pcc=MIN_PCC)
    logger.info("precomputed vs projected -- same weights, so this is the wiring check")
    assert_quality(projected_out, precomputed_out, pcc=MIN_PCC_PATHS_AGREE)
