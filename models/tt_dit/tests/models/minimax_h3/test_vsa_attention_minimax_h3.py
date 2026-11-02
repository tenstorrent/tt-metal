# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMaxH3Attention on the VSA path vs the torch VSA oracle: pre-pass, coarse branch, gate and fine attention.

The torch reference is the diffusers attention with dense SDPA swapped for the oracle in device row order inside a
4x8 ladder rung, plus the rule the ring op adds: a query block that keeps nothing in a shard also computes that
shard's block 0 (one exact block per ring iteration).
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed, _apply_rotary_emb
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ....models.transformers.minimax_h3.vsa_prepass import MiniMaxH3VSAConfig, VSAPrepass
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_BUCKET_LADDER_4X8
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry, vsa_min_padded_len
from ....utils.tensor import bf16_tensor_2dshard
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, ROPE_FREQ_DIM, ROPE_THETA, randomize_norm_weights, upload_rope
from .test_sol_bench_minimax_h3 import _position_ids
from .test_transformer_minimax_h3 import logical_length_tensor
from .test_vsa_ring_sdpa_minimax_h3 import kept_to_indices, with_shard_fallback
from .vsa_oracle import VSA_INDEX_SENTINEL, coarse_output, coarse_scores, fine_attention_batched, select_index_rows

MIN_PCC = 0.995
SPARSITY = 0.9
NUM_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _torch_reference(torch_model, x, rope_cos, rope_sin, geometry, gate_weight) -> torch.Tensor:
    with torch.no_grad():
        heads = lambda t: t.unflatten(-1, (NUM_HEADS, HEAD_DIM))
        q = _apply_rotary_emb(torch_model.norm_q(heads(torch_model.to_q(x))), rope_cos, rope_sin)
        k = _apply_rotary_emb(torch_model.norm_k(heads(torch_model.to_k(x))), rope_cos, rope_sin)
        v = heads(torch_model.to_v(x))
        gate = heads(x @ gate_weight.T)
        # Packed [1, S, H, D] -> [1, H, rung, D] in device row order, zeros at pad rows.
        to_device_order = lambda t: geometry.pack_rows(t[0]).permute(1, 0, 2).unsqueeze(0)
        qt, kt, vt, gt = (to_device_order(t) for t in (q, k, v, gate))

        scores = coarse_scores(qt, kt, geometry)
        indices, _ = select_index_rows(scores, geometry, SPARSITY, with_sets=False)
        listed = indices[0].long()
        n_blocks = geometry.n_blocks
        kept = torch.zeros(NUM_HEADS, n_blocks, n_blocks + 1, dtype=torch.bool)
        kept.scatter_(-1, torch.where(listed == VSA_INDEX_SENTINEL, n_blocks, listed), True)
        indices = kept_to_indices(with_shard_fallback(kept[..., :n_blocks], geometry))
        fine = fine_attention_batched(qt, kt, vt, indices, geometry)
        coarse = coarse_output(scores, vt, geometry, fine.dtype)
        out = geometry.unpack_rows((fine + gt * coarse)[0].permute(1, 0, 2))
        return torch_model.to_out[0](out.reshape(1, -1, NUM_HEADS * HEAD_DIM))[0]


@GALAXY_RING
@pytest.mark.parametrize(
    ("prefix_segments", "grid", "rung"),
    [
        pytest.param((512, 0, 0), (8, 16, 16), 18944, id="quick-aligned"),
        pytest.param((70, 5, 130), (9, 10, 13), 20224, id="quick-ragged-straggler"),
    ],
)
def test_vsa_attention(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, prefix_segments, grid, rung, reset_seeds
) -> None:
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    assert rung in MINIMAX_H3_BUCKET_LADDER_4X8 and rung >= vsa_min_padded_len(prefix_segments, grid, sp)
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)

    torch.manual_seed(0)
    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_HEADS, dim_head=HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()
    gate_weight = 0.02 * torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)
    state = dict(torch_model.state_dict())
    state["to_gate_compress.weight"] = gate_weight

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    rope_cos, rope_sin = rope(_position_ids(geometry.seq_len))
    x = torch.randn(1, geometry.seq_len, HIDDEN_SIZE)
    reference = _torch_reference(torch_model, x, rope_cos, rope_sin, geometry, gate_weight)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp),
        cfg_parallel=None,
    )
    tt_model = MiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=rope_cos.shape[-1],
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        vsa_config=MiniMaxH3VSAConfig(sparsity=SPARSITY),
    )
    tt_model.load_torch_state_dict(state)
    prepass = VSAPrepass(
        rung=rung,
        sp_factor=sp,
        sparsity=SPARSITY,
        num_heads=NUM_HEADS // tp,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
    )
    prepass.update(geometry, traced=False)
    tt_model.set_vsa_prepass(prepass)

    # Device row order: pad rows repeat a real row of their tile, for the input and the rotary tables alike.
    cos_t, sin_t = prepare_rope_tables(
        geometry.permute_metadata(rope_cos), geometry.permute_metadata(rope_sin), HEAD_DIM
    )
    tt_cos, tt_sin = upload_rope(cos_t, sin_t, mesh_device=mesh_device, sp_axis=sp_axis)
    tt_x = bf16_tensor_2dshard(
        geometry.permute_metadata(x[0]).reshape(1, 1, geometry.padded_len, HIDDEN_SIZE),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_out = tt_model(
        tt_x, logical_n=logical_length_tensor(mesh_device, geometry.logical_n), rope_cos=tt_cos, rope_sin=tt_sin
    )

    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 3
    out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )
    out = geometry.unpack_rows(out.reshape(geometry.padded_len, HIDDEN_SIZE).float())
    pcc = _pcc(out, reference)
    logger.info(f"{geometry.n_real_tiles} tiles at rung {rung}: PCC vs torch VSA oracle {pcc:.6f}")
    assert pcc >= MIN_PCC, pcc
