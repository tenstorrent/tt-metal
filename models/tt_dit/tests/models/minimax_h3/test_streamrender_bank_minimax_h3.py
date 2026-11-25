# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One-way prefix attention through the K/V bank (`KVBankSlot`), against the torch reference.

A prefix forward captures K/V; a continuation forward attends to `[bank ; itself]`. Within one attention layer
K and V are row-local projections, so the continuation rows of a one-way-masked window equal the same rows of
an unmasked full forward, and the prefix rows equal a forward over the prefix alone. Both lengths are chosen off
the `sp * TILE` grid so the pad keys of both blocks are live in the mask.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention as TorchMiniMaxH3Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import KVBankSlot, MiniMaxH3Attention, prepare_rope_tables
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.streamrender.dit_adapter import prefix_bank_mask, sparse_block_tables
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, ROPE_FREQ_DIM, ROPE_THETA, randomize_norm_weights, upload_rope

HIDDEN_SIZE = REAL_BLOCK_CONFIG["hidden_size"]
NUM_HEADS = REAL_BLOCK_CONFIG["num_attention_heads"]
HEAD_DIM = REAL_BLOCK_CONFIG["attention_head_dim"]
QK_NORM_EPS = REAL_BLOCK_CONFIG["qk_norm_eps"]


def _padded(rows: int, sp_factor: int) -> int:
    alignment = sp_factor * ttnn.TILE_SIZE
    return -(-rows // alignment) * alignment


@GALAXY_RING
@pytest.mark.parametrize(
    ("prefix_rows", "dynamic_rows", "sparse"),
    [
        pytest.param(1000, 1700, False, id="p1000_d1700-masked"),
        # The sparse path needs whole 64-row blocks per own-row shard: 2048 / 8 = 256.
        pytest.param(1000, 2000, True, id="p1000_d2000-sparse"),
    ],
)
def test_minimax_h3_attention_prefix_bank(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    prefix_rows: int,
    dynamic_rows: int,
    sparse: bool,
    reset_seeds,
) -> None:
    MIN_PCC = 0.995

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    rows = prefix_rows + dynamic_rows

    torch_model = TorchMiniMaxH3Attention(
        hidden_size=HIDDEN_SIZE, heads=NUM_HEADS, dim_head=HEAD_DIM, qk_norm_eps=QK_NORM_EPS
    ).to(torch.float32)
    randomize_norm_weights(torch_model)
    torch_model.eval()

    position_ids = torch.stack([torch.randint(0, 64, (rows,)) for _ in range(3)], dim=-1)
    rope_cos, rope_sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)(position_ids)
    hidden = torch.randn((1, rows, HIDDEN_SIZE), dtype=torch.float32)
    with torch.no_grad():
        full = torch_model(hidden_states=hidden, rotary_emb=(rope_cos, rope_sin), attention_mask=None)
        prefix_only = torch_model(
            hidden_states=hidden[:, :prefix_rows],
            rotary_emb=(rope_cos[:prefix_rows], rope_sin[:prefix_rows]),
            attention_mask=None,
        )

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
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
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3

    def run(start: int, stop: int) -> torch.Tensor:
        count = stop - start
        pad_to = _padded(count, sp_factor)
        block = torch.zeros(1, 1, pad_to, HIDDEN_SIZE)
        block[0, 0, :count] = hidden[0, start:stop]
        cos, sin = (torch.cat([t[start:stop], torch.zeros(pad_to - count, t.shape[-1])]) for t in (rope_cos, rope_sin))
        tt_cos, tt_sin = upload_rope(*prepare_rope_tables(cos, sin, HEAD_DIM), mesh_device=mesh_device, sp_axis=sp_axis)
        out = tt_model(
            bf16_tensor_2dshard(block, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}),
            logical_n=from_torch(
                torch.tensor([count], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.Layout.ROW_MAJOR,
                mesh_axes=[..., None, None],
            ),
            rope_cos=tt_cos,
            rope_sin=tt_sin,
        )
        out = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        return out[0, :, :count, :]

    slot = KVBankSlot(capture=True)
    tt_model.kv_bank = slot
    tt_prefix = run(0, prefix_rows)
    bank_padded = _padded(prefix_rows, sp_factor)
    assert slot.key is not None and slot.key.shape[2] == bank_padded // sp_factor

    slot.capture = False
    dynamic_padded = _padded(dynamic_rows, sp_factor)
    replicated = dict(device=mesh_device, mesh_axes=[..., None, None])
    if sparse:
        block_mask, block_counts = sparse_block_tables(
            prefix_rows, bank_padded, dynamic_rows, dynamic_padded, sp_factor, tt_model.n_local_heads
        )
        slot.block_mask, slot.block_counts = (
            from_torch(t, dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, **replicated)
            for t in (block_mask, block_counts)
        )
        slot.query_pad = from_torch(
            torch.zeros(1, tt_model.n_local_heads, bank_padded // sp_factor, HEAD_DIM),
            dtype=ttnn.bfloat16,
            **replicated,
        )
    else:
        slot.mask = from_torch(
            prefix_bank_mask(prefix_rows, bank_padded, dynamic_rows, dynamic_padded, sp_factor),
            dtype=ttnn.bfloat16,
            **replicated,
        )
    tt_dynamic = run(prefix_rows, rows)
    tt_model.kv_bank = None

    assert_quality(prefix_only, tt_prefix, pcc=MIN_PCC)
    assert_quality(full[:, prefix_rows:], tt_dynamic, pcc=MIN_PCC)
