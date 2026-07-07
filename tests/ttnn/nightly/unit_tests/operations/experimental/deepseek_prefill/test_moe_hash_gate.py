# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ttnn.experimental.deepseek_prefill.moe_hash_gate().

The hash gate routes each token to a fixed set of experts via a static tid2eid[input_ids] lookup
(fused into the op's reader kernel) instead of top-k. The per-expert weights are still
score_func(logits) gathered at those indices, normalized, and scaled. We validate the expert
indices exactly (recall == 1.0, since selection is deterministic) and the weights via PCC.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_gate_output,
    build_padding_config,
    distinct_logits,
    hash_gate_golden_act,
)


# (seq_len,) with num_batches == batch_size == 1 (single prefill sequence).
SEQ_LENS = [32, 128, 100, 3200]
SEQ_LEN_IDS = ["one_tile", "four_tiles", "remainder", "realistic"]

# (total_experts, n_activated_experts, route_scale, vocab_size)
ROUTING_CONFIGS = [
    (256, 6, 1.5, 2048),  # DeepSeek-V4 Flash
    (384, 6, 2.5, 2048),  # DeepSeek-V4 Pro (384 experts: needs exact uint16 tid2eid, not bf16)
    (256, 8, 1.0, 2048),  # generic
]
ROUTING_CONFIG_IDS = ["dsv4flash-256e", "dsv4pro-384e", "generic-256e8a"]

SCORE_FUNCS = ["sqrtsoftplus", "sigmoid"]

TILE_H = 32


@pytest.mark.parametrize("score_func", SCORE_FUNCS)
@pytest.mark.parametrize(
    "total_experts,n_activated_experts,route_scale,vocab_size",
    ROUTING_CONFIGS,
    ids=ROUTING_CONFIG_IDS,
)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=SEQ_LEN_IDS)
@pytest.mark.parametrize("padded_percent", [0, 50], ids=lambda p: f"pad{p}")
def test_moe_hash_gate(
    device,
    seq_len,
    total_experts,
    n_activated_experts,
    route_scale,
    vocab_size,
    padded_percent,
    score_func,
):
    torch.manual_seed(42)
    epsilon = 1e-20

    logits = distinct_logits((1, 1, seq_len, total_experts))
    input_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int64)
    # Random frozen token-id -> expert-id table; distinct experts per row.
    tid2eid = torch.stack([torch.randperm(total_experts)[:n_activated_experts] for _ in range(vocab_size)]).to(
        torch.int64
    )

    ref_indices, ref_weights = hash_gate_golden_act(
        logits, input_ids, tid2eid, route_scale, epsilon, n_activated_experts, score_func
    )

    # --- Device inputs ---
    ttnn_logits = ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # input_ids: one ROW_MAJOR page per score height tile (32 uint32 ids), zero-padded to tile boundary.
    height_tiles = (seq_len + TILE_H - 1) // TILE_H
    ids_padded = torch.zeros(height_tiles * TILE_H, dtype=torch.int64)
    ids_padded[:seq_len] = input_ids
    ttnn_input_ids = ttnn.from_torch(
        ids_padded.reshape(height_tiles, TILE_H).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # tid2eid: ROW_MAJOR uint16, rows padded to 16 cols (32-byte NoC pages).
    # Build from a 16-bit host tensor: ttnn.from_torch does not narrow int32 -> uint16, so an int32
    # source would leave each expert id as 4 bytes (id, 0) and the uint16 reader would read every
    # other slot as 0. int16 matches the on-device uint16 width exactly (expert ids <= 383 fit).
    tid2eid_padded = torch.zeros(vocab_size, 16, dtype=torch.int16)
    tid2eid_padded[:, :n_activated_experts] = tid2eid.to(torch.int16)
    ttnn_tid2eid = ttnn.from_torch(
        tid2eid_padded,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Right-padding: real tokens occupy leading rows.
    num_real = seq_len - int(seq_len * padded_percent / 100)
    apply_padding = 0 < num_real < seq_len
    padding_config = build_padding_config(device, num_real) if apply_padding else None

    ttnn_weights_out, ttnn_indices_out = ttnn.experimental.deepseek_prefill.moe_hash_gate(
        ttnn_logits,
        ttnn_input_ids,
        ttnn_tid2eid,
        n_activated_experts=n_activated_experts,
        route_scale=route_scale,
        epsilon=epsilon,
        score_func=score_func,
        padding_config=padding_config,
    )

    tt_weights = ttnn.to_torch(ttnn_weights_out)[:1, :1, :seq_len, :n_activated_experts]
    tt_indices = ttnn.to_torch(ttnn_indices_out)[:1, :1, :seq_len, :n_activated_experts]

    # Hash routing is deterministic, so the selected experts must match exactly (recall == 1).
    assert_gate_output(
        tt_indices,
        tt_weights,
        ref_indices,
        ref_weights,
        n_activated_experts,
        total_experts,
        num_real,
        apply_padding,
        exact_recall=True,
        pcc_threshold=0.97,
    )
