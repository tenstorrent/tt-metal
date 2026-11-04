# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that deepseek_prefill ops reject arguments their host-side sizing divides by."""

import torch

import ttnn


def _row_major_int(device, shape):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def test_combine_rejects_zero_experts_per_chip(device, expect_error):
    dispatched = ttnn.from_torch(torch.zeros((1, 1, 32, 64)).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    metadata = _row_major_int(device, (1, 1, 32, 8))
    counts = _row_major_int(device, (1, 1, 8))
    offsets = _row_major_int(device, (1, 1, 8))
    with expect_error(RuntimeError, "must be divisible by experts_per_chip=0"):
        ttnn.experimental.deepseek_prefill.combine(
            dispatched,
            metadata,
            counts,
            offsets,
            dispatch_group_size=1,
            experts_per_chip=0,
            num_experts_per_tok=1,
            seq_len_per_chip=32,
        )


def test_forwarding_buffer_rejects_zero_combine_links(device, expect_error):
    with expect_error(RuntimeError, r"num_links must be between 1 and 4 \(got 0\)"):
        ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe_forwarding_buffer(
            device,
            emb_dim=64,
            experts_per_chip=1,
            num_experts_per_tok=1,
            seq_len_per_chip=32,
            combine_axis=0,
            combine_num_links=0,
        )
