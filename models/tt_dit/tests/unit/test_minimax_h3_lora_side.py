# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MiniMax-H3 LoRA side-branch layouts: the augmented weights, fed the inputs the device
assembles (all-gathered `[x_d, u_d]` for column-parallel, per-device `[x_d, p_d]` then a sum for row-parallel),
give `W x + B (A x)`."""

import pytest
import torch
from safetensors.torch import save_file

from models.tt_dit.experimental.lora.h3_adapter_loader import (
    h3_side_rank,
    h3_side_tensors,
    is_side_target,
    iter_h3_host_deltas,
)
from models.tt_dit.models.transformers.minimax_h3.lora_side import (
    augment_col_weight,
    augment_row_weight,
    col_suffix_cols,
    fit_k_block,
    rank_columns,
)

TP = 4


def test_col_suffix_keeps_a_k_block_of_three():
    # H3: 5376 / 4 = 42 tiles per device; q/k/v stack 3 * 128 ranks (3 tiles), ff1 128 (1 tile, padded to 3).
    assert col_suffix_cols(1344, 3 * 128, TP) == 96
    assert col_suffix_cols(1344, 128, TP) == 96
    assert fit_k_block(6, 45) == 5 and fit_k_block(7, 45) == 5 and fit_k_block(3, 45) == 3
    # row-parallel: K block must divide both the base tiles (the concat seam) and the augmented ones
    assert fit_k_block(8, 56, 60) == 4 and fit_k_block(4, 112, 116) == 4


def test_column_parallel_layout():
    g = torch.Generator().manual_seed(0)
    n, k, out, rank, suffix = 5, 16, 12, 6, 2
    x = torch.randn(n, k, generator=g, dtype=torch.float64)
    w = torch.randn(out, k, generator=g, dtype=torch.float64)
    a = torch.randn(rank, k, generator=g, dtype=torch.float64)
    b = torch.randn(out, rank, generator=g, dtype=torch.float64)
    width = TP * suffix
    a_pad = torch.cat([a, torch.zeros(width - rank, k, dtype=a.dtype)])
    u = x @ a_pad.T  # reduce-scatter of the per-device partials: device d keeps columns [d * suffix, (d+1) * suffix)
    k_local = k // TP
    gathered = torch.cat(
        [
            torch.cat([x[:, d * k_local : (d + 1) * k_local], u[:, d * suffix : (d + 1) * suffix]], dim=1)
            for d in range(TP)
        ],
        dim=1,
    )
    w_aug = augment_col_weight(w, rank_columns(b, 0, width), TP)
    torch.testing.assert_close(gathered @ w_aug.T, x @ w.T + (x @ a.T) @ b.T)


def test_column_parallel_qkv_rank_columns():
    g = torch.Generator().manual_seed(1)
    n, k, out, rank = 3, 8, 4, 2
    x = torch.randn(n, k, generator=g, dtype=torch.float64)
    a_list = [torch.randn(rank, k, generator=g, dtype=torch.float64) for _ in range(3)]
    b_list = [torch.randn(out, rank, generator=g, dtype=torch.float64) for _ in range(3)]
    u = x @ torch.cat(a_list).T
    for position, (a, b) in enumerate(zip(a_list, b_list)):
        torch.testing.assert_close(u @ rank_columns(b, position * rank, 3 * rank).T, (x @ a.T) @ b.T)


def test_row_parallel_layout():
    g = torch.Generator().manual_seed(2)
    n, k, out, rank = 5, 16, 12, 3
    x = torch.randn(n, k, generator=g, dtype=torch.float64)
    w = torch.randn(out, k, generator=g, dtype=torch.float64)
    a = torch.randn(rank, k, generator=g, dtype=torch.float64)
    b = torch.randn(out, rank, generator=g, dtype=torch.float64)
    k_local = k // TP
    w_aug = augment_row_weight(w, b, TP)
    shard = k_local + rank
    total = torch.zeros(n, out, dtype=torch.float64)
    for d in range(TP):
        x_d = x[:, d * k_local : (d + 1) * k_local]
        p_d = x_d @ a[:, d * k_local : (d + 1) * k_local].T
        total += torch.cat([x_d, p_d], dim=1) @ w_aug[:, d * shard : (d + 1) * shard].T
    torch.testing.assert_close(total, x @ w.T + (x @ a.T) @ b.T)


@pytest.mark.parametrize(
    ("module", "side"),
    [
        ("transformer_blocks.0.attn.to_q", True),
        ("transformer_blocks.49.attn.to_v", True),
        ("transformer_blocks.3.attn.to_out.0", True),
        ("transformer_blocks.3.ff.net.0.proj", True),
        ("transformer_blocks.3.ff.net.2", True),
        ("transformer_blocks.3.adaln_proj.linear", False),
        ("token_refiner.refiner_blocks.0.attn.to_q", False),
        ("norm_out.linear", False),
    ],
)
def test_side_targets(module, side):
    assert is_side_target(module) is side


def test_side_split_and_scale(tmp_path):
    g = torch.Generator().manual_seed(3)
    rank = 32
    tensors = {}
    for base, (out, inp) in {
        "transformer_blocks.0.attn.to_q": (8, 6),
        "transformer_blocks.0.adaln_proj.linear": (10, 4),
    }.items():
        tensors[f"{base}.lora_A.weight"] = torch.randn(rank, inp, generator=g).to(torch.bfloat16)
        tensors[f"{base}.lora_B.weight"] = torch.randn(out, rank, generator=g).to(torch.bfloat16)
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path), metadata={"alpha": "16"})

    assert h3_side_rank(str(path)) == rank
    side = h3_side_tensors(str(path), scale=2.0)
    assert set(side) == {"transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight"}
    b = tensors["transformer_blocks.0.attn.to_q.lora_B.weight"].float()
    torch.testing.assert_close(side["transformer_blocks.0.attn.to_q.lora_B.weight"], 2.0 * 16 / rank * b)
    host = dict(iter_h3_host_deltas(str(path), ("",), scale=2.0, keep=lambda base: not is_side_target(base)))
    assert set(host) == {"transformer_blocks.0.adaln_proj.linear.weight"}
