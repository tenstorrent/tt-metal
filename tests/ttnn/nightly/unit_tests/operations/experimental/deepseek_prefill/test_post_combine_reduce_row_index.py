# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""post_combine_reduce's optional per-token weights and row-indexed combine, against the gathered path.

Gathered path (the default): combine_output is [1, 1, T, k, emb] (each token's k expert rows gathered side by side)
and weights is [1, 1, T, k, 1] (one weight per page).
Per-token weights: weights is [1, 1, T, k] (one token's k weights per page, read 32 tokens at a time).
Row-indexed combine: combine_output is the expert output itself, a [rows, emb] ROW_MAJOR table, and slot j of token t
reads row combine_row_index[t, j]; no gathered copy. Slots of non-local experts point at skip_row, which the kernel
never reads. Two ways drop those slots: the dispatch table's -1 (DeepSeek: indices + expert_dispatch_table given), or
a zero weight (gpt-oss: neither given).

Both new modes must be bit-identical to the gathered path, and all must match a torch weighted sum.
"""

import pytest
import torch

import ttnn

T = 640  # tokens
E = 32  # routed experts (gpt-oss-20b)
LOCAL = 8  # experts on this chip


def _rm(t, device, dtype=None):
    return ttnn.from_torch(
        t, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("skip", ["dispatch_table", "zero_weight"])
@pytest.mark.parametrize("K", [4, 8], ids=["top4", "top8"])
@pytest.mark.parametrize("emb", [2880, 7168], ids=["gptoss", "dsv3"])
def test_row_indexed_combine_matches_gathered(device, emb, K, skip):
    torch.manual_seed(3)
    indices = torch.stack([torch.randperm(E)[:K] for _ in range(T)])  # [T, K] global expert ids
    local = indices < LOCAL  # experts 0..7 on this chip, the rest elsewhere
    weights = torch.softmax(torch.randn(T, K), dim=-1)
    if skip == "zero_weight":
        weights = torch.where(local, weights, 0.0)
    weights = weights.to(torch.bfloat16)

    # The expert output table: one row per local (token, slot) pair in a shuffled order, then a spare row that every
    # non-local slot points at. The spare row holds huge values: reading it would show in the result.
    pairs = int(local.sum())
    spare = pairs
    rows = -(-(pairs + 1) // 32) * 32
    table = torch.randn(rows, emb).to(torch.bfloat16)
    table[spare] = 1e30
    row_index = torch.full((T, K), spare, dtype=torch.int64)
    row_index[local] = torch.randperm(pairs)

    expected = torch.zeros(T, emb)
    for t in range(T):
        for j in range(K):
            if local[t, j]:
                expected[t] += weights[t, j].float() * table[row_index[t, j]].float()

    common = dict(output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if skip == "dispatch_table":
        dispatch_table = torch.where(torch.arange(E) < LOCAL, 0, -1).to(torch.int32)
        common.update(
            indices=_rm(indices.reshape(1, 1, T, K).to(torch.uint16), device, ttnn.uint16),
            expert_dispatch_table=_rm(dispatch_table, device),
        )

    # Gathered path: [1, 1, T, K, emb] copy, one weight per page.
    gathered = table[row_index].reshape(1, 1, T, K, emb)
    out_gathered = ttnn.experimental.deepseek_prefill.post_combine_reduce(
        _rm(gathered, device), _rm(weights.reshape(1, 1, T, K, 1), device), expert_dim=3, **common
    )
    out_gathered = ttnn.to_torch(out_gathered).reshape(T, emb)

    # Per-token weights, gathered combine.
    out_per_token = ttnn.experimental.deepseek_prefill.post_combine_reduce(
        _rm(gathered, device), _rm(weights.reshape(1, 1, T, K), device), expert_dim=3, **common
    )
    out_per_token = ttnn.to_torch(out_per_token).reshape(T, emb)

    # Row-indexed combine straight from the table, per-token weights.
    out_indexed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
        _rm(table, device),
        _rm(weights.reshape(1, 1, T, K), device),
        expert_dim=3,
        combine_row_index=_rm(row_index.to(torch.int32).reshape(1, 1, T, K), device, ttnn.uint32),
        skip_row=spare,
        **common,
    )
    out_indexed = ttnn.to_torch(out_indexed).reshape(T, emb)

    assert torch.isfinite(out_gathered).all()
    pcc = torch.corrcoef(torch.stack([out_gathered.flatten().float(), expected.flatten()]))[0, 1].item()
    assert pcc > 0.999, f"gathered path vs torch: PCC {pcc}"
    assert torch.equal(out_per_token, out_gathered), "per-token weights differ from the per-page weights"
    assert torch.equal(out_indexed, out_gathered), "row-indexed combine differs from the gathered combine"


def test_row_indexed_combine_rejects_bad_inputs(device, expect_error):
    emb = 2880
    table = _rm(torch.zeros(64, emb).to(torch.bfloat16), device)
    weights = _rm(torch.zeros(1, 1, 32, 4).to(torch.bfloat16), device)
    row_index = torch.zeros(1, 1, 32, 4, dtype=torch.int32)
    pcr = ttnn.experimental.deepseek_prefill.post_combine_reduce

    with expect_error(RuntimeError, "skip_row needs combine_row_index"):
        pcr(_rm(torch.zeros(1, 1, 32, 4, emb).to(torch.bfloat16), device), weights, expert_dim=3, skip_row=0)
    with expect_error(RuntimeError, "combine_row_index must be uint32"):
        pcr(table, weights, expert_dim=3, combine_row_index=_rm(row_index, device, ttnn.int32))
    with expect_error(RuntimeError, "must be a \\[rows, emb_dim\\] table"):
        pcr(
            _rm(torch.zeros(1, 64, emb).to(torch.bfloat16), device),
            weights,
            expert_dim=3,
            combine_row_index=_rm(row_index, device, ttnn.uint32),
        )
