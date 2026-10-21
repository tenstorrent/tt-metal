# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""deepseek_v4_prefill_indexer_select: fused lightning-indexer score + top-512 for a prefill chunk (Blackhole only)."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from ttnn.operations.transformer_golden import MASKED_INDEX, deepseek_v4_prefill_indexer_scores
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import to_dev

pytestmark = pytest.mark.use_module_device

TOPK = 512
H, D = 32, 128  # DeepSeek V4.1 index_n_heads, index_head_dim


def make_inputs(T, E, seed=0):
    """Post-RoPE indexer q, scaled gates (index_head_dim^-0.5 * n_heads^-0.5 folded in) and indexer keys."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, T, D, generator=gen).to(torch.bfloat16)
    weights = (torch.randn(1, 1, T, H, generator=gen) * (D**-0.5 * H**-0.5)).to(torch.bfloat16)
    keys = torch.randn(1, 1, E, D, generator=gen).to(torch.bfloat16)
    return q, weights, keys


def start_tensor(start, device):
    return to_dev(torch.tensor([[[[start]]]], dtype=torch.int32), device, ttnn.uint32)


def to_tile(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def run(device, q, weights, keys, start, rate):
    out = ttnn.transformer.deepseek_v4_prefill_indexer_select(
        to_tile(q, device),
        to_tile(weights, device),
        to_dev(keys, device, ttnn.bfloat16),
        start_tensor(start, device),
        compress_ratio=rate,
    )
    return ttnn.to_torch(out).to(torch.int64).reshape(q.shape[2], TOPK) & 0xFFFFFFFF


def check(indices, q, weights, keys, start, rate):
    """Per query: exactly min(512, visible) distinct visible entries, the rest sentinels, and a selection whose
    reference scores match the reference top-512's up to bf16 near-ties at the cut."""
    scores = deepseek_v4_prefill_indexer_scores(q, weights, keys, start, rate)
    T, E = scores.shape
    limits = ((start + torch.arange(T) + 1) // rate).clamp(max=E)
    worst_recall = 1.0
    for s in range(T):
        row = indices[s]
        valid = row[row != MASKED_INDEX]
        n = min(TOPK, int(limits[s]))
        assert valid.numel() == n, f"query {s}: {valid.numel()} valid slots, expected {n}"
        if n == 0:
            continue
        assert valid.unique().numel() == n, f"query {s}: duplicate indices"
        assert int(valid.max()) < int(limits[s]), f"query {s}: selected an entry it cannot see"
        ref_values, ref_idx = scores[s, : limits[s]].topk(n)
        got = scores[s, valid]
        # Device scores are bf16 products at HiFi2 (like indexer_score_dsa's default): anything selected must be
        # within that rounding (relative to the row's score magnitude) of the reference cut.
        tol = 4e-2 * scores[s, : limits[s]].abs().max() + 1e-6
        assert got.min() >= ref_values[-1] - tol, f"query {s}: selected {got.min()} below the cut {ref_values[-1]}"
        assert got.sum() >= ref_values.sum() - n * tol / 8, f"query {s}: selection score sum too low"
        recall = len(set(valid.tolist()) & set(ref_idx.tolist())) / n
        worst_recall = min(worst_recall, recall)
    assert worst_recall >= 0.95, f"worst per-query recall {worst_recall}"
    return worst_recall


def reference_recall(indices, q, weights, keys, start, rate):
    """Worst per-query recall of the reference top-512, for the baseline (which returns -inf entries, not
    sentinels, past the visible ones)."""
    scores = deepseek_v4_prefill_indexer_scores(q, weights, keys, start, rate)
    limits = ((start + torch.arange(scores.shape[0]) + 1) // rate).clamp(max=scores.shape[1])
    worst = 1.0
    for s in range(scores.shape[0]):
        n = min(TOPK, int(limits[s]))
        if n:
            ref = set(scores[s, : limits[s]].topk(n).indices.tolist())
            worst = min(worst, len(ref & set(indices[s].tolist())) / n)
    return worst


@run_for_blackhole()
@pytest.mark.parametrize(
    "T, E, start, rate",
    [
        (32, 512, 480, 1),  # one block, every query sees at least one full block's worth
        (32, 512, 0, 1),  # first chunk: query s sees s + 1 entries
        (32, 600, 0, 4),  # queries 0..2 see nothing: all sentinels
        (64, 1536, 1400, 1),  # three blocks, partial last block
        (128, 4096, 6000, 2),  # rate 2, many blocks
        (256, 8192, 8000, 1),  # few query tiles: blocks split across cores and merged
        (32, 20000, 19900, 1),  # one query tile, maximum split
        (4096, 2048, 0, 2),  # one query tile per core
        (8192, 1024, 0, 8),  # more query tiles than cores
    ],
)
def test_deepseek_v4_prefill_indexer_select(device, T, E, start, rate):
    q, weights, keys = make_inputs(T, E)
    indices = run(device, q, weights, keys, start, rate)
    check(indices, q, weights, keys, start, rate)


@run_for_blackhole()
def test_deepseek_v4_prefill_indexer_select_runtime_start_and_length(device):
    """start and the entry count are runtime values: new positions and a longer buffer reuse the program."""
    T, rate = 64, 2
    q, weights, keys = make_inputs(T, 4096, seed=1)
    device.clear_program_cache()
    for start, E in [(100, 1024), (3000, 2048), (7000, 4096)]:
        indices = run(device, q, weights, keys[:, :, :E], start, rate)
        check(indices, q, weights, keys[:, :, :E], start, rate)
    assert device.num_program_cache_entries() == 1


def baseline_select(q_tt, w_tt, pad_rm, cut, cap):
    """The capacity-sized path the fused op replaces (the model's index-source body): tile the [cap + T]-row key
    pad, score every one of its `cap` entries for every query, add the causal cut, then top-k over `cap`."""
    num_tokens = q_tt.shape[2]
    keys = ttnn.to_layout(pad_rm, ttnn.TILE_LAYOUT)
    logits = ttnn.experimental.indexer_score_dsa(
        q_tt,
        keys,
        w_tt,
        chunk_start_idx=cap,
        kv_len=pad_rm.shape[2],
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=64, head_group_size=0),
    )
    scores = ttnn.to_layout(ttnn.slice(logits, [0, 0, 0, 0], [1, 1, num_tokens, cap]), ttnn.TILE_LAYOUT)
    flat = ttnn.to_layout(ttnn.add(scores, cut), ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.experimental.topk_large_indices(flat, k=TOPK)


@run_for_blackhole()
@pytest.mark.parametrize(
    "T, start, rate, cap",
    [
        (1024, 7168, 1, 8192),  # entries visible == capacity
        (1024, 7168, 1, 65536),  # a 64K-entry tier serving an 8K-entry prefix
        (2048, 30720, 2, 16384),  # rate 2, visible == capacity
        (2048, 30720, 2, 131072),  # rate 2, 128K-entry tier
    ],
)
def test_deepseek_v4_prefill_indexer_select_perf(device, T, start, rate, cap):
    """Runs the fused op and the capacity-sized baseline (indexer_score_dsa + cut + topk_large_indices) on the same
    chunk; run under the device profiler to compare device kernel time."""
    E = (start + T) // rate
    q, weights, keys = make_inputs(T, E, seed=2)
    q_tt, w_tt = to_tile(q, device), to_tile(weights, device)
    # The fused op reads the capacity-sized entry buffer itself; the baseline reads the model's [cap + T]-row pad.
    keys_full = torch.zeros(1, 1, cap + T, D, dtype=torch.bfloat16)
    keys_full[:, :, :E] = keys
    keys_rm = to_dev(keys_full[:, :, :cap], device, ttnn.bfloat16)
    pad_rm = to_dev(keys_full, device, ttnn.bfloat16)
    limits = (start + torch.arange(T) + 1) // rate
    cut = torch.where(torch.arange(cap).view(1, -1) < limits.view(-1, 1), 0.0, float("-inf"))
    cut_tt = to_tile(cut.view(1, 1, T, cap), device)
    start_tt = start_tensor(start, device)

    for _ in range(2):  # the second run is the warm one
        fused = ttnn.transformer.deepseek_v4_prefill_indexer_select(q_tt, w_tt, keys_rm, start_tt, compress_ratio=rate)
        base = baseline_select(q_tt, w_tt, pad_rm, cut_tt, cap)
    ttnn.synchronize_device(device)

    fused = ttnn.to_torch(fused).to(torch.int64).reshape(T, TOPK) & 0xFFFFFFFF
    base = ttnn.to_torch(base).to(torch.int64).reshape(T, TOPK) & 0xFFFFFFFF
    fused_recall = check(fused, q, weights, keys, start, rate)
    base_recall = reference_recall(base, q, weights, keys, start, rate)
    assert fused_recall >= base_recall - 0.01, f"fused worst recall {fused_recall}, baseline {base_recall}"
