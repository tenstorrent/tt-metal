# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.moe.fused_experts (full preloaded-experts FFN).

The op takes *all* experts' weights and the router's selection -- each token's expert ids plus the
score row they index, or the E-wide score row to rank on -- and decides which experts to run from
that, deriving the per-token weights (the selected scores renormalized and scaled) itself. For the
routing-selected ("hit") experts, in ascending hit-id order, it computes the gate_up matmul, the
SwiGLU gate, the down matmul, *and* the routing-weighted accumulation into a single output tile row
on device:

    gu     = x @ gate_up_w[hit_ids[i]]                          # [B, H] @ [H, 2I] -> [B, 2I]
    act    = silu(clamp(gu[:, :I], max=L)) * clamp(gu[:, I:], -L, L)  # -> [B, I]
    output = sum_i w[:, hit_ids[i]] * (act @ down_w[hit_ids[i]])      # -> [B, H]

The selection reaches the op in one of two mutually exclusive forms:

* ``routing_indices``: the router's own ``ttnn.topk`` ids, with ``routing_scores`` the score row they
  index (the original form);
* ``ranking_scores``: the E-wide score row the router would have ranked (typically
  ``routing_scores + bias``), with ``top_k`` given explicitly. The op finds each token's top-k
  itself, inside the leader kernel that already reads the whole score row -- no separate
  ``ttnn.topk`` launch and no DRAM round-trip of its id output.

B tokens (<= 32) are packed into dim -2 and processed together. The hit set is the *union* of the
tokens' selections, so an expert several tokens picked has its weights fetched from DRAM once and its
matmuls run once; the tokens are separated only by the per-token routing weight in the final
accumulation (0 for a token that did not select the expert).

The I SwiGLU columns are distributed across the compute grid: each SwiGLU core owns a
2-tile (64-column) slice and needs both the gate columns [64c, 64c+64) and the paired up
columns [I+64c, I+64c+64) of the gate_up weight, kept in a *single* [H, 128] DRAM shard
(host-permuted into per-core [gate_64 | up_64] blocks). The down matmul contracts over the
full I, so each SwiGLU core scatters its activation slice to core {0,0}, which gathers the
full activation and broadcasts it to every core; each core then multiplies it by its
[I, H/64] down shard to produce its 64-column slice of each expert's [B, H] rows, scales it
by the per-token routing weights for that expert and accumulates across experts. The output
tensor is [1, B, H] in TILE layout (the B token rows padded to a 32-row tile), BFLOAT16.
"""

import pytest
import torch
import ttnn
import random

from models.common.utility_functions import comp_pcc, comp_allclose


# fused_experts DRAM: one [H, 64] [gate_32|up_32] shard per I-tile, and 64 [I, H/64]
# down shards. With 6 selected experts the op runs on a 12x8 = 96-core grid (16 cores
# per expert). Each core covers (I/32)/16 gate_up shards and 4 down shards.
FUSED_EXPERTS_GRID = 8
FUSED_EXPERTS_NUM_CORES = FUSED_EXPERTS_GRID * FUSED_EXPERTS_GRID
BH_NUM_DRAM_BANKS = 8
TILE = 32


def _swiglu_cols_per_core(intermediate: int) -> int:
    """SwiGLU output columns per core: the I dim is spread over all 64 cores so that every
    core fetches gate_up weights during the DRAM-bound phase 1."""
    return TILE * max(1, (intermediate // TILE) // FUSED_EXPERTS_NUM_CORES)


def _nd_sharded_dram_memory_config(
    rows: int, cols: int, shard_width: int, dram_core_range_set: ttnn.CoreRangeSet
) -> ttnn.MemoryConfig:
    """ND-sharded DRAM: ``rows`` × ``shard_width`` per shard, round-robin over the DRAM banks."""
    assert cols % shard_width == 0, f"last dim {cols} must divide evenly into shards of {shard_width}"
    dram_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[rows, shard_width],
        grid=dram_core_range_set,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, dram_nd_shard_spec)


def _interleave_gate_up(w: torch.Tensor, block: int) -> torch.Tensor:
    """Permute a [K, 2I] gate_up weight into per-core [gate_block | up_block] order so each
    [K, 2*block] shard holds a core's gate columns followed by its paired up columns.

    gate = w[:, :I], up = w[:, I:]; output column (c*2*block + h*block + t) == w[:, h*I + c*block + t].
    """
    k, two_i = w.shape
    intermediate = two_i // 2
    blocks = intermediate // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def _swiglu(gu: torch.Tensor, intermediate: int, limit: float) -> torch.Tensor:
    """Reference SwiGLU on a [tokens, 2I] gate_up output -> [tokens, I]."""
    gate = torch.clamp(gu[:, :intermediate], max=limit)
    up = torch.clamp(gu[:, intermediate:], min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


def _expert_weights(device, hidden: int, intermediate: int, num_experts: int):
    """Random per-expert gate_up / down weights, on host and as the DRAM ND-sharded bf4 tensors
    the op requires (each shard exactly one core's slice, read in a single NoC read)."""
    two_intermediate = 2 * intermediate
    gate_up = [(torch.rand((hidden, two_intermediate), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)]
    down = [(torch.rand((intermediate, hidden), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)]

    dram_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)]
    )
    swiglu_cols = _swiglu_cols_per_core(intermediate)
    gate_up_mem_config = _nd_sharded_dram_memory_config(hidden, two_intermediate, 2 * swiglu_cols, dram_core_range_set)
    down_mem_config = _nd_sharded_dram_memory_config(
        intermediate, hidden, hidden // FUSED_EXPERTS_NUM_CORES, dram_core_range_set
    )

    def to_tt(t, memory_config):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
        )

    gate_up_tt = [to_tt(_interleave_gate_up(w, swiglu_cols), gate_up_mem_config) for w in gate_up]
    down_tt = [to_tt(w, down_mem_config) for w in down]
    return gate_up, down, gate_up_tt, down_tt


@pytest.mark.parametrize(
    "hidden, intermediate, num_experts, top_k",
    [
        # DeepSeek-V4-Flash config sizes (hidden_size=4096, moe_intermediate_size=2048).
        # The model has n_routed_experts=256; we use fewer here to keep DRAM/host memory
        # tractable for a unit test (each [4096, 4096] gate_up weight is ~32 MB).
        (4096, 2048, 64, 6),
    ],
)
# The B tokens are the rows of dim -2 and share one 32-row tile. Every token is scaled by its own
# routing weights, and experts selected by several tokens are fetched and multiplied ONCE for the
# batch (the op iterates the deduplicated union of the rows' selections), so the cases below
# deliberately include tokens with fully shared and only partly shared expert sets.
@pytest.mark.parametrize("batch", (1, 2, 32), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("share_experts", (True, False), ids=("shared_experts", "disjoint_rows"))
# ``experts_block_size`` bounds how many experts' activations are held in L1 at once, so the op runs
# the selected experts in blocks of that size instead of all at once. It must not change the result.
# 0 is the default single block; with 6 hit experts, 2 gives three full blocks (so a block reuses an
# earlier block's activation slot, the case the inter-block handoff exists for) and 4 gives a short
# final block.
@pytest.mark.parametrize("experts_block_size", (0, 2, 4), ids=lambda b: f"block{b}")
def test_fused_experts_gate_up(
    device, hidden, intermediate, num_experts, top_k, batch, share_experts, experts_block_size
):
    torch.manual_seed(0)
    limit = 7.0
    tokens = batch
    two_intermediate = 2 * intermediate
    scaling = 1.0
    eps = 1e-20

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_flat = x.reshape(1, 1, tokens, hidden)

    # Each token names its own ``top_k`` experts; the union over tokens is the routing-selected
    # ("hit") set, which the op runs once each in ascending hit-id order for all tokens.
    #
    # ``share_experts``: every token picks the same experts, so the union is top_k however large the
    # batch is (the weight-sharing case). Otherwise the tokens rotate through a pool one wider, so
    # rows differ while the union stays small -- which also covers a token contributing nothing to
    # an expert that other tokens did select.
    pool = random.sample(range(num_experts), top_k + 1)
    ids = torch.stack(
        [
            torch.tensor(pool[:top_k] if share_experts else [pool[(t + j) % len(pool)] for j in range(top_k)])
            for t in range(tokens)
        ]
    )
    scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.5
    hit_ids = sorted(set(ids.flatten().tolist()))
    num_active = len(hit_ids)

    gate_up_weights = [
        (torch.rand((hidden, two_intermediate), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]
    down_weights = [
        (torch.rand((intermediate, hidden), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]
    # Permute each gate_up weight into per-core [gate | up] blocks so each shard holds
    # everything a core needs for its SwiGLU output slice in one NoC read.
    swiglu_cols = _swiglu_cols_per_core(intermediate)
    gate_up_perm = [_interleave_gate_up(w, swiglu_cols) for w in gate_up_weights]

    def to_tt(t, layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=layout, memory_config=memory_config)

    dram_core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_ranges)

    # Each gate_up shard is one core's [H, 2*swiglu_cols] (gate | up) slice.
    gate_up_mem_config = _nd_sharded_dram_memory_config(hidden, two_intermediate, 2 * swiglu_cols, dram_core_range_set)
    down_mem_config = _nd_sharded_dram_memory_config(
        intermediate, hidden, hidden // FUSED_EXPERTS_NUM_CORES, dram_core_range_set
    )

    x_tt = ttnn.from_torch(
        x_flat,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ids_tt = to_tt(ids.to(torch.int32).reshape(1, 1, tokens, top_k), ttnn.TILE_LAYOUT, dtype=ttnn.uint16)
    scores_tt = to_tt(scores.reshape(1, 1, tokens, num_experts), ttnn.TILE_LAYOUT)
    gate_up_tt = [
        to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, memory_config=gate_up_mem_config) for w in gate_up_perm
    ]
    down_tt = [to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, memory_config=down_mem_config) for w in down_weights]

    tt_out = ttnn.experimental.deepseek.moe.fused_experts(
        x_tt,
        routing_indices=ids_tt,
        routing_scores=scores_tt,
        gate_up_weights=gate_up_tt,
        down_weights=down_tt,
        num_experts=num_active,
        intermediate_size=intermediate,
        swiglu_limit=limit,
        top_k=top_k,
        routed_scaling_factor=scaling,
        routing_eps=eps,
        experts_block_size=experts_block_size,
    )

    out_torch = ttnn.to_torch(tt_out).float()  # [1, B, H]
    assert list(out_torch.shape) == [1, tokens, hidden], f"unexpected output shape {out_torch.shape}"

    # The op returns, per token, the routing-weighted sum over the selected experts:
    #   out[b] = sum_i w[b, hit_ids[i]] * (swiglu(x[b] @ gate_up_w) @ down_w).
    # Every hit expert is evaluated for every token and masked by that token's weight, which is
    # exactly how one shared fetch serves several tokens. Reference uses the bf16-rounded input and
    # scores to match the device path; the chained bf4 matmuls add quantization error, so PCC (not
    # exact match) is checked.
    x_dev = ttnn.to_torch(x_tt).float().reshape(tokens, hidden)
    scores_dev = ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts)
    selected_dev = torch.gather(scores_dev, -1, ids)
    rw_dev = torch.zeros((tokens, num_experts), dtype=torch.float32)
    rw_dev.scatter_(-1, ids, scaling * selected_dev / (selected_dev.sum(dim=-1, keepdim=True) + eps))
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in hit_ids:
        gu = (x_dev @ gate_up_weights[e]).reshape(tokens, two_intermediate)  # [B, 2I]
        act = _swiglu(gu, intermediate, limit)  # [B, I]
        ref = ref + rw_dev[:, e : e + 1] * (act @ down_weights[e])  # [B, H], weighted-accumulated

    got = out_torch.reshape(tokens, hidden)
    passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
    assert passing, f"weighted-sum output mismatch: {pcc_msg} | {comp_allclose(ref, got)}"


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 64, 6)])
# ``share_experts`` fixes whether the tokens select the same experts (union == top_k, the
# weight-sharing case) or rotate through different ones (union grows with the batch), which is what
# the sparse path's dedup has to collapse.
@pytest.mark.parametrize("batch", (1, 2, 8), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("share_experts", (True, False), ids=("shared_experts", "disjoint_rows"))
# The router may rank by a bias-corrected score while weighting by the uncorrected one, so the ids
# do not simply point at each token's largest weights -- the op must use the scores it is handed.
@pytest.mark.parametrize("use_bias", (False, True), ids=("no_bias", "correction_bias"))
def test_fused_experts_sparse_routing(device, hidden, intermediate, num_experts, top_k, batch, share_experts, use_bias):
    """Routing straight off ``ttnn.topk``: the op takes its ids plus the score row and derives the
    hit set and the per-token weights itself.

    The selection here comes from a real topk on device (optionally ranked on a bias-corrected copy
    of the scores), so this covers the router's actual output -- shape, dtype and tie-breaking --
    rather than ids the test made up.
    """
    torch.manual_seed(0)
    limit = 7.0
    tokens = batch
    scaling = 2.5
    eps = 1e-20

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, tokens, hidden),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Router scores are non-negative (the model's are sqrt(softplus(logits))). Boosting a per-token
    # set of columns is what steers the selection, and hence the size of the union.
    scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() * 0.1
    for t in range(tokens):
        first = 0 if share_experts else t
        for j in range(top_k):
            scores[t, (first + j) % num_experts] += 1.0
    scores_tt = ttnn.from_torch(
        scores.reshape(1, 1, tokens, num_experts),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Rank on the bias-corrected copy but weight with the uncorrected scores, as the model does.
    ranked_tt = scores_tt
    if use_bias:
        bias = ttnn.from_torch(
            (torch.rand((1, 1, 1, num_experts), dtype=torch.bfloat16).float() - 0.5) * 0.2,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ranked_tt = ttnn.add(scores_tt, bias)
    _, idx_tt = ttnn.topk(ranked_tt, top_k, dim=-1)

    # The device's own selection drives the reference: bf16 ties are common at this width, so which
    # of several equal scores topk returns is its business.
    ids = ttnn.to_torch(idx_tt).to(torch.int64).reshape(tokens, top_k)
    scores_dev = ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts)
    hit_ids = sorted(set(ids.flatten().tolist()))
    num_active = len(hit_ids)

    # The weights the op should derive: each token's selected scores renormalized and scaled,
    # widened here only so the reference below can index them expert-major.
    selected = torch.gather(scores_dev, -1, ids)  # [T, k]
    weights = torch.zeros((tokens, num_experts), dtype=torch.float32)
    weights.scatter_(-1, ids, scaling * selected / (selected.sum(dim=-1, keepdim=True) + eps))

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)

    # A block bounds the resident activation, which otherwise scales with the (batch-dependent) union.
    experts_block_size = 4

    out = ttnn.experimental.deepseek.moe.fused_experts(
        x_tt,
        routing_indices=idx_tt,
        routing_scores=scores_tt,
        gate_up_weights=gate_up_tt,
        down_weights=down_tt,
        num_experts=num_active,
        intermediate_size=intermediate,
        swiglu_limit=limit,
        top_k=top_k,
        routed_scaling_factor=scaling,
        routing_eps=eps,
        experts_block_size=experts_block_size,
    )
    got = ttnn.to_torch(out).float().reshape(tokens, hidden)

    # Golden: every hit expert evaluated for every token, scaled by that token's weight for it.
    x_dev = ttnn.to_torch(x_tt).float().reshape(tokens, hidden)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in hit_ids:
        act = _swiglu((x_dev @ gate_up[e]).reshape(tokens, 2 * intermediate), intermediate, limit)
        ref = ref + weights[:, e : e + 1] * (act @ down[e])
    passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
    assert passing, f"topk routing vs torch golden: {pcc_msg} | {comp_allclose(ref, got)}"


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 64, 6)])
@pytest.mark.parametrize("batch", (1, 8), ids=lambda b: f"batch{b}")
# A frozen table can name the same expert twice for one token. The reference collapses the repeat
# (it scatters the selection into a one-hot mask), so the op has to collapse it too.
@pytest.mark.parametrize("duplicate_ids", (False, True), ids=("distinct_ids", "repeated_id"))
def test_fused_experts_bf16_indices(device, hidden, intermediate, num_experts, top_k, batch, duplicate_ids):
    """Routing with the ids delivered as bf16 rather than uint16.

    That is the form a table-driven router produces: ``ttnn.embedding`` only gathers from a
    bfloat16 table, so its frozen token-id -> expert-id table hands the ids over as bf16 values
    (exact for E <= 256). The ids here are gathered exactly that way.
    """
    torch.manual_seed(0)
    limit = 7.0
    tokens = batch
    scaling = 2.5
    eps = 1e-20
    vocab = 128

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, tokens, hidden),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.1
    scores_tt = ttnn.from_torch(
        scores.reshape(1, 1, tokens, num_experts),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # The router's frozen table, and the ids gathered from it for this step's tokens.
    tid2eid = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(vocab)])
    if duplicate_ids:
        tid2eid[:, -1] = tid2eid[:, 0]
    token_ids = torch.randint(0, vocab, (1, tokens), dtype=torch.int32)
    table_tt = ttnn.from_torch(tid2eid.float(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ids_tt = ttnn.from_torch(token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx_tt = ttnn.embedding(ids_tt, table_tt, layout=ttnn.TILE_LAYOUT)  # [1, T, k] bf16 expert ids
    idx_tt = ttnn.reshape(idx_tt, [1, 1, tokens, top_k])

    ids = tid2eid[token_ids.reshape(-1).long()]  # [T, k]
    hit_ids = sorted(set(ids.flatten().tolist()))

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)

    got = (
        ttnn.to_torch(
            ttnn.experimental.deepseek.moe.fused_experts(
                x_tt,
                routing_indices=idx_tt,
                routing_scores=scores_tt,
                gate_up_weights=gate_up_tt,
                down_weights=down_tt,
                num_experts=len(hit_ids),
                intermediate_size=intermediate,
                swiglu_limit=limit,
                top_k=top_k,
                routed_scaling_factor=scaling,
                routing_eps=eps,
                experts_block_size=4,
            )
        )
        .float()
        .reshape(tokens, hidden)
    )

    # Reference weights: a one-hot selection mask (which is where a repeated id collapses), then
    # normalize-and-scale over the masked scores.
    scores_dev = ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts)
    mask = torch.zeros((tokens, num_experts), dtype=torch.float32)
    mask.scatter_(-1, ids, 1.0)
    masked = scores_dev * mask
    weights = scaling * masked / (masked.sum(dim=-1, keepdim=True) + eps)

    x_dev = ttnn.to_torch(x_tt).float().reshape(tokens, hidden)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in hit_ids:
        act = _swiglu((x_dev @ gate_up[e]).reshape(tokens, 2 * intermediate), intermediate, limit)
        ref = ref + weights[:, e : e + 1] * (act @ down[e])
    passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
    assert passing, f"bf16-id routing vs torch golden: {pcc_msg} | {comp_allclose(ref, got)}"


def _ranking_rows(tokens: int, num_experts: int, top_k: int, share: bool):
    """A ``[tokens, E]`` ranking row whose top-``top_k`` winners are known exactly.

    Every entry is an integer, so all values are distinct within a row and exactly representable in
    bf16. That is what keeps the device's on-kernel top-k and the torch reference from disagreeing on
    a tie -- and lets the test assert on the union size -- while the selected experts are lifted
    clear of the ``0..E-1`` base values so they are unambiguously the largest.

    ``share`` makes every token rank the same ``top_k`` experts (union == top_k, the weight-sharing
    case); otherwise the tokens rotate through the expert range so the union grows with the batch,
    which is what the kernel's dedup has to collapse.
    """
    ranking = torch.arange(num_experts, dtype=torch.float32).repeat(tokens, 1).clone()
    chosen = []
    for t in range(tokens):
        sel = [((0 if share else t) * top_k + j) % num_experts for j in range(top_k)]
        for j, e in enumerate(sel):
            ranking[t, e] = num_experts + (top_k - j)
        chosen.append(sel)
    hit_ids = sorted({e for sel in chosen for e in sel})
    return ranking, chosen, hit_ids


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 8, 4)])
@pytest.mark.parametrize("batch", (1, 2, 8), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("share_ranking", (True, False), ids=("shared_ranking", "disjoint_ranking"))
# Ranking on one row and weighting with another is the model's bias-corrected case ("biased_ranking");
# passing one tensor as both is the aliased read, where the row is fetched once and ranked in place.
@pytest.mark.parametrize("rank_is_weight", (False, True), ids=("biased_ranking", "rank_is_weight"))
# ROW_MAJOR scores are the single-token decode stick, so they only pair with batch == 1.
@pytest.mark.parametrize("row_major_scores", (False, True), ids=("tile_scores", "rm_decode_scores"))
def test_fused_experts_ranking_scores(
    device, hidden, intermediate, num_experts, top_k, batch, share_ranking, rank_is_weight, row_major_scores
):
    """``ranking_scores``: the op finds each token's top-k itself instead of being handed ids.

    The ranking row is the *bias-corrected* row the router would have ranked, while the weights come
    from the unbiased ``routing_scores`` at the winners -- so the winners must not simply be the
    token's largest weights, which is why the two rows are unrelated here. With
    ``rank_is_weight`` the same tensor is passed for both, which is the aliased (read-once) path.

    The golden is the op's own normalize-and-scale tail over the selected scores, with every hit
    expert evaluated for every token and masked by that token's weight, exactly as in the ids form.
    """
    if row_major_scores and batch != 1:
        pytest.skip("ROW_MAJOR scores are the single-token decode path")

    torch.manual_seed(0)
    limit = 7.0
    tokens = batch
    scaling = 2.5
    eps = 1e-20
    experts_block_size = 2

    ranking, _, expected_hit_ids = _ranking_rows(tokens, num_experts, top_k, share_ranking)
    num_active = len(expected_hit_ids)

    # The unbiased scores that become the weights.
    weight_scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.25

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, tokens, hidden),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    scores_layout = ttnn.ROW_MAJOR_LAYOUT if row_major_scores else ttnn.TILE_LAYOUT

    def to_scores_tt(host):
        return ttnn.from_torch(
            host.reshape(1, 1, tokens, num_experts),
            dtype=ttnn.bfloat16,
            device=device,
            layout=scores_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    rank_tt = to_scores_tt(ranking)
    scores_tt = rank_tt if rank_is_weight else to_scores_tt(weight_scores)

    # The device's own ranking drives the reference -- which of several equal scores wins is the
    # op's business -- so the ids are read back out of the tensor the op receives. The constructed
    # rows are tie-free, so the union is known and can be asserted rather than assumed.
    rank_dev = ttnn.to_torch(rank_tt).float().reshape(tokens, num_experts)
    ids = torch.topk(rank_dev, top_k, dim=-1).indices
    assert sorted(set(ids.flatten().tolist())) == expected_hit_ids, "the ranking rows must be tie-free"

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)

    got = (
        ttnn.to_torch(
            ttnn.experimental.deepseek.moe.fused_experts(
                x_tt,
                routing_scores=scores_tt,
                ranking_scores=rank_tt,
                gate_up_weights=gate_up_tt,
                down_weights=down_tt,
                num_experts=num_active,
                intermediate_size=intermediate,
                swiglu_limit=limit,
                top_k=top_k,
                routed_scaling_factor=scaling,
                routing_eps=eps,
                experts_block_size=experts_block_size,
            )
        )
        .float()
        .reshape(tokens, hidden)
    )

    scores_dev = ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts)
    selected = torch.gather(scores_dev, -1, ids)
    weights = torch.zeros((tokens, num_experts), dtype=torch.float32)
    weights.scatter_(-1, ids, scaling * selected / (selected.sum(dim=-1, keepdim=True) + eps))

    x_dev = ttnn.to_torch(x_tt).float().reshape(tokens, hidden)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in expected_hit_ids:
        act = _swiglu((x_dev @ gate_up[e]).reshape(tokens, 2 * intermediate), intermediate, limit)
        ref = ref + weights[:, e : e + 1] * (act @ down[e])
    passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
    assert passing, f"ranking_scores routing vs torch golden: {pcc_msg} | {comp_allclose(ref, got)}"


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 8, 4)])
@pytest.mark.parametrize("batch", (1, 2, 8), ids=lambda b: f"batch{b}")
@pytest.mark.parametrize("share_ranking", (True, False), ids=("shared_ranking", "disjoint_ranking"))
def test_fused_experts_ranking_scores_match_topk_ids(
    device, hidden, intermediate, num_experts, top_k, batch, share_ranking
):
    """Ranking inside the op must land on the same experts and weights as the ids form.

    Both calls are handed the same ranking row and the same unbiased score row; one is given
    ``ttnn.topk``'s ids (the original interface) and the other ``ranking_scores`` (the op's own
    selection). With tie-free ranking rows the two selections are identical, so the outputs have to
    agree -- which is both the equivalence check for the new path and the backward-compatibility
    check for the old one.
    """
    torch.manual_seed(0)
    limit = 7.0
    tokens = batch
    scaling = 2.5
    eps = 1e-20
    experts_block_size = 2

    ranking, _, expected_hit_ids = _ranking_rows(tokens, num_experts, top_k, share_ranking)
    num_active = len(expected_hit_ids)
    weight_scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.25
    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()

    def to_tt(host, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            host.reshape(1, 1, tokens, num_experts),
            dtype=dtype,
            device=device,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, tokens, hidden),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rank_tt = to_tt(ranking)
    scores_tt = to_tt(weight_scores)
    _, idx_tt = ttnn.topk(rank_tt, top_k, dim=-1)

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)

    def run(**selection):
        return (
            ttnn.to_torch(
                ttnn.experimental.deepseek.moe.fused_experts(
                    x_tt,
                    routing_scores=scores_tt,
                    gate_up_weights=gate_up_tt,
                    down_weights=down_tt,
                    num_experts=num_active,
                    intermediate_size=intermediate,
                    swiglu_limit=limit,
                    top_k=top_k,
                    routed_scaling_factor=scaling,
                    routing_eps=eps,
                    experts_block_size=experts_block_size,
                    **selection,
                )
            )
            .float()
            .reshape(tokens, hidden)
        )

    got_ids = run(routing_indices=idx_tt)
    got_ranked = run(ranking_scores=rank_tt)

    passing, pcc_msg = comp_pcc(got_ids, got_ranked, pcc=0.999)
    assert passing, f"ranking_scores and topk-ids routing disagree: {pcc_msg} | {comp_allclose(got_ids, got_ranked)}"

    # Anchor both against the golden too, so a shared error cannot hide behind the comparison.
    ids = ttnn.to_torch(idx_tt).to(torch.int64).reshape(tokens, top_k)
    assert sorted(set(ids.flatten().tolist())) == expected_hit_ids
    ref, hit_ids = _gather_reference(
        ttnn.to_torch(x_tt).float().reshape(tokens, hidden),
        ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts),
        ids,
        gate_up,
        down,
        intermediate,
        limit,
        scaling,
        eps,
    )
    assert hit_ids == expected_hit_ids
    for name, got in (("routing_indices", got_ids), ("ranking_scores", got_ranked)):
        passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
        assert passing, f"{name} vs torch golden: {pcc_msg} | {comp_allclose(ref, got)}"


def _replicated_row(device, x_row: torch.Tensor, grid: ttnn.CoreRangeSet) -> ttnn.Tensor:
    """Replicate one ROW_MAJOR token row onto ``grid`` in the layout ``matmul_decode`` consumes in
    place: L1 HEIGHT_SHARDED, ROW_MAJOR orientation, shard == the whole ``[1, H]`` row.

    The row starts life on a single core (the only sharded input ``all_gather_for_matmul`` accepts
    for the multicast path) and is replicated from there.
    """
    height, width = x_row.shape[-2], x_row.shape[-1]
    assert height == 1, "the replicated ROW_MAJOR path is the single-token 1x32 decode form"
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    shard_spec = ttnn.ShardSpec(one_core, (height, width), ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    single = ttnn.from_torch(
        x_row,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )
    return ttnn.experimental.deepseek.all_gather_for_matmul(single, grid)


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 8, 2)])
def test_fused_experts_replicated_input(device, hidden, intermediate, num_experts, top_k):
    """A ROW_MAJOR HEIGHT_SHARDED L1 replica is consumed in place, like matmul_decode's replicated A.

    Every core already holds the row, so the op neither reads it from DRAM nor runs its input
    broadcast -- cb_input is aliased over the local shard. The result must be the same as the DRAM
    TILE row (which does go through the broadcast) and must match the torch golden, and the op has
    to take its token count from the shard height, since dim -2 of the replica carries
    ``num_cores * B``.
    """
    torch.manual_seed(0)
    limit = 7.0
    scaling = 2.5
    eps = 1e-20
    tokens = 1  # ROW_MAJOR is the 1x32 decode path: one token row per call.

    # Two distinct ids for the one token -> num_active == 2, so the op stays on the 8x8 grid (the
    # 6-expert / 96-core path would need a 12x8 replica).
    ids = torch.tensor([[0, 1]], dtype=torch.int64)
    hit_ids = sorted(set(ids.flatten().tolist()))
    scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.5
    x = (torch.rand((1, 1, tokens, hidden), dtype=torch.bfloat16) - 0.5).float()

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32).reshape(1, 1, tokens, top_k),
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores_tt = ttnn.from_torch(
        scores.reshape(1, 1, tokens, num_experts),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(FUSED_EXPERTS_GRID - 1, FUSED_EXPERTS_GRID - 1))]
    )

    def run(x_tt):
        return (
            ttnn.to_torch(
                ttnn.experimental.deepseek.moe.fused_experts(
                    x_tt,
                    routing_indices=ids_tt,
                    routing_scores=scores_tt,
                    gate_up_weights=gate_up_tt,
                    down_weights=down_tt,
                    num_experts=len(hit_ids),
                    intermediate_size=intermediate,
                    swiglu_limit=limit,
                    top_k=top_k,
                    routed_scaling_factor=scaling,
                    routing_eps=eps,
                )
            )
            .float()
            .reshape(tokens, hidden)
        )

    x_tile = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_repl = _replicated_row(device, x, grid)
    # The replica is what the op expects: ROW_MAJOR, L1, HEIGHT_SHARDED, shard == the full row, and
    # the logical height carrying the core count.
    assert x_repl.layout == ttnn.ROW_MAJOR_LAYOUT
    assert x_repl.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert tuple(x_repl.memory_config().shard_spec.shape) == (tokens, hidden)
    # dim -2 carries the replica factor, so the op has to take the row count from the shard height.
    assert list(x_repl.shape) == [1, 1, tokens * grid.num_cores(), hidden]

    got_repl = run(x_repl)
    got_tile = run(x_tile)
    # The two input paths (broadcast from DRAM vs read locally) must agree, and both have to match
    # the torch golden -- a shared bug would not survive the second check.
    passing, pcc_msg = comp_pcc(got_tile, got_repl, pcc=0.999)
    assert (
        passing
    ), f"replicated input differs from the DRAM TILE input: {pcc_msg} | {comp_allclose(got_tile, got_repl)}"

    scores_dev = ttnn.to_torch(scores_tt).float().reshape(tokens, num_experts)
    selected_dev = torch.gather(scores_dev, -1, ids)
    rw_dev = torch.zeros((tokens, num_experts), dtype=torch.float32)
    rw_dev.scatter_(-1, ids, scaling * selected_dev / (selected_dev.sum(dim=-1, keepdim=True) + eps))
    x_dev = ttnn.to_torch(x_tile).float().reshape(tokens, hidden)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in hit_ids:
        act = _swiglu((x_dev @ gate_up[e]).reshape(tokens, 2 * intermediate), intermediate, limit)
        ref = ref + rw_dev[:, e : e + 1] * (act @ down[e])
    passing, pcc_msg = comp_pcc(ref, got_repl, pcc=0.98)
    assert passing, f"replicated input vs torch golden: {pcc_msg} | {comp_allclose(ref, got_repl)}"


def _gather_reference(x_dev, scores_dev, ids, gate_up, down, intermediate, limit, scaling, eps):
    """Torch golden for the routing-weighted expert sum: the op's own normalize-and-scale tail over
    the selected scores, then every hit expert evaluated for every token (one matmul per expert,
    shared by all tokens) and masked by that token's weight. Returns (reference, hit_ids)."""
    tokens = x_dev.shape[0]
    selected = torch.gather(scores_dev, -1, ids)
    weights = torch.zeros_like(scores_dev, dtype=torch.float32)
    weights.scatter_(-1, ids, scaling * selected / (selected.sum(dim=-1, keepdim=True) + eps))
    hit_ids = sorted(set(ids.flatten().tolist()))
    ref = torch.zeros((tokens, x_dev.shape[-1]), dtype=torch.float32)
    for e in hit_ids:
        act = _swiglu((x_dev @ gate_up[e]).reshape(tokens, 2 * intermediate), intermediate, limit)
        ref = ref + weights[:, e : e + 1] * (act @ down[e])
    return ref, hit_ids


def _run_both_gather_shapes(device, hidden, intermediate, num_experts, top_k, experts_block_size):
    """Build one case and run it through ``fused_experts`` twice -- a two-hub gather and the
    single-hub fallback -- returning ``(torch golden, two-hub output, single-hub output)``.

    One token row is enough: the gather accounting is per expert block, not per token, so ``top_k``
    (the number of distinct experts) is what sets the hub work -- and choosing it above
    ``experts_block_size`` is what puts the op into its multi-block path.
    """
    torch.manual_seed(0)
    limit = 7.0
    scaling = 2.5
    eps = 1e-20
    tokens = 1

    # Distinct ids, so the union is exactly top_k and every core's chunk of every expert is gathered.
    ids = torch.arange(top_k, dtype=torch.int64).reshape(tokens, top_k)
    scores = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.5
    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()

    gate_up, down, gate_up_tt, down_tt = _expert_weights(device, hidden, intermediate, num_experts)
    ref, hit_ids = _gather_reference(x, scores, ids, gate_up, down, intermediate, limit, scaling, eps)
    assert len(hit_ids) == top_k, "the ids must name distinct experts for the union to be top_k"

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, tokens, hidden),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32).reshape(1, 1, tokens, top_k),
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scores_tt = ttnn.from_torch(
        scores.reshape(1, 1, tokens, num_experts),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run(two_hub_gather):
        return (
            ttnn.to_torch(
                ttnn.experimental.deepseek.moe.fused_experts(
                    x_tt,
                    routing_indices=ids_tt,
                    routing_scores=scores_tt,
                    gate_up_weights=gate_up_tt,
                    down_weights=down_tt,
                    num_experts=len(hit_ids),
                    intermediate_size=intermediate,
                    swiglu_limit=limit,
                    top_k=top_k,
                    routed_scaling_factor=scaling,
                    routing_eps=eps,
                    experts_block_size=experts_block_size,
                    two_hub_gather=two_hub_gather,
                )
            )
            .float()
            .reshape(tokens, hidden)
        )

    return ref, run(True), run(False)


def _assert_gather_shapes_agree(ref, got_two, got_one):
    """Both gather shapes must agree with each other and with the golden.

    They cannot differ in arithmetic -- same fetch set, same matmuls, same accumulation order; only
    the set of cores that gather half the activation and multicast it changes -- so agreement is the
    whole contract of the split. Checking each against the torch golden as well stops an error the
    two paths share from hiding behind the comparison.
    """
    passing, pcc_msg = comp_pcc(got_one, got_two, pcc=0.999)
    assert passing, f"two-hub and single-hub gathers disagree: {pcc_msg} | {comp_allclose(got_one, got_two)}"
    for name, got in (("two_hub", got_two), ("one_hub", got_one)):
        passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
        assert passing, f"{name} gather vs torch golden: {pcc_msg} | {comp_allclose(ref, got)}"


@pytest.mark.parametrize(
    "hidden, intermediate",
    [
        # I == 2048: all 64 cores own a SwiGLU slice, so both hubs gather only from producers and no
        # core ever sends a slot-free ack. The I split lands at tile 32, so each hub multicasts half
        # of every expert of the block.
        (4096, 2048),
        # TP-sized I: only 16 of the 64 cores own a slice. hub1 -- the grid's far corner, core 63 --
        # therefore owns none, so it owes its peer a per-block ack and must not wait on its own.
        (4096, 512),
    ],
    ids=("i2048", "i512_hub1_owns_no_slice"),
)
@pytest.mark.parametrize("experts_block_size", (0, 2), ids=("one_block", "two_blocks"))
def test_fused_experts_gather_hubs(device, hidden, intermediate, experts_block_size):
    """The two-hub activation gather/broadcast must be invisible in the result.

    ``fused_experts`` splits each expert block's I dim across two hubs -- the opposite corners of the
    multicast rectangle -- each gathering its half onto itself and multicasting it back on its own
    NoC (hub0 on NoC 0, hub1 on NoC 1, because two multicast senders into one rectangle on one NoC
    circular-wait on overlapping path reservations). Every core then waits for both halves before
    publishing the slot to its compute. This runs each configuration through both the two-hub path
    and the single-hub fallback (``two_hub_gather=False``) and requires them to agree.

    The cases are chosen for the code they reach:

    * ``i512_hub1_owns_no_slice`` is the one that matters most. At I == 512 only 16 of the 64 cores
      own a SwiGLU slice, so hub1 owns no chunk at all: it must still ack its peer each block, or the
      peer's gather target is never reached. This is the accounting that hangs when it is wrong.
    * ``two_blocks`` runs 4 hit experts in blocks of 2, which is what exercises the deferred
      ``push_back`` into the double-buffered ``cb_act``, the ``reserve_back(current + next)`` that
      replaces the single-hub pipeline's push-then-reserve order, and the per-block accumulation of
      each hub's gather target.
    * ``i2048`` is the all-producers case, where the two halves are cut mid-I and each hub
      multicasts a strict sub-range of every expert.

    NOTE: a liveness bug in the gather shows up as a hang, not as a wrong number -- the op will sit
    in a semaphore wait. The single-hub run is the control: if it hangs too, the fault is in the
    shared machinery rather than the split.
    """
    ref, got_two, got_one = _run_both_gather_shapes(device, hidden, intermediate, 8, 4, experts_block_size)
    _assert_gather_shapes_agree(ref, got_two, got_one)


@pytest.mark.parametrize("hidden, intermediate, num_experts, top_k", [(4096, 2048, 64, 6)])
def test_fused_experts_gather_hubs_grouped(device, hidden, intermediate, num_experts, top_k):
    """The 6-expert / 96-core path gives every expert its own 16-core group, and therefore its own
    hub pair: the two corners of that group's 2x8 multicast rectangle, not the whole grid's.

    That path carries hub coordinates, a destination count and a gather target per group, all of
    them runtime args (a single kernel instance covers all six groups), and all of them different
    from the whole-grid case above -- so it needs its own coverage rather than riding on it.
    """
    grid = device.compute_with_storage_grid_size()
    if grid.x < 12 or grid.y < 8:
        pytest.skip(f"the 6-expert / 96-core path needs a 12x8 compute grid, got {grid.x}x{grid.y}")

    # Six distinct experts on one token is exactly what selects the 96-core path.
    ref, got_two, got_one = _run_both_gather_shapes(device, hidden, intermediate, num_experts, top_k, 0)
    _assert_gather_shapes_agree(ref, got_two, got_one)
