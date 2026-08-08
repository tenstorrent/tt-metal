# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.moe.fused_experts (full preloaded-experts FFN).

The op takes *all* experts' weights and the router's selection -- each token's expert ids plus the
score row they index -- and decides which experts to run from that, deriving the per-token weights
(the selected scores renormalized and scaled) itself. For the routing-selected ("hit") experts, in
ascending hit-id order, it computes the gate_up matmul, the SwiGLU gate, the down matmul, *and* the
routing-weighted accumulation into a single output tile row on device:

    gu     = x @ gate_up_w[hit_ids[i]]                          # [B, H] @ [H, 2I] -> [B, 2I]
    act    = silu(clamp(gu[:, :I], max=L)) * clamp(gu[:, I:], -L, L)  # -> [B, I]
    output = sum_i w[:, hit_ids[i]] * (act @ down_w[hit_ids[i]])      # -> [B, H]

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


# fused_experts uses an 8x8 compute grid; each active core owns a 2-tile SwiGLU output
# slice (64 cols), reading a [H, 128] (gate 64 | up 64) gate_up shard.
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

    x_tt = to_tt(x_flat, ttnn.TILE_LAYOUT)
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
