# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""KV capacity arithmetic of the served Gemma 4 engines (host-only, leaf): what a pool costs per chip, how many
users it holds, and what a chip's DRAM allows.

The vllm-tt-plugin turns ``GEMMA4_MAX_TOKENS_ALL_USERS`` into one block pool (``plugin_pool_tokens``) and applies it to
EVERY submesh un-divided: on the lane engine each lane's KV manager and each mesh column's device buffers hold the
whole pool, so the box holds ``lanes x`` what vLLM's capacity line ("Maximum concurrency for N tokens per request")
reports -- that line is one lane's view. Per chip, the paged KV is the global-attention layers' pages (one local KV
head per chip when the model has fewer global KV heads than the TP group: the head is REPLICATED across the group,
which is where the lane engine's bytes exceed the sharded minimum) plus the sliding layers' window-sized caches.
"""

import math

BFP8_BYTES = 1.0625  # bfloat8_b: 1 byte per element plus the shared exponent per 16
BF16_BYTES = 2.0


def plugin_pool_tokens(max_tokens_all_users, per_lane_max_num_seqs, block_size, *, sliding_window, hybrid_groups):
    """The pool the plugin allocates (``get_num_available_blocks_tt``): the model's budget plus one output
    reservation block per user, plus, with hybrid KV groups, ``sliding_window x users x 8`` of sliding headroom;
    rounded up to whole blocks."""
    tokens = int(max_tokens_all_users) + int(block_size) * int(per_lane_max_num_seqs)
    if hybrid_groups and sliding_window:
        tokens += int(sliding_window) * int(per_lane_max_num_seqs) * 8
    blocks = math.ceil(tokens / int(block_size))
    return blocks * int(block_size), blocks


def users_at_context(pool_tokens, context, block_size, *, sliding_window=0, sliding_groups=0, output_tokens=0):
    """How many requests of ``context`` tokens one pool holds: a request takes its full-attention blocks plus one
    window per sliding group (vLLM's hybrid manager frees the rest) plus its output reservation."""
    per_request = math.ceil((int(context) + int(output_tokens)) / int(block_size))
    if sliding_groups and sliding_window:
        per_request += int(sliding_groups) * (math.ceil(int(sliding_window) / int(block_size)) + 1)
    return int(pool_tokens) // (per_request * int(block_size))


def lane_engine_capacity(pool_tokens, lanes, context, block_size, **kw):
    """``(per_lane_users, box_users)`` of the lane engine at ``context``: vLLM's capacity line is the per-lane
    number; every lane holds its own copy of the pool."""
    per_lane = users_at_context(pool_tokens, context, block_size, **kw)
    return per_lane, per_lane * int(lanes)


def local_kv_heads(kv_heads, tp):
    """Heads a chip stores for one layer: ``kv_heads / tp`` when they divide, else ONE (the head replicated over
    the ``tp / kv_heads`` chips that share it) -- ``init_kv_cache``'s rule."""
    kv_heads, tp = int(kv_heads), int(tp)
    return 1 if kv_heads < tp else kv_heads // tp


def kv_bytes_per_token_per_chip(*, global_layers, global_kv_heads, global_head_dim, tp, bytes_per_elem=BFP8_BYTES):
    """Paged KV bytes one token costs on one chip: K and V of the chip's local global-attention heads, every
    global layer."""
    heads = local_kv_heads(global_kv_heads, tp)
    return int(global_layers) * 2 * heads * int(global_head_dim) * float(bytes_per_elem)


def sharded_minimum_bytes_per_token_per_chip(
    *, global_layers, global_kv_heads, global_head_dim, tp, bytes_per_elem=BFP8_BYTES
):
    """The same bytes with the global heads split evenly over the TP group (no replication): the floor a
    sequence- or dim-sharded KV layout could reach."""
    return int(global_layers) * 2 * (int(global_kv_heads) / int(tp)) * int(global_head_dim) * float(bytes_per_elem)


def sliding_cache_bytes_per_chip(
    *,
    sliding_layers,
    sliding_kv_heads,
    sliding_head_dim,
    tp,
    users,
    sliding_window,
    block_size,
    bytes_per_elem=BFP8_BYTES,
):
    """The window-sized sliding caches per chip: every sliding layer keeps one window (plus a block) per user."""
    heads = local_kv_heads(sliding_kv_heads, tp)
    tokens = int(users) * (int(sliding_window) + int(block_size))
    return int(sliding_layers) * 2 * heads * int(sliding_head_dim) * float(bytes_per_elem) * tokens


def pool_bytes_per_chip(pool_tokens, bytes_per_token):
    return int(pool_tokens) * float(bytes_per_token)


def pool_tokens_for_dram(dram_bytes, *, weights_bytes, reserve_bytes, sliding_bytes, bytes_per_token, block_size):
    """The largest pool (whole blocks) a chip's DRAM allows after the weights, the fixed reserve (traces,
    activations, scratch) and the sliding caches."""
    free = float(dram_bytes) - float(weights_bytes) - float(reserve_bytes) - float(sliding_bytes)
    if free <= 0:
        return 0
    blocks = int(free // (float(bytes_per_token) * int(block_size)))
    return blocks * int(block_size)


def engine_capacity_plan(
    *,
    model,
    dram_bytes,
    weights_bytes,
    reserve_bytes,
    context,
    block_size,
    users_for_sliding,
    tp,
    bytes_per_elem=BFP8_BYTES,
):
    """Pool and users at ``context`` one engine's chips allow for ``model`` (a geometry dict: global_layers,
    global_kv_heads, global_head_dim, sliding_layers, sliding_kv_heads, sliding_head_dim, sliding_window)."""
    per_token = kv_bytes_per_token_per_chip(
        global_layers=model["global_layers"],
        global_kv_heads=model["global_kv_heads"],
        global_head_dim=model["global_head_dim"],
        tp=tp,
        bytes_per_elem=bytes_per_elem,
    )
    sliding = sliding_cache_bytes_per_chip(
        sliding_layers=model["sliding_layers"],
        sliding_kv_heads=model["sliding_kv_heads"],
        sliding_head_dim=model["sliding_head_dim"],
        tp=tp,
        users=users_for_sliding,
        sliding_window=model["sliding_window"],
        block_size=block_size,
        bytes_per_elem=bytes_per_elem,
    )
    pool = pool_tokens_for_dram(
        dram_bytes,
        weights_bytes=weights_bytes,
        reserve_bytes=reserve_bytes,
        sliding_bytes=sliding,
        bytes_per_token=per_token,
        block_size=block_size,
    )
    users = users_at_context(pool, context, block_size)
    return {
        "bytes_per_token": per_token,
        "sliding_bytes": sliding,
        "pool_tokens": pool,
        "users": users,
        "env_pool": users * int(context),
    }


# The three Gemma 4 text models (HF text configs, 2026-10-24).
GEMMA4_GEOMETRY = {
    "26B": dict(
        global_layers=5,
        global_kv_heads=2,
        global_head_dim=512,
        sliding_layers=25,
        sliding_kv_heads=8,
        sliding_head_dim=256,
        sliding_window=1024,
    ),
    "31B": dict(
        global_layers=10,
        global_kv_heads=4,
        global_head_dim=512,
        sliding_layers=50,
        sliding_kv_heads=16,
        sliding_head_dim=256,
        sliding_window=1024,
    ),
    "12B": dict(
        global_layers=8,
        global_kv_heads=1,
        global_head_dim=512,
        sliding_layers=40,
        sliding_kv_heads=8,
        sliding_head_dim=256,
        sliding_window=1024,
    ),
}
