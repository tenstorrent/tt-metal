# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Token-grouped (routed) expert prefill built from existing ttnn ops.

The sparse_matmul prefill in ``prefill.py`` runs every local expert on every token and zeroes the inactive ones
with the routing weights, so it does about E_loc / (experts a token routes to on this chip) times the necessary
work (16x for gemma-4-26B-A4B: 128 experts, top-8). This path computes only the routed (token, expert) pairs:

1. dispatch: for every local expert, the ids of the tokens routed to it (stable sort of the routed mask), gathered
   with ``ttnn.embedding`` into an expert-major buffer ``[1, E_loc, C, H]``;
2. one batched matmul over experts for the fused gate_up projection, GeGLU, one batched matmul for the down
   projection;
3. combine: for every token its (at most ``KL``) routed experts' rows, gathered slot-major ``[KL, S, H]``,
   scaled by the routing weights and summed with ``fast_reduce_nc``.

All index metadata is built on device from the dense routing weights (compare, cumsum for the in-expert rank,
stable sort for the dispatch ids, topk + gather for the combine ids). The capacity ``C`` of the expert-major buffer
is static (``RoutedPrefillConfig.capacity``):

* ``None`` (default): ``"exact"`` for eager prefill. A call recorded under a device trace (``prefill_traced``, set
  from the model's ``_prefill_trace_mode``) keeps the all-experts path unless the config enables the routed path
  explicitly, in which case ``"full"`` is the default there;
* ``"full"``: ``C`` is the chunk length, exact for any routing, no host interaction, traceable (the same
  matmul work as the all-experts path, so no speedup);
* ``"exact"``: eager runs only, one scalar host readback of the busiest expert's token count sizes ``C``;
* an int: ``C`` is that many rows (rounded up to a tile); tokens routed to an expert beyond its first ``C`` are
  dropped (their routing weight is zeroed), as with a capacity factor.

``GEMMA4_MOE_ROUTED_PREFILL=0`` is the one environment override: it keeps the all-experts sparse_matmul path.
"""

import os
from dataclasses import dataclass
from typing import Optional, Union

import ttnn

from .prefill import apply_geglu_fused

TILE_SIZE = 32


@dataclass(frozen=True)
class RoutedPrefillConfig:
    """Routed prefill settings. ``enabled=None`` follows ``GEMMA4_MOE_ROUTED_PREFILL`` (default on) for eager
    prefill and stays off for traced prefill (the all-experts path is kept bit-identical there); ``True`` forces the
    routed path, also under a trace."""

    enabled: Optional[bool] = None
    min_tokens: int = 128  # shorter calls (the decode frame) keep sparse_matmul
    chunk_size: int = 4096  # tokens per chunk, bounds the expert-major buffers
    # None: "exact" for eager prefill, "full" under a device trace; or "full", "exact" (eager readback), rows per expert
    capacity: Union[None, str, int] = None

    def is_enabled(self, seq_len, prefill_traced=False):
        enabled = self.enabled
        if enabled is None:
            if prefill_traced:
                return False
            enabled = os.environ.get("GEMMA4_MOE_ROUTED_PREFILL", "1") == "1"
        return bool(enabled) and seq_len >= self.min_tokens


def resolve_routed_config(routed_prefill):
    """None -> defaults, bool -> enabled flag (any length), RoutedPrefillConfig -> itself."""
    if isinstance(routed_prefill, RoutedPrefillConfig):
        return routed_prefill
    if routed_prefill is None:
        return RoutedPrefillConfig()
    return RoutedPrefillConfig(enabled=bool(routed_prefill), min_tokens=TILE_SIZE if routed_prefill else 128)


def routed_prefill_enabled(seq_len, routed_prefill=None, prefill_traced=False):
    """True when a prefill call of ``seq_len`` rows takes the routed path."""
    return resolve_routed_config(routed_prefill).is_enabled(seq_len, prefill_traced)


def _round_up(n, m=TILE_SIZE):
    return (n + m - 1) // m * m


def _host_max(tensor, mesh_device):
    """Largest element of a small device tensor, over all devices of the mesh."""
    if hasattr(mesh_device, "shape"):
        host = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    else:
        host = ttnn.to_torch(tensor)
    return int(host.float().max().item())


def _capacity(chunk_len, capacity, counts_max_fn):
    """Rows per expert in the expert-major buffer for a chunk of ``chunk_len`` tokens."""
    if capacity == "full":
        return chunk_len
    if capacity == "exact":
        return min(chunk_len, _round_up(max(1, counts_max_fn())))
    return min(chunk_len, _round_up(max(1, int(capacity))))


def build_routing_metadata(local_routing, num_local, top_k, capacity="full"):
    """Index tensors of the routed path, all computed on device.

    ``local_routing`` [1, 1, n, E_loc] dense (bfloat16, TILE) routing weights of this chip's experts (zero where the
    token does not route to the expert). Returns a dict:

    * ``capacity`` C (python int, multiple of 32, <= n);
    * ``dispatch`` [1, E_loc * C] UINT32 ROW_MAJOR: token id of every expert-major row (0 for unused rows, whose
      outputs are never gathered);
    * ``combine`` [1, KL * n] UINT32 ROW_MAJOR: expert-major row of every (slot, token), slot-major (0 for unused
      slots, whose weight is zero);
    * ``slot_weights`` [1, KL, n, 1] bfloat16 TILE: routing weight of every (slot, token), zero for unused or
      capacity-dropped slots;
    * ``kl`` slots per token.
    """
    n = int(local_routing.shape[2])
    kl = max(1, min(int(top_k), int(num_local)))

    mask = ttnn.gt(local_routing, 0.0)  # 1 where the token routes to the expert
    mask_f32 = ttnn.typecast(mask, ttnn.float32)
    csum = ttnn.cumsum(mask_f32, dim=2, dtype=ttnn.float32)  # inclusive count of routed tokens up to each token
    mask_f32.deallocate(True)
    rank = ttnn.subtract(csum, 1.0)  # position of a routed token inside its expert's list

    capacity = _capacity(n, capacity, lambda: _host_max(ttnn.max(csum, dim=2, keepdim=True), local_routing.device()))
    csum.deallocate(True)

    # dispatch: a stable descending sort of the mask puts every expert's routed tokens first, in token order
    mask_t = ttnn.permute(mask, (0, 1, 3, 2))  # [1, 1, E_loc, n]
    mask.deallocate(True)
    sorted_mask, sorted_ids = ttnn.sort(mask_t, dim=-1, descending=True, stable=True)
    mask_t.deallocate(True)
    if capacity < n:
        sorted_mask = ttnn.slice(sorted_mask, [0, 0, 0, 0], [1, 1, num_local, capacity])
        sorted_ids = ttnn.slice(sorted_ids, [0, 0, 0, 0], [1, 1, num_local, capacity])
    valid = ttnn.typecast(sorted_mask, ttnn.float32)
    ids = ttnn.typecast(sorted_ids, ttnn.float32)  # token ids < 2**24: exact
    sorted_mask.deallocate(True)
    sorted_ids.deallocate(True)
    dispatch = ttnn.typecast(ttnn.multiply(ids, valid), ttnn.uint32)
    valid.deallocate(True)
    ids.deallocate(True)
    dispatch = ttnn.reshape(ttnn.to_layout(dispatch, ttnn.ROW_MAJOR_LAYOUT), (1, num_local * capacity))

    # combine: each token's routed experts, their in-expert row and weight
    top_w, top_e = ttnn.topk(local_routing, kl, dim=-1)  # [1, 1, n, KL]
    slot_rank = ttnn.gather(rank, -1, top_e)  # in-expert row of the token, per slot
    rank.deallocate(True)
    routed = ttnn.typecast(ttnn.gt(top_w, 0.0), ttnn.float32)
    fits = ttnn.typecast(ttnn.lt(slot_rank, float(capacity)), ttnn.float32)
    keep = ttnn.multiply(routed, fits)  # 1 for slots that are routed and inside the capacity
    routed.deallocate(True)
    fits.deallocate(True)
    row = ttnn.add(ttnn.multiply(ttnn.typecast(top_e, ttnn.float32), float(capacity)), slot_rank)
    slot_rank.deallocate(True)
    top_e.deallocate(True)
    combine = ttnn.typecast(ttnn.multiply(row, keep), ttnn.uint32)
    row.deallocate(True)
    combine = ttnn.permute(combine, (0, 1, 3, 2))  # [1, 1, KL, n]: slot-major
    combine = ttnn.reshape(ttnn.to_layout(combine, ttnn.ROW_MAJOR_LAYOUT), (1, kl * n))
    slot_weights = ttnn.multiply(top_w, ttnn.typecast(keep, top_w.dtype))
    top_w.deallocate(True)
    keep.deallocate(True)
    slot_weights = ttnn.permute(slot_weights, (0, 3, 2, 1))  # [1, KL, n, 1]
    return {
        "capacity": capacity,
        "dispatch": dispatch,
        "combine": combine,
        "slot_weights": slot_weights,
        "kl": kl,
    }


def _routed_chunk(hidden_states, local_routing, weights, config, compute_kernel_config, capacity):
    """One chunk: hidden [1,1,n,H], local routing [1,1,n,E_loc] -> [1,1,n,H] (this chip's partial sum)."""
    n = int(hidden_states.shape[2])
    hidden = config.hidden_size
    num_local = weights.num_local_experts
    intermediate = weights.intermediate_size_per_device
    meta = build_routing_metadata(local_routing, num_local, config.top_k, capacity)
    cap, kl = meta["capacity"], meta["kl"]

    x_rm = ttnn.reshape(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT), (n, hidden))
    xs = ttnn.embedding(meta["dispatch"], x_rm, layout=ttnn.TILE_LAYOUT)  # [1, E_loc * C, H]
    meta["dispatch"].deallocate(True)
    xs = ttnn.reshape(xs, (1, num_local, cap, hidden))

    up_gate = ttnn.matmul(
        xs,
        weights.gate_up_proj,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    xs.deallocate(True)
    act = apply_geglu_fused(up_gate)
    up_gate.deallocate(True)
    act = ttnn.reshape(act, (1, num_local, cap, intermediate))
    down = ttnn.matmul(
        act,
        weights.down_proj,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    act.deallocate(True)

    down_rm = ttnn.to_layout(ttnn.reshape(down, (1, 1, num_local * cap, hidden)), ttnn.ROW_MAJOR_LAYOUT)
    down_rm = ttnn.reshape(down_rm, (num_local * cap, hidden))
    gathered = ttnn.embedding(meta["combine"], down_rm, layout=ttnn.TILE_LAYOUT)  # [1, KL * n, H]
    meta["combine"].deallocate(True)
    down_rm.deallocate(True)
    gathered = ttnn.reshape(gathered, (1, kl, n, hidden))
    gathered = ttnn.multiply(gathered, meta["slot_weights"])
    meta["slot_weights"].deallocate(True)
    if kl == 1:
        return ttnn.reshape(gathered, (1, 1, n, hidden))
    out = ttnn.experimental.fast_reduce_nc(gathered, dims=[1])
    gathered.deallocate(True)
    return ttnn.reshape(out, (1, 1, n, hidden))


def routed_prefill_forward(hidden_states, local_routing, weights, config, routed_prefill=None, prefill_traced=False):
    """Routed expert forward for a prefill call: hidden [1,1,S,H], this chip's routing [1,1,S,E_loc] ->
    [1,1,S,H] partial sum over this chip's experts (the caller combines across chips)."""
    seq_len = int(hidden_states.shape[2])
    device = hidden_states.device()
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    routed = resolve_routed_config(routed_prefill)
    chunk = max(TILE_SIZE, routed.chunk_size // TILE_SIZE * TILE_SIZE)
    capacity = routed.capacity if routed.capacity is not None else ("full" if prefill_traced else "exact")
    if seq_len <= chunk:
        return _routed_chunk(hidden_states, local_routing, weights, config, compute_kernel_config, capacity)
    hidden_chunks = ttnn.split(hidden_states, chunk, dim=2)
    routing_chunks = ttnn.split(local_routing, chunk, dim=2)
    results = []
    for h_chunk, r_chunk in zip(hidden_chunks, routing_chunks):
        results.append(_routed_chunk(h_chunk, r_chunk, weights, config, compute_kernel_config, capacity))
        h_chunk.deallocate(True)
        r_chunk.deallocate(True)
    out = ttnn.concat(results, dim=2)
    for r in results:
        r.deallocate(True)
    return out
