# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE for DeepSeek-V4-Flash: one block, many tokens per call.

:class:`DeepSeekV4PrefillMoE` is the multi-token counterpart of the decode
:class:`~..decode.moe.DeepSeekV4SparseMoeBlock`. It consumes a chunk of ``T`` hidden states
``[1, 1, T, D]`` and returns ``routed(x) + shared(x)`` in the same shape, the math of the reference
``DeepseekV4SparseMoeBlock`` (``modular_deepseek_v4.py``)::

    scores  = sqrt(softplus(x @ Wgate))                       # per-expert sqrtsoftplus scores
    select  = top_k(scores + e_score_correction_bias)         # learned layers
              tid2eid[token_id]                               # hash layers (first num_hash_layers)
    weights = routed_scaling_factor * scores[select] / (sum(scores[select]) + eps)
    routed  = sum_{e in select} weights_e * down_e(clamped_swiglu(gate_up_e(x)))
    shared  = down(silu(gate(x)) * up(x))                     # one always-on dense expert
    out     = routed + shared                                 # (+ TP all-reduce)

The pieces, per call:

* **Router** (plain ``ttnn`` ops on the whole chunk): the gate matmul, ``sqrt(softplus(.))`` and, for
  learned layers, the bias-corrected ranking row; hash layers gather the selected ids from the frozen
  ``tid2eid`` table with ``ttnn.embedding``. Nothing is read back to the host. The result is the same
  :class:`~..decode.moe.SparseRouting` contract the decode ``fused_experts`` op consumes.
* **Routed experts**: the custom ``ttnn.experimental.deepseek_prefill.fused_experts_prefill`` op. It
  reads the *decode* expert weights (the ND-sharded bf4 tensors of
  :class:`~..decode.moe.DeepSeekV4PreloadedExperts`) in place, so prefill and decode share one copy of
  the ~20 GB of routed weights. On device it computes the routing weights, gathers every expert's
  token rows, tilizes them, runs the clamped-SwiGLU FFN and returns the routing-weighted sum over each
  token's ``top_k`` experts.
* **Shared expert**: the decode :class:`~..decode.moe.DeepSeekV4MLP` with plain (non-prefetcher)
  projections, run on the whole chunk. Like the decode block it applies no swiglu clamp.
* **Combine**: ``routed + shared``, then under tensor parallelism one all-reduce over the TP axis
  (the routed op and the shared expert both emit row-parallel down-projection partials).

Supported shapes:

* ``T`` a multiple of 32. The routed op handles up to ``MAX_OP_TOKENS`` (512) tokens per call, so a
  longer chunk is fed to it in slices of that size; the router and the shared expert see all ``T``
  rows at once.
* The routed op is built for the decode topology: ``D == 4096`` and a per-chip ``I_local == 512``
  (``I = 2048`` under TP=4), up to 256 experts, ``top_k <= 16``. ``experts`` must be the decode
  :class:`~..decode.moe.DeepSeekV4PreloadedExperts` built for the same ``tp_size``.
"""

from typing import Optional

import torch

import ttnn

from ..common import DeepSeekV4Module
from ..decode.moe import (
    DeepSeekV4HashRouter,
    DeepSeekV4MLP,
    DeepSeekV4PreloadedExperts,
    DeepSeekV4TopKRouter,
    SparseRouting,
    _tp_all_reduce,
)
from ..weight_cache import WeightCache, _as_cache

# Most token rows one ``fused_experts_prefill`` call takes (the op keeps every token's routing entry in
# L1: ``kMaxTokens`` in ``fused_experts_prefill_types.hpp``).
MAX_OP_TOKENS = 512
# Token rows are tile aligned everywhere in this module.
ALIGNMENT = 32


def _slice_rows(tensor: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    """Rows ``[start, end)`` of a ``[1, 1, T, X]`` tensor (tile aligned bounds on a TILE tensor)."""
    return ttnn.slice(tensor, [0, 0, start, 0], [1, 1, end, tensor.shape[-1]])


class DeepSeekV4PrefillMoE(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4SparseMoeBlock`` (see the module docstring).

    ``weights`` is the same HF-named dict the decode block takes, each value a torch tensor (or a
    thunk returning one), ``nn.Linear`` layout ``[out, in]``::

        gate.weight  gate.e_score_correction_bias                      # every layer
        gate.tid2eid                                                   # hash layers only
        shared_experts.gate_proj.weight  shared_experts.up_proj.weight  shared_experts.down_proj.weight

    ``experts`` is the layer's :class:`~..decode.moe.DeepSeekV4PreloadedExperts` (decode already owns
    it; prefill reads its weights without copying them). ``gate`` may be injected -- a
    :class:`~..decode.moe.DeepSeekV4HashRouter` for the first ``num_hash_layers`` layers -- otherwise
    the learned :class:`~..decode.moe.DeepSeekV4TopKRouter` is built from ``weights``. Only those
    routers' *weights* are used (their gate projection, the correction bias and the hash table);
    the decode routers' activation contract is decode shaped, so the routing math is redone here for
    ``[1, 1, T, D]`` TILE activations.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        experts: DeepSeekV4PreloadedExperts,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
    ):
        """Build the router weights and the shared expert; the routed weights come from ``experts``."""
        if experts.tp_size != tp_size:
            raise ValueError(f"experts were built for tp_size={experts.tp_size}, this block for tp_size={tp_size}")
        self.device = device
        self.hidden = config.hidden_size
        self.tp_size = tp_size
        self.experts = experts
        self.num_experts = config.num_local_experts
        self.top_k = experts.top_k
        cache = _as_cache(cache)

        self.gate = (
            gate
            if gate is not None
            else DeepSeekV4TopKRouter(config, weights, device, cache=cache, weight_dtype=weight_dtype)
        )
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        if self.gate.num_experts != len(experts.gate_up_weights):
            raise ValueError(
                f"router has {self.gate.num_experts} experts but the routed weights hold {len(experts.gate_up_weights)}"
            )
        # The ranking row is scores + bias on [1, 1, T, E] TILE scores, so the bias is a TILE row too
        # (the decode router keeps it ROW_MAJOR for its ROW_MAJOR score rows).
        # Hash layers select by table lookup and have no bias.
        self._bias_tile = None if self.is_hash else ttnn.to_layout(self.gate.e_score_correction_bias, ttnn.TILE_LAYOUT)

        self.shared_experts = DeepSeekV4MLP(
            weights,
            "shared_experts",
            device,
            cache=cache,
            weight_dtype=weight_dtype,
            config=config,
            use_prefetcher=False,
            tp_size=tp_size,
        )

    # ------------------------------------------------------------------ router
    def _token_ids(self, token_ids, t: int) -> ttnn.Tensor:
        """The token ids of a hash layer as a ``[1, T]`` uint32 ROW_MAJOR device tensor."""
        if isinstance(token_ids, ttnn.Tensor):
            return token_ids
        ids = torch.as_tensor(token_ids).reshape(1, t).to(torch.int32)
        return ttnn.from_torch(
            ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.tp_size > 1 else None,
        )

    def route(self, x_tile: ttnn.Tensor, token_ids=None) -> SparseRouting:
        """``x_tile`` ``[1, 1, T, D]`` TILE -> the routing decision for all ``T`` tokens.

        ``scores`` are the unbiased ``sqrtsoftplus`` scores (they become the routing weights);
        learned layers add the bias-corrected ``ranking`` row (the op takes the top-k on device),
        hash layers the ``indices`` looked up from ``tid2eid[token_ids]``.
        """
        t = x_tile.shape[2]
        logits = self.gate.gate(x_tile)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        ttnn.deallocate(logits)
        if self.is_hash:
            if token_ids is None:
                raise ValueError("a hash-routed layer needs the token ids")
            ids = ttnn.embedding(self._token_ids(token_ids, t), self.gate.eid_table, layout=ttnn.TILE_LAYOUT)
            return SparseRouting(scores=scores, indices=ttnn.reshape(ids, [1, 1, t, self.top_k]))
        return SparseRouting(scores=scores, ranking=ttnn.add(scores, self._bias_tile))

    # ------------------------------------------------------------------ routed experts
    def _routed(self, x_rm: ttnn.Tensor, routing: SparseRouting) -> ttnn.Tensor:
        """The routed experts' weighted sum ``[1, 1, T, D]`` (TILE), in slices of ``MAX_OP_TOKENS``."""
        t = x_rm.shape[2]
        outs = []
        for start in range(0, t, MAX_OP_TOKENS):
            end = min(t, start + MAX_OP_TOKENS)
            whole = start == 0 and end == t

            def part(tensor):
                return None if tensor is None else (tensor if whole else _slice_rows(tensor, start, end))

            out = ttnn.experimental.deepseek_prefill.fused_experts_prefill(
                part(x_rm),
                part(routing.scores),
                self.experts.gate_up_weights,
                self.experts.down_weights,
                self.experts.intermediate,
                self.experts.limit,
                self.top_k,
                self.experts.routed_scaling_factor,
                self.experts.routing_eps,
                routing_indices=part(routing.indices),
                ranking_scores=part(routing.ranking),
            )
            outs.append(out)
        if len(outs) == 1:
            return outs[0]
        combined = ttnn.concat(outs, dim=2)
        for out in outs:
            ttnn.deallocate(out)
        return combined

    # ------------------------------------------------------------------ block
    def forward(self, hidden: ttnn.Tensor, token_ids=None) -> ttnn.Tensor:
        """``hidden`` ``[1, 1, T, D]`` (TILE or ROW_MAJOR, bf16) -> ``[1, 1, T, D]`` TILE.

        ``token_ids`` (``[T]`` / ``[1, T]`` torch ints, or a ``[1, T]`` uint32 ROW_MAJOR device
        tensor) is required for hash layers and ignored otherwise. Under TP the result is the
        all-reduced, replicated residual.
        """
        shape = tuple(hidden.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[3] != self.hidden:
            raise ValueError(f"hidden must be [1, 1, T, {self.hidden}], got {shape}")
        t = shape[2]
        if t % ALIGNMENT:
            raise ValueError(f"T={t} must be a multiple of {ALIGNMENT}")

        x_tile = hidden if hidden.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(hidden, ttnn.TILE_LAYOUT)
        x_rm = hidden if hidden.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.to_memory_config(x_rm, ttnn.DRAM_MEMORY_CONFIG)  # the op gathers rows from DRAM

        routing = self.route(x_tile, token_ids)
        routed = self._routed(x_rm, routing)
        shared = self.shared_experts(x_tile)
        combined = ttnn.add(routed, shared)
        for tensor in (routed, shared, routing.scores, routing.ranking, routing.indices):
            if tensor is not None:
                ttnn.deallocate(tensor)
        if x_tile is not hidden:
            ttnn.deallocate(x_tile)
        if x_rm is not hidden:
            ttnn.deallocate(x_rm)

        combined = ttnn.to_memory_config(combined, ttnn.DRAM_MEMORY_CONFIG)
        if self.tp_size > 1:
            combined = _tp_all_reduce(combined, self.device)
        return combined
