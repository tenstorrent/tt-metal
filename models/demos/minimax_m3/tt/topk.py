# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 MoE router (gate).

Routing:
  * sigmoid scoring over all experts (the gate Linear has no bias)
  * a separate ``e_score_correction_bias`` is added to the sigmoid scores FOR
    SELECTION ONLY (picks which experts win, not the returned weights)
  * the returned top-k weights are the UNBIASED sigmoid values gathered at the
    selected indices, then normalized to sum to 1

HF reference (MiniMaxM3SparseMoeBlock.route_tokens_to_experts):
    routing_weights = sigmoid(router_logits.float())
    scores_for_choice = routing_weights + e_score_correction_bias
    _, top_k_index = topk(scores_for_choice, top_k)
    top_k_weights = routing_weights.gather(1, top_k_index)
    top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

Two implementations of that rule, selected by `use_fused_gate()`:

  * FUSED (default) — one `deepseek_prefill.moe_grouped_topk` call. The op's single-group path
    (n_groups=1) IS this rule: activation -> +bias -> top-k -> gather the UNBIASED activation output
    -> normalize -> * route_scale, all in fp32 internally with bf16 only at the inputs.
  * LEGACY (M3_FUSED_GATE=0) — the original 8-op chain, kept for A/B because the two do NOT select
    the same experts. See use_fused_gate() for the measured difference.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.utils.general_utils import cache_file_exists, get_cache_file_name

# The fused path is the DEFAULT because it is both faster and closer to the reference, which is worth
# spelling out since it does change routing. Measured against a torch reference at M3's exact shape
# (640 tokens, E=128, top-4, bf16 inputs; tests/unit/test_fused_gate_vs_ref.py):
#
#   expert-set agreement with an fp32 reference    fused 99.4%      legacy 96.4%
#   expert-set agreement between the two paths                96.1%
#
# So ~4 % of tokens change their selected expert set on the switch, but the move is TOWARD the
# reference: the legacy chain runs sigmoid AND the bias add in bf16, whose ~0.004 resolution around
# O(1) scores makes near-ties at the top-k boundary resolve arbitrarily across 128 experts. The op
# computes in fp32 and disagrees with the reference on only the genuinely tied ~0.6 %.
#
# Expected to help KV PCC (the harness compares against HF fp32) — but "more faithful" is not the same
# as "better end-to-end", so re-measure V and keep M3_FUSED_GATE=0 available to bisect if it drops.
DEFAULT_USE_FUSED_GATE = True


def use_fused_gate() -> bool:
    """True -> single fused moe_grouped_topk; False -> the legacy bf16 op chain. See above."""
    return os.environ.get("M3_FUSED_GATE", "1" if DEFAULT_USE_FUSED_GATE else "0") not in ("0", "false", "False")


def route_tokens_to_experts_fused(
    router_logits, experts_per_token, score_bias_wide, routed_scaling_factor, padding_config=None
):
    """MiniMax-M3 routing in ONE device op — the ~8-op chain below collapses to this.

    `moe_grouped_topk`'s single-group path (n_groups=1) IS M3's rule: activation (sigmoid) -> + bias ->
    top-k -> gather the UNBIASED activation output at the selected indices -> normalize -> * route_scale.
    The DeepSeek-specific constraints (n_groups==8, experts==256, n_activated_experts==8) live behind the
    op's GROUPED branch; the single-group branch requires only a tile-aligned expert count and k <= 64,
    and 128/4 both pass. `summed_experts_per_group` / `topk_groups` are unread here, so 1 is honest.

    Reach for `deepseek_prefill.moe_grouped_topk`, NOT the older `ttnn.experimental.deepseek_grouped_gate`
    that TtMoEGatePrefill itself calls — that one asserts all three DeepSeek constants unconditionally
    and M3 fails every one. Easy to hit and it makes this look like a kernel change.

    score_bias_wide must be the full [tokens, num_experts] broadcast: the op requires
    bias.logical_shape() == scores.logical_shape(). TopKRouter._wide_bias builds and caches it.

    ``padding_config`` (ROW_MAJOR UINT32 per-device ``[num_real_tokens, pad_side]``) makes the op
    SENTINEL-MARK the padded rows so they route nowhere. It must be the SAME tensor the dispatch op
    gets: the two are only consistent together — the gate marks the pad rows and dispatch shortens its
    token loop to match. Marking without shortening (or the reverse) is worse than doing neither, which
    is why DeepSeek gates the whole thing on its gate mode (tt/moe/tt_moe.py:530). None => every row is
    treated as real: correct, but it does the padded work.

    Returns (indices UINT16 TILE, weights BFLOAT16 TILE). TILE indices are what masked_bincount
    consumes natively, so nothing has to untilize them for the routing setup.
    """
    weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        router_logits,
        score_bias_wide,
        n_groups=1,
        summed_experts_per_group=1,
        topk_groups=1,
        n_activated_experts=experts_per_token,
        route_scale=routed_scaling_factor,
        epsilon=1e-20,
        padding_config=padding_config,
    )
    return indices, weights


def route_tokens_to_experts(
    router_logits, experts_per_token, num_experts, score_bias, use_throughput_experts, routed_scaling_factor=1.0
):
    """Apply MiniMax-M3 routing to gate logits [tokens, num_experts] — legacy multi-op path."""
    if router_logits.dtype != ttnn.bfloat16:
        router_logits = ttnn.typecast(router_logits, dtype=ttnn.bfloat16)

    # Sigmoid scores over all experts.
    routing_weights = ttnn.sigmoid(router_logits)

    # Selection scores = sigmoid + correction bias (bias affects WHICH experts are
    # chosen, not the returned weights).
    if score_bias is not None:
        scores_for_choice = ttnn.add(routing_weights, score_bias)
    else:
        scores_for_choice = routing_weights

    _, expert_indices = ttnn.topk(scores_for_choice, k=experts_per_token, dim=-1, sorted=True)
    if scores_for_choice is not routing_weights:
        scores_for_choice.deallocate(True)

    # Gather the UNBIASED sigmoid weights at the selected indices, normalize to sum 1.
    top_k_weights = ttnn.gather(routing_weights, dim=-1, index=expert_indices)
    denom = ttnn.sum(top_k_weights, dim=-1, keepdim=True)
    top_k_weights = ttnn.div(top_k_weights, denom)
    denom.deallocate(True)

    # M3: scale the routed weights by routed_scaling_factor (2.0), applied AFTER normalize
    # (DeepSeek/M3 convention; shared expert is added later, unscaled). A factor of 1.0 -> no-op.
    if routed_scaling_factor != 1.0:
        top_k_weights = ttnn.mul(top_k_weights, routed_scaling_factor)

    if use_throughput_experts:
        routing_weights.deallocate(True)
        return expert_indices, top_k_weights

    # Dense [tokens, num_experts] weights (normalized value at selected experts, 0 elsewhere).
    dense_weights = ttnn.scatter(ttnn.zeros_like(routing_weights), dim=1, index=expert_indices, src=top_k_weights)
    routing_weights.deallocate(True)
    top_k_weights.deallocate(True)
    return expert_indices, dense_weights


class TopKRouter:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, num_tokens=None):
        """num_tokens: tokens per device per forward (chunk_size // sp_factor). Required for the fused
        gate, whose bias must be a full [num_tokens, num_experts] tensor; None keeps the legacy path."""
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.num_tokens = num_tokens
        # M3: routed-expert output is scaled by routed_scaling_factor (2.0, from config; 1.0 if absent).
        self.routed_scaling_factor = getattr(hf_config, "routed_scaling_factor", 1.0)
        self.tensor_cache_path = tensor_cache_path

        # MiniMax-M3 gate Linear has no bias; weight is [num_experts, hidden] -> [hidden, num_experts].
        torch_weight = state_dict["weight"].transpose(0, 1) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # e_score_correction_bias [num_experts] -> [1, num_experts], replicated; added to
        # selection scores only. Absent in some checkpoints -> no correction.
        score_bias_torch = None
        if state_dict and "e_score_correction_bias" in state_dict:
            score_bias_torch = state_dict["e_score_correction_bias"].reshape(1, -1)
        bias_cache_file = get_cache_file_name(tensor_cache_path, "e_score_correction_bias")
        # Build the bias tensor when we have the source weight, OR (cache-only loading, empty
        # state_dict) when it was previously cached — torch=None then loads it straight from disk.
        # Whether the checkpoint HAS a correction bias can't be known without the source, so the
        # cached file's existence is the signal.
        build_bias = score_bias_torch is not None or (not state_dict and cache_file_exists(bias_cache_file))
        self.score_bias = (
            ttnn.as_tensor(
                score_bias_torch,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=bias_cache_file,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh_device) if isinstance(mesh_device, ttnn.MeshDevice) else None
                ),
            )
            if build_bias
            else None
        )

        # The fused gate needs the bias at the FULL [num_tokens, num_experts] shape: the op asserts
        # bias.logical_shape() == scores.logical_shape(). Built ONCE here, never per call, and never via
        # a host round-trip — one 640x128 bf16 tensor (~164 KiB) per layer, constant for the whole run.
        #
        # It is derived on DEVICE with a single ttnn.repeat off the narrow bias rather than expanded on
        # the host, because the production path is CACHE-ONLY: the tilized weight cache is complete, so
        # state_dict is empty and score_bias_torch is None. An earlier version built this from the torch
        # source only, which meant the fused gate was silently disabled on exactly the path that matters
        # — a full-model KV-PCC run came back digit-for-digit identical to the legacy baseline, which
        # looked like "the change is accuracy-neutral" and was really "the change never ran".
        self.score_bias_wide = None
        if build_bias and num_tokens and self.score_bias is not None:
            self.score_bias_wide = ttnn.repeat(self.score_bias, ttnn.Shape([num_tokens, 1]))

        # Padding-config memo, keyed by real-token count. See build_padding_config.
        self.mesh_device = mesh_device
        self._padding_config_cache = {}

        # Custom compute configs can degrade routing quality; keep the default.
        self.compute_config = None
        # One log line per router recording which gate path actually ran (see __call__).
        self._gate_path_logged = False

    def build_padding_config(self, actual_isl):
        """Per-device ``[num_real_tokens, pad_side]`` for a chunk with ``actual_isl`` real tokens, or
        None when the chunk is full (nothing to mark).

        The SAME tensor must go to the gate topk AND the dispatch op — see
        route_tokens_to_experts_fused. Owned here and memoized per real-token count, so a chunked
        prefill builds each distinct config once instead of per chunk.

        M3's SP sharding within a chunk is CONTIGUOUS and right-padded (tt/attention/msa.py: "SP
        sharding is CONTIGUOUS, no zigzag/balancing"), so chip c holds chunk tokens
        [c*tokens_per_chip, (c+1)*tokens_per_chip) and its real count is the clamped remainder. That is
        DeepSeek's non-rotated branch; M3 needs neither its zigzag nor its rotated-chunk case.

        NOTE this ends in a ttnn.from_torch, i.e. a host->device write. Fine untraced — it is memoized,
        so it costs one tiny [sp, 2] write per distinct count — but a traced runtime must derive the
        row on-device instead (DeepSeek's build_padding_config_device / the moe_padding_config op).
        Port that alongside any trace work.
        """
        tokens_per_chip = self.num_tokens
        sp_factor = self.mesh_device.shape[0] if isinstance(self.mesh_device, ttnn.MeshDevice) else 1
        if actual_isl is None or not tokens_per_chip or actual_isl >= sp_factor * tokens_per_chip:
            return None  # full chunk: every row is real
        if actual_isl not in self._padding_config_cache:
            rows = torch.zeros((sp_factor, 2), dtype=torch.int32)
            for c in range(sp_factor):
                rows[c, 0] = max(0, min(tokens_per_chip, actual_isl - c * tokens_per_chip))
                rows[c, 1] = 0  # right padding
            self._padding_config_cache[actual_isl] = ttnn.from_torch(
                rows,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape),
            )
            logger.info(
                f"[TopKRouter] padding config built for actual_isl={actual_isl} "
                f"({sp_factor} x {tokens_per_chip} tokens): per-chip real counts "
                f"{[int(rows[c, 0]) for c in range(sp_factor)]}"
            )
        return self._padding_config_cache[actual_isl]

    def __call__(self, hidden_states, use_throughput_experts, padding_config=None):
        # Actual token count from volume (shape[0] after reshape is tile-padded).
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # L1 for decode (small), DRAM for prefill (large sequences).
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,  # no bias (MiniMax-M3)
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        # Fused single-op gate when it is available for this shape: needs the wide bias (built at init
        # for self.num_tokens) and the throughput-expert output form (indices + top-k weights, which is
        # the EP path; the dense [tokens, E] scatter form the non-EP path wants is not what the op emits).
        if use_fused_gate() and use_throughput_experts:
            usable = self.score_bias_wide is not None and actual_tokens == self.num_tokens
            if usable:
                if not self._gate_path_logged:
                    self._gate_path_logged = True
                    logger.info(f"[TopKRouter] routing via the FUSED moe_grouped_topk gate ({actual_tokens} tokens)")
                expert_indices, expert_weights = route_tokens_to_experts_fused(
                    router_logits, self.top_k, self.score_bias_wide, self.routed_scaling_factor, padding_config
                )
                ttnn.deallocate(router_logits)
                return expert_indices, expert_weights
            # Never fall through quietly: a silently-disabled fused gate is indistinguishable from a
            # working one in every metric except the op count, which is how it went unnoticed once.
            if not self._gate_path_logged:
                self._gate_path_logged = True
                reason = (
                    f"no wide bias (score_bias={'present' if self.score_bias is not None else 'None'}, "
                    f"num_tokens={self.num_tokens})"
                    if self.score_bias_wide is None
                    else f"token count {actual_tokens} != the {self.num_tokens} the wide bias was built for"
                )
                logger.warning(
                    f"[TopKRouter] fused gate REQUESTED but UNAVAILABLE ({reason}); falling back to the "
                    f"legacy multi-op chain. Perf and routing both differ from the fused path."
                )
        elif not self._gate_path_logged:
            self._gate_path_logged = True
            logger.info("[TopKRouter] routing via the LEGACY multi-op gate chain")

        expert_indices, expert_weights = route_tokens_to_experts(
            router_logits,
            self.top_k,
            self.num_experts,
            self.score_bias,
            use_throughput_experts,
            self.routed_scaling_factor,
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights
