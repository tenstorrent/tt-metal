# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Mixture-of-Experts module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

`MoE1D` factors the **router + routed-experts** compute shared by the Gemma4 and GPT-OSS MoE
blocks into one generalizable, topology-aware module. Both references implement the same dense-
routing `sparse_matmul` expert pipeline; they differ only in the router math and the expert
activation / bias. Those differences are absorbed into construction-time **strategy enums**
(`RoutingStrategy`, `ExpertActivation`) and **optional config fields** — never into `if`-on-config
branches in the hot path (Zen #2).

Execution paths (a straight line of compute per mode):
  Router (shared):  [rms_norm] -> [*scale] -> [*scalar] -> linear(+bias) -> <routing_strategy>
                    -> dense_routing [1,1,S,E] -> [*per_expert_scale]
  Experts decode:   sparse_matmul(gate) (+bias) -> sparse_matmul(up) (+bias) -> <activation>
                    -> sparse_matmul(down, is_input_a_sparse) (+bias) -> mul(routing) -> sum(E)
                    -> [all_reduce if >1 device]
  Experts prefill:  per 32-token group: sparse_matmul(gate/up, all-ones sparsity) (+bias)
                    -> <activation> -> sparse_matmul(down, is_input_a_sparse) (+bias)
                    -> mul(routing) -> fast_reduce_nc(E) -> concat chunks -> [all_reduce]

`forward(router_input, expert_input, mode)` takes **two** inputs: the router and the experts may be
fed different tensors (Gemma4 routes the pre-norm residual while the experts see a separately-normed
input that the *caller* owns). GPT-OSS passes the same tensor for both. The router's *own* pre-linear
normalize/scale (internal to Gemma4's router) is owned here as optional config; everything the
reference layer keeps outside the MoE block (expert-input norm, post norms, shared MLP, residual
combine, layer scalar) stays the caller's responsibility, as does all weight unfusing — weights enter
as already-shaped LazyWeights (`gate/up [1,E,H,I]`, `down [1,E,I,H]`, optional biases, router `[H,E]`).

See `dev-tools/agentic-tttv2/...` MODULE_REFERENCES for the full cross-reference spec.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl
from models.tt_transformers.tt.common import Mode

TILE_SIZE = 32

# =============================================================================
# Strategy enums (bound to callables at construction — Zen #2)
# =============================================================================


class RoutingStrategy(Enum):
    """How the router turns linear logits into dense per-expert weights.

    - SOFTMAX_TOPK_SUMNORM: softmax over ALL experts -> top-k -> divide by the top-k sum
      (linear renormalization, NOT a second softmax). Gemma4.
    - TOPK_SOFTMAX: top-k over raw logits -> softmax over the k selected values. GPT-OSS.

    Both scatter the per-token weights into a dense [1,1,S,E] tensor (0 at unselected experts).
    """

    SOFTMAX_TOPK_SUMNORM = "softmax_topk_sumnorm"
    TOPK_SOFTMAX = "topk_softmax"


class ExpertActivation(Enum):
    """Gated activation applied to (gate, up) before the down projection.

    - GEGLU: gelu(gate, tanh-approx) * up. Gemma4 (gelu_pytorch_tanh).
    - SWIGLU_CLAMP: clamp(gate, max=limit); clamp(up, ±limit); (up+1) * (gate*sigmoid(alpha*gate)).
      GPT-OSS (needs `swiglu_limit`, `swiglu_alpha`).
    """

    GEGLU = "geglu"
    SWIGLU_CLAMP = "swiglu_clamp"


# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass
class MoE1DConfig:
    """Single source of truth for MoE1D.

    Simple usage (Gemma4-style defaults: GeGLU + softmax->topk->sum-norm, no bias):
        config = MoE1DConfig(gate_proj, up_proj, down_proj, router_weight, top_k=8)

    GPT-OSS usage (override the strategy knobs + biases):
        config = MoE1DConfig(
            gate_proj, up_proj, down_proj, router_weight, top_k=4,
            routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
            activation_strategy=ExpertActivation.SWIGLU_CLAMP,
            swiglu_limit=7.0, swiglu_alpha=1.702,
            router_bias=router_bias, gate_bias=gb, up_bias=ub, down_bias=db,
        )
    """

    # --- Required weights ---
    gate_proj: LazyWeight  # [1, E, H, I] column-parallel
    up_proj: LazyWeight  # [1, E, H, I] column-parallel
    down_proj: LazyWeight  # [1, E, I, H] row-parallel
    router_weight: LazyWeight  # [..., H, E] (ttnn linear layout)

    # --- Required scalar ---
    top_k: int | None = None  # experts per token; cannot be derived from weights

    # --- Strategy selection (bound to callables in _resolve) ---
    routing_strategy: RoutingStrategy = RoutingStrategy.SOFTMAX_TOPK_SUMNORM
    activation_strategy: ExpertActivation = ExpertActivation.GEGLU

    # --- Derived dims (from weights if None) ---
    num_experts: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None

    # --- Device / collectives ---
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None  # None = auto-detect (single device: stays None)
    num_links: int = 1

    # --- Optional router pre-linear transforms (None -> step skipped) ---
    router_prenorm_eps: float | None = None  # Gemma4: rms_norm_eps; GPT-OSS: None
    router_scale: LazyWeight | None = None  # Gemma4 [1,1,1,H]; GPT-OSS None
    router_input_scalar: float | None = None  # Gemma4 hidden**-0.5; GPT-OSS None
    router_bias: LazyWeight | None = None  # GPT-OSS [..,E]; Gemma4 None
    per_expert_scale: LazyWeight | None = None  # Gemma4 [1,1,1,E]; GPT-OSS None

    # --- Optional expert biases (None -> no bias add) ---
    gate_bias: LazyWeight | None = None  # [1, E, I]
    up_bias: LazyWeight | None = None  # [1, E, I]
    down_bias: LazyWeight | None = None  # [1, E, H]

    # --- Activation params (SWIGLU_CLAMP only) ---
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None

    # --- dtypes ---
    expert_weight_dtype: ttnn.DataType | None = None  # Gemma4 bf8; GPT-OSS bf4
    router_weight_dtype: ttnn.DataType | None = None
    bias_dtype: ttnn.DataType | None = None
    sparse_matmul_dtype: ttnn.DataType | None = None  # Gemma4 bf16; GPT-OSS bf8
    routing_dtype: ttnn.DataType | None = None  # dense routing weights dtype

    # --- sparse_matmul program configs (overridable callables: (m, n, k) -> prg_config) ---
    decode_gate_up_prg_config: Callable[[int, int, int], object] | None = None
    decode_down_prg_config: Callable[[int, int, int], object] | None = None
    prefill_gate_up_prg_config: Callable[[int, int, int], object] | None = None
    prefill_down_prg_config: Callable[[int, int, int], object] | None = None

    # --- sparse_matmul tuning knobs used by the default program-config builder ---
    gate_up_in0_block_w: int = 1  # autoport Gemma uses 8
    down_in0_block_w: int = 1  # autoport Gemma uses 3

    # --- memory configs ---
    decode_sparse_memcfg: ttnn.MemoryConfig | None = None  # default L1
    prefill_sparse_memcfg: ttnn.MemoryConfig | None = None  # default DRAM
    output_memcfg: ttnn.MemoryConfig | None = None  # default DRAM

    # --- prefill chunking ---
    prefill_chunk_size: int = TILE_SIZE  # group dim = chunk/32; 32 keeps groups=1 (always fits)

    # Fields that are *legitimately* optional (None is a valid resolved value).
    _OPTIONAL_FIELDS = (
        "router_prenorm_eps",
        "router_scale",
        "router_input_scalar",
        "router_bias",
        "per_expert_scale",
        "gate_bias",
        "up_bias",
        "down_bias",
        "swiglu_limit",
        "swiglu_alpha",
        "decode_sparse_memcfg",
        "prefill_sparse_memcfg",
    )

    def is_resolved(self) -> bool:
        """All non-optional fields must be filled before device work."""
        optional = set(self._OPTIONAL_FIELDS)
        # topology stays None on a single device (no CCL needed).
        if self.mesh_device is not None and self.mesh_device.get_num_devices() == 1:
            optional.add("topology")
        return all(
            getattr(self, f) is not None
            for f in self.__dataclass_fields__
            if f not in optional and not f.startswith("_")
        )


# =============================================================================
# MoE1D
# =============================================================================


class MoE1D(LightweightModule):
    """Router + routed experts for 1D-topology devices, prefill + decode.

    Simple API (90% of users — Gemma4-style defaults):
        moe = MoE1D(gate_proj, up_proj, down_proj, router_weight, top_k=8)
        out = moe.forward(router_input, expert_input, mode)   # mode = "decode" | "prefill"

    Power API (10% — full control incl. GPT-OSS config and all program/memory/dtype overrides):
        moe = MoE1D.from_config(MoE1DConfig(...))
    """

    def __init__(
        self,
        gate_proj: LazyWeight,
        up_proj: LazyWeight,
        down_proj: LazyWeight,
        router_weight: LazyWeight,
        top_k: int,
    ):
        """Simple constructor. Derives dims from weight shapes; uses Gemma4-style strategy defaults.

        Args:
            gate_proj/up_proj: [1, E, H, I] expert weights (column-parallel layout).
            down_proj: [1, E, I, H] expert weight (row-parallel layout).
            router_weight: [..., H, E] router linear weight (ttnn layout).
            top_k: experts selected per token.
        """
        super().__init__()
        self.config = _resolve_moe1d_config(
            MoE1DConfig(
                gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj, router_weight=router_weight, top_k=top_k
            )
        )
        self._bind_strategies()
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: MoE1DConfig):
        """Power-user constructor: full customization via a (possibly partial) MoE1DConfig."""
        instance = object.__new__(cls)
        super(MoE1D, instance).__init__()
        instance.config = _resolve_moe1d_config(config)
        instance._bind_strategies()
        instance._device_weights_loaded = False
        return instance

    # --- strategy binding (construction-time, not in forward) ---

    def _bind_strategies(self):
        cfg = self.config
        self._route_select = {
            RoutingStrategy.SOFTMAX_TOPK_SUMNORM: self._select_softmax_topk_sumnorm,
            RoutingStrategy.TOPK_SOFTMAX: self._select_topk_softmax,
        }[cfg.routing_strategy]
        self._activation = {
            ExpertActivation.GEGLU: self._act_geglu,
            ExpertActivation.SWIGLU_CLAMP: self._act_swiglu_clamp,
        }[cfg.activation_strategy]

    def load_device_weights(self):
        """Materialize LazyWeights onto device. Called on first forward; idempotent."""
        if self._device_weights_loaded:
            return
        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        cfg = self.config
        self.gate_proj = cfg.gate_proj.get_device_weight()
        self.up_proj = cfg.up_proj.get_device_weight()
        self.down_proj = cfg.down_proj.get_device_weight()
        self.router_weight = cfg.router_weight.get_device_weight()
        self.router_scale = cfg.router_scale.get_device_weight() if cfg.router_scale is not None else None
        self.router_bias = cfg.router_bias.get_device_weight() if cfg.router_bias is not None else None
        self.per_expert_scale = cfg.per_expert_scale.get_device_weight() if cfg.per_expert_scale is not None else None
        self.gate_bias = cfg.gate_bias.get_device_weight() if cfg.gate_bias is not None else None
        self.up_bias = cfg.up_bias.get_device_weight() if cfg.up_bias is not None else None
        self.down_bias = cfg.down_bias.get_device_weight() if cfg.down_bias is not None else None
        self._device_weights_loaded = True

    # =========================================================================
    # Router (shared across modes) -> dense_routing [1, 1, S, E]
    # =========================================================================

    def _route(self, router_input: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        x = router_input
        if cfg.router_prenorm_eps is not None:
            x = ttnn.rms_norm(x, epsilon=cfg.router_prenorm_eps)
        if self.router_scale is not None:
            x = ttnn.mul(x, self.router_scale)
        if cfg.router_input_scalar is not None:
            x = ttnn.mul(x, cfg.router_input_scalar)

        logits = ttnn.linear(x, self.router_weight, bias=self.router_bias)
        dense = self._route_select(logits, cfg.top_k)
        if self.per_expert_scale is not None:
            dense = ttnn.mul(dense, self.per_expert_scale)
        return dense

    def _select_softmax_topk_sumnorm(self, logits: ttnn.Tensor, top_k: int) -> ttnn.Tensor:
        """Gemma4: softmax(all) -> topk -> divide by top-k sum -> scatter dense."""
        probs = ttnn.softmax(logits, dim=-1)
        ttnn.deallocate(logits)
        top_vals, top_idx = ttnn.topk(probs, k=top_k, dim=-1)
        top_sum = ttnn.sum(top_vals, dim=-1, keepdim=True)
        top_vals = ttnn.div(top_vals, top_sum)
        ttnn.deallocate(top_sum)
        dense = ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=top_idx, src=top_vals)
        ttnn.deallocate(probs)
        ttnn.deallocate(top_vals)
        ttnn.deallocate(top_idx)
        return dense

    def _select_topk_softmax(self, logits: ttnn.Tensor, top_k: int) -> ttnn.Tensor:
        """GPT-OSS: topk(raw logits) -> softmax over the k selected -> scatter dense."""
        logits_bf16 = logits if logits.dtype == ttnn.bfloat16 else ttnn.typecast(logits, dtype=ttnn.bfloat16)
        top_vals, top_idx = ttnn.topk(logits_bf16, k=top_k, dim=-1, sorted=True)
        top_vals = ttnn.softmax(top_vals, dim=-1, numeric_stable=True)
        dense = ttnn.scatter(ttnn.zeros_like(logits_bf16), dim=-1, index=top_idx, src=top_vals)
        if logits_bf16 is not logits:
            ttnn.deallocate(logits_bf16)
        ttnn.deallocate(logits)
        ttnn.deallocate(top_vals)
        ttnn.deallocate(top_idx)
        return dense

    # =========================================================================
    # Expert activations (bound at construction)
    # =========================================================================

    def _act_geglu(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        """GeGLU: gelu(gate, tanh-approx) * up."""
        activated = ttnn.gelu(gate, fast_and_approximate_mode=True)
        return ttnn.mul(activated, up)

    def _act_swiglu_clamp(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        """SwiGLU (clamped): (up+1) * (gate*sigmoid(alpha*gate)), with clamping."""
        cfg = self.config
        gate = ttnn.clamp(gate, min=None, max=cfg.swiglu_limit)
        up = ttnn.clamp(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        gate_sig = ttnn.sigmoid(ttnn.mul(gate, cfg.swiglu_alpha))
        glu = ttnn.mul(gate, gate_sig)
        up = ttnn.add(up, 1)
        return ttnn.mul(up, glu)

    @staticmethod
    def _add_optional_bias(t: ttnn.Tensor, bias: ttnn.Tensor | None) -> ttnn.Tensor:
        """Add a per-expert bias if present (runtime None-check, not a static-config branch)."""
        if bias is None:
            return t
        return ttnn.add(t, bias)

    # =========================================================================
    # Forward dispatch
    # =========================================================================

    def forward(self, router_input: ttnn.Tensor, expert_input: ttnn.Tensor, mode: str | Mode) -> ttnn.Tensor:
        """Dispatch to decode/prefill (the only allowed static `if mode`)."""
        if isinstance(mode, Mode):
            mode = mode.value
        if mode == "decode":
            return self.decode_forward(router_input, expert_input)
        return self.prefill_forward(router_input, expert_input)

    def decode_forward(self, router_input: ttnn.Tensor, expert_input: ttnn.Tensor) -> ttnn.Tensor:
        """Decode (seq_len == 1). Straight line of compute."""
        self.load_device_weights()
        cfg = self.config
        E, H, I, k = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size, cfg.top_k

        dense_routing = self._route(router_input)
        seq = expert_input.shape[2]
        sparsity = ttnn.to_layout(dense_routing, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])

        gate = ttnn.sparse_matmul(
            expert_input,
            self.gate_proj,
            sparsity=sparsity,
            nnz=None,
            memory_config=cfg.decode_sparse_memcfg,
            output_tile=output_tile,
            program_config=cfg.decode_gate_up_prg_config(seq, I, H),
            dtype=cfg.sparse_matmul_dtype,
        )
        sm_i = gate.shape[-1]
        gate = ttnn.reshape(gate, (seq, E, 1, sm_i))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (seq, E, sm_i))
        gate = self._add_optional_bias(gate, self.gate_bias)

        up = ttnn.sparse_matmul(
            expert_input,
            self.up_proj,
            sparsity=sparsity,
            nnz=None,
            memory_config=cfg.decode_sparse_memcfg,
            output_tile=output_tile,
            program_config=cfg.decode_gate_up_prg_config(seq, I, H),
            dtype=cfg.sparse_matmul_dtype,
        )
        up = ttnn.reshape(up, (seq, E, 1, sm_i))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (seq, E, sm_i))
        up = self._add_optional_bias(up, self.up_bias)

        down_input = self._activation(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, E, seq, sm_i))

        down = ttnn.sparse_matmul(
            down_input,
            self.down_proj,
            sparsity=sparsity,
            nnz=None,
            memory_config=cfg.decode_sparse_memcfg,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=cfg.decode_down_prg_config(seq, H, I),
            dtype=cfg.sparse_matmul_dtype,
        )
        ttnn.deallocate(down_input)
        ttnn.deallocate(sparsity)

        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (seq, E, H))
        next_states = self._add_optional_bias(next_states, self.down_bias)

        routing_3d = ttnn.reshape(dense_routing, (seq, E, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        next_states = ttnn.reshape(next_states, (1, 1, seq, H), (1, 1, max(TILE_SIZE, seq), H))
        ttnn.deallocate(dense_routing)

        next_states = self._maybe_all_reduce(next_states)
        next_states = ttnn.to_memory_config(next_states, cfg.output_memcfg)
        return next_states

    def prefill_forward(self, router_input: ttnn.Tensor, expert_input: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill (seq_len > 1, multiple of 32). Chunk the sequence, process per chunk, concat."""
        self.load_device_weights()
        cfg = self.config
        seq_len = expert_input.shape[2]
        assert seq_len % TILE_SIZE == 0, f"prefill seq_len must be a multiple of {TILE_SIZE}, got {seq_len}"

        dense_routing = self._route(router_input)
        chunk = cfg.prefill_chunk_size

        if seq_len > chunk:
            h_chunks = ttnn.split(expert_input, chunk, dim=2)
            r_chunks = ttnn.split(dense_routing, chunk, dim=2)
        else:
            h_chunks = [expert_input]
            r_chunks = [dense_routing]

        result = None
        for h_chunk, r_chunk in zip(h_chunks, r_chunks):
            chunk_out = self._prefill_chunk(h_chunk, r_chunk)
            if result is None:
                result = chunk_out
            else:
                merged = ttnn.concat([result, chunk_out], dim=2)
                ttnn.deallocate(result)
                ttnn.deallocate(chunk_out)
                result = merged

        result = self._maybe_all_reduce(result)
        result = ttnn.to_memory_config(result, cfg.output_memcfg)
        return result

    def _prefill_chunk(self, expert_input: ttnn.Tensor, dense_routing: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
        chunk_len = expert_input.shape[2]
        groups = chunk_len // TILE_SIZE

        hidden_grouped = ttnn.reshape(expert_input, (1, groups, TILE_SIZE, H))
        # All-ones sparsity: compute every expert for every group; routing zeroes inactive ones after.
        sparsity = self._prefill_sparsity(E)
        sparsity_rep = ttnn.repeat(sparsity, (1, 1, groups, 1))
        nnz = E * groups
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])

        gate = ttnn.sparse_matmul(
            hidden_grouped,
            self.gate_proj,
            sparsity=sparsity_rep,
            nnz=nnz,
            memory_config=cfg.prefill_sparse_memcfg,
            output_tile=output_tile,
            program_config=cfg.prefill_gate_up_prg_config(TILE_SIZE, I, H),
            dtype=cfg.sparse_matmul_dtype,
        )
        sm_i = gate.shape[-1]
        gate = ttnn.transpose(gate, 1, 3)
        gate = ttnn.reshape(gate, (1, E, chunk_len, sm_i))
        gate = self._add_optional_bias(gate, self._bias_transposed(self.gate_bias))

        up = ttnn.sparse_matmul(
            hidden_grouped,
            self.up_proj,
            sparsity=sparsity_rep,
            nnz=nnz,
            memory_config=cfg.prefill_sparse_memcfg,
            output_tile=output_tile,
            program_config=cfg.prefill_gate_up_prg_config(TILE_SIZE, I, H),
            dtype=cfg.sparse_matmul_dtype,
        )
        ttnn.deallocate(hidden_grouped)
        up = ttnn.transpose(up, 1, 3)
        up = ttnn.reshape(up, (1, E, chunk_len, sm_i))
        up = self._add_optional_bias(up, self._bias_transposed(self.up_bias))

        down_input = self._activation(gate, up)
        down_input = ttnn.reshape(down_input, (1, E, chunk_len, sm_i))

        down = ttnn.sparse_matmul(
            down_input,
            self.down_proj,
            sparsity=sparsity,
            nnz=E,
            memory_config=cfg.prefill_sparse_memcfg,
            output_tile=output_tile,
            is_input_a_sparse=True,
            program_config=cfg.prefill_down_prg_config(chunk_len, H, I),
            dtype=cfg.sparse_matmul_dtype,
        )
        ttnn.deallocate(down_input)

        next_states = ttnn.reshape(down, (1, E, chunk_len, H))
        next_states = self._add_optional_bias(next_states, self._bias_transposed(self.down_bias))
        routing_perm = ttnn.permute(dense_routing, (0, 3, 2, 1))  # [1, E, S, 1]
        next_states = ttnn.mul(next_states, routing_perm)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        next_states = ttnn.reshape(next_states, (1, 1, chunk_len, H))
        return next_states

    def _prefill_sparsity(self, num_experts: int) -> ttnn.Tensor:
        """Cached all-ones [1,1,1,E] ROW_MAJOR bf16 sparsity (all experts active in prefill)."""
        if getattr(self, "_cached_prefill_sparsity", None) is None:
            self._cached_prefill_sparsity = ttnn.ones(
                (1, 1, 1, num_experts),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                device=self.config.mesh_device,
            )
        return self._cached_prefill_sparsity

    @staticmethod
    def _bias_transposed(bias: ttnn.Tensor | None) -> ttnn.Tensor | None:
        """Prefill biases are [1,E,I]/[1,E,H]; transpose dims 0/1 to broadcast over the chunk dim."""
        if bias is None:
            return None
        return ttnn.transpose(bias, 1, 0)

    def _maybe_all_reduce(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Row-parallel down_proj partials -> all-reduce across TP devices. No-op on a single device."""
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
            return x
        cluster_axis = 1 if cfg.mesh_device.shape[1] > 1 else 0
        reduced = ttnn.all_reduce(
            x,
            cluster_axis=cluster_axis,
            num_links=cfg.num_links,
            topology=cfg.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        return reduced


# =============================================================================
# Config helpers
# =============================================================================


def _build_sparse_matmul_config(m: int, n: int, in0_block_w: int = 1, k: int | None = None):
    """Default sparse_matmul program config: auto-pick the largest n-tile-dividing core grid in 8x8.

    Ported from the Gemma4 reference `_build_sparse_matmul_config`, with the autoport k-snapping:
    `Kt % in0_block_w == 0` is required by the sparse-matmul kernel — snap in0_block_w down to the
    largest divisor of Kt not exceeding the configured value (no-op when already divisible).
    """
    n_tiles = int(math.ceil(n / TILE_SIZE))
    best_cores, best_cx, best_cy = 1, 1, 1
    for num_cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % num_cores != 0:
            continue
        for cy in range(1, 9):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= 8 and num_cores > best_cores:
                    best_cores, best_cx, best_cy = num_cores, cx, cy
                    break
    per_core_N = n_tiles // best_cores

    if k is not None:
        k_tiles = int(math.ceil(k / TILE_SIZE))
        if k_tiles % in0_block_w != 0:
            divisors = [d for d in range(min(k_tiles, in0_block_w), 0, -1) if k_tiles % d == 0]
            in0_block_w = divisors[0] if divisors else 1

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=max(TILE_SIZE, m) // TILE_SIZE,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _resolve_moe1d_config(config: MoE1DConfig) -> MoE1DConfig:
    """Materialize the config to known-good defaults (foundational -> defaults -> resolve weights)."""
    to_set = {}

    assert config.top_k is not None, "top_k must be provided (cannot be derived from weights)"

    # --- Phase 1: dims from gate_proj [1, E, H, I] ---
    g_shape = config.gate_proj.source.shape
    assert len(g_shape) == 4, f"gate_proj must be 4D [1,E,H,I], got {tuple(g_shape)}"
    num_experts = config.num_experts if config.num_experts is not None else g_shape[1]
    hidden_size = config.hidden_size if config.hidden_size is not None else g_shape[2]
    intermediate_size = config.intermediate_size if config.intermediate_size is not None else g_shape[3]
    if config.num_experts is None:
        to_set["num_experts"] = num_experts
    if config.hidden_size is None:
        to_set["hidden_size"] = hidden_size
    if config.intermediate_size is None:
        to_set["intermediate_size"] = intermediate_size

    # --- Phase 1b: device + ccl + topology ---
    mesh_device = config.mesh_device or config.gate_proj.device or ttnn.GetDefaultDevice()
    assert mesh_device is not None, "mesh_device must be available!"
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device
    num_devices = mesh_device.get_num_devices()

    tt_ccl = config.tt_ccl
    if tt_ccl is None and num_devices > 1:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl
    elif tt_ccl is None:
        # Single device: no CCL needed, but is_resolved requires non-None — use the shared instance.
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl

    if config.topology is None and num_devices > 1:
        to_set["topology"] = default_topology(mesh_device)

    # --- Phase 2: dtypes ---
    if config.expert_weight_dtype is None:
        to_set["expert_weight_dtype"] = ttnn.bfloat8_b
    if config.router_weight_dtype is None:
        to_set["router_weight_dtype"] = ttnn.bfloat16
    if config.bias_dtype is None:
        to_set["bias_dtype"] = ttnn.bfloat16
    if config.sparse_matmul_dtype is None:
        to_set["sparse_matmul_dtype"] = ttnn.bfloat16
    if config.routing_dtype is None:
        to_set["routing_dtype"] = ttnn.bfloat16

    # --- Phase 3: validate / default activation params ---
    if config.activation_strategy == ExpertActivation.SWIGLU_CLAMP:
        assert (
            config.swiglu_limit is not None and config.swiglu_alpha is not None
        ), "SWIGLU_CLAMP requires swiglu_limit and swiglu_alpha"

    # --- Phase 4: memory configs ---
    if config.decode_sparse_memcfg is None:
        to_set["decode_sparse_memcfg"] = ttnn.L1_MEMORY_CONFIG
    if config.prefill_sparse_memcfg is None:
        to_set["prefill_sparse_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 5: sparse_matmul program-config callables ---
    gu_blk = config.gate_up_in0_block_w
    dn_blk = config.down_in0_block_w
    if config.decode_gate_up_prg_config is None:
        to_set["decode_gate_up_prg_config"] = lambda m, n, k: _build_sparse_matmul_config(m, n, gu_blk, k)
    if config.decode_down_prg_config is None:
        to_set["decode_down_prg_config"] = lambda m, n, k: _build_sparse_matmul_config(m, n, dn_blk, k)
    if config.prefill_gate_up_prg_config is None:
        to_set["prefill_gate_up_prg_config"] = lambda m, n, k: _build_sparse_matmul_config(m, n, gu_blk, k)
    if config.prefill_down_prg_config is None:
        to_set["prefill_down_prg_config"] = lambda m, n, k: _build_sparse_matmul_config(m, n, dn_blk, k)

    # --- Phase 6: resolve weights ---
    # Single device: replicate (mesh_mapper_config=None). Multi-device: column-parallel gate/up
    # (shard intermediate dim -1), row-parallel down (shard intermediate dim -2). Caller pre-pads
    # the per-device intermediate to tile alignment.
    replicate_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate()], mesh_shape_override=ttnn.MeshShape([num_devices])
    )
    if num_devices > 1:
        col_mapper = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([num_devices])
        )
        row_mapper = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-2)], mesh_shape_override=ttnn.MeshShape([num_devices])
        )
    else:
        col_mapper = replicate_mapper
        row_mapper = replicate_mapper

    ew_dtype = config.expert_weight_dtype or to_set.get("expert_weight_dtype")
    rw_dtype = config.router_weight_dtype or to_set.get("router_weight_dtype")
    bias_dtype = config.bias_dtype or to_set.get("bias_dtype")

    to_set["gate_proj"] = resolve_lazy_weight(
        config.gate_proj,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=col_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ew_dtype,
    )
    to_set["up_proj"] = resolve_lazy_weight(
        config.up_proj,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=col_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ew_dtype,
    )
    to_set["down_proj"] = resolve_lazy_weight(
        config.down_proj,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=row_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ew_dtype,
    )
    to_set["router_weight"] = resolve_lazy_weight(
        config.router_weight,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=replicate_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=rw_dtype,
    )

    # Optional weights — replicated across the mesh, resolved only when present.
    def _resolve_opt(w, dtype):
        if w is None:
            return None
        return resolve_lazy_weight(
            w,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=replicate_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    if config.router_scale is not None:
        to_set["router_scale"] = _resolve_opt(config.router_scale, ttnn.bfloat16)
    if config.router_bias is not None:
        to_set["router_bias"] = _resolve_opt(config.router_bias, bias_dtype)
    if config.per_expert_scale is not None:
        to_set["per_expert_scale"] = _resolve_opt(config.per_expert_scale, ttnn.bfloat16)
    # Expert biases: gate/up are column-parallel (shard -1); down is row-parallel-ish (replicated +
    # device-0 trick handled by the caller). On single device all replicate.
    if config.gate_bias is not None:
        to_set["gate_bias"] = resolve_lazy_weight(
            config.gate_bias,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=col_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=bias_dtype,
        )
    if config.up_bias is not None:
        to_set["up_bias"] = resolve_lazy_weight(
            config.up_bias,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=col_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=bias_dtype,
        )
    if config.down_bias is not None:
        to_set["down_bias"] = _resolve_opt(config.down_bias, bias_dtype)

    resolved = replace(config, **to_set)
    assert resolved.gate_proj.is_resolved() and resolved.down_proj.is_resolved(), "weights must resolve!"
    assert resolved.is_resolved(), "MoE1DConfig must be resolved!"
    return resolved
