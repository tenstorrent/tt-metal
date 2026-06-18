# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MoE (Mixture of Experts) module for 1D-topology devices: N150 (1x1),
N300 (1x2), T3K (1x8).

Scope (M1/M2): **router + routed experts only**, tensor-parallel (TP) on 1D meshes.
The caller owns the shared MLP, the residual adds, the post-expert norms, and the LM
head. Expert-parallel (EP) and the GPT-OSS all-to-all "throughput" path are Galaxy/2D
concerns and live in a future ``MoE2D`` — see
``dev-tools/agents-context/tttv2-module-bringup/moe/MODULE_REFERENCES.md``.

One ``MoE1D`` class generalizes several reference families via config strategy:

  - **Gemma4**: GeGLU, no bias, ``softmax->topk->renorm`` routing, router RMSNorm +
    input/logit scales + per-expert scale, expert RMSNorm, unfused gate/up.
  - **GPT-OSS**: clamped-SwiGLU (alpha, limit), bias on every projection,
    ``topk->softmax`` routing, no router/expert pre-norm (caller pre-normed), fused gate/up.
  - **Qwen3-MoE**: plain SwiGLU (``silu(gate)*up``), no bias, ``softmax->topk->renorm``
    routing, no pre-norms.
  - **Granite-4-H**: plain SwiGLU, no bias, ``topk->softmax`` routing (the routed-expert
    block only — the shared expert / Mamba mixer are caller-owned).
  - **North-Mini**: plain SwiGLU, no bias, ``topk->sigmoid`` routing (independent
    per-expert gate, no renorm).

Forward is a single input (``forward(x, mode)``). When the optional router/expert norm
weights are present (Gemma), they are applied internally so the same ``x`` feeds both the
router and the experts under their distinct norms — reproducing the two tensors Gemma's
decoder layer builds, without a second forward argument.

NOTE (M2 scaffold): ``prefill_forward`` / ``decode_forward`` are stubs. Forward compute
(sparse_matmul expert path) lands in M3, tuned to PCC >= 0.99 against the torch golden in
``models/common/tests/modules/moe/reference_moe.py``.
"""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl
from models.tt_transformers.tt.common import Mode

# =============================================================================
# Strategy enums (resolved at construction; never branched in the hot path)
# =============================================================================


class ExpertActivation(Enum):
    """Expert gated-MLP activation. GEGLU = gelu(gate)*up (Gemma). SWIGLU = silu(gate)*up
    (plain SwiGLU — Qwen3-MoE / Granite-4 / North-Mini). CLAMPED_SWIGLU =
    (clamp(up)+1)*clamp(gate)*sigmoid(alpha*clamp(gate)) (GPT-OSS)."""

    GEGLU = "geglu"
    SWIGLU = "swiglu"
    CLAMPED_SWIGLU = "clamped_swiglu"


class RoutingNorm(Enum):
    """Router weight normalization. SOFTMAX_TOPK_RENORM: softmax over all experts,
    topk, renormalize topk to sum 1 (Gemma / Qwen3-MoE). TOPK_SOFTMAX: topk over raw
    logits, then softmax over the topk (GPT-OSS / Granite-4). TOPK_SIGMOID: topk over raw
    logits, then independent per-expert sigmoid on the topk, no renorm (North-Mini)."""

    SOFTMAX_TOPK_RENORM = "softmax_topk_renorm"
    TOPK_SOFTMAX = "topk_softmax"
    TOPK_SIGMOID = "topk_sigmoid"


# =============================================================================
# Top-level config dataclass — single source of truth
# =============================================================================


@dataclass
class MoEConfig:
    """Central configuration for MoE1D.

    Simple usage (Gemma-shaped defaults):
        config = MoEConfig(router_weight, gate_proj, up_proj, down_proj, top_k=8)

    GPT-OSS:
        config = MoEConfig(
            router_weight, gate_proj, up_proj, down_proj, top_k=4,
            router_bias=..., gate_bias=..., up_bias=..., down_bias=...,
            expert_activation=ExpertActivation.CLAMPED_SWIGLU,
            routing_norm=RoutingNorm.TOPK_SOFTMAX,
            fuse_gate_up=True,
        )

    Expected weight layouts (TTNN layout — caller transposes from HF):
        router_weight : (H, E)           replicated
        gate_proj     : (1, E, H, I)     column-parallel (shard -1)
        up_proj       : (1, E, H, I)     column-parallel (shard -1)
        down_proj     : (1, E, I, H)     row-parallel    (shard -2)
        *_bias        : matching, optional
        router_norm_weight / expert_norm_weight / router_input_scale : (H,) optional
        per_expert_scale : (E,) optional
    """

    # Required: routing + expert weights (LazyWeight)
    router_weight: LazyWeight
    gate_proj: LazyWeight
    up_proj: LazyWeight
    down_proj: LazyWeight

    # Essential routing dim (not derivable from weights)
    top_k: int | None = None

    # Optional biases (GPT-OSS)
    router_bias: LazyWeight | None = None
    gate_bias: LazyWeight | None = None
    up_bias: LazyWeight | None = None
    down_bias: LazyWeight | None = None

    # Optional router pre-processing (Gemma). None => step skipped.
    router_norm_weight: LazyWeight | None = None
    router_input_scale: LazyWeight | None = None
    router_logit_scale: float | None = None  # scalar, e.g. hidden_size**-0.5
    per_expert_scale: LazyWeight | None = None

    # Optional expert pre-norm (Gemma pre_feedforward_layernorm_2). None => experts use x.
    expert_norm_weight: LazyWeight | None = None

    # Strategy (resolved at construction)
    expert_activation: ExpertActivation = ExpertActivation.GEGLU
    routing_norm: RoutingNorm = RoutingNorm.SOFTMAX_TOPK_RENORM
    fuse_gate_up: bool = False

    # Scalars
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    rms_norm_eps: float = 1e-6

    # Derived dims (None => derive from weights)
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_experts: int | None = None

    # Device + collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None
    num_reduce_scatter_links: int = 1

    # Dtypes (None => resolve defaults)
    expert_weight_dtype: ttnn.DataType | None = None
    router_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None

    # Prefill chunking (sparse_matmul groups). None => resolve default.
    prefill_chunk_size: int | None = None

    # Power-user program/memory/compute-kernel overrides (None => resolve in forward, M3)
    gate_up_program_config: object | None = None
    down_program_config: object | None = None
    decode_memory_config: ttnn.MemoryConfig | None = None
    prefill_memory_config: ttnn.MemoryConfig | None = None
    expert_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    router_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    # Fields not required for is_resolved() (genuinely optional / model-specific).
    _OPTIONAL_FIELDS = {
        "router_bias",
        "gate_bias",
        "up_bias",
        "down_bias",
        "router_norm_weight",
        "router_input_scale",
        "router_logit_scale",
        "per_expert_scale",
        "expert_norm_weight",
        "activation_dtype",
        "gate_up_program_config",
        "down_program_config",
        "decode_memory_config",
        "prefill_memory_config",
        "expert_compute_kernel_cfg",
        "router_compute_kernel_cfg",
    }

    def is_resolved(self) -> bool:
        """All non-optional fields populated (topology optional on single device)."""
        optional = set(self._OPTIONAL_FIELDS)
        if self.mesh_device is not None and self.mesh_device.get_num_devices() == 1:
            optional.add("topology")
        return all(getattr(self, f) is not None for f in self.__dataclass_fields__ if f not in optional)


# =============================================================================
# MoE1D
# =============================================================================


class MoE1D(LightweightModule):
    """Mixture-of-Experts (router + routed experts) for 1D-topology devices, TP only.

    Simple API (Gemma-shaped):
        moe = MoE1D(router_weight, gate_proj, up_proj, down_proj, top_k=8)

    Power API (GPT-OSS / overrides):
        moe = MoE1D.from_config(MoEConfig(...))
    """

    def __init__(
        self,
        router_weight: LazyWeight,
        gate_proj: LazyWeight,
        up_proj: LazyWeight,
        down_proj: LazyWeight,
        top_k: int,
    ):
        super().__init__()
        self.config = _resolve_moe1d_config(
            MoEConfig(
                router_weight=router_weight,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
                top_k=top_k,
            )
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: MoEConfig):
        instance = object.__new__(cls)
        super(MoE1D, instance).__init__()
        instance.config = _resolve_moe1d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        """Materialize LazyWeights onto device. Called on first forward; idempotent."""
        if self._device_weights_loaded:
            return
        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        cfg = self.config

        self.router_weight = cfg.router_weight.get_device_weight()
        self.gate_proj = cfg.gate_proj.get_device_weight()
        self.up_proj = cfg.up_proj.get_device_weight()
        self.down_proj = cfg.down_proj.get_device_weight()

        # Optional weights — load when present.
        def _opt(w):
            return w.get_device_weight() if w is not None else None

        self.router_bias = _opt(cfg.router_bias)
        self.gate_bias = _opt(cfg.gate_bias)
        self.up_bias = _opt(cfg.up_bias)
        self.down_bias = _opt(cfg.down_bias)

        # down_bias is added to each device's row-parallel partial *before* the TP
        # all-reduce, so a replicated bias would be summed once per device. Pre-scale by
        # 1/num_devices so the all-reduce sum reconstructs the bias exactly once. (gate/up
        # bias are column-sharded and added pre-contraction, so they don't double-count.)
        num_devices = cfg.mesh_device.get_num_devices()
        if self.down_bias is not None and num_devices > 1:
            self.down_bias = ttnn.mul(self.down_bias, 1.0 / num_devices)
        self.router_norm_weight = _opt(cfg.router_norm_weight)
        self.router_input_scale = _opt(cfg.router_input_scale)
        self.per_expert_scale = _opt(cfg.per_expert_scale)
        self.expert_norm_weight = _opt(cfg.expert_norm_weight)

        # Per-device (possibly padded) intermediate width — the sparse_matmul N dim
        # for gate/up. Read from the resolved device tensor so TP padding is honored.
        self._intermediate_per_device = self.gate_proj.shape[-1]

        self._device_weights_loaded = True

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str | Mode) -> ttnn.Tensor:
        """Dispatch by mode. Single input — router/expert norms (if configured) applied
        internally."""
        if isinstance(mode, Mode):
            mode = mode.value
        if mode == "decode":
            return self.decode_forward(x)
        return self.prefill_forward(x)

    def decode_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode (seq_len==1): sparse_matmul with top-k sparsity (only selected experts)."""
        self.load_device_weights()
        dense_routing = self._router_forward(x)
        expert_input = self._maybe_expert_norm(x)
        return self._experts_decode(expert_input, dense_routing)

    def prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill (seq_len%32==0): all-ones sparsity (compute all experts), mask after."""
        self.load_device_weights()
        dense_routing = self._router_forward(x)
        expert_input = self._maybe_expert_norm(x)
        return self._experts_prefill(expert_input, dense_routing)

    # ── Router ────────────────────────────────────────────────────────────

    def _router_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,S,H] -> dense routing [1,1,S,E] (zeros at non-selected experts).

        Optional pre-norm/scales (Gemma) -> linear(+bias) -> routing-norm strategy
        -> scatter -> optional per-expert scale.
        """
        cfg = self.config
        h = x
        if self.router_norm_weight is not None:
            h = ttnn.rms_norm(h, epsilon=cfg.rms_norm_eps, weight=self.router_norm_weight)
        if self.router_input_scale is not None:
            h = ttnn.mul(h, self.router_input_scale)
        if cfg.router_logit_scale is not None:
            h = ttnn.mul(h, cfg.router_logit_scale)

        # Routing is an argmax (topk): bf16 logit rounding can flip near-tie expert
        # selections, and a single mis-route tanks PCC. HF computes the router in fp32;
        # we do the same — fp32 logits keep the top-k selection stable.
        logits = ttnn.linear(
            h,
            self.router_weight,
            bias=self.router_bias,
            compute_kernel_config=cfg.router_compute_kernel_cfg,
            dtype=ttnn.float32,
        )  # [1,1,S,E] fp32

        # ttnn.topk and ttnn.scatter require bf16 (not fp32+TILE). Softmax/renorm run in
        # fp32 for value accuracy; the discrete selection + dense scatter run in bf16
        # (dense routing multiplies bf16 experts downstream anyway).
        if cfg.routing_norm == RoutingNorm.SOFTMAX_TOPK_RENORM:
            probs = ttnn.softmax(logits, dim=-1)
            probs_bf16 = ttnn.typecast(probs, ttnn.bfloat16)
            top_v, top_i = ttnn.topk(probs_bf16, k=cfg.top_k, dim=-1)
            top_v = ttnn.typecast(top_v, ttnn.float32)
            top_sum = ttnn.sum(top_v, dim=-1, keepdim=True)
            top_v = ttnn.div(top_v, top_sum)
            dense = ttnn.scatter(
                ttnn.zeros_like(probs_bf16), dim=-1, index=top_i, src=ttnn.typecast(top_v, ttnn.bfloat16)
            )
        elif cfg.routing_norm == RoutingNorm.TOPK_SOFTMAX:
            logits_bf16 = ttnn.typecast(logits, ttnn.bfloat16)
            top_v, top_i = ttnn.topk(logits_bf16, k=cfg.top_k, dim=-1)
            top_v = ttnn.softmax(ttnn.typecast(top_v, ttnn.float32), dim=-1)
            dense = ttnn.scatter(
                ttnn.zeros_like(logits_bf16), dim=-1, index=top_i, src=ttnn.typecast(top_v, ttnn.bfloat16)
            )
        else:  # TOPK_SIGMOID — independent per-expert gate, no renorm (North-Mini)
            logits_bf16 = ttnn.typecast(logits, ttnn.bfloat16)
            top_v, top_i = ttnn.topk(logits_bf16, k=cfg.top_k, dim=-1)
            top_v = ttnn.sigmoid(ttnn.typecast(top_v, ttnn.float32))
            dense = ttnn.scatter(
                ttnn.zeros_like(logits_bf16), dim=-1, index=top_i, src=ttnn.typecast(top_v, ttnn.bfloat16)
            )

        if self.per_expert_scale is not None:
            dense = ttnn.mul(dense, self.per_expert_scale)
        return dense

    def _maybe_expert_norm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.expert_norm_weight is not None:
            return ttnn.rms_norm(x, epsilon=self.config.rms_norm_eps, weight=self.expert_norm_weight)
        return x

    # ── Expert activation ─────────────────────────────────────────────────

    def _apply_activation(self, gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        if cfg.expert_activation == ExpertActivation.GEGLU:
            return ttnn.mul(ttnn.gelu(gate, fast_and_approximate_mode=True), up)
        if cfg.expert_activation == ExpertActivation.SWIGLU:
            # Plain SwiGLU: silu(gate) * up (Qwen3-MoE / Granite-4 / North-Mini).
            return ttnn.mul(ttnn.silu(gate), up)
        # CLAMPED_SWIGLU (GPT-OSS): (clamp(up)+1) * clamp(gate) * sigmoid(alpha*clamp(gate))
        g = ttnn.clamp(gate, max=cfg.swiglu_limit)
        u = ttnn.clamp(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        glu = ttnn.mul(g, ttnn.sigmoid(ttnn.mul(g, cfg.swiglu_alpha)))
        return ttnn.mul(ttnn.add(u, 1.0), glu)

    @staticmethod
    def _add_bias(t: ttnn.Tensor, bias: Optional[ttnn.Tensor], grouped: bool) -> ttnn.Tensor:
        """Add a per-expert bias. ``bias`` is [1,E,N]; for the grouped (prefill) case the
        expert tensor is [1,E,chunk,N] so insert a unit chunk dim for broadcast."""
        if bias is None:
            return t
        if grouped:
            # ttnn.reshape only accepts bf16/fp32/int32/uint32. With block-float expert
            # dtypes (e.g. bfloat4_b/bfloat8_b for GPT-OSS) the bias is loaded at the same
            # dtype, so promote to bf16 for the broadcast reshape (decode skips the reshape
            # and adds directly, so it's unaffected).
            if bias.dtype not in (ttnn.bfloat16, ttnn.float32):
                bias = ttnn.typecast(bias, ttnn.bfloat16)
            bias = ttnn.reshape(bias, (1, bias.shape[-2], 1, bias.shape[-1]))
        return ttnn.add(t, bias)

    # ── Experts: decode (top-k sparsity) ──────────────────────────────────

    def _experts_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        E = cfg.num_experts
        top_k = cfg.top_k
        H = cfg.hidden_size
        I = self._intermediate_per_device
        batch = hidden_states.shape[2]  # 1 for decode

        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        # gate/up contract over H; down contracts over the per-device intermediate I.
        gate_up_cfg = _build_sparse_matmul_config(batch, I, k=H)
        down_cfg = _build_sparse_matmul_config(batch, H, k=I)

        gate = ttnn.sparse_matmul(
            hidden_states,
            self.gate_proj,
            sparsity=sparsity,
            nnz=top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_cfg,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )
        sm_i = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, E, 1, sm_i))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, E, sm_i))
        gate = self._add_bias(gate, self.gate_bias, grouped=False)

        up = ttnn.sparse_matmul(
            hidden_states,
            self.up_proj,
            sparsity=sparsity,
            nnz=top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_cfg,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )
        up = ttnn.reshape(up, (batch, E, 1, sm_i))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, E, sm_i))
        up = self._add_bias(up, self.up_bias, grouped=False)

        down_input = self._apply_activation(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, E, batch, sm_i))

        down = ttnn.sparse_matmul(
            down_input,
            self.down_proj,
            sparsity=sparsity,
            nnz=top_k,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_cfg,
            is_input_a_sparse=True,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )

        next_states = ttnn.permute(down, (0, 2, 1, 3))  # [1,S,E,H]
        next_states = ttnn.reshape(next_states, (batch, E, H))
        next_states = self._add_bias(next_states, self.down_bias, grouped=False)
        routing_3d = ttnn.reshape(routing_weights, (batch, E, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        next_states = ttnn.reshape(next_states, (1, 1, batch, H), (1, 1, max(32, batch), H))
        return self._maybe_all_reduce(next_states)

    # ── Experts: prefill (all-ones sparsity, mask after) ──────────────────

    def _experts_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        chunk = cfg.prefill_chunk_size
        seq_len = hidden_states.shape[2]
        assert seq_len % 32 == 0, f"prefill seq_len must be multiple of 32, got {seq_len}"

        if seq_len > chunk:
            h_chunks = ttnn.split(hidden_states, chunk, dim=2)
            r_chunks = ttnn.split(routing_weights, chunk, dim=2)
        else:
            h_chunks = [hidden_states]
            r_chunks = [routing_weights]

        result = None
        for h_c, r_c in zip(h_chunks, r_chunks):
            chunk_out = self._process_prefill_chunk(h_c, r_c)
            if result is None:
                result = chunk_out
            else:
                cat = ttnn.concat([result, chunk_out], dim=2)
                result.deallocate(True)
                chunk_out.deallocate(True)
                result = cat
        return self._maybe_all_reduce(result)

    def _process_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        E = cfg.num_experts
        H = cfg.hidden_size
        I = self._intermediate_per_device
        chunk_len = hidden_states.shape[2]
        group_size = chunk_len // 32

        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, 32, H))
        sparsity = ttnn.repeat(self._prefill_sparsity(), (1, 1, group_size, 1))
        nnz = E * group_size
        output_tile = ttnn.Tile([32, 32])
        # gate/up: M-per-group is a single 32-token tile (the `group_size` groups ride the
        # sparse batch dim), so m=32. down: its input is [1,E,chunk_len,sm_i] with
        # is_input_a_sparse=True, so the matmul M-dim is the whole chunk (group_size·32).
        # Pass m=chunk_len so per_core_M=group_size and num_blocks_y collapses to 1 — else
        # num_blocks_y scales with group_size and overruns the core grid (matches the
        # reference, which passes the real seq_len to its down program config).
        gate_up_cfg = _build_sparse_matmul_config(32, I, k=H)
        down_cfg = _build_sparse_matmul_config(chunk_len, H, k=I)

        gate = ttnn.sparse_matmul(
            hidden_grouped,
            self.gate_proj,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_cfg,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )
        sm_i = gate.shape[-1]
        gate = ttnn.transpose(gate, 1, 3)
        gate = ttnn.reshape(gate, (1, E, chunk_len, sm_i))
        gate = self._add_bias(gate, self.gate_bias, grouped=True)

        up = ttnn.sparse_matmul(
            hidden_grouped,
            self.up_proj,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_cfg,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )
        hidden_grouped.deallocate(True)
        up = ttnn.transpose(up, 1, 3)
        up = ttnn.reshape(up, (1, E, chunk_len, sm_i))
        up = self._add_bias(up, self.up_bias, grouped=True)

        down_input = self._apply_activation(gate, up)
        down_input = ttnn.reshape(down_input, (1, E, chunk_len, sm_i))

        down = ttnn.sparse_matmul(
            down_input,
            self.down_proj,
            sparsity=self._prefill_sparsity(),
            nnz=E,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_cfg,
            is_input_a_sparse=True,
            dtype=cfg.activation_dtype,
            compute_kernel_config=cfg.expert_compute_kernel_cfg,
        )
        down_input.deallocate(True)

        next_states = ttnn.reshape(down, (1, E, chunk_len, H))
        next_states = self._add_bias(next_states, self.down_bias, grouped=True)
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))  # [1,E,S,1]
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        next_states = ttnn.reshape(next_states, (1, 1, chunk_len, H))
        return next_states

    def _prefill_sparsity(self) -> ttnn.Tensor:
        """Cached all-ones [1,1,1,E] ROW_MAJOR bf16 sparsity (prefill computes all experts)."""
        cached = getattr(self, "_prefill_sparsity_tensor", None)
        if cached is not None:
            return cached
        import torch

        cfg = self.config
        ones = torch.ones(1, 1, 1, cfg.num_experts, dtype=torch.bfloat16)
        is_mesh = hasattr(self.config.mesh_device, "shape")
        sp = ttnn.from_torch(
            ones,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=cfg.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(cfg.mesh_device) if is_mesh else None,
        )
        self._prefill_sparsity_tensor = sp
        return sp

    def _maybe_all_reduce(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce the row-parallel down_proj partial sums across the TP axis.

        gate/up are column-parallel (each device holds I/tp of the intermediate); the
        row-parallel down_proj then produces a partial [.,.,S,H] per device, summed here
        to the full hidden. No-op on single device (TP=1). Output is replicated full-H.
        """
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
            return x
        # 1D mesh exposed as (1, N) → TP reduce runs along the N-device axis.
        tp_axis = 1 if cfg.mesh_device.shape[0] == 1 else 0
        return ttnn.all_reduce(
            x,
            cluster_axis=tp_axis,
            num_links=cfg.num_reduce_scatter_links,
            topology=cfg.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# =============================================================================
# sparse_matmul program config
# =============================================================================


# Cap on the K-blocking factor (in0_block_w, in tiles). Bounds the in0/in1 L1
# circular-buffer footprint while still amortizing K reloads. The GPT-OSS reference
# uses up to 30; 32 covers it and stays L1-safe for the K dims MoE1D targets.
_IN0_BLOCK_W_CAP = 32


def _largest_divisor_leq(value, cap):
    """Largest divisor of ``value`` not exceeding ``cap`` (>=1)."""
    hi = min(value, cap)
    for d in range(hi, 0, -1):
        if value % d == 0:
            return d
    return 1


def _build_sparse_matmul_config(m, n, k=None, in0_block_w=None):
    """1D multicast program config for ttnn.sparse_matmul (ported from Gemma4 experts).

    Picks the largest core count that divides the N tile count and fits an 8x8 grid.
    ``k`` (the contraction dim) sets ``in0_block_w`` to the largest divisor of ``Kt``
    under ``_IN0_BLOCK_W_CAP`` so the matmul blocks K instead of looping it one tile at
    a time (``in0_block_w=1`` was ~3-4x slower than the reference's K-blocked config).
    The kernel requires ``Kt % in0_block_w == 0``, so a divisor is mandatory.
    """
    n_tiles = int(math.ceil(n / 32))
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
    if in0_block_w is None:
        in0_block_w = 1 if k is None else _largest_divisor_leq(int(math.ceil(k / 32)), _IN0_BLOCK_W_CAP)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=max(32, m) // 32,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_moe1d_config(config: MoEConfig) -> MoEConfig:
    """Materialize MoEConfig defaults in phases: dims -> device -> dtypes -> LazyWeights.

    Program/memory/compute-kernel configs for the sparse_matmul path are resolved in
    forward (M3) where seq-len / per-device intermediate are known; they stay None here.
    """
    to_set = {}

    # --- Phase 1: dims (weights are in TTNN layout) ---
    # router_weight: (H, E) ; gate_proj: (1, E, H, I)
    hidden_size = config.hidden_size
    if hidden_size is None:
        hidden_size = config.router_weight.source.shape[-2]
        to_set["hidden_size"] = hidden_size

    num_experts = config.num_experts
    if num_experts is None:
        num_experts = config.router_weight.source.shape[-1]
        to_set["num_experts"] = num_experts

    intermediate_size = config.intermediate_size
    if intermediate_size is None:
        intermediate_size = config.gate_proj.source.shape[-1]
        to_set["intermediate_size"] = intermediate_size

    assert config.top_k is not None, "top_k must be provided (not derivable from weights)!"

    # --- Phase 2: device / ccl / topology ---
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.router_weight.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device
    assert mesh_device is not None, "mesh_device must be available at this point!"

    tt_ccl = config.tt_ccl
    if config.tt_ccl is None:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl
    assert tt_ccl.mesh_device == mesh_device, "tt_ccl must match mesh_device!"

    if config.topology is None:
        to_set["topology"] = default_topology(mesh_device)

    num_devices = mesh_device.get_num_devices()

    # --- Phase 3: dtypes ---
    if config.expert_weight_dtype is None:
        to_set["expert_weight_dtype"] = ttnn.bfloat8_b
    if config.router_dtype is None:
        to_set["router_dtype"] = ttnn.bfloat16
    # sparse_matmul output (activation) dtype. The reference GPT-OSS experts pack
    # activations as bfloat8_b; default to bf16 here for headroom and let GPT-OSS-style
    # configs opt into bf8 to match the reference. Wired into every expert sparse_matmul.
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16
    if config.prefill_chunk_size is None:
        # Prefill computes ALL experts densely; peak intermediate ∝ E · group_size
        # (group_size = chunk/32 tiles of tokens). Bound that by a fixed expert·token-tile
        # budget so few-expert models (GPT-OSS, E=8) single-pass the whole sequence in one
        # sparse_matmul — matching the reference's single-pass chunking — while many-expert
        # models (Gemma, E=128) stay at one 32-token tile group to cap DRAM. B=64 → E=8
        # gives chunk=256 (seq≤256 single-pass); E>=64 gives chunk=32 (prior behavior).
        EXPERT_TILE_BUDGET = 64
        group_size = max(1, EXPERT_TILE_BUDGET // num_experts)
        to_set["prefill_chunk_size"] = group_size * 32

    expert_dtype = config.expert_weight_dtype or to_set.get("expert_weight_dtype")
    router_dtype = config.router_dtype or to_set.get("router_dtype")

    # HiFi4 + fp32 dest accumulation for the expert matmuls and router projection.
    # The default (LoFi) sparse_matmul drifts per-op enough to cap the layer at ~0.98
    # PCC; promoting fidelity recovers >= 0.99 (see MODULE_BRINGUP gotcha on math fidelity).
    def _hifi4():
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if config.expert_compute_kernel_cfg is None:
        to_set["expert_compute_kernel_cfg"] = _hifi4()
    if config.router_compute_kernel_cfg is None:
        to_set["router_compute_kernel_cfg"] = _hifi4()

    # --- Phase 4: resolve LazyWeights with TP mesh mappers ---
    # gate/up: column-parallel (shard intermediate dim -1); down: row-parallel (shard -2).
    col_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([num_devices]),
    )
    row_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2)],
        mesh_shape_override=ttnn.MeshShape([num_devices]),
    )

    common = dict(device=mesh_device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    to_set["gate_proj"] = resolve_lazy_weight(
        config.gate_proj, mesh_mapper_config=col_mapper, dtype=expert_dtype, **common
    )
    to_set["up_proj"] = resolve_lazy_weight(config.up_proj, mesh_mapper_config=col_mapper, dtype=expert_dtype, **common)
    to_set["down_proj"] = resolve_lazy_weight(
        config.down_proj, mesh_mapper_config=row_mapper, dtype=expert_dtype, **common
    )
    # router weight + norm/scale vectors: replicated (mesh_mapper_config=None).
    to_set["router_weight"] = resolve_lazy_weight(
        config.router_weight, mesh_mapper_config=None, dtype=router_dtype, **common
    )

    def _resolve_opt(w, mapper, dtype, layout=ttnn.TILE_LAYOUT):
        return (
            None
            if w is None
            else resolve_lazy_weight(
                w,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper_config=mapper,
                dtype=dtype,
                layout=layout,
            )
        )

    # Biases: gate/up bias shard with their matmul output (-1); down/router bias replicated. TILE.
    if config.gate_bias is not None:
        to_set["gate_bias"] = _resolve_opt(config.gate_bias, col_mapper, expert_dtype)
    if config.up_bias is not None:
        to_set["up_bias"] = _resolve_opt(config.up_bias, col_mapper, expert_dtype)
    if config.down_bias is not None:
        to_set["down_bias"] = _resolve_opt(config.down_bias, None, expert_dtype)
    if config.router_bias is not None:
        to_set["router_bias"] = _resolve_opt(config.router_bias, None, router_dtype)
    # Norm gammas feed ttnn.rms_norm: ROW_MAJOR, shape (1,1,dim//32,32), bf16, replicated.
    if config.router_norm_weight is not None:
        to_set["router_norm_weight"] = _resolve_opt(
            config.router_norm_weight, None, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    if config.expert_norm_weight is not None:
        to_set["expert_norm_weight"] = _resolve_opt(
            config.expert_norm_weight, None, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    # Scale vectors feed ttnn.mul broadcast: TILE, shape (1,1,1,dim), bf16, replicated.
    if config.router_input_scale is not None:
        to_set["router_input_scale"] = _resolve_opt(config.router_input_scale, None, ttnn.bfloat16)
    if config.per_expert_scale is not None:
        to_set["per_expert_scale"] = _resolve_opt(config.per_expert_scale, None, ttnn.bfloat16)

    resolved = replace(config, **to_set)
    assert resolved.is_resolved(), "Config must be resolved!"
    return resolved
