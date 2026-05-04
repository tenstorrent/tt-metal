# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni thinker MoE TTNN modules."""

import os

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule, DeviceArch, run_on_devices
from models.experimental.tt_symbiote.modules.moe import (
    TTNNExperts,
    _consolidate_talker_experts_from_module_list,
    _to_torch_any,
    _to_ttnn_raw,
)


def _thinker_gate_router_attrs(thinker_mlp):
    """HF thinker MoE gate may be ``Qwen3OmniMoeThinkerTextTopKRouter`` or a plain ``nn.Linear`` (router logits)."""
    g = thinker_mlp.gate
    cfg = getattr(thinker_mlp, "config", None)

    top_k = getattr(g, "top_k", None)
    if top_k is None and cfg is not None:
        top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_top_k", None)
    if top_k is None:
        top_k = 8

    norm_topk_prob = getattr(g, "norm_topk_prob", None)
    if norm_topk_prob is None and cfg is not None:
        norm_topk_prob = getattr(cfg, "norm_topk_prob", True)
    if norm_topk_prob is None:
        norm_topk_prob = True

    num_experts = getattr(g, "num_experts", None)
    if num_experts is None and isinstance(g, torch.nn.Linear):
        num_experts = int(g.out_features)
    if num_experts is None and cfg is not None:
        num_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
    if num_experts is None:
        ex = thinker_mlp.experts
        if hasattr(ex, "gate_up_proj"):
            num_experts = int(ex.gate_up_proj.shape[0])
        else:
            num_experts = len(ex)

    return int(top_k), bool(norm_topk_prob), int(num_experts)


def _thinker_config_fallback_from_modules(thinker_mlp):
    """Infer MoE config when HF omits ``config`` on ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` (ModuleList experts)."""
    g = thinker_mlp.gate
    hf_experts = thinker_mlp.experts
    n = len(hf_experts)
    if n == 0:
        raise ValueError("Thinker sparse MoE has empty experts ModuleList")
    ex0 = hf_experts[0]

    hidden = None
    if isinstance(g, torch.nn.Linear):
        hidden = int(g.in_features)
    if hidden is None and hasattr(ex0, "gate_proj"):
        hidden = int(ex0.gate_proj.in_features)
    if hidden is None and hasattr(ex0, "down_proj"):
        hidden = int(ex0.down_proj.out_features)
    if hidden is None:
        raise ValueError("Cannot infer thinker hidden_size from gate/expert weights")

    interm = None
    if hasattr(ex0, "gate_proj"):
        interm = int(ex0.gate_proj.out_features)
    if interm is None:
        raise ValueError("Cannot infer thinker moe_intermediate_size from expert MLP")

    top_k, _, _ = _thinker_gate_router_attrs(thinker_mlp)
    cfg = type("ThinkerMoEInferredConfig", (), {})()
    cfg.hidden_size = hidden
    cfg.moe_intermediate_size = interm
    cfg.n_routed_experts = n
    cfg.num_experts = n
    cfg.num_experts_per_tok = top_k
    return cfg


def _thinker_experts_adapter(thinker_mlp):
    """Adapt HF thinker experts for TTNNExperts (stacked tensors or ModuleList → consolidated tensors)."""
    hf_experts = thinker_mlp.experts
    layer_cfg = getattr(thinker_mlp, "config", None) or getattr(hf_experts, "config", None)

    if hasattr(hf_experts, "gate_up_proj") and hasattr(hf_experts, "down_proj"):
        cfg = getattr(hf_experts, "config", None)
        if cfg is None:
            cfg = type("ThinkerExpertsConfig", (), {})()
        cfg.hidden_size = getattr(cfg, "hidden_size", hf_experts.gate_up_proj.shape[2])
        cfg.moe_intermediate_size = getattr(cfg, "moe_intermediate_size", hf_experts.gate_up_proj.shape[1] // 2)
        cfg.n_routed_experts = getattr(cfg, "n_routed_experts", hf_experts.gate_up_proj.shape[0])
        cfg.num_experts_per_tok = (
            getattr(cfg, "num_experts_per_tok", None) or _thinker_gate_router_attrs(thinker_mlp)[0]
        )

        adapter = type("ThinkerExpertsAdapter", (), {})()
        adapter.gate_up_proj = hf_experts.gate_up_proj
        adapter.down_proj = hf_experts.down_proj
        adapter.config = cfg
        return adapter

    # Newer HF: nn.ModuleList of Qwen3OmniMoeThinkerTextMLP (per-expert gate/up/down Linear).
    if layer_cfg is None:
        layer_cfg = _thinker_config_fallback_from_modules(thinker_mlp)
    if getattr(layer_cfg, "moe_intermediate_size", None) is None and len(hf_experts):
        ex0 = hf_experts[0]
        if hasattr(ex0, "gate_proj"):
            mi = int(ex0.gate_proj.out_features)
            try:
                object.__setattr__(layer_cfg, "moe_intermediate_size", mi)
            except Exception:
                setattr(layer_cfg, "moe_intermediate_size", mi)
    consolidated = _consolidate_talker_experts_from_module_list(hf_experts, layer_cfg)
    if getattr(consolidated.config, "n_routed_experts", None) is None:
        consolidated.config.n_routed_experts = getattr(consolidated.config, "num_experts", len(hf_experts))
    if getattr(consolidated.config, "num_experts_per_tok", None) is None:
        consolidated.config.num_experts_per_tok = _thinker_gate_router_attrs(thinker_mlp)[0]
    return consolidated


class TTNNQwen3OmniMoeThinkerTextSparseMoeBlock(TTNNModule):
    """Thinker sparse MoE: torch gate + TTNNExperts (HF-compatible wrapper)."""

    @classmethod
    def from_torch(cls, thinker_mlp):
        """From Qwen3OmniMoeThinkerTextSparseMoeBlock."""
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        module.gate = thinker_mlp.gate
        experts_for_tt = _thinker_experts_adapter(thinker_mlp)
        module.experts = TTNNExperts.from_torch(experts_for_tt)
        return module

    def preprocess_weights_impl(self):
        self.experts.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.experts.move_weights_to_device()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        if not self._is_distributed:
            return tensor
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        """Run gate on torch, experts on TT; return torch tensor for downstream layers."""
        hidden_states_torch = _to_torch_any(hidden_states)
        x_flat = hidden_states_torch.reshape(-1, hidden_states_torch.shape[-1])
        with torch.no_grad():
            _, routing_weights, selected_experts = self.gate(x_flat)
        hidden_states_tt = _to_ttnn_raw(hidden_states)
        hidden_states_tt = self._maybe_all_gather(hidden_states_tt)
        if len(hidden_states_tt.shape) == 3:
            b, s, h = (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[1]), int(hidden_states_tt.shape[2]))
            hidden_states_tt = ttnn.reshape(hidden_states_tt, ttnn.Shape((b, 1, s, h)))
        else:
            b, s, h = (
                int(hidden_states_tt.shape[0]),
                int(hidden_states_tt.shape[2]),
                int(hidden_states_tt.shape[3]),
            )

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        topk_idx_tt = ttnn.from_torch(
            selected_experts.to(torch.int64),
            device=self.device,
            mesh_mapper=mesh_mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        topk_w_tt = ttnn.from_torch(
            routing_weights.to(torch.bfloat16),
            device=self.device,
            mesh_mapper=mesh_mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Call forward directly to avoid wrapping outputs into TorchTTNNTensor.
        expert_out = self.experts.forward(hidden_states_tt, topk_idx_tt, topk_w_tt)
        # Be defensive: forward may still return TorchTTNNTensor depending on internal ops.
        try:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(expert_out, TorchTTNNTensor):
                expert_out = expert_out.to_ttnn
        except Exception:
            pass
        expert_out = _to_ttnn_raw(expert_out)
        return ttnn.reshape(expert_out, ttnn.Shape((b, s, h)))


class TTNNQwen3OmniThinkerMoE(TTNNModule):
    """Thinker MoE: HF-style linear→softmax→top-k router on device; TTNNExperts dispatch/combine. Returns torch tensor for decoder."""

    @classmethod
    def from_torch(cls, thinker_mlp):
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        g = thinker_mlp.gate
        if not hasattr(g, "weight"):
            raise TypeError(f"Thinker MoE gate must expose ``weight`` (Linear or TopKRouter); got {type(g).__name__}")
        module._gate_w_torch = g.weight.data.clone()
        top_k, norm_topk_prob, num_experts = _thinker_gate_router_attrs(thinker_mlp)
        module.top_k = top_k
        module.norm_topk_prob = norm_topk_prob
        module.num_experts = num_experts
        experts_for_tt = _thinker_experts_adapter(thinker_mlp)
        module.experts = TTNNExperts.from_torch(experts_for_tt)
        return module

    def preprocess_weights_impl(self):
        self._gate_tt_host = preprocess_linear_weight(self._gate_w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        del self._gate_w_torch
        self.experts.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.gate_weight_tt = ttnn.to_device(self._gate_tt_host, self.device)
        self.experts.move_weights_to_device()

    def deallocate_weights_impl(self):
        gw = getattr(self, "gate_weight_tt", None)
        if gw is not None:
            ttnn.deallocate(gw)
            self.gate_weight_tt = None
        self.experts.deallocate_weights()

    @property
    def _is_distributed(self):
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        if not self._is_distributed:
            return tensor
        return ttnn.experimental.all_gather_async(
            tensor,
            dim=-1,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

    def _moe_from_tiled_4d(self, hidden_states_tile, b, s, h, orig_batch, out_dtype):
        """Run gate + experts on TILE activations (b, 1, s, h). Returns torch (b, s, hidden_size)."""
        t = b * s
        x_2d = ttnn.reshape(hidden_states_tile, ttnn.Shape((t, h)))
        gate_logits = ttnn.linear(x_2d, self.gate_weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        probs = ttnn.softmax(gate_logits, dim=-1)
        ttnn.deallocate(gate_logits)

        topk_vals, topk_idx = ttnn.topk(probs, k=self.top_k, dim=-1)
        ttnn.deallocate(probs)

        if self.norm_topk_prob:
            denom = ttnn.sum(topk_vals, dim=-1, keepdim=True)
            topk_vals = ttnn.div(topk_vals, denom)
            ttnn.deallocate(denom)

        topk_idx = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
        topk_vals = ttnn.to_layout(topk_vals, ttnn.ROW_MAJOR_LAYOUT)
        topk_idx = ttnn.reshape(topk_idx, ttnn.Shape((t, self.top_k)))
        topk_vals = ttnn.reshape(topk_vals, ttnn.Shape((t, self.top_k)))

        expert_out = self.experts.forward(hidden_states_tile, topk_idx, topk_vals)
        try:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(expert_out, TorchTTNNTensor):
                expert_out = expert_out.to_ttnn
        except Exception:
            pass
        expert_out = _to_ttnn_raw(expert_out)
        h_out = int(self.experts.hidden_size)
        expert_out = ttnn.reshape(expert_out, ttnn.Shape((b, s, h_out)))

        mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self.device.get_num_devices() > 1 else None
        out_torch = ttnn.to_torch(expert_out, mesh_composer=mesh_composer).to(out_dtype)
        ttnn.deallocate(expert_out)
        if mesh_composer is not None:
            out_torch = out_torch.narrow(0, 0, int(orig_batch))
        return out_torch

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        hidden_states_torch = _to_torch_any(hidden_states)
        orig_shape = hidden_states_torch.shape
        out_dtype = hidden_states_torch.dtype
        orig_batch = int(orig_shape[0])

        hidden_states_tt = _to_ttnn_raw(hidden_states)
        hidden_states_tt = self._maybe_all_gather(hidden_states_tt)
        if len(hidden_states_tt.shape) == 3:
            b, s, h = (int(hidden_states_tt.shape[0]), int(hidden_states_tt.shape[1]), int(hidden_states_tt.shape[2]))
            hidden_states_tt = ttnn.reshape(hidden_states_tt, ttnn.Shape((b, 1, s, h)))
        else:
            b, s, h = (
                int(hidden_states_tt.shape[0]),
                int(hidden_states_tt.shape[2]),
                int(hidden_states_tt.shape[3]),
            )

        seq_chunk = int(os.environ.get("TT_SYMBIOTE_MOE_SEQ_CHUNK", "1024"))
        if seq_chunk <= 0:
            seq_chunk = s + 1

        # Long prefill: drop full-sequence TILE if present (dense RM is usually smaller), then tile only each chunk.
        # Set TT_SYMBIOTE_MOE_SEQ_CHUNK=0 to force the single-shot path (legacy behavior).
        if s > seq_chunk:
            if hidden_states_tt.layout == ttnn.TILE_LAYOUT:
                hidden_rm = ttnn.to_layout(
                    hidden_states_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                try:
                    ttnn.deallocate(hidden_states_tt)
                except Exception:
                    pass
                hidden_states_tt = hidden_rm
            parts = []
            for s0 in range(0, s, seq_chunk):
                s1 = min(s0 + seq_chunk, s)
                sc = s1 - s0
                h_rm = ttnn.slice(hidden_states_tt, (0, 0, s0, 0), (b, 1, s1, h))
                h_tile = ttnn.to_layout(h_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                try:
                    ttnn.deallocate(h_rm)
                except Exception:
                    pass
                parts.append(self._moe_from_tiled_4d(h_tile, b, sc, h, orig_batch, out_dtype))
                try:
                    ttnn.deallocate(h_tile)
                except Exception:
                    pass
            out_torch = torch.cat(parts, dim=1)
            return out_torch.reshape(orig_shape)

        if hidden_states_tt.layout != ttnn.TILE_LAYOUT:
            hidden_states_tt = ttnn.to_layout(hidden_states_tt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out_torch = self._moe_from_tiled_4d(hidden_states_tt, b, s, h, orig_batch, out_dtype)
        return out_torch.reshape(orig_shape)
