# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni thinker MoE TTNN modules."""

import os

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule, DeviceArch, run_on_devices
from models.experimental.tt_symbiote.models.qwen_omni.qwen_omni_modules import _to_ttnn_raw
from models.experimental.tt_symbiote.modules.moe import TTNNExperts, _to_torch_any


def _thinker_experts_adapter(thinker_mlp):
    """Adapt HF thinker experts for TTNNExperts (needs config + gate_up/down tensors)."""
    hf_experts = thinker_mlp.experts
    cfg = getattr(hf_experts, "config", None)
    if cfg is None:
        cfg = type("ThinkerExpertsConfig", (), {})()
    cfg.hidden_size = getattr(cfg, "hidden_size", hf_experts.gate_up_proj.shape[2])
    cfg.moe_intermediate_size = getattr(cfg, "moe_intermediate_size", hf_experts.gate_up_proj.shape[1] // 2)
    cfg.n_routed_experts = getattr(cfg, "n_routed_experts", hf_experts.gate_up_proj.shape[0])
    cfg.num_experts_per_tok = getattr(cfg, "num_experts_per_tok", None) or getattr(thinker_mlp.gate, "top_k", 8)

    adapter = type("ThinkerExpertsAdapter", (), {})()
    adapter.gate_up_proj = hf_experts.gate_up_proj
    adapter.down_proj = hf_experts.down_proj
    adapter.config = cfg
    return adapter


class TTNNQwen3OmniThinkerMoE(TTNNModule):
    """Thinker MoE: HF-style linear→softmax→top-k router on device; TTNNExperts dispatch/combine. Returns torch tensor for decoder."""

    @classmethod
    def from_torch(cls, thinker_mlp):
        module = cls()
        module._fallback_torch_layer = thinker_mlp
        g = thinker_mlp.gate
        module._gate_w_torch = g.weight.data.clone()
        module.top_k = int(g.top_k)
        module.norm_topk_prob = bool(g.norm_topk_prob)
        module.num_experts = int(g.num_experts)
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
