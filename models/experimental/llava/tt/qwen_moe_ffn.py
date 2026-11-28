# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen1-MoE feed-forward for MoE-LLaVA-Qwen-1.8B-4e on n150 (1 chip) and n300 (2 chips).

Why this exists: the shared tt_transformers MoE path (``TtMoeLayer`` +
``TtMixtralMLP``) is Mixtral-specific. It asserts ``num_devices == 8``, hardcodes
8 experts with 4096/14336 shapes, and ``TransformerBlock`` turns it on for
*every* layer when ``args.is_mixture_of_experts`` is set. MoE-LLaVA-Qwen has 4
experts (top-2) on the even layers only, and dense SwiGLU on the odd layers.

Design (kept inside models/experimental so shared code is untouched):

* ``QwenMoeTransformerBlock`` is passed to ``Transformer(block_class=...)``. Dense
  layers are built exactly as before. MoE layers swap in ``TtQwenMoeFFN``.
* Experts are split evenly across devices: ``L = num_experts // num_devices``
  local "slots" per device (n150: 4 slots, n300: 2 slots). Slot ``k`` on device
  ``d`` holds global expert ``d * L + k``.
* Routing reuses the proven Mixtral device pattern (``topk`` over 64 padded
  logits, top-2 mask, ``softmax``, ``eqz``). For each slot, the router columns
  are permuted per device so that slot's expert sits at index 0, so
  ``eqz(topk_indices)`` selects exactly that expert. The softmax over the top-2
  logits equals the reference routing in reference/qwen_moe_reference.py
  (``softmax(topk(logits, 2).values)``).
* Each device computes ``sum_k w_k * expert_k(x)`` for its local experts. Then
  ``tt_all_reduce(cluster_axis=0, dim=3)`` sums across devices. On n300 that
  reduce-scatters into the same fractured residual layout as the dense ``MLP``.
  On n150 it is a no-op.

Every token is run through all local experts (dense compute, masked by routing
weight). For 4 small experts this costs 2x the FLOPs of sparse top-2 dispatch,
but it needs no data-dependent gather/scatter, so it stays trace-friendly.
"""

from contextlib import contextmanager

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt import decoder as tt_decoder_module
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.decoder import TransformerBlock

_ROUTER_PAD = 64  # ttnn.topk needs a last dim of at least 64
_TOPK_WIDTH = 32  # topk output width (one tile), as in tt_transformers mixtral_moe.py


class TtQwenMoeFFN(LightweightModule):
    """Top-k-of-E MoE FFN with experts split across a 1D mesh (n150 / n300)."""

    def __init__(self, mesh_device, tt_ccl, args, state_dict, layer_num, dtype, num_experts, num_experts_per_tok):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num
        self.num_devices = mesh_device.get_num_devices()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts % self.num_devices != 0:
            raise ValueError(f"num_experts={num_experts} must divide evenly across {self.num_devices} devices")
        if num_experts > _ROUTER_PAD:
            raise ValueError(f"num_experts={num_experts} exceeds router pad width {_ROUTER_PAD}")
        if not 1 <= self.top_k <= num_experts:
            raise ValueError(f"num_experts_per_tok={self.top_k} out of range for {num_experts} experts")
        self.local_experts = num_experts // self.num_devices

        prefix = f"layers.{layer_num}.block_sparse_moe"
        cache_dir = None if args.dummy_weights else args.weight_cache_path(dtype)

        def cache_name(name):
            if cache_dir is None:
                return None
            return cache_dir / f"{prefix}.qwen_moe_d{self.num_devices}.{name}"

        # ---- Router: one [dim, 64] gate per (slot, device), slot's expert moved to column 0 ----
        gate_w = state_dict[f"{prefix}.gate.weight"]  # [E, dim]
        assert gate_w.shape[0] == num_experts, f"gate has {gate_w.shape[0]} experts, expected {num_experts}"
        gate_padded = torch.nn.functional.pad(gate_w.permute(1, 0), (0, _ROUTER_PAD - num_experts), value=0.0)
        gate_padded = gate_padded.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, 64]

        self.gates = []
        for k in range(self.local_experts):
            per_device = []
            for d in range(self.num_devices):
                g = d * self.local_experts + k
                perm = gate_padded.clone()
                perm[..., [0, g]] = perm[..., [g, 0]]
                per_device.append(perm)
            self.gates.append(
                ttnn.as_tensor(
                    torch.cat(per_device, dim=1),  # [1, D, dim, 64] -> each device gets [1, 1, dim, 64]
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
                    cache_file_name=cache_name(f"gate.slot{k}"),
                )
            )

        # Padded router columns (experts >= E) must never win top-k.
        router_mask = torch.full((1, 1, 1, _ROUTER_PAD), torch.finfo(torch.float).min)
        router_mask[..., :num_experts] = 0.0
        self.router_mask = self._replicated_row(router_mask)

        # Keep only the top_k entries of the 32-wide topk output.
        topk_mask = torch.full((1, 1, 1, _TOPK_WIDTH), torch.finfo(torch.float).min)
        topk_mask[..., : self.top_k] = 0.0
        self.topk_mask = self._replicated_row(topk_mask)

        # ---- Expert weights: slot k -> [D, 1, in, out], device d gets global expert d*L + k ----
        def expert_weight(k, name):
            return torch.cat(
                [
                    state_dict[f"{prefix}.experts.{d * self.local_experts + k}.{name}.weight"]
                    .permute(1, 0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    for d in range(self.num_devices)
                ],
                dim=0,
            )

        def as_expert_tensor(k, name):
            return ttnn.as_tensor(
                expert_weight(k, name),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
                cache_file_name=cache_name(f"experts.slot{k}.{name}"),
            )

        # Meta naming (see qwen1_moe_checkpoint.py): w1 = gate (silu), w3 = up, w2 = down.
        self.w1 = [as_expert_tensor(k, "w1") for k in range(self.local_experts)]
        self.w3 = [as_expert_tensor(k, "w3") for k in range(self.local_experts)]
        self.w2 = [as_expert_tensor(k, "w2") for k in range(self.local_experts)]

        self.router_kernel = args.compute_kernel_config_hifi4
        self.mlp_kernel = args.compute_kernel_config_hifi2

    def _replicated_row(self, row):
        t = ttnn.from_torch(
            row,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # Same trick as mixtral_moe.py: reduce the tile-padded rows so the mask broadcasts over S.
        return ttnn.sum(t, dim=2, keepdim=True)

    def _routing_weight(self, x, k):
        """Routing weight [1, 1, S, 1] of this device's slot-k expert (0 when not in the top-k)."""
        logits = ttnn.linear(
            x,
            self.gates[k],
            compute_kernel_config=self.router_kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.add(logits, self.router_mask)
        topk_values, topk_indices = ttnn.topk(logits, _TOPK_WIDTH)
        logits.deallocate(True)
        topk_values = ttnn.add(topk_values, self.topk_mask)
        probs = ttnn.softmax(topk_values, dim=-1)
        is_slot = ttnn.typecast(ttnn.eqz(topk_indices), dtype=ttnn.bfloat16)
        weight = ttnn.sum(ttnn.mul(probs, is_slot), dim=3, keepdim=True)
        for t in (topk_values, topk_indices, probs, is_slot):
            t.deallocate(True)
        return weight

    def _expert(self, x, k):
        gate = ttnn.linear(
            x,
            self.w1[k],
            compute_kernel_config=self.mlp_kernel,
            dtype=ttnn.bfloat16,
            activation="silu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.w3[k],
            compute_kernel_config=self.mlp_kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(
            hidden,
            self.w2[k],
            compute_kernel_config=self.mlp_kernel,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden.deallocate(True)
        return out

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        # Decode inputs arrive L1-sharded; this layer works on DRAM-interleaved tensors.
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        acc = None
        for k in range(self.local_experts):
            weight = self._routing_weight(x, k)
            y = self._expert(x, k)
            y_weighted = ttnn.mul(y, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y.deallocate(True)
            weight.deallocate(True)
            if acc is None:
                acc = y_weighted
            else:
                new_acc = ttnn.add(acc, y_weighted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                acc.deallocate(True)
                y_weighted.deallocate(True)
                acc = new_acc

        # Sum expert contributions across devices. On n300 this reduce-scatters on dim 3,
        # matching the fractured residual layout the dense MLP returns. No-op on n150.
        out = tt_all_reduce(
            acc,
            self.mesh_device,
            tt_ccl=self.tt_ccl,
            cluster_axis=0,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            topology=self.args.ccl_topology(),
        )
        shape = out.shape
        out = ttnn.reshape(out, (1, 1, shape[-4] * shape[-3] * shape[-2], shape[-1]))

        if mode == Mode.DECODE:
            out = ttnn.to_memory_config(out, self.args.get_mlp_output_mem_config(mode, None))
        return out


class _DeferredFeedForward:
    """Stand-in for the dense MLP while a MoE layer's TransformerBlock is constructed."""

    def __init__(self, *args, **kwargs):
        pass


@contextmanager
def _skip_dense_mlp():
    # TransformerBlock always builds a dense MLP when is_mixture_of_experts is False, which
    # would KeyError on MoE layers (they have no feed_forward.* weights). Swap the MLP class
    # for a no-op only while that one block is constructed; the real FFN is set afterwards.
    original = tt_decoder_module.MLP
    tt_decoder_module.MLP = _DeferredFeedForward
    try:
        yield
    finally:
        tt_decoder_module.MLP = original


class QwenMoeTransformerBlock(TransformerBlock):
    """TransformerBlock that uses TtQwenMoeFFN on ``args.qwen_moe_layer_ids`` and dense MLP elsewhere.

    Requires ``args.qwen_moe_layer_ids``, ``args.qwen_num_experts`` and
    ``args.qwen_num_experts_per_tok`` (set by llm_wrapper.create_qwen_moe_llm),
    and ``args.is_mixture_of_experts`` False so the base class never builds the
    Mixtral-only TtMoeLayer.
    """

    def __init__(self, args, mesh_device, tt_ccl, dtype, state_dict, layer_num, weight_cache_path, **kwargs):
        assert not getattr(
            args, "is_mixture_of_experts", False
        ), "Leave args.is_mixture_of_experts False; it selects the Mixtral-only (8-device) TtMoeLayer for every layer"
        is_moe = layer_num in args.qwen_moe_layer_ids
        if not is_moe:
            super().__init__(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_num,
                weight_cache_path=weight_cache_path,
                **kwargs,
            )
            return

        with _skip_dense_mlp():
            super().__init__(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_num,
                weight_cache_path=weight_cache_path,
                **kwargs,
            )
        self.is_mixture_of_experts = True
        self.feed_forward = TtQwenMoeFFN(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
            num_experts=args.qwen_num_experts,
            num_experts_per_tok=args.qwen_num_experts_per_tok,
        )
