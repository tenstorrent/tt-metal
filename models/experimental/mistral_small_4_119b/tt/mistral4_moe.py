# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 Mixture-of-Experts (MoE) layer — prefill mode, no PyTorch fallback.

Architecture:
  • 128 routed experts  (Mistral4NaiveMoe)
  • 4 active experts per token  (Mistral4TopkRouter)
  • 1 shared expert  (Mistral4MLP, always active)
  • Each expert / shared expert: gate_proj + up_proj → SiLU gate → down_proj
    dimensions: HIDDEN_SIZE(4096) → EXPERT_INTERMEDIATE_SIZE(2048) → HIDDEN_SIZE(4096)

Sharding strategy for N-device mesh:
  Expert weights are sharded along the expert dimension (dim=0):
    device k holds experts  [ k * experts_per_device : (k+1) * experts_per_device ]
  Each device computes its local experts' weighted outputs, then an
  all_gather + sum across devices produces the full MoE output.

  Shared-expert and gate weights are replicated on every device.

Routing:
  use_ttnn_moe=True   (default):
    Gate logits and softmax computed on device via TTNN;
    top-k indices are pulled to host only to build the routing-weight mask
    (a small [seq, num_experts] gather), which is pushed back as a
    device-sharded tensor for the expert loop.

  use_ttnn_moe=False  (host routing / reference):
    Gate logits computed on device; top-k + softmax on host; routing mask
    pushed back to device.  Useful for PCC comparison runs.

  moe_hf_torch_routing=True (for PCC validation only):
    HF model router is called externally; caller supplies routing_weights_tt.
    Not used in normal inference.

Weight loading:
  Hugging Face ``Mistral4NaiveMoe`` (Mistral-Small-4) uses fused parameters:
      ``mlp.experts.gate_up_proj``  [num_experts, 2 * intermediate, hidden]
      ``mlp.experts.down_proj``     [num_experts, hidden, intermediate]
  We split ``gate_up_proj`` for gate vs up.  Alternatively supports stacked
  ``mlp.experts.gate_proj.weight`` / per-expert ``mlp.experts.{i}.gate_proj.weight``.
"""

from __future__ import annotations


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    EXPERT_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    NUM_ACTIVE_EXPERTS,
    NUM_EXPERTS,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
)


# ── Weight loading helpers ─────────────────────────────────────────────────


def _bf16(t: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """Cast to bfloat16, dequantizing FP8 weights if scale_inv is provided.

    scale_inv may be scalar () or per-expert [N]; it is reshaped to broadcast
    against the weight tensor correctly.
    """
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.to(torch.float32)
        if scale_inv is not None:
            s = scale_inv.to(torch.float32)
            # Reshape for broadcasting: expand trailing dims to match weight ndim
            while s.dim() < t.dim():
                s = s.unsqueeze(-1)
            t = t * s
    return t.to(torch.bfloat16).contiguous()


def _load_stacked_experts(
    state_dict: dict,
    prefix: str,
    proj_name: str,
    num_experts: int,
    mesh_device: ttnn.MeshDevice,
    num_devices: int,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """
    Load stacked expert projection weights to device, sharded along expert axis.

    HF ``Mistral4NaiveMoe`` (default for Mistral-Small-4 checkpoints):
      ``{prefix}experts.gate_up_proj`` — split for ``proj_name`` ``gate_proj`` / ``up_proj``
      ``{prefix}experts.down_proj`` — used when ``proj_name`` is ``down_proj``

    Legacy / alternate formats:
      ``{prefix}experts.{proj_name}.weight``  [num_experts, out_features, in_features]
      ``{prefix}experts.{proj_name}``         same, as ``nn.Parameter`` (no ``.weight`` suffix)
      Per-expert: ``{prefix}experts.{i}.{proj_name}.weight``  [out_features, in_features]

    Returns ttnn tensor sharded on dim=0 across devices:
      each device holds [num_experts // num_devices, in_features, out_features]
    """
    half = EXPERT_INTERMEDIATE_SIZE
    gate_up_key = f"{prefix}experts.gate_up_proj"

    if proj_name in ("gate_proj", "up_proj") and gate_up_key in state_dict:
        scale_inv = state_dict.get(f"{prefix}experts.gate_up_proj_scale_inv")
        gu = _bf16(state_dict[gate_up_key], scale_inv)
        if gu.shape[0] != num_experts or gu.shape[1] != 2 * half:
            raise ValueError(f"{gate_up_key}: expected shape ({num_experts}, {2 * half}, *), got {tuple(gu.shape)}")
        w = gu[:, :half, :] if proj_name == "gate_proj" else gu[:, half:, :]
        w = w.permute(0, 2, 1).contiguous()  # [num_experts, in, out] = [E, H, I]
    else:
        stacked_key = f"{prefix}experts.{proj_name}.weight"
        bare_key = f"{prefix}experts.{proj_name}"

        if stacked_key in state_dict:
            scale_inv = state_dict.get(stacked_key.replace(".weight", ".weight_scale_inv"))
            w = _bf16(state_dict[stacked_key], scale_inv)
            w = w.permute(0, 2, 1).contiguous()
        elif bare_key in state_dict:
            scale_inv = state_dict.get(f"{bare_key}_scale_inv")
            w = _bf16(state_dict[bare_key], scale_inv)
            w = w.permute(0, 2, 1).contiguous()
        else:
            experts = []
            for i in range(num_experts):
                key = f"{prefix}experts.{i}.{proj_name}.weight"
                if key not in state_dict:
                    key = f"{prefix}experts.{i}.{proj_name}"
                scale_inv = state_dict.get(key.replace(".weight", ".weight_scale_inv"))
                experts.append(_bf16(state_dict[key], scale_inv).T.contiguous())  # [in, out]
            w = torch.stack(experts, dim=0)  # [num_experts, in, out]

    # For ttnn batched matmul we need 4D: [num_experts, 1, in, out]
    w = w.unsqueeze(1)  # [num_experts, 1, in, out]

    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _load_replicated(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    scale_inv = state_dict.get(key.replace(".weight", ".weight_scale_inv"))
    w = _bf16(state_dict[key], scale_inv)
    if transpose:
        w = w.T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _load_norm_weight_1d(
    state_dict: dict,
    key: str,
    dim: int,
    mesh_device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    w = _bf16(state_dict[key]).reshape(1, 1, dim)
    return ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ── All-reduce helper for 2-device mesh ────────────────────────────────────


def _all_reduce_sum(
    partial: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    num_devices: int,
) -> ttnn.Tensor:
    """
    All-reduce (sum) across all devices in the mesh.

    Each device has [1, 1, seq, hidden] partial result.
    After all_gather(dim=0): [num_devices, 1, seq, hidden]
    After sum(dim=0):        [1, 1, seq, hidden]  (full result, on every device)
    """
    if num_devices == 1:
        return partial

    gathered = ttnn.all_gather(
        partial,
        dim=0,
        num_links=1,
        topology=ttnn.Topology.Ring,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [num_devices, 1, seq, hidden]

    full = ttnn.sum(gathered, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(gathered)
    # sum reduces dim 0 → [1, seq, hidden]; reshape back to [1, 1, seq, hidden]
    seq_len = partial.shape[2]
    full = ttnn.reshape(full, [1, 1, seq_len, HIDDEN_SIZE])
    return full


# ── Shared (always-active) Expert MLP ─────────────────────────────────────


class TtMistral4SharedMLP(LightweightModule):
    """Always-active shared expert: gate_proj / up_proj / down_proj."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        prefix: str,
        intermediate_size: int,
        dtype: ttnn.DataType,
        compute_kernel_config,
    ):
        super().__init__()
        self.compute_kernel_config = compute_kernel_config

        self.gate_proj = _load_replicated(
            state_dict,
            prefix + "gate_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [HIDDEN_SIZE, intermediate_size]

        self.up_proj = _load_replicated(
            state_dict,
            prefix + "up_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [HIDDEN_SIZE, intermediate_size]

        self.down_proj = _load_replicated(
            state_dict,
            prefix + "down_proj.weight",
            transpose=True,
            dtype=dtype,
            mesh_device=mesh_device,
        )  # [intermediate_size, HIDDEN_SIZE]

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:  x: [1, 1, seq, HIDDEN_SIZE]
        Returns:   [1, 1, seq, HIDDEN_SIZE]
        """
        gate = ttnn.linear(
            x,
            self.gate_proj,
            activation="silu",
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.multiply(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out


# ── Main MoE Layer ─────────────────────────────────────────────────────────


class TtMistral4MoELayer(LightweightModule):
    """
    Mistral4 MoE: 128 routed experts (device-sharded) + 1 shared expert.

    For a 2-device mesh [1, 2]:
      - experts_per_device = 64
      - Device 0: experts 0..63  |  Device 1: experts 64..127
      - Routing weights are determined on host (host top-k), pushed back as
        a device-local sharded tensor [1, 1, seq, experts_per_device].
      - Each device sums its local experts' weighted outputs.
      - All-reduce (all_gather + sum) yields the full routed output.
      - Shared expert is added on both devices (replicated).
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        use_ttnn_moe: bool = True,
        moe_hf_torch_routing: bool = False,
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.num_experts = NUM_EXPERTS
        self.num_active = NUM_ACTIVE_EXPERTS
        self.use_ttnn_moe = use_ttnn_moe
        self.moe_hf_torch_routing = moe_hf_torch_routing

        assert self.num_experts % self.num_devices == 0, (
            f"num_experts ({self.num_experts}) must be divisible by " f"num_devices ({self.num_devices})"
        )
        self.experts_per_device = self.num_experts // self.num_devices

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        mlp_prefix = layer_prefix + "mlp."

        # ── Gate (router) weight ───────────────────────────────────────────
        # shape: [num_experts, HIDDEN_SIZE] → [HIDDEN_SIZE, num_experts] for matmul
        self.gate_weight = _load_replicated(
            state_dict,
            mlp_prefix + "gate.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )  # [HIDDEN_SIZE, NUM_EXPERTS]

        # Gate correction bias (additive, applied before softmax; uploaded to device or None)
        gate_bias_key = mlp_prefix + "gate.e_score_correction_bias"
        if gate_bias_key in state_dict:
            gate_bias_t = state_dict[gate_bias_key].to(torch.bfloat16).reshape(1, 1, 1, -1)
            self.gate_bias_tt = ttnn.as_tensor(
                gate_bias_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )  # [1, 1, 1, NUM_EXPERTS] replicated on all devices
        else:
            self.gate_bias_tt = None

        # ── Stacked routed expert weights (sharded on expert axis) ─────────
        self.expert_gate = _load_stacked_experts(
            state_dict,
            mlp_prefix,
            "gate_proj",
            NUM_EXPERTS,
            mesh_device,
            self.num_devices,
            expert_dtype,
        )  # per device: [experts_per_device, 1, HIDDEN_SIZE, intermediate]

        self.expert_up = _load_stacked_experts(
            state_dict,
            mlp_prefix,
            "up_proj",
            NUM_EXPERTS,
            mesh_device,
            self.num_devices,
            expert_dtype,
        )  # per device: [experts_per_device, 1, HIDDEN_SIZE, intermediate]

        self.expert_down = _load_stacked_experts(
            state_dict,
            mlp_prefix,
            "down_proj",
            NUM_EXPERTS,
            mesh_device,
            self.num_devices,
            expert_dtype,
        )  # per device: [experts_per_device, 1, intermediate, HIDDEN_SIZE]

        # ── Shared expert ──────────────────────────────────────────────────
        self.shared_expert = TtMistral4SharedMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            prefix=mlp_prefix + "shared_experts.",
            intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
        )

    # ── Routing ────────────────────────────────────────────────────────────

    def _compute_routing_weights(
        self,
        x: ttnn.Tensor,
        seq_len: int,
    ) -> ttnn.Tensor:
        """
        Compute per-device routing-weight tensors — fully on device.

        Returns a TTNN tensor sharded on dim=3 across devices:
          each device's slice: [1, 1, seq_len, experts_per_device]
          device k holds the weights for its local experts k*E..(k+1)*E-1.

        Routing algorithm (all TTNN ops):
          1. Gate logits on device (HiFi2).
          2. Add e_score_correction_bias if present (replicated device tensor).
          3. Softmax over all experts (on device).
          4. TopK on device → values + indices.
          5. Sum-normalize top-k weights on device.
          6. Scatter into dense [1, 1, seq, NUM_EXPERTS] on device.
          7. To-host / re-upload only for sharding along expert axis (data movement, no compute).
        """
        # 1. Gate logits on device: [1, 1, seq, NUM_EXPERTS]
        gate_logits_tt = ttnn.linear(
            x,
            self.gate_weight,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 2. Apply correction bias if present (device tensor, broadcast [1,1,1,E] → [1,1,S,E])
        if self.gate_bias_tt is not None:
            gate_logits_tt = ttnn.add(gate_logits_tt, self.gate_bias_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # 3. Softmax over all experts on device
        probs_tt = ttnn.softmax(gate_logits_tt, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_logits_tt)

        # 4. TopK on device → values [1,1,seq,k], indices [1,1,seq,k]
        topk_vals_tt, topk_idx_tt = ttnn.topk(probs_tt, k=self.num_active, dim=-1)

        # 5. Sum-normalize top-k weights on device
        topk_sum_tt = ttnn.sum(topk_vals_tt, dim=-1, keepdim=True)
        topk_vals_tt = ttnn.div(topk_vals_tt, topk_sum_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_sum_tt)

        # 6. Scatter into dense [1, 1, seq, NUM_EXPERTS] on device
        dense_routing_tt = ttnn.scatter(
            ttnn.zeros_like(probs_tt),
            dim=-1,
            index=topk_idx_tt,
            src=topk_vals_tt,
        )
        ttnn.deallocate(probs_tt)
        ttnn.deallocate(topk_vals_tt)
        ttnn.deallocate(topk_idx_tt)

        # 7. Re-shard along expert axis so each device sees only its local expert slice.
        #    This requires a host round-trip (data movement only, no compute on host).
        routing_host = ttnn.to_torch(
            dense_routing_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )[
            0:1
        ].to(torch.bfloat16)
        ttnn.deallocate(dense_routing_tt)

        return ttnn.as_tensor(
            routing_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=3),
        )

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:  x: [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        Returns:   [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        """
        seq_len = x.shape[2]

        # ── Routing weights ────────────────────────────────────────────────
        routing_weights = self._compute_routing_weights(x, seq_len)
        # routing_weights: [1, 1, seq, experts_per_device] per device

        # ── Dense expert loop (Python-level, TTNN compute) ─────────────────
        # Input must be replicated; gate/up weight shape per expert:
        #   [1, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE]
        # Expand x to [experts_per_device, 1, seq, HIDDEN] for bmm, or loop.
        #
        # We use a Python loop over local experts (64 per device) for clarity.
        # Each iter: gate(x), up(x), silu*up, down → scale by routing weight.

        partial = None  # accumulates weighted expert outputs on each device

        for local_e in range(self.experts_per_device):
            # Routing weight for this expert: [1, 1, seq, 1]
            w_e = ttnn.slice(
                routing_weights,
                [0, 0, 0, local_e],
                [1, 1, seq_len, local_e + 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Check if any token routes to this expert (all-zero weight → skip)
            # For correctness, always compute (skip optimisation for bring-up).

            # Expert weights for local_e: slice [local_e:local_e+1, 1, in, out]
            gate_w = ttnn.slice(
                self.expert_gate,
                [local_e, 0, 0, 0],
                [local_e + 1, 1, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, 1, HIDDEN_SIZE, intermediate]
            gate_w = ttnn.reshape(gate_w, [HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE])

            up_w = ttnn.slice(
                self.expert_up,
                [local_e, 0, 0, 0],
                [local_e + 1, 1, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            up_w = ttnn.reshape(up_w, [HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE])

            down_w = ttnn.slice(
                self.expert_down,
                [local_e, 0, 0, 0],
                [local_e + 1, 1, EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            down_w = ttnn.reshape(down_w, [EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE])

            # Expert forward: SiLU(gate_proj(x)) * up_proj(x) → down_proj
            gate_out = ttnn.linear(
                x,
                gate_w,
                activation="silu",
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, 1, seq, intermediate]
            ttnn.deallocate(gate_w)

            up_out = ttnn.linear(
                x,
                up_w,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, 1, seq, intermediate]
            ttnn.deallocate(up_w)

            hidden = ttnn.multiply(gate_out, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(gate_out)
            ttnn.deallocate(up_out)

            expert_out = ttnn.linear(
                hidden,
                down_w,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, 1, seq, HIDDEN_SIZE]
            ttnn.deallocate(hidden)
            ttnn.deallocate(down_w)

            # Scale by routing weight (broadcast [1,1,seq,1] → [1,1,seq,H])
            weighted = ttnn.multiply(expert_out, w_e, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(expert_out)
            ttnn.deallocate(w_e)

            if partial is None:
                partial = weighted
            else:
                partial = ttnn.add(partial, weighted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(weighted)

        ttnn.deallocate(routing_weights)

        # If all routing weights are zero (edge case with experts_per_device > num_active)
        if partial is None:
            partial = ttnn.zeros_like(x)

        # ── All-reduce routed expert outputs across devices ─────────────────
        routed_out = _all_reduce_sum(partial, self.mesh_device, self.num_devices)
        if partial is not routed_out:
            ttnn.deallocate(partial)

        # ── Shared expert (always active, replicated) ───────────────────────
        shared_out = self.shared_expert.forward(x)

        # Final MoE output = routed + shared
        out = ttnn.add(routed_out, shared_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(routed_out)
        ttnn.deallocate(shared_out)

        return out  # [1, 1, seq, HIDDEN_SIZE] replicated on all devices
