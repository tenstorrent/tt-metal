# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 Mixture-of-Experts (MoE) layer — fully on device, no host fallback.

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

Routing (fully on device):
  Gate logits, softmax, top-k, sum-normalisation, and scatter into a dense
  [1, 1, seq, NUM_EXPERTS] weight tensor are all TTNN ops.
  A pre-computed per-device block-column selection matrix (routing_shard_proj,
  created once at __init__ via ShardTensorToMesh) projects the replicated
  [1,1,seq,128] routing weights to a per-device [1,1,seq,EPD] slice on device
  via a single ttnn.matmul — no host round-trip in the forward path.

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

    if proj_name == "gate_up" and gate_up_key in state_dict:
        # Fused gate+up: load full [E, 2*I, H] and permute to [E, H, 2*I].
        # Caller slices [:I] for gate, [I:] for up after the matmul.
        scale_inv = state_dict.get(f"{prefix}experts.gate_up_proj_scale_inv")
        gu = _bf16(state_dict[gate_up_key], scale_inv)
        if gu.shape[0] != num_experts or gu.shape[1] != 2 * half:
            raise ValueError(f"{gate_up_key}: expected shape ({num_experts}, {2 * half}, *), got {tuple(gu.shape)}")
        w = gu.permute(0, 2, 1).contiguous()  # [E, H, 2*I]
    elif proj_name in ("gate_proj", "up_proj") and gate_up_key in state_dict:
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
        self.intermediate_size = intermediate_size

        # Fused gate+up weight: swiglu(z) = z[:I] * SiLU(z[I:])
        # Place up in the first half and gate in the second so that
        # swiglu produces SiLU(gate_proj(x)) * up_proj(x).
        gate_w = _bf16(
            state_dict[prefix + "gate_proj.weight"],
            state_dict.get(prefix + "gate_proj.weight_scale_inv"),
        )  # [I, H]
        up_w = _bf16(
            state_dict[prefix + "up_proj.weight"],
            state_dict.get(prefix + "up_proj.weight_scale_inv"),
        )  # [I, H]
        gate_up_w = torch.cat([up_w, gate_w], dim=0).T.contiguous()  # [H, 2I]
        self.gate_up_proj = ttnn.as_tensor(
            gate_up_w,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )  # [HIDDEN_SIZE, 2 * intermediate_size]

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
        seq_len = x.shape[2]
        I = self.intermediate_size
        gate_up = ttnn.linear(
            x,
            self.gate_up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, 2I] — up in [0:I], gate in [I:2I]

        # Split along the last dim (tile-aligned at I=2048); avoids ttnn.swiglu which
        # pads the seq dim to tile size and breaks non-tile-aligned seq lengths.
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, seq_len, I], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.slice(gate_up, [0, 0, 0, I], [1, 1, seq_len, 2 * I], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_up)

        gate_silu = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        hidden = ttnn.multiply(gate_silu, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_silu)
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

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode variant: all activations in L1 (seq=1 tensors are tiny)."""
        _mem = ttnn.L1_MEMORY_CONFIG
        I = self.intermediate_size
        gate_up = ttnn.linear(
            x,
            self.gate_up_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
        )
        up = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, 1, I], memory_config=_mem)
        gate = ttnn.slice(gate_up, [0, 0, 0, I], [1, 1, 1, 2 * I], memory_config=_mem)
        ttnn.deallocate(gate_up)
        gate_silu = ttnn.silu(gate, memory_config=_mem)
        ttnn.deallocate(gate)
        hidden = ttnn.multiply(gate_silu, up, memory_config=_mem)
        ttnn.deallocate(gate_silu)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
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
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.num_experts = NUM_EXPERTS
        self.num_active = NUM_ACTIVE_EXPERTS

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
        # LoFi is safe for bf4 expert weights: quantization error (~0.0625) dominates
        # HiFi2's extra FPU precision, so halving FPU cycles has no meaningful PCC cost.
        self.expert_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        mlp_prefix = layer_prefix + "mlp."

        # ── Gate (router) weight ───────────────────────────────────────────
        # shape: [num_experts, HIDDEN_SIZE] → [HIDDEN_SIZE, num_experts] for matmul
        # bfloat8_b: saves 18 MB / 12 banks ≈ 1.5 MB/bank across 36 layers.
        self.gate_weight = _load_replicated(
            state_dict,
            mlp_prefix + "gate.weight",
            transpose=True,
            dtype=ttnn.bfloat8_b,
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

        # ── Per-device routing shard selector (one-time init) ──────────────
        # At runtime, _compute_routing_weights produces a replicated dense
        # [1, 1, seq, NUM_EXPERTS] routing tensor.  To give each device only
        # its local EPD columns without a host round-trip, we pre-build a
        # per-device block-column selection matrix [1, 1, NUM_EXPERTS, EPD]:
        #   device k's matrix has ones at rows [k*EPD : (k+1)*EPD], zeros elsewhere.
        # ttnn.matmul([1,1,seq,E], [1,1,E,EPD]) → [1,1,seq,EPD] per device.
        sel = torch.zeros(
            self.num_devices,
            1,
            self.num_experts,
            self.experts_per_device,
            dtype=torch.bfloat16,
        )
        for k in range(self.num_devices):
            for i in range(self.experts_per_device):
                sel[k, 0, k * self.experts_per_device + i, i] = 1.0
        self.routing_shard_proj = ttnn.as_tensor(
            sel,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )  # per device: [1, 1, NUM_EXPERTS, experts_per_device]

        # ── Stacked routed expert weights (sharded on expert axis) ─────────
        self.expert_gate_up = _load_stacked_experts(
            state_dict,
            mlp_prefix,
            "gate_up",
            NUM_EXPERTS,
            mesh_device,
            self.num_devices,
            expert_dtype,
        )  # per device: [experts_per_device, 1, HIDDEN_SIZE, 2 * intermediate]

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
        # bfloat8_b: 36 layers × 3 shared-expert weights × 8 MB = 0.86 GB vs 1.73 GB at bf16.
        self.shared_expert = TtMistral4SharedMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            prefix=mlp_prefix + "shared_experts.",
            intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
            dtype=ttnn.bfloat8_b,
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
        Compute per-device routing-weight tensors — fully on device, no host round-trip.

        Returns [1, 1, seq_len, experts_per_device] per device:
          device k holds the weights for its local experts k*EPD..(k+1)*EPD-1.

        Routing algorithm (all TTNN ops):
          1. Gate logits on device (HiFi2).
          2. Add e_score_correction_bias if present (replicated device tensor).
          3. Softmax over all experts (on device).
          4. TopK on device → values + indices.
          5. Sum-normalize top-k weights on device.
          6. Scatter into dense [1, 1, seq, NUM_EXPERTS] on device.
          7. matmul with pre-computed per-device block-column selector → [1,1,seq,EPD].
        """
        # 1. Gate logits on device: [1, 1, seq, NUM_EXPERTS]
        gate_logits_tt = ttnn.linear(
            x,
            self.gate_weight,
            compute_kernel_config=self.compute_kernel_config,
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

        # 7. Project [1,1,seq,E] → [1,1,seq,EPD] per device.
        #    routing_shard_proj is [1,1,E,EPD] on each device (different per device),
        #    pre-computed at __init__ via ShardTensorToMesh — no host round-trip.
        routing_local = ttnn.matmul(
            dense_routing_tt,
            self.routing_shard_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(dense_routing_tt)
        return routing_local

    def _compute_routing_topk(self, x: ttnn.Tensor):
        """
        Compute top-k routing without scatter — for sparse decode.

        Returns:
            topk_vals_tt: [1, 1, 1, num_active] normalized weights (device, DRAM)
            topk_idx_host: list of num_active global expert indices (Python ints)
        """
        gate_logits_tt = ttnn.linear(
            x,
            self.gate_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.gate_bias_tt is not None:
            gate_logits_tt = ttnn.add(gate_logits_tt, self.gate_bias_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probs_tt = ttnn.softmax(gate_logits_tt, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_logits_tt)

        topk_vals_tt, topk_idx_tt = ttnn.topk(probs_tt, k=self.num_active, dim=-1)
        ttnn.deallocate(probs_tt)

        topk_sum_tt = ttnn.sum(topk_vals_tt, dim=-1, keepdim=True)
        topk_vals_tt = ttnn.div(topk_vals_tt, topk_sum_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(topk_sum_tt)

        # Pull the k indices to host — 4 int32 values, negligible PCIe cost.
        topk_idx_host = (
            ttnn.to_torch(topk_idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            .flatten()
            .tolist()
        )
        ttnn.deallocate(topk_idx_tt)
        return topk_vals_tt, topk_idx_host

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:  x: [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        Returns:   [1, 1, seq, HIDDEN_SIZE]  (replicated on all devices)
        """
        seq_len = x.shape[2]

        # routing_weights: [1, 1, seq, experts_per_device] per device
        routing_weights = self._compute_routing_weights(x, seq_len)

        # ── Batched expert matmul ──────────────────────────────────────────
        # Expand x to [EPD, 1, seq, H] so batch dims match the stacked weights.
        # Use concat along dim=0 instead of ttnn.repeat: repeat is implemented
        # in ROW_MAJOR and forces an expensive untilize+tilize cycle (~1.1ms
        # at seq=128). Concat on the outer batch dim preserves TILE_LAYOUT
        # since it doesn't touch the inner tile boundaries.
        x_exp = ttnn.concat([x] * self.experts_per_device, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        I = EXPERT_INTERMEDIATE_SIZE
        # Fused gate+up: [EPD, 1, seq, H] × [EPD, 1, H, 2*I] → [EPD, 1, seq, 2*I]
        # Use LoFi: BFP4 quantization error dominates HiFi2's extra precision, so
        # LoFi halves FPU cycles at no meaningful PCC cost.
        gate_up_all = ttnn.matmul(
            x_exp,
            self.expert_gate_up,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_exp)

        gate_all = ttnn.slice(
            gate_up_all, [0, 0, 0, 0], [self.experts_per_device, 1, seq_len, I], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        up_all = ttnn.slice(
            gate_up_all,
            [0, 0, 0, I],
            [self.experts_per_device, 1, seq_len, 2 * I],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate_up_all)

        gate_silu = ttnn.silu(gate_all, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_all)
        hidden_all = ttnn.multiply(gate_silu, up_all, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_silu)
        ttnn.deallocate(up_all)

        # [EPD, 1, seq, I] × [EPD, 1, I, H] → [EPD, 1, seq, H]
        expert_out_all = ttnn.matmul(
            hidden_all,
            self.expert_down,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [EPD, 1, seq, H]
        ttnn.deallocate(hidden_all)

        # Scale each expert's output by its routing weight.
        # routing_weights [1, 1, seq, EPD] → [EPD, 1, seq, 1]
        routing_t = ttnn.permute(routing_weights, [3, 0, 2, 1])
        ttnn.deallocate(routing_weights)

        # [EPD, 1, seq, H] * [EPD, 1, seq, 1] (broadcasts last dim)
        weighted_all = ttnn.multiply(expert_out_all, routing_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(expert_out_all)
        ttnn.deallocate(routing_t)

        # Sum across local experts: [EPD, 1, seq, H] → [1, seq, H] → [1, 1, seq, H]
        partial = ttnn.sum(weighted_all, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(weighted_all)
        partial = ttnn.reshape(partial, [1, 1, seq_len, HIDDEN_SIZE])

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

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Single-token decode step.

        Single-device: sparse execution — only the num_active (4) top-k experts
        are computed, reducing expert matmuls from 3×batch-128 to 2×4 individual
        matmuls and eliminating the scatter-based dense routing entirely.

        Multi-device: dense batched execution (original approach) using fused gate+up.
        """
        if self.num_devices == 1:
            return self._forward_decode_sparse(x)
        return self._forward_decode_dense(x)

    def _forward_decode_sparse(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Sparse decode: compute only the top-k active experts (single device)."""
        _mem = ttnn.L1_MEMORY_CONFIG
        I = EXPERT_INTERMEDIATE_SIZE
        H = HIDDEN_SIZE

        topk_vals_tt, topk_idx_host = self._compute_routing_topk(x)

        partial = None
        for rank, expert_idx in enumerate(topk_idx_host):
            # Slice this expert's fused gate+up weight: [1, 1, H, 2*I]
            gate_up_w = ttnn.slice(
                self.expert_gate_up,
                [expert_idx, 0, 0, 0],
                [expert_idx + 1, 1, H, 2 * I],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_up = ttnn.matmul(
                x,
                gate_up_w,
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=_mem,
            )
            ttnn.deallocate(gate_up_w)

            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, 1, 1, I], memory_config=_mem)
            up = ttnn.slice(gate_up, [0, 0, 0, I], [1, 1, 1, 2 * I], memory_config=_mem)
            ttnn.deallocate(gate_up)
            gate_silu = ttnn.silu(gate, memory_config=_mem)
            ttnn.deallocate(gate)
            hidden = ttnn.multiply(gate_silu, up, memory_config=_mem)
            ttnn.deallocate(gate_silu)
            ttnn.deallocate(up)

            # Slice this expert's down weight: [1, 1, I, H]
            down_w = ttnn.slice(
                self.expert_down,
                [expert_idx, 0, 0, 0],
                [expert_idx + 1, 1, I, H],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = ttnn.matmul(
                hidden,
                down_w,
                compute_kernel_config=self.expert_compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=_mem,
            )
            ttnn.deallocate(hidden)
            ttnn.deallocate(down_w)

            # Scale by this expert's normalized routing weight [1, 1, 1, 1]
            w_i = ttnn.slice(topk_vals_tt, [0, 0, 0, rank], [1, 1, 1, rank + 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scaled = ttnn.multiply(out, w_i, memory_config=_mem)
            ttnn.deallocate(out)
            ttnn.deallocate(w_i)

            if partial is None:
                partial = scaled
            else:
                new_partial = ttnn.add(partial, scaled, memory_config=_mem)
                ttnn.deallocate(partial)
                ttnn.deallocate(scaled)
                partial = new_partial

        ttnn.deallocate(topk_vals_tt)

        # all_reduce requires DRAM; for 1 device this is a no-op pass-through.
        partial_dram = ttnn.to_memory_config(partial, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        routed_out = _all_reduce_sum(partial_dram, self.mesh_device, self.num_devices)

        shared_out = self.shared_expert.forward_decode(x)
        out = ttnn.add(routed_out, shared_out, memory_config=_mem)
        ttnn.deallocate(routed_out)
        ttnn.deallocate(shared_out)
        return out

    def _forward_decode_dense(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Dense decode for multi-device mesh: all experts computed, fused gate+up."""
        _mem = ttnn.L1_MEMORY_CONFIG
        I = EXPERT_INTERMEDIATE_SIZE

        routing_weights = self._compute_routing_weights(x, 1)

        # See comment in forward(): concat preserves TILE_LAYOUT, avoiding
        # the untilize+tilize cycle that ttnn.repeat triggers.
        x_exp = ttnn.concat([x] * self.experts_per_device, dim=0, memory_config=_mem)

        gate_up_all = ttnn.matmul(
            x_exp,
            self.expert_gate_up,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
        )
        ttnn.deallocate(x_exp)

        gate_all = ttnn.slice(gate_up_all, [0, 0, 0, 0], [self.experts_per_device, 1, 1, I], memory_config=_mem)
        up_all = ttnn.slice(gate_up_all, [0, 0, 0, I], [self.experts_per_device, 1, 1, 2 * I], memory_config=_mem)
        ttnn.deallocate(gate_up_all)

        gate_silu = ttnn.silu(gate_all, memory_config=_mem)
        ttnn.deallocate(gate_all)
        hidden_all = ttnn.multiply(gate_silu, up_all, memory_config=_mem)
        ttnn.deallocate(gate_silu)
        ttnn.deallocate(up_all)

        expert_out_all = ttnn.matmul(
            hidden_all,
            self.expert_down,
            compute_kernel_config=self.expert_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
        )
        ttnn.deallocate(hidden_all)

        routing_t = ttnn.permute(routing_weights, [3, 0, 2, 1], memory_config=_mem)
        ttnn.deallocate(routing_weights)
        weighted_all = ttnn.multiply(expert_out_all, routing_t, memory_config=_mem)
        ttnn.deallocate(expert_out_all)
        ttnn.deallocate(routing_t)

        partial = ttnn.sum(weighted_all, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(weighted_all)
        partial = ttnn.reshape(partial, [1, 1, 1, HIDDEN_SIZE])

        routed_out = _all_reduce_sum(partial, self.mesh_device, self.num_devices)
        if partial is not routed_out:
            ttnn.deallocate(partial)

        shared_out = self.shared_expert.forward_decode(x)
        out = ttnn.add(routed_out, shared_out, memory_config=_mem)
        ttnn.deallocate(routed_out)
        ttnn.deallocate(shared_out)
        return out
