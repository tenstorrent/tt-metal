# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 MoE block — Galaxy mesh (8,4) with EP=8 + TP=4.

Parallelism strategy on (rows=8, cols=4) mesh:
  EP=8 (rows):  256 experts / 8 rows = 32 experts per row device
  TP=4 (cols):  intermediate_size FF=1536 / 4 cols = 384 per device

Expert weight sharding (4D tensors — gpt_oss convention):
  w1 (gate): [1, E, H, FF]  column+EP → [1, E/EP, H, FF/TP] = [1, 32, 3072, 384] per device
  w3 (up):   same as w1
  w2 (down): [1, E, FF, H]  dims=(1,-2) → [1, E/EP, FF/TP, H] = [1, 32, 384, 3072] per device

Router gate weight [E, H] = [256, 3072]: replicated (only 1.5 MB)
Routing bias [E] = [256]: replicated

Forward pass:
  1. Router: linear(x) + sigmoid + routing_bias → top-k selection (on device, replicated)
  2. Build sparse routing_weights [T, E] → [1, E_local, T, 1] for local experts
  3. Dense batched expert compute:
       gate = x ⊗ w1  [1, E_local, T, FF_local]   (broadcast x: [1,1,T,H] → [1,E_local,T,H])
       up   = x ⊗ w3
       hidden = silu(gate) * up
       down   = hidden ⊗ w2  [1, E_local, T, H]
  4. Apply local routing weights + sum over E_local → [1, 1, T, H]
  5. EP all-reduce (sum across rows)
  6. TP all-reduce (sum across cols, since down proj is [FF/TP, H] giving partial H sum)

Routing: sigmoid (NOT softmax), with additive routing bias for top-k selection.
         Weights for aggregation come from sigmoid (no bias).
"""

import torch
import torch.nn.functional as F

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts.operations import apply_expert_parallel_allreduce, apply_tensor_parallel_allreduce

from .model_config import MiniMaxM2TTConfig


class TtMiniMaxMoE:
    """
    MiniMax-M2.5 MoE block — on-device expert computation with EP+TP sharding.

    All expert weights live on device; no CPU fallback.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        layer_idx: int,
        mesh_config: MeshConfig = None,
        ccl_manager: CCLManager = None,
    ):
        self.config = config
        self.device = device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        E = config.num_local_experts
        FF = config.intermediate_size
        H = config.hidden_size

        ep = mesh_config.ep if mesh_config else 1
        tp = mesh_config.tp if mesh_config else 1
        E_local = E // ep  # experts per row
        FF_local = FF // tp  # intermediate per col device

        prefix = f"model.layers.{layer_idx}.block_sparse_moe."

        self._E = E
        self._E_local = E_local
        self._FF = FF
        self._H = H
        self._ep = ep
        self._tp = tp

        # ---- Router gate weight [E, H]: replicated (small) ----
        gate_w = state_dict[prefix + "gate.weight"].to(torch.bfloat16)  # [E, H]
        rep_mapper = ttnn.ReplicateTensorToMesh(device) if self._is_mesh else None
        self.gate_weight = ttnn.from_torch(
            gate_w.T,  # [H, E] for ttnn.linear
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_mapper,
        )

        # ---- Routing bias [E]: replicated ----
        routing_bias = state_dict[prefix + "e_score_correction_bias"].to(torch.bfloat16)  # [E]
        self.routing_bias_cpu = routing_bias.float().cpu()  # keep on CPU for top-k

        # ---- Expert weights: EP + TP 2D sharding ----
        # gpt_oss convention: [1, E, H, FF] for gate/up, [1, E, FF, H] for down
        # 2D mesh sharding: dims=(EP_dim, TP_dim)
        #   EP shards along dim 1 (E) across rows
        #   TP shards along dim 3 (FF for gate/up) or dim 2 (FF for down) across cols
        if self._is_mesh:
            # dims=(row_dim, col_dim): shard E across rows (dim=1), FF across cols (dim=-1 or -2)
            ep_tp_col_mapper = ttnn.ShardTensor2dMesh(
                device, dims=(1, -1), mesh_shape=device.shape
            )  # gate/up: [1, E, H, FF] → [1, E/EP, H, FF/TP] per device
            ep_tp_row_mapper = ttnn.ShardTensor2dMesh(
                device, dims=(1, -2), mesh_shape=device.shape
            )  # down:   [1, E, FF, H] → [1, E/EP, FF/TP, H] per device
        else:
            ep_tp_col_mapper = None
            ep_tp_row_mapper = None

        # Stack expert weights into [1, E, H, FF] / [1, E, FF, H]
        w1_stack = torch.stack(
            [state_dict[prefix + f"experts.{j}.w1.weight"] for j in range(E)]
        )  # [E, FF, H]  (w1 is [FF, H] in HF convention)
        w2_stack = torch.stack(
            [state_dict[prefix + f"experts.{j}.w2.weight"] for j in range(E)]
        )  # [E, H, FF]  (w2 is [H, FF] in HF convention — down proj)
        w3_stack = torch.stack([state_dict[prefix + f"experts.{j}.w3.weight"] for j in range(E)])  # [E, FF, H]

        # Convert to gpt_oss convention [1, E, H, FF] for gate/up (transpose w1, w3)
        # w1: [E, FF, H] → transpose → [E, H, FF] → unsqueeze → [1, E, H, FF]
        w1_4d = w1_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)  # [1, E, H, FF]
        w3_4d = w3_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)  # [1, E, H, FF]
        # w2: [E, H, FF] → transpose → [E, FF, H] → unsqueeze → [1, E, FF, H]
        w2_4d = w2_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)  # [1, E, FF, H]

        self.w1 = ttnn.from_torch(
            w1_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_col_mapper,
        )
        self.w3 = ttnn.from_torch(
            w3_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_col_mapper,
        )
        self.w2 = ttnn.from_torch(
            w2_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_row_mapper,
        )

    # ------------------------------------------------------------------
    # Router (on device)
    # ------------------------------------------------------------------

    def _route(self, x_flat: ttnn.Tensor, T: int, T_pad: int):
        """Compute routing: returns routing_weights_tt [1, E, T_pad, 1] for local experts.

        x_flat: [1, 1, T_pad, H] replicated on all devices.
        T:     original (unpadded) token count
        T_pad: padded token count (multiple of 32)

        Returns:
            routing_tt: [1, E_local, T_pad, 1]  — routing weights for local EP experts
        """
        E = self._E
        E_local = self._E_local
        ep = self._ep

        # Router: [1, 1, T_pad, H] × [H, E] → [1, 1, T_pad, E]
        logits = ttnn.linear(x_flat, self.gate_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Sigmoid → [1, 1, T_pad, E]
        rw = ttnn.sigmoid(logits)
        logits.deallocate(True)

        # Move to CPU for top-k selection with routing bias; only process non-padded tokens
        if self._is_mesh:
            rw_pt = ttnn.to_torch(ttnn.get_device_tensors(rw)[0]).float()
        else:
            rw_pt = ttnn.to_torch(rw).float()
        rw_pt = rw_pt.reshape(T_pad, E)[:T]  # [T, E] — discard padded rows
        rw.deallocate(True)

        top_k = self.config.num_experts_per_tok
        scores = rw_pt + self.routing_bias_cpu.unsqueeze(0)  # [T, E]
        _, top_k_idx = torch.topk(scores, top_k, dim=-1, sorted=False)  # [T, top_k]
        top_k_weights = rw_pt.gather(1, top_k_idx)  # [T, top_k]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Build sparse [T, E] routing weights
        sparse_w = torch.zeros(T, E, dtype=torch.bfloat16)
        sparse_w.scatter_(1, top_k_idx, top_k_weights.to(torch.bfloat16))  # [T, E]

        # Split into EP local shard — each device only uses its E_local rows
        # With ShardTensor2dMesh(dims=(1,-1)) on the weight [1, E, T, 1],
        # device[row, col] gets experts [row*E_local : (row+1)*E_local].
        # We push the full [1, E, T, 1] and let the mapper shard along dim 1.
        routing_4d = sparse_w.T.reshape(1, E, T, 1)  # [1, E, T, 1]

        # Pad T to tile boundary for TTNN
        T_pad = max(32, ((T + 31) // 32) * 32)
        if T_pad != T:
            routing_4d = F.pad(routing_4d, (0, 0, 0, T_pad - T))

        if self._is_mesh:
            ep_tp_col_mapper = ttnn.ShardTensor2dMesh(
                self.device, dims=(1, None), mesh_shape=self.device.shape
            )  # shard E across rows; replicate across cols
            routing_tt = ttnn.from_torch(
                routing_4d,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ep_tp_col_mapper,
            )
        else:
            routing_tt = ttnn.from_torch(
                routing_4d,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return routing_tt  # [1, E_local, T_pad, 1] per device

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: [B, S, H]

        Returns:
            [B, S, H]
        """
        cfg = self.config
        H = self._H
        tp = self._tp
        ep = self._ep
        E_local = self._E_local

        orig_shape = list(x.shape)
        if len(orig_shape) == 3:
            T = orig_shape[0] * orig_shape[1]
        elif len(orig_shape) == 4:
            T = orig_shape[0] * orig_shape[1] * orig_shape[2]
        else:
            raise ValueError(f"Unexpected x.shape {orig_shape}")

        T_pad = max(32, ((T + 31) // 32) * 32)

        # Reshape to [1, 1, T, H] for matmul
        x_flat = ttnn.reshape(x, (1, 1, T, H))
        if T_pad != T:
            x_flat = ttnn.pad(x_flat, padding=[(0, 0), (0, 0), (0, T_pad - T), (0, 0)], value=0.0)

        # ------ Router ------
        routing_tt = self._route(x_flat, T, T_pad)  # [1, E_local, T_pad, 1] per device

        # ------ Expand x for batched expert matmul ------
        # x_flat [1, 1, T_pad, H] → [1, E_local, T_pad, H]
        # (TTNN matmul requires exact batch dims — no broadcast support)
        x_exp = ttnn.repeat(x_flat, ttnn.Shape([1, E_local, 1, 1]))

        # ------ Gate/Up projections: [1, E_local, T_pad, H] × [1, E_local, H, FF_local] ------
        gate = ttnn.matmul(x_exp, self.w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(x_exp, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_exp.deallocate(True)

        # ------ SwiGLU: silu(gate) * up ------
        gate_silu = ttnn.silu(gate)
        gate.deallocate(True)
        hidden = ttnn.mul(gate_silu, up)
        gate_silu.deallocate(True)
        up.deallocate(True)

        # ------ Down projection: [1, E_local, T_pad, FF_local] × [1, E_local, FF_local, H] ------
        down = ttnn.matmul(hidden, self.w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden.deallocate(True)

        # ------ Apply routing weights: [1, E_local, T_pad, H] * [1, E_local, T_pad, 1] ------
        down = ttnn.mul(down, routing_tt)
        routing_tt.deallocate(True)

        # ------ Sum over local experts using fast_reduce_nc (dim=1) ------
        # fast_reduce_nc on [1, E_local, T_pad, H] reduces dim=1 → [1, T_pad, H]
        # unsqueeze_to_4D → [1, 1, T_pad, H]
        out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(down, dims=[1]))
        down.deallocate(True)

        # ------ EP all-reduce: sum contributions across rows ------
        if self._is_mesh and self.mesh_config and self.ccl_manager and ep > 1:
            out = apply_expert_parallel_allreduce(out, self.mesh_config, self.ccl_manager)

        out = ttnn.unsqueeze_to_4D(out)

        # ------ TP all-reduce: sum partial H sums across cols ------
        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out = apply_tensor_parallel_allreduce(out, self.mesh_config, self.device, T_pad, self.ccl_manager)

        # ------ Reshape back to input shape ------
        if T_pad != T:
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, T, H))

        out = ttnn.reshape(out, orig_shape)
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
