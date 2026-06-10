# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text MLP (Qwen2 SwiGLU FFN) for dots.ocr.

Qwen2MLP (the dots.ocr text decoder): ``down_proj(silu(gate_proj(x)) *
up_proj(x))``, 1536 -> 8960 -> 1536, no biases — the same SwiGLU shape as
reference_impl models/tt_transformers/tt/mlp.py (w1=gate, w3=up, w2=down).

TTNN mapping: two sibling ``ttnn.linear`` branches sharing the input,
explicit ``ttnn.silu`` on the gate branch, elementwise ``ttnn.mul`` with the
up branch, then the down ``ttnn.linear`` — the KB ttnn_silu_2 SwiGLU
replacement (``out = ttnn.mul(ttnn.silu(gate), up)``). KB ttnn_mul_1's fused
variant (``input_tensor_a_activations=[ttnn.UnaryOpType.SILU]``, as the
tt_transformers reference uses) computes the same thing in one op; deferred
to the optimization phase since the mlp guard requires a traced silu/gelu
kernel.

Parallelism plan (ARCHITECTURE.md / inventory notes): placement=shard —
gate/up are COLUMN-parallel (output-feature dim sharded 4-way,
``ShardTensorToMesh(dim=-1)``; per-chip intermediate slice 8960/4 = 2240),
the elementwise silu/mul stay chip-local on the matching slices, and down is
ROW-parallel (input-feature dim sharded, ``dim=-2``), producing per-chip
PARTIAL [.., hidden] sums combined with an all-reduce
(``ttnn.reduce_scatter`` + ``ttnn.all_gather``, fp32 fabric accumulation;
swapped from all_gather + local adds in the optimization phase per
tp-guidance), the same idiom as this model's text_attention o_proj. On a
single device the sharding degenerates to the replicated full computation
and the CCL is skipped.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtTextMLP(LightweightModule):
    """dots.ocr text SwiGLU FFN: down(silu(gate(x)) * up(x)), no biases, TP-sharded.

    Args:
        mesh_device: ttnn mesh device handle (1xN line; weights TP-sharded).
        state_dict: {"gate_proj.weight": [8960, 1536], "up_proj.weight":
            [8960, 1536], "down_proj.weight": [1536, 8960]} torch tensors
            (HF keys model.layers.N.mlp.*).
        dtype: on-device weight/activation dtype.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        num_devices = mesh_device.get_num_devices()
        self.num_devices = num_devices

        shard_cols = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        shard_rows = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        # Transpose [out, in] -> [in, out] for x @ W^T, then shard:
        # gate/up column-parallel on the OUTPUT feature dim, down
        # row-parallel on the INPUT feature dim (per-chip rows match the
        # per-chip silu*mul slice, so the matmul yields a PARTIAL sum).
        def as_weight(name, mapper):
            return ttnn.from_torch(
                state_dict[name].transpose(-2, -1).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper if num_devices > 1 else replicate,
            )

        self.w1 = as_weight("gate_proj.weight", shard_cols)  # [hidden, inter/N]
        self.w3 = as_weight("up_proj.weight", shard_cols)  # [hidden, inter/N]
        self.w2 = as_weight("down_proj.weight", shard_rows)  # [inter/N, hidden]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, seq, hidden] TILE_LAYOUT, replicated across the mesh.

        Returns: [1, 1, seq, hidden], replicated (all-reduced down_proj output).
        """
        gate = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Row-parallel down_proj: per-chip PARTIAL sum over its 2240 rows.
        out = ttnn.linear(
            h,
            self.w2,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)

        if self.num_devices > 1:
            # All-reduce of the per-chip partials: reduce_scatter (each chip
            # sums its hidden/N shard, fp32 fabric accumulation) + all_gather
            # to re-replicate. Replaces the original all_gather + N slices +
            # N-1 local adds (full 4*hidden payload gathered then summed on
            # 110-core BinaryNg) — same swap as text_attention o_proj
            # (tick-28); tracy tick-29 A/B on this block: per-device kernel
            # 368.5 -> 287.5 us (-22%), CCL cluster 171.9 -> 90.6 us, PCC
            # unchanged.
            reduced = ttnn.reduce_scatter(out, dim=3, topology=ttnn.Topology.Linear)
            ttnn.deallocate(out)
            out = ttnn.all_gather(reduced, dim=3, topology=ttnn.Topology.Linear)
            ttnn.deallocate(reduced)
        return out
