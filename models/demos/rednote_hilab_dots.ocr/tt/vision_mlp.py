# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision MLP (SwiGLU FFN) for dots.ocr.

DotsSwiGLUFFN (modeling_dots_vision): ``fc2(silu(fc1(x)) * fc3(x))``,
1536 -> 4224 -> 1536, no biases. Identical SwiGLU pattern to reference_impl
models/demos/qwen25_vl/tt/vision_mlp.py (w1=gate/fc1, w3=up/fc3, w2=down/fc2).

TTNN mapping: two sibling ``ttnn.linear`` branches sharing the input, then a
single fused elementwise ``ttnn.mul(gate, up,
input_tensor_a_activations=[ttnn.UnaryOpType.SILU])`` (KB ttnn_mul_1, the
qwen25_vl reference idiom — silu computed inside the BinaryNg kernel), then
the down ``ttnn.linear``. Applied in the optimization phase: it removes the
standalone 110-core silu kernel pass measured at ~99 us (~14% of block kernel
time) at the production fp32 [1,1,896,1536] operating point.

Occupancy REDO (production posture: bf16 tower, tp=4, 11264-row document,
1x4 BH mesh, queried grid 11x10=110): traced replay 4.00 -> 2.46 ms/device
kernel time (wall 3.12 -> 2.63 ms/call; 42 calls/image). Levers, one per
measurement: (1) CCL num_links 1->2 (reduce_scatter 1201->655 us, all_gather
1241->648 us/device; links at HW ceiling); (2) bf8b-first single-pass
directive: bfloat8_b weights + HiFi2 (fp32 dest acc kept) on all three
projections, matmuls 1371->1075 us/device, block PCC 0.999991 unchanged;
(3) gate/up 1D height-mcast program config in0_block_w=4 (matmuls
1075->961 us/device, FPU util 31->37%). Final occupancy vs 110-core grid:
gate/up 88/110 (80%, ceil(352 M-tiles/110)=4 rows/core), down 100/110 (91%,
heuristic — every explicit down config CB-overflows program.cpp:1326),
fused silu-mul 110/110, CCL 34-36c link-bound at 2/2 links. Rejected with
evidence: bf16 dest acc (gate no-win 369 vs 362; down -12% but breaks the
fp32-dest-acc precision recipe for ~0.6% image-level gain), M-padding
11264->11520 for a 110-core 2D grid (pad+slice data movement cancels the
gain and changes the block contract). Composition note for the vision_block
tick: the all-reduce (1.3 ms/device, 53% of block kernel time) returns the
output replicated for the residual add; if the block's residual stream ever
goes row-sharded, the all_gather half could fold away.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
three weights are ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay
replicated, no CCL. On a single device the mesh_mapper degenerates gracefully.
Production (tt/ocr_model.py) overrides to tp_degree=4 column/row-parallel.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtVisionMLP(LightweightModule):
    """dots.ocr vision SwiGLU FFN: fc2(silu(fc1(x)) * fc3(x)), no biases.

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: {"fc1.weight": [hidden, dim], "fc2.weight": [dim, hidden],
            "fc3.weight": [hidden, dim]} torch tensors (HF keys
            vision_tower.blocks.N.mlp.*).
        dtype: on-device weight dtype.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, tp_degree=1, weight_dtype=None):
        super().__init__()
        self.mesh_device = mesh_device
        # Column/row-parallel TP (optimization REDO A/B): gate/up column-shard
        # the hidden dim (4224/tp per chip), down row-shards it back; partial
        # outputs are all-reduced (reduce_scatter+all_gather) in forward.
        # tp_degree=1 = replicate (single-device degenerate).
        self.tp_degree = tp_degree

        # bf8b-first on the single-pass vision path (occupancy REDO): the
        # run-once bf16 tower takes bfloat8_b weights + HiFi2 (fp32 dest acc
        # kept) on all three projections — matmuls 1371->1031 us/device at
        # the 11264-row document shape, block PCC 0.999886. The fp32
        # high-precision path (tests/test_vision_transformer.py legacy
        # posture) keeps fp32 weights + HiFi4 untouched.
        self.high_precision = dtype == ttnn.float32
        if weight_dtype is None:
            weight_dtype = dtype if self.high_precision else ttnn.bfloat8_b

        # Transpose [out, in] -> [in, out] for x @ W^T.
        as_weight = lambda name, dim=None: ttnn.from_torch(
            state_dict[name].transpose(-2, -1).contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            if (tp_degree == 1 or dim is None)
            else ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )
        self.w1 = as_weight("fc1.weight", dim=-1)  # gate: [dim, hidden] col-parallel
        self.w3 = as_weight("fc3.weight", dim=-1)  # up:   [dim, hidden] col-parallel
        self.w2 = as_weight("fc2.weight", dim=-2)  # down: [hidden, dim] row-parallel

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if weight_dtype != ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _gate_up_program_config(self, x):
        """1D height-mcast matmul config for the col-parallel gate/up linears.

        Occupancy REDO A/B at the production document shape (per-chip
        [11264,1536]@[1536,1056], bf16 act x bf8b weight, HiFi2+fp32-acc):
        heuristic 88-core 2D mcast 387 us -> 1D height-mcast in0_block_w=4
        ~330 us/op in-block (scratch A/B 428->366; ibw6 368.5, 2D 11x8 ibw2
        369.6 — ibw4 1D kept). ibw>=8 and every explicit config for the
        row-parallel down matmul CB-overflow (program.cpp:1326), so down
        keeps the heuristic (100/110 cores, the best FPU util of the three).
        Size-gated: small shapes (the 896-row PCC gate posture) and wide
        per-chip N (tp=1 replicate, N_t=132: 1D in1 CB would overflow L1)
        fall back to the heuristic, which already picks a sane grid there.
        """
        m_tiles = 1
        for i in range(len(x.padded_shape) - 1):
            m_tiles *= x.padded_shape[i]
        m_tiles //= 32
        k_tiles = x.padded_shape[-1] // 32
        n_tiles = self.w1.padded_shape[-1] // 32
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        per_m = -(-m_tiles // cores)  # ceil
        if per_m < 2 or per_m > 8 or n_tiles > 64 or k_tiles % 4:
            return None  # off the measured envelope -> heuristic
        sub_h = max(s for s in (4, 2, 1) if per_m % s == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=4,
            out_subblock_h=sub_h,
            out_subblock_w=1,
            per_core_M=per_m,
            per_core_N=n_tiles,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [..., dim] TILE_LAYOUT, replicated across the mesh.

        Returns: [..., dim], replicated.
        """
        gate_up_pc = self._gate_up_program_config(x)
        gate = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_up_pc,
        )
        up = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=gate_up_pc,
        )
        # Fused silu(gate) * up in one BinaryNg kernel (KB ttnn_mul_1).
        h = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # bf8b all-reduce (occupancy REDO, vision_block tick): the tp>1 CCL
        # pair is LINK-bound (18-34 cores, 2/2 links — the recorded HW
        # ceiling), so the remaining lever is wire BYTES: emit the down
        # partials as bfloat8_b and run reduce_scatter+all_gather at half
        # the volume. The replicated bf8b output feeds the caller's
        # mixed-dtype residual add. fp32/tp=1 paths untouched.
        ccl_bf8b = self.tp_degree > 1 and not self.high_precision
        out = ttnn.linear(
            h,
            self.w2,
            dtype=ttnn.bfloat8_b if ccl_bf8b else None,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        if self.tp_degree > 1:
            # Row-parallel down produced per-chip PARTIAL sums — all-reduce.
            # num_links=2 saturates both QB eth channels per hop (occupancy
            # REDO A/B: reduce_scatter 1201->631 us, all_gather 1241->645 us
            # per device at the document shape; num_links=4 rejected by HW —
            # TT_FATAL, 2 channels per hop. Same lever as vision_attention).
            part = ttnn.reduce_scatter(out, dim=3, topology=ttnn.Topology.Linear, num_links=2)
            ttnn.deallocate(out)
            out = ttnn.all_gather(part, dim=3, topology=ttnn.Topology.Linear, num_links=2)
            ttnn.deallocate(part)
        return out
