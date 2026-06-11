# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text MLP (Qwen2 SwiGLU FFN) for dots.ocr.

Qwen2MLP (the dots.ocr text decoder): ``down_proj(silu(gate_proj(x)) *
up_proj(x))``, 1536 -> 8960 -> 1536, no biases — the same SwiGLU shape as
reference_impl models/tt_transformers/tt/mlp.py (w1=gate, w3=up, w2=down).

TTNN mapping: two sibling ``ttnn.linear`` branches sharing the input, the
KB ttnn_mul_1 fused-SiLU ``ttnn.mul`` (``input_tensor_a_activations=
[ttnn.UnaryOpType.SILU]``, as the tt_transformers reference uses), then the
down ``ttnn.linear``. Optimization phase added a DECODE fast path (padded
seq <= 32): DRAM-WIDTH-SHARDED weight copies + HiFi2
MatmulMultiCoreReuseMultiCastDRAMSharded matmuls on a 12-core L1
width-sharded chain (see ``_forward_decode``); decode block kernel time
131.9 -> 94.3 us/device (-28.5%) at the production bf16-row/bfp8-weight
operating point.

Parallelism plan (ARCHITECTURE.md / inventory notes): placement=shard —
gate/up are COLUMN-parallel (output-feature dim sharded 4-way,
``ShardTensorToMesh(dim=-1)``; per-chip intermediate slice 8960/4 = 2240),
the elementwise silu/mul stay chip-local on the matching slices, and down is
ROW-parallel (input-feature dim sharded, ``dim=-2``), producing per-chip
PARTIAL [.., hidden] sums combined with an all-reduce. Prefill uses
``ttnn.reduce_scatter`` + ``ttnn.all_gather`` (wire-bound regime, same
idiom as this model's text_attention o_proj); the decode row uses
``ttnn.all_gather(dim=1)`` + ``fast_reduce_nc`` (latency-bound regime —
24.0 vs 30.8 us/pair measured at [1,1,1,1536] bf16, perf tick 67). On a
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
        gate_up_dtype: gate/up weight storage override (None -> dtype) —
            perf-phase knob for trying bfloat8_b on the two column-parallel
            branches while down_proj (the contraction back into the residual
            stream) keeps ``dtype``.
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, gate_up_dtype=None):
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
        def as_weight(name, mapper, w_dtype):
            return ttnn.from_torch(
                state_dict[name].transpose(-2, -1).contiguous(),
                dtype=w_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper if num_devices > 1 else replicate,
            )

        gate_up_dtype = gate_up_dtype or dtype
        self.w1 = as_weight("gate_proj.weight", shard_cols, gate_up_dtype)  # [hidden, inter/N]
        self.w3 = as_weight("up_proj.weight", shard_cols, gate_up_dtype)  # [hidden, inter/N]
        self.w2 = as_weight("down_proj.weight", shard_rows, dtype)  # [inter/N, hidden]

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # ---- decode DRAM-sharded matmuls (occupancy REDO) ----------------
        # M=1 decode matmuls are DRAM weight-streaming bound: the default
        # interleaved kernels read the bfp8 weights at ~100-125 GB/s
        # (tracy baseline: gate/up 29.6/29.8 us @70c, down 37.0 us @48c per
        # device). Same lever as text_attention o_proj/QKV: weights stored
        # WIDTH_SHARDED across the chip's DRAM banks and run through
        # MatmulMultiCoreReuseMultiCastDRAMSharded with the activation row
        # width-sharded in L1. The per-chip inter slice 2240 (70 tiles) is
        # zero-padded to 2304 (72 tiles) so ONE core grid divides both
        # hidden (48t) and inter (72t) exactly; the zero gate columns stay
        # zero through fused-silu*up and meet zero down rows, so the output
        # is exact. Core-count A/B (see _forward_decode): 12 cores wins.
        import torch

        hidden = state_dict["gate_proj.weight"].shape[1]
        inter = state_dict["gate_proj.weight"].shape[0]
        inter_pd = inter // num_devices
        self._dec_cores = 12
        inter_pad = ((inter_pd + 32 * self._dec_cores - 1) // (32 * self._dec_cores)) * (32 * self._dec_cores)
        compute_grid = mesh_device.compute_with_storage_grid_size()
        dram_grid = mesh_device.dram_grid_size()
        dram_weight_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid.x - 1, dram_grid.y - 1))}
        )

        def _dram_ws_mc(k, n):
            pad_n = ((n + 32 * dram_grid.x - 1) // (32 * dram_grid.x)) * (32 * dram_grid.x)
            spec = ttnn.ShardSpec(dram_weight_grid, (k, pad_n // dram_grid.x), ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)

        def _rect_grid(n):
            # (0,0)-anchored RECTANGLE, nearest-square (not the row-wise
            # wrap set): the decoder layer's canonical residual shard feeds
            # these cores straight from the sharded RMSNorm, whose program
            # config takes a rectangular grid size. Same helper as
            # text_attention._rect_grid.
            best = None
            for gy in range(1, compute_grid.y + 1):
                if n % gy == 0 and n // gy <= compute_grid.x:
                    cand = (n // gy, gy)
                    if best is None or abs(cand[0] - cand[1]) < abs(best[0] - best[1]):
                        best = cand
            assert best is not None, f"{n} cores do not fit {compute_grid.x}x{compute_grid.y}"
            return best

        def _l1_ws_mc(width):
            gx, gy = _rect_grid(self._dec_cores)
            return ttnn.create_sharded_memory_config(
                shape=(32, width // self._dec_cores),
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )

        def _pad_chunks(w, dim):
            # Zero-pad each per-device slice inter_pd -> inter_pad along dim.
            chunks = w.chunk(num_devices, dim=dim)
            pad_shape = list(chunks[0].shape)
            pad_shape[dim] = inter_pad - inter_pd
            zero = torch.zeros(pad_shape, dtype=w.dtype)
            return torch.cat([torch.cat([c, zero], dim=dim) for c in chunks], dim=dim)

        def _dec_weight(torch_w, mapper, k, n, w_dtype):
            return ttnn.from_torch(
                torch_w.contiguous(),
                dtype=w_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=_dram_ws_mc(k, n),
                mesh_mapper=mapper if num_devices > 1 else replicate,
            )

        # gate/up [inter, hidden] -> pad the per-chip out slices -> transpose
        # to [hidden, N*inter_pad]; column-parallel shard on the padded dim.
        gu_pad = lambda name: _pad_chunks(state_dict[name], 0).transpose(-2, -1)
        self.w1_dec = _dec_weight(gu_pad("gate_proj.weight"), shard_cols, hidden, inter_pad, gate_up_dtype)
        self.w3_dec = _dec_weight(gu_pad("up_proj.weight"), shard_cols, hidden, inter_pad, gate_up_dtype)
        # down [hidden, inter] -> pad the per-chip in slices -> transpose to
        # [N*inter_pad, hidden]; row-parallel shard on the padded dim.
        dn_pad = _pad_chunks(state_dict["down_proj.weight"], 1).transpose(-2, -1)
        self.w2_dec = _dec_weight(dn_pad, shard_rows, inter_pad, hidden, dtype)

        h_t, i_t = hidden // 32, inter_pad // 32
        self._dec_x_mc = _l1_ws_mc(hidden)
        self._dec_h_mc = _l1_ws_mc(inter_pad)
        self._dec_out_mc = _l1_ws_mc(hidden)
        # NOTE measured A/Bs (12c, per device): fusing SILU into the gate
        # matmul's fused_activation DOUBLES the matmul (16.9 -> 34.6 us, the
        # SFPU pass serializes the BW-bound kernel) and was reverted; the
        # SILU stays fused in the binary mul (KB ttnn_mul_1) which costs
        # 11.2 us at 12c vs plain mul 4.7 us — still the cheaper total.
        self._dec_pc13 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=h_t // self._dec_cores,
            per_core_M=1,
            per_core_N=i_t // self._dec_cores,
            fused_activation=None,
        )
        self._dec_pc2 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=i_t // self._dec_cores,
            per_core_M=1,
            per_core_N=h_t // self._dec_cores,
            fused_activation=None,
        )
        # Decode matmuls are FLOP-bound at HiFi4 on the small DRAM-sharded
        # grid (tracy: ~30 us unchanged vs interleaved). tt_transformers'
        # decode posture (model_config get_mlp_*_prg_config comment: "These
        # use HiFi2; ... would be FLOP-bound on 12 cores with HiFi4") drops
        # to HiFi2 with fp32 accumulation kept; the 8-row decode PCC drift
        # gate covers the precision change.
        self._dec_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode fast path: one padded tile row [1, 1, <=32, hidden].

        i2s L1 width-shard (12 cores; skipped when the decoder layer hands
        the residual stream in ALREADY width-sharded — the layer-scope
        sharded-residual lever) -> DRAM-sharded HiFi2 gate/up matmuls
        -> fused-SiLU mul (stays width-sharded) -> DRAM-sharded down matmul
        -> s2i DRAM for the CCL all-reduce. Occupancy-redo A/B at the
        queried 11x10 grid: the M=1 matmuls are DRAM weight-streaming bound
        and prefer FEW cores (per-matmul: 110c interleaved 30.0 / 24c 20.7 /
        12c 16.9 / 8c 16.2 us — flat 12->8, a measured BW plateau ~210-230
        GB/s), while the eltwise mul scales with per-core tiles and prefers
        MANY (24c 6.2 / 12c 11.2 / 8c 16.4 us). 12 cores minimizes the
        block total (94.3 us/device vs 24c 100.6 / 8c 97.7).
        """
        x_ws = x if x.is_sharded() else ttnn.interleaved_to_sharded(x, self._dec_x_mc)
        gate = ttnn.linear(
            x_ws,
            self.w1_dec,
            program_config=self._dec_pc13,
            compute_kernel_config=self._dec_compute_kernel_config,
            memory_config=self._dec_h_mc,
        )
        up = ttnn.linear(
            x_ws,
            self.w3_dec,
            program_config=self._dec_pc13,
            compute_kernel_config=self._dec_compute_kernel_config,
            memory_config=self._dec_h_mc,
        )
        if x_ws is not x:
            ttnn.deallocate(x_ws)
        # KB ttnn_mul_1: SiLU fused into the binary mul (the tt_transformers
        # decode posture). Measured A/Bs at 12c per device: fused-SiLU mul
        # 11.2 us beats standalone in-place silu + plain mul (7.1 + 4.9 us)
        # and beats SiLU fused into the gate matmul (matmul 16.9 -> 34.6 us).
        h = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=self._dec_h_mc,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(
            h,
            self.w2_dec,
            program_config=self._dec_pc2,
            compute_kernel_config=self._dec_compute_kernel_config,
            memory_config=self._dec_out_mc,
        )
        ttnn.deallocate(h)
        out_i = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return out_i

    def forward(self, x: ttnn.Tensor, out_mc=None) -> ttnn.Tensor:
        """x: [1, 1, seq, hidden] TILE_LAYOUT, replicated across the mesh.

        out_mc: optional L1 width-sharded MemoryConfig for the decode-path
            output (the decoder layer's canonical residual shard) — the
            all_gather re-replication lands straight in that shard so the
            caller's residual add runs shard-to-shard. None keeps the DRAM
            interleaved block-output contract.
        Returns: [1, 1, seq, hidden], replicated (all-reduced down_proj output).
        """
        # Decode posture: one padded tile row (logical seq <= 32), DRAM
        # interleaved or already width-sharded (layer-scope sharded
        # residual), takes the DRAM-sharded L1 fast path.
        mem_layout = x.memory_config().memory_layout
        if x.padded_shape[-2] <= 32 and mem_layout in (
            ttnn.TensorMemoryLayout.INTERLEAVED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ):
            out = self._forward_decode(x)
            if self.num_devices > 1:
                # Decode all-reduce: all_gather(dim=1) + fast_reduce_nc
                # (perf tick-67 A/B at the [1,1,1,1536] bf16 row, traced
                # replay, num_links=2): 24.0 us vs 30.8 us for the sync
                # RS+AG pair and 29.8 us for reduce_scatter_minimal_async +
                # all_gather_async — at the 1-row shape the RS hop is
                # latency-bound, while one dim=1 gather + a local 4-slice
                # reduction is cheaper. Numerics: identical to RS+AG within
                # bf16 rounding (max|diff| vs 4x host truth 0.0625 for BOTH
                # forms). Prefill below keeps the RS+AG pair (wire-bound
                # regime; the dim=1 gather would move 4x the bytes).
                g = ttnn.all_gather(out, dim=1, num_links=2, topology=ttnn.Topology.Linear)
                ttnn.deallocate(out)
                red = ttnn.experimental.fast_reduce_nc(
                    g, dims=[1], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(g)
                # fast_reduce_nc keeps the 32-row padded face as its logical
                # shape; slice row 0 back to [1, 1, 1, hidden] (straight into
                # the caller's residual shard when out_mc is given).
                if out_mc is not None:
                    out = ttnn.slice(red, [0, 0, 0, 0], [1, 1, 1, red.shape[-1]], memory_config=out_mc)
                else:
                    out = ttnn.slice(red, [0, 0, 0, 0], [1, 1, 1, red.shape[-1]])
                ttnn.deallocate(red)
            elif out_mc is not None:
                out_sh = ttnn.interleaved_to_sharded(out, out_mc)
                ttnn.deallocate(out)
                out = out_sh
            return out

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
        # KB ttnn_mul_1: SiLU fused into the binary mul via
        # input_tensor_a_activations (same as reference_impl tt_transformers
        # mlp.py w2_in) — one eltwise launch instead of silu + mul.
        h = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Row-parallel down_proj: per-chip PARTIAL sum over its 2240 rows.
        # Core grid is at its structural cap (N_t=48 -> 48 cores; explicit
        # core_grid 10x10 measured WORSE in composed-decoder tracy, tick-30:
        # 69.5 -> 98.4 us; reverted).
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
            reduced = ttnn.reduce_scatter(out, dim=3, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(out)
            out = ttnn.all_gather(reduced, dim=3, num_links=2, topology=ttnn.Topology.Linear)
            ttnn.deallocate(reduced)
        return out
