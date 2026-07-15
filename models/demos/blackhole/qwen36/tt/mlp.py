# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: down(silu(gate(x)) * up(x)).

9B (single device): dense matmuls, full weights.
27B TP (1,4 mesh): w1/w3 column-parallel, w2 row-parallel; tt_all_reduce
reduce-scatters on meshes with a dim-1 shape (e.g. P150x4), fracturing hidden.
"""
import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix — layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        if dram_sharded:
            return MLPWeights(
                w1=tpc.shard_w(
                    state_dict["gate_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w1_weight_memcfg,
                    cache_path=cache("gate_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=ttnn.bfloat8_b,
                ),
            )

        # Default: INTERLEAVED DRAM shards; ttnn.linear works for decode and prefill.
        return MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=ttnn.bfloat8_b,
            ),
        )

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # 1D-decode (default): small-grid 1D matmuls beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny decode MLP matmuls (see test_mlp_matmul_sweep). Forces interleaved weights.
        self._mlp_1d_decode = args is not None and getattr(args, "mlp_1d_decode", False)
        # Match load_mlp_weights dram_sharded condition for layout consistency.
        self._dram_sharded = (
            self.num_devices > 1
            and args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not self._mlp_1d_decode
        )
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        if self.num_devices > 1:
            return self._forward_tp(x)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        down_pc = None
        if (
            T > 1
            and getattr(self.args, "prefill_progcfg", None) is not None
            and os.environ.get("QWEN9B_MLP_DOWN_AUTO") != "1"
        ):
            down_pc = self.args.prefill_progcfg(T, hidden.shape[-1], w.w2.shape[-1])
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        ttnn.deallocate(hidden)
        return output

    def _forward_tp(self, x):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        if getattr(self, "_dram_sharded", False):
            # DRAM-WIDTH_SHARDED w1/w3. Sharded kernel needs M=1 tile (32 rows, decode).
            # Use x.shape[-2] for seq/M (not x.shape[1], Z=1 in both modes). Prefill
            # (M>1 tile) uses 2D progcfg. Convert outputs to DRAM-interleaved for rest of path.
            seq = x.shape[-2]
            if seq <= ttnn.TILE_SIZE:
                x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
                w1_out = ttnn.linear(
                    x_sh,
                    w.w1,
                    compute_kernel_config=ckc,
                    program_config=args.mlp_w1_progcfg,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                w3_out = ttnn.linear(
                    x_sh,
                    w.w3,
                    compute_kernel_config=ckc,
                    program_config=args.mlp_w3_progcfg,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                )
                ttnn.deallocate(x_sh)
                # Decode: keep gate/up outputs in L1 (they feed the L1 mul → w2 matmul, all
                # consumed before the all-reduce), avoiding the L1→DRAM→L1 round-trip.
                w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
                w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
            else:
                # NOT fused via all_gather_minimal_matmul_async: w1+w3 are two consumers of the gathered
                # norm output, and the op can't return the gathered activation, so fusing = 2 gathers
                # (regressed TTFT). The ff_norm's single all-gather here is already the one-gather optimum.
                # Fused SILU in progcfg; sharded/2D kernel rejects activation= kwarg with progcfg.
                pc_gate = tpc.create_prefill_matmul_program_config(
                    seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU
                )
                pc_up = args.prefill_progcfg(seq, args.dim, w.w3.shape[-1])
                w1_out = ttnn.linear(x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=mc)
                w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=mc)
                _silu_fused = True
        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            # Prefill (M>1 tile, compute-bound): FPU-tuned 2D config (grid width -> 1x4 subblock,
            # in0_block_w=4) beats ttnn-auto's 1x1 stall ~2.7x (test_mlp_matmul_sweep_prefill). SILU fused.
            seq = x.shape[-2]
            # max_cols = device worker-grid width (11 on BH): wide grid (gate/up -> 9x10) vs old 8-wide.
            _gw = getattr(args, "decode_grid_w", 8)
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(seq, args.dim, w.w3.shape[-1], max_cols=_gw)
            # L1 output (gate/up outputs; down output via mc_out below): +FPU, avoids the DRAM round-trip
            # (test_mlp_matmul_sweep_prefill *_outL1). The [seq,N] tensors fit L1 at the prefill chunk.
            w1_out = ttnn.linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _silu_fused = True
        else:
            # Interleaved weights: auto matmul program for decode and prefill.
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        # gated activation (down-proj INPUT): L1 in decode, DRAM in prefill. The L1 win is OUTPUT-only;
        # keeping both down input (hidden) and output (partial) in L1 at seq 2048 overflows L1.
        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
        # gate * up. Standalone silu only on DRAM-sharded decode path (SILU not fused there).
        if _silu_fused:
            hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out)
            ttnn.deallocate(w1_out)
        else:
            w1_act = ttnn.silu(w1_out, memory_config=mc_out)
            ttnn.deallocate(w1_out)
            hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out)
            ttnn.deallocate(w1_act)
        ttnn.deallocate(w3_out)
        # Prefill w2: 2D progcfg on (8,10); decode (M<=32) keeps ttnn-auto.
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            w2_pc = args.mlp_w2_decode_1d_progcfg
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            # Prefill down-proj: subblock-tuned 2D config with the wide grid (max_cols=device width),
            # off the generic 8-wide prefill_progcfg. Output L1 via mc_w2_out below.
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2], hidden.shape[-1], w.w2.shape[-1], max_cols=getattr(args, "decode_grid_w", 8)
            )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial).
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc)
        ttnn.deallocate(hidden)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
