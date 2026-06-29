# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP for Qwen3.5: down(silu(gate(x)) * up(x)).

Single device (9B): full weights, dense matmuls (unchanged validated path).
Tensor-parallel (27B on a (1,4) mesh): w1/w3 column-parallel, w2 row-parallel,
followed by tt_all_reduce (which reduce-scatters on a mesh with a 1 in its shape,
e.g. P150x4) so the output is fractured along the hidden dim.
"""
import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj  [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj  [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj    [in, out], bfloat4_b


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """state_dict is the per-layer mlp substate: keys 'gate_proj.weight', 'down_proj.weight', 'up_proj.weight'."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # Tensor-parallel: column-parallel w1/w3 (shard output dim), row-parallel
        # w2 (shard input dim). DRAM-sharded weight memcfgs come from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # Store w1/w3 (gate/up) DRAM-WIDTH_SHARDED so the decode matmul (M=1 tile)
        # reads weights at near-peak DRAM bandwidth (~+10% decode tok/s). w2 (down)
        # stays interleaved. The DRAM-sharded weight is layout-incompatible with an
        # interleaved cache file, so its cache name is qualified with `.dramshard`
        # (ttnn.as_tensor reloads a cache file as-is, ignoring the requested
        # memory_config — see weight_cache_path note). Fall back to interleaved if
        # the DRAM-sharded memcfgs are unavailable (e.g. single device).
        dram_sharded = args is not None and getattr(args, "mlp_w1_weight_memcfg", None) is not None

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

        # Default: keep each device's shard INTERLEAVED in DRAM so a regular
        # ttnn.linear serves both decode (M=1) and prefill (M=seq_len).
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

    # Gate and up projections use bfloat4_b (halves memory bandwidth for these large matmuls).
    # Down projection stays at bfloat8_b (on the critical accuracy path).
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
        # Must mirror the `dram_sharded` condition in load_mlp_weights so the
        # forward path matches the loaded weight layout.
        self._dram_sharded = (
            self.num_devices > 1 and args is not None and getattr(args, "mlp_w1_weight_memcfg", None) is not None
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
        """Tensor-parallel forward. Input x is replicated (full hidden dim) on
        every device; output is fractured along the hidden dim (reduce-scatter)."""
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        if getattr(self, "_dram_sharded", False):
            # w1/w3 weights are DRAM-WIDTH_SHARDED. The DRAM-sharded matmul kernel
            # only supports a single input tile-row (M == 1 tile = 32 rows), which
            # is exactly the decode shape (B<=32 users padded to one tile). Gate on
            # the real sequence/M dim x.shape[-2] (32 in decode, 2048 in prefill) —
            # NOT x.shape[1], which is the Z dim and is 1 in BOTH modes. Prefill
            # (M>1 tile) uses the 2D program config. Outputs are converted back to
            # DRAM-interleaved so silu/mul/w2/reduce-scatter are unchanged.
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
                w1_out = ttnn.to_memory_config(w1_out, mc)
                w3_out = ttnn.to_memory_config(w3_out, mc)
            else:
                pc = args.prefill_progcfg(seq, args.dim, w.w1.shape[-1])
                w1_out = ttnn.linear(x, w.w1, compute_kernel_config=ckc, program_config=pc, memory_config=mc)
                w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, program_config=pc, memory_config=mc)
        else:
            # Interleaved weights → let ttnn auto-select the matmul program (serves
            # both decode and prefill).
            w1_out = ttnn.linear(x, w.w1, compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)

        # SILU applied separately, then gate * up.
        w1_act = ttnn.silu(w1_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        hidden = ttnn.mul(w1_act, w3_out, memory_config=mc)
        ttnn.deallocate(w1_act)
        ttnn.deallocate(w3_out)
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)

        # Reduce across devices. On a (1,4) mesh tt_all_reduce reduce-scatters,
        # leaving the output fractured along the hidden dim (dim=3).
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
