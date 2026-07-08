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
    w1: ttnn.Tensor  # gate_proj  [in, out], bfloat16
    w2: ttnn.Tensor  # down_proj  [in, out], bfloat16
    w3: ttnn.Tensor  # up_proj    [in, out], bfloat16


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """state_dict is the per-layer mlp substate: keys 'gate_proj.weight', 'down_proj.weight', 'up_proj.weight'."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # Tensor-parallel: column-parallel w1/w3 (shard output dim), row-parallel
        # w2 (shard input dim). DRAM-sharded weight memcfgs come from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        def cache(name):
            return str(tensor_cache_path / f"mlp.{name}.weight.tp") if tensor_cache_path else None

        # Shard across devices (column-parallel w1/w3, row-parallel w2) but keep
        # each device's shard INTERLEAVED in DRAM so a regular ttnn.linear serves
        # both decode (M=1) and prefill (M=seq_len). DRAM-width-sharding the
        # weights for a faster decode matmul is a later optimization.
        return MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat16,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat16,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=ttnn.bfloat16,
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

    # All three projections use bfloat16 for maximum precision (previously gate/up
    # were bfloat4_b and down was bfloat8_b to save memory bandwidth). This roughly
    # doubles/quadruples the MLP weight footprint but keeps full bf16 precision.
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat16),
        w2=load("down_proj", ttnn.bfloat16),
        w3=load("up_proj", ttnn.bfloat16),
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        # HiFi2 (was LoFi): LoFi only consumes the top mantissa bits of the inputs,
        # which is fine for bf4/bf8 weights but discards most of the precision of the
        # bf16 weights now used above. HiFi2 processes more mantissa bits so the bf16
        # upgrade is not wasted (at some throughput cost).
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
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

        # Interleaved weights → let ttnn auto-select the matmul program (serves
        # both decode and prefill). SILU applied separately, then gate * up.
        mc = ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
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
