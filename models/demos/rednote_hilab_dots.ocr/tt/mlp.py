# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr language-model Qwen2 MLP block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`mlp_forward`

Qwen2MLP SwiGLU (no bias):

    h   = silu(gate_proj(x)) * up_proj(x)
    out = down_proj(h)

hidden_size 1536, intermediate_size 8960, no bias.

The forward runs entirely with ttnn ops (linear / silu / mul); no host-side
matmul or activation. gate_proj and up_proj share the same input and output
width, so they are fused into a single [dim, 2*intermediate] linear and split on
device — one matmul instead of two — before applying SiLU to the gate half.

Reference TTNN impl this follows: models/demos/rednote_hilab_dots.ocr/tt/vision_mlp.py
"""
import importlib.util
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_TT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sc = _load_sibling("dots_mlp_sharded_config", "sharded_config.py")


class TtMLP(LightweightModule):
    """dots.ocr LM Qwen2 SwiGLU MLP.

    Args:
        device: ttnn Device or MeshDevice.
        gate_weight: torch.Tensor [intermediate, dim] (gate proj, no bias).
        up_weight: torch.Tensor [intermediate, dim] (up proj, no bias).
        down_weight: torch.Tensor [dim, intermediate] (down proj, no bias).
        dtype: legacy activation dtype hint (bf16); retained for API compat.
        weight_dtype: storage dtype for the gate_up + down weight tensors.
            Defaults to bfloat8_b: at decode (seq=1) both MLP matmuls are
            DRAM-bandwidth-bound on the weight read (gate_up ~27.5M params,
            down ~13.8M params), so halving the weight bytes to bf8 removes the
            dominant read cost — the same win proven on lm_head (-36%, PCC
            0.99997). Activations + HiFi4 fp32_dest_acc compute stay unchanged.
    """

    def __init__(
        self,
        device,
        gate_weight,
        up_weight,
        down_weight,
        dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sharded_decode=False,
    ):
        super().__init__()
        self.device = device
        self.sharded_decode = sharded_decode

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        intermediate = gate_weight.shape[0]
        self.intermediate = intermediate

        # Fuse gate_proj and up_proj into one [dim, 2*intermediate] linear.
        # ttnn.linear computes x @ W when W is [in, out]; pass the torch weight
        # transposed. Concatenate along the output dim: [gate | up].
        gate_t = gate_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        up_t = up_weight.transpose(0, 1).contiguous()  # [dim, intermediate]
        gate_up = torch.cat([gate_t, up_t], dim=-1)  # [dim, 2*intermediate]
        self.gate_up_weight = ttnn.as_tensor(
            gate_up,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        self.down_weight = ttnn.as_tensor(
            down_weight.transpose(0, 1).contiguous(),  # [intermediate, dim]
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # fp32 compute to match the reference float path.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # ---- DRAM-sharded decode weight DUPS (bf8), built alongside the bf16/bf8
        # interleaved prefill weights above which are LEFT UNTOUCHED. Only built
        # when sharded_decode=True. gate_up: k=dim, n=2*intermediate; down:
        # k=intermediate, n=dim. ~+20 MB DRAM total at bf8.
        self.dim = down_weight.shape[0]
        self._dec_gate_up = None
        self._dec_down = None
        self._dec_gu_cores = None
        self._dec_down_cores = None
        if sharded_decode:
            # HiFi2 for the DRAM-sharded decode matmuls (flop-bound otherwise).
            self.decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            gu_k, gu_n = self.dim, 2 * intermediate
            self._dec_gu_cores, _ = _sc.sharded_matmul_plan(device, 32, gu_k, gu_n)
            if self._dec_gu_cores is not None:
                self._dec_gate_up = ttnn.as_tensor(
                    gate_up,
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_sc.create_dram_sharded_mem_config(device, gu_k, gu_n),
                    mesh_mapper=mesh_mapper,
                )
            d_k, d_n = intermediate, self.dim
            self._dec_down_cores, _ = _sc.sharded_matmul_plan(device, 32, d_k, d_n)
            if self._dec_down_cores is not None:
                self._dec_down = ttnn.as_tensor(
                    down_weight.transpose(0, 1).contiguous(),
                    device=device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=_sc.create_dram_sharded_mem_config(device, d_k, d_n),
                    mesh_mapper=mesh_mapper,
                )

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Width-sharded DRAM-sharded-matmul decode MLP. x: [32, dim] -> [32, dim].

        Falls back to interleaved per-matmul if that shape was not expressible as a
        DRAM-sharded matmul (recorded at construction). The SwiGLU elementwise chain
        runs on the width-sharded intermediate to avoid an extra reshard.
        """
        intermediate = self.intermediate
        # ---- gate_up (DRAM-sharded if available) ----
        if self._dec_gate_up is not None:
            x_ws = ttnn.to_memory_config(x, _sc.width_sharded_l1_config(32, self.dim, self._dec_gu_cores))
            gate_up = ttnn.linear(
                x_ws,
                self._dec_gate_up,
                program_config=_sc.dram_matmul_config(32, self.dim, 2 * intermediate, self._dec_gu_cores),
                compute_kernel_config=self.decode_compute_kernel_config,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_ws)
        else:
            gate_up = ttnn.linear(
                x,
                self.gate_up_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # SwiGLU on the (sharded or interleaved) gate_up intermediate. slice/silu/mul
        # keep whatever layout gate_up has; interleave back for the down matmul input.
        mem = gate_up.memory_config()
        gate = ttnn.slice(gate_up, [0, 0], [gate_up.shape[0], intermediate], memory_config=mem)
        up = ttnn.slice(gate_up, [0, intermediate], [gate_up.shape[0], 2 * intermediate], memory_config=mem)
        ttnn.deallocate(gate_up)
        h = ttnn.mul(ttnn.silu(gate, memory_config=mem), up, memory_config=mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # ---- down (DRAM-sharded if available) ----
        if self._dec_down is not None:
            # down needs the input width-sharded on its own core count (8); reshard.
            h_ws = ttnn.to_memory_config(h, _sc.width_sharded_l1_config(32, intermediate, self._dec_down_cores))
            ttnn.deallocate(h)
            out = ttnn.linear(
                h_ws,
                self._dec_down,
                program_config=_sc.dram_matmul_config(32, intermediate, self.dim, self._dec_down_cores),
                compute_kernel_config=self.decode_compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(h_ws)
            # Hand a normal interleaved L1 tensor back to the residual add.
            out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
            return out
        else:
            h_il = ttnn.sharded_to_interleaved(h, ttnn.L1_MEMORY_CONFIG) if h.is_sharded() else h
            out = ttnn.linear(
                h_il,
                self.down_weight,
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (TILE layout) -> [seq, dim]."""
        # Pin the gate/up split + SwiGLU elementwise chain to L1 for decode and
        # short prefills (the [seq, 8960] bf16 intermediate fits L1). At a large
        # prefill (e.g. a full-document vision prompt, seq in the thousands) that
        # intermediate is tens of MB and overflows L1, so fall back to DRAM above
        # a tile-friendly threshold. Decode (seq=1) keeps the L1 fast path.
        mem = ttnn.L1_MEMORY_CONFIG if x.shape[0] <= 1024 else ttnn.DRAM_MEMORY_CONFIG

        # Fused gate/up projection: [seq, dim] @ [dim, 2*intermediate].
        # bf8 intermediate: the [seq, 2*intermediate] gate_up tensor is the widest
        # activation in the block; emitting it bf8 halves its write AND every
        # downstream read (slice/silu/mul/down all flow bf8). The residual-stream
        # output (down, below) stays bf16. PCC headroom is ample (~0.9999).
        gate_up = ttnn.linear(
            x,
            self.gate_up_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
        )
        gate = ttnn.slice(
            gate_up,
            [0, 0],
            [gate_up.shape[0], self.intermediate],
            memory_config=mem,
        )  # gate_proj
        up = ttnn.slice(
            gate_up,
            [0, self.intermediate],
            [gate_up.shape[0], 2 * self.intermediate],
            memory_config=mem,
        )  # up_proj

        # SwiGLU: silu(gate) * up.
        h = ttnn.mul(
            ttnn.silu(gate, memory_config=mem),
            up,
            memory_config=mem,
        )

        # Down projection: [seq, intermediate] @ [intermediate, dim].
        out = ttnn.linear(
            h,
            self.down_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )
        return out
