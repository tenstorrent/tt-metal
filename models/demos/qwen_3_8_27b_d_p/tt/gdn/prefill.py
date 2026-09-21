# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet token mixer for the 48 ``linear_attention`` layers.

    in_proj_qkv (SP-local) -> ALL-GATHER over SP -> 4-tap causal conv + SiLU (whole chunk)
    -> chunked gated delta rule with a carried recurrent state -> mesh_partition back to SP
    -> silu-gated per-head RMSNorm against z -> out_proj -> TP all-reduce

**Why the all-gather.** This is a recurrent token mixer, not attention: the delta rule's state
update is sequential in the sequence and *state-dependent*, so it does not decompose into per-rank
pieces the way online-softmax attention does. Running the scan per SP rank with a zero initial
state gives each rank the wrong answer, and composing the ranks afterwards would need each rank's
``[k_dim, k_dim]`` transition matrix, which the op does not return.

So each SP row gathers the whole chunk's projected q/k/v (and beta/g), runs the scan over all
``chunk_size`` tokens, and keeps its own token block with ``mesh_partition`` (the inverse of
all-gather; a per-device slice, no fabric). The scan is therefore computed ``sp`` times over.
That costs **no wall-clock**: the op places one Tensix core per (batch x value head), which is 12
cores of the pod's ~130 per chip, and the other SP rows have nothing else to run at that moment.
It costs DRAM bandwidth and one collective per layer. Recorded as a known gap in ``README.md``.

The projections and ``z`` stay SP-local — gathering *after* them moves 2560 columns per chip
instead of the residual's 5120, and avoids doing the projection maths 8 times.

**Why the conv runs on the gathered chunk.** A 4-tap causal conv needs 3 tokens of left context.
On the gathered chunk there is no halo problem at all: the only history that crosses a boundary is
the previous *prefill chunk*'s, which is exactly ``GdnState.conv_state``.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule

from ...config import MeshConfig
from ...reference.config import Qwen35TextConfig
from ..caches import GdnState
from ..compute import matmul_compute_config
from ..context import ChunkContext
from .weights import GdnWeights, load_gdn_weights

# The chunked delta rule's tiling width. 32 rather than the reference's 64 because the device op's
# flat (relayout-free) q/k/v path requires it: the in-kernel L2 norm that path depends on is only
# available at chunk_size 32. The factorization is exact at any chunk size — the reference agrees
# with itself at 32 vs 64 to 1e-10 (test_gdn_scan_chunk_size_is_exact).
DEVICE_DELTA_CHUNK = 32


class GatedDeltaNet(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg: Qwen35TextConfig,
        state_dict: dict,
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        layer_idx: int,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path: Optional[str] = None,
        weights: Optional[GdnWeights] = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.layer_idx = layer_idx
        self.activation_dtype = activation_dtype

        tp = mesh_config.tp
        self.n_k_local = cfg.linear_num_key_heads // tp
        self.n_v_local = cfg.linear_num_value_heads // tp
        self.q_width = self.n_k_local * cfg.linear_key_head_dim
        self.k_width = self.q_width
        self.v_width = self.n_v_local * cfg.linear_value_head_dim
        self.conv_width = self.q_width + self.k_width + self.v_width
        self.kernel = cfg.linear_conv_kernel_dim

        self.weights = weights or load_gdn_weights(
            mesh_device,
            cfg,
            state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        self.matmul_config = matmul_compute_config(mesh_device)
        self.conv_program_config = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=min(512, self.conv_width))

    # --- pieces, each with its own PCC test ------------------------------------------------
    def gates(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """-> (beta ``[1,1,S,n_v_local]`` fp32, g ``[1,1,S,n_v_local]`` fp32 log-decay).

        ``g = -exp(A_log) * softplus(a + dt_bias)`` with ``-exp(A_log)`` folded at load. The whole
        expression stays fp32: in fp16 ``exp(A_log)`` can reach inf and every later chunk of the
        scan inherits a NaN state.
        """
        b = ttnn.linear(x, self.weights.in_proj_b, dtype=ttnn.float32, compute_kernel_config=self.matmul_config)
        beta = ttnn.sigmoid(b)
        b.deallocate(True)

        a = ttnn.linear(x, self.weights.in_proj_a, dtype=ttnn.float32, compute_kernel_config=self.matmul_config)
        shifted = ttnn.add(a, self.weights.dt_bias)
        a.deallocate(True)
        sp_a = ttnn.softplus(shifted)
        shifted.deallocate(True)
        g = ttnn.multiply(sp_a, self.weights.neg_a_exp)
        sp_a.deallocate(True)
        return beta, g

    def gather_sequence(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather an SP-local activation to the whole chunk, on the sequence dim."""
        if self.mesh_config.sp == 1:
            return x
        return self.mesh_config.allgather(x, self.ccl_manager, axis=self.mesh_config.sp_axis, dim=2)

    def causal_conv(
        self, mixed_qkv: ttnn.Tensor, history: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """``mixed_qkv`` ``[1, 1, T, conv_width]`` TILE + 3-token history -> (q, k, v, next_history).

        ``qkv_causal_conv1d_silu`` takes ROW_MAJOR ``[1, T, W]`` and returns TILE ``[1, T, *]``
        splits, so the untilize happens once here for all three outputs. The next chunk's history
        is the last ``kernel-1`` **pre-conv** tokens — the conv's own input, not its output.
        """
        t = mixed_qkv.shape[-2]
        rm = ttnn.to_layout(mixed_qkv, ttnn.ROW_MAJOR_LAYOUT)
        rm = ttnn.reshape(rm, [1, t, self.conv_width])
        next_history = ttnn.slice(rm, [0, t - (self.kernel - 1), 0], [1, t, self.conv_width])

        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            rm,
            history,
            self.weights.conv_taps[0],
            self.weights.conv_taps[1],
            self.weights.conv_taps[2],
            self.weights.conv_taps[3],
            self.q_width,
            self.k_width,
            self.v_width,
            program_config=self.conv_program_config,
        )
        rm.deallocate(True)
        return q, k, v, next_history

    def scan(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        g: ttnn.Tensor,
        beta: ttnn.Tensor,
        initial_state: Optional[ttnn.Tensor],
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Chunked gated delta rule. -> (o head-major ``[n_v_local, T, dv]``, final state).

        q/k/v go in **flat** (rank-3 ``[1, T, H*D]``): the op's prep reader tile-addresses them,
        does the GQA head map (value head ``hv`` reads key head ``hv // 3``) and the L2 norm in
        kernel, so nothing here has to relayout a 128-wide head out of a 512-wide row.

        ``output_head_major`` returns ``[B*HV, T, V]`` by pure metadata reshape and is exactly the
        layout the gated norm wants — the default token-major path would permute to ``[B,T,HV,V]``
        in ROW_MAJOR and the norm would permute straight back.
        """
        t = q.shape[-2]
        o, final_state = ttnn.transformer.chunk_gated_delta_rule(
            ttnn.reshape(q, [1, t, self.q_width]),
            ttnn.reshape(k, [1, t, self.k_width]),
            ttnn.reshape(v, [1, t, self.v_width]),
            ttnn.reshape(g, [1, t, self.n_v_local]),
            ttnn.reshape(beta, [1, t, self.n_v_local]),
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=DEVICE_DELTA_CHUNK,
            output_head_major=True,
        )
        return o, final_state

    def gated_norm(self, o_head_major: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        """Per-head RMSNorm then the **silu** gate, in the layout the scan already produced.

        ``ttnn.experimental.kda.sigmoid_gated_rms_norm`` computes ``normalized * weight *
        sigmoid(z)`` and, usefully, converts head-first ``[B*H, T, V]`` into token-first
        ``[B, T, H*V]`` on the way out — exactly the transpose ``out_proj`` needs. Qwen3.5 gates
        with ``silu(z) = z * sigmoid(z)``, so the remaining factor is one multiply by ``z``:

            normalized * weight * silu(z) == (normalized * weight * sigmoid(z)) * z

        which is why the sigmoid-gated kernel is usable here at all, and why calling it *without*
        that trailing multiply would be the silent-PCC-loss version of this block.
        """
        out = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o_head_major,
            z,
            self.weights.norm_weight,
            self.n_v_local,
            epsilon=self.cfg.rms_norm_eps,
            output_dtype=ttnn.bfloat16,
        )
        gated = ttnn.multiply(out, z)
        out.deallocate(True)
        return gated

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        state: Optional[GdnState] = None,
    ) -> ttnn.Tensor:
        """``x`` full-emb replicated, SP-sharded on the sequence -> same layout out.

        When ``state`` is given it is read for this chunk's initial conv/recurrent state and
        **updated in place** with the chunk's final state, which is how the runtime threads state
        across chunks without rebuilding the model.
        """
        s_local = x.shape[-2]
        mixed_local = ttnn.linear(
            x, self.weights.in_proj_qkv, dtype=self.activation_dtype, compute_kernel_config=self.matmul_config
        )
        z = ttnn.linear(
            x, self.weights.in_proj_z, dtype=self.activation_dtype, compute_kernel_config=self.matmul_config
        )
        beta_local, g_local = self.gates(x)

        mixed = self.gather_sequence(mixed_local)
        mixed_local.deallocate(True)
        beta = self.gather_sequence(beta_local)
        beta_local.deallocate(True)
        g = self.gather_sequence(g_local)
        g_local.deallocate(True)

        history = state.conv_state if state is not None else self._zero_history()
        q, k, v, next_history = self.causal_conv(mixed, history)
        mixed.deallocate(True)

        o, final_state = self.scan(q, k, v, g, beta, state.recurrent if state is not None else None)
        for t in (q, k, v, g, beta):
            t.deallocate(True)

        if self.mesh_config.sp > 1:
            o_local = ttnn.mesh_partition(o, 1, cluster_axis=self.mesh_config.sp_axis)
            o.deallocate(True)
        else:
            o_local = o
        assert o_local.shape[-2] == s_local, f"partitioned scan output {o_local.shape} != s_local {s_local}"

        z3 = ttnn.reshape(z, [1, s_local, self.v_width])
        gated = self.gated_norm(o_local, z3)
        o_local.deallocate(True)
        z.deallocate(True)

        out = ttnn.linear(
            ttnn.reshape(gated, [1, 1, s_local, self.v_width]),
            self.weights.out_proj,
            compute_kernel_config=self.matmul_config,
        )
        gated.deallocate(True)
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)

        if state is not None:
            state.conv_state.deallocate(True)
            state.recurrent.deallocate(True)
            state.conv_state = next_history
            state.recurrent = final_state
            state.seeded = True
        else:
            next_history.deallocate(True)
            final_state.deallocate(True)
        return out

    def _zero_history(self) -> ttnn.Tensor:
        """A stateless call still needs the conv's left context; all-zero is exactly what a
        one-shot forward's ``padding=kernel-1`` produces."""
        import torch

        return ttnn.from_torch(
            torch.zeros(1, self.kernel - 1, self.cfg.gdn_conv_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.shard_mapper(self.mesh_device, tensor_dim=-1),
        )

    def mix(self, x: ttnn.Tensor, ctx: ChunkContext) -> ttnn.Tensor:
        """The uniform token-mixer entry point ``DecoderLayer`` calls (see ``tt/context.py``)."""
        state = ctx.caches.gdn[self.layer_idx] if ctx.caches is not None else None
        return self.forward(x, state=state)
