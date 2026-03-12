# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTNN implementation of Qwen3.5 GatedDeltaNet (linear attention) layer.

Phase-2 implementation strategy
---------------------------------
Linear projections (in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj)
run as ttnn.linear matmuls on-device with DRAM-sharded weights and
MatmulMultiCoreReuseMultiCastProgramConfig (prefill) or
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig (decode).

The causal conv1d and the gated delta-rule recurrent step are mathematically novel
operations that do not yet have corresponding TTNN primitives.  They remain on CPU
via ttnn -> torch -> ttnn bridges.  A later phase will map these to Tensix kernels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step or multi-step gated delta rule on CPU.

    query/key/value : (B, T, n_heads, head_dim)
    g/beta          : (B, T, n_heads)
    initial_state   : (B, n_heads, k_dim, v_dim) or None
    Returns (output, new_state)  where output is (B, T, n_heads, v_dim).
    """
    query = _l2norm(query)
    key = _l2norm(key)
    # Transpose to (B, n_heads, T, dim)
    query, key, value, beta, g = [x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)]
    B, H, T, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = k_dim**-0.5
    query = query * scale
    state = (
        torch.zeros(B, H, k_dim, v_dim, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    outputs = torch.zeros(B, H, T, v_dim, dtype=torch.float32, device=query.device)
    for t in range(T):
        q_t = query[:, :, t]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    outputs = outputs.transpose(1, 2).to(query.dtype)
    return outputs, state


def _rms_norm_gated_cpu(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm with SiLU gate: x_normed * w * silu(gate)."""
    dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    x = weight * x.to(dtype)
    return x * F.silu(gate.float()).to(dtype)


class GatedDeltaNetTT(LightweightModule):
    """TTNN GatedDeltaNet layer for Qwen3.5 linear_attention layers.

    Weight keys (after map_hf_to_meta_keys_qwen3_5):
        {prefix}.attention.in_proj_qkv.weight
        {prefix}.attention.in_proj_z.weight
        {prefix}.attention.in_proj_b.weight
        {prefix}.attention.in_proj_a.weight
        {prefix}.attention.conv1d.weight
        {prefix}.attention.out_proj.weight
        {prefix}.attention.norm.weight
        {prefix}.attention.A_log
        {prefix}.attention.dt_bias
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict: dict,
        weight_cache_path,
        layer_num: int,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.hidden_size = args.dim
        self.norm_eps = args.norm_eps

        prefix = args.get_state_dict_prefix("Attention", layer_num)

        def load_tensor(key_suffix, weight_key="weight"):
            full_key = f"{prefix}.{key_suffix}.{weight_key}"
            return state_dict[full_key]

        if weight_cache_path is None or args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        # Projection dimensions:
        #   in_proj_qkv : K=hidden_size  N=conv_dim   (10240 for Qwen3.5-27B)
        #   in_proj_z   : K=hidden_size  N=value_dim  (6144)
        #   in_proj_b/a : K=hidden_size  N=num_v_heads (48 — too small for DRAM sharding)
        #   out_proj    : K=value_dim    N=hidden_size (5120)
        _h = self.hidden_size  # 5120
        _c = self.conv_dim  # 10240
        _v = self.value_dim  # 6144

        def as_tt_dram_sharded(name, suffix, weight, k: int, n: int):
            """Load weight transposed as (1,1,k,n) into DRAM width-sharded memory.

            DRAM-sharded layout enables the DRAMSharded matmul kernel in decode mode.
            """
            w = torch.transpose(weight, -2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=args.create_dram_sharded_mem_config(k, n),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(f"{name}.{suffix}"),
            )

        def as_tt_dram(name, suffix, weight):
            """Load weight transposed as DRAM-interleaved (for small N weights)."""
            w = torch.transpose(weight, -2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(f"{name}.{suffix}"),
            )

        # Large projections → DRAM-sharded for efficient DRAMSharded decode kernel
        self.in_proj_qkv = as_tt_dram_sharded("in_proj_qkv", "weight", load_tensor("in_proj_qkv"), _h, _c)
        self.in_proj_z = as_tt_dram_sharded("in_proj_z", "weight", load_tensor("in_proj_z"), _h, _v)
        self.out_proj = as_tt_dram_sharded("out_proj", "weight", load_tensor("out_proj"), _v, _h)

        # Small projections → regular DRAM-interleaved (N=48 not DRAM-shard-aligned)
        self.in_proj_b = as_tt_dram("in_proj_b", "weight", load_tensor("in_proj_b"))
        self.in_proj_a = as_tt_dram("in_proj_a", "weight", load_tensor("in_proj_a"))

        # CPU tensors for conv1d / recurrent / norm (no TTNN primitives yet)
        self._conv1d_weight = load_tensor("conv1d").float()  # (conv_dim, 1, kernel_size)
        self._norm_weight = load_tensor("norm").float()  # (head_v_dim,)
        self._A_log = state_dict[f"{prefix}.A_log"].float()
        self._dt_bias = state_dict[f"{prefix}.dt_bias"].float()

        # Compute kernel: HiFi2, no fp32 accumulation (avoids L1 overflow)
        self._compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Memory config for out_proj input in decode mode.
        # Must use the same core grid as _decode_pc (attn_input_grid) so the
        # DRAMSharded matmul's in0 sharding matches the compute grid.
        _out_proj_cores = args.attn_input_grid  # CoreGrid(x=8, y=4) = 32 cores on WH B0
        self._out_proj_input_mem_decode = ttnn.create_sharded_memory_config(
            (args.tile_padded_batch_rows, self.value_dim // _out_proj_cores.num_cores),
            _out_proj_cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # ------------------------------------------------------------------
    def _prefill_pc(self, seq_len: int, n_out: int):
        """MatmulMultiCoreReuseMultiCast program config for prefill."""
        tile = self.args.tile_size  # 32
        per_core_M = max(1, math.ceil(seq_len / (tile * 8)))
        per_core_N = math.ceil(n_out / (tile * self.args.dram_shard_grid_width))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= 2048,
        )

    def _decode_pc(self, k: int, n: int):
        """DRAMSharded program config for decode.

        Use attn_input_grid.num_cores as explicit num_cores so every projection's
        compute grid matches the input L1 sharding (from DistributedNorm "attn" output
        or _out_proj_input_mem_decode).  dram_shard_core_grid_for_k_and_n can return a
        different count (e.g. 40 for in_proj_qkv), causing a shard/block-width mismatch.
        """
        num_cores = self.args.attn_input_grid.num_cores
        return self.args.dram_matmul_config(
            m=self.args.tile_padded_batch_rows,
            k=k,
            n=n,
            num_cores=num_cores,
        )

    def _small_pc(self, seq_len: int, n_out: int):
        """Program config for small N projections (in_proj_b/a with N=48)."""
        # N=48 is 1.5 tiles wide - just use a simple 1D matmul
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(1, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(1, math.ceil(seq_len / self.args.tile_size)),
            per_core_N=math.ceil(n_out / self.args.tile_size),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )

    def _linear_large(
        self,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        mode: Mode,
        k: int,
        n: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        """TTNN linear for DRAM-sharded large projections.

        In decode mode:
          - DRAMSharded program config requires both input AND output to be L1 sharded.
          - Input arrives sharded (from residual-sharded attention input or out_proj sharding).
          - Output uses L1_WIDTH_SHARDED_MEMORY_CONFIG (TTNN fills in shard spec from matmul grid).
        In prefill mode:
          - MatmulMultiCoreReuseMultiCast with DRAM interleaved I/O.
        """
        if mode == Mode.PREFILL:
            pc = self._prefill_pc(seq_len, n)
            out_mem = ttnn.DRAM_MEMORY_CONFIG
        else:
            pc = self._decode_pc(k, n)
            out_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        return ttnn.linear(
            x,
            w,
            program_config=pc,
            compute_kernel_config=self._compute_cfg,
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
        )

    def _linear_small(
        self,
        x: ttnn.Tensor,
        w: ttnn.Tensor,
        seq_len: int,
        n_out: int,
    ) -> ttnn.Tensor:
        """TTNN linear for small DRAM-interleaved projections (in_proj_b/a).

        MatmulMultiCoreReuseMultiCast requires interleaved (non-sharded) input, so we
        de-shard x when it arrives sharded (decode mode with residual-sharded input).
        """
        if x.is_sharded():
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        pc = self._small_pc(seq_len, n_out)
        return ttnn.linear(
            x,
            w,
            program_config=pc,
            compute_kernel_config=self._compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def _tt_to_cpu(self, t: ttnn.Tensor) -> torch.Tensor:
        """Collect mesh tensor to CPU torch, shape (1, T, dim)."""
        out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return out.squeeze(0).squeeze(0).unsqueeze(0)  # (1, 1, BT, D) -> (1, BT, D)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,
        mode: Mode,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, torch.Tensor, torch.Tensor]:
        """
        x               : ttnn tensor (1, 1, B*T, hidden_size)
        conv_state      : (B, conv_dim, kernel-1)  CPU torch
        recurrent_state : (B, n_v_heads, k_dim, v_dim)  CPU torch
        Returns (output_ttnn, new_conv_state, new_recurrent_state).
        """
        # ---- Determine seq_len from TTNN tensor shape ----
        seq_len = x.shape[-2]  # B*T for prefill; tile_padded_rows for decode
        if mode == Mode.DECODE:
            seq_len_cpu = 1
        else:
            seq_len_cpu = seq_len

        # ---- On-device projections ----
        mixed_qkv_tt = self._linear_large(x, self.in_proj_qkv, mode, self.hidden_size, self.conv_dim, seq_len)
        z_tt = self._linear_large(x, self.in_proj_z, mode, self.hidden_size, self.value_dim, seq_len)
        b_tt = self._linear_small(x, self.in_proj_b, seq_len, self.num_v_heads)
        a_tt = self._linear_small(x, self.in_proj_a, seq_len, self.num_v_heads)

        # ---- Bring projections to CPU for conv1d / recurrent ----
        mixed_qkv = self._tt_to_cpu(mixed_qkv_tt)[:, :seq_len_cpu, :]  # (1, T, conv_dim)
        z = self._tt_to_cpu(z_tt)[:, :seq_len_cpu, :]  # (1, T, value_dim)
        b = self._tt_to_cpu(b_tt)[:, :seq_len_cpu, :]  # (1, T, num_v_heads)
        a = self._tt_to_cpu(a_tt)[:, :seq_len_cpu, :]  # (1, T, num_v_heads)
        ttnn.deallocate(mixed_qkv_tt)
        ttnn.deallocate(z_tt)
        ttnn.deallocate(b_tt)
        ttnn.deallocate(a_tt)

        batch, seq_len_cpu, _ = mixed_qkv.shape  # B=1 for now

        # Transpose for conv1d: (B, conv_dim, T)
        mixed_qkv_t = mixed_qkv.transpose(1, 2).float()

        # ---- Causal depthwise conv1d (CPU) ----
        if conv_state is None:
            conv_state = torch.zeros(batch, self.conv_dim, self.conv_kernel - 1, dtype=torch.float32)
        else:
            conv_state = conv_state.float()

        if seq_len_cpu == 1:
            # Decode: rolling buffer
            combined = torch.cat([conv_state, mixed_qkv_t], dim=-1)  # (B, conv_dim, kernel_size)
            conv_out = F.conv1d(
                combined,
                self._conv1d_weight.squeeze(1).unsqueeze(1),
                padding=0,
                groups=self.conv_dim,
            )
            mixed_qkv_conv = F.silu(conv_out[:, :, -1:]).transpose(1, 2)  # (B, 1, conv_dim)
            conv_state_new = combined[:, :, -(self.conv_kernel - 1) :]
        else:
            # Prefill: prepend state for causal context
            padded = torch.cat([conv_state, mixed_qkv_t], dim=-1)  # (B, conv_dim, kernel-1+T)
            conv_out = F.conv1d(
                padded,
                self._conv1d_weight.squeeze(1).unsqueeze(1),
                padding=0,
                groups=self.conv_dim,
            )
            mixed_qkv_conv = F.silu(conv_out[:, :, -seq_len_cpu:]).transpose(1, 2)  # (B, T, conv_dim)
            conv_state_new = padded[:, :, -(self.conv_kernel - 1) :]

        # ---- Split conv output → Q, K, V ----
        query, key, value = torch.split(mixed_qkv_conv.float(), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.view(batch, seq_len_cpu, self.num_k_heads, self.head_k_dim)
        key = key.view(batch, seq_len_cpu, self.num_k_heads, self.head_k_dim)
        value = value.view(batch, seq_len_cpu, self.num_v_heads, self.head_v_dim)

        # Gates
        beta = b.sigmoid()
        g = -self._A_log.float().exp() * F.softplus(a.float() + self._dt_bias.float())

        # GQA expansion
        n_rep = self.num_v_heads // self.num_k_heads
        if n_rep > 1:
            query = query.repeat_interleave(n_rep, dim=2)
            key = key.repeat_interleave(n_rep, dim=2)

        # ---- Gated delta rule (CPU) ----
        core_out, recurrent_state_new = _recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state=recurrent_state
        )

        # ---- Gated RMSNorm (CPU) ----
        core_out = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim).float()
        core_out = _rms_norm_gated_cpu(core_out.float(), z_flat, self._norm_weight, self.norm_eps)
        core_out = core_out.reshape(1, 1, batch * seq_len_cpu, self.value_dim).to(torch.bfloat16)

        # ---- Put intermediate back on device for out_proj ----
        if mode == Mode.DECODE:
            # Pad to tile_padded_batch_rows so the DRAMSharded out_proj matmul sees M=32.
            pad_rows = self.args.tile_padded_batch_rows - batch * seq_len_cpu
            if pad_rows > 0:
                pad = torch.zeros(1, 1, pad_rows, self.value_dim, dtype=torch.bfloat16)
                core_out = torch.cat([core_out, pad], dim=-2)
            core_out_tt = ttnn.from_torch(
                core_out,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            core_out_tt = ttnn.to_memory_config(core_out_tt, self._out_proj_input_mem_decode)
        else:
            core_out_tt = ttnn.from_torch(
                core_out,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        out_tt = self._linear_large(
            core_out_tt, self.out_proj, mode, self.value_dim, self.hidden_size, batch * seq_len_cpu
        )
        ttnn.deallocate(core_out_tt)

        return out_tt, conv_state_new, recurrent_state_new
