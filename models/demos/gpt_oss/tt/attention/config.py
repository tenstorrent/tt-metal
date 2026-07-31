# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import ttnn


@dataclass
class AttentionConfig:
    """Core attention configuration - model agnostic"""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_local_batch_size: int

    users_row_sharded: bool = False
    sliding_window: int | None = None
    scaling: float | None = None  # Computed if None

    def __post_init__(self):
        """Compute scaling factor if not provided"""
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5


@dataclass
class ProgramConfig:
    """
    Base configuration for SDPA program configs.

    Models just need to specify chunk sizes and compute settings.
    The boilerplate SDPA config generation is handled automatically.
    """

    # Decode SDPA config
    decode_q_chunk_size: int = 0
    decode_k_chunk_size: int = 128

    # Prefill SDPA config
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    # Matmul program config parameters (optional - None means no program config)
    # Decode QKV projection
    # qkv [32,2880]x[2880,5120], IN0 is L1_INTERLEAVED in-model -- which is the
    # layout the offline sweep used, so the sweep result is applicable here (unlike
    # o_proj, whose IN0 is L1_WIDTH_SHARDED and takes a different program factory).
    # .auto/qkv_sweep.py: 8x5 cores, in0_block_w=2, out_subblock_w=4 -> 50.32 us/op
    # vs 63.66 us/op default, pcc 0.9998 vs the device baseline.
    # per_core_N = Nt/cores = 160/40 = 4 exactly, so the floor-division bug that
    # silently invalidated the o_proj config does not apply. out_block_w = osw = 4
    # gives num_blocks_w_dim = 1, so the PR #51514 corruption cannot trigger.
    decode_qkv_cores: tuple[int, int] | None = (8, 5)
    decode_qkv_in0_block_w: int = 2
    decode_qkv_out_subblock_h: int = 1
    decode_qkv_out_subblock_w: int = 4

    # Decode output projection
    decode_out_cores: tuple[int, int] | None = None
    decode_out_in0_block_w: int = 1
    decode_out_out_subblock_h: int = 1
    decode_out_out_subblock_w: int = 1

    # Prefill QKV projection
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_qkv_in0_block_w: int = 1
    prefill_qkv_out_subblock_h: int = 1
    prefill_qkv_out_subblock_w: int = 1

    # Prefill output projection
    prefill_out_cores: tuple[int, int] | None = None
    prefill_out_in0_block_w: int = 1
    prefill_out_out_subblock_h: int = 1
    prefill_out_out_subblock_w: int = 1

    def __post_init__(self):
        """Validate configuration on creation"""
        if self.decode_q_chunk_size < 0 or self.decode_k_chunk_size <= 0:
            raise ValueError("Decode chunk sizes must be non-negative (q) and positive (k)")

        if self.prefill_q_chunk_size_small <= 0 or self.prefill_k_chunk_size_small <= 0:
            raise ValueError("Prefill small chunk sizes must be positive")

        if self.prefill_q_chunk_size_large <= 0 or self.prefill_k_chunk_size_large <= 0:
            raise ValueError("Prefill large chunk sizes must be positive")

        if self.prefill_threshold <= 0:
            raise ValueError("Prefill threshold must be positive")

        # Validate math_fidelity
        valid_fidelities = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
        if self.math_fidelity not in valid_fidelities:
            raise ValueError(f"math_fidelity must be one of {valid_fidelities}, got {self.math_fidelity}")

    def get_decode_sdpa_config(self, mesh_device) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for decode mode"""
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=self.decode_q_chunk_size,
            k_chunk_size=self.decode_k_chunk_size,
            exp_approx_mode=False,
        )

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for prefill mode based on sequence length"""
        if seq_len >= self.prefill_threshold:
            q_chunk = self.prefill_q_chunk_size_large
            k_chunk = self.prefill_k_chunk_size_large
        else:
            q_chunk = self.prefill_q_chunk_size_small
            k_chunk = self.prefill_k_chunk_size_small

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """Get compute kernel config"""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        k: int,
        in0_block_w: int,
        out_subblock_h: int,
        out_subblock_w: int,
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Build matmul program config for attention projections"""
        core_x, core_y = cores
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            # out_block_w must follow out_subblock_w: in1_num_subblocks is
            # out_block_w / out_subblock_w, so a hardcoded 1 makes that 0 for any
            # out_subblock_w > 1 (the PR #51514 defect, also present in the expert
            # path until it was fixed there).
            out_block_w=out_subblock_w,
            # CEIL on the row count too: decode passes the LOGICAL seq len (1), so
            # floor gave per_core_M = 1//32 = 0 and the whole config was invalid --
            # ttnn silently fell back to its default heuristic. The tensor is
            # tile-padded to 32 rows, so 1 row still needs per_core_M = 1.
            per_core_M=max(1, -(-m // 32)),
            # CEIL, not floor. Nt is not always divisible by the core count, and a
            # too-small per_core_N silently produces an invalid config that ttnn
            # ignores (this is what made the o_proj attempt a no-op).
            per_core_N=-(-(n // 32) // (core_x * core_y)),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_qkv_config(self, m: int, n: int, k: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for decode QKV projection"""
        if self.decode_qkv_cores is None:
            return None
        return self._build_matmul_config(
            self.decode_qkv_cores,
            m,
            n,
            k,
            self.decode_qkv_in0_block_w,
            self.decode_qkv_out_subblock_h,
            self.decode_qkv_out_subblock_w,
        )

    def get_decode_out_config(self, m: int, n: int, k: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for decode output projection"""
        if self.decode_out_cores is None:
            return None
        return self._build_matmul_config(
            self.decode_out_cores,
            m,
            n,
            k,
            self.decode_out_in0_block_w,
            self.decode_out_out_subblock_h,
            self.decode_out_out_subblock_w,
        )

    def get_prefill_qkv_config(
        self, m: int, n: int, k: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for prefill QKV projection"""
        if self.prefill_qkv_cores is None:
            return None
        return self._build_matmul_config(
            self.prefill_qkv_cores,
            m,
            n,
            k,
            self.prefill_qkv_in0_block_w,
            self.prefill_qkv_out_subblock_h,
            self.prefill_qkv_out_subblock_w,
        )

    def get_prefill_out_config(
        self, m: int, n: int, k: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for prefill output projection"""
        if self.prefill_out_cores is None:
            return None
        return self._build_matmul_config(
            self.prefill_out_cores,
            m,
            n,
            k,
            self.prefill_out_in0_block_w,
            self.prefill_out_out_subblock_h,
            self.prefill_out_out_subblock_w,
        )
