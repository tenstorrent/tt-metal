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
    decode_qkv_cores: tuple[int, int] | None = None
    decode_qkv_in0_block_w: int = 1
    decode_qkv_out_subblock_h: int = 1
    decode_qkv_out_subblock_w: int = 1

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

    @staticmethod
    def get_decode_user_grid(mesh_device, batch_size: int):
        """Per-user core placement for decode: (cores holding user b's Q/K/V shard, SDPA program grid).

        Three consumers read a user's data from a core they *compute* rather than from the shard spec,
        so they must all agree: rotary_embedding_llama borrows cos/sin/trans_mat from the shard resident
        in the local core's L1 (RotarySetup places them, see models/tt_transformers/tt/rope.py
        get_batch_grid); paged SDPA decode's reducer/output core for batch b is (b % grid.x, b // grid.x)
        of its program grid and reads Q from that core's L1; and nlp_create_qkv_heads_decode places user
        b on the b-th core (row-major) of the grid it is given. RotarySetup's rule is: an 8x8 grid when
        the batch is a multiple of 32, otherwise the device compute grid (8x8 on Wormhole, 13x10 on
        Blackhole), row-major. Mirror it here and pick the SDPA grid with the matching width. Up to 8
        users both layouts coincide (first row), so the 8x8 SDPA grid is kept for them.
        """
        device_grid = mesh_device.compute_with_storage_grid_size()
        grid_8x8 = ttnn.CoreCoord(8, 8)
        # RotarySetup's rule (see docstring): 8x8 for multiples of 32; the first row of any grid coincides with the
        # 8x8 layout for <= 8 users; on an 8-wide device grid both layouts are the same anyway.
        rope_uses_8x8 = batch_size % 32 == 0
        fits_first_row = batch_size <= 8
        device_is_8_wide = device_grid.x == grid_8x8.x
        uses_8x8_grid = rope_uses_8x8 or fits_first_row or device_is_8_wide
        sdpa_grid = grid_8x8 if uses_8x8_grid else ttnn.CoreCoord(device_grid.x, device_grid.y)
        user_cores = ttnn.num_cores_to_corerangeset(batch_size, sdpa_grid, row_wise=True)
        return user_cores, sdpa_grid

    @staticmethod
    def get_decode_concat_grid(batch_size: int):
        """Single-rectangle core grid with one core per user for nlp_concat_heads_decode's input.

        The op derives its output grid from the bounding box of the input grid, so the input must be
        exactly one CoreRange (16 users laid out row-major on a 13-wide grid, 13 + 3, is not). Pick the
        widest w <= 8 that divides the batch with at most 8 rows; every practical batch (powers of two,
        multiples of 8) gets a full rectangle, and a batch that fits in one row stays a row.
        """
        if batch_size <= 8:
            return ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(8, 8), row_wise=True)
        for width in range(8, 0, -1):
            if batch_size % width == 0 and batch_size // width <= 8:
                return ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, batch_size // width - 1))}
                )
        raise ValueError(
            f"batch size {batch_size}: nlp_concat_heads_decode needs one core per user on a single rectangle of "
            "at most 8x8 cores and no w <= 8 with batch/w <= 8 divides this batch; pad the batch to a multiple "
            "of 8 or a power of two (<= 32)."
        )

    def get_decode_sdpa_config(self, mesh_device, batch_size: int = 1) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for decode mode"""
        _, sdpa_grid = self.get_decode_user_grid(mesh_device, batch_size)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
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
            out_block_w=1,
            per_core_M=m // 32,
            per_core_N=n // 32 // (core_x * core_y),
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
