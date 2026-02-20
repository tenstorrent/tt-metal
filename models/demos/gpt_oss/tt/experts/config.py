# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Generic expert configuration interfaces.
No model-specific code - models provide their own ProgramConfig implementations.
"""

import math
from dataclasses import dataclass

import ttnn


@dataclass
class ExpertConfig:
    """Core expert configuration - model agnostic"""

    intermediate_size: int
    num_experts: int
    hidden_size: int
    num_experts_per_tok: int
    swiglu_limit: float
    alpha: float = 1.702


@dataclass
class ProgramConfig:
    """
    Base configuration for expert program configs.

    Models just need to specify grid sizes and chunking parameters.
    The boilerplate MatmulProgramConfig generation is handled automatically.

    Example:
        # GPT-OSS config
        config = ProgramConfig(
            decode_gate_up_cores=(3, 4),
            decode_down_cores=(5, 6),
        )

        # Mixtral with different settings
        config = ProgramConfig(
            decode_gate_up_cores=(4, 6),
            decode_down_cores=(6, 8),
            sequence_chunk_size=2048,
        )
    """

    # Core grid sizes for decode
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_down_cores: tuple[int, int] = (5, 6)

    # Core grid sizes for prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_down_cores: tuple[int, int] = (5, 6)

    # Sparse matmul subblock widths
    decode_gate_up_subblock_w: int = 1
    decode_down_subblock_w: int = 1
    prefill_gate_up_subblock_w: int = 1
    prefill_down_subblock_w: int = 1

    # Input block widths (in0_block_w)
    decode_gate_up_in0_block_w: int = 1
    decode_down_in0_block_w: int = 1
    prefill_gate_up_in0_block_w: int = 1
    prefill_down_in0_block_w: int = 1

    # Chunking parameters
    sequence_chunk_size: int = 4 * 1024
    down_split_size: int = 1024

    def __post_init__(self):
        """Validate configuration on creation"""
        self._validate_cores("decode_gate_up_cores", self.decode_gate_up_cores)
        self._validate_cores("decode_down_cores", self.decode_down_cores)
        self._validate_cores("prefill_gate_up_cores", self.prefill_gate_up_cores)
        self._validate_cores("prefill_down_cores", self.prefill_down_cores)

        if self.sequence_chunk_size <= 0:
            raise ValueError(f"sequence_chunk_size must be positive, got {self.sequence_chunk_size}")
        if self.sequence_chunk_size % 32 != 0:
            raise ValueError(f"sequence_chunk_size must be multiple of 32, got {self.sequence_chunk_size}")

        if self.down_split_size <= 0:
            raise ValueError(f"down_split_size must be positive, got {self.down_split_size}")
        if self.down_split_size % 32 != 0:
            raise ValueError(f"down_split_size must be multiple of 32, got {self.down_split_size}")

    def _validate_cores(self, name: str, cores: tuple[int, int]):
        """Validate core grid dimensions"""
        if not isinstance(cores, tuple) or len(cores) != 2:
            raise ValueError(f"{name} must be a tuple of (x, y), got {cores}")

        core_x, core_y = cores
        if core_x <= 0 or core_y <= 0:
            raise ValueError(f"{name} must have positive dimensions, got {cores}")

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        in0_block_w: int = 1,
        out_subblock_w: int = 1,
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """
        Build MatmulProgramConfig with standard settings.

        This is the single source of truth for matmul config generation.
        All get_*_config methods use this builder.

        Args:
            cores: (core_x, core_y) grid dimensions
            m: Input height dimension
            n: Output width dimension
            in0_block_w: Block width for input tensor
            out_subblock_w: Output subblock width (for sparse matmuls)

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = cores
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=1,
            per_core_M=max(32, m) // 32,
            per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_gate_up_config(self, m: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode gate/up projections"""
        return self._build_matmul_config(
            self.decode_gate_up_cores,
            m,
            n,
            in0_block_w=self.decode_gate_up_in0_block_w,
            out_subblock_w=self.decode_gate_up_subblock_w,
        )

    def get_decode_down_config(self, m: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode down projection"""
        return self._build_matmul_config(
            self.decode_down_cores,
            m,
            n,
            in0_block_w=self.decode_down_in0_block_w,
            out_subblock_w=self.decode_down_subblock_w,
        )

    def get_prefill_gate_up_config(self, m: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill gate/up projections"""
        return self._build_matmul_config(
            self.prefill_gate_up_cores,
            m,
            n,
            in0_block_w=self.prefill_gate_up_in0_block_w,
            out_subblock_w=self.prefill_gate_up_subblock_w,
        )

    def get_prefill_down_config(self, m: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill down projection"""
        return self._build_matmul_config(
            self.prefill_down_cores,
            m,
            n,
            in0_block_w=self.prefill_down_in0_block_w,
            out_subblock_w=self.prefill_down_subblock_w,
        )
