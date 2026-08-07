# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    base_down_split_size: int = 1024

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

        if self.base_down_split_size <= 0:
            raise ValueError(f"down_split_size must be positive, got {self.base_down_split_size}")
        if self.base_down_split_size % 32 != 0:
            raise ValueError(f"down_split_size must be multiple of 32, got {self.base_down_split_size}")

    def _validate_cores(self, name: str, cores: tuple[int, int]):
        """Validate core grid dimensions"""
        if not isinstance(cores, tuple) or len(cores) != 2:
            raise ValueError(f"{name} must be a tuple of (x, y), got {cores}")

        core_x, core_y = cores
        if core_x <= 0 or core_y <= 0:
            raise ValueError(f"{name} must have positive dimensions, got {cores}")

    def get_down_split_size(self, seqlen: int) -> int:
        if seqlen <= 32 * 1024:
            return self.base_down_split_size
        else:
            # For very long sequences, decrease split size to avoid OOM
            return self.base_down_split_size // 2

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        in0_block_w: int = 1,
        out_subblock_w: int = 1,
        k: int = None,
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
            k: Input contraction dimension (used to snap in0_block_w to a
                divisor of Kt; passing None preserves the configured value).

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = cores
        num_cores = core_x * core_y
        Nt = int(math.ceil(n / 32))
        # Ceiling division: per_core_N = ceil(Nt / num_cores). The kernel then
        # computes num_blocks_x = ceil(Nt / per_core_N) and asserts it fits in
        # num_cores. Using floor division here breaks when Nt is not a multiple
        # of num_cores (e.g. tp=1 on a single Blackhole card: Nt=90, cores=12 →
        # floor=7 → 13 blocks > 12 cores). For all current Wormhole configs Nt
        # is divisible by num_cores so ceil and floor agree — no WH change.
        per_core_N = (Nt + num_cores - 1) // num_cores

        # The sparse matmul kernel asserts `Kt % in0_block_w == 0`. Different
        # tp factors produce different Kt (e.g. down's K = intermediate/tp:
        # tp=8 → Kt=12, tp=1 → Kt=90), and the configured in0_block_w may not
        # divide them all. Snap to the largest divisor of Kt that does not
        # exceed the configured ceiling; when Kt is already divisible by the
        # configured value this is a no-op (so existing Wormhole tunings are
        # preserved). When Kt is prime (or coprime with every value ≤
        # configured, e.g. Kt=23 on tp=4) the only divisor under the ceiling
        # is 1, which collapses the matmul to a tile-by-tile inner loop and
        # roughly halves prefill throughput. In that case fall back to Kt
        # itself — always a divisor, and small enough to fit in L1 for the
        # Kt range produced by realistic gpt-oss TP shardings (Kt ≤ ~90).
        if k is not None:
            Kt = int(math.ceil(k / 32))
            if Kt % in0_block_w != 0:
                divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
                in0_block_w = max(divisors) if divisors else Kt

        # Derive out_block_w so the subblock axis is expressible instead of
        # hardcoding it to 1. The mcast_in0 compute kernel bakes
        # in1_num_subblocks = out_block_w // out_subblock_w in as a compile-time
        # arg; if out_subblock_w does not divide out_block_w that integer
        # division yields 0, the in1_subblock loop runs zero times, cb_out is
        # never pushed, and the in1 writer parks forever on cb_out.wait_front()
        # -- a silent device hang. (The same expression also underflows uint32
        # in the last-block padded skip count.) The host guard added in
        # sparse_matmul_device_operation.cpp now TT_FATALs on that illegal combo
        # instead of hanging; here we keep the model from generating one.
        #
        # dst-register file limit: out_subblock_h * out_subblock_w <= 8.
        # out_subblock_h is 1 for decode (M=1 tile), so cap out_subblock_w at 8.
        out_subblock_w = min(out_subblock_w, 8)
        # out_subblock_w must divide per_core_N so that out_block_w (a multiple
        # of it) can also divide per_core_N. If it does not, snap it DOWN to the
        # largest divisor of per_core_N that does not exceed the requested value
        # -- mirroring the in0_block_w snap-to-divisor idiom above (1 always
        # divides, so a legal value always exists).
        if per_core_N % out_subblock_w != 0:
            divisors = [d for d in range(1, out_subblock_w) if per_core_N % d == 0]
            out_subblock_w = max(divisors)
        # out_block_w is the smallest legal block that holds one full subblock,
        # i.e. out_block_w == out_subblock_w. This guarantees both
        # out_block_w % out_subblock_w == 0 and per_core_N % out_block_w == 0, so
        # in1_num_subblocks is never 0. With the shipped default out_subblock_w=1
        # this yields out_block_w=1 -- byte-identical to the previous hardcode.
        out_block_w = out_subblock_w

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_block_w,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_gate_up_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode gate/up projections"""
        return self._build_matmul_config(
            self.decode_gate_up_cores,
            m,
            n,
            in0_block_w=self.decode_gate_up_in0_block_w,
            out_subblock_w=self.decode_gate_up_subblock_w,
            k=k,
        )

    def get_decode_down_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode down projection"""
        return self._build_matmul_config(
            self.decode_down_cores,
            m,
            n,
            in0_block_w=self.decode_down_in0_block_w,
            out_subblock_w=self.decode_down_subblock_w,
            k=k,
        )

    def get_prefill_gate_up_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill gate/up projections"""
        return self._build_matmul_config(
            self.prefill_gate_up_cores,
            m,
            n,
            in0_block_w=self.prefill_gate_up_in0_block_w,
            out_subblock_w=self.prefill_gate_up_subblock_w,
            k=k,
        )

    def get_prefill_down_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill down projection"""
        return self._build_matmul_config(
            self.prefill_down_cores,
            m,
            n,
            in0_block_w=self.prefill_down_in0_block_w,
            out_subblock_w=self.prefill_down_subblock_w,
            k=k,
        )
