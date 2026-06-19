# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic expert configuration interfaces.
No model-specific code - models provide their own ProgramConfig implementations.
"""

import math
from dataclasses import dataclass

import ttnn


def _grid_dividing(nt: int, fallback: tuple[int, int], max_dim: int = 8) -> tuple[int, int]:
    """Pick a rectangular core grid (cx, cy), cx,cy <= max_dim, whose core count
    DIVIDES nt, maximizing utilization.

    The sparse_matmul in0-mcast requires num_blocks_x == num_cores (rectangular):
    per_core_N = ceil(nt / cores) and num_blocks_x = ceil(nt / per_core_N), which
    only equals `cores` when cores | nt. the configured fixed grids assume TP-sharded
    dims; under different TP (e.g. TP=1, nt=96) a fixed grid like (5,6)=30 does NOT
    divide nt and the kernel asserts. Auto-selecting a dividing grid makes the
    experts run correctly at any TP (perf tuning is separate).
    """
    best = None
    for cy in range(1, max_dim + 1):
        for cx in range(1, max_dim + 1):
            c = cx * cy
            if nt % c == 0 and (best is None or c > best[0]):
                best = (c, cx, cy)
    return (best[1], best[2]) if best is not None else fallback


@dataclass
class ExpertConfig:
    """Core expert configuration.

    MiniMax-M3 uses the clamped "swigluoai" SwiGLU (gpt-oss variant):
    out = (up_clamped + 1) * (gate_clamped * sigmoid(alpha * gate_clamped)),
    with gate clamped to max=swiglu_limit and up clamped to [-swiglu_limit, swiglu_limit].
    (M2 used plain SiLU SwiGLU: silu(gate) * up — no clamp, no alpha, no (up+1).)
    Defaults match the M3 text_config (swiglu_alpha=1.702, swiglu_limit=7.0).
    """

    intermediate_size: int
    num_experts: int
    hidden_size: int
    num_experts_per_tok: int
    swiglu_limit: float = 7.0
    alpha: float = 1.702


@dataclass
class ProgramConfig:
    """
    Base configuration for expert program configs.

    Models just need to specify grid sizes and chunking parameters.
    The boilerplate MatmulProgramConfig generation is handled automatically.
    """

    # Core grid sizes for prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_down_cores: tuple[int, int] = (5, 6)

    # Sparse matmul subblock widths
    prefill_gate_up_subblock_w: int = 1
    prefill_down_subblock_w: int = 1

    # Input block widths (in0_block_w)
    prefill_gate_up_in0_block_w: int = 1
    prefill_down_in0_block_w: int = 1

    # Chunking parameters
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024

    def __post_init__(self):
        """Validate configuration on creation"""
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
        # Preserve tuned grids when they divide Nt; otherwise snap to a dividing grid
        # so the sparse_matmul stays rectangular at any TP (e.g. TP=1, Nt=96).
        if Nt % num_cores != 0:
            core_x, core_y = _grid_dividing(Nt, fallback=cores, max_dim=8)
            num_cores = core_x * core_y
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
        # Kt range produced by realistic TP shardings (Kt ≤ ~90).
        if k is not None:
            Kt = int(math.ceil(k / 32))
            if Kt % in0_block_w != 0:
                divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
                in0_block_w = max(divisors) if divisors else Kt

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=1,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
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
