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


# Prefill QKV / o_proj 2D multicast matmul geometry per role: (grid columns, N tiles per core, in0_block_w). The ttnn
# default for these shapes picks in0_block_w=1 (one K tile per multicast step); in0_block_w 8-9 and an out_block_h of at
# most 32 tiles measured QKV 0.35 -> 0.24 ms at S = 8192 and 1.54 -> 1.01 ms at 32768, o_proj 0.31 -> 0.24 and
# 1.44 -> 1.00 ms (BH, per-device 1x4 shapes, l11/probe_mm.py). Keyed by the per-device (K, N) of each projection.
PREFILL_PROJ_2D = {
    "qkv": (10, 4, 9),  # [S, 2880] x [2880, 1280]: N = 40 tiles over 10 columns, K = 90 tiles
    "o": (9, 10, 8),  # [S, 1024] x [1024, 2880]: N = 90 tiles over 9 columns, K = 32 tiles
}
PREFILL_PROJ_L1_BYTES = 1_100_000
PREFILL_PROJ_L1_MARGIN = 32 * 1024
# Tensor parallelism the geometries above were measured at (1x4); any other TP keeps the ttnn default.
PREFILL_PROJ_TP = 4


def prefill_proj_program_config(grid, m_tiles, k_tiles, n_tiles, role, tp, in0_dtype, weight_dtype, l1_bytes=None):
    """2D multicast program config for a prefill projection of m_tiles x k_tiles x n_tiles, or None (ttnn default)
    when the role's geometry does not fit (other shapes or a smaller grid), at a TP other than PREFILL_PROJ_TP, or when
    in0 or the weight is not bfloat8_b (the L1 model below sizes their tiles as bf8: a bf16 in0 doubles the in0
    buffers and can overflow L1 at long sequences).

    M splits over at most grid.y rows. out_block_h is the largest divisor of per_core_M (at most 32) whose circular
    buffers (in0 + in1 double-buffered, output + bf16 intermediate block) fit PREFILL_PROJ_L1_BYTES; when that leaves
    fewer than 12 rows, per_core_M is rounded up (up to 8 tiles) to a value with a larger such divisor
    (out_block_h = 1 measured 2-3x slower, and in0_block_w 8 with the whole 26-tile block overflows L1 at S = 8192).
    l1_bytes: the circular-buffer budget instead of PREFILL_PROJ_L1_BYTES when lower (L1 held by resident buffers)."""
    budget = PREFILL_PROJ_L1_BYTES if l1_bytes is None else min(l1_bytes, PREFILL_PROJ_L1_BYTES)
    if tp != PREFILL_PROJ_TP or in0_dtype != ttnn.bfloat8_b or weight_dtype != ttnn.bfloat8_b:
        return None
    gx, pcn, kb = PREFILL_PROJ_2D[role]
    if gx > grid.x or gx * pcn != n_tiles or k_tiles % kb:
        return None
    out_tile = 2048 if role == "qkv" else 1088  # QKV writes bf16, o_proj bf8

    def block_h(pcm):
        for obh in range(min(pcm, 32), 0, -1):
            l1 = obh * kb * 2 * 1088 + kb * pcn * 2 * 1088 + obh * pcn * (out_tile + 2048)
            if pcm % obh == 0 and l1 <= budget:
                return obh
        return 1

    pcm = -(-m_tiles // grid.y)
    obh = block_h(pcm)
    if obh < 12 and pcm > 12:
        pcm, obh = max(((p, block_h(p)) for p in range(pcm, pcm + 9)), key=lambda t: (t[1], -t[0]))
    gy = -(-m_tiles // pcm)
    sw = 4 if pcn % 4 == 0 else 2 if pcn % 2 == 0 else 1
    sh = next(h for h in (4, 2, 1) if obh % h == 0 and h * sw <= 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=obh,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


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
    # Optional per-length chunks for full-attention layers, replacing the small/large rule above when set:
    # ((min_seq_len, q_chunk, k_chunk), ...) with ascending min_seq_len starting at 0; the last entry whose
    # min_seq_len <= seq_len applies.
    prefill_chunk_regimes: tuple[tuple[int, int, int], ...] | None = None
    # Same format, for sliding-window layers; None uses the full-attention chunks.
    prefill_sliding_chunk_regimes: tuple[tuple[int, int, int], ...] | None = None

    # Compute config
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False
    # Prefill SDPA math fidelity on Blackhole; None uses math_fidelity (decode SDPA always uses math_fidelity).
    prefill_sdpa_math_fidelity: str | None = None

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

    # Prefill QKV / o_proj on Blackhole: 2D multicast program configs from prefill_proj_program_config instead of the
    # ttnn default (False keeps the default everywhere).
    prefill_proj_2d: bool = False
    # L1 address the prefill projections' circular buffers must end below (0: the whole L1); set by
    # Model.set_prefill_l1_end when buffers stay resident in L1 (DFlash).
    prefill_l1_end: int = 0

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

        for name in ("prefill_chunk_regimes", "prefill_sliding_chunk_regimes"):
            regimes = getattr(self, name)
            if regimes is None:
                continue
            starts = [r[0] for r in regimes]
            if not starts or starts[0] != 0 or starts != sorted(set(starts)):
                raise ValueError(f"{name} must start at 0 and ascend: {regimes}")
            if any(q <= 0 or k <= 0 for _, q, k in regimes):
                raise ValueError(f"{name} chunk sizes must be positive: {regimes}")

        # Validate math_fidelity
        valid_fidelities = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
        if self.math_fidelity not in valid_fidelities:
            raise ValueError(f"math_fidelity must be one of {valid_fidelities}, got {self.math_fidelity}")
        if self.prefill_sdpa_math_fidelity is not None and self.prefill_sdpa_math_fidelity not in valid_fidelities:
            raise ValueError(
                f"prefill_sdpa_math_fidelity must be None or one of {valid_fidelities}, "
                f"got {self.prefill_sdpa_math_fidelity}"
            )

    def get_decode_sdpa_config(self, mesh_device) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for decode mode"""
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=self.decode_q_chunk_size,
            k_chunk_size=self.decode_k_chunk_size,
            exp_approx_mode=False,
        )

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int, sliding_window=None) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for prefill mode based on sequence length and layer type (sliding_window set or None)"""
        return self.prefill_sdpa_config_candidates(mesh_device, seq_len, sliding_window)[0]

    def prefill_sdpa_config_candidates(self, mesh_device, seq_len: int, sliding_window=None):
        """The prefill SDPA configs to try in order: the tuned chunks first; with L1 held resident by a caller
        (prefill_l1_end, e.g. the batched DFlash runners), then halved chunks (k first, then q, down to one tile), whose
        circular buffers fit below less L1 (the SDPA's L1 use is not modelled here: attention/prefill.py takes the first
        that builds)."""
        # The per-length chunk tables and the full compute grid were tuned on Blackhole only; other architectures
        # keep the small/large chunk rule on an 8x8 grid.
        blackhole = ttnn.device.is_blackhole(mesh_device)
        regimes = self.prefill_chunk_regimes if blackhole else None
        if blackhole and sliding_window is not None and self.prefill_sliding_chunk_regimes is not None:
            regimes = self.prefill_sliding_chunk_regimes
        if regimes is not None:
            _, q_chunk, k_chunk = [r for r in regimes if r[0] <= seq_len][-1]
        elif seq_len >= self.prefill_threshold:
            q_chunk = self.prefill_q_chunk_size_large
            k_chunk = self.prefill_k_chunk_size_large
        else:
            q_chunk = self.prefill_q_chunk_size_small
            k_chunk = self.prefill_k_chunk_size_small

        # On Blackhole, prefill SDPA spreads batch x heads x q-chunks over the whole compute grid (P150: 11x10 = 110
        # cores, not 64), so long sequences get ~1.7x the cores.
        grid = mesh_device.compute_with_storage_grid_size() if blackhole else ttnn.CoreCoord(8, 8)
        chunks = [(q_chunk, k_chunk)]
        while self.prefill_l1_end and (q_chunk > ttnn.TILE_SIZE or k_chunk > ttnn.TILE_SIZE):
            if k_chunk >= q_chunk and k_chunk > ttnn.TILE_SIZE:
                k_chunk //= 2
            else:
                q_chunk //= 2
            chunks.append((q_chunk, k_chunk))
        return [
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, exp_approx_mode=False, q_chunk_size=q, k_chunk_size=k
            )
            for q, k in chunks
        ]

    def get_prefill_sdpa_compute_kernel_config(self, mesh_device) -> ttnn.WormholeComputeKernelConfig:
        """Compute kernel config of the prefill SDPA: prefill_sdpa_math_fidelity on Blackhole, else as decode"""
        fidelity = self.math_fidelity
        if self.prefill_sdpa_math_fidelity is not None and ttnn.device.is_blackhole(mesh_device):
            fidelity = self.prefill_sdpa_math_fidelity
        return self.get_compute_kernel_config(fidelity)

    def get_prefill_proj_config(self, mesh_device, role: str, x, weight, tp, in0_dtype=None):
        """Program config of the prefill QKV (role "qkv") or o_proj ("o") linear of x [1, 1, M, K] by weight [K, N]:
        a 2D multicast config on Blackhole when prefill_proj_2d is set and prefill_proj_program_config accepts the
        shape, TP and dtypes, else None (ttnn default). in0_dtype: the dtype the matmul reads x in, when the caller
        casts x before the matmul (default x.dtype)."""
        if not self.prefill_proj_2d or not ttnn.device.is_blackhole(mesh_device):
            return None
        shape = x.padded_shape
        if any(shape[i] != 1 for i in range(len(shape) - 2)):
            return None
        l1_bytes = None
        if self.prefill_l1_end:
            # (the 2D configs' L1 model counts its circular buffers only: keep the matmul's small extras clear)
            l1_bytes = self.prefill_l1_end - mesh_device.l1_allocator_base() - PREFILL_PROJ_L1_MARGIN
        return prefill_proj_program_config(
            mesh_device.compute_with_storage_grid_size(),
            shape[-2] // ttnn.TILE_SIZE,
            shape[-1] // ttnn.TILE_SIZE,
            weight.padded_shape[-1] // ttnn.TILE_SIZE,
            role,
            tp=tp,
            in0_dtype=x.dtype if in0_dtype is None else in0_dtype,
            weight_dtype=weight.dtype,
            l1_bytes=l1_bytes,
        )

    def get_compute_kernel_config(self, math_fidelity: str | None = None) -> ttnn.WormholeComputeKernelConfig:
        """Get compute kernel config (math_fidelity overrides the configured fidelity)"""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, math_fidelity or self.math_fidelity),
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
