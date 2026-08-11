# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized TTNN decoder layer for ``meta-models/Muse-Glimmer-30B`` (text decoder).

Stage 02 of the repo-local TTNN autoport pipeline. Same model math, same public
contract and the same correctness floor as
:mod:`models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder`, but with
the per-device performance work the functional stage deliberately deferred:

* a **named precision/fidelity policy** per tensor group (attention weights, MLP
  weights, KV cache, activations, norms) instead of one global BF16 policy;
* **L1 width-sharded decode residual** carried through both pre-norms, both
  post-norms, both residual adds, the attention gate multiply and every decode
  projection, with sharded ``LayerNormShardedMultiCoreProgramConfig`` norms;
* **DRAM-sharded decode matmuls** (weights width-sharded over the 8 Blackhole DRAM
  banks, activations/outputs width-sharded in L1 on the matching core grid) with an
  explicit, swept ``in0_block_w`` / ``per_core_N`` geometry;
* **explicit 2D prefill program configs** over a large Blackhole core grid, with the
  sequence folded into ``prefill_matmul_cutoff``-row blocks so the per-core output
  block fits L1;
* **fused elementwise**: SiLU folded into the SwiGLU multiply and sigmoid folded into
  the attention-gate multiply, removing two unary ops per layer per phase.

Everything that decides correctness — chunked prefill, the sliding-window overlap
trim, the paged-cache fill/update contract, the SDPA chunk-size and page-table-capacity
guards, RoPE/NoPE dispatch, the ``qk_scale_factor`` placement, non-aligned sequence
lengths — is preserved exactly as the functional stage validated it. Where a guard
exists it is because a real device bug was measured; see the functional stage's
``doc/functional_decoder/work_log.md``.

Public contract
---------------
Identical to :class:`~...functional_decoder.FunctionalDecoder` with two documented
additions:

* ``decode_forward`` accepts a DRAM/L1 interleaved **or** already-residual-sharded
  hidden state and returns the tensor in the layer's decode residual memory config
  (L1 width-sharded). That is the optimized layer-to-layer contract: stacking layers
  needs no reshard. ``decoder.decode_residual_memory_config`` exposes it, and
  ``ttnn.to_torch`` reads it back unchanged, so every functional test still applies.
* ``precision`` selects the named precision/fidelity policy (see
  :class:`PrecisionPolicy`).

Prefill still takes and returns ``[batch, 1, seq_len, hidden_size]`` DRAM-interleaved
tensors, because prefill activations are large and belong in DRAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
    DEFAULT_BLOCK_SIZE,
    PREFILL_CHUNK_SIZE,
    PREFILL_SDPA_K_CHUNK,
    PREFILL_SDPA_Q_CHUNK,
    SDPA_CHUNK_CANDIDATES,
    SDPA_MAX_SEQ,
    TILE,
    MuseGlimmerDecoderConfig,
    _lcm,
    _round_up,
)
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.model_config import num_to_corerange

# Decode SDPA grid / K chunk: inherited from the functional stage's measured sweep
# (doc/functional_decoder/perf/core_grid_sweep.md) and re-measured here.
DECODE_SDPA_GRID = (8, 4)
DECODE_SDPA_K_CHUNK = 64

# Decode activation/residual core count. 6656 hidden = 208 tiles = 16 * 13, so 16 is the
# largest core count that both divides the hidden tile count *and* forms a rectangle
# inside the 11x10 Blackhole compute grid (26/52/104 would need a 13-wide row). 16 cores
# give 13 K-tiles per core, which is exactly what lets the dominant DRAM-sharded decode
# matmuls run at in0_block_w=13 instead of the 1-2 a wider grid would force.
DECODE_RESIDUAL_GRID = (8, 2)

# Decode SwiGLU working-shard core count. gcd(hidden_tiles 208, intermediate_tiles 624)
# is 208, so the MLP projections can use many more cores than the attention ones; swept
# over {16, 26, 52, 104} in scripts/sweep_optimized_decoder.py.
DECODE_MLP_CORES = 52

# Prefill: fold the sequence into blocks of this many rows before the big 2D matmuls, so
# per_core_M * per_core_N output tiles fit L1. 512 is the Blackhole default in
# models/common/modules/mlp/mlp_1d.py and the measured best here (512: 54.1 ms at 8192
# tokens, 1024: 57.8, 2048: 57.8, 256: 79.5).
#
# The grid cap is (11, 8), not the full (11, 10) compute grid. `grid_x` is forced to the
# DRAM bank count (8) by the DRAM-width-sharded weight contract - see `_prefill_grid_x`.
# `grid_y` is capped at 8 because 10 measured 48% *slower* at 8192 tokens (80.4 ms vs
# 54.1 ms) for the same per_core_M and in0_block_w: the extra two multicast rows cost
# more than the 25% extra cores gain on this Blackhole grid.
PREFILL_MATMUL_CUTOFF = 512
PREFILL_MATMUL_GRID = (11, 8)
# Upper bound on the prefill K block. The per-role value is derived from an L1 budget,
# so this only bounds the search; 26 measured best (52.6-53.1 ms at 8192 tokens against
# 53.9-53.9 at 8), because it lets the three attention projections use a 26-tile K block
# while the L1 model still holds the 19968-wide SwiGLU projections at 8. Above 26 the
# attention rows regress sharply (52 and 104 both measured 62.2 ms).
PREFILL_IN0_BLOCK_W_CAP = 26
PREFILL_OUT_SUBBLOCK_MAX = 8
# Output subblock height for the prefill 2D matmuls. `out_subblock_h * out_subblock_w` must
# fit the destination register file (8 tiles without FP32 accumulation), so h>1 trades width
# for height; swept in scripts/sweep_optimized_decoder.py.
PREFILL_OUT_SUBBLOCK_H = 1


# ---------------------------------------------------------------- precision policy


@dataclass(frozen=True)
class PrecisionPolicy:
    """Per-tensor-group precision and math fidelity.

    One knob per tensor group so a regression can be assigned to the group that caused
    it, as ``$optimize`` requires. ``attn_*`` covers the fused QKV projection, the
    attention gate projection and the output projection; ``mlp_*`` covers the SwiGLU
    gate/up projections; ``mlp_down_*`` covers the down projection, which is the more
    precision-sensitive of the three.
    """

    name: str
    activation_dtype: "ttnn.DataType" = None  # set in __post_init__-ish factories
    attn_weight_dtype: "ttnn.DataType" = None
    attn_fidelity: "ttnn.MathFidelity" = None
    mlp_weight_dtype: "ttnn.DataType" = None
    mlp_fidelity: "ttnn.MathFidelity" = None
    mlp_down_weight_dtype: "ttnn.DataType" = None
    mlp_down_fidelity: "ttnn.MathFidelity" = None
    kv_cache_dtype: "ttnn.DataType" = None
    norm_dtype: "ttnn.DataType" = None
    sdpa_fidelity: "ttnn.MathFidelity" = None
    norm_fidelity: "ttnn.MathFidelity" = None
    # Prefill can legally use a different fidelity from decode: prefill matmuls are
    # FLOP-bound, decode matmuls are DRAM-bound.
    prefill_attn_fidelity: "ttnn.MathFidelity" = None
    prefill_mlp_fidelity: "ttnn.MathFidelity" = None
    prefill_mlp_down_fidelity: "ttnn.MathFidelity" = None

    def evolve(self, **kwargs) -> "PrecisionPolicy":
        return replace(self, **kwargs)


def _policy(name: str, **kwargs) -> PrecisionPolicy:
    base = dict(
        activation_dtype=ttnn.bfloat16,
        attn_weight_dtype=ttnn.bfloat8_b,
        attn_fidelity=ttnn.MathFidelity.LoFi,
        mlp_weight_dtype=ttnn.bfloat4_b,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
        mlp_down_weight_dtype=ttnn.bfloat4_b,
        mlp_down_fidelity=ttnn.MathFidelity.LoFi,
        kv_cache_dtype=ttnn.bfloat8_b,
        norm_dtype=ttnn.bfloat16,
        sdpa_fidelity=ttnn.MathFidelity.HiFi4,
        norm_fidelity=ttnn.MathFidelity.HiFi4,
        prefill_attn_fidelity=ttnn.MathFidelity.LoFi,
        prefill_mlp_fidelity=ttnn.MathFidelity.LoFi,
        prefill_mlp_down_fidelity=ttnn.MathFidelity.LoFi,
    )
    base.update(kwargs)
    return PrecisionPolicy(name=name, **base)


#: Named policies. ``bf16_baseline`` reproduces the functional stage's numerics so the
#: optimized layout work can be measured without a dtype change; the others are the
#: precision candidates the stage swept. ``bfp4_all`` is the selected default; see
#: ``doc/optimized_decoder/pcc/policy_sweep.json`` for every policy on both weight sources.
POLICIES: dict[str, PrecisionPolicy] = {
    "bf16_baseline": _policy(
        "bf16_baseline",
        attn_weight_dtype=ttnn.bfloat16,
        attn_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_weight_dtype=ttnn.bfloat16,
        mlp_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_down_weight_dtype=ttnn.bfloat16,
        mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
        kv_cache_dtype=ttnn.bfloat16,
        prefill_attn_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp8_all_hifi2": _policy(
        "bfp8_all_hifi2",
        attn_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_weight_dtype=ttnn.bfloat8_b,
        mlp_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
        mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_attn_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp8_all_lofi": _policy(
        "bfp8_all_lofi",
        mlp_weight_dtype=ttnn.bfloat8_b,
        mlp_down_weight_dtype=ttnn.bfloat8_b,
    ),
    "bfp8_attn_bfp4_mlp": _policy("bfp8_attn_bfp4_mlp"),
    "bfp8_attn_bfp4_gateup": _policy(
        "bfp8_attn_bfp4_gateup",
        mlp_down_weight_dtype=ttnn.bfloat8_b,
    ),
    "bfp4_all": _policy(
        "bfp4_all",
        attn_weight_dtype=ttnn.bfloat4_b,
    ),
    # Per-tensor-group fidelity probes for the selected dtype policy: the whole-policy
    # sweep above only compares LoFi against HiFi2 at BFP8 weights, and math fidelity is a
    # knob independent of dtype, so these two isolate it per projection group at BFP4.
    "bfp4_all_hifi2_attn": _policy(
        "bfp4_all_hifi2_attn",
        attn_weight_dtype=ttnn.bfloat4_b,
        attn_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_hifi2_mlp": _policy(
        "bfp4_all_hifi2_mlp",
        attn_weight_dtype=ttnn.bfloat4_b,
        mlp_fidelity=ttnn.MathFidelity.HiFi2,
        mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_hifi2_gateup": _policy(
        "bfp4_all_hifi2_gateup",
        attn_weight_dtype=ttnn.bfloat4_b,
        mlp_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_hifi2_down": _policy(
        "bfp4_all_hifi2_down",
        attn_weight_dtype=ttnn.bfloat4_b,
        mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_prefill_hifi2_attn": _policy(
        "bfp4_all_prefill_hifi2_attn",
        attn_weight_dtype=ttnn.bfloat4_b,
        prefill_attn_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_prefill_hifi2_mlp": _policy(
        "bfp4_all_prefill_hifi2_mlp",
        attn_weight_dtype=ttnn.bfloat4_b,
        prefill_mlp_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
    "bfp4_all_prefill_hifi2": _policy(
        "bfp4_all_prefill_hifi2",
        attn_weight_dtype=ttnn.bfloat4_b,
        prefill_attn_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_fidelity=ttnn.MathFidelity.HiFi2,
        prefill_mlp_down_fidelity=ttnn.MathFidelity.HiFi2,
    ),
}

DEFAULT_POLICY = "bfp4_all"


#: Blackhole L1 is 1499 KiB per core; leave headroom for the runtime's own allocations.
L1_MATMUL_CB_BUDGET = 1_450_000

#: Bytes per 32x32 tile, including the block-float exponent section.
_TILE_BYTES = {
    "DataType.BFLOAT16": 2048,
    "DataType.FLOAT32": 4096,
    "DataType.BFLOAT8_B": 1024 + 64,
    "DataType.BFLOAT4_B": 512 + 64,
}


def _tile_bytes(dtype) -> int:
    return _TILE_BYTES.get(str(dtype), 2048)


def _largest_divisor(value: int, cap: int) -> int:
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _out_subblock_w(per_core_n: int, out_subblock_h: int = 1, max_hw: int = 4) -> int:
    """Largest legal output subblock width.

    ``out_subblock_h * out_subblock_w`` must fit the destination register file: 8 tiles
    without FP32 accumulation, 4 with it. The prefill matmuls run with
    ``fp32_dest_acc_en=False``, so 8 is available - and it matters, because a
    ``per_core_N`` like 63 has no divisor in {2, 3, 4} better than 3 but does have 7.
    """
    width = min(max_hw, per_core_n)
    while width > 1:
        if width * out_subblock_h <= max_hw and per_core_n % width == 0:
            break
        width -= 1
    return width


class OptimizedDecoder(LightweightModule):
    """One Muse-Glimmer text decoder layer, optimized for Blackhole."""

    def __init__(
        self,
        *,
        config: MuseGlimmerDecoderConfig,
        mesh_device,
        weights: dict,
        rope_cache: dict | None,
        precision: PrecisionPolicy,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        sdpa_core_grid: tuple[int, int] | None = None,
        prefill_sdpa_q_chunk: int = PREFILL_SDPA_Q_CHUNK,
        prefill_sdpa_k_chunk: int = PREFILL_SDPA_K_CHUNK,
        decode_sdpa_core_grid: tuple[int, int] | None = None,
        decode_sdpa_k_chunk: int | None = None,
        decode_residual_grid: tuple[int, int] = DECODE_RESIDUAL_GRID,
        decode_mlp_cores: int = DECODE_MLP_CORES,
        decode_matmul: str = "dram_sharded",
        decode_in0_block_w: dict | None = None,
        prefill_matmul_cutoff: int = PREFILL_MATMUL_CUTOFF,
        prefill_matmul_grid: tuple[int, int] = PREFILL_MATMUL_GRID,
        prefill_in0_block_w_cap: int = PREFILL_IN0_BLOCK_W_CAP,
        prefill_out_subblock_max: int = PREFILL_OUT_SUBBLOCK_MAX,
        prefill_out_subblock_h: int = PREFILL_OUT_SUBBLOCK_H,
        pack_qkv_gate: bool = False,
        pack_mlp_gate_up: bool = False,
        decode_packer_l1_acc: bool = True,
        decode_fp32_acc: bool = False,
        decode_sdpa_output_l1: bool = True,
        decode_rope_pad_to_tile: bool = True,
        weight_memory: str = "dram_sharded",
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.precision = precision
        self.cache_dtype = precision.kv_cache_dtype
        self.block_size = block_size
        self.prefill_chunk_size = prefill_chunk_size

        if block_size % TILE != 0:
            raise ValueError(f"block_size {block_size} must be a multiple of {TILE}")
        if prefill_chunk_size % block_size != 0:
            raise ValueError(f"prefill_chunk_size {prefill_chunk_size} must be a multiple of block_size {block_size}")
        if prefill_chunk_size % prefill_sdpa_q_chunk != 0 or prefill_chunk_size % prefill_sdpa_k_chunk != 0:
            raise ValueError(
                f"prefill_chunk_size {prefill_chunk_size} must be a multiple of the SDPA chunk sizes "
                f"({prefill_sdpa_q_chunk}, {prefill_sdpa_k_chunk})"
            )
        window = config.sliding_window or 0
        if prefill_chunk_size + window > SDPA_MAX_SEQ:
            raise ValueError(
                f"prefill_chunk_size {prefill_chunk_size} + sliding_window {window} exceeds the "
                f"non-chunked SDPA correctness bound {SDPA_MAX_SEQ}"
            )

        self.pack_qkv_gate = pack_qkv_gate
        self.pack_mlp_gate_up = pack_mlp_gate_up
        self.wqkv = weights.get("wqkv")
        self.w_attn_gate = weights.get("w_attn_gate")
        self.wqkvg = weights.get("wqkvg")
        self.w_mlp_gate_up = weights.get("w_mlp_gate_up")
        self.wo = weights["wo"]
        self.w_mlp_gate = weights.get("w_mlp_gate")
        self.w_mlp_up = weights.get("w_mlp_up")
        self.w_mlp_down = weights["w_mlp_down"]
        self.input_norm_w = weights["input_norm_w"]
        self.post_attn_norm_w = weights["post_attn_norm_w"]
        self.pre_ff_norm_w = weights["pre_ff_norm_w"]
        self.post_ff_norm_w = weights["post_ff_norm_w"]
        self.rope_cache = rope_cache

        grid = mesh_device.compute_with_storage_grid_size()
        self.device_grid = (grid.x, grid.y)
        self.sdpa_core_grid = sdpa_core_grid or self.device_grid
        self.prefill_sdpa_q_chunk = prefill_sdpa_q_chunk
        self.prefill_sdpa_k_chunk = prefill_sdpa_k_chunk
        self.decode_sdpa_core_grid = decode_sdpa_core_grid or (
            min(DECODE_SDPA_GRID[0], grid.x),
            min(DECODE_SDPA_GRID[1], grid.y),
        )
        self.decode_sdpa_k_chunk = decode_sdpa_k_chunk or DECODE_SDPA_K_CHUNK

        if decode_matmul not in ("dram_sharded", "mcast1d", "interleaved"):
            raise ValueError(f"unknown decode_matmul {decode_matmul!r}")
        if decode_matmul == "mcast1d" and (pack_qkv_gate or pack_mlp_gate_up):
            raise ValueError(
                "projection packing is only implemented for the DRAM-sharded decode matmul family; "
                "the 1D multicast family needs num_cores to divide the tiled N, which the packed "
                "widths do not satisfy on a legal rectangle"
            )
        self.decode_matmul = decode_matmul
        self.decode_residual_grid = decode_residual_grid
        self.decode_cores = decode_residual_grid[0] * decode_residual_grid[1]
        self.decode_mlp_cores = decode_mlp_cores
        self.prefill_matmul_cutoff = prefill_matmul_cutoff
        self.prefill_matmul_grid = prefill_matmul_grid
        self.prefill_in0_block_w_cap = prefill_in0_block_w_cap
        self.prefill_out_subblock_max = prefill_out_subblock_max
        self.prefill_out_subblock_h = prefill_out_subblock_h
        self.weight_memory = weight_memory
        self.decode_sdpa_output_l1 = decode_sdpa_output_l1
        self.decode_rope_pad_to_tile = decode_rope_pad_to_tile
        # Per-role in0_block_w overrides for the geometry sweep. Roles:
        # "qkv", "attn_gate", "wo", "mlp_gate_up", "mlp_down".
        self.decode_in0_block_w = dict(decode_in0_block_w or {})

        arch = mesh_device.arch()
        self._arch = arch
        self.norm_kernel_config = self._kernel_config(precision.norm_fidelity, fp32_dest_acc_en=True)
        self.sdpa_kernel_config = self._kernel_config(precision.sdpa_fidelity, fp32_dest_acc_en=True)
        # Decode matmul fidelity per tensor group. fp32_dest_acc_en is off for the
        # reduced-precision decode matmuls: they are DRAM-bound and fp32 accumulation
        # halves the DST throughput without changing the DRAM traffic.
        self.decode_packer_l1_acc = decode_packer_l1_acc
        self.decode_fp32_acc = decode_fp32_acc
        decode_flags = dict(fp32_dest_acc_en=decode_fp32_acc, packer_l1_acc=decode_packer_l1_acc)
        self.attn_kernel_config = self._kernel_config(precision.attn_fidelity, **decode_flags)
        self.mlp_kernel_config = self._kernel_config(precision.mlp_fidelity, **decode_flags)
        self.mlp_down_kernel_config = self._kernel_config(precision.mlp_down_fidelity, **decode_flags)
        self.prefill_attn_kernel_config = self._kernel_config(precision.prefill_attn_fidelity)
        self.prefill_mlp_kernel_config = self._kernel_config(precision.prefill_mlp_fidelity)
        self.prefill_mlp_down_kernel_config = self._kernel_config(precision.prefill_mlp_down_fidelity)

        self._build_decode_configs()

    def _kernel_config(self, fidelity, *, fp32_dest_acc_en: bool = False, packer_l1_acc: bool = True):
        return ttnn.init_device_compute_kernel_config(
            self._arch,
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=packer_l1_acc,
        )

    # ------------------------------------------------------- decode layout config

    @staticmethod
    def _rectangle(grid: tuple[int, int]):
        """Explicit ``grid[0] x grid[1]`` core rectangle anchored at (0, 0).

        ``ttnn.CoreGrid`` alone is not enough: ``create_sharded_memory_config`` lays
        ``grid.num_cores`` shards out row-major across the *device* grid width, so an
        ``8x2`` CoreGrid on this 11-wide Blackhole grid yields a 16-core set whose
        bounding box is 11x2 — which the sharded layernorm rejects ("Sharded layernorm
        does not support non-rectangular core grids").
        """
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})

    def _width_sharded_memcfg(self, width: int, cores: int, grid: tuple[int, int], *, rows: int = TILE):
        if width % (cores * TILE) != 0:
            raise ValueError(f"width {width} does not split into {cores} tile-aligned width shards")
        return ttnn.create_sharded_memory_config(
            shape=(rows, width // cores),
            core_grid=self._rectangle(grid),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _matmul_output_memcfg(self, width: int, cores: int, grid: tuple[int, int], *, rows: int = TILE):
        """The width-sharded L1 layout a DRAM-sharded matmul actually produces.

        ``MatmulMultiCoreReuseMultiCastDRAMSharded`` ignores the requested output memory
        config and builds its own from ``num_cores_to_corerangeset(num_cores, grid,
        row_wise=True)``. On an 11x10 grid that is a rectangle only for ``num_cores <= 11``
        or multiples of 11; for 16 cores it is the 11+5 set whose bounding box spans 22
        cores, which the sharded layernorm rejects. Modelling it here keeps the elementwise
        consumers on exactly the producer's layout and confines resharding to the two
        places where the residual contract needs it.
        """
        if width % (cores * TILE) != 0:
            raise ValueError(f"width {width} does not split into {cores} tile-aligned width shards")
        device_grid = ttnn.CoreCoord(self.device_grid[0], self.device_grid[1])
        core_set = ttnn.num_cores_to_corerangeset(cores, device_grid, row_wise=True)
        return ttnn.create_sharded_memory_config(
            shape=(rows, width // cores),
            core_grid=core_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _norm_program_config(self, width: int, cores: int, grid: tuple[int, int]):
        block_w = width // cores // TILE
        subblock_w = _largest_divisor(block_w, 4)
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid[0], grid[1]],
            subblock_w=subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )

    def _dram_matmul_pc(self, role: str, *, m: int, k: int, n: int, cores: int, weight_dtype, fused_activation=None):
        """DRAM-sharded decode matmul program config with an L1-accurate ``in0_block_w``.

        Two facts about ``matmul_multicore_reuse_mcast_dram_sharded`` drive this:

        * the op picks its **own** worker set, from
          ``get_optimal_dram_bank_to_reader_assignment`` - 12 cores on this p300c, which is
          what the profiler's ``Cores`` column reports, and *not* the same as the 8 DRAM
          banks the weights are width-sharded over. The compute width per worker is
          ``ceil(N_tiles / workers)``, not ``per_core_N``: ``per_core_N`` in the program
          config only sets the output storage shard. So a larger ``cores`` does not add
          compute; it only narrows the in0 shard and therefore shrinks the legal
          ``in0_block_w``.
        * the weight circular buffer is *triple* buffered:
          ``3 * per_core_N_compute * in0_block_w * dram_aligned_tile_bytes``. That is what
          makes the K block dtype-dependent: at BF16 the 19968-wide SwiGLU projection
          only affords ``in0_block_w=2``, at BFP4 it affords 8.
        """
        per_core_m = math.ceil(m / TILE)
        per_core_n_storage = math.ceil(n / (TILE * cores))
        k_tiles_per_core = k // (TILE * cores)
        # The DRAM bank count is a deliberately conservative proxy for the op's reader
        # worker count (8 against the 12 it actually picks): it over-estimates
        # per_core_N_compute and therefore the weight circular buffer, so the in0_block_w
        # this model accepts is always legal, never optimistic. The per-role sweep in
        # doc/optimized_decoder/perf/candidates.json confirms the values it picks win.
        banks = self.mesh_device.dram_grid_size().x
        per_core_n_compute = math.ceil(n / TILE / banks)

        act_tile = _tile_bytes(self.precision.activation_dtype)
        w_tile = _tile_bytes(weight_dtype)
        # out CB + interm0 CB (BF16 because packer_l1_acc is on and fp32_dest_acc is off)
        # + in2 (the local in0 shard) + the output-reshard CB.
        fixed = (
            per_core_m * per_core_n_compute * 2 * act_tile
            + per_core_m * k_tiles_per_core * act_tile
            + per_core_m * per_core_n_storage * act_tile
        )
        budget = L1_MATMUL_CB_BUDGET - fixed
        default_block_w = 1
        for candidate in range(k_tiles_per_core, 0, -1):
            if k_tiles_per_core % candidate:
                continue
            cb = 3 * per_core_n_compute * candidate * w_tile + 2 * per_core_m * candidate * act_tile
            if cb <= budget:
                default_block_w = candidate
                break
        in0_block_w = self.decode_in0_block_w.get(role, default_block_w)
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n_storage,
            fused_activation=fused_activation,
        )

    def _mcast1d_pc(self, role: str, *, m: int, k: int, n: int, cores: int, weight_dtype, fused_activation=None):
        """1D multicast decode matmul program config (the non-DRAM-sharded candidate family).

        Unlike the DRAM-sharded op, this one computes on the grid it is given, so it is the
        way to put more than the DRAM-sharded op's 12 reader cores on a dominant decode
        matmul. It needs ``num_cores`` to divide both the tiled K (for the width-sharded
        in0) and the tiled N (for the width-sharded output), and the grid must be a
        rectangle - which on this model pins every role to 16 cores, because
        ``gcd(208, 144) = gcd(208, 128) = gcd(128, 208) = 16`` and 26/52/104 have no
        rectangle inside an 11x10 grid.
        """
        per_core_m = math.ceil(m / TILE)
        per_core_n = n // TILE // cores
        k_tiles_per_core = k // (TILE * cores)
        act_tile = _tile_bytes(self.precision.activation_dtype)
        w_tile = _tile_bytes(weight_dtype)
        fixed = per_core_m * per_core_n * 2 * act_tile
        budget = L1_MATMUL_CB_BUDGET - fixed
        block_w = 1
        for candidate in range(k_tiles_per_core, 0, -1):
            if k_tiles_per_core % candidate:
                continue
            if 2 * candidate * (per_core_m * act_tile + per_core_n * w_tile) <= budget:
                block_w = candidate
                break
        block_w = self.decode_in0_block_w.get(role, block_w)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.decode_residual_grid,
            in0_block_w=block_w,
            out_subblock_h=1,
            out_subblock_w=_out_subblock_w(per_core_n, 1, 8),
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )

    def _build_decode_configs(self):
        cfg = self.config
        cores = self.decode_cores
        grid = self.decode_residual_grid
        hidden = cfg.hidden_size

        self.decode_residual_memcfg = self._width_sharded_memcfg(hidden, cores, grid)
        self.decode_norm_pc = self._norm_program_config(hidden, cores, grid)

        # The attention-gate working shard has to match whatever
        # ``MatmulMultiCoreReuseMultiCastDRAMSharded`` produces for that projection,
        # because the gate multiply is elementwise against it. The op does *not* honour a
        # requested output memory config: it derives ``num_cores = N_tiles / per_core_N``
        # and lays them out row-major over the compute grid, which on an 11-wide Blackhole
        # grid is only a rectangle when ``num_cores <= 11`` or is a multiple of 11. See
        # ``_matmul_output_memcfg``.
        self.decode_attn_gate_memcfg = self._matmul_output_memcfg(cfg.num_attention_heads * cfg.head_dim, cores, grid)

        attn_gate_width = cfg.num_attention_heads * cfg.head_dim
        self.decode_qkvg_pc = self._dram_matmul_pc(
            "qkvg",
            m=TILE,
            k=hidden,
            n=cfg.qkv_width + attn_gate_width,
            cores=cores,
            weight_dtype=self.precision.attn_weight_dtype,
        )
        self.decode_qkv_pc = self._dram_matmul_pc(
            "qkv",
            m=TILE,
            k=hidden,
            n=cfg.qkv_width,
            cores=cores,
            weight_dtype=self.precision.attn_weight_dtype,
        )
        self.decode_attn_gate_pc = self._dram_matmul_pc(
            "attn_gate",
            m=TILE,
            k=hidden,
            n=cfg.num_attention_heads * cfg.head_dim,
            cores=cores,
            weight_dtype=self.precision.attn_weight_dtype,
        )
        self.decode_wo_pc = self._dram_matmul_pc(
            "wo",
            m=TILE,
            k=cfg.num_attention_heads * cfg.head_dim,
            n=hidden,
            cores=cores,
            weight_dtype=self.precision.attn_weight_dtype,
        )

        if self.decode_matmul == "mcast1d":
            # One coherent 16-core rectangle for every role, so there is no SwiGLU
            # working-shard reshard at all and every matmul output is already on the
            # residual grid the sharded norms need.
            self.decode_qkv_pc = self._mcast1d_pc(
                "qkv",
                m=TILE,
                k=hidden,
                n=cfg.qkv_width,
                cores=cores,
                weight_dtype=self.precision.attn_weight_dtype,
            )
            self.decode_attn_gate_pc = self._mcast1d_pc(
                "attn_gate",
                m=TILE,
                k=hidden,
                n=attn_gate_width,
                cores=cores,
                weight_dtype=self.precision.attn_weight_dtype,
            )
            self.decode_wo_pc = self._mcast1d_pc(
                "wo",
                m=TILE,
                k=attn_gate_width,
                n=hidden,
                cores=cores,
                weight_dtype=self.precision.attn_weight_dtype,
            )
            self.decode_mlp_gate_up_pc = self._mcast1d_pc(
                "mlp_gate_up",
                m=TILE,
                k=hidden,
                n=cfg.intermediate_size,
                cores=cores,
                weight_dtype=self.precision.mlp_weight_dtype,
            )
            self.decode_mlp_down_pc = self._mcast1d_pc(
                "mlp_down",
                m=TILE,
                k=cfg.intermediate_size,
                n=hidden,
                cores=cores,
                weight_dtype=self.precision.mlp_down_weight_dtype,
            )
            self.decode_mlp_packed_pc = None
            self.decode_qkvg_pc = None
            self.decode_mlp_in_memcfg = self.decode_residual_memcfg
            self.decode_attn_gate_memcfg = self._width_sharded_memcfg(attn_gate_width, cores, grid)
            self.decode_mlp_out_memcfg = self._width_sharded_memcfg(cfg.intermediate_size, cores, grid)
            self.decode_qkv_out_memcfg = self._width_sharded_memcfg(cfg.qkv_width, cores, grid)
            return

        # Phase-specific MLP working shard (OPT-011). The attention projections are
        # pinned to 16 cores by gcd(hidden_tiles=208, qkv_tiles=144) = 16, but the SwiGLU
        # projections share gcd(208, 624) = 208, so they can use far more cores. At 16
        # cores the gate/up matmul needs per_core_N = 39 output tiles and its in1 circular
        # buffers grow to 2080640 B against a 1572864 B L1 - the op does not fit at all.
        # Resharding the post-norm activation onto `decode_mlp_cores` before the SwiGLU
        # block and back to the residual grid after the down projection costs two
        # ~425 KB L1->L1 reshards and unlocks the whole family.
        mlp_cores = self.decode_mlp_cores
        self.decode_mlp_in_memcfg = self._matmul_output_memcfg(hidden, mlp_cores, grid)
        self.decode_mlp_gate_up_pc = self._dram_matmul_pc(
            "mlp_gate_up",
            m=TILE,
            k=hidden,
            n=cfg.intermediate_size,
            cores=mlp_cores,
            weight_dtype=self.precision.mlp_weight_dtype,
        )
        self.decode_mlp_packed_pc = self._dram_matmul_pc(
            "mlp_gate_up_packed",
            m=TILE,
            k=hidden,
            n=2 * cfg.intermediate_size,
            cores=mlp_cores,
            weight_dtype=self.precision.mlp_weight_dtype,
        )
        self.decode_mlp_down_pc = self._dram_matmul_pc(
            "mlp_down",
            m=TILE,
            k=cfg.intermediate_size,
            n=hidden,
            cores=mlp_cores,
            weight_dtype=self.precision.mlp_down_weight_dtype,
        )

    @property
    def decode_residual_memory_config(self):
        """The layer-to-layer decode activation contract (L1 width-sharded)."""
        return self.decode_residual_memcfg

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        state_dict_prefix: str | None = None,
        precision: PrecisionPolicy | str = DEFAULT_POLICY,
        rope_dtype=ttnn.bfloat16,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        rope_max_seq_len: int | None = None,
        sdpa_core_grid: tuple[int, int] | None = None,
        prefill_sdpa_q_chunk: int = PREFILL_SDPA_Q_CHUNK,
        prefill_sdpa_k_chunk: int = PREFILL_SDPA_K_CHUNK,
        decode_sdpa_core_grid: tuple[int, int] | None = None,
        decode_sdpa_k_chunk: int | None = None,
        decode_residual_grid: tuple[int, int] = DECODE_RESIDUAL_GRID,
        decode_mlp_cores: int = DECODE_MLP_CORES,
        decode_matmul: str = "dram_sharded",
        decode_in0_block_w: dict | None = None,
        prefill_matmul_cutoff: int = PREFILL_MATMUL_CUTOFF,
        prefill_matmul_grid: tuple[int, int] = PREFILL_MATMUL_GRID,
        prefill_in0_block_w_cap: int = PREFILL_IN0_BLOCK_W_CAP,
        prefill_out_subblock_max: int = PREFILL_OUT_SUBBLOCK_MAX,
        prefill_out_subblock_h: int = PREFILL_OUT_SUBBLOCK_H,
        weight_memory: str | None = None,
        pack_qkv_gate: bool = False,
        pack_mlp_gate_up: bool = False,
        decode_packer_l1_acc: bool = True,
        decode_fp32_acc: bool = False,
        decode_sdpa_output_l1: bool = True,
        decode_rope_pad_to_tile: bool = True,
        weight_dtype=None,
        cache_dtype=None,
    ) -> "OptimizedDecoder":
        """Build the optimized layer from an HF state dict.

        ``precision`` is a :class:`PrecisionPolicy` or the name of one in
        :data:`POLICIES`. ``weight_dtype`` / ``cache_dtype`` are accepted for
        signature compatibility with the functional layer and, when given, override
        the policy's weight / KV-cache dtypes (used by the dtype sweep).
        """
        import torch  # setup-time only: weight conversion never happens at runtime

        if isinstance(precision, str):
            if precision not in POLICIES:
                raise ValueError(f"unknown precision policy {precision!r}; have {sorted(POLICIES)}")
            precision = POLICIES[precision]
        if weight_dtype is not None:
            precision = precision.evolve(
                attn_weight_dtype=weight_dtype,
                mlp_weight_dtype=weight_dtype,
                mlp_down_weight_dtype=weight_dtype,
            )
        if cache_dtype is not None:
            precision = precision.evolve(kv_cache_dtype=cache_dtype)

        config = MuseGlimmerDecoderConfig.from_hf_config(hf_config, layer_idx)

        if state_dict_prefix is None:
            default_prefix = f"model.language_model.layers.{layer_idx}."
            state_dict_prefix = default_prefix if any(k.startswith(default_prefix) for k in state_dict) else ""

        def get(name: str) -> "torch.Tensor":
            key = f"{state_dict_prefix}{name}"
            if key not in state_dict:
                raise KeyError(f"missing weight {key!r} in state dict")
            return state_dict[key].to(torch.float32)

        hidden = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads
        head_dim = config.head_dim

        q_proj = get("self_attn.q_proj.weight")
        k_proj = get("self_attn.k_proj.weight")
        v_proj = get("self_attn.v_proj.weight")
        o_proj = get("self_attn.o_proj.weight")
        attn_gate = get("self_attn.gate_proj.weight")
        mlp_gate = get("mlp.gate_proj.weight")
        mlp_up = get("mlp.up_proj.weight")
        mlp_down = get("mlp.down_proj.weight")

        expected = {
            "self_attn.q_proj.weight": (n_heads * head_dim, hidden),
            "self_attn.k_proj.weight": (n_kv * head_dim, hidden),
            "self_attn.v_proj.weight": (n_kv * head_dim, hidden),
            "self_attn.o_proj.weight": (hidden, n_heads * head_dim),
            "self_attn.gate_proj.weight": (n_heads * head_dim, hidden),
            "mlp.gate_proj.weight": (config.intermediate_size, hidden),
            "mlp.up_proj.weight": (config.intermediate_size, hidden),
            "mlp.down_proj.weight": (hidden, config.intermediate_size),
        }
        actual = {
            "self_attn.q_proj.weight": q_proj,
            "self_attn.k_proj.weight": k_proj,
            "self_attn.v_proj.weight": v_proj,
            "self_attn.o_proj.weight": o_proj,
            "self_attn.gate_proj.weight": attn_gate,
            "mlp.gate_proj.weight": mlp_gate,
            "mlp.up_proj.weight": mlp_up,
            "mlp.down_proj.weight": mlp_down,
        }
        for name, shape in expected.items():
            if tuple(actual[name].shape) != shape:
                raise ValueError(f"{name} has shape {tuple(actual[name].shape)}, expected {shape}")

        # A throwaway instance would be needed to call _dram_sharded_weight_memcfg before
        # __init__, so build the DRAM-sharded weight memory configs here with the same
        # arithmetic (dram_cores = mesh_device.dram_grid_size().x).
        dram = mesh_device.dram_grid_size()
        dram_cores = dram.x
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})

        def dram_sharded_memcfg(k: int, n: int):
            padded_n = _round_up(n, TILE * dram_cores)
            shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        if weight_memory is None:
            weight_memory = "dram_sharded" if decode_matmul == "dram_sharded" else "interleaved"
        if weight_memory not in ("dram_sharded", "interleaved"):
            raise ValueError(f"unknown weight_memory {weight_memory!r}")

        def as_weight(torch_tensor, dtype, *, dram_sharded=None):
            dram_sharded = (weight_memory == "dram_sharded") if dram_sharded is None else dram_sharded
            k, n = int(torch_tensor.shape[-2]), int(torch_tensor.shape[-1])
            memory_config = dram_sharded_memcfg(k, n) if dram_sharded else ttnn.DRAM_MEMORY_CONFIG
            # Tilize on host, then place. Handing ``from_torch`` a row-major tensor plus a
            # DRAM width-sharded memory config makes it tilize on device, and a
            # 6656x2496 BF16 shard needs 2667264 B of circular buffer on one core against
            # a 1572864 B L1. BFP8/BFP4 happen to convert on host and hide the problem.
            host = ttnn.from_torch(
                torch_tensor,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            return ttnn.to_device(host, mesh_device, memory_config=memory_config)

        def as_device_tensor(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                torch_tensor,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        # Fused QKV, ordered [q_heads*head_dim, k_heads*head_dim, v_heads*head_dim] as
        # ttnn.experimental.nlp_create_qkv_heads* expects.
        wqkv = torch.cat([q_proj, k_proj, v_proj], dim=0).transpose(0, 1).contiguous()

        def norm_weight(name: str):
            # HF MuseGlimmerTextCenteredRMSNorm: normed * (1 + weight); ttnn.rms_norm
            # computes normed * weight, so the unit offset is folded in at setup time.
            w = get(name)
            if tuple(w.shape) != (hidden,):
                raise ValueError(f"{name} has shape {tuple(w.shape)}, expected {(hidden,)}")
            return as_device_tensor((1.0 + w).reshape(1, 1, 1, hidden), dtype=precision.norm_dtype)

        attn_gate_t = attn_gate.transpose(0, 1).contiguous()
        mlp_gate_t = mlp_gate.transpose(0, 1).contiguous()
        mlp_up_t = mlp_up.transpose(0, 1).contiguous()
        weights = {
            "wo": as_weight(o_proj.transpose(0, 1).contiguous(), precision.attn_weight_dtype),
            "w_mlp_down": as_weight(mlp_down.transpose(0, 1).contiguous(), precision.mlp_down_weight_dtype),
            "input_norm_w": norm_weight("input_layernorm.weight"),
            "post_attn_norm_w": norm_weight("post_attention_layernorm.weight"),
            "pre_ff_norm_w": norm_weight("pre_feedforward_layernorm.weight"),
            "post_ff_norm_w": norm_weight("post_feedforward_layernorm.weight"),
        }
        # Same-input projection packing candidates (OPT-001 / OPT-010). Q/K/V are always
        # packed; `pack_qkv_gate` additionally folds the attention gate projection - the
        # fourth projection of the same post-norm activation - into the same matmul, and
        # `pack_mlp_gate_up` folds the SwiGLU gate/up pair.
        if pack_qkv_gate:
            weights["wqkvg"] = as_weight(torch.cat([wqkv, attn_gate_t], dim=-1), precision.attn_weight_dtype)
        else:
            weights["wqkv"] = as_weight(wqkv, precision.attn_weight_dtype)
            weights["w_attn_gate"] = as_weight(attn_gate_t, precision.attn_weight_dtype)
        if pack_mlp_gate_up:
            weights["w_mlp_gate_up"] = as_weight(torch.cat([mlp_gate_t, mlp_up_t], dim=-1), precision.mlp_weight_dtype)
        else:
            weights["w_mlp_gate"] = as_weight(mlp_gate_t, precision.mlp_weight_dtype)
            weights["w_mlp_up"] = as_weight(mlp_up_t, precision.mlp_weight_dtype)

        rope_cache = None
        if config.uses_rope:
            max_pos = rope_max_seq_len or config.max_position_embeddings
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            positions = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            rope_cache = {
                "cos_prefill": as_device_tensor(cos.reshape(1, 1, max_pos, head_dim), dtype=rope_dtype),
                "sin_prefill": as_device_tensor(sin.reshape(1, 1, max_pos, head_dim), dtype=rope_dtype),
                "cos_decode": as_device_tensor(cos, dtype=rope_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "sin_decode": as_device_tensor(sin, dtype=rope_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "max_seq_len": max_pos,
            }

        return cls(
            config=config,
            mesh_device=mesh_device,
            weights=weights,
            rope_cache=rope_cache,
            precision=precision,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            sdpa_core_grid=sdpa_core_grid,
            prefill_sdpa_q_chunk=prefill_sdpa_q_chunk,
            prefill_sdpa_k_chunk=prefill_sdpa_k_chunk,
            decode_sdpa_core_grid=decode_sdpa_core_grid,
            decode_sdpa_k_chunk=decode_sdpa_k_chunk,
            decode_residual_grid=decode_residual_grid,
            decode_mlp_cores=decode_mlp_cores,
            decode_matmul=decode_matmul,
            decode_in0_block_w=decode_in0_block_w,
            prefill_matmul_cutoff=prefill_matmul_cutoff,
            prefill_matmul_grid=prefill_matmul_grid,
            prefill_in0_block_w_cap=prefill_in0_block_w_cap,
            prefill_out_subblock_max=prefill_out_subblock_max,
            prefill_out_subblock_h=prefill_out_subblock_h,
            pack_qkv_gate=pack_qkv_gate,
            pack_mlp_gate_up=pack_mlp_gate_up,
            decode_packer_l1_acc=decode_packer_l1_acc,
            decode_fp32_acc=decode_fp32_acc,
            decode_sdpa_output_l1=decode_sdpa_output_l1,
            decode_rope_pad_to_tile=decode_rope_pad_to_tile,
            weight_memory=weight_memory,
        )

    # ----------------------------------------------------------- kv cache api

    def allocate_kv_cache(self, *, batch_size: int, max_seq_len: int, num_blocks: int | None = None):
        """Allocate an empty paged K/V cache pair, ``[num_blocks, kv_heads, block_size, head_dim]``."""
        import torch  # setup-time only

        blocks_per_seq = self.blocks_per_seq(max_seq_len)
        total_blocks = num_blocks if num_blocks is not None else blocks_per_seq * batch_size
        if total_blocks < blocks_per_seq * batch_size:
            raise ValueError(
                f"num_blocks {total_blocks} cannot hold {batch_size} sequences of {max_seq_len} tokens "
                f"({blocks_per_seq} blocks each)"
            )
        shape = (total_blocks, self.config.num_key_value_heads, self.block_size, self.config.head_dim)
        caches = []
        for _ in range(2):
            caches.append(
                ttnn.as_tensor(
                    torch.zeros(shape, dtype=torch.float32),
                    dtype=self.cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        return caches[0], caches[1]

    # ------------------------------------------------------------- primitives

    @staticmethod
    def _slice_view(tensor, begins, ends):
        """``ttnn.slice`` that reports whether a new buffer was produced (see functional stage)."""
        shape = list(tensor.shape)
        if all(b == 0 for b in begins) and all(e == s for e, s in zip(ends, shape)):
            return tensor, False
        return ttnn.slice(tensor, begins, ends), True

    def _norm(self, x, weight, eps):
        """Interleaved RMSNorm (prefill)."""
        return ttnn.rms_norm(
            x,
            weight=weight,
            epsilon=eps,
            compute_kernel_config=self.norm_kernel_config,
        )

    def _sharded_norm(self, x, weight, eps):
        """L1 width-sharded RMSNorm over the decode residual grid (decode)."""
        return ttnn.rms_norm(
            x,
            weight=weight,
            epsilon=eps,
            program_config=self.decode_norm_pc,
            memory_config=self.decode_residual_memcfg,
            compute_kernel_config=self.norm_kernel_config,
        )

    def _qk_norm(self, x):
        """RMSNorm over ``head_dim`` with no scale (HF ``MuseGlimmerRMSNorm(with_scale=False)``)."""
        return ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.norm_kernel_config,
        )

    def _sdpa_program_config(self, *, q_chunk: int, k_chunk: int, grid: tuple[int, int] | None = None):
        grid = grid or self.sdpa_core_grid
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    # ---------------------------------------------------------------- prefill

    def _prefill_grid_x(self, n_tiles: int) -> int:
        """Compute-grid width for a prefill 2D matmul.

        With DRAM width-sharded weights this MUST equal the DRAM bank count, so that
        ``per_core_N`` is exactly the in1 DRAM shard width in tiles. Anything else makes
        ``MatmulMultiCoreReuseMultiCastProgramConfig`` return **NaN**, silently: a 9-wide
        grid on the 144-tile QKV projection covers N exactly (``per_core_N = 16``) and
        still produced NaN, dropping whole-layer prefill PCC to 0.765 and filling the
        paged K cache with NaN at every sequence length - while looking ~5% *faster* than
        the legal 8-wide grid in a wall-clock sweep. The same program config with a
        DRAM-interleaved in1 is correct at every grid width. Reproduced standalone by
        ``scripts/repro_prefill_matmul_grid_x9.py``; worth an upstream validation check.
        """
        if self.weight_memory == "dram_sharded":
            grid_x = min(self.mesh_device.dram_grid_size().x, self.device_grid[0])
            if n_tiles % grid_x:
                raise ValueError(
                    f"tiled N {n_tiles} is not a multiple of the DRAM bank count {grid_x}; the "
                    "2D matmul cannot consume DRAM width-sharded weights for this shape"
                )
            return grid_x
        max_x = min(self.prefill_matmul_grid[0], self.device_grid[0])
        return _largest_divisor(n_tiles, max_x)

    def _prefill_cb_bytes(self, *, per_core_m: int, per_core_n: int, in0_block_w: int, weight_dtype) -> int:
        act_tile = _tile_bytes(self.precision.activation_dtype)
        w_tile = _tile_bytes(weight_dtype)
        # out CB + interm0 CB (BF16: packer_l1_acc on, fp32_dest_acc off) + double-buffered
        # in0/in1 blocks.
        return per_core_m * per_core_n * 2 * act_tile + 2 * in0_block_w * (per_core_m * act_tile + per_core_n * w_tile)

    def _prefill_in0_block_w(self, *, k_tiles: int, per_core_m: int, per_core_n: int, weight_dtype) -> int:
        """Largest legal ``in0_block_w`` whose circular buffers still fit Blackhole L1.

        ``in0_block_w`` must divide the tiled K dimension. For the 19968-wide SwiGLU
        projections at BF16 the budget caps it at 2; at BFP4 the same budget allows 8 -
        i.e. the reduced weight dtype buys a larger K block as well as less DRAM traffic,
        so the value is derived from the budget rather than pinned to a constant.
        """
        cap = min(self.prefill_in0_block_w_cap, k_tiles)
        for candidate in range(cap, 0, -1):
            if k_tiles % candidate:
                continue
            if (
                self._prefill_cb_bytes(
                    per_core_m=per_core_m,
                    per_core_n=per_core_n,
                    in0_block_w=candidate,
                    weight_dtype=weight_dtype,
                )
                <= L1_MATMUL_CB_BUDGET
            ):
                return candidate
        return 0

    def _prefill_fold(self, *, rows: int, k_tiles: int, n_tiles: int, weight_dtype):
        """Pick the row-fold and 2D grid for one prefill matmul.

        The sequence is folded into ``block_rows``-row blocks along a leading batch
        dimension so that ``per_core_M x per_core_N`` output tiles fit L1: a single
        ``M = 8192`` (or ``M = 2080``) matmul against the 19968-wide SwiGLU projection
        needs megabytes of output circular buffer per core and simply does not fit. The
        block size must tile the row count, so when the logical length has no useful
        divisor - 3008 rows is 2 x 47 tiles, whose only fold options are 1 and 2 tiles,
        i.e. 8 or 16 of the 110 cores - the rows are zero-padded up to a multiple of a
        larger block and the padding is sliced off the output. Padding is mathematically
        inert here: zero activation rows produce zero output rows, and they are removed
        before anything else in the layer sees them.

        Candidates are first filtered to those that still support the largest legal
        ``in0_block_w`` any fold achieves for this shape - a fold with a big
        ``per_core_M`` can only afford ``in0_block_w=1``, which costs far more than the
        extra cores gain - and then ranked by ``grid_x * grid_y * rows / padded_rows``
        (cores actually used, discounted by wasted padded rows), preferring the larger
        ``per_core_M`` on ties because it re-reads the weights fewer times.
        """
        grid_x = self._prefill_grid_x(n_tiles)
        per_core_n = n_tiles // grid_x
        max_y = min(self.prefill_matmul_grid[1], self.device_grid[1])
        max_block_tiles = max(1, self.prefill_matmul_cutoff // TILE)
        candidates = []
        for block_tiles in range(1, max_block_tiles + 1):
            grid_y = _largest_divisor(block_tiles, max_y)
            per_core_m = block_tiles // grid_y
            in0_block_w = self._prefill_in0_block_w(
                k_tiles=k_tiles, per_core_m=per_core_m, per_core_n=per_core_n, weight_dtype=weight_dtype
            )
            if not in0_block_w:
                continue
            block_rows = block_tiles * TILE
            padded_rows = _round_up(rows, block_rows)
            score = grid_x * grid_y * rows / padded_rows
            candidates.append((in0_block_w, score, per_core_m, block_rows, padded_rows, grid_x, grid_y, per_core_n))
        if not candidates:
            raise ValueError(f"no legal prefill matmul fold for rows={rows}, k_tiles={k_tiles}, n_tiles={n_tiles}")
        best_block_w = max(c[0] for c in candidates)
        candidates = [c for c in candidates if c[0] == best_block_w]
        in0_block_w, _, per_core_m, block_rows, padded_rows, grid_x, grid_y, per_core_n = max(
            candidates, key=lambda c: (c[1], c[2])
        )
        return block_rows, padded_rows, grid_x, grid_y, per_core_m, per_core_n, in0_block_w

    def _prefill_linear(self, x, weight, *, n: int, kernel_config, weight_dtype, fused_activation=None):
        """2D-program-config prefill matmul with the sequence folded into L1-sized blocks."""
        shape = list(x.shape)
        batch, seq, k = shape[0], shape[-2], shape[-1]
        rows = batch * seq
        block_rows, padded_rows, grid_x, grid_y, per_core_m, per_core_n, in0_block_w = self._prefill_fold(
            rows=rows, k_tiles=k // TILE, n_tiles=n // TILE, weight_dtype=weight_dtype
        )
        # out_subblock_h is capped by per_core_M as well as by the register file: a taller
        # subblock than the per-core output block does not exist.
        out_subblock_h = min(self.prefill_out_subblock_h, per_core_m)
        while out_subblock_h > 1 and per_core_m % out_subblock_h:
            out_subblock_h -= 1
        out_subblock_w = _out_subblock_w(per_core_n, out_subblock_h, self.prefill_out_subblock_max)
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=False,
        )

        # ttnn.reshape returns a view over the same buffer, so a view is never deallocated
        # independently of its source; only the pad/slice copies are owned.
        flat = ttnn.reshape(x, [1, 1, rows, k])
        padded = flat
        if padded_rows != rows:
            padded = ttnn.pad(flat, [(0, 0), (0, 0), (0, padded_rows - rows), (0, 0)], value=0.0)
        folded = ttnn.reshape(padded, [1, padded_rows // block_rows, block_rows, k])
        out = ttnn.linear(
            folded,
            weight,
            program_config=program_config,
            compute_kernel_config=kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision.activation_dtype,
        )
        if padded is not flat:
            padded.deallocate(True)
        out_flat = ttnn.reshape(out, [1, 1, padded_rows, n])
        if padded_rows != rows:
            trimmed = ttnn.slice(out_flat, [0, 0, 0, 0], [1, 1, rows, n])
            out.deallocate(True)
            out_flat = trimmed
        return ttnn.reshape(out_flat, [batch, 1, seq, n])

    def _mlp_prefill(self, x):
        cfg = self.config
        if self.pack_mlp_gate_up:
            packed = self._prefill_linear(
                x,
                self.w_mlp_gate_up,
                n=2 * cfg.intermediate_size,
                kernel_config=self.prefill_mlp_kernel_config,
                weight_dtype=self.precision.mlp_weight_dtype,
            )
            b, seq = packed.shape[0], packed.shape[-2]
            gate = ttnn.slice(packed, [0, 0, 0, 0], [b, 1, seq, cfg.intermediate_size])
            up = ttnn.slice(packed, [0, 0, 0, cfg.intermediate_size], [b, 1, seq, 2 * cfg.intermediate_size])
            packed.deallocate(True)
        else:
            gate = self._prefill_linear(
                x,
                self.w_mlp_gate,
                n=cfg.intermediate_size,
                kernel_config=self.prefill_mlp_kernel_config,
                weight_dtype=self.precision.mlp_weight_dtype,
            )
            up = self._prefill_linear(
                x,
                self.w_mlp_up,
                n=cfg.intermediate_size,
                kernel_config=self.prefill_mlp_kernel_config,
                weight_dtype=self.precision.mlp_weight_dtype,
            )
        # SiLU fused into the binary multiply: removes a full-size unary pass over the
        # 19968-wide intermediate (3.2% of functional prefill device time).
        activated = ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=self.precision.activation_dtype
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = self._prefill_linear(
            activated,
            self.w_mlp_down,
            n=cfg.hidden_size,
            kernel_config=self.prefill_mlp_down_kernel_config,
            weight_dtype=self.precision.mlp_down_weight_dtype,
        )
        activated.deallocate(True)
        return out

    def prefill_forward(
        self,
        hidden_states,
        *,
        kv_cache,
        page_table,
        user_ids=None,
        seq_len: int | None = None,
        start_pos: int = 0,
    ):
        """Multi-token prefill for one or more users. Contract identical to the functional layer."""
        cfg = self.config
        batch = hidden_states.shape[0]
        rows = list(range(batch)) if user_ids is None else [int(u) for u in user_ids]
        if len(rows) != batch:
            raise ValueError(f"user_ids has {len(rows)} entries for a batch of {batch}")
        if len(set(rows)) != len(rows):
            raise ValueError(f"user_ids must be distinct cache slots, got {rows}")
        if max(rows) >= page_table.shape[0]:
            raise ValueError(f"user_ids {rows} exceed the {page_table.shape[0]} page-table rows")
        logical_seq = seq_len if seq_len is not None else hidden_states.shape[-2]
        if logical_seq <= 0:
            raise ValueError(f"seq_len must be positive, got {logical_seq}")
        if hidden_states.shape[-2] < logical_seq:
            raise ValueError(f"hidden_states has {hidden_states.shape[-2]} rows, need at least {logical_seq}")
        if start_pos % TILE != 0 or start_pos % self.block_size != 0:
            raise ValueError(f"start_pos {start_pos} must be a multiple of {TILE} and block_size {self.block_size}")
        if start_pos and cfg.is_sliding:
            raise ValueError(
                f"start_pos={start_pos} (continued prefill) is not supported on a "
                f"sliding_attention layer: the first {cfg.sliding_window} positions of the new "
                "segment would attend to a truncated window. Prefill the whole prompt in one "
                "call (this layer already chunks internally), or extend the sliding path to "
                "read its window prefix from the paged cache."
            )
        if start_pos + logical_seq > cfg.max_position_embeddings:
            raise ValueError(
                f"start_pos + seq_len = {start_pos + logical_seq} exceeds max_position_embeddings "
                f"{cfg.max_position_embeddings}"
            )

        padded_seq = _round_up(logical_seq, TILE)
        x = hidden_states
        x_owned = False
        if x.shape[-2] < padded_seq:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded_seq - x.shape[-2]), (0, 0)], value=0.0)
            x_owned = True
        elif x.shape[-2] > padded_seq:
            x, x_owned = self._slice_view(x, [0, 0, 0, 0], [batch, x.shape[1], padded_seq, cfg.hidden_size])

        hist = cfg.sliding_window if cfg.is_sliding else 0
        chunk = self.prefill_chunk_size
        outputs = []
        chunk_start = 0
        while chunk_start < padded_seq:
            chunk_len = min(chunk, padded_seq - chunk_start)
            ext_start = max(0, chunk_start - hist) if cfg.is_sliding else chunk_start
            outputs.append(
                self._prefill_chunk(
                    x,
                    kv_cache=kv_cache,
                    page_table=page_table,
                    rows=rows,
                    ext_start=ext_start,
                    chunk_start=chunk_start,
                    chunk_len=chunk_len,
                    start_pos=start_pos,
                )
            )
            chunk_start += chunk_len

        if x_owned:
            x.deallocate(True)

        out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        if len(outputs) > 1:
            for chunk_out in outputs:
                chunk_out.deallocate(True)
        if out.shape[-2] != logical_seq:
            sliced, owned = self._slice_view(out, [0, 0, 0, 0], [batch, 1, logical_seq, cfg.hidden_size])
            if owned:
                out.deallocate(True)
            out = sliced
        return out

    def _prefill_chunk(
        self,
        x,
        *,
        kv_cache,
        page_table,
        rows,
        ext_start: int,
        chunk_start: int,
        chunk_len: int,
        start_pos: int,
    ):
        cfg = self.config
        batch = x.shape[0]
        hidden = cfg.hidden_size
        ext_len = chunk_start + chunk_len - ext_start
        keep_from = chunk_start - ext_start

        x_ext = (
            x
            if (ext_start == 0 and ext_len == x.shape[-2])
            else ttnn.slice(x, [0, 0, ext_start, 0], [batch, 1, ext_start + ext_len, hidden])
        )
        xn_ext = self._norm(x_ext, self.input_norm_w, cfg.rms_norm_eps)

        gate_ext = None
        if self.pack_qkv_gate:
            qkvg = self._prefill_linear(
                xn_ext,
                self.wqkvg,
                n=cfg.qkv_width + cfg.num_attention_heads * cfg.head_dim,
                kernel_config=self.prefill_attn_kernel_config,
                weight_dtype=self.precision.attn_weight_dtype,
            )
            packed_rows = qkvg.shape[-2]
            xqkv = ttnn.slice(qkvg, [0, 0, 0, 0], [batch, 1, packed_rows, cfg.qkv_width])
            gate_ext = ttnn.slice(
                qkvg,
                [0, 0, 0, cfg.qkv_width],
                [batch, 1, packed_rows, cfg.qkv_width + cfg.num_attention_heads * cfg.head_dim],
            )
            qkvg.deallocate(True)
        else:
            xqkv = self._prefill_linear(
                xn_ext,
                self.wqkv,
                n=cfg.qkv_width,
                kernel_config=self.prefill_attn_kernel_config,
                weight_dtype=self.precision.attn_weight_dtype,
            )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)

        q = self._qk_norm(q)
        k = self._qk_norm(k)
        # HF scales Q by qk_scale_factor after the QK norm and before RoPE.
        q = ttnn.multiply(q, cfg.qk_scale_factor)

        if cfg.uses_rope:
            q, k = self._rope_prefill(q, k, position_start=start_pos + ext_start, length=ext_len)

        k_new = k if keep_from == 0 else ttnn.slice(k, [0, 0, keep_from, 0], [batch, k.shape[1], ext_len, cfg.head_dim])
        v_new = v if keep_from == 0 else ttnn.slice(v, [0, 0, keep_from, 0], [batch, v.shape[1], ext_len, cfg.head_dim])
        self._fill_paged_cache(
            kv_cache,
            k_new,
            v_new,
            page_table=page_table,
            rows=rows,
            absolute_start=start_pos + chunk_start,
        )
        if k_new is not k:
            k_new.deallocate(True)
        if v_new is not v:
            v_new.deallocate(True)

        if cfg.is_sliding:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=cfg.sdpa_scale,
                sliding_window_size=cfg.sliding_window,
                program_config=self._sdpa_program_config(
                    q_chunk=self._sdpa_chunk_for_length(ext_len, self.prefill_sdpa_q_chunk),
                    k_chunk=self._sdpa_chunk_for_length(ext_len, self.prefill_sdpa_k_chunk),
                ),
                compute_kernel_config=self.sdpa_kernel_config,
            )
            q.deallocate(True)
            k.deallocate(True)
            v.deallocate(True)
            if keep_from:
                trimmed = ttnn.slice(
                    attn, [0, 0, keep_from, 0], [batch, cfg.num_attention_heads, ext_len, cfg.head_dim]
                )
                attn.deallocate(True)
                attn = trimmed
        else:
            k.deallocate(True)
            v.deallocate(True)
            attn = self._chunked_prefill_sdpa(
                q,
                kv_cache=kv_cache,
                page_table=page_table,
                rows=rows,
                chunk_start_idx=start_pos + chunk_start,
                chunk_len=chunk_len,
            )
            q.deallocate(True)

        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)

        if gate_ext is not None:
            xn = xn_ext
            gate = (
                gate_ext
                if keep_from == 0
                else ttnn.slice(
                    gate_ext,
                    [0, 0, keep_from, 0],
                    [batch, 1, ext_len, cfg.num_attention_heads * cfg.head_dim],
                )
            )
            if gate is not gate_ext:
                gate_ext.deallocate(True)
        else:
            xn = xn_ext if keep_from == 0 else ttnn.slice(xn_ext, [0, 0, keep_from, 0], [batch, 1, ext_len, hidden])
            gate = self._prefill_linear(
                xn,
                self.w_attn_gate,
                n=cfg.num_attention_heads * cfg.head_dim,
                kernel_config=self.prefill_attn_kernel_config,
                weight_dtype=self.precision.attn_weight_dtype,
            )
        # Sigmoid fused into the gate multiply (one fewer full-size unary pass).
        gated = ttnn.multiply(
            attn_cat,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            dtype=self.precision.activation_dtype,
        )
        gate.deallocate(True)
        attn_cat.deallocate(True)
        attn_cat = gated
        if xn is not xn_ext:
            xn.deallocate(True)
        xn_ext.deallocate(True)

        attn_out = self._prefill_linear(
            attn_cat,
            self.wo,
            n=hidden,
            kernel_config=self.prefill_attn_kernel_config,
            weight_dtype=self.precision.attn_weight_dtype,
        )
        attn_cat.deallocate(True)
        attn_out = self._norm(attn_out, self.post_attn_norm_w, cfg.post_norm_eps)

        x_chunk = x_ext if keep_from == 0 else ttnn.slice(x_ext, [0, 0, keep_from, 0], [batch, 1, ext_len, hidden])
        h = ttnn.add(x_chunk, attn_out)
        attn_out.deallocate(True)
        if x_chunk is not x_ext:
            x_chunk.deallocate(True)
        if x_ext is not x:
            x_ext.deallocate(True)

        y = self._norm(h, self.pre_ff_norm_w, cfg.rms_norm_eps)
        y = self._mlp_prefill(y)
        y = self._norm(y, self.post_ff_norm_w, cfg.post_norm_eps)
        out = ttnn.add(h, y)
        h.deallocate(True)
        y.deallocate(True)
        return out

    def _rope_prefill(self, q, k, *, position_start: int, length: int):
        cfg = self.config
        cache = self.rope_cache
        if position_start % TILE != 0:
            raise ValueError(f"rope position_start {position_start} must be a multiple of {TILE}")
        begins = [0, 0, position_start, 0]
        ends = [1, 1, position_start + length, cfg.head_dim]
        cos, cos_owned = self._slice_view(cache["cos_prefill"], begins, ends)
        sin, sin_owned = self._slice_view(cache["sin_prefill"], begins, ends)
        q_out = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k_out = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        q.deallocate(True)
        k.deallocate(True)
        if cos_owned:
            cos.deallocate(True)
        if sin_owned:
            sin.deallocate(True)
        return q_out, k_out

    def _fill_paged_cache(self, kv_cache, k, v, *, page_table, rows, absolute_start: int):
        """Write ``k``/``v`` into the paged cache at absolute positions ``[absolute_start, ...)``."""
        k_cache, v_cache = kv_cache
        batch = k.shape[0]
        first_block = absolute_start // self.block_size
        blocks_needed = _round_up(k.shape[-2], self.block_size) // self.block_size
        if first_block + blocks_needed > page_table.shape[1]:
            raise ValueError(
                f"page table has {page_table.shape[1]} blocks per sequence, need "
                f"{first_block + blocks_needed} for positions [{absolute_start}, "
                f"{absolute_start + k.shape[-2]})"
            )

        # paged_fill_cache wants the fill tensor in the cache dtype (BFP8 here).
        k_fill = k if k.dtype == self.cache_dtype else ttnn.typecast(k, self.cache_dtype)
        v_fill = v if v.dtype == self.cache_dtype else ttnn.typecast(v, self.cache_dtype)
        for index, row in enumerate(rows):
            row_page_table, pt_owned = self._slice_view(
                page_table, [row, first_block], [row + 1, first_block + blocks_needed]
            )
            if batch == 1:
                k_user, v_user, user_owned = k_fill, v_fill, False
            else:
                k_user, user_owned = self._slice_view(
                    k_fill, [index, 0, 0, 0], [index + 1, k.shape[1], k.shape[2], k.shape[3]]
                )
                v_user, _ = self._slice_view(v_fill, [index, 0, 0, 0], [index + 1, v.shape[1], v.shape[2], v.shape[3]])
            ttnn.experimental.paged_fill_cache(k_cache, k_user, row_page_table, block_size=self.block_size)
            ttnn.experimental.paged_fill_cache(v_cache, v_user, row_page_table, block_size=self.block_size)
            if pt_owned:
                row_page_table.deallocate(True)
            if user_owned:
                k_user.deallocate(True)
                v_user.deallocate(True)
        if k_fill is not k:
            k_fill.deallocate(True)
        if v_fill is not v:
            v_fill.deallocate(True)

    @staticmethod
    def _sdpa_chunk_for_length(length: int, cap: int) -> int:
        """Largest candidate chunk size ``<= cap`` that divides ``length`` (non-dividing hangs the op)."""
        if length % TILE != 0:
            raise ValueError(f"SDPA sequence length {length} must be tile-aligned")
        for candidate in SDPA_CHUNK_CANDIDATES:
            if candidate <= cap and candidate <= length and length % candidate == 0:
                return candidate
        return TILE

    def _chunked_sdpa_chunk_sizes(self, chunk_len: int, chunk_start_idx: int, kv_length: int) -> tuple[int, int]:
        """``(q_chunk, k_chunk)`` for ``chunked_scaled_dot_product_attention``.

        The K candidate additionally keeps the op's *rounded* K extent inside the page
        table: the program factory rounds ``Sk`` up to ``k_chunk_size`` and the reader
        walks one page-table entry per ``block_size`` of the rounded extent with no bound
        check (see the functional stage's work log, section 9).
        """
        q_candidates = [
            c
            for c in SDPA_CHUNK_CANDIDATES
            if c <= self.prefill_sdpa_q_chunk and chunk_len % c == 0 and chunk_start_idx % c == 0
        ]
        k_candidates = [
            c
            for c in SDPA_CHUNK_CANDIDATES
            if c <= self.prefill_sdpa_k_chunk
            and chunk_start_idx % c == 0
            and _round_up(chunk_start_idx + chunk_len, c) <= kv_length
        ]
        if not q_candidates or not k_candidates:
            raise ValueError(
                f"cannot pick SDPA chunk sizes for chunk_len={chunk_len}, "
                f"chunk_start_idx={chunk_start_idx}, kv_length={kv_length}"
            )
        return max(q_candidates), max(k_candidates)

    def _rows_page_table(self, page_table, rows):
        """Page table restricted to ``rows``, in input-batch order. Returns ``(tensor, owned)``."""
        if rows == list(range(page_table.shape[0])):
            return page_table, False
        blocks = page_table.shape[1]
        row_slices = [self._slice_view(page_table, [row, 0], [row + 1, blocks]) for row in rows]
        if len(row_slices) == 1:
            return row_slices[0]
        gathered = ttnn.concat([tensor for tensor, _ in row_slices], dim=0)
        for tensor, owned in row_slices:
            if owned:
                tensor.deallocate(True)
        return gathered, True

    def _chunked_prefill_sdpa(self, q, *, kv_cache, page_table, rows, chunk_start_idx: int, chunk_len: int):
        """Causal SDPA for one Q chunk against the full paged prefix (full-attention layers)."""
        k_cache, v_cache = kv_cache
        kv_length = page_table.shape[1] * self.block_size
        q_chunk, k_chunk = self._chunked_sdpa_chunk_sizes(chunk_len, chunk_start_idx, kv_length)
        program_config = self._sdpa_program_config(q_chunk=q_chunk, k_chunk=k_chunk)
        sdpa_page_table, owned = self._rows_page_table(page_table, rows)

        # No `scale=`: the op's nanobind binding marks scale `.noconvert()`, so a Python
        # float cannot be passed. Its default is 1/sqrt(head_dim) == cfg.sdpa_scale and Q
        # already carries qk_scale_factor, so the omission is exact.
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            sdpa_page_table,
            chunk_start_idx,
            program_config=program_config,
            compute_kernel_config=self.sdpa_kernel_config,
        )
        if owned:
            sdpa_page_table.deallocate(True)
        return out

    # ----------------------------------------------------------------- decode

    def _to_residual(self, x):
        """Put a tensor on the decode residual contract, resharding only if it is not already."""
        if x.memory_config() == self.decode_residual_memcfg:
            return x
        out = ttnn.to_memory_config(x, self.decode_residual_memcfg)
        x.deallocate(True)
        return out

    def _decode_out_memcfg(self, role: str):
        """Output memory config for a decode matmul role.

        The DRAM-sharded op ignores this and builds its own layout, so that family just
        asks for ``L1_WIDTH_SHARDED`` and the consumers follow the producer. The 1D
        multicast family does honour it, and every role lands on the same 16-core
        rectangle as the residual, which is what removes the SwiGLU reshards.
        """
        if self.decode_matmul != "mcast1d":
            return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        return {
            "qkv": self.decode_qkv_out_memcfg,
            "attn_gate": self.decode_attn_gate_memcfg,
            "wo": self.decode_residual_memcfg,
            "mlp_gate_up": self.decode_mlp_out_memcfg,
            "mlp_down": self.decode_residual_memcfg,
        }[role]

    def _decode_linear(self, x, weight, *, role: str, program_config, memory_config, kernel_config):
        if self.decode_matmul in ("dram_sharded", "mcast1d"):
            return ttnn.linear(
                x,
                weight,
                program_config=program_config,
                compute_kernel_config=kernel_config,
                memory_config=memory_config,
                dtype=self.precision.activation_dtype,
            )
        # "interleaved": the functional stage's default-program-config path, kept as the
        # no-explicit-config baseline for the family comparison.
        return ttnn.linear(
            x,
            weight,
            compute_kernel_config=kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.precision.activation_dtype,
        )

    def _mlp_decode(self, x):
        cfg = self.config
        if self.decode_matmul == "dram_sharded" and x.memory_config() != self.decode_mlp_in_memcfg:
            x_mlp = ttnn.to_memory_config(x, self.decode_mlp_in_memcfg)
            x.deallocate(True)
            x = x_mlp
        if self.pack_mlp_gate_up:
            packed = self._decode_linear(
                x,
                self.w_mlp_gate_up,
                role="mlp_gate_up_packed",
                program_config=self.decode_mlp_packed_pc,
                memory_config=self._decode_out_memcfg("mlp_gate_up_packed"),
                kernel_config=self.mlp_kernel_config,
            )
            rows = packed.shape[-2]
            gate = ttnn.slice(packed, [0, 0, 0, 0], [1, 1, rows, cfg.intermediate_size])
            up = ttnn.slice(packed, [0, 0, 0, cfg.intermediate_size], [1, 1, rows, 2 * cfg.intermediate_size])
            packed.deallocate(True)
        else:
            gate = self._decode_linear(
                x,
                self.w_mlp_gate,
                role="mlp_gate_up",
                program_config=self.decode_mlp_gate_up_pc,
                memory_config=self._decode_out_memcfg("mlp_gate_up"),
                kernel_config=self.mlp_kernel_config,
            )
            up = self._decode_linear(
                x,
                self.w_mlp_up,
                role="mlp_gate_up",
                program_config=self.decode_mlp_gate_up_pc,
                memory_config=self._decode_out_memcfg("mlp_gate_up"),
                kernel_config=self.mlp_kernel_config,
            )
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=gate.memory_config(),
            dtype=self.precision.activation_dtype,
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = self._decode_linear(
            activated,
            self.w_mlp_down,
            role="mlp_down",
            program_config=self.decode_mlp_down_pc,
            memory_config=self._decode_out_memcfg("mlp_down"),
            kernel_config=self.mlp_down_kernel_config,
        )
        activated.deallocate(True)
        return self._to_residual(out)

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, rope_idxs=None):
        """Single-token decode for ``batch`` users.

        ``hidden_states``: ``[1, 1, batch, hidden_size]``, TILE. Interleaved or already in
        :attr:`decode_residual_memory_config`; the output is always in that config.
        """
        cfg = self.config
        batch = hidden_states.shape[-2]
        k_cache, v_cache = kv_cache
        if page_table.shape[0] != batch:
            raise ValueError(
                f"decode needs one page-table row per user: got {page_table.shape[0]} rows for batch {batch}"
            )
        if current_pos.shape[-1] != batch:
            raise ValueError(f"current_pos has {current_pos.shape[-1]} entries for batch {batch}")
        self._check_decode_page_table_capacity(page_table.shape[1] * self.block_size)

        x = hidden_states
        x_owned = False
        if x.memory_config() != self.decode_residual_memcfg:
            x = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
            x_owned = True

        xn = self._sharded_norm(x, self.input_norm_w, cfg.rms_norm_eps)

        if self.pack_qkv_gate:
            qkvg = self._decode_linear(
                xn,
                self.wqkvg,
                role="qkvg",
                program_config=self.decode_qkvg_pc,
                memory_config=self._decode_out_memcfg("qkvg"),
                kernel_config=self.attn_kernel_config,
            )
            rows = qkvg.shape[-2]
            gate_width = cfg.num_attention_heads * cfg.head_dim
            xqkv = ttnn.slice(qkvg, [0, 0, 0, 0], [1, 1, rows, cfg.qkv_width])
            gate = ttnn.slice(qkvg, [0, 0, 0, cfg.qkv_width], [1, 1, rows, cfg.qkv_width + gate_width])
            qkvg.deallocate(True)
        else:
            xqkv = self._decode_linear(
                xn,
                self.wqkv,
                role="qkv",
                program_config=self.decode_qkv_pc,
                memory_config=self._decode_out_memcfg("qkv"),
                kernel_config=self.attn_kernel_config,
            )
            gate = self._decode_linear(
                xn,
                self.w_attn_gate,
                role="attn_gate",
                program_config=self.decode_attn_gate_pc,
                memory_config=self._decode_out_memcfg("attn_gate"),
                kernel_config=self.attn_kernel_config,
            )
        xn.deallocate(True)

        # nlp_create_qkv_heads_decode wants an L1 tensor: its interleaved-DRAM reader
        # zeroes odd Q rows on Blackhole (tt-metal #16667).
        xqkv_l1 = ttnn.sharded_to_interleaved(xqkv, ttnn.L1_MEMORY_CONFIG) if xqkv.is_sharded() else xqkv
        if xqkv_l1 is not xqkv:
            xqkv.deallocate(True)
        elif xqkv_l1.memory_config().buffer_type == ttnn.BufferType.DRAM:
            moved = ttnn.to_memory_config(xqkv_l1, ttnn.L1_MEMORY_CONFIG)
            xqkv_l1.deallocate(True)
            xqkv_l1 = moved

        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_l1,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv_l1.deallocate(True)

        q = self._norm_sharded(q, self._qk_norm)
        k = self._norm_sharded(k, self._qk_norm)
        q_scaled = ttnn.multiply(q, cfg.qk_scale_factor, memory_config=q.memory_config())
        q.deallocate(True)
        q = q_scaled

        if cfg.uses_rope:
            if rope_idxs is None:
                raise ValueError("rope_idxs is required for a RoPE (sliding_attention) layer")
            cos, sin = self._rope_decode_mats(rope_idxs, batch=batch, like=q)
            q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
            k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
            q.deallocate(True)
            k.deallocate(True)
            cos.deallocate(True)
            sin.deallocate(True)
            q, k = q_rot, k_rot

        # paged_update_cache takes BF16/FLOAT32 inputs even for a reduced-precision cache.
        ttnn.experimental.paged_update_cache(
            k_cache,
            k,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        k.deallocate(True)
        v.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=cfg.sdpa_scale,
            sliding_window_size=cfg.sliding_window,
            # The next op reshards this to one core per user anyway, so the interleaved
            # buffer type is a candidate, not a contract; measured in the candidate table.
            memory_config=ttnn.L1_MEMORY_CONFIG if self.decode_sdpa_output_l1 else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._decode_sdpa_program_config(batch),
            compute_kernel_config=self.sdpa_kernel_config,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        q.deallocate(True)

        attn_cat = self._concat_heads_decode(attn, batch, gate.memory_config())
        attn.deallocate(True)

        # Sigmoid fused into the gate multiply; both operands are width-sharded on the
        # same grid, so the whole gated attention output stays in L1.
        gated = ttnn.multiply(
            attn_cat,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            memory_config=gate.memory_config(),
            dtype=self.precision.activation_dtype,
        )
        attn_cat.deallocate(True)
        gate.deallocate(True)

        attn_out = self._decode_linear(
            gated,
            self.wo,
            role="wo",
            program_config=self.decode_wo_pc,
            memory_config=self._decode_out_memcfg("wo"),
            kernel_config=self.attn_kernel_config,
        )
        gated.deallocate(True)
        attn_out = self._to_residual(attn_out)
        attn_out = self._sharded_norm(attn_out, self.post_attn_norm_w, cfg.post_norm_eps)
        h = ttnn.add(x, attn_out, memory_config=self.decode_residual_memcfg)
        attn_out.deallocate(True)
        if x_owned:
            x.deallocate(True)

        y = self._sharded_norm(h, self.pre_ff_norm_w, cfg.rms_norm_eps)
        y = self._mlp_decode(y)
        y = self._sharded_norm(y, self.post_ff_norm_w, cfg.post_norm_eps)
        out = ttnn.add(h, y, memory_config=self.decode_residual_memcfg)
        h.deallocate(True)
        y.deallocate(True)
        return out

    def _norm_sharded(self, x, norm_fn):
        """Run a norm on a height-sharded decode tensor, restoring the shard config."""
        mem_config = x.memory_config()
        x_int = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x.deallocate(True)
        normed = norm_fn(x_int)
        x_int.deallocate(True)
        out = ttnn.to_memory_config(normed, mem_config)
        normed.deallocate(True)
        return out

    def _rope_decode_mats(self, rope_idxs, *, batch: int, like):
        """Gather per-user cos/sin rows on device and shard them exactly like ``like``.

        The sharded ``rotary_embedding_hf`` decode kernel pairs shard *i* of cos/sin with
        shard *i* of Q, so the layout must be derived from Q's own shard spec rather than
        constructed (``nlp_create_qkv_heads_decode`` picks a non-rectangular set for some
        batches; using a tidy rectangle instead rotates users by each other's positions).
        """
        cfg = self.config
        cache = self.rope_cache
        idxs, idxs_owned = rope_idxs, False
        if self.decode_rope_pad_to_tile and rope_idxs.shape[-1] < ttnn.TILE_SIZE:
            # Candidate for the per-token TilizeWithValPadding the batch-1 gather pays:
            # gather a whole tile row of positions so the tilize has nothing to pad. The
            # extra rows index position 0, which is a valid row of the cache, and they are
            # sliced off below.
            idxs = ttnn.pad(rope_idxs, [(0, 0), (0, ttnn.TILE_SIZE - rope_idxs.shape[-1])], value=0)
            idxs_owned = True
        cos = ttnn.embedding(idxs, cache["cos_decode"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idxs, cache["sin_decode"], layout=ttnn.TILE_LAYOUT)
        if idxs_owned:
            idxs.deallocate(True)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        if cos.shape[1] != batch:
            cos = cos[:, :batch, :, :]
            sin = sin[:, :batch, :, :]
        shard_spec = like.memory_config().shard_spec
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, cfg.head_dim),
            core_grid=shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=shard_spec.orientation,
            use_height_and_width_as_shard_shape=True,
        )
        cos_sh = ttnn.to_memory_config(cos, shard_cfg)
        sin_sh = ttnn.to_memory_config(sin, shard_cfg)
        cos.deallocate(True)
        sin.deallocate(True)
        return cos_sh, sin_sh

    def _decode_core_range_set(self, batch: int):
        """One core per user, arranged as a single rectangle (nlp_concat_heads_decode needs one)."""
        grid_x, grid_y = self.device_grid
        if batch > grid_x * grid_y:
            raise ValueError(f"batch {batch} exceeds the {grid_x}x{grid_y} core grid")
        width = min(batch, grid_x)
        if batch % width != 0 or batch // width > grid_y:
            candidates = [c for c in range(width, 0, -1) if batch % c == 0 and batch // c <= grid_y]
            if not candidates:
                raise ValueError(f"cannot fit batch {batch} in a rectangle on a {grid_x}x{grid_y} grid")
            width = candidates[0]
        return ttnn.CoreRangeSet({num_to_corerange(batch, grid_x=width, grid_y=grid_y)})

    def _decode_batch_shard_config(self, batch: int, width: int):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width),
            core_grid=self._decode_core_range_set(batch),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def blocks_per_seq(self, max_seq_len: int) -> int:
        """Page-table width (blocks per sequence), rounded so capacity is a whole number of K chunks."""
        blocks = _round_up(max_seq_len, self.block_size) // self.block_size
        capacity_step = _lcm(self.block_size, self.decode_sdpa_k_chunk)
        return _round_up(blocks * self.block_size, capacity_step) // self.block_size

    def _check_decode_page_table_capacity(self, kv_length: int) -> None:
        """The decode SDPA rounds its K extent up; the page table must cover the rounded value."""
        if kv_length % self.decode_sdpa_k_chunk == 0:
            return
        needed = _round_up(kv_length, _lcm(self.block_size, self.decode_sdpa_k_chunk))
        raise ValueError(
            f"decode page-table capacity {kv_length} tokens "
            f"({kv_length // self.block_size} blocks x block_size {self.block_size}) is not a "
            f"whole number of decode K chunks ({self.decode_sdpa_k_chunk}); the SDPA decode op "
            "rounds its K extent up to the K chunk and reads the page table past its last "
            f"entry. Allocate {needed // self.block_size} blocks per sequence instead "
            "(OptimizedDecoder.blocks_per_seq does this)."
        )

    def _decode_sdpa_program_config(self, batch: int):
        """Decode SDPA grid. Needs one core per (user, KV head); see the functional stage's bug 8."""
        cfg = self.config
        min_cores = batch * cfg.num_key_value_heads
        grid_x, grid_y = self.device_grid
        grid = self.decode_sdpa_core_grid
        if grid[0] * grid[1] < min_cores:
            grid = (grid_x, grid_y)
        if grid[0] * grid[1] < min_cores:
            raise ValueError(
                f"decode batch {batch} needs {min_cores} cores (batch * num_key_value_heads) "
                f"but the compute grid only has {grid_x * grid_y}"
            )
        return self._sdpa_program_config(q_chunk=TILE, k_chunk=self.decode_sdpa_k_chunk, grid=grid)

    def _concat_heads_decode(self, attn, batch: int, target_memcfg):
        """Concat decode heads and land the result on the attention-gate projection's shard."""
        cfg = self.config
        shard_cfg = self._decode_batch_shard_config(batch, cfg.head_dim)
        attn_sh = ttnn.to_memory_config(attn, shard_cfg)
        out_sh = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=cfg.num_attention_heads)
        attn_sh.deallocate(True)
        if out_sh.shape[2] != batch:  # the op pads the batch dim up to a tile
            trimmed = out_sh[:, :, :batch, :]
            out_sh.deallocate(True)
            out_sh = trimmed
        out = ttnn.to_memory_config(out_sh, target_memcfg)
        out_sh.deallocate(True)
        return out

    # ------------------------------------------------------------------ hosts

    def forward(self, *args, mode: str = "decode", **kwargs):
        """Dispatch to :meth:`prefill_forward` / :meth:`decode_forward`."""
        if mode == "prefill":
            return self.prefill_forward(*args, **kwargs)
        if mode == "decode":
            return self.decode_forward(*args, **kwargs)
        raise ValueError(f"unknown mode {mode!r}")

    # ------------------------------------------------------------- reporting

    def config_summary(self) -> dict:
        """Structured description of the selected precision/layout/program configs."""
        from models.common.tensor_utils import program_config_to_dict

        def dt(x):
            return str(x)

        return {
            "precision_policy": self.precision.name,
            "dtypes": {
                "activation": dt(self.precision.activation_dtype),
                "attn_weights": dt(self.precision.attn_weight_dtype),
                "mlp_gate_up_weights": dt(self.precision.mlp_weight_dtype),
                "mlp_down_weights": dt(self.precision.mlp_down_weight_dtype),
                "kv_cache": dt(self.precision.kv_cache_dtype),
                "norm_weights": dt(self.precision.norm_dtype),
            },
            "fidelity": {
                "attn_decode": dt(self.precision.attn_fidelity),
                "mlp_decode": dt(self.precision.mlp_fidelity),
                "mlp_down_decode": dt(self.precision.mlp_down_fidelity),
                "attn_prefill": dt(self.precision.prefill_attn_fidelity),
                "mlp_prefill": dt(self.precision.prefill_mlp_fidelity),
                "mlp_down_prefill": dt(self.precision.prefill_mlp_down_fidelity),
                "norm": dt(self.precision.norm_fidelity),
                "sdpa": dt(self.precision.sdpa_fidelity),
            },
            "decode": {
                "matmul_family": self.decode_matmul,
                "residual_grid": list(self.decode_residual_grid),
                "residual_cores": self.decode_cores,
                "residual_shard_width": self.config.hidden_size // self.decode_cores,
                "attn_gate_shard_width": self.decode_attn_gate_memcfg.shard_spec.shape[1],
                "mlp_working_cores": self.decode_mlp_cores,
                "mlp_working_shard_width": self.config.hidden_size // self.decode_mlp_cores,
                "sdpa_grid": list(self.decode_sdpa_core_grid),
                "sdpa_k_chunk": self.decode_sdpa_k_chunk,
                "packer_l1_acc": self.decode_packer_l1_acc,
                "fp32_dest_acc_en": self.decode_fp32_acc,
                "pack_qkv_gate": self.pack_qkv_gate,
                "pack_mlp_gate_up": self.pack_mlp_gate_up,
                "sdpa_output_l1": self.decode_sdpa_output_l1,
                "rope_pad_to_tile": self.decode_rope_pad_to_tile,
                "program_configs": {
                    role: program_config_to_dict(pc)
                    for role, pc in (
                        ("qkv", self.decode_qkv_pc),
                        ("attn_gate", self.decode_attn_gate_pc),
                        ("wo", self.decode_wo_pc),
                        ("mlp_gate_up", self.decode_mlp_gate_up_pc),
                        ("mlp_down", self.decode_mlp_down_pc),
                    )
                },
            },
            "prefill": {
                "matmul_grid": list(self.prefill_matmul_grid),
                "matmul_cutoff": self.prefill_matmul_cutoff,
                "in0_block_w_cap": self.prefill_in0_block_w_cap,
                "out_subblock_max": self.prefill_out_subblock_max,
                "out_subblock_h": self.prefill_out_subblock_h,
                "weight_memory": self.weight_memory,
                "chunk_size": self.prefill_chunk_size,
                "sdpa_q_chunk": self.prefill_sdpa_q_chunk,
                "sdpa_k_chunk": self.prefill_sdpa_k_chunk,
            },
            "block_size": self.block_size,
        }
