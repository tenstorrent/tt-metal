# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phi-3.5-mini (``microsoft/Phi-3.5-mini-instruct``) — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``). Target: one Blackhole P150 (mesh ``(1, 1)``).

Phi-3.5-mini is a Phi3-architecture 3.8B dense decoder (32 layers, hidden 3072, 32 attn / 32 KV heads,
head_dim 96, SwiGLU MLP intermediate 8192, vocab 32064, ``rope_theta=10000`` with longrope scaling,
full RoPE, no QKV bias, no q/k norm). The HF checkpoint ships **fused** ``qkv_proj`` and
``gate_up_proj`` — :mod:`weight_utils` splits them host-side before this builds TTNN modules.

RoPE: the tables hold the longrope *short* factors (positions below ``original_max_position_embeddings``,
4096), which is what Hugging Face applies while a sequence stays within 4096 tokens. Contexts beyond 4096
(long factors) are not supported by this port; ``hf_adaptor.from_pretrained`` rejects them.

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

The tensor graph is provider-neutral. ``hf_adaptor.py`` owns Hugging Face loading,
checkpoint validation, and tensor conversion; this module only builds and executes
the TT graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.lightweightmodule import LightweightModule
from models.common.models.phi35_mini import weight_utils
from models.common.models.phi35_mini.attention import PREFILL_1D_SEQ_LEN, Phi35MiniAttention, Phi35MiniAttentionConfig
from models.common.models.phi35_mini.mlp import Phi35MiniMLP, Phi35MiniMLPConfig
from models.common.models.phi35_mini.rope import Phi35MiniRotarySetup
from models.common.models.phi35_mini.sampling import Phi35MiniSampling
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _nearest_32
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig, _create_sharded_norm_program_config
from models.common.modules.rope.rope_1d import Rope1DConfig
from models.common.modules.sampling.sampling_1d import Sampling1DConfig
from models.common.modules.tt_ccl import default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim



# phi-3.5-mini S11: TT_PHI35_GRID12=1 adapts the P150 (13 x 10) configs to a 12 x 10 Galaxy Blackhole chip.
_S11_GRID12 = __import__("os").environ.get("TT_PHI35_GRID12") == "1"

def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Phi35MiniPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Phi35MiniTransformerConfig:
    """Complete provider-neutral tensor-graph configuration."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    dim: int
    num_devices: int
    mesh_device: ttnn.MeshDevice
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[Phi35MiniDecoderLayerConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)
    tt_ccl: Any = None
    cache_path: str | None = None


@dataclass(frozen=True)
class Phi35MiniModelParameters:
    """Provider-neutral Phi-3.5-mini dimensions consumed by the TT config builder."""

    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_batch_size: int
    max_seq_len: int


@dataclass(frozen=True)
class Phi35MiniLayerWeights:
    wqkv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    attention_norm: torch.Tensor
    ff_norm: torch.Tensor


@dataclass(frozen=True)
class Phi35MiniWeights:
    embedding: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: tuple[Phi35MiniLayerWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
"""LoFi + packer L1 acc for the BFP4 MLP FF1/FF3 matmuls in performance mode."""

_HIFI4_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
"""HiFi4 + fp32 accumulation for the BF16 attention matmuls and SDPA in accuracy mode."""

_HIFI2_FP32_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
"""HiFi2 with fp32 accumulation: the 128-row QKV and W2 on the DRAM-streamed program."""


@dataclass(frozen=True)
class Phi35MiniPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Phi-3.5-mini.

    Mirrors what TTTv1 ``DecodersPrecision`` resolves for ``Phi-3.5-mini-instruct``
    (``models/tt_transformers/tt/model_config.py``, ``ModelOptimizations``):

      * **Accuracy**: Phi-3.5 is not in the Llama-3 / Mistral / Phi-3-mini BFP8-attention branch, so it takes
        the generic branch: BF16 WQKV / WO / KV cache with HiFi4 QKV, SDPA and output matmuls in decode and
        prefill; BFP8 MLP with the HIFI2_FP16 default.
      * **Performance**: BFP4 + LoFi for the MLP FF1/FF3 matmuls; BFP8 FF2 at the ``MLP1D`` HIFI2_FP16 default;
        BFP8 attention weights and KV cache at the ``Attention1D`` HiFi2 default. TTTv1 also puts FF2 in BFP4 +
        LoFi for Phi-3.5; on this port that scores top-1 87.9% on the book reference, below the 89% target
        (``models/model_targets.yaml``), so FF2 stays BFP8 (91.6%).

    Two module-level recipes are exposed: :data:`PHI35_MINI_ACCURACY` and :data:`PHI35_MINI_PERFORMANCE`.
    Use ``dataclasses.replace`` to customize a single field.
    """

    wqkv_dtype: ttnn.DataType = ttnn.bfloat8_b
    wo_dtype: ttnn.DataType = ttnn.bfloat8_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_w2_dtype: ttnn.DataType = ttnn.bfloat8_b

    # ``None`` keeps the module default (MLP1D: HIFI2_FP16; Attention1D: HiFi2).
    mlp_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    mlp_ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    attention_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b


PHI35_MINI_ACCURACY = Phi35MiniPrecisionConfig(
    wqkv_dtype=ttnn.bfloat16,
    wo_dtype=ttnn.bfloat16,
    kv_cache_dtype=ttnn.bfloat16,
    attention_compute_kernel_cfg=_HIFI4_COMPUTE_KERNEL_CFG,
)

PHI35_MINI_PERFORMANCE = Phi35MiniPrecisionConfig(
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_ff1_3_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
)


# Decode Q/K core grid. The rope's cos/sin, the rope transformation matrix and the fused-QK head split must put
# user u's Q and K on the same cores. Attention1D lays the transformation matrix out over an 8-wide grid; the
# rope defaults to the full 13-wide P150 grid, which put every user past the first row on other cores' cos/sin.
_DECODE_QK_CORE_GRID = ttnn.CoreCoord(8, 8)

# From this many users up, the paged KV cache is kept as DRAM ND shards and decode SDPA runs over the whole compute
# grid (see _kv_cache_memory_config). Below it both stay as they were: one user's SDPA is fastest on its 8x8 grid
# with Q sharded (17.3 us vs 19.3 us on the whole grid with Q in DRAM).
_KV_ND_SHARD_MIN_BATCH = 16
_KV_ND_SHARD_HEADS = 8


def _kv_cache_nd_sharded(max_batch_size: int, n_kv_heads: int) -> bool:
    return max_batch_size >= _KV_ND_SHARD_MIN_BATCH and n_kv_heads % _KV_ND_SHARD_HEADS == 0


def _kv_cache_memory_config(
    mesh_device: ttnn.MeshDevice, *, max_batch_size: int, n_kv_heads: int, block_size: int, head_dim: int
) -> ttnn.MemoryConfig:
    """Memory config of each layer's paged K and V cache, [blocks, kv heads, block size, head dim].

    Interleaved, consecutive tiles of a (block, head) row land in consecutive DRAM banks, and batch-32 decode SDPA
    reads them at about 45% of DRAM bandwidth. As ND shards of 8 heads of one block (24 tiles, contiguous in one bank,
    shards round-robin over the banks) with SDPA over the whole grid, batch-32 decode SDPA on a P150 takes
    83-94 / 150-165 / 285-308 us at positions 100 / 227 / 500 against 111-116 / 205-211 / 394-417 us interleaved on
    8x8, with contiguous or shuffled page tables. Whole blocks per shard (32 heads) or fewer heads per shard (1-4)
    are slower.
    """
    if not _kv_cache_nd_sharded(max_batch_size, n_kv_heads):
        return ttnn.DRAM_MEMORY_CONFIG
    banks = mesh_device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            ttnn.Shape([1, _KV_ND_SHARD_HEADS, block_size, head_dim]), grid, ttnn.ShardOrientation.ROW_MAJOR
        ),
    )


def _decode_dram_matmul_config(
    *,
    k: int,
    n: int,
    in0_cores: int,
    workers_per_bank: int,
    in0_block_w: int,
    per_core_n: int | None = None,
    fuse_swiglu: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    """Batch-1..32 decode matmul (M = one tile) on the multi-core DRAM-sharded pipeline.

    ``workers_per_bank`` cores next to each DRAM bank split that bank's weight columns, read their activation
    rows straight from the width-sharded L1 input, and stream the weight on both data-movement RISCs. The
    single-reader program cannot retire tiles as fast as one bank delivers them at M = one tile.

    Worker counts and K-block widths were swept per projection on a P150 (traced back-to-back matmuls at batch 1,
    output PCC unchanged, inputs on the layouts the model uses); the chosen points read DRAM at 80-93% of 512 GB/s,
    against 47-56% for the single-reader default:
      QKV 3072x9216 BFP8 (6 workers, block 1): 81.5 -> 62.7 us; WO 3072x3072 BFP8 (2, 2): 27.5 -> 24.8 us;
      fused W1/W3/SwiGLU 3072x16384 BFP4 (8, 1): 61.2 us; W2 8192x3072 BFP8 (3, 2): 65.9 -> 56.8 us;
      LM head 3072x8192 BFP8 split (8, 2): 94.7 -> 56.9 us.
    The pipeline only engages for an L1 width-sharded output and no fused activation or bias; the decode
    projections here all qualify. The output is width-sharded ``per_core_n`` tiles per core (default: the
    activation shard count's share of N); the requested memory config's shard spec does not change it.
    ``fuse_swiglu``: the weight's column tiles are gate / up pairs and the output is SiLU(gate) * up, N / 2 wide.
    """
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=per_core_n if per_core_n is not None else math.ceil(n / (TILE_SIZE * in0_cores)),
        fused_activation=None,
        num_workers_per_dram_bank=workers_per_bank,
        use_multicore_pipeline=True,
        fuse_swiglu=fuse_swiglu,
    )


def _prefill_minimal_matmul_config(
    mesh_device: ttnn.MeshDevice, m_block: int, k_block: int, n_block: int, subblock_w: int = 4
) -> ttnn.MinimalMatmulConfig:
    """minimal_matmul blocking for prefills above 128 rows, with 1 x 4 output subblocks.

    The modules' defaults use 8 x 8 x 8 blocks and 1 x 1 subblocks, so every output tile is packed on its own. Swept
    per projection on a P150 for the batch-32 prefill shapes (32 users x 128 rows; output PCC unchanged), per call:
      QKV 2048 x 3072 x 9216 (8, 4, 8): 588 -> 529 us;  WO 1024 x 3072 x 3072 (8, 4, 8): 124 -> 117 us;
      fused W1/W3 SwiGLU 4096 x 3072 x 16384 (8, 8, 16): 1675 -> 944 us;  W2 4096 x 8192 x 3072 (16, 8, 8): 878 -> 800 us.
    1 x 4 subblocks stay within Dest with or without fp32 accumulation and are within 1% of the best subblock at HiFi2.
    Not taken: QKV 1 x 8 (847 us) and W1/W3 8 x 4 x 8 with 1 x 8 (816 us) - with them token accuracy fell to
    90.8 / 98.4% (smaller K blocks also mean more partial-sum accumulation rounds in the output format).
    """
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=1,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
    )


def _decode_residual_grid(mesh_device: ttnn.MeshDevice, dim: int) -> ttnn.CoreGrid:
    """Cores of the decode residual: the first cores of row 0 of the compute grid.

    The DRAM-sharded decode matmuls write their output width-sharded over the first ``N / per_core_N`` cores in
    row-major order of the compute grid. With the residual on the first cores of row 0, WO and W2
    (``per_core_N = dim / 32 / cores``) write straight into the residual layout and need no reshard; on the previous
    8x4 grid each of them was followed by one (1.2 us plus a dispatch gap each). The decode norms and the matmuls
    that read the normed residual (QKV, W1/W3, LM head) take this layout as their input. Picks the most cores that
    fit in one row and split ``dim`` into whole tiles: 12 cores x 8 tiles for dim 3072 on the 13-wide P150 grid.
    RMSNorm on these 12 cores takes the same time as on 8x4 (5.4 us).
    """
    width = mesh_device.compute_with_storage_grid_size().x
    tiles = dim // TILE_SIZE
    return ttnn.CoreGrid(y=1, x=max(n for n in range(1, width + 1) if tiles % n == 0))


def _decode_residual_memcfg(mesh_device: ttnn.MeshDevice, dim: int, max_batch_size: int) -> ttnn.MemoryConfig:
    """L1 width-sharded decode residual on :func:`_decode_residual_grid`.

    Attention1D and MLP1D end decode on this layout (their ``decode_residual_memcfg``) and the RMSNorm1D decode
    inputs and outputs use it too, so a residual kept here needs no DRAM round trip: the residual adds run sharded in
    L1 and the norms take their input as is.
    """
    grid = _decode_residual_grid(mesh_device, dim)
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE), dim // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_width_sharded_memcfg(mesh_device: ttnn.MeshDevice, k: int, n: int) -> ttnn.MemoryConfig:
    """Weight layout of the DRAM-sharded decode matmuls: N split evenly over the DRAM banks, full K per bank."""
    banks = mesh_device.dram_grid_size().x
    padded_n = math.ceil(n / (TILE_SIZE * banks)) * TILE_SIZE * banks
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded_n // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _lazy_decode_weight(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """DRAM width-sharded copy of a projection weight for the decode matmuls (see attention.py / mlp.py)."""
    k, n = tensor.shape[-2], tensor.shape[-1]
    return _lazy_placed(
        tensor,
        dtype=dtype,
        mesh_device=mesh_device,
        memory_config=_dram_width_sharded_memcfg(mesh_device, k, n),
        cache=cache,
    )


def _lazy_placed(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    memory_config: ttnn.MemoryConfig,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Fully placed LazyWeight for a model-owned weight that no TTTv2 module resolves."""
    return LazyWeight(
        source=tensor,
        dtype=dtype,
        device=mesh_device,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_dir_weight_name=cache,
    )


def _prefill_128_1d_config(
    *,
    n: int,
    num_cores: int,
    grid: tuple[int, int],
    in0_block_w: int,
    fused_activation=None,
    dram_stream: tuple[int, int, int] | None = None,
):
    """1D-multicast matmul for the 128-row prefill bucket: the activation is multicast, N split over cores.

    At 128 rows the projections are weight-read bound. Swept on a P150 against DRAM-interleaved weights (traced
    back-to-back matmuls, output PCC unchanged), the chosen configs take QKV 262 -> 94 us, WO 102 -> 43 us,
    W1/W3 200 -> 63 us and W2 244 -> 98 us against the 2D configs on the DRAM-sharded decode weights.

    ``dram_stream`` = (cores per DRAM bank, activation distributors, K tiles per activation multicast) runs the
    multi-row DRAM-streamed program instead (``dram_stream_workers_per_bank``), which needs the DRAM width-sharded
    decode copy of the weight; ``in0_block_w`` is then the K rows per streamed weight block.
    """
    m_tiles = PREFILL_1D_SEQ_LEN // TILE_SIZE
    per_core_n = n // TILE_SIZE // num_cores
    out_subblock_w = max(w for w in (4, 3, 2, 1) if per_core_n % w == 0 and w * m_tiles <= 8)
    assert num_cores <= grid[0] * grid[1] and n == per_core_n * TILE_SIZE * num_cores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
        **(
            dict(
                dram_stream_workers_per_bank=dram_stream[0],
                dram_stream_distributors=dram_stream[1],
                dram_stream_chunk=dram_stream[2],
            )
            if dram_stream
            else {}
        ),
    )


def _prefill_128_residual_memcfg(mesh_device: ttnn.MeshDevice, dim: int) -> ttnn.MemoryConfig:
    """L1 block-sharded layout of the 128-row residual stream and its norms: one tile row of the 128 per row of
    cores, the hidden size split over the decode residual's core count along it (12 x 4 cores, 32 x 256 each)."""
    cols = _decode_residual_grid(mesh_device, dim).num_cores
    rows = PREFILL_1D_SEQ_LEN // TILE_SIZE
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, dim // cols),
        grid,
        ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _prefill_rms_norm(
    norm: RMSNorm1D,
    x: ttnn.Tensor,
    long_prefill_dtype: ttnn.DataType = ttnn.bfloat16,
    delta: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """Prefill RMSNorm; the 128-row bucket runs block-sharded in L1 and returns L1-interleaved, longer prefills write
    ``long_prefill_dtype``.

    ``RMSNorm1D`` prefill normalizes the DRAM-interleaved tensor with no program config, which splits the work
    by tile rows: 4 cores for 128 rows, 50 us per norm on a P150. Here each tile row of the 128 goes to its own row
    of cores and the hidden size is split over the residual's core count along it (12 x 4 cores, 32 x 256 each):
    8.3 us including the reshards in and out, against 12.2 us width-sharded over 32 cores. The 128-row layer keeps
    its activations in L1 (see ``Phi35MiniDecoderLayer.prefill_forward``).

    ``delta`` (128 rows, in the residual's block-sharded layout like ``x``): normalize ``x + delta`` and leave the sum
    in ``x``, the residual add folded into the norm (``Phi35MiniDecoderLayer.prefill_forward_128_deferred``).
    """
    if x.shape[-2] != PREFILL_1D_SEQ_LEN:
        norm.load_device_weights()
        cfg = norm.config
        if cfg.mesh_device.get_num_devices() != 1:
            return norm.prefill_forward(x)
        return ttnn.rms_norm(
            x,
            epsilon=cfg.eps,
            weight=norm.weight,
            compute_kernel_config=cfg.compute_kernel_config,
            dtype=long_prefill_dtype,
        )
    norm.load_device_weights()
    cfg = norm.config
    dim = x.shape[-1]
    memcfg = _prefill_128_residual_memcfg(cfg.mesh_device, dim)
    cols = _decode_residual_grid(cfg.mesh_device, dim).num_cores
    rows = PREFILL_1D_SEQ_LEN // TILE_SIZE
    block_w = dim // cols // TILE_SIZE
    # The 128-row layer keeps its residual stream in this layout, so the input is usually already sharded.
    x_sharded = x if x.memory_config() == memcfg else ttnn.to_memory_config(x, memcfg)
    if delta is not None and x_sharded is not x:
        raise ValueError("the fused residual add needs x in the residual's block-sharded layout")
    add_kwargs = dict(residual_input_tensor=delta, write_pre_add_sum=True) if delta is not None else {}
    y = ttnn.rms_norm(
        x_sharded,
        epsilon=cfg.eps,
        weight=norm.weight,
        **add_kwargs,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(cols, rows),
            subblock_w=max(w for w in (4, 3, 2, 1) if block_w % w == 0),
            block_h=1,
            block_w=block_w,
            inplace=False,
        ),
        # The sharded norm writes its output L1 interleaved itself (the 1D-multicast matmuls read that), instead of a
        # sharded output and a sharded_to_interleaved op: 8.4 -> 7.6 us on a P150, one op less.
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=cfg.compute_kernel_config,
    )
    if x_sharded is not x:
        ttnn.deallocate(x_sharded)
    return y


def _interleave_swiglu_tiles(w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """[K, N] gate and up weights -> [K, 2N] with column tile 2p from gate tile p and 2p+1 from up tile p.

    The column layout ``ttnn.experimental.minimal_matmul(..., fuse_swiglu=True)`` reads: it computes
    silu(gate) * up per tile pair and writes N columns.
    """
    k, n = w_gate.shape[-2], w_gate.shape[-1]
    assert w_up.shape == w_gate.shape and n % TILE_SIZE == 0
    gate = w_gate.reshape(k, n // TILE_SIZE, 1, TILE_SIZE)
    up = w_up.reshape(k, n // TILE_SIZE, 1, TILE_SIZE)
    return torch.cat([gate, up], dim=2).reshape(k, 2 * n)


def _slice_last_token_tile(x: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
    """Slice the 32-row tile containing ``last_token_idx`` from ``[1, 1, S, W]``.

    Width-sharded LM matmul M tile rows must equal ``LMHead1D`` program-config tile rows.
    """
    floor = (last_token_idx // 32) * 32
    return ttnn.slice(x, (0, 0, floor, 0), (1, 1, floor + 32, x.shape[-1]))


def _decode_norm_configs(
    mesh_device: ttnn.MeshDevice, dim: int, max_batch_size: int
) -> dict[str, ttnn.LayerNormShardedMultiCoreProgramConfig | ttnn.MemoryConfig]:
    """RMSNorm1D decode program / memory configs for every decode norm: input and output on the residual layout.

    The norm outputs feed QKV, W1/W3 and the LM head, whose DRAM-sharded matmuls are built for activations sharded
    on the same cores; a norm output on another grid silently corrupts decode activations (observed on the 7B / 14B
    ports).
    """
    grid = _decode_residual_grid(mesh_device, dim)
    tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
    return dict(
        decode_program_config=_create_sharded_norm_program_config(dim, grid, tile_padded_batch_rows, TILE_SIZE),
        decode_memory_config=_decode_residual_memcfg(mesh_device, dim, max_batch_size),
    )


def _decode_rms_norm_add(norm: RMSNorm1D, x: ttnn.Tensor, delta: ttnn.Tensor) -> ttnn.Tensor:
    """Decode RMSNorm of ``x + delta`` that also leaves ``x + delta`` in ``x`` (the residual stream), one op instead
    of a residual add and the norm. ``x`` and ``delta`` are on the norm's decode layout (the residual's).
    P150, 32 x 3072 on 12 cores, traced: 9.4 -> 7.0 us per norm; the sum is bit-identical to the add."""
    norm.load_device_weights()
    cfg = norm.config
    return ttnn.rms_norm(
        x,
        epsilon=cfg.eps,
        weight=norm.weight,
        residual_input_tensor=delta,
        program_config=cfg.decode_program_config,
        memory_config=cfg.decode_memory_config,
        compute_kernel_config=cfg.compute_kernel_config,
        write_pre_add_sum=True,
    )


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )


@dataclass
class _Phi35MiniTuning:
    """L1 / cutoff tuning resolved at build time.

    Precision-vs-fidelity knobs live on :class:`Phi35MiniPrecisionConfig`. This dataclass only carries
    non-mode-dependent L1 footprint controls: MLP prefill cutoff and the optional W1→DRAM spill on
    decode.
    """

    mlp_prefill_len_cutoff: int | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    # ttnn.experimental.minimal_matmul for the QKV + W2 prefill matmuls above seq_len > 128 (TTTv1 parity).
    prefill_minimal_matmul: bool = True


def _resolve_phi35_mini_tuning(*, num_dev: int, max_batch_size: int) -> _Phi35MiniTuning:
    """Pick L1 tuning knobs for Phi-3.5-mini on one P150.

    ``mlp_prefill_len_cutoff=None`` keeps the ``MLP1D`` default; the 3.8B model's MLP is small enough that
    the Phi-4 L1 cutoff is not needed.
    """
    del max_batch_size
    t = _Phi35MiniTuning(mlp_prefill_len_cutoff=None, mlp_decode_spill_w1_to_dram=False)
    logger.info(
        f"L1 tuning for Phi-3.5-mini on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, "
        f"decode_spill_w1_to_dram={t.mlp_decode_spill_w1_to_dram}, "
        f"prefill_minimal_matmul={t.prefill_minimal_matmul}"
    )
    return t


@dataclass
class Phi35MiniDecoderLayerConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Phi35MiniAttentionConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: Phi35MiniMLPConfig


def _build_decoder_layer(
    *,
    idx: int,
    weights: Phi35MiniLayerWeights,
    mcfg: Phi35MiniModelParameters,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    precision: Phi35MiniPrecisionConfig,
    paged_cfg: Phi35MiniPagedAttentionConfig,
    cache_path: Path | None,
    tuning: _Phi35MiniTuning,
) -> Phi35MiniDecoderLayerConfig:
    """Build one decoder-layer config from provider-converted tensors."""
    prefix = f"layer{idx}"

    lazy_wqkv = _lazy(
        weights.wqkv,
        dtype=precision.wqkv_dtype,
        cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None,
    )
    lazy_wo = _lazy(
        weights.wo,
        dtype=precision.wo_dtype,
        cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None,
    )

    qkv_size = mcfg.head_dim * (mcfg.n_heads + 2 * mcfg.n_kv_heads)
    residual_cores = _decode_residual_grid(mesh_device, mcfg.dim).num_cores
    attention_config = Phi35MiniAttentionConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        # The prefill rotary op's transformation matrix in L1 (Attention1D's default is DRAM): the 128-row rope reads it
        # with cos / sin (also L1, see prepare_prefill_rot_mats) in 11.1 us instead of 12.1 us on a P150.
        transformation_mat_prefill=ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        # Prefill reads interleaved weights (1D-multicast matmuls); decode reads the DRAM-sharded copies.
        wqkv_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        wo_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        wqkv_decode=_lazy_decode_weight(
            weights.wqkv,
            dtype=precision.wqkv_dtype,
            mesh_device=mesh_device,
            cache=(cache_path / "attn", f"{prefix}_wqkv_decode") if cache_path else None,
        ),
        wo_decode=_lazy_decode_weight(
            weights.wo,
            dtype=precision.wo_dtype,
            mesh_device=mesh_device,
            cache=(cache_path / "attn", f"{prefix}_wo_decode") if cache_path else None,
        ),
        # Single-user 128-row QKV on the multi-row DRAM-streamed program over the DRAM width-sharded decode copy (12
        # cores per bank, one activation distributor, 4 K tiles per multicast). fp32 accumulation: 16-bit Dest takes 8
        # K tiles at a time into a Float32 L1 accumulator, since a core's 4 x 3 output tiles exceed the fp32 Dest. P150,
        # traced, 80.5 -> 66.0 us (pcc 0.999926 against the 1D matmul). Whole-K 16-bit Dest accumulation ran 65.2 us but
        # cost acc128 1.0 / 0.4 points (91.8 / 99.2 -> 90.8 / 98.8).
        prefill_128_xqkv_prg_config=_prefill_128_1d_config(n=qkv_size, num_cores=72, grid=(12, 6), in0_block_w=2),
        prefill_128_qkv_stream_prg_config=_prefill_128_1d_config(
            n=qkv_size, num_cores=72, grid=(12, 6), in0_block_w=1, dram_stream=(12, 1, 4)
        ),
        prefill_128_qkv_compute_kernel_cfg=_HIFI2_FP32_COMPUTE_KERNEL_CFG,
        prefill_128_wo_prg_config=_prefill_128_1d_config(n=mcfg.dim, num_cores=48, grid=(12, 4), in0_block_w=8),
        # Single-user 128-row WO on the multi-row DRAM-streamed program over wo_decode (6 cores per bank, 2 K rows a
        # weight block, two activation distributors taking turns, 8 K tiles per multicast), fp32 Dest: 33.0 -> 26.4
        # us, bit-identical to the 1D matmul.
        prefill_128_wo_stream_prg_config=_prefill_128_1d_config(
            n=mcfg.dim, num_cores=48, grid=(12, 4), in0_block_w=2, dram_stream=(6, 2, 8)
        ),
        prefill_128_wo_compute_kernel_cfg=_HIFI2_FP32_COMPUTE_KERNEL_CFG,
        # 128-row single-user SDPA: 32-row Q x 128-row K chunks with approximate exp on a bfloat8_b Q, 12.5 us (traced,
        # P150, 32 heads, fp32 accumulation; pcc vs torch 0.99981) against 19.6 us for 32 x 64 with exact exp and
        # 27.7 us for Attention1D's 64 x 64. acc128 91.8 / 99.4%.
        prefill_128_sdpa_prg_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=True
        ),
        # Batched prefill QKV output in bfloat8_b (the KV cache dtype). 32 users x 512 rows, teacher forcing: top-1 /
        # top-5 91.8 / 99.4% against 91.2 / 99.0% with bfloat16 and Attention1D's casts.
        prefill_batched_qkv_dtype=ttnn.bfloat8_b,
        # WO with the residual add fused in, 12 x 4 x 8 blocks (16 x 4 x 8 overflows the L1 left in the model).
        # P150, 4096 rows, LoFi: 321 us against 378 us with 8 x 4 x 8 (HiFi2 and a separate add: 559 us).
        prefill_batched_wo_residual_minimal_matmul_config=_prefill_minimal_matmul_config(mesh_device, 12, 4, 8),
        # LoFi for the batched-prefill WO (compute-bound at 32 users x 128 rows). With W2 LoFi as well: TTFT 4.1 ->
        # 3.9 ms/user at batch 32; 32-user teacher forcing 92.0 / 99.2%. QKV LoFi too brought top-5 to 98.6%.
        prefill_batched_wo_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
        # Q enters the batched-prefill SDPA in bfloat8_b (SDPA's output follows, so the head concat moves half the bytes
        # and WO reads bfloat8_b): TTFT 3.7 -> 3.6 ms/user at batch 32; 32-user teacher forcing 92.6 / 99.4%.
        prefill_batched_q_dtype=ttnn.bfloat8_b,
        # Batched prefill QKV output / heads / rotated Q, K / SDPA output in L1 up to 32 users x 128 rows: TTFT 3.6 ->
        # 3.3 ms/user at batch 32.
        prefill_batched_l1_max_rows=4096,
        # 128 x 128 chunks (one per user and head), approximate exp, HiFi4 with fp32 accumulation, on a bfloat8_b Q with
        # Q / K / V in L1: P150, 32 users x 32 heads x 128 rows, 154 us (128 x 64) -> 102 us. HiFi2 (84 us) brought
        # 32-user teacher forcing to 90.2 / 99.0% (HiFi4: 92.2 / 99.2%). Precise exp at 128 x 64 was slower than 64 x 64.
        prefill_batched_sdpa_prg_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(mesh_device.compute_with_storage_grid_size().x,
                                            mesh_device.compute_with_storage_grid_size().y),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=True,
        ),
        prefill_batched_sdpa_compute_kernel_cfg=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
        prefill_wo_minimal_matmul=True,
        prefill_xqkv_minimal_matmul_config=lambda seq_len: _prefill_minimal_matmul_config(mesh_device, 8, 4, 8),
        prefill_wo_minimal_matmul_config=lambda seq_len: _prefill_minimal_matmul_config(mesh_device, 8, 4, 8),
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        n_heads=mcfg.n_heads,
        n_kv_heads=mcfg.n_kv_heads,
        head_dim=mcfg.head_dim,
        max_batch_size=mcfg.max_batch_size,
        max_seq_len=mcfg.max_seq_len,
        q_norm_config=None,
        k_norm_config=None,
        wqkv_bias=None,
        use_vllm_paged_kv_cache=True,
        paged_attention_config=paged_cfg,
        kv_cache=None,
        kv_cache_dtype=precision.kv_cache_dtype,
        # Many users: SDPA over the whole grid, reading the ND-sharded cache (_kv_cache_memory_config). Off the 8x8
        # grid it needs Q in DRAM rather than sharded on the 8-wide head grid.
        **(
            dict(
                # Approximate exp: at 32 users each core runs 8 heads one after another and the per-head compute,
                # not the KV read, sets the time. P150: 99.9 -> 93.4 us at position 85, 238 -> 206 us at 300
                # (worst per-user pcc 0.99964 vs torch over 160 positions).
                decode_sdpa_prg_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=0,
                    k_chunk_size=0,
                    exp_approx_mode=True,
                    max_cores_per_head_batch=16,
                ),
                decode_sdpa_q_in_dram=True,
            )
            if _kv_cache_nd_sharded(mcfg.max_batch_size, mcfg.n_kv_heads)
            # Few users: 8x8 grid with one core per head (no cross-core tree reduction at decode lengths). P150, batch
            # 1, demo decode: 8.55 -> 8.49 ms a token. Approximate exp as well ran 8.41 ms but cost token accuracy
            # (91.6 / 99.2 -> 91.0 / 98.8%).
            else dict(
                decode_sdpa_prg_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    exp_approx_mode=False,
                    q_chunk_size=0,
                    k_chunk_size=0,
                    max_cores_per_head_batch=1,
                )
            )
        ),
        use_qk_fused=True,
        decode_transformation_core_grid=_DECODE_QK_CORE_GRID,
        decode_residual_memcfg=_decode_residual_memcfg(mesh_device, mcfg.dim, mcfg.max_batch_size),
        decode_input_memcfg=_decode_residual_memcfg(mesh_device, mcfg.dim, mcfg.max_batch_size),
        prefill_qkv_minimal_matmul=tuning.prefill_minimal_matmul,
        # One head per output core: the fused-QK head split (Phi35MiniAttention) takes the sharded output only
        # when every shard divides one head.
        decode_xqkv_prg_config=_decode_dram_matmul_config(
            k=mcfg.dim,
            n=qkv_size,
            in0_cores=residual_cores,
            workers_per_bank=6,
            in0_block_w=1,
            per_core_n=mcfg.head_dim // TILE_SIZE,
        ),
        # Output straight onto the residual cores (see _decode_residual_grid).
        decode_attn_output_prg_config=_decode_dram_matmul_config(
            k=mcfg.n_heads * mcfg.head_dim,
            n=mcfg.dim,
            in0_cores=mcfg.n_heads,
            workers_per_bank=2,
            in0_block_w=2,
            per_core_n=mcfg.dim // TILE_SIZE // residual_cores,
        ),
        li_qkv_decode_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
        sdpa_decode_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
        li_o_decode_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
        li_qkv_prefill_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
        sdpa_prefill_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
        li_o_prefill_compute_kernel_cfg=precision.attention_compute_kernel_cfg,
    )

    padded_hidden = get_padded_hidden_dim(mcfg.hidden_dim, num_dev, TILE_SIZE)
    # The fused W1/W3/SwiGLU decode matmul writes 2 tiles per core (128 cores); W2 reads that layout as is.
    mlp_w13_per_core_n = 2 if not _S11_GRID12 else 4  # S11: 256 output tiles on 64 cores (128 > 120 on a 12 x 10 Galaxy chip)
    mlp_w2_in0_cores = padded_hidden // num_dev // TILE_SIZE // mlp_w13_per_core_n
    # [W1 | W3] as gate / up column tile pairs, for the fused SwiGLU matmuls of prefill and decode.
    w13_swiglu = _interleave_swiglu_tiles(weights.w1, weights.w3)
    mlp_config = Phi35MiniMLPConfig(
        w1=_lazy(
            weights.w1,
            dtype=precision.mlp_w1_w3_dtype,
            cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None,
        ),
        w2=_lazy(
            weights.w2,
            dtype=precision.mlp_w2_dtype,
            cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None,
        ),
        w3=_lazy(
            weights.w3,
            dtype=precision.mlp_w1_w3_dtype,
            cache=(cache_path / "mlp", f"{prefix}_w3") if cache_path else None,
        ),
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        max_batch_size=mcfg.max_batch_size,
        prefill_len_cutoff=tuning.mlp_prefill_len_cutoff,
        w1_w3_dtype=precision.mlp_w1_w3_dtype,
        w2_dtype=precision.mlp_w2_dtype,
        ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
        decode_ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
        ff2_compute_kernel_cfg=precision.mlp_ff2_compute_kernel_cfg,
        decode_ff2_compute_kernel_cfg=precision.mlp_ff2_compute_kernel_cfg,
        decode_spill_w1_to_dram_before_w3=tuning.mlp_decode_spill_w1_to_dram,
        decode_residual_memcfg=_decode_residual_memcfg(mesh_device, mcfg.dim, mcfg.max_batch_size),
        decode_input_memcfg=_decode_residual_memcfg(mesh_device, mcfg.dim, mcfg.max_batch_size),
        w1_w3_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        w2_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        w13_swiglu_decode=_lazy_decode_weight(
            w13_swiglu,
            dtype=precision.mlp_w1_w3_dtype,
            mesh_device=mesh_device,
            cache=(cache_path / "mlp", f"{prefix}_w13_swiglu_decode") if cache_path else None,
        ),
        w2_decode=_lazy_decode_weight(
            weights.w2,
            dtype=precision.mlp_w2_dtype,
            mesh_device=mesh_device,
            cache=(cache_path / "mlp", f"{prefix}_w2_decode") if cache_path else None,
        ),
        # W1 / W3 write bfloat8_b for the SiLU-multiply at 128 rows: TTFT 13.4 -> 13.2 ms at batch 1; 128-row
        # prefill teacher forcing (acc128) 91.8 / 99.0% against 91.2 / 99.0% with bfloat16.
        prefill_128_w1_w3_dtype=ttnn.bfloat8_b,
        # 128-row W1 / W3 / SiLU-multiply as one 1D-multicast matmul with fuse_swiglu (13 x 10 grid, 4 output tiles a
        # core, i.e. 2 gate / up pairs): P150, traced, 114.8 -> 103.6 us (pcc vs torch 0.99977 -> 0.99982). On the
        # DRAM width-sharded decode copy, which the matmul reads one K row of a core's 4 tiles per NOC request instead
        # of one request per 576-byte bfloat4_b tile: 103.9 -> 78.7 us with 4 K tiles a block (output bit-identical to
        # the interleaved weight at the same config).
        prefill_128_w13_swiglu_prg_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            # phi-3.5-mini S11 (Galaxy Blackhole, 12 x 10 grid): 13 x 10 / 16-per-bank stream need 130 / 128 cores.
            compute_with_storage_grid_size=(13, 10) if not _S11_GRID12 else (8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=PREFILL_1D_SEQ_LEN // TILE_SIZE,
            per_core_N=4 if not _S11_GRID12 else 8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
            fuse_swiglu=True,
            # Multi-row DRAM-streamed program: 16 cores per bank, one activation distributor, 4 K tiles per
            # multicast, the whole K in 16-bit Dest (ff1_3_compute_kernel_cfg): 79.0 -> 65.8 us (pcc 0.99919 against
            # the 1D matmul). fp32 accumulation through a Float32 L1 accumulator ran 67.0 us and scored acc128 no
            # better (90.8 / 98.6 against 91.8 / 99.0).
            dram_stream_workers_per_bank=16 if not _S11_GRID12 else 0,
            dram_stream_distributors=1,
            dram_stream_chunk=4,
        ),
        prefill_128_w1_w3_prg_config=_prefill_128_1d_config(
            n=padded_hidden // num_dev, num_cores=64, grid=(8, 8), in0_block_w=4
        ),
        # 128-row W2 on the multi-row DRAM-streamed program over w2_decode (6 cores per bank, two activation
        # distributors taking turns, 8 K tiles per multicast), fp32 Dest over K = 8192: 81.5 -> 58 us (pcc 0.99990
        # against the 1D matmul).
        prefill_128_w2_prg_config=_prefill_128_1d_config(n=mcfg.dim, num_cores=96, grid=(13, 10) if not _S11_GRID12 else (12, 10), in0_block_w=8),
        prefill_128_w2_stream_prg_config=_prefill_128_1d_config(
            n=mcfg.dim, num_cores=96, grid=(13, 10) if not _S11_GRID12 else (12, 10), in0_block_w=1, dram_stream=(6, 2, 8)
        ),
        prefill_128_w2_compute_kernel_cfg=_HIFI2_FP32_COMPUTE_KERNEL_CFG,
        w13_swiglu_prefill=_lazy_placed(
            w13_swiglu,
            dtype=precision.mlp_w1_w3_dtype,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache=(cache_path / "mlp", f"{prefix}_w13_swiglu") if cache_path else None,
        ),
        prefill_swiglu_minimal_matmul_config=_prefill_minimal_matmul_config(mesh_device, 8, 8, 16),
        # One matmul over the gate / up pairs with SiLU(gate) * up applied in Dest (fuse_swiglu), instead of W1, W3
        # and a SiLU-multiply that reads both back. P150, batch 1, traced: 61.2 us against 73.2 us for the three ops.
        # 8 workers per bank leave each core 4 gate / up pairs, so the SwiGLU tail after the last weight block is short.
        decode_w13_swiglu_prg_config=_decode_dram_matmul_config(
            k=mcfg.dim,
            n=2 * padded_hidden // num_dev,
            in0_cores=residual_cores,
            workers_per_bank=8 if not _S11_GRID12 else 4,
            in0_block_w=1,
            per_core_n=mlp_w13_per_core_n,
            fuse_swiglu=True,
        ),
        # Input: the fused matmul's output (128 cores x 2 tiles), so no reshard before W2 (+1.0 us of matmul time
        # against an 8x4 input, -1.2 us of reshard and a dispatch gap). Output straight onto the residual cores.
        decode_w2_prg_config=_decode_dram_matmul_config(
            k=padded_hidden // num_dev,
            n=mcfg.dim,
            in0_cores=mlp_w2_in0_cores,
            workers_per_bank=3,
            in0_block_w=2,
            per_core_n=mcfg.dim // TILE_SIZE // residual_cores,
        ),
        prefill_w2_minimal_matmul=tuning.prefill_minimal_matmul,
        prefill_w2_minimal_matmul_config=lambda seq_len: _prefill_minimal_matmul_config(mesh_device, 16, 8, 8),
        # W2 with the residual add fused in: 16 x 8 x 4 blocks (16 x 8 x 8 plus the residual block overflows L1).
        prefill_w2_residual_minimal_matmul_config=_prefill_minimal_matmul_config(mesh_device, 16, 8, 4),
        # LoFi above 128 rows (see prefill_batched_wo_compute_kernel_cfg).
        prefill_w2_residual_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
    )

    norm_decode_configs = _decode_norm_configs(mesh_device, mcfg.dim, mcfg.max_batch_size)

    def _build_norm(weight: torch.Tensor, name: str, **extra: Any) -> RMSNorm1DConfig:
        lw = _lazy(
            weight,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", f"{prefix}_{name}") if cache_path else None,
        )
        return RMSNorm1DConfig(
            weight=lw,
            mesh_device=mesh_device,
            eps=mcfg.rms_norm_eps,
            max_batch_size=mcfg.max_batch_size,
            tt_ccl=tt_ccl,
            **extra,
        )

    return Phi35MiniDecoderLayerConfig(
        attention_norm_config=_build_norm(weights.attention_norm, "pre_attn", **norm_decode_configs),
        attention_config=attention_config,
        ff_norm_config=_build_norm(weights.ff_norm, "post_attn", **norm_decode_configs),
        mlp_config=mlp_config,
    )


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    lm_head_weight: torch.Tensor,
    mcfg: Phi35MiniModelParameters,
    lm_head_dtype: ttnn.DataType,
    cache_path: Path | None,
) -> LMHead1DConfig:
    """Build the vocab-sharded LM head with DRAM-matmul program configs.

    LM head DRAM matmul is sized for decode batch tiles (``max_batch_size``). Prefill logits
    use a single 32-row tile via ``post_process_prefill_output`` / :func:`_slice_last_token_tile`.
    """
    lm_splits, lm_split_sizes, lm_weights_memcfgs = weight_utils.build_lm_head_lazy_weights(
        mesh_device,
        lm_head_weight,
        dim=mcfg.dim,
        vocab_size=mcfg.vocab_size,
        dtype=lm_head_dtype,
        cache_dir=cache_path / "lm_head" if cache_path else None,
    )
    lm_head_core_grid = _decode_residual_grid(mesh_device, mcfg.dim)
    tile = ttnn.TILE_SIZE
    tile_padded_batch_rows = tile * math.ceil(mcfg.max_batch_size / tile)
    lm_prog_configs = [
        _decode_dram_matmul_config(
            k=mcfg.dim, n=ss, in0_cores=lm_head_core_grid.num_cores, workers_per_bank=8, in0_block_w=2
        )
        for ss in lm_split_sizes
    ]
    lm_input_memcfg = ttnn.create_sharded_memory_config(
        (tile_padded_batch_rows, _nearest_32(mcfg.dim // lm_head_core_grid.num_cores)),
        lm_head_core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return LMHead1DConfig(
        output_weights=lm_splits,
        mesh_device=mesh_device,
        dim=mcfg.dim,
        max_batch_size=mcfg.max_batch_size,
        lm_head_dtype=lm_head_dtype,
        program_configs=lm_prog_configs,
        input_memcfg=lm_input_memcfg,
        weights_memcfgs=lm_weights_memcfgs,
    )


def build_phi35_mini_transformer_config(
    *,
    mesh_device: ttnn.MeshDevice,
    params: Phi35MiniModelParameters,
    weights: Phi35MiniWeights,
    n_layers: int,
    precision: Phi35MiniPrecisionConfig,
    cache_path: Path | None,
    paged_attention_config: Phi35MiniPagedAttentionConfig,
) -> Phi35MiniTransformerConfig:
    """Build the TT tensor graph from provider-neutral dimensions and converted tensors."""

    num_devices = mesh_device.get_num_devices()
    if params.n_heads % num_devices or params.n_kv_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({params.n_heads}/{params.n_kv_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if len(weights.layers) != n_layers:
        raise ValueError(f"Expected {n_layers} decoder layer weight sets, got {len(weights.layers)}")

    tt_ccl = get_tt_ccl(mesh_device) if num_devices > 1 else None
    topology = default_topology(mesh_device)
    tuning = _resolve_phi35_mini_tuning(num_dev=num_devices, max_batch_size=params.max_batch_size)
    block_configs = [
        _build_decoder_layer(
            idx=index,
            weights=weights.layers[index],
            mcfg=params,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            num_dev=num_devices,
            precision=precision,
            paged_cfg=paged_attention_config,
            cache_path=cache_path,
            tuning=tuning,
        )
        for index in range(n_layers)
    ]
    return Phi35MiniTransformerConfig(
        n_layers=n_layers,
        vocab_size=params.vocab_size,
        max_batch_size=params.max_batch_size,
        max_seq_len=params.max_seq_len,
        dim=params.dim,
        num_devices=num_devices,
        mesh_device=mesh_device,
        embedding_config=Embedding1DConfig(
            weights=_lazy(
                weights.embedding,
                dtype=ttnn.bfloat16,
                cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
            ),
            mesh_device=mesh_device,
            embed_scale=1.0,
        ),
        rope_config=Rope1DConfig(
            cos_matrix=_lazy(
                weights.rope_cos,
                dtype=ttnn.bfloat16,
                cache=(cache_path / "rope", "cos") if cache_path else None,
            ),
            sin_matrix=_lazy(
                weights.rope_sin,
                dtype=ttnn.bfloat16,
                cache=(cache_path / "rope", "sin") if cache_path else None,
            ),
            max_batch_size=params.max_batch_size,
            head_dim=params.head_dim,
            device=mesh_device,
            # Fused Q/K rope + fused K/V paged cache update in decode (one op each instead of two).
            use_qk_fused=True,
            core_grid=_DECODE_QK_CORE_GRID,
        ),
        block_configs=block_configs,
        norm_config=RMSNorm1DConfig(
            weight=_lazy(
                weights.final_norm,
                dtype=ttnn.bfloat16,
                cache=(cache_path / "norm", "final") if cache_path else None,
            ),
            mesh_device=mesh_device,
            eps=params.rms_norm_eps,
            max_batch_size=params.max_batch_size,
            tt_ccl=tt_ccl,
            **_decode_norm_configs(mesh_device, params.dim, params.max_batch_size),
        ),
        lm_head_config=_build_lm_head(
            mesh_device=mesh_device,
            lm_head_weight=weights.lm_head,
            mcfg=params,
            lm_head_dtype=precision.lm_head_dtype,
            cache_path=cache_path,
        ),
        sampling_config=Sampling1DConfig(
            vocab_size=params.vocab_size,
            valid_vocab_size=params.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            max_batch_size=_nearest_32(params.max_batch_size),
            allow_force_argmax=False,
            pad_to_power_of_2=True,
        ),
        decode_residual_memcfg=_decode_residual_memcfg(mesh_device, params.dim, params.max_batch_size),
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtypes=[None] * n_layers,
        tt_ccl=tt_ccl,
        cache_path=str(cache_path) if cache_path is not None else None,
    )


class Phi35MiniDecoderLayer(LightweightModule):
    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        attention: Phi35MiniAttention,
        post_attention_layernorm: RMSNorm1D,
        mlp: Phi35MiniMLP,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.attention_norm = input_layernorm
        self.attention = attention
        self.ff_norm = post_attention_layernorm
        self.feed_forward = mlp

    @classmethod
    def from_config(cls, config: Phi35MiniDecoderLayerConfig) -> Phi35MiniDecoderLayer:
        return cls(
            input_layernorm=RMSNorm1D.from_config(config.attention_norm_config),
            attention=Phi35MiniAttention.from_config(config.attention_config),
            post_attention_layernorm=RMSNorm1D.from_config(config.ff_norm_config),
            mlp=Phi35MiniMLP.from_config(config.mlp_config),
        )

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        batch_size: int = 1,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        row_map: ttnn.Tensor | None = None,
        padded_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        # For batched prefill (batch_size > 1) x is the folded [1,1,B*S,dim] hidden state; norm,
        # residual add and MLP are row-independent so they treat B*S as one long sequence unchanged.
        # Only attention unfolds the batch axis internally (see Attention1D.prefill_forward).
        # Fractured embed / norm activations must be all-gathered to full ``dim`` before
        # Attention1D / MLP1D (QKV matmul expects width ``dim``).
        # One 128-row user keeps the residual stream and the attention / MLP inputs and outputs in L1 (0.75 MB per
        # tensor); longer or batched prefills use DRAM.
        # The 128-row residual stream stays block-sharded in L1 between layers, in the norms' layout: the residual
        # adds write it there and the norms read it without a reshard (2 x 1.9-3.6 us per layer on a P150). The
        # model's prefill_forward returns it to L1 interleaved after the last layer.
        residual_memcfg = (
            _prefill_128_residual_memcfg(self.attention.config.mesh_device, x.shape[-1])
            if x.shape[-2] == PREFILL_1D_SEQ_LEN
            and batch_size == 1
            and chunk_start_idx is None
            and chunk_start_idx_tensor is None
            else ttnn.DRAM_MEMORY_CONFIG
        )
        # The residual adds happen inside the attention / MLP calls: the batched attention path and MLPs above 128 rows
        # fuse them into the WO / W2 matmuls, the other paths add after.
        r = _prefill_rms_norm(self.input_layernorm, x)
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r)
        h = self.attention.prefill_forward_with_residual(
            r,
            x,
            rot_mats,
            residual_memcfg=residual_memcfg,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            batch_size=batch_size,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            row_map=row_map,
            padded_sequence_length=padded_sequence_length,
        )
        # The MLP norm writes bfloat8_b above 128 rows: the LoFi fused W1/W3 matmul then reads half the bytes (P150,
        # 4096 rows: norm 141 -> 117 us, W1/W3 943 -> 844 us). The attention norm stays bfloat16: a bfloat8_b QKV
        # input (HiFi2) as well brought 32-user top-5 accuracy to 98.6%.
        r2 = _prefill_rms_norm(self.post_attention_layernorm, h, long_prefill_dtype=ttnn.bfloat8_b)
        r2 = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, r2)
        return self.mlp.prefill_forward_with_residual(r2, h, residual_memcfg=residual_memcfg)

    def supports_128_deferred_add(self) -> bool:
        return self.attention.supports_128_deferred_add() and self.mlp.config.prefill_128_w2_stream_prg_config is not None

    def prefill_forward_128_deferred(
        self,
        x: ttnn.Tensor,
        delta: ttnn.Tensor | None,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Single-user 128-row layer with both residual adds folded into the following norms, as in decode
        (decode_forward_deferred_add).

        ``x`` is the residual stream (block-sharded in L1) and ``delta`` the previous layer's MLP output still to be
        added to it (None for the first layer). The streamed WO and W2 write their outputs straight into the residual's
        layout; the input norm adds ``delta`` and the post-attention norm adds the WO output, each leaving the sum in
        ``x`` (write_pre_add_sum). Returns ``x`` and this layer's MLP output, which the next layer's input norm (or the
        model) adds. Removes the two residual-add ops a layer (3.9 us each plus a dispatch gap on a P150).
        """
        memcfg = _prefill_128_residual_memcfg(self.attention.config.mesh_device, x.shape[-1])
        if x.memory_config() != memcfg:
            # The first layer's input (the embedding output) is left as it is, as the non-deferred path does.
            x = ttnn.to_memory_config(x, memcfg)
        r = _prefill_rms_norm(self.input_layernorm, x, delta=delta)
        if delta is not None:
            ttnn.deallocate(delta)
        attn_out = self.attention.prefill_forward(
            r, rot_mats, user_id=user_id, page_table=page_table, chunk_page_table=chunk_page_table, batch_size=1,
            wo_memory_config=memcfg,
        )
        r2 = _prefill_rms_norm(self.post_attention_layernorm, x, delta=attn_out)
        ttnn.deallocate(attn_out)
        return x, self.mlp.prefill_128_to(r2, memcfg)

    def decode_forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        xa = _all_gather_rmsnorm_tensor(
            self.input_layernorm, x, memory_config=self.input_layernorm.config.decode_memory_config
        )
        r = self.input_layernorm.forward(xa, "decode")
        r = self.attention.forward(r, current_pos, rot_mats, mode="decode", page_table=page_table)
        residual_memcfg = self.attention.config.decode_residual_memcfg
        h = ttnn.add(x, r, memory_config=residual_memcfg)
        hf = _all_gather_rmsnorm_tensor(
            self.post_attention_layernorm, h, memory_config=self.post_attention_layernorm.config.decode_memory_config
        )
        r2 = self.post_attention_layernorm.forward(hf, "decode")
        r2 = self.mlp.forward(r2, "decode")
        return ttnn.add(h, r2, memory_config=residual_memcfg)

    def decode_forward_deferred_add(
        self,
        x: ttnn.Tensor,
        delta: ttnn.Tensor | None,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Single-device decode layer with the residual adds folded into the norms (_decode_rms_norm_add).

        ``x`` is the residual stream and ``delta`` the previous layer's MLP output still to be added to it (None for
        the first layer). Returns the residual stream (``x`` itself, updated in place) and this layer's MLP output,
        which the next layer's input norm (or the model's final norm) adds.
        """
        if delta is None:
            r = self.input_layernorm.forward(x, "decode")
        else:
            r = _decode_rms_norm_add(self.input_layernorm, x, delta)
            ttnn.deallocate(delta)
        r = self.attention.forward(r, current_pos, rot_mats, mode="decode", page_table=page_table)
        r2 = _decode_rms_norm_add(self.post_attention_layernorm, x, r)
        ttnn.deallocate(r)
        return x, self.mlp.forward(r2, "decode")


class Phi35MiniTransformer(LightweightModule):
    """
    Full decoder for Phi-3.5-mini (TTTv2 modules only) on one P150.

    Prefill/decode on **embedded** activations match the model-owned runtime. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(self, config: Phi35MiniTransformerConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config
        self.embed = Embedding1D.from_config(config.embedding_config)
        self.embedding = self.embed
        self.rope_setup = Phi35MiniRotarySetup.from_config(config.rope_config)
        self.layers = [
            Phi35MiniDecoderLayer.from_config(config.block_configs[index])
            for index in tqdm(range(config.n_layers), desc="Building layers")
        ]
        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)
        self.sampling = (
            Phi35MiniSampling.from_config(config.sampling_config) if config.sampling_config is not None else None
        )
        self.supports_on_device_sampling = self.sampling is not None
        self.mesh_device = config.mesh_device
        self.tt_ccl = config.tt_ccl or (get_tt_ccl(config.mesh_device) if config.num_devices > 1 else None)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers
        self.model_args = None

    @property
    def n_kv_heads(self) -> int:
        return self.config.block_configs[0].attention_config.n_kv_heads

    def kv_cache_memory_config(self, block_size: int) -> ttnn.MemoryConfig:
        """Memory config for the paged KV cache tensors (see ``_kv_cache_memory_config``)."""
        attn = self.config.block_configs[0].attention_config
        return _kv_cache_memory_config(
            self.mesh_device,
            max_batch_size=attn.max_batch_size,
            n_kv_heads=attn.n_kv_heads,
            block_size=block_size,
            head_dim=attn.head_dim,
        )

    def iter_executor_named_modules(self):
        if not hasattr(self, "layers"):
            return
        for index, layer in enumerate(self.layers):
            for suffix, submodule in (
                ("attn_norm", getattr(layer, "attention_norm", None)),
                ("attention", getattr(layer, "attention", None)),
                ("ff_norm", getattr(layer, "ff_norm", None)),
                ("mlp", getattr(layer, "feed_forward", None)),
            ):
                if submodule is not None:
                    yield f"layer[{index}].{suffix}", submodule
        if hasattr(self, "norm"):
            yield "final_norm", self.norm
        if hasattr(self, "lm_head"):
            yield "lm_head", self.lm_head

    def configure_paged_attention(self, *, block_size: int, max_num_blocks: int) -> None:
        for name, value in (("block_size", block_size), ("max_num_blocks", max_num_blocks)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        live_configs = tuple(layer.attention.config for layer in self.layers)
        for layer_index, config in enumerate(live_configs):
            if not config.use_vllm_paged_kv_cache or config.paged_attention_config is None:
                raise RuntimeError(f"Model layer {layer_index} is not configured for externally managed paged KV cache")
            if config.kv_cache is not None or getattr(self.layers[layer_index].attention, "kv_cache", None) is not None:
                raise RuntimeError(f"Model layer {layer_index} already has a bound KV cache")
        construction_configs = tuple(block.attention_config for block in self.config.block_configs)
        for config in tuple({id(item): item for item in (*construction_configs, *live_configs)}.values()):
            config.paged_attention_config = replace(
                config.paged_attention_config,
                block_size=block_size,
                max_num_blocks=max_num_blocks,
            )

    def set_kv_cache(self, kv_cache: list | None) -> None:
        if kv_cache is None:
            for layer in self.layers:
                layer.attention.config.kv_cache = None
                if hasattr(layer.attention, "kv_cache"):
                    layer.attention.kv_cache = None
            return
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers")
        cache_pairs = []
        for index, value in enumerate(kv_cache):
            try:
                pair = tuple(value)
            except TypeError as error:
                raise TypeError(f"kv_cache layer {index} must provide an iterable K/V tensor pair") from error
            if len(pair) != 2:
                raise ValueError(f"kv_cache layer {index} must contain exactly two K/V tensors")
            cache_pairs.append(pair)
        for layer, pair in zip(self.layers, cache_pairs):
            layer.attention.config.kv_cache = pair
            if hasattr(layer.attention, "kv_cache"):
                layer.attention.kv_cache = pair

    def prepare_prefill_rot_mats(self, position_indices: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.rope_setup.load_device_weights()
        cos = None
        sin = None
        try:
            # Up to the 128-row bucket the tables (2 x 24 KB) stay in L1, where the prefill rotary op reads them faster.
            memcfg = ttnn.L1_MEMORY_CONFIG if position_indices.shape[-1] <= PREFILL_1D_SEQ_LEN else None
            cos = ttnn.embedding(
                position_indices, self.rope_setup.cos_matrix_rm, layout=ttnn.TILE_LAYOUT, memory_config=memcfg
            )
            sin = ttnn.embedding(
                position_indices, self.rope_setup.sin_matrix_rm, layout=ttnn.TILE_LAYOUT, memory_config=memcfg
            )
            return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)
        except BaseException:
            for tensor in (sin, cos):
                if tensor is not None:
                    try:
                        ttnn.deallocate(tensor)
                    except BaseException:
                        pass
            raise

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Decode token embedding, looked up straight into the residual's L1 width-sharded layout.

        With a width-sharded output the embedding op splits the hidden columns over the residual cores (12 on a P150:
        2.4 us); ``Embedding1D`` looks up into DRAM on one core (14.7 us) and the residual then needs a reshard
        (1.4 us).
        """
        if self.embed.config.embed_scale != 1.0 or not self.decode_residual_memcfg.is_sharded():
            x = ttnn.unsqueeze_to_4D(self.embed.forward(tokens))
            return ttnn.to_memory_config(x, self.decode_residual_memcfg)
        self.embed.load_device_weights()
        x = ttnn.embedding(tokens, self.embed.weights, layout=ttnn.TILE_LAYOUT, memory_config=self.decode_residual_memcfg)
        return ttnn.unsqueeze_to_4D(x)

    @property
    def supports_packed_batched_prefill(self) -> bool:
        """Batched prefill may run on packed rows: the shared prefill runtime then lays the users' 32-row tiles out back
        to back and passes ``row_map`` / ``padded_sequence_length`` to ``prefill_forward``. Norms and MLPs run on the
        packed rows only; the attention's QKV matmul writes the padded per-user heads through the row map and its WO
        reads them back. P150, 32 users of 61-103 tokens: 97 of 128 row tiles."""
        return self.num_devices == 1

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed.forward(tokens)
        return ttnn.unsqueeze_to_4D(x)

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
        batch_size: int = 1,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
        row_map: ttnn.Tensor | None = None,
        padded_sequence_length: int | None = None,
    ) -> ttnn.Tensor:
        # batch_size > 1: x_embed is the folded [1,1,B*S,dim] tensor (B users). With ``row_map`` (packed batched
        # prefill, see ``supports_packed_batched_prefill``) it holds only the users' 32-row tiles back to back. The batched path always
        # returns the full hidden state when no runtime last-token slice is supplied; the executor does
        # per-slot last-token extraction + norm/lm_head so those stages stay bit-identical to the single-user path.
        # Batched prefill runs its residual stream in bfloat8_b: the fused WO / W2 residual matmuls and both norms read
        # (and the matmuls write) half the bytes. P150, 32 users x 128 rows: device 100.2 -> 95.7 ms (WO 286 -> 236 us,
        # W2 542 -> 496, norms 136 / 112 -> 116 / 86); 32-user teacher forcing 91.0 / 99.2% (was 91.8 / 99.2). A
        # single long prompt keeps bfloat16: with bfloat8_b its token accuracy fell to 91.4 / 98.6%.
        x = x_embed
        if batch_size > 1 and x.dtype != ttnn.bfloat8_b:
            x = ttnn.typecast(x_embed, ttnn.bfloat8_b)
            ttnn.deallocate(x_embed)
        deferred = (
            x.shape[-2] == PREFILL_1D_SEQ_LEN
            and batch_size == 1
            and chunk_start_idx is None
            and chunk_start_idx_tensor is None
            and row_map is None
            and all(layer.supports_128_deferred_add() for layer in self.layers)
        )
        if deferred:
            # Single-user 128 rows: residual adds folded into the norms (prefill_forward_128_deferred); the last
            # layer's MLP output is added here, writing the L1-interleaved hidden state in the same op.
            delta = None
            for layer in self.layers:
                x, delta = layer.prefill_forward_128_deferred(
                    x, delta, rot_mats, user_id=user_id, page_table=page_table, chunk_page_table=chunk_page_table
                )
            old = x
            x = ttnn.add(old, delta, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(old)
            ttnn.deallocate(delta)
        for layer in self.layers if not deferred else ():
            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                batch_size=batch_size,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                row_map=row_map,
                padded_sequence_length=padded_sequence_length,
            )
        if x.is_sharded():  # the 128-row residual stream (Phi35MiniDecoderLayer.prefill_forward)
            sharded = x
            x = ttnn.to_memory_config(sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(sharded)

        if last_token_index is not None and last_token_slice is None:
            raise ValueError("last_token_index is required with a runtime last_token_slice")
        if get_last_token == -1 and last_token_slice is None:
            return x

        old = x
        if last_token_slice is None:
            x_tile = _slice_last_token_tile(old, get_last_token)
        else:
            x_tile = ttnn.slice(
                old,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(old.shape[2]) // 32,
            )
        ttnn.deallocate(old)
        if last_token_index is not None:
            if x_tile.dtype != ttnn.bfloat16:
                old_tile = x_tile
                x_tile = ttnn.typecast(x_tile, ttnn.bfloat16)
                ttnn.deallocate(old_tile)
            old_tile = x_tile
            x_tile = ttnn.embedding(last_token_index, x_tile, layout=ttnn.TILE_LAYOUT)
            x_tile = ttnn.unsqueeze_to_4D(x_tile)
            ttnn.deallocate(old_tile)
        return self._last_tile_logits(x_tile)

    def post_process_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        if last_token_slice is None:
            x = _slice_last_token_tile(hidden_states, last_token_idx)
        else:
            x = ttnn.slice(
                hidden_states,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(hidden_states.shape[2]) // 32,
            )
        if last_token_index is not None:
            if x.dtype != ttnn.bfloat16:
                old = x
                x = ttnn.typecast(x, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x
            x = ttnn.embedding(last_token_index, x, layout=ttnn.TILE_LAYOUT)
            x = ttnn.unsqueeze_to_4D(x)
            ttnn.deallocate(old)
        return self._last_tile_logits(x)

    def _last_tile_logits(self, x_tile: ttnn.Tensor) -> ttnn.Tensor:
        """Final-norm + all-gather + LM-head on a 32-row tile. ``x_tile`` shape ``[1, 1, 32, dim]``.

        On one device the tile takes decode's route: sharded onto the norm's decode cores, the sharded decode norm
        (5.6 us) and the LM head on its output, instead of the interleaved prefill norm, which runs a 32-row tile on
        one core (47.8 us on a P150).
        """
        if self.num_devices == 1 and self.norm.config.decode_memory_config is not None:
            x = ttnn.to_memory_config(x_tile, self.norm.config.decode_memory_config)
            ttnn.deallocate(x_tile)
            x = self.norm.decode_forward(x)
            return self.lm_head.forward(x)
        x = self.norm.prefill_forward(x_tile)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def post_process_batched_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx_list: list[int],
        padded_batch: int,
        prefill_seq_len: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        del last_token_slice, last_token_index
        fold_len = int(hidden_states.shape[-2])
        selector = torch.zeros(1, 1, 32, fold_len, dtype=torch.bfloat16)
        packed = fold_len != padded_batch * prefill_seq_len
        row_start = 0
        for local_row, last_token_idx in enumerate(last_token_idx_list):
            # Packed rows: each user takes ceil(length / 32) tiles back to back (plan.packed_prefill_layout).
            start = row_start if packed else local_row * prefill_seq_len
            selector[0, 0, local_row, start + last_token_idx] = 1.0
            row_start += math.ceil((last_token_idx + 1) / TILE_SIZE) * TILE_SIZE
        selector = ttnn.from_torch(
            selector,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.matmul(
            selector,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(selector)
        return self._last_tile_logits(x)

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        x = x_embed
        if self.num_devices == 1:
            # Residual adds folded into the next norm (Phi35MiniDecoderLayer.decode_forward_deferred_add).
            delta = None
            for layer in self.layers:
                x, delta = layer.decode_forward_deferred_add(x, delta, current_pos, rot_mats, page_table=page_table)
            x = _decode_rms_norm_add(self.norm, x, delta)
            ttnn.deallocate(delta)
            return self.lm_head.forward(x)
        for layer in self.layers:
            x = layer.decode_forward(x, current_pos, rot_mats, page_table=page_table)
        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        return self.lm_head.forward(x)

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        if self.num_devices > 1 and self.tt_ccl is not None:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=default_topology(self.mesh_device),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=24,
                num_workers_per_link=4,
                num_buffers_per_channel=2,
            )
        return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor) -> None:
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)
