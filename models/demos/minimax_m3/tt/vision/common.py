# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the MiniMax-M3 vision modules: uploads, compute configs, the weight matmul and program configs.

block_placement decides whether an encoder block keeps its activations in L1 or in DRAM.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
from typing import Callable, NamedTuple, Optional

import torch

import ttnn
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE

# Circular-buffer bytes per core for a weight matmul. It stays under a Blackhole core's free L1 (about 1.39 MiB) and
# leaves room for the tensors in L1: the RoPE tables and, where block_placement puts them there, the activations.
# It also sets the K block: the largest that fits.
LINEAR_L1_BYTES = 1_000_000
# The largest K block (in0_block_w), in tiles.
LINEAR_MAX_IN0_BLOCK_W = 8
# The fewest tiles of M per core with which a weight matmul may use transpose_mcast.
LINEAR_MIN_TRANSPOSED_PER_CORE_M = 4
# Output subblocks (h, w) in the order ttnn's own matmul configs try them.
_SUBBLOCKS = (
    (4, 2), (2, 4), (8, 1), (1, 8), (7, 1), (1, 7), (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5), (2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1),
)  # fmt: skip


def to_device(tensor: torch.Tensor, mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Upload a host tensor to DRAM as a tile tensor of dtype, replicated over the mesh.

    The tensor is rounded to bf16 on the host first.
    """
    return ttnn.from_torch(
        tensor.to(torch.bfloat16).contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    """Upload an nn.Linear weight [out, in] as the [in, out] tile tensor that ttnn.linear takes."""
    return to_device(torch_w.transpose(-2, -1), mesh_device, dtype)


def as_linear_bias(mesh_device, torch_b: torch.Tensor, dtype) -> ttnn.Tensor:
    """Upload a bias [out] as a [1, 1, 1, out] tile tensor, replicated over the mesh."""
    return to_device(torch_b.view(1, 1, 1, -1), mesh_device, dtype)


def hifi4_compute_config(packer_l1_acc: bool = False):
    """HiFi4 compute config with exact math and fp32 accumulation, as LayerNorm, RoPE and SDPA use it."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def linear_compute_config():
    """hifi4_compute_config for the weight matmuls, with partial sums accumulated in L1 (packer_l1_acc)."""
    return hifi4_compute_config(packer_l1_acc=True)


def memory_config_kwargs(memory_config: Optional[ttnn.MemoryConfig]) -> dict:
    """Return {"memory_config": memory_config}, or {} for None so that the op writes where its default puts it."""
    return {} if memory_config is None else {"memory_config": memory_config}


def linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    b: Optional[ttnn.Tensor],
    compute_kernel_config,
    activation: Optional[Callable[..., ttnn.Tensor]] = None,
    fused_activation: Optional[ttnn.UnaryWithParam] = None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    activation_memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """Run the weight matmul x @ w + b into memory_config, then activation, if given, as a separate op.

    w is [in, out] and b is [1, 1, 1, out] or None, as as_linear_weight and as_linear_bias upload them. The program
    config is linear_program_config's for the padded shapes, and for x and the output in block shards or not.
    activation runs as activation(out, memory_config=activation_memory_config); None there writes where out is.
    fused_activation is the same activation in a program config's form, and needs activation. Where
    linear_program_config gives a config, that config carries fused_activation instead, and the matmul writes
    activation_memory_config, or memory_config when that is None.

    Raises:
        ValueError: for x or the output in block shards when linear_program_config leaves the config to ttnn.
    """
    assert fused_activation is None or activation is not None, "fused_activation needs activation"
    in0_sharded = x.memory_config().is_sharded()

    def program_config_for(out_memory_config):
        return linear_program_config(
            x.device(),
            x.padded_shape[-2],
            w.padded_shape[-2],
            w.padded_shape[-1],
            compute_kernel_config,
            in0_sharded=in0_sharded,
            out_sharded=out_memory_config.is_sharded(),
        )

    fused_memory_config = memory_config if activation_memory_config is None else activation_memory_config
    program_config = None if fused_activation is None else program_config_for(fused_memory_config)
    fused = program_config is not None
    if fused:
        # linear_program_config builds a new config at each call, so no other matmul gets the activation.
        program_config.fused_activation = fused_activation
        memory_config = fused_memory_config
    else:
        program_config = program_config_for(memory_config)
    if program_config is None and (in0_sharded or memory_config.is_sharded()):
        raise ValueError(
            f"a matmul of {x.padded_shape[-2]} rows by {tuple(w.padded_shape[-2:])} with a sharded input or output"
            " needs linear_program_config's config, and it leaves this one to ttnn"
        )
    out = ttnn.linear(
        x,
        w,
        bias=b,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
        memory_config=memory_config,
    )
    return out if activation is None or fused else activation(out, memory_config=activation_memory_config)


def linear_program_config(
    mesh_device,
    rows: int,
    in_features: int,
    out_features: int,
    compute_kernel_config,
    *,
    in0_sharded: bool = False,
    out_sharded: bool = False,
):
    """Return the program config for a weight matmul of (rows, in_features) by (in_features, out_features).

    Operands and output are bf16 or bfloat8_b, interleaved in DRAM or L1, or in L1 block shards. The config is a 2D
    multicast with one output block per core. _linear_blocking picks the split and the blocks within the LINEAR_*
    limits. Returns None to leave the choice to ttnn in two cases. Either a 1D multicast would give each core no more
    output tiles, as for small images, or no K block fits in LINEAR_L1_BYTES.

    The sharding flags change neither the split nor the K blocks, so every output tile is the same sum.

    Args:
        in0_sharded: the input is in block shards. Sets fuse_batch, which the matmul needs then.
        out_sharded: the output is in block shards, one block per core. Takes a subblock one tile high or as wide as the
            block, which the matmul needs then.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    blocking = _linear_blocking(
        math.ceil(rows / ttnn.TILE_SIZE),
        math.ceil(in_features / ttnn.TILE_SIZE),
        math.ceil(out_features / ttnn.TILE_SIZE),
        grid.x,
        grid.y,
        bool(compute_kernel_config.fp32_dest_acc_en),
        out_sharded=out_sharded,
    )
    if blocking is None:
        return None
    per_core_m, per_core_n, in0_block_w, subblock_h, subblock_w, transpose_mcast = blocking
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=transpose_mcast,
        fuse_batch=in0_sharded,
    )


class _LinearBlocking(NamedTuple):
    """linear_program_config's choice, in tiles."""

    per_core_m: int
    per_core_n: int
    in0_block_w: int
    subblock_h: int
    subblock_w: int
    transpose_mcast: bool


@functools.lru_cache(maxsize=None)
def _linear_blocking(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    fp32_dest_acc: bool,
    *,
    out_sharded: bool = False,
    l1_bytes: int = LINEAR_L1_BYTES,
    max_in0_block_w: int = LINEAR_MAX_IN0_BLOCK_W,
    min_transposed_per_core_m: int = LINEAR_MIN_TRANSPOSED_PER_CORE_M,
):
    """Return linear_program_config's choice in tiles, as a _LinearBlocking, or None.

    out_sharded changes only the subblock. The limits default to the LINEAR_* values at import time, and are part of
    the cache key.
    """
    cores = grid_x * grid_y
    # Output tiles per core of a 1D multicast: all the rows and a slice of the columns on each core, or vice versa.
    one_d = min(m_tiles * math.ceil(n_tiles / cores), math.ceil(m_tiles / cores) * n_tiles)
    # Without transpose_mcast the grid's rows (y) take M and its columns (x) take N; with it, the other way round.
    splits = [(math.ceil(m_tiles / grid_y), math.ceil(n_tiles / grid_x), False)]
    if math.ceil(m_tiles / grid_x) >= min_transposed_per_core_m:
        splits.append((math.ceil(m_tiles / grid_x), math.ceil(n_tiles / grid_y), True))
    per_core_m, per_core_n, transpose = min(splits, key=lambda s: (s[0] * s[1], s[2]))
    if per_core_m * per_core_n >= one_d:
        return None

    def cb_bytes(in0_block_w):
        # The 2D multicast factory's circular buffers, in 2 KB bf16 tiles: double-buffered in0 and in1 K blocks, the
        # output block and a bias row. With fp32 accumulation the partial sums get their own 4 KB fp32 tiles.
        block = per_core_m * per_core_n
        partials = block * 4096 if fp32_dest_acc else 0
        return 2048 * (2 * in0_block_w * (per_core_m + per_core_n) + block + per_core_n) + partials

    fitting = [w for w in range(1, min(k_tiles, max_in0_block_w) + 1) if k_tiles % w == 0 and cb_bytes(w) <= l1_bytes]
    if not fitting:
        return None
    dest_tiles = 4 if fp32_dest_acc else 8
    subblock_h, subblock_w = next(
        (h, w)
        for h, w in _SUBBLOCKS
        if h * w <= dest_tiles
        and per_core_m % h == 0
        and per_core_n % w == 0
        and (not out_sharded or h == 1 or w == per_core_n)
    )
    return _LinearBlocking(per_core_m, per_core_n, fitting[-1], subblock_h, subblock_w, transpose)


# The q chunks sdpa_program_config picks from. Chunks below 128 are left out: on a P150 a q chunk of 64 saved 0.4 ms
# of the 3.7 ms its query rows predict at 672x672, and none has been timed at a size where the rule would pick it.
SDPA_Q_CHUNK_SIZES = (128, 160, 192, 256)
# The k chunk of sdpa_program_config. SDPA rescales its running softmax once per k chunk, so chunks of 256 do that half
# as often as chunks of 128. Larger chunks take more circular-buffer room, which the L1 budget test checks.
SDPA_K_CHUNK_SIZE = 256


def sdpa_program_config(mesh_device, seq_len: int, num_heads: int):
    """SDPA program config for seq_len queries and keys in each of num_heads heads, on the whole compute grid.

    k chunks are SDPA_K_CHUNK_SIZE; _sdpa_q_chunk_size picks the q chunk. Each call returns a new config, so a caller
    may change its own. exp_approx_mode is left unset, so SDPA keeps the op's default, approximate exp.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=_sdpa_q_chunk_size(seq_len, num_heads, grid.x * grid.y),
        k_chunk_size=SDPA_K_CHUNK_SIZE,
    )


# Unbounded: one int per patch count and grid, and ttnn keeps a compiled SDPA program for each anyway.
@functools.lru_cache(maxsize=None)
def _sdpa_q_chunk_size(seq_len: int, num_heads: int, cores: int) -> int:
    """sdpa_program_config's q chunk, from SDPA_Q_CHUNK_SIZES, on a grid of `cores` cores.

    Takes the chunk that leaves the slowest core the fewest query rows. On a tie it takes the larger chunk, which makes
    fewer passes over k and v.
    """
    # The op deals the num_heads x ceil(seq_len / q) q chunks evenly over the cores, so the slowest core runs
    # ceil(num_heads * ceil(seq_len / q) / cores) chunks of q rows.
    return min(SDPA_Q_CHUNK_SIZES, key=lambda q: (math.ceil(num_heads * math.ceil(seq_len / q) / cores) * q, -q))


@dataclasses.dataclass(frozen=True)
class BlockPlacement:
    """Where an encoder block's ops write their outputs, as the memory_config each passes to ttnn.

    None means that the op gets no memory_config and writes where its input is. The blocks' LayerNorms and RoPE ops
    never get one, and neither does SDPA, whose output stays in DRAM. block_placement picks one per image size and grid.

    Attributes:
        name: the placement's name, for logs.
        block_input: pre_layrnorm's output, the first block's input.
        residual: the residual stream: both adds' outputs, and the out_proj and fc2 outputs they add.
        qkv: QKV's output and the q, k and v heads split from it.
        fc1_out: fc1's output.
        gelu_out: GELU's output, which fc2 reads.
    """

    name: str
    block_input: Optional[ttnn.MemoryConfig]
    residual: ttnn.MemoryConfig
    qkv: ttnn.MemoryConfig
    fc1_out: ttnn.MemoryConfig
    gelu_out: Optional[ttnn.MemoryConfig]


# Every activation in DRAM: pre_layrnorm and GELU get no memory config and write where their input is.
DRAM_PLACEMENT = BlockPlacement(
    name="dram",
    block_input=None,
    residual=ttnn.DRAM_MEMORY_CONFIG,
    qkv=ttnn.DRAM_MEMORY_CONFIG,
    fc1_out=ttnn.DRAM_MEMORY_CONFIG,
    gelu_out=None,
)
# The block's activations in L1 interleaved, except the inputs of out_proj and fc2: SDPA's and GELU's outputs stay in
# DRAM (block_placement).
L1_INTERLEAVED_PLACEMENT = BlockPlacement(
    name="l1_interleaved",
    block_input=ttnn.L1_MEMORY_CONFIG,
    residual=ttnn.L1_MEMORY_CONFIG,
    qkv=ttnn.L1_MEMORY_CONFIG,
    fc1_out=ttnn.L1_MEMORY_CONFIG,
    gelu_out=ttnn.DRAM_MEMORY_CONFIG,
)
# As L1_INTERLEAVED_PLACEMENT, but fc1 writes block shards in L1, one block of its program config per core. GELU
# keeps them, and fc2 reads them.
L1_SHARDED_MLP_PLACEMENT = dataclasses.replace(
    L1_INTERLEAVED_PLACEMENT,
    name="l1_sharded_mlp",
    fc1_out=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
    gelu_out=None,
)
# The values of M3VLVisionModel.activations. "l1" keeps the blocks' activations in L1 where block_placement proves
# they fit, with the MLP's hidden ones in block shards where those fit. "l1_interleaved" does the same without
# shards, and "dram" keeps every activation in DRAM.
ACTIVATIONS = ("l1", "l1_interleaved", "dram")
# The core grids where block_placement may use L1: a Blackhole chip's, by harvesting. The L1 budget test covers each.
L1_GRIDS = ((11, 10), (12, 10), (13, 10))
# The most rows block_placement keeps in L1: those of 672x672, the largest image the processor makes. Larger images
# run in DRAM, as some of them would not fit: an op's circular buffers would overlap the activations.
L1_MAX_ROWS = 2304
# The (in_features, out_features, fp32_dest_acc) of QKV, out_proj, fc1 and fc2 that block_placement may use L1 for:
# the checkpoint's, with heads padded to 96 and fp32 accumulation. The L1 budget test covers only these widths.
L1_LINEARS = ((1280, 4608, True), (1536, 1280, True), (1280, 5120, True), (5120, 1280, True))
# A P150's L1 bank with no L1_SMALL region, as measured on one.
P150_L1_BANK_BYTES = 1_460_864
# Room the proof leaves in each L1 bank for the runtime to take more than when P150_L1_BANK_BYTES was measured.
PROOF_L1_BANK_SLACK_BYTES = 16 * 1024
# The L1 bank size that block_placement is proven for: a P150's, less the L1_SMALL region that every M3 mesh open
# reserves, less PROOF_L1_BANK_SLACK_BYTES. A larger bank only adds room.
PROOF_L1_BANK_BYTES = P150_L1_BANK_BYTES - L1_SMALL_SIZE - PROOF_L1_BANK_SLACK_BYTES
# How much of the top of each L1 bank other tensors may take when a forward starts. Every margin of the proof exceeds
# it.
PROOF_L1_FOREIGN_BYTES = 64 * 1024
# The variable that makes tt-metal open its devices with the HYBRID allocator, when it starts with 1. That allocator's
# per-core tensors do not show in ttnn.get_memory_view, so the L1 check fails while its value starts with 1.
HYBRID_ALLOCATOR_ENV = "TT_METAL_ALLOCATOR_MODE_HYBRID"


def block_placement(block, rows: int, activations: str) -> BlockPlacement:
    """Return the BlockPlacement of an encoder block for an input of `rows` rows, padded or not.

    activations is one of ACTIVATIONS. "l1" gives L1_SHARDED_MLP_PLACEMENT or L1_INTERLEAVED_PLACEMENT, and
    "l1_interleaved" the latter, where the host test of the L1 budget proves that they fit. Elsewhere, and for "dram",
    it gives DRAM_PLACEMENT. The choice is cached on the block's linear widths, the tile rows and the grid.

    L1 is used only for the linears of L1_LINEARS, on the grids of L1_GRIDS, up to L1_MAX_ROWS rows. All four block
    linears must also get a config from linear_program_config, as ttnn's own configs plan by the free L1.
    L1_INTERLEAVED_PLACEMENT leaves the inputs of out_proj and fc2 in DRAM, as those matmuls ran slower reading L1
    interleaved. "l1" also shards the MLP's hidden activations, for fc2 to read, where each core's block of fc1's
    output is the shard of fc2's input that the core reads.

    Raises:
        ValueError: for activations not in ACTIVATIONS.
    """
    if activations not in ACTIVATIONS:
        raise ValueError(f"activations must be one of {ACTIVATIONS}, got {activations!r}")
    if activations == "dram":
        return DRAM_PLACEMENT
    grid = block_linears(block)[0][0].device().compute_with_storage_grid_size()
    widths = block_linear_widths(block)
    return _l1_block_placement(math.ceil(rows / ttnn.TILE_SIZE), widths, grid.x, grid.y, activations == "l1")


def block_linears(block) -> tuple:
    """Return the (weight, compute config) of the block's QKV, out_proj, fc1 and fc2, as passed to linear()."""
    attention, mlp = block.attention, block.mlp
    return (
        (attention.wqkv, attention.compute_kernel_config),
        (attention.wo, attention.compute_kernel_config),
        (mlp.fc1_weight, mlp.compute_kernel_config),
        (mlp.fc2_weight, mlp.compute_kernel_config),
    )


def block_linear_widths(block) -> tuple[tuple[int, int, bool], ...]:
    """Return the (in_features, out_features, fp32_dest_acc) of each of block_linears(block), as linear() reads them."""
    return tuple(
        (w.padded_shape[-2], w.padded_shape[-1], bool(config.fp32_dest_acc_en)) for w, config in block_linears(block)
    )


@functools.lru_cache(maxsize=None)
def _l1_block_placement(
    m_tiles: int, linears: tuple[tuple[int, int, bool], ...], grid_x: int, grid_y: int, shard_mlp: bool
) -> BlockPlacement:
    """Return block_placement's choice for m_tiles tile rows.

    linears is what block_linear_widths gives, and shard_mlp is True for "l1".
    """
    if linears != L1_LINEARS or (grid_x, grid_y) not in L1_GRIDS or m_tiles * ttnn.TILE_SIZE > L1_MAX_ROWS:
        return DRAM_PLACEMENT
    n_tiles, blockings = [], []
    for in_features, out_features, fp32_dest_acc in linears:
        k_tiles = math.ceil(in_features / ttnn.TILE_SIZE)
        n_tiles.append(math.ceil(out_features / ttnn.TILE_SIZE))
        # linear_program_config's choice: None leaves the config to ttnn.
        blockings.append(_linear_blocking(m_tiles, k_tiles, n_tiles[-1], grid_x, grid_y, fp32_dest_acc))
        if blockings[-1] is None:
            return DRAM_PLACEMENT
    if not shard_mlp:
        return L1_INTERLEAVED_PLACEMENT
    (fc1, fc2), (fc1_n_tiles, fc2_n_tiles) = blockings[2:], n_tiles[2:]
    # Shard the MLP only where each core's block of fc1's output is the shard of fc2's input that the core reads. A
    # sharded input needs COL_MAJOR shards, hence transposed splits with the same rows per core.
    if (
        fc1.transpose_mcast
        and fc2.transpose_mcast
        and fc1.per_core_m == fc2.per_core_m
        and fc1_n_tiles % fc1.per_core_n == 0
        and fc1.per_core_n % fc2.in0_block_w == 0
        and math.ceil(fc1_n_tiles / fc1.per_core_n) == math.ceil(fc2_n_tiles / fc2.per_core_n)
    ):
        return L1_SHARDED_MLP_PLACEMENT
    return L1_INTERLEAVED_PLACEMENT


def l1_free_for_activations(mesh_device) -> tuple[bool, str]:
    """Check that mesh_device's L1 is as block_placement's proof assumes, and return (ok, the reason if not).

    The proof assumes the lockstep allocator, one L1 bank per core, at least PROOF_L1_BANK_BYTES per bank, and no tensor
    more than PROOF_L1_FOREIGN_BYTES below the top of a bank. Circular buffers grow up from the bottom of a bank and
    tensors down from its top, so even a small tensor low in the bank cuts every margin.
    """
    if os.environ.get(HYBRID_ALLOCATOR_ENV, "").startswith("1"):
        return False, f"{HYBRID_ALLOCATOR_ENV} is set, and the L1 memory view does not show per-core tensors"
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    grid = mesh_device.compute_with_storage_grid_size()
    if view.num_banks != grid.x * grid.y:
        return False, f"{view.num_banks} L1 banks for {grid.x * grid.y} cores"
    if view.total_bytes_per_bank < PROOF_L1_BANK_BYTES:
        return False, f"{view.total_bytes_per_bank} B per L1 bank, under the {PROOF_L1_BANK_BYTES} B of the proof"
    # The block table lists the bank's free and allocated blocks, at addresses from the bottom of the bank.
    allocated = [int(block["address"]) for block in view.block_table if block["allocated"] == "yes"]
    taken = view.total_bytes_per_bank - min(allocated, default=view.total_bytes_per_bank)
    if taken > PROOF_L1_FOREIGN_BYTES:
        return False, f"an L1 tensor {taken} B below the top of each bank, over {PROOF_L1_FOREIGN_BYTES} B"
    return True, ""
