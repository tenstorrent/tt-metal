# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Preparation of the tensors the TTNN ops consume.

Weight reorientation and expert packing run once at load and never touch the device; the rotary
tables and the attention mask are built on the host and returned on device, the tables once per
model through RotaryTables; the two reshapes take device tensors.

Two layouts carry the activation, and only the first crosses a block boundary:

    (B, 1, S, H)      every block boundary, and inside attention and pooling
    (1, 1, B*S, H)    inside the MoE layer, one flat token axis

Neither is a preference. Attention mixes tokens along S, so flattening the batch away lets one
text attend to another: PCC 0.71 against per-sequence attention, a wrong answer rather than a
less precise one. Pooling also reduces along S, and has to produce one mean per text rather
than one per batch.

The MoE layer forces the opposite: the routed expert pass gathers its rows from one (1, 1, T, H)
operand by token index.

So tt/moe.py flattens on entry and unflattens on exit, which is where the reference does its own
x.view(-1, H), and nothing else in the encoder reshapes: ttnn.linear, ttnn.layer_norm and
ttnn.gelu are token-wise and take the batch-separated form unchanged. That confines the round
trip to the six MoE layers, 12 reshapes rather than the 24 a flat-everywhere contract would
need around attention.

Sub-blocks take other shapes internally, (B, A, S, D) head-split, (B*A, 1, S, D) for rotary at
small batches and (1, 1, R, H) and (1, 1, R, F) across the experts, each produced and consumed by
the op that owns it.

The flat form keeps tokens batch-major, matching the reference's own x.view(-1, H), so the
router's per-token weights stay aligned with the expert outputs.

The round trip is exact at every length tested, including ones that are not tile multiples. It
is free only when B is 1 or S is a multiple of 32, where the batch rows concatenate in place
and the reshape is pure metadata. Otherwise the tile padding sits in different places in the
two layouts and the data is physically copied.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import torch

import ttnn

from models.common.tensor_utils import get_rot_transformation_mat
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import build_extended_attention_mask
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import SMALL_M_TILES
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import (
    ACTIVATION_DTYPE,
    LAYOUT,
    MEMORY_CONFIG,
)
from models.experimental.nomic_embed_text_v2_moe.tt.pooling import MASK_SUM_FLOOR
from models.ttt_compat.tt.rope import get_rot_mats


def to_device(
    tensor: torch.Tensor,
    device,
    dtype: ttnn.DataType = ACTIVATION_DTYPE,
    layout: ttnn.Layout = LAYOUT,
    memory_config: ttnn.MemoryConfig = MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Move a torch tensor onto the device under this port's defaults."""
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


# Up to this many tile rows of M the residual stream (the block and embedding norms' outputs) and
# the expert sum go to L1, where the next op reads them: 20 to 36 us a forward from 1x32 to 1x320,
# measured with the dense experts the routed ones replaced; 3% to 4.8% of a forward from 2048 to
# 8192 tokens (2026-10-27), the norms 31% faster. Every op then plans its buffers below the residual
# (matmul_config.l1_floor), and each block frees its input as soon as norm1 has read it, which keeps
# the residual near the top of L1.
ACTIVATION_L1_MAX_TILES = 256


def m_tiles(shape: ttnn.Shape | tuple[int, ...]) -> int:
    """Tile rows of M of an activation, each sequence at its tile-padded length: B * ceil(S / 32) for
    (B, 1, S, W), ceil(T / 32) for the flat (1, 1, T, W)."""
    return shape[0] * ttnn.core.divup(shape[-2], ttnn.TILE_SIZE)


def activation_memory_config(shape: ttnn.Shape | tuple[int, ...]) -> ttnn.MemoryConfig:
    """L1 for an activation of at most ACTIVATION_L1_MAX_TILES tile rows of M, DRAM above.

    M counts each sequence at its tile-padded length: B * ceil(S / 32) rows for (B, 1, S, W), and
    ceil(T / 32) for the flat (1, 1, T, W). It takes a shape, so an op's output can be placed before
    it exists.
    """
    return ttnn.L1_MEMORY_CONFIG if m_tiles(shape) <= ACTIVATION_L1_MAX_TILES else ttnn.DRAM_MEMORY_CONFIG


def transpose_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """Reorient an (out_features, in_features) checkpoint weight for ttnn.linear.

    torch applies x @ w.T; ttnn.linear multiplies by the weight as given, so it wants
    (in_features, out_features).

    Skipping this raises on four of the five projections. attn.out_proj is 768x768, so there the
    untransposed weight typechecks and computes noise instead: PCC 0.001 to 0.004 across seeds,
    uncorrelated in every one.

    contiguous() is not required, since ttnn.from_torch reads the strided view correctly. It
    keeps the copy explicit and at load time.
    """
    return weight.transpose(0, 1).contiguous()


def pack_expert_weights(
    w1: torch.Tensor, w2: torch.Tensor, config: NomicMoEConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the two (E*F, H) expert blocks into broadcast-batch matmul operands.

    The expert axis is outer, each expert's slab stored (F, H). w1 is applied transposed and w2
    is not, so only w1 needs the transpose. The results are (1, E, H, F) and (1, E, F, H), ready
    to broadcast against a (1, 1, T, H) activation.

    Viewing either as (E, H, F) is an equally legal reshape, since E*F*H is symmetric in F and H.
    In torch that mistake is silent: the wrong slab plus a .T has the right shape and returns
    noise, which test_w2_transposed_view_typechecks_but_is_garbage pins. The 4D operand is what
    makes it loud, since there is no .T to paper over it and the inner dimensions stop agreeing;
    test_transposed_expert_weights_are_a_shape_error asserts it raises. TtNomicExperts transposes
    w2 once more for its own programs, back to the (E, H, F) shape of the mistake, so there the
    module PCC tests are the guard.
    """
    expert_shape = (config.num_experts, config.intermediate_size, config.hidden_size)
    return (
        w1.view(*expert_shape).transpose(1, 2).unsqueeze(0).contiguous(),
        w2.view(*expert_shape).unsqueeze(0).contiguous(),
    )


def stack_expert_columns(per_expert: torch.Tensor) -> torch.Tensor:
    """(1, E, R, C) per-expert slabs -> (1, 1, R, E*C), expert e in columns e*C .. (e+1)*C.

    The slabs sit side by side along the last axis, so a bfloat8_b conversion groups the same 16
    values per exponent as it does in the per-expert tensor, C being a multiple of 16: both forms
    hold the same quantized weights.
    """
    _, experts, rows, cols = per_expert.shape
    return per_expert[0].permute(1, 0, 2).reshape(1, 1, rows, experts * cols).contiguous()


def additive_attention_mask(
    attention_mask: torch.Tensor,
    device,
    dtype: ttnn.DataType = ACTIVATION_DTYPE,
    mask_dtype: ttnn.DataType | None = None,
) -> ttnn.Tensor:
    """Turn a (B, S) keep-mask into the (B, 1, S, S) additive mask SDPA takes, on device.

    Shares build_extended_attention_mask with the reference rather than restating it, so the two
    sides derive the fill value the same way, then materialises the query axis. Torch SDPA
    broadcasts a (B, 1, 1, S) mask over queries; ttnn's rejects it with
    mask_shape[2] == q_shape[2], so only the head axis may stay singleton.

    The tile padding has to be dtype-min too, not the 0 a TILE conversion defaults to. S rounds
    up to a multiple of 32, and 0 means "attend here", so SDPA otherwise counts the pad columns
    in the softmax denominator: at S=37 the output norm drops to 0.69x. PCC is nearly blind to
    it, moving only 0.9998 to 0.9974, so the guard is
    test_all_ones_mask_matches_no_mask rather than a PCC gate.

    attention_mask holds 1 for real tokens and 0 for padding. dtype has to match the SDPA
    operands, an fp32 mask against bf16 q/k/v being rejected, and the fill value is built in
    that dtype rather than cast down to it: fp32's finfo.min is outside bf16's range and would
    round to -inf. The result is 0.0 at real-token keys and dtype-min everywhere else.

    mask_dtype, when given, is the dtype the mask is cast to after tiling: SDPA also takes a
    bfloat8_b or bfloat4_b mask, whose shared exponents hold 0 and dtype-min exactly.
    """
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    seqlen = attention_mask.shape[-1]
    mask = build_extended_attention_mask(attention_mask, torch_dtype).expand(-1, -1, seqlen, -1)
    row_major = ttnn.from_torch(mask.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tiled = ttnn.to_layout(row_major, LAYOUT, pad_value=float(torch.finfo(torch_dtype).min))
    if mask_dtype is None or mask_dtype == dtype:
        return tiled
    cast = ttnn.typecast(tiled, mask_dtype)
    ttnn.deallocate(tiled)
    return cast


def rotary_pair_permutation(head_dim: int) -> torch.Tensor:
    """The head-dim order rotary_embedding_llama rotates in: new 2i holds old i, new 2i + 1 old i + D/2.

    The reference rotates dimension i with i + D/2 (NeoX rotate-half); the llama kernel rotates
    adjacent pairs, through a 32 x 32 matmul. q and k permuted alike leave q.k unchanged.
    """
    half = head_dim // 2
    return torch.stack((torch.arange(half), torch.arange(half) + half), dim=1).flatten()


def rotary_tables(
    device, config: NomicMoEConfig, seqlen: int, dtype: ttnn.DataType = ACTIVATION_DTYPE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build the (1, 1, S, head_dim) cos/sin tables rotary_embedding_llama reads, in L1.

    Delegates to ttt_compat's get_rot_mats, whose unscaled path computes the same inv_freq and
    outer product as NomicBertRotaryEmbedding and widens each frequency into an adjacent pair,
    [c0, c0, c1, c1, ...]: the reference's concat-widened [c0 .. c31, c0 .. c31] in
    rotary_pair_permutation's order. test_rotary_tables_match_the_reference holds them bit-exact in fp32.

    The widening has to match the pairing of q and k: concat-widened tables under the adjacent-pair
    kernel are not a rotation and do not preserve the per-plane norm.

    L1 because every core of the op reads them, and re-reads them every tile row once it has more than
    eight. The two tables are 64 KB each at S=512.
    """
    tables = []
    for table in get_rot_mats(
        head_dim=config.head_dim,
        device=device,
        seq_len=seqlen,
        theta=config.rotary_emb_base,
        rope_scaling=None,
        datatype=dtype,
    ):
        tables.append(ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG))
        ttnn.deallocate(table)
    cos, sin = tables
    return cos, sin


def rotary_rotation_matrix(device, grid: ttnn.CoreCoord) -> ttnn.Tensor:
    """rotary_embedding_llama's 32 x 32 rotation matrix, one copy in every core's L1, 2 KB a core.

    With a copy on every core of the grid the prefill program reads it in place. From one DRAM copy
    every core fetches it from one bank: rotary took 152 against 83 us a forward at 1x32, 684 against
    624 at 8x512.
    """
    cores = grid.x * grid.y
    return ttnn.from_torch(
        get_rot_transformation_mat().repeat(1, 1, cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )


class RotaryTables:
    """The cos/sin tables of the longest sequence seen, kept on device and handed out per call.

    Built per forward on the host they cost 0.17 ms and left the device idle for 0.66 ms at 8x512
    before the first block. The rotary op accepts tables longer than the sequence, but their tile
    padding is load-bearing: the rotation multiplies the padding rows of q and k by it, and only
    zeros there keep those rows finite. The MoE layer's unflatten_tokens leaves its output's padding
    unwritten; that reaches q and k as inf one block later, and with real cos/sin in the padding the
    inf keys broke SDPA on every row of the shorter sequence of a padded batch (two retrieval texts
    at 1 - cos 0.94 and 0.70). So a sequence on the tile grid, which has no padding rows, reads the
    long tables as they are, and any other gets them cut to S on device with the padding zeroed:
    four small ops and no host work. test_rotary_zeroes_the_padding_of_q_only_with_zero_padded_tables
    and test_sdpa_is_moved_by_non_finite_padded_keys pin the two halves of the hazard.
    """

    def __init__(self, device, config: NomicMoEConfig, dtype: ttnn.DataType = ACTIVATION_DTYPE):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.seqlen = 0
        self.tables: tuple[ttnn.Tensor, ttnn.Tensor] | None = None

    def __call__(self, seqlen: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(cos, sin) for a sequence of seqlen; hand the pair back to release() once the forward is done."""
        if seqlen > self.seqlen:
            if self.tables is not None:
                for table in self.tables:
                    ttnn.deallocate(table)
            self.tables = rotary_tables(self.device, self.config, seqlen, dtype=self.dtype)
            self.seqlen = seqlen
        if seqlen == self.seqlen or seqlen % ttnn.TILE_SIZE == 0:
            return self.tables
        cut = []
        for table in self.tables:
            sliced = ttnn.slice(
                table, [0, 0, 0, 0], [1, 1, seqlen, table.shape[-1]], memory_config=ttnn.L1_MEMORY_CONFIG
            )
            # In place: the fill returns a tensor on the same buffer.
            cut.append(ttnn.fill_implicit_tile_padding(sliced, 0.0))
        return tuple(cut)

    def release(self, tables: tuple[ttnn.Tensor, ttnn.Tensor]) -> None:
        """Free a pair __call__ cut for one forward; the kept pair stays."""
        if tables is not self.tables:
            for table in tables:
                ttnn.deallocate(table)


def table_tensors(tables) -> tuple[ttnn.Tensor, ...]:
    """The device tensors of a generated op's prepared object: a tensor, a dataclass holding them (such as
    nomic_qkv_rotary_heads' RotaryTables), or any nesting of tuples and lists of those."""
    if isinstance(tables, ttnn.Tensor):
        return (tables,)
    if dataclasses.is_dataclass(tables) and not isinstance(tables, type):
        return tuple(
            tensor for field in dataclasses.fields(tables) for tensor in table_tensors(getattr(tables, field.name))
        )
    if isinstance(tables, (tuple, list)):
        return tuple(tensor for table in tables for tensor in table_tensors(table))
    return ()


# The generated attention front (ttnn.operations.nomic_qkv_rotary_heads) covers 32 to 128 tile rows of M.
FUSED_ATTENTION_MIN_TILES, FUSED_ATTENTION_MAX_TILES = 32, 128


class FusedRotaryTables:
    """nomic_qkv_rotary_heads' tables (prepare_rotary_tables, its own format) for the longest sequence seen, kept on
    the device and shared by the 12 layers: the op takes tables at least as long as S, and zeroes the padding rows of
    q, k and v itself, so a shorter pass reads the long tables as they are.

    Built before the pass's activations, as RotaryTables are (tt.model): L1 tensors are allocated from the top
    down, and tables made after the embeddings would sit below the residual stream.
    """

    def __init__(self, device):
        self.device = device
        self.seqlen = None
        self.tables = None

    def __call__(self, seqlen: int):
        if self.seqlen is None or seqlen > self.seqlen:
            self.release()
            from ttnn.operations.nomic_qkv_rotary_heads import prepare_rotary_tables

            self.tables = prepare_rotary_tables(seqlen, self.device)
            self.seqlen = seqlen
        return self.tables

    def release(self) -> None:
        for tensor in table_tensors(self.tables):
            ttnn.deallocate(tensor)
        self.seqlen, self.tables = None, None


# Up to this many tile rows of M a layer norm splits the hidden axis over a block-sharded copy, in column-major
# order: SHARDED_NORM_COLUMNS core rows take 3 of the 24 hidden tiles each, and M spreads over the grid's columns,
# ceil(M / columns) tile rows a core, where the interleaved program gives each core whole rows. At 2x512 (32 tile
# rows) a norm takes 12.7 to 13.0 us against 20.4, and the 25 norms with their conversions 47 us less a forward,
# each norm's split output the next one's residual (tt/block.py). At 64 tile rows the conversions outweigh the
# gain (a block's two norms 56.8 against 45.8 us), and from 192 no split fits L1 (2026-10-27).
SHARDED_NORM_MAX_TILES = 32
SHARDED_NORM_COLUMNS = 8
# Above matmul_config.SMALL_M_TILES and up to this many tile rows the split norm runs without converting its
# output back: out_proj, fc2 and the MoE combine write its layout, and QKV and fc1 read its output as it stands. At
# 128 tile rows the norm takes 30.2 against 44.0 us interleaved, and its neighbours take the layout at no cost.
SHARDED_NORM_DIRECT_MAX_TILES = 128


def sharded_norm_statistics_scale(width: int) -> float:
    """How many times the true mean and variance the block-sharded program computes over `width` columns.

    It scales its sums by C/W on each core and 1/C across them, each rounded to bfloat16, C being
    SHARDED_NORM_COLUMNS. At W = 768 = 3 x 256 one of the two carries a factor 1/3, which bfloat16 rounds
    up by 2^-9; uncorrected, every norm output then shrinks by 0.1%, and the norm's error against
    float64 is 2.08x the bfloat16 floor against 1.91x interleaved. LayerNormParameters multiplies gamma
    by the square root, kept in float32 at the 10-bit mantissa the FPU reads it at: 1.53x (2026-10-22).
    What remains is (scale - 1) x a row's mean over its standard deviation, under 0.2 at every norm of
    this model.
    """

    def rounding(value: float) -> float:
        return torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).item() / value

    return rounding(SHARDED_NORM_COLUMNS / width) * rounding(1 / SHARDED_NORM_COLUMNS)


def round_to_tf32(tensor: torch.Tensor) -> torch.Tensor:
    """float32 rounded to nearest even at TF32's 10-bit mantissa, the precision the FPU reads an operand at."""
    bits = tensor.float().contiguous().view(torch.int32)
    bits = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return bits.view(torch.float32)


# LayerNormParameters' crossover from the tiled weight and bias to the row-major ones, in tile rows of M.
NORM_ROW_MAJOR_ABOVE_TILES = 32


class LayerNormParameters:
    """One layer norm's weight and bias, in each of the two forms the norm reads fastest by input size.

    ttnn.layer_norm has every core read the whole weight and bias. Tiled, a (W,) parameter is padded
    to 32 rows, 48 KB at W=768, and all 110 cores read it from the same DRAM banks: 0.48 ms of the
    1.94 ms the norms took a forward at 8x512. As (1, 1, W/32, 32) row-major rows in L1 the read is
    1.5 KB a core, but issued a 64-byte row per barrier, about 1 us a call where a core holds one or
    two tile rows. The crossover sits between 32 tile rows of M (+4% for row-major) and 64 (-18% to
    -27%), so the norm reads the row-major copy above NORM_ROW_MAJOR_ABOVE_TILES. The 50 row-major tensors take 0.7 KB of each L1 bank; tiled in L1 they took 22 KB and
    clashed with the L1 plan of SDPA's buffers.

    The block-sharded norm below SHARDED_NORM_MAX_TILES reads `sharded`: gamma corrected for that
    program's statistics scale, in float32, and the tiled bias.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, device, dtype: ttnn.DataType):
        self.tiled = tuple(to_device(tensor, device, dtype=dtype) for tensor in (weight, bias))
        corrected = round_to_tf32(weight.float() * sharded_norm_statistics_scale(weight.numel()) ** 0.5)
        self.sharded = (to_device(corrected, device, dtype=ttnn.float32), self.tiled[1])
        # The generated norm (nd_norm_layout) computes exact statistics, so it reads the plain gamma, float32 at the
        # mantissa the FPU reads, and the tiled bias.
        self.plain = (to_device(round_to_tf32(weight.float()), device, dtype=ttnn.float32), self.tiled[1])
        self.row_major = tuple(
            to_device(
                tensor.reshape(1, 1, -1, ttnn.TILE_SIZE),
                device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for tensor in (weight, bias)
        )

    def for_input(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(weight, bias) for a layer norm over x, (B, 1, S, W)."""
        return self.row_major if m_tiles(x.shape) > NORM_ROW_MAJOR_ABOVE_TILES else self.tiled


def sharded_norm_split(m_tiles: int, width_tiles: int, grid: ttnn.CoreCoord) -> tuple[int, int] | None:
    """(columns of cores, tile rows of M a core) of the hidden-axis split, or None for the interleaved call."""
    if m_tiles > SHARDED_NORM_DIRECT_MAX_TILES or SHARDED_NORM_COLUMNS > grid.y or width_tiles % SHARDED_NORM_COLUMNS:
        return None
    block_h = ttnn.core.divup(m_tiles, grid.x)
    return ttnn.core.divup(m_tiles, block_h), block_h


@cache
def sharded_norm_configs(
    core_columns: int, block_h: int, width_tiles: int, inplace: bool = False
) -> tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]:
    """The block-sharded layout and program of a layer norm on SHARDED_NORM_COLUMNS x core_columns cores.

    Column-major, the hidden axis on the grid's rows and M on its columns. Built directly:
    create_sharded_memory_config takes a column-major shard shape as (width, height). inplace writes the
    output into the input's buffer, bit for bit the same.
    """
    block_w = width_tiles // SHARDED_NORM_COLUMNS
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_columns - 1, SHARDED_NORM_COLUMNS - 1))}
    )
    shard = ttnn.ShardSpec(grid, (block_h * ttnn.TILE_SIZE, block_w * ttnn.TILE_SIZE), ttnn.ShardOrientation.COL_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_columns, SHARDED_NORM_COLUMNS),
        subblock_w=block_w,
        block_h=block_h,
        block_w=block_w,
        inplace=inplace,
    )
    return memory_config, program_config


def sharded_norm_layout(
    shape: ttnn.Shape | tuple[int, ...], grid: ttnn.CoreCoord
) -> tuple[ttnn.MemoryConfig, int, int, int] | None:
    """(memory config, core columns, tile rows a core, core rows) of the split norm over an activation of
    `shape`, for a producer to write its output in, or None where the norm runs interleaved."""
    width_tiles = shape[-1] // ttnn.TILE_SIZE
    split = sharded_norm_split(m_tiles(shape), width_tiles, grid)
    if split is None:
        return None
    return sharded_norm_configs(*split, width_tiles)[0], *split, SHARDED_NORM_COLUMNS


# The generated layer norm (ttnn/ttnn/operations/nomic_residual_layer_norm) covers 32 to 128 tile rows of M on the
# 11x10 grid, in its own ND-sharded layout: 2 or 3 cores a row group, 1 to 4 tile rows a core, picked per shape.
# Every op on the residual stream then reads and writes that layout (tt.attention, tt.mlp, tt.moe), and a
# forward either runs entirely on it or not at all.
ND_NORM_MIN_TILES, ND_NORM_MAX_TILES = 32, 128


class _GridOnly:
    """What the generated ops' layout functions read of a device: its compute grid."""

    def __init__(self, x: int, y: int):
        self._grid = ttnn.CoreCoord(x, y)

    def compute_with_storage_grid_size(self) -> ttnn.CoreCoord:
        return self._grid


@cache
def _nd_norm_layout(shape: tuple[int, ...], grid_x: int, grid_y: int) -> ttnn.MemoryConfig | None:
    from ttnn.operations._op_contract import SupportRefusal
    from ttnn.operations.nomic_residual_layer_norm import input_memory_config

    try:
        return input_memory_config(shape, _GridOnly(grid_x, grid_y))
    except SupportRefusal:
        return None


def nd_norm_layout(shape: ttnn.Shape | tuple[int, ...], tt_config) -> ttnn.MemoryConfig | None:
    """The generated norm's ND-sharded layout for an activation of `shape`, (B, 1, S, H), or None where the
    residual stream keeps the split layout: tt_config.nd_residual off, M outside 32 to 128 tile rows, or a batch of
    sequences off the tile grid, whose MoE flatten would copy an ND tensor (tt.moe)."""
    if not tt_config.nd_residual:
        return None
    batch, seqlen = shape[0], shape[-2]
    if not ND_NORM_MIN_TILES <= m_tiles(shape) <= ND_NORM_MAX_TILES or (batch > 1 and seqlen % ttnn.TILE_SIZE):
        return None
    grid = tt_config.core_grid
    return _nd_norm_layout((shape[0], shape[1], shape[2], shape[3]), grid.x, grid.y)


def _nd_layer_norm(
    x: ttnn.Tensor,
    parameters: LayerNormParameters,
    nd_layout: ttnn.MemoryConfig,
    *,
    epsilon: float,
    memory_config: ttnn.MemoryConfig,
    residual: ttnn.Tensor | None,
    keep_sharded: bool,
    consume_input: bool,
    in_place: bool,
) -> ttnn.Tensor | tuple[ttnn.Tensor, None]:
    """layer_norm on the generated norm, x and the residual in (or converted to) nd_layout, the output in nd_layout
    unless memory_config is DRAM, where it is copied (the encoder's last output). keep_sharded returns (out, None):
    the output is already every reader's layout."""
    from ttnn.operations.nomic_residual_layer_norm import nomic_residual_layer_norm

    x_nd = x if x.memory_config() == nd_layout else ttnn.to_memory_config(x, nd_layout)
    converted = x_nd is not x
    residual_nd = residual
    if residual is not None and residual.memory_config() != nd_layout:
        residual_nd = ttnn.to_memory_config(residual, nd_layout)
    gamma, beta = parameters.plain
    # In place on a converted x, or on a consumed one unless in_place is False: the output takes x's slot, which
    # decides the L1 plans after it (tt/block.py).
    overwrite = converted or (consume_input and in_place)
    out = nomic_residual_layer_norm(
        x_nd, gamma, beta, residual=residual_nd, epsilon=epsilon, output_tensor=x_nd if overwrite else None
    )
    if consume_input:
        if converted:
            ttnn.deallocate(x)
        elif not overwrite:
            ttnn.deallocate(x_nd)
    if residual_nd is not residual:
        ttnn.deallocate(residual_nd)
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        moved = ttnn.to_memory_config(out, memory_config)
        ttnn.deallocate(out)
        out = moved
    return (out, None) if keep_sharded else out


def layer_norm(
    x: ttnn.Tensor,
    parameters: LayerNormParameters,
    *,
    epsilon: float,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    residual: ttnn.Tensor | None = None,
    keep_sharded: bool = False,
    consume_input: bool = False,
    in_place: bool = True,
    nd_layout: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor | None]:
    """layer_norm(x + residual) over the last axis of x, (B, 1, S, W), written where memory_config says.

    With nd_layout (nd_norm_layout's, for x's shape) it runs on the generated norm in that layout instead
    (_nd_layer_norm), and the rest of this docstring is the split layout's.

    Up to SHARDED_NORM_DIRECT_MAX_TILES tile rows of M it runs block-sharded with the corrected gamma,
    converting x and the residual to that layout and the output back. An x or residual already in it,
    such as a previous call's keep_sharded output for the same shape, is taken as it is. Above SMALL_M_TILES,
    where the dense projections are the port's own matmuls, a keep_sharded output is not converted back: its
    readers take the layout (at 32 tile rows 2.8 us a conversion, 18 a forward).

    consume_input hands x over: it is freed here, or overwritten by the output. Above SHARDED_NORM_MAX_TILES
    the split norm runs in place on a converted x, and on a consumed one unless in_place is False. Where the
    output lands decides the L1 plans after it: in place it takes x's slot, otherwise the highest free block
    it fits, below x and the residual or in a hole above them (tt/block.py).

    Returns:
        The output; with keep_sharded=True, (output, its block-sharded form, or None where there is
        none or the output is already in that layout).
    """
    if nd_layout is not None:
        return _nd_layer_norm(
            x,
            parameters,
            nd_layout,
            epsilon=epsilon,
            memory_config=memory_config,
            residual=residual,
            keep_sharded=keep_sharded,
            consume_input=consume_input,
            in_place=in_place,
        )
    width_tiles = x.shape[-1] // ttnn.TILE_SIZE
    rows = m_tiles(x.shape)
    split = sharded_norm_split(rows, width_tiles, x.device().compute_with_storage_grid_size())
    if split is None:
        weight, bias = parameters.for_input(x)
        out = ttnn.layer_norm(
            x,
            residual_input_tensor=residual,
            weight=weight,
            bias=bias,
            epsilon=epsilon,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )
        if consume_input:
            ttnn.deallocate(x)
        return (out, None) if keep_sharded else out

    sharded_config = sharded_norm_configs(*split, width_tiles)[0]
    x_sharded = x if x.memory_config() == sharded_config else ttnn.to_memory_config(x, sharded_config)
    converted = x_sharded is not x
    inplace = rows > SHARDED_NORM_MAX_TILES and (converted or (consume_input and in_place))
    program_config = sharded_norm_configs(*split, width_tiles, inplace)[1]
    residual_sharded = residual
    if residual is not None and residual.memory_config() != sharded_config:
        residual_sharded = ttnn.to_memory_config(residual, sharded_config)
    weight, bias = parameters.sharded
    out_sharded = ttnn.layer_norm(
        x_sharded,
        residual_input_tensor=residual_sharded,
        weight=weight,
        bias=bias,
        epsilon=epsilon,
        memory_config=sharded_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    if not inplace and (converted or consume_input):
        ttnn.deallocate(x_sharded)
    if converted and consume_input:
        ttnn.deallocate(x)
    if residual_sharded is not residual:
        ttnn.deallocate(residual_sharded)
    if keep_sharded and rows > SMALL_M_TILES:
        return out_sharded, None
    out = ttnn.to_memory_config(out_sharded, memory_config)
    if keep_sharded:
        return out, out_sharded
    ttnn.deallocate(out_sharded)
    return out


def flatten_tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """(B, 1, S, H) -> (1, 1, B*S, H), batch-major. At B = 1 that is x itself, and the call is skipped."""
    batch = x.shape[0]
    if batch == 1:
        return x
    return ttnn.reshape(x, (1, 1, batch * x.shape[-2], x.shape[-1]))


def unflatten_tokens(x: ttnn.Tensor, batch: int, seqlen: int) -> ttnn.Tensor:
    """(1, 1, B*S, H) -> (B, 1, S, H). B and S are arguments because the flat form has lost them.

    Where the reshape copies (B > 1, S off the tile grid) it leaves the tile padding of the copy
    unwritten, up to 4e37 measured. Nothing reads that padding into a logical value; RotaryTables
    has why that holds for q and k. At B = 1 there is nothing to reshape.
    """
    if batch == 1:
        return x
    return ttnn.reshape(x, (batch, 1, seqlen, x.shape[-1]))


def prepare_token_ids(input_ids: torch.Tensor, device) -> ttnn.Tensor:
    """Move (B, S) token ids onto the device in the form ttnn.embedding indexes with.

    ttnn.embedding requires a uint32 index tensor in ROW_MAJOR; a TILE index or an int32 one is
    rejected. The ids are bounded by the tokenizer's 250002 entries, so uint32 is lossless.
    """
    return ttnn.from_torch(input_ids.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def pooling_mask(attention_mask: torch.Tensor, device, dtype: ttnn.DataType = ttnn.float32) -> ttnn.Tensor:
    """Turn a (B, S) keep-mask into the (B, 1, 1, S) weights mean_pool multiplies the tokens by.

    Each row is mask / count: 1/count at real tokens, 0 at padding. The counts are known here on the
    host, so the divisor is folded into the weights and the mean is one batched matmul. The count is
    floored as reference/postprocessing.mean_pool floors it, so a row that keeps nothing gets zero
    weights rather than a division by zero. fp32 so that 1/count does not round to bfloat16.
    """
    batch, seqlen = attention_mask.shape
    mask = attention_mask.float()
    weights = mask / mask.sum(dim=1, keepdim=True).clamp(min=MASK_SUM_FLOOR)
    return to_device(weights.reshape(batch, 1, 1, seqlen), device, dtype=dtype)
