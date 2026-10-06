# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional MHA, the TTNN form of reference.NomicBertAttention.

    x (B, 1, S, H)
      Wqkv                    -> (B, 1, S, 3H)   three-major: [q | k | v]
      nlp_create_qkv_heads    -> three (B, A, S, D)
      rotary_embedding_hf     -> q and k rotated, v untouched
      SDPA, is_causal=False   -> (B, A, S, D)
      nlp_concat_heads        -> (B, 1, S, H)
      out_proj                -> (B, 1, S, H)

The rotary tables and the additive mask come from the caller, not from here: both depend only on
S, and building them per block would repeat the same host work 12 times.
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import dense_linear
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup

# The largest fp32 score block, in tiles, that fit L1 beside the rest of SDPA's buffers.
_FP32_SCORE_TILES = 144


def _round_up_to_tile(n: int) -> int:
    return ttnn.core.divup(n, ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _balanced_chunk(length: int, limit: int) -> int:
    """The tile-multiple chunk covering `length` in the fewest chunks of at most `limit`, split evenly."""
    return _round_up_to_tile(ttnn.core.divup(length, ttnn.core.divup(length, limit)))


@cache
def sdpa_program_config(
    batch: int, seqlen: int, heads: int, grid: ttnn.CoreCoord, fp32_dest_acc: bool, masked: bool
) -> ttnn.SDPAProgramConfig:
    """Query and key chunk sizes for one (B, A, S, D) call.

    SDPA hands each core a contiguous run of the B * A * ceil(S / q_chunk) query chunks. The
    fastest query chunk measured is the smallest that still leaves no core a second one, and with
    more heads than cores the whole sequence, up to 512; a whole-sequence key chunk suits a
    whole-sequence query chunk, and 256 or 128 the rest. Over 54 shapes from 1x32 to 32x512 this
    lands within 8% of the best pair, and on the best one at 8x512, 8x384, 1x128 and 2x37. A query
    chunk splits the sequence evenly, so 17 tiles run as two chunks of 9 rather than 16 and 1. A
    key chunk does not have to: the bfloat16 kernel skips the padded key tiles of the last chunk,
    and 256 ran faster than an even 192 at S=384.

    With fp32 accumulation the kernel keeps its score block in fp32, and past 144 tiles (384 x 384)
    it no longer fits L1: 480 x 480 and 512 x 512 fail to allocate. There the query chunk shrinks
    to a split of at most 256 while the heads fit on the grid, and the key chunk once they do not,
    each the faster of the two where measured. The cap holds with no mask or a bfloat4_b one; a
    bfloat16 mask fails at 256 x 512 too, and only attention_placement's fit check catches that.

    A mask adds a double-buffered block of the score's size. On the bfloat16 kernel that leaves
    512 x 512 unable to allocate, and with a mask every shape measured ran fastest, or within 7%,
    at a key chunk of at most 256.
    """
    whole = _round_up_to_tile(seqlen)
    chunks_per_head = (grid.x * grid.y) // (batch * heads)
    q_limit = 512 if not chunks_per_head else min(512, _round_up_to_tile(ttnn.core.divup(seqlen, chunks_per_head)))
    q_chunk = _balanced_chunk(whole, q_limit)
    if fp32_dest_acc:
        k_chunk = _balanced_chunk(whole, 512)
        if (q_chunk // ttnn.TILE_SIZE) * (k_chunk // ttnn.TILE_SIZE) > _FP32_SCORE_TILES:
            if chunks_per_head:
                q_chunk = _balanced_chunk(whole, 256)
            else:
                k_chunk = _balanced_chunk(whole, 256)
    elif q_chunk <= 64 and whole > 256:
        k_chunk = 128
    elif masked or (whole > 256 and q_chunk != whole):
        k_chunk = min(whole, 256)
    else:
        k_chunk = min(whole, 512)
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk)


def sdpa_circular_buffer_bytes(
    seqlen: int,
    q_chunk: int,
    k_chunk: int,
    head_dim: int,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    double_buffered_q: bool,
) -> int:
    """SDPA's circular-buffer bytes per core, as its program factory allocates them for bf16 q, k, v.

    A padded sequence without a mask gets a generated one: a single tile on the bfloat16 kernel,
    a full double-buffered block of bfloat4_b on the fp32 one. Against the two regions the
    allocator reported, 8x512 at 512 x 512 on the bfloat16 kernel and 8x450 at 256 x 480 on the
    fp32 one, this puts both regions' base at 110 KB to within 2 KB.
    """
    tile = ttnn.tile_size(ttnn.bfloat16)
    wide = ttnn.tile_size(ttnn.float32) if fp32_dest_acc else tile
    q_tiles, k_tiles, d_tiles = q_chunk // ttnn.TILE_SIZE, k_chunk // ttnn.TILE_SIZE, head_dim // ttnn.TILE_SIZE
    total = q_tiles * d_tiles * tile * (2 if double_buffered_q else 1)
    total += 2 * 2 * k_tiles * d_tiles * tile  # k and v, double-buffered
    total += q_tiles * k_tiles * wide  # the score block
    total += 2 * q_tiles * d_tiles * tile  # the two running outputs
    total += 3 * q_tiles * tile + 2 * q_tiles * wide  # the running maxima and their difference, the sums
    total += (q_tiles * d_tiles if fp32_dest_acc else 8) * tile  # the output block, a ping-pong when streaming
    total += 3 * tile  # the scalars
    padded = seqlen % q_chunk or seqlen % k_chunk
    if mask_dtype is not None:
        total += 2 * q_tiles * k_tiles * ttnn.tile_size(mask_dtype)
    elif padded:
        total += 2 * q_tiles * k_tiles * ttnn.tile_size(ttnn.bfloat4_b) if fp32_dest_acc else 2 * tile
    return total


# The L1 the head tensors may span beside SDPA's buffers, in head tensors: q, k, v and their rotated
# copies overlap, and the ones freed leave holes. Measured up to 5.9 at the allocator's clashes.
_HEAD_TENSOR_SPAN = 7

# Head room below the circular-buffer limit: the region started 6 KiB above the reported limit.
_L1_MARGIN_BYTES = 16 * 1024


@cache
def attention_placement(
    batch: int,
    seqlen: int,
    heads: int,
    head_dim: int,
    grid: ttnn.CoreCoord,
    fp32_dest_acc: bool,
    mask_dtype: ttnn.DataType | None,
    cb_bytes: int,
    banks: int,
) -> tuple[ttnn.SDPAProgramConfig, ttnn.MemoryConfig]:
    """SDPA's program config and where the head tensors and SDPA's output live, for one call.

    In L1 the head split, both rotary calls, SDPA and the head concat all read and write L1: at
    8x384 that took them from 3.46 to 2.30 ms a forward. They need room beside SDPA's buffers,
    which at 8x512 take 1.07 MB of a core on the fp32 kernel, so the key chunk shrinks, to 256 and
    then 128, before the tensors go back to DRAM. On the fp32 kernel the smaller chunks split the
    sequence evenly, since it computes the padded keys of its last chunk; the bfloat16 kernel
    skips them.

    The query chunk is never split instead, although q128 with a whole-sequence key chunk is faster
    beside L1 heads (63.5 us against q384/k192's 72.7 at 8x384): with three query chunks a head at
    10 or more sequences SDPA returns wrong rows (10x384, 12x384, 16x384, 12x352), in DRAM as in
    L1. See test_sdpa_where_three_query_chunks_a_head_are_wrong.
    """
    config = sdpa_program_config(batch, seqlen, heads, grid, fp32_dest_acc, mask_dtype is not None)
    whole = _round_up_to_tile(seqlen)
    head_tiles = batch * ttnn.core.divup(seqlen, ttnn.TILE_SIZE) * heads * head_dim // ttnn.TILE_SIZE
    head_bank = ttnn.core.divup(head_tiles, banks) * ttnn.tile_size(ttnn.bfloat16)
    chunks = batch * heads * ttnn.core.divup(seqlen, config.q_chunk_size)
    double_buffered_q = chunks > grid.x * grid.y
    budget = cb_bytes - _L1_MARGIN_BYTES - _HEAD_TENSOR_SPAN * head_bank
    smaller = (_balanced_chunk(whole, limit) if fp32_dest_acc else limit for limit in (256, 128))
    for k_chunk in [config.k_chunk_size, *(k for k in smaller if k < config.k_chunk_size)]:
        footprint = sdpa_circular_buffer_bytes(
            seqlen, config.q_chunk_size, k_chunk, head_dim, fp32_dest_acc, mask_dtype, double_buffered_q
        )
        if footprint <= budget:
            candidate = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid,
                q_chunk_size=config.q_chunk_size,
                k_chunk_size=k_chunk,
                exp_approx_mode=config.exp_approx_mode,
            )
            return candidate, ttnn.L1_MEMORY_CONFIG
    return config, ttnn.DRAM_MEMORY_CONFIG


class TtNomicBertAttention(LightweightModule):
    """Fused three-major QKV projection, full-head rotary, then bidirectional SDPA.

    Two defaults differ from torch and both fail silently:

      - ttnn's SDPA defaults is_causal to True where torch defaults it to False. Left alone it
        applies a decoder mask to an encoder and still returns finite output, PCC 0.44.
      - the mask must materialise the query axis as (B, 1, S, S). Torch broadcasts (B, 1, 1, S)
        over queries; ttnn rejects that shape outright, which at least is loud.

    SDPA's 1/sqrt(head_dim) scale is folded into the q third of Wqkv instead, and the call takes
    scale=1.0. Given a mask and any other scale, SDPA multiplies the mask by 1/scale on every call,
    24 us a layer at 8x512. 1/sqrt(64) is a power of two, so q comes out scaled to the bit.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        def weight(name, group, tensor=None):
            return to_device(
                transpose_linear_weight(state_dict[f"{state_dict_prefix}{name}.weight"] if tensor is None else tensor),
                device,
                dtype=tt_config.matmul_weight_dtype(group),
            )

        def bias(name, tensor=None):
            return to_device(
                state_dict[f"{state_dict_prefix}{name}.bias"] if tensor is None else tensor,
                device,
                dtype=tt_config.weight_dtype,
            )

        # Wqkv is three-major, so the q projection is its first hidden_size output rows.
        query_rows = slice(0, config.hidden_size)
        qkv_weight = state_dict[f"{state_dict_prefix}Wqkv.weight"].clone()
        qkv_bias = state_dict[f"{state_dict_prefix}Wqkv.bias"].clone()
        qkv_weight[query_rows] *= self.head_dim**-0.5
        qkv_bias[query_rows] *= self.head_dim**-0.5
        self.qkv_weight, self.qkv_bias = weight("Wqkv", OpGroup.QKV, qkv_weight), bias("Wqkv", qkv_bias)
        self.out_weight, self.out_bias = weight("out_proj", OpGroup.ATTN_OUT), bias("out_proj")

    def _rotate(
        self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """Apply rotary position embedding to one of q or k.

        rotary_embedding_hf's prefill mode wants a leading batch of 1, so the batch is folded
        into the head axis; cos/sin broadcast over it, applying the same table to every row.

        Args:
            x: (B, A, S, D) queries or keys.
            cos: (1, 1, S, D) cosine table from tt.common.RotaryTables, or longer when S is on the
                tile grid.
            sin: the sine table, the same shape.

        Returns:
            ttnn.Tensor: (B, A, S, D), rotated.
        """
        batch, heads, seqlen, head_dim = x.shape
        folded = ttnn.reshape(x, (1, batch * heads, seqlen, head_dim))
        rotated = ttnn.experimental.rotary_embedding_hf(
            folded, cos, sin, is_decode_mode=False, memory_config=memory_config
        )
        return ttnn.reshape(rotated, (batch, heads, seqlen, head_dim))

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Project, rotate, attend and project back.

        Args:
            x: (B, 1, S, H) block input.
            rot_mats: (cos, sin), each (1, 1, S, D) or longer, from tt.common.RotaryTables.
            attn_mask: (B, 1, S, S) additive mask from tt.common.additive_attention_mask, or
                None for no masking.

        Returns:
            ttnn.Tensor: (B, 1, S, H).
        """
        qkv = dense_linear(x, self.qkv_weight, self.qkv_bias, OpGroup.QKV, self.tt_config)

        compute_kernel_config = self.tt_config.compute_kernel_config(OpGroup.SDPA)
        batch, _, seqlen, _ = x.shape
        program_config = sdpa_program_config(
            batch,
            seqlen,
            self.num_heads,
            self.tt_config.core_grid,
            compute_kernel_config.fp32_dest_acc_en,
            attn_mask is not None,
        )
        heads_memory = ttnn.DRAM_MEMORY_CONFIG
        if self.tt_config.attention_l1:
            program_config, heads_memory = attention_placement(
                batch,
                seqlen,
                self.num_heads,
                self.head_dim,
                self.tt_config.core_grid,
                compute_kernel_config.fp32_dest_acc_en,
                None if attn_mask is None else attn_mask.dtype,
                self.tt_config.l1_cb_bytes,
                self.tt_config.l1_banks,
            )

        # Three-major, heads contiguous inside each of q, k and v. transpose_k_heads stays False
        # because SDPA wants K as (B, A, S, D), not pre-transposed. qkv may sit in L1
        # (dense_linear); the heads are placed explicitly rather than inherit that.
        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=heads_memory,
        )
        ttnn.deallocate(qkv)

        cos, sin = rot_mats
        rotated_query = self._rotate(query, cos, sin, heads_memory)
        rotated_key = self._rotate(key, cos, sin, heads_memory)
        # Freed here rather than inside _rotate: the reshape there aliases this buffer, so the
        # pre-rotation tensor is the one thing that owns it.
        ttnn.deallocate(query)
        ttnn.deallocate(key)

        context = ttnn.transformer.scaled_dot_product_attention(
            rotated_query,
            rotated_key,
            value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=heads_memory,
        )
        for tensor in (rotated_query, rotated_key, value):
            ttnn.deallocate(tensor)

        concatenated = ttnn.experimental.nlp_concat_heads(context, memory_config=heads_memory)
        ttnn.deallocate(context)

        out = dense_linear(concatenated, self.out_weight, self.out_bias, OpGroup.ATTN_OUT, self.tt_config)
        ttnn.deallocate(concatenated)
        return out
