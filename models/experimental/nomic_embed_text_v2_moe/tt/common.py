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

The MoE matmuls force the opposite. sparse_matmul takes the tokens as one (1, 1, T, H) operand
against the (1, E, H, F) weights, and a transposed pass multiplies the stacked weight by one
(1, 1, H, T) x^T. The spare dim at position 1 is what the expert axis expands into and what
fast_reduce_nc collapses again.

So tt/moe.py flattens on entry and unflattens on exit, which is where the reference does its own
x.view(-1, H), and nothing else in the encoder reshapes: ttnn.linear, ttnn.layer_norm and
ttnn.gelu are token-wise and take the batch-separated form unchanged. That confines the round
trip to the six MoE layers, 12 reshapes rather than the 24 a flat-everywhere contract would
need around attention.

Sub-blocks take other shapes internally, (B, A, S, D) head-split, (1, B*A, S, D) for rotary and
(1, E, T, F) across the experts, each produced and consumed by the op that owns it.

The flat form keeps tokens batch-major, matching the reference's own x.view(-1, H), so the
router's per-token weights stay aligned with the expert outputs.

The round trip is exact at every length tested, including ones that are not tile multiples. It
is free only when B is 1 or S is a multiple of 32, where the batch rows concatenate in place
and the reshape is pure metadata. Otherwise the tile padding sits in different places in the
two layouts and the data is physically copied.
"""

from __future__ import annotations

import torch

import ttnn

from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import build_extended_attention_mask
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import SMALL_M_TILES
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import (
    ACTIVATION_DTYPE,
    LAYOUT,
    MEMORY_CONFIG,
)
from models.experimental.nomic_embed_text_v2_moe.tt.pooling import MASK_SUM_FLOOR
from models.tt_transformers.tt.rope import get_rot_mats_hf


def to_device(
    tensor: torch.Tensor,
    device,
    dtype: ttnn.DataType = ACTIVATION_DTYPE,
    layout: ttnn.Layout = LAYOUT,
    memory_config: ttnn.MemoryConfig = MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Move a torch tensor onto the device under this port's defaults."""
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


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


def rotary_tables(
    device, config: NomicMoEConfig, seqlen: int, dtype: ttnn.DataType = ACTIVATION_DTYPE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build the (1, 1, S, head_dim) cos/sin tables rotary_embedding_hf reads, in L1.

    Delegates to tt_transformers' get_rot_mats_hf, whose unscaled path computes the same
    inv_freq, outer product and concat widening as NomicBertRotaryEmbedding plus
    apply_rotary_emb. test_rotary_tables_match_the_reference holds them bit-exact in fp32.

    The concat widening is load-bearing: interleaving gives the GPT-J lane pairing, which under
    the kernel's NeoX rotate-half is not a rotation and does not preserve the per-plane norm.

    L1 because rotary_embedding_hf reads the tables once per folded head: from DRAM they held it at
    55% of the DRAM peak (tt-npe) although q, k and its output are in L1. In L1 rotary takes 0.97 ms
    a forward at 8x512 against 1.27. The two tables are 64 KB each at S=512.
    """
    tables = []
    for table in get_rot_mats_hf(
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


class LayerNormParameters:
    """One layer norm's weight and bias, in each of the two forms the norm reads fastest by input size.

    ttnn.layer_norm has every core read the whole weight and bias. Tiled, a (W,) parameter is padded
    to 32 rows, 48 KB at W=768, and all 110 cores read it from the same DRAM banks: 0.48 ms of the
    1.94 ms the norms took a forward at 8x512. As (1, 1, W/32, 32) row-major rows in L1 the read is
    1.5 KB a core, but issued a 64-byte row per barrier, about 1 us a call where a core holds one or
    two tile rows. The crossover sits between 32 tile rows of M (+4% for row-major) and 64 (-18% to
    -27%), so the norm reads the row-major copy above SMALL_M_TILES, the dense matmuls' small-M
    threshold. The 50 row-major tensors take 0.7 KB of each L1 bank; tiled in L1 they took 22 KB and
    clashed with the L1 plan of SDPA's buffers.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, device, dtype: ttnn.DataType):
        self.tiled = tuple(to_device(tensor, device, dtype=dtype) for tensor in (weight, bias))
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
        batch, _, seqlen, _ = x.shape
        return self.row_major if batch * ttnn.core.divup(seqlen, ttnn.TILE_SIZE) > SMALL_M_TILES else self.tiled


def flatten_tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """(B, 1, S, H) -> (1, 1, B*S, H), batch-major."""
    batch, _, seqlen, hidden = x.shape
    return ttnn.reshape(x, (1, 1, batch * seqlen, hidden))


def unflatten_tokens(x: ttnn.Tensor, batch: int, seqlen: int) -> ttnn.Tensor:
    """(1, 1, B*S, H) -> (B, 1, S, H). B and S are arguments because the flat form has lost them.

    Where the reshape copies (B > 1, S off the tile grid) it leaves the tile padding of the copy
    unwritten, up to 4e37 measured. Nothing reads that padding into a logical value; RotaryTables
    has why that holds for q and k.
    """
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
