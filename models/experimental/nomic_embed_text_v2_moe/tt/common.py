# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Preparation of the tensors the TTNN ops consume.

Weight reorientation and expert packing run once at load and never touch the device; the rotary
tables and the attention mask are built on the host and returned on device; the two reshapes
take device tensors.

Two layouts carry the activation between sub-blocks:

    (B, 1, S, H)      attention and pooling, where the batch axis has to stay separate
    (1, 1, B*S, H)    everything else, one flat token axis

Neither is a preference. Attention mixes tokens along S, so flattening the batch away lets one
text attend to another: PCC 0.71 against per-sequence attention, a wrong answer rather than a
less precise one. Pooling also reduces along S, and has to produce one mean per text rather
than one per batch.

The MoE matmuls force the opposite. ttnn.matmul broadcasts a weight's batch dims only when
every batch dim of the activation is 1, so (1, 1, T, H) x (1, E, H, F) gives (1, E, T, F) while
(B, 1, S, H) raises outright. The spare dim at position 1 is what the expert axis expands into
and what fast_reduce_nc collapses again.

Those two are what crosses a sub-block boundary, not every shape in the model. Sub-blocks take
others internally, (B, A, S, D) head-split, (1, B*A, S, D) for rotary and (1, E, T, F) across
the experts, each produced and consumed by the op that owns it.

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
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import (
    ACTIVATION_DTYPE,
    LAYOUT,
    MEMORY_CONFIG,
)
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
    test_transposed_expert_weights_are_a_shape_error asserts it raises.
    """
    expert_shape = (config.num_experts, config.intermediate_size, config.hidden_size)
    return (
        w1.view(*expert_shape).transpose(1, 2).unsqueeze(0).contiguous(),
        w2.view(*expert_shape).unsqueeze(0).contiguous(),
    )


def additive_attention_mask(
    attention_mask: torch.Tensor, device, dtype: ttnn.DataType = ACTIVATION_DTYPE
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
    """
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    seqlen = attention_mask.shape[-1]
    mask = build_extended_attention_mask(attention_mask, torch_dtype).expand(-1, -1, seqlen, -1)
    row_major = ttnn.from_torch(mask.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_layout(row_major, LAYOUT, pad_value=float(torch.finfo(torch_dtype).min))


def rotary_tables(
    device, config: NomicMoEConfig, seqlen: int, dtype: ttnn.DataType = ACTIVATION_DTYPE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build the (1, 1, S, head_dim) cos/sin tables rotary_embedding_hf reads, on device.

    Delegates to tt_transformers' get_rot_mats_hf, whose unscaled path computes the same
    inv_freq, outer product and concat widening as NomicBertRotaryEmbedding plus
    apply_rotary_emb. test_rotary_tables_match_the_reference holds them bit-exact in fp32.

    The concat widening is load-bearing: interleaving gives the GPT-J lane pairing, which under
    the kernel's NeoX rotate-half is not a rotation and does not preserve the per-plane norm.
    """
    cos, sin = get_rot_mats_hf(
        head_dim=config.head_dim,
        device=device,
        seq_len=seqlen,
        theta=config.rotary_emb_base,
        rope_scaling=None,
        datatype=dtype,
    )
    return cos, sin


def flatten_tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """(B, 1, S, H) -> (1, 1, B*S, H), batch-major."""
    batch, _, seqlen, hidden = x.shape
    return ttnn.reshape(x, (1, 1, batch * seqlen, hidden))


def unflatten_tokens(x: ttnn.Tensor, batch: int, seqlen: int) -> ttnn.Tensor:
    """(1, 1, B*S, H) -> (B, 1, S, H). B and S are arguments because the flat form has lost them."""
    return ttnn.reshape(x, (batch, 1, seqlen, x.shape[-1]))
