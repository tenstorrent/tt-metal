# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The whole model, the TTNN form of reference.NomicBertModel, plus the text-to-embedding driver.

    TtNomicBertModel, the backbone:

      input_ids (B, S) int64
        embeddings           -> (B, 1, S, H)
        emb_ln               -> (B, 1, S, H)
        encoder, 12 blocks   -> (B, 1, S, H)   last_hidden_state

    encode(), the driver, mirroring reference/embedding.py stage for stage:

      texts
        tokenize             -> input_ids, attention_mask
        embed(), for each length-sorted chunk cut to its longest text:
          the backbone          -> (b, 1, s, H)
          mean_pool             -> (b, 1, 1, H)   fp32, padding excluded
          matryoshka_truncate   -> (b, 1, 1, dim) optional
          l2_normalize          -> (b, 1, 1, dim) unit norm
        to_torch, input order   -> (B, dim)

Large inputs run in chunks of whole sequences (tt/chunking.py): the backbone splits a batch past
tt_config.max_chunk_tokens computed tokens into contiguous chunks and joins their outputs, and embed()
sorts the texts by length so that each chunk also drops the padding its longest text did not need.

The split is the reference's: the model is the backbone and nothing else, and the pooling stages
are called directly by the one driver rather than wrapped in a method of their own. Anyone
holding token ids and wanting a pooled vector calls tt.pooling the way a reference user calls
reference.postprocessing.

This is the host boundary. Every module below it takes and returns device tensors; this class
takes torch token ids, because that is what a tokenizer produces and what the reference's own
entry point takes, and because the attention mask is a host build that depends on S, which varies
per call. The rotary tables depend on S too, but are kept on the model (RotaryTables) rather than
rebuilt per call.

encode() mirrors reference/embedding.py and reuses reference/preprocessing.py verbatim for the
prefixes and tokenization: that stage is pure host text handling with no device equivalent, and
running the identical code on both sides is what makes a parity difference attributable to the
backbone rather than to tokenization.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import (
    MAX_SEQ_LENGTH,
    NomicPromptPrefix,
    apply_prompt,
    tokenize,
)
from models.experimental.nomic_embed_text_v2_moe.tt import pooling
from models.experimental.nomic_embed_text_v2_moe.tt.chunking import plan_chunks, plan_sorted_chunks
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    LayerNormParameters,
    RotaryTables,
    activation_memory_config,
    additive_attention_mask,
    pooling_mask,
    prepare_token_ids,
)
from models.experimental.nomic_embed_text_v2_moe.tt.embeddings import TtNomicBertEmbeddings
from models.experimental.nomic_embed_text_v2_moe.tt.encoder import TtNomicBertEncoder
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup


class TtNomicBertModel(LightweightModule):
    """Encoder-only backbone returning last_hidden_state, with pooling available on top.

    No pooler and no task head: the checkpoint ships neither. Pooling to one vector per text is
    `embed`, matching the checkpoint's own modules.json, which declares Pooling and Normalize as
    separate stages after the backbone.

    The weights are the whole 951 MB of the checkpoint, moved to device once at construction, so
    build this once per device and call it many times.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix=""):
        super().__init__()
        self.device = device
        self.config = config
        self.tt_config = tt_config

        self.embeddings = TtNomicBertEmbeddings(
            device, config, tt_config, state_dict, f"{state_dict_prefix}embeddings."
        )
        self.encoder = TtNomicBertEncoder(device, config, tt_config, state_dict, f"{state_dict_prefix}encoder.")
        self.emb_ln = LayerNormParameters(
            state_dict[f"{state_dict_prefix}emb_ln.weight"],
            state_dict[f"{state_dict_prefix}emb_ln.bias"],
            device,
            tt_config.weight_dtype,
        )
        # The activation dtype, so lowering tt_config.activation_dtype moves the tables with it.
        self.rotary_tables = RotaryTables(device, config, dtype=tt_config.activation_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Embed the tokens and run the encoder stack, in chunks of whole sequences when the batch is large.

        Past tt_config.max_chunk_tokens computed tokens the batch runs as contiguous chunks
        (tt.chunking.plan_chunks), each through the whole backbone, and their outputs are joined in DRAM.
        Each sequence's output is what the backbone computes for it in its own chunk.

        Args:
            input_ids: (B, S) int64 token ids, from reference.preprocessing.tokenize.
            attention_mask: (B, S) int64, 1 for real tokens and 0 for padding. None means no
                masking. An all-ones mask is treated the same way: it is a proven no-op
                (test_an_all_ones_mask_is_a_no_op), while materialising it would cost a
                (B, 1, S, S) tensor read by every head of every SDPA call, which doubled SDPA's
                time at 8x512. SDPA masks the tile padding of S on its own.
            token_type_ids: Accepted for parity with the reference. type_vocab_size is 1, so 0 is
                the only legal value and the embeddings module has already folded that row into
                the word table.

        Returns:
            ttnn.Tensor: (B, 1, S, H) last hidden state, one contextual vector per token.

        Raises:
            ValueError: If token_type_ids holds anything but zeros, which this checkpoint cannot
                represent.
        """
        if token_type_ids is not None and bool(token_type_ids.any()):
            raise ValueError(
                f"type_vocab_size is {self.config.type_vocab_size}, so token_type_ids must be all "
                "zeros; the token-type row is folded into the word embedding table at load"
            )

        batch, seqlen = input_ids.shape
        outputs = []
        for chunk in plan_chunks(batch, seqlen, self.tt_config.max_chunk_tokens):
            rows = slice(chunk.start, chunk.start + chunk.count)
            outputs.append(self.backbone(input_ids[rows], None if attention_mask is None else attention_mask[rows]))
        if len(outputs) == 1:
            return outputs[0]
        out = ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for piece in outputs:
            ttnn.deallocate(piece)
        return out

    def backbone(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """One pass of the backbone over the whole of input_ids, (B, S) -> (B, 1, S, H); see forward."""
        seqlen = input_ids.shape[-1]
        hidden = self.embeddings(prepare_token_ids(input_ids, self.device))
        weight, bias = self.emb_ln.for_input(hidden)
        normalized = ttnn.layer_norm(
            hidden,
            weight=weight,
            bias=bias,
            epsilon=self.config.layer_norm_epsilon,
            memory_config=activation_memory_config(hidden.shape),
            compute_kernel_config=self.tt_config.compute_kernel_config(OpGroup.NORM),
        )
        ttnn.deallocate(hidden)

        # dtype is passed explicitly rather than left to the helper's default, so lowering
        # tt_config.activation_dtype moves the mask's fill value with it.
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            mask = additive_attention_mask(
                attention_mask,
                self.device,
                dtype=self.tt_config.activation_dtype,
                mask_dtype=self.tt_config.attention_mask_dtype,
            )
        rot_mats = self.rotary_tables(seqlen)
        out = self.encoder(normalized, rot_mats, mask)
        self.rotary_tables.release(rot_mats)
        ttnn.deallocate(normalized)
        return out


def encode(
    model: TtNomicBertModel,
    tokenizer,
    texts: Sequence[str],
    prompt_prefix: Optional[Union[NomicPromptPrefix, Sequence[Optional[NomicPromptPrefix]]]] = None,
    matryoshka_dim: Optional[int] = None,
    max_length: int = MAX_SEQ_LENGTH,
) -> torch.Tensor:
    """Turn text into normalized embeddings on device, the counterpart of reference.embedding.encode.

    Row i is the embedding of texts[i]: padding is masked out at pooling, so a short text gets
    the same vector, to within 1e-2 in cosine, whether encoded alone or in a ragged batch. It is
    not bit-identical, since the matmul blockings are chosen from the batch's shape. The dot
    product of two rows is their cosine similarity.

    Args:
        model: A constructed TtNomicBertModel.
        tokenizer: The XLMRobertaTokenizerFast loaded from the checkpoint.
        texts: The input strings, length B.
        prompt_prefix: One NomicPromptPrefix for every text, a sequence of B of them for one per
            text, or None for no prefix. The prefixes are trained in, so the same string embeds
            differently as a query than as a document.
        matryoshka_dim: Target embedding width, at most 768, or None for the full 768.
        max_length: Tokenizer truncation limit, defaulting to MAX_SEQ_LENGTH (512).

    Returns:
        torch.Tensor: (B, matryoshka_dim or 768) fp32, unit norm, back on the host.
    """
    encoded = tokenize(tokenizer, apply_prompt(texts, prompt_prefix), max_length=max_length)
    return embed(model, encoded["input_ids"], encoded["attention_mask"], matryoshka_dim)


def embed(
    model: TtNomicBertModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    matryoshka_dim: Optional[int] = None,
) -> torch.Tensor:
    """Token ids to normalized embeddings, in length-sorted chunks each cut to its longest sequence.

    The sequences are sorted longest first and split into chunks of at most tt_config.max_chunk_tokens
    computed tokens (tt.chunking.plan_sorted_chunks). Each chunk runs at its own longest sequence's length,
    so a short text no longer computes the padding its batch's longest text set, and each is pooled and
    normalized on its own; the rows come back in input order. All chunks are queued before the first is
    read back.

    Args:
        model: A constructed TtNomicBertModel.
        input_ids: (B, S) int64 token ids, right-padded as reference.preprocessing.tokenize returns them;
            with any other padding every chunk keeps the full S.
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding; None for no padding.
        matryoshka_dim: Target embedding width, at most 768, or None for the full 768.

    Returns:
        torch.Tensor: (B, matryoshka_dim or 768) fp32, unit norm.
    """
    batch, seqlen = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones((batch, seqlen), dtype=torch.long)
    kernel_config = model.tt_config.compute_kernel_config(OpGroup.REDUCE)
    lengths = attention_mask.sum(dim=-1).long()
    # Cutting a chunk to its longest sequence assumes right padding; any other mask keeps the full width.
    if not torch.equal(attention_mask.bool(), torch.arange(seqlen) < lengths[:, None]):
        lengths = torch.full_like(lengths, seqlen)
    order = torch.argsort(lengths, descending=True, stable=True)
    chunks = plan_sorted_chunks(lengths[order].tolist(), model.tt_config.max_chunk_tokens)
    pieces = []
    for chunk in chunks:
        rows = order[chunk.start : chunk.start + chunk.count]
        # At least one token a row: a fully padded row pools to zeros either way.
        width = max(chunk.seqlen, 1)
        chunk_mask = attention_mask[rows, :width]
        hidden = model(input_ids[rows, :width], chunk_mask)
        # Pooling needs a mask even where the backbone did not: padded positions must not enter the
        # mean, because the <pad> embedding is trained and non-zero, so counting it would make one
        # text's embedding depend on how long its batch-mates are.
        pooled = pooling.mean_pool(hidden, pooling_mask(chunk_mask, model.device), compute_kernel_config=kernel_config)
        ttnn.deallocate(hidden)
        pieces.append(
            pooling.l2_normalize(
                pooling.matryoshka_truncate(pooled, matryoshka_dim), compute_kernel_config=kernel_config
            )
        )
        ttnn.deallocate(pooled)
    out = torch.empty((batch, pieces[0].shape[-1]), dtype=torch.float32)
    for chunk, piece in zip(chunks, pieces):
        out[order[chunk.start : chunk.start + chunk.count]] = ttnn.to_torch(piece).float().reshape(chunk.count, -1)
        ttnn.deallocate(piece)
    return out
