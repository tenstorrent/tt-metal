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
        the backbone         -> (B, 1, S, H)
        mean_pool            -> (B, 1, 1, H)   padding excluded
        matryoshka_truncate  -> (B, 1, 1, dim) optional
        l2_normalize         -> (B, 1, 1, dim) unit norm
        to_torch             -> (B, dim)

The split is the reference's: the model is the backbone and nothing else, and the pooling stages
are called directly by the one driver rather than wrapped in a method of their own. Anyone
holding token ids and wanting a pooled vector calls tt.pooling the way a reference user calls
reference.postprocessing.

This is the host boundary. Every module below it takes and returns device tensors; this class
takes torch token ids, because that is what a tokenizer produces and what the reference's own
entry point takes, and because the rotary tables and the attention mask are host builds that
depend on S, which varies per call.

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
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    additive_attention_mask,
    pooling_mask,
    prepare_token_ids,
    rotary_tables,
    to_device,
)
from models.experimental.nomic_embed_text_v2_moe.tt.embeddings import TtNomicBertEmbeddings
from models.experimental.nomic_embed_text_v2_moe.tt.encoder import TtNomicBertEncoder


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
        self.emb_ln_weight = to_device(
            state_dict[f"{state_dict_prefix}emb_ln.weight"], device, dtype=tt_config.weight_dtype
        )
        self.emb_ln_bias = to_device(
            state_dict[f"{state_dict_prefix}emb_ln.bias"], device, dtype=tt_config.weight_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Embed the tokens and run the encoder stack.

        Args:
            input_ids: (B, S) int64 token ids, from reference.preprocessing.tokenize.
            attention_mask: (B, S) int64, 1 for real tokens and 0 for padding. None means no
                masking, which is equivalent to all-ones and cheaper: an all-ones mask is a
                proven no-op (test_an_all_ones_mask_is_a_no_op) and materialising it would cost
                a (B, 1, S, S) tensor, 1 MB at B=2 S=512.
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

        seqlen = input_ids.shape[-1]
        hidden = self.embeddings(prepare_token_ids(input_ids, self.device))
        normalized = ttnn.layer_norm(
            hidden,
            weight=self.emb_ln_weight,
            bias=self.emb_ln_bias,
            epsilon=self.config.layer_norm_epsilon,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        ttnn.deallocate(hidden)

        # dtype is passed explicitly rather than left to each helper's default, so lowering
        # tt_config.activation_dtype moves the mask and the rotary tables with it. A mismatch here
        # surfaces inside SDPA, which rejects a mask whose dtype differs from q/k/v.
        dtype = self.tt_config.activation_dtype
        mask = None if attention_mask is None else additive_attention_mask(attention_mask, self.device, dtype=dtype)
        out = self.encoder(normalized, rotary_tables(self.device, self.config, seqlen, dtype=dtype), mask)
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

    Row i is the embedding of texts[i] and is independent of the other rows: padding is masked
    out at pooling, so a short text gets the same vector whether encoded alone or in a ragged
    batch. The dot product of two rows is their cosine similarity.

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
    attention_mask = encoded["attention_mask"]
    kernel_config = model.tt_config.compute_kernel_config

    hidden = model(encoded["input_ids"], attention_mask)

    # Pooling needs a mask even where the backbone did not: padded positions must not enter the
    # mean, because the <pad> embedding is trained and non-zero, so counting it would make one
    # text's embedding depend on how long its batch-mates are.
    pooled = pooling.mean_pool(
        hidden,
        pooling_mask(attention_mask, model.device, dtype=model.tt_config.activation_dtype),
        compute_kernel_config=kernel_config,
    )
    ttnn.deallocate(hidden)

    embeddings = pooling.l2_normalize(
        pooling.matryoshka_truncate(pooled, matryoshka_dim), compute_kernel_config=kernel_config
    )

    return ttnn.to_torch(embeddings).float().reshape(len(texts), -1)
