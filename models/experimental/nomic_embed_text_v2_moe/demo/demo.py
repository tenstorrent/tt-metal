# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-embedding demo for the nomic-embed-text-v2-moe TTNN port.

Embeds the queries and passages given, prints the similarity matrix, and reports the top passage
per query. With --compare it runs the PyTorch reference over the same inputs and prints the
per-row cosine between the two.

    python models/experimental/nomic_embed_text_v2_moe/demo/demo.py
    python models/experimental/nomic_embed_text_v2_moe/demo/demo.py \
        --query "What does Tenstorrent make?" \
        --passage "Tenstorrent builds AI accelerators." --compare

Queries and passages carry different trained prefixes, so the same string embeds differently as
one or the other; that asymmetry is what the model was trained with. Do not set
TT_VISIBLE_DEVICES: on a p300c it fails with "Custom fabric mesh graph descriptor path must be
specified for CUSTOM cluster type".
"""

from __future__ import annotations

import argparse

from loguru import logger
import torch

import ttnn

from models.experimental.nomic_embed_text_v2_moe.common import load_tokenizer, resolve_checkpoint
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import load_vendored_config
from models.experimental.nomic_embed_text_v2_moe.reference.loader import load_state_dict_from_safetensors
from models.experimental.nomic_embed_text_v2_moe.reference.postprocessing import cosine_similarity_matrix
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import NomicPromptPrefix
from models.experimental.nomic_embed_text_v2_moe.tt.model import TtNomicBertModel, encode
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import TtModelConfig

DEFAULT_QUERIES = [
    "What does Tenstorrent make?",
    "How does expert routing work?",
]

DEFAULT_PASSAGES = [
    "Tenstorrent builds AI accelerators for efficient inference.",
    "Mixture-of-experts models route each token to a subset of experts.",
    "The quick brown fox jumps over the lazy dog.",
    "Bonjour, comment allez-vous aujourd'hui?",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", action="append", dest="queries", help="repeatable; defaults to a built-in set")
    parser.add_argument("--passage", action="append", dest="passages", help="repeatable; defaults to a built-in set")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--matryoshka-dim", type=int, default=None, help="truncate embeddings to this width")
    parser.add_argument("--compare", action="store_true", help="also run the PyTorch reference and report cosine")
    return parser.parse_args()


def main():
    args = parse_args()
    queries = args.queries or DEFAULT_QUERIES
    passages = args.passages or DEFAULT_PASSAGES

    config = load_vendored_config()
    tokenizer = load_tokenizer()
    logger.info("reading the checkpoint")
    state_dict = load_state_dict_from_safetensors(resolve_checkpoint())

    device = ttnn.open_device(device_id=args.device_id)
    try:
        logger.info("moving weights to device")
        model = TtNomicBertModel(device, config, TtModelConfig.from_device(device), state_dict)

        embedded_queries = encode(
            model, tokenizer, queries, prompt_prefix=NomicPromptPrefix.QUERY, matryoshka_dim=args.matryoshka_dim
        )
        embedded_passages = encode(
            model, tokenizer, passages, prompt_prefix=NomicPromptPrefix.PASSAGE, matryoshka_dim=args.matryoshka_dim
        )
    finally:
        ttnn.close_device(device)

    similarity = cosine_similarity_matrix(embedded_queries, embedded_passages)
    logger.info(f"embeddings: queries {tuple(embedded_queries.shape)}, passages {tuple(embedded_passages.shape)}")

    for idx, query in enumerate(queries):
        best = int(similarity[idx].argmax())
        logger.info(f"query: {query}")
        for passage_idx, passage in enumerate(passages):
            marker = " <- top" if passage_idx == best else ""
            logger.info(f"    {similarity[idx, passage_idx]:+.4f}  {passage}{marker}")

    if args.compare:
        from models.experimental.nomic_embed_text_v2_moe.reference import embedding as reference_embedding
        from models.experimental.nomic_embed_text_v2_moe.reference.loader import load_pretrained_reference_model

        logger.info("running the PyTorch reference over the same inputs")
        reference = load_pretrained_reference_model()
        pairs = (
            (queries, embedded_queries, NomicPromptPrefix.QUERY),
            (passages, embedded_passages, NomicPromptPrefix.PASSAGE),
        )
        for texts, embedded, prefix in pairs:
            expected = reference_embedding.encode(
                reference, tokenizer, texts, prompt_prefix=prefix, matryoshka_dim=args.matryoshka_dim
            )
            cosine = (embedded * expected).sum(-1)
            logger.info(f"{prefix.name.lower()} cosine vs reference: min {float(cosine.min()):.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
