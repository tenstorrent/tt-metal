"""Shared inputs for the embeddinggemma-2 serving-path comparison (bare tt-metal / FastAPI / tt-media-server)."""

import os

MODEL_ID = "google/embeddinggemma-2"
REVISION = os.environ.get("EMBEDDING_GEMMA2_REVISION", "914f7f89142e33e77833254d9c9b90c3cef7303b")
# (batch, tokens incl. BOS/EOS); batch * bucket <= 8192 except batch 1, as the model's serving shapes allow.
POINTS = [(1, 32), (1, 128), (1, 512), (1, 2048), (1, 8192), (8, 128), (8, 512), (32, 128)]

_PASSAGE = (
    "The northern lights are caused by charged particles from the sun striking atoms in the upper atmosphere. "
    "Tenstorrent builds AI accelerators on RISC-V cores with a mesh of Tensix compute units and on-chip SRAM. "
    "A search engine ranks documents by comparing the embedding of a query with the embeddings of candidates. "
    "Rivers carve valleys over millions of years while glaciers leave behind lakes, moraines and polished rock. "
)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)


def make_texts(tokenizer, n_tokens: int, count: int, seed: int = 0) -> list[str]:
    """`count` different texts that each tokenize to exactly `n_tokens` ids (BOS ... EOS) with `tokenizer`."""
    body = tokenizer(_PASSAGE * (n_tokens // 40 + 4), add_special_tokens=False).input_ids
    specials = len(tokenizer("a").input_ids) - len(tokenizer("a", add_special_tokens=False).input_ids)
    want = n_tokens - specials
    texts = []
    for i in range(count):
        start = (seed * 7919 + i * 13) % 97
        ids = body[start : start + want]
        text = tokenizer.decode(ids)
        # re-tokenization can shift by a few ids: trim or extend until exact
        for _ in range(20):
            n = len(tokenizer(text).input_ids)
            if n == n_tokens:
                break
            ids = ids[: len(ids) + (n_tokens - n)] if n > n_tokens else body[start : start + len(ids) + (n_tokens - n)]
            text = tokenizer.decode(ids)
        texts.append(text)
    return texts
