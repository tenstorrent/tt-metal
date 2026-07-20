# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 *text* input path — text ids -> GPT input.

This is the text branch that feeds the XTTS GPT decoder: a text token embedding
summed with a *learned* position embedding
(``LearnedPositionEmbeddings``: an ``nn.Embedding`` indexed by position), i.e.

    text_emb = text_embedding(text_ids) + text_pos_embedding(0 .. text_len)

``text_emb`` is exactly the text portion of the ``[text] + [mel]`` stream that
becomes the input to the 30-block GPT-2 decoder (see ``xtts_gpt_model``).

Weights come from the upstream checkpoint at https://huggingface.co/coqui/XTTS-v2
(``gpt.text_embedding.weight``, ``gpt.text_pos_embedding.emb.weight``).
"""

import functools

import torch
from torch import nn

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, HIDDEN_SIZE, MAX_TEXT_POS
from models.experimental.xtts.reference.xtts_gpt_model import NUM_TEXT_TOKENS

VOCAB_FILE = "vocab.json"  # XTTS-v2 BPE tokenizer, alongside model.pth in the HF repo


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    """Load the XTTS-v2 BPE tokenizer from the HF repo's ``vocab.json``.

    ``vocab.json`` is a standard HuggingFace ``tokenizers`` file, so we load it
    directly (no coqui-tts dependency). Its vocab size is exactly
    ``NUM_TEXT_TOKENS`` (6681), matching ``gpt.text_embedding``.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(repo_id=HF_REPO_ID, filename=VOCAB_FILE))


def preprocess_text(text, lang="en"):
    """XTTS text preprocessing: prepend the ``[lang]`` tag, convert spaces to the explicit
    ``[SPACE]`` token, then BPE-tokenize — matching coqui ``VoiceBpeTokenizer.encode``.

    Returns token ids as a ``LongTensor`` of shape ``[1, seq]`` — the input the
    text embedding expects.

    CRITICAL: the XTTS ``vocab.json`` uses a Whitespace pre-tokenizer that DISCARDS raw
    spaces, and ``[SPACE]`` (id 2) is a real vocab token the GPT was trained on as the
    word delimiter. So coqui does ``txt.replace(" ", "[SPACE]")`` before encoding; without
    it every word boundary is lost and the GPT receives out-of-distribution run-together
    subwords (slurred/merged words, wrong prosody, early stops). We replicate that exactly.

    NOTE: upstream coqui also expands numbers/abbreviations/symbols per language
    (``multilingual_cleaners``) before this; that normalization source is not vendored here
    (only ``model.pth`` and ``vocab.json``), so pass already-normalized text for numeric prose.
    """
    tokenizer = _load_tokenizer()
    txt = f"[{lang}]{text.strip().lower()}".replace(" ", "[SPACE]")
    ids = tokenizer.encode(txt).ids
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class XttsReferenceTextEmbedding(nn.Module):
    """Text token embedding + learned position embedding.

    ``forward(text_ids)`` takes integer ids ``[batch, text_len]`` (values in
    ``[0, NUM_TEXT_TOKENS)``) and returns ``[batch, text_len, HIDDEN_SIZE]``.
    """

    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(NUM_TEXT_TOKENS, HIDDEN_SIZE)
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_POS, HIDDEN_SIZE)

    def forward(self, text_ids):
        pos = torch.arange(text_ids.shape[1], device=text_ids.device)
        return self.text_embedding(text_ids) + self.text_pos_embedding(pos)


def reference_text_embedding(state_dict):
    """Build the text input path with real weights, in eval mode."""
    module = XttsReferenceTextEmbedding()
    module.text_embedding.load_state_dict({"weight": state_dict["gpt.text_embedding.weight"]})
    module.text_pos_embedding.load_state_dict({"weight": state_dict["gpt.text_pos_embedding.emb.weight"]})
    module.eval()
    return module
