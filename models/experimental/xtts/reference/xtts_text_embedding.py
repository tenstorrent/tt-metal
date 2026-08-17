# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
from torch import nn

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, HF_REVISION, HIDDEN_SIZE, MAX_TEXT_POS
from models.experimental.xtts.reference.xtts_gpt_model import NUM_TEXT_TOKENS

from models.experimental.xtts.config import DEFAULT_LANGUAGE, VOCAB_FILE  # noqa: F401


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    """Load and cache the XTTS HuggingFace tokenizer."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download(repo_id=HF_REPO_ID, filename=VOCAB_FILE, revision=HF_REVISION))


def preprocess_text(text, lang=DEFAULT_LANGUAGE):
    """Tokenize lowercased text with language tag and space markers."""
    # vocab Whitespace pre-tokenizer discards spaces; [SPACE] must be substituted before BPE.
    tokenizer = _load_tokenizer()
    txt = f"[{lang}]{text.strip().lower()}".replace(" ", "[SPACE]")
    ids = tokenizer.encode(txt).ids
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class XttsReferenceTextEmbedding(nn.Module):
    def __init__(self):
        """Build text and positional embedding layers."""
        super().__init__()
        self.text_embedding = nn.Embedding(NUM_TEXT_TOKENS, HIDDEN_SIZE)
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_POS, HIDDEN_SIZE)

    def forward(self, text_ids):
        """Embed text ids and add learned positional embeddings."""
        pos = torch.arange(text_ids.shape[1], device=text_ids.device)
        return self.text_embedding(text_ids) + self.text_pos_embedding(pos)


def reference_text_embedding(state_dict):
    """Load reference text embedding weights from a checkpoint state dict."""
    module = XttsReferenceTextEmbedding()
    module.text_embedding.load_state_dict({"weight": state_dict["gpt.text_embedding.weight"]})
    module.text_pos_embedding.load_state_dict({"weight": state_dict["gpt.text_pos_embedding.emb.weight"]})
    module.eval()
    return module
