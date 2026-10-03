# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model / mesh / paged-KV configuration of the DeepSeek-V4.1-Flash demo (the counterpart of GPT-OSS ``model_config.py`` + the tt_transformers
``ModelArgs`` surface that ``preprocess_inputs_prefill`` and the generator use: ``tokenizer``, ``encode_prompt``, ``max_context_len``,
``max_seq_len``, ``max_batch_size``, ``n_layers``, ``base_model_name``)."""

import os
import sys
from dataclasses import dataclass

CKPT = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")
VOCAB = 129280
PAGE_TOKENS = 128  # one page = 128 tokens (kv_paged.PageLayout); the demo's ``page_block_size`` must equal it


@dataclass
class DSV41PageParams:
    """Paged KV pool parameters (the DSV4.1 pool is one pool per mesh row: ``max_num_blocks_per_row`` pages of 128 tokens shared by the users of
    that row, see docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md)."""

    block_size: int = PAGE_TOKENS
    max_num_blocks: int = 0  # pages per mesh row; 0 = users_per_row * ceil(max_seq_len / 128)


class DSV41ModelArgs:
    base_model_name = "DeepSeek-V4.1-Flash"

    def __init__(self, mesh_device, max_batch_size, max_seq_len, layer_ids=None, paged_attention_config=None):
        from transformers import AutoTokenizer

        self.mesh_device = mesh_device
        self.mesh_rows, self.mesh_cols = tuple(mesh_device.shape)
        self.max_batch_size = max_batch_size  # users per mesh-row group * rows (padded batch)
        self.users_per_row = max(1, -(-max_batch_size // self.mesh_rows))
        self.max_batch_size = self.users_per_row * self.mesh_rows
        self.max_seq_len = max_seq_len
        self.max_context_len = max(max_seq_len, 1 << 20)  # the checkpoint supports 1M; the pool capacity is what limits
        self.layer_ids = list(range(40)) if layer_ids is None else list(layer_ids)
        self.n_layers = len(self.layer_ids)
        self.model_name = self.base_model_name
        self.vocab_size = VOCAB
        self.paged_attention_config = paged_attention_config
        self.tokenizer = AutoTokenizer.from_pretrained(CKPT)
        self.processor = None
        self.eos_token_id = self.tokenizer.eos_token_id
        enc = os.path.join(CKPT, "encoding")
        if enc not in sys.path:
            sys.path.insert(0, enc)

    def encode_prompt(self, prompt, instruct=True):
        """Chat template (non-thinking) of the checkpoint's ``encoding.py`` when ``instruct``; else BOS + raw text."""
        if not instruct:
            return self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        from encoding import encode_messages

        return self.tokenizer.encode(
            encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat"), add_special_tokens=False
        )


def dense_context_limit(layer_ids):
    """Longest context (tokens) the model handles exactly without the indexer: ratio-1 layers select all compressed entries while there are <= 512 (ctx <= 512),
    ratio-2 layers while ctx <= 1024; window-only layers have no limit."""
    from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R

    ratios = {R.model_args().compress_ratios[L] for L in layer_ids}
    return 512 if 1 in ratios else 1024 if 2 in ratios else 1 << 30
