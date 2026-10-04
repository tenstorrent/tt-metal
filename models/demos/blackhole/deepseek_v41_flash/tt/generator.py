# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generation pipeline of DeepSeek-V4.1-Flash with the entry points of ``models/tt_transformers/tt/generator.py`` that the GPT-OSS demo uses:
``prefill_forward_text`` (batched prefill of ragged prompts into the paged pool) and ``decode_forward`` (traced device-loop decode step with
on-device greedy sampling feedback). The same ``Generator`` is what a spec-decode driver will wrap."""

import torch
from loguru import logger


class Generator:
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        self.model = model if isinstance(model, list) else [model]
        self.model_args = model_args if isinstance(model_args, list) else [model_args]
        self.mesh_device = mesh_device
        self.processor, self.tokenizer = processor, tokenizer
        self.prev_page_table = None
        self.prefill_chunk = (
            None  # tokens per user per prefill chunk (multiple of 128); None = auto from the row token budget
        )

    @property
    def m(self):
        return self.model[0]

    def auto_chunk(self, max_len, budget_tokens_per_row=None):
        """Chunk of tokens per user so that users_per_row * chunk <= the per-row token budget of one prefill pass (activation memory). The streams of a
        chunk are fp32 [32,1,4,5120] tiles (4 -> 32 row padding): ~80 MiB/bank per 1024 tokens/row for the captured chunk + ~55 MiB eager compile peak, so the
        budget shrinks with the context (the sparse-prefill tables grow with S_pad): 4096 tokens/row up to 16k, 2048 above (measured: 32k at 4096/row OOMs
        in the compile pass with 713 MiB/bank free, 64k at 2048/row fits with 341 MiB free). DSV41_PREFILL_ROW_TOKENS overrides.
        """
        import os

        from models.demos.blackhole.deepseek_v41_flash.tt.common import get_padded_prefill_len

        if budget_tokens_per_row is None:
            env = os.environ.get("DSV41_PREFILL_ROW_TOKENS")
            budget_tokens_per_row = int(env) if env else (4096 if max_len <= 16384 else 2048)
        c = max(128, (budget_tokens_per_row // self.m.U) // 128 * 128)
        return None if c >= get_padded_prefill_len(max_len) else c

    def prefill_forward_text(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=True,
        warmup_prefill=False,
        sampling_params=None,
        chunk=None,
        return_logits=False,
        **kwargs,
    ):
        """tokens [B, L] right-padded prompts, prompt_lens [B]. Returns logits [B, vocab] (host) of every user's LAST prompt token, or, when
        ``sampling_params`` is given, a tuple (first tokens [B, 1], None) sampled greedily on the device. KV / state land in the paged pool.
        """
        lens = torch.as_tensor(prompt_lens if prompt_lens is not None else [tokens.shape[1]] * tokens.shape[0])
        chunk = chunk or self.prefill_chunk or self.auto_chunk(int(lens.max()))
        want = sampling_params is None or return_logits
        first, logits = self.m.prefill_forward(tokens, lens, chunk=chunk, want_logits=want, enable_trace=enable_trace)
        logger.info(f"prefill timing {{{', '.join(f'{k}: {v:.2f}' for k, v in self.m.timing.items())}}} chunk={chunk}")
        if sampling_params is not None:
            return first.reshape(-1, 1), logits
        return logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        *,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """One decode step. tokens [B] (the token at position start_pos [B]) -> (next tokens [B], None). Greedy only (``sampling_params`` temperature 0)."""
        if sampling_params is not None:
            t = sampling_params.temperature
            t = t[0] if isinstance(t, (list, tuple)) else t
            assert (
                t == 0
            ), "DSV4.1 decode samples greedily on the device (temperature 0); top-k / top-p sampling is not implemented"
        out = self.m.decode_forward(
            tokens.reshape(-1), start_pos, enable_trace=enable_trace, reload_inputs=reload_inputs
        )
        return out, None
