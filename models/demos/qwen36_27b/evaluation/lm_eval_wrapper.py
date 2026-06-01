# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
lm-evaluation-harness wrapper for Qwen3.6-27B on TT P150a.

Adapts Qwen36Generator to the lm_eval.api.model.LM interface.
Follows the pattern from models/demos/wormhole/mamba/benchmarks/lm_harness_eval.py.

Usage:
    # Register and run via lm-eval CLI:
    lm_eval --model qwen36_tt --tasks mmlu_pro --limit 10

    # Or use programmatically:
    from models.demos.qwen36_27b.evaluation.lm_eval_wrapper import Qwen36TTEvalWrapper
    model = Qwen36TTEvalWrapper(max_layers=4, dummy_weights=True)
    # ... use with lm_eval.evaluator.simple_evaluate()
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("qwen36_tt")
class Qwen36TTEvalWrapper(LM):
    def __init__(
        self,
        max_layers: int | None = None,
        dummy_weights: bool = False,
        device_id: int = 0,
        batch_size: int = 1,
        max_length: int = 8192,
    ):
        super().__init__()

        import ttnn
        from transformers import AutoTokenizer

        from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
        from models.demos.qwen36_27b.tt.load_weights import load_state_dict, create_dummy_state_dict
        from models.demos.qwen36_27b.tt.model import TtQwen36Model
        from models.demos.qwen36_27b.tt.generator import Qwen36Generator

        self.config = Qwen36ModelConfig()
        if max_layers is not None:
            self.config.num_hidden_layers = max_layers

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._max_length = max_length

        if dummy_weights:
            state_dict = create_dummy_state_dict(self.config, num_layers=self.config.num_hidden_layers)
        else:
            state_dict = load_state_dict(self.config, max_layers=max_layers)

        self.tt_device = ttnn.open_device(device_id=device_id)
        self.model = TtQwen36Model(self.tt_device, state_dict, self.config)
        self.generator = Qwen36Generator(self.model, self.config, tokenizer=self.tokenizer)
        del state_dict

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 512

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "tt"

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        total = len(ctx_ids) + len(cont_ids)
        if total > self._max_length:
            ctx_ids = ctx_ids[-(self._max_length - len(cont_ids)):]
        return ctx_ids, cont_ids

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for instance in tqdm(requests, desc="loglikelihood"):
            context, continuation = instance.arguments
            ctx_ids, cont_ids = self._encode_pair(context, continuation)

            full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
            self.generator.reset()

            logits = self.generator.get_logits_for_sequence(full_ids)
            # logits[t] predicts token at position t+1
            # We want logits for positions that predict the continuation tokens

            vocab_size = self.config.vocab_size
            cont_start = len(ctx_ids)
            cont_len = len(cont_ids)

            log_probs = F.log_softmax(logits[:, :vocab_size], dim=-1)

            total_ll = 0.0
            is_greedy = True
            for i in range(cont_len):
                pred_pos = cont_start + i - 1  # logits at position p predict token p+1
                if pred_pos < 0:
                    pred_pos = 0
                target_token = cont_ids[i]
                token_ll = log_probs[pred_pos, target_token].item()
                total_ll += token_ll

                greedy_token = torch.argmax(log_probs[pred_pos]).item()
                if greedy_token != target_token:
                    is_greedy = False

            results.append((total_ll, is_greedy))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for instance in tqdm(requests, desc="generate_until"):
            context = instance.arguments[0]
            gen_kwargs = instance.arguments[1] if len(instance.arguments) > 1 else {}

            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)

            input_ids = torch.tensor([self.tok_encode(context)], dtype=torch.long)
            if input_ids.shape[1] > self._max_length - max_gen:
                input_ids = input_ids[:, -(self._max_length - max_gen):]

            self.generator.reset()
            generated = self.generator.generate(
                input_ids,
                max_new_tokens=max_gen,
                temperature=temperature,
            )

            output_text = self.tok_decode(generated)

            for stop in until:
                if stop in output_text:
                    output_text = output_text[:output_text.index(stop)]
                    break

            results.append(output_text)

        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for instance in tqdm(requests, desc="loglikelihood_rolling"):
            text = instance.arguments[0]
            token_ids = self.tok_encode(text)
            if len(token_ids) > self._max_length:
                token_ids = token_ids[:self._max_length]

            full_ids = torch.tensor([token_ids], dtype=torch.long)
            self.generator.reset()
            logits = self.generator.get_logits_for_sequence(full_ids)

            vocab_size = self.config.vocab_size
            log_probs = F.log_softmax(logits[:, :vocab_size], dim=-1)

            total_ll = 0.0
            for i in range(1, len(token_ids)):
                total_ll += log_probs[i - 1, token_ids[i]].item()

            results.append((total_ll, True))

        return results

    def close(self):
        import ttnn
        if hasattr(self, "tt_device") and self.tt_device is not None:
            ttnn.close_device(self.tt_device)
            self.tt_device = None

    def __del__(self):
        self.close()
