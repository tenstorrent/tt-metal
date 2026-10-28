# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""d1-3B's prompt and readout, host side.

Rendering and readout call d1's own `prompt.py` / `api.py` from the checkpoint: the model was post-trained on
that exact text, so any drift in a newline or an option code moves the probabilities.

A decision is a softmax over each option's best token at the answer slot (the prompt's last token). d1-3B
ships no calibration, so the log-softmax normaliser cancels and only the option tokens' logits are needed.
"""

from models.demos.multimodal.d1_3b.tt.checkpoint import remote_modules, resolve_checkpoint


class D1Prompt:
    def __init__(self, ckpt_dir=None, tokenizer=None, prompt_module=None, api_module=None):
        self.ckpt_dir = resolve_checkpoint(ckpt_dir)
        if prompt_module is None or api_module is None:
            prompt_module, api_module = remote_modules(self.ckpt_dir)
        self.prompt, self.api = prompt_module, api_module
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.ckpt_dir)
        self.tokenizer = tokenizer
        bos = getattr(tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.lead = self.prompt.default_lead("lfm2_vl")

    def question(self, q):
        return self.prompt.as_question(q)

    def render(self, state, q, image_markup=""):
        return self.prompt.render(self.tokenizer, state, self.question(q), self.bos, self.lead, images=image_markup)

    def input_ids(self, state, q):
        return self.tokenizer.encode(self.render(state, q), add_special_tokens=False)

    def readout_ids(self, q):
        return self.prompt.readout_ids(self.tokenizer, self.question(q))

    def option_token_ids(self, q):
        """Every token id the readout of `q` looks at, in a fixed order."""
        return sorted({i for group in self.readout_ids(q) for i in group})

    def probabilities(self, q, scores):
        """`scores` maps token id to its logit or log-probability at the answer slot (only option ids needed)."""
        return self.prompt.readout(self.tokenizer, self.question(q), scores)

    def answer(self, q, probabilities):
        return self.api.answer(self.question(q), probabilities)
