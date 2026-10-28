# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The System One API of d1-3B on a Wormhole chip, answer for answer like HF's `model.system_one`.

    d1 = TtD1.from_pretrained(mesh_device)
    d1.system_one("I was charged twice, please refund one.", {"refund": {"type": "noul", "instructions": "..."}})
    # {"answers": {"refund": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 39, "output_tokens": 0}}

HF reads several questions over one state in one packed pass; here each question is its own traced prefill of
state + question, which is the same computation (verified to 1e-6 on the fp32 reference).
"""

from models.demos.multimodal.d1_3b.tt.decision import D1Prompt
from models.demos.multimodal.d1_3b.tt.model import TtD1Model


class TtD1:
    def __init__(self, model: TtD1Model, prompt: D1Prompt):
        self.model, self.prompt = model, prompt

    @classmethod
    def from_pretrained(cls, mesh_device, trace=True, **model_kwargs):
        model = TtD1Model(mesh_device, **model_kwargs)
        if trace:
            model.capture_traces()
        return cls(model, D1Prompt(tokenizer=model.args.tokenizer))

    def _check_images(self, images):
        if images:
            raise NotImplementedError("image states are not supported yet on device")

    def probabilities(self, state, questions, images=None):
        """Each question's distribution over its options (`yes`, `no` for a noul), in the questions' order."""
        self._check_images(images)
        out = []
        for q in questions:
            hidden = self.model.hidden(self.prompt.input_ids(state, q))
            out.append(self.prompt.probabilities(q, self.model.option_logits(hidden, self.prompt.option_token_ids(q))))
        return out

    def input_tokens(self, state, questions):
        """The tokens HF's `system_one` reports: the prompt for one question; the state once plus every question
        for several."""
        tok, p = self.prompt.tokenizer, self.prompt.prompt
        if len(questions) == 1:
            return len(self.prompt.input_ids(state, questions[0]))
        prefix = tok.encode(p.prefix_text(tok, state, self.prompt.bos), add_special_tokens=False)
        suffixes = [
            tok.encode(p.suffix_text(tok, self.prompt.question(q), self.prompt.lead), add_special_tokens=False)
            for q in questions
        ]
        return len(prefix) + sum(map(len, suffixes))

    def system_one(self, state, questions, images=None):
        """Named questions over one state: `{"answers": {name: answer}, "usage": {...}}`."""
        names, qs = list(questions), list(questions.values())
        probs = self.probabilities(state, qs, images)
        return {
            "answers": {n: self.prompt.answer(q, p) for n, q, p in zip(names, qs, probs)},
            "usage": {"input_tokens": self.input_tokens(state, qs), "output_tokens": 0},
        }

    def system_one_batch(self, requests):
        """`(state, questions)` or `(state, questions, images)` requests, one response each."""
        return [self.system_one(r[0], r[1], r[2] if len(r) > 2 else None) for r in requests]

    def release(self):
        self.model.release_traces()
