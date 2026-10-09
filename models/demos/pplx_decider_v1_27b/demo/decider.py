# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TTDecider``: the reference app's ``Decider.predict`` on a Tenstorrent device.

Mirrors the snapshot ``inference.py``::

    row = {"state": state, "question": question}
    probabilities = DecisionModel.predict([row], batch_size=1)[0]
    return answer(question, probabilities)

Prompt rendering, tokenization, option counting and ``answer`` come from the snapshot's own
``autojev`` package, loaded exactly as ``reference/decision_prompts.py`` does (``AppTokenizer``,
including its Python-3.10 ``autojev.types`` stub). ``DecisionModel.predict`` is replaced by
``PplxDeciderModel.decide``: token upload -> device forward -> one readback of the probabilities.
Text only (images are the vision-tower stage).
"""

from __future__ import annotations

from pathlib import Path

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT, AppTokenizer
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel

# The text example of the snapshot inference.py (main()).
DEMO_STATE = "My Stripe integration keeps failing. I'm losing sales. Please help ASAP."


def demo_questions() -> dict:
    return {
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "routing": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Charges and refunds",
                "technical_support": "Integration errors",
                "sales": "Questions about buying a product",
            },
        },
    }


class TTDecider:
    def __init__(self, model: PplxDeciderModel, tokenizer: AppTokenizer):
        if abs(model.config.temperature - tokenizer.temperature) > 1e-12:
            raise ValueError("Model and tokenizer read different decision_config.json temperatures")
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, device, snapshot: Path | str = DEFAULT_SNAPSHOT, **model_kwargs) -> "TTDecider":
        from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

        model = PplxDeciderModel.from_snapshot(device, SnapshotReader(snapshot), **model_kwargs)
        return cls(model, AppTokenizer(Path(snapshot)))

    def prepare(self, state: str, question) -> tuple[dict, list[int], int]:
        """The app's row, its token ids and option count (``DecisionModel.prepare`` at batch 1)."""
        row = {"state": state, "question": question}
        return row, self.tokenizer.input_ids(row), self.tokenizer.count(row)

    def predict_probabilities(self, state: str, question) -> list[float]:
        _, ids, count = self.prepare(state, question)
        return self.model.decide(ids, count)

    def predict(self, state: str, question):
        """Return a choice, yes/no probability, or score using the saved temperature (``Decider.predict``)."""
        return self.tokenizer.autojev.answer(question, self.predict_probabilities(state, question))
