# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TTDecider``: the reference app's ``Decider.predict`` on a Tenstorrent device.

Mirrors the snapshot ``inference.py``::

    row = {"state": state, "question": question}
    if images:
        row["images"] = list(images)
    probabilities = DecisionModel.predict([row], batch_size=1)[0]
    return answer(question, probabilities)

Prompt rendering, tokenization, image loading (``open_image``: path, data URL or PIL image),
option counting and ``answer`` come from the snapshot's own ``autojev`` package, loaded exactly as
``reference/decision_prompts.py`` does (``AppTokenizer``, including its Python-3.10
``autojev.types`` stub). The processor gets the app's pixel budget
(``image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}``, ``model.py:139``) and
the chat template runs with ``enable_thinking=False``. ``DecisionModel.predict`` is replaced by
``PplxDeciderModel.decide_encoded``: input prep (3D position ids, splice index, vision inputs) ->
device forward (vision tower -> splice -> 64 layers -> head) -> one readback of the probabilities.
A request without images takes the text-only path (``PplxDeciderModel.decide``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT, AppTokenizer
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel

# The text example of the snapshot inference.py (main()).
DEMO_STATE = "My Stripe integration keeps failing. I'm losing sales. Please help ASAP."
# The image example of the snapshot inference.py (``--image``).
IMAGE_STATE = "Look at the supplied image."
APP_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 262144}


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


def image_question() -> dict:
    """The ``inference.py --image`` question."""
    return {
        "type": "choice",
        "instructions": "What is the dominant color?",
        "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "other": "Another color"},
    }


class TTDecider:
    def __init__(self, model: PplxDeciderModel, tokenizer: AppTokenizer):
        if abs(model.config.temperature - tokenizer.temperature) > 1e-12:
            raise ValueError("Model and tokenizer read different decision_config.json temperatures")
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.processor.image_processor.size = dict(APP_IMAGE_SIZE)

    @classmethod
    def from_pretrained(
        cls, device, snapshot: Path | str = DEFAULT_SNAPSHOT, *, vision: bool = True, **model_kwargs
    ) -> "TTDecider":
        """``vision=False`` skips the vision tower (text-only use, ~0.96 GiB less DRAM)."""
        from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

        model = PplxDeciderModel.from_snapshot(device, SnapshotReader(snapshot), vision=vision, **model_kwargs)
        return cls(model, AppTokenizer(Path(snapshot)))

    def row(self, state: str, question, images: Sequence = ()) -> dict:
        row = {"state": state, "question": question}
        if images:
            row["images"] = list(images)
        return row

    def prepare(self, state: str, question) -> tuple[dict, list[int], int]:
        """The app's row, its token ids and option count (``DecisionModel.prepare`` at batch 1, text only)."""
        row = self.row(state, question)
        return row, self.tokenizer.input_ids(row), self.tokenizer.count(row)

    def encode(self, row: dict) -> dict:
        """``DecisionModel.prepare`` for one row: the processor output (images opened by ``open_image``)."""
        from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import encode

        return encode(self.tokenizer, row)

    def predict_probabilities(self, state: str, question, *, images: Sequence = ()) -> list[float]:
        if not images:
            _, ids, count = self.prepare(state, question)
            return self.model.decide(ids, count)
        row = self.row(state, question, images)
        return self.model.decide_encoded(self.encode(row), self.tokenizer.count(row))

    def predict(self, state: str, question, *, images: Sequence = ()):
        """Return a choice, yes/no probability, or score using the saved temperature (``Decider.predict``)."""
        return self.tokenizer.autojev.answer(question, self.predict_probabilities(state, question, images=images))
