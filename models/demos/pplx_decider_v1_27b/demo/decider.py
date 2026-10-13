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

from loguru import logger

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import AppTokenizer
from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel, batch_cap, bucket_for, buckets_up_to
from models.demos.pplx_decider_v1_27b.tt.paths import resolve_model_path

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
        cls,
        mesh_device,
        model_path: Path | str | None = None,
        max_batch_size: int = 1,
        max_seq_len: int = 8192,
        vision: bool = True,
        *,
        snapshot: Path | str | None = None,
        **model_kwargs,
    ) -> "TTDecider":
        """Load the model on ``mesh_device`` (one device), the ``models/demos/gemma4`` way.

        ``model_path``: a local checkpoint directory or an HF repo id. ``None`` resolves, in order,
        ``$HF_MODEL``, ``$PPLX_DECIDER_SNAPSHOT`` (legacy), then the HF cache at the pinned revision
        (``tt/paths.py: resolve_model_path``). An HF repo id is looked up in the HF cache only
        (``perplexity-ai/pplx-decider-v1-27b`` at the pinned revision); a missing one raises with the
        ``hf download`` command.

        The device weight cache is located from ``model_path`` (``tt/paths.py:
        resolve_model_cache_path``): ``$TT_CACHE_PATH``, else next to the weights
        (``<model dir>/tt_weight_cache/<key>``). The first load fills it (27 GB, ~160 s).

        ``max_batch_size``: the most requests ``predict_batch`` runs in one forward, 1..32. Per bucket it
        is clamped to the measured DRAM cap (``tt/model.py: BATCH_CAPS``). ``predict`` is always batch 1.

        ``max_seq_len``: the largest prefill bucket used, one of 128 / 1024 / 2048 / 4096 / 8192.
        Prompts longer than it are rejected (nothing is truncated); smaller buckets are unchanged.

        ``vision=False`` skips the vision tower (text-only use, ~0.96 GiB less DRAM).
        ``vision_precision="bfp8"`` (a ``model_kwargs`` entry, or ``$PPLX_DECIDER_VISION_PRECISION``)
        is the opt-in BFP8 vision-weight setting; the default is ``"bf16"`` (doc/vision/README.md).
        ``snapshot=`` is a deprecated alias of ``model_path``.
        """
        from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

        if snapshot is not None:
            logger.warning("TTDecider.from_pretrained(snapshot=...) is deprecated; use model_path=")
            model_path = model_path or snapshot
        model_dir, source = resolve_model_path(model_path)
        logger.info(
            f"TTDecider: model path {model_dir} (from {source}), max_batch_size {max_batch_size}, "
            f"max_seq_len {max_seq_len}, vision {vision}"
        )
        model = PplxDeciderModel.from_snapshot(
            mesh_device,
            SnapshotReader(model_dir),
            vision=vision,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **model_kwargs,
        )
        caps = {b: batch_cap(b, max_batch_size) for b in buckets_up_to(max_seq_len)}
        logger.info(f"TTDecider: rows per forward by bucket {caps}")
        return cls(model, AppTokenizer(model_dir))

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

    # -- batches ----------------------------------------------------------------------------------
    def plan_batches(self, rows: Sequence[dict]) -> tuple[list[dict], list[int], list[list[int]]]:
        """Input prep and scheduling of ``predict_batch``, without the device.

        Returns ``(encoded, counts, batches)``: per row its model input (``{"input_ids"}`` for text, the
        processor output for an image row) and option count, and the batches as lists of row indices.
        Rows are grouped by prefill bucket (a row runs at its own bucket, so batching pads no row further
        than batch 1 does), in input order within a bucket, and each bucket's rows are split into
        batches of ``batch_cap(bucket, max_batch_size)``. Text and image rows of one bucket share batches.
        Every row is validated first: a row over ``max_seq_len`` tokens rejects the whole call.
        """
        import torch

        encoded, counts, by_bucket = [], [], {}
        for i, row in enumerate(rows):
            row = self.row(row["state"], row["question"], row.get("images", ()))
            if "images" in row:
                enc = self.encode(row)
            else:
                enc = {"input_ids": torch.tensor([self.tokenizer.input_ids(row)], dtype=torch.int64)}
            length = int(torch.as_tensor(enc["input_ids"]).numel())
            bucket = bucket_for(length, self.model.config.max_seq_len)
            encoded.append(enc)
            counts.append(self.tokenizer.count(row))
            by_bucket.setdefault(bucket, []).append(i)
        batches = []
        for bucket in sorted(by_bucket):
            cap = batch_cap(bucket, self.model.config.max_batch_size)
            members = by_bucket[bucket]
            batches += [members[s : s + cap] for s in range(0, len(members), cap)]
        return encoded, counts, batches

    def predict_probabilities_batch(self, rows: Sequence[dict]) -> list[list[float]]:
        """Probabilities per row (``{"state", "question", "images"?}``), in input order. See ``plan_batches``."""
        encoded, counts, batches = self.plan_batches(rows)
        out = [None] * len(encoded)
        for batch in batches:
            probs = self.model.decide_batch([encoded[i] for i in batch], [counts[i] for i in batch])
            for i, p in zip(batch, probs):
                out[i] = p
        return out

    def predict_batch(self, rows: Sequence[dict]) -> list:
        """``predict`` for many rows: one answer per row, in input order.

        ``rows``: dicts with ``state``, ``question`` and optional ``images`` (as ``predict``). Up to
        ``max_batch_size`` rows of one prefill bucket share a forward (``plan_batches``); each row reads its
        own last real token and applies its own option count and mask, with the checkpoint's temperature.
        """
        probs = self.predict_probabilities_batch(rows)
        return [self.tokenizer.autojev.answer(r["question"], p) for r, p in zip(rows, probs)]
