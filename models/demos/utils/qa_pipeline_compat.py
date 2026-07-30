# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Standalone extractive question-answering pipeline, vendored for transformers 5.x.

transformers 5.x removed the extractive ``"question-answering"`` pipeline
(``transformers.pipelines.question_answering`` / ``QuestionAnsweringPipeline``);
only ``table-`` and ``document-question-answering`` remain. Several tt-metal BERT
demos rely on that pipeline's helper methods (``_sanitize_parameters``,
``_args_parser``, ``preprocess``, ``postprocess``) to do the QA feature-prep and
answer-span decoding around the TT model.

This is a faithful copy of transformers 4.53.0's ``QuestionAnsweringPipeline``
(Apache-2.0, https://github.com/huggingface/transformers, tag v4.53.0,
``src/transformers/pipelines/question_answering.py``) reduced to a standalone class
that does NOT subclass the (heavily refactored) 5.x ``Pipeline`` base — the demos
only call the four methods above directly, never the orchestration (``__call__`` /
``run`` / ``_forward`` via the base). Behavior is preserved so the SQuAD-v2
accuracy numbers match the pre-bump pipeline. All transformers building blocks it
needs (``SquadExample``/``SquadFeatures``/``squad_convert_examples_to_features``,
``ArgumentHandler``, ``PaddingStrategy``) still ship in 5.x.
"""

from __future__ import annotations

import inspect
import types
import warnings
from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import torch
from transformers.data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from transformers.pipelines.base import ArgumentHandler
from transformers.utils import PaddingStrategy, logging

logger = logging.get_logger(__name__)

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch always present in CI
    Dataset = None


def decode_spans(
    start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int, undesired_tokens: np.ndarray
) -> tuple:
    """Generate the top-k answer spans from start/end logits (see transformers 4.53)."""
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
    starts = starts[desired_spans]
    ends = ends[desired_spans]
    scores = candidates[0, starts, ends]

    return starts, ends, scores


def select_starts_ends(
    start,
    end,
    p_mask,
    attention_mask,
    min_null_score=1000000,
    top_k=1,
    handle_impossible_answer=False,
    max_answer_len=15,
):
    """Normalize raw QA logits and decode answer spans (see transformers 4.53)."""
    # The tt BERT demos feed start/end logits as a 1-D tensor padded to max_seq_len (e.g. (384,)),
    # while p_mask/attention_mask carry the tokenizer's per-feature shape (1, seq) — and `seq` may be
    # shorter than the padded length (slow vs fast tokenizer / squadv2 path). Align start/end to
    # p_mask's shape (slice to its seq length, restore the batch dim) so the np.where masking, the
    # softmax (axis=-1), the CLS masking (start[0, 0]) and decode_spans all operate consistently.
    # (The real QA pipeline's _forward returned model-shaped logits, so this never tripped there.)
    p_mask = np.asarray(p_mask)
    _seq = p_mask.shape[-1]
    start = np.asarray(start).reshape(-1)[:_seq].reshape(p_mask.shape)
    end = np.asarray(end).reshape(-1)[:_seq].reshape(p_mask.shape)
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask).reshape(-1)[:_seq].reshape(p_mask.shape)

    # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
    undesired_tokens = np.abs(np.array(p_mask) - 1)

    if attention_mask is not None:
        undesired_tokens = undesired_tokens & attention_mask

    # Generate mask
    undesired_tokens_mask = undesired_tokens == 0.0

    # Make sure non-context indexes in the tensor cannot contribute to the softmax
    start = np.where(undesired_tokens_mask, -10000.0, start)
    end = np.where(undesired_tokens_mask, -10000.0, end)

    # Normalize logits and spans to retrieve the answer
    start = np.exp(start - start.max(axis=-1, keepdims=True))
    start = start / start.sum()

    end = np.exp(end - end.max(axis=-1, keepdims=True))
    end = end / end.sum()

    if handle_impossible_answer:
        min_null_score = min(min_null_score, (start[0, 0] * end[0, 0]).item())

    # Mask CLS
    start[0, 0] = end[0, 0] = 0.0

    starts, ends, scores = decode_spans(start, end, top_k, max_answer_len, undesired_tokens)
    return starts, ends, scores, min_null_score


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """Maps (question, context) arguments to internal ``SquadExample`` objects (transformers 4.53)."""

    def normalize(self, item):
        if isinstance(item, SquadExample):
            return item
        elif isinstance(item, dict):
            for k in ["question", "context"]:
                if k not in item:
                    raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")
                elif item[k] is None:
                    raise ValueError(f"`{k}` cannot be None")
                elif isinstance(item[k], str) and len(item[k]) == 0:
                    raise ValueError(f"`{k}` cannot be empty")

            return QuestionAnsweringPipeline.create_sample(**item)
        raise ValueError(f"{item} argument needs to be of type (SquadExample, dict)")

    def __call__(self, *args, **kwargs):
        # Detect where the actual inputs are
        if args is not None and len(args) > 0:
            if len(args) == 1:
                inputs = args[0]
            elif len(args) == 2 and {type(el) for el in args} == {str}:
                inputs = [{"question": args[0], "context": args[1]}]
            else:
                inputs = list(args)
        elif "X" in kwargs:
            inputs = kwargs["X"]
        elif "data" in kwargs:
            inputs = kwargs["data"]
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], list) and isinstance(kwargs["context"], str):
                inputs = [{"question": Q, "context": kwargs["context"]} for Q in kwargs["question"]]
            elif isinstance(kwargs["question"], list) and isinstance(kwargs["context"], list):
                if len(kwargs["question"]) != len(kwargs["context"]):
                    raise ValueError("Questions and contexts don't have the same lengths")

                inputs = [{"question": Q, "context": C} for Q, C in zip(kwargs["question"], kwargs["context"])]
            elif isinstance(kwargs["question"], str) and isinstance(kwargs["context"], str):
                inputs = [{"question": kwargs["question"], "context": kwargs["context"]}]
            else:
                raise ValueError("Arguments can't be understood")
        else:
            raise ValueError(f"Unknown arguments {kwargs}")

        # When user is sending a generator we need to trust it's a valid example
        generator_types = (types.GeneratorType, Dataset) if Dataset is not None else (types.GeneratorType,)
        if isinstance(inputs, generator_types):
            return inputs

        # Normalize inputs
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif isinstance(inputs, Iterable):
            # Copy to avoid overriding arguments
            inputs = list(inputs)
        else:
            raise ValueError(f"Invalid arguments {kwargs}")

        for i, item in enumerate(inputs):
            inputs[i] = self.normalize(item)

        return inputs


class QuestionAnsweringPipeline:
    """Extractive QA pipeline vendored from transformers 4.53 (standalone; torch-only).

    Construct as ``QuestionAnsweringPipeline(model=..., tokenizer=...)`` and use exactly
    like the removed ``pipeline("question-answering", ...)`` for the helper-method flow:
    ``_sanitize_parameters`` / ``_args_parser`` / ``preprocess`` / ``postprocess``.
    """

    default_input_names = "question,context"
    handle_impossible_answer = False
    framework = "pt"  # transformers 5.x is torch-only; the base no longer tracks `framework`

    def __init__(self, model, tokenizer: PreTrainedTokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self._args_parser = QuestionAnsweringArgumentHandler()

    def __call__(self, *args, **kwargs):
        """Drop-in for transformers' ``QuestionAnsweringPipeline.__call__``.

        transformers 5.x removed the extractive pipeline; this standalone shim doesn't subclass
        the base ``Pipeline`` (which previously supplied ``__call__``), so run the full
        question/context -> answer flow here by chaining the shim's own methods
        (``_args_parser`` -> ``preprocess`` -> ``_forward`` -> ``postprocess``). Returns a single
        answer dict for one example, or a list for batched inputs (matching the 4.53 behavior).
        """
        examples = self._args_parser(*args, **kwargs)
        preprocess_params, _, postprocess_params = self._sanitize_parameters()
        answers = []
        for example in examples:
            features = list(self.preprocess(example, **preprocess_params))
            # Run the reference model under no_grad — otherwise its outputs require grad and
            # postprocess's .numpy() raises "Can't call numpy() on Tensor that requires grad"
            # (the real pipeline ran inference in no-grad context).
            with torch.no_grad():
                model_outputs = [self._forward(feature) for feature in features]
            answers.append(self.postprocess(model_outputs, **postprocess_params))
        return answers[0] if len(answers) == 1 else answers

    @staticmethod
    def create_sample(
        question: Union[str, list[str]], context: Union[str, list[str]]
    ) -> Union[SquadExample, list[SquadExample]]:
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def _sanitize_parameters(
        self,
        padding=None,
        topk=None,
        top_k=None,
        doc_stride=None,
        max_answer_len=None,
        max_seq_len=None,
        max_question_len=None,
        handle_impossible_answer=None,
        align_to_words=None,
        **kwargs,
    ):
        # Set defaults values
        preprocess_params = {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if doc_stride is not None:
            preprocess_params["doc_stride"] = doc_stride
        if max_question_len is not None:
            preprocess_params["max_question_len"] = max_question_len
        if max_seq_len is not None:
            preprocess_params["max_seq_len"] = max_seq_len

        postprocess_params = {}
        if topk is not None and top_k is None:
            warnings.warn("topk parameter is deprecated, use top_k instead", UserWarning)
            top_k = topk
        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            if max_answer_len < 1:
                raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer
        if align_to_words is not None:
            postprocess_params["align_to_words"] = align_to_words
        return preprocess_params, {}, postprocess_params

    def preprocess(self, example, padding="do_not_pad", doc_stride=None, max_question_len=64, max_seq_len=None):
        # XXX: This is special, args_parser will not handle anything generator or dataset like
        # For those we expect user to send a simple valid example either directly as a SquadExample or simple dict.
        if isinstance(example, dict):
            example = SquadExample(None, example["question"], example["context"], None, None, None)

        if max_seq_len is None:
            max_seq_len = min(self.tokenizer.model_max_length, 384)
        if doc_stride is None:
            doc_stride = min(max_seq_len // 2, 128)

        if doc_stride > max_seq_len:
            raise ValueError(f"`doc_stride` ({doc_stride}) is larger than `max_seq_len` ({max_seq_len})")

        if not self.tokenizer.is_fast:
            features = squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_seq_length=max_seq_len,
                doc_stride=doc_stride,
                max_query_length=max_question_len,
                padding_strategy=PaddingStrategy.MAX_LENGTH,
                is_training=False,
                tqdm_enabled=False,
            )
        else:
            # Define the side we want to truncate / pad and the text/pair sorting
            question_first = self.tokenizer.padding_side == "right"

            encoded_inputs = self.tokenizer(
                text=example.question_text if question_first else example.context_text,
                text_pair=example.context_text if question_first else example.question_text,
                padding=padding,
                truncation="only_second" if question_first else "only_first",
                max_length=max_seq_len,
                stride=doc_stride,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            num_spans = len(encoded_inputs["input_ids"])

            # p_mask: 1 for tokens that cannot be in the answer (question + special tokens), 0 for context.
            p_mask = [
                [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(span_id)]
                for span_id in range(num_spans)
            ]

            features = []
            for span_idx in range(num_spans):
                input_ids_span_idx = encoded_inputs["input_ids"][span_idx]
                attention_mask_span_idx = (
                    encoded_inputs["attention_mask"][span_idx] if "attention_mask" in encoded_inputs else None
                )
                token_type_ids_span_idx = (
                    encoded_inputs["token_type_ids"][span_idx] if "token_type_ids" in encoded_inputs else None
                )
                # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
                if self.tokenizer.cls_token_id is not None:
                    cls_indices = np.nonzero(np.array(input_ids_span_idx) == self.tokenizer.cls_token_id)[0]
                    for cls_index in cls_indices:
                        p_mask[span_idx][cls_index] = 0
                submask = p_mask[span_idx]
                features.append(
                    SquadFeatures(
                        input_ids=input_ids_span_idx,
                        attention_mask=attention_mask_span_idx,
                        token_type_ids=token_type_ids_span_idx,
                        p_mask=submask,
                        encoding=encoded_inputs[span_idx],
                        cls_index=None,
                        token_to_orig_map={},
                        example_index=0,
                        unique_id=0,
                        paragraph_len=0,
                        token_is_max_context=0,
                        tokens=[],
                        start_position=0,
                        end_position=0,
                        is_impossible=False,
                        qas_id=None,
                    )
                )

        for i, feature in enumerate(features):
            fw_args = {}
            others = {}
            model_input_names = self.tokenizer.model_input_names + ["p_mask", "token_type_ids"]

            for k, v in feature.__dict__.items():
                if k in model_input_names:
                    tensor = torch.tensor(v)
                    if tensor.dtype == torch.int32:
                        tensor = tensor.long()
                    fw_args[k] = tensor.unsqueeze(0)
                else:
                    others[k] = v

            is_last = i == len(features) - 1
            yield {"example": example, "is_last": is_last, **fw_args, **others}

    def _forward(self, inputs):
        example = inputs["example"]
        model_inputs = {k: inputs[k] for k in self.tokenizer.model_input_names}
        # `XXXForSequenceClassification` models should not use `use_cache=True` even if it's supported
        model_forward = self.model.forward
        if "use_cache" in inspect.signature(model_forward).parameters.keys():
            model_inputs["use_cache"] = False
        output = self.model(**model_inputs)
        if isinstance(output, dict):
            return {"start": output["start_logits"], "end": output["end_logits"], "example": example, **inputs}
        else:
            start, end = output[:2]
            return {"start": start, "end": end, "example": example, **inputs}

    def postprocess(
        self,
        model_outputs,
        top_k=1,
        handle_impossible_answer=False,
        max_answer_len=15,
        align_to_words=True,
    ):
        min_null_score = 1000000  # large and positive
        answers = []
        for output in model_outputs:
            if output["start"].dtype == torch.bfloat16:
                start_ = output["start"].to(torch.float32)
                end_ = output["end"].to(torch.float32)
            else:
                start_ = output["start"]
                end_ = output["end"]
            example = output["example"]
            p_mask = output["p_mask"]
            attention_mask = (
                output["attention_mask"].numpy() if output.get("attention_mask", None) is not None else None
            )

            starts, ends, scores, min_null_score = select_starts_ends(
                start_, end_, p_mask, attention_mask, min_null_score, top_k, handle_impossible_answer, max_answer_len
            )

            if not self.tokenizer.is_fast:
                char_to_word = np.array(example.char_to_word_offset)
                for s, e, score in zip(starts, ends, scores):
                    token_to_orig_map = output["token_to_orig_map"]
                    answers.append(
                        {
                            "score": score.item(),
                            "start": np.where(char_to_word == token_to_orig_map[s])[0][0].item(),
                            "end": np.where(char_to_word == token_to_orig_map[e])[0][-1].item(),
                            "answer": " ".join(example.doc_tokens[token_to_orig_map[s] : token_to_orig_map[e] + 1]),
                        }
                    )
            else:
                question_first = bool(self.tokenizer.padding_side == "right")
                enc = output["encoding"]

                # Encoding was *not* padded, input_ids *might* be (left-padding shifts offsets).
                if self.tokenizer.padding_side == "left":
                    offset = (output["input_ids"] == self.tokenizer.pad_token_id).numpy().sum()
                else:
                    offset = 0

                sequence_index = 1 if question_first else 0

                for s, e, score in zip(starts, ends, scores):
                    s = s - offset
                    e = e - offset

                    start_index, end_index = self.get_indices(enc, s, e, sequence_index, align_to_words)

                    target_answer = example.context_text[start_index:end_index]
                    answer = self.get_answer(answers, target_answer)

                    if answer:
                        answer["score"] += score.item()
                    else:
                        answers.append(
                            {
                                "score": score.item(),
                                "start": start_index,
                                "end": end_index,
                                "answer": example.context_text[start_index:end_index],
                            }
                        )

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:top_k]
        if len(answers) == 1:
            return answers[0]
        return answers

    def get_answer(self, answers: list[dict], target: str) -> Optional[dict]:
        for answer in answers:
            if answer["answer"].lower() == target.lower():
                return answer
        return None

    def get_indices(self, enc, s: int, e: int, sequence_index: int, align_to_words: bool) -> tuple[int, int]:
        if align_to_words:
            try:
                start_word = enc.token_to_word(s)
                end_word = enc.token_to_word(e)
                start_index = enc.word_to_chars(start_word, sequence_index=sequence_index)[0]
                end_index = enc.word_to_chars(end_word, sequence_index=sequence_index)[1]
            except Exception:
                # Some tokenizers don't really handle words. Keep to offsets then.
                start_index = enc.offsets[s][0]
                end_index = enc.offsets[e][1]
        else:
            start_index = enc.offsets[s][0]
            end_index = enc.offsets[e][1]
        return start_index, end_index
