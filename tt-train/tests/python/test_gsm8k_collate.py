# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the GSM8K recipe collate (no device required)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ttnn

from ttml.datasets import Batch

EXAMPLE = Path(__file__).resolve().parents[2] / "sources" / "examples" / "gsm8k_finetune" / "gsm8k_finetune.py"
EOS = 2


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @staticmethod
    def from_numpy(arr, layout, dtype):
        return _FakeTensor(arr)

    def to_numpy(self, dtype=None, composer=None):
        return self._arr


# The collate looks up `ttml.autograd.Tensor.from_numpy` in its module globals, so it is given a stand-in
# `ttml` namespace instead of patching the (nanobind) ttml types.
_FAKE_TTML = SimpleNamespace(autograd=SimpleNamespace(Tensor=_FakeTensor))


def _load_collate():
    """Compile only gsm8k_collate_fn, so the script's heavy imports are not needed."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gsm8k_collate_fn")
    ns = {"np": np, "ttml": _FAKE_TTML, "ttnn": ttnn, "Batch": Batch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(EXAMPLE), "exec"), ns)
    return ns["gsm8k_collate_fn"]


def _trained_pairs(batch, row):
    labels = batch.labels.to_numpy()[row]
    mask = batch.loss_mask.to_numpy()[row, 0, :, 0]
    return [(t, int(labels[t])) for t in range(len(labels)) if mask[t] > 0]


def _hf_pairs(question, answer, max_len):
    """HF convention: labels = input_ids with the prompt set to -100, shifted inside the loss."""
    ids = list(question) + list(answer)
    if len(ids) > max_len:
        keep = max_len - len(answer)
        ids = list(question[:keep]) + list(answer) if keep > 0 else list(answer[:max_len])
        n_prompt = max(keep, 0)
    else:
        n_prompt = len(question)
    return [(k - 1, ids[k]) for k in range(max(n_prompt, 1), len(ids))]


QUESTION = [10, 11, 12]
ANSWER = [20, 21, 22, 23]


# The collate appends EOS, so the answer is 5 tokens. At 5 the whole answer fits and only the question is dropped;
# at 4 and 3 the answer itself is truncated.
@pytest.mark.parametrize("max_len", [10, 8, 7, 6, 5, 4, 3])
def test_trained_pairs_match_hf_convention(max_len):
    collate = _load_collate()
    batch = collate([(QUESTION, ANSWER)], EOS, max_len)
    assert _trained_pairs(batch, 0) == _hf_pairs(QUESTION, ANSWER + [EOS], max_len)


def test_mask_is_normalised_over_the_batch():
    collate = _load_collate()
    batch = collate([(QUESTION, ANSWER), (QUESTION[:2], ANSWER[:2])], EOS, 10)
    mask = batch.loss_mask.to_numpy()
    assert mask.sum() == pytest.approx(2 * 10)
    assert _trained_pairs(batch, 1) == [(1, 20), (2, 21), (3, EOS)]
