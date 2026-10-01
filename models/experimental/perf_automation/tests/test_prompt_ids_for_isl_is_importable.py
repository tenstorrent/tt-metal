# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A generated perf test may import ``prompt_ids_for_isl`` from perf_test_gen at module top level.

The helper builds a prompt of EXACTLY the requested ISL from the model's own tokenizer, so the
input length is the tool's measurement condition rather than an example sentence written into each
test. Generated tests carry it as an inline ``try/except`` fallback, but at least one generated
file imported it unconditionally at module scope -- and because it was NOT a top-level export of
perf_test_gen, that ImportError failed collection of the ENTIRE tests directory, wedging an
optimize run at perf-test build (observed 2026-09-22: test_text_generation_perf.py line 28). It is
now a real export, so either import shape resolves and one generated file can no longer take the
whole directory's collection down with it.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


class _Tok:
    def encode(self, s, add_special_tokens=False):
        return [11, 12, 13, 14]


def test_prompt_ids_for_isl_is_a_top_level_export():
    from agent.perf_test_gen import prompt_ids_for_isl  # must resolve at module scope

    assert callable(prompt_ids_for_isl)


def test_prompt_ids_for_isl_returns_exactly_n_tokens():
    import torch

    from agent.perf_test_gen import prompt_ids_for_isl

    for n in (1, 4, 8, 128):
        out = prompt_ids_for_isl(_Tok(), n)
        assert out.dtype == torch.long
        assert int(out.numel()) == n, (n, out.numel())


def test_prompt_ids_for_isl_tolerates_a_tokenizer_without_kwarg():
    from agent.perf_test_gen import prompt_ids_for_isl

    class _Old:
        def encode(self, s):  # no add_special_tokens kwarg
            return [7, 8]

    assert int(prompt_ids_for_isl(_Old(), 5).numel()) == 5
