# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PPPrefill.run_span refuses a span that cannot continue the prompt, also under python -O (review finding 3: they
were bare asserts). CPU only: the checks run before any device work (a stub prefill stops at _use).

usage: PYTHONPATH=<tt-metal>:<tt-metal>/ttnn python -m pytest --noconftest -c /dev/null qwen36_resident/tests/test_pp_run_span.py
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Reached(Exception):
    """the span passed every check and would now run on the device"""


def stub():
    def use(C):
        raise Reached(C)

    return SimpleNamespace(
        _geo={1024: {}, 512: {}},
        max_len=8192,
        C=1024,
        n=4,
        _use=use,
        _check_trace_layout=lambda: None,
        is_long=lambda n_tokens, chunk: False,
    )


def span(pp, T, first_chunk, n_tokens, drain, chunk=1024):
    from qwen36_resident.prefill.pp_prefill import PPPrefill

    PPPrefill.run_span(pp, torch.zeros(T, dtype=torch.int64), first_chunk, n_tokens, drain=drain, chunk=chunk)


CASES = [
    # (T, first_chunk, n_tokens, drain, chunk, accepted)
    (2048, 0, 2048, False, 1024, True),  # two whole chunks
    (2047, 0, 2047, False, 1024, False),  # a partial chunk without the drain
    (1000, 2, 3048, True, 1024, True),  # the draining tail
    (1000, 2, 3049, True, 1024, False),  # does not end at n_tokens
    (1024, 8, 9216, False, 1024, False),  # past max_len
    (1024, 0, 1024, False, 300, False),  # not a chunk size of the prefill
    (1024, 4, 3072, False, 512, True),  # a unit-aligned span at a smaller chunk (first_chunk counts in it)
    (0, 2, 1024, True, 512, True),  # an empty draining span (the final chunk is the prompt's last token)
]


@pytest.mark.parametrize("T,first,n,drain,chunk,ok", CASES)
def test_span_checks(T, first, n, drain, chunk, ok):
    pp = stub()
    if first:  # the prompt's first span fixed the chunk size
        pp._span_chunk = chunk
    with pytest.raises(Reached if ok else ValueError):  # allow-pytest.raises: run with --noconftest
        span(pp, T, first, n, drain, chunk)


def test_the_chunk_size_is_fixed_by_the_first_span():
    pp = stub()
    with pytest.raises(Reached):  # allow-pytest.raises: run with --noconftest
        span(pp, 1024, 0, 1024, False, 1024)
    with pytest.raises(ValueError, match="differs"):  # allow-pytest.raises: run with --noconftest
        span(pp, 512, 2, 1536, False, 512)


def test_a_span_checks_the_trace_layout_as_run_does():
    """contract section 6a: run_span replays the captured ticks too, so it refuses a changed L1 layout before any"""
    pp = stub()

    def changed():
        raise RuntimeError("the decode models' L1 layout changed")

    pp._check_trace_layout = changed
    with pytest.raises(RuntimeError, match="L1 layout"):  # allow-pytest.raises: run with --noconftest
        span(pp, 1024, 0, 1024, False, 1024)


def test_the_checks_survive_python_O():
    code = (
        "import sys, torch; sys.path[:0] = sys.argv[1:]\n"
        "from qwen36_resident.tests.test_pp_run_span import stub, span\n"
        "try:\n    span(stub(), 2047, 0, 2047, False)\nexcept ValueError:\n    print('RAISED')\n"
    )
    p = subprocess.run([sys.executable, "-O", "-c", code, WT, f"{WT}/ttnn"], capture_output=True, text=True)
    assert "RAISED" in p.stdout, p.stdout + p.stderr[-2000:]


def test_a_tick_without_taps_does_no_tap_work():
    """the DFlash2 drafter's taps (PPPrefill(taps=...)) cost a prefill without a drafter nothing: no tap ring and no
    tap page table (both only with taps: tap_cache, tap_pt), no per-tick tap write (here: _control, shared by run and
    run_span), no device tap fill (_body: `if j in self.tap_cache`)"""
    from unittest.mock import patch

    from qwen36_resident.prefill import pp_prefill as PP

    def pp(taps):
        ns = SimpleNamespace(
            n=4, C=128, block=64, blocks=32, scratch0=32, tap_ring=34, mesh=None, tap_cache={0: "ring"} if taps else {},
            ctrl="ctrl", cos="cos", sin="sin", fill_pt="fill", read_pt="read", cstart="cstart",
            tap_pt="tap_pt" if taps else None,
        )  # fmt: skip
        ns._control_chips = lambda work: PP.PPPrefill._control_chips(ns, work)  # (_control builds on it)
        return ns

    with patch.object(PP.ttnn, "ShardTensorToMesh", lambda mesh, dim: None), patch.object(
        PP.ttnn, "from_torch", lambda t, **kw: t
    ):
        plain = PP.PPPrefill._control(pp(False), 1, 300)
        drafted = PP.PPPrefill._control(pp(True), 1, 300)
    assert [dst for _, dst in plain] == ["ctrl", "cos", "sin", "fill", "read", "cstart"]
    assert [dst for _, dst in drafted] == ["ctrl", "cos", "sin", "fill", "read", "cstart", "tap_pt"]
    # chip 1 runs chunk 0 at tick 1: its tap ring blocks 0, 1
    assert drafted[-1][0][1].tolist() == [0, 1]
