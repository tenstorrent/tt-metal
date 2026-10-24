# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the generator re-captures a gpt_oss decode trace when programs were compiled since its capture.

Programs compiled after a decode trace capture (the first prefill of a new length) can hold long-lived device
resources, e.g. the global semaphores of the prefill's ttnn.all_reduce, that land in the trace's scratch region; every
decode replay then overwrites them and the next prefill of that length hangs or decodes garbage. Models that set
``_tt_recapture_decode_trace_on_new_programs`` get their decode trace (and its sampling traces) released at the start
of the next request and re-captured over the current allocations. No device needed.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.gpt_oss.tt.model import Model
from models.tt_transformers.tt.generator import Generator


class _Mesh:
    def __init__(self):
        self.entries = 10

    def num_program_cache_entries(self):
        return self.entries


class _Sampling:
    def __init__(self):
        self.resets = 0

    def reset_trace(self):
        self.resets += 1


def _generator(model_cls):
    mesh = _Mesh()
    model = model_cls.__new__(model_cls)
    model.sampling = _Sampling()
    gen = Generator.__new__(Generator)
    gen.model = [model]
    gen.model_args = [SimpleNamespace(mesh_device=mesh)]
    gen.data_parallel = 1
    gen.trace_ids_decode = defaultdict(lambda: None)
    gen.trace_inputs_decode = defaultdict(lambda: None)
    gen.trace_output_decode = defaultdict(lambda: None)
    gen._decode_trace_cache_entries = {}
    gen.trace_ids_decode[True] = {0: "decode-trace"}
    gen.trace_inputs_decode[True] = ["inputs"]
    gen.trace_output_decode[True] = ["logits"]
    gen._decode_trace_cache_entries[True] = gen._program_cache_entries()
    return gen, mesh, model


def test_gpt_oss_opts_in():
    assert Model._tt_recapture_decode_trace_on_new_programs is True


def test_stale_after_new_programs_and_release(monkeypatch):
    released = []
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: released.append(trace_id))
    gen, mesh, model = _generator(Model)

    assert not gen._decode_trace_is_stale(True)
    mesh.entries += 3  # a prefill of a new length compiled programs after the capture
    assert gen._decode_trace_is_stale(True)

    gen._release_decode_trace(True)
    assert released == ["decode-trace"]
    assert model.sampling.resets == 1
    assert gen.trace_ids_decode[True] is None and gen.trace_inputs_decode[True] is None
    assert True not in gen._decode_trace_cache_entries
    assert not gen._decode_trace_is_stale(True)  # nothing recorded until the re-captured trace's first replay


@pytest.mark.parametrize("opt_in", [False, None])
def test_models_that_do_not_opt_in_keep_their_trace(opt_in):
    attrs = {} if opt_in is None else {"_tt_recapture_decode_trace_on_new_programs": opt_in}
    gen, mesh, _ = _generator(type("OtherModel", (), attrs))
    mesh.entries += 3
    assert not gen._decode_trace_is_stale(True)
