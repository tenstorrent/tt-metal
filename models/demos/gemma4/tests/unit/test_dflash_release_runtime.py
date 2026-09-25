# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``release_persistent_capture`` on the base ``Generator`` and DiffusionGemma.

These import the production modules, so they need the TT runtime and run on a
device host; the host-only chain tests are in ``test_dflash_capture_release.py``.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("ttnn.device", reason="needs the TT runtime")
import ttnn  # noqa: E402


def _base_generator():
    from models.ttt_compat.tt.generator import Generator

    generator = Generator.__new__(Generator)
    released = []
    generator.model_args = [SimpleNamespace(mesh_device="mesh")]
    generator.model = [SimpleNamespace(sampling=SimpleNamespace(reset_trace=lambda: released.append("sampling")))]
    generator.trace_id_prefill = {"128_0": 11}
    generator.trace_id_prefill_sampling = {}
    generator.trace_ids_decode = {}
    generator.data_parallel = 1
    return generator, released


def test_base_release_runs_once_and_the_destructor_only_runs_the_base(monkeypatch):
    generator, released = _base_generator()
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: released.append(tid))
    generator.release_persistent_capture()
    generator.release_persistent_capture()
    assert released.count(11) == 1 and released.count("sampling") == 1
    generator.__del__()
    assert released.count(11) == 1


def test_diffusion_gemma_release_reaches_the_base():
    from models.experimental.diffusion_gemma.tt.generator_vllm import DiffusionGemmaForCausalLM

    model = DiffusionGemmaForCausalLM.__new__(DiffusionGemmaForCausalLM)
    model._sessions = {}
    model._persistent_adapter = None
    model.model_args = []
    model.model = []
    model.release_persistent_capture()
    assert getattr(model, "_generator_capture_released", False) is True
