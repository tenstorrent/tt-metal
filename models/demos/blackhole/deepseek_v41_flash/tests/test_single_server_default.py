# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only test of the server default of the single prefill chunk trace (``prefill_single.resolve_server_default``): when it is switched on it also pins DSV41_KEY_EXPORT_FILL_MAX
(the default 524288 sent the warm-up export of the 1M decode key slabs through slice_write; in that DRAM layout the single chunk trace gave wrong, run-to-run different prefill logits: needle 4817
lost at 8k / 16k, 4 identical prompts decoding differently; device regression: tests/test_chunk_probe.py with PROBE_WARM=1 PROBE_IDS=<needle tokens> PROBE_REPS=3 PROBE_RELEASE=1 under the server env,
first token must be 24243 every repetition), and every missing prerequisite falls back to the per-S_pad traces with a reason."""

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS

ENV = ("DSV41_PF_SINGLE", "DSV41_PF_PAGED", "DSV41_PF_PAGED_FP8", "DSV41_PREFILL_DYN", "DSV41_KEY_EXPORT_FILL_MAX")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(PS, "kv_len_tensor_available", lambda: True)


def test_default_on():
    import os

    on, why = PS.resolve_server_default(512, 1048576)
    assert on and os.environ["DSV41_PF_SINGLE"] == "1"
    assert (
        "DSV41_KEY_EXPORT_FILL_MAX" not in os.environ
    )  # (the model's default max(524288, slab rows) keeps the export in place)


@pytest.mark.parametrize(
    "env,chunk,ctx,needle",
    [
        ({"DSV41_PF_SINGLE": "0"}, 512, 1048576, "DSV41_PF_SINGLE=0"),
        ({"DSV41_PF_PAGED": "0"}, 512, 1048576, "read mode"),
        ({"DSV41_PF_PAGED_FP8": "1"}, 512, 1048576, "fp8"),
        ({}, 256, 1048576, "multiple of 512"),
        ({}, 512, 8192, "16384"),
    ],
)
def test_fallbacks(monkeypatch, env, chunk, ctx, needle):
    import os

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    on, why = PS.resolve_server_default(chunk, ctx)
    assert not on and needle in why
    assert os.environ["DSV41_PF_SINGLE"] == "0"


def test_missing_op_falls_back(monkeypatch):
    monkeypatch.setattr(PS, "kv_len_tensor_available", lambda: False)
    on, why = PS.resolve_server_default(512, 1048576)
    assert not on and "kv_len_tensor" in why


def test_slot_s0_and_per_row_switch(monkeypatch):
    assert PS.slot_s0(512, 4) == [512] * 4
    assert PS.slot_s0([0, 512, 1024, 1536], 4) == [0, 512, 1024, 1536]
    try:
        PS.slot_s0([0, 512], 4)
        raised = False
    except AssertionError:
        raised = True
    assert raised
    monkeypatch.setenv("DSV41_PF_SINGLE", "1")
    assert PS.per_row_enabled(1)
    assert not PS.per_row_enabled(2)  # one prefill slot per mesh row only
    monkeypatch.setenv("DSV41_PF_PERROW", "0")
    assert not PS.per_row_enabled(1)
    monkeypatch.delenv("DSV41_PF_PERROW")
    monkeypatch.setenv("DSV41_VISION", "1")
    assert PS.per_row_enabled(1)  # image prompts too
