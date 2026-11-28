# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stage outside the autoregressive loop is not charged for a KV cache it does not have.

`_seq` in _bytes_for is the WORKLOAD's prompt, and active_bytes turns any non-zero seq_len into a KV
read of 2 x layers x kv_heads x head_dim x seq_len x batch. Voxtral's audio encoder was charged
32 layers x 20 kv heads x 64 head_dim x 128 TEXT TOKENS x 8 -- 413 MB measured directly -- for a
stage that never sees the text prompt and keeps no cache at all. A Whisper-style encoder attends
bidirectionally inside one forward pass; its K and V are intermediates, and the activation term
already prices those.

WHICH STAGES ARE IN THE LOOP IS RECORDED, NOT GUESSED, and needs no stage names:

  the recurring stage      retires one item per unit, and reads the accumulated history
  the prompt-consuming one is named by the per-request marker, and writes the cache the
                           recurring stage then reads

Anything else -- an encoder, a vocoder, a diffusion step -- gets seq_len 0, which zeroes the KV term
and leaves its activations untouched.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_BLOCK = {"layers": 32, "hidden_size": 1280, "intermediate_size": 5120, "kv_heads": 20, "head_dim": 64}
_MF = {"total_params": 637_000_000, "dominant_dtype": "bfloat16", "kv_dtype": "bfloat16"}


def test_the_kv_term_is_what_a_non_ar_stage_was_being_charged():
    """The size of the error, measured on voxtral's audio tower."""
    from agent.perf_target import active_bytes

    with_kv = active_bytes(_MF, seq_len=128, batch=8, items=187, block=_BLOCK)
    without = active_bytes(_MF, seq_len=0, batch=8, items=187, block=_BLOCK)

    assert with_kv - without > 400e6, (with_kv, without)
    assert without > 0, "zeroing the KV term must not zero the activations"


def test_a_stage_outside_the_loop_gets_no_kv(monkeypatch):
    import cc_optimize.summary as S
    import cc_optimize.perf_mcp as PM

    monkeypatch.setattr(S, "_prompt_tokens", lambda: 128)
    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {"encode": 1500, "prefill": 4096, "decode": 1})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {"prefill": 128})
    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1_328_000_000, "language_model": 8_028_000_000})
    mf = {
        "device_weight_bytes": 1_718_081_696,
        "dominant_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "blocks": {
            "audio_tower": _BLOCK,
            "language_model": {
                "layers": 30,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "kv_heads": 8,
                "head_dim": 128,
                "params": 4_014_000_000,
            },
        },
        "stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"},
    }
    monkeypatch.setattr(S, "_model_facts", lambda: mf)
    monkeypatch.setattr(S, "_fidelity_breakdown", lambda p: ([("hifi2", 1.0, 351.0e12, 0.0)], None))

    stage_ms = {"encode": 3.9087, "prefill": 42.6459, "decode": 2.5261}
    with_prefill = S._stage_roofs(1_718_081_696, 512.0, 1, "tok/s/u", {"buckets": []}, stage_ms)

    # prefill is named by the per-request marker, so it IS in the loop and keeps its KV
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {})
    without_prefill = S._stage_roofs(1_718_081_696, 512.0, 1, "tok/s/u", {"buckets": []}, stage_ms)

    assert (
        with_prefill["prefill"]["bytes"] > without_prefill["prefill"]["bytes"]
    ), "the prompt-consuming stage lost the KV it writes"
    assert (
        with_prefill["decode"]["bytes"] == without_prefill["decode"]["bytes"]
    ), "the recurring stage's KV depends on a marker it should not need"
    assert (
        with_prefill["encode"]["bytes"] == without_prefill["encode"]["bytes"]
    ), "the encoder's bytes move with a marker that has nothing to do with it"


def test_the_recurring_stage_keeps_its_kv_without_any_marker(monkeypatch):
    """items == 1 is what makes a stage recurring; it must not need to be named anywhere."""
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_per_request_stages", lambda: set())
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("_in_kv_loop =")
    line = src[i : src.index("\n", i)]
    assert "_items" in line and "<= 1" in line, line
    assert "_per_request_stages()" in line, line


def test_the_test_does_not_ask_what_a_stage_is_called():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("_in_kv_loop =")
    line = src[i : src.index("\n", i)]
    for name in ("decode", "prefill", "encode"):
        assert '"%s"' % name not in line, line
