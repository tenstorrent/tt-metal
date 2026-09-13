# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A lookup table is resident, not streamed, and the ceiling has to know the difference.

The byte model's rule for weights is that one unit of work reads its subtree's weights once. True of
every weight a matmul consumes; false of an embedding, which is indexed. Voxtral's embed_tokens is
[131072, 3072] -- 805 MB -- and a decode step reads the ROW for its token, about 6 KB.

That is roughly a quarter of decode's modelled read set, which is why decode alone printed above
100% of DRAM peak: prefill carries the same error against a read set five times larger (5%), and
encode not at all, since audio_tower has no lookup table.

THE TOOL CANNOT DERIVE IT. embed_tokens and lm_head are both [131072, 3072], the same size, in the
same checkpoint section -- one gathered, one streamed by the vocabulary matmul. The profiler could
tell them apart (EmbeddingsDeviceOperation against MatmulDeviceOperation) but its buckets carry no
byte counts. The pipeline holds both tensors, so it states which is which, keyed by the checkpoint's
own subtree name -- the same key stage_roots resolves a stage to.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_BLOCKS = {
    "audio_tower": {"layers": 32, "hidden_size": 1280, "intermediate_size": 5120, "kv_heads": 20, "head_dim": 64},
    "language_model": {
        "layers": 30,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "kv_heads": 8,
        "head_dim": 128,
        "params": 4_014_000_000,
    },
}
_ROOTS = {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"}
_SECS = {"audio_tower": 380_081_696, "language_model": 1_338_000_000}


def _roofs(monkeypatch, gathered):
    import cc_optimize.perf_mcp as PM
    import cc_optimize.summary as S

    monkeypatch.setattr(S, "_prompt_tokens", lambda: 128)
    monkeypatch.setattr(S, "_request_batch", lambda: 8)
    monkeypatch.setattr(S, "_SECTION_BYTES", {"audio_tower": 1_328_000_000, "language_model": 8_028_000_000})
    monkeypatch.setattr(PM, "read_stage_isl_map", lambda *a, **k: {"encode": 1500, "prefill": 4096, "decode": 1})
    monkeypatch.setattr(PM, "read_stage_isl_per_request_map", lambda *a, **k: {"prefill": 128})
    monkeypatch.setattr(S, "_fidelity_breakdown", lambda p: ([("hifi2", 1.0, 351.0e12, 0.0)], None))
    mf = {
        "device_weight_bytes": 1_718_081_696,
        "device_section_bytes": _SECS,
        "dominant_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "blocks": _BLOCKS,
        "stage_roots": _ROOTS,
    }
    if gathered:
        mf["gathered_weight_bytes"] = gathered
    monkeypatch.setattr(S, "_model_facts", lambda: mf)
    return S._stage_roofs(
        1_718_081_696,
        512.0,
        1,
        "tok/s/u",
        {"buckets": []},
        {"encode": 3.9087, "prefill": 42.6459, "decode": 2.5261},
    )


def test_a_gathered_table_comes_off_the_stages_that_run_its_subtree(monkeypatch):
    before = _roofs(monkeypatch, None)
    after = _roofs(monkeypatch, {"language_model": 805_306_368})

    assert after["decode"]["bytes"] < before["decode"]["bytes"]
    assert before["decode"]["bytes"] - after["decode"]["bytes"] == 805_306_368
    assert after["prefill"]["bytes"] < before["prefill"]["bytes"], "prefill runs the same subtree"


def test_a_subtree_with_no_lookup_is_untouched(monkeypatch):
    before = _roofs(monkeypatch, None)
    after = _roofs(monkeypatch, {"language_model": 805_306_368})

    assert after["encode"]["bytes"] == before["encode"]["bytes"], "audio_tower has no lookup table"


def test_it_moves_decode_under_its_measurement(monkeypatch):
    """The point of the exercise: 2.5261 ms measured against a floor that used to be above it."""
    before = _roofs(monkeypatch, None)["decode"]["memory_ms"]
    after = _roofs(monkeypatch, {"language_model": 805_306_368})["decode"]["memory_ms"]

    assert before > 2.5261, "the premise changed: decode no longer exceeded its ceiling"
    assert after < 2.5261, (before, after)


def test_a_pipeline_that_states_nothing_changes_no_number(monkeypatch):
    assert _roofs(monkeypatch, {})["decode"]["bytes"] == _roofs(monkeypatch, None)["decode"]["bytes"]


def test_the_marker_round_trips():
    from cc_optimize.perf_mcp import _parse_gathered_weights

    assert _parse_gathered_weights("TRACE_GATHERED_WEIGHTS=language_model:805306368\n") == {"language_model": 805306368}
    assert _parse_gathered_weights("nothing here") == {}
    assert _parse_gathered_weights("TRACE_GATHERED_WEIGHTS=bad:notanint") == {}


def test_the_subtraction_names_no_stage():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("gathered_weight_bytes")
    line = src[src.rindex("\n", 0, i) + 1 : src.index("\n", i)]
    for name in ("decode", "prefill", "encode"):
        assert '"%s"' % name not in line, line
    assert "_root_of(stage)" in line, line
