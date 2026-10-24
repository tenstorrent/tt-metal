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
byte counts. The pipeline holds both tensors, so it declares its tables ONCE, as tensors
(stage_seams.GATHERED_TABLES_ATTR). trace_replay reads the declaration after setup() and hands it to
two readers: the census, which credits the tables' bytes under the same subtree names the weights are
credited to -- the key stage_roots resolves a stage to -- and prints them for the ESTIMATE path below;
and the per-stage hook, which charges an op reading a table beside an integer index for the rows
that came out (test_a_stage_streams_its_weights_not_what_it_writes). It used to be two declarations,
a method returning bytes per subtree that no pipeline ever wrote and an op-name set in the hook that
a model's own generic_op gather defeated.
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


# ------------------------------------------------- one declaration: the tensors, read once, two readers


class _Tbl:
    """A resident tensor with a buffer address, as the census and the hook both recognise one."""

    def __init__(self, numel, addr, dtype="bfloat16"):
        self.shape = (numel,)
        self.dtype = dtype
        self._addr = addr

    def storage_type(self):
        return "DEVICE"

    def buffer_address(self):
        return self._addr


class _M:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_the_census_credits_a_declared_table_to_the_subtrees_it_sits_in():
    """embed_tokens and lm_head: the same shape, the same size, the same subtree. The declared one is
    credited to every section name on its path, beside the weights, and the other is not."""
    from agent.weight_census import census, gathered_marker
    from cc_optimize.perf_mcp import _parse_gathered_weights

    table = _Tbl(131072 * 3072, 0x10)
    head = _Tbl(131072 * 3072, 0x20)
    c = census(_M(language_model=_M(embed_tokens=_M(weight=table), lm_head=_M(weight=head))), gathered=[table])
    nb = 131072 * 3072 * 2
    assert c["gathered_sections"] == {"language_model": nb, "embed_tokens": nb}
    assert c["sections"]["language_model"] == 2 * nb, "the weight groups still count every weight"
    assert _parse_gathered_weights(gathered_marker(c) + "\n") == {"language_model": nb, "embed_tokens": nb}


def test_a_table_declared_through_another_reference_still_matches():
    """Keyed by buffer, not by Python object: the pipeline may hand the census a view of its table."""
    from agent.weight_census import census

    c = census(_M(tower=_M(w=_Tbl(1000, 0x10))), gathered=[_Tbl(1000, 0x10)])
    assert c["gathered_sections"] == {"tower": 2000}


def test_nothing_declared_credits_nothing_and_prints_nothing():
    from agent.weight_census import census, gathered_marker

    c = census(_M(tower=_M(w=_Tbl(1000, 0x10))))
    assert c["gathered_sections"] == {}
    assert gathered_marker(c) == ""


def test_the_reader_flattens_a_method_or_an_attribute_and_forgives_a_failure():
    from agent.stage_marks import pipeline_gathered_tables

    t1, t2, t3 = object(), object(), object()

    class _Method:
        def gathered_tables(self):
            return {"embed": [t1, t2], "rope": (t3,)}

    class _Attr:
        gathered_tables = [t1, None, [t2]]

    class _Raises:
        def gathered_tables(self):
            raise RuntimeError("not built yet")

    assert pipeline_gathered_tables(_Method()) == [t1, t2, t3]
    assert pipeline_gathered_tables(_Attr()) == [t1, t2]
    assert pipeline_gathered_tables(_Raises()) == []
    assert pipeline_gathered_tables(object()) == []
    assert pipeline_gathered_tables(None) == []


def test_the_reader_spells_the_attribute_from_the_seams_module():
    from agent import stage_seams
    from agent.stage_marks import pipeline_gathered_tables

    t = object()
    assert pipeline_gathered_tables(type("P", (), {stage_seams.GATHERED_TABLES_ATTR: [t]})()) == [t]


def test_the_declaration_is_read_once_after_setup_and_handed_to_both_readers():
    """trace_replay reads the tables once setup() has made them, and hands the same list to the census
    and to every stage's hook. No method returning bytes per subtree and no op-name set survives."""
    src = (_PA / "agent" / "trace_replay.py").read_text()
    assert "pipeline_gathered_tables(" in src
    assert "gathered=_tables" in src, "the census is not handed the declaration"
    assert "_measure_stage(device, st, _stage_budget, tables=_tables)" in src, "the stage hook is not handed it"
    assert "gathered_weight_bytes" not in src, "a second declaration"
    assert "_GATHER_OPS" not in src, "a second rule"
    i = src.index("pipeline_gathered_tables(")
    assert "adapter.setup(device)" in src[:i], "the tables are read before setup() made them"
