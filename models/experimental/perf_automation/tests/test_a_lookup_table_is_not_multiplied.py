"""The compute floor multiplied an embedding table.

`2 x params x tokens` counts a multiply-accumulate for every parameter once per token -- right for a
WEIGHT in a matmul, wrong for a lookup table. An embedding is read by INDEX: one row per token, no
multiply. blocks[root]["params"] is the tower's SIZE, so it includes the table.

Voxtral prefill: 4.014B instead of 3.611B, so 2 x 0.403B x 4096 = 3.30 TFLOP of phantom matmul --
18.8 ms of a 222.61 ms floor, 9.2% too slow, making the stage read closer to its ceiling than it is.
Encode is unaffected: an audio tower has no token embedding.

The rule already existed (model_bytes._LOOKUP_ONLY, which total_params applies); blocks[] arrived
later for multi-tower models and recorded only the size."""
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


def test_the_generator_carries_the_tensor_name():
    """A consumer cannot tell a table from a weight without it; the section alone cannot say."""
    from agent.weight_census import _checkpoint_tensor_sections

    import inspect

    src = inspect.getsource(_checkpoint_tensor_sections)
    assert 'yield numel, str(name).split(".", 1)[0], str(name)' in src


def test_the_producer_records_both_counts():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index('_geo["params"] = int(_pp[_root])')
    stanza = src[i : i + 400]
    assert '_geo["matmul_params"]' in stanza, "the tower still records only its size"
    assert '_geo["lookup_params"]' in stanza


def test_it_uses_the_same_rule_as_model_bytes():
    """Two definitions of 'lookup-only' would drift; there must be one."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "from agent.model_bytes import _LOOKUP_ONLY" in src


def test_the_compute_floor_prefers_matmul_params():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index('_mm = int(_blk.get("matmul_params") or 0)')
    stanza = src[i : i + 400]
    assert '_params = _mm or int(_blk.get("params") or 0)' in stanza


def test_a_tower_with_no_table_is_untouched():
    """An audio encoder has no token embedding, so nothing is subtracted and encode does not move."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("if _lk.get(_root):")
    assert '_geo["matmul_params"]' in src[i : i + 300], "matmul_params is set even with no lookup tensors"


def test_a_producer_that_predates_this_still_works():
    """An older facts file has neither key; the floor must fall back to the raw size, not to zero."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index('_mm = int(_blk.get("matmul_params") or 0)')
    assert "or int(params or 0)" in src[i : i + 400]


# ---------------------------------------------------------- observed, not named


def _run_mod():
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("cc_run_obs", _PA / "cc_optimize" / "run.py")
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_PROF = {
    "buckets": [
        {
            "id": "embedding",
            "top_ops": [
                {"op_code": "EmbeddingsDeviceOperation", "shape": "1x1 @ 131072x3072", "count": 127},
                {"op_code": "EmbeddingsDeviceOperation", "shape": "1x1 @ 640x128", "count": 254},
            ],
        },
        {"id": "matmul", "top_ops": [{"op_code": "MatmulDeviceOperation", "shape": "32x3072 @ 3072x131072"}]},
    ]
}


def test_gathers_are_read_from_the_profile_not_from_names():
    """A model that calls its table something new is invisible to a name list. A gather is an op the
    device RAN, so the profile can say it whatever the tensor is called."""
    m = _run_mod()
    assert sorted(m.observed_gathered_numels(_PROF)) == [81920, 131072 * 3072]


def test_only_the_embedding_bucket_counts():
    """The matmul bucket carries the SAME 131072x3072 operand -- lm_head. It must not be read as a
    gather; the op class is what separates them."""
    m = _run_mod()
    only_matmul = {"buckets": [b for b in _PROF["buckets"] if b["id"] == "matmul"]}
    assert m.observed_gathered_numels(only_matmul) == []


def test_an_observed_size_is_spent_once():
    """embed_tokens and lm_head are both 131072x3072 on this model, so the shape is ambiguous by
    construction. One gather was observed, so ONE tensor of that size is excluded and the head stays
    counted -- 4.014B - 0.403B = 3.611B, not 3.208B."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("for _n in _obs:")
    stanza = src[i : i + 320]
    assert "break" in stanza, "every matching tensor is excluded, not just one"


def test_the_name_rule_remains_as_the_pre_profile_fallback():
    """The first emitter call happens before any baseline, so a ceiling must still exist then."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("if not _lk and _LO is not None:")
    assert "_LO.search(_nm)" in src[i : i + 400]


def test_no_profile_is_not_an_error():
    m = _run_mod()
    assert m.observed_gathered_numels(None) == []
    assert m.observed_gathered_numels({}) == []
    assert m.observed_gathered_numels({"buckets": [{"id": "embedding", "top_ops": [{"shape": "junk"}]}]}) == []
