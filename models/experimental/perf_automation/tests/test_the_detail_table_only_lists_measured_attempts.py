"""The per-attempt detail table is a table of measurements, so an unmeasured attempt does not belong.

Its "1CQ delta vs current" column answers one question: what did this attempt do to the end-to-end
number. An attempt that ran no end-to-end of its own has no answer, and printed "n/m" -- honest, but
on gemma-3-12b-it that was 97 of 105 rows. A reader scrolling ninety-seven n/m's to reach eight real
deltas is not reading a measurement table.

They cannot be given a number. Those rows are the RUN 20 attempts restored after the 2026-08-02 host
crash, recorded before record_kernel_attempt required an end-to-end; subtracting a baseline they
never measured against is how a fake win gets manufactured. So they are dropped from the DETAIL
table, and dropping is all it is:

- the op x rung matrix above still marks each one tried, so nothing looks untried and the ladder
  cannot re-offer a rung on the strength of this table
- the kernel log is untouched -- resume still reads all 105
- a one-line footer states how many were omitted and why, because a table that silently shrinks reads
  as a run that tried less than it did

Self-limiting: the attempt gate now refuses an attempt owning no end-to-end measurement, so every row
written from here on carries a real delta and the footer goes to zero on its own.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize import summary as sm  # noqa: E402


def _measured(sig, kind, ms, delta):
    return {
        "op_signature": sig,
        "kernel_kind": kind,
        "measured_ms": ms,
        "note": "measured",
        "fullpipe_ms": 35.5,
        "fullpipe_best_ms": 35.0,
        "fullpipe_delta_ms": delta,
        "fullpipe_measured_here": True,
    }


def _unmeasured(sig, kind, ms):
    return {
        "op_signature": sig,
        "kernel_kind": kind,
        "measured_ms": ms,
        "note": "RUN 20, restored after the host crash",
        "fullpipe_measured_here": False,
    }


def _render(attempts, tmp_path):
    klog = tmp_path / "kl.json"
    klog.write_text(json.dumps(attempts))
    return sm.render_summary(str(klog), model="gemma3", task="main", finalized=True, metric="device_ms")


def _detail(text):
    return text.split("Per-attempt detail", 1)[1] if "Per-attempt detail" in text else ""


# ---------------------------------------------------------------- the n/m rows go


def test_an_unmeasured_attempt_is_not_listed(tmp_path):
    text = _render([_unmeasured("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 400.27)], tmp_path)
    assert "n/m" not in _detail(text), _detail(text)[:400]


def test_the_measured_ones_stay(tmp_path):
    text = _render([_measured("LayerNormDeviceOperation 1024 x 3840", "shard", 378.04, -0.76)], tmp_path)
    assert "-0.76 ms" in _detail(text)


def test_a_mixed_log_keeps_only_the_measured_rows(tmp_path):
    text = _render(
        [
            _unmeasured("MatmulDeviceOperation 32 x 3840 x 15360", "grid", 400.27),
            _measured("LayerNormDeviceOperation 1024 x 3840", "shard", 378.04, -0.76),
            _unmeasured("SDPAOperation", "fidelity", 399.98),
        ],
        tmp_path,
    )
    d = _detail(text)
    assert "-0.76 ms" in d and "400.27" not in d and "399.98" not in d, d[:500]


# ---------------------------------------------------------------- but they are accounted for


def test_the_omission_is_stated_with_a_count(tmp_path):
    """A table that silently shrinks reads as a run that tried less than it did."""
    text = _render([_unmeasured("SDPAOperation", "fidelity", 399.98) for _ in range(3)], tmp_path)
    assert "3" in _detail(text) and "omitted" in _detail(text).lower(), _detail(text)[:400]


def test_no_footer_when_everything_was_measured(tmp_path):
    """Self-limiting: once the attempt gate has been in force for a whole run this line disappears."""
    text = _render([_measured("LayerNormDeviceOperation 1024 x 3840", "shard", 378.04, -0.76)], tmp_path)
    assert "omitted" not in _detail(text).lower()


def test_the_matrix_still_marks_them_tried(tmp_path):
    """Dropping a row from the detail table must not make the op look untried -- the matrix above is
    what the reader (and the ladder) uses to see coverage."""
    text = _render([_unmeasured("SDPAOperation", "fidelity", 399.98)], tmp_path)
    assert "SDPA" in text.split("Per-attempt detail", 1)[0]


def test_a_win_is_never_dropped(tmp_path):
    """A win owns an end-to-end by definition, so it can never fall into the unmeasured bucket."""
    text = _render([_measured("BinaryNgDeviceOperation 128 x 15360", "cpp", 381.23, -1.05)], tmp_path)
    assert "-1.05 ms" in _detail(text) and "win" in _detail(text)


def test_a_legacy_row_with_numbers_is_kept(tmp_path):
    """Rows predating the stamp still render via the subtract fallback -- they HAVE a comparison, so
    they are measured rows and stay."""
    a = _measured("MatmulDeviceOperation 128 x 3840 x 15360", "grid", 400.41, None)
    del a["fullpipe_delta_ms"]
    del a["fullpipe_measured_here"]
    text = _render([a], tmp_path)
    assert "+0.50 ms" in _detail(text), _detail(text)[:300]
