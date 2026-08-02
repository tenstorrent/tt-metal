"""A change that alters the whole model is not a row in a per-op ladder.

The op × rung matrix is one op walking its own ladder: grid/fidelity/dtype/shard change THAT op's
config, tt-lang/cpp replace THAT op's kernel. Some levers are not shaped like that. The DRAM
prefetcher is built once, every layer's weights are registered with it, and `global_cb` goes to all
the decode matmuls together -- it changes six ops at once. Removing the host round-trips from the
decode loop is the same: one edit, no owning op.

With nowhere to put them, they get filed under whichever op happened to be the target. Run 20's
prefetcher attempt was recorded as `Matmul 32x3840x15360 / shard` -- true of where the agent was
standing, useless to anyone looking for whether the prefetcher had been tried. Attaching it to all
six matmuls instead would double-count one change six times.

So they get their own block, between Block-level timing and the matrix, keyed by a `model:` prefix on
the op_signature. The matrix stays honest -- every row is one op's ladder -- and a model-wide lever is
recorded once, cleared once, and reported with its own end-to-end delta.

host_overhead already lives in the matrix as a pseudo-op with `trace` as its rung. That is the shape
this generalises: it stays where it is, and genuinely model-scoped levers move out.
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.cc_optimize import summary as sm  # noqa: E402


def _attempt(sig, kind, ms=380.0, **kw):
    a = {"op_signature": sig, "kernel_kind": kind, "measured_ms": ms, "note": "n"}
    a.update(kw)
    return a


def _render(attempts, tmp_path, **kw):
    klog = tmp_path / "kl.json"
    klog.write_text(json.dumps(attempts))
    kwargs = dict(model="gemma3", task="main", finalized=True, metric="device_ms")
    kwargs.update(kw)
    return sm.render_summary(str(klog), **kwargs)


# ---------------------------------------------------------------- the block exists


def test_a_model_level_lever_gets_its_own_section(tmp_path):
    text = _render([_attempt("model:prefetch", "prefetch", 368.42)], tmp_path)
    assert "Model-level levers" in text, text[:400]


def test_it_is_not_a_row_in_the_per_op_matrix(tmp_path):
    """The matrix is one ladder per op. A model-wide lever in it claims an ownership it does not
    have -- which is how run 20's prefetcher ended up looking like a property of one matmul."""
    text = _render(
        [_attempt("model:prefetch", "prefetch"), _attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid")],
        tmp_path,
    )
    matrix = text.split("Model-level levers", 1)[1]
    matrix = matrix[matrix.index("\nop ") :] if "\nop " in matrix else ""
    assert "model:prefetch" not in matrix, matrix[:300]


def test_the_section_sits_between_block_timing_and_the_matrix(tmp_path):
    """Where time goes, then model-wide levers, then the per-op ladder."""
    text = _render([_attempt("model:prefetch", "prefetch"), _attempt("LayerNormDeviceOperation", "grid")], tmp_path)
    assert "Model-level levers" in text and "\nop " in text
    assert text.index("Model-level levers") < text.index("\nop ")


def test_the_lever_name_is_shown_without_the_prefix(tmp_path):
    """`model:` is a routing key, not something a reader should have to parse."""
    text = _render([_attempt("model:prefetch", "prefetch")], tmp_path)
    block = text.split("Model-level levers", 1)[1].split("\nop ", 1)[0]
    assert "prefetch" in block and "model:prefetch" not in block, block


# ---------------------------------------------------------------- what it reports


def test_the_measurement_and_verdict_are_shown(tmp_path):
    text = _render(
        [_attempt("model:prefetch", "prefetch", 368.42, fullpipe_ms=33.4, fullpipe_delta_ms=-1.84)], tmp_path
    )
    block = text.split("Model-level levers", 1)[1].split("\nop ", 1)[0]
    assert "-1.84" in block, block


def test_an_untried_run_has_no_block(tmp_path):
    """No model-level attempts -> no empty heading."""
    text = _render([_attempt("LayerNormDeviceOperation", "grid")], tmp_path)
    assert "Model-level levers" not in text


def test_several_levers_are_listed_together(tmp_path):
    text = _render([_attempt("model:prefetch", "prefetch"), _attempt("model:host-loop", "host")], tmp_path)
    block = text.split("Model-level levers", 1)[1].split("\nop ", 1)[0]
    assert "prefetch" in block and "host-loop" in block, block


# ---------------------------------------------------------------- the rest is untouched


def test_host_overhead_stays_in_the_matrix(tmp_path):
    """It is a pseudo-op the ranker already selects, with `trace` as its rung. Not model-scoped --
    moving it would be a second change dressed as this one."""
    text = _render([_attempt("host_overhead", "trace")], tmp_path)
    assert "host_overhead" in text
    assert "Model-level levers" not in text


def test_a_normal_op_still_renders_in_the_matrix(tmp_path):
    text = _render([_attempt("MatmulDeviceOperation 32 x 3840 x 15360", "grid")], tmp_path)
    assert "Matmul 32x3840x15360" in text or "MatmulDeviceOperation" in text
