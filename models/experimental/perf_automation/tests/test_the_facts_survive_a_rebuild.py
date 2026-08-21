"""A rebuilt facts file must not silently become a different model.

Run 13, 2026-08-21: a git_revert after a no-gain attempt deleted perf_target_inputs.json (untracked,
in the directory the loop reverts). perf_mcp rebuilt it -- correctly, that rebuild exists for exactly
this -- but the rebuild produces only what _perf_target_inputs can derive, and `blocks` needs a
resolvable model id while `stage_roots` is merged in from discovery by a different path entirely. So
the multi-tower shape became a flat one: layers 32 from the audio tower beside hidden_size 3072 from
the language model. Every stage then fell back to that geometry and to total_params, pricing the
audio encoder with the language model's 3.611B instead of its own 0.637B -- 5.7x the real work, and
a report showing encode at 321% of a 702 TFLOPS peak.

The no-downgrade guard could not catch it: it checks weight_bytes/total_params/active_params, which
the rebuild DOES produce, so nothing looked lost."""
import ast
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _fn(path, name):
    src = Path(path).read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "\n".join(src.splitlines()[n.lineno - 1 : n.end_lineno])
    raise AssertionError("%s not found" % name)


def test_a_rebuild_carries_the_structural_facts_forward():
    body = _fn(_PA / "cc_optimize" / "run.py", "_emit_perf_target_inputs")
    assert 'for _k in ("blocks", "stage_roots")' in body, "a rebuild still drops the multi-tower facts"
    assert "facts[_k] = _prev[_k]" in body


def test_it_carries_forward_rather_than_refusing_the_write():
    """Refusing would block a legitimate geometry refresh over a key this producer never emits."""
    body = _fn(_PA / "cc_optimize" / "run.py", "_emit_perf_target_inputs")
    i = body.index('for _k in ("blocks", "stage_roots")')
    assert "return" not in body[i : i + 200], "the carry-forward turned into a refusal"


def test_the_divisor_guard_is_untouched():
    """It was written for the gemma-3 incident and still has its own job."""
    body = _fn(_PA / "cc_optimize" / "run.py", "_emit_perf_target_inputs")
    assert '"weight_bytes", "total_params", "active_params"' in body


def test_the_rebuild_passes_a_model_id():
    """blocks needs the HF config behind the id; this path passed None, so it could never rebuild
    them -- only preserve what was already there."""
    body = _fn(_PA / "cc_optimize" / "perf_mcp.py", "_load_perf_target_inputs")
    assert "_emit_perf_target_inputs(_MODEL_ROOT, _MODEL_ROOT, None" not in body, "still passing None"
    assert "_model_id_for_facts(_MODEL_ROOT)" in body


def test_a_single_tower_model_is_unaffected():
    """It writes the flat shape legitimately -- which is why this hid for four days. Carrying
    forward only fires when the OLD file had a key the new facts lack."""
    body = _fn(_PA / "cc_optimize" / "run.py", "_emit_perf_target_inputs")
    i = body.index('for _k in ("blocks", "stage_roots")')
    assert "if _prev.get(_k) and not facts.get(_k)" in body[i : i + 200]
