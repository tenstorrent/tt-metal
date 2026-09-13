"""`total_params` holds a READ SET, and nothing said so.

model_bytes counts params for the observed unit and deliberately drops two families: lookup-only
tensors (a token reads one row of an embedding table) and tower-only tensors (an encoder runs per
clip, not per token). Both exclusions are correct and documented in model_bytes -- but the result is
written under the name `total_params`, so on Voxtral-Mini-3B-2507 the file reads

    total_params   3,611,483,136        <- 3.611B
    blocks: audio_tower 0.637B, language_model 4.014B

a total smaller than its own parts. It is right (4.014B - 0.403B embedding), and the only way to
establish that was to rediscover both exclusions in another module."""
import ast
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _fn_src(path, name):
    tree = ast.parse(Path(path).read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "\n".join(Path(path).read_text().splitlines()[n.lineno - 1 : n.end_lineno])
    raise AssertionError("%s not found in %s" % (name, path))


def test_the_facts_state_which_count_they_carry():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index('facts["total_params"] = int(total_params)')
    stanza = src[i : i + 2200]
    assert 'facts["params_basis"]' in stanza, "the file still gives a read set the name of a total"
    assert "read set for unit=" in stanza


def test_the_basis_distinguishes_measured_from_name_derived():
    """A count parsed out of the model NAME is a different kind of claim from a header walk, and a
    reader checking a ceiling needs to know which it got."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index('facts["params_basis"]')
    stanza = src[i : i + 500]
    assert "if analytic_params" in stanza
    assert "model name" in stanza


def test_unit_cannot_be_unbound_when_the_basis_is_built():
    """_unit is assigned inside a try that can raise before reaching it; params_basis reads it."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    guard = src.index('_unit = ""')
    use = src.index('% (_unit or "unknown")')
    assert guard < use, "params_basis can read an unbound _unit"


def test_nothing_was_renamed():
    """total_params is read by perf_target.ceiling_params, simple_active_bytes, two places in
    summary, and by every perf_target_inputs.json already on disk. Stating the basis must not become
    a compatibility break."""
    for rel, needle in (
        (("agent", "perf_target.py"), 'mf.get("total_params", 0)'),
        (("cc_optimize", "run.py"), 'facts["total_params"]'),
    ):
        assert needle in (_PA.joinpath(*rel)).read_text(), "%s stopped using total_params" % (rel,)
