"""The gate that says "FULL depth" was timing two layers.

An uncapped TRACY capture overflows the 12000-marker buffer, so the profiling runs are capped and
must stay capped. _run_full_pipeline_ms is not a profiling run -- it pops TT_METAL_DEVICE_PROFILER
and drives trace_replay, a stopwatch that prints three numbers. The cap has no purpose there.

It inherited one anyway: `env = dict(os.environ)`, and the MCP config hands this process
TT_PERF_STACK<i>_LAYERS=2 as LOOSE variables beside the PERF_MCP_PROFILE_ENV json that is the
intended channel. The generated perf test reads exactly those names and passes them to
build_pipeline.

Measured on Voxtral 2026-08-21: the gate reported 2.47 ms/token where a full-depth capture recorded
55.68 tok/s (17.96 ms) -- so the roofline compared a 2-layer time against a 62-layer ceiling and
printed decode at 539% of peak, and every win was banked against a 7x-optimistic proxy."""
import ast
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _fn(name):
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "\n".join(src.splitlines()[n.lineno - 1 : n.end_lineno])
    raise AssertionError("%s not found" % name)


def test_the_gate_drops_the_depth_cap():
    body = _fn("_run_full_pipeline_ms")
    assert "_depth_vars" in body, "the gate still inherits the profiling depth cap"
    assert "env.pop(k, None)" in body or "env.pop(_k" in body


def test_it_still_disables_tracy():
    """The two belong together: the cap is only droppable BECAUSE the profiler is off here."""
    body = _fn("_run_full_pipeline_ms")
    assert 'env.pop("TT_METAL_DEVICE_PROFILER", None)' in body


def test_the_names_are_derived_not_hardcoded():
    """layer_depth owns the spelling and the model owns which stacks exist -- a model with other
    stack names must be covered without editing a list here."""
    body = _fn("_run_full_pipeline_ms")
    assert "stage_layers_var" in body and "stack_layers_var" in body
    assert "_declared_stack_count" in body, "the stack count is picked here instead of read"
    assert "range(8)" not in body, "a hardcoded stack count is back"


def test_it_says_so_out_loud():
    """A silent change of what is being measured is how this went unnoticed for a whole run."""
    body = _fn("_run_full_pipeline_ms")
    assert "measuring at FULL depth" in body


def test_the_profiling_path_keeps_its_cap():
    """Tracy MUST stay capped -- an uncapped capture overflows the marker buffer. The fix must not
    touch it."""
    probes = (_PA / "agent" / "probes.py").read_text()
    assert 'env["TT_METAL_DEVICE_PROFILER"] = "1"' in probes
    i = probes.index('env["TT_METAL_DEVICE_PROFILER"] = "1"')
    assert "_depth_vars" not in probes[max(0, i - 2000) : i + 2000], "the tracy path lost its cap"
