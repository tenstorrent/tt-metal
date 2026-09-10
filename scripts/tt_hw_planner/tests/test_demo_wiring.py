"""Unit tests for `scripts.tt_hw_planner.demo_wiring`.

Pins the multi-component wiring behavior: every graduated component on
the maximal antichain ends up in the demo's WIRED_COMPONENTS table, and
nested graduated components are subsumed by their graduated parent.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.tt_hw_planner.demo_wiring import (  # noqa: E402
    GraduatedComponent,
    _is_ancestor_or_equal,
    build_wiring_specs,
    collect_graduated_components,
    format_wiring_literal,
    select_maximal_antichain,
)


# ----------------------------------------------------------------------------
# _is_ancestor_or_equal
# ----------------------------------------------------------------------------


def test_is_ancestor_self_match() -> None:
    assert _is_ancestor_or_equal("vision_encoder", "vision_encoder") is True


def test_is_ancestor_strict_parent() -> None:
    assert _is_ancestor_or_equal("vision_encoder", "vision_encoder.neck") is True


def test_is_ancestor_indexed_child() -> None:
    assert _is_ancestor_or_equal("encoder.layers", "encoder.layers[0]") is True
    assert _is_ancestor_or_equal("encoder.layers[0]", "encoder.layers[0].mlp") is True


def test_is_ancestor_not_a_word_prefix() -> None:
    # `vision_encoder` should NOT be considered ancestor of `vision_encoder_2.neck`
    assert _is_ancestor_or_equal("vision_encoder", "vision_encoder_2.neck") is False


def test_is_ancestor_child_is_not_ancestor_of_parent() -> None:
    assert _is_ancestor_or_equal("vision_encoder.neck", "vision_encoder") is False


def test_is_ancestor_disjoint_paths() -> None:
    assert _is_ancestor_or_equal("vision_encoder", "mask_decoder") is False
    assert _is_ancestor_or_equal("a.b.c", "a.b.d") is False


def test_is_ancestor_empty_strings() -> None:
    assert _is_ancestor_or_equal("", "vision_encoder") is False
    assert _is_ancestor_or_equal("vision_encoder", "") is False
    assert _is_ancestor_or_equal("", "") is False


# ----------------------------------------------------------------------------
# select_maximal_antichain
# ----------------------------------------------------------------------------


def _mk(name: str, path: str, ops: int = 0) -> GraduatedComponent:
    return GraduatedComponent(name=name, safe=name, submodule_path=path, op_count=ops)


def test_antichain_disjoint_components_all_selected() -> None:
    """Three sibling components — none subsumes another — all wired."""
    comps = [
        _mk("vision_config", "vision_encoder"),
        _mk("mask_decoder_config", "mask_decoder"),
        _mk("prompt_encoder_config", "prompt_encoder"),
    ]
    out = select_maximal_antichain(comps)
    assert sorted(c.name for c in out) == sorted(["vision_config", "mask_decoder_config", "prompt_encoder_config"])


def test_antichain_parent_subsumes_child() -> None:
    """When both vision_encoder AND vision_encoder.neck graduate, the
    parent wins. The child stays validated by its isolated PCC test but
    isn't wired separately."""
    comps = [
        _mk("vision_config", "vision_encoder"),
        _mk("vision_neck", "vision_encoder.neck"),
    ]
    out = select_maximal_antichain(comps)
    assert len(out) == 1
    assert out[0].name == "vision_config"


def test_antichain_deeply_nested_chain_picks_outermost() -> None:
    comps = [
        _mk("a", "encoder"),
        _mk("b", "encoder.layers"),
        _mk("c", "encoder.layers[0]"),
        _mk("d", "encoder.layers[0].mlp"),
    ]
    out = select_maximal_antichain(comps)
    assert len(out) == 1
    assert out[0].name == "a"


def test_antichain_multiple_roots_with_nested_children() -> None:
    comps = [
        _mk("ve", "vision_encoder"),
        _mk("ve_neck", "vision_encoder.neck"),
        _mk("md", "mask_decoder"),
        _mk("md_head", "mask_decoder.head"),
    ]
    out = select_maximal_antichain(comps)
    assert sorted(c.name for c in out) == ["md", "ve"]


def test_antichain_same_path_dedup_tiebreaker_alphabetical() -> None:
    """When two components have the exact same path, only ONE survives,
    chosen alphabetically by name (deterministic)."""
    comps = [
        _mk("vision_model", "vision_encoder"),
        _mk("vision_config", "vision_encoder"),
    ]
    out = select_maximal_antichain(comps)
    assert len(out) == 1
    assert out[0].name == "vision_config"  # alphabetically first


def test_antichain_empty_input() -> None:
    assert select_maximal_antichain([]) == []


def test_antichain_is_deterministic_across_input_orderings() -> None:
    """The selection result must depend only on the SET of inputs, not
    the order they're passed in."""
    forward = [
        _mk("a", "encoder"),
        _mk("b", "encoder.layers[0].mlp"),
        _mk("c", "mask_decoder"),
    ]
    reverse = list(reversed(forward))
    sel_fwd = [c.name for c in select_maximal_antichain(forward)]
    sel_rev = [c.name for c in select_maximal_antichain(reverse)]
    assert sel_fwd == sel_rev


# ----------------------------------------------------------------------------
# format_wiring_literal
# ----------------------------------------------------------------------------


def test_format_wiring_literal_empty_returns_empty_list() -> None:
    assert format_wiring_literal([]) == "[]"


def test_format_wiring_literal_emits_tuples() -> None:
    specs = [
        {
            "path": "vision_encoder",
            "import": "models.demos.sam2.demo._stubs.vision_config",
            "name": "vision_config",
            "safe": "vision_config",
        },
        {
            "path": "mask_decoder",
            "import": "models.demos.sam2.demo._stubs.decoder_head",
            "name": "decoder_head",
            "safe": "decoder_head",
        },
    ]
    lit = format_wiring_literal(specs)
    # Must be valid Python that evaluates to a list of 3-tuples
    parsed = eval(lit, {"__builtins__": {}})  # noqa: S307 — controlled input
    assert parsed == [
        ("vision_encoder", "models.demos.sam2.demo._stubs.vision_config", "vision_config"),
        ("mask_decoder", "models.demos.sam2.demo._stubs.decoder_head", "decoder_head"),
    ]


# ----------------------------------------------------------------------------
# collect_graduated_components — end-to-end on a fake demo dir
# ----------------------------------------------------------------------------


def _make_fake_demo(tmp_path: Path, comps: List[Dict[str, object]], graduated: Dict[str, str]) -> Path:
    """Build a fake demo dir with the on-disk artifacts the collector reads.

    ``graduated`` maps safe_id -> submodule_path. Each entry gets a
    graduated stub (no `_get_torch_submodule` reference), args.pt /
    kwargs.pt / manifest.json. Components in ``comps`` but NOT in
    ``graduated`` get an autofill stub (collector should skip them).
    """
    demo_dir = tmp_path / "fake_demo"
    demo_dir.mkdir()
    (demo_dir / "_stubs").mkdir()
    (demo_dir / "_captured").mkdir()
    (demo_dir / "bringup_status.json").write_text(json.dumps({"components": comps}))
    for comp in comps:
        safe = comp["name"]
        stub = demo_dir / "_stubs" / f"{safe}.py"
        if safe in graduated:
            # Graduated: native ttnn (no torch fallback markers).
            stub_body = (
                "import ttnn\n\ndef build(device, m):\n    return _Port(device, m)\n\n"
                "class _Port:\n    def __init__(self, d, m): pass\n"
                "    def __call__(self, *a, **k): return None\n"
            )
            stub.write_text(stub_body)
            # Simulate _snapshot_native_stub having run (component passed PCC).
            # _stub_has_graduated_from_autofill requires this snapshot as
            # positive graduation evidence.
            stub.with_suffix(".py.last_good_native").write_text(stub_body)
            cap = demo_dir / "_captured" / safe
            cap.mkdir()
            (cap / "args.pt").write_text("fake")
            (cap / "kwargs.pt").write_text("fake")
            (cap / "manifest.json").write_text(json.dumps({"submodule_path": graduated[safe]}))
        else:
            # Autofill / not graduated.
            stub.write_text("def _get_torch_submodule(): pass\n")
    return demo_dir


def test_collect_skips_non_NEW_status(tmp_path: Path) -> None:
    comps = [
        {"name": "reused_comp", "status": "REUSE"},
        {"name": "adapted_comp", "status": "ADAPT"},
    ]
    demo_dir = _make_fake_demo(tmp_path, comps, graduated={})
    assert collect_graduated_components(demo_dir, comps) == []


def test_collect_skips_components_missing_capture_artifacts(tmp_path: Path) -> None:
    comps = [{"name": "vision_config", "status": "NEW"}]
    demo_dir = _make_fake_demo(tmp_path, comps, graduated={"vision_config": "vision_encoder"})
    # Remove capture artifacts to simulate a NEW + graduated stub without
    # captured inputs (e.g. capture-inputs hasn't run).
    (demo_dir / "_captured" / "vision_config" / "args.pt").unlink()
    out = collect_graduated_components(demo_dir, comps)
    assert out == []


def test_collect_skips_non_graduated_stubs(tmp_path: Path) -> None:
    comps = [{"name": "vision_config", "status": "NEW"}]
    demo_dir = _make_fake_demo(tmp_path, comps, graduated={})
    out = collect_graduated_components(demo_dir, comps)
    assert out == []


def test_collect_skips_components_with_empty_submodule_path(tmp_path: Path) -> None:
    comps = [{"name": "vision_config", "status": "NEW"}]
    demo_dir = _make_fake_demo(tmp_path, comps, graduated={"vision_config": ""})
    out = collect_graduated_components(demo_dir, comps)
    assert out == []


def test_collect_returns_graduated_with_paths(tmp_path: Path) -> None:
    comps = [
        {"name": "vision_config", "status": "NEW"},
        {"name": "mask_decoder_config", "status": "NEW"},
    ]
    demo_dir = _make_fake_demo(
        tmp_path,
        comps,
        graduated={"vision_config": "vision_encoder", "mask_decoder_config": "mask_decoder"},
    )
    out = collect_graduated_components(demo_dir, comps)
    by_name = {c.name: c for c in out}
    assert set(by_name) == {"vision_config", "mask_decoder_config"}
    assert by_name["vision_config"].submodule_path == "vision_encoder"
    assert by_name["mask_decoder_config"].submodule_path == "mask_decoder"


# ----------------------------------------------------------------------------
# build_wiring_specs — end-to-end
# ----------------------------------------------------------------------------


def test_build_wiring_specs_end_to_end_antichain(tmp_path: Path) -> None:
    """When parent + child both graduate, build_wiring_specs returns
    ONLY the parent in the wiring spec."""
    comps = [
        {"name": "vision_config", "status": "NEW"},
        {"name": "vision_neck", "status": "NEW"},
        {"name": "mask_decoder_config", "status": "NEW"},
    ]
    demo_dir = _make_fake_demo(
        tmp_path,
        comps,
        graduated={
            "vision_config": "vision_encoder",
            "vision_neck": "vision_encoder.neck",
            "mask_decoder_config": "mask_decoder",
        },
    )
    specs = build_wiring_specs(demo_dir=demo_dir, components=comps, repo_root=tmp_path)
    by_name = {s["name"]: s for s in specs}
    assert set(by_name) == {"vision_config", "mask_decoder_config"}
    assert "vision_neck" not in by_name


def test_build_wiring_specs_returns_empty_when_nothing_graduated(tmp_path: Path) -> None:
    comps = [{"name": "vision_config", "status": "NEW"}]
    demo_dir = _make_fake_demo(tmp_path, comps, graduated={})
    assert build_wiring_specs(demo_dir=demo_dir, components=comps, repo_root=tmp_path) == []


# ----------------------------------------------------------------------------
# bringup_loop integration: emit_runnable_demo now uses demo_wiring
# ----------------------------------------------------------------------------


def test_bringup_loop_imports_demo_wiring() -> None:
    """The emit_runnable_demo function must import and use demo_wiring."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src = Path(bl.__file__).read_text()
    # The function body must reference both public helpers.
    func_idx = src.find("def emit_runnable_demo(")
    assert func_idx >= 0
    body = src[func_idx : func_idx + 4000]
    assert "build_wiring_specs" in body, "emit_runnable_demo must call build_wiring_specs"
    assert "format_wiring_literal" in body, "emit_runnable_demo must call format_wiring_literal"
    assert "_MIXED_EXEC_DEMO_TEMPLATE" in body, "emit_runnable_demo must use the mixed-execution template"


# ----------------------------------------------------------------------------
# Template runtime helpers: extract _resolve / _set_submodule / _tokenize_path
# from the embedded template source and exec them in a clean namespace.
# Pins behavior for indexed paths (`encoder.layers[0]`) which a previous
# rpartition(".")-based implementation got wrong.
# ----------------------------------------------------------------------------


def _exec_template_helpers() -> Dict[str, object]:
    """Lift `_tokenize_path`, `_resolve`, `_set_submodule`, `_TTModuleShim`
    out of the _MIXED_EXEC_DEMO_TEMPLATE source and exec them in a clean
    namespace so we can unit-test their behavior without firing up ttnn.

    The template references `torch` at module-level for `_TTModuleShim`
    inheritance and isinstance checks, so we inject torch into the
    namespace before exec."""
    import torch as _torch_for_exec

    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src = Path(bl.__file__).read_text()
    start = src.find("def _tokenize_path(dotted: str):")
    assert start >= 0, "_tokenize_path must be defined in the demo template"
    end = src.find("def _build_hf_model(", start)
    assert end > start, "_build_hf_model must come after _set_submodule in the template"
    snippet = src[start:end]
    ns: Dict[str, object] = {"torch": _torch_for_exec}
    exec(snippet, ns)
    return ns


def test_template_tokenize_path_handles_indexed_paths() -> None:
    ns = _exec_template_helpers()
    tok = ns["_tokenize_path"]
    assert tok("encoder.layers[0].mlp") == ["encoder", "layers", 0, "mlp"]
    assert tok("vision_encoder") == ["vision_encoder"]
    assert tok("encoder.layers[0]") == ["encoder", "layers", 0]
    assert tok("") == []


def test_template_set_submodule_handles_indexed_paths() -> None:
    """Regression: the previous rpartition(".") implementation set a
    literal attribute named "layers[0]" instead of replacing layers[0]
    via list indexing. This test pins the fix.

    Non-Module replacements get wrapped in _TTModuleShim (Bug N fix),
    so we check the wrapped port via `._tt_port`."""
    import torch  # noqa: F401 — used implicitly by template

    ns = _exec_template_helpers()
    set_sub = ns["_set_submodule"]
    resolve = ns["_resolve"]
    shim_cls = ns["_TTModuleShim"]

    class _ListContainer:
        def __init__(self):
            self.layers = ["L0", "L1", "L2"]

    class _Model:
        def __init__(self):
            self.encoder = _ListContainer()
            self.head = "HEAD"

    m = _Model()
    set_sub(m, "encoder.layers[0]", "REPLACED")
    # After: m.encoder.layers[0] must be a shim wrapping "REPLACED"
    installed = m.encoder.layers[0]
    assert isinstance(installed, shim_cls)
    assert installed._tt_port == "REPLACED"
    assert m.encoder.layers[1] == "L1"
    # AND no literal "layers[0]" attribute should have been set
    assert not hasattr(m.encoder, "layers[0]")
    resolved = resolve(m, "encoder.layers[0]")
    assert isinstance(resolved, shim_cls)
    assert resolved._tt_port == "REPLACED"


def test_template_set_submodule_handles_dotted_paths() -> None:
    ns = _exec_template_helpers()
    set_sub = ns["_set_submodule"]
    resolve = ns["_resolve"]
    shim_cls = ns["_TTModuleShim"]

    class _Neck:
        def __init__(self):
            self.dense = "DENSE_ORIG"

    class _Encoder:
        def __init__(self):
            self.neck = _Neck()

    class _Model:
        def __init__(self):
            self.encoder = _Encoder()

    m = _Model()
    set_sub(m, "encoder.neck", "NEW_NECK")
    installed = m.encoder.neck
    assert isinstance(installed, shim_cls)
    assert installed._tt_port == "NEW_NECK"
    resolved = resolve(m, "encoder.neck")
    assert resolved._tt_port == "NEW_NECK"


def test_template_set_submodule_handles_root_attr_replacement() -> None:
    """Single-token path replaces a top-level attribute on the model.
    Non-Module replacement gets shim-wrapped."""
    ns = _exec_template_helpers()
    set_sub = ns["_set_submodule"]
    shim_cls = ns["_TTModuleShim"]

    class _Model:
        def __init__(self):
            self.encoder = "ORIG"

    m = _Model()
    set_sub(m, "encoder", "NEW")
    assert isinstance(m.encoder, shim_cls)
    assert m.encoder._tt_port == "NEW"


def test_collect_includes_all_graduated_components(tmp_path: Path, monkeypatch) -> None:
    """Pin: graduation is the contract. Every component with a graduated
    stub (native ttnn + passed PCC) lands in the demo wiring pool. No
    workload / bench / placement signal can demote a graduated stub —
    the only path off device is a verified missing TTNN kernel."""
    from scripts.tt_hw_planner import overlay_manager as om

    monkeypatch.setattr(om, "_OVERLAYS_DIR", tmp_path / "overlays")

    comps = [
        {"name": "fast_comp", "status": "NEW", "submodule_path": "fast"},
        {"name": "slow_comp", "status": "NEW", "submodule_path": "slow"},
    ]
    demo_dir = _make_fake_demo(
        tmp_path,
        comps,
        graduated={"fast_comp": "fast", "slow_comp": "slow"},
    )

    out = collect_graduated_components(demo_dir, comps, model_id="test/m")
    names = {c.name for c in out}
    assert "fast_comp" in names
    assert "slow_comp" in names


def test_template_set_submodule_wraps_non_module_ports_in_shim() -> None:
    """Bug N regression: PyTorch's `nn.Module.__setattr__` rejects non-
    Module values when the attribute is registered as a child module.
    The previous template did `setattr(parent, attr, tt_port)` directly
    and crashed with:

        TypeError: cannot assign 'DecoderHead' as child module 'mask_decoder'
        (torch.nn.Module or None expected)

    Fix: wrap non-Module values in `_TTModuleShim(nn.Module)` so
    PyTorch accepts the assignment while still routing `__call__` to
    the wrapped TT port. Caught by actually running pytest on the
    auto-emitted demo — not by 6 prior static audit passes.
    """
    import torch

    ns = _exec_template_helpers()
    set_sub = ns["_set_submodule"]
    shim_cls = ns["_TTModuleShim"]
    assert shim_cls is not None

    class _NotAModule:
        """Plain Python class — not nn.Module."""

        def __init__(self, tag):
            self.tag = tag

        def __call__(self, *args, **kwargs):
            return f"called-{self.tag}"

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()
            self.head = torch.nn.Linear(4, 1)

    m = _Model()
    # Direct setattr WITHOUT shim would raise TypeError. The fixed
    # _set_submodule transparently wraps the non-Module in a shim.
    set_sub(m, "head", _NotAModule("tt-port"))
    # After wiring, attribute lookup returns the shim
    assert isinstance(m.head, shim_cls)
    # Calling the model's submodule routes through the shim to the port
    assert m.head() == "called-tt-port"


def test_template_set_submodule_passes_module_through_unchanged() -> None:
    """When the new module IS already an nn.Module, no shim wrapping —
    install it directly. (Otherwise we'd double-wrap on idempotent
    re-installations.)"""
    import torch

    ns = _exec_template_helpers()
    set_sub = ns["_set_submodule"]
    shim_cls = ns["_TTModuleShim"]

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 1)

    real_module = torch.nn.Linear(4, 1)
    m = _Model()
    set_sub(m, "head", real_module)
    assert m.head is real_module
    assert not isinstance(m.head, shim_cls)


def test_template_shim_call_dispatches_to_wrapped_port() -> None:
    """The shim's `forward` (and thus its `__call__` via nn.Module.__call__)
    must dispatch transparently to the wrapped TT port."""
    ns = _exec_template_helpers()
    shim_cls = ns["_TTModuleShim"]

    calls = []

    class _Port:
        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "port-output"

    shim = shim_cls(_Port())
    out = shim(1, 2, key="value")
    assert out == "port-output"
    assert calls == [((1, 2), {"key": "value"})]
