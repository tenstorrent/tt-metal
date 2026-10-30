"""Discovery's env diagnosis must classify the ROOT exception, not the wrapper.

Regression pins for the FLUX.2 misdiagnosis: diffusers wraps a real
ModuleNotFoundError in ``RuntimeError: Failed to import X because ...`` whose text
ends in ``'flash_attn'``. Classifying that OUTER text matched a hardcoded
"needs transformers < 5" rule, so the tool confidently told the user to downgrade
transformers -- which was wrong, would have broken the shared env, and hid the
actual cause (diffusers too old to contain the module).
"""

from __future__ import annotations

from scripts.tt_hw_planner.module_tree import _reference_loader_next_steps


def _wrapped(inner: BaseException, outer_msg: str) -> BaseException:
    """Build a real chained exception (``raise ... from inner``)."""
    try:
        try:
            raise inner
        except BaseException as e:
            raise RuntimeError(outer_msg) from e
    except BaseException as exc:
        return exc


FLUX_TARGET = "diffusers.models.transformers.transformer_flux2"

FLUX_OUTER = (
    "Failed to import diffusers.models.transformers.transformer_flux2 because of "
    "the following error (look up to see its traceback):\n"
    "Failed to import diffusers.loaders.single_file_model because of the "
    "following error (look up to see its traceback):\n'flash_attn'"
)


def test_wrapped_missing_submodule_reports_upgrade_not_downgrade(monkeypatch) -> None:
    """The exact FLUX failure: installed package, missing submodule.

    THE ABSENCE IS STUBBED, not borrowed from the interpreter. This asked the real environment
    whether diffusers provides transformer_flux2, so it only tested anything on a machine whose
    diffusers predates that module -- and stopped testing anything the moment 0.38.0 landed here,
    where the submodule EXISTS and the code correctly declines to call it missing. The scenario is
    "installed package, missing submodule"; that has to be stated, not hoped for.
    """
    import importlib.util as _ilu

    _real = _ilu.find_spec
    monkeypatch.setattr(_ilu, "find_spec", lambda n, *a, **k: None if n == FLUX_TARGET else _real(n, *a, **k))
    out = _reference_loader_next_steps(
        "black-forest-labs/FLUX.2-klein-9B", _wrapped(KeyError("flash_attn"), FLUX_OUTER), None
    )
    assert "does not provide" in out
    assert "diffusers.models.transformers.transformer_flux2" in out
    # The invariant is "get a NEWER diffusers" -- the exact form varies with what
    # is discoverable: a floor from the model's own config ("diffusers>=0.37.0"),
    # a pin to newest on PyPI, or a bare "--upgrade diffusers" when offline.
    fix = out.split("SUGGESTED FIX")[1]
    assert "diffusers" in fix
    assert ("--upgrade" in fix) or (">=" in fix) or ("==" in fix)
    # the specific wrong advice that shipped before
    assert "transformers<5" not in out
    assert "needs transformers < 5" not in out


def test_root_cause_is_surfaced_in_the_cause_line() -> None:
    out = _reference_loader_next_steps("x/y", _wrapped(KeyError("flash_attn"), FLUX_OUTER), None)
    assert "root cause:" in out
    assert "KeyError" in out


def test_genuinely_missing_package_still_says_install() -> None:
    exc = ModuleNotFoundError("No module named 'totally_absent_pkg_xyz'", name="totally_absent_pkg_xyz")
    out = _reference_loader_next_steps("x/y", exc, None)
    assert "not installed" in out
    assert "pip install totally_absent_pkg_xyz" in out
    assert "--upgrade" not in out


def test_deeply_chained_exception_still_finds_the_root() -> None:
    """Two wrapper layers must not defeat the unwrap."""
    inner = _wrapped(ModuleNotFoundError("No module named 'absent_pkg_qq'", name="absent_pkg_qq"), "layer one failed")
    outer = _wrapped(inner, "layer two failed")
    out = _reference_loader_next_steps("x/y", outer, None)
    assert "absent_pkg_qq" in out


def test_never_promises_the_fix_will_work() -> None:
    """A heuristic must not be phrased as a guarantee."""
    for exc in (
        _wrapped(KeyError("flash_attn"), FLUX_OUTER),
        ModuleNotFoundError("No module named 'absent_pkg_zz'", name="absent_pkg_zz"),
        RuntimeError("something else entirely"),
    ):
        out = _reference_loader_next_steps("x/y", exc, None)
        assert "and it will work" not in out
        assert "SUGGESTED FIX" in out


def test_no_package_specific_rules_in_the_logic() -> None:
    """The diagnosis must not name any package it was not told about.

    The original bug was a hardcoded rule ("if the text mentions flash_attn, the
    answer is transformers<5"). Nothing in the output may name a package that did
    not come from the exception or the traceback."""
    out = _reference_loader_next_steps("x/y", RuntimeError("'flash_attn'"), None)
    # Echoing the error text is fine; inventing a package is not. Nothing named
    # 'transformers' appears anywhere in the exception, so it must not appear in
    # the diagnosis or the suggested command.
    assert "transformers" not in out


def test_error_raised_inside_an_installed_package_blames_that_package() -> None:
    """Derived from the traceback, so it works for any dependency."""
    import json as _json

    try:
        _json.loads("{definitely not json")
    except BaseException as exc:
        out = _reference_loader_next_steps("x/y", exc, None)
    # json is stdlib, not site-packages -> no blame, generic fallback
    assert "could not be built" in out or "raised inside installed package" in out
