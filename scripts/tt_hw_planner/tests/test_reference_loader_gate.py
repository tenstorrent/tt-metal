"""The reference-loader gate must not bank a file that defines no loader.

Every per-component PCC score for a non-transformers checkpoint is measured against whatever
`_reference_loader.py` returns, so "is this a loader?" is the question the whole gate rests on. It
used to be answered with ``"def load_reference_model" in source`` -- a substring, which the name
merely being MENTIONED satisfied. A file whose only occurrence was in a comment (the shape an agent
leaves when it writes a TODO and stops) defined nothing, yet resolved=True and bring-up moved on.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import importlib.util

import pytest

from scripts.tt_hw_planner.reference_loader_resolver import (
    _resolved,
    _validates,
    loader_path,
    uses_random_weights,
    verify,
    weight_provenance,
)

# Scoped to the tests that need it: the structural checks below must keep running on a box with no
# torch, which is exactly where a contributor is most likely to run this file.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None,
    reason="needs torch + safetensors",
)


def _write(tmp_path: Path, src: str) -> Path:
    p = loader_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return tmp_path


def test_real_loader_validates(tmp_path: Path) -> None:
    assert _validates(
        _write(
            tmp_path,
            """
            def load_reference_model(model_id: str):
                return object()
            """,
        )
    )


def test_name_only_in_a_comment_is_not_a_loader(tmp_path: Path) -> None:
    # The substring gate accepted this: nothing is defined, so the import the PCC template does
    # would fail LATER, far from the cause, with bring-up already recorded as resolved.
    assert not _validates(
        _write(
            tmp_path,
            """
            # def load_reference_model(model_id: str): -- TODO, not written yet
            PLACEHOLDER = True
            """,
        )
    )


def test_name_only_inside_a_string_is_not_a_loader(tmp_path: Path) -> None:
    assert not _validates(
        _write(
            tmp_path,
            '''
            HELP = """
            def load_reference_model(model_id): ...
            """
            ''',
        )
    )


def test_zero_arg_stub_is_not_a_loader(tmp_path: Path) -> None:
    # Callers pass the model id; a no-arg def would TypeError on first use.
    assert not _validates(
        _write(
            tmp_path,
            """
            def load_reference_model():
                return object()
            """,
        )
    )


def test_unparseable_and_missing_files_are_not_loaders(tmp_path: Path) -> None:
    assert not _validates(tmp_path)  # nothing written at all
    assert not _validates(_write(tmp_path, "def load_reference_model(  <<< syntax error\n"))


def test_random_weight_fallback_travels_with_the_result(tmp_path: Path) -> None:
    """Strategy 5 builds the reference from random weights, so PCC against it verifies STRUCTURE
    only. That used to be recorded in a module docstring, which nothing reads -- a run could be
    scored against weights unrelated to the checkpoint and the result looked identical."""
    d = _write(
        tmp_path,
        """
        REFERENCE_USES_RANDOM_WEIGHTS = True

        def load_reference_model(model_id: str):
            return object()
        """,
    )
    assert uses_random_weights(d)
    out = _resolved(d, "loader written")
    assert out["resolved"] is True and out["random_weights"] is True
    assert "RANDOM weights" in out["caveat"]


def test_real_weight_loader_carries_no_caveat(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        """
        def load_reference_model(model_id: str):
            return object()
        """,
    )
    assert not uses_random_weights(d)
    assert "random_weights" not in _resolved(d, "loader written")


# --- runtime gate: a file that parses is still not a loader that WORKS ------------------------
# Each case below is a loader that sails through the structural check and would have been banked as
# resolved, then failed somewhere downstream where the cause is no longer obvious.


@requires_torch
@pytest.mark.parametrize(
    ("body", "expect"),
    [
        ("raise RuntimeError('no weights here')", "RuntimeError"),
        ("return None", "returned None"),
        ("return {'state_dict': {}}", "not nn.Module"),
        ("import torch; return torch.nn.Module()", "no parameters"),
    ],
    ids=["raises", "returns-none", "returns-non-module", "no-parameters"],
)
def test_runtime_gate_rejects_loaders_that_cannot_produce_a_model(tmp_path: Path, body: str, expect: str) -> None:
    d = _write(tmp_path, f"def load_reference_model(model_id):\n    {body}\n")
    assert _validates(d), "precondition: the structural check accepts all of these"
    v = verify(d, "some/model")
    assert v["ok"] is False and v["status"] == "broken"
    assert expect in v["reason"], v["reason"]


@requires_torch
def test_runtime_gate_accepts_a_loader_that_returns_a_real_module(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        """
        import torch

        def load_reference_model(model_id):
            return torch.nn.Linear(8, 8)
        """,
    )
    v = verify(d, "some/model")
    assert v["ok"] is True and v["status"] == "verified", v["reason"]


# --- provenance: break the model on purpose ---------------------------------------------------


def _checkpoint(tmp_path: Path, tensors: dict) -> str:
    from safetensors.torch import save_file

    d = tmp_path / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "model.safetensors"))
    return str(d)


def _module_with(weight) -> "torch.nn.Module":
    import torch

    m = torch.nn.Module()
    m.w = torch.nn.Parameter(weight, requires_grad=False)
    return m


@requires_torch
def test_provenance_confirms_weights_that_came_from_the_checkpoint(tmp_path: Path) -> None:
    import torch

    w = torch.randn(4096)
    out = weight_provenance(_checkpoint(tmp_path, {"w": w}), _module_with(w.clone()))
    assert out["status"] == "from_checkpoint", out


@requires_torch
def test_provenance_flags_a_reference_that_never_loaded_the_weights(tmp_path: Path) -> None:
    """THE case worth catching: architecture built from config, weights left at random init.

    Such a reference loads cleanly, has the right shapes, and passes every other check -- while
    every PCC measured against it is meaningless.
    """
    import torch

    ckpt = _checkpoint(tmp_path, {"w": torch.randn(4096)})
    random_init = _module_with(torch.randn(4096) * 0.02)  # never read the checkpoint
    out = weight_provenance(ckpt, random_init)
    assert out["status"] == "no_match", out
    assert "randomly initialised" in out["reason"]


@requires_torch
def test_provenance_tolerates_a_permuted_but_correct_conversion(tmp_path: Path) -> None:
    """A correct loader may reorder weights (RoPE layouts differ); that must not read as no_match."""
    import torch

    w = torch.randn(4096)
    permuted = w[torch.randperm(w.numel())]
    out = weight_provenance(_checkpoint(tmp_path, {"w": w}), _module_with(permuted))
    assert out["status"] == "from_checkpoint", out


@requires_torch
def test_unreachable_checkpoint_is_unverified_not_a_failure(tmp_path: Path) -> None:
    """An environment that cannot check must not be reported as a bad loader."""
    import torch

    out = weight_provenance(str(tmp_path / "nothing-here"), _module_with(torch.randn(4096)))
    assert out["status"] == "unverified", out
