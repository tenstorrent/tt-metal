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

from scripts.tt_hw_planner.reference_loader_resolver import (
    _resolved,
    _validates,
    loader_path,
    uses_random_weights,
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
