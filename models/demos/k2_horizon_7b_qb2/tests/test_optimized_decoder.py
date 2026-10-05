"""Real-weight acceptance invokes the delivered optimized runtime directly."""

import ast
from argparse import Namespace
from pathlib import Path

import pytest

from .run_optimized import run


def test_optimized_implementation_is_standalone():
    path = Path(__file__).resolve().parents[1] / "tt/optimized_decoder.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OptimizedDecoder")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {"prefill_forward", "decode_forward", "_finish", "_norm"} <= methods
    assert not any(isinstance(n, ast.Name) and n.id in {"FusedDecoder", "FunctionalDecoder"} for n in ast.walk(tree))


@pytest.mark.parametrize("batch,split", [(1, None), (2, 31), (32, None)])
def test_optimized_dense_decoder(tmp_path, batch, split):
    run(
        Namespace(
            lengths=[257],
            batch=batch,
            decode_steps=3,
            synthetic=False,
            profile=False,
            split=split,
            remap=True,
            audit=True,
            unchunked_control=batch == 1,
            candidate=None,
            baseline_fused=False,
            output=str(tmp_path / f"dense_{batch}.json"),
        )
    )
