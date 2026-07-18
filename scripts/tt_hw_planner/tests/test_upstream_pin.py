"""transformers pin tracks upstream tt-metal via registry_sync.

auto-sync already fetches current main; now it also fetches
tt_metal/python_env/requirements-dev.txt and parses the pinned transformers
requirement, so the tool's transformers version follows upstream instead of a
hardcoded constant. Parsing is offline-testable against a synthetic req file.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _rs():
    return importlib.import_module("scripts.tt_hw_planner.registry_sync")


def _tree(tmp: Path, body: str) -> Path:
    req = tmp / "tt_metal" / "python_env" / "requirements-dev.txt"
    req.parent.mkdir(parents=True)
    req.write_text(body)
    return tmp


def test_parses_pinned_transformers(tmp_path):
    rs = _rs()
    root = _tree(
        tmp_path,
        "numpy==1.26\n# For TT-Transformers Qwen3 support\ntransformers == 5.10.2\ntorch==2.4\n",
    )
    assert rs.upstream_transformers_pin(root) == "transformers==5.10.2"


def test_parses_no_space_variant(tmp_path):
    rs = _rs()
    root = _tree(tmp_path, "transformers==5.11.0\n")
    assert rs.upstream_transformers_pin(root) == "transformers==5.11.0"


def test_none_when_absent_or_unpinned(tmp_path):
    rs = _rs()
    assert rs.upstream_transformers_pin(_tree(tmp_path, "numpy==1.26\ntorch==2.4\n")) is None
    root2 = tmp_path / "b"
    (root2 / "tt_metal" / "python_env").mkdir(parents=True)
    (root2 / "tt_metal" / "python_env" / "requirements-dev.txt").write_text("transformers>=5.0\n")
    assert rs.upstream_transformers_pin(root2) is None


def test_missing_file_returns_none(tmp_path):
    rs = _rs()
    assert rs.upstream_transformers_pin(tmp_path) is None
