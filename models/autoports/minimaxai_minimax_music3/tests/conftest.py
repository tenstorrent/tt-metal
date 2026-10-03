import os
from pathlib import Path

import pytest


def _snapshot():
    p = os.environ.get("MUSIC3_SNAPSHOT")
    if p and Path(p).is_dir():
        return Path(p)
    try:
        from models.autoports.minimaxai_minimax_music3.config import resolve_snapshot

        return resolve_snapshot(allow_download=False)
    except Exception:
        return None


@pytest.fixture(scope="session")
def snapshot():
    s = _snapshot()
    if s is None:
        pytest.skip("MiniMax-Music3 snapshot not in the HF cache (set MUSIC3_SNAPSHOT)")
    return s


@pytest.fixture(scope="session")
def golden_root():
    p = Path(os.environ.get("MUSIC3_GOLDEN_ROOT", Path.home() / "music3-bringup" / "artifacts" / "golden"))
    if not p.is_dir():
        pytest.skip(f"golden root {p} missing (stage 01)")
    return p


@pytest.fixture(scope="session")
def diffusers_mm3():
    """The diffusers MiniMax-Music3 integration (reference venv only)."""
    try:
        from diffusers.modular_pipelines.minimax_music3 import encoders  # noqa

        return encoders
    except Exception:
        pytest.skip("diffusers minimax_music3 not importable in this venv")
