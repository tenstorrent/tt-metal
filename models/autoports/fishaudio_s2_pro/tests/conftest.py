import os
from pathlib import Path

import pytest


def _snapshot():
    p = os.environ.get("FISH_S2_SNAPSHOT")
    if p and Path(p).is_dir():
        return Path(p)
    try:
        from models.autoports.fishaudio_s2_pro.tt.weights import resolve_snapshot

        return resolve_snapshot(allow_download=False)
    except Exception:
        return None


@pytest.fixture(scope="session")
def snapshot():
    s = _snapshot()
    if s is None:
        pytest.skip("fishaudio/s2-pro snapshot not in the HF cache (set FISH_S2_SNAPSHOT)")
    return s


@pytest.fixture(scope="session")
def upstream():
    """The upstream fish-speech package (only in the reference venv)."""
    try:
        import fish_speech  # noqa
        from fish_speech.conversation import Conversation  # noqa

        return fish_speech
    except Exception:
        pytest.skip("upstream fish_speech not importable in this venv")


@pytest.fixture(scope="session")
def golden_root():
    p = Path(os.environ.get("FISH_S2_GOLDEN_ROOT", Path.home() / "s2pro-bringup" / "artifacts" / "golden"))
    if not p.is_dir():
        pytest.skip(f"golden root {p} missing (stage 01)")
    return p
