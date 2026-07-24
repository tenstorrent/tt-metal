# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# CI pytest wrapper for the base HunyuanImage-3.0 text-to-image demo (demo/demo.py).
#
# demo.py is an env/argv-driven ``__main__`` script: its module-level globals
# (PROMPT, HY_STEPS, HY_NUM_LAYERS, ...) and ``ensure_base_weights()`` bake in the
# HY_* environment at *import* time. So rather than import it, this wrapper drives
# it as a subprocess with the CI configuration and asserts the whole prompt ->
# recaption(off) -> denoise -> VAE -> PNG chain runs end-to-end on the 2x2 mesh.
#
# Fast/real knobs come from the HY_* env (see demo.py header); the yaml CI entry
# sets HY_STEPS / HY_NUM_LAYERS / HY_GUIDANCE. Defaults here keep a bare
# ``pytest test_demo.py`` cheap (8 denoise steps) while still exercising the full
# 32-layer backbone.

import os
import subprocess
import sys
from pathlib import Path

import pytest

_DEMO = Path(__file__).resolve().parent / "demo.py"
_ROOT = Path(__file__).resolve().parents[4]  # tt-metal repo root


@pytest.mark.parametrize("prompt", ["a photo of a cat, studio lighting"], ids=["cat"])
def test_t2i_demo(prompt, tmp_path):
    """Prompt-only base T2I demo end-to-end on the 2x2 mesh -> PNG.

    Verifies the demo exits 0 and writes a non-empty output image. Denoise steps /
    backbone layers are overridable via HY_STEPS / HY_NUM_LAYERS (fast CI defaults
    applied when the env does not already set them)."""
    out_png = tmp_path / "hy_t2i_ci.png"
    env = {
        **os.environ,
        "HY_STEPS": os.environ.get("HY_STEPS", "8"),
        "HY_NUM_LAYERS": os.environ.get("HY_NUM_LAYERS", "32"),
        "HY_GUIDANCE": os.environ.get("HY_GUIDANCE", "5.0"),
        "HY_OUT": str(out_png),
    }
    result = subprocess.run([sys.executable, str(_DEMO), prompt], cwd=str(_ROOT), env=env)
    assert result.returncode == 0, f"demo.py exited with {result.returncode}"
    assert out_png.is_file() and out_png.stat().st_size > 0, f"no output image written to {out_png}"
