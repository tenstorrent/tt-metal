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

from models.experimental.hunyuan_image_3_0.ref.weights import ENV_BASE, HF_REPO_BASE, validate_env_checkpoint_dir

_DEMO = Path(__file__).resolve().parent / "demo.py"
_ROOT = Path(__file__).resolve().parents[4]  # tt-metal repo root


@pytest.fixture(scope="session", autouse=True)
def require_staged_checkpoint():
    if os.environ.get(ENV_BASE):
        validate_env_checkpoint_dir(ENV_BASE, HF_REPO_BASE)


@pytest.mark.parametrize("prompt", ["a photo of a cat, studio lighting"], ids=["cat"])
def test_t2i_demo(prompt, tmp_path):
    """Prompt-only base T2I demo end-to-end on the 2x2 mesh -> PNG.

    Verifies the demo exits 0 and writes a non-empty output image. Denoise steps /
    backbone layers are overridable via HY_STEPS / HY_NUM_LAYERS (fast CI defaults
    applied when the env does not already set them)."""
    out_png = tmp_path / "hy_t2i_ci.png"
    # The prompt travels in HY_PROMPT rather than argv, so the argv list stays fully
    # literal (interpreter + resolved demo.py path). demo.py's precedence is
    # HY_PROMPT_FILE > argv[1] > HY_PROMPT, so with no argv[1] this is what it reads.
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HY_SKIP_WEIGHT_DOWNLOAD": "1",
        "HY_STEPS": os.environ.get("HY_STEPS", "8"),
        "HY_NUM_LAYERS": os.environ.get("HY_NUM_LAYERS", "32"),
        "HY_GUIDANCE": os.environ.get("HY_GUIDANCE", "5.0"),
        "HY_OUT": str(out_png),
        "HY_PROMPT": prompt,
    }
    env.pop("HY_PROMPT_FILE", None)
    result = subprocess.run([sys.executable, str(_DEMO)], cwd=str(_ROOT), env=env, shell=False)
    assert result.returncode == 0, f"demo.py exited with {result.returncode}"
    assert out_png.is_file() and out_png.stat().st_size > 0, f"no output image written to {out_png}"
