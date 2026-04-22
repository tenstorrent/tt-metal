# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: run the HF-only demo path (torch + transformers), no TTNN / no device.

This mirrors ``python -m models.demos.dots_ocr.demo.demo --backend hf`` but stays hermetic:
- No ``MESH_DEVICE`` required
- No ``import ttnn`` in the demo path itself

**How to run without the Tenstorrent wheel:** the repo root ``conftest.py`` imports ``ttnn`` at
import time. Use ``--confcutdir`` so only ``models/demos/dots_ocr/tests/conftest.py`` is loaded::

    cd tt-metal && PYTHONPATH=. python3 -m pytest \\
      models/demos/dots_ocr/tests/test_demo_hf_torch_only.py \\
      --confcutdir=models/demos/dots_ocr/tests -v

By default uses ``hf-internal-testing/tiny-random-LlamaForCausalLM`` (override with ``HF_MODEL``).
For **image → text** readout, set ``HF_MODEL`` to a multimodal checkpoint (e.g. ``dots.mocr``);
the test then writes a small PNG under ``tmp_path`` unless ``DOTS_HF_TEST_IMAGE`` points to a file.

Set ``RUN_DOTS_HF_DEMO_FULL=1`` to enable the optional full ``dots.mocr`` integration test (large download).
"""

from __future__ import annotations

import os

import pytest

_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _is_tiny_text_only_model(model_id: str) -> bool:
    return model_id == _TINY_LLAMA or "tiny-random" in model_id


def test_hf_demo_backend_torch_only_smoke(tmp_path):
    """
    Exercise :func:`models.demos.dots_ocr.demo.demo.run_hf_backend`.

    - **Tiny default model:** text-only prompt (plain CausalLM cannot run vision).
    - **Multimodal ``HF_MODEL``:** passes a real image path, decodes generation to a string (readout).

    Optional: ``DOTS_HF_TEST_IMAGE`` — path to an existing image file (overrides synthetic PNG).
    Optional: ``DOTS_HF_TEST_PROMPT`` — prompt string (defaults differ for text vs image).
    """
    model_id = os.environ.get("HF_MODEL", _TINY_LLAMA)
    _orig = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = model_id
    try:
        from models.demos.dots_ocr.demo.demo import run_hf_backend

        env_image = os.environ.get("DOTS_HF_TEST_IMAGE")
        if env_image:
            print("Case where env image .... DOTS_HF_TEST_IMAGE \n")
            assert os.path.isfile(env_image), f"DOTS_HF_TEST_IMAGE is not a file: {env_image}"
            if _is_tiny_text_only_model(model_id):
                pytest.skip(
                    "DOTS_HF_TEST_IMAGE requires a multimodal HF_MODEL "
                    "(e.g. rednote-hilab/dots.mocr); tiny CausalLM cannot run vision."
                )
            image_path = env_image
            prompt = os.environ.get("DOTS_HF_TEST_PROMPT", "Describe this image briefly.")
            max_new = int(os.environ.get("DOTS_HF_TEST_MAX_NEW_TOKENS", "32"))
        elif _is_tiny_text_only_model(model_id):
            image_path = ""
            print("Case where _is_tiny_text_only_model \n")
            prompt = "Hello"
            max_new = 4
        else:
            from PIL import Image

            print("Case where Else ,,,, torch only smoke \n")
            png = tmp_path / "hf_torch_only_smoke.png"
            Image.new("RGB", (128, 128), color=(248, 248, 248)).save(png)
            image_path = str(png)
            prompt = os.environ.get("DOTS_HF_TEST_PROMPT", "Read any text in the image.")
            max_new = int(os.environ.get("DOTS_HF_TEST_MAX_NEW_TOKENS", "32"))

        text = run_hf_backend(
            model_id=model_id,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new,
        )
        # Readout: new tokens only via ``decode_generated_suffix`` (full ``generate`` output includes the prompt).
        assert isinstance(text, str)
    finally:
        if _orig is None:
            os.environ.pop("HF_MODEL", None)
        else:
            os.environ["HF_MODEL"] = _orig


@pytest.mark.skipif(
    os.environ.get("RUN_DOTS_HF_DEMO_FULL") != "1",
    reason="Set RUN_DOTS_HF_DEMO_FULL=1 to run HF demo against rednote-hilab/dots.mocr (large download).",
)
def test_hf_demo_dots_mocr_optional(tmp_path):
    """Optional: full dots.mocr HF path with a synthetic image and decoded readout."""
    model_id = os.environ.get("HF_MODEL", "rednote-hilab/dots.mocr")
    from PIL import Image

    from models.demos.dots_ocr.demo.demo import run_hf_backend

    png = tmp_path / "dots_mocr_hf_optional.png"
    Image.new("RGB", (128, 128), color=(245, 245, 245)).save(png)
    image_path = os.environ.get("DOTS_HF_TEST_IMAGE") or str(png)
    if not os.path.isfile(image_path):
        pytest.fail(f"Image path is not a file: {image_path}")

    text = run_hf_backend(
        model_id=model_id,
        image_path=image_path,
        prompt=os.environ.get("DOTS_HF_TEST_PROMPT", "Read any text in the image."),
        max_new_tokens=int(os.environ.get("DOTS_HF_TEST_MAX_NEW_TOKENS", "32")),
    )
    assert isinstance(text, str)
