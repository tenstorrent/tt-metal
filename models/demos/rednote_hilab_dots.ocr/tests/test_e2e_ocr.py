# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""E2E HF-parity gate for the dots.ocr ``ocr`` use case.

Runs the full TTNN pipeline (tt/ocr_model.py: vision tower -> embedding
splice -> 28-layer decoder -> lm_head -> greedy AR loop) and the HF
reference (DotsOCRForCausalLM, bf16 CPU greedy) over the same synthetic
document samples, computes corpus WER for both against the rendered
ground truth, and asserts the inventory gate
``validation_threshold = "HF + 0.05"``: ttnn_wer <= hf_wer + 0.05.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch  # noqa: F401  (keeps torch initialized before ttnn)

import ttnn

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
SAMPLES_PATH = MODEL_DIR / "demo" / "inputs" / "ocr_samples.json"
MAX_NEW_TOKENS = 32  # short-form gate; perf is the perf phase's job
WER_TOLERANCE = 0.05  # "HF + 0.05" from use_case.validation_threshold


def _load_by_path(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


demo_ocr = _load_by_path("dots_ocr_demo_ocr", MODEL_DIR / "demo" / "demo_ocr.py")
validate = _load_by_path("dots_ocr_demo_validate", MODEL_DIR / "demo" / "validate.py")
ocr_mod = _load_by_path("dots_ocr_tt_ocr_model", MODEL_DIR / "tt" / "ocr_model.py")


@pytest.fixture(scope="module")
def samples():
    if not SAMPLES_PATH.exists():
        _load_by_path("dots_ocr_make_samples", MODEL_DIR / "demo" / "inputs" / "make_samples.py").main()
    entries = json.loads(SAMPLES_PATH.read_text())
    return [{"image": str(SAMPLES_PATH.parent / e["image"]), "ref": e["ref"]} for e in entries]


@pytest.fixture(scope="module")
def hf_outputs(samples):
    # HF reference once per module (loads + frees the bf16 model inside).
    return demo_ocr.run_hf_reference([s["image"] for s in samples], max_new_tokens=MAX_NEW_TOKENS)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ocr_wer_matches_hf(samples, hf_outputs, mesh_device):
    from PIL import Image

    tokenizer, image_processor, chat_template = demo_ocr.load_host_processors()
    model = ocr_mod.TtOCRModel(mesh_device, tokenizer, image_processor, chat_template)
    ttnn_outputs = [model.ocr(Image.open(s["image"]).convert("RGB"), max_new_tokens=MAX_NEW_TOKENS) for s in samples]

    refs = [s["ref"] for s in samples]
    ttnn_wer = validate.wer(ttnn_outputs, refs)
    hf_wer = validate.wer(hf_outputs, refs)

    print(
        f"TTNN wer={ttnn_wer:.4f}  HF wer={hf_wer:.4f}  " f"drift={ttnn_wer - hf_wer:.4f}  (tolerance={WER_TOLERANCE})"
    )
    for i, (s, hf_t, tt_t) in enumerate(zip(samples, hf_outputs, ttnn_outputs)):
        print(f"  [{i}] ref={s['ref']!r}\n      HF ={hf_t!r}\n      TT ={tt_t!r}")

    assert ttnn_wer <= hf_wer + WER_TOLERANCE, (
        f"TTNN wer {ttnn_wer:.4f} exceeds HF wer {hf_wer:.4f} + {WER_TOLERANCE} — "
        f"investigate AR drift (first diverging argmax step) per the generation skill."
    )
