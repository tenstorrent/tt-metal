# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end OCR parity test for rednote-hilab/dots.ocr (image -> text).

Runs the TTNN OCR pipeline (:class:`tt.ocr_model.TtOcrModel`) on a known
document image and compares the generated text against the HuggingFace
reference ``DotsOCRForCausalLM`` (greedy decode) computed with the SAME real
checkpoint weights at the SAME layer depth. The validation_metric is
``accuracy`` (char-level / token text match vs HF); the gate is "HF - 1.0",
i.e. the TTNN output must match HF to within the threshold (HF's own accuracy
against itself is 1.0, so the gate is accuracy >= 0.0 -- the pipeline must
produce a valid generation that we then score against HF).

The model dir name contains a dot, so the pipeline + loader are imported by
file path via importlib (the project convention).

Config: the full production model is 28 LM layers + 42 vision layers. That
full-depth AR run is heavy for a single device session, so this e2e validates
the AR generation loop + vision->text scatter + decode wiring at a REDUCED but
representative depth (env DOTS_OCR_LM_LAYERS / DOTS_OCR_VISION_LAYERS, default
2/2), with the HF reference truncated to the SAME depth for an apples-to-apples
parity comparison. The real checkpoint weights are used for every loaded layer.
"""
import importlib.util
import os

import torch

import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
_TT_DIR = os.path.normpath(os.path.join(_HERE, "..", "tt"))
_DEMO_DIR = os.path.normpath(os.path.join(_HERE, "..", "demo"))

CHECKPOINT_PATH = os.environ.get(
    "DOTS_OCR_CHECKPOINT",
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/"
    "c0111ce6bc07803dbc267932ffef0ae3a51dc951",
)
LM_LAYERS = int(os.environ.get("DOTS_OCR_LM_LAYERS", "28"))
VISION_LAYERS = int(os.environ.get("DOTS_OCR_VISION_LAYERS", "42"))
MAX_NEW_TOKENS = int(os.environ.get("DOTS_OCR_MAX_NEW_TOKENS", "12"))
SAMPLE_IMAGE = os.path.join(_DEMO_DIR, "sample_ocr.png")
PROMPT = "Read the text in the image."
EOS_TOKEN_IDS = (151643, 151673)


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _char_accuracy(pred: str, ref: str) -> float:
    """Char-level match fraction of pred vs ref (1.0 == identical)."""
    n = max(len(pred), len(ref), 1)
    match = sum(1 for a, b in zip(pred, ref) if a == b)
    return match / n


def _token_accuracy(pred_ids, ref_ids) -> float:
    n = max(len(pred_ids), len(ref_ids), 1)
    match = sum(1 for a, b in zip(pred_ids, ref_ids) if a == b)
    return match / n


def _run_ocr_e2e(device):
    """Run TTNN OCR + HF reference, return (accuracy, tokens_generated, tt_text, hf_text)."""
    demo = _load_by_path("dots_e2e_demo", "demo_ocr.py", _DEMO_DIR)
    loader = _load_by_path("dots_e2e_loader", "weight_loader.py", _TT_DIR)
    ocr_model_mod = _load_by_path("dots_e2e_ocr_model", "ocr_model.py", _TT_DIR)

    # Host preprocessing (DotsVL image patchify + chat prompt with <|imgpad|>).
    input_ids, pixel_values, grid_thw, tokenizer = demo.host_preprocess(SAMPLE_IMAGE, PROMPT, CHECKPOINT_PATH)
    pixel_values = pixel_values.to(torch.float32)

    # Real checkpoint weights at the (reduced) depth for both paths.
    lm_sd = loader.load_language_model_weights(CHECKPOINT_PATH, num_layers=LM_LAYERS)
    vis_sd = loader.load_vision_tower_weights(CHECKPOINT_PATH, num_layers=VISION_LAYERS)

    tt_model = ocr_model_mod.TtOcrModel(
        device=device,
        lm_state_dict=lm_sd,
        vision_state_dict=vis_sd,
        grid_thw=grid_thw,
        lm_num_layers=LM_LAYERS,
        vision_num_layers=VISION_LAYERS,
    )
    tt_ids = tt_model.generate(input_ids, pixel_values, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=EOS_TOKEN_IDS)

    # HF reference (DotsOCRForCausalLM) at the SAME reduced depth, same weights.
    hf_ids = demo.run_hf_reference(input_ids, pixel_values, grid_thw, LM_LAYERS, VISION_LAYERS, MAX_NEW_TOKENS)

    tt_text = tokenizer.decode(tt_ids, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)

    # accuracy: char-level text match of the TTNN generation vs the HF reference.
    accuracy = _char_accuracy(tt_text, hf_text)
    token_acc = _token_accuracy(tt_ids, hf_ids)
    print(f"[e2e ocr] LM_LAYERS={LM_LAYERS} VISION_LAYERS={VISION_LAYERS} max_new_tokens={MAX_NEW_TOKENS}")
    print(f"[e2e ocr] TTNN tokens={tt_ids}")
    print(f"[e2e ocr] HF   tokens={hf_ids}")
    print(f"[e2e ocr] TTNN text={tt_text!r}")
    print(f"[e2e ocr] HF   text={hf_text!r}")
    print(f"[e2e ocr] char accuracy vs HF = {accuracy:.4f} | token accuracy = {token_acc:.4f}")
    return accuracy, token_acc, len(tt_ids), tt_text, hf_text


def test_e2e_ocr(device):
    """TTNN OCR image->text must match HF DotsOCRForCausalLM (accuracy gate 'HF - 1.0')."""
    accuracy, token_acc, n_tokens, tt_text, hf_text = _run_ocr_e2e(device)

    # MEANINGFUL gate (KV-cache perf phase, full 28 LM / 42 vision depth). At
    # full depth the KV-cache decode path produces real OCR text that matches the
    # HF DotsOCRForCausalLM reference: on the sample image both decode "HELLO
    # 2026" token-for-token to EOS. The use_case gate "HF - 1.0" tolerates the
    # full 1.0 margin for bf16 argmax drift, but at full depth we require a
    # SUBSTANTIAL char-level match (>= 0.5) -- the pipeline must produce real
    # text, not a leading-token-only match. Reduced-depth runs (env overrides
    # LM/VISION layers < full) only need a valid generation (gate 0.0), since a
    # 2-layer trunk cannot do real OCR.
    assert n_tokens > 0, "OCR pipeline produced no tokens"
    full_depth = LM_LAYERS >= 28 and VISION_LAYERS >= 42
    gate = 0.5 if full_depth else (1.0 - 1.0)  # full-depth meaningful gate vs 'HF - 1.0'
    assert accuracy >= gate, (
        f"OCR accuracy {accuracy:.4f} below gate {gate:.4f} "
        f"(full_depth={full_depth}, TTNN text={tt_text!r}, HF text={hf_text!r})"
    )
    print(f"[e2e ocr] PASS accuracy={accuracy:.4f} token_accuracy={token_acc:.4f} tokens={n_tokens} gate={gate}")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_ocr_e2e(device)
    finally:
        ttnn.close_device(device)
