# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E TTNN pipeline tests for dots.ocr (text-only and vision+text)."""

import os
import time

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.models.dots_ocr import TTNNDotsOCRPipeline


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}


def _resolve_mesh_device_shape():
    mesh_device = os.environ.get("MESH_DEVICE")
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP":
        return DOTS_OCR_DP_MESH_DEVICE_MAP.get(
            mesh_device, MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))
        )
    return MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))


def _dots_ocr_mesh_num_devices():
    sh = _resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        if len(sh) >= 2:
            return int(sh[0]) * int(sh[1])
        if len(sh) == 1:
            return int(sh[0])
    return 1


def _dots_ocr_device_params():
    dp = {"trace_region_size": 300000000, "num_command_queues": 1}
    if _dots_ocr_mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"

DOTS_OCR_LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
    "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
    "'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)


def _resolve_model_path():
    """Resolve dots.ocr model path: env var > HF cache > model ID for auto-download."""
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


def _dots_ocr_pipeline_batch_size():
    """Match ``TTNNDotsOCRPipeline`` batch to mesh size when DP is requested.

    DP sharding in the pipeline requires ``batch_size == num_devices`` on the
    mesh. ``DOTS_OCR_PARALLELISM=DP`` alone only changes the *fixture* mesh
    shape (e.g. N300 ``(2, 1)``); without this, tests still run batch 1.
    """
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() != "DP":
        return 1
    n = _dots_ocr_mesh_num_devices()
    return n if n > 1 else 1


def _dots_ocr_stack_input_ids_for_dp(input_ids: torch.Tensor) -> torch.Tensor:
    """Turn ``[1, S]`` into ``[B, S]`` by repeating the same prompt on each stream."""
    bs = _dots_ocr_pipeline_batch_size()
    if bs <= 1 or input_ids.shape[0] == bs:
        return input_ids
    if input_ids.shape[0] != 1:
        raise ValueError(f"DP batch stacking expects base shape [1, S], got {tuple(input_ids.shape)}")
    return input_ids.expand(bs, -1).contiguous()


def _parse_predicted_layout(text):
    """Extract the JSON layout array from a model output. Returns list[dict] or None.

    The model is asked to emit a single JSON array. In practice the output may be
    wrapped in markdown fences or prose; we try a direct parse first, then fall
    back to the widest ``[ ... ]`` slice in the string.
    """
    import json
    import re

    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    first, last = text.find("["), text.rfind("]")
    if first != -1 and last > first:
        try:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _layout_concat_text(layout):
    """Concatenate the ``text`` fields of a layout list in reading order."""
    parts = []
    for elem in layout or []:
        if isinstance(elem, dict):
            t = elem.get("text")
            if isinstance(t, str) and t:
                parts.append(t)
    return "\n".join(parts)


def _norm_text(s):
    """Collapse whitespace for fuzzy comparison."""
    return " ".join((s or "").split())


def _compare_layout(pred_layout, gt_layout):
    """Compute simple comparison metrics between two layout lists."""
    import difflib
    from collections import Counter

    pred_cats = Counter((e.get("category") or "<missing>") for e in (pred_layout or []) if isinstance(e, dict))
    gt_cats = Counter((e.get("category") or "<missing>") for e in (gt_layout or []) if isinstance(e, dict))
    pred_text = _norm_text(_layout_concat_text(pred_layout))
    gt_text = _norm_text(_layout_concat_text(gt_layout))
    text_ratio = difflib.SequenceMatcher(None, pred_text, gt_text).ratio() if (pred_text and gt_text) else 0.0
    missing_cats = sorted((gt_cats - pred_cats).keys())
    extra_cats = sorted((pred_cats - gt_cats).keys())
    return {
        "pred_count": len(pred_layout) if pred_layout else 0,
        "gt_count": len(gt_layout) if gt_layout else 0,
        "pred_categories": dict(pred_cats),
        "gt_categories": dict(gt_cats),
        "missing_categories": missing_cats,
        "extra_categories": extra_cats,
        "text_similarity": text_ratio,
    }


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
def test_dots_ocr_text(mesh_device):
    """Test standalone TTNN pipeline for dots.ocr (text-only, no vision)."""

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
    )

    tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "What is optical character recognition and how does it work?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])

    pipeline.warmup(input_ids)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(input_ids, max_new_tokens=128)
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        text = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline TEXT OUTPUT: {text}")

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Text Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(text.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_text_timing_stats.csv")
    pipeline.release()


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_link",
    [
        "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg",
    ],
)
def test_dots_ocr_vision(mesh_device, image_link):
    """Test standalone TTNN pipeline for dots.ocr with vision (image + text).

    Default mesh comes from ``MESH_DEVICE`` (e.g. N300 ``(1, 2)``). With
    ``DOTS_OCR_PARALLELISM=DP``, the fixture uses ``DOTS_OCR_DP_MESH_DEVICE_MAP``
    (N300 ``(2, 1)``). In DP mode the test sets ``batch_size == num_devices`` and
    repeats the same prompt on each stream so dual-stream sharding is exercised.
    """
    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import requests

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
    )

    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    # Load and crop the image
    image = Image.open(requests.get(image_link, stream=True).raw)
    original_width, original_height = image.size

    # Crop to 57.5% of original height from the top
    new_height = int(original_height * 0.575)
    top = 0
    bottom = new_height

    # Crop box: (left, top, right, bottom)
    image = image.crop((0, top, original_width, bottom))

    print(f"Cropped image from {original_width}x{original_height} to {original_width}x{new_height}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        max_length=2800,
        return_tensors="pt",
    )

    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=180,
        stop_on_eos=False,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        decoded = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)
    print(f"Pipeline VISION OUTPUT: {decoded}")

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    ms_per_token = total_time / num_tokens * 1000

    print(f"\n{'='*60}")
    print(f"dots.ocr Pipeline Vision Performance Summary")
    print(f"{'='*60}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_vision_timing_stats.csv")
    pipeline.release()


@pytest.mark.parametrize(
    "device_params",
    [_dots_ocr_device_params()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [_resolve_mesh_device_shape()],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_image_path, ground_truth_json_path, output_pred_path",
    [
        (
            os.environ.get("DOTS_OCR_INPUT_IMAGE", ""),
            os.environ.get("DOTS_OCR_GROUND_TRUTH_JSON", ""),
            os.environ.get("DOTS_OCR_OUTPUT_PRED", ""),
        ),
    ],
)
def test_dots_ocr_layout(mesh_device, input_image_path, ground_truth_json_path, output_pred_path):
    """Run dots.ocr layout OCR on a local image and (optionally) compare to ground truth.

    Inputs are driven by env vars (or pytest parametrize override):
      - DOTS_OCR_INPUT_IMAGE       : absolute path to the page image (jpg/png/...).
      - DOTS_OCR_GROUND_TRUTH_JSON : optional path to ground-truth layout JSON
        (list of ``{bbox, category, text}``). When provided, the test parses the
        model output as JSON and prints/saves comparison metrics.
      - DOTS_OCR_OUTPUT_PRED       : absolute path for the predicted output file
        (defaults to ``<input_stem>_pred.md`` next to the image).

    The test is skipped if any *provided* input path is missing.
    """
    if not input_image_path or not os.path.isfile(input_image_path):
        pytest.skip(f"Input image not found: {input_image_path!r}")
    if ground_truth_json_path and not os.path.isfile(ground_truth_json_path):
        pytest.skip(f"Ground-truth JSON not found: {ground_truth_json_path!r}")

    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    if not output_pred_path:
        stem, _ = os.path.splitext(input_image_path)
        output_pred_path = f"{stem}_pred.md"

    pbatch = _dots_ocr_pipeline_batch_size()
    pipeline = TTNNDotsOCRPipeline.from_hf_model(
        model_path=DOTS_OCR_LOCAL_PATH,
        device=mesh_device,
        batch_size=pbatch,
    )

    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    image = Image.open(input_image_path).convert("RGB")
    print(f"Loaded input image: {input_image_path} ({image.size[0]}x{image.size[1]})")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DOTS_OCR_LAYOUT_PROMPT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = _dots_ocr_stack_input_ids_for_dp(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]

    pipeline.warmup(input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    DispatchManager.clear_timings()
    start_time = time.time()
    generated_ids = pipeline.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=2048,
        stop_on_eos=True,
    )
    ttnn.synchronize_device(mesh_device)
    end_time = time.time()

    if isinstance(generated_ids[0], list):
        streams = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
        decoded = "\n--- stream ---\n".join(streams)
        num_tokens = sum(len(seq) for seq in generated_ids)
    else:
        decoded = processor.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time if total_time > 0 else 0.0
    ms_per_token = total_time / num_tokens * 1000 if num_tokens > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"dots.ocr Layout OCR Performance Summary")
    print(f"{'='*60}")
    print(f"Input image:          {input_image_path}")
    print(f"Output prediction:    {output_pred_path}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.1f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_pred_path)) or ".", exist_ok=True)
    with open(output_pred_path, "w", encoding="utf-8") as f:
        f.write(decoded)
    print(f"Wrote prediction to: {output_pred_path}")

    if ground_truth_json_path:
        with open(ground_truth_json_path, "r", encoding="utf-8") as f:
            gt_layout = json.load(f)

        # In DP, every stream sees the same input — score the first one and note
        # if other streams diverge (any non-determinism would surface here).
        if isinstance(generated_ids[0], list):
            stream_texts = [processor.decode(seq, skip_special_tokens=True) for seq in generated_ids]
            primary_text = stream_texts[0]
            num_unique_streams = len(set(stream_texts))
        else:
            primary_text = decoded
            num_unique_streams = 1

        pred_layout = _parse_predicted_layout(primary_text)
        metrics = _compare_layout(pred_layout, gt_layout) if pred_layout is not None else None

        print(f"\n{'='*60}")
        print(f"dots.ocr Layout OCR Comparison vs Ground Truth")
        print(f"{'='*60}")
        print(f"Ground truth:         {ground_truth_json_path}")
        print(f"Unique DP streams:    {num_unique_streams}")
        if pred_layout is None:
            print("Could not parse model output as a JSON list — skipping structural metrics.")
        else:
            print(f"Element count:        pred={metrics['pred_count']}  gt={metrics['gt_count']}")
            print(f"Text similarity:      {metrics['text_similarity']:.4f}  (difflib ratio over reading-order text)")
            print(f"Pred categories:      {metrics['pred_categories']}")
            print(f"GT categories:        {metrics['gt_categories']}")
            if metrics["missing_categories"]:
                print(f"Categories missing in pred: {metrics['missing_categories']}")
            if metrics["extra_categories"]:
                print(f"Categories extra in pred:   {metrics['extra_categories']}")
        print(f"{'='*60}\n")

        metrics_path = f"{os.path.splitext(output_pred_path)[0]}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input_image": input_image_path,
                    "ground_truth_json": ground_truth_json_path,
                    "output_prediction": output_pred_path,
                    "num_unique_dp_streams": num_unique_streams,
                    "parsed_prediction_as_json": pred_layout is not None,
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote metrics to:     {metrics_path}")

    print(f"\n{'='*60}")
    print(f"dots.ocr Layout OCR Predicted Output")
    print(f"{'='*60}")
    print(decoded)
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_layout_timing_stats.csv")
    pipeline.release()


@pytest.mark.parametrize(
    "input_image_path, output_pred_path",
    [
        (
            os.environ.get("DOTS_OCR_INPUT_IMAGE", ""),
            os.environ.get("DOTS_OCR_OUTPUT_PRED_CPU", ""),
        ),
    ],
)
def test_dots_ocr_layout_cpu(input_image_path, output_pred_path):
    """CPU reference inference of dots.ocr (no TTNN device required).

    Mirrors ``test_dots_ocr_layout`` but runs the HF ``DotsOCRForCausalLM``
    on CPU so the output can be diffed against the TTNN pipeline result for
    head-to-head quality comparison. Skipped if ``DOTS_OCR_INPUT_IMAGE`` is
    unset or missing.

    Knobs:
      - DOTS_OCR_INPUT_IMAGE        : page image path (required).
      - DOTS_OCR_OUTPUT_PRED_CPU    : output prediction path (defaults to
        ``<input_stem>_pred_cpu.md``).
      - DOTS_OCR_CPU_MAX_NEW_TOKENS : decode budget; default 1024. CPU
        inference is slow (~minutes per few hundred tokens), so the default
        is lower than the TTNN test's 2048.
    """
    if not input_image_path or not os.path.isfile(input_image_path):
        pytest.skip(f"Input image not found: {input_image_path!r}")

    pytest.importorskip("qwen_vl_utils")
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    from transformers import AutoModelForCausalLM

    if not output_pred_path:
        stem, _ = os.path.splitext(input_image_path)
        output_pred_path = f"{stem}_pred_cpu.md"

    max_new_tokens = int(os.environ.get("DOTS_OCR_CPU_MAX_NEW_TOKENS", "1024"))

    import json
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(DOTS_OCR_LOCAL_PATH, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(DOTS_OCR_LOCAL_PATH)
    with open(os.path.join(DOTS_OCR_LOCAL_PATH, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, _tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = "<|imgpad|>"
    processor.image_token_id = 151665

    image = Image.open(input_image_path).convert("RGB")
    print(f"Loaded input image: {input_image_path} ({image.size[0]}x{image.size[1]})")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DOTS_OCR_LAYOUT_PROMPT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prefill_len = int(inputs["input_ids"].shape[-1])
    print(f"Prefill seq_len: {prefill_len}, max_new_tokens: {max_new_tokens}")

    print(f"Loading {DOTS_OCR_MODEL_ID} on CPU from {DOTS_OCR_LOCAL_PATH} ...")
    load_start = time.time()
    cpu_dtype_env = os.environ.get("DOTS_OCR_CPU_DTYPE", "bf16").lower()
    cpu_dtype = torch.float32 if cpu_dtype_env in ("fp32", "float32") else torch.bfloat16
    print(f"CPU model dtype: {cpu_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        DOTS_OCR_LOCAL_PATH,
        torch_dtype=cpu_dtype,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    print(f"Model loaded in {time.time() - load_start:.1f}s")

    # The vision tower's forward() hard-codes ``bf16=True`` as the default
    # (modeling_dots_vision.py:492-494) and casts pixel_values to bf16 on
    # entry. When we want a pure fp32 run, override that default to False so
    # the input stays in the model's loaded dtype.
    if cpu_dtype == torch.float32:
        import functools

        _orig_vision_forward = model.vision_tower.forward
        model.vision_tower.forward = functools.partial(_orig_vision_forward, bf16=False)
        print("Patched vision_tower.forward to bf16=False (fp32 mode)")

    # Bypass transformers.generate() — newer versions strip pixel_values from
    # prepare_inputs_for_generation's scope, but DotsOCRForCausalLM.forward()
    # still needs it directly. Manual prefill + greedy decode loop instead.
    eos = model.config.eos_token_id
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())

    print(f"Running CPU prefill (seq_len={prefill_len}) ...")
    start_time = time.time()
    # Match pixel_values dtype to model dtype so the vision conv weights and
    # input agree (bf16 path also exists if the vision tower's bf16 cast is
    # not patched off).
    px = inputs["pixel_values"].to(cpu_dtype)
    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=px,
            image_grid_thw=inputs["image_grid_thw"],
            use_cache=True,
            return_dict=True,
        )
    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prefill_end = time.time()
    print(f"Prefill done in {prefill_end - start_time:.1f}s")

    generated_ids = [int(next_token.item())]
    print(f"Running CPU decode (max_new_tokens={max_new_tokens}; this is slow on CPU) ...")
    decode_start = time.time()
    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_token.item())
        generated_ids.append(tok)
        if (step + 1) % 25 == 0:
            elapsed = time.time() - decode_start
            rate = (step + 2) / elapsed
            print(f"  decoded {step+2}/{max_new_tokens} tokens in {elapsed:.1f}s ({rate:.2f} tok/s)")
        if tok in eos_set:
            print(f"  EOS at decode step {step+1}")
            break
    end_time = time.time()

    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)

    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time if total_time > 0 else 0.0
    ms_per_token = total_time / num_tokens * 1000 if num_tokens > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"dots.ocr Layout OCR CPU Reference Performance Summary")
    print(f"{'='*60}")
    print(f"Input image:          {input_image_path}")
    print(f"Output prediction:    {output_pred_path}")
    print(f"Prefill seq_len:      {prefill_len}")
    print(f"Generated tokens:     {num_tokens}")
    print(f"Total time:           {total_time:.3f} s")
    print(f"Throughput:           {tokens_per_second:.2f} tok/s")
    print(f"Avg time per token:   {ms_per_token:.1f} ms/tok")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_pred_path)) or ".", exist_ok=True)
    with open(output_pred_path, "w", encoding="utf-8") as f:
        f.write(decoded)
    print(f"Wrote prediction to: {output_pred_path}")

    print(f"\n{'='*60}")
    print(f"dots.ocr Layout OCR CPU Reference Predicted Output")
    print(f"{'='*60}")
    print(decoded)
    print(f"{'='*60}\n")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"
