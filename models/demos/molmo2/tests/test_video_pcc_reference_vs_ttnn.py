# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Video parity smoke test: TTNN video demo + HF top-5 next-token predictions.

This test runs the TTNN path through ``run_video_demo`` (same entrypoint used by the CLI demo) and then runs
the HF reference model on the *same* video + prompt. It logs HF top-5 first-token predictions in the same spirit
as TTNN demo diagnostics so outputs can be compared quickly.

**Prompt:** Must include the literal ``<|video|>`` marker (start of user content is typical). If you omit it, the
processor may still return video tensors while ``input_ids`` has no video span, and HF will assert
``Expected 0 videos, but got 1``. The test prepends ``<|video|>\\n`` when missing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytest
import torch
from loguru import logger

from models.demos.molmo2.demo.demo import run_video_demo
from models.demos.molmo2.reference.model import Molmo2Reference


def _video_pcc_num_frames() -> int:
    """Frames for HF + TTNN (default 80 for T3K video PCC)."""
    raw = os.environ.get("MOLMO2_VIDEO_PCC_NUM_FRAMES", "368")
    try:
        n = int(raw)
    except ValueError:
        return 80
    return max(1, n)


def _video_demo_max_new_tokens() -> int:
    raw = os.environ.get("MOLMO2_VIDEO_DEMO_MAX_NEW_TOKENS", "16")
    try:
        n = int(raw)
    except ValueError:
        return 16
    return max(1, n)


def _video_demo_max_fps() -> Optional[float]:
    raw = os.environ.get("MOLMO2_VIDEO_DEMO_MAX_FPS", "2.0")
    if raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _normalize_molmo2_video_prompt(prompt: str) -> str:
    """
    Molmo2 video conditioning requires ``<|video|>`` in the user prompt so ``input_ids`` contain the video
    placeholder / end markers that match ``pixel_values_videos`` / ``video_grids``.
    """
    p = prompt.strip()
    if "<|video|>" in p:
        return prompt
    logger.warning("Prompt missing <|video|>; prepending '<|video|>\\n' so HF and demo use valid video conditioning.")
    return "<|video|>\n" + p


def _hf_video_batch_from_processor(
    processor,
    video_path: str,
    prompt: str,
    num_frames: int = 384,
    max_fps: Optional[float] = 2.0,
) -> Dict[str, torch.Tensor]:
    """Build keyword args for ``Molmo2ForConditionalGeneration`` video forward (torch tensors on CPU)."""
    messages = [{"role": "user", "content": prompt}]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    call_kwargs: Dict[str, Any] = {
        "text": text,
        "videos": video_path,
        "return_tensors": "pt",
        "num_frames": num_frames,
    }
    if max_fps is not None:
        call_kwargs["max_fps"] = max_fps
    result = processor(**call_kwargs)
    batch: Dict[str, Any] = {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
        "pixel_values_videos": result["pixel_values_videos"],
        "video_token_pooling": result["video_token_pooling"],
        "video_grids": result["video_grids"],
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if "token_type_ids" in result:
        batch["token_type_ids"] = result["token_type_ids"].long()
    return batch


@pytest.mark.parametrize("num_frames", [_video_pcc_num_frames()])
def test_video_pcc_reference_vs_ttnn_single_flow(
    molmo2_video_path,
    molmo2_video_prompt,
    model_location,
    num_frames,
):
    """Run TTNN video demo + HF top-5 first-token logits on same input."""
    max_new_tokens = _video_demo_max_new_tokens()
    max_fps = _video_demo_max_fps()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_location, trust_remote_code=True)

    video_prompt = """<|video|>
The person uses multiple similar objects to play an occlusion game. Where is the hidden object at the end of the game from the person's point of view?
\nA. Under the first object from the left.
\nB. Under the third object from the left.
\nC. Under the second object from the left.
\nPlease respond with only the letter of the correct answer."""
    logger.info(
        "Video demo parity run — video={}\nprompt={!r}\nnum_frames={}\nmax_fps={}\nmax_new_tokens={}",
        molmo2_video_path,
        video_prompt,
        num_frames,
        max_fps,
        max_new_tokens,
    )

    # TTNN path: run the actual demo entrypoint (opens/closes T3K mesh internally).
    ttnn_output, perf = run_video_demo(
        video_path=molmo2_video_path,
        prompt=video_prompt,
        max_new_tokens=max_new_tokens,
        num_layers=None,
        max_seq_len=65536,
        max_frames=num_frames,
        max_fps=10,
        native_video_fps=False,
        use_trace=False,
        use_decode_trace=False,
        use_vision_trace=False,
        use_unified_trace=False,
        use_dp_vision_trace=True,
        use_paged_attention=True,
        batch_size=1,
        num_devices=8,
        use_data_parallel=True,
        frames_per_device=8,
        repetition_penalty=1.0,
        use_async_ccl=False,
    )
    logger.info("TTNN demo output: {!r}", ttnn_output)
    logger.info(
        "TTNN perf: frames={}, ttft_ms={:.2f}, tok_per_s={:.2f}",
        perf.get("n_frames", -1),
        float(perf.get("ttft_ms", 0.0)),
        float(perf.get("tokens_per_sec", 0.0)),
    )

    # HF path: same prompt/video and frame sampling settings, then print top-5 first-token logits.
    hf_batch = _hf_video_batch_from_processor(
        processor,
        molmo2_video_path,
        video_prompt,
        num_frames=num_frames,
        max_fps=max_fps,
    )
    ref = Molmo2Reference(model_location, torch_dtype=torch.bfloat16)
    hf_model = ref.model
    hf_model.eval()

    device_torch = next(hf_model.parameters()).device
    hf_batch = {k: v.to(device_torch) if torch.is_tensor(v) else v for k, v in hf_batch.items()}

    with torch.no_grad():
        hf_out = hf_model(**hf_batch)

    hf_logits_last = hf_out.logits[0, -1, :].float().cpu()
    logger.info(
        "HF next_token_logits stats: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}",
        float(hf_logits_last.mean().item()),
        float(hf_logits_last.std().item()),
        float(hf_logits_last.min().item()),
        float(hf_logits_last.max().item()),
    )

    top5_values, top5_indices = torch.topk(hf_logits_last, 5)
    logger.info("HF Top 5 predictions:")
    for i, (val, idx) in enumerate(zip(top5_values.tolist(), top5_indices.tolist())):
        decoded = processor.tokenizer.decode([idx])
        logger.info("  {}. token={}, logit={:.2f}, decoded={!r}", i + 1, idx, val, decoded)

    hf_first = int(torch.argmax(hf_logits_last).item())
    hf_first_text = processor.tokenizer.decode([hf_first])
    logger.info("HF selected first token: {} -> {!r}", hf_first, hf_first_text)

    assert isinstance(ttnn_output, str)
    assert ttnn_output != ""
