# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A video's prompt tokens as the checkpoint's own processor writes them.

MiMo's chat template renders a video as ``<|vision_start|><|video_pad|><|vision_end|>``. The checkpoint's processor
(``MiMoOmniProcessor``, vllm/transformers_utils/processors/mimo_v2_omni.py) replaces that whole placeholder by

    <|mimo_video_start|> 00:00 <|vision_start|> pad x n <|vision_end|> 00:02 <|vision_start|> ... <|mimo_video_end|>

(``process``: ``_ids = [video_start_token_id] ... + [video_end_token_id]``), as it does for images and audio, whose
expansions bring their own start / end tokens. vLLM 0.31.0's ``MiMoV2OmniMultiModalProcessor._get_prompt_updates``
targets only ``<|video_pad|>``, so when vLLM applies the update itself (its processor-cache path, the default) the
template's outer pair stays around the expansion:

    <|vision_start|> <|mimo_video_start|> ... <|mimo_video_end|> <|vision_end|>

With MIMO_VIDEO_PROMPT=hf (the default) the update targets the whole placeholder, so the prompt is the checkpoint
processor's (tests/test_video_prompt_host.py), and Xiaomi's API's (its prompt for the same 32 frames is 2 tokens
shorter than upstream's); ``vllm`` keeps upstream's. Images and audio are the same either way (their targets already
give the processor's tokens).

Measured on Video-MME short docs 0-59 (one row engine, concurrency 8, 2026-11-07; Xiaomi's API: 75.0 %):

* 32 frames, no audio (three runs each): ``vllm`` 63.3 / 63.3 / 60.0 %, ``hf`` 60.0 / 58.3 / 58.3 %; verbose answers
  ("Based on the video, ...") 13-15 vs 12-15 of 60.
* 2 fps, 280 tokens a frame pair, no audio: 60.0 % either way (verbose 5 vs 4).
* 2 fps, 280 tokens a frame pair, with the audio track (tt/video_audio.py): ``vllm`` 70.0 / 68.3 % with 9 / 10 verbose
  answers, ``hf`` 71.7 % with none: the audio blocks inside an upstream prompt's extra vision pair push the model to
  explain. Hence ``hf``."""

import os

from vllm.multimodal.processing import PromptReplacement

VIDEO_PROMPT = os.environ.get("MIMO_VIDEO_PROMPT", "hf")
if VIDEO_PROMPT not in ("hf", "vllm"):
    raise ValueError(f"MIMO_VIDEO_PROMPT={VIDEO_PROMPT!r}: hf or vllm")


def video_placeholder_ids(hf_processor, tokenizer):
    """The token ids of the chat template's video placeholder: vision start, video pad, vision end."""
    p = hf_processor.mimo_processor
    return [p.vision_start_token_id, tokenizer.get_vocab()[hf_processor.video_token], p.vision_end_token_id]


def whole_placeholder_updates(updates, video_target):
    """Upstream's prompt updates with the video one retargeted to the whole placeholder ``video_target``."""
    return [
        PromptReplacement(modality=u.modality, target=list(video_target), replacement=u.replacement)
        if u.modality == "video"
        else u
        for u in updates
    ]


class VideoPromptMixin:
    """For a MiMoV2OmniMultiModalProcessor subclass: with MIMO_VIDEO_PROMPT=hf (default), video prompt tokens as the
    checkpoint's processor writes them (module docstring); upstream's otherwise."""

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        if VIDEO_PROMPT != "hf":
            return updates
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        target = video_placeholder_ids(hf_processor, self.info.get_tokenizer())
        return whole_placeholder_updates(updates, target)
