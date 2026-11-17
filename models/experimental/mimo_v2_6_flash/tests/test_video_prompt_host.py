# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A served video prompt is the checkpoint processor's (tt/video_prompt.py; host, needs vLLM 0.31.0 and the checkpoint).

The chat template renders a video as ``<|vision_start|><|video_pad|><|vision_end|>`` and the checkpoint's
``MiMoOmniProcessor`` replaces all three tokens by ``<|mimo_video_start|> (timestamp <|vision_start|> pads
<|vision_end|>)* <|mimo_video_end|>``. vLLM's upstream processor replaces only the pad, leaving the outer pair around
the expansion (MIMO_VIDEO_PROMPT=vllm, the default); with MIMO_VIDEO_PROMPT=hf the served prompt equals the checkpoint
processor's token for token, and the item's placeholder range and embedding rows still cover exactly its video pads."""

import os

import pytest
import torch

pytest.importorskip("vllm")

from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vi  # noqa: E402
from models.experimental.mimo_v2_6_flash.tests.common import model_path  # noqa: E402

PROMPT = "<|im_start|>user\n" + vi.VIDEO_PLACEHOLDER + "What happens in the video?<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def processors():
    path = model_path()
    if not os.path.exists(os.path.join(path, "config.json")):
        pytest.skip(f"no checkpoint at {path}")
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniMultiModalProcessor

    from models.experimental.mimo_v2_6_flash.tt.video_prompt import VideoPromptMixin

    class Processor(VideoPromptMixin, MiMoV2OmniMultiModalProcessor):
        pass

    upstream = vi.vllm_processor(path)
    return upstream, Processor(upstream.info, upstream.dummy_inputs)


def run(proc, frames, prompt=PROMPT):
    out = proc(prompt, proc.info.parse_mm_data({"video": [frames]}), hf_processor_mm_kwargs={})
    (ph,) = out["mm_placeholders"]["video"]
    return out["prompt_token_ids"], ph


def checkpoint_processor_ids(proc, frames, prompt=PROMPT):
    """The checkpoint processor's own token ids for the prompt (MiMoOmniProcessor.__call__ on the text)."""
    hf = proc.info.get_hf_processor()
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    video = (t, torch.arange(t.shape[0], dtype=torch.float32))  # 1 fps, as vLLM's processor hands frames over
    out = hf(text=prompt, videos=[video], return_tensors="pt")
    return out["input_ids"].reshape(-1).tolist()


@pytest.mark.parametrize("num_frames", [4, 32])
def test_video_prompt_is_the_checkpoint_processors(processors, num_frames, monkeypatch):
    from models.experimental.mimo_v2_6_flash.tt import video_prompt

    upstream, ours = processors
    frames = vi.sample_video(num_frames=num_frames, size=(320, 192))
    tok = upstream.info.get_tokenizer()
    vocab = tok.get_vocab()
    vs, ve = vocab["<|vision_start|>"], vocab["<|vision_end|>"]
    video_start, video_end = vocab["<|mimo_video_start|>"], vocab["<|mimo_video_end|>"]
    pad = vocab["<|video_pad|>"]

    monkeypatch.setattr(video_prompt, "VIDEO_PROMPT", "hf")
    ids, ph = run(ours, frames)
    assert ids == checkpoint_processor_ids(upstream, frames)
    # The placeholder range is the expansion, its embedding rows exactly its pads.
    item = ids[ph.offset : ph.offset + ph.length]
    assert item[0] == video_start and item[-1] == video_end
    assert ids[ph.offset - 1] != vs and ids[ph.offset + ph.length] != ve
    is_embed = torch.as_tensor(ph.is_embed)
    assert torch.equal(is_embed, torch.tensor(item) == pad)
    assert int(is_embed.sum()) == sum(1 for t in ids if t == pad)

    # Upstream (and MIMO_VIDEO_PROMPT=vllm): the same expansion wrapped in the template's outer pair.
    up_ids, up_ph = run(upstream, frames)
    assert up_ids[up_ph.offset - 1] == vs and up_ids[up_ph.offset + up_ph.length] == ve
    assert (
        up_ids[: up_ph.offset - 1] + up_ids[up_ph.offset : up_ph.offset + up_ph.length] == ids[: ph.offset + ph.length]
    )
    assert len(up_ids) == len(ids) + 2
    monkeypatch.setattr(video_prompt, "VIDEO_PROMPT", "vllm")
    assert run(ours, frames)[0] == up_ids


def test_image_prompts_unchanged(processors):
    upstream, ours = processors
    img = vi.sample_image(size=(224, 160))
    prompt = "<|im_start|>user\n" + vi.IMAGE_PLACEHOLDER + "Describe.<|im_end|>\n"
    a = upstream(prompt, upstream.info.parse_mm_data({"image": [img]}), hf_processor_mm_kwargs={})
    b = ours(prompt, ours.info.parse_mm_data({"image": [img]}), hf_processor_mm_kwargs={})
    assert a["prompt_token_ids"] == b["prompt_token_ids"]
    assert a["mm_placeholders"]["image"][0].offset == b["mm_placeholders"]["image"][0].offset
