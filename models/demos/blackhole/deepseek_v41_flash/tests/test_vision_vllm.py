# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM side of image input without a device: the multimodal processor (tt/vision_vllm.py) on a ModelConfig built from the serving shim (vllm_config), and the chat template vs the
checkpoint's ``encoding.encode_messages``. Needs vLLM (the server venv); skipped without it."""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vllm")

from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(HERE, "vllm_config")
PH = "<｜deepseek_image｜>"


def _img(w, h, seed):
    r = np.random.RandomState(seed)
    return Image.fromarray(r.randint(0, 255, (h, w, 3)).astype(np.uint8))


def _processor():
    import torch.nn as nn
    from vllm.config import ModelConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.interfaces import SupportsMultiModal
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import InputProcessingContext
    from vllm.tokenizers import cached_tokenizer_from_config

    from models.demos.blackhole.deepseek_v41_flash.tt.vision_vllm import (
        DSV41DummyInputsBuilder,
        DSV41MultiModalProcessor,
        DSV41ProcessingInfo,
    )

    @MULTIMODAL_REGISTRY.register_processor(
        DSV41MultiModalProcessor, info=DSV41ProcessingInfo, dummy_inputs=DSV41DummyInputsBuilder
    )
    class Stub(
        nn.Module, SupportsMultiModal
    ):  # (the real class: tt/generator_vllm.DeepseekV41ForCausalLM carries the same decorator)
        @classmethod
        def get_placeholder_str(cls, modality, i):
            return PH

        def embed_input_ids(self, input_ids, multimodal_embeddings=None, **kw):
            raise NotImplementedError

        def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kw):
            raise NotImplementedError

        def compute_logits(self, hidden_states):
            raise NotImplementedError

    ModelRegistry.register_model("DSV41StubForCausalLM", Stub)
    cfg = ModelConfig(
        model=CKPT,
        hf_config_path=CFG,
        tokenizer=CKPT,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        hf_overrides={"architectures": ["DSV41StubForCausalLM"]},
    )
    ctx = InputProcessingContext(model_config=cfg, tokenizer=cached_tokenizer_from_config(cfg))
    info = DSV41ProcessingInfo(ctx)
    return info, DSV41MultiModalProcessor(info, DSV41DummyInputsBuilder(info), cache=None)


def test_processor_expands_placeholders():
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.processor import TimingContext

    from models.demos.blackhole.deepseek_v41_flash.tt import vision_prompt as VP
    from models.demos.blackhole.deepseek_v41_flash.tt.vision_args import VisionArgs

    info, proc = _processor()
    va = VisionArgs()

    def run(prompt, imgs):
        items = info.get_data_parser().parse_mm_data({"image": imgs} if imgs else {})
        return proc.apply(ProcessorInputs(prompt, items), TimingContext(enabled=False))

    t = run("<｜User｜>hello<｜Assistant｜></think>", [])
    assert not t["mm_placeholders"] and va.image_token_id not in t["prompt_token_ids"]
    sizes = [(640, 480), (300, 900)]
    t = run(f"<｜User｜>{PH}\n\n{PH}\n\ncompare<｜Assistant｜>", [_img(w, h, i) for i, (w, h) in enumerate(sizes)])
    ids, ph = t["prompt_token_ids"], t["mm_placeholders"]["image"]
    assert [p.length for p in ph] == [VP.image_span_len(w, h, va) for w, h in sizes]
    for p in ph:
        assert all(i == va.image_token_id for i in ids[p.offset : p.offset + p.length])
    assert sum(i == va.image_token_id for i in ids) == sum(p.length for p in ph)
    items = t["mm_kwargs"]["image"]
    assert len(items) == 2 and all("pixel_values" in it and "image_grid_thw" in it for it in items)
    for it, (w, h) in zip(items, sizes):
        _, n_h, n_w = [int(x) for x in it["image_grid_thw"].data]
        assert it["pixel_values"].data.shape == (n_h * n_w, 3 * 14 * 14)


def test_chat_template_matches_encoding():
    import jinja2
    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.join(CKPT, "encoding"))
    from encoding import encode_messages

    tok = AutoTokenizer.from_pretrained(CKPT)
    tpl = jinja2.Environment().from_string(open(os.path.join(CFG, "chat_template.jinja")).read())
    img = lambda u: {"type": "image_url", "image_url": {"url": u}}
    cases = [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "again"},
        ],
        [{"role": "user", "content": [img("x.png"), {"type": "text", "text": "what is this?"}]}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    img("x.png"),
                    {"type": "text", "text": "and"},
                    img("y.png"),
                ],
            }
        ],
    ]
    for m in cases:
        want = encode_messages(json.loads(json.dumps(m)), thinking_mode="chat")
        got = tpl.render(messages=m, bos_token=tok.bos_token, eos_token=tok.eos_token, add_generation_prompt=True)
        assert got == want, (want, got)
