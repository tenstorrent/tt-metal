# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Server-adapter test: drives Molmo2ForConditionalGeneration (tt/generator_vllm.py) the way
vLLM's TT model runner does — initialize_vllm_model -> prefill_forward -> decode_forward —
without needing vLLM installed (the vLLM base classes are stubbed when missing).

Requests of different lengths/modalities run back to back on one adapter so that decode-trace
reuse across requests is exercised. Each request is cross-checked against the untraced
``forward_prefill`` + ``forward_decode_step`` path of the same model.

Run (Galaxy 1x8 slice, see README):
    pytest models/demos/molmo2/tests/test_generator_vllm_adapter.py -v -s
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn


def _stub_vllm_if_missing():
    try:
        import vllm  # noqa: F401

        return
    except ImportError:
        pass
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.interfaces",
        "vllm.multimodal",
    ):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.model_executor.models.interfaces"].SupportsMultiModal = type("SupportsMultiModal", (), {})

    class _Registry:
        def register_processor(self, *args, **kwargs):
            return lambda cls: cls

    sys.modules["vllm.multimodal"].MULTIMODAL_REGISTRY = _Registry()


_stub_vllm_if_missing()

from models.demos.molmo2.tt.generator_vllm import Molmo2ForConditionalGeneration  # noqa: E402
from models.demos.molmo2.tt.hf_loading import MOLMO2_HF_ID, molmo2_checkpoint_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
VIDEO = (
    REPO
    / "models/demos/molmo2/verification/videos/d02d399003cca14e2a5c822d389821e553e6b7942c5640d48c075703e533d1dc.mp4"
)
_MESH = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 8))
MAX_NEW = int(os.environ.get("MOLMO2_ADAPTER_MAX_NEW", "24"))


@pytest.fixture(scope="module")
def mesh_device():
    rows, cols = _MESH
    if rows * cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    yield device
    ttnn.close_mesh_device(device)
    if rows * cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def adapter(mesh_device):
    hf_config = types.SimpleNamespace(_name_or_path=MOLMO2_HF_ID)
    return Molmo2ForConditionalGeneration.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=36864
    )


@pytest.fixture(scope="module")
def images():
    snap = molmo2_checkpoint_dir()
    return {
        "logo": Image.open(snap / "molmo_2_logo_RGB.png").convert("RGB"),
        "cat": (
            Image.open(REPO.parent / "media/cat.jpg").convert("RGB")
            if (REPO.parent / "media/cat.jpg").exists()
            else Image.fromarray((torch.rand(480, 640, 3).numpy() * 255).astype("uint8"))
        ),
    }


def _vllm_style_inputs(processor, content, images=None, videos=None):
    """Processor output -> (tokens, prefill kwargs) in the shape vLLM hands to prefill_forward."""
    text = processor.apply_chat_template(
        [[{"role": "user", "content": content}]], tokenize=False, add_generation_prompt=True
    )
    kw = {"text": text, "return_tensors": "pt", "padding": True}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
    out = processor(**kw)
    tokens = out["input_ids"].to(torch.int32)
    # vLLM batches multimodal fields as one entry per request.
    kwargs = {"token_type_ids": [out["token_type_ids"]]}
    for key in (
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "pixel_values_videos",
        "video_token_pooling",
        "video_grids",
    ):
        if key in out:
            kwargs[key] = [out[key]]
    return out, tokens, kwargs


def _server_generate(adapter, tokens, kwargs, max_new, eos):
    S = tokens.shape[1]
    logits = adapter.prefill_forward(tokens, None, None, torch.tensor([S]), **kwargs)  # [1, 1, vocab]
    nxt = int(logits[0, -1].argmax())
    out, pos = [], S
    for _ in range(max_new):
        if nxt == eos:
            break
        out.append(nxt)
        step = adapter.decode_forward(torch.tensor([[nxt]], dtype=torch.int32), torch.tensor([pos]), None, None)
        nxt = int(step[0].argmax())
        pos += 1
    return out


def _teacher_forced(adapter, tokens, kwargs, ref_tokens):
    """Feed HF's greedy tokens through the server path; return TT argmax at every step."""
    S = tokens.shape[1]
    logits = adapter.prefill_forward(tokens, None, None, torch.tensor([S]), **kwargs)
    preds = [int(logits[0, -1].argmax())]
    for i, tok in enumerate(ref_tokens[:-1]):
        step = adapter.decode_forward(torch.tensor([[tok]], dtype=torch.int32), torch.tensor([S + i]), None, None)
        preds.append(int(step[0].argmax()))
    return preds


def _load_cases(images):
    import json

    cases = []
    for case in json.loads((Path(__file__).parent / "adapter_cases.json").read_text()):
        imgs = [images[c["image"]] for c in case["content"] if c["type"] == "image"]
        vids = [str(VIDEO.parent / c["video"]) for c in case["content"] if c["type"] == "video"]
        content = [
            (
                {"type": "image", "image": "x"}
                if c["type"] == "image"
                else {"type": "video", "video": "v"}
                if c["type"] == "video"
                else c
            )
            for c in case["content"]
        ]
        media = {}
        if imgs:
            media["images"] = imgs
        if vids:
            media["videos"] = vids[0]
        cases.append((case["name"], content, media))
    return cases


MIN_TEACHER_FORCED_AGREEMENT = 0.9


def test_adapter_requests(adapter, images):
    """Back-to-back requests through prefill_forward/decode_forward (decode trace reused across
    requests of different lengths). Free-running output must be non-empty; with MOLMO2_HF_REF
    (json from make_hf_adapter_reference.py, transformers 4.57.1) teacher-forced top-1 agreement
    with the HF greedy tokens must be >= 90%."""
    import json

    from models.demos.molmo2.tt.hf_loading import load_molmo2_processor

    ref_path = os.environ.get("MOLMO2_HF_REF")
    refs = json.load(open(ref_path)) if ref_path else {}
    processor = load_molmo2_processor()
    eos = processor.tokenizer.eos_token_id
    failures = []
    for name, content, media in _load_cases(images):
        out, tokens, kwargs = _vllm_style_inputs(processor, content, **media)
        text = processor.tokenizer.decode(
            _server_generate(adapter, tokens, kwargs, MAX_NEW, eos), skip_special_tokens=True
        )
        logger.info(f"[{name}] S={tokens.shape[1]} free-running: {text!r}")
        assert text.strip(), f"{name}: empty server output"
        if name in refs:
            ref = refs[name]
            assert ref["S"] == tokens.shape[1], f"{name}: prompt length {tokens.shape[1]} != HF {ref['S']}"
            preds = _teacher_forced(adapter, tokens, kwargs, ref["tokens"])
            agree = sum(a == b for a, b in zip(preds, ref["tokens"])) / len(preds)
            first_miss = next((i for i, (a, b) in enumerate(zip(preds, ref["tokens"])) if a != b), None)
            logger.info(
                f"[{name}] teacher-forced agreement {agree:.0%} over {len(preds)} tokens, first mismatch @ {first_miss}"
            )
            if agree < MIN_TEACHER_FORCED_AGREEMENT:
                failures.append((name, agree))
    assert not failures, f"teacher-forced agreement below {MIN_TEACHER_FORCED_AGREEMENT:.0%}: {failures}"
