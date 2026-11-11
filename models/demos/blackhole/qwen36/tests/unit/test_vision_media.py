# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 (FLUX.3 prompt upsampler) media module and stored-value format, host only.

The reference is the upsampler's own request path: the FLUX.3 reference-image policy (flux3/pup.py, below verbatim)
on a PIL image, then the HF Qwen2VL image processor with the upsampler checkpoint's preprocessor_config.json. The
token test also needs the checkpoint's tokenizer (QWEN35_PUP_SNAPSHOT = its directory) and skips without it.
"""

import io
import math
import os

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.vision import format as media_format
from models.demos.blackhole.qwen36.vision.media import VISION_END, VISION_START, Qwen35Image, Qwen35Media, pixel_values

IMAGE_PAD = 248056
CONFIG = {
    "image_token_id": IMAGE_PAD,
    "video_token_id": 248057,
    "vision_start_token_id": VISION_START,
    "vision_end_token_id": VISION_END,
    "vision_config": {
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 64,
        "deepstack_visual_indexes": [],
    },
}
PREPROCESSOR = dict(  # the upsampler checkpoint's preprocessor_config.json
    do_convert_rgb=True,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    merge_size=2,
    patch_size=16,
    resample=3,
    rescale_factor=1 / 255,
    size={"longest_edge": 16777216, "shortest_edge": 65536},
    temporal_patch_size=2,
)

# ---- flux3/pup.py (black-forest-labs/flux-3-image-klein e53f4ea, Apache-2.0): the reference-image policy, verbatim
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, 11)}}
JPEG_QUALITY = 92


def bfl_reference_size(width, height, n_refs):
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def bfl_policy_reference(image, n_refs):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = bfl_reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


# ----


def rgb(h, w, seed=0):
    """A smooth image plus noise (pure noise would make the JPEG round trip the only thing tested)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([x / max(w, 1), y / max(h, 1), (x + y) / max(h + w, 1)], axis=-1) * 200
    return np.clip(base + rng.normal(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor(**PREPROCESSOR)


def hf_reference(processor, images, n_refs):
    from PIL import Image

    refs = [bfl_policy_reference(Image.fromarray(im), n_refs) for im in images]
    return processor(images=refs, return_tensors="pt")


@pytest.mark.parametrize(
    "sizes",
    [
        [(1024, 1024)],  # 1 ref, already at the cap
        [(2048, 1536)],  # 1 ref, downscaled to the 1 MP cap, non-square grid
        [(877, 1531)],  # 1 ref, odd size
        [(1000, 1000), (640, 900)],  # 2 refs: 724^2 cap
        [(1024, 1024)] * 10,  # 10 refs: 512^2 each
        [(512, 512), (300, 700), (1200, 800)],  # 3 refs, mixed
        [(100, 120)],  # under 256^2 after the policy: HF's own bicubic upscale
        [(40, 2000)],  # tall strip
    ],
)
def test_prepare_matches_hf(processor, sizes):
    media = Qwen35Media(CONFIG)
    images = [rgb(h, w, seed=i) for i, (h, w) in enumerate(sizes)]
    items = [media.prepare("image", im, n_refs=len(images)) for im in images]
    want = hf_reference(processor, images, len(images))
    assert want["image_grid_thw"].tolist() == [list(it.grid_thw) for it in items]
    got = torch.cat([pixel_values(it.patches) for it in items])
    assert torch.equal(got, want["pixel_values"])
    got16 = torch.cat([pixel_values(it.patches, torch.bfloat16) for it in items])
    assert torch.equal(got16.view(torch.int16), want["pixel_values"].to(torch.bfloat16).view(torch.int16))
    for it in items:
        assert media.expansion(it) == [(IMAGE_PAD, it.grid_h * it.grid_w // 4)]
        assert media.value_size(it) == 64 + it.n_tokens * 64 * 2  # header + bf16 rows


def test_pixel_caps_by_ref_count():
    media = Qwen35Media(CONFIG)
    image = rgb(2048, 2048)
    tokens = {n: media.prepare("image", image, n_refs=n).n_tokens for n in (1, 2, 3, 10)}
    assert tokens == {1: 1024, 2: 484, 3: 256, 10: 256}  # 1024^2, 704^2, 512^2, 512^2
    salts = {n: Qwen35Media.media_key_salt(media.prepare("image", image, n_refs=n)) for n in (1, 2, 3, 10)}
    assert len({salts[1], salts[2], salts[3]}) == 3 and salts[3] == salts[10]


@pytest.mark.skipif(not os.environ.get("QWEN35_PUP_SNAPSHOT"), reason="needs the upsampler checkpoint's tokenizer")
@pytest.mark.parametrize("n_refs", [1, 2, 10])
def test_expansion_matches_hf_processor_ids(n_refs):
    """The chat template's marker replaced by the expansion == the HF processor's input_ids."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(os.environ["QWEN35_PUP_SNAPSHOT"], local_files_only=True)
    media = Qwen35Media(CONFIG)
    images = [rgb(900 + 37 * i, 1200 - 41 * i, seed=i) for i in range(n_refs)]
    messages = [
        {"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": "a watercolor"}]}
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    refs = [bfl_policy_reference(Image.fromarray(im), n_refs) for im in images]
    want = proc(text=text, images=refs, return_tensors="pt")["input_ids"][0].tolist()

    template_ids = proc.tokenizer(text, add_special_tokens=False)["input_ids"]
    items = [media.prepare("image", im, n_refs=n_refs) for im in images]
    marker = media.placeholder_token_ids["image"]
    assert template_ids.count(marker) == n_refs
    got, k = [], 0
    for t in template_ids:  # the adapter's splice
        if t == marker:
            for token, count in media.expansion(items[k]):
                got += [token] * count
            k += 1
        else:
            got.append(t)
    assert got == want
    # the template wraps each marker in vision_start / vision_end already
    for i, t in enumerate(template_ids):
        if t == marker:
            assert (template_ids[i - 1], template_ids[i + 1]) == (VISION_START, VISION_END)


@pytest.mark.parametrize(
    "kind,image,n_refs,message",
    [
        ("video", rgb(64, 64), 1, "does not accept video"),
        ("image", rgb(1, 201), 1, "aspect ratio"),
        ("image", np.zeros((64, 64, 4), np.uint8), 1, "RGB"),
        ("image", np.zeros((64, 64, 3), np.float32), 1, "RGB"),
        ("image", rgb(0, 28), 1, "non-empty"),
        ("image", rgb(64, 64), 11, "1 to 10"),
        ("image", rgb(64, 64), 0, "1 to 10"),
    ],
)
def test_prepare_rejects(expect_error, kind, image, n_refs, message):
    with expect_error(ValueError, message):
        Qwen35Media(CONFIG).prepare(kind, image, n_refs=n_refs)


def test_stub_encode_writes_a_value_prefill_reads(expect_error):
    media = Qwen35Media(CONFIG)
    with expect_error(ValueError, "does not accept video"):
        media.load(None, "", "video", {"stub_embeddings": True})
    media.load(None, "", "image", {"stub_embeddings": True})
    item = media.prepare("image", rgb(300, 500))
    bufs = [memoryview(bytearray(media.stored_nbytes(item))) for _ in range(2)]
    media.encode([item], bufs[:1])
    media.encode([item], bufs[1:])
    rows = media_format.embeddings(bufs[0], 64)
    assert rows.shape == (item.n_tokens, 64) and rows.any()
    assert bytes(bufs[0]) == bytes(bufs[1])
    with expect_error(ValueError, "rows for an item"):
        media.encode([item], [memoryview(bytearray(media_format.stored_nbytes(item.n_tokens - 1, 64)))])
    other = Qwen35Image(item.patches[::-1].copy(), item.grid_h, item.grid_w)
    buf = memoryview(bytearray(media.stored_nbytes(other)))
    media.encode([other], [buf])
    assert bytes(buf) != bytes(bufs[0])
    media.close()
    with expect_error(AssertionError, "encode before load"):
        media.encode([item], bufs[:1])


def test_format_rejects_bad_values(expect_error):
    buf = memoryview(bytearray(media_format.stored_nbytes(6, 64)))
    assert media_format.embeddings(buf, 64).shape == (6, 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:-2], 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:0], 64)
