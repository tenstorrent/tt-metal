# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The image request path on the host: the protocol's image parts and refusals, the processor geometry
(``smart_resize``, grids, the ``detail`` cap), decoding and digests.  No device, no checkpoint."""

import base64
import hashlib

import pytest

from models.demos.blackhole.qwen38_flash_next.mrope import IMAGE_TOKEN_ID, Qwen38ImageGrid
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vision_inputs as inputs
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import Qwen38ChatRequestRejected

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _data_url(data: bytes, media: str = "image/png") -> str:
    return f"data:{media};base64,{base64.b64encode(data).decode()}"


def _user(*parts):
    return [{"role": "user", "content": list(parts)}]


# -- the processor geometry ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, resized, tokens",
    [
        ((100, 100), (256, 256), 64),  # under the minimum: scaled up
        ((480, 640), (480, 640), 300),
        ((768, 1024), (768, 1024), 768),
        ((1024, 1024), (1024, 1024), 1024),
        ((1080, 1920), (1088, 1920), 2040),
        ((1536, 2048), (1536, 2048), 3072),
        ((2160, 3840), (2176, 3840), 8160),
        ((4096, 4096), (4096, 4096), 16384),  # the stock maximum
        ((4320, 7680), (3072, 5440), 16320),  # over the maximum: scaled down
        ((300, 50), (640, 128), 80),
        ((1365, 2048), (1376, 2048), 2752),  # the tree's demo.jpeg
    ],
)
def test_smart_resize_matches_the_processor_table(size, resized, tokens):
    height, width = size
    assert inputs.smart_resize(height, width) == resized
    grid = inputs.image_grid(height, width)
    assert grid.as_tuple() == (1, resized[0] // 16, resized[1] // 16)
    assert grid.merged_tokens == tokens


def test_low_detail_caps_the_pixels_and_the_tokens(expect_error):
    assert inputs.image_grid(1024, 1024, "low") == Qwen38ImageGrid(1, 32, 32)
    assert inputs.image_grid(1024, 1024, "low").merged_tokens == 256
    assert inputs.image_grid(4320, 7680, "low").merged_tokens <= 256
    assert inputs.image_grid(1024, 1024, "high") == inputs.image_grid(1024, 1024, "auto")
    with expect_error(inputs.Qwen38ImageError, match="detail"):
        inputs.image_grid(64, 64, "medium")
    with expect_error(inputs.Qwen38ImageError, match="aspect ratio"):
        inputs.smart_resize(10, 4000)
    with expect_error(inputs.Qwen38ImageError, match="positive"):
        inputs.smart_resize(0, 10)


def test_decode_image_and_the_request_digest(expect_error):
    decoded = inputs.decode_image(PNG_1X1)
    assert (decoded.width, decoded.height) == (1, 1) and decoded.detail == "auto"
    assert decoded.grid == Qwen38ImageGrid(1, 16, 16) and decoded.merged_tokens == 64  # 1x1 scales to 256x256
    assert len(decoded.sha256) == 64
    low = inputs.decode_image(PNG_1X1, "low")
    assert inputs.request_digest([decoded]) != inputs.request_digest([low])  # the detail is part of the key
    assert inputs.request_digest([decoded, low]) != inputs.request_digest([low, decoded])  # and the order
    with expect_error(inputs.Qwen38ImageError, match="decoded"):
        inputs.decode_image(b"not an image")


# -- the protocol ------------------------------------------------------------------------------------------------


def test_image_parts_are_collected_in_prompt_order_and_rendered_as_image_items():
    images: list[protocol.Qwen38ImagePart] = []
    messages = _user(
        {"type": "text", "text": "Compare:"},
        {"type": "image_url", "image_url": {"url": _data_url(PNG_1X1), "detail": "low"}},
        {"type": "image_url", "image_url": _data_url(PNG_1X1, "image/jpeg")},  # the string shorthand
    )
    normalized = protocol.normalize_messages(messages, images=images)
    assert normalized[0]["content"] == [{"type": "text", "text": "Compare:"}, {"type": "image"}, {"type": "image"}]
    assert [(image.detail, image.media_type, image.part_index) for image in images] == [
        ("low", "image/png", 1),
        ("auto", "image/jpeg", 2),
    ]
    assert images[0].data == PNG_1X1 and images[0].message_index == 0


@pytest.mark.parametrize(
    "part, fragment",
    [
        ({"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}, "data: URL"),
        ({"type": "image_url", "image_url": {"url": "data:text/plain;base64,aGk="}}, "must be data:<image"),
        ({"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}, "not valid base64"),
        ({"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}, "empty image"),
        ({"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk=", "detail": "medium"}}, "detail"),
        ({"type": "image_url", "image_url": 5}, "image_url must be"),
        ({"type": "video", "video": "x"}, "video parts"),
        ({"type": "video_url", "video_url": {"url": "x"}}, "video parts"),
        ({"type": "input_video", "input_video": {"data": "x"}}, "video parts"),
        ({"type": "text", "text": "t", "video": "x"}, "video parts"),
        ({"type": "input_audio", "input_audio": {}}, "must be 'text' or 'image_url'"),
        ("plain", "content part object"),
    ],
)
def test_refused_parts(part, fragment):
    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        Qwen38ChatRequestRejected, match=fragment
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        protocol.normalize_messages(_user(part), images=[])
    assert caught.value.param == "messages[0].content[0]" or caught.value.param.startswith("messages[0].content[0].")


def test_images_are_refused_without_a_collector_and_outside_user_messages(expect_error):
    part = {"type": "image_url", "image_url": {"url": _data_url(PNG_1X1)}}
    with expect_error(Qwen38ChatRequestRejected, match="text-only"):
        protocol.normalize_messages(_user(part))
    with expect_error(Qwen38ChatRequestRejected, match="belong to user messages"):
        protocol.normalize_messages([{"role": "system", "content": [part]}], images=[])
    with expect_error(Qwen38ChatRequestRejected, match="belong to user messages"):
        protocol.normalize_messages(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": [part]}], images=[]
        )
    assert protocol.MAX_IMAGE_BYTES == 16 << 20 and protocol.IMAGE_DETAILS == ("auto", "low", "high")


class _FakeTokenizer:
    """The template's render as a fixed string; ids are the characters' ordinals with the special tokens mapped."""

    SPECIAL = {"<|image_pad|>": IMAGE_TOKEN_ID, "<|vision_start|>": 248_053, "<|vision_end|>": 248_054}

    def apply_chat_template(self, messages, **kwargs):
        text = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                text += content
            else:
                for item in content:
                    text += item["text"] if item["type"] == "text" else "<|vision_start|><|image_pad|><|vision_end|>"
        return text + ("<think>\n" if kwargs["enable_thinking"] else "<think>\n\n</think>\n\n")

    def __call__(self, text, add_special_tokens=False):
        ids = []
        while text:
            for token, value in self.SPECIAL.items():
                if text.startswith(token):
                    ids.append(value)
                    text = text[len(token) :]
                    break
            else:
                ids.append(ord(text[0]) % 1000 + 1)
                text = text[1:]
        return type("Encoded", (), {"input_ids": ids})()


def test_render_prompt_expands_the_pads_by_the_grids(expect_error):
    tokenizer = _FakeTokenizer()
    grid = Qwen38ImageGrid(1, 4, 6)  # 6 merged tokens
    messages = _user({"type": "image_url", "image_url": {"url": _data_url(PNG_1X1)}}, {"type": "text", "text": "Hi"})
    images: list[protocol.Qwen38ImagePart] = []
    ids = protocol.render_prompt(
        tokenizer, messages, None, enable_thinking=False, reasoning_effort="medium", images=images, image_grids=[grid]
    )
    assert len(images) == 1 and ids.count(IMAGE_TOKEN_ID) == 6
    assert ids.index(248_053) + 1 == ids.index(IMAGE_TOKEN_ID) and ids[ids.index(IMAGE_TOKEN_ID) + 6] == 248_054
    with expect_error(protocol.Qwen38ChatFormatError, match="1 image parts but 0 image grids"):
        protocol.render_prompt(tokenizer, messages, None, enable_thinking=False, reasoning_effort="medium", images=[])
    text_only = protocol.render_prompt(
        tokenizer, _user({"type": "text", "text": "Hi"}), None, enable_thinking=False, reasoning_effort="medium"
    )
    assert IMAGE_TOKEN_ID not in text_only


def test_image_digests_key_each_image_and_the_request_digest_is_their_hash() -> None:
    from types import SimpleNamespace

    grid = Qwen38ImageGrid(1, 32, 32)
    a = SimpleNamespace(sha256="aa", detail="auto", grid=grid)
    b = SimpleNamespace(sha256="bb", detail="low", grid=grid)
    keys = inputs.image_digests([a, b])
    assert keys == ("aa:auto:(1, 32, 32)", "bb:low:(1, 32, 32)")
    assert inputs.request_digest([a, b]) == hashlib.sha256("".join(f"{k};" for k in keys).encode()).hexdigest()
    # two images of one grid render identical pad ids: the bytes' digest in the key is what tells them apart
    other = SimpleNamespace(sha256="cc", detail="auto", grid=grid)
    assert (
        inputs.image_digests([a]) != inputs.image_digests([other])
        and inputs.request_digest([]) == hashlib.sha256().hexdigest()
    )
