# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the Kimi-K3 media module (`service/media.py`) and its stored-value format, against the HF processor
copied into `reference/`. The device side, load and encode, is `test_media_encoder.py`."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import CHECKPOINT_ENV
from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor
from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media, _stub_rows, tokenizer_source
from models.experimental.kimi_k3_moonvit_v2.tests.common import MEDIA_CONFIG, MEDIA_PAD, random_array
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import pixels_to_patches
from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import check_output_rows

REFERENCE_DIR = Path(__file__).parents[1] / "reference"
TEXT_HIDDEN = 7168


@pytest.fixture(scope="module")
def processor() -> KimiK3VisionProcessor:
    cfg = json.loads((REFERENCE_DIR / "preprocessor_config.json").read_text())["media_proc_cfg"]
    return KimiK3VisionProcessor(media_proc_cfg=cfg)


# (height, width): unresized, odd sizes the padding rounds up, and resized by the side limit (7168 px).
@pytest.mark.parametrize("hw", [(1024, 1024), (20, 33), (48, 49), (1080, 1920), (300, 8000), (1, 1)])
def test_prepare_matches_hf(processor, hw):
    media = KimiK3Media(MEDIA_CONFIG)
    array = random_array(*hw)
    item = media.prepare("image", array)

    # The item holds uint8 pixels; encode makes the patches from them.
    assert item.pixels.dtype == np.uint8
    pixel_values, grid_thws = pixels_to_patches(item.pixels, media.config)
    want = processor.preprocess([{"type": "image", "image": Image.fromarray(array)}])
    assert torch.equal(pixel_values, want["pixel_values"])
    assert grid_thws.tolist() == want["grid_thws"].tolist()
    _, h, w = grid_thws[0].tolist()
    assert (item.grid_h, item.grid_w) == (h // 2, w // 2)
    assert item.n_tokens == h * w // 4
    assert media.placeholder_token_ids == {"image": MEDIA_PAD}
    assert (item.height, item.width) == hw
    assert media.stored_nbytes(item) == 16 + item.n_tokens * TEXT_HIDDEN * 2


def test_expansion_matches_hf_prompt():
    """The frontend renders the chat with one <|media_pad|> per image, and expansion replaces each: that must give
    the ids HF gets from the same chat with its processor's image prompts, each <|media_pad|> widened to the image's
    embeddings as the model does."""
    media = KimiK3Media(MEDIA_CONFIG)
    tokenizer = media.tokenizer
    # (height, width): unresized, padded, resized by the side limit, and sizes whose digits cross 3-digit groups.
    items = [media.prepare("image", random_array(*hw)) for hw in [(1080, 1920), (20, 33), (300, 8000), (1, 1)]]
    content = [{"type": "image_url", "image_url": {"url": f"file://{i}.png"}} for i in range(len(items))]
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": "What is in these images?"}]}]

    hf_prompts = [KimiK3VisionProcessor.make_image_prompt(item.width, item.height) for item in items]
    hf_ids, widths = [], iter(item.n_tokens for item in items)
    for token in tokenizer.apply_chat_template(messages, image_prompts=hf_prompts, tokenize=True):
        hf_ids += [token] * (next(widths) if token == MEDIA_PAD else 1)

    ours, runs = [], iter(media.expansion(item) for item in items)
    pads = ["<|media_pad|>"] * len(items)
    for token in tokenizer.apply_chat_template(messages, image_prompts=pads, tokenize=True):
        ours += [t for t, n in next(runs) for _ in range(n)] if token == MEDIA_PAD else [token]
    assert ours == hf_ids


def test_tokenizer_source(monkeypatch):
    """The local checkpoint folder when $KIMI_K3_HF_MODEL names one, else the Hugging Face repo."""
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    assert tokenizer_source() == "moonshotai/Kimi-K3"
    monkeypatch.setenv(CHECKPOINT_ENV, "/weights/Kimi-K3")
    assert tokenizer_source() == "/weights/Kimi-K3"


def test_constructor_loads_the_tokenizer(expect_error):
    """The constructor loads the tokenizer, once per process, and checks it against config.json's placeholder."""
    media = KimiK3Media(MEDIA_CONFIG)
    assert media.tokenizer is KimiK3Media(MEDIA_CONFIG).tokenizer
    assert media.tokenizer.convert_tokens_to_ids("<|media_pad|>") == MEDIA_PAD
    with expect_error(ValueError, "to 163605, config.json's media_placeholder_token_id is 163606"):
        KimiK3Media({**MEDIA_CONFIG, "media_placeholder_token_id": MEDIA_PAD + 1})


@pytest.mark.parametrize(
    "kind,array,message",
    [
        ("video", random_array(28, 28), "does not accept video"),
        ("image", np.zeros((28, 28, 4), np.uint8), "RGB"),
        ("image", np.zeros((28, 28, 3), np.float32), "uint8"),
        ("image", np.zeros((28, 28), np.uint8), "uint8"),
        ("image", random_array(0, 28), "non-empty"),
        # 258 x 258 patches after the 28 px padding: over 65,536, rejected rather than shrunk.
        ("image", np.zeros((3590, 3590, 3), np.uint8), "image 3590x3590: grid 258x258 exceeds"),
    ],
)
def test_prepare_rejects(expect_error, kind, array, message):
    with expect_error(ValueError, message):
        KimiK3Media(MEDIA_CONFIG).prepare(kind, array)


def test_prepare_takes_read_only_and_flipped_images():
    """prepare takes a read-only image, and a flipped or channel-swapped view like its copy."""
    media = KimiK3Media(MEDIA_CONFIG)
    image = random_array(100, 60)
    read_only = image.copy()
    read_only.flags.writeable = False
    assert np.array_equal(media.prepare("image", read_only).pixels, media.prepare("image", image).pixels)
    for view in (image[::-1], image[:, ::-1], image[..., ::-1]):
        assert np.array_equal(media.prepare("image", view).pixels, media.prepare("image", view.copy()).pixels)


def test_image_pickles_out_of_band():
    """The service sends items between processes: the pixels go out of band, so the stream stays small."""
    item = KimiK3Media(MEDIA_CONFIG).prepare("image", random_array(672, 896))
    buffers = []
    stream = pickle.dumps(item, protocol=5, buffer_callback=buffers.append)
    assert len(stream) < 4096
    back = pickle.loads(stream, buffers=buffers)
    assert (back.grid_h, back.grid_w) == (item.grid_h, item.grid_w)
    assert np.array_equal(back.pixels, item.pixels)


def test_prepare_does_not_import_ttnn():
    """The constructor, prepare and stored_nbytes run where the service prepares images, which may have no ttnn
    (expansion too; the constructor imports only transformers and the tokenizer). A fresh interpreter, since this one
    has ttnn imported already."""
    code = (
        "import sys, numpy as np\n"
        "from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media\n"
        f"media = KimiK3Media({MEDIA_CONFIG!r})\n"
        "item = media.prepare('image', np.zeros((30, 40, 3), np.uint8))\n"
        "media.stored_nbytes(item)\n"
        "assert 'ttnn' not in sys.modules, 'ttnn imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[4])


def test_device_params():
    import ttnn

    params = KimiK3Media(MEDIA_CONFIG).device_params
    assert params["fabric_config"] == ttnn.FabricConfig.FABRIC_2D_TORUS_X
    assert params["trace_region_size"] > 0


def test_encode_before_load(expect_error):
    media = KimiK3Media(MEDIA_CONFIG)
    item = media.prepare("image", random_array(28, 28))
    with expect_error(RuntimeError, "encode before load"):
        media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])
    media.close()  # without a load: a no-op


def test_format_rejects_bad_values(expect_error):
    assert media_format.stored_nbytes(6, 64) == 16 + 6 * 64 * 2
    buf = memoryview(bytearray(media_format.stored_nbytes(6, 64)))
    media_format.Header("image").write(buf)
    assert media_format.Header.read(buf) == media_format.Header("image")
    rows = media_format.embeddings(buf, 64)
    assert rows.shape == (6, 64)
    rows[:] = 0xFFFF
    assert bytes(buf[:16]) == bytes(16)  # the rows start after the header
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:-2], 64)
    with expect_error(ValueError, "whole bf16 rows"):
        media_format.embeddings(buf[:16], 64)
    with expect_error(ValueError, "shorter than its 16 B header"):
        media_format.Header.read(buf[:15])
    buf[0] = len(media_format.KINDS)
    with expect_error(ValueError, "unknown media kind"):
        media_format.Header.read(buf)


@pytest.mark.parametrize("dtype", ["uint16", "int16", "bfloat16"])
def test_output_rows_take_bf16_holders(dtype):
    """read_output copies into an array of any dtype that holds bf16 bits, as format.embeddings returns uint16."""
    if dtype == "bfloat16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    check_output_rows(np.zeros((3, 8), dtype))


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@pytest.mark.parametrize(
    "rows,error,message",
    [
        (torch.zeros(3, 8, dtype=torch.int16), TypeError, "numpy array of uint16, int16, bfloat16"),
        (np.zeros((3, 8), np.float32), TypeError, "got float32"),
        (np.zeros((3, 8, 1), np.uint16), ValueError, "shape \\(3, 8, 1\\)"),
        (np.zeros((0, 8), np.uint16), ValueError, "logical_n > 0"),
        (np.zeros((3, 16), np.uint16)[:, ::2], ValueError, "C-contiguous False"),
        (_read_only(np.zeros((3, 8), np.uint16)), ValueError, "writable False"),
    ],
    ids=["torch", "float32", "3d", "empty", "strided", "read_only"],
)
def test_output_rows_rejects(expect_error, rows, error, message):
    """read_output raises for these before it reads the chips."""
    with expect_error(error, message):
        check_output_rows(rows)


def stub_media() -> KimiK3Media:
    media = KimiK3Media(MEDIA_CONFIG)
    media.load(None, "", "image", {"stub_embeddings": True})
    return media


def slot(media: KimiK3Media, item, extra: int = 0) -> memoryview:
    return memoryview(bytearray(media.stored_nbytes(item) + extra))


def test_stub_encode_is_deterministic():
    """With stub_embeddings, load and encode use no chip: the stored value is a header and seeded rows."""
    items = [KimiK3Media(MEDIA_CONFIG).prepare("image", random_array(56, 84, seed=s)) for s in (0, 1)]
    values = []
    for _ in range(2):
        media = stub_media()
        bufs = [slot(media, item) for item in items]
        media.encode(items, bufs)
        media.close()
        values.append([bytes(buf) for buf in bufs])
    assert values[0] == values[1]
    first, second = values[0]
    assert first != second  # seeded by the image
    assert media_format.Header.read(memoryview(first)).kind == "image"
    rows = torch.from_numpy(media_format.embeddings(memoryview(first), TEXT_HIDDEN).copy()).view(torch.bfloat16)
    assert rows.shape == (items[0].n_tokens, TEXT_HIDDEN)
    assert torch.isfinite(rows).all() and rows.abs().max() > 0


def test_encode_checks_every_slot_first(expect_error):
    """A slot of the wrong size, read-only or not C-contiguous raises before any slot is written."""
    media = stub_media()
    items = [media.prepare("image", random_array(56, 84, seed=s)) for s in (0, 1)]
    good, size = slot(media, items[0]), media.stored_nbytes(items[1])
    bad_slots = (
        ("a slot of", slot(media, items[1], extra=2)),
        ("read-only", memoryview(bytes(size))),
        ("not C-contiguous", memoryview(bytearray(2 * size))[::2]),  # the right size, every other byte
    )
    for message, bad in bad_slots:
        with expect_error(ValueError, message):
            media.encode(items, [good, bad])
        assert bytes(good) == bytes(len(good)), "encode wrote into a slot before it checked them all"
    bufs = [good, slot(media, items[1])]
    with expect_error(ValueError, "2 items for 1 slots"):
        media.encode(items, bufs[:1])
    media.close()


def test_encode_takes_slots_of_any_format():
    """A slot is cast to bytes: a uint16 memoryview gets the same value as a byte one."""
    media = stub_media()
    item = media.prepare("image", random_array(56, 84))
    as_bytes, as_uint16 = slot(media, item), slot(media, item).cast("H")
    media.encode([item, item], [as_bytes, as_uint16])
    assert bytes(as_bytes) == as_uint16.tobytes()
    media.close()


def test_load_twice(expect_error):
    media = stub_media()
    with expect_error(RuntimeError, "load after load: close first"):
        media.load(None, "", "image", {"stub_embeddings": True})
    media.close()
    media.load(None, "", "image", {"stub_embeddings": True})  # after close, a load is fine
    media.close()
    item = media.prepare("image", random_array(28, 28))
    with expect_error(RuntimeError, "after close"):
        media.encode([item], [slot(media, item)])


@pytest.mark.parametrize(
    "weights_path,mesh,options,message",
    [
        ("", object(), {}, "weights_path must be a Kimi-K3 checkpoint folder"),
        ("/nonexistent/k3", object(), {}, "weights_path must be a Kimi-K3 checkpoint folder"),
        (".", None, {}, "mesh_device is None"),
        (".", object(), {"cache_dir": 3}, "cache_dir must be a folder path"),
        (".", object(), {"stub_embeddings": "yes"}, "stub_embeddings must be a bool"),
    ],
)
def test_load_rejects_before_ttnn(expect_error, weights_path, mesh, options, message):
    """Bad inputs raise before load touches ttnn or the chips."""
    media = KimiK3Media(MEDIA_CONFIG)
    with expect_error(ValueError, message):
        media.load(mesh, weights_path, "image", options)
    with expect_error(ValueError, "does not accept video"):
        media.load(mesh, weights_path, "video", options)


def test_stub_does_not_import_ttnn():
    """A stub load and encode, as the service runs them without a ring, import no ttnn."""
    code = (
        "import sys, numpy as np\n"
        "from models.experimental.kimi_k3_moonvit_v2.service.media import KimiK3Media\n"
        f"media = KimiK3Media({MEDIA_CONFIG!r})\n"
        "media.load(None, '', 'image', {'stub_embeddings': True})\n"
        "item = media.prepare('image', np.zeros((30, 40, 3), np.uint8))\n"
        "media.encode([item], [memoryview(bytearray(media.stored_nbytes(item)))])\n"
        "media.close()\n"
        "assert 'ttnn' not in sys.modules, 'ttnn imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[4])


def test_encode_writes_every_item_of_a_call():
    """An encode call of two images writes each slot in full, header and rows, as a call of that image alone does."""
    media = stub_media()
    items = [media.prepare("image", random_array(56, 84)), media.prepare("image", random_array(84, 140, seed=1))]
    together = [memoryview(bytearray(b"\xff" * media.stored_nbytes(item))) for item in items]
    media.encode(items, together)
    for item, value in zip(items, together):
        alone = memoryview(bytearray(b"\xff" * media.stored_nbytes(item)))
        media.encode([item], [alone])
        assert bytes(value) == bytes(alone)
        assert bytes(value[:16]) == bytes(16)  # the header: kind 0 ("image"), then zero padding
        assert np.array_equal(media_format.embeddings(value, TEXT_HIDDEN), _stub_rows(item, TEXT_HIDDEN))
    media.close()


def test_stub_encode_writes_a_value_prefill_reads():
    """The prefill reads a stored value from the store, read-only, with format.Header.read and format.embeddings, and
    expects one row per <|media_pad|> of the image's expansion (the media ref's num_tokens)."""
    media = stub_media()
    item = media.prepare("image", random_array(100, 60))
    buf = memoryview(bytearray(media.stored_nbytes(item)))
    media.encode([item], [buf])
    media.close()

    fetched = memoryview(bytes(buf))  # as the store hands it back
    assert media_format.Header.read(fetched).kind == "image"
    rows = media_format.embeddings(fetched, TEXT_HIDDEN)
    (pads,) = [n for token, n in media.expansion(item) if token == MEDIA_PAD]
    assert rows.shape == (pads, TEXT_HIDDEN) == (item.n_tokens, TEXT_HIDDEN)
    assert np.array_equal(rows, _stub_rows(item, TEXT_HIDDEN))
