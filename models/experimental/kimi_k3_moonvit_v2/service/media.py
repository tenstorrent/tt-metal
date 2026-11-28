# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 media module for the encoder service.

The service loads the class by import path (`models.experimental.kimi_k3_moonvit_v2.service.media:KimiK3Media`),
builds it with the checkpoint's config.json and calls only the members below. The media key, the store, request
limits, image decoding and batching belong to the service; this module enforces only the model's own rules, in
`prepare`.

The encoder runs on a (1, 4) ring of Blackhole chips, since attention is ring attention. Every image is padded to
`max_patches` (67,584) patch rows and `logical_n` is a device tensor, so the one trace captured in `load` serves every
image size.

ttnn is imported only where the chips are used, in `device_params` and `load`: `prepare`, `expansion` and
`stored_nbytes` must also run in a process without ttnn. With the option `stub_embeddings`, `load` and `encode` write
seeded stand-in embeddings and use no chip, so the service and the prefill side can be tested without a ring.

The constructor loads Kimi's own tokenizer (`tokenizer_source`; it needs `transformers` and `tiktoken`), which
`expansion` uses to tokenize each image's prompt. Each source is loaded once per process.

The service must not call `load`, `encode` or `close` from several threads at once: they share the device inputs and
the one captured trace.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import os
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import CHECKPOINT_ENV
from models.experimental.kimi_k3_moonvit_v2.service import format as media_format
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import resize_array
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import RING_SIZE, MoonViTConfig

if TYPE_CHECKING:
    from models.experimental.kimi_k3_moonvit_v2.tt.trace import Trace
    from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model

# The ring is along mesh axis 1.
SP_AXIS = 1
# The Hugging Face repo id: the tokenizer's source when no local checkpoint folder is given.
MODEL_NAME = "moonshotai/Kimi-K3"


# TODO(#66023): settle how production finds the tokenizer instead of the env var with a download fallback.
def tokenizer_source() -> str:
    """Where `AutoTokenizer.from_pretrained` loads Kimi's tokenizer from: the local checkpoint folder that
    $KIMI_K3_HF_MODEL names, which saves a download when the tokenizer ships with the weights, else MODEL_NAME (the
    files are then downloaded into the Hugging Face cache)."""
    return os.environ.get(CHECKPOINT_ENV) or MODEL_NAME


@functools.cache
def load_tokenizer(source: str):
    """Kimi-K3's tokenizer from `source`, loaded once per process."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


@dataclass(frozen=True)
class K3Image:
    pixels: np.ndarray
    """Resized and padded RGB image, uint8 [height, width, 3]; height and width are multiples of 28. encode
    normalizes it and cuts it into patches straight into the device inputs (`TT_MoonViT_Vision_Model.prepare_image`)."""
    grid_h: int
    """Image-token rows: height / 28 (14x14 patches merged 2x2)."""
    grid_w: int
    """Image-token columns: width / 28."""
    width: int
    """Width in pixels of the image given to prepare, before the resize. The prompt states it (expansion)."""
    height: int
    """Height in pixels of the image given to prepare, before the resize."""

    @property
    def n_tokens(self) -> int:
        return self.grid_h * self.grid_w


def _clear_frames(error: BaseException) -> None:
    """Clear the local variables of the frames in error's traceback, and in those of the errors it chains.

    The frames of a failed load or encode hold device tensors, which the caller holding the error would keep alive
    after close. The tracebacks still show their lines."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending += [error.__cause__, error.__context__]


def _stub_rows(item: K3Image, hidden: int) -> np.ndarray:
    """Stand-in embeddings for item, seeded by its pixels: bf16 bits as uint16, [n_tokens, hidden]."""
    digest = hashlib.sha256(np.asarray(item.pixels.shape).tobytes() + item.pixels.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rows = torch.randn(item.n_tokens, hidden, generator=torch.Generator().manual_seed(seed)) * 0.02
    return rows.to(torch.bfloat16).view(torch.uint16).numpy()


def _check_options(options: dict) -> None:
    if not isinstance(options, dict):
        raise ValueError(f"options must be a dict, got {type(options).__name__}")
    cache_dir = options.get("cache_dir")
    if cache_dir is not None and not isinstance(cache_dir, (str, os.PathLike)):
        raise ValueError(f"cache_dir must be a folder path, got {cache_dir!r}")
    if not isinstance(options.get("stub_embeddings", False), bool):
        raise ValueError(f"stub_embeddings must be a bool, got {options['stub_embeddings']!r}")


class KimiK3Media:
    """The Kimi-K3 media module that the encoder service calls.

    Not thread safe: the service must not call load, encode or close from several threads at once."""

    @property
    def device_params(self) -> dict:
        """How the service opens the mesh: the ring's fabric, and DRAM for the one captured trace.

        Reading it imports ttnn (the fabric config is a ttnn enum); prepare, expansion and stored_nbytes do not."""
        from models.experimental.kimi_k3_moonvit_v2.tt.ccl import RING_FABRIC_CONFIG
        from models.experimental.kimi_k3_moonvit_v2.tt.trace import TRACE_REGION_SIZE

        return {"fabric_config": RING_FABRIC_CONFIG, "trace_region_size": TRACE_REGION_SIZE}

    def __init__(self, hf_config: dict):
        # Raises for a vision_config that is not MoonViT-V2.
        self.config = MoonViTConfig.from_hf_config(hf_config)
        self.hidden = self.config.text_hidden_size
        # <|media_pad|>: the frontend renders the chat with one <|media_pad|> per image as its image_prompts, so each
        # image arrives as this one token. expansion replaces it with the image's whole prompt.
        self.placeholder_token_ids = {"image": hf_config["media_placeholder_token_id"]}
        # Loaded here, not in load: the service also calls expansion in processes that never load the encoder.
        source = tokenizer_source()
        self.tokenizer = load_tokenizer(source)
        pad = self.tokenizer.convert_tokens_to_ids("<|media_pad|>")
        if pad != self.placeholder_token_ids["image"]:
            raise ValueError(
                f"the tokenizer of {source} maps <|media_pad|> to {pad}, config.json's media_placeholder_token_id is"
                f" {self.placeholder_token_ids['image']}"
            )
        self._model: TT_MoonViT_Vision_Model | None = None
        self._trace: Trace | None = None
        self._stub = False

    def prepare(self, kind: str, media: np.ndarray) -> K3Image:
        """Resize and pad one image on the CPU as HF's processor does, bit for bit (`tt/image_prep.py:resize_array`).
        encode does the rest of HF's processing (normalize, patchify), so the item stays uint8 and pickles small.

        media: non-empty uint8 RGB [height, width, 3], any size: the largest grid HF's resize makes (67,320
        patches) fits the device's 67,584 rows.
        """
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"Kimi-K3 does not accept {kind}")
        # TODO(#66023): Kimi-K3's reference handles non-RGB images (alpha onto a chessboard, after the resize), which
        # an RGB array can't reproduce. Either the service does it, or this interface takes alpha or the original image.
        pixels = resize_array(media, self.config)  # raises for anything but a non-empty uint8 RGB array
        fh, fw = (k * self.config.patch_size for k in self.config.merge_kernel_size)
        height, width = media.shape[:2]
        return K3Image(pixels, pixels.shape[0] // fh, pixels.shape[1] // fw, width, height)

    def expansion(self, item: K3Image) -> list[tuple[int, int]]:
        """The tokens that replace the image's one <|media_pad|>: the prompt HF's processor gives the image
        (`reference/kimi_k3_vision_processing.py:make_image_prompt`, with the size before the resize), tokenized by
        Kimi's tokenizer, with its one <|media_pad|> repeated once per embedding, as the model widens it
        (`_merge_input_ids_with_image_features`). Only the pads get embeddings."""
        from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor

        pad = self.placeholder_token_ids["image"]
        ids = self.tokenizer.encode(KimiK3VisionProcessor.make_image_prompt(item.width, item.height))
        if ids.count(pad) != 1:
            raise ValueError(f"the image prompt has {ids.count(pad)} <|media_pad|> tokens, not 1: {ids}")
        at = ids.index(pad)
        return [*((t, 1) for t in ids[:at]), (pad, item.n_tokens), *((t, 1) for t in ids[at + 1 :])]

    def stored_nbytes(self, item: K3Image) -> int:
        return media_format.stored_nbytes(item.n_tokens, self.hidden)

    def load(self, mesh_device, weights_path: str, kind: str, options: dict) -> None:
        """Put the vision weights on `mesh_device`, run the encoder once and capture its trace.

        mesh_device: the (1, 4) ring, opened by the service with `device_params`; it is not opened, closed or reset
        here, and any other width along axis 1 raises ValueError. weights_path: the Kimi-K3 checkpoint folder; only
        the shards that hold the vision tower and projector are read. options:
        - `cache_dir` (optional): a folder that keeps the converted weights for the next load
          (`tt/module.py:load_cached`); valid only for this mesh shape.
        - `stub_embeddings` (optional, bool): encode writes seeded stand-in embeddings; mesh_device and weights_path
          are not used.

        Raises ValueError before ttnn is imported for another kind, bad options, a weights_path that is not a folder or
        a mesh_device of None; RuntimeError for a second load before close. A failed load frees what it put on the
        chips before it raises.
        """
        if kind not in self.placeholder_token_ids:
            raise ValueError(f"Kimi-K3 does not accept {kind}")
        if self._stub or self._model is not None or self._trace is not None:
            raise RuntimeError("load after load: close first")
        _check_options(options)
        if options.get("stub_embeddings"):
            self._stub = True
            return
        if not weights_path or not os.path.isdir(weights_path):
            raise ValueError(f"weights_path must be a Kimi-K3 checkpoint folder, got {weights_path!r}")
        if mesh_device is None:
            raise ValueError("mesh_device is None: load runs the encoder on the ring that the service opened")
        import ttnn
        from models.experimental.kimi_k3_moonvit_v2.reference.checkpoint import load_vision_state_dict
        from models.experimental.kimi_k3_moonvit_v2.tt.ccl import RING_NUM_LINKS, TT_CCL, ParallelFactor
        from models.experimental.kimi_k3_moonvit_v2.tt.module import load_cached
        from models.experimental.kimi_k3_moonvit_v2.tt.trace import Trace
        from models.experimental.kimi_k3_moonvit_v2.tt.vision_model import TT_MoonViT_Vision_Model

        ccl = model = inputs = None
        try:
            ccl = TT_CCL(mesh_device, num_links=RING_NUM_LINKS, topology=ttnn.Topology.Ring)
            model = self._model = TT_MoonViT_Vision_Model(
                self.config, mesh_device=mesh_device, sequence_parallel=ParallelFactor(RING_SIZE, SP_AXIS), ccl=ccl
            )
            load_cached(model, options.get("cache_dir"), lambda: load_vision_state_dict(weights_path))
            # The inputs are written in place on every call, so a trace captured on this image replays any other.
            inputs = model.prepare_image(np.zeros((28, 28, 3), np.uint8))
            # Untraced first: that compiles the programs and allocates the ring-attention buffers, neither of which
            # can happen inside a capture. forward untilizes on the chips, inside the trace, so the replay's output
            # is read back as plain rows.
            ttnn.deallocate(model(**inputs))
            self._trace = Trace(mesh_device, lambda: model(**inputs))
        except BaseException as error:
            # This frame is still running, so _clear_frames cannot clear it: drop its device objects here. model is
            # also a closure cell of the trace's forward.
            ccl = model = inputs = None
            self.close()
            _clear_frames(error)  # the failed call's frames hold device tensors
            gc.collect()
            raise

    def encode(self, items: list[K3Image], out: list[memoryview]) -> None:
        """Write each item's stored value into its slot in out: format.Header("image"), then its bf16 embeddings.

        out holds one slot per item: a writable, C-contiguous memoryview of `stored_nbytes(item)` bytes, in any format.
        Every slot is checked (writable, C-contiguous, the right size) before anything is written. encode normalizes
        and patchifies each image on the CPU straight into the device inputs (`TT_MoonViT_Vision_Model.prepare_image`;
        the position embeddings and RoPE tables are rebuilt and copied only when the grid differs from the previous
        image's), then the rows of the chips that hold them are copied straight into the slot
        (`TT_MoonViT_Vision_Model.read_output`); the waits on the chips release the GIL. If encode raises, the caller
        must not use the slots of that call.

        Raises ValueError for a slot that is read-only, not C-contiguous or of the wrong size, and RuntimeError before
        load or after close.
        """
        if len(items) != len(out):
            raise ValueError(f"{len(items)} items for {len(out)} slots")
        for item, buf in zip(items, out):
            if buf.readonly:
                raise ValueError("a read-only slot: encode writes into its slots")
            if not buf.c_contiguous:
                raise ValueError("a slot that is not C-contiguous: a stored value is one run of bytes")
            size = self.stored_nbytes(item)
            if buf.nbytes != size:
                raise ValueError(
                    f"a slot of {buf.nbytes} B for an image of {item.n_tokens} tokens, which needs {size} B"
                )
        if not self._stub and self._trace is None:
            raise RuntimeError("encode before load, or after close")
        for item, buf in zip(items, out):
            buf = buf.cast("B")  # the header and the rows are at byte offsets, whatever the slot's format
            media_format.Header("image").write(buf)
            rows = media_format.embeddings(buf, self.hidden)
            if self._stub:
                rows[:] = _stub_rows(item, self.hidden)
                continue
            try:
                self._model.prepare_image(item.pixels)
                self._model.read_output(self._trace.replay(), rows)
            except BaseException as error:
                _clear_frames(error)  # so that close, while the caller holds the error, frees the device tensors
                raise

    def close(self) -> None:
        """Release the trace and drop the weights and the device inputs. Safe after a failed load or encode, and
        when called twice. The mesh is the service's: it stays open."""
        trace, self._trace = self._trace, None
        loaded = trace is not None or self._model is not None
        self._model = None
        self._stub = False
        if trace is not None:
            trace.release()
        if loaded:
            gc.collect()  # device tensors that only reference cycles still hold
