# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image + video + text (multimodal) prefill path for GLM-5.3-Flash on Tenstorrent.

HF (``Glm5NextModel.forward``)::

    inputs_embeds = embed_tokens(input_ids)                                  # bf16 [1, S, 4096]
    image_embeds  = visual(pixel_values, grid_thw).pooler_output             # [N_img, 4096], cast to bf16
    video_embeds  = visual(pixel_values_videos, [1, h, w] * t).pooler_output # one window per 2-frame group
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)   # image tokens outside video spans
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)   # image tokens inside video spans
    hidden        = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)   # the 4 mHC streams

The text model has no rotary embedding (``qk_rope_head_dim == 0``): vision tokens need no position
handling beyond their order in the sequence, so the only multimodal work on the text side is the
input streams. On device the text rows keep the exact ``TtEmbedding`` path (``ttnn.embedding`` of
the bf16 table into fp32 + ``mhc_expand``) and the vision rows are overwritten with the tower output
(bf16, as HF's cast) by one masked multiply + add on the packed streams ``[1, 1, T_pad, 4*4096]``
fp32 (fp32 ops with 1.0 / 0.0 operands: bit-exact on the text rows).

Video = frames in pairs (``temporal_patch_size`` 2): the processor stacks every 2 consecutive sampled frames
into one ``[h*w, 1176]`` patch group and the prompt holds one run of ``h*w/4`` image tokens per group (plus a
``"t.t seconds"`` timestamp); the tower runs each group as a ``(1, h, w)`` window (HF ``get_video_features``
flattens the grid), so on device a video is ``t`` images whose features are scattered in prompt order. All of
that bookkeeping (which image-token rows belong to which item, chunk splitting) is host-only in
``tt/mm_inputs.py``; this module owns the device work:

* ``encode_multimodal`` / ``encode_multimodal_video`` / ``encode_prompt``: processor + chat template -> ids,
  pixel patches, grids (``MMItem`` list in prompt order);
* ``vision_features`` (``image_features`` / ``video_features``): the ``tt/vision.py`` tower on device over whole
  attention windows batched to ``max_patches`` per call (bounded transients), per-item host features;
* ``multimodal_streams`` / ``chunk_streams``: the prefill input streams of a whole prompt or of one device chunk
  of it (the served adapter's 16 K chunks; an item straddling a chunk boundary takes its rows chunk by chunk);
* ``prefill_multimodal``: the layers from the whole-prompt streams, same return as the text ``prefill``;
* ``VisionRuntime``: the served adapter's tower (resident on the mesh, or built per request from host-cached
  weights inside the prefill reserve: the 1 M profile), its warm-up and the planner's byte counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import device_tensors_of
from models.demos.glm53_flash.tt.memory_plan import vision_tower_bytes
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.mm_inputs import (
    IMAGE_TOKEN_ID,
    MMItem,
    ScatterEntry,
    chunk_scatter,
    feature_rows,
    flatten_grid,
)
from models.demos.glm53_flash.tt.vision import GLM53VisionConfig, TtVisionTower, load_vision_weights
from models.demos.glm53_flash.tt.vllm_config import DEFAULT_VISION_BATCH_PATCHES, eager_pad_length

TILE = 32
THINK_END = "</think>"
DEFAULT_VIDEO_FPS = 2.0  # the checkpoint's video_processor fps (one sampled frame per 0.5 s)


# ---------------------------------------------------------------------------------------------
# host: prompt encoding
# ---------------------------------------------------------------------------------------------
@dataclass
class MMEncoding:
    """A prompt's ids ``[1, S]`` and its multimodal items in prompt order (``MMItem``: image / video patches +
    grid); ``pixel_values`` / ``image_grid_thw`` / ``pixel_values_videos`` / ``video_grid_thw`` are the processor's
    batched tensors (None when the modality is absent)."""

    input_ids: torch.Tensor
    items: list[MMItem] = field(default_factory=list)
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    pixel_values_videos: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None

    @property
    def features_order(self) -> list[str]:
        return [it.kind for it in self.items]


def _load_image(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image


def load_frames(video) -> np.ndarray:
    """Video frames as ``[T, H, W, 3]`` uint8: an ndarray / tensor of that shape, a list of PIL images or arrays,
    or a directory of frame image files (sorted by name). Video files are not decoded here (pass frames)."""
    if isinstance(video, (str, Path)):
        p = Path(video)
        if not p.is_dir():
            raise ValueError(f"{p}: pass a directory of frame images or the frames themselves (no video decoding)")
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if not files:
            raise ValueError(f"{p}: no frame images")
        return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    if torch.is_tensor(video):
        video = video.cpu().numpy()
    if isinstance(video, np.ndarray):
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"frames must be [T, H, W, 3], got {video.shape}")
        return video.astype(np.uint8)
    frames = [np.asarray(_load_image(f)) if not isinstance(f, np.ndarray) else f for f in video]
    return np.stack(frames).astype(np.uint8)


def synthetic_clip(image, n_frames: int = 4, step_deg: float = 10.0) -> np.ndarray:
    """A deterministic short clip from one image: frame ``k`` is the image rotated by ``k * step_deg`` degrees
    (``[T, H, W, 3]`` uint8). The reference, the demo and the tests share it (no video decoding needed)."""
    img = _load_image(image)
    return np.stack([np.asarray(img.rotate(step_deg * k)) for k in range(n_frames)]).astype(np.uint8)


def _video_metadata(n_frames: int, fps: float):
    from transformers.video_utils import VideoMetadata

    return VideoMetadata(total_num_frames=int(n_frames), fps=float(fps), duration=n_frames / fps, video_backend="np")


def encode_prompt(processor, content: list[dict], *, fps: float = DEFAULT_VIDEO_FPS, think: bool = False) -> MMEncoding:
    """One user turn of ``content`` entries (``{"type": "image", "image": ...}``, ``{"type": "video", "video":
    frames}``, ``{"type": "text", "text": ...}`` in prompt order) through the chat template + processor. Images:
    PIL / path; videos: frames (``load_frames``) sampled at ``fps`` (the processor keeps every frame at the
    checkpoint's 2 fps, pairs them into temporal groups and writes ``"<t>.<d> seconds"`` after each group's run of
    image tokens). ``think=False`` appends ``</think>`` so the answer starts immediately."""
    images, videos, template = [], [], []
    for entry in content:
        kind = entry.get("type")
        if kind in ("image", "image_url"):
            images.append(_load_image(entry.get("image", entry.get("image_url"))))
            template.append({"type": "image"})
        elif kind in ("video", "video_url"):
            videos.append(load_frames(entry.get("video", entry.get("video_url"))))
            template.append({"type": "video"})
        else:
            template.append({"type": "text", "text": entry["text"]})
    messages = [{"role": "user", "content": template}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kw: dict = {"text": [text], "return_tensors": "pt"}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
        kw["video_metadata"] = [_video_metadata(v.shape[0], fps) for v in videos]
    enc = processor(**kw)
    ids = enc["input_ids"]
    if not think:
        end = processor.tokenizer.convert_tokens_to_ids(THINK_END)
        ids = torch.cat([ids, torch.tensor([[end]], dtype=ids.dtype)], dim=1)
    pv, ig = enc.get("pixel_values"), enc.get("image_grid_thw")
    pvv, vg = enc.get("pixel_values_videos"), enc.get("video_grid_thw")
    items: list[MMItem] = []
    i_img = i_vid = 0
    img_off = vid_off = 0
    for entry in template:
        if entry["type"] == "image":
            n = int(ig[i_img].prod())
            items.append(MMItem("image", pv[img_off : img_off + n].float(), ig[i_img]))
            img_off += n
            i_img += 1
        elif entry["type"] == "video":
            n = int(vg[i_vid].prod())
            items.append(MMItem("video", pvv[vid_off : vid_off + n].float(), vg[i_vid]))
            vid_off += n
            i_vid += 1
    return MMEncoding(ids, items, pv, ig, pvv, vg)


def encode_multimodal(
    processor, image: Image.Image | str | Path, question: str, *, think: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One image + question through the chat template -> ``(input_ids [1, S] int64, pixel_values
    [N_patches, 1176] fp32, image_grid_thw [1, 3] int64)``.

    The template renders ``<|begin_of_image|><|image|><|end_of_image|>`` before the question and the
    processor expands ``<|image|>`` into ``grid.prod() // merge**2`` copies of ``image_token_id`` (256 for
    a 448x448 image). The generation prompt ends with ``<|assistant|><think>``; ``think=False`` (default)
    appends ``</think>`` so the model answers directly (a checkable caption within a few tokens instead
    of a reasoning trace). The same encoding feeds the HF reference and the device."""
    image = _load_image(image)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    ids = enc["input_ids"]
    if not think:
        end = processor.tokenizer.convert_tokens_to_ids(THINK_END)
        ids = torch.cat([ids, torch.tensor([[end]], dtype=ids.dtype)], dim=1)
    return ids, enc["pixel_values"], enc["image_grid_thw"]


def encode_multimodal_video(
    processor, video, question: str, *, fps: float = DEFAULT_VIDEO_FPS, think: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One video (frames, ``load_frames``) + question -> ``(input_ids [1, S], pixel_values_videos [N_patches,
    1176] fp32, video_grid_thw [1, 3])``: ``<|begin_of_video|>`` + per temporal group ``<|begin_of_image|>`` +
    ``h*w/4`` image tokens + ``<|end_of_image|>`` + the group's timestamp + ``<|end_of_video|>`` + the question."""
    enc = encode_prompt(
        processor, [{"type": "video", "video": video}, {"type": "text", "text": question}], fps=fps, think=think
    )
    return enc.input_ids, enc.pixel_values_videos, enc.video_grid_thw


# ---------------------------------------------------------------------------------------------
# device: vision tower
# ---------------------------------------------------------------------------------------------
def build_vision_tower(device, snapshot: Path | None = None, weights: dict | None = None, **kw) -> TtVisionTower:
    """``TtVisionTower`` with the checkpoint's ``model.visual.*`` weights on ``device`` (a mesh device
    replicates the weights on every chip: ``from_torch`` without a mapper). ``weights``: the host tensors of
    ``load_vision_weights`` when the caller keeps them (the per-request serving mode)."""
    cfg = GLM53VisionConfig.from_snapshot(snapshot)
    return TtVisionTower(device, cfg, load_vision_weights(snapshot) if weights is None else weights, **kw)


def release_vision_tower(tower: TtVisionTower) -> int:
    """Deallocate every device tensor of the tower (the single-chip model streams the text layers through
    the same DRAM). Returns the number of buffers released; the object is unusable afterwards."""
    n = 0
    for t in device_tensors_of(tower):
        if t.is_allocated():
            ttnn.deallocate(t)
            n += 1
    return n


def _first_shard_to_torch(t: ttnn.Tensor) -> torch.Tensor:
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0] if shards else t)


VISION_PATCH_BUCKET = 1024  # tower calls are padded to a multiple of this many patches (the 448x448 image's count)


def vision_pad_patches(n_patches: int, bucket: int = VISION_PATCH_BUCKET) -> int:
    """The padded row count of a tower call holding ONE window of ``n_patches``: the next multiple of ``bucket``
    (1024, 2048, 3072, 4096 for the default 4096-patch cap) plus one 32-row tile of pad rows, so the call always
    has exactly two attention windows (the image, the pad) and its window bounds one shape: the tower's programs
    (patch embed, the windowed SDPA, the merge) exist at a handful of shapes that the boot warm-up builds
    (``VisionRuntime.warmup``) and no image or clip geometry compiles at request time (a 2-frame clip batched as
    two windows compiled its SDPA under the sealed cache, 2026-10-28 8x4)."""
    return max(bucket, -(-int(n_patches) // bucket) * bucket) + TILE


def _tower_call(
    tower: TtVisionTower, pixel_values: torch.Tensor, grid_rows: torch.Tensor, pad_to: int | None = None
) -> list[torch.Tensor]:
    """One tower call over whole windows (``grid_rows`` ``[n, 3]`` with t == 1): per-window ``[h*w/4, 4096]``
    host features (fp32 holding bf16 values). Every device temporary is freed before returning."""
    inputs = tower.prepare_inputs(pixel_values, grid_rows, pad_to=pad_to)
    out = tower(inputs)
    flat = _first_shard_to_torch(out).float().reshape(-1, tower.cfg.out_hidden_size)[: inputs.num_tokens]
    feats = [f.clone() for f in torch.split(flat, inputs.split_sizes)]
    ttnn.deallocate(out)
    for t in (inputs.patches, inputs.cos, inputs.sin, inputs.cu_window):
        if t is not None:
            ttnn.deallocate(t)
    return feats


def vision_features(
    tower: TtVisionTower, items: list[MMItem], *, max_patches: int = DEFAULT_VISION_BATCH_PATCHES
) -> list[torch.Tensor]:
    """HF ``get_image_features`` / ``get_video_features`` on device for ``items`` (in prompt order): per item the
    host features ``[item.n_tokens, 4096]`` fp32 holding bf16 values (the tower's ``out_dtype``, HF's cast to the
    bf16 embeddings). Every item's grid is flattened to one ``(1, h, w)`` window per temporal group (a video of
    ``t`` groups = ``t`` windows, an image = 1) and consecutive windows are batched into tower calls of at most
    ``max_patches`` patches (``mm_inputs.vision_batches``): the device transient is bounded by the batch, not by
    the prompt's media. On a mesh every chip runs the tower (replicated weights, identical inputs) and chip 0's
    result is read."""
    windows: list[tuple[int, torch.Tensor, torch.Tensor]] = []  # (item index, pixel slice, [1, h, w])
    for k, it in enumerate(items):
        off = 0
        for row in flatten_grid(it.grid_thw):
            n = int(row.prod())
            windows.append((k, it.pixel_values[off : off + n], row))
            off += n
        if off != it.n_patches:
            raise ValueError(f"item {k}: {off} patches over its windows vs {it.n_patches}")
    if not windows:
        return []
    grid_flat = torch.stack([w[2] for w in windows])
    per_window: list[torch.Tensor | None] = [None] * len(windows)
    # one tower call per window (an image, a clip's temporal group): every call is one window + the pad window,
    # padded to its patch bucket (``vision_pad_patches``), so the tower runs a fixed, warmed set of shapes whatever
    # the media mix; ``max_patches`` caps a single window
    for i, (_, pv, row) in enumerate(windows):
        n = int(pv.shape[0])
        if n > max_patches:
            raise ValueError(f"a vision window of {n} patches exceeds the {max_patches}-patch tower call cap")
        per_window[i] = _tower_call(tower, pv, grid_flat[i : i + 1], pad_to=vision_pad_patches(n))[0]
    out: list[torch.Tensor] = []
    for k, it in enumerate(items):
        parts = [per_window[i] for i, w in enumerate(windows) if w[0] == k]
        feats = torch.cat(parts, dim=0)
        if feats.shape[0] != it.n_tokens:
            raise RuntimeError(f"item {k}: {feats.shape[0]} feature rows vs {it.n_tokens} tokens")
        out.append(feats)
    return out


def image_features(
    device,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    tower: TtVisionTower | None = None,
    snapshot: Path | None = None,
    max_patches: int = DEFAULT_VISION_BATCH_PATCHES,
) -> list[torch.Tensor]:
    """HF ``get_image_features`` on device: the processor's ``pixel_values`` ``[N_patches, 1176]`` and
    ``image_grid_thw`` ``[n, 3]`` through ``tt/vision.py`` -> per-image host features ``[tokens, 4096]`` fp32
    holding bf16 values. ``tower``: a resident tower (built once, demo / mesh); without one the tower is
    built, run and released here (the single-chip model needs the DRAM for the text layers). A grid row
    with ``t > 1`` (a video passed here) yields that video's features (its ``t`` windows concatenated)."""
    own = tower is None
    if own:
        tower = build_vision_tower(device, snapshot)
    grid = torch.as_tensor(grid_thw).reshape(-1, 3).long()
    items, off = [], 0
    for row in grid:
        n = int(row.prod())
        items.append(MMItem("video" if int(row[0]) > 1 else "image", pixel_values[off : off + n].float(), row))
        off += n
    if off != int(pixel_values.shape[0]):
        raise ValueError(f"pixel_values hold {int(pixel_values.shape[0])} patches, the grids need {off}")
    feats = vision_features(tower, items, max_patches=max_patches)
    if own:
        release_vision_tower(tower)
    return feats


def video_features(
    device,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    *,
    tower: TtVisionTower | None = None,
    snapshot: Path | None = None,
    max_patches: int = DEFAULT_VISION_BATCH_PATCHES,
) -> list[torch.Tensor]:
    """HF ``get_video_features`` on device: per video ``[t * h * w / 4, 4096]`` host features (the ``t`` temporal
    groups' windows in order)."""
    return image_features(
        device, pixel_values_videos, video_grid_thw, tower=tower, snapshot=snapshot, max_patches=max_patches
    )


# ---------------------------------------------------------------------------------------------
# device: input streams
# ---------------------------------------------------------------------------------------------
def image_rows(input_ids: torch.Tensor, image_token_id: int = IMAGE_TOKEN_ID) -> torch.Tensor:
    """Row indices (prompt positions) of the vision placeholder tokens (images and video frames alike)."""
    return torch.nonzero(input_ids.reshape(-1) == image_token_id).reshape(-1)


def _streams_with_rows(model, ids: torch.Tensor, rows: torch.Tensor, feats: torch.Tensor, T_pad: int) -> ttnn.Tensor:
    """The packed streams of ``ids`` (``[S]``, padded to ``T_pad`` rows) with ``feats`` ``[len(rows), 4096]``
    scattered at ``rows`` (positions within the chunk, ascending): the text path on device for every row, then
    ``streams * keep + img`` with host-built masks (1.0 / 0.0 operands: bit-exact on the text rows)."""
    emb: TtEmbedding = model.embedding
    mesh: GLMMesh = model.mesh
    S = int(ids.numel())
    unit = TILE * mesh.rows
    if T_pad < S or T_pad % unit:
        raise ValueError(f"pad length {T_pad} must hold the {S} tokens and be a multiple of {unit}")
    width = emb.hc_mult * emb.hidden
    if rows.numel() and (int(rows.max()) >= S or int(rows.min()) < 0):
        raise ValueError(f"scatter rows {int(rows.min())}..{int(rows.max())} outside the {S}-token chunk")
    if feats.shape[0] != rows.numel() or feats.shape[-1] != emb.hidden:
        raise ValueError(f"{rows.numel()} vision rows vs features {tuple(feats.shape)}")

    # text path on device: identical to GLM53Model.prefill / GLM53MeshModel.prefill
    host_ids = emb.tokens_to_host(ids, pad_to=T_pad)
    if mesh.rows > 1:
        ids_dev = mesh.upload(host_ids, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        ids_dev = ttnn.from_torch(host_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
    streams = emb(ids_dev)
    ttnn.deallocate(ids_dev)
    if rows.numel() == 0:
        return streams

    # host: the masks over the packed layout (row = prompt position, the 4 streams side by side)
    keep = torch.ones(1, 1, T_pad, width, dtype=torch.float32)
    keep[0, 0, rows] = 0.0
    img = torch.zeros(1, 1, T_pad, width, dtype=torch.float32)
    img[0, 0, rows] = feats.to(torch.bfloat16).float().repeat(1, emb.hc_mult)
    if mesh.rows > 1:
        keep_dev = mesh.upload(keep, ttnn.float32, rows=2)
        img_dev = mesh.upload(img, ttnn.float32, rows=2)
    else:
        keep_dev = ttnn.from_torch(keep, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=model.device)
        img_dev = ttnn.from_torch(img, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=model.device)
    streams = ttnn.multiply(streams, keep_dev, output_tensor=streams)
    streams = ttnn.add(streams, img_dev, output_tensor=streams)
    ttnn.deallocate(keep_dev)
    ttnn.deallocate(img_dev)
    return streams


def multimodal_streams(
    model,
    input_ids: torch.Tensor,
    features: list[torch.Tensor] | torch.Tensor,
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
    pad_to: int | None = None,
) -> ttnn.Tensor:
    """Prefill input streams for a whole image / video + text prompt: the packed mHC streams ``[1, 1, T_pad,
    4*4096]`` fp32 (``T_pad`` = ``pad_to`` or ``S`` rounded up to ``32 * rows``; on a multi-row mesh each row
    holds its slab ``[1, 1, T_pad / rows, 4*4096]``, the layout ``GLM53MeshModel.prefill`` builds for text).

    Text rows: exactly the text path (``model.embedding`` of the ids on device: bf16 table rows widened to
    fp32, repeated over the 4 streams). Vision rows (``input_ids == image_token_id``, images and video frames
    alike): ``features`` (the per-item ``[tokens, 4096]`` tensors of ``vision_features`` / HF, concatenated in
    prompt order) rounded to bf16 as HF does, on all 4 streams. ``model``: ``GLM53Model`` / ``GLM53MeshModel``
    (anything with ``embedding``, ``mesh`` and ``device``)."""
    ids = input_ids.reshape(-1).long()
    S = int(ids.numel())
    unit = TILE * model.mesh.rows
    T_pad = eager_pad_length(S, unit) if pad_to is None else int(pad_to)  # a warmed bucket, never a new length
    rows = image_rows(ids, image_token_id)
    feats = torch.cat(list(features), dim=0) if not torch.is_tensor(features) else features
    if feats.shape[0] != rows.numel():
        raise ValueError(f"{rows.numel()} vision tokens in the prompt vs features {tuple(feats.shape)}")
    return _streams_with_rows(model, ids, rows, feats, T_pad)


def chunk_streams(
    model, chunk_ids: torch.Tensor, entries: list[tuple[ScatterEntry, torch.Tensor]], pad_to: int
) -> ttnn.Tensor:
    """The prefill input streams of ONE device chunk of a prompt (``chunk_ids`` ``[n]``, padded to ``pad_to``
    rows): ``entries`` pair every ``mm_inputs.chunk_scatter`` entry with its item's features (the entry's
    ``local_rows`` take ``feats[entry.lo:entry.hi]``). Rows past the chunk are never written (``local_rows`` lie
    inside it by construction); an entry-less chunk is the plain text embedding."""
    ids = chunk_ids.reshape(-1).long()
    if not entries:
        return _streams_with_rows(
            model, ids, torch.zeros(0, dtype=torch.long), torch.zeros(0, model.embedding.hidden), pad_to
        )
    rows = torch.cat([e.local_rows.reshape(-1) for e, _ in entries])
    feats = torch.cat([f[e.lo : e.hi] for e, f in entries], dim=0)
    order = torch.argsort(rows)
    return _streams_with_rows(model, ids, rows[order], feats[order], pad_to)


# ---------------------------------------------------------------------------------------------
# device: prefill
# ---------------------------------------------------------------------------------------------
def prefill_multimodal(
    model,
    input_ids: torch.Tensor,
    features: list[torch.Tensor] | torch.Tensor,
    *,
    user: int = 0,
    image_token_id: int = IMAGE_TOKEN_ID,
    **kw,
):
    """The text ``prefill`` of ``model`` from ``multimodal_streams``: ``GLM53Model`` -> logits of the last
    token (``[1, 1, 1, vocab]`` bf16; ``all_rows=`` / ``on_layer=`` pass through), ``GLM53MeshModel`` ->
    ``(streams_local, logits_last, owner_row)``. One whole prompt (``start = 0``), image and video items alike."""
    S = int(input_ids.numel())
    streams = multimodal_streams(model, input_ids, features, image_token_id=image_token_id)
    return model.prefill_streams(streams, S, user=user, **kw)


# ---------------------------------------------------------------------------------------------
# serving: the adapter's tower
# ---------------------------------------------------------------------------------------------
class VisionRuntime:
    """The served adapter's vision tower (``tt/generator_vllm.py``): ``resident=True`` builds it once on the mesh
    (before any trace capture; its bytes are the planner's ``vision_bytes``), ``resident=False`` keeps the
    checkpoint tensors on the host and builds / runs / releases the tower per request inside the prefill reserve
    (the 1 M profile, where 1.05 GiB per chip of resident weights would cost the fourth 1 M user). ``encode``
    turns a request's items into host features, ``chunk_streams`` the device chunk's input streams."""

    def __init__(
        self,
        device,
        snapshot: Path | None = None,
        *,
        resident: bool = True,
        max_patches: int = DEFAULT_VISION_BATCH_PATCHES,
        weight_dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.snapshot = snapshot
        self.resident = resident
        self.max_patches = int(max_patches)
        self.weight_dtype = weight_dtype
        self.cfg = GLM53VisionConfig.from_snapshot(snapshot)
        self.weights: dict | None = None  # host tensors, kept in the per-request mode
        self.tower: TtVisionTower | None = None
        self.calls = 0
        self.encode_s = 0.0
        if resident:
            t0 = time.perf_counter()
            self.tower = self._build()
            logger.info(
                f"GLM53 vision: tower resident on the mesh ({self.weight_bytes / 2**30:.2f} GiB per chip) in {time.perf_counter() - t0:.1f} s"
            )
        else:
            self.weights = load_vision_weights(snapshot)
            logger.info(
                f"GLM53 vision: per-request tower ({self.weight_bytes / 2**30:.2f} GiB per chip uploaded per image request)"
            )

    @property
    def weight_bytes(self) -> float:
        c = self.cfg
        wb = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625}.get(self.weight_dtype, 2.0)
        return vision_tower_bytes(
            c.depth,
            c.hidden_size,
            c.intermediate_size,
            c.out_hidden_size,
            c.projection_intermediate_size,
            c.patch_dim_padded,
            c.head_dim,
            wb,
        )

    def _build(self) -> TtVisionTower:
        if self.weights is None:
            self.weights = load_vision_weights(self.snapshot)
        tower = build_vision_tower(self.device, self.snapshot, weights=self.weights, weight_dtype=self.weight_dtype)
        if self.resident:
            self.weights = None  # the host copy is not needed again
        return tower

    def encode(self, items: list[MMItem]) -> list[torch.Tensor]:
        """Per item (prompt order) the host features ``[n_tokens, 4096]``; eager (never inside a trace), every
        tower temporary freed, the tower itself released in the per-request mode."""
        if not items:
            return []
        t0 = time.perf_counter()
        tower = self.tower if self.tower is not None else self._build()
        try:
            feats = vision_features(tower, items, max_patches=self.max_patches)
        finally:
            if not self.resident:
                release_vision_tower(tower)
        self.calls += 1
        self.encode_s += time.perf_counter() - t0
        return feats

    def chunk_streams(self, model, chunk_ids: torch.Tensor, entries, pad_to: int) -> ttnn.Tensor:
        return chunk_streams(model, chunk_ids, entries, pad_to)

    def dummy_item(self, side: int = 448) -> MMItem:
        """A 448x448-shaped image item of random patches (the demo image's geometry: grid (1, 32, 32), 1024
        patches, 256 tokens) for the boot warm-up."""
        g = side // self.cfg.patch_size
        grid = torch.tensor([[1, g, g]])
        return MMItem("image", torch.randn(g * g, self.cfg.patch_dim, generator=torch.Generator().manual_seed(0)), grid)

    def warmup(self) -> float:
        """Compile the tower's programs (and, per request mode, exercise the build + release) at EVERY patch
        bucket a tower call can take (``vision_pad_patches``: 1024 .. ``max_patches`` in steps of 1024, one
        dummy item of exactly that many patches each); returns the seconds taken."""
        t0 = time.perf_counter()
        for n in range(VISION_PATCH_BUCKET, self.max_patches + 1, VISION_PATCH_BUCKET):
            self.encode([self.dummy_item_patches(n)])
        return time.perf_counter() - t0

    def dummy_item_patches(self, n_patches: int) -> MMItem:
        """A random image item of exactly ``n_patches`` patches (grid ``(1, 32, n / 32)``; ``n`` a multiple of 128,
        so the 2x2 merge and the 32-row tiles divide it) for the tower warm-up at one patch bucket."""
        if n_patches % 128:
            raise ValueError(f"a dummy tower item needs a multiple of 128 patches, got {n_patches}")
        grid = torch.tensor([[1, 32, n_patches // 32]])
        return MMItem(
            "image", torch.randn(n_patches, self.cfg.patch_dim, generator=torch.Generator().manual_seed(1)), grid
        )

    def release(self) -> None:
        if self.tower is not None:
            release_vision_tower(self.tower)
            self.tower = None


__all__ = [
    "MMEncoding",
    "DEFAULT_VIDEO_FPS",
    "load_frames",
    "synthetic_clip",
    "encode_prompt",
    "encode_multimodal",
    "encode_multimodal_video",
    "build_vision_tower",
    "release_vision_tower",
    "vision_features",
    "image_features",
    "video_features",
    "image_rows",
    "multimodal_streams",
    "chunk_streams",
    "prefill_multimodal",
    "VisionRuntime",
    "feature_rows",
    "chunk_scatter",
]
