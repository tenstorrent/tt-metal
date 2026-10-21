# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image + text (multimodal) prefill path for GLM-5.3-Flash on Tenstorrent.

HF (``Glm5NextModel.forward``)::

    inputs_embeds = embed_tokens(input_ids)                                  # bf16 [1, S, 4096]
    image_embeds  = visual(pixel_values, grid_thw).pooler_output             # [N_img, 4096], cast to bf16
    inputs_embeds = inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)
    hidden        = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)   # the 4 mHC streams

The text model has no rotary embedding (``qk_rope_head_dim == 0``): image tokens need no position
handling beyond their order in the sequence, so the only multimodal work on the text side is the
input streams. On device the text rows keep the exact ``TtEmbedding`` path (``ttnn.embedding`` of
the bf16 table into fp32 + ``mhc_expand``) and the image rows are overwritten with the tower output
(bf16, as HF's cast) by one masked multiply + add on the packed streams ``[1, 1, T_pad, 4*4096]``
fp32 (fp32 ops with 1.0 / 0.0 operands: bit-exact on the text rows).

Pieces: ``encode_multimodal`` (processor + chat template -> ids / pixel patches / grid),
``image_features`` (``tt/vision.py`` tower on device -> host features, per image), ``multimodal_streams``
(streams on device for ``GLM53Model`` or ``GLM53MeshModel``; on a multi-row mesh the rows hold
consecutive slabs of the padded prompt, the layout ``GLM53MeshModel.prefill`` expects) and
``prefill_multimodal`` (the layers from those streams, same return as the text ``prefill``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import ttnn
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import device_tensors_of
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.vision import GLM53VisionConfig, TtVisionTower, load_vision_weights

TILE = 32
THINK_END = "</think>"


# ---------------------------------------------------------------------------------------------
# host: prompt encoding
# ---------------------------------------------------------------------------------------------
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
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    ids = enc["input_ids"]
    if not think:
        end = processor.tokenizer.convert_tokens_to_ids(THINK_END)
        ids = torch.cat([ids, torch.tensor([[end]], dtype=ids.dtype)], dim=1)
    return ids, enc["pixel_values"], enc["image_grid_thw"]


# ---------------------------------------------------------------------------------------------
# device: vision tower
# ---------------------------------------------------------------------------------------------
def build_vision_tower(device, snapshot: Path | None = None, **kw) -> TtVisionTower:
    """``TtVisionTower`` with the checkpoint's ``model.visual.*`` weights on ``device`` (a mesh device
    replicates the weights on every chip: ``from_torch`` without a mapper)."""
    cfg = GLM53VisionConfig.from_snapshot(snapshot)
    return TtVisionTower(device, cfg, load_vision_weights(snapshot), **kw)


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


def image_features(
    device,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    tower: TtVisionTower | None = None,
    snapshot: Path | None = None,
) -> list[torch.Tensor]:
    """HF ``get_image_features`` on device: the processor's ``pixel_values`` ``[N_patches, 1176]`` and
    ``image_grid_thw`` through ``tt/vision.py`` -> per-image host features ``[tokens, 4096]`` fp32
    holding bf16 values (the tower's ``out_dtype``, HF's cast of the features to the bf16 embeddings).

    ``tower``: a resident tower (built once, demo / mesh); without one the tower is built, run and
    released here (the single-chip model needs the DRAM for the text layers). On a mesh the whole mesh
    runs the tower (replicated weights, identical inputs on every chip) and chip 0's result is read."""
    own = tower is None
    if own:
        tower = build_vision_tower(device, snapshot)
    inputs = tower.prepare_inputs(pixel_values, grid_thw)
    out = tower(inputs)
    flat = _first_shard_to_torch(out).float().reshape(-1, tower.cfg.out_hidden_size)[: inputs.num_tokens]
    feats = [f.clone() for f in torch.split(flat, inputs.split_sizes)]
    ttnn.deallocate(out)
    for t in (inputs.patches, inputs.cos, inputs.sin, inputs.cu_window):
        if t is not None:
            ttnn.deallocate(t)
    if own:
        release_vision_tower(tower)
    return feats


# ---------------------------------------------------------------------------------------------
# device: input streams
# ---------------------------------------------------------------------------------------------
def image_rows(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Row indices (prompt positions) of the image placeholder tokens."""
    return torch.nonzero(input_ids.reshape(-1) == image_token_id).reshape(-1)


def multimodal_streams(
    model,
    input_ids: torch.Tensor,
    features: list[torch.Tensor] | torch.Tensor,
    *,
    image_token_id: int = GLM53VisionConfig.image_token_id,
) -> ttnn.Tensor:
    """Prefill input streams for an image + text prompt: the packed mHC streams ``[1, 1, T_pad, 4*4096]``
    fp32 (``T_pad`` = ``S`` rounded up to ``32 * rows``; on a multi-row mesh each row holds its slab
    ``[1, 1, T_pad / rows, 4*4096]``, the layout ``GLM53MeshModel.prefill`` builds for text).

    Text rows: exactly the text path (``model.embedding`` of the ids on device: bf16 table rows widened to
    fp32, repeated over the 4 streams). Image rows (``input_ids == image_token_id``): ``features`` (the
    per-image ``[tokens, 4096]`` tensors of ``image_features`` / HF, concatenated in prompt order) rounded
    to bf16 as HF does, on all 4 streams. The merge is ``streams * keep + img`` with host-built ``keep``
    (1.0 text / 0.0 image rows) and ``img`` (features at the image rows, 0 elsewhere): fp32 ops whose
    operands are 1.0 / 0.0, bit-exact on the text rows. ``model``: ``GLM53Model`` / ``GLM53MeshModel``
    (anything with ``embedding``, ``mesh`` and ``device``)."""
    emb: TtEmbedding = model.embedding
    mesh: GLMMesh = model.mesh
    ids = input_ids.reshape(-1).long()
    S = int(ids.numel())
    unit = TILE * mesh.rows
    T_pad = -(-S // unit) * unit
    rows = image_rows(ids, image_token_id)
    feats = torch.cat(list(features), dim=0) if not torch.is_tensor(features) else features
    if feats.shape[0] != rows.numel() or feats.shape[-1] != emb.hidden:
        raise ValueError(f"{rows.numel()} image tokens in the prompt vs features {tuple(feats.shape)}")
    width = emb.hc_mult * emb.hidden

    # text path on device: identical to GLM53Model.prefill / GLM53MeshModel.prefill
    host_ids = emb.tokens_to_host(ids, pad_to=T_pad)
    if mesh.rows > 1:
        ids_dev = mesh.upload(host_ids, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        ids_dev = ttnn.from_torch(host_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
    streams = emb(ids_dev)
    ttnn.deallocate(ids_dev)

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


# ---------------------------------------------------------------------------------------------
# device: prefill
# ---------------------------------------------------------------------------------------------
def prefill_multimodal(
    model,
    input_ids: torch.Tensor,
    features: list[torch.Tensor] | torch.Tensor,
    *,
    user: int = 0,
    image_token_id: int = GLM53VisionConfig.image_token_id,
    **kw,
):
    """The text ``prefill`` of ``model`` from ``multimodal_streams``: ``GLM53Model`` -> logits of the last
    token (``[1, 1, 1, vocab]`` bf16; ``all_rows=`` / ``on_layer=`` pass through), ``GLM53MeshModel`` ->
    ``(streams_local, logits_last, owner_row)``. One whole prompt (``start = 0``)."""
    S = int(input_ids.numel())
    streams = multimodal_streams(model, input_ids, features, image_token_id=image_token_id)
    return model.prefill_streams(streams, S, user=user, **kw)


__all__ = [
    "encode_multimodal",
    "build_vision_tower",
    "release_vision_tower",
    "image_features",
    "image_rows",
    "multimodal_streams",
    "prefill_multimodal",
]
