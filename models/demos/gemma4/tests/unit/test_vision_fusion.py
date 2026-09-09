# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the vision->text fusion glue in ``Gemma4Model``.

The vision tower has its own PCC coverage; these tests cover the three steps
that sit between it and the text model, none of which the tower tests touch:

* ``_compact_valid_vision_tokens`` -- HF ``hidden_states[pooler_mask]``
* ``_apply_embed_vision``          -- HF ``Gemma4MultimodalEmbedder.forward``
* ``_scatter_vision_tokens``       -- HF ``inputs_embeds.masked_scatter(...)``

Each method is exercised on a bare ``Gemma4Model`` instance carrying only the
attributes it reads, so the tests need no checkpoint and no text model. That
also means they never touch the weight cache, which is what a full-model test
would have to do.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.vision.vision_model_config import vision_mesh_shape_from_env

IMAGE_TOKEN_ID = 258880
EPS = 1e-6


def _stub(mesh_device, **attrs):
    """A ``Gemma4Model`` with just the attributes the fusion helpers read."""
    model = object.__new__(Gemma4Model)
    model.mesh_device = mesh_device
    model.image_token_id = IMAGE_TOKEN_ID
    model._vision_scatter_offset = 0
    model.embed_vision_eps = EPS
    model.embed_vision_weight = None
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


def _is_mesh(mesh_device):
    return hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1


def _replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None


def _from_torch(t, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate(mesh_device),
    )


def _read(t, mesh_device):
    """Read one replica back. Every tensor under test is replicated."""
    if _is_mesh(mesh_device):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return ttnn.to_torch(t)


def _check(name, tt_out, reference, pcc, exact):
    """PCC plus, for pure data movement, a bit-exactness check."""
    assert tuple(tt_out.shape) == tuple(reference.shape), f"{name}: shape {tt_out.shape} != {reference.shape}"
    passing, msg = comp_pcc(reference, tt_out, pcc)
    logger.info(f"{name}: {msg}")
    logger.info(comp_allclose(reference, tt_out))
    if exact:
        diff = (tt_out.float() - reference.float()).abs().max().item()
        assert diff == 0.0, f"{name}: data movement altered values (max abs diff {diff})"
    assert passing, f"{name}: PCC below {pcc}"


# ---------------------------------------------------------------- compaction


def _compaction_mask(kind, batch, length, n_valid):
    mask = torch.zeros(batch, length, dtype=torch.bool)
    if kind == "prefix":
        # What the Gemma4 pooler actually produces: valid buckets are a
        # contiguous prefix, padded soft tokens trail. Hits the slice fast path.
        mask.reshape(-1)[:n_valid] = True
    elif kind == "all_valid":
        mask[:] = True
    elif kind == "scattered":
        # Holes inside the valid range force the ttnn.embedding gather path.
        idx = torch.randperm(batch * length)[:n_valid]
        mask.reshape(-1)[idx] = True
    elif kind == "per_user_prefix":
        # Multi-image batch: each row is a prefix, so the *flattened* mask is
        # not, which is the case the fast path must not claim.
        per_user = n_valid // batch
        mask[:, :per_user] = True
    else:
        raise ValueError(kind)
    return mask


# -------------------------------------------------------------- embed_vision


def _embed_vision_reference(vision_tokens, weight, eps):
    """HF ``Gemma4MultimodalEmbedder``: scaleless RMSNorm then bias-free Linear.

    ``Gemma4RMSNorm(with_scale=False)`` normalizes in float32 as
    ``x * (mean(x^2) + eps) ** -0.5``. ``weight`` is the on-device
    ``[vis_H, text_H]`` orientation, i.e. the HF Linear weight transposed.
    """
    x = vision_tokens.float()
    x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (x.to(torch.bfloat16).float() @ weight.float()).to(torch.bfloat16)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "n_valid",
    [260, 256, 1],
    ids=["unaligned_260", "tile_aligned_256", "single_token"],
)
def test_apply_embed_vision(mesh_device, reset_seeds, ensure_gc, n_valid):
    """Reference: HF ``Gemma4MultimodalEmbedder.forward``.

    ``n_valid=260`` is the real 31B count and is not a tile multiple, so it
    exercises the pad-to-32 / slice-back path.
    """
    vis_h, text_h = 128, 192
    vision_tokens = torch.randn(n_valid, vis_h, dtype=torch.bfloat16)
    # Same scale as the checkpoint's embedding_projection (std ~0.03).
    weight = (torch.randn(vis_h, text_h) * 0.03).to(torch.bfloat16)

    reference = _embed_vision_reference(vision_tokens, weight, EPS)

    model = _stub(
        mesh_device,
        embed_vision_weight=_from_torch(weight.reshape(1, 1, vis_h, text_h), mesh_device),
    )
    tt_in = _from_torch(vision_tokens, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_out = model._apply_embed_vision(tt_in)
    _check("embed_vision", _read(tt_out, mesh_device), reference, pcc=0.99, exact=False)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_apply_embed_vision_rejects_zero_projection(mesh_device, reset_seeds, ensure_gc):
    """A zero projection yields zero soft tokens -- the model then sees a blank
    image with no other failure signal, so pin the behaviour explicitly."""
    vis_h, text_h, n_valid = 128, 192, 260
    model = _stub(
        mesh_device,
        embed_vision_weight=_from_torch(torch.zeros(1, 1, vis_h, text_h, dtype=torch.bfloat16), mesh_device),
    )
    tt_in = _from_torch(torch.randn(n_valid, vis_h, dtype=torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = _read(model._apply_embed_vision(tt_in), mesh_device)
    assert out.abs().max().item() == 0.0, "zero projection must produce zero tokens"


# ------------------------------------------------------------------- scatter


def _placeholder_ids(kind, seq_len, n_image, batch=1):
    """Text ids with ``n_image`` image placeholders laid out per ``kind``."""
    ids = torch.randint(100, 5000, (batch, seq_len), dtype=torch.int32)
    if kind == "contiguous":
        ids[:, 20 : 20 + n_image] = IMAGE_TOKEN_ID
    elif kind == "two_runs":
        half = n_image // 2
        ids[:, 8 : 8 + half] = IMAGE_TOKEN_ID
        ids[:, 8 + half + 30 : 8 + half + 30 + (n_image - half)] = IMAGE_TOKEN_ID
    elif kind == "none":
        pass
    else:
        raise ValueError(kind)
    return ids


def _masked_scatter_reference(x, ids, vision_tokens, user_idx=0, offset=0):
    """HF ``inputs_embeds.masked_scatter(image_mask, image_features)``."""
    hidden = x.shape[-1]
    out = x.clone().reshape(-1, hidden)
    row_base = user_idx * ids.shape[-1] if out.shape[0] > ids.shape[-1] else 0
    pos = (ids[user_idx] == IMAGE_TOKEN_ID).nonzero().reshape(-1) + row_base
    out[pos] = vision_tokens[offset : offset + pos.numel()]
    return out.reshape(x.shape)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "kind, seq_len, n_image",
    [
        ("contiguous", 1024, 260),  # the 31B single-image prompt
        ("two_runs", 512, 64),  # two images in one prompt
        ("none", 256, 0),  # text-only must pass through untouched
    ],
    ids=["contiguous", "two_runs", "text_only"],
)
def test_scatter_vision_tokens(mesh_device, reset_seeds, ensure_gc, kind, seq_len, n_image):
    hidden = 192
    x = torch.randn(1, 1, seq_len, hidden, dtype=torch.bfloat16)
    ids = _placeholder_ids(kind, seq_len, n_image)
    vision_tokens = torch.randn(max(n_image, 1), hidden, dtype=torch.bfloat16)

    reference = _masked_scatter_reference(x, ids, vision_tokens)

    model = _stub(mesh_device)
    tt_out = model._scatter_vision_tokens(
        _from_torch(x, mesh_device),
        ids,
        _from_torch(vision_tokens, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    _check("scatter", _read(tt_out, mesh_device), reference, pcc=0.9999, exact=True)
    if n_image:
        assert model._vision_scatter_offset == n_image, "offset must advance by the placeholder count"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_scatter_vision_tokens_chunked(mesh_device, reset_seeds, ensure_gc):
    """Chunked prefill: one image spanning two chunks consumes rows in order."""
    chunk, hidden, n_first, n_second = 256, 192, 40, 24
    vision_tokens = torch.randn(n_first + n_second, hidden, dtype=torch.bfloat16)
    tt_vision = _from_torch(vision_tokens, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    ids_a = torch.randint(100, 5000, (1, chunk), dtype=torch.int32)
    ids_a[:, chunk - n_first :] = IMAGE_TOKEN_ID  # image starts at the chunk tail
    ids_b = torch.randint(100, 5000, (1, chunk), dtype=torch.int32)
    ids_b[:, :n_second] = IMAGE_TOKEN_ID  # and continues into the next chunk

    model = _stub(mesh_device)
    for label, ids, offset in (("chunk0", ids_a, 0), ("chunk1", ids_b, n_first)):
        x = torch.randn(1, 1, chunk, hidden, dtype=torch.bfloat16)
        reference = _masked_scatter_reference(x, ids, vision_tokens, offset=offset)
        tt_out = model._scatter_vision_tokens(_from_torch(x, mesh_device), ids, tt_vision)
        _check(f"scatter_{label}", _read(tt_out, mesh_device), reference, pcc=0.9999, exact=True)
    assert model._vision_scatter_offset == n_first + n_second


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("user_idx", [0, 1], ids=["user0", "user1"])
def test_scatter_vision_tokens_batched_user(mesh_device, reset_seeds, ensure_gc, user_idx):
    """Batched prefill: only the addressed user's rows may change."""
    batch, seq_len, hidden, n_image = 2, 256, 192, 48
    x = torch.randn(batch, 1, seq_len, hidden, dtype=torch.bfloat16)
    ids = _placeholder_ids("contiguous", seq_len, n_image, batch=batch)
    vision_tokens = torch.randn(n_image, hidden, dtype=torch.bfloat16)

    reference = _masked_scatter_reference(x, ids, vision_tokens, user_idx=user_idx)

    model = _stub(mesh_device)
    tt_out = model._scatter_vision_tokens(
        _from_torch(x, mesh_device),
        ids,
        _from_torch(vision_tokens, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
        user_idx=user_idx,
    )
    got = _read(tt_out, mesh_device)
    _check(f"scatter_user{user_idx}", got, reference, pcc=0.9999, exact=True)
    other = 1 - user_idx
    assert (got[other].float() - x[other].float()).abs().max().item() == 0.0, "other user's rows were modified"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_scatter_vision_tokens_rejects_count_mismatch(mesh_device, reset_seeds, ensure_gc):
    """Fewer soft tokens than placeholders must raise, not silently misalign."""
    seq_len, hidden, n_image = 256, 192, 48
    x = _from_torch(torch.randn(1, 1, seq_len, hidden, dtype=torch.bfloat16), mesh_device)
    ids = _placeholder_ids("contiguous", seq_len, n_image)
    short = _from_torch(
        torch.randn(n_image - 1, hidden, dtype=torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    model = _stub(mesh_device)
    with pytest.raises(AssertionError, match="image-token positions"):
        model._scatter_vision_tokens(x, ids, short)


# ---------------------------------------------------------------- end to end


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [vision_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_fusion_end_to_end(mesh_device, reset_seeds, ensure_gc):
    """compact -> embed_vision -> scatter against one torch reference.

    Shapes mirror the 31B single-image prompt: 280 pooled soft tokens of which
    260 are valid, projected into the text hidden size and spliced over the 260
    image placeholders of a 1024-token prompt.
    """
    length, n_valid, vis_h = 280, 260, 128
    seq_len, text_h = 1024, 192

    pooled = torch.randn(1, 1, length, vis_h, dtype=torch.bfloat16)
    mask = _compaction_mask("prefix", 1, length, n_valid)
    weight = (torch.randn(vis_h, text_h) * 0.03).to(torch.bfloat16)
    x = torch.randn(1, 1, seq_len, text_h, dtype=torch.bfloat16)
    ids = _placeholder_ids("contiguous", seq_len, n_valid)

    compacted_ref = pooled.reshape(length, vis_h)[mask.reshape(-1)]
    tokens_ref = _embed_vision_reference(compacted_ref, weight, EPS)
    reference = _masked_scatter_reference(x, ids, tokens_ref)

    model = _stub(
        mesh_device,
        embed_vision_weight=_from_torch(weight.reshape(1, 1, vis_h, text_h), mesh_device),
    )
    compacted = model._compact_valid_vision_tokens(_from_torch(pooled, mesh_device), mask)
    tokens = model._apply_embed_vision(compacted)
    tt_out = model._scatter_vision_tokens(_from_torch(x, mesh_device), ids, tokens)

    got = _read(tt_out, mesh_device)
    _check("fusion_e2e", got, reference, pcc=0.99, exact=False)
    # The placeholder rows must actually carry vision content, not text or zeros.
    pos = (ids[0] == IMAGE_TOKEN_ID).nonzero().reshape(-1)
    assert got[0, 0, pos].abs().max().item() > 0.0, "vision rows are zero"
    text_pos = (ids[0] != IMAGE_TOKEN_ID).nonzero().reshape(-1)
    text_diff = (got[0, 0, text_pos].float() - x[0, 0, text_pos].float()).abs().max().item()
    assert text_diff == 0.0, f"text rows were modified (max abs diff {text_diff})"
