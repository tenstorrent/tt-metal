# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Strict loading of local or MMDetection3D-style BEVFormer checkpoints."""

from pathlib import Path


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a parameter dictionary")
    for key in ("state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    return checkpoint


def _strip_distributed_prefix(key):
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _mmdet3d_key(key):
    """Map the upstream detector/head ownership to this reference model's ownership."""
    key = _strip_distributed_prefix(key)
    mappings = (
        ("pts_bbox_head.transformer.encoder.", "transformer.encoder."),
        ("pts_bbox_head.transformer.can_bus_mlp.", "transformer.can_bus_mlp."),
        ("pts_bbox_head.transformer.level_embeds", "transformer.level_embeds"),
        ("pts_bbox_head.transformer.cams_embeds", "transformer.cams_embeds"),
        ("pts_bbox_head.transformer.decoder.", "head.decoder."),
        ("pts_bbox_head.transformer.reference_points.", "head.reference_points."),
        ("pts_bbox_head.query_embedding.", "head.query_embedding."),
        ("pts_bbox_head.cls_branches.", "head.cls_branches."),
        ("pts_bbox_head.reg_branches.", "head.reg_branches."),
        ("pts_bbox_head.bev_embedding.", "bev_embedding."),
        ("pts_bbox_head.positional_encoding.", "positional_encoding."),
    )
    for source, target in mappings:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def load_model_checkpoint(model, path: str | Path, checkpoint_format="mmdet3d"):
    """Load ``path`` into ``model`` with no silently random parameters.

    Shape mismatches, missing keys and unexpected keys are fatal. This is especially important for
    class-count, BEV-grid and variant mismatches that otherwise produce plausible-looking output.
    """
    import torch

    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older supported torch versions have no weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = _unwrap_checkpoint(checkpoint)
    if checkpoint_format == "local":
        state_dict = {_strip_distributed_prefix(key): value for key, value in state_dict.items()}
    elif checkpoint_format == "mmdet3d":
        # Training-only regression loss weights are saved by the upstream head but are not read by
        # inference and therefore do not exist in the reduced reference model.
        ignored = {"pts_bbox_head.code_weights"}
        state_dict = {
            _mmdet3d_key(key): value
            for key, value in state_dict.items()
            if _strip_distributed_prefix(key) not in ignored
        }
    else:
        raise ValueError(f"unknown checkpoint format {checkpoint_format!r}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"checkpoint {path} is incompatible with the selected model config; "
            "verify the variant, class count and checkpoint format\n"
            f"{error}"
        ) from error
    return model
