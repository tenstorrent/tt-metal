# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision half of the encoder backend (doc/multimodal_plan.md, ``EncoderBackend``): the ViT on one 1x8 row.

    encoder = MiMoVisionEncoder(row_mesh, model_path)            # row_mesh: a (1, 8) MeshDevice / submesh
    encoder.warm()                                               # every bucket's programs, before trace capture
    image_embeds = encoder.embed_image(pixel_values, image_grid_thw)              # host bf16 [sum(t*h*w) / 4, 4096]
    video_embeds = encoder.embed_video(pixel_values_videos, video_grid_thw)

Inputs are the vLLM processor's fields as the plugin hands them over (any float dtype; served pixels are bf16), for
one or several items; outputs are the items' merged-token embeddings back to back, in item order (split them with
``t * h * w // 4`` rows per item). A video's rows are frame after frame, as its placeholder's ``is_embed`` rows.

Programs: every device pass is padded to one of ``encoder.buckets`` (``ROW_BUCKETS``: 1024 .. 32768 patch rows), so
the programs a call can run are a fixed set; ``warm()`` compiles them all (or a subset of buckets) and keeps their
binaries in DRAM (~11 MB per chip for all buckets). Calls after that compile nothing and allocate nothing that
outlives them.

Weights: converted from the checkpoint on first use into ``cache_dir`` (``MIMO_VISION_CACHE``/<checkpoint name> by
default; never the LLM's cache) under the same lock as the LLM's weights: 201 MB per chip in BF16, less with
``weight_dtype=ttnn.bfloat8_b``. Every call runs eagerly (no trace) and leaves no device tensor behind, so it can run
between the LLM's traced decode steps.
"""

import os

import torch

import ttnn

from ...reference.vision import as_grid_list
from .vit import DEFAULT_MAX_ROWS, DEFAULT_NUMERICS, NUMERICS, ROW_BUCKETS, TtMiMoVisionTransformer


def default_cache_dir(model_path):
    root = os.environ.get("MIMO_VISION_CACHE")
    if not root:
        return None
    return os.path.join(root, os.path.basename(os.path.normpath(model_path)))


class MiMoVisionEncoder:
    """``mesh_device``: one 1x8 row (the requesting replica's submesh). ``max_rows``: patch rows per device pass
    (whole frames; an image always runs in one pass; at most the largest bucket). ``weight_dtype``: matmul weights
    (bf16 or bfloat8_b). ``numerics``: a ``vit.NUMERICS`` preset; "balanced" (HiFi4 linears, fp32 residual stream,
    bf16 all-reduce) by default; "accurate" adds fp32 all-reduces, each followed by a device sync (see
    ``vit.VisionNumerics``). ``buckets``: pass row counts (None: one program set per input size, unbounded)."""

    modalities = ("image", "video")

    def __init__(
        self,
        mesh_device,
        model_path,
        cache_dir=None,
        max_rows=DEFAULT_MAX_ROWS,
        weight_dtype=ttnn.bfloat16,
        numerics=DEFAULT_NUMERICS,
        buckets=ROW_BUCKETS,
    ):
        cache_dir = cache_dir if cache_dir is not None else default_cache_dir(model_path)
        self.vit = TtMiMoVisionTransformer.from_checkpoint(
            mesh_device,
            model_path,
            cache_dir=cache_dir,
            max_rows=max_rows,
            weight_dtype=weight_dtype,
            numerics=NUMERICS[numerics] if isinstance(numerics, str) else numerics,
            buckets=buckets,
        )
        self.cfg = self.vit.cfg

    @property
    def buckets(self):
        """Pass row counts: every call runs the programs of these sizes only (inputs are padded up to one)."""
        return self.vit.buckets

    def warm(self, buckets=None):
        """Compile every bucket's programs (default: all) before the LLM captures its traces. Returns {rows: s}."""
        return self.vit.warm(buckets)

    def _embed(self, pixel_values, grid_thw):
        grids = as_grid_list(grid_thw)
        n = sum(t * h * w for t, h, w in grids)
        assert pixel_values.shape == (n, self.cfg.patch_dim), (tuple(pixel_values.shape), grids)
        with torch.no_grad():
            return self.vit(pixel_values, grids)

    def embed_image(self, pixel_values, grid_thw):
        """pixel_values [sum(t*h*w), 1536], image_grid_thw [items, 3] -> bf16 [sum(t*h*w) / 4, 4096] on host."""
        return self._embed(pixel_values, grid_thw)

    def embed_video(self, pixel_values_videos, grid_thw):
        """pixel_values_videos [sum(t*h*w), 1536], video_grid_thw [items, 3] -> bf16 [sum(t*h*w) / 4, 4096]."""
        return self._embed(pixel_values_videos, grid_thw)
