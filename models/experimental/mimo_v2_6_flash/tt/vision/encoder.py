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
binaries in DRAM (~20 MB per chip for all buckets). Calls after that compile nothing.

Traces: passes of the buckets up to ``trace_max_rows`` (``MIMO_VIT_TRACE_MAX_ROWS``, default 4096: images up to
1024x1024; 0 turns tracing off) replay a trace instead of enqueueing ~900 ops from the host (bit-identical; a
448x448 image takes 18.8 ms end to end on bhg2, 16.3 ms of it device time). ``warm()`` also allocates their
persistent inputs and outputs (29 MiB per chip for the 9 buckets), before the LLM's traces are captured;
``capture_traces()`` captures them after the LLM's (a bucket not captured by then is captured on its first call). They
take 71 MiB of trace region per chip (``trace_info()``); a capture that does not fit leaves its bucket eager. Larger
passes run eagerly and leave no device tensor behind.

Weights: converted from the checkpoint on first use into ``cache_dir`` (``MIMO_VISION_CACHE``/<checkpoint name> by
default; never the LLM's cache) under the same lock as the LLM's weights: 201 MB per chip in BF16, less with
``weight_dtype=ttnn.bfloat8_b``.
"""

import os

import torch

import ttnn

from ...reference.vision import as_grid_list
from .vit import DEFAULT_MAX_ROWS, DEFAULT_NUMERICS, NUMERICS, ROW_BUCKETS, TRACE_MAX_ROWS, TtMiMoVisionTransformer

# Largest traced bucket (patch rows); 0: no traces.
VIT_TRACE_MAX_ROWS = int(os.environ.get("MIMO_VIT_TRACE_MAX_ROWS", str(TRACE_MAX_ROWS)))


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
    ``vit.VisionNumerics``; not traceable). ``buckets``: pass row counts (None: one program set per input size,
    unbounded, no traces). ``trace_max_rows``: the largest traced bucket (0: none)."""

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
        trace_max_rows=None,
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
        limit = VIT_TRACE_MAX_ROWS if trace_max_rows is None else trace_max_rows
        self.trace_buckets = tuple(b for b in self.vit.buckets if b <= limit) if self.vit.traceable() else ()

    @property
    def buckets(self):
        """Pass row counts: every call runs the programs of these sizes only (inputs are padded up to one)."""
        return self.vit.buckets

    def warm(self, buckets=None):
        """Compile every bucket's programs (default: all) and allocate the traced buckets' persistent buffers, before
        the LLM captures its traces. Returns {rows: s}."""
        times = self.vit.warm(buckets)
        self.vit.prepare_traces(self.trace_buckets)
        return times

    def capture_traces(self):
        """Capture the traced buckets' passes (after warm(), once the LLM's traces are captured). Returns {rows: trace
        region bytes per chip}.

        ``warm()`` must have run, before the LLM captured its traces: it allocates the traced buckets' persistent
        pixel / input / output buffers. Allocating them here instead, after the LLM's captures, would put them where
        an LLM trace replay writes its temporaries, and the vision inputs and outputs would be overwritten under a
        running pass. So this refuses rather than allocate."""
        missing = [rows for rows in self.trace_buckets if rows not in self.vit.prepared_traces()]
        if missing:
            raise RuntimeError(
                f"MiMoVisionEncoder.capture_traces() before warm(): the traced buckets {missing} have no persistent "
                "buffers yet, and they must be allocated before the LLM captures its traces (call warm() first)"
            )
        return self.vit.capture_traces()

    def trace_info(self):
        """{rows: (captured, trace region bytes per chip, recaptures)}."""
        return self.vit.trace_info()

    def release_traces(self):
        self.vit.release_traces()

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
