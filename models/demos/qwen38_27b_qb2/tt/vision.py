# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 vision encoder using existing Qwen3.5 blocks and explicit frame masks.

This feature is hardware-unqualified. It is constructed only by the separate
multimodal adapter; the qualified text adapter allocates no vision weights.
"""

import hashlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.vision.functional import qwen3_5_vision_transformer_preprocess
from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs
from models.demos.qwen38_27b_qb2.tt.multimodal import validate_grid, vision_boundaries
from models.demos.qwen38_27b_qb2.tt.vision_weights import load_reference_vision
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

_ARGS_ENV_LOCK = threading.Lock()


@contextmanager
def _args_environment(snapshot, cache_root):
    # Shared ModelArgs still reads these environment variables. Bound this
    # startup-only compatibility section and restore the exact previous values.
    with _ARGS_ENV_LOCK:
        original = {name: os.environ.get(name) for name in ("HF_MODEL", "TT_CACHE_PATH")}
        os.environ.update(HF_MODEL=str(snapshot), TT_CACHE_PATH=str(cache_root))
        try:
            yield
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class Qwen38VisionEncoder:
    def __init__(self, model, config, *, max_patches=32768):
        if tuple(model.mesh.shape) != (1, 4):
            raise ValueError("Vision encoder requires the same TP4 mesh as the text model")
        if config.vision_config.out_hidden_size != model.config.hidden_size:
            raise ValueError("Vision merger width differs from the text embedding width")
        if getattr(config.vision_config, "deepstack_visual_indexes", []):
            raise ValueError("DeepStack visual injection is unsupported by this Qwen3.8 checkpoint adapter")
        self.mesh = model.mesh
        self.config = config.vision_config
        self.max_patches = max_patches
        snapshot = Path(model.snapshot)
        identity = hashlib.sha256(
            str(snapshot.resolve()).encode()
            + (snapshot / "config.json").read_bytes()
            + (snapshot / "model.safetensors.index.json").read_bytes()
        ).hexdigest()[:20]
        # Distinct worker dirs prevent eight replicas from racing on tensorbin
        # creation. Everything stays under the host's task-local cache root.
        cache = Path(os.getenv("QWEN_VISION_CACHE_DIR", os.getenv("TT_METAL_CACHE", "/tmp")))
        cache = cache / "qwen38-vision" / identity / str(os.getpid())
        with _args_environment(snapshot, cache):
            args = VisionModelArgs(model.mesh, max_batch_size=1, max_seq_len=max_patches)
        self.reference = load_reference_vision(snapshot, self.config)
        self.wrapper = DropInVisionTransformer(self.reference, args, dtype=ttnn.bfloat8_b, tt_ccl=model.ccl)
        self.tower = self.wrapper.tt_model

    @torch.inference_mode()
    def __call__(self, pixels, grid):
        grid = validate_grid(grid, merge=self.config.spatial_merge_size)
        actual = int(grid.prod(-1).sum())
        if actual > self.max_patches:
            raise ValueError(f"Visual input has {actual} patches; maximum is {self.max_patches}")
        patch_width = self.config.in_channels * self.config.temporal_patch_size * self.config.patch_size**2
        if pixels.ndim != 2 or tuple(pixels.shape) != (actual, patch_width):
            raise ValueError("Visual patch tensor shape disagrees with checkpoint patch geometry")
        padded = ((actual + 2047) // 2048) * 2048
        boundaries = vision_boundaries(grid, padded, merge=self.config.spatial_merge_size)
        patches = self.reference.patch_embed(pixels.to(device="cpu"))
        position_embeddings = self.reference.fast_pos_embed_interpolate(grid)
        patches = patches + position_embeddings.to(patches.dtype)
        _, (cos, sin) = qwen3_5_vision_transformer_preprocess(
            seq_len=actual,
            grid_thw=grid,
            head_dim=self.config.hidden_size // self.config.num_heads,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        cos, sin = convert_rope_style_hf_to_meta(cos, sin)
        rot_mats = [
            ttnn.from_torch(
                torch.nn.functional.pad(value, (0, 0, 0, padded - actual), value=padding)[None, None],
                device=self.mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            for value, padding in ((cos, 1), (sin, 0))
        ]
        windows = ttnn.from_torch(
            boundaries,
            device=self.mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # The inherited DropIn forward discarded cu_seqlens. Call the same
        # blocks with the supported native window argument so real frames never
        # attend another frame or padded keys. Padding owns a separate window.
        x = self.tower.prepare_input(patches, padded)
        for block in self.tower.blocks:
            x = block(x, rot_mats=rot_mats, cu_window_seqlens=windows)
        merged = self.tower.patch_merger(x[:, :, :actual, :])
        host = ttnn.to_torch(merged, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
        expected = actual // self.config.spatial_merge_size**2
        if host.numel() != expected * self.config.out_hidden_size:
            raise ValueError("Vision output cannot be mapped to the expected merged tokens")
        return host.reshape(expected, self.config.out_hidden_size).to(torch.bfloat16).clone()
