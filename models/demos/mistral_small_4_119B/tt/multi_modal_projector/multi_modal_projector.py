# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT implementation of Mistral3MultiModalProjector.

Pipeline (prefill-only — images are never processed during decode):
    vision_features
        → RMSNorm(vision_hidden_size)
        → spatial patch merge on host  [total_patches → total_merged_patches,
                                         vision_hidden → vision_hidden * spatial_merge_size²]
        → merging_layer linear         [vision_hidden * s² → vision_hidden]
        → linear_1                     [vision_hidden → text_hidden]
        → GELU
        → linear_2                     [text_hidden → text_hidden]
        → projected image tokens

Config dimensions (real model):
    vision_hidden_size   = 1024
    text_hidden_size     = 4096
    spatial_merge_size   = 2   →  s² = 4
    merging_layer input  = 1024 * 4 = 4096   output = 1024
    linear_1 input       = 1024              output = 4096
    linear_2 input       = 4096              output = 4096

All weights are replicated across the full mesh (shard_dims=(None, None)).
The module is not tensor-parallel because it runs once per image during prefill
and its weight tensors are small compared to the language model.

Runtime note: ``forward_prefill`` requires ``cfg["image_sizes"]`` to be set by
the caller to a list of ``(H, W)`` pixel-dimension tuples (one per image in
the batch) before each invocation.  All other cfg entries come from
``create_run_config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    RMSNormConfig,
)
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
    shard_and_save,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Mistral3MultiModalProjector(AbstractModule):
    """TT implementation of Mistral3MultiModalProjector.

    Bridges the Pixtral vision encoder output into the Mistral4 language-model
    token embedding space.  All weights are replicated across the mesh; no CCL
    collectives are needed.

    Only ``forward_prefill`` is implemented.  ``forward_decode`` raises
    ``NotImplementedError`` because vision features are spliced into the text
    embedding sequence exactly once, during the first prefill step.
    """

    # ─── Config accessors ────────────────────────────────────────────────

    @classmethod
    def _vision_hidden(cls, hf_config: PretrainedConfig) -> int:
        return int(hf_config.vision_config.hidden_size)

    @classmethod
    def _text_hidden(cls, hf_config: PretrainedConfig) -> int:
        return int(hf_config.text_config.hidden_size)

    @classmethod
    def _rms_eps(cls, hf_config: PretrainedConfig) -> float:
        return float(hf_config.text_config.rms_norm_eps)

    @classmethod
    def _spatial_merge_size(cls, hf_config: PretrainedConfig) -> int:
        return int(getattr(hf_config, "spatial_merge_size", 2))

    @classmethod
    def _patch_size(cls, hf_config: PretrainedConfig) -> int:
        return int(hf_config.vision_config.patch_size)

    @classmethod
    def _num_feature_layers(cls, hf_config: PretrainedConfig) -> int:
        vfl = hf_config.vision_feature_layer
        return 1 if isinstance(vfl, int) else len(vfl)

    # ─── Weight conversion ───────────────────────────────────────────────

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        """Convert HF multi_modal_projector weights to TTNN tensors.

        All four weight tensors are replicated to every device on the mesh.

        Expected keys in ``state_dicts[0]``:
            ``norm.weight``                        – [vision_hidden]
            ``patch_merger.merging_layer.weight``  – [vision_hidden, vision_hidden * s²]
            ``linear_1.weight``                    – [text_hidden, vision_hidden * num_feature_layers]
            ``linear_2.weight``                    – [text_hidden, text_hidden]
        """
        sd = state_dicts[0]
        assert sd is not None, "state_dicts[0] must not be None for Mistral3MultiModalProjector"

        tile = int(ttnn.TILE_SIZE)  # 32

        # ── norm.weight ──────────────────────────────────────────────────
        # [vision_hidden] → [1, 1, vision_hidden//tile, tile] row-major, replicated
        norm_w = sd["norm.weight"].to(torch.bfloat16)
        norm_weight_cfg = {
            "weight": shard_and_save(
                output_path / "norm.weight",
                norm_w.reshape(1, 1, -1, tile),
                shard_dims=(None, None),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        }

        # ── patch_merger.merging_layer.weight ────────────────────────────
        # [vision_hidden, vision_hidden*s²] → [1, 1, vision_hidden, vision_hidden*s²]
        # With transpose_b=True: linear maps [..., vision_hidden*s²] → [..., vision_hidden]
        merge_w = sd["patch_merger.merging_layer.weight"].to(torch.bfloat16)
        merging_layer_weight_cfg = {
            "input_tensor_b": shard_and_save(
                output_path / "merging_layer.input_tensor_b",
                merge_w.unsqueeze(0).unsqueeze(0).contiguous(),
                shard_dims=(None, None),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        }

        # ── linear_1.weight ──────────────────────────────────────────────
        # [text_hidden, vision_hidden * num_feature_layers] → [1, 1, text_hidden, vision_hidden * nfl]
        # With transpose_b=True: maps [..., vision_hidden*nfl] → [..., text_hidden]
        linear_1_w = sd["linear_1.weight"].to(torch.bfloat16)
        linear_1_weight_cfg = {
            "input_tensor_b": shard_and_save(
                output_path / "linear_1.input_tensor_b",
                linear_1_w.unsqueeze(0).unsqueeze(0).contiguous(),
                shard_dims=(None, None),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        }

        # ── linear_2.weight ──────────────────────────────────────────────
        # [text_hidden, text_hidden] → [1, 1, text_hidden, text_hidden]
        linear_2_w = sd["linear_2.weight"].to(torch.bfloat16)
        linear_2_weight_cfg = {
            "input_tensor_b": shard_and_save(
                output_path / "linear_2.input_tensor_b",
                linear_2_w.unsqueeze(0).unsqueeze(0).contiguous(),
                shard_dims=(None, None),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        }

        return {
            "norm": norm_weight_cfg,
            "merging_layer": merging_layer_weight_cfg,
            "linear_1": linear_1_weight_cfg,
            "linear_2": linear_2_weight_cfg,
        }

    # ─── Model configs ───────────────────────────────────────────────────

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        """Static operator configs for prefill mode.

        Stores ``spatial_merge_size``, ``vision_hidden_size``, and ``patch_size``
        as plain integers in the config dict so ``forward_prefill`` can read them
        without accessing ``hf_config`` at runtime.
        """
        mesh_stub = MeshDeviceStub(mesh_device.shape)

        return {
            "norm": RMSNormConfig(
                epsilon=cls._rms_eps(hf_config),
                weight=FromWeightConfig(mesh_stub),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
            ),
            "merging_layer": LinearConfig(
                input_tensor_b=FromWeightConfig(mesh_stub),
                transpose_b=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "linear_1": LinearConfig(
                input_tensor_b=FromWeightConfig(mesh_stub),
                transpose_b=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "linear_2": LinearConfig(
                input_tensor_b=FromWeightConfig(mesh_stub),
                transpose_b=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            # Runtime-readable scalar parameters
            "spatial_merge_size": cls._spatial_merge_size(hf_config),
            "vision_hidden_size": cls._vision_hidden(hf_config),
            "patch_size": cls._patch_size(hf_config),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        **kwargs,
    ) -> ModelDecodeConfig:
        raise NotImplementedError(
            "Mistral3MultiModalProjector.decode_model_config: " "vision projection only runs during prefill."
        )

    # ─── State ───────────────────────────────────────────────────────────

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        *args,
        **kwargs,
    ) -> ModelState:
        # MESH_DEVICE_STATE_DICT_KEY is consumed by create_run_config to resolve
        # MeshDeviceStub references and is then removed from the final cfg dict.
        # Store mesh_device under a second key so forward_prefill can retrieve it.
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "projector_mesh_device": mesh_device,
        }

    # ─── Spatial merge helper ─────────────────────────────────────────────

    @classmethod
    def _spatial_merge_on_host(
        cls,
        x: ttnn.Tensor,
        image_sizes: List[Tuple[int, int]],
        patch_size: int,
        spatial_merge_size: int,
        vision_hidden_size: int,
        mesh_device: ttnn.MeshDevice,
    ) -> ttnn.Tensor:
        """Merge spatial_merge_size×spatial_merge_size neighbouring patches on CPU.

        Implements the same logic as ``Mistral3PatchMerger.forward``:
          1. For each image, reshape the flat patch sequence into a (h, w) grid.
          2. Apply a 2-D unfold with kernel=stride=spatial_merge_size to create
             non-overlapping blocks.
          3. Concatenate the s² patches in each block into a single embedding.

        Args:
            x:                 TTNN tensor [1, 1, total_patches, vision_hidden],
                               replicated across all devices.
            image_sizes:       List of (H_pixels, W_pixels) for each image.
            patch_size:        Vision encoder patch size in pixels.
            spatial_merge_size: Number of patches to merge per spatial dimension.
            vision_hidden_size: Vision encoder hidden dimension (= x.shape[-1]).
            mesh_device:       Used to extract the raw torch tensor and to
                               re-upload the merged result.

        Returns:
            TTNN tensor [1, 1, total_merged_patches, vision_hidden * s²],
            replicated across all devices, in bfloat16 / TILE_LAYOUT / DRAM.
        """
        R, C = tuple(mesh_device.shape)

        # Pull to host.  For a replicated [1,1,N,H] tensor on a (R,C) mesh,
        # ConcatMesh2dToTensor produces [R, 1, N, H*C].  We extract the first
        # replica: row 0, columns 0..vision_hidden_size-1.
        x_torch = ttnn.to_torch(
            x,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=(R, C)),
        )
        # x_torch: [R, 1, N, H*C]  (replicated → all rows/cols identical)
        x_torch = x_torch[0, 0, :, :vision_hidden_size]  # [N, H]

        # Compute per-image patch grid sizes.
        grid_sizes = [(h // patch_size, w // patch_size) for h, w in image_sizes]
        tokens_per_image = [gh * gw for gh, gw in grid_sizes]
        d = vision_hidden_size
        s = spatial_merge_size

        merged_parts: list[torch.Tensor] = []
        offset = 0
        for idx, (gh, gw) in enumerate(grid_sizes):
            n_tokens = tokens_per_image[idx]
            image_tokens = x_torch[offset : offset + n_tokens]  # [gh*gw, d]
            offset += n_tokens

            # Reshape flat patch sequence → spatial grid [1, d, gh, gw]
            image_grid = image_tokens.view(gh, gw, d).permute(2, 0, 1).unsqueeze(0)

            # Non-overlapping unfold: groups each s×s neighbourhood into one row.
            # Output: [1, d*s², (gh//s)*(gw//s)]
            grid = torch.nn.functional.unfold(image_grid.float(), kernel_size=s, stride=s)
            # Reshape → [(gh//s)*(gw//s), d*s²]
            grid = grid.view(d * s * s, -1).t().to(torch.bfloat16)
            merged_parts.append(grid)

        merged = torch.cat(merged_parts, dim=0)  # [total_merged, d*s²]
        merged = merged.unsqueeze(0).unsqueeze(0)  # [1, 1, total_merged, d*s²]

        # Pad sequence length to a multiple of TILE_SIZE so TTNN can tile it.
        tile = int(ttnn.TILE_SIZE)
        total_merged, ds2 = merged.shape[2], merged.shape[3]
        pad_seq = (tile - total_merged % tile) % tile
        if pad_seq > 0:
            merged = torch.nn.functional.pad(merged, (0, 0, 0, pad_seq))

        return ttnn.from_torch(
            merged,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    # ─── Forward ─────────────────────────────────────────────────────────

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Prefill forward: norm → spatial merge → merging_layer → linear_1 → GELU → linear_2.

        The caller **must** inject runtime image metadata into the cfg dict before
        calling this method::

            cfg["image_sizes"] = [(H0, W0), (H1, W1), ...]  # pixels per image

        Args:
            x:   Vision encoder output [1, 1, total_patches, vision_hidden_size],
                 replicated across all devices.
            cfg: Run config produced by ``create_run_config``, augmented with
                 ``image_sizes`` by the caller.

        Returns:
            Projected image tokens [1, 1, total_merged_patches, text_hidden_size],
            replicated across all devices.
        """
        image_sizes: List[Tuple[int, int]] = cfg["image_sizes"]
        patch_size = int(cfg["patch_size"])
        s = int(cfg["spatial_merge_size"])
        vision_h = int(cfg["vision_hidden_size"])
        mesh_device: ttnn.MeshDevice = cfg["projector_mesh_device"]

        # 1. RMSNorm over vision_hidden_size dimension.
        x = ttnn.rms_norm(x, program_config=ttnn.LayerNormDefaultProgramConfig(), **cfg["norm"])

        # 2. Spatial patch merge (host-side).  Returns [1, 1, merged_patches, vision_h*s²].
        x_merged = cls._spatial_merge_on_host(x, image_sizes, patch_size, s, vision_h, mesh_device)
        ttnn.deallocate(x)

        # 3. merging_layer: [merged_patches, vision_h*s²] → [merged_patches, vision_h]
        x_proj = ttnn.linear(x_merged, **cfg["merging_layer"])
        ttnn.deallocate(x_merged)

        # 4. linear_1: [merged_patches, vision_h] → [merged_patches, text_h]
        x_l1 = ttnn.linear(x_proj, **cfg["linear_1"])
        ttnn.deallocate(x_proj)

        # 5. GELU activation.
        x_act = ttnn.gelu(x_l1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_l1)

        # 6. linear_2: [merged_patches, text_h] → [merged_patches, text_h]
        x_out = ttnn.linear(x_act, **cfg["linear_2"])
        ttnn.deallocate(x_act)

        return x_out

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        raise NotImplementedError(
            "Mistral3MultiModalProjector.forward_decode: " "vision projection only runs during prefill, not decode."
        )
