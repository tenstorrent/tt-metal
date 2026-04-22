# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Vision Model Configuration for Dots OCR.

This configures the full TTNN vision stack for Dots.mocr:
- 42 transformer layers
- Hidden size: 1536
- 12 attention heads
- Patch size: 14
- Spatial merge size: 2
- Post-norm RMSNorm architecture
"""

from __future__ import annotations

import math
import types
from typing import Optional

from loguru import logger

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.model_config import DotsModelArgs
from models.demos.dots_ocr.tt.vision_config_dataclass import DotsVisionConfig


def _nearest_multiple(x: int, multiple_of: int) -> int:
    return int(math.ceil(x / multiple_of) * multiple_of)


class DotsVisionModelArgs(DotsModelArgs):
    """
    Vision-specific model arguments for Dots OCR.

    Extends :class:`DotsModelArgs` so it inherits the Dots-specific ``_set_hf_params``
    override (which enables ``trust_remote_code=True`` and knows how to merge the Dots
    vision config), then adds vision-specific convenience attributes
    (``vision_dim``, ``vision_n_heads``, ``patch_size``, etc.) extracted from the HF
    ``DotsVisionConfig``.
    """

    def __init__(
        self,
        *args,
        hf_config=None,
        mesh_device=None,
        **kwargs,
    ):
        # Forward ``mesh_device`` positionally (including ``None`` for CPU-only test paths).
        # ``DotsModelArgs`` handles ``trust_remote_code_hf`` for us. For vision-only / smoke
        # paths that do not pass ``hf_config`` and don't set ``HF_MODEL``, seed it to the
        # canonical Dots OCR checkpoint so the parent can resolve it (mirrors the convention
        # already used by ``DotsModelArgs`` itself).
        import os

        if hf_config is None and os.getenv("HF_MODEL") is None:
            os.environ["HF_MODEL"] = "rednote-hilab/dots.mocr"

        if not args:
            super().__init__(mesh_device, hf_config=hf_config, **kwargs)
        else:
            super().__init__(*args, hf_config=hf_config, **kwargs)

        if hf_config is None:
            hf_config = getattr(self, "hf_config", None)
        self.vision_config = self._extract_vision_config(hf_config)

        # Core vision dimensions
        self.vision_dim = self.vision_config.hidden_size
        self.vision_n_heads = self.vision_config.num_attention_heads
        self.vision_n_kv_heads = self.vision_config.num_attention_heads  # Same for Dots
        self.vision_head_dim = self.vision_dim // self.vision_n_heads

        # MLP dimensions
        self.vision_intermediate_size = self.vision_config.intermediate_size
        self.vision_mlp_dim = self.vision_intermediate_size

        # Intermediate MLP width (``vision_config.intermediate_size``), used for tile padding.
        self.vision_unpadded_hidden_dim = self.vision_intermediate_size

        # Patching configuration
        self.patch_size = self.vision_config.patch_size
        self.spatial_merge_size = self.vision_config.spatial_merge_size
        self.temporal_patch_size = getattr(self.vision_config, "temporal_patch_size", 1)

        # Normalization
        self.rms_norm_eps = self.vision_config.rms_norm_eps
        self.post_norm = getattr(self.vision_config, "post_norm", True)

        # Other parameters
        self.initializer_range = getattr(self.vision_config, "initializer_range", 0.02)
        self.init_merger_std = getattr(self.vision_config, "init_merger_std", 0.02)

        # Convenience aliases used by tests and the VisionTransformerTT driver.
        self.num_hidden_layers = self.vision_config.num_hidden_layers

        # TTNN-specific configurations
        self.tile_size = 32  # Standard TTNN tile size

        # Pad hidden dim to a tile multiple for TTNN matmuls
        num_dev = max(getattr(self, "num_devices", 1), 1)
        self.vision_hidden_dim = _nearest_multiple(self.vision_unpadded_hidden_dim, self.tile_size * num_dev)
        if self.vision_hidden_dim != self.vision_unpadded_hidden_dim:
            logger.info(
                f"Vision: padding hidden dim from {self.vision_unpadded_hidden_dim} to {self.vision_hidden_dim}"
            )

        self.vision_padded_head_dim = math.ceil(self.vision_head_dim / self.tile_size) * self.tile_size

        if self.vision_padded_head_dim != self.vision_head_dim:
            logger.info(f"Vision: padding head dim from {self.vision_head_dim} to {self.vision_padded_head_dim}")

        # Aggregate QKV size if using padded head dims (legacy bring-up field)
        self.vision_qkv_size = self.vision_padded_head_dim * (2 * self.vision_n_kv_heads + self.vision_n_heads)

        # Long-sequence / prefill shims (optional; used by some matmul configs)
        self.MAX_QKV_MM_SEQ_LEN = getattr(self, "MAX_QKV_MM_SEQ_LEN", 1024)
        self.min_kv_prefill_shard_seqlen = getattr(self, "min_kv_prefill_shard_seqlen", 0)
        self.ccl_dtype = getattr(self, "ccl_dtype", None)

        # Optional prefill program config for large vision sequences.
        try:
            num_rows = lambda seq_len: min(seq_len, 2048)
            k_dim = self.vision_dim
            n_dim = self.vision_dim
            self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=num_rows(seq_len),
                k=k_dim,
                n=n_dim,
                grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
                in0_block_w=max(1, self.vision_dim // 1024),
                fuse_batch=seq_len <= 1024,
            )
        except Exception:
            pass

        # MLP dtype hint (``bfp4_mlp``) for paths that read ``args.optimizations``.
        opt = getattr(self, "optimizations", None)
        if opt is None or not hasattr(opt, "bfp4_mlp"):
            self.optimizations = types.SimpleNamespace(bfp4_mlp=False)

        # Some TT helpers read ``model_config["DECODERS_OPTIMIZATIONS"]`` per layer; Dots text config
        # may not index vision layer ids. Provide a small shim for vision bring-up.
        ttnn = get_ttnn()

        class _VisionOptimShim:
            def get_tensor_dtype(self, decoder_id, tensor):
                # Prefer bf8 weights for throughput; activations/outputs can be bf8 as well.
                if ttnn is None:
                    return None
                return ttnn.bfloat8_b

            def get_math_fidelity(self, decoder_id, op, configuration):
                # Use the highest fidelity config available on the configuration.
                return getattr(configuration, "compute_kernel_config_hifi4", None) or getattr(
                    configuration, "compute_kernel_config_hifi2_fp16", None
                )

        try:
            self.model_config["DECODERS_OPTIMIZATIONS"] = _VisionOptimShim()
        except Exception:
            pass

        logger.info(
            f"DotsVisionModelArgs: dim={self.vision_dim}, layers={self.vision_config.num_hidden_layers}, "
            f"heads={self.vision_n_heads}, patch_size={self.patch_size}, "
            f"spatial_merge={self.spatial_merge_size}"
        )

    def _extract_vision_config(self, hf_config):
        """Extract or create vision configuration from HF config."""
        if hf_config is None:
            # Default configuration for Dots.mocr
            return DotsVisionConfig()

        # Try to get vision_config from HF config
        if hasattr(hf_config, "vision_config") and hf_config.vision_config is not None:
            vc = hf_config.vision_config
            return DotsVisionConfig(
                hidden_size=getattr(vc, "hidden_size", 1536),
                num_hidden_layers=getattr(vc, "num_hidden_layers", 42),
                num_attention_heads=getattr(vc, "num_attention_heads", 12),
                intermediate_size=getattr(vc, "intermediate_size", 4224),
                patch_size=getattr(vc, "patch_size", 14),
                spatial_merge_size=getattr(vc, "spatial_merge_size", 2),
                rms_norm_eps=getattr(vc, "rms_norm_eps", 1e-5),
                initializer_range=getattr(vc, "initializer_range", 0.02),
                init_merger_std=getattr(vc, "init_merger_std", 0.02),
                post_norm=getattr(vc, "post_norm", True),
                num_channels=getattr(vc, "num_channels", 3),
                temporal_patch_size=getattr(vc, "temporal_patch_size", 1),
            )

        # Fallback to default
        logger.warning("Could not extract vision config from HF, using defaults")
        return DotsVisionConfig()

    def get_state_dict_prefix(self, module_name: str, layer_num: Optional[int] = None) -> str:
        """
        Get state dict prefix for vision components.

        Keys follow the HF Dots ``vision_tower`` layout (``feed_forward``, ``self_attn`` → ``attention`` in maps).
        """
        layer_prefix = f"vision_tower.blocks.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "VisionAttention": "attention",
            "VisionBlock": "",
            "VisionTransformer": "vision_tower",
            "PatchMerger": "vision_tower.merger",
            "PatchEmbed": "vision_tower.patch_embed",
            "": "",
        }
        base = layer_prefix + module_map.get(module_name, "")
        # Most call sites expect prefixes to end with "." so they can append "weight"/"bias" keys.
        # Keep "" unchanged for modules that intentionally return empty prefix.
        if base and not base.endswith("."):
            base = base + "."
        return base

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated: bool = False):
        """
        Prepare vision tensor for TTNN prefill.

        For vision, we typically want this in DRAM for the large activations.
        """
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("TTNN is not available (import failed or incompatible install).")

        x_1BSH = x_bsh.unsqueeze(0) if x_bsh.dim() == 3 else x_bsh

        memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return xs_1BSH

    def is_distributed_norm(self, mode: str) -> bool:
        """Vision typically doesn't use distributed norm."""
        return False


# Convenience function to create vision args
def create_dots_vision_args(mesh_device, hf_config=None, **kwargs):
    """Create DotsVisionModelArgs with default settings, overridable via ``**kwargs``."""
    kwargs.setdefault("max_batch_size", 1)
    kwargs.setdefault("max_seq_len", 2048)  # typical vision sequence length
    args = DotsVisionModelArgs(
        mesh_device=mesh_device,
        hf_config=hf_config,
        **kwargs,
    )
    args.mesh_device = mesh_device
    return args
