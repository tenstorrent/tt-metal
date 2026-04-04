# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT-Metal model configuration adapter for Qwen3-Coder-Next.

Bridges the HF-oriented Qwen3CoderNextConfig with tt_transformers patterns,
providing device-aware sharding, layer-type dispatch, and CCL topology.
Does NOT modify the base tt_transformers ModelArgs -- instead provides a
standalone config that downstream TT modules (decoder, attention, MoE) consume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# Layer type constants used by model.py for hybrid dispatch
LAYER_TYPE_DELTANET = "linear_attention"
LAYER_TYPE_GQA = "full_attention"


@dataclass
class Qwen3CoderNextTTConfig(Qwen3CoderNextConfig):
    """TT-Metal-aware configuration extending the HF config parser.

    Adds device topology, expert sharding, batch partitioning, and
    layer-type classification needed by the TT model implementation.

    Compatibility flags:
        is_qwen3_coder_next: Identifies this model for Qwen3-Coder-Next-specific paths.
        is_qwen35: Enables reuse of DeltaNet/GatedAttention config parsing from Qwen3.5.
        is_mixture_of_experts: Set False to suppress Mixtral-style MoE auto-detection
            in tt_transformers ModelArgs.load_state_dict (which infers MoE from ".experts."
            keys and assumes Mixtral structure). We handle MoE routing ourselves.
    """

    # --- Compatibility flags ---
    is_qwen3_coder_next: bool = True
    is_qwen35: bool = True
    is_mixture_of_experts: bool = False

    # --- Device topology (set by from_pretrained) ---
    num_devices: int = 1
    mesh_device: object = field(default=None, repr=False)
    device_name: str = ""

    # --- Sequence / batch limits ---
    max_batch_size: int = 32
    max_seq_len: int = 8192

    # --- Derived sharding (populated in __post_init__) ---
    experts_per_device: int = 0
    batch_size_per_device_group: int = 0
    layer_types: List[str] = field(default_factory=list)
    tile_padded_batch_rows: int = 32

    # ------------------------------------------------------------------
    # Layer-type helpers
    # ------------------------------------------------------------------

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """True if ``layer_idx`` uses Gated DeltaNet (linear) attention."""
        return not self.is_full_attention_layer(layer_idx)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """True if ``layer_idx`` uses full GQA (softmax) attention.

        Pattern: every ``full_attention_interval``-th layer (0-indexed from the end
        of each interval) is GQA.  i.e. indices 3, 7, 11, ... for interval=4.
        """
        return layer_idx % self.full_attention_interval == (self.full_attention_interval - 1)

    def get_layer_type(self, layer_idx: int) -> str:
        """Return the layer-type string for ``layer_idx``."""
        if self.is_full_attention_layer(layer_idx):
            return LAYER_TYPE_GQA
        return LAYER_TYPE_DELTANET

    # ------------------------------------------------------------------
    # CCL topology
    # ------------------------------------------------------------------

    def ccl_topology(self):
        """Return the CCL topology for collective ops.

        Ring for T3K (8 devices) or multi-chip >= 8; Linear for smaller meshes;
        None for single device.
        """
        try:
            import ttnn

            if self.num_devices >= 8:
                return ttnn.Topology.Ring
            if self.num_devices > 1:
                return ttnn.Topology.Linear
        except ImportError:
            pass
        return None

    # ------------------------------------------------------------------
    # State-dict prefix (matches Qwen35 / tt_transformers convention)
    # ------------------------------------------------------------------

    @staticmethod
    def get_state_dict_prefix(module_name: str, layer_num: Optional[int]) -> str:
        """Return the HF state-dict key prefix for a given module.

        Follows the same convention as Qwen3.5 / tt_transformers so that
        weight-loading utilities can locate tensors consistently.
        """
        module_map = {
            "GatedDeltaNet": "linear_attn",
            "GatedAttention": "attention",
            "Attention": "attention",
            "MoE": "mlp",
            "MLP": "mlp",
            "RMSNorm": "",
        }
        prefix = module_map.get(module_name, module_name.lower())
        if layer_num is not None:
            if prefix:
                return f"layers.{layer_num}.{prefix}"
            return f"layers.{layer_num}"
        return prefix

    # ------------------------------------------------------------------
    # Weight cache path helper
    # ------------------------------------------------------------------

    def weight_cache_path(self, dtype=None) -> Optional[Path]:
        """Return the weight cache directory, or None if no cache configured."""
        cache_root = os.getenv("TT_CACHE_PATH")
        if cache_root is None:
            return None
        suffix = ""
        if dtype is not None:
            suffix = f"/{dtype}"
        device_tag = self.device_name or f"{self.num_devices}dev"
        return Path(cache_root) / "qwen3_coder_next" / device_tag / suffix

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self):
        """Compute derived fields after dataclass init."""
        # Parent validation (Qwen3CoderNextConfig.__post_init__ calls validate + logs)
        super().__post_init__()

        # Expert sharding across devices
        if self.num_devices > 0:
            self.experts_per_device = self.num_experts // self.num_devices
        else:
            self.experts_per_device = self.num_experts

        # Batch-parallel KV: split batch across devices
        if self.num_devices > 0:
            self.batch_size_per_device_group = max(self.max_batch_size // self.num_devices, 1)
        else:
            self.batch_size_per_device_group = self.max_batch_size

        # Build layer_types list for model.py hybrid dispatch
        self.layer_types = [self.get_layer_type(i) for i in range(self.num_hidden_layers)]

        # Expose dim / n_layers / n_heads / n_kv_heads aliases that tt_transformers modules expect
        self.dim = self.hidden_size
        self.n_layers = self.num_hidden_layers
        self.n_heads = self.num_attention_heads
        self.n_kv_heads = self.num_key_value_heads
        self.norm_eps = self.rms_norm_eps
        # linear_value_head_dim = 128 from parent (NOT head_dim=256 which is GQA only)
        # Verified: in_proj_qkvz shape (12288,2048) = 2*key_dim(2048) + 2*value_dim(4096), value_dim=32*128
        self.partial_rotary_factor = self.partial_rotary_factor
        self.rope_theta = self.rope_theta
        self.rope_scaling = None

        # Padded batch rows (tile-aligned)
        self.tile_padded_batch_rows = max(self.max_batch_size, 32)
        self.tile_size = 32

        # Additional flags expected by tt_transformers modules
        self.dummy_weights = False
        self.is_distributed_norm = self.num_devices > 1
        self.is_multichip = self.num_devices > 1
        self.is_galaxy = self.num_devices == 32
        self.rms_norm_add_unit_offset = False

        # Model config dict (expected by TTNN layers for memory configs)
        self.model_config = {}

    # ------------------------------------------------------------------
    # Interface methods expected by TTNN layers (gated_deltanet.py, etc.)
    # ------------------------------------------------------------------

    def get_model_config(self):
        """Return model config dict for TTNN memory/program configs."""
        return self.model_config

    def get_residual_mem_config(self, mode=None):
        """Return memory config for residual tensors."""
        try:
            import ttnn

            return ttnn.DRAM_MEMORY_CONFIG
        except ImportError:
            return None

    def get_norm_config(self, mode=None):
        """Return memory config for norm operations."""
        try:
            import ttnn

            return ttnn.DRAM_MEMORY_CONFIG
        except ImportError:
            return None

    def create_dram_sharded_mem_config(self, *args, **kwargs):
        """Return DRAM sharded memory config. Falls back to interleaved DRAM."""
        try:
            import ttnn

            return ttnn.DRAM_MEMORY_CONFIG
        except ImportError:
            return None

    def dram_matmul_config(self, *args, **kwargs):
        """Return matmul program config for DRAM-sharded weights. None = auto-select."""
        return None

    @property
    def attn_input_grid(self):
        """Core grid for attention input. Auto-configured based on device count."""
        try:
            import ttnn

            if self.num_devices >= 8:
                return ttnn.CoreGrid(x=8, y=4)
            return ttnn.CoreGrid(x=4, y=4)
        except ImportError:
            return None

    @property
    def decoders_optimizations(self):
        """Return optimization settings stub for TTNN layers."""
        return None

    @classmethod
    def from_hf_config_dict(cls, hf_dict, mesh_device=None, max_batch_size=32, max_seq_len=8192):
        """Create config from HF config dict (used by generator_vllm.py)."""
        base = Qwen3CoderNextConfig.from_hf_config(hf_dict)
        hf_fields = {f: getattr(base, f) for f in Qwen3CoderNextConfig.__dataclass_fields__}

        if mesh_device is not None:
            num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
            device_name = _determine_device_name(mesh_device)
        else:
            num_devices = 0
            device_name = "cpu"

        return cls(
            **hf_fields,
            num_devices=num_devices,
            mesh_device=mesh_device,
            device_name=device_name,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        logger.info(
            f"Qwen3CoderNextTTConfig: {self.num_devices} devices, "
            f"{self.experts_per_device} experts/device, "
            f"batch_per_group={self.batch_size_per_device_group}, "
            f"layer_types={self.num_deltanet_layers}xDeltaNet+{self.num_gqa_layers}xGQA"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        mesh_device=None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
    ) -> "Qwen3CoderNextTTConfig":
        """Load config from HF model name/path and attach device topology.

        Args:
            model_name: HuggingFace model name or local path containing config.json.
            mesh_device: TT mesh device handle (or None for CPU-only config).
            max_batch_size: Maximum batch size for decode.
            max_seq_len: Maximum sequence length.

        Returns:
            Qwen3CoderNextTTConfig with all TT-specific fields populated.
        """
        # Load base HF config values
        hf_config = Qwen3CoderNextConfig.from_pretrained(model_name)

        # Extract HF fields into a dict
        hf_fields = {}
        for f in Qwen3CoderNextConfig.__dataclass_fields__:
            hf_fields[f] = getattr(hf_config, f)

        # Determine device topology
        if mesh_device is not None:
            num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
            device_name = _determine_device_name(mesh_device)
        else:
            num_devices = 0
            device_name = "cpu"

        return cls(
            **hf_fields,
            num_devices=num_devices,
            mesh_device=mesh_device,
            device_name=device_name,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_defaults(
        cls,
        mesh_device=None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
    ) -> "Qwen3CoderNextTTConfig":
        """Create config with default Qwen3-Coder-Next dimensions (no HF download).

        Useful for testing and development without model weights.
        """
        if mesh_device is not None:
            num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
            device_name = _determine_device_name(mesh_device)
        else:
            num_devices = 0
            device_name = "cpu"

        return cls(
            num_devices=num_devices,
            mesh_device=mesh_device,
            device_name=device_name,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )


def _determine_device_name(mesh_device) -> str:
    """Infer a short device name string from the mesh device handle."""
    try:
        n = mesh_device.get_num_devices()
        if n == 8:
            return "T3K"
        elif n == 32:
            return "TG"
        elif n == 2:
            return "N300"
        elif n == 1:
            return "N150"
        return f"{n}dev"
    except Exception:
        return "unknown"
