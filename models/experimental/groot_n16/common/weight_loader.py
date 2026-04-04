# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loader for GR00T N1.6-3B.

Downloads from HuggingFace (nvidia/GR00T-N1.6-3B) and categorizes weights
into backbone (vision + language), action head (DiT), and embodiment MLPs.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from safetensors.torch import load_file as safetensors_load_file

logger = logging.getLogger(__name__)


class Gr00tN16WeightLoader:
    """Load and categorize GR00T N1.6 weights from HuggingFace."""

    # Weight prefix mapping
    VISION_PREFIX = "backbone.vision_encoder."
    LM_PREFIX = "backbone.language_model."
    CONNECTOR_PREFIX = "backbone.connector."
    DIT_PREFIX = "action_head.diffusion_model."
    STATE_ENCODER_PREFIX = "action_head.state_encoder."
    ACTION_ENCODER_PREFIX = "action_head.action_encoder."
    ACTION_DECODER_PREFIX = "action_head.action_decoder."
    VL_LAYERNORM_PREFIX = "action_head.vl_layernorm."
    POS_EMBED_PREFIX = "action_head.pos_embed"
    TIMESTEP_PREFIX = "action_head.timestep_encoder."

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_model_id: str = "nvidia/GR00T-N1.6-3B",
        cache_dir: Optional[str] = None,
    ):
        self.hf_model_id = hf_model_id
        self.model_path = model_path
        self.cache_dir = cache_dir
        self._state_dict: Optional[Dict[str, torch.Tensor]] = None

    def load(self) -> Dict[str, torch.Tensor]:
        """Load all weights, downloading from HF if needed."""
        if self._state_dict is not None:
            return self._state_dict

        if self.model_path and Path(self.model_path).exists():
            self._state_dict = self._load_from_path(self.model_path)
        else:
            self._state_dict = self._load_from_hf()

        logger.info(f"Loaded {len(self._state_dict)} weight tensors")
        return self._state_dict

    def _load_from_path(self, path: str) -> Dict[str, torch.Tensor]:
        """Load from local safetensors files."""
        p = Path(path)
        state_dict = {}
        if p.is_file():
            state_dict = safetensors_load_file(str(p))
        else:
            for sf_file in sorted(p.glob("*.safetensors")):
                logger.info(f"Loading {sf_file.name}")
                state_dict.update(safetensors_load_file(str(sf_file)))
        return state_dict

    def _load_from_hf(self) -> Dict[str, torch.Tensor]:
        """Download and load from HuggingFace."""
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading {self.hf_model_id} from HuggingFace...")
        local_dir = snapshot_download(
            repo_id=self.hf_model_id,
            cache_dir=self.cache_dir,
            allow_patterns=["*.safetensors", "*.json"],
        )
        return self._load_from_path(local_dir)

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        if self._state_dict is None:
            self.load()
        return self._state_dict

    def get_vision_weights(self) -> Dict[str, torch.Tensor]:
        """Extract SigLIP2 vision encoder weights."""
        prefix = self.VISION_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_language_weights(self) -> Dict[str, torch.Tensor]:
        """Extract Qwen3-1.7B language model weights."""
        prefix = self.LM_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_connector_weights(self) -> Dict[str, torch.Tensor]:
        """Extract MLP connector weights (vision -> language dim)."""
        prefix = self.CONNECTOR_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_dit_weights(self) -> Dict[str, torch.Tensor]:
        """Extract AlternateVLDiT action head weights."""
        prefix = self.DIT_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_state_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract per-embodiment state encoder MLP weights."""
        prefix = self.STATE_ENCODER_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_action_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract multi-embodiment action encoder weights."""
        prefix = self.ACTION_ENCODER_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_action_decoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract per-embodiment action decoder MLP weights."""
        prefix = self.ACTION_DECODER_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_vl_layernorm_weights(self) -> Dict[str, torch.Tensor]:
        """Extract VL LayerNorm weights for backbone features."""
        prefix = self.VL_LAYERNORM_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_pos_embed(self) -> Optional[torch.Tensor]:
        """Get action positional embedding."""
        for k, v in self.state_dict.items():
            if k.startswith(self.POS_EMBED_PREFIX):
                return v
        return None

    def get_timestep_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract timestep encoder weights."""
        prefix = self.TIMESTEP_PREFIX
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def print_weight_summary(self):
        """Print summary of weight categories."""
        categories = {
            "Vision (SigLIP2)": self.VISION_PREFIX,
            "Language (Qwen3)": self.LM_PREFIX,
            "Connector": self.CONNECTOR_PREFIX,
            "DiT": self.DIT_PREFIX,
            "State encoder": self.STATE_ENCODER_PREFIX,
            "Action encoder": self.ACTION_ENCODER_PREFIX,
            "Action decoder": self.ACTION_DECODER_PREFIX,
            "VL LayerNorm": self.VL_LAYERNORM_PREFIX,
            "Timestep encoder": self.TIMESTEP_PREFIX,
        }

        total = 0
        for name, prefix in categories.items():
            weights = {k: v for k, v in self.state_dict.items() if k.startswith(prefix)}
            num_params = sum(v.numel() for v in weights.values())
            total += num_params
            logger.info(f"  {name}: {len(weights)} tensors, {num_params/1e6:.1f}M params")

        # Uncategorized
        all_prefixes = list(categories.values()) + [self.POS_EMBED_PREFIX]
        uncategorized = {
            k: v for k, v in self.state_dict.items()
            if not any(k.startswith(p) for p in all_prefixes)
        }
        if uncategorized:
            num_params = sum(v.numel() for v in uncategorized.values())
            total += num_params
            logger.info(f"  Uncategorized: {len(uncategorized)} tensors, {num_params/1e6:.1f}M params")
            for k in sorted(uncategorized.keys())[:10]:
                logger.info(f"    {k}: {uncategorized[k].shape}")

        logger.info(f"  Total: {total/1e6:.1f}M params")
