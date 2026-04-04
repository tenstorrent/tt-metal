# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loader for GR00T N1.6-3B.

Downloads from HuggingFace (nvidia/GR00T-N1.6-3B) and categorizes weights
into backbone (vision + language), action head (DiT), and embodiment MLPs.

Actual weight key prefixes (from model.safetensors.index.json):
    backbone.model.vision_model.vision_model.*       - SigLIP2 vision encoder
    backbone.model.language_model.*                   - Qwen3-1.7B LM
    backbone.model.mlp1.*                             - MLP connector
    action_head.model.transformer_blocks.*            - DiT blocks
    action_head.model.proj_out_1/2.*                  - DiT output projections
    action_head.model.timestep_encoder.*              - Timestep encoder
    action_head.state_encoder.layer{1,2}.{W,b}        - State encoder MLP
    action_head.action_encoder.W{1,2,3}.{W,b}         - Action encoder
    action_head.action_decoder.layer{1,2}.{W,b}        - Action decoder MLP
    action_head.vlln.{weight,bias}                    - VL LayerNorm
    action_head.position_embedding.weight             - Positional embedding
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file as safetensors_load_file

logger = logging.getLogger(__name__)


class Gr00tN16WeightLoader:
    """Load and categorize GR00T N1.6 weights from HuggingFace."""

    # Actual weight key prefixes from the model
    VISION_PREFIX = "backbone.model.vision_model.vision_model."
    LM_PREFIX = "backbone.model.language_model."
    CONNECTOR_PREFIX = "backbone.model.mlp1."
    DIT_PREFIX = "action_head.model."
    STATE_ENCODER_PREFIX = "action_head.state_encoder."
    ACTION_ENCODER_PREFIX = "action_head.action_encoder."
    ACTION_DECODER_PREFIX = "action_head.action_decoder."
    VL_LAYERNORM_PREFIX = "action_head.vlln."
    POS_EMBED_KEY = "action_head.position_embedding.weight"

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

    def _extract(self, prefix: str) -> Dict[str, torch.Tensor]:
        """Extract weights with given prefix, stripping the prefix from keys."""
        return {k[len(prefix) :]: v for k, v in self.state_dict.items() if k.startswith(prefix)}

    def get_vision_weights(self) -> Dict[str, torch.Tensor]:
        """Extract SigLIP2 vision encoder weights.

        Keys like: embeddings.patch_embedding.weight, encoder.layers.0.*, post_layernorm.*, head.*
        """
        return self._extract(self.VISION_PREFIX)

    def get_language_weights(self) -> Dict[str, torch.Tensor]:
        """Extract Qwen3-1.7B language model weights.

        Keys like: model.embed_tokens.weight, model.layers.0.*, lm_head.weight
        Note: Qwen3 uses q_norm/k_norm (QK normalization) and RMSNorm (no bias).
        """
        return self._extract(self.LM_PREFIX)

    def get_connector_weights(self) -> Dict[str, torch.Tensor]:
        """Extract MLP connector weights (vision -> language dim).

        Keys like: 0.weight, 0.bias, 2.weight, 2.bias (2-layer MLP)
        """
        return self._extract(self.CONNECTOR_PREFIX)

    def get_dit_weights(self) -> Dict[str, torch.Tensor]:
        """Extract AlternateVLDiT weights (blocks + proj_out + timestep_encoder).

        Keys like:
            transformer_blocks.0.attn1.to_q.weight
            transformer_blocks.0.ff.net.0.proj.weight
            transformer_blocks.0.norm1.linear.weight
            proj_out_1.weight, proj_out_2.weight
            timestep_encoder.timestep_embedder.linear_1.weight
        """
        return self._extract(self.DIT_PREFIX)

    def get_state_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract per-embodiment state encoder weights.

        Keys: layer1.W [num_cat, hidden, input], layer1.b [num_cat, hidden],
              layer2.W [num_cat, output, hidden], layer2.b [num_cat, output]
        """
        return self._extract(self.STATE_ENCODER_PREFIX)

    def get_action_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract multi-embodiment action encoder weights.

        Keys: W1.W, W1.b, W2.W, W2.b, W3.W, W3.b
        """
        return self._extract(self.ACTION_ENCODER_PREFIX)

    def get_action_decoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract per-embodiment action decoder weights.

        Keys: layer1.W, layer1.b, layer2.W, layer2.b
        """
        return self._extract(self.ACTION_DECODER_PREFIX)

    def get_vl_layernorm_weights(self) -> Dict[str, torch.Tensor]:
        """Extract VL LayerNorm weights: weight, bias."""
        return self._extract(self.VL_LAYERNORM_PREFIX)

    def get_pos_embed(self) -> Optional[torch.Tensor]:
        """Get action positional embedding tensor."""
        return self.state_dict.get(self.POS_EMBED_KEY)

    def get_timestep_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Extract timestep encoder weights (under action_head.model.timestep_encoder).

        Keys: timestep_embedder.linear_1.weight, timestep_embedder.linear_1.bias, etc.
        """
        prefix = self.DIT_PREFIX + "timestep_encoder."
        return {k[len(prefix) :]: v for k, v in self.state_dict.items() if k.startswith(prefix)}

    def print_weight_summary(self):
        """Print summary of weight categories."""
        categories = {
            "Vision (SigLIP2)": self.VISION_PREFIX,
            "Language (Qwen3)": self.LM_PREFIX,
            "Connector (MLP)": self.CONNECTOR_PREFIX,
            "DiT (action head)": self.DIT_PREFIX,
            "State encoder": self.STATE_ENCODER_PREFIX,
            "Action encoder": self.ACTION_ENCODER_PREFIX,
            "Action decoder": self.ACTION_DECODER_PREFIX,
            "VL LayerNorm": self.VL_LAYERNORM_PREFIX,
        }

        total = 0
        for name, prefix in categories.items():
            weights = {k: v for k, v in self.state_dict.items() if k.startswith(prefix)}
            num_params = sum(v.numel() for v in weights.values())
            total += num_params
            logger.info(f"  {name}: {len(weights)} tensors, {num_params/1e6:.1f}M params")

        pos = self.get_pos_embed()
        if pos is not None:
            total += pos.numel()
            logger.info(f"  Pos embed: 1 tensor, {pos.numel()/1e6:.1f}M params")

        logger.info(f"  Total: {total/1e6:.1f}M params")
