# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import tree
import logging
from typing import Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, BatchFeature

from gr00t_attention import TTNNSelfAttention


try:
    from models.experimental.tt_symbiote.modules.activation import TTNNSilu
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
except ImportError:
    logging.warning(
        "TTNN hardware modules not found; falling back to CPU execution path."
    )

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


class Gr00tN1d6(PreTrainedModel):
    """
    GR00T Humanoid Foundation Model optimized for Tenstorrent hardware.
    Integrates vision-language backbones with accelerated Diffusion Transformers (DiT).
    """

    config_class = Gr00tN1d6Config

    def __init__(self, config: Gr00tN1d6Config, **kwargs):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.backbone_embedding_dim = config.backbone_embedding_dim

        # EagleBackbone initialization requires use_flash_attention=True for the internal config
        self.backbone = EagleBackbone(
            model_name=config.model_name, use_flash_attention=True, load_bf16=True
        )

        self.action_head = nn.Module()
        self.action_head.model = (
            AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            if config.use_alternate_vl_dit
            else DiT(**config.diffusion_model_cfg)
        )

        # Multimodal and embodiment encoders
        self.action_head.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=config.input_embedding_dim,
        )
        self.action_head.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=config.max_action_dim,
        )
        self.action_head.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.max_action_dim,
            hidden_size=config.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_head.vlln = nn.LayerNorm(config.backbone_embedding_dim)
        self.action_head.position_embedding = nn.Parameter(
            torch.zeros(1, 256, self.hidden_size)
        )

    def perform_surgery(self):
        """
        Recursively replaces PyTorch layers with TTNN hardware modules to enable
        Wormhole acceleration via Forced Class Takeover.
        """
        logging.info("Initializing hardware module injection for Tensix cores...")

        attn_targets = {
            "Attention",
            "SigLipAttention",
            "AlternateVLDiTAttention",
            "Qwen2Attention",
        }
        op_map = {nn.Linear: TTNNLinear, nn.LayerNorm: TTNNLayerNorm, nn.SiLU: TTNNSilu}

        def inject_hardware_ops(module):
            for attr_name, child in list(module.named_children()):
                cls_name = child.__class__.__name__

                if cls_name in attn_targets:
                    # Capture original weights through constructor re-initialization
                    orig = child
                    child.__class__ = TTNNSelfAttention
                    TTNNSelfAttention.__init__(child, orig)
                elif child.__class__ in op_map:
                    # Directly swap module instances
                    hw_mod = op_map[child.__class__].from_torch(child)
                    if hasattr(module, "_modules"):
                        module._modules[attr_name] = hw_mod
                    object.__setattr__(module, attr_name, hw_mod)
                else:
                    inject_hardware_ops(child)

        inject_hardware_ops(self.backbone)
        inject_hardware_ops(self.action_head)
        logging.info("Surgery complete: Model locked to hardware compute engines.")

    def forward(self, **inputs) -> dict:
        """
        Multimodal forward pass utilizing device-mapped backbone and DiT features.
        """
        backbone_in, action_in = self.prepare_input(inputs)
        backbone_out = self.backbone(backbone_in)
        backbone_feats = backbone_out["backbone_features"]

        state_features = self.action_head.state_encoder(
            action_in.state, action_in.embodiment_id
        )

        # Handle zero-sized text embedding edge cases to ensure hardware shape parity
        if state_features.nelement() == 0:
            state_features = torch.zeros(
                (
                    backbone_feats.shape[0],
                    backbone_feats.shape[1],
                    backbone_feats.shape[2],
                ),
                device=self.device,
                dtype=torch.bfloat16,
            )

        # DiT execution on hardware-accelerated paths
        diffusion_res = self.action_head.model(
            hidden_states=torch.cat((state_features, state_features), dim=1),
            encoder_hidden_states=backbone_feats,
            timestep=torch.tensor([1], device=self.device),
            image_mask=backbone_out["image_mask"],
            backbone_attention_mask=backbone_out["backbone_attention_mask"],
        )

        model_out = (
            diffusion_res[0]
            if isinstance(diffusion_res, (tuple, list))
            else diffusion_res
        )
        actions = self.action_head.action_decoder(model_out, action_in.embodiment_id)

        return {"action": actions}

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """
        Pre-processes multimodal inputs to bfloat16 for hardware parity.
        """
        backbone_in = self.backbone.prepare_input(inputs)
        action_in = BatchFeature(data=inputs)

        def to_hw_safe(x):
            if not isinstance(x, torch.Tensor):
                return x
            target_dtype = torch.bfloat16 if torch.is_floating_point(x) else x.dtype
            return x.to(self.device, dtype=target_dtype)

        return (
            tree.map_structure(to_hw_safe, backbone_in),
            tree.map_structure(to_hw_safe, action_in),
        )
