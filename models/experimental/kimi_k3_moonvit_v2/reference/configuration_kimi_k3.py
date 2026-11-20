# SPDX-FileCopyrightText: © 2025-2026 The Moonshot AI Team.
# SPDX-License-Identifier: LicenseRef-LICENSE-FILE
# Vendored from https://huggingface.co/moonshotai/Kimi-K3 under the Kimi K3 License (LICENSE-Kimi-K3 beside this file).
# Modified: the language model's configs (KimiLinearConfig, KimiK3Config) removed.

from transformers.configuration_utils import PretrainedConfig


class KimiK3VisionConfig(PretrainedConfig):
    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple = (2, 2),
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        # MM Projector parameters
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        # vision tower parameters
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        # Other parameters
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        text_hidden_size=7168,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = merge_kernel_size
        self.merge_type = merge_type
        self._attn_implementation = _attn_implementation

        # MM Projector config
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        # vision tower parameters
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        super().__init__(**kwargs)
