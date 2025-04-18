# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.attention import Attention as BaseAttention  # renamed for clarity
from models.demos.glm4.tt.model_config import Glm4ModelArgs  # Import Glm4ModelArgs

# Assume get_rot_transformation_mat will be available here after refactoring common.py
from models.demos.glm4.tt.common import get_rot_transformation_mat


class Glm4Attention(BaseAttention):
    """
    glm4 specific attention mechanism, inheriting from base attention
    includes logic for partial rotary embeddings
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration: Glm4ModelArgs,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Call base class init - note: transformation_mats might be None here
        # The base class will create default ones (with factor=1.0) if None
        super().__init__(
            mesh_device,
            state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            transformation_mats,
            configuration,
            paged_attention_config,
            use_paged_kv_cache,
        )

        # override transformation_mats using the partial_rotary_factor from Glm4ModelArgs
        # this ensures the correct RoPE matrix is used regardless of what base init did
        partial_factor = (
            configuration.partial_rotary_factor
            if configuration.is_glm4 and configuration.partial_rotary_factor is not None
            else 1.0
        )
        self.transformation_mats = {
            "prefill": get_rot_transformation_mat(self.head_dim, partial_rotary_factor=partial_factor),
            "decode": get_rot_transformation_mat(self.head_dim, partial_rotary_factor=partial_factor),
        }

        # add other glm4 specific initializations if needed later

    # override methods like forward if glm4 requires different logic
    def forward(self, x: ttnn.Tensor, start_pos: int, freqs_cis: ttnn.Tensor, mask: ttnn.Tensor):
        # TODO: implement glm4 specific forward pass, potentially calling super().forward()
        # or reimplementing parts with partial rope logic
        # For now, just use the overridden transformation_mats via the base forward
        return super().forward(x, start_pos, freqs_cis, mask)
