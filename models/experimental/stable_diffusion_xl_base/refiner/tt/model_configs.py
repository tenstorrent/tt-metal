# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations


class RefinerModelOptimisations(ModelOptimisations):
    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat16,
        ff_weights_dtype=ttnn.bfloat8_b,
    ):
        super().__init__(conv_act_dtype, conv_w_dtype, attention_weights_dtype, ff_weights_dtype)

    def get_matmul_config(self, matmul_path):
        return None

    def get_mm_compute_config(self, module_path):
        return None

    def get_conv_config(self, conv_path):
        return None

    def get_conv_compute_config(self, module_path):
        return None

    def get_conv_output_dtype(self):
        return self.conv_output_dtype
