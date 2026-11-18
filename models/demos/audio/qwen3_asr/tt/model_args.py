# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ModelArgs for the Qwen3-ASR text decoder (Qwen3-1.7B shape) with its N150 decode settings.

The settings were tuned on this model's N150 decode speed and transcripts, so they apply
only through this class; other models with the same shape keep the shared defaults.
"""
import ttnn
from models.tt_transformers.tt.model_config import (
    MathFidelitySetting,
    ModelArgs,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


class Qwen3ASRModelArgs(ModelArgs):
    def _apply_model_overrides(self):
        self.fuse_w13_decode = True
        self.qkv_width_sharded_create_heads = True
        if self.num_devices != 1 or self.is_galaxy:
            return
        self.lm_head_dtype = ttnn.bfloat8_b
        self.sdpa_prefill_exp_approx = True
        # 16-core grids for RMSNorm and the DRAM-sharded MLP / attention-input matmuls that
        # share them; 32 cores for lm_head.
        self.attn_input_grid = self.core_grid_for_count(16)
        self.mlp_core_grid = self.core_grid_for_count(16)
        self.mlp2_core_grid = self.core_grid_for_count(16)
        self.lm_head_core_grid = self.core_grid_for_count(32)
        self.max_columns_per_device_lm_head = self.get_lm_head_max_columns_per_device(
            self.lm_head_core_grid, self.prefetcher
        )
        for conf in self.optimizations.decoder_optimizations.values():
            conf.op_fidelity_settings[OpGroup.LI_FF2] = MathFidelitySetting.LOFI
            conf.op_fidelity_settings[OpGroup.LI_FF1_FF3] = MathFidelitySetting.LOFI
            conf.op_fidelity_settings[OpGroup.SDPA_DECODE] = MathFidelitySetting.HIFI2
            conf.tensor_dtype_settings[TensorGroup.KV_CACHE] = PrecisionSetting.BFP8
            conf.tensor_dtype_settings[TensorGroup.WQKV] = PrecisionSetting.BFP8
            conf.tensor_dtype_settings[TensorGroup.WO] = PrecisionSetting.BFP8
