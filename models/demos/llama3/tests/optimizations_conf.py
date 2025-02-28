# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from models.demos.llama3.tt.model_config import DecodersPrecision, LlamaOptimizations, LayerGroup, PrecisionSetting


def precision_conf():
    num_decoders = 32

    return [
        pytest.param(
            DecodersPrecision(
                num_decoders,
                LlamaOptimizations(
                    {
                        LayerGroup.FF1: PrecisionSetting.BFP4_LOFI,
                        LayerGroup.FF2: PrecisionSetting.BFP4_LOFI,
                        LayerGroup.FF3: PrecisionSetting.BFP4_LOFI,
                    }
                ),
            ),
            id="precision_ff1_bfp4_lofi_ff2_bfp4_lofi_ff3_bfp4_lofi",
        ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BFP4_LOFI,
        #             LayerGroup.FF2: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF3: PrecisionSetting.BFP4_LOFI,
        #         }
        #     ),
        #     id="precision_ff1_bfp4_lofi_ff2_bfp8_hifi2_ff3_bfp4_lofi",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BFP4_LOFI,
        #             LayerGroup.FF2: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF3: PrecisionSetting.BFP4_LOFI,
        #         }
        #     ),
        #     id="precision_ff1_bfp4_lofi_ff2_bf16_hifi4_ff3_bfp4_lofi",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF2: PrecisionSetting.BFP4_LOFI,
        #             LayerGroup.FF3: PrecisionSetting.BFP8_HIFI2,
        #         }
        #     ),
        #     id="precision_ff1_bfp8_hifi2_ff2_bfp4_lofi_ff3_bfp8_hifi2",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF2: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF3: PrecisionSetting.BFP8_HIFI2,
        #         }
        #     ),
        #     id="precision_ff1_bfp8_hifi2_ff2_bfp8_hifi2_ff3_bfp8_hifi2",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF2: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF3: PrecisionSetting.BFP8_HIFI2,
        #         }
        #     ),
        #     id="precision_ff1_bfp8_hifi2_ff2_bf16_hifi4_ff3_bfp8_hifi2",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF2: PrecisionSetting.BFP4_LOFI,
        #             LayerGroup.FF3: PrecisionSetting.BF16_HIFI4,
        #         }
        #     ),
        #     id="precision_ff1_bf16_hifi4_ff2_bfp4_lofi_ff3_bf16_hifi4",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF2: PrecisionSetting.BFP8_HIFI2,
        #             LayerGroup.FF3: PrecisionSetting.BF16_HIFI4,
        #         }
        #     ),
        #     id="precision_ff1_bf16_hifi4_ff2_bfp8_hifi2_ff3_bf16_hifi4",
        # ),
        # pytest.param(
        #     LlamaOptimizations(
        #         {
        #             LayerGroup.FF1: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF2: PrecisionSetting.BF16_HIFI4,
        #             LayerGroup.FF3: PrecisionSetting.BF16_HIFI4,
        #         }
        #     ),
        #     id="precision_ff1_bf16_hifi4_ff2_bf16_hifi4_ff3_bf16_hifi4",
        # ),
        pytest.param(DecodersPrecision(num_decoders, LlamaOptimizations.accuracy()), id="accuracy"),
        pytest.param(DecodersPrecision(num_decoders, LlamaOptimizations.performance()), id="performance"),
    ]
