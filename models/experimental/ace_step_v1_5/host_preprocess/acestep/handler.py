# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Business Logic Handler
Encapsulates all data processing and business logic as a bridge between model and UI
"""
import os

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

import torch
from acestep.core.generation.handler import (
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    GenerateMusicMixin,
    GenerateMusicRequestMixin,
    InitServiceMixin,
    LyricAlignmentCommonMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateRequestMixin,
    TaskUtilsMixin,
)

warnings.filterwarnings("ignore")


class AceStepHandler(
    GenerateMusicMixin,
    GenerateMusicRequestMixin,
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    InitServiceMixin,
    LyricAlignmentCommonMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    TaskUtilsMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
):
    """ACE-Step Business Logic Handler"""

    def __init__(self):
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32
        # TTNN demos use ``preprocess_only=True`` in ``initialize_service`` (CPU batching only).
        self.preprocess_only = False
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_vae = None
        self.use_mlx_vae = False

        self.vae = None
        self.text_encoder = None
        self.text_tokenizer = None
        self.silence_latent = None
        self.sample_rate = 48000

        from acestep.core.generation.handler.lyric_alignment_common import _DEFAULT_LAYERS_CONFIG

        self.custom_layers_config = dict(_DEFAULT_LAYERS_CONFIG)
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.compiled = False
        self.last_init_params = None
        self.quantization = None
