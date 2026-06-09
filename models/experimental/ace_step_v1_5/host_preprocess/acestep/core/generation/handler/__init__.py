# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Handler decomposition for TTNN demo preprocessing.

TTNN demos call ``handler_prepare_condition_payload`` only. That path uses:

  ``generate_music_request``, ``service_generate_request``, ``service_generate_execute``,
  ``batch_prep``, ``conditioning_*``, ``conditioning_embed``, ``audio_codes``,
  ``init_service*``, ``memory_utils``, ``metadata_utils``, ``padding_utils``,
  ``prompt_utils``, ``progress``, ``task_utils``, ``lyric_alignment_common``.

Full ``generate_music`` PyTorch diffusion is not available here.
"""

from .audio_codes import AudioCodesMixin
from .batch_prep import BatchPrepMixin
from .conditioning_batch import ConditioningBatchMixin
from .conditioning_embed import ConditioningEmbedMixin
from .conditioning_masks import ConditioningMaskMixin
from .conditioning_target import ConditioningTargetMixin
from .conditioning_text import ConditioningTextMixin
from .generate_music import GenerateMusicMixin
from .generate_music_request import GenerateMusicRequestMixin
from .init_service import InitServiceMixin
from .lyric_alignment_common import LyricAlignmentCommonMixin
from .memory_utils import MemoryUtilsMixin
from .metadata_utils import MetadataMixin
from .padding_utils import PaddingMixin
from .prompt_utils import PromptMixin
from .progress import ProgressMixin
from .service_generate_execute import ServiceGenerateExecuteMixin
from .service_generate_request import ServiceGenerateRequestMixin
from .task_utils import TaskUtilsMixin

__all__ = [
    "AudioCodesMixin",
    "BatchPrepMixin",
    "ConditioningBatchMixin",
    "ConditioningEmbedMixin",
    "ConditioningMaskMixin",
    "ConditioningTargetMixin",
    "ConditioningTextMixin",
    "GenerateMusicMixin",
    "GenerateMusicRequestMixin",
    "InitServiceMixin",
    "LyricAlignmentCommonMixin",
    "MemoryUtilsMixin",
    "MetadataMixin",
    "PaddingMixin",
    "PromptMixin",
    "ProgressMixin",
    "ServiceGenerateExecuteMixin",
    "ServiceGenerateRequestMixin",
    "TaskUtilsMixin",
]
