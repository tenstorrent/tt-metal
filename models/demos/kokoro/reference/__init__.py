# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M reference (mirrors `models/experimental/speecht5_tts/reference/` layout).

- Op-by-op PyTorch from Hugging Face weights (`kokoro_plbert`, `kokoro_predictor`, `kokoro_istftnet`)
- Full composed reference (`kokoro_full_model`)
- Upstream `kokoro` `KModel` wrapper (`kokoro_model`) for PCC vs the official implementation
- Pipeline wrapper for G2P/voices (`kokoro_pipeline`)
"""

from .kokoro_config import KokoroConfig
from .kokoro_full_model import (
    KokoroFullOutput,
    KokoroFullReference,
    load_full_reference_from_huggingface,
    load_full_reference_model,
)
from .kokoro_istftnet import KokoroIstftNet, KokoroIstftNetOutput, load_decoder_from_huggingface
from .kokoro_model import KokoroModelReference, load_reference_kmodel, load_reference_model
from .kokoro_pipeline import KokoroPipelineReference, load_reference_pipeline
from .kokoro_plbert import KokoroPlBert, KokoroPlBertOutput, load_plbert_from_huggingface
from .kokoro_predictor import KokoroPredictor, KokoroPredictorOutput, load_predictor_from_huggingface

__all__ = [
    "KokoroConfig",
    "KokoroFullOutput",
    "KokoroFullReference",
    "KokoroIstftNet",
    "KokoroIstftNetOutput",
    "KokoroModelReference",
    "KokoroPlBert",
    "KokoroPlBertOutput",
    "KokoroPipelineReference",
    "KokoroPredictor",
    "KokoroPredictorOutput",
    "load_decoder_from_huggingface",
    "load_full_reference_from_huggingface",
    "load_full_reference_model",
    "load_plbert_from_huggingface",
    "load_predictor_from_huggingface",
    "load_reference_kmodel",
    "load_reference_model",
    "load_reference_pipeline",
]
