# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
OpenVoice V2 TTNN Implementation

Port of MyShell's OpenVoice V2 voice cloning model to Tenstorrent TTNN APIs.
Supports voice conversion, cross-lingual synthesis, and style manipulation.

Quick Start:
    ```python
    from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter

    # Initialize converter
    converter = TTNNToneColorConverter("config.json", device=device)
    converter.load_checkpoint("checkpoint.pth")

    # Extract speaker embedding
    target_se = converter.extract_se(["reference.wav"])

    # Convert voice
    converter.convert(
        source_audio="source.wav",
        src_se=src_se,
        tgt_se=target_se,
        output_path="output.wav",
    )
    ```
"""

__version__ = "0.1.0"

# Models
from models.demos.openvoice.tt.synthesizer import TTNNSynthesizerTrn
from models.demos.openvoice.tt.tone_color_converter import TTNNToneColorConverter
from models.demos.openvoice.tt.reference_encoder import TTNNReferenceEncoder
from models.demos.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder
from models.demos.openvoice.tt.generator import TTNNGenerator
from models.demos.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock
from models.demos.openvoice.tt.text_encoder import TTNNTextEncoder
from models.demos.openvoice.tt.duration_predictor import TTNNDurationPredictor
from models.demos.openvoice.tt.transformer_flow import TTNNTransformerCouplingBlock
from models.demos.openvoice.tt.melo_tts import TTNNMeloTTS

# Utils
from models.demos.openvoice.utils.weight_loader import load_openvoice_checkpoint
from models.demos.openvoice.utils.audio import AudioProcessor, load_audio, save_audio

__all__ = [
    # Version
    "__version__",
    # Models
    "TTNNSynthesizerTrn",
    "TTNNToneColorConverter",
    "TTNNReferenceEncoder",
    "TTNNPosteriorEncoder",
    "TTNNGenerator",
    "TTNNResidualCouplingBlock",
    "TTNNTextEncoder",
    "TTNNDurationPredictor",
    "TTNNTransformerCouplingBlock",
    "TTNNMeloTTS",
    # Utils
    "load_openvoice_checkpoint",
    "AudioProcessor",
    "load_audio",
    "save_audio",
]
