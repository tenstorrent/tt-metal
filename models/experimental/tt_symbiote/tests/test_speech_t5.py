# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Speech T5 model with TTNN backend."""

import soundfile as sf
import torch
from torch import nn
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_speech_t5(device):
    """Test SpeechT5 model with TTNN acceleration."""

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.LayerNorm: TTNNLayerNorm,
    }
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(dtype=torch.bfloat16)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    inputs = processor(
        text="What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        return_tensors="pt",
    )
    inputs["speaker_embeddings"] = torch.randn((1, 512), dtype=torch.bfloat16)
    model.eval()  # Disables dropout, batch norm updates
    vocoder.eval()
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    register_module_replacement_dict(vocoder, nn_to_ttnn, model_config=None)
    set_device(model, device)
    set_device(vocoder, device)
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    speech = model.generate_speech(**inputs, vocoder=vocoder)
    DispatchManager.save_stats_to_file("speech_t5_timing_stats.csv")
    sf.write("output.wav", speech.squeeze().detach().numpy(), samplerate=16000)
    print("Speech T5 TTNN test passed, output.wav generated.")
