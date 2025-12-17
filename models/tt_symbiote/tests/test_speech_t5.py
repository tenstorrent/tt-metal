"""Test for Speech T5 model with TTNN backend."""

import torch
from torch import nn
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_speech_t5(device):
    """Test SpeechT5 model with TTNN acceleration."""

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.LayerNorm: TTNNLayerNorm,
    }
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
    inputs["speaker_embeddings"] = torch.randn((1, 512))
    model.eval()  # Disables dropout, batch norm updates
    vocoder.eval()
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    register_module_replacement_dict(vocoder, nn_to_ttnn, model_config=None)
    set_device(model, device)
    set_device(vocoder, device)
    torch.set_grad_enabled(False)  # Disables autograd overhead
    speech = model.generate_speech(**inputs, vocoder=vocoder)
