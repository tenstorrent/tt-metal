"""Test for Whisper 3 model with TTNN backend."""

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_whisper3(device):
    """Test Whisper 3 model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
    }
    torch_dtype = torch.bfloat16

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(dtype=torch_dtype)
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
    )
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]

    result = pipe(sample, return_timestamps=True)
    print(result["text"])
