# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Omni model with TTNN backend."""

import os
import pytest
import soundfile as sf
import torch
from torch import nn
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLama
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_qwen_omni(device):
    """Test Qwen3-Omni model with TTNN acceleration."""
    assert (
        os.environ.get("TT_SYMBIOTE_RUN_MODE") == "CPU"
    ), f"Expected TT_SYMBIOTE_RUN_MODE environment variable to be 'CPU', got {os.environ.get('TT_SYMBIOTE_RUN_MODE')}"
    # Define module replacement mapping
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
    }

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    print(f"Loading Qwen3-Omni model from {MODEL_PATH}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(dtype=torch.bfloat16)

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
                {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Register module replacements
    print("Registering TTNN module replacements...")
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Set device for all TTNN modules
    print("Setting device for TTNN modules...")
    set_device(model, device)

    # Preprocess and move weights to device
    print("Preprocessing and moving weights to device...")
    # for k, v in tqdm(modules.items(), desc="Processing modules"):
    #     v.preprocess_weights()
    #     v.move_weights_to_device()

    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Preparation for inference
    print("Preparing inputs...")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    print("Running inference...")
    DispatchManager.clear_timings()

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(
        **inputs, speaker="Ethan", thinker_return_dict_in_generate=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    DispatchManager.save_stats_to_file("qwen_omni_timing_stats.csv")

    text = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"QWEN-OMNI OUTPUT: {text}")

    if audio is not None:
        sf.write(
            "output.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print("Audio output saved to output.wav")
