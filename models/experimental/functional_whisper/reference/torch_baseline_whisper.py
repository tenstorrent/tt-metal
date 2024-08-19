# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoFeatureExtractor, WhisperModel
from datasets import load_dataset
import torch


# Generates the baseline expected results for tests within ttnn
if __name__ == "__main__":
    model_name = "openai/whisper-base"
    model = WhisperModel.from_pretrained(model_name).to(torch.bfloat16).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.type(torch.bfloat16)
    decoder_input_ids = torch.ones(1, 32).type(torch.int32) * model.config.decoder_start_token_id
    parameters = model.state_dict()
    print(parameters.keys())
    last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    print(last_hidden_state.shape)
    last_three = last_hidden_state[0, -1, -3:]
    print(last_three)
