# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from datasets import load_dataset
from loguru import logger
from transformers import WhisperModel, AutoFeatureExtractor

import ttnn

from models.experimental.whisper.tt.whisper_model import TtWhisperModel
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
    skip_for_wormhole_b0,
)


def run_whisper_model(device):
    pytorch_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = pytorch_model.config

    pytorch_model.eval()
    state_dict = pytorch_model.state_dict()

    """
    Original example from Huggingface documentation:

    Example:
        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperModel
        >>> from datasets import load_dataset

        >>> model = WhisperModel.from_pretrained("openai/whisper-base")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 512]
        ```
    """

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    input_features = inputs.input_features

    dec_seq_len = 32
    decoder_input_ids = (
        torch.tensor(
            [
                [
                    1,
                ]
                * dec_seq_len
            ]
        )
        * pytorch_model.config.decoder_start_token_id
    )

    with torch.no_grad():
        pytorch_output = pytorch_model(input_features=input_features, decoder_input_ids=decoder_input_ids)

    tt_whisper = TtWhisperModel(state_dict=state_dict, device=device, config=pytorch_model.config)
    tt_whisper.eval()

    with torch.no_grad():
        input_features = torch2tt_tensor(input_features, device, ttnn.ROW_MAJOR_LAYOUT)
        logger.info("Running tt whisper model")

        ttm_output = tt_whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)

    # Check correlations
    tt_out_to_torch = tt2torch_tensor(ttm_output.last_hidden_state)
    tt_out_to_torch = torch.squeeze(tt_out_to_torch, 0)

    does_pass, pcc_message = comp_pcc(pytorch_output.last_hidden_state, tt_out_to_torch, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info("Model Passed!")
    else:
        logger.warning("Model Failed!")

    assert does_pass


@skip_for_wormhole_b0()
def test_WhipserModel_inference(device):
    torch.manual_seed(1234)
    run_whisper_model(device=device)
