from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import tt_lib
import torch
from loguru import logger

from transformers import (
    WhisperModel,
    WhisperForAudioClassification,
    AutoFeatureExtractor,
)
from datasets import load_dataset

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.whisper.whisper_encoder import (
    TtWhisperEncoder,
    TtWhisperEncoderOutput,
)
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_encoder(device, for_audio_classification=False, encoder_layers=1):
    if for_audio_classification:
        model = WhisperForAudioClassification.from_pretrained(
            "sanchit-gandhi/whisper-medium-fleurs-lang-id"
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "sanchit-gandhi/whisper-medium-fleurs-lang-id"
        )
    else:
        model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "openai/whisper-tiny.en"
        )

    configuration = model.config
    if encoder_layers != configuration.encoder_layers:
        configuration.encoder_layers = encoder_layers
        model = WhisperForAudioClassification(configuration)

    pytorch_model = model.encoder
    pytorch_model.eval()

    base_address = "encoder"
    state_dict = model.state_dict()

    use_real_inputs = True

    if use_real_inputs:
        ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        sample = next(iter(ds))
        inputs = feature_extractor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt",
        )
        input_features = inputs.input_features
        logger.debug(f"Input of size {input_features.size()}")
        logger.debug("Input audio language:")
        logger.debug(sample["language"])
        head_mask = None
    else:
        batch = 1
        feature_size = 80
        seq_len = 3000
        input_features = torch.rand((batch, feature_size, seq_len))
        head_mask = None

    with torch.no_grad():
        pytorch_output = pytorch_model(
            input_features=input_features,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        tt_whisper_encoder = TtWhisperEncoder(
            base_address=base_address,
            state_dict=state_dict,
            device=device,
            config=pytorch_model.config,
        )
        tt_whisper_encoder.eval()

        ttm_output = tt_whisper_encoder(
            input_features=input_features,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=False,
        )

        logger.debug(f"Encoder returned {ttm_output.last_hidden_state.shape()}")

        # TT Output To Torch
        input_tensor_shape = [
            1,
            1,
            configuration.max_source_positions,
            configuration.d_model,
        ]
        ttm_output_pt = torch.Tensor(ttm_output.last_hidden_state.data()).reshape(
            *input_tensor_shape
        )
        ttm_output_pt = torch.squeeze(ttm_output_pt, 0)

        logger.debug(f"Encoder output to torch {ttm_output_pt.size()}")
        # else:
        #     ttm_output_pt = tt2torch_tensor(ttm_output.last_hidden_state)

        does_pass, pcc_message = comp_pcc(
            pytorch_output.last_hidden_state, ttm_output_pt, 0.98
        )
        logger.info(pcc_message)

        if does_pass:
            logger.info("Encoder Passed!")
        else:
            logger.warning("Encoder Failed!")

        assert does_pass


def test_WhipserEncoder_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder(device=device, for_audio_classification=False)
    tt_lib.device.CloseDevice(device)


def test_WhipserEncoderForAudioClassification_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=24)
    tt_lib.device.CloseDevice(device)


def test_WhipserEncoderForAudioClassification_one_layer_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=1)
    tt_lib.device.CloseDevice(device)


def test_WhipserEncoderForAudioClassification_two_layers_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=2)
    tt_lib.device.CloseDevice(device)


def test_WhipserEncoderForAudioClassification_three_layers_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_encoder(device=device, for_audio_classification=True, encoder_layers=3)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_WhipserEncoder_inference()
    test_WhipserEncoderForAudioClassification_one_layer_inference()
    test_WhipserEncoderForAudioClassification_two_layers_inference()
    test_WhipserEncoderForAudioClassification_three_layers_inference()
    test_WhipserEncoderForAudioClassification_inference()
