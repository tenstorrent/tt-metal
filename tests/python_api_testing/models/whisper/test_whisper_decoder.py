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
from transformers import WhisperModel, WhisperConfig

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
    create_padded_tensor,
    create_unpadded_tensor,
)
from python_api_testing.models.whisper.whisper_decoder import TtWhisperDecoder
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def run_whisper_decoder(device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    base_address = "decoder"

    configuration = model.config

    pytorch_model = model.decoder
    pytorch_model.eval()
    state_dict = model.state_dict()

    padding = True
    if padding:
        enc_seq_len = 1500
        pad = 4
    else:
        enc_seq_len = 32

    batch = 1
    embed_dim = configuration.d_model
    dec_seq_len = 32

    # (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    # Indices of input sequence tokens in the vocabulary
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

    encoder_hidden_states = torch.rand(batch, enc_seq_len, embed_dim)
    output_tensor_shape = list(encoder_hidden_states.size())
    while len(output_tensor_shape) < 4:
        output_tensor_shape.insert(0, 1)
    output_tensor_shape[-2] = enc_seq_len + pad
    ttm_encoder_hidden_states = create_padded_tensor(
        list(encoder_hidden_states.size()),
        encoder_hidden_states,
        output_tensor_shape,
        pad_value=0.0,
        device=device,
    )

    with torch.no_grad():
        pytorch_output = pytorch_model(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            output_hidden_states=True,
        )

    logger.info("Running tt whisper decoder")

    tt_whisper_decoder = TtWhisperDecoder(
        base_address=base_address,
        state_dict=state_dict,
        device=device,
        config=model.config,
        already_padded_inputs=True,
    )
    tt_whisper_decoder.eval()

    with torch.no_grad():
        ttm_output = tt_whisper_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=ttm_encoder_hidden_states,
            output_attentions=True,
            output_hidden_states=True,
        )

    # Check last_hidden_state
    ttm_output_to_torch = tt2torch_tensor(ttm_output.last_hidden_state)
    ttm_output_to_torch = torch.squeeze(ttm_output_to_torch, 0)

    does_pass, pcc_message = comp_pcc(
        pytorch_output.last_hidden_state, ttm_output_to_torch, 0.97
    )
    logger.info(pcc_message)

    if does_pass:
        logger.info("Decoder Passed!")
    else:
        logger.warning("Decoder Failed!")

    assert does_pass


def test_WhipserDecoder_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_whisper_decoder(device=device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_WhipserDecoder_inference()
