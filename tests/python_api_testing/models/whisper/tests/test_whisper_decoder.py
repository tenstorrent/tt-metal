import tt_lib
import torch
from loguru import logger
from transformers import WhisperModel, WhisperConfig

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tt_models.whisper.tt.whisper_decoder import TtWhisperDecoder
from tt_models.utility_functions import (
    comp_allclose,
    comp_pcc,
)


def run_whisper_decoder(device):
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    base_address = "decoder"

    configuration = model.config

    pytorch_model = model.decoder
    pytorch_model.eval()
    state_dict = model.state_dict()

    enc_seq_len = 1500
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
    )
    tt_whisper_decoder.eval()

    ttm_encoder_hidden_states = torch2tt_tensor(
        encoder_hidden_states, device, tt_lib.tensor.Layout.ROW_MAJOR
    )
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
        pytorch_output.last_hidden_state, ttm_output_to_torch, 0.98
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
