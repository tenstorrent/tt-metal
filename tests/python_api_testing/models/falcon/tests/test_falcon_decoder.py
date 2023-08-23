import torch
import pytest
from loguru import logger

import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_decoder import TtFalconDecoderLayer
from tests.python_api_testing.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.decoder = hf_reference_model.transformer.h[layer_num]

        # Disable dropout
        self.decoder.eval()

    def forward(self, x, alibi, attention_mask, layer_past):
        result = self.decoder(
            hidden_states=x,
            alibi=alibi,
            attention_mask=attention_mask,
            layer_past=layer_past,
        )[0]
        return result


def run_test_FalconDecoder_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ========================================================================
    torch.manual_seed(0)
    decoder_input = (torch.rand(batch, seq_len, configuration.hidden_size) * 2) - 1
    base_url = "transformer.h"
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.n_head

    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.ones(batch, 1, q_len, kv_len, dtype=bool).triu(
            diagonal=1
        )
        layer_past = None

        tt_decoder_input = torch2tt_tensor(decoder_input.unsqueeze(1), device)
        tt_attention_mask = torch2tt_tensor(
            (attention_mask_bool * -100000).expand(-1, configuration.n_head, -1, -1),
            device,
        )
        tt_layer_past = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.zeros(batch, 1, q_len, kv_len, dtype=bool)
        attention_mask_bool[:, :, :, -1] = True
        k_cache = torch.rand(batch, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, kv_cache_len, head_dim)
        layer_past = (k_cache, v_cache)

        tt_decoder_input = torch2tt_tensor(
            decoder_input.unsqueeze(1).transpose(0, 2), device
        )

        kv_len_padded = (kv_len + 31) // 32 * 32
        attention_mask_bool_padded = torch.cat(
            (
                attention_mask_bool,
                torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
            ),
            dim=-1,
        )
        tt_attention_mask = torch2tt_tensor(
            (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                -1, configuration.n_head, -1, -1
            ),
            device,
        )
        tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_k_cache[:, :kv_cache_len, :] = k_cache
        tt_v_cache[:, :kv_cache_len, :] = v_cache
        tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
        tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
        tt_layer_past = (tt_k_cache, tt_v_cache)

    else:
        raise NotImplementedError(
            f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode."
        )

    # PyTorch output =======================================================================
    pytorch_FalconDecoder_model = PytorchFalconDecoderModel(
        hugging_face_reference_model, layer_num
    )
    pytorch_out = pytorch_FalconDecoder_model(
        x=decoder_input,
        alibi=None,
        attention_mask=attention_mask_bool,
        layer_past=layer_past,
    )

    # TT hardware execution =================================================================
    tt_FalconDecoder_model = TtFalconDecoderLayer(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration,
        max_position_embeddings,
        llm_mode,
        model_config,
        tt_cache_path,
    )

    tt_out, layer_present = tt_FalconDecoder_model(
        hidden_states=tt_decoder_input,
        alibi=None,
        attention_mask=tt_attention_mask,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
    )
    tt_out = tt2torch_tensor(tt_out).squeeze(1)
    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon Decoder output Passed!")
    else:
        logger.warning("Falcon Decoder output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_version, layer_num, pcc",
    (("tiiuae/falcon-7b-instruct", 0, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_FalconDecoder_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    pcc,
    model_config_str,
    model_location_generator,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_test_FalconDecoder_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        layer_num,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
    tt_lib.device.CloseDevice(device)
