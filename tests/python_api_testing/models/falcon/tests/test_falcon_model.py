import torch
import pytest
from loguru import logger
import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_model import TtFalconModel
from tests.python_api_testing.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconModel(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers):
        super().__init__()
        self.model = hf_reference_model.transformer
        self.model.h = self.model.h[:num_layers]
        self.model.eval()

    def forward(self, input_ids, past_key_values):
        result = self.model(input_ids=input_ids, past_key_values=past_key_values)[0]

        return result


def run_test_FalconModel_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
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

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = "transformer"
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.n_head

    if 1:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(
            batch, seq_len
        )

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None
        tt_layer_past = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        k_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        past_key_values = ((k_cache, v_cache),)

        tt_k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
        tt_k_cache[:, :, :kv_cache_len, :] = k_cache
        tt_v_cache[:, :, :kv_cache_len, :] = v_cache
        tt_k_cache = torch2tt_tensor(tt_k_cache, device)
        tt_v_cache = torch2tt_tensor(tt_v_cache, device)
        tt_layer_past = (tt_k_cache, tt_v_cache)

    else:
        raise NotImplementedError(
            f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode."
        )

    # Prepare output -----------------------------------------------------------------------
    pytorch_FalconModel = PytorchFalconModel(hugging_face_reference_model, num_layers)
    pytorch_out = pytorch_FalconModel(
        input_ids=model_input, past_key_values=past_key_values
    )
    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    tt_FalconModel = TtFalconModel(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        llm_mode,
        model_config,
        tt_cache_path,
    )
    # TODO: Generate embeddings and attention_mask on device
    tt_embeddings, tt_attention_mask = tt_FalconModel.model_preprocessing(
        model_input, kv_cache_len
    )
    tt_out = tt_FalconModel(
        input_embeddings=tt_embeddings,
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
        logger.info("Falcon Model Passed!")
    else:
        logger.warning("Falcon Model Failed!")
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
    "num_layers, pcc",
    ((2, 0.98), (32, 0.98)),
    ids=["layers_2", "layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconModel_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
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
    run_test_FalconModel_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
    tt_lib.device.CloseDevice(device)
