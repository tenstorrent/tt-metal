import torch
import pytest
from loguru import logger

import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_causallm import TtFalconCausalLM

from tests.python_api_testing.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconCausalLM(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers):
        super().__init__()
        self.model = hf_reference_model
        self.model.transformer.h = self.model.transformer.h[:num_layers]

        # Disable dropout
        self.model.eval()

    def forward(self, input_ids):
        result = self.model(input_ids=input_ids)[0]

        return result


def run_test_FalconCausalLM_inference(
    device,
    model_version,
    batch,
    seq_len,
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

    # Prepare input ========================================================================
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048

    if 1:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(
            batch, seq_len
        )

    # PyTorch output =======================================================================
    pytorch_FalconCausalLM = PytorchFalconCausalLM(
        hugging_face_reference_model, num_layers
    )
    pytorch_out = pytorch_FalconCausalLM(input_ids=model_input)

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders

    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )

    # TODO: Generate embeddings and attention_mask on device
    tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
        model_input
    )

    tt_out = tt_FalconCausalLM(
        input_embeddings=tt_embeddings, attention_mask=tt_attention_mask
    )
    tt_out = tt2torch_tensor(tt_out).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon CausalLM Passed!")
    else:
        logger.warning("Falcon CausalLM Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 128),),
    ids=["batch1_seqlen128"],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((2, 0.98), (32, 0.86)),
    ids=["layers_2", "layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_FalconCausalLM_inference(
    model_version,
    batch,
    seq_len,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tt_lib.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/falcon-7b_{request.node.callspec.id}"
    )

    run_test_FalconCausalLM_inference(
        device,
        model_version,
        batch,
        seq_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
    tt_lib.device.CloseDevice(device)
