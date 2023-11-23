# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json

from models.experimental.mistral.tt.mistral_attention import TtAttention
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.model import Attention
from models.utility_functions import torch_to_tt_tensor_rm
from models.experimental.mistral.mistral_helper_funcs import unpad_from_zero, get_freqs_cis, format_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "rotary_embedding_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "softmax_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "empty_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "scatter_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "dtype",
    (tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.9793647197892646),),
)
def test_mistral_attention_inference(
    pcc,
    model_location_generator,
    device,
    reset_seeds,
    empty_ondevice,
    rotary_embedding_ondevice,
    softmax_ondevice,
    scatter_ondevice,
    dtype,
):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f"layers.0.attention."
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    if True:
        state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}
    model_args.max_batch_size = 1
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(state_dict)

    model_args.FALLBACK_ROTARY_EMBEDDING = rotary_embedding_ondevice
    model_args.FALLBACK_SOFTMAX = softmax_ondevice
    model_args.FALLBACK_EMPTY = empty_ondevice
    model_args.FALLBACK_SCATTER = scatter_ondevice
    model_args.WEIGHTS_DTYPE = dtype
    output_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    )
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/Mistral/"
    tt_model = TtAttention(
        args=model_args,
        device=device,
        base_address=base_address,
        tt_cache_path=tt_cache_path,
        output_mem_config=output_mem_config,
    )
    input = torch.randn(1, 11, 4096)
    seqlen = input.shape[1]
    empty_tensor = torch.zeros((11, 64))
    freqs_cis = torch.complex(empty_tensor, empty_tensor)
    query_shape = [1, 11, model_args.n_heads, model_args.head_dim // 2]
    key_shape = [1, 11, model_args.n_kv_heads, model_args.head_dim // 2]
    bcast_freq_xq, bcast_freq_xk = get_freqs_cis(freqs_cis, query_shape, key_shape, device, output_mem_config)
    positions = torch.arange(0, 11)
    mask = torch.randn(11, 11)

    reference_output = reference_model(input, freqs_cis, positions, mask=mask)
    del reference_model
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_position = torch_to_tt_tensor_rm(positions, device, put_on_device=False)
    mask = torch_to_tt_tensor_rm(mask, device, put_on_device=False)
    mask = format_tensor(mask, tt_lib.tensor.Layout.TILE, device, output_mem_config, pad_value=-10000)
    tt_input = format_tensor(tt_input, tt_lib.tensor.Layout.TILE, device, output_mem_config)
    tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)
    desired_shape = list(reference_output.shape)
    desired_shape.insert(0, 1)
    tt_output_torch = unpad_from_zero(tt_output, desired_shape).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Attention Passed!")
    else:
        logger.warning("Mistral_Attention Failed!")

    assert passing, f"Mistral_Attention output does not meet PCC requirement {pcc}."
