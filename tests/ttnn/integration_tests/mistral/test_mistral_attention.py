# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
import ttnn.deprecated
from ttnn.model_preprocessing import preprocess_model_parameters

import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.reference.model import Attention
from models.experimental.mistral.mistral_helper_funcs import get_freqs_cis
from models.experimental.functional_mistral.tt.ttnn_functional_attention import attention
from models.utility_functions import skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-metal/issues/9076")
@skip_for_wormhole_b0()
def test_mistral_attention_inference(model_location_generator, device, reset_seeds):
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=1, is_whole_model=False)
    state_dict = torch.load(model_path / "consolidated.00.pth")
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    ref_model = transformer.layers[0].attention
    ref_model.eval()

    model_args.max_batch_size = 1
    model_args.n_layers = 32

    model_args.max_batch_size = 1

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(state_dict)
    output_mem_config = ttnn.experimental.tensor.MemoryConfig(
        ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: ref_model,
        device=device,
    )

    input = torch.rand(1, 11, 4096)
    seqlen = input.shape[1]
    empty_tensor = torch.zeros((11, 64))
    freqs_cis = torch.complex(empty_tensor, empty_tensor)
    query_shape = [1, 11, model_args.n_heads, model_args.head_dim // 2]
    key_shape = [1, 11, model_args.n_kv_heads, model_args.head_dim // 2]
    bcast_freq_xq, bcast_freq_xk = get_freqs_cis(freqs_cis, query_shape, key_shape, device, output_mem_config)
    positions = torch.arange(0, 11)
    mask = torch.randn(11, 11)
    reference_ouput = reference_model(input, freqs_cis, positions, mask=mask)

    ttnn_input = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(input, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )
    positions = ttnn.from_torch(positions, dtype=ttnn.bfloat16)
    mask = ttnn.to_layout(ttnn.from_torch(mask, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT)
    output = attention(
        model_args,
        ttnn_input,
        bcast_freq_xq,
        bcast_freq_xk,
        positions,
        mask,
        seqlen,
        parameters=parameters,
        device=device,
        mem_config=output_mem_config,
    )

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)

    output = ttnn.to_torch(output).squeeze(0)
    assert_with_pcc(reference_ouput, output.to(reference_ouput.dtype), 0.98)
