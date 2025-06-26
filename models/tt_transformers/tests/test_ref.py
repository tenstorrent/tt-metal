# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    ref_model = model_args.reference_attention()
    ref_model.load_state_dict(partial_state_dict)

    from transformers import AutoModelForCausalLM

    hf_transformer = AutoModelForCausalLM.from_pretrained(model_args.CKPT_DIR)
    hf_model = hf_transformer.model.layers[0].self_attn
    hf_model.eval()

    # Get the state dicts
    ref_state_dict = ref_model.attention.state_dict()  # should contain hf keys and weights
    hf_state_dict = hf_model.state_dict()

    for key in ["k_proj", "q_proj"]:
        for suffix in ["weight", "bias"]:
            print(
                f"{key}.{suffix}: ref matches hf : {torch.allclose(ref_state_dict[key + '.' + suffix], hf_state_dict[key + '.' + suffix])}"
            )

    print(" ".join(f"{x:+3.1f}" for x in ref_state_dict["k_proj.bias"]))
    print(" ".join(f"{x:+3.1f}" for x in hf_state_dict["k_proj.bias"]))
