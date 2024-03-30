# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch
from models.experimental.mamba.tt_opt import model_config
from transformers import AutoTokenizer
import pytest



from models.experimental.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model(version):
    from models.experimental.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(f"state-spaces/{version}")


def get_tt_metal_model(device, configs, version):
    from models.experimental.mamba.tt_opt.full_model import MambaTT


    torch.manual_seed(0)
    

    reference_model = get_cpu_reference_model(version)
    cache_path = f"/tmp/state-spaces/{version}"

    model = MambaTT(reference_model, device, configs, cache_path, 1)
    return model


def run_demo(device, num_users, hidden_size, profile):
    #return
    configs = model_config.create_model_config(num_users, hidden_size)
    model = get_tt_metal_model(device, configs, 'mamba-2.8b-slimpj')

    # evaluate model:
    model.eval()

    with torch.no_grad():
        # create random torch tensor of hidden size and num_users, with datatype bfloat16

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        
        input_data = tokenizer("Hello", return_tensors="pt")["input_ids"]
        input_data = input_data.repeat(num_users, 1)
        out_data = model(input_data)     

        if profile == 1:
            input_data = tokenizer("Hello", return_tensors="pt")["input_ids"]
            input_data = input_data.repeat(num_users, 1)
            out_data = model(input_data)

    

    
    return out_data


def main_func(device, num_users, hidden_size, profile):
    assert num_users == 32
    assert (hidden_size // 8) % 32 == 0
    return run_demo(device, num_users, hidden_size, profile)

@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("hidden_size", [2560])
@pytest.mark.parametrize("profile", [0, 1])
def test_mamba(
    device, use_program_cache, num_users, hidden_size, profile
):
    main_func(device, num_users, hidden_size, profile)


