# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import sys
import ttnn
import torch
import model_config
from transformers import AutoTokenizer


from models.experimental.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model(version, batch_size):
    from models.experimental.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(f"state-spaces/{version}", batch_size=batch_size)


def get_tt_metal_model(batch_size, hidden_size, configs, version):
    from models.experimental.mamba.tt_opt.full_model import MambaTT

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    torch.manual_seed(0)
    ttnn.enable_program_cache(device)

    reference_model = get_cpu_reference_model(version, batch_size)
    cache_path = f"/tmp/state-spaces/{version}"

    model = MambaTT(reference_model, device, configs, cache_path, 1)
    return model, device


def run_demo(batch_size, hidden_size, profile):
    configs = model_config.create_model_config(batch_size, hidden_size)
    model, device = get_tt_metal_model(batch_size, hidden_size, configs, "mamba-2.8b-slimpj")

    # evaluate model:
    model.eval()

    with torch.no_grad():
        # create random torch tensor of hidden size and batch size, with datatype bfloat16

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        input_data = tokenizer("Hello", return_tensors="pt")["input_ids"]
        input_data = input_data.repeat(batch_size, 1)

        if profile == 1:
            out_data = model(input_data)
        out_data = model(input_data)

    ttnn.close_device(device)

    return out_data


def main():
    batch_size = int(sys.argv[1])
    hidden_size = int(sys.argv[2])
    profile = int(sys.argv[3])
    assert batch_size == 32
    assert (hidden_size // 8) % 32 == 0
    return run_demo(batch_size, hidden_size, profile)


if __name__ == "__main__":
    main()
