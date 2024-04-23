# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.utility_functions import get_devices_for_t3000
from models.demos.t3000.llama2_70b.tt.llama_common import get_llama_path
from models.demos.t3000.llama2_70b.tt.model_config import get_model_config
from models.demos.t3000.llama2_70b.demo.demo import build_generator, load_prompts_file, run_decode, construct_arg


def main(args):
    # Set random reproducible seed
    torch.manual_seed(0)

    generator = build_generator(args)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    tokenized, prompts = load_prompts_file(args, tokenizer)

    # Run decode
    with torch.no_grad():
        all_text = run_decode(args=args, model=model, tokenizer=tokenizer, prompt_tokens=tokenized, prompts=prompts)

        if args.output_at_end:
            with open("models/demos/t3000/llama3_70b/demo/data/demo_user_output.txt", "w") as f:
                for i, text in enumerate(all_text):
                    f.write(f"User {i}: {text}\n")


@pytest.mark.timeout(240000)
# @pytest.mark.parametrize("decode_only", (True, False), ids=["decode_only", "prefill_decode"])
@pytest.mark.parametrize("num_layers", (1, 2, 10, 80), ids=["1L", "2L", "10L", "80L"])
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    [
        ("tt", False, 8),
        ("meta", False, 8),
    ],
    ids=["tt-70b", "meta-70b"],
)
@pytest.mark.parametrize(
    "num_tokens, prompts_file, output_at_end, top_p, top_k, temperature",
    [
        (128, "models/demos/t3000/llama3_70b/demo/data/multi_prompt.json", True, 1, 1, 1.0),
        (128, "models/demos/t3000/llama3_70b/demo/data/multi_prompt.json", True, 0.9, 10, 1.0),
    ],
    ids=["greedy", "sampling"],
)
def test_LlamaModel_demo(
    implementation,
    skip_model_load,
    num_layers,
    num_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    all_devices,
    n_devices,
    # decode_only,
    use_program_cache,
):
    ## Get model config
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices)
    model_config_default = get_model_config()

    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(devices) < n_devices:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if (
        compute_grid_size.x < model_config_default["MAX_GRID_SIZE"][0]
        or compute_grid_size.y < model_config_default["MAX_GRID_SIZE"][1]
    ):
        pytest.skip(f"Requires grid size of at least {model_config_default['MAX_GRID_SIZE']} to run")

    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config_default, n_devices, False)

    args = construct_arg(
        implementation=implementation,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        num_tokens=num_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        devices=devices,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=True,
    )
    main(args)
