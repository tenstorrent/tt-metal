# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import parse_optimizations


# These inputs override the default inputs used by simple_text_demo.py. Check the main demo to see the default values.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="input prompts json file")
    parser.addoption("--instruct", action="store", type=int, help="Use instruct weights")
    parser.addoption("--repeat_batches", action="store", type=int, help="Number of consecutive batches of users to run")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length supported by the model")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch ")
    parser.addoption(
        "--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate for each user"
    )
    parser.addoption("--data_parallel", action="store", type=int, help="Number of data parallel workers")
    parser.addoption(
        "--paged_attention", action="store", type=bool, help="Whether to use paged attention or default attention"
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
