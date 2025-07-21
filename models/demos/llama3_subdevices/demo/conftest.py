# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


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
    parser.addoption(
        "--paged_attention", action="store", type=bool, help="Whether to use paged attention or default attention"
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--disable_pf_perf_mode", action="store_true", default=False, help="Enable performance mode for prefetcher"
    )
    parser.addoption(
        "--print_outputs",
        action="store",
        default=False,
        type=bool,
        help="Whether to print token output every decode iteration",
    )
