# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import parse_optimizations


def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="Input prompts json file")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length")
    parser.addoption("--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate")
    parser.addoption("--paged_attention", action="store", type=bool, help="Whether to use paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters")
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default",
    )
