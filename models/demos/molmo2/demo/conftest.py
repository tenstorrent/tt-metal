# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Register demo-specific CLI options so pytest_addoption in demo.py works
# when running from the repo root (root conftest.py takes priority otherwise).


def pytest_addoption(parser):
    parser.addoption("--input_prompts", default=None, help="JSON prompt file override")
    parser.addoption("--max_new_tokens", default=None, type=int)
    parser.addoption("--batch_size", default=None, type=int)
