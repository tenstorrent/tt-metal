# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


# These inputs override the parametrized defaults in demo.py.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", default=None, help="input prompts json file")
    parser.addoption("--max_new_tokens", action="store", default=None, type=int, help="tokens to generate per user")
    parser.addoption("--batch_size", action="store", default=None, type=int, help="number of users in a batch")
