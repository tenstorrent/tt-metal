# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None
