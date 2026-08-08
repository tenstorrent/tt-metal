# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small host-side helpers shared across the model.

  * general_utils.py — get_cache_file_name: build the on-disk path for a cached
                       (sharded/converted) weight tensor under the model's cache dir.
"""
