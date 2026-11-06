# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the QWEN38_PAGED_KV test switch: the env parsing, the pool geometry and the block id choice."""

import pytest

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    paged_kv_block_ids,
    paged_kv_paging,
    paged_kv_settings,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import paged_table_width


def test_switch_is_off_by_default():
    assert paged_kv_settings({}) is None
    assert paged_kv_settings({"QWEN38_PAGED_KV": "0", "QWEN38_PAGED_SEED": "5"}) is None


def test_switch_defaults_and_overrides():
    assert paged_kv_settings({"QWEN38_PAGED_KV": "1"}) == (128, 0)
    env = {"QWEN38_PAGED_KV": "1", "QWEN38_PAGED_BLOCK_SIZE": "512", "QWEN38_PAGED_SEED": "7"}
    assert paged_kv_settings(env) == (512, 7)


@pytest.mark.parametrize("context, block_size", [(8192, 128), (32768, 128), (32768, 2048), (262144, 4096)])
def test_geometry_and_ids(context, block_size):
    paging = paged_kv_paging(context, block_size)
    width = paged_table_width(context, block_size)
    assert (paging.page_size, paging.max_blocks, paging.num_pages) == (block_size, width, width + 9)
    ids = paged_kv_block_ids(paging, 3)
    assert len(ids) == width
    assert len(set(ids)) == width
    assert all(1 <= block < paging.num_pages for block in ids)
    assert ids == paged_kv_block_ids(paging, 3)
    assert ids != paged_kv_block_ids(paging, 4)
