# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``create_tt_model`` and the small helpers of the DeepSeek-V4.1-Flash demo (counterpart of ``models/demos/gpt_oss/tt/common.py``)."""

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.model_args import PAGE_TOKENS, DSV41ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig


def get_padded_prefill_len(seq_len: int) -> int:
    """Padded prompt length of the DSV4.1 prefill: a multiple of 128 (the chunk / halo / ring granularity), not a power of two (no per-bucket
    traces: the whole prompt is processed eagerly in chunks)."""
    return -(-seq_len // PAGE_TOKENS) * PAGE_TOKENS


def default_page_params(max_seq_len, users_per_row):
    """Pages (of 128 tokens) per mesh row for ``users_per_row`` users that may all reach ``max_seq_len`` tokens."""
    return {
        "page_block_size": PAGE_TOKENS,
        "page_max_num_blocks_per_dp": users_per_row * (-(-(max_seq_len + 128) // PAGE_TOKENS)),
    }


def create_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    layer_ids=None,
    kv_dtype=ttnn.bfloat16,
    log=print,
):
    """-> (model_args, model, tt_kv_cache, state_dict). ``tt_kv_cache`` is the ``PagedKVPool`` (the one pool prefill writes and decode reads);
    weights are read layer by layer from the checkpoint (no ``state_dict`` object): ``state_dict`` is None."""
    import os

    from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import Model

    if (
        int(os.environ.get("DSV41_SPEC", "0")) > 0
    ):  # speculative decoding needs the window ring to hold 128 + k rows (tt/spec_paged.py RING_SPEC)
        os.environ.setdefault("DSV41_RING_ROWS", "160")

    args = DSV41ModelArgs(mesh_device, max_batch_size, max_seq_len, layer_ids, paged_attention_config)
    if paged_attention_config is not None:
        assert paged_attention_config.block_size == PAGE_TOKENS, "the DSV4.1 paged pool uses 128-token pages"
    num_pages = paged_attention_config.max_num_blocks if paged_attention_config is not None else None
    model = Model(mesh_device, args, max_ctx=max_seq_len, num_pages=num_pages, kv_dtype=kv_dtype, log=log)
    return args, model, model.pool, None
