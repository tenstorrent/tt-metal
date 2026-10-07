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


def _env_setup(kv_dtype, batch=None, max_seq_len=None, mesh_rows=4):
    """Build-time environment. The spec default of this batch size (tt/spec_policy.py, the same resolution the demo and ``Generator.enable_spec`` use) decides DSV41_RING_ROWS=288
    (read by the pool at build / reconfigure time; only when spec decode will run) and DSV41_SPEC_ROWS (B >= 64 opt-in).
    """
    import os

    if batch is not None:
        from models.demos.blackhole.deepseek_v41_flash.tt import spec_policy

        spec_policy.resolve_apply(batch, mesh_rows, max_seq_len)
    if os.environ.get("DSV41_POOL_DTYPE", "bf16") == "fp8":  # fp8_e4m3 KV pool (halves the pool: batch 128 at 64k)
        kv_dtype = ttnn.fp8_e4m3
    return kv_dtype


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

    from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import Model

    kv_dtype = _env_setup(kv_dtype, max_batch_size, max_seq_len, mesh_device.shape[0])

    args = DSV41ModelArgs(mesh_device, max_batch_size, max_seq_len, layer_ids, paged_attention_config)
    if paged_attention_config is not None:
        assert paged_attention_config.block_size == PAGE_TOKENS, "the DSV4.1 paged pool uses 128-token pages"
    num_pages = paged_attention_config.max_num_blocks if paged_attention_config is not None else None
    from models.demos.blackhole.deepseek_v41_flash.tt.build_slots import build_slot

    n_layers = args.n_layers
    with build_slot(
        n_layers, log=log
    ):  # cluster-wide cap on concurrent full builds (NFS weight reads), see tt/build_slots.py
        model = Model(mesh_device, args, max_ctx=max_seq_len, num_pages=num_pages, kv_dtype=kv_dtype, log=log)
    return args, model, model.pool, None


def reconfigure_tt_model(
    model,
    mesh_device,
    max_batch_size,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    layer_ids=None,
    kv_dtype=ttnn.bfloat16,
    log=print,
    generator_hooks=(),
):
    """-> (model_args, model, tt_kv_cache, state_dict) like ``create_tt_model``, but for an EXISTING ``model`` (same layers): its batch dependent device
    state is released and rebuilt for ``max_batch_size`` / ``max_seq_len`` while the weights stay on the device (``Model.reconfigure``). Takes minutes
    instead of the 60-90 minutes of a full build and does not run out of DRAM like a second ``create_tt_model`` in the same process.
    """
    kv_dtype = _env_setup(kv_dtype, max_batch_size, max_seq_len, mesh_device.shape[0])
    args = DSV41ModelArgs(mesh_device, max_batch_size, max_seq_len, layer_ids, paged_attention_config)
    assert list(args.layer_ids) == list(model.layer_ids), "reconfigure keeps the weights: the layer set cannot change"
    if paged_attention_config is not None:
        assert paged_attention_config.block_size == PAGE_TOKENS, "the DSV4.1 paged pool uses 128-token pages"
    num_pages = paged_attention_config.max_num_blocks if paged_attention_config is not None else None
    model.log = log
    model.reconfigure(args, max_seq_len, num_pages=num_pages, kv_dtype=kv_dtype, generator_hooks=generator_hooks)
    return args, model, model.pool, None
