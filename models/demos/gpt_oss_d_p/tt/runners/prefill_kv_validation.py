# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""GPT-OSS device-less KV read-back + golden PCC for the common prefill producer.

Called from ``models/demos/common/prefill/runners/prefill_producer.py`` in Gate 1 (mock-migration)
and Gate 2b (burst-loopback anchored to the golden). Mirrors
``minimax_m3/tt/runners/prefill_kv_validation.py``, minus ``index_k`` — GPT-OSS is dense GQA with no
sparse lightning indexer, matching ``tt/runners/kv_chunk_table.py``'s 2N-config layout
(K_h0..N-1 = 0..N-1, V_h0..N-1 = N..2N-1).

Device K stores the rotary slice Meta-interleaved (see ``model_config.convert_hf_qkv_to_meta_format``),
while the golden trace is HF half-split; we permute the golden's rotary slice before comparing. V is
raw (no swizzle).
"""

from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open


def _hf_to_meta_rotary_perm(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """HF half-split -> device Meta interleaved (identity tail if partial rotary). Same as
    ``minimax_m3/tt/runners/prefill_kv_validation.py::_hf_to_meta_rotary_perm`` and the inline
    permutation in ``TtPrefillRuntime.kv_cache_pcc_check``."""
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    return torch.tensor(src, dtype=torch.long)


def read_slot_kv_and_check_pcc(
    *,
    table,
    device_map,
    slot_id: int,
    real_len: int,
    trace_dir: str,
    num_layers: int,
    read_kv_slice,
    decode_bfp8_chunk,
):
    """Read slot ``slot_id``'s KV over [0, real_len) via the address table and PCC vs the golden trace.
    Returns the min PCC across K, V, and all layers. ``read_kv_slice`` and ``decode_bfp8_chunk`` are the
    common producer's private helpers threaded in as callables so this module doesn't reach into it."""
    from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    n_kv = GptOss120BConfig.NUM_KEY_VALUE_HEADS
    head_dim = GptOss120BConfig.HEAD_DIM
    rotary_dim = getattr(GptOss120BConfig, "ROTARY_DIM", head_dim)  # full rotary for GPT-OSS
    perm = _hf_to_meta_rotary_perm(head_dim, rotary_dim)

    read_len = ((real_len + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1) // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * (
        NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    kv_dir = Path(trace_dir) / "kv_cache"
    mins = {"k": 1.0, "v": 1.0}
    for layer in range(num_layers):
        dev_k = torch.stack(
            [read_kv_slice(table, device_map, h, layer, slot_id, read_len, head_dim, decode_bfp8_chunk) for h in range(n_kv)],
            dim=0,
        )[:, :real_len]
        dev_v = torch.stack(
            [
                read_kv_slice(table, device_map, n_kv + h, layer, slot_id, read_len, head_dim, decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as fh:
            g_k = fh.get_tensor(f"key_cache_layer_{layer}").float()[0, :, :real_len, :][..., perm]  # HF -> Meta
            g_v = fh.get_tensor(f"value_cache_layer_{layer}").float()[0, :, :real_len, :]

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        logger.info(f"  layer {layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")

    min_pcc = min(mins.values())
    logger.info(
        f"[producer] slot {slot_id} GPT-OSS KV PCC over [0,{real_len}) across {num_layers} layers -> "
        f"K={mins['k']:.5f} V={mins['v']:.5f} (min {min_pcc:.6f})"
    )
    return min_pcc
