# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate 1 read-back for DeepSeek-V4: the producer reads each migrated config through the KV chunk table,
off the runner's device DRAM, and grades it against the vLLM golden trace (``tt/v4/kv_caches.py`` for the
configs, ``tests/v4/golden.py`` for the trace's streams).

  HCA / CSA compressed rows -> ``compressed_entries/compressed_entries_layer_<L>``, ``[entries, head_dim]``,
      graded per half (nope, then the natively interleaved rope), as ``test_block_chunked`` grades HCA;
  CSA index keys -> ``indexer_key_cache/indexer_key_cache_layer_<L>``, ``[entries, index_head_dim]``. The
      device stores them Hadamard-rotated, so they are rotated back first, as the MLA read-back does.

A run of ``real_len`` tokens fills ``real_len // rate`` entries, and the golden holds the cache after the whole
prompt, so the first that many golden rows are the reference. A layer whose stream the trace does not keep is
logged and left ungraded; a layer on another host's chips is not this producer's to read.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import CACHE_KINDS, CSA, HCA
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, read_sharded_rows

COMPRESSED_STREAM = ("compressed_entries", "compressed_entries_layer_{}")
INDEX_STREAM = ("indexer_key_cache", "indexer_key_cache_layer_{}")


def golden_rows(trace_dir, stream, layer: int, count: int):
    """The first ``count`` rows of one layer's stream, or None when the trace does not keep it."""
    group, name = stream[0], stream[1].format(layer)
    path = Path(trace_dir) / group / name
    return read_sharded_rows(path, name, 0, count) if path.is_dir() else None


def _decode_row(raw: bytes, width: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().view(1, width)


def read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir, model_config, num_layers):
    """``{kind: min PCC}`` over this host's layers of ``slot_id``'s first ``real_len`` tokens."""
    from models.demos.common.prefill.runners.prefill_producer import (
        _decode_bf16_chunk,
        _decode_bfp8_chunk,
        _read_kv_slice,
        _resolve_unique_id,
    )
    from tests.ttnn.utils_for_testing import comp_pcc

    config = v4_hf_config(model_config)
    nope = config.head_dim - config.qk_rope_head_dim
    unrotate = normalized_hadamard_matrix(config.index_head_dim).float()
    page = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    # kind -> (config id, attention, row width, decoder, rows a chunk, golden stream)
    kinds = {
        CACHE_KINDS[0]: (0, HCA, config.head_dim, _decode_bf16_chunk, page, COMPRESSED_STREAM),
        CACHE_KINDS[1]: (1, CSA, config.head_dim, _decode_row, 1, COMPRESSED_STREAM),
        CACHE_KINDS[2]: (2, CSA, config.index_head_dim, _decode_bfp8_chunk, page, INDEX_STREAM),
    }

    mins, ungraded = {}, []
    for layer in range(num_layers):
        for kind, (config_id, attention, width, decode, step, stream) in kinds.items():
            if config.layer_types[layer] != attention:
                continue
            loc = table.lookup(layer, 0, slot_id, config_id)
            try:
                _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            except KeyError:
                continue
            entries = real_len // config.compress_rates[attention]
            golden = golden_rows(trace_dir, stream, layer, entries)
            if golden is None:
                ungraded.append(f"{layer}:{kind}")
                continue
            device = _read_kv_slice(table, device_map, config_id, layer, slot_id, entries, width, decode, step=step)
            if kind == CACHE_KINDS[2]:
                pcc = float(comp_pcc(golden, device.float() @ unrotate)[1])
            else:
                pcc = min(cache_half_pccs(golden, device.float(), nope, pe_interleave=False))
            mins[kind] = min(mins.get(kind, 1.0), pcc)
            logger.info(f"[producer] slot {slot_id} layer {layer:>2} {kind} over {entries} entries: PCC {pcc:.5f}")

    if ungraded:
        logger.warning(f"[producer] slot {slot_id}: no golden for {len(ungraded)} layer cache(s), ungraded: {ungraded}")
    if not mins:
        raise RuntimeError(f"slot {slot_id}: no local layer had a golden to grade against (nothing verified)")
    logger.info(f"[producer] slot {slot_id} V4 KV PCC over [0,{real_len}): {mins}")
    return mins
