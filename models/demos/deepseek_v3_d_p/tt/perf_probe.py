# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Env-gated perf/inspection probes for the Kimi prefill runner-vs-test investigation.

All probes are OFF by default and add ZERO overhead unless their env var is set, so this
module can sit in the hot path permanently. Both the runner (pipeline.prefill -> forward_chunk)
and the no-PCC transformer test call TtPrefillTransformer.forward_chunk, so instrumenting
forward_chunk / TtPrefillBlock.forward / ttMLA covers BOTH paths identically — the whole point
is an apples-to-apples comparison.

Toggles:
  PREFILL_SECTION_TIMING=1    -> per-chunk section breakdown (embed / mla / moe / dense), with a
                                 synchronize_device at each section boundary. Serializes the pipeline
                                 (inflates absolute ms) but attributes time to sections; since BOTH
                                 paths use the SAME instrumentation, the DIFFERENCE localizes the gap.
  PREFILL_DUMP_CONSTRUCTION=1 -> one-shot dump of TtPrefillTransformer construction config + the
                                 first forward_chunk's input/cache tensor specs. Diff runner vs test.
"""

import os
import time
from collections import defaultdict

from loguru import logger

import ttnn

SECTION_TIMING = os.environ.get("PREFILL_SECTION_TIMING", "0") == "1"
DUMP_CONSTRUCTION = os.environ.get("PREFILL_DUMP_CONSTRUCTION", "0") == "1"

_acc = defaultdict(float)
_count = defaultdict(int)
_dumped_input = False


class section:
    """Context manager: when SECTION_TIMING, sync-bracket a code region and accumulate its ms.

    Usage:  with section("mla", mesh_device): ...
    Sync on enter and exit so the measured wall time is device-completion time for this region
    (not async dispatch). No-op (no sync, no timing) when the toggle is off.
    """

    __slots__ = ("name", "dev", "t0")

    def __init__(self, name, mesh_device):
        self.name = name
        self.dev = mesh_device

    def __enter__(self):
        if SECTION_TIMING:
            ttnn.synchronize_device(self.dev)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if SECTION_TIMING:
            ttnn.synchronize_device(self.dev)
            dt = (time.perf_counter() - self.t0) * 1000.0
            _acc[self.name] += dt
            _count[self.name] += 1
        return False


def flush_sections(tag=""):
    """Emit the accumulated per-section ms for one chunk and reset. No-op when off."""
    if not SECTION_TIMING:
        return
    if not _acc:
        return
    total = sum(_acc.values())
    parts = " ".join(f"{k}={_acc[k]:.1f}(n={_count[k]})" for k in sorted(_acc))
    logger.info(f"[section-timing]{tag} total={total:.1f}ms {parts}")
    _acc.clear()
    _count.clear()


def _spec(t):
    try:
        return f"shape={list(t.shape)} dtype={t.dtype} layout={t.layout} mem={t.memory_config()}"
    except Exception as e:  # noqa: BLE001
        return f"<spec error: {e}>"


def dump_construction(xf):
    """One-shot dump of the constructed transformer's config knobs + shared KV buffer spec.

    Called at the end of TtPrefillTransformer.__init__. Diff the runner's dump vs the test's dump
    to find any construction difference (the open hypothesis is mla_seq_len: 61440 vs 56320).
    """
    if not DUMP_CONSTRUCTION:
        return
    lines = [
        "[construction-dump] TtPrefillTransformer:",
        f"  num_layers={xf.num_layers} seq_len={xf.seq_len} is_chunked={xf.is_chunked} "
        f"is_balanced={xf.is_balanced} padding_side={xf.padding_side}",
    ]
    # Pull a representative block's MLA knobs (all layers share construction params).
    try:
        blk = xf.layers[0]
        mla = blk.mla
        lines.append(
            f"  block0: is_moe={getattr(blk, 'is_moe', '?')} slot_num={getattr(blk, 'slot_num', '?')} "
            f"layer_num={getattr(blk, 'layer_num', '?')}"
        )
        lines.append(
            f"  mla: mla_seq_len={getattr(mla, 'mla_seq_len', '?')} seq_len={getattr(mla, 'seq_len', '?')} "
            f"sp_factor={getattr(mla, 'sp_factor', '?')} sp_axis={getattr(mla, 'sp_axis', '?')} "
            f"kv_lora_rank={getattr(mla, 'kv_lora_rank', '?')} scale={getattr(mla, 'scale', '?')}"
        )
        buf = getattr(mla, "_chunked_kv_buf", None)
        if buf is not None:
            lines.append(f"  shared _chunked_kv_buf: {_spec(buf)}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  <block/mla introspection error: {e}>")
    logger.info("\n".join(lines))


def dump_first_chunk_inputs(token_ids, kvpe_cache, kv_actual_isl, cache_user_id):
    """One-shot dump of the first forward_chunk's input + cache tensor specs. No-op when off."""
    global _dumped_input
    if not DUMP_CONSTRUCTION or _dumped_input:
        return
    _dumped_input = True
    logger.info(
        "[input-dump] forward_chunk first call:\n"
        f"  kv_actual_isl={kv_actual_isl} cache_user_id={cache_user_id}\n"
        f"  token_ids: {_spec(token_ids)}\n"
        f"  kvpe_cache: {_spec(kvpe_cache)}"
    )
