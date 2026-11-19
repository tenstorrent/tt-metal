# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated TP4 SiLU movement candidates; lookup/compute arithmetic is unchanged.

Call ``install(dec)`` during setup for eight workers with initialization once
per physical CB page. ``install(dec, variant="tile_workers")`` instead uses 32
workers, one (head, tile) per worker and width shards [32,32]. Measure the latter
through the following projection, including any resulting layout conversion.

The existing immutable table is reused by the eight-worker candidate. Outputs
retain the baseline transient allocation/trace lifetime. No extra persistent
runtime tensor or host conversion is introduced. The 32-worker candidate owns
a setup-loaded compact lookup table on all 32 workers, retained by the helper.
"""

import copy
import types

from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import (
    LOCAL_SILU_COMPUTE_SOURCE,
    LOCAL_SILU_READER_SOURCE,
    SILU_COMPACT_ENTRIES,
    compact_silu_math_table,
    local_silu_math_table,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import TP4LocalSiluMultiply


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("SiLU probe source contract changed; inspect before adapting the candidate")
    return source.replace(old, new, 1)


def once_per_page_reader(source):
    """Both pages are initialized before the first publication to compute.

    reserve_back does not advance the writer pointer. Its two-page reservation
    proves the contiguous ring is free at entry; per-tile reserve/push remains
    unchanged. At tiles 2/3, reuse overwrites every valid row word, while all
    other words retain their initial zero. Compute never writes these buffers.
    """
    source = _replace_once(
        source,
        " for(uint32_t tile=0;tile<4;++tile) {",
        """ // CBs 0 and 1 each contain exactly two contiguous 4096-byte pages.
 weights.reserve_back(2);gates.reserve_back(2);
 auto* all_w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
 auto* all_g=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 for(uint32_t j=0;j<2048;++j) {all_w[j]=0;all_g[j]=0;}
 for(uint32_t tile=0;tile<4;++tile) {""",
    )
    return _replace_once(source, "  for(uint32_t j=0;j<1024;++j) {w[j]=0;g[j]=0;}\n", "")


def tile_worker_sources(reader, compute):
    reader = _replace_once(
        reader,
        " const uint32_t head=get_arg_val<uint32_t>(3);",
        " const uint32_t task=get_arg_val<uint32_t>(3),head=task/4,first_tile=task%4;",
    )
    reader = _replace_once(
        reader, " for(uint32_t tile=0;tile<4;++tile) {", " for(uint32_t tile=first_tile;tile<first_tile+1;++tile) {"
    )
    compute = _replace_once(compute, " for(uint32_t tile=0;tile<4;++tile) {", " for(uint32_t tile=0;tile<1;++tile) {")
    # This independent family uses only the first page of each unchanged
    # two-page CB; its original per-tile initialization happens exactly once.
    return reader, compute


def _clone_call(reader, compute):
    original = TP4LocalSiluMultiply.__call__
    namespace = dict(original.__globals__, LOCAL_SILU_READER_SOURCE=reader, LOCAL_SILU_COMPUTE_SOURCE=compute)
    function = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    function.__kwdefaults__ = original.__kwdefaults__
    return function


class _OncePerPageSilu(TP4LocalSiluMultiply):
    __call__ = _clone_call(once_per_page_reader(LOCAL_SILU_READER_SOURCE), LOCAL_SILU_COMPUTE_SOURCE)


class _TileWorkerSilu(TP4LocalSiluMultiply):
    __call__ = _clone_call(*tile_worker_sources(LOCAL_SILU_READER_SOURCE, LOCAL_SILU_COMPUTE_SOURCE))


def install(dec, variant="once_per_page"):
    """Replace only this decoder's helper and return the previous helper.

    The return value permits an explicit component A/B with identical weighted
    and Z inputs. Compare both output and optional debug-gate bits on all four
    chips; include repeated traced invocations with changing inputs. For
    tile_workers, convert both results to the same layout outside the component
    correctness comparison, but include that conversion in chain timing.
    """
    if dec.kind != "linear_attention":
        raise ValueError("SiLU probe applies only to TP4 linear-attention decoders")
    if variant not in ("once_per_page", "tile_workers"):
        raise ValueError("unknown SiLU probe variant")
    original = dec.decode_local_silu
    if type(original) is not TP4LocalSiluMultiply:
        raise ValueError("install SiLU probe once on the unmodified baseline helper")
    candidate = copy.copy(original)
    candidate.__class__ = _OncePerPageSilu if variant == "once_per_page" else _TileWorkerSilu
    if variant == "tile_workers":
        t, mesh = candidate.t, candidate.mesh
        grid = mesh.compute_with_storage_grid_size()
        if grid.x * grid.y < 32:
            raise ValueError("tile-worker SiLU requires 32 compute cores")
        candidate.coordinates = [(i % grid.x, i // grid.x) for i in range(32)]
        candidate.cores = t.num_cores_to_corerangeset(32, grid, row_wise=True)
        candidate.output_memory = t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(candidate.cores, (32, 32), t.ShardOrientation.ROW_MAJOR),
        )
        table_memory = t.MemoryConfig(
            t.TensorMemoryLayout.HEIGHT_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(candidate.cores, (1, SILU_COMPACT_ENTRIES), t.ShardOrientation.ROW_MAJOR),
        )
        candidate.table = t.from_torch(
            compact_silu_math_table(local_silu_math_table()).expand(32, SILU_COMPACT_ENTRIES).contiguous(),
            dtype=t.float32,
            layout=t.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=table_memory,
        )
    dec.decode_local_silu = candidate
    dec.precision_policy["experimental_local_silu"] = {
        "variant": variant,
        "workers": len(candidate.coordinates),
        "output_shard": [32, 128 if variant == "once_per_page" else 32],
        "lookup_and_arithmetic": "unchanged",
        "buffers": "original two-page FP32 weights/gates; original aligned BF16 code scratch",
    }
    return original
