# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 G2 capacity stand-in: the stock prefill runner on GLM-5.3 layers 26-53, reshaped to G2's memory.

GLM-5.4's loop pass 2 (G2) does not exist yet (q171.2.5 / q171.2.6 / q171.2.7). For the capacity bisection
(tt_metal_tracker-q171.3.5) this launcher patches the GLM-5.3 runner before it starts, then runs it unchanged:

* every layer of the probe's range is ``shared`` (pass 2 runs no indexer, A5): no indexer weights load, no index-K
  cache and no indexer score buffer are allocated, and the offset first rank feeds the layers placeholder top-k
  (``TtPrefillRuntime.make_placeholder_indices``, passed to ``prefill_chunk``), the way G2 would consume the sets
  G1 forwards;
* next to the KVPE cache (G2's bank A: the pass-1 copy, same shape as G1's), it allocates what G2 adds that the
  runner does not: the pass-2 window, 128 rows x 704 B per loop layer per user, replicated on every chip (an upper
  bound), and one chunk's receive buffers for the 7 top-k sets and the 28 layers' pass-1 KVPE rows.

Not represented: the two-bank attention's own working buffers and the damp gate (scalar, negligible). Use with
``STAGE_PROBE_RUNNER_MODULE=models.demos.deepseek_v3_d_p.scripts.stage_probe.capacity_probe_runner`` and
``PREFILL_PP_LAYER_OFFSET=26 PREFILL_NUM_LAYERS=28``.
"""

from __future__ import annotations

import os
import runpy

from loguru import logger

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config

LOOP = GLM54Config.loop_layers()
WINDOW = GLM54Config.LOOP_WINDOW_SIZE
LOOP_FULL_LAYERS = sum(1 for i in LOOP if GLM53Config.indexer_types()[i] == "full")

# Allocations kept alive for the runner's lifetime.
_EXTRA = []


def _all_shared_loop() -> None:
    types = GLM53Config.indexer_types()
    GLM53Config.INDEX_TOPK_PATTERN = "".join(
        "S" if i in LOOP else ("F" if t == "full" else "S") for i, t in enumerate(types)
    )
    logger.info(f"[g2-standin] indexer map forced shared on layers {LOOP.start}-{LOOP.stop - 1}")


def _patch_allocate_kv_cache() -> None:
    import ttnn
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, init_mla_kv_cache

    def allocate_kv_cache(self, *, mesh_device, hf_config, params):
        first, last = params.first_layer_idx, params.first_layer_idx + params.num_layers - 1
        assert (first, last) == (LOOP.start, LOOP.stop - 1) and not params.mtp_levels, "G2 stand-in is layers 26-53"
        slots = self.kv_cache_slots(params)
        kvpe = init_mla_kv_cache(
            cache_format=self.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
            hf_config=hf_config,
            mesh_device=mesh_device,
            seq_len=params.max_seq_len,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=params.num_layers,
            num_users=slots,
            tp_axis=params.tp_axis,
            bank_aligned=True,
        )
        geometry = MlaKvCacheGeometry.from_config(hf_config)
        row = -(-geometry.packed_row_bytes // 64) * 64
        sp, tp = params.mesh_shape[params.sp_axis], params.mesh_shape[params.tp_axis]
        q_rows = params.chunk_size // (sp * tp)

        def alloc(shape, dtype, what):
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            _EXTRA.append(t)
            logger.info(f"[g2-standin] {what}: {list(shape)} {dtype} per chip")

        # Window: 128 rows of one packed row per loop layer per user, bytes as uint8.
        alloc([slots * params.num_layers, 1, WINDOW, row], ttnn.uint8, "pass-2 window")
        alloc([LOOP_FULL_LAYERS, 1, q_rows, GLM53Config.INDEX_TOPK], ttnn.uint32, "top-k receive (7 sets)")
        alloc([params.num_layers, 1, q_rows, row], ttnn.uint8, "pass-1 KVPE receive")
        return MlaKvCaches(kvpe=kvpe, index=None)

    GLM53Adapter.allocate_kv_cache = allocate_kv_cache


def _patch_placeholder_indices() -> None:
    """prefill_runner passes top-k only from an upstream rank's D2D segment, and an offset first rank has none,
    yet ``prefill_chunk`` requires them on a rank that starts on a shared layer. Hand it a fresh placeholder set
    (zeros, ``make_placeholder_indices``) per chunk: ``prefill_chunk`` consumes (copies, then deallocates) the
    set it is given, as it does a received D2D segment."""
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    prefill_chunk = TtPrefillRuntime.prefill_chunk

    def prefill_chunk_with_indices(self, *args, **kwargs):
        if self.receives_indexer_indices and kwargs.get("indexer_indices") is None:
            kwargs["indexer_indices"] = self.make_placeholder_indices()
        return prefill_chunk(self, *args, **kwargs)

    TtPrefillRuntime.prefill_chunk = prefill_chunk_with_indices


def main() -> None:
    assert os.environ.get("PREFILL_MODEL", "glm_5_3") == "glm_5_3"
    _all_shared_loop()
    _patch_allocate_kv_cache()
    _patch_placeholder_indices()
    runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__")


if __name__ == "__main__":
    main()
