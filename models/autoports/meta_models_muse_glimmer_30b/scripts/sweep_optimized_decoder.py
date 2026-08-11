# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced warmed decode and warmed prefill latency for optimized-decoder candidates.

``CANDIDATES`` is a JSON list of dicts, each one a set of ``OptimizedDecoder.from_state_dict``
keyword overrides plus an optional ``label``. Every candidate is built and measured in the
same process on the same device, so the numbers are directly comparable; a candidate that
fails to build or to run is recorded with its error instead of aborting the sweep.

This is the harness behind ``doc/optimized_decoder/perf/candidates.json``. It uses the same
measured window shape as ``tests/test_optimized_decoder_perf.py`` (compile+warm,
synchronize, measure, synchronize) but without Tracy, because a sweep only needs relative
wall clock.

    CANDIDATES='[{"label":"bfp4_all","precision":"bfp4_all"},
                 {"label":"bfp8_lofi","precision":"bfp8_all_lofi"}]' \
    KINDS=sliding_rope DECODE_ITERS=32 CONTEXT=4096 PREFILL_SEQ=8192 \
      python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_optimized_decoder.py

Environment: ``CANDIDATES``, ``KINDS``, ``CONTEXT``, ``BATCH``, ``DECODE_ITERS``,
``PREFILL_ITERS``, ``PREFILL_SEQ`` (0 skips prefill timing), ``OUT``.
"""

import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder

ART = Path("models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/weight_stats.json")
CANDIDATES = json.loads(os.environ.get("CANDIDATES", '[{"precision": "bfp8_attn_bfp4_mlp"}]'))
KINDS = os.environ.get("KINDS", "sliding_rope").split(",")
CONTEXT = int(os.environ.get("CONTEXT", "4096"))
BATCH = int(os.environ.get("BATCH", "1"))
DECODE_ITERS = int(os.environ.get("DECODE_ITERS", "32"))
PREFILL_ITERS = int(os.environ.get("PREFILL_ITERS", "3"))
PREFILL_SEQ = int(os.environ.get("PREFILL_SEQ", "0"))  # 0 = skip prefill timing
OUT = Path(os.environ.get("OUT", "/tmp/perf_probe.json"))

R.configure_host_threads()
text_config = R.load_text_config()
kinds = R.layer_kinds(text_config)
stats = json.loads(ART.read_text())

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
rows = []
try:
    for kind_id in KINDS:
        kind = kinds[kind_id]
        sd = R.synthetic_layer_state_dict(stats["layers"][str(kind.layer_idx)]["tensors"], seed=7)
        for cand in CANDIDATES:
            label = cand.get("label", json.dumps({k: v for k, v in cand.items() if k != "label"}))
            kwargs = {k: v for k, v in cand.items() if k != "label"}
            if "decode_residual_grid" in kwargs:
                kwargs["decode_residual_grid"] = tuple(kwargs["decode_residual_grid"])
            if "prefill_matmul_grid" in kwargs:
                kwargs["prefill_matmul_grid"] = tuple(kwargs["prefill_matmul_grid"])
            if "decode_sdpa_core_grid" in kwargs:
                kwargs["decode_sdpa_core_grid"] = tuple(kwargs["decode_sdpa_core_grid"])
            for key in ("cache_dtype", "weight_dtype"):
                if isinstance(kwargs.get(key), str):
                    kwargs[key] = getattr(ttnn, kwargs[key])
            try:
                dec = OptimizedDecoder.from_state_dict(
                    sd,
                    hf_config=text_config,
                    layer_idx=kind.layer_idx,
                    mesh_device=mesh,
                    block_size=64,
                    **kwargs,
                )
            except Exception as exc:  # candidate is illegal at build time
                print(f"PERF {kind_id} {label} BUILD_FAIL {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                rows.append(dict(kind=kind_id, label=label, error=f"build: {str(exc)[:400]}"))
                continue

            row = dict(kind=kind_id, label=label, batch=BATCH, context=CONTEXT, config=dec.config_summary())
            blocks = dec.blocks_per_seq(CONTEXT + 1)
            total_blocks = blocks * BATCH + 2
            pt = U.build_page_table(batch=BATCH, blocks_per_seq=blocks, total_blocks=total_blocks, seed=4242)
            kv = dec.allocate_kv_cache(batch_size=BATCH, max_seq_len=blocks * 64, num_blocks=total_blocks)
            ptt = U.page_table_to_device(pt, mesh)
            try:
                prompt = R.unit_rms_hidden_states((1, CONTEXT, text_config.hidden_size), seed=62)
                prompt_tt = U.prefill_input(prompt, mesh)
                for user in range(BATCH):
                    dec.prefill_forward(prompt_tt, kv_cache=kv, page_table=ptt, user_ids=[user]).deallocate(True)
                prompt_tt.deallocate(True)
                ttnn.synchronize_device(mesh)

                if PREFILL_SEQ:
                    ph = R.unit_rms_hidden_states((1, PREFILL_SEQ, text_config.hidden_size), seed=61)
                    pblocks = dec.blocks_per_seq(PREFILL_SEQ + 1)
                    ppt = U.page_table_to_device(
                        U.build_page_table(batch=1, blocks_per_seq=pblocks, total_blocks=pblocks + 2, seed=7), mesh
                    )
                    pkv = dec.allocate_kv_cache(batch_size=1, max_seq_len=pblocks * 64, num_blocks=pblocks + 2)
                    pt_tt = U.prefill_input(ph, mesh)
                    dec.prefill_forward(pt_tt, kv_cache=pkv, page_table=ppt).deallocate(True)
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(PREFILL_ITERS):
                        dec.prefill_forward(pt_tt, kv_cache=pkv, page_table=ppt).deallocate(True)
                    ttnn.synchronize_device(mesh)
                    row["prefill_ms"] = 1000.0 * (time.perf_counter() - start) / PREFILL_ITERS
                    row["prefill_seq"] = PREFILL_SEQ
                    for t in (pt_tt, ppt, pkv[0], pkv[1]):
                        t.deallocate(True)

                hidden_d = R.unit_rms_hidden_states((BATCH, 1, text_config.hidden_size), seed=63)
                x_dev = U.decode_input(hidden_d, mesh)
                pos_dev, rope_dev = U.position_tensors([CONTEXT] * BATCH, mesh)
                dec.decode_forward(
                    x_dev, kv_cache=kv, page_table=ptt, current_pos=pos_dev, rope_idxs=rope_dev
                ).deallocate(True)
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out_dev = dec.decode_forward(
                    x_dev, kv_cache=kv, page_table=ptt, current_pos=pos_dev, rope_idxs=rope_dev
                )
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(DECODE_ITERS):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                row["decode_ms"] = 1000.0 * (time.perf_counter() - start) / DECODE_ITERS
                ttnn.release_trace(mesh, tid)
                out_dev.deallocate(True)
                x_dev.deallocate(True)
                print(
                    f"PERF {kind_id} {label} decode_ms={row.get('decode_ms'):.4f} "
                    f"prefill_ms={row.get('prefill_ms')}",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
                print(f"PERF {kind_id} {label} RUN_FAIL {row['error'][:200]}", flush=True)
            rows.append(row)
            for t in (kv[0], kv[1], ptt):
                t.deallocate(True)
            del dec
finally:
    ttnn.close_mesh_device(mesh)
OUT.write_text(json.dumps(rows, indent=2))
print("DONE")
