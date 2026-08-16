# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The **full 48-layer prefill** pass, under the op-level profiler.

Stage 05 shipped with prefill unprofiled and disclosed it as a gap: the only
op-level evidence in ``doc/full_model/`` is a 2-layer *decode* window, and
``doc/optimized_multichip_decoder/`` profiles a single prefill *layer*, not the
model. So the TTFT this project publishes has never had an op-level breakdown
behind it. This closes that.

Two differences from ``profile_full_model_48.py``, both forced by prefill being
eager rather than traced:

* **the boundary cannot be found by an embedding triple.** Decode's window
  starts at the three ``EmbeddingsDeviceOperation`` gathers ``decode_hidden``
  opens with; prefill's rotary comes from a slice of the precomputed tables, not
  from a per-token gather, so there is no such marker. Instead this script runs
  the *same* prefill twice back to back with no ``reset()`` in between, which
  makes the capture two identical program sequences per device, and
  ``window_full_model_48_prefill.py`` splits it in half and **asserts the two
  halves are the same sequence of ops, row for row** before publishing the
  second one. That is a stronger boundary check than decode's tallies, not a
  weaker one: it compares every row, in order, not ten counts;
* ``--sync-host-device`` is dropped for the same reason it is dropped in the
  decode capture -- absolute microseconds are the question and the sync inflates
  every collective.

``PROMPT_LEN`` is **128** so the window is the one behind the published warmed
TTFT figure (``probes/perf_full_model_p128_argmaxrows.json``, ``ttft_ms``
125.43), and not some other length.

    python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_pf \\
        profile_full_model_48_prefill.py
    python window_full_model_48_prefill.py \\
        /tmp/prof_fm48_pf/reports/*/ops_perf_results_*.csv \\
        --out /tmp/fm48_prefill_window.csv --layers 48
    tt-perf-report /tmp/fm48_prefill_window.csv

Never combine with the watcher, per this project's standing rule.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[3]
LAYERS = 48
CONTEXT = 4096
#: The prompt length the published TTFT is measured at.
PROMPT_LEN = 128
#: Measured prefills after the warm-up. The windower halves the capture, so
#: this must stay at 2.
ITERATIONS = 2


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, override_num_layers=LAYERS, max_context_len=CONTEXT, max_batch_size=1
        )
        print(f"weight load {time.perf_counter() - t0:.1f} s ({LAYERS} layers)", flush=True)

        kv_cache = gen._ensure_kv_cache()
        tokens = torch.arange(1000, 1000 + PROMPT_LEN, dtype=torch.long).unsqueeze(0)
        page_table = gen.make_page_table([PROMPT_LEN + 2])

        # One reset, before everything. A reset *between* the measured
        # iterations would put FillDeviceOperation rows between them and break
        # the halving; re-prefilling the same prompt into the same pages writes
        # the same K/V twice, which is exactly what makes the two iterations
        # identical.
        gen.reset()

        # Warm-up: compiles every program the measured iterations then re-run.
        gen.prefill_forward(
            tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=[PROMPT_LEN], sampling_mode="device"
        )
        ttnn.synchronize_device(mesh)
        print("warm-up prefill done", flush=True)

        for iteration in range(ITERATIONS):
            t = time.perf_counter()
            gen.prefill_forward(
                tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=[PROMPT_LEN], sampling_mode="device"
            )
            ttnn.synchronize_device(mesh)
            print(f"prefill iteration {iteration} done, {1e3 * (time.perf_counter() - t):.3f} ms wall", flush=True)

        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
