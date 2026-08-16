# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The **full 48-layer** decode step, under the op-level profiler.

Stage 05's `profile_full_model.py` is the *reduced* variant: 2 layers, which is
the right window for looking at the terminal path but over-weights it by
construction (the terminal path is paid once whether there are 2 layers or 48).
Stage 06's question is the opposite one -- where does the time go when 48 layers
run back-to-back -- so this captures all 48.

Two deliberate differences from the stage-05 script:

* ``LAYERS = 48`` and the profiler's program-support count has to be raised
  well above its 1000 default (``--op-support-count``); one 48-layer decode
  iteration alone is ~3.6k programs per device;
* **no ``--sync-host-device``**. Stage 05's report carries the warning "read the
  ranking, not the absolute microseconds: ``--sync-host-device`` inflates every
  collective". At 48 layers there are 96 collectives in the window and the
  absolute microseconds are the whole question, so the sync is dropped. The
  cross-check that this was the right call is that the windowed device time then
  has to land on the independently measured 20.22 ms model-trace figure.

    python -m tracy -v -r -p --op-support-count 32000 -o /tmp/prof_fm48_dec \\
        profile_full_model_48.py
    python window_full_model_48.py /tmp/prof_fm48_dec/reports/*/ops_perf_results_*.csv \\
        --out /tmp/fm48_decode_window.csv --layers 48
    tt-perf-report /tmp/fm48_decode_window.csv

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
#: Kept short on purpose: prefill is ~3.4k more programs per device and this
#: capture is about decode. 128 matches the perf probe's prompt length.
PROMPT_LEN = 128
#: Warm-up installs + replays the trace once; this many measured replays follow,
#: so the capture holds ITERATIONS+1 complete decode iterations and the windower
#: can take the last one with a whole iteration in front of it.
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

        prompt = list(range(1000, 1000 + PROMPT_LEN))
        horizon = len(prompt) + ITERATIONS + 2
        page_table = gen.make_page_table([horizon])
        gen.reset()
        gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt)],
            sampling_mode="device",
        )
        ttnn.synchronize_device(mesh)
        print("prefill done", flush=True)

        # Installing the trace warms every program the capture will contain and
        # performs the first replay.
        gen.decode_forward(
            None,
            torch.tensor([len(prompt)]),
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
            decode_horizon=horizon,
        )
        ttnn.synchronize_device(mesh)
        print("trace installed + first replay done", flush=True)

        for iteration in range(ITERATIONS):
            t = time.perf_counter()
            gen.decode_forward(
                None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True
            )
            ttnn.synchronize_device(mesh)
            print(f"decode iteration {iteration} done, {1e3 * (time.perf_counter() - t):.3f} ms wall", flush=True)

        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
