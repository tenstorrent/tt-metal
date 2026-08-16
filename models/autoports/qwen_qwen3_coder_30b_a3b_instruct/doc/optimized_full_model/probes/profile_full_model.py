# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One clean full-model pass for the op-level profiler, at the reduced tier.

`doc/optimized_multichip_decoder/` has a `tt-perf-report` for its layer;
`doc/full_model/` had none, so the ops stage 05 actually adds -- embedding, the
final norm, the column-parallel LM head and the sampler -- had no op-level
evidence at all. This closes that.

It is the **reduced** variant on purpose: one layer of each kind (there is only
one kind here) rather than 48, exactly as the reduced test tier is defined. The
48 layers are 48 copies of a body stage 04 already profiled row by row; what has
never been profiled is the *terminal* path, and a 2-layer capture puts those
rows in a CSV small enough to read without the layer body burying them.

    python -m tracy -v -r -p --sync-host-device -o /tmp/prof_fm_dec \\
        profile_full_model.py decode
    python -m tracy -v -r -p --sync-host-device -o /tmp/prof_fm_pf \\
        profile_full_model.py prefill
    tt-perf-report /tmp/prof_fm_dec/reports/*/ops_perf_results_*.csv

Modes:

* ``decode``  -- embedding -> 2 layers -> final norm -> LM head -> sampler,
                 the token-out path, warmed then captured;
* ``prefill`` -- the same stack over a 512-token prompt, ending at the LM head.

Never combine with the watcher, per this project's standing rule.

The warm-up pass before the measured one keeps kernel compilation out of the
profile, and the published window is always the **last** iteration.
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
LAYERS = 2
CONTEXT = 4096
PREFILL_LEN = 512
ITERATIONS = 2


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "decode"
    assert mode in ("decode", "prefill"), mode

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MC.MESH_SHAPE), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        t0 = time.perf_counter()
        gen = build_generator(
            str(MODEL_DIR), mesh, override_num_layers=LAYERS, max_context_len=CONTEXT, max_batch_size=1
        )
        print(f"weight load {time.perf_counter() - t0:.1f} s ({LAYERS} layers)", flush=True)

        kv_cache = gen._ensure_kv_cache()

        if mode == "prefill":
            tokens = torch.arange(1000, 1000 + PREFILL_LEN, dtype=torch.long).unsqueeze(0)
            page_table = gen.make_page_table([PREFILL_LEN + 1])
            for iteration in range(ITERATIONS):
                gen.reset()
                gen.prefill_forward(
                    tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=[PREFILL_LEN],
                    sampling_mode="device",
                )
                ttnn.synchronize_device(mesh)
                print(f"prefill iteration {iteration} done", flush=True)
        else:
            prompt = list(range(1000, 1128))
            page_table = gen.make_page_table([len(prompt) + ITERATIONS + 2])
            gen.reset()
            gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[len(prompt)],
                sampling_mode="device",
            )
            # Installing the trace warms every program the capture will contain
            # and performs the first replay.
            gen.decode_forward(
                None,
                torch.tensor([len(prompt)]),
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_mode="device",
                enable_trace=True,
                active_batch=1,
                decode_horizon=len(prompt) + ITERATIONS + 2,
            )
            ttnn.synchronize_device(mesh)
            for iteration in range(ITERATIONS):
                gen.decode_forward(
                    None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True
                )
                ttnn.synchronize_device(mesh)
                print(f"decode iteration {iteration} done", flush=True)

        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
