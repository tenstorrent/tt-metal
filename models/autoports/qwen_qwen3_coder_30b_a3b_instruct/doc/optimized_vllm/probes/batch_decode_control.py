# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The standalone batch-32 decode control stage 08 never took.

Stage 08 observed that a ``max_num_seqs=32`` server decodes at ~263 ms/token
whether **one** user or **thirty-two** are active, and attributed that to MoE
decode batch scaling in ``tt/model.py`` rather than to the serving adapter. That
attribution rested on inference: no standalone measurement of a 32-slot traced
decode existed in any stage.

This is that measurement, outside vLLM entirely. It builds the generator at
``--slots`` configured decode rows -- exactly what ``max_num_seqs`` does through
``initialize_vllm_model`` -- prefills ``k`` of them, installs the decode trace
with the remaining ``32-k`` rows carrying the inactive sentinel ``current_pos =
-1`` (the same convention vLLM's padded decode batch uses), and times the traced
replay for each ``k`` in the sweep.

The shape of the curve is the finding:

* **flat in k** means the configured slot count, not the active one, sets the
  per-step cost -- i.e. the inactive rows are doing full expert work;
* **rising in k** means only the active rows cost anything.

No profiler is involved: this is host wall-clock around ``ttnn.execute_trace``
plus one ``synchronize_device``, which is the same harness
``doc/optimized_full_model/probes/perf_full_model.py`` uses for ``token_out``.

Writes ``batch_decode_control{tag}.json`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]


def _median_ms(fn, reps: int) -> dict:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "reps": reps,
    }


def _serving_shaped_page_table(gen, slots: int, active: int, blocks_per_row: int) -> torch.Tensor:
    """A vLLM-shaped block table: real disjoint blocks for the active rows, 0 elsewhere.

    vLLM zero-fills the unused entries of its own block tables rather than using
    the standalone ``-1``, because the paged decode SDPA kernel dereferences
    every page in its rounded read window before causal masking. This mirrors
    that exactly so the control measures the serving shape, not a nicer one.
    """
    table = torch.zeros((slots, gen.pages_per_user), dtype=torch.int32)
    for row in range(active):
        start = 1 + row * blocks_per_row  # block 0 stays vLLM's null block
        table[row, :blocks_per_row] = torch.arange(start, start + blocks_per_row, dtype=torch.int32)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=32, help="configured decode rows == max_num_seqs")
    parser.add_argument("--active", default="1,2,4,8,16,32", help="comma-separated active-row counts to sweep")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    actives = [int(v) for v in args.active.split(",") if v]
    if any(not 1 <= a <= args.slots for a in actives):
        raise SystemExit(f"--active values must be in [1,{args.slots}]")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary = {
        "what": (
            "standalone traced decode at a fixed configured slot count, sweeping how many "
            "of those slots carry a live request. No vLLM, no profiler."
        ),
        "slots": args.slots,
        "layers": args.layers,
        "prompt_len": args.prompt_len,
        "context": args.context,
        "mesh": "1x4 P300_X2, FABRIC_1D_RING",
        "sampling": "device, greedy (top_k=1), split-sampling trace, tt_out_tok feedback",
        "rows": [],
    }
    try:
        gen = build_generator(
            MODEL_DIR,
            mesh,
            override_num_layers=args.layers,
            max_context_len=args.context,
            max_batch_size=args.slots,
        )
        summary["model_max_batch_size"] = gen.model.max_batch_size
        summary["pages_per_user"] = gen.pages_per_user
        kv_cache = gen._ensure_kv_cache()
        horizon = args.prompt_len + args.gen_len
        blocks_per_row = gen._sdpa_rounded_page_count(horizon)
        prompt = list(range(1000, 1000 + args.prompt_len))

        for active in actives:
            gen.reset()
            page_table = _serving_shaped_page_table(gen, args.slots, active, blocks_per_row)
            gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=active)
            gen.prefill_forward(
                torch.tensor([prompt] * active),
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[args.prompt_len] * active,
                sampling_mode="device",
            )
            positions = torch.full((args.slots,), -1, dtype=torch.int64)
            positions[:active] = args.prompt_len
            gen.decode_forward(
                None,
                positions,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_mode="device",
                enable_trace=True,
                active_batch=args.slots,
                decode_horizon=horizon,
                validate_page_coverage=False,
            )
            ttnn.synchronize_device(mesh)

            def model_trace_only():
                ttnn.execute_trace(mesh, gen._trace_model_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

            def token_out():
                ttnn.execute_trace(mesh, gen._trace_model_id, cq_id=0, blocking=False)
                ttnn.execute_trace(mesh, gen._trace_sampling_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

            model_trace_only()
            token_out()
            row = {"active": active}
            for name, fn in (("model_trace", model_trace_only), ("token_out", token_out)):
                timing = _median_ms(fn, args.reps)
                row[name] = timing
                row[f"{name}_tps_user"] = 1e3 / timing["ms"]
            row["aggregate_tps"] = active * row["token_out_tps_user"]
            summary["rows"].append(row)
            print(
                f"slots={args.slots} active={active:>2}  "
                f"model_trace {row['model_trace']['ms']:.3f} ms  "
                f"token_out {row['token_out']['ms']:.3f} ms  "
                f"{row['token_out_tps_user']:.3f} t/s/u  "
                f"{row['aggregate_tps']:.2f} tok/s aggregate",
                flush=True,
            )

        summary["trace_stats"] = dict(gen.trace_stats)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out = HERE / f"batch_decode_control{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
