# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model performance: TTFT and the two decode metrics, plus the sampler A/B.

Workload shape is the vLLM primary single-user profile -- **prompt 128,
generate 128, batch 1** -- and every row of the CSV records it, because the
three decode numbers below are not comparable without it:

``model_trace``
    the model decode trace alone: token in -> final norm -> LM head ->
    sampler-ready local logits, **no sampling, no token feedback, no readback**.
    This is the logits-only figure that is comparable to a PERF.md-style
    decoder-stack number and to the layer-stack lower bound.
``token_out``
    model trace + split-sampling trace, the sampled token fed back on device
    through ``tt_out_tok``, **no host readback** -- the serving steady state.
``token_out_readback``
    the same plus the per-token host readback a standalone generator needs.

Also measures the two common sampler paths at the same logits, so the choice is
priced rather than asserted:

* ``Sampling1D`` split sampling -- local top-32 per die, all-gather 32 values
  and 32 indices, ``ttnn.sampling`` with k=1. **What ships.**
* ``Sampling1D`` force-argmax -- all-gather the whole 37984-wide local logit
  shard from all four dies, untilize 151936 columns, ``ttnn.argmax``.

Writes ``perf_full_model.csv`` and ``perf_full_model.json`` next to this file.
"""

from __future__ import annotations

import argparse
import csv
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


def _median_ms(fn, reps: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ttft-reps", type=int, default=5)
    # Stage 06: the same workload at several prompt lengths is the point of the
    # paged-SDPA lever, so each run needs its own artifact instead of
    # overwriting the last. Default keeps the stage-05 filenames exactly.
    parser.add_argument("--tag", default="", help="suffix for perf_full_model{tag}.csv/.json")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    rows = []
    summary = {
        "workload": f"prompt {args.prompt_len} / generate {args.gen_len} / batch 1",
        "layers": args.layers,
        "context": args.context,
        "mesh": "1x4 P300_X2, FABRIC_1D_RING",
    }
    try:
        gen = build_generator(
            MODEL_DIR, mesh, override_num_layers=args.layers, max_context_len=args.context, max_batch_size=1
        )
        prompt = list(range(1000, 1000 + args.prompt_len))
        kv_cache = gen._ensure_kv_cache()
        page_table = gen.make_page_table([args.prompt_len + args.gen_len])
        gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=1)

        # --- TTFT: warmed, prompt-in to first-token-out, device sampling ------
        ttft_samples = []
        for _ in range(args.ttft_reps):
            gen.reset()
            t0 = time.perf_counter()
            sampled = gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[args.prompt_len],
                sampling_mode="device",
            )
            int(gen._sampled_to_torch(sampled)[0].item())
            ttft_samples.append((time.perf_counter() - t0) * 1e3)
        # The first pass compiles; report the warmed median and keep the cold one.
        summary["ttft_cold_ms"] = ttft_samples[0]
        summary["ttft_ms"] = statistics.median(ttft_samples[1:]) if len(ttft_samples) > 1 else ttft_samples[0]
        summary["ttft_samples_ms"] = ttft_samples
        # ``iterations`` is the number of timed repetitions behind the ``ms``
        # column, for every row. It used to hold ``args.prompt_len`` here, which
        # is a prompt length wearing an iteration count's name.
        rows.append(
            (
                "ttft",
                f"prefill+lm_head+sample+readback, prompt {args.prompt_len}",
                summary["ttft_ms"],
                max(1, len(ttft_samples) - 1),
            )
        )

        # --- install the decode traces ---------------------------------------
        gen.decode_forward(
            None,
            torch.tensor([args.prompt_len]),
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
        )
        ttnn.synchronize_device(mesh)

        def token_out_readback():
            gen.decode_forward(
                None, None, page_table=None, kv_cache=kv_cache, sampling_mode="device", enable_trace=True
            )
            gen._sampled_to_torch(gen._trace_sampled)

        def token_out():
            ttnn.execute_trace(mesh, gen._trace_model_id, cq_id=0, blocking=False)
            ttnn.execute_trace(mesh, gen._trace_sampling_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        def model_trace_only():
            ttnn.execute_trace(mesh, gen._trace_model_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        for name, fn in (
            ("model_trace", model_trace_only),
            ("token_out", token_out),
            ("token_out_readback", token_out_readback),
        ):
            fn()  # warm
            median, samples = _median_ms(fn, args.gen_len)
            summary[f"{name}_ms"] = median
            summary[f"{name}_tps_user"] = 1e3 / median
            summary[f"{name}_min_ms"] = min(samples)
            summary[f"{name}_max_ms"] = max(samples)
            rows.append((name, "traced decode, batch 1", median, args.gen_len))

        # --- sampler A/B at the shipped logits --------------------------------
        gen.reset()
        sampled = gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[args.prompt_len],
            sampling_mode="device",
        )
        gen.decode_forward(
            None,
            torch.tensor([args.prompt_len]),
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
        )
        logits = gen._trace_logits
        k, p, temp = gen._ensure_sampling_params()

        def split():
            return gen.model.sample_split(logits, k=k, p=p, temp=temp)

        def force_argmax():
            return gen.model.sample_greedy_argmax(logits)

        for name, fn in (("sampler_split", split), ("sampler_force_argmax", force_argmax)):
            try:
                out = fn()
                ttnn.synchronize_device(mesh)
                token = int(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[0].item())
                reps = 30
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(reps):
                    fn()
                ttnn.synchronize_device(mesh)
                elapsed = (time.perf_counter() - t0) / reps * 1e3
                summary[f"{name}_ms"] = elapsed
                summary[f"{name}_token"] = token
                rows.append((name, "eager, same logits", elapsed, 1))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                summary[f"{name}_error"] = repr(exc)

        summary["trace_stats"] = dict(gen.trace_stats)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    with (HERE / f"perf_full_model{args.tag}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        # ``iterations`` = timed repetitions behind ``ms``; the workload the
        # figure describes lives in ``what``.
        writer.writerow(["metric", "what", "ms", "iterations"])
        writer.writerows(rows)
    (HERE / f"perf_full_model{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
