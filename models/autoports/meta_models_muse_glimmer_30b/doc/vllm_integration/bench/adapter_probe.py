# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drive ``tt/generator_vllm.py`` exactly the way the TT vLLM plugin does, without vLLM.

The plugin's call sequence is short and completely determined by
``vllm_tt_plugin/model_runner.py`` and ``vllm_tt_plugin/async_decode.py``:

    initialize_vllm_model -> allocate_kv_cache -> warmup_model_prefill/decode
      -> prefill_forward(sampling_params=...)                       [one per request]
      -> decode_forward(read_from_device=False) -> read_decode_output(async_read=True)
         -> process_decode_output_host(is_tokens=...)               [one per token]

Reproducing it here is what makes the adapter debuggable on the reduced target and
what pins the parts a live server cannot show directly: which inputs are refreshed
from host on which step, what the counters say a steady-state token costs, and
whether the stale-input contract actually holds when the host tensors are wrong on
purpose.

Usage:
    python .../adapter_probe.py --layers 0,3 --out probe_reduced.json
    python .../adapter_probe.py --out probe_full.json      # all 52 layers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from loguru import logger  # noqa: E402


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="", help="reduced layer indices, e.g. '0,3'")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--kv-token-budget", type=int, default=None)
    parser.add_argument("--prompt-lens", type=str, default="128,37,4097")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--stale-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Corrupt the host token/position tensors after the first steady decode step. "
            "The adapter's contract says a device-sampled step with an unchanged batch layout "
            "must ignore them; --no-stale-inputs is the control that says whether a failure "
            "is caused by that rule."
        ),
    )
    parser.add_argument(
        "--read-mode",
        choices=("async", "sync"),
        default="async",
        help=(
            "async: decode_forward(read_from_device=False) -> read_decode_output(async_read=True) "
            "-> process_decode_output_host, i.e. the vLLM async split. sync: read_from_device=True, "
            "so the generator reads inside the call. The control for 'is the deferred read racing "
            "the next trace replay'."
        ),
    )
    parser.add_argument(
        "--drain-per-step",
        action="store_true",
        help="ttnn.synchronize_device after every decode step; the control for host-outruns-device.",
    )
    parser.add_argument(
        "--keep-stale-for-host-sampling",
        action="store_true",
        help=(
            "Hand the deliberately-corrupted host positions to the final host-sampling step too. "
            "That step samples on host, so the adapter restages from the caller by contract and the "
            "stale-input rule does not cover it; this switch exists only to reproduce the original "
            "hang and to prove the port now refuses an illegal position instead of hanging the mesh."
        ),
    )
    parser.add_argument(
        "--verbose-steps",
        action="store_true",
        help=(
            "Print a flushed marker around every decode step and read back the four persistent "
            "decode trace inputs before each one, so a hang names the step and shows what the "
            "device actually held going into it."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.layers:
        os.environ["MUSE_GLIMMER_VLLM_LAYER_INDICES"] = args.layers
    if args.kv_token_budget:
        os.environ["MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET"] = str(args.kv_token_budget)

    from transformers import AutoConfig

    import ttnn
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
        DEFAULT_TRACE_REGION_SIZE,
        close_generator_mesh,
        open_generator_mesh,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import (
        MuseGlimmerForConditionalGeneration as Adapter,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import DECODE_ROWS, HF_MODEL_ID, weights_snapshot_dir
    from models.common.sampling.generator import SamplingParams

    block_size = 64
    report: dict = {
        "layers": args.layers or "all",
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "block_size": block_size,
    }
    hf_config = AutoConfig.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)
    hf_config._name_or_path = HF_MODEL_ID

    mesh = open_generator_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        started = time.perf_counter()
        model = Adapter.initialize_vllm_model(
            hf_config,
            mesh,
            args.max_num_seqs,
            max_seq_len=args.max_model_len,
        )
        report["build_s"] = round(time.perf_counter() - started, 2)

        # --- what the worker computes, with the plugin's own arithmetic ---------
        tokens_all_users = Adapter.get_max_tokens_all_users(
            model_name=HF_MODEL_ID,
            num_devices=mesh.get_num_devices(),
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
        num_blocks = -(-(tokens_all_users + block_size * args.max_num_seqs) // block_size)
        report["kv"] = {"tokens_all_users": tokens_all_users, "num_blocks": num_blocks}

        num_layers = len(model.generator.model.layers)
        kv_cache = model.allocate_kv_cache((num_blocks, 1, block_size, 128), torch.bfloat16, num_layers)
        report["kv"]["allocated_layers"] = len(kv_cache)
        report["kv"]["cache_shape"] = list(kv_cache[0][0].shape)
        report["kv"]["cache_dtype"] = str(kv_cache[0][0].dtype)
        report["kv"]["model_max_num_blocks"] = model.generator.model.config.max_num_blocks
        report["kv"]["model_cache_slots"] = model.generator.model.config.max_batch_size
        report["kv"]["binds_the_allocated_buffers"] = [
            int(kv_cache[i][0].buffer_address()) == int(model.generator.model.layers[i].k_cache.buffer_address())
            for i in range(num_layers)
        ]

        # --- warmup, exactly as TTModelRunner.warmup_model sequences it ---------
        started = time.perf_counter()
        model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
        model.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=args.max_num_seqs,
            num_blocks=-(-args.max_model_len // block_size),
            can_sample_on_device=True,
            enable_trace=False,
        )
        model.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=args.max_num_seqs,
            num_blocks=-(-args.max_model_len // block_size),
            can_sample_on_device=True,
            enable_trace=True,
        )
        report["warmup_s"] = round(time.perf_counter() - started, 2)
        report["decode_trace_captured"] = model.generator._trace_id is not None
        report["sampling_trace_captured"] = bool(model.generator._sampling_captured)
        # The optimized-vLLM stage's prefill traces are captured at the end of the
        # ``enable_trace=True`` decode warmup, i.e. by the call above, so this is where
        # the bucket set lands.  ``requested`` vs ``captured`` is the whole capacity
        # story: a short ``captured`` list means the trace region ran out and the
        # remaining buckets serve eagerly.
        _vllm_caps = model.capability_report().get("vllm", {})
        report["prefill_trace"] = {
            "enabled": _vllm_caps.get("prefill_trace_enabled"),
            "buckets_requested": _vllm_caps.get("prefill_trace_buckets_requested"),
            "buckets_resident": list(model.generator.prefill_trace_buckets),
            "capture_failures": int(model.generator._prefill_capture_failures),
            "capture_disabled_after_failure": bool(model.generator._prefill_capture_disabled),
            "max_padded_len": int(model.generator.gen_config.prefill_trace_max_padded_len),
        }

        blocks_per_seq = -(-args.max_model_len // block_size)

        def greedy(rows: int):
            return SamplingParams(temperature=[0.0] * rows, top_k=[1] * rows, top_p=[1.0] * rows, seed=[None] * rows)

        def dump_device_inputs(tag: str) -> None:
            """What the four persistent decode trace inputs hold, read from device 0."""
            inputs = getattr(model.generator, "_device_inputs", None)
            if not inputs:
                print(f"[probe] {tag}: device inputs not allocated yet", flush=True)
                return
            cur = ttnn.to_torch(ttnn.get_device_tensors(inputs["current_pos"])[0]).reshape(-1)
            rope = ttnn.to_torch(ttnn.get_device_tensors(inputs["rope_pos_ids"])[0]).reshape(-1)
            tok = ttnn.to_torch(ttnn.get_device_tensors(inputs["tokens"])[0]).reshape(-1)
            table = ttnn.to_torch(ttnn.get_device_tensors(inputs["page_table"])[0]).reshape(DECODE_ROWS, -1)
            print(
                f"[probe] {tag}: current_pos={cur[:6].tolist()} rope={rope[:6].tolist()} "
                f"tok={tok[:6].tolist()} page_table[:4,:6]={table[:4, :6].tolist()}",
                flush=True,
            )

        def step_decode(**kwargs):
            """One decode step through whichever read path --read-mode selects."""
            if args.read_mode == "sync":
                got = model.decode_forward(read_from_device=True, **kwargs)
            else:
                out = model.decode_forward(read_from_device=False, **kwargs)
                host, events = model.read_decode_output(out, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                got = model.process_decode_output_host(host, is_tokens=True)
            if args.drain_per_step:
                ttnn.synchronize_device(mesh)
            return got

        def block_table(rows: int, blocks: int, base: int = 0) -> torch.Tensor:
            table = torch.zeros(rows, blocks_per_seq, dtype=torch.int32)
            for row in range(rows):
                start = base + row * blocks
                table[row, :blocks] = torch.arange(start, start + blocks, dtype=torch.int32)
            return table

        # --- one request per prompt length, batch 1 -----------------------------
        report["requests"] = []
        base_block = 0
        for prompt_len in [int(v) for v in args.prompt_lens.split(",") if v]:
            blocks = -(-prompt_len // block_size) + 2  # room for the generated tail
            table = block_table(1, blocks, base_block)
            base_block += blocks
            ids = torch.arange(1000, 1000 + prompt_len, dtype=torch.int32).reshape(1, prompt_len)
            model.generator.reset_counters()
            first = model.prefill_forward(
                tokens=ids,
                page_table=table,
                kv_cache=kv_cache,
                enable_trace=False,
                prompt_lens=[prompt_len],
                start_pos=torch.zeros(1, dtype=torch.int32),
                sampling_params=greedy(1),
            )
            prefill_counters = dict(model.generator.counters)
            emitted = [int(first[0])]

            decode_table = torch.zeros(args.max_num_seqs, blocks_per_seq, dtype=torch.int32)
            decode_table[:1] = table
            positions = torch.full((args.max_num_seqs,), -1, dtype=torch.int32)
            positions[0] = prompt_len
            decode_tokens = torch.zeros(args.max_num_seqs, 1, dtype=torch.int32)
            decode_tokens[0, 0] = emitted[0]

            model.generator.reset_counters()
            for step in range(args.decode_steps):
                if args.verbose_steps:
                    dump_device_inputs(f"single len={prompt_len} step={step} before")
                    print(f"[probe] single len={prompt_len} step={step} submit", flush=True)
                got = step_decode(
                    tokens=decode_tokens,
                    page_table=decode_table,
                    kv_cache=kv_cache,
                    start_pos=positions,
                    enable_trace=True,
                    sampling_params=greedy(args.max_num_seqs),
                    reset_batch=(step == 0),
                )
                if args.verbose_steps:
                    print(f"[probe] single len={prompt_len} step={step} done", flush=True)
                emitted.append(int(got[0]))
                if args.stale_inputs:
                    # Deliberately WRONG host inputs from here on: the contract says a
                    # steady device-sampled step must ignore them.
                    decode_tokens[0, 0] = 0
                    positions[0] = -7 if step else prompt_len + 1
                else:
                    decode_tokens[0, 0] = emitted[-1]
                    positions[0] = prompt_len + step + 1
            report["requests"].append(
                {
                    "prompt_len": prompt_len,
                    "aligned": {
                        "tile32": prompt_len % 32 == 0,
                        "page64": prompt_len % block_size == 0,
                        "chunk8192": prompt_len % 8192 == 0,
                    },
                    "prefill_counters": prefill_counters,
                    "decode_counters": dict(model.generator.counters),
                    "decode_serving_counters": dict(model.generator.serving_counters),
                    "tokens": emitted,
                    "distinct_tokens": len(set(emitted)),
                    "max_token_id": max(emitted),
                    "in_vocab": max(emitted) < model.generator.model.config.vocab_size,
                }
            )
            logger.info(f"prompt_len={prompt_len} -> {emitted}")

        # --- multi-request smoke: three concurrent slots ------------------------
        rows = min(3, args.max_num_seqs)
        prompt_lens = [96, 130, 61][:rows]
        blocks = max(-(-p // block_size) for p in prompt_lens) + 2
        table = block_table(rows, blocks, base_block)
        max_len = max(prompt_lens)
        ids = torch.zeros(rows, max_len, dtype=torch.int32)
        for row, length in enumerate(prompt_lens):
            ids[row, :length] = torch.arange(2000 + row * 50, 2000 + row * 50 + length, dtype=torch.int32)
        model.generator.reset_counters()
        if args.verbose_steps:
            print(f"[probe] multi prefill submit rows={rows} lens={prompt_lens}", flush=True)
        first = model.prefill_forward(
            tokens=ids,
            page_table=table,
            kv_cache=kv_cache,
            enable_trace=False,
            prompt_lens=prompt_lens,
            start_pos=torch.zeros(rows, dtype=torch.int32),
            sampling_params=greedy(rows),
        )
        if args.verbose_steps:
            print(f"[probe] multi prefill done -> {first[:rows].tolist()}", flush=True)
        decode_table = torch.zeros(args.max_num_seqs, blocks_per_seq, dtype=torch.int32)
        decode_table[:rows] = table
        positions = torch.full((args.max_num_seqs,), -1, dtype=torch.int32)
        positions[:rows] = torch.tensor(prompt_lens, dtype=torch.int32)
        decode_tokens = torch.zeros(args.max_num_seqs, 1, dtype=torch.int32)
        decode_tokens[:rows, 0] = first[:rows].to(torch.int32)
        per_row = [[int(first[row])] for row in range(rows)]
        model.generator.reset_counters()
        for step in range(args.decode_steps):
            if args.verbose_steps:
                dump_device_inputs(f"multi step={step} before")
                print(f"[probe] multi step={step} submit", flush=True)
            got = step_decode(
                tokens=decode_tokens,
                page_table=decode_table,
                kv_cache=kv_cache,
                start_pos=positions,
                enable_trace=True,
                sampling_params=greedy(args.max_num_seqs),
                reset_batch=(step == 0),
            )
            if args.verbose_steps:
                print(f"[probe] multi step={step} done", flush=True)
            for row in range(rows):
                per_row[row].append(int(got[row]))
            if args.stale_inputs:
                decode_tokens[:rows, 0] = 0
                positions[:rows] = -7
            else:
                for row in range(rows):
                    decode_tokens[row, 0] = per_row[row][-1]
                    positions[row] = prompt_lens[row] + step + 1
        report["stale_inputs"] = bool(args.stale_inputs)
        report["read_mode"] = args.read_mode
        report["drain_per_step"] = bool(args.drain_per_step)
        report["keep_stale_for_host_sampling"] = bool(args.keep_stale_for_host_sampling)
        report["multi_request"] = {
            "prompt_lens": prompt_lens,
            "rows": rows,
            "tokens": per_row,
            "rows_are_distinct": len({tuple(r) for r in per_row}) == rows,
            "decode_counters": dict(model.generator.counters),
            "decode_serving_counters": dict(model.generator.serving_counters),
        }

        # --- page-table refresh: changed and unchanged, read back off the device --
        #
        # The stale-input rule covers tokens and positions: a steady device-sampled
        # step must ignore the host copies of those.  The page table is the explicit
        # exception -- it changes when a sequence crosses a block boundary, which has
        # nothing to do with the sampled token, so it is compared and staged on every
        # step.  Both halves of that need proving through the adapter, not just at the
        # generator: an unchanged table must cost no copy, and a changed one must
        # actually reach the device.
        #
        # The change is a block id in the *tail* of row 0 -- past what this sequence's
        # positions reach -- so it is observable in the staged tensor while provably
        # not changing what any op reads.  ``ok`` below asserts the refresh *counts* and
        # the staged device tensor, which is what this section is for; the emitted tokens
        # of both windows are recorded alongside them for a reader to compare, and are
        # deliberately not asserted equal -- the two windows decode from different
        # positions, so equality is not the expected relation.
        def device_page_table():
            handle = model.generator._device_inputs["page_table"]
            return ttnn.to_torch(ttnn.get_device_tensors(handle)[0]).reshape(DECODE_ROWS, -1).clone()

        pt_report = {"rows": rows}
        positions[:rows] = torch.tensor(
            [prompt_lens[row] + args.decode_steps for row in range(rows)], dtype=torch.int32
        )
        for row in range(rows):
            decode_tokens[row, 0] = per_row[row][-1]
        # One restaging step to put the caller's state back after the stale-input loop.
        step_decode(
            tokens=decode_tokens,
            page_table=decode_table,
            kv_cache=kv_cache,
            start_pos=positions,
            enable_trace=True,
            sampling_params=greedy(args.max_num_seqs),
            reset_batch=True,
        )
        model.generator.reset_counters()
        before_table = device_page_table()
        unchanged_tokens = []
        for _ in range(3):
            got = step_decode(
                tokens=decode_tokens,
                page_table=decode_table,
                kv_cache=kv_cache,
                start_pos=positions,
                enable_trace=True,
                sampling_params=greedy(args.max_num_seqs),
                reset_batch=False,
            )
            unchanged_tokens.append([int(got[row]) for row in range(rows)])
        pt_report["unchanged"] = {
            "steps": 3,
            "page_table_refreshes": model.generator.counters["page_table_refreshes"],
            "trace_replays": model.generator.counters["trace_replays"],
            "device_table_unchanged": bool(torch.equal(before_table, device_page_table())),
            "tokens": unchanged_tokens,
        }

        # Now change a tail block id for row 0 and step again.
        changed_table = decode_table.clone()
        tail = changed_table.shape[1] - 1
        spare = int(changed_table.max()) + 1
        changed_table[0, tail] = spare
        model.generator.reset_counters()
        changed_tokens = []
        for step in range(3):
            got = step_decode(
                tokens=decode_tokens,
                page_table=changed_table,
                kv_cache=kv_cache,
                start_pos=positions,
                enable_trace=True,
                sampling_params=greedy(args.max_num_seqs),
                reset_batch=False,
            )
            changed_tokens.append([int(got[row]) for row in range(rows)])
        staged = device_page_table()
        pt_report["changed"] = {
            "steps": 3,
            "changed_cell": [0, tail, spare],
            # One copy for the step whose table differed, none for the two after it.
            "page_table_refreshes": model.generator.counters["page_table_refreshes"],
            "trace_replays": model.generator.counters["trace_replays"],
            "device_table_matches_new_host_table": bool(
                torch.equal(staged[:rows], changed_table[:rows].to(staged.dtype))
            ),
            "device_table_differs_from_old": bool(not torch.equal(staged, before_table)),
            "tokens": changed_tokens,
        }
        pt_report["ok"] = bool(
            pt_report["unchanged"]["page_table_refreshes"] == 0
            and pt_report["unchanged"]["device_table_unchanged"]
            and pt_report["changed"]["page_table_refreshes"] == 1
            and pt_report["changed"]["device_table_matches_new_host_table"]
        )
        report["page_table_refresh"] = pt_report

        # --- host-sampling compatibility mode (what vLLM falls back to) ---------
        #
        # This step is *not* covered by the stale-input rule: it samples on host, so
        # the adapter restages tokens and positions from the caller.  Feeding it the
        # deliberately-corrupted values would be feeding an illegal position to a path
        # whose contract is that the host values are authoritative.
        print("[probe] multi decode section done; host-sampling step submit", flush=True)
        if not args.keep_stale_for_host_sampling:
            positions[:rows] = torch.tensor(
                [prompt_lens[row] + args.decode_steps for row in range(rows)], dtype=torch.int32
            )
            for row in range(rows):
                decode_tokens[row, 0] = per_row[row][-1]
        model.generator.reset_counters()
        logits = model.decode_forward(
            tokens=decode_tokens,
            page_table=decode_table,
            kv_cache=kv_cache,
            start_pos=positions,
            enable_trace=True,
            read_from_device=False,
            sampling_params=None,
            reset_batch=True,
        )
        host_logits = model.process_decode_output_host(logits, is_tokens=False)
        print("[probe] host-sampling step done", flush=True)
        report["host_sampling"] = {
            "logits_shape": list(host_logits.shape),
            "finite": bool(torch.isfinite(host_logits).all()),
            "counters": dict(model.generator.counters),
        }

        report["capability_report"] = model.capability_report()
        report["status"] = "ok"
    finally:
        try:
            model.generator.teardown()
        except Exception:  # noqa: BLE001
            pass
        close_generator_mesh(mesh)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "capability_report"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
