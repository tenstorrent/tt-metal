# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the largest serving KV pool this model can actually carry.

`KV_CACHE_TOKEN_BUDGET` in ``tt/generator_vllm.py`` was a constant derived from a
byte budget on paper. The stage review's objection is fair: "largest feasible
value" is a claim, and a claim about a device needs a measurement. Allocation
alone is not the bar either -- a pool that allocates but leaves no room for the
prefill working set would fail on the first long request, which is exactly the
kind of failure that shows up in production and not in a smoke test.

So each ladder rung is only counted as feasible if, at that pool size, the model
can still:

  * allocate all 104 cache tensors and adopt them,
  * capture the decode and sampling traces (they come out of the trace region,
    but their persistent outputs do not),
  * run a **full prefill chunk** (8192 tokens) -- the largest single activation
    working set the serving path ever builds, and
  * replay several traced decode steps on top of it.

The ladder descends, so the first success is the largest feasible rung and it is
measured on the least-fragmented allocator. Free DRAM is read from the allocator
via ``ttnn.get_memory_view`` rather than computed, before and after.

Usage::

    python doc/vllm_integration/bench/kv_budget_probe.py --out doc/vllm_integration/kv_budget_probe.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--blocks",
        default="28672,24576,20480,16416",
        help="descending ladder of pool sizes in 64-token blocks; first full success wins",
    )
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    import torch
    from loguru import logger
    from transformers import AutoConfig

    import ttnn
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
        DEFAULT_TRACE_REGION_SIZE,
        close_generator_mesh,
        open_generator_mesh,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import (
        BYTES_PER_BLOCK_PER_DEVICE,
        KV_CACHE_TOKEN_BUDGET,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import (
        MuseGlimmerForConditionalGeneration as Adapter,
    )
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import (
        HF_MODEL_ID,
        dram_capacity_bytes,
        weights_snapshot_dir,
    )
    from models.common.sampling.generator import SamplingParams

    block_size = 64
    blocks_per_seq = args.max_model_len // block_size
    ladder = [int(b) for b in args.blocks.split(",") if b]

    hf_config = AutoConfig.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)
    hf_config._name_or_path = HF_MODEL_ID

    def free_dram(mesh) -> int:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        try:
            return int(view.total_bytes_per_bank - view.total_bytes_allocated_per_bank) * int(view.num_banks)
        except Exception:  # noqa: BLE001 -- older/newer view APIs
            return -1

    report: dict = {
        "shipped_budget_tokens": KV_CACHE_TOKEN_BUDGET,
        # Not tokens/64: the TT worker adds one block per request slot as scheduling
        # headroom before converting, so blocks > tokens/block_size by max_num_seqs.
        "shipped_budget_blocks": -(-(KV_CACHE_TOKEN_BUDGET + block_size * args.max_num_seqs) // block_size),
        "shipped_budget_blocks_formula": "ceil((budget_tokens + block_size * max_num_seqs) / block_size)",
        "bytes_per_block_per_device": BYTES_PER_BLOCK_PER_DEVICE,
        "max_model_len": args.max_model_len,
        "blocks_per_seq": blocks_per_seq,
        "ladder": ladder,
        "attempts": [],
    }

    mesh = open_generator_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    model = None
    try:
        started = time.perf_counter()
        model = Adapter.initialize_vllm_model(hf_config, mesh, args.max_num_seqs, max_seq_len=args.max_model_len)
        report["build_s"] = round(time.perf_counter() - started, 1)
        report["dram_capacity_bytes_per_device"] = dram_capacity_bytes(mesh)
        report["free_dram_after_weights_and_build_pool"] = free_dram(mesh)

        num_layers = len(model.generator.model.layers)
        for blocks in ladder:
            attempt: dict = {
                "blocks": blocks,
                "tokens": blocks * block_size,
                "bytes_per_device": blocks * BYTES_PER_BLOCK_PER_DEVICE,
            }
            kv_cache = None
            try:
                kv_cache = model.allocate_kv_cache((blocks, 1, block_size, 128), torch.bfloat16, num_layers)
                attempt["allocated"] = True
                attempt["free_dram_after_alloc"] = free_dram(mesh)

                model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
                model.warmup_model_decode(
                    kv_cache=kv_cache,
                    max_batch_size=args.max_num_seqs,
                    num_blocks=blocks_per_seq,
                    can_sample_on_device=True,
                    enable_trace=True,
                )
                attempt["traces_captured"] = True
                attempt["free_dram_after_warmup"] = free_dram(mesh)

                # The largest activation working set the serving path ever builds.
                chunk = model.generator.model.config.prefill_chunk_size
                need = -(-chunk // block_size) + 2
                table = torch.zeros(1, blocks_per_seq, dtype=torch.int32)
                table[0, :need] = torch.arange(blocks - need, blocks, dtype=torch.int32)
                ids = torch.arange(1000, 1000 + chunk, dtype=torch.int32).reshape(1, chunk)
                greedy = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], seed=[None])
                first = model.prefill_forward(
                    tokens=ids,
                    page_table=table,
                    kv_cache=kv_cache,
                    enable_trace=False,
                    prompt_lens=[chunk],
                    start_pos=torch.zeros(1, dtype=torch.int32),
                    sampling_params=greedy,
                )
                attempt["prefill_tokens"] = chunk
                attempt["free_dram_at_prefill_peak"] = free_dram(mesh)

                dt = torch.zeros(args.max_num_seqs, blocks_per_seq, dtype=torch.int32)
                dt[0] = table[0]
                pos = torch.full((args.max_num_seqs,), -1, dtype=torch.int32)
                pos[0] = chunk
                toks = torch.zeros(args.max_num_seqs, 1, dtype=torch.int32)
                toks[0, 0] = int(first[0])
                emitted = []
                for step in range(args.decode_steps):
                    out = model.decode_forward(
                        tokens=toks,
                        page_table=dt,
                        kv_cache=kv_cache,
                        start_pos=pos,
                        enable_trace=True,
                        read_from_device=False,
                        sampling_params=SamplingParams(
                            temperature=[0.0] * args.max_num_seqs,
                            top_k=[1] * args.max_num_seqs,
                            top_p=[1.0] * args.max_num_seqs,
                            seed=[None] * args.max_num_seqs,
                        ),
                        reset_batch=(step == 0),
                    )
                    host, events = model.read_decode_output(out, async_read=True)
                    for e in events:
                        ttnn.event_synchronize(e)
                    emitted.append(int(model.process_decode_output_host(host, is_tokens=True)[0]))
                attempt["decode_tokens"] = emitted
                attempt["feasible"] = True
                logger.info(f"KV_BUDGET rung {blocks} blocks: FEASIBLE")
            except Exception as exc:  # noqa: BLE001
                attempt["feasible"] = False
                attempt["error"] = f"{type(exc).__name__}: {exc}"[:600]
                logger.warning(f"KV_BUDGET rung {blocks} blocks: NOT feasible -- {attempt['error'][:200]}")
            finally:
                try:
                    model.generator.teardown()
                except Exception:  # noqa: BLE001
                    pass
                if kv_cache is not None:
                    for pair in kv_cache:
                        for t in pair:
                            try:
                                ttnn.deallocate(t)
                            except Exception:  # noqa: BLE001
                                pass
                report["attempts"].append(attempt)
            if attempt.get("feasible"):
                break

        feasible = [a for a in report["attempts"] if a.get("feasible")]
        # The ladder descends and stops at the first success, so this is a proven lower
        # bound on the ceiling -- if the top rung passed, larger sizes were never tried.
        report["largest_feasible_blocks"] = feasible[0]["blocks"] if feasible else None
        report["largest_feasible_tokens"] = feasible[0]["tokens"] if feasible else None
        report["is_proven_lower_bound_not_ceiling"] = bool(feasible and feasible[0]["blocks"] == ladder[0])
        report["status"] = "ok"
    finally:
        try:
            if model is not None:
                model.generator.teardown()
        except Exception:  # noqa: BLE001
            pass
        close_generator_mesh(mesh)

    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}, indent=2, default=str), flush=True)
    for a in report["attempts"]:
        print(
            f"  rung {a['blocks']:>6} blocks ({a['tokens']:>9} tok, "
            f"{a['bytes_per_device']/1e9:5.2f} GB/dev): feasible={a.get('feasible')} "
            f"{a.get('error','')[:120]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
