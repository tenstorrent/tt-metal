# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Localise the multi-slot serving hang to one variable.

``adapter_probe.py --layers 0,3`` hangs deterministically (twice, across a device
reset, with byte-identical batch-1 output before it) at the *first traced decode
step of the multi-request case*: ``dump_running_operations`` reports
``PagedUpdateCacheDeviceOperation (trace id: 0)`` RUNNING on 3 cores per device
while the fabric routers sit in ``transaction_flushed`` with unretired NOC reads.

Three things change at once between the batch-1 case that works and the
multi-request case that hangs:

  A. three prefills run back to back, with no drain between them, before any
     decode (batch-1 always ran ``prefill -> 16 decodes -> prefill``);
  B. the traced decode step has three *active* rows instead of one;
  C. those rows sit at three different positions in three different page-table
     rows.

Each arm below moves exactly one of them.  ``--arm`` picks one; the runner script
runs them in separate processes so a hang in one does not mask the next.

    prefill3_decode1   A alone: three prefills, then decode with ONE active row.
    prefill1_decode3   B/C alone: one prefill, then decode with three active rows.
    prefill3_drain     A plus an explicit drain between prefills.
    prefill3_decode3   the failing combination, as a control.
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

ARMS = (
    "prefill3_decode1",
    "prefill1_decode3",
    "prefill3_drain",
    "prefill3_decode3",
    # Second round, once the first localised the failure to "one batched
    # prefill_forward call carrying three rows" rather than to three prefills:
    "prefill3_separate",  # three separate calls, NO drain -- drain vs batching
    "prefill3_batched_drain",  # one batched call, drained afterwards
    "prefill2_decode2",  # does it need three rows, or just more than one?
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.environ["MUSE_GLIMMER_VLLM_LAYER_INDICES"] = args.layers
    os.environ["MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET"] = "262144"

    import torch
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
    from models.autoports.meta_models_muse_glimmer_30b.tt.model import HF_MODEL_ID, weights_snapshot_dir
    from models.common.sampling.generator import SamplingParams

    block_size, max_len, seqs = 64, 131072, 32
    blocks_per_seq = max_len // block_size
    hf_config = AutoConfig.from_pretrained(str(weights_snapshot_dir(HF_MODEL_ID)), local_files_only=True)
    hf_config._name_or_path = HF_MODEL_ID

    mesh = open_generator_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    report = {"arm": args.arm, "layers": args.layers}
    try:
        model = Adapter.initialize_vllm_model(hf_config, mesh, seqs, max_seq_len=max_len)
        num_blocks = 4128
        kv_cache = model.allocate_kv_cache(
            (num_blocks, 1, block_size, 128), torch.bfloat16, len(model.generator.model.layers)
        )
        model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
        model.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=seqs,
            num_blocks=blocks_per_seq,
            can_sample_on_device=True,
            enable_trace=False,
        )
        model.warmup_model_decode(
            kv_cache=kv_cache,
            max_batch_size=seqs,
            num_blocks=blocks_per_seq,
            can_sample_on_device=True,
            enable_trace=True,
        )
        print("READY", flush=True)

        def greedy(rows: int):
            return SamplingParams(temperature=[0.0] * rows, top_k=[1] * rows, top_p=[1.0] * rows, seed=[None] * rows)

        prompt_lens = [96, 130, 61]
        rows = 1 if args.arm == "prefill1_decode3" else 3
        prefill_rows = 1 if args.arm == "prefill1_decode3" else 3
        decode_rows = 1 if args.arm == "prefill3_decode1" else 3
        per_row_blocks = 5
        table = torch.zeros(3, blocks_per_seq, dtype=torch.int32)
        for row in range(3):
            start = 100 + row * per_row_blocks
            table[row, :per_row_blocks] = torch.arange(start, start + per_row_blocks, dtype=torch.int32)

        if args.arm == "prefill3_drain":
            first_ids = []
            for row in range(3):
                ids = torch.arange(2000 + row * 50, 2000 + row * 50 + prompt_lens[row], dtype=torch.int32).reshape(
                    1, -1
                )
                out = model.prefill_forward(
                    tokens=ids,
                    page_table=table[row : row + 1],
                    kv_cache=kv_cache,
                    enable_trace=False,
                    prompt_lens=[prompt_lens[row]],
                    start_pos=torch.zeros(1, dtype=torch.int32),
                    sampling_params=greedy(1),
                )
                ttnn.synchronize_device(mesh)
                first_ids.append(int(out[0]))
            first = torch.tensor(first_ids, dtype=torch.int64)
            print(f"PREFILL_DONE(drained) {first_ids}", flush=True)
        else:
            max_prompt = max(prompt_lens[:prefill_rows])
            ids = torch.zeros(prefill_rows, max_prompt, dtype=torch.int32)
            for row in range(prefill_rows):
                ids[row, : prompt_lens[row]] = torch.arange(
                    2000 + row * 50, 2000 + row * 50 + prompt_lens[row], dtype=torch.int32
                )
            first = model.prefill_forward(
                tokens=ids,
                page_table=table[:prefill_rows],
                kv_cache=kv_cache,
                enable_trace=False,
                prompt_lens=prompt_lens[:prefill_rows],
                start_pos=torch.zeros(prefill_rows, dtype=torch.int32),
                sampling_params=greedy(prefill_rows),
            )
            print(f"PREFILL_DONE {first.tolist()}", flush=True)

        decode_table = torch.zeros(seqs, blocks_per_seq, dtype=torch.int32)
        decode_table[:3] = table
        positions = torch.full((seqs,), -1, dtype=torch.int32)
        decode_tokens = torch.zeros(seqs, 1, dtype=torch.int32)
        for row in range(decode_rows):
            positions[row] = prompt_lens[row]
            decode_tokens[row, 0] = int(first[min(row, first.numel() - 1)])

        emitted = [[] for _ in range(decode_rows)]
        started = time.perf_counter()
        for step in range(args.steps):
            out = model.decode_forward(
                tokens=decode_tokens,
                page_table=decode_table,
                kv_cache=kv_cache,
                start_pos=positions,
                enable_trace=True,
                read_from_device=False,
                sampling_params=greedy(seqs),
                reset_batch=(step == 0),
            )
            host, events = model.read_decode_output(out, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            got = model.process_decode_output_host(host, is_tokens=True)
            for row in range(decode_rows):
                emitted[row].append(int(got[row]))
            print(f"DECODE step={step} {[e[-1] for e in emitted]}", flush=True)
        report["decode_s"] = round(time.perf_counter() - started, 2)
        report["tokens"] = emitted
        report["status"] = "ok"
        print("ARM_OK", flush=True)
    finally:
        try:
            model.generator.teardown()
        except Exception:  # noqa: BLE001
            pass
        close_generator_mesh(mesh)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
