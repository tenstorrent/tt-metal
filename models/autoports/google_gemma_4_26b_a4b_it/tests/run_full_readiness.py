# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the standard readiness entry checks on the model's TP4 mesh."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualitative", action="store_true")
    p.add_argument("--performance", action="store_true")
    p.add_argument("--skip-readiness", action="store_true")
    p.add_argument(
        "--reference", type=Path, default=Path("models/autoports/google_gemma_4_26b_a4b_it/readiness_aime24_chat.refpt")
    )
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {}
    try:
        gen = build_generator(None, mesh, max_seq_len=8192)
        if not args.skip_readiness:
            reference = load_reference(args.reference)
            report["prefill"] = [
                _run_one_entry_prefill(generator=gen, entry=e, reference=reference) for e in reference.entries
            ]
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("PREFILL", report["prefill"], flush=True)
            gen.reset()
            acc = TokenAccuracy(args.reference)
            report["decode"] = [_run_one_entry(generator=gen, acc=acc, entry_idx=i) for i in range(acc.num_entries)]
            report["decode_counters"] = gen.counters
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("DECODE", report["decode"], flush=True)
            for phase in ("prefill", "decode"):
                assert all(r["top5"] >= 0.98 and r["top100"] == 1.0 for r in report[phase]), phase
        if args.qualitative:
            controls = json.loads((args.output.parent / "qualitative_hf.json").read_text())
            rows = []
            for row in controls["prompts"]:
                generated = gen.generate(row["prompt_tokens"], 128)
                rows.append(
                    {
                        **row,
                        "tt_tokens": generated,
                        "tt_completion": gen.tokenizer.decode(generated, skip_special_tokens=False),
                        "metrics": gen.metrics,
                    }
                )
                (args.output.parent / "qualitative_tt.json").write_text(
                    json.dumps({**controls, "prompts": rows}, indent=2) + "\n"
                )
                print("QUALITATIVE", row["id"], rows[-1]["tt_completion"], flush=True)
        if args.performance:
            prompt = gen.tokenizer.encode("This is a document about numbers and arithmetic. " * 1024)[:4096]
            assert len(prompt) == 4096
            gen.generate(prompt, 8, stop_on_eos=False)
            generated = gen.generate(prompt, 128, stop_on_eos=False)
            result = {
                **gen.metrics,
                "concurrency": 1,
                "tokens": generated,
                "completion": gen.tokenizer.decode(generated),
            }
            (args.output.parent / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
            print("PERFORMANCE", gen.metrics, flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
