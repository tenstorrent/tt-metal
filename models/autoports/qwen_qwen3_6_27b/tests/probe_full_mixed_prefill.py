# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mixed request rows equal solo rows before and after traced decode."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import source_identity
from models.autoports.qwen_qwen3_6_27b.tests.probe_bounded_prefill import logit_metrics
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--full-prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {"all_layers": args.all_layers, "full_prepare": args.full_prepare, "cases": [], "passed": False}
    report["source_sha256"] = source_identity()
    try:
        model = Model(mesh, **({} if args.all_layers else {"layer_indices": [0, 3]}))
        # Caller-owned 64-page physical pool backs 96 virtual page entries.
        # Only the logical pages needed by each request are unique; unused
        # virtual tails alias private scratch pages. Request rows move with
        # their tables when fixed state slots change.
        physical = torch.randperm(64, generator=torch.Generator().manual_seed(42)).to(torch.int32)
        table = torch.stack((physical[:16].repeat(2), physical[16:32].repeat(2), physical[32:64]))
        external_cache = model.create_cache(table)
        gen = QwenGenerator(model, batch_size=3, max_seq_len=1024, kv_cache=external_cache, page_table=table)
        report.update(external_cache_identity=gen.cache is external_cache, physical_pages=64, virtual_pages=96)
        if not args.full_prepare:
            for helper in (gen.prefill_single, gen.prefill_uniform):
                helper.prepare(_diagnostic_lengths=[1, 33, 128, 512])
        gen.prepare()
        programs = int(mesh.num_program_cache_entries())
        traces = dict(gen.traces)
        lengths = [33, 128, 513]
        prompts = torch.stack([torch.arange(513, dtype=torch.int32) + 100 + 1000 * i for i in range(3)])
        uniform_solo = []
        for row in range(3):
            gen.reset()
            logits = gen.prefill_forward(
                prompts[row : row + 1],
                page_table=table[row : row + 1],
                kv_cache=gen.cache,
                prompt_lens=[33],
                empty_slots=[0],
            )
            uniform_solo.append(logits[0].clone())
        gen.reset()
        uniform = gen.prefill_forward(prompts, page_table=table, kv_cache=gen.cache, prompt_lens=[33] * 3)
        uniform_metrics = [logit_metrics(uniform_solo[row], uniform[row]) for row in range(3)]
        report["uniform_bounded_pool"] = {
            "rows": uniform_metrics,
            "bulk_enabled": bool(gen.prefill_uniform._bulk_fill_enabled),
            "passed": all(row["passed"] for row in uniform_metrics) and gen.prefill_uniform._bulk_fill_enabled,
        }
        assert report["uniform_bounded_pool"]["passed"], report["uniform_bounded_pool"]
        solo = []
        for row, length in enumerate(lengths):
            gen.reset()
            prefill = gen.prefill_forward(
                prompts[row : row + 1],
                page_table=table[row : row + 1],
                kv_cache=gen.cache,
                prompt_lens=[length],
                empty_slots=[0],
            )
            tokens = torch.zeros((3, 1), dtype=torch.int32)
            tokens[0, 0] = prefill.argmax(-1).item()
            decode = gen.decode_forward(
                tokens,
                torch.tensor([length, -1, -1]),
                page_table=gen.page_host,
                kv_cache=gen.cache,
                sample_on_device=False,
            )
            solo.append((prefill[0].clone(), decode[0].clone()))
        for order in ([0, 1, 2], [2, 0, 1], [0, 1, 2]):
            gen.reset()
            prefill = gen.prefill_forward(
                prompts[order], page_table=table[order], kv_cache=gen.cache, prompt_lens=[lengths[i] for i in order]
            )
            tokens = prefill.argmax(-1).to(torch.int32)
            decode = gen.decode_forward(
                tokens,
                torch.tensor([lengths[i] for i in order]),
                page_table=gen.page_host,
                kv_cache=gen.cache,
                sample_on_device=False,
            )
            metrics = [
                {
                    "request": request,
                    "slot": slot,
                    "prefill": logit_metrics(solo[request][0], prefill[slot]),
                    "decode": logit_metrics(solo[request][1], decode[slot]),
                }
                for slot, request in enumerate(order)
            ]
            case = {
                "order": order,
                "rows": metrics,
                "passed": all(m[phase]["passed"] for m in metrics for phase in ("prefill", "decode")),
            }
            report["cases"].append(case)
            print(f"MIXED order={order} passed={case['passed']}", flush=True)

        # Preemption discards GDN state, then reconstructs from full history.
        # Other slots stay live; only the preempted slot is decoded afterward.
        def first(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()

        protected = [
            (state, first(state.conv), first(state.recurrent)) for state in gen.cache if hasattr(state, "conv")
        ]
        gen.release_request(2)
        rebuilt = gen.prefill_forward(
            prompts[2:3], page_table=table[2:3], kv_cache=gen.cache, prompt_lens=[513], empty_slots=[2]
        )
        tokens = torch.zeros((3, 1), dtype=torch.int32)
        tokens[2, 0] = rebuilt.argmax(-1).item()
        resumed = gen.decode_forward(
            tokens, torch.tensor([-1, -1, 513]), page_table=gen.page_host, kv_cache=gen.cache, sample_on_device=False
        )
        other_slots_preserved = all(
            torch.equal(conv[:2], first(state.conv)[:2]) and torch.equal(recurrent[:2], first(state.recurrent)[:2])
            for state, conv, recurrent in protected
        )
        preemption = dict(
            prefill=logit_metrics(solo[2][0], rebuilt[0]),
            decode=logit_metrics(solo[2][1], resumed[2]),
            other_slots_preserved=other_slots_preserved,
        )
        preemption["passed"] = (
            other_slots_preserved and preemption["prefill"]["passed"] and preemption["decode"]["passed"]
        )
        report["preemption_rebuild"] = preemption
        assert preemption["passed"], preemption
        report.update(
            program_cache_before=programs,
            program_cache_after=int(mesh.num_program_cache_entries()),
            traces_unchanged=gen.traces == traces,
            counters=dict(gen.counters),
        )
        report["passed"] = all(case["passed"] for case in report["cases"]) and gen.traces == traces
        report["passed"] &= programs == report["program_cache_after"]
        assert report["passed"], report
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
