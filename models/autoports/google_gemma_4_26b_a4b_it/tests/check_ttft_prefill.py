# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced prefill trace timing/state probe; this is not a serving benchmark."""

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--baseline-only", action="store_true", help="Measure unchanged eager APIs and exit")
    parser.add_argument("--skip-long", action="store_true", help="Explicitly omit the 1025-token fallback control")
    parser.add_argument(
        "--wire-sampling", action="store_true", help="Use 32-row prefill tables and neutral padded decode parameters"
    )
    parser.add_argument(
        "--transitions", action="store_true", help="Check sampling/reset transitions and lengths 1023/1024"
    )
    args = parser.parse_args()
    if args.decode_steps < 1 or args.decode_steps > 32:
        parser.error("--decode-steps must be in 1..32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Reduced layers 0 and 5, direct adapter; synchronized timings are not real-server TTFT",
        layers=[0, 5],
        max_seq_len=2048,
        trace_region_size=1024**3,
        baseline_only=args.baseline_only,
        transitions_enabled=args.transitions,
        wire_sampling=args.wire_sampling,
        skipped=["1025-token fallback"] if args.skip_long else [],
        eager_phases=[],
        controls=[],
        traced=[],
        transition_controls={},
        transitions=[],
        prefill_only=[],
        passed=False,
    )
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1024**3)
    gen = None

    def timed(function):
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        value = function()
        ttnn.synchronize_device(mesh)
        return value, (time.perf_counter() - started) * 1000

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        gen = Gemma4Generator(mesh, max_seq_len=2048, layer_indices=(0, 5))
        gen.prefill_trace_enabled = False
        adapter = AutoportGemma4ForCausalLM(gen, 2)
        # Two requests, each with 64 sliding and 64 full-attention pages.
        # The physical pool is shared, but live logical page IDs are disjoint.
        specs = [((256, 2, 32, 256), torch.bfloat16, i) for i in range(30)]
        specs[5] = ((256, 1, 32, 512), torch.bfloat16, 0)
        caches = [adapter.allocate_kv_cache_per_layer(specs) for _ in range(2)]
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        modes = dict(
            greedy=params,
            sampled=SamplingParams(temperature=0.7, top_k=16, top_p=0.9, seed=71),
            penalty=SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                seed=71,
                presence_penalty=0.5,
                frequency_penalty=0.3,
                repetition_penalty=1.1,
            ),
            host=None,
        )

        def inputs(length, batch=1, token=100, changed_pages=False, cache_index=0):
            ids = torch.full((batch, length), token, dtype=torch.long)
            ids[:, 0] = 2
            if batch == 2:
                ids[1, 1:] += 1
            table = torch.arange(64, dtype=torch.int32).repeat(batch, 1)
            table += torch.arange(batch, dtype=torch.int32)[:, None] * 128
            if changed_pages:
                table = table.flip(1)
            full = table + 64
            tables = [table] * 30
            tables[5] = full
            return ids, tables, caches[cache_index]

        def request(case, mode="greedy", decode_steps=None):
            ids, tables, cache = inputs(**case)
            lengths = [ids.shape[1]] * ids.shape[0]
            if args.wire_sampling:
                sliding = torch.nn.functional.pad(tables[0], (0, 0, 0, 32 - len(lengths)))
                full = torch.nn.functional.pad(tables[5], (0, 0, 0, 32 - len(lengths)))
                tables = [sliding] * 30
                tables[5] = full
            request_params = modes[mode]
            before = dict(gen.counters)
            # Match the adapter's compact lookup, while giving its public API
            # the scheduler's padded rows below so compaction is exercised.
            reusable = gen.can_reuse_serving_prefill(
                ids,
                page_table=(tables[0][: len(lengths)], tables[5][: len(lengths)]),
                kv_cache=cache,
                prompt_lens=lengths,
            )
            output, prefill_ms = timed(
                lambda: adapter.prefill_forward(
                    ids,
                    tables[0],
                    cache,
                    lengths,
                    sampling_params=request_params,
                    page_tables_per_layer=tables,
                    empty_slots=list(range(ids.shape[0])),
                )
            )
            if mode == "host":
                output = output.argmax(dim=-1)
            observed = [[int(value)] for value in output.flatten()]
            decode_ms = []
            decode_params = request_params
            if args.wire_sampling and request_params is not None:
                inactive = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
                decode_params = SamplingParams(
                    **{
                        field.name: [getattr(request_params, field.name)] * len(lengths)
                        + [getattr(inactive, field.name)] * (32 - len(lengths))
                        for field in fields(SamplingParams)
                    }
                )
            for step in range(args.decode_steps if decode_steps is None else decode_steps):
                output, elapsed = timed(
                    lambda: adapter.decode_forward(
                        output,
                        torch.tensor([length + step for length in lengths], dtype=torch.int32),
                        tables[0],
                        cache,
                        sampling_params=decode_params,
                        page_tables_per_layer=tables,
                        reset_batch=step == 0,
                        **(
                            dict(
                                prompt_tokens=ids,
                                output_tokens=torch.tensor(observed, dtype=torch.long),
                                output_token_counts=torch.full((len(lengths),), step + 1, dtype=torch.int64),
                            )
                            if mode in ("sampled", "penalty")
                            else {}
                        ),
                    )
                )
                if mode == "host":
                    output = output.argmax(dim=-1)
                decode_ms.append(elapsed)
                for row, value in zip(observed, output.flatten()):
                    row.append(int(value))
            if decode_steps != 0:
                assert gen.cache is cache, "External cache binding changed"
            return dict(
                case=case,
                mode=mode,
                prefill_table_shape=list(tables[0].shape),
                can_reuse_before=reusable,
                tokens=observed,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                counter_delta={name: value - before.get(name, 0) for name, value in gen.counters.items()},
            )

        # Isolate eager preparation, model, sampler and host transfer costs.
        # The first pass includes compilation; the second uses warmed programs.
        for length in (32, 128, 129, 256):
            ids, tables, cache = inputs(length)
            expected = None
            for label in ("first_use", "warm_1", "warm_2"):
                phases = dict(label=label, length=length)
                _, phases["configure_ms"] = timed(lambda: gen.configure_sampling(params, prompt_tokens=ids))
                logits, phases["model_ms"] = timed(
                    lambda: gen.prefill_forward(
                        ids, page_table=(tables[0], tables[5]), kv_cache=cache, prompt_lens=[length]
                    )
                )
                sampled, phases["sampling_ms"] = timed(lambda: gen.sample_prefill(logits))
                host, phases["read_ms"] = timed(lambda: ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).clone())
                phases["token"] = int(host.flatten()[0])
                phases["sum_ms"] = sum(phases[key] for key in ("configure_ms", "model_ms", "sampling_ms", "read_ms"))
                report["eager_phases"].append(phases)
                if expected is None:
                    expected = phases["token"]
                assert phases["token"] == expected, phases
                del logits, sampled, host
                save()
        if args.baseline_only:
            report["passed"] = True
            return

        cases = [dict(length=length) for length in (32, 33, 127, 128, 129)]
        cases += [
            dict(length=129, token=113),
            dict(length=129, token=113, changed_pages=True),
            dict(length=129, token=113, changed_pages=True, cache_index=1),
            dict(length=33, batch=2),
        ]
        if args.transitions:
            cases.extend(dict(length=length) for length in (1023, 1024))
        if not args.skip_long:
            cases.append(dict(length=1025))
        for case in cases:
            report["controls"].append(request(case))
            save()

        gen._release_trace()
        gen.prefill_trace_enabled = True
        for case, control in zip(cases, report["controls"]):
            for repeat in range(2):
                result = request(case)
                result["repeat"] = repeat
                result["matches_eager"] = result["tokens"] == control["tokens"]
                report["traced"].append(result)
                save()
                assert result["matches_eager"], result
                eligible = case.get("batch", 1) == 1 and case["length"] <= 1024
                if not eligible:
                    assert not result["can_reuse_before"], result
                elif repeat:
                    assert result["can_reuse_before"], "Repeated eligible prefill did not retain its trace"
        if args.transitions:
            adapter.allow_host_sampling = True
            transition_case = dict(length=33, token=117)
            gen._release_trace()
            gen.prefill_trace_enabled = False
            for mode in modes:
                report["transition_controls"][mode] = request(transition_case, mode)
                save()
            gen._release_trace()
            gen.prefill_trace_enabled = True
            sequence = [
                ("greedy_initial", "greedy"),
                ("greedy_replay", "greedy"),
                ("sampled_fallback", "sampled"),
                ("greedy_after_sampled", "greedy"),
                ("greedy_replay_after_sampled", "greedy"),
                ("penalty_fallback", "penalty"),
                ("greedy_after_penalty", "greedy"),
                ("greedy_replay_after_penalty", "greedy"),
                ("host_compatibility", "host"),
                ("greedy_after_host", "greedy"),
                ("greedy_replay_after_host", "greedy"),
                ("greedy_replay_after_reset", "greedy"),
            ]
            for label, mode in sequence:
                if label == "greedy_replay_after_reset":
                    gen.reset()
                result = request(transition_case, mode)
                result["label"] = label
                result["matches_eager"] = result["tokens"] == report["transition_controls"][mode]["tokens"]
                report["transitions"].append(result)
                save()
                assert result["matches_eager"], result
                if "replay" in label:
                    assert result["can_reuse_before"], result
                    assert result["counter_delta"].get("prefill_replays", 0) == 1, result
                elif mode != "greedy":
                    assert result["counter_delta"].get("prefill_replays", 0) == 0, result
                    assert gen.prefill_prepared is None, "Fallback retained the greedy prefill graph"
        # OSL1 streams must warm without relying on a later decode call, then
        # safely transition to the normal decode-owned persistent buffers.
        case = dict(length=128, token=119)
        gen._release_trace()
        gen.prefill_prepared = None
        gen.prefill_trace_enabled = False
        control = request(case)
        gen._release_trace()
        gen.prefill_prepared = None
        gen.prefill_trace_enabled = True
        for repeat in range(3):
            result = request(case, decode_steps=0)
            result["repeat"] = repeat
            result["matches_eager"] = result["tokens"] == [row[:1] for row in control["tokens"]]
            report["prefill_only"].append(result)
            save()
            assert result["matches_eager"], result
            if repeat:
                assert result["counter_delta"].get("prefill_replays", 0) == 1, result
        result = request(case)
        result["label"] = "decode_after_prefill_only"
        result["matches_eager"] = result["tokens"] == control["tokens"]
        report["prefill_only"].append(result)
        assert result["matches_eager"], result
        report["passed"] = True
        report["counters"] = dict(gen.counters)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
