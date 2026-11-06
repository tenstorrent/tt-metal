# SPDX-License-Identifier: Apache-2.0
"""Generator batched/solo exactness and replay acceptance on real model weights.

Reduced layers are explicit by default. This is correctness evidence, never a
full-model serving performance measurement. Run with allocation tracking before
TTNN import; the coordinator supplies the outer timeout and hardware ownership.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

import ttnn

from ..tt.generator import Gemma4Generator

LENGTHS = (5, 31, 32, 33, 127, 128, 129, 257, 640, 1000, 1024)


def cache_snapshot(generator, lengths):
    result = []
    for layer_index, (pair, pages) in enumerate(zip(generator.kv_cache, generator.prefill.host_pages)):
        for kind, tensor in enumerate(pair):
            for device, shard in enumerate(ttnn.get_device_tensors(tensor)):
                # Read whole caches once. Per-page device slices would introduce
                # page-address-specific programs after trace preparation.
                full = ttnn.to_torch(shard)
                for row, length in enumerate(lengths):
                    count = (length + 31) // 32
                    valid = full[pages[row, :count].long()].permute(1, 0, 2, 3).flatten(1, 2)[:, :length].clone()
                    result.append(((generator.model.layer_indices[layer_index], kind, device, row), valid))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(2, 4, 8, 32), default=8)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    parser.add_argument("--layers", default="0,5", help="Comma-separated real layer indices or all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-source", choices=("random", "chat"), default="random")
    parser.add_argument(
        "--mixed", action="store_true", help="Cycle selected lengths across rows rather than uniform groups"
    )
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--positive-control", action="store_true")
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(",")]
    if any(v <= 0 or v > 1024 for v in lengths):
        raise ValueError("Acceptance lengths must be within 1..1024")
    os.environ["GEMMA4_BATCHED_PREFILL"] = "1"
    torch.set_num_threads(8)
    torch.manual_seed(729)
    configured_batch = 8 if args.batch <= 8 else 32
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    report = {
        "logical_batch": args.batch,
        "decode_batch": configured_batch,
        "layers": args.layers,
        "prompt_source": args.prompt_source,
        "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "cases": [],
        "status": "running",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    gen = None
    try:
        gen = Gemma4Generator(
            mesh,
            batch_size=configured_batch,
            max_seq_len=2048,
            layer_indices=None if args.layers == "all" else tuple(map(int, args.layers.split(","))),
        )
        gen.prepare_decode()
        candidate = gen.batched_prefill
        if candidate is None:
            raise AssertionError("Batched preparation was not enabled")
        trace_ids = tuple(str(t) for t in (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace))
        initial_captures = gen.counters["captures"]
        initial_programs = mesh.num_program_cache_entries()
        report.update(
            prepared_buckets=list(candidate.inputs),
            trace_ids=trace_ids,
            captures=initial_captures,
            programs_after_preparation=initial_programs,
        )

        def arm(tokens, row_lengths, enabled):
            gen.batched_prefill = candidate if enabled else None
            gen.reset()
            gen.set_sampling(temperature=0.0, top_k=1, top_p=1.0, seed=0)
            first = gen.prefill_forward(
                tokens,
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=np.asarray(row_lengths, dtype=np.int64),
            )
            # All four vocabulary shards, including every submitted row.
            logits = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(gen.logits)], dim=-1)
            logits = logits.reshape(configured_batch, -1)[: args.batch].clone()
            caches = cache_snapshot(gen, row_lengths)
            prefill_programs = mesh.num_program_cache_entries()
            generated = [[int(t)] for t in first[: args.batch]]
            positions = row_lengths + [-1] * (configured_batch - args.batch)
            for step in range(31):
                values = gen.decode_forward(
                    first,
                    positions,
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    reload_inputs=step == 0,
                )
                for row in range(args.batch):
                    generated[row].append(int(values[row]))
            # The final workload action is a decode replay, so a late unsafe
            # allocation in any prefill helper is checked before acceptance.
            return logits, caches, generated, prefill_programs, mesh.num_program_cache_entries()

        for offset, length in enumerate(lengths):
            row_lengths = (
                [lengths[(offset + r) % len(lengths)] for r in range(args.batch)]
                if args.mixed
                else [length] * args.batch
            )
            prompts = []
            for row, count in enumerate(row_lengths):
                if args.prompt_source == "random":
                    prompt = torch.randint(3, gen.model.config.vocab_size, (count,), dtype=torch.int32)
                else:
                    rendered = gen.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"Explain why the sky looks blue to observer {row}. "
                                + "Please provide scientific details and examples. " * 400,
                            }
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    if len(rendered) < count:
                        raise ValueError("Rendered chat is shorter than requested probe")
                    # Exact-length cache stress retains the real rendered prefix;
                    # these cut prompts are not qualitative-quality evidence.
                    prompt = torch.tensor(rendered[:count], dtype=torch.int32)
                prompts.append(prompt)
            for order in (list(range(args.batch)), list(reversed(range(args.batch)))):
                ordered_lengths = [row_lengths[r] for r in order]
                tokens = torch.zeros(args.batch, max(ordered_lengths), dtype=torch.int32)
                for row, source in enumerate(order):
                    tokens[row, : row_lengths[source]] = prompts[source]
                solo = arm(tokens, ordered_lengths, False)
                batch = arm(tokens, ordered_lengths, True)
                cache_rows = []
                for (key, a), (other_key, b) in zip(solo[1], batch[1]):
                    assert key == other_key
                    cache_rows.append(
                        {
                            "layer": key[0],
                            "kind": "KV"[key[1]],
                            "device": key[2],
                            "row": key[3],
                            "equal": torch.equal(a, b),
                        }
                    )
                divergences = [
                    next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(solo[2], batch[2])
                ]
                entry = {
                    "lengths": ordered_lengths,
                    "row_order": order,
                    "groups": candidate.groups(ordered_lengths, [0] * args.batch),
                    "logits_equal_per_row": [torch.equal(a, b) for a, b in zip(solo[0], batch[0])],
                    "max_logit_delta": (solo[0].float() - batch[0].float()).abs().max().item(),
                    "cache": cache_rows,
                    "greedy_first_divergence": divergences,
                    "solo_tokens": solo[2],
                    "batched_tokens": batch[2],
                    "program_counts": {
                        "solo_prefill": solo[3],
                        "solo_final_decode": solo[4],
                        "batched_prefill": batch[3],
                        "batched_final_decode": batch[4],
                    },
                    "trace_ids": [str(t) for t in (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace)],
                    "captures": gen.counters["captures"],
                }
                entry["pass"] = (
                    all(entry["logits_equal_per_row"])
                    and all(r["equal"] for r in cache_rows)
                    and all(d is None or d >= 8 for d in divergences)
                    and all(c == initial_programs for c in (solo[3], solo[4], batch[3], batch[4]))
                    and tuple(entry["trace_ids"]) == trace_ids
                    and entry["captures"] == initial_captures
                )
                report["cases"].append(entry)
                save()
                print(json.dumps({"lengths": ordered_lengths, "row_order": order, "pass": entry["pass"]}), flush=True)
                if not entry["pass"] and not args.continue_on_failure:
                    raise AssertionError("Batched prefill acceptance failed; see JSON")
        if args.positive_control:
            if report["tracking"] != "1" or report["skip_program_cache"] != "0":
                raise ValueError("Positive control requires tracking=1 and skip_program_cache=0")
            report["positive_control"] = {"status": "running"}
            save()
            # A host-to-device row-major copy creates only the deliberate
            # buffer. Tile conversion would also compile a persistent program
            # after capture, which cannot be removed by deleting this tensor.
            live_buffer = gen.model.upload(
                torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            try:
                ttnn.execute_trace(mesh, gen.model_trace, cq_id=0, blocking=False)
            except RuntimeError as error:
                message = str(error)
                if (
                    "device buffer(s) still alive before trace replay. These will be corrupted on replay."
                    not in message
                ):
                    report["positive_control"] = {"status": "fail", "unexpected_error": message}
                    raise
                report["positive_control"] = {"status": "detected", "expected_error": message}
            else:
                report["positive_control"] = {
                    "status": "fail",
                    "error": "Unsafe live-buffer replay unexpectedly succeeded",
                }
                raise AssertionError("Trace allocation positive control failed to detect live buffer")
            finally:
                del live_buffer
                gc.collect()
                save()
            # No exemption: releasing the deliberate allocation must restore a
            # clean replay under the same tracker and existing trace lifetime.
            gen._replay()
            report["positive_control"].update(status="pass", clean_replay_after_release=True)
        report["status"] = "pass" if all(c["pass"] for c in report["cases"]) else "fail"
        save()
        assert report["status"] == "pass"
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if gen is not None:
            for name in ("model_trace", "sample_trace", "buffered_sample_trace"):
                trace = getattr(gen, name, None)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
