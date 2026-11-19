# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full real layer-1 PLE continuation and live-trace coexistence contract.

The TT boundary is hidden states plus raw token words. The entire checkpoint
embedding domain is resident in the packed table; no selected-row substitute
or prepared embedding is passed to TT. HF reads exact checkpoint rows lazily.
All request buffers, hash buckets, input shapes, and decode signatures are
prepared and warmed before capture. Other-request prefills then run while the
decode trace remains alive, with only the exact trace output acknowledged.

This short-context diagnostic does not establish the advertised context limit
or public batched API support. It opens only a 1x1 mesh and obeys TT_VISIBLE.
"""

import argparse
import json
import os
import time
from pathlib import Path

# Enable the allocator's live-trace check before the first TTNN import. An
# explicit external disable is rejected below instead of silently claiming it.
os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")

import torch
from transformers import AutoConfig, DynamicCache
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import measure, runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference, token_rows
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding

STATE_NAMES = ("conv", "recurrent", "ple_conv", "token_history")
CONTINUATION_STARTS = (1, 31, 32, 33)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_nf4_absmax"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="1,31,32,33,65")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) < 1:
        parser.error("--lengths requires positive lengths")
    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        parser.error("TT_METAL_TRACE_ALLOC_TRACKING must be enabled for this contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    torch.manual_seed(1753)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    if max(lengths) + 3 > config.max_position_embeddings:
        parser.error("Requested prefix plus decode exceeds the target context")
    reference, weights = load_reference(args.snapshot, config)
    full_length = max(65, *lengths)
    hidden_width = config.hidden_size * config.hc_count
    prompt = (torch.randn(1, full_length, hidden_width) * 0.02).bfloat16()
    decode_inputs = [(torch.randn(1, 1, hidden_width) * 0.02).bfloat16() for _ in range(3)]
    ids = torch.randint(0, config.vocab_size, (1, full_length + 3))
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    # The continuation controls retain nontrivial lexical history. EOS is
    # covered near the end of prefill and again in the changed decode inputs.
    ids[0, full_length - 2] = eos
    ids[0, full_length + 1] = eos

    def hf(value, tokens, cache=None):
        return reference(
            value,
            ple_input_ids=tokens,
            position_embeddings=(None, None),
            past_key_values=cache if cache is not None else DynamicCache(config=config),
        )

    with torch.inference_mode():
        expected = {length: hf(prompt[:, :length], ids[:, :length]) for length in sorted(set([full_length, *lengths]))}
        reference_cache = DynamicCache(config=config)
        hf(prompt, ids[:, :full_length], reference_cache)
        expected_decode = [
            hf(value, ids[:, full_length + step : full_length + step + 1], reference_cache)
            for step, value in enumerate(decode_inputs)
        ]
        hf_continuation = {}
        for start in CONTINUATION_STARTS:
            cache = DynamicCache(config=config)
            hf(prompt[:, :start], ids[:, :start], cache)
            residual = prompt[:, start : start + 8]
            resumed = hf(residual, ids[:, start : start + 8], cache)
            dropped = hf(residual, ids[:, start : start + 8])
            target = expected[full_length][:, start : start + 8]
            hf_continuation[str(start)] = {
                "resumed": measure(target, resumed, residual),
                "dropped_history": measure(target, dropped, residual),
            }
    print("PLE_CONTRACT_HF_READY", flush=True)
    report = {
        "scope": "complete layer1; device hash, full resident packed table, real nonembedding weights",
        "snapshot": str(args.snapshot),
        "cache_dir": str(args.cache_dir),
        "layer_idx": 1,
        "weights": "real",
        "expert_dtype": args.expert_dtype,
        "threshold": 0.995,
        "context_scope": "short-context contract probe; no advertised-context acceptance claim",
        "request_scope": "two independent request states executed serially on one loaded instance",
        "raw_token_boundary": "UINT32 RM [1,S,16], word0 raw ID, word1 valid=1",
        "state_buffers_restored": list(STATE_NAMES),
        "hf_continuation_controls": hf_continuation,
        "sweep": [],
        "continuation": [],
        "decode": [],
        "live_trace_prefills": [],
        "runtime_audit": {
            "torch_function_mode": True,
            "forbid_from_torch_to_torch": True,
            "throw_exception_on_fallback": True,
            "completed_calls": 0,
        },
        "trace_allocation_tracking": True,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "passed": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def signpost(name):
        if args.profile:
            from tracy import signpost as mark

            mark(name)

    save()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace_id = None
    try:
        packed = PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=1,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        # Include every short bucket used in either request or continuation.
        buckets = tuple(sorted({1, 8, 31, 32, *(length % 32 or 32 for length in lengths)}))
        state = model.create_state(hash_lengths=buckets)
        other_state = model.create_state(hash_lengths=buckets)

        def host(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

        def device(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.to_device(host(value, dtype=dtype, layout=layout), mesh)

        def prepare(value, tokens):
            return (
                device(value.unsqueeze(0)),
                device(token_rows(tokens), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            )

        def snapshot(target):
            return {
                name: host(
                    ttnn.to_torch(getattr(target, name)),
                    dtype=getattr(target, name).dtype,
                    layout=getattr(target, name).layout,
                )
                for name in STATE_NAMES
            }

        def raw_state(target):
            return {name: ttnn.to_torch(getattr(target, name)).clone() for name in STATE_NAMES}

        def states_equal(left, right):
            return all(torch.equal(left[name], right[name]) for name in STATE_NAMES)

        def restore(target, saved):
            for name, tensor in saved.items():
                ttnn.copy_host_to_device_tensor(tensor, getattr(target, name))

        initial, other_initial = snapshot(state), snapshot(other_state)
        prepared = {
            length: prepare(prompt[:, :length], ids[:, :length]) for length in sorted(set([full_length, *lengths]))
        }
        continuation_inputs = {
            start: (
                prepare(prompt[:, :start], ids[:, :start]),
                prepare(prompt[:, start : start + 8], ids[:, start : start + 8]),
            )
            for start in CONTINUATION_STARTS
        }
        decode_input, decode_ids = prepare(decode_inputs[0], ids[:, full_length : full_length + 1])
        hidden_refresh = [host(value.unsqueeze(0)) for value in decode_inputs]
        token_refresh = [
            host(
                token_rows(ids[:, full_length + step : full_length + step + 1]),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for step in range(3)
        ]

        def prefill(pair, target=state):
            with runtime_audit():
                output = model.prefill_forward(pair[0], state=target, token_ids=pair[1])
            report["runtime_audit"]["completed_calls"] += 1
            return output

        def consume(output):
            actual = ttnn.to_torch(output)
            ttnn.deallocate(output)
            return actual

        def decode():
            with runtime_audit():
                output = model.decode_forward(decode_input, state=state, token_ids=decode_ids)
            report["runtime_audit"]["completed_calls"] += 1
            return output

        # Increasing lengths on the same weights/state allocations, followed
        # by a repeat after reset. Reset does not rebuild the loaded decoder.
        first_results = {}
        for length in [*lengths, lengths[0]]:
            restore(state, initial)
            actual = consume(prefill(prepared[length]))
            entry = dict(length=length, **measure(expected[length], actual, prompt[:, :length]))
            if length in first_results:
                entry["reset_equal"] = torch.equal(first_results[length], actual)
            else:
                first_results[length] = actual.clone()
            report["sweep"].append(entry)
            save()
            print("PLE_CONTRACT_SWEEP", json.dumps(entry), flush=True)

        for start, (prefix, resumed_input) in continuation_inputs.items():
            restore(state, initial)
            ttnn.deallocate(prefill(prefix))
            resumed = consume(prefill(resumed_input))
            # Drop ALL history, including PLE convolution and lexical hashes.
            restore(state, initial)
            dropped = consume(prefill(resumed_input))
            target, residual = expected[full_length][:, start : start + 8], prompt[:, start : start + 8]
            dropped_metrics = measure(target, dropped, residual)
            entry = {
                "start": start,
                "rows": 8,
                "correct": measure(target, resumed, residual),
                "dropped_all_history_negative_control": dropped_metrics,
                "negative_control_detected": dropped_metrics["finite"]
                and dropped_metrics["contribution_min_row_pcc"] < 0.995,
            }
            report["continuation"].append(entry)
            save()
            print("PLE_CONTRACT_CONTINUATION", json.dumps(entry), flush=True)

        # Warm every distinct live-trace prefill shape on its actual independent
        # state. No state, input tensor, or hash operation is created later.
        for length in sorted(set(lengths)):
            restore(other_state, other_initial)
            consume(prefill(prepared[length], other_state))
        restore(other_state, other_initial)
        restore(state, initial)
        ttnn.deallocate(prefill(prepared[full_length]))
        if args.profile:
            restore(state, initial)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            started = time.perf_counter()
            measured = prefill(prepared[full_length])
            ttnn.synchronize_device(mesh)
            report["warmed_prefill_host_ms"] = (time.perf_counter() - started) * 1000
            signpost("PERF_PREFILL_END")
            consume(measured)
        prefix_snapshot = snapshot(state)
        consume(decode())
        restore(state, prefix_snapshot)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        # Exact tensor only. Consume it before any later work or replay; its
        # backing storage may be overwritten by unrelated eager allocations.
        trace_allocation_tracker.acknowledge_corruptible(trace_output)
        ttnn.to_torch(trace_output)
        restore(state, prefix_snapshot)
        with runtime_audit():
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        report["runtime_audit"]["completed_calls"] += 1
        ttnn.to_torch(trace_output)

        saved_outputs, saved_states = [], []
        for repetition in range(2):
            restore(state, prefix_snapshot)
            outputs = []
            for step in range(3):
                # Exercise all five default prefill lengths across six replay
                # windows while keeping this one decode trace alive.
                length = lengths[(repetition * 3 + step) % len(lengths)]
                before = raw_state(state)
                restore(other_state, other_initial)
                actual_prefill = consume(prefill(prepared[length], other_state))
                after = raw_state(state)
                other_before_replay = raw_state(other_state)
                interleaved = {
                    "repetition": repetition,
                    "before_decode_step": step,
                    "length": length,
                    "decode_state_unchanged_by_prefill": states_equal(before, after),
                    **measure(expected[length], actual_prefill, prompt[:, :length]),
                }
                # All temporary eager outputs have been consumed/deallocated
                # before execute_trace invokes the allocator safety check.
                ttnn.copy_host_to_device_tensor(hidden_refresh[step], decode_input)
                ttnn.copy_host_to_device_tensor(token_refresh[step], decode_ids)
                ttnn.synchronize_device(mesh)
                if repetition == 0:
                    signpost("PERF_DECODE" if step == 0 else f"PERF_DECODE_STEP_{step}")
                started = time.perf_counter()
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                report["runtime_audit"]["completed_calls"] += 1
                elapsed = (time.perf_counter() - started) * 1000
                if repetition == 0:
                    signpost("PERF_DECODE_END" if step == 0 else f"PERF_DECODE_STEP_{step}_END")
                actual = ttnn.to_torch(trace_output)
                outputs.append(actual.clone())
                interleaved["prefill_state_unchanged_by_decode"] = states_equal(
                    other_before_replay, raw_state(other_state)
                )
                report["live_trace_prefills"].append(interleaved)
                if repetition == 0:
                    report["decode"].append(
                        dict(step=step, host_ms=elapsed, **measure(expected_decode[step], actual, decode_inputs[step]))
                    )
                save()
                print(
                    "PLE_CONTRACT_REPLAY",
                    json.dumps({"repetition": repetition, "step": step, "prefill": interleaved}),
                    flush=True,
                )
            saved_outputs.append(outputs)
            saved_states.append(raw_state(state))
        report["decode_reset_equal"] = all(torch.equal(a, b) for a, b in zip(*saved_outputs))
        report["decode_state_reset_equal"] = {
            name: torch.equal(saved_states[0][name], saved_states[1][name]) for name in STATE_NAMES
        }
        report["trace"] = {
            "captures": 1,
            "replays": 7,
            "unmeasured_warmup_replays": 1,
            "trace_id": str(trace_id),
            "acknowledgment": "sole exact output; consumed before unrelated prefill and next replay",
            "prefill_shapes_warmed_before_capture": sorted(set(lengths)),
            "hash_buckets_prepared_before_capture": list(buckets),
        }
        report["passed"] = (
            all(row["passed"] and row.get("reset_equal", True) for row in report["sweep"])
            and all(row["correct"]["passed"] and row["negative_control_detected"] for row in report["continuation"])
            and all(row["resumed"]["passed"] for row in hf_continuation.values())
            and all(row["passed"] for row in report["decode"])
            and all(
                row["passed"] and row["decode_state_unchanged_by_prefill"] and row["prefill_state_unchanged_by_decode"]
                for row in report["live_trace_prefills"]
            )
            and report["decode_reset_equal"]
            and all(report["decode_state_reset_equal"].values())
        )
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        save()
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
