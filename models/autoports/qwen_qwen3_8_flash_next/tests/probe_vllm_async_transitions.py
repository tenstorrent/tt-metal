# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Script identical resident-feedback transitions with immediate/queued reads.

Run serialized by the device owner under timeout3600 in the serving environment.
This reconstructs two diagnostic histories; it does NOT reproduce missing v3/v4
request/page metadata. Reduced layers0/1/3, context1024, scheduler-owned pool80.
No full logits or host sampler is used. Authoritative host tokens are supplied
only at the same drained admission/remap boundaries in both timing columns.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_vllm_async import (
    check_position,
    counter_delta,
    equal_state,
    read_ranks,
    snapshot,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import DecodeSubmission
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
from models.common.sampling.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--read-order",
        choices=("immediate-first", "queued-first", "immediate-twice", "queued-twice"),
        default="immediate-first",
        help="Diagnostic controls separate overlap from cold/warm execution order; acceptance uses immediate-first.",
    )
    parser.add_argument("--sampling-top-k", type=int, choices=(1, 40), default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh artifact path")
    from vllm.transformers_utils.config import get_config

    os.environ.setdefault("QWEN_VLLM_LAYER_INDICES", "0,1,3")
    checkpoint = os.environ.get(
        "QWEN_CHECKPOINT",
        "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
    )
    report = dict(
        passed=False,
        scope="reconstructed_matched_history_adapter_diagnostic_not_serving_acceptance",
        context=1024,
        pool_blocks=80,
        max_batch_size=2,
        layers=os.environ["QWEN_VLLM_LAYER_INDICES"],
        profiles={
            "temperature": 0.0 if args.sampling_top_k == 1 else 1.0,
            "top_k": args.sampling_top_k,
            "top_p": 1.0 if args.sampling_top_k == 1 else 0.95,
            "seed": 42,
        },
        regimes=["immediate", "two_queued"],
        runs=[],
        comparisons=[],
        read_order=args.read_order,
    )
    mesh = adapter = None
    started = time.monotonic()
    try:
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(
            get_config(checkpoint, trust_remote_code=False), mesh, 2, 1024
        )
        gen = adapter.generator
        pool = adapter.allocate_kv_cache_per_layer([((80, 1, 32, 256), torch.bfloat16, i) for i in range(12)])
        assert pool is gen.shared_pool
        adapter.warmup_model_prefill(kv_cache=pool)
        profile = report["profiles"]
        sampling = SamplingParams(**profile)
        # The common formatter deliberately does not broadcast scalar seeds:
        # scalar42 seeds lane0 only, leaving lane1 entropy-seeded at every
        # authoritative reload. Match explicit request rows for both lanes so
        # the exact RNG-state oracle compares deterministic initial conditions.
        decode_sampling = SamplingParams(**{key: [value, value] for key, value in profile.items()})
        report["seed_contract"] = dict(
            prefill_single_request_seed=42,
            decode_row_seeds=[42, 42],
            formatter_source="models/common/sampling/generator.py:format_sampling_params",
            generator_source="tt/generator.py:QwenGenerator.set_sampling_params",
            prior_v1_fixture_issue=(
                "Scalar decode seed42 formats as [42,None,...]; row1 authoritative resets draw fresh entropy. "
                "Greedy outputs can match while its logical RNG state differs. Explicit two-row parameters "
                "make both decode rows seeded; the exact full-state oracle remains unchanged."
            ),
        )
        prompts = {
            "A": ([785, 13458, 13] * 171)[:513],
            "B": ([785, 3432, 13] * 43)[:129],
            "C": ([785, 9403, 13] * 6)[:17],
        }
        report["prompts"] = prompts

        def run(history, queued):
            # Shared pages are not cleared by request release. Initialize them
            # identically so complete final-page comparisons include equal tails.
            for slot in range(2):
                adapter.release_request(slot)
            for pair in pool.pairs.values():
                for tensor in pair:
                    ttnn.full_like(tensor, 0.0, optional_tensor=tensor)
            tables = torch.tensor([list(range(1, 33)), list(range(33, 65))], dtype=torch.int32)
            owners = ["A", None]
            last_tokens = {}
            outputs = {name: [] for name in prompts}
            pending = []
            ledger = []
            run_report = dict(
                history=history,
                read_mode="two_queued" if queued else "immediate",
                passed=False,
                ledger=ledger,
                tokens=outputs,
                checks={},
                max_pending_reads=0,
            )
            report["runs"].append(run_report)
            before_run = dict(gen.counters)

            def flush():
                if not pending:
                    return
                for _, _, events in pending:
                    for event in events:
                        ttnn.event_synchronize(event)
                formatted = {}
                # Reverse formatting catches overwritten host data/metadata.
                for entry, host, _ in reversed(pending):
                    result = adapter.process_decode_output_host(host)
                    assert tuple(result.shape) == (2, 1)
                    assert torch.equal(adapter.process_decode_output_host(host), result)
                    formatted[entry["sequence"]] = result.reshape(-1).tolist()
                for entry, _, _ in pending:
                    tokens = formatted[entry["sequence"]]
                    entry["output_token_ids"] = tokens
                    for row, name in enumerate(entry["requests"]):
                        if entry["submitted_positions"][row] < 0:
                            assert tokens[row] == -1
                        else:
                            assert name is not None and 0 <= tokens[row] < gen.model.config.vocab_size
                            outputs[name].append(tokens[row])
                            last_tokens[name] = tokens[row]
                pending.clear()

            def decode(label, active=(0, 1), *, reload=False, page_only=False, remap=None):
                if reload or remap is not None:
                    flush()  # Model the runner's required transition drain.
                positions = [cache.position for cache in gen.caches]
                if remap is not None:
                    positions = [positions[source] for source in remap]
                submitted = [positions[slot] if reload else 777 for slot in range(2)]
                host_tokens = [last_tokens.get(name, 1234) if reload else 1234 for name in owners]
                for slot in range(2):
                    if slot not in active:
                        submitted[slot] = -1
                entry = dict(
                    sequence=len(ledger),
                    kind="decode",
                    label=label,
                    requests=list(owners),
                    active_slots=list(active),
                    resident_positions=positions,
                    submitted_positions=submitted,
                    submitted_token_ids=host_tokens,
                    reload_inputs=reload,
                    reload_page_table=page_only,
                    slot_remap=remap,
                    sampling_seeds=[42, 42],
                    page_tables=tables.tolist(),
                )
                ledger.append(entry)
                before = dict(gen.counters)
                submission = adapter.decode_forward(
                    tokens=torch.tensor(host_tokens, dtype=torch.long),
                    start_pos=submitted,
                    page_table=tables,
                    kv_cache=pool,
                    slots=(0, 1),
                    sampling_params=decode_sampling,
                    reload_inputs=reload,
                    reload_page_table=page_only,
                    slot_remap=remap,
                    reload_sampling_params=reload,
                    reset_sampling_state=reload,
                    read_from_device=False,
                )
                assert isinstance(submission, DecodeSubmission) and submission.is_tokens
                assert submission.active_rows == active
                if reload:
                    assert gen.request_seeds[:2] == [42, 42], "Fixture left an active decode row unseeded"
                assert gen.counters["readbacks"] == before["readbacks"]
                if not reload:
                    for name in ("token_refreshes", "position_refreshes"):
                        assert gen.counters[name] == before[name], f"{label}: resident step refreshed {name}"
                if queued:
                    host, events = adapter.read_decode_output(submission, async_read=True)
                    assert events and host.tensor.storage_type() == ttnn.StorageType.HOST
                    assert all(host.tensor is not other.tensor for _, other, _ in pending)
                    assert gen.counters["synchronizations"] == before["synchronizations"]
                else:
                    host, events = adapter.read_decode_output(submission), []
                entry["counter_delta"] = counter_delta(before, gen.counters)
                pending.append((entry, host, events))
                run_report["max_pending_reads"] = max(run_report["max_pending_reads"], len(pending))
                if not queued or len(pending) == 2:
                    flush()

            def prefill(name, slot, start, end, final):
                flush()
                survivor_slots = [i for i, owner in enumerate(owners) if i != slot and owner is not None]
                survivor = {i: snapshot(gen.caches[i], tables[i].tolist()) for i in survivor_slots}
                feedback = read_ranks(gen.decode_tokens)
                seeds = read_ranks(gen.seed_state.state)
                entry = dict(
                    sequence=len(ledger),
                    kind="prefill",
                    request=name,
                    slot=slot,
                    start=start,
                    end=end,
                    prefill_sample_mask=[final],
                    page_table=tables[slot].tolist(),
                    surviving_requests={i: owners[i] for i in survivor_slots},
                )
                ledger.append(entry)
                before = dict(gen.counters)
                result, rope_delta = adapter.prefill_forward(
                    tokens=torch.tensor([prompts[name]], dtype=torch.long),
                    prompt_lens=[end],
                    start_pos=[start],
                    empty_slots=(slot,),
                    page_table=tables[slot : slot + 1],
                    kv_cache=pool,
                    sampling_params=sampling,
                    prefill_sample_mask=[final],
                )
                assert rope_delta.tolist() == [0]
                if final:
                    assert result.shape == (1, 1)
                    token = int(result[0, 0])
                    outputs[name].append(token)
                    last_tokens[name] = token
                    entry["output_token_ids"] = [token]
                else:
                    assert result is None, "All-intermediate prefill must return no token/logits output"
                    for counter in ("sampling_replays", "readbacks"):
                        assert gen.counters[counter] == before[counter], f"Intermediate prefill changed {counter}"
                    entry["output_token_ids"] = []
                for i in survivor_slots:
                    equal_state(survivor[i], snapshot(gen.caches[i], tables[i].tolist()), f"Prefill preserves slot{i}")
                for old, new in zip(feedback, read_ranks(gen.decode_tokens), strict=True):
                    for i in survivor_slots:
                        assert old.flatten()[i] == new.flatten()[i], "Prefill overwrote survivor feedback"
                for old, new in zip(seeds, read_ranks(gen.seed_state.state), strict=True):
                    for i in survivor_slots:
                        assert torch.equal(old[i], new[i]), "Prefill changed survivor RNG"
                entry["counter_delta"] = counter_delta(before, gen.counters)
                check_position(gen.caches[slot], end)

            prefill("A", 0, 0, 256, False)
            prefill("A", 0, 256, 512, False)
            prefill("A", 0, 512, 513, True)
            decode("initial_singleton", active=(0,), reload=True)
            if history == "A_extra_singleton":
                decode("extra_singleton", active=(0,))
            flush()
            owners[1] = "B"
            prefill("B", 1, 0, 64, False)
            prefill("B", 1, 64, 128, False)
            prefill("B", 1, 128, 129, True)
            decode("admit_B", reload=True)
            # B next consumes position130. End at position159, leaving a
            # queued resident pair at159/160 around its new logical page5.
            for position in range(130, 159):
                decode(f"joint_B_position_{position}")
            flush()
            old_physical = {}
            for index, layer in enumerate(gen.caches[1].layers):
                if layer.attention is not None:
                    for name in ("k_cache", "v_cache"):
                        old_physical[index, name] = read_ranks(getattr(layer.attention, name))
            decode("before_page_growth_B159")
            tables[1, 5] = 70
            decode("page_growth_B160", page_only=True)
            flush()
            for index, layer in enumerate(gen.caches[1].layers):
                if layer.attention is not None:
                    for name in ("k_cache", "v_cache"):
                        for rank, current in enumerate(read_ranks(getattr(layer.attention, name))):
                            previous = old_physical[index, name][rank]
                            assert torch.equal(current[38], previous[38]), "Growth wrote old physical page38"
                            assert not torch.equal(current[70], previous[70]), "Growth did not write physical page70"
            check_position(gen.caches[1], 161)
            run_report["checks"]["queued_page_only_growth"] = dict(
                passed=True, logical_position=160, old_physical_page=38, new_physical_page=70
            )

            owners[:] = owners[::-1]
            tables = tables.flip(0).clone()
            decode("live_slot_swap", reload=True, remap=(1, 0))
            flush()  # A transition output is not promised overlap-safe.
            decode("resident_after_swap_1")
            decode("resident_after_swap_2")
            flush()
            survivor = snapshot(gen.caches[0], tables[0].tolist())
            before_release = dict(gen.counters)
            adapter.release_request(1)  # A retires; B stays live in slot0.
            owners[1] = None
            equal_state(survivor, snapshot(gen.caches[0], tables[0].tolist()), "Retirement preserves B")
            assert gen.counters["releases"] == before_release["releases"], "Request retirement released traces"
            ledger.append(dict(sequence=len(ledger), kind="release", request="A", slot=1, surviving_request="B"))
            decode("survivor_B_1", active=(0,))
            decode("survivor_B_2", active=(0,))
            flush()
            owners[1] = "C"
            prefill("C", 1, 0, 16, False)
            prefill("C", 1, 16, 17, True)
            decode("reuse_slot_for_C", reload=True)
            flush()
            decode("resident_after_reuse_1")
            decode("resident_after_reuse_2")
            flush()
            final_state = {
                f"slot{slot}.{name}": values
                for slot, cache in enumerate(gen.caches)
                for name, values in snapshot(cache, tables[slot].tolist()).items()
            }
            final_state["feedback"] = read_ranks(gen.decode_tokens)
            final_state["rng"] = read_ranks(gen.seed_state.state)
            run_report["final_rng_rows"] = [
                dict(rank=rank, rows=value[:2, :5].tolist()) for rank, value in enumerate(final_state["rng"])
            ]
            run_report["final_owners"] = owners
            run_report["final_positions"] = [cache.position for cache in gen.caches]
            run_report["counter_delta"] = counter_delta(before_run, gen.counters)
            run_report["passed"] = True
            return run_report, final_state

        read_modes = {
            "immediate-first": (False, True),
            "queued-first": (True, False),
            "immediate-twice": (False, False),
            "queued-twice": (True, True),
        }[args.read_order]
        for history in ("C_singleton", "A_extra_singleton"):
            immediate, state = run(history, queued=read_modes[0])
            queued, queued_state = run(history, queued=read_modes[1])
            assert immediate["max_pending_reads"] == (2 if read_modes[0] else 1)
            assert queued["max_pending_reads"] == (2 if read_modes[1] else 1)
            assert immediate["tokens"] == queued["tokens"], f"{history}: token IDs differ across read timing"
            assert immediate["final_positions"] == queued["final_positions"]
            # Counters and completion timing are deliberately different; all
            # execution inputs, request ownership and output IDs must match.
            strip = lambda entries: [{k: v for k, v in entry.items() if k != "counter_delta"} for entry in entries]
            assert strip(immediate["ledger"]) == strip(queued["ledger"]), f"{history}: histories were not identical"
            report["comparisons"].append(
                dict(
                    history=history,
                    exact_tokens=True,
                    exact_ledger=True,
                    exact_state=False,
                    rng_differences=[
                        dict(
                            rank=rank,
                            slot=row,
                            differing_words=(left[row] != right[row]).nonzero().flatten().tolist(),
                            expected=left[row].tolist(),
                            actual=right[row].tolist(),
                        )
                        for rank, (left, right) in enumerate(zip(state["rng"], queued_state["rng"], strict=True))
                        for row in range(left.shape[0])
                        if not torch.equal(left[row], right[row])
                    ],
                )
            )
            equal_state(state, queued_state, f"{history}: final state")
            report["comparisons"][-1]["exact_state"] = True
            del state, queued_state
        first, second = report["runs"][0]["tokens"], report["runs"][2]["tokens"]
        report["cross_history"] = {
            name: dict(
                equal=first[name] == second[name],
                first_different_token=next(
                    (index for index, (a, b) in enumerate(zip(first[name], second[name])) if a != b), None
                ),
                lengths=[len(first[name]), len(second[name])],
            )
            for name in prompts
        }
        report[
            "cross_history_interpretation"
        ] = "Descriptive only: these reconstructed histories cannot attribute original v3/v4 serving differences."
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if adapter is not None:
                adapter.release_persistent_capture()
        except Exception:
            report["passed"] = False
            report["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            except Exception:
                report["passed"] = False
                report["mesh_close_error"] = traceback.format_exc()
                raise
            finally:
                report["seconds"] = time.monotonic() - started
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ADAPTER_ASYNC_TRANSITIONS_PASS", flush=True)


if __name__ == "__main__":
    main()
