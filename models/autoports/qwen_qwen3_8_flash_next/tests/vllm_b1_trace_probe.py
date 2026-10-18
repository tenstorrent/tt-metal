# SPDX-License-Identifier: Apache-2.0
"""Tracked reduced B1 adapter coverage; exclusive devices required except --cpu-check.

Real layers0,1,3 and full LM head/sampling, C262144, all prefill buckets1..512.
This is not full-model quality, server scheduler, concurrency or performance proof.
"""

import argparse
import copy
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 262144
BUCKETS = [1 << i for i in range(10)]
SIGNATURES = {"decode", "greedy", "sample", "argmax"} | {f"prefill_{n}" for n in BUCKETS}


def chunks(ends):
    result, start = [], 0
    for end in ends:
        if not start < end <= CONTEXT:
            raise ValueError("Absolute ends must strictly increase within context")
        while start < end:
            length = max(n for n in BUCKETS if n <= end - start)
            result.append(dict(start=start, length=length, slot=0))
            start += length
    return result


def workload():
    # Controls reload authoritative host inputs; repeats submit the next device
    # step before resolving the preceding output snapshot, with poisoned inputs.
    specs = [
        ("control_511", [511], "greedy", 8, None),
        ("control_512", [512], "greedy", 8, None),
        ("control_513", [512, 513], "greedy", 8, None),
        ("control_129", [129], "greedy", 8, None),
        ("control_130", [130], "greedy", 66, None),
        ("sample_131", [131], "sample", 34, None),
        ("unaligned_513", [129, 513], "greedy", 8, None),
        ("control_645", [512, 645], "greedy", 8, None),
        ("repeat_511", [511], "greedy", 8, "control_511"),
        ("repeat_sample_131", [131], "sample", 34, "sample_131"),
        ("repeat_130", [130], "greedy", 66, "control_130"),
        ("repeat_129", [129], "greedy", 8, "control_129"),
        ("repeat_unaligned_513", [129, 513], "greedy", 8, "unaligned_513"),
    ]
    return [dict(name=name, ends=ends, mode=mode, steps=steps, reference=ref) for name, ends, mode, steps, ref in specs]


class Pages:
    """Tiny single-request allocator; does not stand in for vLLM scheduler proof."""

    def __init__(self):
        self.blocks, self.events = [], []

    def ensure(self, position, step):
        old = len(self.blocks)
        while len(self.blocks) <= position // 32:
            self.blocks.append(CONTEXT // 32 - len(self.blocks))
        if len(self.blocks) != old:
            self.events.append(dict(step=step, consumed_position=position, added=self.blocks[old:]))
        return len(self.blocks) != old


def cpu_check():
    plans = workload()
    assert {e["length"] for p in plans for e in chunks(p["ends"])} == set(BUCKETS)
    assert {129, 130, 131, 511, 512, 513, 645} <= {p["ends"][-1] for p in plans}
    indexed = {p["name"]: p for p in plans}
    for plan in plans:
        if plan["reference"]:
            control = indexed[plan["reference"]]
            assert all(plan[key] == control[key] for key in ("ends", "mode", "steps"))
        pages = Pages()
        pages.ensure(plan["ends"][-1] - 1, -1)
        for step in range(plan["steps"]):
            pages.ensure(plan["ends"][-1] + step, step)
        assert len(set(pages.blocks)) == len(pages.blocks)
        assert all(0 < block <= CONTEXT // 32 for block in pages.blocks)
        if plan["steps"] == 66:
            assert sum(e["step"] >= 0 for e in pages.events) >= 2
    assert chunks([512, 513])[-1] == dict(start=512, length=1, slot=0)
    assert chunks([129, 513])[2:] == [dict(start=129, length=256, slot=0), dict(start=385, length=128, slot=0)]
    audit_checks = cpu_audit_check()
    return dict(
        status="cpu_plan_pass",
        phases=len(plans),
        decode_steps=sum(p["steps"] for p in plans),
        buckets=BUCKETS,
        audit_checks=audit_checks,
    )


def audit_events(events, baseline, phases):
    # generator.event uses json.dumps(default=str). Convert only the trusted
    # live baseline at this boundary; never coerce malformed event identifiers.
    baseline = {signature: str(trace_id) for signature, trace_id in baseline.items()}
    captures = [e for e in events if e["event"] == "capture"]
    assert Counter(e["signature"] for e in captures) == Counter({k: 1 for k in SIGNATURES})
    assert {e["signature"]: e["trace_id"] for e in captures} == baseline
    prepared = [e for e in events if e["event"] == "prepared"]
    assert len(prepared) == 1 and prepared[0]["batch"] == 1 and prepared[0]["buckets"] == BUCKETS
    assert prepared[0]["traces"] == baseline
    assert all(e["time"] < prepared[0]["time"] for e in captures)
    warm = [e for e in events if e["event"] == "preparation"]
    assert Counter(e["signature"] for e in warm) == Counter({k: 1 for k in SIGNATURES})
    assert max(e["time"] for e in warm) < min(e["time"] for e in captures)
    assert not any(
        e["event"] in {"recapture", "retirement", "retire", "invalidation", "eviction", "shutdown_release"}
        for e in events
    )
    for event in events:
        if event["event"] == "replay":
            assert baseline[event["signature"]] == event["trace_id"]
    marks = [(i, e["phase"]) for i, e in enumerate(events) if e["event"] == "b1_probe_phase"]
    assert [name for _, name in marks] == [p["name"] for p in phases]
    for index, (start, _) in enumerate(marks):
        segment = events[start + 1 : marks[index + 1][0] if index + 1 < len(marks) else len(events)]
        actual = []
        for event in segment:
            if event["event"] == "prefill_chunk":
                assert event["active"] == [True] and event["slots"] == [0]
                actual.append(dict(start=event["start_positions"][0], length=event["chunk_length"], slot=0))
        assert actual == chunks(phases[index]["ends"])
        phases[index]["actual_chunks"] = actual
    return [
        dict(
            signature=k,
            trace_id=baseline[k],
            captures=1,
            retirements=0,
            replays=sum(e["event"] == "replay" and e["signature"] == k for e in events),
        )
        for k in sorted(baseline)
    ]


def cpu_audit_check():
    class MeshTraceIdLike:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"MeshTraceId({self.value})"

    baseline = {signature: MeshTraceIdLike(i) for i, signature in enumerate(sorted(SIGNATURES))}
    original = baseline.copy()
    events = [dict(event="preparation", signature=k, time=i) for i, k in enumerate(baseline)]
    events += [
        dict(event="capture", signature=k, trace_id=str(v), time=20 + i) for i, (k, v) in enumerate(baseline.items())
    ]
    events += [
        dict(event="prepared", batch=1, buckets=BUCKETS, time=40, traces={k: str(v) for k, v in baseline.items()}),
        dict(event="b1_probe_phase", phase="cpu_fixture", time=41),
        dict(event="replay", signature="decode", trace_id=str(baseline["decode"]), time=42),
    ]
    events += [
        dict(event="prefill_chunk", active=[True], slots=[0], start_positions=[e["start"]], chunk_length=e["length"])
        for e in chunks([512, 513])
    ]
    phases = [dict(name="cpu_fixture", ends=[512, 513])]
    assert len(audit_events(events, baseline, copy.deepcopy(phases))) == 14
    assert all(baseline[k] is original[k] for k in baseline)
    for mutation in ("capture_id", "replay_id", "prepared_id", "nonstring_id", "chunk_start"):
        changed = copy.deepcopy(events)
        if mutation in ("capture_id", "nonstring_id"):
            event = next(e for e in changed if e["event"] == "capture")
            event["trace_id"] = "MeshTraceId(999)" if mutation == "capture_id" else 0
        elif mutation == "replay_id":
            next(e for e in changed if e["event"] == "replay")["trace_id"] = "MeshTraceId(999)"
        elif mutation == "prepared_id":
            next(e for e in changed if e["event"] == "prepared")["traces"]["decode"] = "MeshTraceId(999)"
        else:
            changed[-1]["start_positions"] = [511]
        try:
            audit_events(changed, baseline, copy.deepcopy(phases))
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Audit accepted corrupted {mutation}")
    return dict(live_object_baseline=True, baseline_identity_preserved=True, rejected_mutations=5)


def run(args):
    # These settings precede every import capable of loading TTNN.
    for key, value in (("TT_METAL_TRACE_ALLOC_TRACKING", "1"), ("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")):
        os.environ.setdefault(key, value)
        if os.environ[key] != value:
            raise RuntimeError(f"{key} must be {value}")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in (None, "0"):
        raise RuntimeError("Disable device profiling")
    os.environ.update(
        QWEN_FLASH_SERVING_LAYERS="0,1,3", QWEN_FLASH_CHECKPOINT=args.checkpoint, QWEN_FLASH_ALLOW_HOST_SAMPLING="0"
    )
    event_path = args.output.with_suffix(".events.jsonl")
    runtime_path = args.output.with_suffix(".runtime.json")
    for path in (args.output, event_path, runtime_path):
        if path.exists():
            raise FileExistsError(f"Preserve evidence; use a fresh path: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(QWEN_FLASH_TRACE_EVENTS=str(event_path), QWEN_FLASH_RUNTIME_REPORT=str(runtime_path))
    import torch
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from vllm_tt_plugin.model_input import TTSamplingParams

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view
    from models.autoports.qwen_qwen3_8_flash_next.tests.vllm_adapter_device_probe import check_delta
    from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
    from models.autoports.qwen_qwen3_8_flash_next.tt.precision import runtime_summary

    report = dict(
        status="running",
        scope=__doc__,
        args=vars(args),
        batch=1,
        context=CONTEXT,
        layers=[0, 1, 3],
        phases=[],
        closed_mesh=False,
        tracking={
            k: os.environ[k] for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "tests/vllm_adapter_device_probe.py"]
            + [
                ROOT / "tt" / n
                for n in (
                    "generator_vllm.py",
                    "generator.py",
                    "model.py",
                    "multichip_decoder.py",
                    "qsa.py",
                    "paged_qsa.py",
                    "precision.py",
                )
            ]
        },
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    mesh, adapter = None, None
    persist()
    try:
        torch.set_num_threads(8)
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        config = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(config, mesh, 1, CONTEXT)
        cache = adapter.allocate_kv_cache((CONTEXT // 32 + 1, 1, 32, 256), torch.float8_e4m3fn, 1)
        g = adapter.generator
        assert cache is g.kv_cache is g.model.cache
        assert g.model.precision_policy == json.loads(
            (ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text()
        )
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        adapter.warmup_model_decode(cache, enable_trace=True, can_sample_on_device=True)
        baseline, counts = dict(g.traces), dict(g.counters)
        assert set(baseline) == SIGNATURES and counts["captures"] == 14
        assert g.buckets == BUCKETS and g.prefill_trace_buckets == BUCKETS
        report.update(
            runtime=runtime_summary(g.model),
            prepared_trace_ids=baseline,
            prepared_counters=counts,
            prepared_memory=view(mesh),
            prepared_trace_memory=view(mesh, ttnn.BufferType.TRACE),
            public_pool_tokens=g.model.public_pool_tokens,
            scratch_page_base=g.model.scratch_page_base,
        )

        def forbidden_logits(*unused, **kwargs):
            raise AssertionError("No full-logits readback is allowed in this token-out probe")

        g._read_logits = forbidden_logits
        references = {}
        for plan in workload():
            g.reset()
            adapter.slot_map = [0]
            if hasattr(adapter, "sampling"):
                del adapter.sampling
            g.event("b1_probe_phase", phase=plan["name"])
            length = plan["ends"][-1]
            prompt = g.tokenizer.encode(
                (f"Request length {length}: explain plants, water, sunlight and roots. " * 200),
                add_special_tokens=False,
            )[:length]
            assert len(prompt) == length
            pages = Pages()
            pages.ensure(length - 1, -1)

            def table():
                result = torch.zeros(1, CONTEXT // 32, dtype=torch.int32)
                result[0, : len(pages.blocks)] = torch.tensor(pages.blocks, dtype=torch.int32)
                return result

            mode = plan["mode"]
            params = TTSamplingParams(
                temperature=[0.0 if mode == "greedy" else 0.8],
                top_k=[1 if mode == "greedy" else 8],
                top_p=[1.0],
                seed=[101],
                enable_log_probs=[False],
            )
            start = 0
            for end in plan["ends"]:
                first, delta = adapter.prefill_forward(
                    torch.tensor([prompt[:end]], dtype=torch.int32),
                    table(),
                    cache,
                    prompt_lens=[end],
                    empty_slots=[0],
                    start_pos=[start],
                    sampling_params=params,
                    enable_trace=True,
                )
                assert delta.tolist() == [0]
                start = end
            latest = int(first.reshape(-1)[0])
            phase = dict(
                **plan,
                prompt_token_ids=prompt,
                tokens=[latest],
                rows=[],
                allocations=pages.events,
                overlapped_submissions=0,
                memory_before=view(mesh),
            )
            report["phases"].append(phase)
            overlap = plan["reference"] is not None
            pending = None

            def retire(item):
                step, host, events = item
                for event in events:
                    ttnn.event_synchronize(event)
                value = int(adapter.process_decode_output_host(host, is_tokens=True).reshape(-1)[0])
                assert len(phase["tokens"]) == step + 1
                phase["tokens"].append(value)
                return value

            for step in range(plan["steps"]):
                position = length + step
                grew = pages.ensure(position, step)
                reload = not overlap or step == 0
                page_only = grew and not reload
                before = dict(g.counters)
                submitted = adapter.decode_forward(
                    tokens=torch.tensor([[latest if reload else 7777]], dtype=torch.int32),
                    start_pos=[position if reload else CONTEXT + 100],
                    page_table=table() if reload or page_only else torch.full((1, 1), -1, dtype=torch.int32),
                    kv_cache=cache,
                    sampling_params=params,
                    enable_trace=True,
                    read_from_device=False,
                    reload_inputs=reload,
                    reload_page_table=page_only,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                )
                host, events = adapter.read_decode_output(submitted, async_read=True)
                current = (step, host, events)
                delta = {k: g.counters[k] - before.get(k, 0) for k in g.counters}
                check_delta(delta, reload_inputs=reload, page_only=page_only, table_changed=grew)
                phase["rows"].append(
                    dict(
                        step=step,
                        consumed_position=position,
                        reload_inputs=reload,
                        page_only=page_only,
                        counter_delta=delta,
                    )
                )
                if overlap:
                    if pending is not None:
                        phase["overlapped_submissions"] += 1
                        latest = retire(pending)
                    pending = current
                else:
                    latest = retire(current)
            if pending is not None:
                latest = retire(pending)
            ttnn.synchronize_device(mesh)
            actual_positions = ttnn.to_torch(ttnn.get_device_tensors(g.tt_pos)[0])[:, 0].tolist()
            assert actual_positions == [length + plan["steps"]]
            assert (
                g.traces == baseline
                and g.counters["captures"] == counts["captures"]
                and g.counters["releases"] == counts.get("releases", 0)
            )
            assert g.counters["logits_readbacks"] == counts.get("logits_readbacks", 0)
            if overlap:
                control = references[plan["reference"]]
                assert phase["tokens"] == control["tokens"], (
                    "Token mismatch",
                    plan["name"],
                    phase["tokens"],
                    control["tokens"],
                )
                assert phase["overlapped_submissions"] == plan["steps"] - 1
                phase["exact_control"] = True
            if plan["steps"] == 66 and overlap:
                assert sum(row["page_only"] for row in phase["rows"]) >= 2
            phase.update(
                status="pass",
                final_positions=actual_positions,
                trace_ids=dict(g.traces),
                counters=dict(g.counters),
                memory_after=view(mesh),
            )
            references[plan["name"]] = phase
            persist()
        events = [json.loads(line) for line in event_path.read_text().splitlines()]
        report["execution_signatures"] = audit_events(events, baseline, report["phases"])
        assert len({p["memory_after"]["total_bytes_allocated_per_bank"] for p in report["phases"]}) == 1
        assert any(len(set(p["tokens"])) > 1 for p in report["phases"] if p["reference"])
        report.update(
            status="pass",
            final_counters=dict(g.counters),
            caveat="Reduced B1 adapter only; automatic scheduler/concurrency, full48 numerics and performance remain separate gates",
        )
    except BaseException:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        persist()
        try:
            if adapter is not None and adapter.generator is not None:
                adapter.release_persistent_capture()
                report["shutdown_counters"] = dict(adapter.generator.counters)
                assert report["shutdown_counters"]["captures"] == report["shutdown_counters"]["releases"]
        except BaseException:
            report.update(status="failed", cleanup_failure=traceback.format_exc())
            raise
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
                    report["closed_mesh"] = True
            except BaseException:
                report.update(status="failed", mesh_close_failure=traceback.format_exc())
                raise
            finally:
                persist()
    print("B1_TRACE_PROBE_PASS", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/b1_trace_probe.json")
    parser.add_argument("--trace-region-size", type=int, default=512000000)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    validation = cpu_check()
    if args.cpu_check:
        print(json.dumps(validation))
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for device execution")
    run(args)


if __name__ == "__main__":
    main()
