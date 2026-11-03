"""Validate and collect buffered full-phase benchmark evidence on a live server."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from benchmark_work import accounting


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--action", choices=["check", "collect"], required=True)
    a = p.parse_args()
    out, n = a.run_dir, a.concurrency
    identity_path = out / f"perf-b{n}-server.json"
    server = json.loads(identity_path.read_text())
    phase_path = out / f"phase-b{n}.jsonl"
    init = rows(phase_path)[0]
    if init["max_num_seqs"] != n or init["layers"] != list(range(48)):
        raise ValueError("incorrect instrumented model identity")
    runtime = json.loads(phase_path.with_suffix(".runtime.json").read_text())
    proof = json.loads(
        (Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_default_performance_proof.json").read_text()
    )["runtime"]
    if runtime != proof:
        raise ValueError("current constructed tensors differ from accounting construction proof")
    if a.action == "check":
        accounting([dict(tokens=512, start_pos=0, head_tokens=1)], [dict(positions=[512], batch_size=1)], n)
        before = time.perf_counter()
        reply = requests.post(
            server["base_url"] + "/v1/completions",
            json=dict(
                model=server["model"],
                prompt=[1000 + i % 100 for i in range(512)],
                max_tokens=3,
                temperature=0,
                ignore_eos=True,
            ),
            timeout=300,
        )
        reply.raise_for_status()
        (out / f"roofline-b{n}-probe.json").write_text(json.dumps(reply.json(), indent=2) + "\n")
        time.sleep(2)
        observed = [row for row in rows(phase_path)[1:] if row["start"] >= before]
        if {r["phase"] for r in observed} != {"prefill", "decode"}:
            raise ValueError("collector smoke did not observe both asynchronous completion phases")
        print(json.dumps(dict(ready=True, observed_steps=len(observed), server_pid=init["pid"])))
        return
    perf_path = out / f"perf-b{n}.json"
    perf = json.loads(perf_path.read_text())
    time.sleep(2)
    all_rows = rows(phase_path)[1:]
    earliest = min(perf["start_times"])
    # Both processes share Linux CLOCK_MONOTONIC/perf_counter. The CLI initial
    # probe has completed before these detailed request starts. Requests are
    # bound by unique server IDs starting after the measured client boundary.
    selected_ids = {
        rid for row in all_rows if row["phase"] == "prefill" and row["start"] >= earliest for rid in row["request_ids"]
    }
    if len(selected_ids) != perf["completed"]:
        raise ValueError(f"request mapping differs: {len(selected_ids)} != {perf['completed']}")
    selected = [r for r in all_rows if set(r["request_ids"]) & selected_ids]
    if any(set(r["request_ids"]) - selected_ids for r in selected):
        raise ValueError("mixed measured/unmeasured physical batch")
    prefill = [
        dict(tokens=x["token_count"], start_pos=x["start_pos"], head_tokens=x["head_token_count"])
        for row in selected
        if row["phase"] == "prefill"
        for x in row["rows"]
    ]
    if sum(r["tokens"] for r in prefill) != perf["total_input_tokens"]:
        raise ValueError("prefill work disagrees with actual input tokens")
    decode = [dict(positions=r["positions"], batch_size=r["batch"]) for r in selected if r["phase"] == "decode"]
    steps_by_id = {
        rid: sum(rid in r["request_ids"] for r in selected if r["phase"] == "decode") for rid in selected_ids
    }
    if any(value < 127 for value in steps_by_id.values()):
        raise ValueError("missing completed decode work")
    seconds = {"prefill": 0.0, "decode": 0.0}
    previous = selected[0]["start"]
    for row in selected:
        if row["completion"] < previous:
            raise ValueError("nonmonotonic completion timeline")
        seconds[row["phase"]] += row["completion"] - previous
        previous = row["completion"]
    work = accounting(prefill, decode, n)
    evidence = out / f"roofline-b{n}-accounting.json"
    evidence.write_text(
        json.dumps(
            dict(
                work=work,
                seconds=seconds,
                request_ids=sorted(selected_ids),
                request_decode_steps=steps_by_id,
                phase_events=selected,
                raw_phase_sha256=sha(phase_path),
                performance_start_times=perf["start_times"],
            ),
            indent=2,
        )
        + "\n"
    )
    common = dict(
        timing_scope="full_phase_wall_time",
        timing_method="Ordered CQ completion/readback observations plus host sampling completion; disjoint intervals from first submit through final completion, including host gaps, all chunks, ops and communication; 0.5ms nonblocking event polling, buffered writes. No device synchronization/profiler added.",
        evidence=evidence.name,
        evidence_sha256=sha(evidence),
    )
    from benchmark_stage.subsets import digest

    entry = dict(
        performance_sha256=sha(perf_path),
        server_identity_sha256=digest(server),
        requests=perf["completed"],
        input_tokens=perf["total_input_tokens"],
        output_tokens=perf["total_output_tokens"],
    )
    entry["prefill"] = dict(
        common,
        flops=work["prefill_useful_flops"],
        seconds=seconds["prefill"],
        peak_flops_per_second=work["hardware_reference"]["mesh_prefill_effective_peak_flops_per_second"],
        work_method="Useful full48 TP4 active-expert projection + causal attention + terminal FLOPs from each actual chunk; fidelity-weighted nominal ceiling (LoFi projections/head, HiFi2 attention), calculation retained in accounting.",
        peak_source=work["hardware_reference"]["compute_source"],
    )
    entry["decode"] = dict(
        common,
        dram_bytes=work["decode_dram_bytes_estimate"],
        seconds=seconds["decode"],
        peak_dram_bytes_per_second=work["hardware_reference"]["mesh_dram_peak_bytes_per_second"],
        work_method=work["decode_dram_bytes_estimate_kind"]
        + "; upper expert-union scenario and all components retained in evidence; full actual batch/positions, tile dtype overhead, KV and material traffic.",
        peak_source="https://docs.tenstorrent.com/aibs/blackhole/p300.html : two cards x1024 GB/s",
    )
    roofline = out / "roofline.json"
    result = json.loads(roofline.read_text()) if roofline.exists() else {}
    result[str(n)] = entry
    roofline.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(seconds=seconds, work=work)))


if __name__ == "__main__":
    main()
