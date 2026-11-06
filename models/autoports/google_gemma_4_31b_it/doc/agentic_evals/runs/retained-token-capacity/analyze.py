"""Host-only serialization calibration; no models or devices imported."""
import copy
import json
import statistics
from pathlib import Path

SOURCES = {
    "B8_live_pair": "bringup/artifacts/model-token-retention/live-pair/candidate/phases.jsonl",
    "B1_benchmarks": "bringup/artifacts/stage12-retained-runtime-benchmark/phase-events.json",
}
measurements = {}
for label, path in SOURCES.items():
    p = Path(path)
    frames = [json.loads(line) for line in p.read_text().splitlines()]
    events = [e for f in frames for e in f.get("events", [])]
    decode = [e for e in events if e.get("phase") == "decode"]
    complete = [e for e in decode if e["event"] == "complete"]
    sizes = {
        kind: [len(json.dumps(e).encode()) + 2 for e in decode if e["event"] == kind] for kind in ["submit", "complete"]
    }
    gaps = [
        (b["monotonic_ns"] - a["monotonic_ns"]) / 1e9
        for a, b in zip(complete, complete[1:])
        if b["monotonic_ns"] > a["monotonic_ns"]
    ]
    times = sorted(e["monotonic_ns"] / 1e9 for e in complete)
    peaks = {}
    for window in (1, 5, 10):
        left = 0
        peak = 0
        for right, t in enumerate(times):
            while times[left] < t - window:
                left += 1
            peak = max(peak, right - left + 1)
        peaks[str(window)] = peak / window
    overheads = [len(json.dumps(dict(f, events=[])).encode()) + 1 for f in frames]
    measurements[label] = dict(
        source=path,
        file_bytes=p.stat().st_size,
        frames=len(frames),
        events=len(events),
        decode_steps=len(complete),
        event_size_bytes={k: dict(min=min(v), mean=statistics.mean(v), max=max(v)) for k, v in sizes.items()},
        frame_empty_bytes=dict(min=min(overheads), mean=statistics.mean(overheads), max=max(overheads)),
        median_completion_gap_seconds=statistics.median(gaps),
        median_gap_rate_steps_s=1 / statistics.median(gaps),
        peak_sliding_rates_steps_s=peaks,
    )
    if label == "B8_live_pair":
        templates = {kind: next(e for e in decode if e["event"] == kind) for kind in ("submit", "complete")}
# Conservative serialized event envelope: 19-digit clocks, 8-digit step IDs,
# 6-digit active positions and every physical sample ID, full 8-entry remap.
# This is an explicit scenario bound, not a bound for arbitrary metadata.
cases = {}
for name, clients, cap in [("SWE_C2_B8", 2, 32768), ("GPQA_C4_B8", 4, 126976)]:
    serial = {}
    for kind, template in templates.items():
        e = copy.deepcopy(template)
        e.update(
            step_id=99999999,
            submit_ns=2**63 - 1,
            monotonic_ns=2**63 - 1,
            positions=[262143] * clients + [-1] * (8 - clients),
            effective_positions=[262143] * clients + [-1] * (8 - clients),
            active_rows=clients,
            reload_inputs=False,
            reload_page_table=False,
            slot_remap=list(range(8)),
        )
        if kind == "complete":
            e["sampled_decode_token_ids"] = [262143] * 8
        serial[kind] = len(json.dumps(e).encode()) + 2
    step_bytes = sum(serial.values())
    rate = measurements["B8_live_pair"]["median_gap_rate_steps_s"]
    # Use full cap steps (one more than cap-minus-prefill native decode) conservatively.
    steps = cap
    frames = steps / rate
    estimated = steps * step_bytes + frames * 4096
    scenarios = {}
    for r in [rate, 40, 60, 80, 100]:
        total = 43200 * (r * step_bytes + 4096)
        scenarios[str(r)] = dict(
            steps_per_second=r, bytes_12h=total, gib_12h=total / 2**30, fits_default_4GiB=total < 4 * 2**30 - 2048
        )
    threshold = ((4 * 2**30 - 2048) / 43200 - 4096) / step_bytes
    cases[name] = dict(
        clients=clients,
        physical_batch=8,
        output_cap=cap,
        conservative_decode_steps=steps,
        event_envelope_bytes=serial,
        bytes_per_step=step_bytes,
        frame_overhead_allowance_bytes_per_s=4096,
        call_bytes_at_measured_B8_step_rate=estimated,
        call_mib=estimated / 2**20,
        full_buffer_step_capacity=1000000 // 2,
        automatic_export_event_capacity_per_s=1024,
        producer_events_s_at_measured_rate=rate * 2,
        remaining_file_required_before_call_bytes=estimated + 2048,
        estimated_prior_consumption_allowed_bytes=4 * 2**30 - estimated - 2048,
        maximum_sustained_step_rate_for_12h_4GiB=threshold,
        run_scenarios=scenarios,
    )
result = dict(
    measurements=measurements,
    estimates=cases,
    defaults=dict(
        max_events=1000000,
        export_chunk_events=1024,
        interval_seconds=1,
        max_file_bytes=4 * 2**30,
        reserved_terminal_bytes=2048,
        max_pending_steps=64,
    ),
    assumptions=[
        "C2 and C4 served on B8, token array includes all8 padded rows",
        "Two events per decode step; all simultaneous clients share that step",
        "4096bytes/s frame allowance exceeds measured empty-frame maxima; prefill event bytes additional",
        "Continuous decode scenarios include earlier decode calls; they are not an unconditional12h bound",
        "Serialized sizes are not Python heap sizes; memory cannot be inferred from JSONbytes",
    ],
)
print(json.dumps(result, indent=2))
