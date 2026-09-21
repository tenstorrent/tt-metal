"""Matched-input accuracy sweep for the six final v3 implementations.

Run only under compute-sprint-v1/run_locked.sh. No performance is inferred here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V3 = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOW = load("pareto_low", V3 / "compensated/benchmark.py")
CANON = LOW.CANONICAL
REPRO = LOW.REPRO
NUM = load("pareto_metrics", V3 / "numerics.py")
GROUPED = V3 / "compensated/group2_valid/compute_streaming.hpp"


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def pins():
    result = LOW.source_hashes(GROUPED)
    paths = [Path(__file__), V3 / "review/compute_valid.cpp"]
    for folder in (V3 / "fp32", ROOT / "experiments/sdpa-l2/compute-sprint-v1/bf16"):
        paths += [p for p in folder.iterdir() if p.suffix in (".cpp", ".hpp") and not p.name.startswith("._")]
    result.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result


def run_variant(device, inputs, variant, k_length):
    prepared = [CANON.prepare(device, x, variant, is_q=i == 0, cores=1) for i, x in enumerate(inputs)]
    before = [digest(ttnn.to_torch(x)) for x in prepared]
    original = ttnn.KernelDescriptor
    descriptors = []

    def descriptor(*args, **kwargs):
        if kwargs.get("kernel_source", "").endswith("/compute.cpp"):
            if variant in "CD":
                flags = ["SDPA_SPRINT_IDENTITY4", "SDPA_SPRINT_STATE_UNPACK_BATCH",
                         "SDPA_SPRINT_LAZY_INIT", "SDPA_SPRINT_MAX_SCAN",
                         "SDPA_V3_L1_INPLACE", "SDPA_V3_L1_BOTH"]
                flags += ["SDPA_V3_O2"] if variant == "D" else ["SDPA_SPRINT_C_REFINE", "SDPA_SPRINT_C_REFINE_HOIST"]
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v3/fp32/compute_early.cpp"
            elif variant == "B":
                flags = ["SDPA_BF16_BLOCK_STATE", "SDPA_BF16_CORRECTION_REUSE",
                         "SDPA_BF16_CORRECTION_FENCE", "SDPA_REVIEW_GROUP2"]
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v3/review/compute_valid.cpp"
            else:
                flags = []
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v1/bf16/fullchip_compute.cpp"
            kwargs["defines"] = list(kwargs["defines"]) + [(flag, "1") for flag in flags]
        descriptors.append({k: kwargs.get(k) for k in ("kernel_source", "defines", "compile_time_args")})
        return original(*args, **kwargs)

    trace = None
    out = traced_out = None
    try:
        if variant in "EG":
            out, invoke, metadata = LOW.build(device, prepared, variant, resident=False,
                jobs=1, k_chunks=k_length // 512, candidate=GROUPED)
            invoke()
            def capture_call():
                invoke()
                return out
        else:
            ttnn.KernelDescriptor = descriptor
            def capture_call():
                return CANON.attention(device, *prepared, variant, max_cores=1,
                                       q_chunk_size=256, k_chunk_size=512, reader_barrier_tiles=2)
            out = capture_call()
            metadata = {"descriptors": descriptors}
        actual = ttnn.to_torch(out)
        assert torch.isfinite(actual).all()
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        traced_out = capture_call()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(ttnn.to_torch(traced_out)) == digest(actual)
        assert [digest(ttnn.to_torch(x)) for x in prepared] == before
        return actual, metadata, before
    finally:
        ttnn.KernelDescriptor = original
        if trace is not None:
            ttnn.release_trace(device, trace)
        if traced_out is not None and traced_out is not out:
            ttnn.deallocate(traced_out)
        if out is not None:
            ttnn.deallocate(out)
        if variant in "EG":
            for x in prepared:
                ttnn.deallocate(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh result filename"
    torch.set_num_threads(8)
    cases = [("core", n, dist) for n in (4096, 32768, 262144)
             for dist in ("normal", "clipped", "scaled_down", "scaled_qk", "outliers", "uniform")]
    cases += [("stress", 32768, dist) for dist in ("common_q", "common_k", "common_v")]
    pinned = pins()
    records = []
    report = {"seed": 20260919, "query_rows": 256, "heads": 1, "dim": 128,
              "q_chunk": 256, "k_chunk": 512, "noncausal": True, "source_sha256": pinned,
              "note": "Matched original BF16 inputs and FP64 reference across all six variants; accuracy only, separate resident throughput.",
              "records": records, "complete": False}
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for suite, n, dist in cases:
            tensors = REPRO.make_inputs(1, 256, n, 128, report["seed"],
                                        "normal" if dist in ("clipped", "scaled_down") else dist)
            if dist == "clipped":
                tensors = [x.clamp(-2, 2) for x in tensors]
            elif dist == "scaled_down":
                tensors[:2] = [(x.float() * 0.25).bfloat16() for x in tensors[:2]]
            hashes = [digest(x) for x in tensors]
            reference = REPRO.reference(*tensors)
            inputs = [ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for x in tensors]
            assert [digest(ttnn.to_torch(x)) for x in inputs] == hashes
            for variant in "DCBAEG":
                actual, metadata, prepared_hashes = run_variant(device, inputs, variant, n)
                assert [digest(ttnn.to_torch(x)) for x in inputs] == hashes
                records.append(dict(variant=variant, suite=suite, distribution=dist, k_length=n,
                    metrics=NUM.metrics(actual, reference), input_sha256=hashes,
                    prepared_sha256=prepared_hashes, output_sha256=digest(actual),
                    raw_trace_equal=True, actual_replays=2, inputs_immutable=True, metadata=metadata))
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(variant, suite, n, dist, records[-1]["metrics"]["l2_pct"], flush=True)
            for x in inputs:
                ttnn.deallocate(x)
        assert pins() == pinned
        report.update(complete=True, selected_sources_immutable=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
