"""Fresh unchanged-A resident control; run only through the shared device lock."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = load("v3_unchanged_a", ROOT / "experiments/sdpa-l2/compute-sprint-v2/review/bench.py")
NUMERICS = load("v3_a_numerics", HERE / "numerics.py")


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    target = HERE / (args.label + ".json")
    assert not target.exists(), "Use a fresh evidence label"
    torch.set_num_threads(8)
    inputs = BENCH.REPRO.make_inputs(1, 256, 512, 128, 1240, "normal")
    hashes = [digest(x) for x in inputs]
    reference = BENCH.REPRO.reference(*inputs)
    build_args = argparse.Namespace(variant="A", q_repeats=8, k_chunks=512, distinct_kv=False)
    sources = [Path(__file__).resolve(), Path(NUMERICS.__file__), Path(BENCH.__file__),
               Path(BENCH.CANON.__file__), Path(BENCH.REPRO.__file__)]
    for directory in ("compute-sprint-v1/bf16", "bfp4-lofi-v2/resident", "single-core-resident-v1/main"):
        sources.extend(p for p in (ROOT / "experiments/sdpa-l2" / directory).rglob("*")
                       if p.suffix in (".cpp", ".h", ".hpp"))
    def source_hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    pinned = source_hashes()
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        tensors = [ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                   for x in inputs]
        assert [digest(ttnn.to_torch(x)) for x in tensors] == hashes
        out, invoke, metadata = BENCH.build(device, build_args, tensors, "canonical")
        invoke()
        eager = ttnn.to_torch(out)
        assert torch.isfinite(eager).all()
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        invoke()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        timings = []
        try:
            for iteration in range(14):
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                elapsed = 1000 * (time.perf_counter() - start)
                if iteration >= 5:
                    timings.append(elapsed)
                if iteration < 2:
                    assert digest(ttnn.to_torch(out)) == digest(eager)
            assert digest(ttnn.to_torch(out)) == digest(eager)
        finally:
            ttnn.release_trace(device, trace)
        assert [digest(ttnn.to_torch(x)) for x in tensors] == hashes
        assert source_hashes() == pinned
        flops = 4 * 256 * 512 * 128 * 8 * 512
        median = statistics.median(timings)
        result = dict(variant="A", implementation="unchanged frozen v1 canonical A",
                      q_chunk=256, k_chunk=512, dim=128, cores=1, q_repeats=8, k_chunks=512,
                      useful_flops=flops, median_ms=median, tflops_per_core=flops / (median * 1e9),
                      replay_ms=timings, actual_replays=14, trace_raw_equal=True,
                      metadata=metadata, input_hashes=hashes, device_inputs_immutable=True,
                      selected_sources_immutable=True, selected_source_sha256=pinned,
                      metrics=NUMERICS.metrics(eager, reference), output_sha256=digest(eager),
                      note="Fresh control only, not an optimization or paired speedup claim; resident repeated KV, preprocessing excluded.")
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
