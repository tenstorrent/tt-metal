"""Independent B qualification: frozen fullchip dataflow and original FP64 reference."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANON = load("v3_review_canonical", "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py")
REPRO = load("v3_review_reference", "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
NUMERICS = load("v3_review_numerics", "experiments/sdpa-l2/compute-sprint-v3/numerics.py")
FLAGS = {"SDPA_BF16_BLOCK_STATE": "1", "SDPA_BF16_CORRECTION_REUSE": "1",
         "SDPA_BF16_CORRECTION_FENCE": "1"}


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def make_inputs(args, distribution):
    custom = distribution in ("growing_max", "identity_transitions", "repeated_kv", "zero_v",
                              "constant_v_scaled", "uniform_constant_v_scaled", "common_qk")
    q, k, v = REPRO.make_inputs(1, args.q_length, args.k_length, 128, args.seed,
                              "normal" if custom else distribution)
    chunks = args.k_length // 512
    if distribution == "growing_max":
        factors = torch.linspace(0.25, 4, chunks).repeat_interleave(512)
        k = (k.float() * factors.reshape(1, 1, args.k_length, 1)).bfloat16()
    elif distribution == "identity_transitions":
        assert chunks <= 16
        factors = (2.0 ** (torch.arange(chunks) // 2)).repeat_interleave(512)
        k = (k[:, :, :512, :].repeat(1, 1, chunks, 1).float() *
             factors.reshape(1, 1, args.k_length, 1)).bfloat16()
    elif distribution == "repeated_kv":
        k = k[:, :, :512, :].repeat(1, 1, chunks, 1)
        v = v[:, :, :512, :].repeat(1, 1, chunks, 1)
    elif distribution == "zero_v":
        v.zero_()
    elif distribution in ("constant_v_scaled", "uniform_constant_v_scaled"):
        v.fill_(3.25)
        if distribution == "uniform_constant_v_scaled":
            q.zero_()
    elif distribution == "common_qk":
        q = (q.float() + 32.0).bfloat16()
        k = (k.float() + 32.0).bfloat16()
    return q, k, v


def source_pins():
    folders = [HERE, ROOT / "experiments/sdpa-l2/compute-sprint-v3/compensated/group2",
               ROOT / "experiments/sdpa-l2/compute-sprint-v1/bf16",
               ROOT / "experiments/sdpa-l2/compute-sprint-v1/lowp",
               ROOT / "experiments/sdpa-l2/compute-sprint-v2/compensated/identity",
               ROOT / "experiments/sdpa-l2/compute-sprint-v2/compensated/identity_early",
               ROOT / "experiments/sdpa-l2/bf16-denom-pair-v3/candidate"]
    paths = {path for folder in folders for path in folder.rglob("*")
             if path.suffix in (".py", ".cpp", ".hpp", ".h")}
    paths |= {ROOT / path for path in (
        "experiments/sdpa-l2/compute-sprint-v3/numerics.py",
        "experiments/sdpa-l2/compute-sprint-v2/review/bench.py",
        "experiments/sdpa-l2/compute-sprint-v2/review/compute.cpp",
        "experiments/sdpa-l2/compute-sprint-v2/review/resident.cpp",
        "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
        "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/reader.cpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/reader_chain.cpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/writer.cpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/resident/reader.cpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/resident/writer.cpp",
        "experiments/sdpa-l2/hybrid-mixed-v1/reader_distinct.cpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/fast_correction.hpp",
        "experiments/sdpa-l2/bfp4-lofi-v2/exp_refiner.hpp",
        "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h")}
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(args):
    torch.set_num_threads(8)
    output = HERE / (args.label + ".json")
    assert not output.exists()
    assert args.q_length % 256 == args.k_length % 512 == 0
    original = ttnn.KernelDescriptor
    mode, descriptors = "v1", []
    numeric = CANON.recipe("B")[2]
    numeric["SDPA_K_CHUNK_TILES"] = "16"
    def construct(*pos, **kwargs):
        if kwargs["kernel_source"].endswith("/compute.cpp"):
            assert dict(kwargs["defines"]) == numeric
            kwargs["defines"] = list(kwargs["defines"]) + list(FLAGS.items())
            if mode == "v1":
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v1/bf16/fullchip_compute.cpp"
            elif mode == "v2_early":
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v2/review/compute.cpp"
                kwargs["defines"].append(("SDPA_REVIEW_IDENTITY_EARLY", "1"))
            else:
                kwargs["kernel_source"] = "experiments/sdpa-l2/compute-sprint-v3/review/compute.cpp"
                kwargs["defines"].append(("SDPA_REVIEW_GROUP2", "1"))
        descriptors.append({key: kwargs.get(key) for key in ("kernel_source", "defines", "compile_time_args")})
        return original(*pos, **kwargs)
    records = []
    pins = source_pins()
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        for distribution in args.distributions.split(","):
            q, k, v = make_inputs(args, distribution)
            input_sha = [digest(x) for x in (q, k, v)]
            reference = REPRO.reference(q, k, v)
            inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
            actuals = {}
            for mode in ("v1", "v2_early", "group2"):
                descriptors.clear()
                ttnn.KernelDescriptor = construct
                try:
                    def invoke():
                        return CANON.attention(device, *inputs, "B", max_cores=args.cores,
                                               q_chunk_size=256, k_chunk_size=512, reader_barrier_tiles=2)
                    out = invoke()
                    actual = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                    trace = ttnn.begin_trace_capture(device, cq_id=0)
                    traced_out = invoke()
                    ttnn.end_trace_capture(device, trace, cq_id=0)
                    replay_equal = []
                    try:
                        for _ in range(2):
                            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                            replay = ttnn.to_torch(traced_out)
                            replay_equal.append(torch.equal(actual.view(torch.uint16), replay.view(torch.uint16)))
                    finally:
                        ttnn.release_trace(device, trace)
                        ttnn.deallocate(traced_out)
                finally:
                    ttnn.KernelDescriptor = original
                actuals[mode] = actual.clone()
                record = dict(distribution=distribution, mode=mode, input_sha256=input_sha,
                              output_sha256=digest(actual), trace_raw_equal=replay_equal,
                              metrics=NUMERICS.metrics(actual, reference), descriptors=list(descriptors))
                if mode != "v1":
                    record["vs_v1"] = NUMERICS.compare(actuals["v1"], actual, reference)
                    record["raw_equal_v1"] = torch.equal(actuals["v1"].view(torch.uint16), actual.view(torch.uint16))
                if mode == "group2":
                    record["vs_v2_early"] = NUMERICS.compare(actuals["v2_early"], actual, reference)
                if distribution == "common_v":
                    centered_ref = reference.double() - 32.0
                    centered_norm = centered_ref.norm().item()
                    record["common_v_diagnostic"] = dict(
                        note="Diagnostic only; the original-reference acceptance gate is unchanged.",
                        removed_common_mode=32.0,
                        metrics_after_common_mode_removal=NUMERICS.metrics(actual.double() - 32.0, centered_ref),
                        candidate_minus_v1_l2_pct_of_centered_reference=(
                            100 * (actual.double() - actuals["v1"].double()).norm().item() / centered_norm
                            if centered_norm else None))
                records.append(record)
                report = dict(arguments=vars(args), results=records, source_sha256=pins,
                              numerical_defines=numeric, implementation_defines=FLAGS,
                              note="Numerical rejection is recorded without failing a cleanly closed device job.")
                output.write_text(json.dumps(report, indent=2) + "\n")
                print("RESULT " + json.dumps({k:v for k,v in record.items() if k != "descriptors"}), flush=True)
                assert all(replay_equal), "Nondeterministic eager/trace output bits"
                if mode == "v2_early":
                    assert record["raw_equal_v1"], "Frozen controls disagree"
            for tensor, expected in zip(inputs, input_sha):
                assert digest(ttnn.to_torch(tensor)) == expected, "Input was mutated"
                ttnn.deallocate(tensor)
        assert pins == source_pins(), "Source changed during execution"
    finally:
        ttnn.KernelDescriptor = original
        ttnn.close_device(device)
    report["completed"] = True
    report["all_numerical_gates_pass"] = all(r["vs_v1"]["acceptance"]["pass"] and
                                             r["vs_v2_early"]["acceptance"]["pass"]
                                             for r in records if r["mode"] == "group2")
    report["input_immutability_pass"] = True
    report["source_immutability_pass"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--q-length", type=int, default=1024)
    p.add_argument("--k-length", type=int, default=1536)
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--distributions", default="normal,constant_v,common_v,growing_max,identity_transitions")
    run(p.parse_args())
