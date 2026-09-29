# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""C08 exact YUV output fusion/replay and inclusive conversion+D2H timing.

CPU: --prepare DIR --case tiny|crop|production. Inputs/oracles are separate files;
the broker collector loads only inputs. C08_CASE, C08_MODE=base|fused|traced,
C08_INPUTS and C08_RESULTS opt into collection. Verify off-device with
--verify RESULT.pt --inputs DIR [--baseline BASE.pt]. The traced route copies a
distinct source buffer into its captured input on every timed iteration, matching
the production tail boundary. Codec encoding is outside this measurement.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CASES = {
    "tiny": (1, 16, 32, 16, 32),
    "crop": (33, 128, 192, 126, 188),
    "production": (145, 1088, 1920, 1088, 1920),
}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _oracle(packed, height, width, out_h, out_w):
    """Independent pixel indexing, then FP32 BT.601 and 2x2 chroma before quantization."""
    frames = packed.shape[1]
    planar = torch.empty(frames, out_h * out_w * 3 // 2, dtype=torch.uint8)
    # Standard coefficients derived for [-1,1] RGB and limited-range BT.601.
    kr, kb = 0.299, 0.114
    kg = 1 - kr - kb
    coeffs = (
        (109.5 * kr, 109.5 * kg, 109.5 * kb, 125.5),
        (-56 * kr / (1 - kb), -56 * kg / (1 - kb), 56, 128),
        (56, -56 * kg / (1 - kr), -56 * kb / (1 - kr), 128),
    )
    for frame in range(frames):
        rgb = torch.empty(3, height, width, dtype=torch.float32)
        for channel in range(3):
            for row in range(4):
                for col in range(4):
                    # Checkpoint output channels are (c,p,r,q), p=1; H uses q, W uses r.
                    rgb[channel, row::4, col::4] = packed[0, frame, :, :, channel * 16 + col * 4 + row]
        rgb.clamp_(-1, 1)
        planes = []
        for index, (wr, wg, wb, offset) in enumerate(coeffs):
            values = wr * rgb[0] + wg * rgb[1] + wb * rgb[2] + offset
            if index:
                values = values.reshape(height // 2, 2, width // 2, 2).mean((1, 3))
                values = values[: out_h // 2, : out_w // 2]
            else:
                values = values[:out_h, :out_w]
            planes.append((values + 0.5).clamp(0, 255).to(torch.uint8).flatten())
        planar[frame] = torch.cat(planes)
    return planar


def _prepare(directory, case):
    frames, height, width, out_h, out_w = CASES[case]
    manifest_path = directory / f"{case}.json"
    assert not manifest_path.exists(), f"refusing to overwrite {manifest_path}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "case": case, "shape": list(CASES[case]), "inputs": {}, "references": {}}
    for name, seed in (("a", 17), ("b", 31)):
        input_path = directory / f"{case}-{name}.pt"
        reference_path = directory / f"{case}-{name}-reference.pt"
        assert not input_path.exists() and not reference_path.exists()
        # Binary fractions spanning [-4,4] exercise clipping before chroma
        # averaging. Different spatial/frame/channel values catch wrong axis order.
        packed = (
            torch.randint(
                -512,
                513,
                (1, frames, height // 4, width // 4, 48),
                dtype=torch.int16,
                generator=torch.Generator().manual_seed(seed),
            )
            .to(torch.bfloat16)
            .div_(128)
        )
        torch.save(packed, input_path)
        reference = _oracle(packed, height, width, out_h, out_w)
        torch.save(reference, reference_path)
        manifest["inputs"][name] = _hash_file(input_path)
        manifest["references"][name] = _hash_file(reference_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"C08_PREPARED {manifest_path}; production fixtures require several GB of host memory/disk")


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    params = {"l1_small_size": 32768, "trace_region_size": 64_000_000}
    metafunc.parametrize(
        "mesh_device,device_params",
        [
            pytest.param((4, 8), {**ring_params_8k_req_exact_devices, **params}, id="galaxy"),
            pytest.param((2, 4), {**line_params_req_exact_devices, **params}, id="f07"),
        ],
        indirect=["mesh_device", "device_params"],
    )


@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    "C08_INPUTS" not in os.environ and "C08_RESULTS" not in os.environ,
    reason="explicit C08 experiment; set C08_INPUTS and C08_RESULTS",
)
def test_collect_yuv_output_fusion(mesh_device, device_params):
    import ttnn
    from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder
    from models.tt_dit.utils.tensor import typed_tensor_2dshard
    from models.tt_dit.utils.tracing import Tracer
    from models.tt_dit.utils.yuv_d2h import fast_device_to_host_yuv, yuv_planes_to_host

    case, mode = os.environ["C08_CASE"], os.environ["C08_MODE"]
    assert case in CASES and mode in {"base", "fused", "traced"}
    profile = os.environ.get("C08_PROFILE", "0") == "1"
    if profile:
        assert mode != "traced", "collect eager device counters separately from trace timings"
    else:
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0"), "profile separately from timings"
    directory = Path(os.environ["C08_INPUTS"])
    manifest_path = directory / f"{case}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == 1 and manifest["shape"] == list(CASES[case])
    result_path = Path(os.environ["C08_RESULTS"]) / f"{case}-{mode}.pt"
    assert not result_path.exists(), f"refusing to overwrite {result_path}"
    frames, height, width, out_h, out_w = CASES[case]
    values = {}
    for name in ("a", "b"):
        path = directory / f"{case}-{name}.pt"
        assert _hash_file(path) == manifest["inputs"][name]
        values[name] = torch.load(path, map_location="cpu", weights_only=True)
    source = typed_tensor_2dshard(
        values["a"], mesh_device, shard_mapping={0: 2, 1: 3}, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    # Prepare both host sharded updates before capture; never allocate a new
    # persistent device tensor after a trace and assume it survives replay.
    host_updates = {
        name: ttnn.from_torch(
            value,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        for name, value in values.items()
    }
    context = SimpleNamespace(patch_size=4)
    tracer = Tracer(
        lambda x: LTXVideoDecoder._unpatch_yuv_device(context, x),
        device=mesh_device,
        prep_run=True,
        clone_prep_inputs=False,
    )
    trace_input = ttnn.clone(source) if mode == "traced" else None
    if trace_input is not None:
        assert trace_input.buffer_address() != source.buffer_address()
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    saved, hashes, warm_ms = {}, {}, []

    def fused(x):
        return LTXVideoDecoder._unpatch_yuv_device(context, x)

    def read_planes(planes):
        return yuv_planes_to_host(planes, mesh_device, logical_h=out_h, logical_w=out_w)

    def run(use_trace=False):
        if mode == "base":
            b, t, h, w, _ = tuple(source.shape)
            expanded = ttnn.reshape(source, (b, t, h, w, 3, 1, 4, 4))
            bcthw = ttnn.permute(expanded, (0, 4, 1, 5, 2, 7, 3, 6))
            bcthw = ttnn.reshape(bcthw, (b, 3, t, h * 4, w * 4))
            return fast_device_to_host_yuv(bcthw, mesh_device, logical_h=out_h, logical_w=out_w)
        return read_planes(tracer(source) if use_trace else fused(source))

    def record(label, value):
        hashes[label] = _hash_array(value)
        if label in ("eager_a", "eager_b"):
            saved[label] = torch.from_numpy(value.copy())

    try:
        for name in ("a", "b"):
            ttnn.copy_host_to_device_tensor(host_updates[name], source)
            output = run()
            if not capture_only:
                record(f"eager_{name}", output)
                if profile:
                    ttnn.ReadDeviceProfiler(mesh_device)
        if capture_only:
            if trace_input is not None:
                ttnn.copy(source, trace_input)  # Record the production replay's input-copy program as well.
            pytest.skip("kernel recipe capture only; no correctness/timing evidence")
        ttnn.copy_host_to_device_tensor(host_updates["a"], source)
        if profile:
            from tracy import signpost

            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            op0 = ttnn._ttnn.get_device_operation_id()
            signpost("start")
            run()
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            signpost("stop")
            print(f"C08_PROFILE_OPID_RANGE={op0},{ttnn._ttnn.get_device_operation_id()}")
        elif mode == "traced":
            # Distinct source→persistent trace input copy is included on every
            # replay, just like a fresh VAE conv_out tensor in production.
            ttnn.copy(source, trace_input)
            record("capture_a", read_planes(tracer(trace_input)))
        else:
            record("capture_a", run())
        if not profile:
            for label, name in (("a0", "a"), ("b", "b"), ("a1", "a")):
                ttnn.copy_host_to_device_tensor(host_updates[name], source)
                record(label, run(use_trace=mode == "traced"))
            repeats = int(os.environ.get("C08_REPEATS", "5"))
            assert repeats >= 5
            for index in range(repeats):
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                output = run(use_trace=mode == "traced")
                ttnn.synchronize_device(mesh_device)
                warm_ms.append((time.perf_counter() - start) * 1000)
                record(f"timed_a{index}", output)
        if mode == "traced":
            assert tracer.trace_captured
        from models.tt_dit.models.vae import vae_ltx
        from models.tt_dit.utils import yuv_d2h

        metadata = {
            "schema": 1,
            "case": case,
            "mode": mode,
            "profile": profile,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sources": {
                "harness": _hash_file(__file__),
                "vae": _hash_file(vae_ltx.__file__),
                "yuv_d2h": _hash_file(yuv_d2h.__file__),
            },
            "manifest_sha256": _hash_file(manifest_path),
            "mesh": list(mesh_device.shape),
            "warm_ms": warm_ms,
            "trace_input_copy_included": mode == "traced",
            "logical_rgb_bytes_all_chips": frames * height * width * 3 * 2,
            "logical_yuv_bytes_all_chips": frames * height * width * 3 // 2,
            "timing_scope": "resident conv_out input through unpatch/clip/YUV, optional trace input copy, async DMA and final cropped planar assembly; excludes VAE, codec, upload, hashing and warmup",
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "hashes": hashes, "saved": saved}, result_path)
        print(f"C08_CORRECTNESS_PENDING {result_path}")
    finally:
        tracer.release_trace()


def _verify(path, directory, baseline):
    verdict = path.with_suffix(".quality.json")
    verdict.unlink(missing_ok=True)
    result = torch.load(path, map_location="cpu", weights_only=True)
    meta, hashes = result["metadata"], result["hashes"]
    case = meta["case"]
    manifest_path = directory / f"{case}.json"
    manifest = json.loads(manifest_path.read_text())
    assert meta["schema"] == 1 and not meta["profile"] and meta["manifest_sha256"] == _hash_file(manifest_path)
    assert meta["mode"] in {"base", "fused", "traced"}
    assert meta["trace_input_copy_included"] == (meta["mode"] == "traced")
    times = meta["warm_ms"]
    assert len(times) >= 5 and all(math.isfinite(t) and t > 0 for t in times)
    required = {"eager_a", "eager_b", "capture_a", "a0", "b", "a1"} | {f"timed_a{i}" for i in range(len(times))}
    assert set(hashes) == required
    assert hashes["eager_a"] != hashes["eager_b"], "changed input does not affect output"
    for label in required:
        assert hashes[label] == hashes["eager_b" if label in ("eager_b", "b") else "eager_a"], label
    metrics = {}
    _, _, _, out_h, out_w = CASES[case]
    offsets = (0, out_h * out_w, out_h * out_w * 5 // 4, out_h * out_w * 3 // 2)
    for name in ("a", "b"):
        ref_path = directory / f"{case}-{name}-reference.pt"
        assert _hash_file(ref_path) == manifest["references"][name]
        ref = torch.load(ref_path, map_location="cpu", weights_only=True)
        actual = result["saved"][f"eager_{name}"]
        assert actual.dtype == torch.uint8 and actual.shape == ref.shape
        assert _hash_array(actual.numpy()) == hashes[f"eager_{name}"], "saved output/hash mismatch"
        error = (actual.to(torch.int16) - ref.to(torch.int16)).abs()
        maxima = [error[:, offsets[i] : offsets[i + 1]].max().item() for i in range(3)]
        assert maxima[0] <= 1 and all(v <= 2 for v in maxima[1:]), f"existing YUV CPU oracle gate failed: {maxima}"
        metrics[name] = {"max_uint8_error_Y_Cb_Cr": maxima}
    if meta["mode"] != "base":
        assert baseline is not None, "candidate requires exact baseline comparison"
        base = torch.load(baseline, map_location="cpu", weights_only=True)
        assert base["metadata"]["mode"] == "base"
        _verify(baseline, directory, None)
        for key in ("commit", "sources", "manifest_sha256", "mesh", "case"):
            assert base["metadata"][key] == meta[key], f"baseline {key} mismatch"
        assert all(
            base["hashes"][label] == value for label, value in hashes.items()
        ), "candidate differs from baseline bytes"
    report = {
        "quality_pass": True,
        "result_sha256": _hash_file(path),
        "metrics": metrics,
        "mean_warm_ms": sum(times) / len(times),
        "exact_baseline_parity": meta["mode"] != "base",
    }
    verdict.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C08_CORRECTNESS_PASS {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--case", choices=list(CASES))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.case is None:
            parser.error("--prepare requires --case")
        _prepare(args.prepare, args.case)
    else:
        if args.inputs is None:
            parser.error("--verify requires --inputs")
        _verify(args.verify, args.inputs, args.baseline)
