# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Collect C04 SDPA evidence at pinned-source per-chip geometry, then verify off-device.

The pipeline uses encoder TP=mesh.shape[1], independently of DiT TP. Galaxy TP8
has Q2/KV1; f07 TP4 has Q4/KV2. Both use B1/S1024/D256, chunks128, HiFi2,
fp32 accumulation. This isolates repeat_interleave+SDPA on one chip; it is not
an end-to-end encoder benchmark or proof of a pipeline latency improvement.

Prepare/verify commands run on the host OUTSIDE the device reservation:
  python -m models.tt_dit.tests.encoders.gemma.test_gemma_native_gqa --prepare /tmp/gqa/fixtures
  LTX_GEMMA_GQA_FIXTURES=/tmp/gqa/fixtures LTX_GEMMA_GQA_RESULTS=/tmp/gqa/results \
    pytest <this-file> -k 'tp8 and leftpad' -s  # through the device broker
  python -m models.tt_dit.tests.encoders.gemma.test_gemma_native_gqa \
    --verify /tmp/gqa/results --fixtures /tmp/gqa/fixtures \
    --manifest /tmp/gqa/manifest.json --execution /tmp/gqa/c04-execution.json

The pytest collection step deliberately emits GQA_CORRECTNESS_PENDING. Only the
offline verifier emits GQA_CORRECTNESS_PASS; collection success is not a gate.
Use a fresh results directory per revision/run. Full Gemma and changed-prompt
pipeline replay/quality remain required before enabling the production flag.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import torch

from models.tt_dit.tests.encoders.gemma.gqa_requalification import (
    ALL_CASES,
    MODES,
    SOURCE_PATHS,
    Recipe,
    digest,
    prepare_fixture,
    require,
    strict_json,
    verify_suite,
)

# Preserve the four original cases; signed controls augment them.
CASES = [(heads, mask) for heads in (2, 4) for mask in ("causal", "leftpad")]


def _case_name(heads, mask):
    return f"tp{16 // heads}-{mask}"


def _sha256(path):
    return digest(path)


def _prepare(directory):
    """BF16 operands, unchanged FP32 oracle, and independent FP64 dense oracle."""
    directory.mkdir(parents=True, exist_ok=False)
    for name in ALL_CASES:
        fixture = prepare_fixture(Recipe(name))
        path = directory / f"{name}.pt"
        with path.open("xb") as stream:
            torch.save(fixture, stream)
        print(f"GQA_FIXTURE {path} sha256={_sha256(path)}")


def _tt(host, device=None):
    import ttnn

    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _read(device_tensor):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(device_tensor)[0]).float()


@pytest.mark.parametrize("name", ALL_CASES)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    "LTX_GEMMA_GQA_FIXTURES" not in os.environ and "LTX_GEMMA_GQA_RESULTS" not in os.environ,
    reason="explicit GQA evidence collection; set both fixture and result directories",
)
def test_collect_gemma_native_gqa(mesh_device, name):
    import ttnn

    fixture_dir = Path(os.environ["LTX_GEMMA_GQA_FIXTURES"])
    # Capture the diagnostic selector once, before constructing configs/traces.
    # Later environment changes cannot mutate or relabel this recipe.
    recipe = Recipe(name, os.environ.get("GQA_EXP_MODE", "accurate"))
    require(os.environ.get("GQA_DIAGNOSTIC_HIFI4", "0") == "0", "C04 requalification requires production HiFi2")
    result_dir = Path(os.environ["LTX_GEMMA_GQA_RESULTS"]) / recipe.exp_mode
    result_path = result_dir / f"{name}.pt"
    assert not result_path.exists(), f"use a fresh results directory; refusing to replace {result_path}"
    fixture_path = fixture_dir / f"{name}.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    require(fixture["case"] == name and fixture["geometry"] == Recipe(name).identity(), "fixture geometry")
    inputs = fixture["cases"]
    batches = int(os.environ.get("GQA_TIMING_BATCHES", "5"))
    replays = int(os.environ.get("GQA_REPLAYS_PER_BATCH", "25"))
    assert batches > 0 and replays > 0
    result_dir.mkdir(parents=True, exist_ok=True)

    # All persistent inputs are allocated before either trace capture. Updates
    # below copy to the same addresses, including the changed padding mask.
    persistent = {key: _tt(inputs[0][key], mesh_device) for key in ("q", "k", "v")}
    if inputs[0]["mask"] is not None:
        persistent["mask"] = _tt(inputs[0]["mask"], mesh_device)
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=recipe.exp_mode == "approximate",
    )
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def forward(native):
        q, k, v = (persistent[key] for key in ("q", "k", "v"))
        if not native:
            k = ttnn.repeat_interleave(k, 2, dim=1)
            v = ttnn.repeat_interleave(v, 2, dim=1)
        q, k, v = (ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) for x in (q, k, v))
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=recipe.causal,
            attn_mask=persistent.get("mask"),
            scale=recipe.scale,
            program_config=program,
            compute_kernel_config=compute,
        )

    if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY") == "1":
        for native in (False, True):
            forward(native)
        pytest.skip("kernel recipe capture only; no correctness or timing result")

    provenance_path = os.environ.get("LTX_ACCEPTANCE_PROVENANCE")
    provenance = strict_json(Path(provenance_path).read_text()) if provenance_path else None
    # The native owner supplies request_provenance(task38_manifest). Collecting
    # without it remains useful for debugging, but cannot qualify offline.
    binaries = {} if provenance is None else provenance["build"]["binary_hashes"]
    request_id = str(uuid.uuid4())
    traces, outputs = {}, {}
    result = {
        "status": "collected",
        "case": name,
        "recipe": recipe.identity(),
        "request_id": request_id,
        "samples": [{"request_id": request_id, "index": i, "seed": seed} for i, seed in enumerate((11, 29, 11))],
        "provenance": provenance,
        "native_binary_sha256": {path: _sha256(Path(path)) for path in binaries},
        "fixture_sha256": _sha256(fixture_path),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path: _sha256(Path(path)) for path in SOURCE_PATHS},
        "tracked_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if (diff := subprocess.check_output(["git", "diff", "HEAD"])) else None
        ),
        "arch": str(mesh_device.arch()),
        "mesh_shape": list(mesh_device.shape),
        "timing_boundary": "synchronized trace replay of KV-repeat+SDPA versus native GQA SDPA",
        "profiler_env": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "replays_per_batch": replays,
        "timing_samples_us": {"expanded": [], "native": []},
        "replay_outputs": [],
        "eager_outputs": {},
    }
    try:
        for route, native in (("expanded", False), ("native", True)):
            eager = forward(native)  # warm compilation, and save an eager reference
            ttnn.synchronize_device(mesh_device)
            result["eager_outputs"][route] = _read(eager)
            ttnn.deallocate(eager)
            traces[route] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs[route] = forward(native)  # retain both outputs across all replays
            ttnn.end_trace_capture(mesh_device, traces[route], cq_id=0)
        for fixture_index in (0, 1, 0):
            sample = inputs[fixture_index]
            for key, device_tensor in persistent.items():
                host_tensor = _tt(sample[key])
                ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
            replay = {"fixture_index": fixture_index}
            for route in traces:
                ttnn.execute_trace(mesh_device, traces[route], cq_id=0, blocking=True)
                replay[route] = _read(outputs[route])
            result["replay_outputs"].append(replay)
        # Alternating AB/BA batches controls order drift. Input0 was restored above.
        for batch in range(batches):
            for route in ("expanded", "native") if batch % 2 == 0 else ("native", "expanded"):
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(replays):
                    ttnn.execute_trace(mesh_device, traces[route], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                result["timing_samples_us"][route].append((time.perf_counter() - start) * 1e6 / replays)
        with result_path.open("xb") as stream:
            torch.save(result, stream)
        print(f"GQA_CORRECTNESS_PENDING {result_path}; run --verify off-device")
        print("GQA_TIMING_SAMPLES " + json.dumps(result["timing_samples_us"]))
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs.values(), *persistent.values()):
            ttnn.deallocate(tensor)


def _verify(result_dir, fixture_dir, mode="accurate", manifest=None, execution=None):
    report = verify_suite(result_dir, fixture_dir, mode, manifest, execution)
    label = "GQA_CORRECTNESS_PASS" if mode == "accurate" else "GQA_APPROXIMATE_DIAGNOSTIC_PASS"
    print(f"{label} all {len(ALL_CASES)} cases; component scope only")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--exp-mode", choices=MODES, default="accurate")
    parser.add_argument("--manifest", type=Path, help="task38 acceptance manifest; native-owner attestation required")
    parser.add_argument("--execution", type=Path, help="hash-bound C04 executed recipe/raw-log receipt")
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    else:
        if not args.fixtures:
            parser.error("--verify requires --fixtures")
        _verify(args.verify, args.fixtures, args.exp_mode, args.manifest, args.execution)
