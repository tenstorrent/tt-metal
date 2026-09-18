# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Collect C04 SDPA evidence at served per-chip geometry, then verify off-device.

The pipeline uses encoder TP=mesh.shape[1], independently of DiT TP. Galaxy TP8
has Q2/KV1; f07 TP4 has Q4/KV2. Both use B1/S1024/D256, chunks128, HiFi2,
fp32 accumulation. This isolates repeat_interleave+SDPA on one chip; it is not
an end-to-end encoder benchmark or proof of a pipeline latency improvement.

Prepare/verify commands run on the host OUTSIDE the device reservation:
  python -m models.tt_dit.tests.encoders.gemma.test_gemma_native_gqa --prepare /tmp/gqa-inputs
  LTX_GEMMA_GQA_FIXTURES=/tmp/gqa-inputs LTX_GEMMA_GQA_RESULTS=/tmp/gqa-results \
    pytest <this-file> -k 'tp8 and leftpad' -s  # through the device broker
  python -m models.tt_dit.tests.encoders.gemma.test_gemma_native_gqa \
    --verify /tmp/gqa-results --fixtures /tmp/gqa-inputs --case tp8-leftpad

The pytest collection step deliberately emits GQA_CORRECTNESS_PENDING. Only the
offline verifier emits GQA_CORRECTNESS_PASS; collection success is not a gate.
Use a fresh results directory per revision/run. Full Gemma and changed-prompt
pipeline replay/quality remain required before enabling the production flag.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

SEQ = 1024
HEAD_DIM = 256
CASES = [(heads, mask) for heads in (2, 4) for mask in ("causal", "leftpad")]


def _case_name(heads, mask):
    return f"tp{16 // heads}-{mask}"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(directory):
    """FP32 CPU oracle from the same quantized Q/K/V and exact production mask."""
    directory.mkdir(parents=True, exist_ok=True)
    for heads, mask_mode in CASES:
        cases = []
        for seed, real_tokens in ((11, 257), (29, 613)):
            generator = torch.Generator().manual_seed(seed)
            q = torch.randn((1, heads, SEQ, HEAD_DIM), generator=generator).bfloat16()
            k = torch.randn((1, heads // 2, SEQ, HEAD_DIM), generator=generator).bfloat16()
            v = torch.randn((1, heads // 2, SEQ, HEAD_DIM), generator=generator).bfloat16()
            mask = None
            if mask_mode == "leftpad":
                # GemmaEncoder.build_attn_mask: causal + left-padding key mask.
                causal = torch.triu(torch.full((SEQ, SEQ), float("-inf")), diagonal=1)[None, None]
                valid = torch.arange(SEQ) >= SEQ - real_tokens
                padding = torch.where(valid[None, None, None, :], 0.0, float("-inf"))
                mask = (causal + padding).bfloat16()
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float().repeat_interleave(2, dim=1),
                v.float().repeat_interleave(2, dim=1),
                attn_mask=None if mask is None else mask.float(),
                is_causal=mask is None,
                scale=1.0 / math.sqrt(HEAD_DIM),
            )
            assert torch.isfinite(reference).all(), "CPU oracle produced non-finite output"
            cases.append(
                {
                    "seed": seed,
                    "q": q,
                    "k": k,
                    "v": v,
                    "mask": mask,
                    "real_tokens": SEQ if mask is None else real_tokens,
                    "reference": reference,
                }
            )
        path = directory / f"{_case_name(heads, mask_mode)}.pt"
        torch.save({"heads": heads, "mask_mode": mask_mode, "cases": cases}, path)
        print(f"GQA_FIXTURE {path} sha256={_sha256(path)}")


def _tt(host, device=None):
    import ttnn

    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _read(device_tensor):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(device_tensor)[0]).float()


@pytest.mark.parametrize("heads,mask_mode", CASES, ids=[_case_name(*case) for case in CASES])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    "LTX_GEMMA_GQA_FIXTURES" not in os.environ and "LTX_GEMMA_GQA_RESULTS" not in os.environ,
    reason="explicit GQA evidence collection; set both fixture and result directories",
)
def test_collect_gemma_native_gqa(mesh_device, heads, mask_mode):
    import ttnn

    fixture_dir = Path(os.environ["LTX_GEMMA_GQA_FIXTURES"])
    result_dir = Path(os.environ["LTX_GEMMA_GQA_RESULTS"])
    name = _case_name(heads, mask_mode)
    result_path = result_dir / f"{name}.pt"
    assert not result_path.exists(), f"use a fresh results directory; refusing to replace {result_path}"
    fixture_path = fixture_dir / f"{name}.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert fixture["heads"] == heads and fixture["mask_mode"] == mask_mode
    inputs = fixture["cases"]
    batches = int(os.environ.get("GQA_TIMING_BATCHES", "5"))
    replays = int(os.environ.get("GQA_REPLAYS_PER_BATCH", "25"))
    assert batches > 0 and replays > 0
    result_dir.mkdir(parents=True, exist_ok=True)

    # All persistent inputs are allocated before either trace capture. Updates
    # below copy to the same addresses, including the changed padding mask.
    persistent = {key: _tt(inputs[0][key], mesh_device) for key in ("q", "k", "v")}
    if mask_mode == "leftpad":
        persistent["mask"] = _tt(inputs[0]["mask"], mesh_device)
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    diagnostic_hifi4 = os.environ.get("GQA_DIAGNOSTIC_HIFI4", "0") == "1"
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if diagnostic_hifi4 else ttnn.MathFidelity.HiFi2,
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
            is_causal=mask_mode == "causal",
            attn_mask=persistent.get("mask"),
            scale=1.0 / math.sqrt(HEAD_DIM),
            program_config=program,
            compute_kernel_config=compute,
        )

    if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY") == "1":
        capture_outputs = [forward(False), forward(True)]
        pytest.skip("kernel recipe capture only; no correctness or timing result")

    traces, outputs = {}, {}
    result = {
        "case": name,
        "fixture_sha256": _sha256(fixture_path),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            path: _sha256(Path(path))
            for path in (
                "models/tt_dit/encoders/gemma/model_gemma.py",
                "models/tt_dit/tests/encoders/gemma/test_gemma_native_gqa.py",
            )
        },
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "arch": str(mesh_device.arch()),
        "mesh_shape": list(mesh_device.shape),
        "local_q_shape": [1, heads, SEQ, HEAD_DIM],
        "local_kv_shape": [1, heads // 2, SEQ, HEAD_DIM],
        "timing_boundary": "synchronized trace replay of KV-repeat+SDPA versus native GQA SDPA",
        "profiler_env": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "diagnostic_hifi4": diagnostic_hifi4,
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
        torch.save(result, result_path)
        print(f"GQA_CORRECTNESS_PENDING {result_path}; run --verify off-device")
        print("GQA_TIMING_SAMPLES " + json.dumps(result["timing_samples_us"]))
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs.values(), *persistent.values()):
            ttnn.deallocate(tensor)


def _verify(result_dir, fixture_dir, name):
    result_path = result_dir / f"{name}.pt"
    fixture_path = fixture_dir / f"{name}.pt"
    report_path = result_dir / f"{name}.json"
    # A failed rerun must not leave an earlier successful summary beside altered
    # evidence. Raw .pt records are immutable during normal device collection.
    report_path.unlink(missing_ok=True)
    result = torch.load(result_path, map_location="cpu", weights_only=True)
    assert result["case"] == name and result["fixture_sha256"] == _sha256(fixture_path), "fixture provenance mismatch"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert [r["fixture_index"] for r in result["replay_outputs"]] == [0, 1, 0]
    metrics = []
    for index, replay in enumerate(result["replay_outputs"]):
        sample = fixture["cases"][replay["fixture_index"]]
        first_real = SEQ - sample["real_tokens"]
        oracle = sample["reference"][..., first_real:, :].flatten()
        for route in ("expanded", "native"):
            output = replay[route]
            assert torch.isfinite(output).all(), f"{name}/{route}/{index}: non-finite output"
            real = output[..., first_real:, :].flatten()
            pcc = torch.corrcoef(torch.stack((oracle, real)))[0, 1].item()
            rel_rmse = ((real - oracle).square().mean().sqrt() / oracle.std()).item()
            metrics.append({"replay": index, "route": route, "pcc": pcc, "relative_rmse": rel_rmse})
            # Noniterative SDPA sanity gate; do not reject the pipeline on a
            # denoising PCC. A failure here requires debugging baseline and native.
            assert pcc >= 0.999 and rel_rmse <= 0.02, f"{name}/{route}: {metrics[-1]}"
        # No mathematical operation changed, so record and demand exact route
        # parity for this bounded kernel change. If it fails, inspect the raw
        # delta/kernel schedules; do not silently weaken this gate or discard.
        torch.testing.assert_close(replay["native"], replay["expanded"], rtol=0, atol=0)
    for route in ("expanded", "native"):
        replay = result["replay_outputs"]
        torch.testing.assert_close(replay[0][route], result["eager_outputs"][route], rtol=0, atol=0)
        torch.testing.assert_close(replay[0][route], replay[2][route], rtol=0, atol=0)
        assert not torch.equal(replay[0][route], replay[1][route]), f"{route}: changed input replay was stale"
    report = {
        "case": name,
        "commit": result["commit"],
        "result_sha256": _sha256(result_path),
        "fixture_sha256": result["fixture_sha256"],
        "quality_pass": True,
        "bitwise_route_equal": True,
        "metrics": metrics,
        "timing_samples_us": result["timing_samples_us"],
        "profiler_env": result["profiler_env"],
        "diagnostic_hifi4": result.get("diagnostic_hifi4", False),
        "timing_boundary": result["timing_boundary"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GQA_CORRECTNESS_PASS {name} commit={result['commit']} bitwise_route_equal=true")
    print("GQA_VERIFIED_EVIDENCE " + json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--case", choices=[_case_name(*case) for case in CASES])
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    else:
        if not (args.fixtures and args.case):
            parser.error("--verify requires --fixtures and --case")
        _verify(args.verify, args.fixtures, args.case)
