# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in C03 audio-placement evidence; CPU preparation/verification run off-device.

Prepare: python -m models.tt_dit.tests.models.ltx.test_audio_submesh --prepare <inputs.pt>
Collect through the broker with C03_INPUTS=<inputs.pt>, C03_RESULTS=<fresh directory>,
LTX_CHECKPOINT=<same checkpoint>, and LTX_AUDIO_SUBMESH=4x8|1x8|1x4|1x1.
Verify: python -m models.tt_dit.tests.models.ltx.test_audio_submesh \
    --verify <results.pt> --inputs <inputs.pt> [--baseline <full-mesh results.pt>]

Collection success is not a quality gate. C03_PROFILE=1 collects an eager per-block
drained profile, separate from traced latency; its output cannot pass the verifier.
Recipe capture exits skipped without writing waveforms, timings, or quality results.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

FRAMES, FPS = 145, 24.0
FIXTURE = Path(__file__).with_name("fixtures") / "girl_audio_latent.npy"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint():
    path = Path(os.environ["LTX_CHECKPOINT"]).expanduser().resolve(strict=True)
    assert path.is_file(), path
    return path


def _prepare(path):
    # Fail on absent oracle dependencies instead of inheriting importorskip's
    # misleading successful pytest exit. This command never opens a device.
    import diffusers.models.autoencoders.autoencoder_kl_ltx2_audio  # noqa: F401
    import diffusers.pipelines.ltx2.vocoder  # noqa: F401

    from models.tt_dit.tests.models.ltx.test_audio_ltx import _decode_audio_reference
    from models.tt_dit.utils.cache import source_id

    assert not path.exists(), f"refusing to overwrite {path}"
    checkpoint = _checkpoint()
    a = torch.from_numpy(np.load(FIXTURE, allow_pickle=False)).float()
    assert tuple(a.shape) == (1, 151, 128) and torch.isfinite(a).all()
    # Deliberately different, deterministic input derived from real content.
    # Its own oracle catches a trace that still serves the captured A buffers.
    inputs = {"a": a, "b": torch.roll(a, shifts=37, dims=1)}
    references = {}
    for name, latent in inputs.items():
        audio = _decode_audio_reference(str(checkpoint), latent, FRAMES, FPS)
        _wave_stats(audio.waveform)
        references[name] = {"waveform": audio.waveform.clone(), "sampling_rate": audio.sampling_rate}
    assert not torch.equal(references["a"]["waveform"], references["b"]["waveform"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "checkpoint_source_id": source_id(checkpoint),
            "fixture_sha256": _sha256(FIXTURE),
            "frames": FRAMES,
            "fps": FPS,
            "inputs": inputs,
            "references": references,
        },
        path,
    )
    print(f"C03_PREPARED {path} sha256={_sha256(path)} shape={tuple(a.shape)}")


def _wave_stats(waveform):
    assert waveform.ndim == 2 and waveform.shape[0] == 2, waveform.shape
    assert torch.isfinite(waveform).all(), "non-finite waveform"
    x = waveform.double()
    rms = x.square().mean().sqrt().item()
    assert rms > 0 and x.std().item() > 0, "flat or silent waveform"
    return {"peak": x.abs().max().item(), "rms": rms, "clipping_fraction": (x.abs() >= 1).double().mean().item()}


def _metrics(reference, waveform, sampling_rate):
    assert waveform.shape == reference.shape, f"sample-count mismatch: {waveform.shape} != {reference.shape}"
    stats = _wave_stats(waveform)
    r, x = reference.double(), waveform.double()
    error = x - r
    mse = error.square().mean().item()
    stats.update(
        psnr_db=math.inf if mse == 0 else 10 * math.log10(r.abs().max().item() ** 2 / mse),
        snr_db=math.inf if mse == 0 else 10 * math.log10(r.square().mean().item() / mse),
        pcc=torch.corrcoef(torch.stack((r.flatten(), x.flatten())))[0, 1].item(),
        max_abs=error.abs().max().item(),
        # Sliding windows expose interior shard seams and both clip boundaries,
        # without assuming that a seam lies at duration / T-shards after padding.
        window_rmse=[
            error[:, start : start + sampling_rate // 10].square().mean().sqrt().item()
            for start in range(0, x.shape[-1], sampling_rate // 20)
        ],
    )
    window = torch.hann_window(1024, dtype=torch.float64)
    r_spec = torch.stft(r, n_fft=1024, hop_length=256, window=window, return_complex=True).abs()
    x_spec = torch.stft(x, n_fft=1024, hop_length=256, window=window, return_complex=True).abs()
    stats["relative_spectrum_l2"] = ((x_spec - r_spec).norm() / r_spec.norm()).item()
    return stats


def _set_trace(pipeline, enabled):
    pipeline.tt_mel_decoder.use_trace = enabled
    pipeline.tt_vocoder_with_bwe.use_trace = enabled
    pipeline.tt_vocoder_with_bwe.use_trace_bwe = enabled


def _resource_estimates(pipeline):
    """Logical parameter bytes only: not padded allocation, peak memory, or energy."""
    import ttnn
    from models.tt_dit.tests.models.ltx.test_audio_ltx import _walk_tt

    mesh = tuple(pipeline.audio_mesh_device.shape)
    result = {}
    for name, root in (("mel", pipeline.tt_mel_decoder), ("vocoder_bwe_stft", pipeline.tt_vocoder_with_bwe)):
        local_bytes = 0
        for module in _walk_tt(root):
            for _, parameter in module.named_parameters():
                byte_width = {ttnn.bfloat16: 2, ttnn.float32: 4}[parameter.dtype]
                local_bytes += math.prod(parameter.local_shape) * byte_width
        result[name] = {
            "logical_parameter_bytes_per_chip": local_bytes,
            "logical_parameter_bytes_all_chips": local_bytes * math.prod(mesh),
        }
    result["active_chips"] = math.prod(mesh)
    result["mel_replicas"] = math.prod(mesh)
    result["vocoder_time_shards"] = max(mesh)
    result["vocoder_sharded_body_replicas"] = min(mesh)
    result[
        "interpretation"
    ] = "source-derived estimates; exclude padding, non-Parameter buffers, traces, and activation peaks; energy unmeasured"
    return result


def pytest_generate_tests(metafunc):
    # Keep the offline verifier importable without TTNN's compiled device module.
    # At pytest collection, use exactly the served Galaxy ring / f07 line fabric.
    import ttnn
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    common = {"l1_small_size": 32768, "trace_region_size": 300_000_000}
    metafunc.parametrize(
        "mesh_device,device_params,parent_topology",
        [
            pytest.param((4, 8), {**ring_params_8k_req_exact_devices, **common}, ttnn.Topology.Ring, id="galaxy"),
            pytest.param((2, 4), {**line_params_req_exact_devices, **common}, ttnn.Topology.Linear, id="f07"),
        ],
        indirect=["mesh_device", "device_params"],
    )


@pytest.mark.skip_post_commit
@pytest.mark.skipif(os.environ.get("C03_LIFECYCLE", "0") != "1", reason="explicit C03 lifecycle regression")
def test_audio_submesh_lifecycle(mesh_device, device_params, parent_topology, monkeypatch):
    """Reproduce shared-CQ ownership at close without model weights or kernels.

    Through the broker, run with C03_LIFECYCLE=1 and select galaxy or f07.
    The mesh fixture closes the child before the parent after this test returns.
    """
    import ttnn
    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

    monkeypatch.setenv("LTX_AUDIO_SUBMESH", "1x4")
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=parent_topology)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=DiTParallelConfig.from_tuples(
            cfg=(1, 0), tp=(mesh_device.shape[0], 0), sp=(mesh_device.shape[1], 1)
        ),
        ccl_manager=ccl_manager,
        checkpoint_name=None,
        traced=False,
    )
    child = pipeline._owned_audio_submesh
    assert child is not None and tuple(child.shape) == (1, 4)
    # Both CCLManager constructors initialized global semaphores through CQ0.
    # Even after synchronization the old cleanup leaves both queues in use.
    try:
        pipeline.release_traces()
    finally:
        pipeline.release_audio_submesh()
    pipeline.release_audio_submesh()  # Repeat cleanup, as an outer finalizer may do.
    assert pipeline.audio_mesh_device is mesh_device
    assert pipeline.audio_ccl_manager is pipeline.vae_ccl_manager

    # Parent work may resume before the caller tears down the still-owned child.
    semaphore = ttnn.create_global_semaphore(mesh_device, ccl_manager.ccl_cores, 0)
    ttnn.synchronize_device(mesh_device)
    del semaphore


@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    "C03_INPUTS" not in os.environ and "C03_RESULTS" not in os.environ,
    reason="explicit audio-placement experiment; set C03_INPUTS and C03_RESULTS",
)
def test_collect_audio_submesh(mesh_device, device_params, parent_topology):
    import ttnn
    from models.tt_dit.tests.models.ltx.test_audio_ltx import (
        _PROF_FLUSH_TYPES,
        _build_pipeline,
        _flush_forward_after,
        _walk_tt,
    )
    from models.tt_dit.utils.cache import source_id
    from models.tt_dit.utils.tracing import set_kernel_prewarm_capturing

    inputs_path = Path(os.environ["C03_INPUTS"])
    output_path = Path(os.environ["C03_RESULTS"]) / "results.pt"
    assert not output_path.exists(), f"refusing to overwrite {output_path}"
    data = torch.load(inputs_path, map_location="cpu", weights_only=True)
    assert data["schema"] == 1 and data["frames"] == FRAMES and data["fps"] == FPS
    assert data["fixture_sha256"] == _sha256(FIXTURE), "wrong real-latent fixture"
    checkpoint = _checkpoint()
    assert data["checkpoint_source_id"] == source_id(checkpoint), "oracle checkpoint changed"
    requested_mesh = tuple(int(v) for v in os.environ["LTX_AUDIO_SUBMESH"].split("x"))
    assert requested_mesh in (tuple(mesh_device.shape), (1, 8), (1, 4), (1, 1)), requested_mesh
    assert all(0 < extent <= full for extent, full in zip(requested_mesh, mesh_device.shape))
    assert os.environ.get("LTX_AUDIO_CHANNEL_TP", "0") == "0", "C03 isolates time sharding with channel TP off"
    assert not os.environ.get("LTX_DUMP_AUDIO_LATENT"), "latent file writes would contaminate timing"
    assert os.environ.get("LTX_TIME_STAGES", "0") == "0", "stage syncs would contaminate timing"
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    profile = os.environ.get("C03_PROFILE", "0") == "1"
    repeats = int(os.environ.get("C03_REPEATS", "5"))
    assert repeats >= 5, "keep at least five independent warm samples"

    pipeline, _ = _build_pipeline(
        mesh_device, sp_axis=1, tp_axis=0, checkpoint=str(checkpoint), num_links=2, topology=parent_topology
    )
    assert tuple(pipeline.audio_mesh_device.shape) == requested_mesh, "submesh request silently ignored"
    assert pipeline.audio_ccl_manager.topology == ttnn.Topology.Linear
    assert pipeline.tt_mel_decoder.mesh_device is pipeline.audio_mesh_device
    assert pipeline.tt_vocoder_with_bwe.mesh_device is pipeline.audio_mesh_device
    audio_device = pipeline.audio_mesh_device
    outputs, times = {}, []
    if profile and not capture_only:
        # Warmup emits instrumented programs too; drain its blocks as well so it
        # cannot overflow the profiler before the measured region even starts.
        for root in (pipeline.tt_mel_decoder, pipeline.tt_vocoder_with_bwe):
            for module in _walk_tt(root):
                if isinstance(module, _PROF_FLUSH_TYPES):
                    _flush_forward_after(module, audio_device)

    def decode(label, name):
        out = pipeline.decode_audio(data["inputs"][name], FRAMES, fps=FPS)
        if not capture_only:
            outputs[label] = {"waveform": out.waveform.clone(), "sampling_rate": out.sampling_rate}
        return out

    try:
        decode("eager_a", "a")
        if capture_only:
            # Record eager + persistent-buffer variants in this disposable process.
            # Values read during recipe discovery are never quality/timing evidence.
            _set_trace(pipeline, True)
            set_kernel_prewarm_capturing(True)
            decode("recipe", "a")
            pytest.skip("kernel recipe capture only; no correctness or timing result")
        decode("eager_b", "b")
        estimates = _resource_estimates(pipeline)
        if profile:
            from tracy import signpost

            ttnn.synchronize_device(audio_device)
            ttnn.ReadDeviceProfiler(audio_device)
            op0 = ttnn._ttnn.get_device_operation_id()
            signpost("start")
            decode("profile_a", "a")
            ttnn.synchronize_device(audio_device)
            ttnn.ReadDeviceProfiler(audio_device)
            signpost("stop")
            print(f"C03_PROFILE_OPID_RANGE={op0},{ttnn._ttnn.get_device_operation_id()}")
        else:
            _set_trace(pipeline, True)
            decode("capture_a", "a")
            decode("replay_a0", "a")  # absorb first replay before timing
            decode("replay_b", "b")
            decode("replay_a1", "a")
            for index in range(repeats):
                ttnn.synchronize_device(audio_device)
                start = time.perf_counter()
                out = pipeline.decode_audio(data["inputs"]["a"], FRAMES, fps=FPS)
                ttnn.synchronize_device(audio_device)
                times.append((time.perf_counter() - start) * 1000)
                outputs[f"timed_a{index}"] = {"waveform": out.waveform.clone(), "sampling_rate": out.sampling_rate}
            for module in (
                pipeline.tt_mel_decoder,
                pipeline.tt_vocoder_with_bwe.vocoder,
                pipeline.tt_vocoder_with_bwe.bwe_generator,
            ):
                tracers = type(module)._forward_device._tracers_keyed.get(module, {})
                assert len(tracers) == 1 and all(
                    t._trace_ids is not None for t in tracers.values()
                ), "missing real audio trace"
        metadata = {
            "schema": 1,
            "profile": profile,
            "capture_only": False,
            "inputs_sha256": _sha256(inputs_path),
            "checkpoint_source_id": source_id(checkpoint),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha256": _sha256(Path(__file__)),
            "parent_mesh": list(mesh_device.shape),
            "audio_mesh": list(audio_device.shape),
            "audio_topology": "Linear",
            "warm_ms": times,
            "resource_estimates": estimates,
            "timing_scope": "synchronized torch-latent-in to torch-waveform-out; includes transfers, host bridges, and trimming; excludes load, warmup, capture and file I/O",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "outputs": outputs}, output_path)
        print(f"C03_CORRECTNESS_PENDING {output_path}")
        print(json.dumps(metadata, sort_keys=True))
    finally:
        set_kernel_prewarm_capturing(False)
        pipeline.release_traces()
        pipeline.release_audio_submesh()


def _validate_result(result, inputs_sha):
    meta, outputs = result["metadata"], result["outputs"]
    assert meta["schema"] == 1 and not meta["capture_only"] and not meta["profile"]
    assert meta["inputs_sha256"] == inputs_sha, "result/oracle input mismatch"
    assert len(meta["warm_ms"]) >= 5 and all(math.isfinite(t) and t > 0 for t in meta["warm_ms"])
    required = {"eager_a", "eager_b", "capture_a", "replay_a0", "replay_b", "replay_a1"}
    required.update(f"timed_a{i}" for i in range(len(meta["warm_ms"])))
    assert set(outputs) == required, "missing eager/capture/changed-input/warm output"
    anchor = outputs["replay_a0"]["waveform"]
    for name in required - {"eager_a", "eager_b", "replay_b"}:
        assert torch.equal(anchor, outputs[name]["waveform"]), f"same-input trace drift in {name}"
    assert not torch.equal(anchor, outputs["replay_b"]["waveform"]), "changed input replayed stale A"
    return meta, outputs


def _verify(result_path, inputs_path, baseline_path=None):
    # A malformed replacement must never leave an earlier green verdict behind.
    report_path = result_path.with_suffix(".quality.json")
    report_path.unlink(missing_ok=True)
    data = torch.load(inputs_path, map_location="cpu", weights_only=True)
    meta, outputs = _validate_result(
        torch.load(result_path, map_location="cpu", weights_only=True), _sha256(inputs_path)
    )
    assert meta["checkpoint_source_id"] == data["checkpoint_source_id"]
    report = {
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "inputs_sha256": _sha256(inputs_path),
        "baseline_result_sha256": None if baseline_path is None else _sha256(baseline_path),
        "mean_warm_ms": sum(meta["warm_ms"]) / len(meta["warm_ms"]),
        "metrics": {},
    }
    failures = []
    for name, output in outputs.items():
        ref = data["references"]["b" if name in ("eager_b", "replay_b") else "a"]
        assert output["sampling_rate"] == ref["sampling_rate"] == 48000
        metrics = _metrics(ref["waveform"], output["waveform"], ref["sampling_rate"])
        report["metrics"][name] = metrics
        if metrics["psnr_db"] < 28.0:  # Existing real-checkpoint full audio reference gate.
            failures.append(f"{name}: CPU PSNR {metrics['psnr_db']:.3f} dB < 28 dB")
    # Eager versus trace isolates replay mistakes from CPU-versus-device numerics.
    for suffix in ("a", "b"):
        replay = outputs["replay_a0" if suffix == "a" else "replay_b"]["waveform"]
        if not torch.equal(outputs[f"eager_{suffix}"]["waveform"], replay):
            failures.append(f"eager/trace mismatch for {suffix}; inspect metrics and debug before accepting")
    if baseline_path:
        base_meta, baseline = _validate_result(
            torch.load(baseline_path, map_location="cpu", weights_only=True), _sha256(inputs_path)
        )
        assert base_meta["audio_mesh"] == base_meta["parent_mesh"] == meta["parent_mesh"]
        assert base_meta["checkpoint_source_id"] == meta["checkpoint_source_id"]
        report["baseline_metrics"] = {}
        for name in ("replay_a0", "replay_b"):
            metrics = _metrics(baseline[name]["waveform"], outputs[name]["waveform"], 48000)
            report["baseline_metrics"][name] = metrics
            # Existing time-sharding max-error guard, complementary to whole-wave PSNR.
            if metrics["max_abs"] >= 5e-3:
                failures.append(f"{name}: full-mesh max error {metrics['max_abs']:.6g} >= 5e-3")
    report["failures"] = failures
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    assert not failures, f"C03 quality unresolved; debug/measure benign controls: {failures}; report={report_path}"
    print(f"C03_CORRECTNESS_PASS {report_path}; mean_warm_ms={report['mean_warm_ms']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    else:
        if args.inputs is None:
            parser.error("--verify requires --inputs")
        _verify(args.verify, args.inputs, args.baseline)
