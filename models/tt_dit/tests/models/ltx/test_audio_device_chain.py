# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""C07: exact causal framing and inclusive full-audio A/B/A, default-off experiments.

CPU preparation/verification never opens a device. Reuse C03's real girl latent and
checkpoint CPU oracle via ``--prepare inputs.pt``. Collect each mode in a fresh
broker job with C07_MODE=base|framing|chain, C07_INPUTS and C07_RESULTS. Mode flags
are set before module construction; audio placement remains the full parent mesh.
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

from models.tt_dit.tests.models.ltx.test_audio_submesh import (
    FPS,
    FRAMES,
    _checkpoint,
    _prepare,
    _set_trace,
    _sha256,
    _validate_result,
    _verify,
)


def pytest_generate_tests(metafunc):
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


def _frame_inputs(length):
    # Distinct channels, nonzero means, signs and exact binary fractions expose
    # stereo interleaving, causal offsets and every frame/tile boundary.
    t = torch.arange(length, dtype=torch.float32)
    a = torch.stack(((t % 1024 - 512) / 1024, (t % 127 + 1) / 128))
    b = torch.zeros_like(a)
    positions = sorted({0, min(79, length - 1), min(80, length - 1), length // 2, length - 1})
    b[0, positions] = 0.75
    b[1, positions] = -0.5
    return {"a": a.unsqueeze(-1), "b": b.unsqueeze(-1)}


def _frames(x):
    return torch.nn.functional.pad(x.squeeze(-1), (432, 0)).unfold(-1, 512, 80).contiguous()


def _collect_framing(mesh_device, capture_only):
    import ttnn
    from models.tt_dit.models.audio_vae.bwe_ltx import _STFTFn

    stft = _STFTFn(filter_length=512, hop_length=80, win_length=512, mesh_device=mesh_device)
    results = {}
    # Causal MelDecoder maps T=151 to 4*T-3=601 mel frames, then *160 samples.
    # Also cover the uncropped 604-frame extent, beyond the production boundary.
    for length in (80, 159, 160, 1025, 96160, 96640):
        values = _frame_inputs(length)
        stft.prepare_device_windows(2, length)
        input_dev = ttnn.from_torch(values["a"], device=mesh_device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        eager = stft._frame_device(input_dev)
        if capture_only:
            continue
        saved = {"eager_a": ttnn.to_torch(ttnn.get_device_tensors(eager)[0]).clone()}
        del eager
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            output = stft._frame_device(input_dev)
        finally:
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            for label, name in (("a0", "a"), ("b", "b"), ("a1", "a")):
                host = ttnn.from_torch(values[name], dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, input_dev)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                saved[label] = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).clone()
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        results[length] = saved
    return results


@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    "C07_INPUTS" not in os.environ and "C07_RESULTS" not in os.environ,
    reason="explicit C07 experiment; set C07_INPUTS and C07_RESULTS",
)
def test_collect_audio_device_chain(mesh_device, device_params, parent_topology, monkeypatch):
    import ttnn
    from models.tt_dit.tests.models.ltx.test_audio_ltx import _build_pipeline
    from models.tt_dit.utils.cache import source_id
    from models.tt_dit.utils.tracing import set_kernel_prewarm_capturing

    mode = os.environ["C07_MODE"]
    assert mode in {"base", "framing", "chain"}
    monkeypatch.setenv("LTX_STFT_DEVICE_FRAMING", "1" if mode == "framing" else "0")
    monkeypatch.setenv("LTX_AUDIO_DEVICE_CHAIN", "1" if mode == "chain" else "0")
    monkeypatch.setenv("LTX_AUDIO_SUBMESH", "x".join(str(v) for v in mesh_device.shape))
    assert os.environ.get("LTX_AUDIO_CHANNEL_TP", "0") == "0"
    assert not os.environ.get("LTX_DUMP_AUDIO_LATENT")
    assert os.environ.get("LTX_TIME_STAGES", "0") == "0"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", ""), "collect counters separately from timings"
    inputs_path = Path(os.environ["C07_INPUTS"])
    result_path = Path(os.environ["C07_RESULTS"]) / f"{mode}.pt"
    assert not result_path.exists(), f"refusing to overwrite {result_path}"
    data = torch.load(inputs_path, map_location="cpu", weights_only=True)
    assert data["schema"] == 1 and data["frames"] == FRAMES and data["fps"] == FPS
    checkpoint = _checkpoint()
    assert data["checkpoint_source_id"] == source_id(checkpoint)
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    framing = _collect_framing(mesh_device, capture_only) if mode != "base" else {}
    pipeline, _ = _build_pipeline(
        mesh_device, sp_axis=1, tp_axis=0, checkpoint=str(checkpoint), num_links=2, topology=parent_topology
    )
    audio_device = pipeline.audio_mesh_device
    assert tuple(audio_device.shape) == tuple(mesh_device.shape), "C07 must not change placement"
    voc = pipeline.tt_vocoder_with_bwe
    assert voc.device_chain == (mode == "chain")
    assert voc.mel_stft.stft_fn.device_framing == (mode != "base")
    outputs, times = {}, []

    def decode(label, name):
        output = pipeline.decode_audio(data["inputs"][name], FRAMES, fps=FPS)
        if not capture_only:
            outputs[label] = {"waveform": output.waveform.clone(), "sampling_rate": output.sampling_rate}

    try:
        _set_trace(pipeline, False)
        decode("eager_a", "a")
        if capture_only:
            _set_trace(pipeline, True)
            set_kernel_prewarm_capturing(True)
            decode("recipe", "a")
            pytest.skip("recipe capture only; no correctness or timing evidence")
        decode("eager_b", "b")
        _set_trace(pipeline, True)
        for label, name in (("capture_a", "a"), ("replay_a0", "a"), ("replay_b", "b"), ("replay_a1", "a")):
            decode(label, name)
        repeats = int(os.environ.get("C07_REPEATS", "5"))
        assert repeats >= 5
        for index in range(repeats):
            ttnn.synchronize_device(audio_device)
            start = time.perf_counter()
            output = pipeline.decode_audio(data["inputs"]["a"], FRAMES, fps=FPS)
            ttnn.synchronize_device(audio_device)
            times.append((time.perf_counter() - start) * 1000)
            outputs[f"timed_a{index}"] = {"waveform": output.waveform.clone(), "sampling_rate": output.sampling_rate}
        traced_methods = [(pipeline.tt_mel_decoder, "_forward_device")]
        traced_methods += (
            [(voc, "_forward_device_chain")]
            if mode == "chain"
            else [(voc.vocoder, "_forward_device"), (voc.bwe_generator, "_forward_device")]
        )
        for module, method in traced_methods:
            tracers = getattr(type(module), method)._tracers_keyed.get(module, {})
            assert len(tracers) == 1 and all(t.trace_captured for t in tracers.values()), "missing real trace"
        from models.tt_dit.models.audio_vae import bwe_ltx

        meta = {
            "schema": 1,
            "profile": False,
            "capture_only": False,
            "mode": mode,
            "inputs_sha256": _sha256(inputs_path),
            "checkpoint_source_id": source_id(checkpoint),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha256": _sha256(__file__),
            "bwe_source_sha256": _sha256(bwe_ltx.__file__),
            "parent_mesh": list(mesh_device.shape),
            "audio_mesh": list(audio_device.shape),
            "warm_ms": times,
            "index_logical_bytes_per_chip": sum(
                math.prod(tuple(x.shape)) * 4 for x in voc.mel_stft.stft_fn._window_indices.values()
            ),
            "timing_scope": "synchronized latent-in to waveform-out, including all transfers, host bridges and trim; excluding warmup/capture/file IO",
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": meta, "outputs": outputs, "framing": framing}, result_path)
        print(f"C07_CORRECTNESS_PENDING {result_path}")
    finally:
        set_kernel_prewarm_capturing(False)
        pipeline.release_traces()
        pipeline.release_audio_submesh()


def _verify_c07(path, inputs, baseline):
    verdict = path.with_suffix(".c07.json")
    verdict.unlink(missing_ok=True)
    path.with_suffix(".quality.json").unlink(missing_ok=True)
    result = torch.load(path, map_location="cpu", weights_only=True)
    meta, _ = _validate_result(result, _sha256(inputs))
    assert meta["mode"] in {"base", "framing", "chain"}
    assert meta["audio_mesh"] == meta["parent_mesh"]
    if meta["mode"] != "base":
        assert baseline is not None, "candidate requires matched host-bridge baseline"
        base = torch.load(baseline, map_location="cpu", weights_only=True)
        assert base["metadata"]["mode"] == "base"
        assert base["metadata"]["commit"] == meta["commit"], "collect both routes at the same source revision"
        assert base["metadata"]["harness_sha256"] == meta["harness_sha256"], "baseline/candidate harness mismatch"
        assert base["metadata"]["bwe_source_sha256"] == meta["bwe_source_sha256"], "baseline/candidate source mismatch"
        assert set(result["framing"]) == {80, 159, 160, 1025, 96160, 96640}
        for length, saved in result["framing"].items():
            values = _frame_inputs(length)
            assert set(saved) == {"eager_a", "a0", "b", "a1"}
            for label, output in saved.items():
                assert torch.equal(output, _frames(values["b" if label == "b" else "a"])), (length, label)
    # Existing 28dB full-checkpoint CPU gate, exact eager/replay/restore, and
    # <=5e-3 baseline max-error guard; metrics include seam windows and spectrum.
    _verify(path, inputs, baseline)
    report = {"result_sha256": _sha256(path), "quality_pass": True, "mode": meta["mode"]}
    verdict.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C07_CORRECTNESS_PASS {verdict}")


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
        _verify_c07(args.verify, args.inputs, args.baseline)
