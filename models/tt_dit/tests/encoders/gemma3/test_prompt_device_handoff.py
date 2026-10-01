# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in C09 sink-bit evidence; invoke only through the device broker.

C09_PROMPTS=<C04 prompts.json> C09_RESULTS=<fresh directory>
C09_PHASE=component|integrated LTX_DEVICE_PROMPT_HANDOFF=1
pytest this_file::test_prompt_device_handoff -s. Component compares real eager
and captured A/B/A handoffs plus a native FP32/BF16 boundary probe. Integrated
uses the production145-frame static pipeline and checks sinks after all actual
consumer traces run. Readbacks are validation outside handoff timing only.
CPU: --verify DIR. Neither phase establishes absolute AV quality or speedup.
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


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bits(value):
    assert value.device.type == "cpu" and value.dtype == torch.bfloat16
    return value.contiguous().view(torch.int16)


def _tensor_sha(value):
    return hashlib.sha256(_bits(value).numpy().tobytes()).hexdigest()


def _cast_values():
    # BF16 halfway boundaries and their FP32 neighbors expose double rounding;
    # signed zeros and exact powers expose copy/representation changes.
    bits = [
        0,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x3F807FFF,
        0x3F808000,
        0x3F808001,
        0x3F817FFF,
        0x3F818000,
        0x3F818001,
        0xBF807FFF,
        0xBF808000,
        0xBF808001,
        0x00800000,
        0x7F000000,
    ]
    signed = [x if x < 2**31 else x - 2**32 for x in bits]
    values = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    return values.repeat(math.ceil(1024 / len(values)))[:1024].reshape(1, 1, 32, 32)


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import ring_params_8k_req_exact_devices

    metafunc.parametrize(
        "mesh_device,device_params",
        [
            pytest.param(
                (4, 8),
                {**ring_params_8k_req_exact_devices, "l1_small_size": 32768, "trace_region_size": 500_000_000},
                id="galaxy",
            )
        ],
        indirect=["mesh_device", "device_params"],
    )


@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    not os.environ.get("C09_PROMPTS") and not os.environ.get("C09_RESULTS"), reason="opt-in C09 evidence"
)
def test_prompt_device_handoff(mesh_device, device_params):
    import ttnn
    from models.tt_dit.encoders.gemma3.encoder_pair import GemmaTokenizerEncoderPair
    from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerModel
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
    from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
    from models.tt_dit.tests.encoders.gemma3.test_gemma_prompt_replay import _prompts, _sources
    from models.tt_dit.utils.tensor import bf16_tensor
    from models.tt_dit.utils.tracing import Tracer, set_kernel_prewarm_capturing

    directory = Path(os.environ["C09_RESULTS"])
    assert not directory.exists(), "use a fresh evidence directory"
    directory.mkdir(parents=True)
    prompts_path = Path(os.environ["C09_PROMPTS"])
    prompts = _prompts(prompts_path)
    checkpoint, gemma, sources = _sources()
    phase = os.environ["C09_PHASE"]
    assert phase in ("component", "integrated")
    assert os.environ.get("LTX_DEVICE_PROMPT_HANDOFF") == "1"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", "")
    assert os.environ.get("LTX_AUDIO_SUBMESH", "4x8") == "4x8"
    if phase == "integrated":
        assert not os.environ.get("LTX_GEN_EAGER_STAGES")
        for name in ("LTX_ITER_FAST", "LTX_VIDEO_ONLY", "LTX_PROFILE_DENOISE_ONLY"):
            assert os.environ.get(name, "0") == "0", name
        for name in ("LTX_VOC_TRACE", "LTX_BWE_TRACE", "LTX_VAE_TRACE"):
            assert os.environ.get(name, "1") == "1", name
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    shape = tuple(mesh_device.shape)
    assert shape == (4, 8)
    rows, samples, media = [], {"baseline_host_roundtrip_ms": [], "device_handoff_ms": []}, []
    pipe = None
    pair = None
    failure = None
    guard_checked = False
    source_paths = [
        __file__,
        "models/tt_dit/tests/encoders/gemma3/test_gemma_prompt_replay.py",
        "models/tt_dit/utils/cache.py",
        "models/tt_dit/encoders/gemma3/encoder_pair.py",
        "models/tt_dit/pipelines/ltx/pipeline_ltx.py",
        "models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py",
        "models/tt_dit/utils/tracing.py",
        "models/tt_dit/utils/tensor.py",
    ]
    metadata = dict(
        schema=1,
        phase=phase,
        capture_only=capture_only,
        sources=sources,
        prompts_sha256=_sha(prompts_path),
        mesh=list(shape),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={p: _sha(p) for p in source_paths},
        binary_sha256=_sha("ttnn/ttnn/_ttnn.so"),
        tracked_status=subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"], text=True
        ),
        cgroup=Path("/proc/self/cgroup").read_text(),
        timing_scope="component synchronized fresh prompt through sink update; validation D2H excluded; five within-process observations, no speed acceptance",
    )

    def checkpoint_evidence():
        if not capture_only:
            torch.save(
                dict(
                    metadata=metadata,
                    rows=rows,
                    samples=samples,
                    media=media,
                    guard_checked=guard_checked,
                    failure=failure,
                ),
                directory / "evidence.pt",
            )

    def retain(value):
        shards = ttnn.get_device_tensors(value)
        coords = list(value.tensor_topology().mesh_coords())
        assert len(shards) == len(coords) == math.prod(shape)
        result = []
        for coord, shard in zip(coords, shards):
            host = ttnn.to_torch(shard).contiguous()
            result.append((tuple(int(x) for x in coord), host))
        assert {coord for coord, _ in result} == {(i, j) for i in range(shape[0]) for j in range(shape[1])}
        return result

    def compare(label, baseline, candidate):
        row = dict(label=label, modalities={})
        passed = True
        for modality in baseline:
            expected = baseline[modality]
            expected_hash = _tensor_sha(expected)
            replicas, failures = [], {}
            for coord, actual in retain(candidate[modality]):
                actual_hash = _tensor_sha(actual)
                exact = actual.shape == expected.shape and actual_hash == expected_hash
                finite = bool(torch.isfinite(actual).all())
                replicas.append(dict(coord=list(coord), sha256=actual_hash, finite=finite, exact=exact))
                if not exact or not finite:
                    failures[str(coord)] = actual.clone()
                    passed = False
            row["modalities"][modality] = dict(
                baseline=expected.clone(),
                baseline_sha256=expected_hash,
                replicas=replicas,
                failed_replica_values=failures,
            )
        rows.append(row)
        checkpoint_evidence()  # Preserve the mismatching arrays before failing.
        assert passed, f"C09 sink-bit mismatch: {label}; evidence saved"

    def all_replicas_baseline(buffers):
        result = {}
        for modality, buffer in buffers.items():
            values = retain(buffer)
            result[modality] = values[0][1].clone()
            expected = _tensor_sha(result[modality])
            if not all(_tensor_sha(value) == expected for _, value in values):
                compare("baseline_broadcast_failure", {modality: result[modality]}, {modality: buffer})
        return result

    def encoder_tracer():
        return GemmaTokenizerEncoderPair._encode_device._tracers.get(pair)

    try:
        # Cheap native cast probe before loading Gemma. It uses the exact copy
        # primitive/host upload dtype transition used by the real handoff.
        raw = _cast_values()
        native = ttnn.from_torch(raw, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        expected = ttnn.from_torch(raw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        actual = ttnn.zeros(raw.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ttnn.copy(native, actual)
        if not capture_only:
            compare("native_cast_boundary", all_replicas_baseline({"cast": expected}), {"cast": actual})
        del native, expected, actual

        # The production constructor performs its mandatory traced warmup,
        # including VAE/audio captures. Reserve control buffers before it too.
        control = {
            key: ttnn.zeros((1, 1, 1024, width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            for key, width in (("video", 4096), ("audio", 2048))
        }
        set_kernel_prewarm_capturing(capture_only)
        pipeline_class = LTXDistilledPipeline if phase == "integrated" else LTXPipeline
        pipe = pipeline_class.create_pipeline(
            mesh_device,
            checkpoint_name=checkpoint if phase == "integrated" else None,
            gemma_path=gemma,
            sp_axis=1,
            tp_axis=0,
            num_links=2,
            dynamic_load=False,
            topology=ttnn.Topology.Ring,
            is_fsdp=False,
            mode="av",
            run_warmup=False,
            traced=True,
            num_frames=145,
            height=1088,
            width=1920,
        )
        pair = pipe.gemma_encoder_pair
        pair.checkpoint_name = checkpoint
        # Both control/candidate sinks precede every producer/consumer trace.
        LTXDistilledPipeline._allocate_device_prompt_buffers(pipe)
        sinks = dict(video=pipe._prompt_v.value, audio=pipe._prompt_a.value)
        addresses = {key: int(value.buffer_address()) for key, value in sinks.items()}
        pair.ensure_loaded()
        pair.defer_trace_capture()

        def baseline(name):
            video, audio = pair.encode([prompts[name]])[0]
            assert audio is not None
            ttnn.copy(pipe._prepare_prompt(video.float()), control["video"])
            ttnn.copy(bf16_tensor(audio.float().unsqueeze(0), device=mesh_device), control["audio"])

        def candidate(name):
            pair.encode_to_device_buffers(prompts[name], sinks["video"], sinks["audio"])
            assert all(int(value.buffer_address()) == addresses[key] for key, value in sinks.items())

        def pair_case(label, name):
            baseline(name)
            candidate(name)
            if not capture_only:
                compare(label, all_replicas_baseline(control), sinks)

        if phase == "component":
            for index, name in enumerate(("a", "b", "a")):
                pair_case(f"eager_{index}_{name}", name)
            pair.open_trace_gate()
            set_kernel_prewarm_capturing(capture_only)
            for index, name in enumerate(("a", "b", "a")):
                pair_case(f"traced_{index}_{name}", name)
            if capture_only:
                pytest.skip("C09 kernel recipes only; no correctness/timing evidence")
            assert encoder_tracer() is not None and encoder_tracer().trace_captured
            for index in range(5):
                for label, operation in (("baseline_host_roundtrip_ms", baseline), ("device_handoff_ms", candidate)):
                    ttnn.synchronize_device(mesh_device)
                    start = time.perf_counter()
                    operation("a")
                    ttnn.synchronize_device(mesh_device)
                    samples[label].append((time.perf_counter() - start) * 1000)
                compare(f"timed_{index}_a", all_replicas_baseline(control), sinks)
        else:
            references = {}
            for name in ("a", "b"):
                baseline(name)
                if not capture_only:
                    references[name] = all_replicas_baseline(control)
            set_kernel_prewarm_capturing(capture_only)
            for index, name in enumerate(("a", "a", "b", "a")):
                output = directory / f"consumer_{index}_{name}.mp4"
                pipe.generate(
                    prompts[name], output_path=str(output), num_frames=145, height=1088, width=1920, seed=10, fps=24
                )
                if not capture_only:
                    compare(f"consumers_{index}_{name}", references[name], sinks)
                    media.append(dict(path=str(output), sha256=_sha(output)))
                    if index >= 1:
                        consumer_tracers = list(
                            LTXTransformerModel.inner_step._tracers_keyed.get(pipe.transformer, {}).values()
                        )
                        assert len(consumer_tracers) >= 2 and all(t.trace_captured for t in consumer_tracers)
                        # The conv decoder runs eagerly unless LTX_VIDEO_VAE_TRACE=1.
                        if pipe.vae_decoder.trace_decode:
                            assert (
                                pipe.vae_decoder._decode_tracer is not None
                                and pipe.vae_decoder._decode_tracer.trace_captured
                            )
                        else:
                            assert pipe.vae_decoder._decode_tracer is None
                        for module in (
                            pipe.tt_mel_decoder,
                            pipe.tt_vocoder_with_bwe.vocoder,
                            pipe.tt_vocoder_with_bwe.bwe_generator,
                        ):
                            tracers = type(module)._forward_device._tracers_keyed.get(module, {})
                            assert tracers and all(t.trace_captured for t in tracers.values())
                        assert encoder_tracer() is not None and encoder_tracer().trace_captured
            if capture_only:
                pytest.skip("C09 integrated kernel recipes only; no correctness/timing evidence")

        # Production cleanup leaves the encoder trace live. Recreating sinks
        # must fail until that final producer trace is explicitly released.
        pipe.release_traces()
        assert encoder_tracer() is not None and encoder_tracer().trace_captured
        assert Tracer._traces_live.get(mesh_device.id(), 0) > 0
        try:
            LTXDistilledPipeline._allocate_device_prompt_buffers(pipe)
        except AssertionError as exc:
            assert "before capture" in str(exc)
        else:
            raise AssertionError("live encoder trace allowed prompt sink recreation")
        encoder_tracer().release_trace()
        assert not Tracer._traces_live.get(mesh_device.id(), 0)
        LTXDistilledPipeline._allocate_device_prompt_buffers(pipe)
        sinks = dict(video=pipe._prompt_v.value, audio=pipe._prompt_a.value)
        addresses = {key: int(value.buffer_address()) for key, value in sinks.items()}
        pair_case("recreated_a", "a")
        guard_checked = True
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            checkpoint_evidence()
        finally:
            set_kernel_prewarm_capturing(False)
            try:
                if pair is not None and encoder_tracer() is not None:
                    encoder_tracer().release_trace()
            finally:
                if pipe is not None:
                    try:
                        pipe.release_traces()
                    finally:
                        pipe.release_audio_submesh()


def _verify(directory):
    verdict = directory / "verified.json"
    verdict.unlink(missing_ok=True)
    data = torch.load(directory / "evidence.pt", map_location="cpu", weights_only=True)
    meta = data["metadata"]
    assert meta["schema"] == 1 and not meta["capture_only"] and meta["mesh"] == [4, 8]
    assert data["failure"] is None and data["guard_checked"]
    phase = meta["phase"]
    required = {"native_cast_boundary", "recreated_a"}
    if phase == "component":
        required |= {f"{p}_{i}_{n}" for p in ("eager", "traced") for i, n in enumerate(("a", "b", "a"))}
        required |= {f"timed_{i}_a" for i in range(5)}
        assert set(data["samples"]) == {"baseline_host_roundtrip_ms", "device_handoff_ms"}
        assert all(len(v) == 5 and all(math.isfinite(x) and x > 0 for x in v) for v in data["samples"].values())
    else:
        assert phase == "integrated" and len(data["media"]) == 4
        required |= {f"consumers_{i}_{n}" for i, n in enumerate(("a", "a", "b", "a"))}
    assert len(data["rows"]) == len(required) and {r["label"] for r in data["rows"]} == required
    coords = {(i, j) for i in range(4) for j in range(8)}
    by_label = {row["label"]: row for row in data["rows"]}
    for row in data["rows"]:
        assert set(row["modalities"]) == ({"cast"} if row["label"] == "native_cast_boundary" else {"video", "audio"})
        for modality, value in row["modalities"].items():
            expected_shape = {"cast": (1, 1, 32, 32), "video": (1, 1, 1024, 4096), "audio": (1, 1, 1024, 2048)}[
                modality
            ]
            assert tuple(value["baseline"].shape) == expected_shape
            assert torch.isfinite(value["baseline"]).all()
            assert value["baseline_sha256"] == _tensor_sha(value["baseline"])
            assert not value["failed_replica_values"]
            replicas = value["replicas"]
            assert len(replicas) == len(coords) and {tuple(r["coord"]) for r in replicas} == coords
            assert all(r["finite"] and r["exact"] and r["sha256"] == value["baseline_sha256"] for r in replicas)
    anchor_a = by_label["eager_0_a" if phase == "component" else "consumers_0_a"]
    anchor_b = by_label["eager_1_b" if phase == "component" else "consumers_2_b"]
    for modality in ("video", "audio"):
        assert (
            anchor_a["modalities"][modality]["baseline_sha256"] != anchor_b["modalities"][modality]["baseline_sha256"]
        )
        for label, row in by_label.items():
            if label == "native_cast_boundary":
                continue
            anchor = anchor_b if label.endswith("_b") else anchor_a
            assert (
                row["modalities"][modality]["baseline_sha256"] == anchor["modalities"][modality]["baseline_sha256"]
            ), f"A/B/A reference drift: {label}/{modality}"
    report = dict(
        status="C09_SINK_BITS_VERIFIED",
        phase=phase,
        evidence_sha256=_sha(directory / "evidence.pt"),
        samples=data["samples"],
        verifier_sha256=_sha(__file__),
        scope="literal every-chip prompt sink equivalence and allocation guard; source/broker binding review and absolute AV quality remain separate",
    )
    verdict.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, required=True)
    _verify(parser.parse_args().verify)
