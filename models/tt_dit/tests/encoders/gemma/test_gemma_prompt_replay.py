# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in full Gemma/connector changed-prompt evidence, with prompt cache disabled.

CPU: --prepare prompts.json. Optional independent CPU oracle: --prepare-reference
reference.pt --prompts prompts.json (loads HF12B; run outside a device lease).
Broker: C04_ENCODER_PROMPTS=prompts.json C04_ENCODER_RESULTS=fresh.pt
LTX_GEMMA_NATIVE_GQA=0|1 GEMMA_PATH=<local dir> LTX_CHECKPOINT=<local file>
pytest '<this-file>::test_collect_gemma_prompt_replay[galaxy]' -s.
Run each route in a separate process, then CPU: --verify base.pt --native native.pt
--prompts prompts.json [--reference reference.pt]. Existing SDPA gates are unchanged.
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepare(path):
    from models.tt_dit.utils.ltx import DEFAULT_LTX_PROMPT, STEADY_STATE_LTX_PROMPT

    assert not path.exists(), f"refusing to overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"a": DEFAULT_LTX_PROMPT, "b": STEADY_STATE_LTX_PROMPT}, indent=2) + "\n")


def _prompts(path):
    prompts = json.loads(path.read_text())
    assert set(prompts) == {"a", "b"} and all(isinstance(s, str) and s.strip() for s in prompts.values())
    assert prompts["a"] != prompts["b"]
    return prompts


def _sources():
    from models.tt_dit.utils.cache import source_id

    checkpoint = Path(os.environ["LTX_CHECKPOINT"]).expanduser().resolve(strict=True)
    gemma = Path(os.environ["GEMMA_PATH"]).expanduser().resolve(strict=True)
    assert checkpoint.is_file() and gemma.is_dir()
    weights = sorted(gemma.glob("*.safetensors"))
    assert weights, "local Gemma checkpoint shards absent"
    sources = {"ltx": source_id(checkpoint)}
    for path in weights + sorted(gemma.glob("*.json")) + sorted(gemma.glob("*.model")):
        sources[f"gemma/{path.name}"] = source_id(path)
    return str(checkpoint), str(gemma), sources


def _prepare_reference(path, prompts_path):
    # Existing reference implementation constructs only CPU torch/HF modules.
    # Its TTNN imports require a host installation but it never opens a device.
    from models.tt_dit.tests.encoders.gemma.test_gemma_full import _encode_prompts_reference

    assert not path.exists(), f"refusing to overwrite {path}"
    ckpt, gemma, sources = _sources()
    prompts = _prompts(prompts_path)
    outputs = {}
    for label, prompt in prompts.items():
        outputs[label] = dict(zip(("video", "audio"), _encode_prompts_reference(ckpt, gemma, [prompt])))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts_sha256": _sha(prompts_path), "sources": sources, "outputs": outputs}, path)
    print(f"C04_ENCODER_CPU_REFERENCE {path} sha256={_sha(path)}")


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    common = {"l1_small_size": 8192, "trace_region_size": 300_000_000}
    metafunc.parametrize(
        "mesh_device,device_params",
        [
            pytest.param((4, 8), {**ring_params_8k_req_exact_devices, **common}, id="galaxy"),
            pytest.param((2, 4), {**line_params_req_exact_devices, **common}, id="f07"),
        ],
        indirect=["mesh_device", "device_params"],
    )


@pytest.mark.skip_post_commit
@pytest.mark.skipif(
    not os.environ.get("C04_ENCODER_PROMPTS") and not os.environ.get("C04_ENCODER_RESULTS"),
    reason="opt-in full encoder changed-prompt evidence",
)
def test_collect_gemma_prompt_replay(mesh_device, device_params):
    import ttnn
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
    from models.tt_dit.utils.tracing import set_kernel_prewarm_capturing

    prompts_path = Path(os.environ["C04_ENCODER_PROMPTS"])
    result_path = Path(os.environ["C04_ENCODER_RESULTS"])
    assert not result_path.exists(), f"refusing to overwrite {result_path}"
    mode = os.environ["LTX_GEMMA_NATIVE_GQA"]
    assert mode in {"0", "1"}
    prompts = _prompts(prompts_path)
    ckpt, gemma, sources = _sources()
    repeats = int(os.environ.get("C04_ENCODER_REPEATS", "5"))
    assert repeats >= 5
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "0", "profile separately from encoder timing"
    pipe = LTXPipeline.create_pipeline(
        mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av", dynamic_load=False
    )
    pair = pipe.gemma_encoder_pair
    pair.checkpoint_name = ckpt
    pair.ensure_loaded()
    assert pair._num_layers == 48 and pair.sequence_length == 1024 and pair._encoder_trace
    assert pair.parallel_config.tensor_parallel.factor == mesh_device.shape[1]
    layers = list(pair.gemma_encoder.layers)
    assert len(layers) == 48 and all(layer.self_attn._native_gqa == (mode == "1") for layer in layers)
    outputs, timings = {}, []

    def save(label, values):
        assert len(values) == 1 and values[0][1] is not None
        outputs[label] = {
            name: torch.as_tensor(value).float().clone() for name, value in zip(("video", "audio"), values[0])
        }

    def encode(label, prompt_label):
        values = pipe.encode_prompts([prompts[prompt_label]], use_cache=False)
        if not capture_only:
            save(label, values)

    try:
        pair.defer_trace_capture()
        encode("eager_a", "a")
        if capture_only:
            set_kernel_prewarm_capturing(True)
            pair.open_trace_gate()
            encode("recipe", "a")
            pytest.skip("kernel recipe capture only; no correctness or timing evidence")
        encode("eager_b", "b")
        pair.open_trace_gate()
        encode("capture_a", "a")
        for label, prompt_label in (("replay_a0", "a"), ("replay_b", "b"), ("replay_a1", "a")):
            encode(label, prompt_label)
        tracer = type(pair)._encode_device._tracers.get(pair)
        assert tracer is not None and tracer.trace_captured, "whole encoder trace absent"
        for index in range(repeats):
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            values = pipe.encode_prompts([prompts["a"]], use_cache=False)
            ttnn.synchronize_device(mesh_device)
            timings.append((time.perf_counter() - start) * 1000)
            save(f"timed_a{index}", values)
        record = {
            "schema": 1,
            "mode": mode,
            "prompts_sha256": _sha(prompts_path),
            "sources": sources,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
            "source_sha256": {
                name: _sha(name)
                for name in (
                    __file__,
                    "models/tt_dit/encoders/gemma/model_gemma.py",
                    "models/tt_dit/encoders/gemma/encoder_pair.py",
                )
            },
            "mesh": list(mesh_device.shape),
            "tp": pair.parallel_config.tensor_parallel.factor,
            "ccl_topology": str(pair.ccl_manager.topology),
            "ccl_num_links": pair.ccl_manager.num_links,
            "diagnostic_hifi4_env": os.environ.get("GQA_DIAGNOSTIC_HIFI4", "0"),
            "trace_captured": True,
            "use_prompt_cache": False,
            "warm_ms": timings,
            "timing_boundary": "synchronized prompt-to-host-embeddings, including tokenization/transfers; excludes load/capture/file I/O",
            "outputs": outputs,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, result_path)
        print(f"C04_ENCODER_CORRECTNESS_PENDING {result_path}; warm_ms={timings}")
    finally:
        set_kernel_prewarm_capturing(False)
        tracer = type(pair)._encode_device._tracers.get(pair)
        if tracer is not None:
            tracer.release_trace()
        pipe.release_traces()


def _validate(record, mode, prompts_sha):
    assert record["schema"] == 1 and record["mode"] == mode and record["prompts_sha256"] == prompts_sha
    assert record["trace_captured"] and not record["use_prompt_cache"]
    assert record["mesh"] in ([4, 8], [2, 4]) and record["tp"] == record["mesh"][1]
    assert len(record["warm_ms"]) >= 5 and all(math.isfinite(t) and t > 0 for t in record["warm_ms"])
    outputs = record["outputs"]
    expected = {"eager_a", "eager_b", "capture_a", "replay_a0", "replay_b", "replay_a1"}
    expected.update(f"timed_a{i}" for i in range(len(record["warm_ms"])))
    assert set(outputs) == expected
    for label, modalities in outputs.items():
        assert set(modalities) == {"video", "audio"}
        for modality, width in (("video", 4096), ("audio", 2048)):
            value = modalities[modality]
            assert tuple(value.shape[-2:]) == (1024, width) and value.numel() == 1024 * width
            assert torch.isfinite(value).all() and value.std() > 0, (label, modality)
            anchor = outputs["eager_b" if label in ("eager_b", "replay_b") else "eager_a"][modality]
            assert torch.equal(value, anchor), f"{label}/{modality}: eager/trace/restoration mismatch"
    assert all(not torch.equal(outputs["eager_a"][m], outputs["eager_b"][m]) for m in ("video", "audio"))
    return outputs


def _verify(base_path, native_path, prompts_path, reference_path=None):
    report_path = native_path.with_suffix(".verified.json")
    report_path.unlink(missing_ok=True)
    _prompts(prompts_path)
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    native = torch.load(native_path, map_location="cpu", weights_only=True)
    baseline = _validate(base, "0", _sha(prompts_path))
    actual = _validate(native, "1", _sha(prompts_path))
    for key in (
        "sources",
        "mesh",
        "tp",
        "ccl_topology",
        "ccl_num_links",
        "diagnostic_hifi4_env",
        "commit",
        "tracked_diff_sha256",
        "source_sha256",
    ):
        assert base[key] == native[key], f"A/B provenance differs: {key}"
    for label in baseline:
        for modality in ("video", "audio"):
            assert torch.equal(baseline[label][modality], actual[label][modality]), (label, modality, "route mismatch")
    reference_metrics = []
    if reference_path:
        ref = torch.load(reference_path, map_location="cpu", weights_only=True)
        assert ref["prompts_sha256"] == _sha(prompts_path) and ref["sources"] == base["sources"]
        for label in ("a", "b"):
            for modality in ("video", "audio"):
                a, b = (
                    actual[f"eager_{label}"][modality].flatten().double(),
                    ref["outputs"][label][modality].flatten().double(),
                )
                assert a.shape == b.shape and torch.isfinite(b).all()
                pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
                assert pcc >= 0.9995, (label, modality, pcc)  # Existing full-encoder reference gate.
                reference_metrics.append({"prompt": label, "modality": modality, "pcc": pcc})
    report = {
        "status": "FULL_ENCODER_EXACT_NONREGRESSION_PASS",
        "base_sha256": _sha(base_path),
        "native_sha256": _sha(native_path),
        "prompts_sha256": _sha(prompts_path),
        "reference_sha256": _sha(reference_path) if reference_path else None,
        "reference_metrics": reference_metrics,
        "commit": native["commit"],
        "mesh": native["mesh"],
        "warm_ms": {"base": base["warm_ms"], "native": native["warm_ms"]},
        "scope": "full Gemma+connectors exact baseline/native eager and changed-prompt A/B/A; not pipeline AV quality",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C04_ENCODER_EXACT_PASS {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--prepare-reference", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    elif args.prepare_reference:
        if not args.prompts:
            parser.error("--prepare-reference requires --prompts")
        _prepare_reference(args.prepare_reference, args.prompts)
    else:
        if not (args.prompts and args.native):
            parser.error("--verify requires --prompts and --native")
        _verify(args.verify, args.native, args.prompts, args.reference)
