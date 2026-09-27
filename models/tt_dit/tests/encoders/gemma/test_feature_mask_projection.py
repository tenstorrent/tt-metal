# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in C13 feature-only experiment with real Gemma states and learned weights.

CPU only: --prepare DIR --prompts prompts.json (two distinct real prompts a/b).
Broker: C13_INPUTS=DIR C13_RESULTS=fresh.pt LTX_FEATURE_MASK_AFTER_PROJECTION=0|1
pytest this_file::test_collect_feature_mask_projection -k galaxy|f07.
CPU only: --verify result.pt --inputs DIR [--baseline baseline.pt].
This verifies isolated literal equivalence; full encoder and AV quality remain required.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

CASES = ("a", "b", "all_valid", "all_pad", "tile_boundary")
SOURCE_FILES = (
    "models/tt_dit/encoders/gemma/feature_extractor.py",
    "models/tt_dit/layers/linear.py",
    "models/tt_dit/layers/module.py",
    "models/tt_dit/utils/tracing.py",
    "models/tt_dit/tests/encoders/gemma/test_feature_mask_projection.py",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bits(value):
    assert value.dtype == torch.bfloat16
    return value.contiguous().view(torch.int16)


def _tensor_sha(value):
    return hashlib.sha256(_bits(value).numpy().tobytes()).hexdigest()


def _diagnostic_metrics(actual, reference):
    a, b = actual.float().flatten().double(), reference.float().flatten().double()
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    pcc_defined = a.numel() > 1 and bool(a.std() > 0 and b.std() > 0)
    reference_norm = b.norm().item()
    relative_l2_defined = reference_norm > 0
    return {
        "pcc_diagnostic": torch.corrcoef(torch.stack((a, b)))[0, 1].item() if pcc_defined else None,
        "pcc_defined": pcc_defined,
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": (a - b).norm().item() / reference_norm if relative_l2_defined else None,
        "relative_l2_defined": relative_l2_defined,
    }


def _tracked_source_binding():
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", *args], text=True).strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    pin_path = os.environ.get("C13_DEPENDENCY_PINS")
    if pin_path is None:
        assert not status, "tracked source edits require exact reviewed dependency pins"
        return {"tracked_status": status, "dependency_pins_sha256": None}
    expected_sha = os.environ["C13_DEPENDENCY_PINS_SHA256"]
    assert _sha(pin_path) == expected_sha, "dependency pin file changed"
    pins = json.loads(Path(pin_path).read_text())
    allowed = {"tt_metal/third_party/tracy", "tt_metal/third_party/tt-cluster-descriptors", "tt_metal/third_party/umd"}
    assert set(pins["dependencies"]) == allowed
    assert status == pins["tracked_status"]
    assert {line.strip() for line in status.splitlines()} == {"T " + path for path in allowed}
    assert _sha("ttnn/ttnn/_ttnn.so") == pins["native_sha256"]
    for name, expected in pins["dependencies"].items():
        path = Path(name)
        assert path.is_symlink() and os.readlink(path) == expected["link"]
        assert str(path.resolve(strict=True)) == expected["resolved"]
        assert git("-C", str(path), "rev-parse", "HEAD") == expected["commit"]
        assert git("-C", str(path), "status", "--porcelain", "--untracked-files=no") == expected["tracked_status"] == ""
    return {"tracked_status": status, "dependency_pins_sha256": expected_sha, "dependencies": pins["dependencies"]}


def _sources():
    # Match production cache.source_id without importing TTNN in CPU preparation.
    def identity(path):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        if path.parent.name == "blobs" and len(path.name) == 64 and all(c in "0123456789abcdef" for c in path.name):
            source_id = "sha256:" + path.name
        else:
            source_id = f"stat:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}"
        return {"resolved": str(path), "source_id": source_id}

    checkpoint = Path(os.environ["LTX_CHECKPOINT"]).resolve(strict=True)
    gemma = Path(os.environ["GEMMA_PATH"]).resolve(strict=True)
    weights = sorted(gemma.glob("*.safetensors"))
    assert checkpoint.is_file() and weights
    sources = {"ltx": identity(checkpoint)}
    for path in weights + sorted(gemma.glob("*.json")) + sorted(gemma.glob("*.model")):
        sources[f"gemma/{path.name}"] = identity(path)
    return checkpoint, gemma, sources


def _mask(case, masks):
    if case in ("a", "b"):
        return masks[case]
    result = torch.ones((1, 1024), dtype=torch.int64)
    if case == "all_pad":
        result.zero_()
    elif case == "tile_boundary":
        result[:, :31] = 0
        result[:, 32:34] = 0
        result[:, 63:65] = 0
        result[:, 1023:] = 0
    else:
        assert case == "all_valid"
    return result


def _prepare(directory, prompts_path):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert not directory.exists(), "use a fresh preparation directory"
    prompts = json.loads(prompts_path.read_text())
    assert set(prompts) == {"a", "b"} and all(isinstance(p, str) and p.strip() for p in prompts.values())
    assert prompts["a"] != prompts["b"]
    checkpoint, gemma, sources = _sources()
    tokenizer = AutoTokenizer.from_pretrained(gemma, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        [prompts[k] for k in ("a", "b")],
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    model = AutoModelForCausalLM.from_pretrained(gemma, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    print("C13_CPU_PREPARE: real HF Gemma, two prompts, all 49 hidden states", flush=True)
    with torch.inference_mode():
        output = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask, output_hidden_states=True)
    assert len(output.hidden_states) == 49
    states = {
        label: torch.stack([h[i : i + 1].contiguous() for h in output.hidden_states]).clone()
        for i, label in enumerate(("a", "b"))
    }
    del output, model
    masks = {label: tokens.attention_mask[i : i + 1].clone() for i, label in enumerate(("a", "b"))}
    for label, value in states.items():
        assert value.shape == (49, 1, 1024, 3840) and value.dtype == torch.bfloat16
        assert torch.isfinite(value).all() and 0 < masks[label].sum() < 1024
    assert not torch.equal(_bits(states["a"]), _bits(states["b"]))
    weights, raw_hashes = {}, {}
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        for axis, dim in (("video", 4096), ("audio", 2048)):
            for suffix in ("weight", "bias"):
                key = f"text_embedding_projection.{axis}_aggregate_embed.{suffix}"
                raw = stream.get_tensor(key).bfloat16()
                raw_hashes[key] = _tensor_sha(raw)
                if suffix == "weight":
                    assert raw.shape == (dim, 188160)
                    # Independent checkpoint D-major -> production layer-major layout.
                    raw = raw.reshape(dim, 3840, 49).permute(0, 2, 1).reshape(dim, 188160).contiguous()
                else:
                    assert raw.numel() == dim
                    raw = raw.reshape(1, dim)
                weights[f"{axis}_aggregate_embed.{suffix}"] = raw
    references = {}
    for case in CASES:
        value = states["b" if case == "b" else "a"]
        normalized = []
        for layer in value:
            layer = layer.float()
            normalized.append((layer * torch.rsqrt(layer.square().mean(-1, keepdim=True) + 1e-6)).bfloat16())
        concat = torch.cat(normalized, dim=-1)
        concat = (concat * _mask(case, masks).unsqueeze(-1)).bfloat16()
        references[case] = {}
        for axis, dim in (("video", 4096), ("audio", 2048)):
            scaled = (concat * math.sqrt(dim / 3840)).bfloat16().float()
            references[case][axis] = torch.nn.functional.linear(
                scaled,
                weights[f"{axis}_aggregate_embed.weight"].float(),
                weights[f"{axis}_aggregate_embed.bias"].float().flatten(),
            ).contiguous()
        print(f"C13_CPU_REFERENCE {case} complete", flush=True)
    assert _sources()[2] == sources, "model source changed during preparation"
    directory.mkdir(parents=True)
    (directory / "preparer-source.py").write_bytes(Path(__file__).read_bytes())
    for label, value in states.items():
        torch.save(value, directory / f"states-{label}.pt")
    torch.save({"masks": masks, "tokens": tokens.input_ids}, directory / "tokens.pt")
    torch.save(weights, directory / "weights.pt")
    torch.save(references, directory / "references.pt")
    manifest = {
        "schema": 1,
        "prompts": prompts,
        "prompts_sha256": _sha(prompts_path),
        "sources": sources,
        "raw_projection_bf16_sha256": raw_hashes,
        "cases": list(CASES),
        "preparer_sha256": _sha(__file__),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
        "reference_scope": "independent BF16-boundary RMS/projection diagnostic; no new quality threshold",
        "files": {p.name: _sha(p) for p in sorted(directory.glob("*.pt"))},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"C13_PREPARED {directory} manifest_sha256={_sha(directory / 'manifest.json')}", flush=True)


def _fixture(directory, *, references=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["schema"] == 1 and manifest["cases"] == list(CASES)
    assert _sha(directory / "preparer-source.py") == manifest["preparer_sha256"]
    expected = {"states-a.pt", "states-b.pt", "tokens.pt", "weights.pt", "references.pt"}
    assert set(manifest["files"]) == expected
    names = expected if references else expected - {"references.pt"}
    for name in names:
        assert _sha(directory / name) == manifest["files"][name], f"fixture changed: {name}"
    return manifest


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    common = {"l1_small_size": 8192, "trace_region_size": 100_000_000}
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
    not os.environ.get("C13_INPUTS") and not os.environ.get("C13_RESULTS"), reason="opt-in C13 experiment"
)
def test_collect_feature_mask_projection(mesh_device, device_params):
    import ttnn
    from models.tt_dit.encoders.gemma.feature_extractor import GemmaFeatureExtractor
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tracing import Tracer

    directory, result = Path(os.environ["C13_INPUTS"]), Path(os.environ["C13_RESULTS"])
    assert not result.exists(), "use a fresh result path"
    manifest = _fixture(directory)
    # Every model tensor comes from this sealed fixture, not host checkpoint paths.
    # Preserve the original preparation source identities across host transfers.
    assert manifest["sources"]["ltx"] and any(k.startswith("gemma/") for k in manifest["sources"])
    mode = os.environ["LTX_FEATURE_MASK_AFTER_PROJECTION"]
    assert mode in ("0", "1")
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", ""), "profile separately"
    source_binding = _tracked_source_binding()
    shape = tuple(mesh_device.shape)
    assert shape in ((4, 8), (2, 4))
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(shape[1], 1))
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    model = GemmaFeatureExtractor(
        input_dim=188160,
        embedding_dim=3840,
        video_dim=4096,
        audio_dim=2048,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        parallel_config=pc,
    )
    assert model._mask_after_projection == (mode == "1")
    weights = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    model.load_torch_state_dict(weights)
    del weights
    states = {
        label: torch.load(directory / f"states-{label}.pt", map_location="cpu", weights_only=True)
        for label in ("a", "b")
    }
    masks = torch.load(directory / "tokens.pt", map_location="cpu", weights_only=True)["masks"]
    host_states = {
        label: [ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for h in value]
        for label, value in states.items()
    }
    host_masks = {
        case: ttnn.from_torch(_mask(case, masks).float().unsqueeze(-1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for case in CASES
    }
    stable_states = [ttnn.to_device(h, mesh_device) for h in host_states["a"]]
    stable_mask = ttnn.to_device(host_masks["a"], mesh_device)
    del states
    physical_ids = sorted(int(x) for x in mesh_device.get_device_ids())
    assert len(set(physical_ids)) == math.prod(shape)
    outputs, hashes, samples = {}, {}, []
    tracer = Tracer(model, device=mesh_device, prep_run=True, clone_prep_inputs=False)

    def replace(case):
        for host, device in zip(host_states["b" if case == "b" else "a"], stable_states):
            ttnn.copy_host_to_device_tensor(host, device)
        ttnn.copy_host_to_device_tensor(host_masks[case], stable_mask)

    def retain(label, values):
        outputs[label], hashes[label] = {}, {}
        observed = {}
        try:
            for axis, value in zip(("video", "audio"), values):
                shards = ttnn.get_device_tensors(value)
                coords = list(value.tensor_topology().mesh_coords())
                assert len(shards) == len(coords) == math.prod(shape)
                rows = []
                for coord, shard in zip(coords, shards):
                    actual = ttnn.to_torch(shard).contiguous()
                    key = axis + "/" + ",".join(str(int(c)) for c in coord)
                    observed[key] = actual.clone()
                    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
                    rows.append({"coord": [int(c) for c in coord], "sha256": _tensor_sha(actual)})
                    if len(rows) == 1:
                        outputs[label][axis] = actual.clone()
                assert {tuple(r["coord"]) for r in rows} == {(i, j) for i in range(shape[0]) for j in range(shape[1])}
                assert len({r["sha256"] for r in rows}) == 1, "mesh replica mismatch"
                hashes[label][axis] = rows
        except BaseException as error:
            # A failed native gate must retain the actual tensors for debugging.
            # This artifact is explicitly invalid and cannot pass _validate.
            result.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "status": "INVALID_COLLECTION_FAILURE",
                    "error": repr(error),
                    "label": label,
                    "observed_shards": observed,
                    "earlier_outputs": outputs,
                    "earlier_hashes": hashes,
                    "mode": mode,
                    "mesh": list(shape),
                    "physical_ids": physical_ids,
                    "manifest_sha256": _sha(directory / "manifest.json"),
                    "source_sha256": {p: _sha(p) for p in SOURCE_FILES},
                    "binary_sha256": _sha("ttnn/ttnn/_ttnn.so"),
                },
                result.with_suffix(".failure.pt"),
            )
            raise

    try:
        if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"):
            replace("a")
            model(stable_states, stable_mask)
            tracer(stable_states, stable_mask)
            pytest.skip("kernel recipes only; no correctness or timing artifact")
        for case in CASES:
            replace(case)
            values = model(stable_states, stable_mask)
            retain(f"eager_{case}", values)
            for value in values:
                ttnn.deallocate(value)
        replace("a")
        retain("capture_a", tracer(stable_states, stable_mask))
        for case in (*CASES, "a"):
            replace(case)
            retain(
                "restored_a" if case == "a" and "replay_a" in outputs else f"replay_{case}",
                tracer(stable_states, stable_mask),
            )
        for index in range(5):
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            values = tracer(stable_states, stable_mask)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000)
            retain(f"timed_a{index}", values)
        record = {
            "schema": 1,
            "mode": mode,
            "manifest_sha256": _sha(directory / "manifest.json"),
            "sources": manifest["sources"],
            "mesh": list(shape),
            "physical_ids": physical_ids,
            "topology": "Linear",
            "num_links": 2,
            "capture_only": False,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {p: _sha(p) for p in SOURCE_FILES},
            "tracked_source_binding": source_binding,
            "binary_sha256": _sha("ttnn/ttnn/_ttnn.so"),
            "cgroup": Path("/proc/self/cgroup").read_text(),
            "trace_device_ms": samples,
            "timing_boundary": "resident inputs; blocking whole-feature Tracer plus synchronize; excludes uploads/D2H",
            "outputs": outputs,
            "hashes": hashes,
        }
        result.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, result)
        print(f"C13_EQUIVALENCE_PENDING {result} trace_device_ms={samples}", flush=True)
    finally:
        tracer.release_trace()


def _validate(record, directory):
    manifest = _fixture(directory, references=True)
    assert record["schema"] == 1 and not record["capture_only"] and record["mode"] in ("0", "1")
    assert record["manifest_sha256"] == _sha(directory / "manifest.json") and record["sources"] == manifest["sources"]
    assert record["mesh"] in ([4, 8], [2, 4]) and record["topology"] == "Linear" and record["num_links"] == 2
    assert len(set(record["physical_ids"])) == math.prod(record["mesh"])
    assert set(record["source_sha256"]) == set(SOURCE_FILES)
    assert len(record["trace_device_ms"]) == 5 and all(math.isfinite(x) and x > 0 for x in record["trace_device_ms"])
    expected = (
        {f"{phase}_{case}" for phase in ("eager", "replay") for case in CASES}
        | {"capture_a", "restored_a"}
        | {f"timed_a{i}" for i in range(5)}
    )
    assert set(record["outputs"]) == set(record["hashes"]) == expected
    weights = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
    masks = torch.load(directory / "tokens.pt", map_location="cpu", weights_only=True)["masks"]
    references = torch.load(directory / "references.pt", map_location="cpu", weights_only=True)
    metrics = {}
    for label, values in record["outputs"].items():
        case = label.split("_", 1)[1] if label.startswith(("eager_", "replay_")) else "a"
        assert set(values) == set(record["hashes"][label]) == {"video", "audio"}
        metrics[label] = {}
        for axis, dim in (("video", 4096), ("audio", 2048)):
            value = values[axis]
            assert value.shape == (1, 1024, dim) and value.dtype == torch.bfloat16 and torch.isfinite(value).all()
            rows = record["hashes"][label][axis]
            coords = {(i, j) for i in range(record["mesh"][0]) for j in range(record["mesh"][1])}
            assert len(rows) == len(coords) and {tuple(x["coord"]) for x in rows} == coords
            assert all(x["sha256"] == _tensor_sha(value) for x in rows)
            assert torch.equal(
                _bits(value), _bits(record["outputs"][f"eager_{case}"][axis])
            ), f"eager/trace/restoration mismatch: {label}/{axis}"
            padded = _mask(case, masks) == 0
            bias = weights[f"{axis}_aggregate_embed.bias"].expand(1, 1024, dim)
            assert torch.equal(
                _bits(value[padded]), _bits(bias[padded])
            ), f"learned padding bias changed: {label}/{axis}"
            metrics[label][axis] = _diagnostic_metrics(value, references[case][axis])
    for axis in ("video", "audio"):
        assert not torch.equal(_bits(record["outputs"]["eager_a"][axis]), _bits(record["outputs"]["eager_b"][axis]))
    return metrics


def _verify(result, directory, baseline=None):
    verdict = result.with_suffix(".equivalence.json")
    verdict.unlink(missing_ok=True)
    record = torch.load(result, map_location="cpu", weights_only=True)
    metrics = _validate(record, directory)
    if baseline is not None:
        other = torch.load(baseline, map_location="cpu", weights_only=True)
        _validate(other, directory)
        assert record["mode"] == "1" and other["mode"] == "0"
        assert record["cgroup"] != other["cgroup"], "independent broker runs required"
        for key in (
            "manifest_sha256",
            "sources",
            "mesh",
            "physical_ids",
            "topology",
            "num_links",
            "commit",
            "source_sha256",
            "tracked_source_binding",
            "binary_sha256",
        ):
            assert record[key] == other[key], f"baseline provenance differs: {key}"
        for label, values in record["outputs"].items():
            for axis, value in values.items():
                assert torch.equal(
                    _bits(value), _bits(other["outputs"][label][axis])
                ), f"literal A/B mismatch: {label}/{axis}"
    else:
        assert record["mode"] == "0", "candidate requires a same-source baseline"
    report = {
        "status": "C13_ISOLATED_LITERAL_EQUIVALENCE" if baseline else "C13_BASELINE_REPLAY_VERIFIED",
        "claim_limit": "CPU metrics diagnostic; broker/source binding review and full encoder/AV quality still required",
        "result_sha256": _sha(result),
        "manifest_sha256": _sha(directory / "manifest.json"),
        "baseline_sha256": _sha(baseline) if baseline else None,
        "verifier_sha256": _sha(__file__),
        "metrics": metrics,
        "trace_device_ms": record["trace_device_ms"],
    }
    verdict.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.prepare:
        assert args.prompts
        _prepare(args.prepare, args.prompts)
    else:
        assert args.inputs
        _verify(args.verify, args.inputs, args.baseline)
