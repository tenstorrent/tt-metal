# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-prompt connector whole-row Q/K statistics experiment.

CPU, off lease: --prepare DIR --prompts actual-prompts.json (HF12B + Diffusers).
Broker: C05_INPUTS=<video|audio>-<full|norm0..7>.pt C05_RESULTS=<fresh.pt>
LTX_CONNECTOR_QK_STATS=0|1 pytest <this-file>::test_collect_connector_qk_stats -k galaxy|f07.
CPU, off lease: --verify result.pt --inputs fixture.pt [--baseline baseline.pt].
An isolated connector pass is not full Gemma or generated video/audio acceptance.
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

from models.tt_dit.tests.encoders.gemma.test_gemma_prompt_replay import _prompts, _sha, _sources


def _tensor_sha(value):
    return hashlib.sha256(value.float().contiguous().numpy().tobytes()).hexdigest()


def _interleave(value):
    """Permute each whole head from checkpoint SPLIT to production INTERLEAVED."""
    dim = value.shape[-1]
    return value.reshape(*value.shape[:-1], 32, 2, dim // 64).transpose(-1, -2).reshape(value.shape)


def _prepare(directory, prompts_path):
    from models.tt_dit.tests.encoders.gemma.test_gemma_full import _encode_prompts_reference

    prompts = _prompts(prompts_path)
    checkpoint, gemma, sources = _sources()
    paths = {
        f"{axis}-{part}": directory / f"{axis}-{part}.pt"
        for axis in ("video", "audio")
        for part in ("full", *(f"norm{i}" for i in range(8)))
    }
    assert all(not path.exists() for path in paths.values()), "use fresh fixture paths"
    records = {}

    def observe(connectors):
        for axis, dim in (("video", 4096), ("audio", 2048)):
            connector = getattr(connectors, f"{axis}_connector")
            assert len(connector.transformer_blocks) == 8

            def before(module, args, *, axis=axis, dim=dim):
                assert len(args) == 2 and args[0].shape == (2, 1024, dim)
                features, additive_mask = args
                features = features.detach().bfloat16()
                binary_mask = (additive_mask >= -9000).reshape(2, 1024).long()
                records[f"{axis}-full"] = {"features": features.clone(), "mask": binary_mask}
                # Quantize only the fixture boundary, so the independent CPU
                # connector consumes the same BF16 features as the device.
                return features.float(), additive_mask

            connector.register_forward_pre_hook(before)
            for index, block in enumerate(connector.transformer_blocks):
                record = {"qkv": {}}
                records[f"{axis}-norm{index}"] = record
                for name in ("q", "k"):
                    norm = getattr(block.attn1, f"norm_{name}")
                    assert tuple(norm.normalized_shape) == (dim,), "reference must normalize the whole inner row"
                    assert math.isclose(norm.eps, 1e-6)
                    record[f"{name}_weight"] = _interleave(norm.weight.detach().bfloat16()).clone()
                for name in ("q", "k", "v"):

                    def projected(module, args, value, *, name=name, record=record):
                        value = value.detach().bfloat16()
                        record["qkv"][name] = (_interleave(value) if name != "v" else value).clone()

                    getattr(block.attn1, f"to_{name}").register_forward_hook(projected)

    print("C05_CPU_REFERENCE: real two-prompt HF12B + all sixteen connector blocks", flush=True)
    references = dict(
        zip(
            ("video", "audio"),
            _encode_prompts_reference(checkpoint, gemma, list(prompts.values()), connector_observer=observe),
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    for key, record in records.items():
        axis, component = key.split("-", 1)
        record.update(
            schema=1,
            axis=axis,
            component=component,
            prompts=prompts,
            prompts_sha256=_sha(prompts_path),
            sources=sources,
        )
        if component == "full":
            record["inputs"] = {
                label: {"features": record["features"][i : i + 1].clone(), "mask": record["mask"][i : i + 1].clone()}
                for i, label in enumerate(prompts)
            }
            record["references"] = {
                label: {"output": references[axis][i : i + 1].clone()} for i, label in enumerate(prompts)
            }
            del record["features"], record["mask"]
        else:
            assert set(record["qkv"]) == {"q", "k", "v"}
            record["inputs"], record["references"] = {}, {}
            for i, label in enumerate(prompts):
                values = {name: value[i : i + 1].clone() for name, value in record["qkv"].items()}
                record["inputs"][label] = values
                ref = {}
                for name, value in values.items():
                    value = value.float()
                    if name != "v":
                        value = (
                            value
                            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                            * record[f"{name}_weight"].float()
                        )
                    ref[name] = value.reshape(1, 1024, 32, -1).transpose(1, 2).contiguous()
                record["references"][label] = ref
            del record["qkv"]
        for values in record["references"].values():
            assert all(torch.isfinite(value).all() and value.std() > 0 for value in values.values())
        assert all(
            not torch.equal(record["references"]["a"][name], value) for name, value in record["references"]["b"].items()
        )
        torch.save(record, paths[key])
        print(f"C05_PREPARED {paths[key]} sha256={_sha(paths[key])}", flush=True)


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
    not os.environ.get("C05_INPUTS") and not os.environ.get("C05_RESULTS"),
    reason="opt-in real-prompt connector Q/K experiment",
)
def test_collect_connector_qk_stats(mesh_device, device_params):
    import ttnn
    from models.tt_dit.encoders.gemma.embeddings_connector import ConnectorBlock, EmbeddingsConnector
    from models.tt_dit.encoders.gemma.encoder_pair import _connector_state_dict, _read_connector_checkpoint
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils import cache
    from models.tt_dit.utils.mochi import get_rot_transformation_mat
    from models.tt_dit.utils.tensor import bf16_tensor

    inputs_path, result_path = Path(os.environ["C05_INPUTS"]), Path(os.environ["C05_RESULTS"])
    assert not result_path.exists(), result_path
    fixture = torch.load(inputs_path, map_location="cpu", weights_only=True)
    checkpoint, _, sources = _sources()
    assert fixture["schema"] == 1 and sources == fixture["sources"]
    axis, component = fixture["axis"], fixture["component"]
    assert axis in ("video", "audio") and component in ("full", *(f"norm{i}" for i in range(8)))
    mode = os.environ["LTX_CONNECTOR_QK_STATS"]
    assert mode in ("0", "1")
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", ""), "profile separately"
    mesh_shape = tuple(mesh_device.shape)
    assert mesh_shape in ((4, 8), (2, 4))
    tp, dim = mesh_shape[1], 4096 if axis == "video" else 2048
    # Serving passes vae_ccl_manager to Gemma+connectors: Linear even when
    # the parent Galaxy device fabric is configured as a ring.
    topology = ttnn.Topology.Linear
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(tp, 1))
    ccl = CCLManager(mesh_device, num_links=2, topology=topology)
    if component == "full":
        model = EmbeddingsConnector(
            output_dim=dim, num_blocks=8, num_heads=32, mesh_device=mesh_device, ccl_manager=ccl, parallel_config=pc
        )
        cache.load_model(
            model,
            model_name=Path(checkpoint).stem,
            subfolder=model.weight_cache_subfolder(axis),
            parallel_config=pc,
            mesh_shape=mesh_shape,
            mesh_device=mesh_device,
            dtype="float32",
            sources=[checkpoint],
            get_torch_state_dict=lambda: _connector_state_dict(_read_connector_checkpoint(checkpoint), axis, 8),
        )
        assert all(block._qk_stats == (mode == "1") for block in model.transformer_1d_blocks)
        transform = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
        # Allocate captured constants before any traced activation arena.
        model._rope_cos_sin(1024)
        model._tiled_registers(1024, dim)

        def upload(label):
            sample = fixture["inputs"][label]
            assert sample["features"].shape == (1, 1024, dim)
            indices, keep = model.build_indices(sample["mask"], 1024)
            return {"features": bf16_tensor(sample["features"], device=mesh_device), "indices": indices, "keep": keep}

        def forward(values):
            # EmbeddingsConnector consumes/deallocates features. Copy the stable
            # trace input on device so subsequent eager/trace A/B/A uses live I/O.
            features = ttnn.clone(values["features"])
            return {"output": model(features, values["indices"], values["keep"], trans_mat=transform)}

        def read(values):
            return {name: ttnn.to_torch(ttnn.get_device_tensors(value)[0]) for name, value in values.items()}

    else:
        model = ConnectorBlock(dim, dim * 4, 32, 1e-6, mesh_device, ccl, pc)
        assert model._qk_stats == (mode == "1")
        # Fixture affine and Q/K already use the production permutation. Full
        # connector runs independently exercise the original checkpoint loader.
        for name in ("q", "k"):
            getattr(model, f"{name}_norm").load_torch_state_dict({"weight": fixture[f"{name}_weight"]})

        def upload(label):
            return {
                name: bf16_tensor(value, device=mesh_device, mesh_axis=1, shard_dim=-1)
                for name, value in fixture["inputs"][label].items()
            }

        def forward(values):
            return dict(
                zip(
                    ("q", "k", "v"),
                    model._normalize_qk_and_split_heads(values["q"], values["k"], values["v"], apply_rope=True),
                )
            )

        def read(values):
            # Select mesh coordinates explicitly; do not infer shard order from
            # the device tensor list. Other mesh rows contain replicas.
            outputs = {}
            for name, value in values.items():
                shards = sorted(
                    (int(coord[1]), shard)
                    for coord, shard in zip(value.tensor_topology().mesh_coords(), ttnn.get_device_tensors(value))
                    if int(coord[0]) == 0
                )
                assert [coord for coord, _ in shards] == list(range(tp))
                outputs[name] = torch.cat([ttnn.to_torch(shard) for _, shard in shards], dim=1)
            return outputs

    norms = (
        [norm for block in model.transformer_1d_blocks for norm in (block.q_norm, block.k_norm)]
        if component == "full"
        else [model.q_norm, model.k_norm]
    )
    expected_width = dim // tp if mode == "1" else dim
    assert all(norm.weight.data.shape[-1] == expected_width for norm in norms)

    stable = upload("a")
    outputs, hashes, trace_ms, eager_ms = {}, {}, [], []
    trace = None

    def replace(label):
        uploaded = upload(label)
        for name, value in uploaded.items():
            ttnn.copy(value, stable[name])

    def save(label, values, retain=False):
        hashes[label] = {name: _tensor_sha(value) for name, value in values.items()}
        if retain:
            outputs[label] = {name: value.float().clone() for name, value in values.items()}

    try:
        if capture_only:
            recipes = [forward(stable), forward(stable)]
            pytest.skip("kernel recipes only; no correctness or timing result")
        for label in ("a", "b"):
            replace(label)
            save(f"eager_{label}", read(forward(stable)), retain=True)
        for index in range(5):
            replace("a")
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            value = forward(stable)
            ttnn.synchronize_device(mesh_device)
            eager_ms.append((time.perf_counter() - start) * 1000)
            save(f"eager_timed_a{index}", read(value))
        replace("a")
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            traced_outputs = [forward(stable), forward(stable)]
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for label, sample in (("capture_a", "a"), ("replay_a0", "a"), ("replay_b", "b"), ("replay_a1", "a")):
            replace(sample)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            for slot, value in enumerate(traced_outputs):
                save(f"{label}_{slot}", read(value), retain=slot == 1 and label in ("replay_a0", "replay_b"))
        for index in range(5):
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            trace_ms.append((time.perf_counter() - start) * 1000 / 2)
            for slot, value in enumerate(traced_outputs):
                save(f"trace_timed_a{index}_{slot}", read(value))
        record = {
            "schema": 1,
            "axis": axis,
            "component": component,
            "mode": mode,
            "inputs_sha256": _sha(inputs_path),
            "sources": sources,
            "mesh": list(mesh_shape),
            "topology": str(topology),
            "num_links": ccl.num_links,
            "whole_row": True,
            "standalone_rope": True,
            "capture_only": False,
            "norm_count": len(norms),
            "norm_weight_local_dim": expected_width,
            "binary_sha256": _sha("ttnn/ttnn/_ttnn.so"),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {
                name: _sha(name)
                for name in (
                    __file__,
                    "models/tt_dit/encoders/gemma/embeddings_connector.py",
                    "models/tt_dit/layers/normalization.py",
                )
            },
            "eager_device_ms": eager_ms,
            "paired_trace_per_forward_ms": trace_ms,
            "outputs": outputs,
            "hashes": hashes,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, result_path)
        print(f"C05_CORRECTNESS_PENDING {result_path}; eager_device_ms={eager_ms}; paired_trace_ms={trace_ms}")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)


def _metrics(actual, reference):
    assert actual.shape == reference.shape and torch.isfinite(actual).all() and actual.std() > 0
    a, b = actual.flatten().double(), reference.flatten().double()
    return {
        "pcc": torch.corrcoef(torch.stack((a, b)))[0, 1].item(),
        "max_abs": (a - b).abs().max().item(),
        "relative_l2": ((a - b).norm() / b.norm()).item(),
    }


def _validate(record, fixture, inputs_path):
    assert record["schema"] == 1 and not record["capture_only"] and record["whole_row"] and record["standalone_rope"]
    assert record["mode"] in ("0", "1") and record["inputs_sha256"] == _sha(inputs_path)
    assert record["sources"] == fixture["sources"]
    assert record["axis"] == fixture["axis"] and record["component"] == fixture["component"]
    assert record["mesh"] in ([4, 8], [2, 4])
    dim = 4096 if fixture["axis"] == "video" else 2048
    assert record["norm_count"] == (16 if fixture["component"] == "full" else 2)
    assert record["norm_weight_local_dim"] == (dim // record["mesh"][1] if record["mode"] == "1" else dim)
    for key in ("eager_device_ms", "paired_trace_per_forward_ms"):
        assert len(record[key]) == 5 and all(math.isfinite(t) and t > 0 for t in record[key])
    hashes, outputs = record["hashes"], record["outputs"]
    required = {"eager_a", "eager_b"} | {f"eager_timed_a{i}" for i in range(5)}
    required |= {f"{label}_{i}" for label in ("capture_a", "replay_a0", "replay_b", "replay_a1") for i in range(2)}
    required |= {f"trace_timed_a{i}_{slot}" for i in range(5) for slot in range(2)}
    assert set(hashes) == required and set(outputs) == {"eager_a", "eager_b", "replay_a0_1", "replay_b_1"}
    for label in required:
        expected = hashes["eager_b" if label == "eager_b" or label.startswith("replay_b_") else "eager_a"]
        assert hashes[label] == expected, f"eager/trace or repeated-input drift: {label}"
    assert all(
        hashes["eager_a"][name] != value for name, value in hashes["eager_b"].items()
    ), "changed real prompt did not change every output"
    metrics = {}
    for label, values in outputs.items():
        ref = fixture["references"]["b" if label in ("eager_b", "replay_b_1") else "a"]
        assert set(values) == set(ref) == set(hashes[label])
        metrics[label] = {}
        for name, value in values.items():
            expected_shape = (1, 1024, dim) if fixture["component"] == "full" else (1, 32, 1024, dim // 32)
            assert tuple(value.shape) == expected_shape, f"not the real connector shape: {value.shape}"
            assert _tensor_sha(value) == hashes[label][name], "saved tensor hash mismatch"
            metrics[label][name] = _metrics(value, ref[name])
            if name == "v":
                assert torch.equal(value, ref[name]), "V-only head split must match the CPU layout exactly"
            assert metrics[label][name]["pcc"] >= (0.998 if fixture["component"] == "full" else 0.999), metrics[label][
                name
            ]
    return metrics


def _verify(result_path, inputs_path, baseline_path=None):
    verdict = result_path.with_suffix(".quality.json")
    verdict.unlink(missing_ok=True)
    fixture = torch.load(inputs_path, map_location="cpu", weights_only=True)
    record = torch.load(result_path, map_location="cpu", weights_only=True)
    metrics = _validate(record, fixture, inputs_path)
    baseline_metrics = {}
    if baseline_path is not None:
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        _validate(baseline, fixture, inputs_path)
        assert baseline["mode"] == "0" and record["mode"] == "1"
        for key in ("mesh", "topology", "num_links", "commit", "source_sha256", "binary_sha256"):
            assert baseline[key] == record[key], f"A/B provenance mismatch: {key}"
        for label, values in record["outputs"].items():
            baseline_metrics[label] = {}
            for name, value in values.items():
                baseline_metrics[label][name] = _metrics(value, baseline["outputs"][label][name])
                assert baseline_metrics[label][name]["pcc"] >= 0.9995, baseline_metrics[label][name]
                if name == "v":
                    assert torch.equal(value, baseline["outputs"][label][name]), "V head split changed"
    report = {
        "status": "C05_ISOLATED_PASS",
        "result_sha256": _sha(result_path),
        "inputs_sha256": _sha(inputs_path),
        "baseline_sha256": None if baseline_path is None else _sha(baseline_path),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "eager_device_ms": record["eager_device_ms"],
        "paired_trace_per_forward_ms": record["paired_trace_per_forward_ms"],
    }
    verdict.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C05_ISOLATED_PASS {verdict}; full encoder and generated AV gates remain required")


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
        if args.prompts is None:
            parser.error("--prepare requires --prompts")
        _prepare(args.prepare, args.prompts)
    else:
        if args.inputs is None:
            parser.error("--verify requires --inputs")
        _verify(args.verify, args.inputs, args.baseline)
