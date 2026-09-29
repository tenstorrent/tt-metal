# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in C06 real-latent GroupNorm/upsampler evidence with an offline CPU oracle.

Prepare on CPU: --prepare DIR --latent actual-stage1.pt --case galaxy-padded.
LTX_UPSAMPLER_CHECKPOINT must name the local real spatial-upscaler safetensors.
Run each C06_COMPONENT=full|pre_norm|post_norm and LTX_UPSAMPLER_W_STATS_GN=0|1
in a separate broker process with C06_INPUTS=<case-component.pt>, C06_RESULTS=<new.pt>.
Verify outside the reservation: --verify candidate.pt --inputs fixture.pt --baseline base.pt.
This isolates the upsampler; it does not establish full video/audio pipeline quality.
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

CASES = {"galaxy-padded": (20, 32), "galaxy-crop": (17, 30), "f07-padded": (18, 32), "f07-crop": (17, 30)}
COMPONENTS = ("full", "pre_norm", "post_norm")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value):
    return hashlib.sha256(value.float().contiguous().numpy().tobytes()).hexdigest()


def _checkpoint():
    path = Path(os.environ["LTX_UPSAMPLER_CHECKPOINT"]).expanduser().absolute()
    assert path.is_file(), path
    return path


def _prepare(directory, latent_path, case):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from safetensors import safe_open
    from safetensors.torch import load_file

    from models.tt_dit.utils.cache import source_id

    paths = {component: directory / f"{case}-{component}.pt" for component in COMPONENTS}
    assert all(not path.exists() for path in paths.values()), "use fresh fixture paths"
    actual = torch.load(latent_path, map_location="cpu", weights_only=True)
    assert actual["schema"] == 1 and actual["kind"] == "actual_stage1_unnormalized_before_replicate_padding"
    x = actual["input"].float()
    assert tuple(x.shape) == (1, 128, 19, 17, 30), x.shape
    assert torch.isfinite(x).all() and x.std() > 0
    assert x.mean(dim=(0, 2, 3, 4)).abs().max() > 0, "need actual nonzero-mean channel statistics"
    checkpoint = _checkpoint()
    assert source_id(checkpoint) == actual["provenance"]["upsampler_source_id"], "tap/upscaler identity mismatch"
    with safe_open(checkpoint, framework="pt") as file:
        config = json.loads(file.metadata()["config"])
    assert config["in_channels"] == 128 and config["mid_channels"] == 1024 and config["num_blocks_per_stage"] == 4
    model = (
        LTX2LatentUpsamplerModel(
            in_channels=config["in_channels"],
            mid_channels=config["mid_channels"],
            num_blocks_per_stage=4,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            rational_spatial_scale=2.0,
            use_rational_resampler=False,
        )
        .float()
        .eval()
    )
    model.load_state_dict(load_file(checkpoint))
    height, width = CASES[case]
    x = torch.nn.functional.pad(x, (0, width - 30, 0, height - 17), mode="replicate")
    # Match the actual first upload's BF16 rounding, before reference inference.
    latent_inputs = {"a": x.bfloat16(), "b": torch.roll(x, shifts=3, dims=2).bfloat16()}
    records = {component: {"inputs": {}, "references": {}} for component in COMPONENTS}
    hooks = []
    current = None
    for component, norm in (("pre_norm", model.initial_norm), ("post_norm", model.post_upsample_res_blocks[0].norm1)):
        records[component]["affine"] = {
            name: getattr(norm, name).detach().bfloat16().clone() for name in ("weight", "bias")
        }

        def before_norm(module, args, *, component=component):
            value = args[0].detach().bfloat16()
            records[component]["inputs"][current] = value.clone()
            affine = records[component]["affine"]
            records[component]["references"][current] = torch.nn.functional.group_norm(
                value.float(), 32, affine["weight"].float(), affine["bias"].float(), eps=1e-5
            )

        hooks.append(norm.register_forward_pre_hook(before_norm))
    with torch.no_grad():
        for label, value in latent_inputs.items():
            current = label
            print(f"C06_CPU_REFERENCE {case} input={label} shape={tuple(value.shape)}", flush=True)
            records["full"]["inputs"][label] = value
            records["full"]["references"][label] = model(value.float()).detach()
    for hook in hooks:
        hook.remove()
    directory.mkdir(parents=True, exist_ok=True)
    for component, record in records.items():
        assert set(record["inputs"]) == set(record["references"]) == {"a", "b"}
        assert not torch.equal(record["references"]["a"], record["references"]["b"])
        for reference in record["references"].values():
            assert torch.isfinite(reference).all() and reference.std() > 0
        record.update(
            schema=1,
            case=case,
            component=component,
            config=config,
            checkpoint_source_id=source_id(checkpoint),
            real_latent_sha256=_sha(latent_path),
            provenance=actual["provenance"],
        )
        torch.save(record, paths[component])
        print(f"C06_PREPARED {paths[component]} sha256={_sha(paths[component])}", flush=True)


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    common = {"trace_region_size": 100_000_000}
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
    "C06_INPUTS" not in os.environ and "C06_RESULTS" not in os.environ,
    reason="opt-in real-latent upsampler stats experiment",
)
def test_collect_upsampler_wstats(mesh_device, device_params):
    import ttnn
    from models.tt_dit.models.upsampler.latent_upsampler_ltx import (
        LTXLatentUpsampler,
        _gn_hw_sharded,
        _upsampler_group_norm,
        _WidthStatsGroupNorm3D,
    )
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.cache import source_id
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
    from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

    inputs_path, output_path = Path(os.environ["C06_INPUTS"]), Path(os.environ["C06_RESULTS"])
    assert not output_path.exists(), output_path
    data = torch.load(inputs_path, map_location="cpu", weights_only=True)
    component = os.environ["C06_COMPONENT"]
    mode = os.environ["LTX_UPSAMPLER_W_STATS_GN"]
    assert component in COMPONENTS and component == data["component"] and mode in ("0", "1")
    if component == "full":
        # The serving upsampler receives host FP32 after unnormalization. Preserve
        # that upload/conversion cost while using the oracle's quantized values.
        data["inputs"] = {key: value.float() for key, value in data["inputs"].items()}
    assert data["schema"] == 1 and data["case"] in CASES
    expected_parent = (4, 8) if data["case"].startswith("galaxy") else (2, 4)
    assert tuple(mesh_device.shape) == expected_parent
    checkpoint = _checkpoint()
    assert data["checkpoint_source_id"] == source_id(checkpoint)
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0"), "profile separately"
    hf, wf = expected_parent
    pc = VaeHWParallelConfig(height_parallel=ParallelFactor(hf, 0), width_parallel=ParallelFactor(wf, 1))
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    _, channels, frames, height, width = data["inputs"]["a"].shape
    if component == "full":
        dit_pc = DiTParallelConfig(
            cfg_parallel=ParallelFactor(1, 0),
            sequence_parallel=ParallelFactor(wf, 1),
            tensor_parallel=ParallelFactor(hf, 0),
        )
        model = LTXLatentUpsampler.from_checkpoint(
            str(checkpoint),
            input_hw=(height, width),
            latent_frames=frames,
            mesh_device=mesh_device,
            parallel_config=pc,
            ccl_manager=ccl,
            dit_parallel_config=dit_pc,
        )
        model.reload_weights()

        def forward(value):
            return model.forward_device(value, height, width)[0]

        output_height, output_width = height * 2, width * 2
        norms = [model.initial_norm] + [
            norm
            for blocks in (model.res_blocks, model.post_upsample_res_blocks)
            for block in blocks
            for norm in (block.norm1, block.norm2)
        ]
    else:
        model = _upsampler_group_norm(
            num_channels=channels,
            num_groups=32,
            input_nhw=frames * height * width,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat16,
            parallel_config=pc,
            ccl_manager=ccl,
        )
        model.load_torch_state_dict(data["affine"])

        def forward(value):
            return _gn_hw_sharded(model, value, pc, ccl, height, width)

        output_height, output_width = height, width
        norms = [model]
    expected_fast = (
        (8 if component == "full" else int(component == "post_norm"))
        if mode == "1" and data["case"].endswith("padded")
        else 0
    )
    assert sum(isinstance(norm, _WidthStatsGroupNorm3D) for norm in norms) == expected_fast

    def upload(label):
        value = data["inputs"][label].permute(0, 2, 3, 4, 1).contiguous()
        value, _ = conv_pad_height(value, hf)
        value, _ = conv_pad_width(value, wf)
        return typed_tensor_2dshard(
            value, mesh_device, shard_mapping={0: 2, 1: 3}, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )

    def read(value):
        out = fast_device_to_host(value, mesh_device, [2, 3], ccl_manager=ccl)
        return out[:, :, :output_height, :output_width, :].permute(0, 4, 1, 2, 3).contiguous()

    x = upload("a")  # stable input allocated before every captured activation
    trace = None
    outputs, hashes, eager_ms, trace_ms = {}, {}, [], []

    def save(label, value, retain=False):
        value = value.float()  # evidence conversion stays outside timing
        hashes[label] = _tensor_sha(value)
        if retain:
            outputs[label] = value

    try:
        if capture_only:
            recipes = [forward(x), forward(x)]
            pytest.skip("kernel recipe capture only; no correctness or timing result")
        # Eager A/B and inclusive timings all precede device trace capture.
        for label in ("a", "b"):
            temp = upload(label)
            ttnn.copy(temp, x)
            del temp
            save(f"eager_{label}", read(forward(x)), retain=True)
        for index in range(5):
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            eager_input = upload("a")
            eager_output = read(forward(eager_input))
            ttnn.synchronize_device(mesh_device)
            eager_ms.append((time.perf_counter() - start) * 1000)
            save(f"eager_timed_a{index}", eager_output)
            del eager_input, eager_output
        temp = upload("a")
        ttnn.copy(temp, x)
        del temp
        # Pair forwards so both semaphore/stats-buffer variants are captured and
        # replay ends on its starting parity. Keep both output tensors alive.
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            traced_outputs = [forward(x), forward(x)]
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for label, input_label in (("capture_a", "a"), ("replay_a0", "a"), ("replay_b", "b"), ("replay_a1", "a")):
            temp = upload(input_label)
            ttnn.copy(temp, x)
            del temp
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            for index, output in enumerate(traced_outputs):
                save(f"{label}_{index}", read(output), retain=label in ("replay_a0", "replay_b") and index == 1)
        for index in range(5):
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            trace_ms.append((time.perf_counter() - start) * 1000 / 2)
            for slot, output in enumerate(traced_outputs):
                save(f"trace_timed_a{index}_{slot}", read(output))
        record = {
            "schema": 1,
            "mode": mode,
            "component": component,
            "case": data["case"],
            "inputs_sha256": _sha(inputs_path),
            "checkpoint_source_id": source_id(checkpoint),
            "real_latent_sha256": data["real_latent_sha256"],
            "mesh": list(mesh_device.shape),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha("models/tt_dit/models/upsampler/latent_upsampler_ltx.py"),
            "stats_norms_per_forward": expected_fast,
            "eager_inclusive_ms": eager_ms,
            "paired_trace_per_forward_ms": trace_ms,
            "outputs": outputs,
            "hashes": hashes,
            "capture_only": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, output_path)
        print(f"C06_CORRECTNESS_PENDING {output_path}; eager_ms={eager_ms}; paired_trace_per_forward_ms={trace_ms}")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)


def _validate(record, data, inputs_path):
    assert record["schema"] == 1 and not record["capture_only"]
    assert record["inputs_sha256"] == _sha(inputs_path)
    assert record["checkpoint_source_id"] == data["checkpoint_source_id"]
    assert record["real_latent_sha256"] == data["real_latent_sha256"]
    assert record["case"] == data["case"] and record["component"] == data["component"]
    for key in ("eager_inclusive_ms", "paired_trace_per_forward_ms"):
        assert len(record[key]) == 5 and all(math.isfinite(t) and t > 0 for t in record[key])
    hashes, outputs = record["hashes"], record["outputs"]
    required = {"eager_a", "eager_b"} | {f"eager_timed_a{i}" for i in range(5)}
    required |= {f"{label}_{i}" for label in ("capture_a", "replay_a0", "replay_b", "replay_a1") for i in range(2)}
    required |= {f"trace_timed_a{i}_{slot}" for i in range(5) for slot in range(2)}
    assert set(hashes) == required and set(outputs) == {"eager_a", "eager_b", "replay_a0_1", "replay_b_1"}
    for label in required:
        expected = hashes["eager_b" if label == "eager_b" or label.startswith("replay_b_") else "eager_a"]
        assert hashes[label] == expected, f"eager/trace or repeated-input drift: {label}"
    assert hashes["eager_a"] != hashes["eager_b"], "changed input did not change output"
    metrics = {}
    for label, value in outputs.items():
        assert _tensor_sha(value) == hashes[label], "saved tensor hash mismatch"
        ref = data["references"]["b" if label in ("eager_b", "replay_b_1") else "a"]
        assert value.shape == ref.shape and torch.isfinite(value).all() and value.std() > 0
        a, b = ref.flatten().double(), value.flatten().double()
        pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
        metrics[label] = {
            "pcc": pcc,
            "max_abs": (a - b).abs().max().item(),
            "relative_l2": ((a - b).norm() / a.norm()).item(),
        }
        assert pcc >= (0.99 if data["component"] == "full" else 0.9993), metrics[label]
    return metrics


def _verify(result_path, inputs_path, baseline_path=None):
    verdict = result_path.with_suffix(".quality.json")
    verdict.unlink(missing_ok=True)
    data = torch.load(inputs_path, map_location="cpu", weights_only=True)
    record = torch.load(result_path, map_location="cpu", weights_only=True)
    metrics = _validate(record, data, inputs_path)
    baseline_metrics = {}
    if baseline_path:
        base = torch.load(baseline_path, map_location="cpu", weights_only=True)
        _validate(base, data, inputs_path)
        assert base["mode"] == "0" and record["mode"] == "1"
        for key in ("mesh", "commit", "source_sha256", "model_source_sha256"):
            assert base[key] == record[key], f"A/B provenance mismatch: {key}"
        for label, value in record["outputs"].items():
            reference = base["outputs"][label]
            a, b = reference.flatten().double(), value.flatten().double()
            pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
            baseline_metrics[label] = {
                "pcc": pcc,
                "max_abs": (a - b).abs().max().item(),
                "relative_l2": ((a - b).norm() / a.norm()).item(),
            }
            assert pcc >= (0.99 if data["component"] == "full" else 0.9995), baseline_metrics[label]
            if data["case"].endswith("crop") or data["component"] == "pre_norm":
                assert torch.equal(reference, value), "fallback must be bit-identical"
    report = {
        "status": "C06_ISOLATED_PASS",
        "result_sha256": _sha(result_path),
        "inputs_sha256": _sha(inputs_path),
        "baseline_sha256": None if baseline_path is None else _sha(baseline_path),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "eager_inclusive_ms": record["eager_inclusive_ms"],
        "paired_trace_per_forward_ms": record["paired_trace_per_forward_ms"],
    }
    verdict.write_text(json.dumps(report, indent=2) + "\n")
    print(f"C06_ISOLATED_PASS {verdict}; full video/audio pipeline gates remain required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--latent", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.latent is None or args.case is None:
            parser.error("--prepare requires --latent and --case")
        _prepare(args.prepare, args.latent, args.case)
    else:
        if args.inputs is None:
            parser.error("--verify requires --inputs")
        _verify(args.verify, args.inputs, args.baseline)
