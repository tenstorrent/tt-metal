# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager TP4 teacher-forced HF comparison; not a quality or performance gate."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.reference_comparison import validate_teacher_forcing, vector_metrics
from models.demos.qwen38_27b_qb2.tt.precision import ROLES, load_precision


def validate_decoder_control(baseline, candidate):
    """Diagnostic-only fidelity/weight controls cannot change state or attention math."""
    allowed = {"config_id", "weight_groups", "compute_fidelities"}
    if set(candidate) != set(baseline) or any(
        candidate[key] != value for key, value in baseline.items() if key not in allowed
    ):
        raise ValueError("Decoder control must retain baseline state, cache and attention settings")
    for key in ("weight_groups", "compute_fidelities"):
        if candidate[key]["head"] != baseline[key]["head"]:
            raise ValueError("Decoder control must retain the baseline head")
    if (
        len({candidate["weight_groups"][role] for role in ROLES}) != 1
        or candidate["weight_groups"]["attention"] not in ("bfloat4_b", "bfloat8_b")
        or any(candidate["compute_fidelities"][role] != "HiFi2" for role in ROLES)
    ):
        raise ValueError("Decoder control requires uniform BFP4 or BFP8 weights and HiFi2")
    return candidate


def main(args):
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.demo.galaxy_serving import qualified_groups, verify_qualified_source
    from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
    from models.demos.qwen38_27b_qb2.tt.model import Qwen38Model

    args.output.mkdir()
    report = dict(
        state="validating_reference",
        scope="One TP4 replica, B1, eager teacher-forced public prompt, all decoder inputs and final norm",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_at=time.time(),
        cleanup_completed=False,
        hardware_opened=False,
        is_gpqa_score=False,
        is_performance_measurement=False,
        steps=[],
    )

    def save():
        path = args.output / "progress.json.tmp"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        path.replace(args.output / "progress.json")

    save()
    parent = None
    try:
        qualification = json.loads(args.qualification.read_text())
        groups = qualified_groups(qualification)
        source = Path(__file__).resolve().parents[1]
        os.environ["QWEN_PRECISION_CONFIG"] = str(args.precision.resolve())
        verify_qualified_source(qualification, source)
        # G0 still authenticates the unchanged model source and baseline policy.
        # An explicit control is a new unqualified experiment, never a substitute
        # G0 receipt for serving or an inherited accuracy claim.
        precision_path = args.precision
        runtime_policy = load_precision(precision_path)
        if runtime_policy != qualification["precision"]:
            raise ValueError("Diagnostic baseline differs from its qualification")
        control = getattr(args, "control_precision", None)
        if control is not None:
            runtime_policy = validate_decoder_control(runtime_policy, load_precision(control))
            precision_path = control
            os.environ["QWEN_PRECISION_CONFIG"] = str(control.resolve())
            report.update(
                diagnostic_control=True,
                serving_qualified=False,
                baseline_precision=qualification["precision"],
                control_precision_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
            )
        reference_report = json.loads((args.reference / "progress.json").read_text())
        reference_file = args.reference / "reference.pt"
        if (
            reference_report.get("state") != "completed"
            or reference_report.get("hardware_opened") is not False
            or reference_report.get("reference_tensor_sha256")
            != hashlib.sha256(reference_file.read_bytes()).hexdigest()
            or reference_report.get("checkpoint_config_sha256")
            != hashlib.sha256((args.weights / "config.json").read_bytes()).hexdigest()
        ):
            raise ValueError("A complete, matching CPU HF reference is required")
        config = json.loads((args.weights / "config.json").read_text())["text_config"]
        torch.set_num_threads(8)
        reference = torch.load(reference_file, map_location="cpu", weights_only=True)
        prompt = qualification["prompt_tokens"]
        capacity = validate_teacher_forcing(
            reference,
            prompt,
            layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            vocab_size=config["vocab_size"],
        )
        for row in reference:
            if any(not torch.isfinite(t).all() for t in [row["logits"], *row["layer_last_hidden"]]):
                raise ValueError("Non-finite HF reference")
        report.update(
            state="loading_model",
            qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
            reference_sha256=reference_report["reference_tensor_sha256"],
            precision=runtime_policy,
            model_source_sha256=qualification["source_sha256"],
            prompt_tokens=len(prompt),
        )
        save()
        if ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
            raise ValueError("This diagnostic requires the allocated Blackhole Galaxy")
        configure_fabric(topology=ttnn.Topology.Linear)
        parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
        report["hardware_opened"] = True
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        if list(mesh.get_device_ids()) != [int(chip) for chip in groups[0].split(",")]:
            raise ValueError("Diagnostic TP4 group differs from the qualified physical group")
        model = Qwen38Model(mesh, snapshot=args.weights, precision_config=precision_path, topology=ttnn.Topology.Linear)
        if model.precision != runtime_policy:
            raise ValueError("Diagnostic loaded a different precision policy")
        cache = model.allocate_cache(batch_size=1, capacity=((capacity + 31) // 32) * 32)
        table = model.upload(
            torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        captured = {}
        active_step = None

        def capture(name, tensor, expected):
            if name in captured:
                raise ValueError(f"Duplicate diagnostic capture: {name}")
            ranks = [ttnn.to_torch(rank).float() for rank in ttnn.get_device_tensors(tensor)]
            if len(ranks) != 4 or any(rank.shape[-1] != config["hidden_size"] for rank in ranks):
                raise ValueError("Capture requires four full-width replicated hidden states")
            actual = [rank.reshape(-1, config["hidden_size"])[-1].tolist() for rank in ranks]
            golden = expected.float().reshape(-1).tolist()
            captured[name] = dict(
                hf_per_rank=[vector_metrics(row, golden) for row in actual],
                rank0_agreement=[vector_metrics(row, actual[0]) for row in actual[1:]],
            )

        def instrument(layer, index, method_name):
            original = getattr(layer, method_name)

            def wrapper(x, *args, **kwargs):
                capture(f"layer_{index:02d}_input", x, active_step["layer_last_hidden"][index])
                return original(x, *args, **kwargs)

            setattr(layer, method_name, wrapper)

        for index, layer in enumerate(model.layers):
            instrument(layer, index, "prefill_forward")
            instrument(layer, index, "decode_forward")
        original_head = model._dram_logits

        def capture_head(hidden):
            capture("final_normalized", hidden, active_step["layer_last_hidden"][-1])
            return original_head(hidden)

        model._dram_logits = capture_head

        def logits_metrics(actual, expected):
            value = ttnn.to_torch(actual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(-1)
            golden = expected.float().reshape(-1)
            if value.numel() != config["vocab_size"]:
                raise ValueError("Device logits do not cover exactly the HF vocabulary")
            metrics = vector_metrics(value.tolist(), golden.tolist())
            metrics.update(
                top1=int(value.argmax().item()),
                hf_top1=int(golden.argmax().item()),
                top20_overlap=len(set(value.topk(20).indices.tolist()) & set(golden.topk(20).indices.tolist())),
                kl_from_hf=float((golden.softmax(-1) * (golden.log_softmax(-1) - value.log_softmax(-1))).sum().item()),
            )
            return metrics

        report.update(state="comparing", device_ids=list(mesh.get_device_ids()), loaded_at=time.time())
        save()
        for step, active_step in enumerate(reference):
            captured.clear()
            ids = active_step["input_ids"].int()
            if step == 0:
                tokens = model.upload(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                logits = model.prefill(tokens, cache=cache, page_table=table, length=len(prompt))
            else:
                padded = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                padded.reshape(-1)[0] = ids.item()
                tokens = model.upload(padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                positions = model.upload(
                    torch.tensor([len(prompt) + step - 1], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                logits = model.decode(tokens, positions, cache=cache, page_table=table)
            if len(captured) != len(model.layers) + 1:
                raise ValueError("Missing decoder-input or final-normalization capture")
            row = dict(step=step, mode="prefill" if step == 0 else "decode", hidden=dict(captured))
            row["full_model_logits"] = logits_metrics(logits, active_step["logits"])
            # Feed the same HF normalized hidden state into the actual device
            # head to separate the head's error from accumulated decoder error.
            injected = model.upload(active_step["layer_last_hidden"][-1].reshape(1, 1, 1, -1))
            injected = ttnn.to_memory_config(injected, model.layers[0]._width_memory(8, 32, 640))
            isolated = original_head(injected)
            row["head_on_hf_hidden"] = logits_metrics(isolated, active_step["logits"])
            report["steps"].append(row)
            save()
            print(
                f"HF_LAYER_STEP_COMPLETE step={step} full_top1={row['full_model_logits']['top1']} "
                f"hf_top1={row['full_model_logits']['hf_top1']}",
                flush=True,
            )
        report.update(
            state="completed",
            interpretation="Diagnostic B1/eager comparison against BF16 HF weights. Weight quantization and kernel math both differ; layer error does not alone identify a bug. This is not GPQA, long-horizon accuracy, traced-serving qualification or a throughput measurement.",
        )
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:2000])
        raise
    finally:
        try:
            if parent is not None:
                ttnn.close_mesh_device(parent)
            report["cleanup_completed"] = True
        except BaseException as error:
            report.update(state="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["finished_at"] = time.time()
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "qualification", "precision", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--control-precision", type=Path)
    main(parser.parse_args())
