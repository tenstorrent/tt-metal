# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full64 AIME24 precision measurement with actual trace and policy evidence."""

import argparse
import faulthandler
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy
from models.common.readiness_check import run_prefill_check as prefill
from models.common.readiness_check import run_teacher_forcing as teacher
from models.common.readiness_check.schema import load_reference


def compute_summary(config):
    return {
        "math_fidelity": str(config.math_fidelity),
        "fp32_dest_acc_en": config.fp32_dest_acc_en,
        "math_approx_mode": config.math_approx_mode,
        "packer_l1_acc": config.packer_l1_acc,
    }


def runtime_summary(gen):
    model = gen.model
    rows = []
    for layer in model.layers:
        requested = decoder_policy(model.precision, layer.layer_idx)
        assert all(layer.policy[k] == v for k, v in requested.items())
        weights = {}
        for name, weight in layer.weights.items():
            if len(weight.shape) == 2:
                role = layer._role(name)
                assert weight.dtype == getattr(ttnn, requested[role + "_dtype"])
                assert layer.dram_weights[name].dtype == weight.dtype
                weights[name] = dict(role=role, dtype=str(weight.dtype), shape=list(weight.shape))
        rows.append(
            dict(
                layer=layer.layer_idx,
                policy=layer.policy,
                actual_weights=weights,
                compute={r: compute_summary(c) for r, c in layer.projection_configs.items()},
            )
        )
    return dict(
        policy=model.precision,
        layers=rows,
        head_dtype=str(model.head_weight.dtype),
        head_compute=compute_summary(model.head_compute),
        final_norm_compute=compute_summary(model.norm_compute),
        context=model.context,
    )


def allocated_state_summary(gen):
    return dict(
        logits_dtype=str(gen.logits.dtype),
        token_dtype=str(gen.tokens.dtype),
        cache_capacity=gen.cache.capacity,
        page_count=gen.cache.num_pages,
        layers=[
            dict(
                layer=layer.layer_idx,
                kind=layer.kind,
                tensors={
                    name: dict(dtype=str(tensor.dtype), shape=list(tensor.shape), layout=str(tensor.layout))
                    for name in ("key", "value", "conv", "recurrent")
                    if (tensor := getattr(state, name)) is not None
                },
            )
            for layer, state in zip(gen.model.layers, gen.cache.layers)
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path("models/autoports/qwen_qwen3_8_27b")
    reference = root / "readiness_aime24_chat.refpt"
    metadata = json.loads(reference.with_suffix(".metadata.json").read_text())
    assert metadata["generation_length"] == 100 and metadata["chat_template"]
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    report = dict(
        reference=str(reference),
        reference_metadata=metadata,
        mesh=[1, 4],
        hardware="4 Blackhole p300c; TP4 Ring; payload8192",
        status="running",
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        faulthandler.dump_traceback_later(180, repeat=True)
        gen = build_generator(root, mesh, layer_indices=[0, 3] if args.smoke else None)
        faulthandler.cancel_dump_traceback_later()
        report["runtime"] = runtime_summary(gen)
        save()
        if args.smoke:
            report["smoke"] = []
            for length in (31, 33, 129):
                tokens = gen.generate([1596] * length, 4)
                report["smoke"].append(dict(length=length, tokens=tokens, perf=gen.last_perf))
            report["runtime"]["allocated_state"] = allocated_state_summary(gen)
            report["status"] = "smoke_pass"
            save()
            return
        with patch.object(prefill, "_import_build_generator", return_value=lambda **kw: gen):
            report["prefill"] = prefill.run_prefill_check(model_dir=root, reference_path=reference, mesh_device=mesh)
        save()
        report["teacher_forcing_runs"] = []
        for _ in range(3):
            with patch.object(teacher, "_import_build_generator", return_value=lambda **kw: gen):
                accuracy = teacher.run_teacher_forcing(model_dir=root, reference_path=reference, mesh_device=mesh)
            perf = gen.last_perf
            counters = perf["steady_state_counters"]
            assert perf["teacher_forcing"] and counters["model_replays"] == 99
            assert counters["sampling_replays"] == 99 and not perf["host_sampling"]
            report["teacher_forcing_runs"].append(dict(accuracy=accuracy, perf=perf, trace_verified=True))
            save()
        report["status"] = (
            "pass"
            if all(
                row["top1"] >= 0.9 and row["top5"] >= 0.98 and row["top100"] == 1
                for row in report["prefill"] + accuracy
            )
            else "accuracy_fail"
        )
        entry = load_reference(reference).entries[0]
        prompt = entry.prompt_tokens[0].tolist()
        forced = entry.generated_tokens[0].tolist()
        report["warmed_teacher_forcing_perf"] = []
        for _ in range(3):
            predicted = gen.generate(prompt, len(forced), next_input=lambda step, predicted: forced[step])
            perf = gen.last_perf
            assert perf["teacher_forcing"] and not perf["host_sampling"]
            assert perf["steady_state_counters"]["model_replays"] == 99
            assert perf["steady_state_counters"]["sampling_replays"] == 99
            report["warmed_teacher_forcing_perf"].append(perf)
        for perf in report["warmed_teacher_forcing_perf"][-2:]:
            assert not perf["steady_state_counters"].get("trace_captures", 0)
        report["runtime"]["allocated_state"] = allocated_state_summary(gen)
        report["teacher_forced_tokens"] = predicted
        report["reference_tokens"] = forced
        report["first_top1_miss"] = next((i for i, (a, b) in enumerate(zip(predicted, forced)) if a != b), None)
        report["workload"] = dict(prompt_len=len(prompt), gen_len=len(forced), batch=1)
        report[
            "measurement_regime"
        ] = "traced teacher forcing, batch1 S203 G100, native sampled-token delivery; median last2 warmed runs"
        save()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
