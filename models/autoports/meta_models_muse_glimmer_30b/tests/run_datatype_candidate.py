# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model precision evidence; selection uses only warmed traced teacher forcing."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check as prefill
from readiness_check import run_teacher_forcing as teacher

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests.run_full_model import SharedGenerator
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator
from models.autoports.meta_models_muse_glimmer_30b.tt.precision import GROUPS, load_precision_config

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def compute_summary(ck):
    return {
        name: str(getattr(ck, name)) if name == "math_fidelity" else getattr(ck, name)
        for name in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
    }


def runtime_summary(g):
    model = g.model
    layers = []
    for layer, cache in zip(model.layers, g.kv_cache):
        groups = {}
        for name in GROUPS:
            ck = layer.mm_ck[name]
            groups[name] = dict(
                weight_dtype=str(layer.weights[name].dtype),
                compute_config=compute_summary(ck),
                prefill_weight_dtype=str(
                    (layer.minimal_gate_up if name == "mlp.gate_up" else layer.prefill_weights[name]).dtype
                ),
            )
            assert layer.weights[name].dtype == getattr(ttnn, layer.policy[name + ".dtype"])
            assert getattr(ck, "math_fidelity") == getattr(ttnn.MathFidelity, layer.policy[name + ".fidelity"])
            assert getattr(ck, "fp32_dest_acc_en") == layer.policy[name + ".fp32"]
        for phase in ("prefill", "decode"):
            qk = layer.norm_configs[phase][None]
            for field, attribute in (("fidelity", "math_fidelity"), ("fp32", "fp32_dest_acc_en")):
                key = "qk_norm." + field
                if key in layer.policy:
                    expected = layer.policy[key]
                    if field == "fidelity":
                        expected = getattr(ttnn.MathFidelity, expected)
                    assert getattr(qk, attribute) == expected, (layer.layer_idx, phase, key)
        assert all(t.dtype == layer.kv_dtype for t in cache)
        layers.append(
            dict(
                layer=layer.layer_idx,
                kind=layer.kind,
                groups=groups,
                attention_compute_config=compute_summary(layer.attention_ck),
                rope_dtype=str(layer.cos.dtype) if layer.use_rope else None,
                norm_weight_dtypes={name: str(t.dtype) for name, t in layer.norm_weights.items()},
                actual_cache_dtypes=[str(t.dtype) for t in cache],
                resolved_runtime_policy=layer.policy,
                norm_configs={
                    phase: {str(k): compute_summary(v) for k, v in cfg.items()}
                    for phase, cfg in layer.norm_configs.items()
                },
            )
        )
    assert model.lm_decode.dtype == getattr(ttnn, model.terminal_policy["weight_dtype"])
    assert model.lm_compute_config.math_fidelity == getattr(ttnn.MathFidelity, model.terminal_policy["fidelity"])
    assert model.lm_compute_config.fp32_dest_acc_en == model.terminal_policy["fp32"]
    head_programs = [
        {name: getattr(pc, name) for name in ("in0_block_w", "per_core_M", "per_core_N", "num_workers_per_dram_bank")}
        for pc in model.lm_head.config.program_configs
    ]
    assert all(p["in0_block_w"] == model.terminal_policy["block_w"] for p in head_programs)
    assert all(p["num_workers_per_dram_bank"] == model.terminal_policy["readers"] for p in head_programs)
    assert model.lm_input_mem.shard_spec.grid.num_cores() == model.terminal_policy["input_cores"]
    assert len(head_programs) == model.terminal_policy["splits"]
    assert g.logits.dtype == model.logits_dtype and g.prefill[128]["hidden"].dtype == model.residual_dtype
    with g._prefill_policy(32):
        short_norm_configs = {
            str(layer.layer_idx): compute_summary(layer.norm_configs["prefill"]["input_layernorm"])
            for layer in model.layers
        }
    return dict(
        short_prefill_norm_configs=short_norm_configs,
        layers=layers,
        terminal_weight_dtype=str(model.lm_decode.dtype),
        terminal_compute_config=compute_summary(model.lm_compute_config),
        terminal_program_configs=head_programs,
        terminal_input_memory=str(model.lm_input_mem),
        embedding_dtype=str(model.embedding_weight.dtype),
        final_norm_dtype=str(model.final_norm.dtype),
        final_norm_compute_config=compute_summary(model.compute_config),
        logits_dtype=str(g.logits.dtype),
        prefill_hidden_dtype=str(g.prefill[128]["hidden"].dtype),
        sampling_parameter_dtype=str(g.p.dtype),
        sampling_index_dtype=str(g.tokens.dtype),
        context=model.config.max_position_embeddings,
        physical_pages=g.num_pages,
        ccl_allocations=[
            dict(signature=str(k), dtype=str(v.dtype))
            for k, v in model.ccl.muse_scratch_buffers.items()
            if hasattr(v, "dtype")
        ],
    )


def main(args):
    torch.set_num_threads(8)
    config = load_precision_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        config_id=config["config_id"],
        precision_config=config,
        config_path=str(args.config),
        command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hardware="QB2/P300x2: 4 Blackhole chips, 2 p300c boards",
        mesh=[1, 4],
        environment={
            k: os.environ.get(k, "unset")
            for k in [
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            ]
        },
        reference_path=str(ROOT / "readiness_aime24_chat.refpt"),
        reference_sha256=hashlib.sha256((ROOT / "readiness_aime24_chat.refpt").read_bytes()).hexdigest(),
        thresholds={"top1": 0.90, "top5": 0.98, "top100": 1.0},
        measurement_regime=(
            "reduced layers0/3 operational smoke, nine prompt lengths and four generated tokens; not full-model accuracy/performance"
            if args.smoke
            else "warmed batch1 AIME24 chat-template, 100 generated tokens; traced decode + sampling + forced-input + readback; median of 3"
        ),
        implementation_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
        },
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    g = mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        start = time.perf_counter()
        g = build_generator(ROOT, mesh, precision_config=config, **({"layer_indices": [0, 3]} if args.smoke else {}))
        result["construction_seconds"] = time.perf_counter() - start
        result["runtime_summary"] = runtime_summary(g)
        result["traces"] = {str(k): str(v) for k, v in g.traces.items()}
        save()
        if args.smoke:
            rows = []
            for n in [31, 32, 33, 127, 128, 129, 4095, 4096, 4097]:
                before = dict(g.counters)
                tokens = g.generate([42] * n, 4, stop_on_eos=False)
                rows.append(
                    dict(length=n, tokens=tokens, counters={k: v - before.get(k, 0) for k, v in g.counters.items()})
                )
            result.update(smoke=rows, status="smoke_pass_not_full_model")
            save()
            return
        shared = SharedGenerator(g)
        with patch.object(prefill, "_import_build_generator", return_value=lambda **kw: shared):
            result["prefill"] = prefill.run_prefill_check(
                model_dir=ROOT, reference_path=Path(result["reference_path"]), mesh_device=mesh
            )
        save()
        trials = []
        for repeat in range(4):
            before = dict(g.counters)
            trace_ids = dict(g.traces)
            with patch.object(teacher, "_import_build_generator", return_value=lambda **kw: shared):
                stats = teacher.run_teacher_forcing(
                    model_dir=ROOT, reference_path=Path(result["reference_path"]), mesh_device=mesh
                )
            delta = {k: v - before.get(k, 0) for k, v in g.counters.items()}
            assert g.traces == trace_ids and delta.get("captures", 0) == 0
            assert delta["trace_replays"] >= 198 and "decode" in g.traces
            assert all(s["total"] == 100 and s["k"] == 100 for s in stats)
            trials.append(
                dict(repeat=repeat, stats=stats, counters=delta, trace_verified=True, excluded_warmup=repeat == 0)
            )
            result["teacher_trials"] = trials
            save()
        measured = trials[1:]
        result.update(
            top1=min(s["top1"] for r in measured for s in r["stats"]),
            top5=min(s["top5"] for r in measured for s in r["stats"]),
            top100=min(s["top100"] for r in measured for s in r["stats"]),
            ttft_ms=statistics.median(s["ttft_ms"] for r in measured for s in r["stats"]),
            decode_tps=statistics.median(s["decode_t/s/u"] for r in measured for s in r["stats"]),
            trace_verified=True,
        )
        result["accuracy_pass"] = all(
            s[k] >= v for s in result["prefill"] + [result] for k, v in result["thresholds"].items()
        )
        result["status"] = "pass" if result["accuracy_pass"] else "accuracy_fail"
        result["counters"] = dict(g.counters)
        save()
        print(
            "CANDIDATE_RESULT",
            {k: result[k] for k in ["config_id", "top1", "top5", "top100", "ttft_ms", "decode_tps", "status"]},
            flush=True,
        )
    except BaseException as error:
        result.update(status="runtime_error", error=repr(error))
        save()
        traceback.print_exc()
        raise
    finally:
        if g is not None:
            g.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
