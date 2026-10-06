# SPDX-License-Identifier: Apache-2.0
"""Full-model terminal policies with immutable decoder weights reused between configurations.

Each precision change synchronizes and retires every old trace, destroys the old
request state, rebuilds the head, then constructs/prepares a new generator. The
selected policy is separately reproduced through fresh default construction.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from readiness_check.run_prefill_check import _run_one_entry_prefill
from readiness_check.run_teacher_forcing import _run_one_entry
from readiness_check.schema import load_reference
from readiness_check.teacher_forcing import TokenAccuracy

import ttnn
from models.common.modules.lm_head.lm_head_1d import LMHead1D

from ..tt.generator import KolibriGenerator, build_generator
from ..tt.precision import load_precision_config
from .datatype_run import runtime_summary, token_out
from .full_memory import memory_views
from .full_provenance import provenance


def replace_terminal(model, config):
    old = model.precision_config
    assert old["mesh_policy"] == config["mesh_policy"]
    assert old["layer_exceptions"] == config["layer_exceptions"]
    variable = {
        "head_weight_dtype",
        "head_output_dtype",
        "head_fidelity",
        "head_fp32",
        "sampling_logits_dtype",
        "kv_cache_dtype",
    }
    assert {k: v for k, v in old["runtime"].items() if k not in variable} == {
        k: v for k, v in config["runtime"].items() if k not in variable
    }
    runtime = config["runtime"]
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, runtime["head_fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=runtime["head_fp32"],
        packer_l1_acc=True,
    )
    # LazyWeight._value is its cached native tensor. Clearing it is essential:
    # copying the wrapper without clearing it would silently retain the old dtype.
    weights = [
        replace(w, dtype=getattr(ttnn, runtime["head_weight_dtype"]), _value=None)
        for w in model.head.config.output_weights
    ]
    head = LMHead1D.from_config(
        replace(
            model.head.config,
            output_weights=weights,
            lm_head_dtype=getattr(ttnn, runtime["head_output_dtype"]),
            compute_kernel_config=compute,
        )
    )
    head.load_device_weights()
    model.head = head
    model.head_compute = compute
    model.precision_config = config
    model.runtime_precision = runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--reduced", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    model_root = Path(__file__).resolve().parents[1]
    stage = model_root / "doc/datatype_sweep"
    reference_path = model_root / "readiness_aime24_chat.refpt"
    reference = load_reference(reference_path)
    configs = [load_precision_config(stage / "configs" / f"{name}.json") for name in args.configs]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    model = None
    retired = []
    decoder_addresses = None
    try:
        for index, config in enumerate(configs):
            out = stage / ("terminal_reconfigure_smoke" if args.reduced else "candidates") / config["config_id"]
            out.mkdir(parents=True, exist_ok=True)
            os.environ["FULL_ARTIFACT_DIR"] = str(out)
            result = dict(
                config_id=config["config_id"],
                precision_config=config,
                provenance=provenance(),
                hardware="QB2 / P300x2 / four Blackhole chips",
                mesh=[1, 4],
                reference=str(reference_path),
                reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                thresholds=dict(top1=0.9, top5=0.98, top100=1.0),
                status="running",
                construction_regime="fresh first model; subsequent head/cache reconstruction after all prior traces are quiesced/released; unchanged decoder weights reused",
                preceding_retired_trace_sets=list(retired),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            command = dict(
                command=[sys.executable, *sys.argv],
                config_id=config["config_id"],
                start=time.time(),
                git_head=result["provenance"]["git_head"],
                environment={k: v for k, v in os.environ.items() if k.startswith(("TT_METAL_", "FULL_", "KOLIBRI_"))},
            )
            (out / "source_snapshots" / (result["harness_sha256"] + ".py.txt")).write_bytes(Path(__file__).read_bytes())
            helper = Path(__file__).with_name("datatype_run.py").read_bytes()
            result["helper_sha256"] = hashlib.sha256(helper).hexdigest()
            (out / "source_snapshots" / (result["helper_sha256"] + ".py.txt")).write_bytes(helper)

            def save():
                (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                (out / "run.command.json").write_text(json.dumps(command, indent=2) + "\n")

            save()
            print("TERMINAL_CONFIG_START", config["config_id"], flush=True)
            try:
                if model is None:
                    gen = build_generator(
                        model_root, mesh, precision_config=config, layer_indices=[0, 4] if args.reduced else None
                    )
                    model = gen.model
                    decoder_addresses = [layer.experts["gate_up"].buffer_address() for layer in model.layers]
                else:
                    replace_terminal(model, config)
                    gc.collect()
                    gen = KolibriGenerator(model, capacity=config["runtime"]["max_context"])
                assert decoder_addresses == [layer.experts["gate_up"].buffer_address() for layer in model.layers]
                gen.prepare()
                result["runtime_summary"] = runtime_summary(gen)
                result["allocator"] = memory_views(mesh)
                result["decoder_weight_addresses"] = decoder_addresses
                traces = dict(gen.traces)
                save()
                if args.reduced:
                    result["smoke"] = [
                        dict(length=n, tokens=gen.generate([42] * n, 4, stop_on_eos=False))
                        for n in (31, 33, 129, 197, 8193)
                    ]
                    result["status"] = "smoke-pass"
                else:
                    result["prefill"] = [
                        _run_one_entry_prefill(generator=gen, entry=e, reference=reference) for e in reference.entries
                    ]
                    save()
                    samples = []
                    for repeat in range(3):
                        acc = TokenAccuracy(reference_path)
                        before = gen.counters.copy()
                        rows = [_run_one_entry(generator=gen, acc=acc, entry_idx=i) for i in range(acc.num_entries)]
                        counts = dict(gen.counters - before)
                        assert counts.get("decode_replays", 0) >= 99
                        samples.append(
                            dict(repeat=repeat, rows=rows, counters=counts, metrics=gen.last_generation_metrics)
                        )
                        result["teacher_forcing_samples"] = samples
                        save()
                    result["decode_t_s_u"] = statistics.median(s["rows"][0]["decode_t/s/u"] for s in samples)
                    result["ttft_ms"] = statistics.median(s["rows"][0]["ttft_ms"] for s in samples)
                    result["accuracy"] = samples[0]["rows"][0]
                    result[
                        "measurement_regime"
                    ] = "warmed trace-verified AIME24 teacher-forcing; host reference input and token readback; B1 P197 G100; full 1M cache; median of three repeats"
                    result["status"] = (
                        "pass"
                        if all(
                            row[k] >= v
                            for row in result["prefill"] + [s["rows"][0] for s in samples]
                            for k, v in result["thresholds"].items()
                        )
                        else "accuracy-fail"
                    )
                    result["token_out"] = token_out(gen, reference.entries[0].prompt_tokens[0].tolist()[:128])
                assert gen.traces == traces
                result["traces"] = {k: str(v) for k, v in traces.items()}
                result["counters"] = dict(gen.counters)
                ttnn.synchronize_device(mesh)
                gen.close()
                result["teardown_counters"] = dict(gen.counters)
                assert gen.counters["teardown_releases"] == len(traces)
                retired.append(result["traces"])
                del gen
                gen = None
                gc.collect()
                command.update(end=time.time(), exit_code=0)
                save()
                print(
                    "TERMINAL_CONFIG_END", config["config_id"], result["status"], result.get("decode_t_s_u"), flush=True
                )
            except Exception as error:
                result.update(status="runtime-fail", error=repr(error))
                command.update(end=time.time(), exit_code=1)
                save()
                raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
