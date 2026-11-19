# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial full-model precision evidence using installed readiness scoring.

Run under an external timeout. Each process owns one mesh and candidate.
Traced resident-logit timing reuses the optimized-full-model benchmark exactly.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from readiness_check.run_prefill_check import _run_one_entry_prefill
from readiness_check.run_teacher_forcing import _run_one_entry
from readiness_check.schema import load_reference
from readiness_check.teacher_forcing import TokenAccuracy

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_runtime_identity import (
    runtime_identity,
    source_input_identity,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance import measure_teacher_forcing
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = json.loads(args.build_kwargs_file.read_text())
    references = args.reference or [ROOT / "readiness_aime24_chat.refpt"]
    reference_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in references}
    source_inputs = [args.build_kwargs_file, Path(kwargs["precision_config"]), *references]
    assert str(ROOT / "readiness_aime24_chat.refpt") in reference_hashes or any(
        path.resolve() == (ROOT / "readiness_aime24_chat.refpt").resolve() for path in references
    ), "Ranking workload must use the scored AIME reference"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        config_id=args.config_id,
        build_kwargs=kwargs,
        command=sys.argv,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hardware="4 Blackhole p300c",
        mesh=[1, 4],
        measurement_regime="serial",
        environment={
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        references=[],
        reference_entry_counts={str(path): len(load_reference(path).entries) for path in references},
        reference_sha256=reference_hashes,
        native_runtime_identity=runtime_identity(),
        python_runtime_identity=source_input_identity(model_root=ROOT, explicit_paths=source_inputs),
        teacher_performance=[],
        status="running",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = generator = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        generator = build_generator(ROOT, mesh, **kwargs)
        model = generator.model
        summary = getattr(model, "precision_summary", None)
        result["runtime_policy"] = summary() if callable(summary) else getattr(model, "precision_policy", None)
        result["layers"] = list(model.layer_indices)
        assert result["layers"] == list(range(40))
        result["loaded_census"] = model.census("datatype_loaded")
        # Exercise all-logit prefill before trace capture, using runner scoring unchanged.
        for path in references:
            reference = load_reference(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == reference_hashes[str(path)]
            row = dict(reference=str(path), sha256=reference_hashes[str(path)], prefill=[], teacher=[])
            for entry in reference.entries:
                row["prefill"].append(_run_one_entry_prefill(generator=generator, entry=entry, reference=reference))
            result["references"].append(row)
            save()
        aime = load_reference(ROOT / "readiness_aime24_chat.refpt").entries[0]
        assert aime.prompt_tokens.numel() == 161 and aime.generated_tokens.numel() == 100
        forced_ids = aime.generated_tokens[0, :99].tolist()
        forced_inputs = []
        for token in forced_ids:
            ids = torch.zeros(tuple(generator.tokens.shape), dtype=torch.int32)
            ids[..., 0] = token
            forced_inputs.append(
                ttnn.from_torch(
                    ids,
                    dtype=generator.tokens.dtype,
                    layout=generator.tokens.layout,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )
        ttnn.copy_host_to_device_tensor(forced_inputs[0], generator.tokens, cq_id=0)

        def position_value():
            return int(ttnn.to_torch(ttnn.get_device_tensors(generator.positions[0])[0]).flatten()[0])

        position_value()
        generator.prepare(prefill_cases=((161, 0),))
        trace_ids = {key: int(value) for key, value in generator.traces.items()}
        assert set(trace_ids) == {"model", "sampling"}
        for path, row in zip(references, result["references"]):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == reference_hashes[str(path)]
            acc = TokenAccuracy(path)
            for entry_idx in range(acc.num_entries):
                before = dict(generator.counters)
                stats = _run_one_entry(generator=generator, acc=acc, entry_idx=entry_idx)
                stats["trace_counter_delta"] = {k: generator.counters[k] - v for k, v in before.items()}
                assert stats["trace_counter_delta"]["model_replays"] == stats["total"] - 1
                stats["top1_binomial_standard_error"] = math.sqrt(stats["top1"] * (1 - stats["top1"]) / stats["total"])
                row["teacher"].append(stats)
            save()
        for repeat in range(args.repeats):
            generator.reset()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for value in generator.prefill_forward(aime.prompt_tokens.int(), prompt_lens=[161], device_output=True):
                ttnn.deallocate(value)
            ttnn.synchronize_device(mesh)
            ttft_ms = (time.perf_counter() - start) * 1000
            generator._copy_host(torch.tensor([161], dtype=torch.int32), generator.positions[0])
            entries = int(mesh.num_program_cache_entries())
            perf = measure_teacher_forcing(
                generator,
                mesh,
                forced_inputs,
                forced_ids,
                ttnn=ttnn,
                device_only=device_only,
                position_value=position_value,
                prepared_entries=entries,
                trace_ids=trace_ids,
            )
            perf.update(
                repeat=repeat,
                ttft_ms=ttft_ms,
                ttft_scope="warmed161-token prefill to resident logits, excludes sampling/readback",
            )
            result["teacher_performance"].append(perf)
            save()
        result["prepared_census"] = model.census("datatype_prepared")
        result["trace_ids"] = trace_ids
        result["runtime_counters"] = dict(generator.counters)
        result["runtime_policy_after"] = model.precision_summary()
        assert result["runtime_policy_after"]["selected_policy"] == result["runtime_policy"]["selected_policy"]
        result["native_runtime_identity_after"] = runtime_identity()
        assert (
            result["native_runtime_identity_after"] == result["native_runtime_identity"]
        ), "Native source/build changed during candidate measurement"
        result["python_runtime_identity_after"] = source_input_identity(model_root=ROOT, explicit_paths=source_inputs)
        assert (
            result["python_runtime_identity_after"] == result["python_runtime_identity"]
        ), "Python source or measurement inputs changed during candidate measurement"
        result["passed"] = all(
            s["top1"] >= 0.90 and s["top5"] >= 0.98 and s["top100"] == 1.0
            for row in result["references"]
            for mode in ("prefill", "teacher")
            for s in row[mode]
        )
        result["status"] = "accuracy_pass" if result["passed"] else "accuracy_fail"
        save()
        print(
            json.dumps(
                dict(
                    config_id=args.config_id,
                    status=result["status"],
                    performance=result["teacher_performance"][-1]["tokens_per_second_per_user"],
                )
            ),
            flush=True,
        )
    except BaseException:
        result["status"] = "runtime_error"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["device_closed"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
