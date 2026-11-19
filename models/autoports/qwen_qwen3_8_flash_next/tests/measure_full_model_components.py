# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Clean component traces from one real reduced 0/1/3 generator pass.

No profiler, HF reference generation, or all48 construction. Persistent inputs
are retained from natural-prompt prefill followed by one teacher-forced decode.
Measured replay freezes each component's input while its device state advances;
this is a representative-layer estimate, not full autoregressive throughput.
Run serially with an external timeout and the accepted 1x4 mesh descriptor.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=127)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve old evidence; select a new output path")
    if not 1 <= args.prompt_length <= 1024 or not 1 <= args.iterations <= 256:
        parser.error("Use prompt-length1..1024 and iterations1..256")
    if not 1 <= args.trials <= 5 or not 1 <= args.warmup <= 8:
        parser.error("Use trials1..5 and warmup1..8")
    model_dir = Path(__file__).resolve().parents[1]
    repo = model_dir.parents[2]
    sources = [
        Path(__file__).resolve(),
        model_dir / "tests/run_full_model_acceptance.py",
        model_dir / "tests/profile_full_model_terminal.py",
        *sorted(p for p in (model_dir / "tt").rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp")),
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
        repo / "models/common/tt_ccl.py",
    ]
    report = {
        "passed": False,
        "regime": "clean_reduced_component_trace_windows",
        "layers": [0, 1, 3],
        "mesh": [1, 4],
        "capacity": 262144,
        "batch_size": 1,
        "prompt_length": args.prompt_length,
        "iterations_per_window": args.iterations,
        "command": [sys.executable, *sys.argv],
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "READINESS_"))},
        "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "components": {},
        "limitations": [
            "Reduced0/1/3 residency/history; captured activations are reachable in this reduced model, not all48",
            "Frozen component input; recurrent/QSA state advances during replay; no generated-token feedback in layer timing",
            "35/1/12 weighting is a representative-stack estimate, not a measured all48 lower bound",
            "Timing is a whole resident replay window with one final sync; no per-operation profiler attribution",
            "No HF accuracy, serving-state, max-context or allocation-tracker acceptance claim",
        ],
    }
    started = time.monotonic()
    mesh = gen = None
    traces, retained_outputs = {}, {}
    phase = "preflight"
    try:
        assert "TT_METAL_WATCHER" not in os.environ, "Watcher must be unset, even empty/0"
        for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
            assert os.environ.get(name) in (None, "", "0"), f"Disable {name} for clean timing"
        assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
        for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS"):
            assert os.environ.get(name) == "0", f"Set {name}=0 for clean timing"
        expected_mesh = repo / "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
        assert Path(os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")).resolve() == expected_mesh
        assert os.environ.get("TT_VISIBLE_DEVICES") == "0,1,2,3"
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()

        import torch
        from ttnn.tools import trace_allocation_tracker

        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tests import run_full_model_acceptance as acceptance
        from models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal import sampler_body
        from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk
        from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
        from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import _state_tensors
        from models.common.sampling.generator import SamplingParams, format_sampling_params

        assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING
        assert not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024, l1_small_size=32768)
        options = {"layer_indices": (0, 1, 3), "max_batch_size": 1, "max_seq_len": 262144}
        for arg, key in ((args.checkpoint, "checkpoint"), (args.packed_cache, "packed_cache")):
            if arg:
                options[key] = arg
        phase = "prepare"
        gen = build_generator(str(model_dir), mesh, **options)
        assert tuple(gen.model.layer_indices) == (0, 1, 3)
        multiplicities = {"layer0": 0, "layer1": 0, "layer3": 0}
        for index, kind in enumerate(gen.model.config.layer_types):
            has_ple = index + 1 in gen.model.config.ple_layer_ids
            if kind == "linear_attention":
                representative = "layer1" if has_ple else "layer0"
            else:
                assert kind == "indexed_attention" and not has_ple
                representative = "layer3"
            multiplicities[representative] += 1
        assert multiplicities == {"layer0": 35, "layer1": 1, "layer3": 12}
        report["full_model_layer_counts_from_config"] = multiplicities
        assert gen.factored_decode is None and gen.prefill_cache is gen.caches[0]
        assert not gen.caches[0].shared_kv
        gen.sampler.reset_sampling_params(
            format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32)
        )
        assert not gen.sampler.tt_sampling.force_argmax_sampling
        gen.prepare()
        assert not gen.traces and not gen.prefill_traces and not gen.sampling_traces

        phase = "natural_prefill_without_capture"
        prompt = acceptance.natural_prompt(gen, model_dir, args.prompt_length + 1)
        gen.reset()
        offset = 0
        while offset < args.prompt_length:
            count = min(32, 1 << ((args.prompt_length - offset).bit_length() - 1))
            tokens = torch.tensor([prompt[offset : offset + count]], dtype=torch.int32)
            words = torch.zeros(1, count, 16, dtype=torch.int32)
            words[..., 0], words[..., 1] = tokens, 1
            gen._refresh(gen.prefill_inputs[count], tokens)
            gen._refresh(gen.prefill_words[count], words)
            gen._prefill_body(count)
            offset += count
        gen.caches[0].position = args.prompt_length
        report["initial_positions"] = acceptance.qsa_positions(gen, args.prompt_length)
        decode = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        decode[..., 0] = prompt[-1]
        gen._refresh(gen.decode_tokens, decode)
        report["prompt_token_ids"] = list(map(int, prompt[:-1]))
        report["teacher_forced_decode_token"] = int(prompt[-1])
        prompt_path = model_dir / "doc/full_model/hf_reference/prompt.txt"
        report["prompt_source_sha256"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()

        # Every mutable state snapshot and retained input is allocated before
        # the first capture. No state copies occur inside measured windows.
        buffers = list(_state_tensors(gen.caches[0]))
        snapshots = [(name, target, ttnn.clone(target)) for name, target in buffers]
        saved_token = ttnn.clone(gen.decode_tokens)
        observed = {}

        def clone_tensors(value):
            if value is None:
                return None
            if isinstance(value, (tuple, list)):
                return type(value)(clone_tensors(item) for item in value)
            return ttnn.clone(value)

        def retain(name, function):
            def wrapped(*call_args, **call_kwargs):
                assert name not in observed
                # A Python reference alone does not freeze data if a returned
                # tensor aliases reusable scratch. Snapshot arguments before
                # the original call and outputs before a later layer executes.
                # Keep original cache/state objects; only tensor inputs and
                # token words receive private, setup-time storage.
                frozen_args = clone_tensors(call_args)
                frozen_kwargs = dict(call_kwargs)
                if "token_ids" in frozen_kwargs:
                    frozen_kwargs["token_ids"] = clone_tensors(frozen_kwargs["token_ids"])
                result = function(*call_args, **call_kwargs)
                observed[name] = (function, frozen_args, frozen_kwargs, clone_tensors(result))
                return result

            return wrapped

        phase = "retain_reachable_component_inputs"
        with ExitStack() as stack:
            for name, obj, attr in (
                ("embedding", gen.model, "embed"),
                ("terminal", gen.model, "logits"),
                *((f"layer{index}", layer, "decode_forward") for index, layer in zip((0, 1, 3), gen.model.layers)),
            ):
                stack.enter_context(patch.object(obj, attr, new=retain(name, getattr(obj, attr))))
            gen._decode_body((0,))
        frozen_logits = ttnn.clone(gen.sample_logits)

        def flat_tensors(value):
            if value is None:
                return []
            if isinstance(value, (tuple, list)):
                return [tensor for item in value for tensor in flat_tensors(item)]
            return [value]

        def read(value):
            return [
                ttnn.to_torch(rank).clone()
                for tensor in flat_tensors(value)
                for rank in ttnn.get_device_tensors(tensor)
            ]

        def exact(actual, expected):
            return len(actual) == len(expected) and all(torch.equal(a, b) for a, b in zip(actual, expected))

        expected_outputs = {name: read(item[3]) for name, item in observed.items()}
        expected_outputs["reduced_model"] = read(gen.sample_logits)
        held_logits = observed["terminal"][3]
        frozen_inputs = {name: (item[1], item[2].get("token_ids")) for name, item in observed.items()}
        frozen_inputs["row_assembly"] = held_logits
        frozen_inputs["logits_restore_source"] = frozen_logits
        frozen_input_values = {name: read(value) for name, value in frozen_inputs.items()}

        def check_frozen_inputs():
            for name, value in frozen_inputs.items():
                assert exact(read(value), frozen_input_values[name]), f"Frozen input overwritten: {name}"

        def entry():
            token = ttnn.reshape(gen.decode_tokens[:, :, :, :1], (1, 1))
            words = ttnn.concat([ttnn.reshape(token, (1, 1, 1)), gen.word_suffix], dim=2)
            return token, words

        def assembly():
            ttnn.full_like(gen.sample_logits, -100.0, optional_tensor=gen.sample_logits)
            write_token_chunk(held_logits, gen.sample_logits, 0)
            return gen.sample_logits

        def sample():
            sampler_body(gen, gen.sampler)
            return gen.decode_tokens

        def reduced_model():
            gen._decode_body((0,))
            return gen.sample_logits

        def restore():
            for _, target, saved in snapshots:
                ttnn.copy(saved, target)
            ttnn.copy(saved_token, gen.decode_tokens)
            ttnn.copy(frozen_logits, gen.sample_logits)
            gen.caches[0].position = args.prompt_length

        operations = {"token_entry": entry}
        for name in ("embedding", "layer0", "layer1", "layer3", "terminal"):
            function, call_args, call_kwargs, _ = observed[name]
            operations[name] = lambda f=function, a=call_args, k=call_kwargs: f(*a, **k)
        operations.update(row_assembly=assembly, split_greedy=sample, reduced_model=reduced_model)
        phase = "warm_all_signatures_before_capture"
        for name, operation in operations.items():
            restore()
            output = operation()
            actual = read(output)
            if name in expected_outputs:
                assert exact(actual, expected_outputs[name]), f"Retained-input eager mismatch: {name}"
            expected_outputs[name] = actual
            del output
        check_frozen_inputs()
        restore()
        ttnn.synchronize_device(mesh)
        cache_entries = mesh.num_program_cache_entries()
        phase = "capture_components"
        for name, operation in operations.items():
            restore()
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                retained_outputs[name] = operation()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace
            assert mesh.num_program_cache_entries() == cache_entries, f"Unwarmed program at capture: {name}"

        for name, trace in traces.items():
            phase = f"measure_{name}"
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            assert exact(read(retained_outputs[name]), expected_outputs[name]), f"First replay mismatch: {name}"
            check_frozen_inputs()
            samples = []
            for trial in range(args.trials):
                restore()
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                restore()
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed_ms = (time.perf_counter() - start) * 1000
                # All diagnostics are host reads after the timing boundary;
                # no new TTNN finiteness/reduction program is compiled here.
                actual = read(retained_outputs[name])
                assert all(bool(torch.isfinite(t).all()) for t in actual), f"Nonfinite output: {name}"
                advanced = name in ("layer3", "reduced_model")
                position = args.prompt_length + (args.iterations if advanced else 0)
                gen.caches[0].position = position
                positions = acceptance.qsa_positions(gen, position)
                check_frozen_inputs()
                samples.append(
                    {
                        "trial": trial,
                        "total_ms": elapsed_ms,
                        "ms_per_replay": elapsed_ms / args.iterations,
                        "positions": positions,
                    }
                )
            report["components"][name] = {
                "trace_id": str(trace),
                "median_ms": statistics.median(row["ms_per_replay"] for row in samples),
                "samples": samples,
                "first_replay_exact": True,
                "all_outputs_finite": True,
                "persistent_input": True,
                "private_input_snapshots_unchanged_after_every_window": True,
                "nonblocking_replay": True,
                "interior_host_reads_or_syncs": 0,
                "state_restoration_inside_window": False,
                "output_shapes": [list(t.shape) for t in flat_tensors(retained_outputs[name])],
            }
            assert mesh.num_program_cache_entries() == cache_entries, f"Late program compilation: {name}"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"component": name, "median_ms": report["components"][name]["median_ms"]}), flush=True)
        times = {name: row["median_ms"] for name, row in report["components"].items()}
        stack_ms = sum(count * times[name] for name, count in multiplicities.items())
        terminal_ms = sum(
            times[name] for name in ("token_entry", "embedding", "terminal", "row_assembly", "split_greedy")
        )
        report["weighted_estimate"] = {
            "multiplicities": multiplicities,
            "decoder_stack_ms": stack_ms,
            "entry_terminal_sampler_ms": terminal_ms,
            "stack_plus_terminal_ms": stack_ms + terminal_ms,
            "sum_of_reduced_components_ms": sum(
                v for k, v in times.items() if k not in ("reduced_model", "split_greedy")
            ),
            "reduced_model_trace_ms": times["reduced_model"],
            "scope": "Representative trace estimate, with frozen real reduced-model component inputs; not all48 measured latency",
        }
        report["program_cache_entries_after_all_warmups"] = cache_entries
        report["all_inputs_snapshots_and_program_warmups_before_first_capture"] = True
        report["passed"] = True
    except BaseException as error:
        report["failure"] = {
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
