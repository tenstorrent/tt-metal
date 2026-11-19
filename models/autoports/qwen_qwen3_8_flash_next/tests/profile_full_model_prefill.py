# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced 0/1/3 full-path eager prefill attribution at C262144.

Six warmed logical buckets are profiled once each, with real natural-prompt
IDs and native generator word construction. This never prepares or captures a
trace and never profiles all48 layers. Results are per-op attribution, not HF
quality, TTFT, traced latency, or full-model acceptance. Run under Tracy with
--op-support-count 8192 and an external timeout; profiler drains occur outside
each warm and measured chunk.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

BUCKETS = (1, 2, 4, 8, 16, 32)


def assert_no_traces(generator):
    assert generator.factored_decode is None
    assert not generator.prepared, "This harness must not invoke full generator.prepare()"
    assert all(not getattr(generator, name) for name in ("traces", "prefill_traces", "sampling_traces"))
    assert not generator.sampler._trace_states, "Sampler unexpectedly owns trace state"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh artifact path; preserve previous profiling evidence")
    model_dir = Path(__file__).resolve().parents[1]
    repo = model_dir.parents[2]
    sources = [
        Path(__file__).resolve(),
        model_dir / "tests/run_full_model_acceptance.py",
        model_dir / "doc/optimized_full_model/run_prefill_profile_v1.sh",
        *sorted(path for path in (model_dir / "tt").rglob("*") if path.suffix in (".py", ".cpp", ".h", ".hpp")),
        repo / "models/common/sampling/generator.py",
        repo / "models/common/sampling/tt_sampling.py",
        repo / "models/common/tt_ccl.py",
    ]
    report = {
        "passed": False,
        "scope": "Reduced real layers0/1/3 full generator prefill body, one eager chunk per bucket",
        "regime": "serial_reduced_eager_prefill_op_profile",
        "layers": [0, 1, 3],
        "mesh": [1, 4],
        "batch_size": 1,
        "capacity": 262144,
        "buckets": list(BUCKETS),
        "physical_output_rows": 32,
        "command": [sys.executable, *sys.argv],
        "environment": {key: value for key, value in os.environ.items() if key.startswith(("TT_", "READINESS_"))},
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "expected_signposts": [
            marker for n in BUCKETS for marker in (f"PREFILL_BUCKET_{n}", f"PREFILL_BUCKET_{n}_END")
        ],
        "measurements": [],
        "limitations": [
            "Eager per-op attribution; no claim of traced end-to-end latency, TTFT, or HF quality",
            "One prefix chunk from position0 per bucket, not an advancing long-prompt prefill",
            "Reduced model residency and activation history differ from all48 layers",
            "No sampling, checkpoint save, scheduler migration, or caller read inside measured body",
            "Full-model embeddings, three complete decoder kinds, final mixer/head and prefill output copy are included",
        ],
    }
    started = time.monotonic()
    gen = mesh = None
    phase = "preflight"
    try:
        assert "TT_METAL_WATCHER" not in os.environ, "Watcher must be unset, including empty/0 values"
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "").startswith("1"), "Run with the device profiler enabled"
        assert (
            int(os.environ.get("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "0")) >= 8192
        ), "Use Tracy --op-support-count 8192"
        for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS"):
            assert os.environ.get(name) == "0", f"Set {name}=0 for profiler attribution"
        for name in ("TT_METAL_TRACE_PROFILER", "TT_METAL_STREAMING_PROFILER"):
            assert os.environ.get(name) in (None, "", "0"), f"Disable {name}"
        assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
        expected = repo / "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
        assert Path(os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")).resolve() == expected
        assert os.environ.get("TT_VISIBLE_DEVICES") == "0,1,2,3"
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()
        report["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True, timeout=10).splitlines()

        import torch
        from tracy import signpost
        from ttnn.tools import trace_allocation_tracker

        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tests import run_full_model_acceptance as acceptance
        from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

        assert not trace_allocation_tracker.TRACE_ALLOC_TRACKING
        assert not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024, l1_small_size=32768)

        def drain():
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)

        phase = "build_reduced_generator"
        options = {"layer_indices": (0, 1, 3), "max_batch_size": 1, "max_seq_len": 262144}
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        gen = build_generator(str(model_dir), mesh, **options)
        assert tuple(gen.model.layer_indices) == (0, 1, 3) and gen.capacity == 262144
        assert gen.prefill_cache is gen.caches[0]
        gen._check_cache_setup()
        assert_no_traces(gen)
        drain()
        prompt = acceptance.natural_prompt(gen, model_dir, max(BUCKETS))
        assert len(prompt) == 32
        prompt_path = model_dir / "doc/full_model/hf_reference/prompt.txt"
        report["prompt"] = {
            "source": str(prompt_path.relative_to(repo)),
            "source_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "token_ids": list(map(int, prompt)),
            "construction": "acceptance.natural_prompt; each bucket uses the first N tokens",
            "token_words": "zeros[1,N,16], words[...,0]=token_ids, words[...,1]=1; identical to prefill_forward",
        }

        def reset_and_upload(count):
            # Exact public prefill token/word staging, outside attribution.
            gen._reset_cache(gen.prefill_cache)
            logical = torch.tensor([prompt[:count]], dtype=torch.int32)
            words = torch.zeros(1, count, 16, dtype=torch.int32)
            words[:, :, 0], words[:, :, 1] = logical, 1
            gen._refresh(gen.prefill_inputs[count], logical)
            gen._refresh(gen.prefill_words[count], words)
            drain()

        for count in BUCKETS:
            phase = f"warm_prefill_{count}"
            reset_and_upload(count)
            gen._prefill_body(count)
            drain()
            assert_no_traces(gen)
            phase = f"measured_prefill_{count}"
            reset_and_upload(count)
            cache_entries = mesh.num_program_cache_entries()
            label = f"PREFILL_BUCKET_{count}"
            signpost(label)
            try:
                gen._prefill_body(count)
            finally:
                # Only the device body is enclosed; synchronization/drain and
                # diagnostics are outside both markers, even on failure.
                signpost(label + "_END")
            drain()
            assert mesh.num_program_cache_entries() == cache_entries, "Measured bucket compiled an unwarmed program"
            assert_no_traces(gen)
            # Scheduler metadata normally advances in public prefill_forward.
            # This diagnostic body bypasses that caller, so update it outside
            # the region before checking device/scheduler agreement.
            gen.prefill_cache.position = count
            positions = acceptance.qsa_positions(gen, count)
            finite = ttnn.min(ttnn.isfinite(gen.prefill_output[:, :, :count, :]), dim=-1, keepdim=True)
            per_rank = [ttnn.to_torch(value).reshape(-1).tolist() for value in ttnn.get_device_tensors(finite)]
            assert len(per_rank) == 4 and all(
                len(values) == count and all(v == 1 for v in values) for values in per_rank
            )
            del finite
            drain()
            report["measurements"].append(
                {
                    "logical_rows": count,
                    "start_position": 0,
                    "warmup_chunks": 1,
                    "measured_chunks": 1,
                    "signposts": [label, label + "_END"],
                    "input_shape": list(gen.prefill_inputs[count].shape),
                    "word_shape": list(gen.prefill_words[count].shape),
                    "output_shape": list(gen.prefill_output.shape),
                    "output_padded_shape": list(gen.prefill_output.padded_shape),
                    "program_cache_entries_after_warmup": cache_entries,
                    "no_measured_program_compilation": True,
                    "all_active_logits_finite": True,
                    "finiteness_per_rank": per_rank,
                    "qsa_positions": positions,
                    "no_live_traces": True,
                }
            )
            print(json.dumps({"bucket": count, "phase": "complete", "no_live_traces": True}), flush=True)
        report["profile_contract"] = {
            "no_live_traces": True,
            "measured_host_uploads": 0,
            "measured_host_reads": 0,
            "generator_prepare_called": False,
            "profile_is_eager_not_trace_latency": True,
            "profiler_drained_before_and_after_each_warm_and_measured_chunk": True,
            "advice_required": True,
        }
        report["passed"] = True
    except BaseException as error:
        report.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(json.dumps({"passed": report["passed"], "phase": phase, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
