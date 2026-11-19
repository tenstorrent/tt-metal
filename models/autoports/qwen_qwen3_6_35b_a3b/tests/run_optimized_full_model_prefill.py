# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial full-stack TTFT boundaries and one optional nonaligned 32K smoke."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.datatype_runtime_identity import (
    runtime_identity,
    source_input_identity,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="64,255,256,257,1025,2049")
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous evidence"
    lengths = [int(x) for x in args.lengths.split(",")]
    assert max(lengths) + 2 <= args.context
    assert not os.environ.get("TT_METAL_WATCHER")
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "0"
    torch.set_num_threads(8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = ROOT / "doc/datatype_sweep/selected_precision_config.json"
    fixture_path = ROOT / "doc/full_model/hf_reference/aime24_prompt_tokens.json"
    selected_sha256 = hashlib.sha256(selected.read_bytes()).hexdigest() if selected.is_file() else None
    provenance_inputs = [Path(__file__), fixture_path] + ([selected] if selected.is_file() else [])
    result = {
        "passed": False,
        "command": sys.argv,
        "time_unix": time.time(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mesh": [1, 4],
        "regime": "serial",
        "layers": list(range(40)),
        "context": args.context,
        "rows": [],
        "selected_precision_sha256_before": selected_sha256,
        "fixture": {"path": str(fixture_path), "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest()},
        "native_runtime_identity": runtime_identity(),
        "source_input_identity": source_input_identity(model_root=ROOT, explicit_paths=provenance_inputs),
        "native_identity_capture": "Actual prefill process, before mesh construction and after teardown",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((ROOT / "tt").glob("*.py"))]
        },
        "scope": "Warmed public prefill through first sampled token delivery; two queued traced decode steps; repeated AIME token fixture, not quality evidence",
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    mesh = generator = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, max_seq_len=args.context)
        result["normal_default_construction"] = True
        result["runtime_precision"] = model.precision_summary()
        if selected.exists():
            result["selected_precision_artifact"] = str(selected)
            result["selected_precision_sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
            assert result["runtime_precision"]["selected_policy"] == json.loads(selected.read_text())
        generator = QwenGenerator(model)
        generator.configure_sampling(temperature=0, top_k=1, seed=0)
        fixture = json.loads(fixture_path.read_text())

        def prefill(length):
            ids = torch.tensor([(fixture * ((length + len(fixture) - 1) // len(fixture)))[:length]])
            logits = generator.prefill_forward(ids, prompt_lens=[length], device_output=True)[0]
            logits = ttnn.reshape(logits, (1, 1, 1, logits.shape[-1]))
            ttnn.copy(ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], 0), generator.sampler_logits)
            ttnn.deallocate(logits)
            generator._copy_host(torch.tensor([length], dtype=torch.int32), generator.positions[0])
            generator.counters["position_refreshes"] += 1
            generator._sample() if generator.prepared else generator._sampling_forward()
            return generator._read_tokens([0])[0]

        # Warm output read layouts before capture; prepare includes every physical bucket.
        prefill(32)
        ttnn.to_torch(ttnn.get_device_tensors(generator.positions[0])[0])
        generator.prepare()
        trace_ids = dict(generator.traces)
        entries = int(mesh.num_program_cache_entries())
        result["prepared_census"] = model.census("stage7_prefill_prepared")
        for length in lengths:
            if length <= 8192:
                generator.reset()
                prefill(length)
            generator.reset()
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            first = prefill(length)
            elapsed = time.perf_counter() - started
            before = dict(generator.counters)
            with device_only():
                for _ in range(2):
                    generator.decode_forward(None, [0], refresh_inputs=False, read_tokens=False)
            ttnn.synchronize_device(mesh)
            delta = {k: generator.counters[k] - v for k, v in before.items()}
            assert delta["model_replays"] == delta["sampling_replays"] == 2
            for key in ("readbacks", "token_refreshes", "position_refreshes", "page_table_refreshes", "captures"):
                assert delta[key] == 0, delta
            final = generator._read_tokens([0])[0]
            position = int(ttnn.to_torch(ttnn.get_device_tensors(generator.positions[0])[0]).flatten()[0])
            assert position == length + 2 and 0 <= first < model.vocab_size and 0 <= final < model.vocab_size
            assert dict(generator.traces) == trace_ids
            assert int(mesh.num_program_cache_entries()) == entries
            row = {
                "logical_length": length,
                "ttft_ms": elapsed * 1000,
                "first_token": first,
                "final_token": final,
                "next_position": position,
                "decode_counter_delta": delta,
                "program_cache_entries": entries,
                "linear_attention_chunk": 64,
                "full_attention_chunk": 256,
                "moe_chunk": 2048,
                "unaligned": bool(length % 32),
                "warmup": "same length once" if length <= 8192 else "all physical variants; one long smoke only",
            }
            result["rows"].append(row)
            print("PREFILL_BOUNDARY", json.dumps(row), flush=True)
            save()
        result["final_census"] = model.census("stage7_prefill_complete")
        result["trace_ids"] = {k: int(v) for k, v in trace_ids.items()}
        result["passed"] = True
    finally:
        try:
            if generator is not None:
                generator.close()
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
        finally:
            provenance_errors = []
            try:
                result["native_runtime_identity_after"] = runtime_identity()
                result["source_input_identity_after"] = source_input_identity(
                    model_root=ROOT, explicit_paths=provenance_inputs
                )
                result["selected_precision_sha256_after"] = (
                    hashlib.sha256(selected.read_bytes()).hexdigest() if selected.is_file() else None
                )
                result["fixture_sha256_after"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
                if result["native_runtime_identity_after"] != result["native_runtime_identity"]:
                    provenance_errors.append("Native source/build identity changed")
                if result["source_input_identity_after"] != result["source_input_identity"]:
                    provenance_errors.append("Source/input identity changed")
                if result["selected_precision_sha256_after"] != selected_sha256:
                    provenance_errors.append("Selected precision artifact changed")
                if result["fixture_sha256_after"] != result["fixture"]["sha256"]:
                    provenance_errors.append("Prompt fixture changed")
            except Exception as exc:
                provenance_errors.append(str(exc))
            if provenance_errors:
                result["passed"] = False
                result["provenance_errors"] = provenance_errors
            save()
            if provenance_errors and sys.exc_info()[0] is None:
                raise RuntimeError("Prefill provenance failed; see artifact provenance_errors")


if __name__ == "__main__":
    main()
