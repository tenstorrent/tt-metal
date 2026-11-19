# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All40 wrapper around the installed resumed-prefill readiness check.

Run serialized after setup:
 timeout 7200 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_resume_check --mode resume

Default natural, non-repeated book text; lengths257/1025, blocks32/128,
64-position windows plus final position, and both negative controls. Packaged
thresholds are unchanged. --reduced selects layers0/3 for diagnostics only.
A numerical pass does not declare recurrent prefix caching supported; serving
must preserve full matching recurrent/conv state and its documented exclusion.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("resume",), default="resume")
    p.add_argument("--lengths", default="257,1025")
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "readiness/prefill_resume_check.json")
    args = p.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    assert lengths and min(lengths) >= 257 and max(lengths) <= args.context
    book = Path.cwd() / "models/ttt_compat/tests/tale-of-two-cities.txt.bz2"
    assert book.is_file(), "Run from the checkout root with the canonical non-repeating readiness book"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch
    from readiness_check import run_prefill_resume_check as readiness

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "command": sys.argv,
        "time_unix": time.time(),
        "layers": [0, 3] if args.reduced else list(range(40)),
        "context": args.context,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(readiness.__file__),
                ROOT / "tt/generator.py",
                ROOT / "tt/model.py",
                ROOT / "tt/prefill.py",
                book,
            ]
        },
    }

    def save():
        output = args.output if result["status"] == "pass" else args.output.with_name(args.output.stem + ".failed.json")
        output.write_text(json.dumps(result, indent=2) + "\n")
        return output

    mesh = generator = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, layer_indices=[0, 3] if args.reduced else None, max_seq_len=args.context)
        generator = QwenGenerator(model, max_batch_size=1)
        generator.prefill_logits([1] * 33)  # Both full and logical-tail readbacks use physical32 logits.
        generator.prepare(prefill_cases=((32, 0),))
        prepared_entries = int(mesh.num_program_cache_entries())
        # The package owns the checker, its gates, controls and teardown; this
        # one-call constructor adapter supplies our explicitly prepared model.
        with patch.object(readiness, "_import_build_generator", lambda _root: lambda **kwargs: generator):
            record = readiness.run_prefill_resume_check(
                model_dir=ROOT,
                mesh_device=mesh,
                lengths=lengths,
                block_sizes=[32, 128],
                sliding_window=None,
                tile=32,
                window=64,
                controls=list(readiness.CONTROL_NAMES),
                thresholds=dict(readiness.DEFAULT_THRESHOLDS),
                vocab_size=None,
                seed=0,
                prompt_source="natural",
                text_file=book,
            )
        result.update(record)
        result["program_cache_entries_after_prepare"] = prepared_entries
        result["program_cache_entries_after_checks"] = int(mesh.num_program_cache_entries())
        result["counters"] = dict(generator.counters)
        result[
            "prefix_caching_declaration"
        ] = "Separate serving capability: recurrent_state exclusion; this validates continuation with the matching full recurrent and KV history."
        assert result["status"] == readiness.STATUS_PASS, result.get("reasons")
    except BaseException:
        result["status"] = "fail"
        result["error"] = traceback.format_exc()
        traceback.print_exc()
        try:
            save()
        except BaseException:
            traceback.print_exc()
        raise
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_errors = []
        if generator is not None and not generator.closed:
            try:
                generator.close()
            except BaseException:
                cleanup_errors.append({"operation": "generator.close", "error": traceback.format_exc()})
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:
                cleanup_errors.append({"operation": "close_mesh_device", "error": traceback.format_exc()})
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["status"] = "fail"
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        try:
            output = save()
            print("PREFILL_RESUME_RESULT", result["status"], str(output), flush=True)
        except BaseException:
            if not original_exception_pending:
                raise
            traceback.print_exc()
        if cleanup_errors and not original_exception_pending:
            raise RuntimeError("Resumed-prefill cleanup failed; see artifact cleanup_errors")


if __name__ == "__main__":
    main()
