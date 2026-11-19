# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-input runtime score checks with changed-input trace replay at real capacity."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_projection_precision import compact_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_score_windows_205504.pt"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    captured = torch.load(args.capture, weights_only=True, mmap=True, map_location="cpu")
    keys = captured["keys"]
    assert keys.shape == (1, 1, 65536, 128)
    report = {"scope": __doc__, "capture": str(args.capture), "physical_context": 262144, "shapes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA

    report["runtime_sha256"] = hashlib.sha256(Path(QSA.indexer_scores.__code__.co_filename).read_bytes()).hexdigest()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:

        def host(value):
            return ttnn.from_torch(value.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        key_tt = ttnn.to_device(host(keys), mesh)
        score_owner = SimpleNamespace(ttnn=ttnn)
        for count in (1, 3, 7, 31, 32):
            start = min(27, 32 - count)
            first = captured["query"][:, :, start : start + count].clone()
            queries = [first, first.flip(-1).contiguous()]
            query_host = [host(query) for query in queries]
            query_tt = ttnn.to_device(query_host[0], mesh)
            expected = [
                (query.float() @ keys.float().transpose(-1, -2)).relu().sum(1, keepdim=True) / math.sqrt(128)
                for query in queries
            ]
            with runtime_audit():
                warm = QSA.indexer_scores(score_owner, query_tt, key_tt)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                output = QSA.indexer_scores(score_owner, query_tt, key_tt)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(output)
            results, actuals = [], []
            for branch in (0, 1, 0):
                ttnn.copy_host_to_device_tensor(query_host[branch], query_tt)
                start_time = time.perf_counter()
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                seconds = time.perf_counter() - start_time
                actual = ttnn.to_torch(output).clone()
                actuals.append(actual)
                same_sets = [
                    set(expected[branch][0, 0, row].topk(512).indices.tolist())
                    == set(actual[0, 0, row].topk(512).indices.tolist())
                    for row in range(count)
                ]
                measured = compact_metrics(expected[branch], actual)
                results.append(
                    {
                        "branch": branch,
                        "traced_seconds": seconds,
                        "accuracy": measured,
                        "top512_sets_equal": same_sets,
                        "passed": measured["maximum_absolute_error"] < 1e-4 and all(same_sets),
                    }
                )
            report["shapes"][str(count)] = {
                "replays": results,
                "repeat_equal": torch.equal(actuals[0], actuals[2]),
                "changed_input_changes_output": not torch.equal(actuals[0], actuals[1]),
                "runtime_fallback_audit": "clean",
            }
            ttnn.release_trace(mesh, trace)
            trace = None
            ttnn.deallocate(output)
            ttnn.deallocate(query_tt)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"rows": count, **report["shapes"][str(count)]}), flush=True)
        report["passed"] = all(
            value["repeat_equal"]
            and value["changed_input_changes_output"]
            and all(replay["passed"] for replay in value["replays"])
            for value in report["shapes"].values()
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
