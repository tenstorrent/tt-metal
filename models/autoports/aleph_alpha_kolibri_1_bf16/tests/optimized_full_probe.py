# SPDX-License-Identifier: Apache-2.0
"""Precision-locked prefill replay comparison on the complete generator contract."""

import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator
from .full_provenance import provenance


def main():
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/optimized_full_model"))
    result = dict(provenance=provenance(), rows=[])
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        setup_started = time.monotonic()
        gen = build_generator(root, mesh, layer_indices=[0, 4])
        # Compare terminal layout with real preceding layers and fixed precision.
        norms = []
        values = []
        for sharded in (False, True):
            gen.model.sharded_terminal_norm = sharded
            gen.bind([42], [0])
            gen._decode()
            values.append(gen.read_logits().clone())
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            gen._decode()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for _ in range(5):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            tick = time.monotonic()
            for _ in range(100):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            norms.append(dict(sharded=sharded, traced_model_ms=(time.monotonic() - tick) * 10))
            ttnn.release_trace(mesh, trace)
        norm_pcc = torch.corrcoef(torch.stack([v.float().flatten() for v in values]))[0, 1].item()
        assert norm_pcc >= 0.9999, norm_pcc
        assert torch.equal(values[0].argmax(-1), values[1].argmax(-1))
        result["terminal_norm"] = dict(rows=norms, pcc=norm_pcc, top1_equal=True)
        del values
        gen.prepare()
        result["setup_seconds"] = time.monotonic() - setup_started
        traces = dict(gen.traces)
        counts = gen.counters.copy()
        # Changed rules: physical buckets, 256-aligned traced starts, 65536
        # precision boundary; unaligned starts continue through decode.
        cases = [(n, 0) for n in (31, 32, 33, 127, 128, 129, 511, 512, 513, 2047, 2048, 2049, 8191, 8192, 8193)]
        cases += [(33, 31), (129, 32), (513, 256), (129, 65280), (129, 65536), (33, 1048543)]
        for length, start in cases:
            outputs = []
            times = []
            for traced in (False, True):
                gen.trace_prefill = traced
                gen.reset()
                # Populate the same previous prefix for offset probes. Last
                # maximum-context probe is an address test, not quality evidence.
                if 0 < start <= 65536:
                    gen._prefill([42] * start)
                ttnn.synchronize_device(mesh)
                tick = time.monotonic()
                first = gen.prefill_forward(
                    torch.full((1, length), 42, dtype=torch.long),
                    page_table=gen.state.host_page_tables,
                    kv_cache=gen.state,
                    prompt_lens=[length],
                    start_pos=[start],
                )
                times.append((time.monotonic() - tick) * 1000)
                logits = gen.read_logits().clone()
                ids = [int(first[0])]
                for _ in range(min(3, gen.logical_capacity - start - length)):
                    ids.append(int(gen.decode_forward(None, None, page_table=None, kv_cache=gen.state)[0]))
                outputs.append((ids, logits))
            assert outputs[0][0] == outputs[1][0], (length, start, outputs[0][0], outputs[1][0])
            assert torch.equal(outputs[0][1], outputs[1][1]), (
                length,
                start,
                (outputs[0][1] - outputs[1][1]).abs().max(),
            )
            assert gen.traces == traces
            row = dict(
                length=length,
                start=start,
                eager_ms=times[0],
                traced_ms=times[1],
                exact_logits=True,
                tokens=outputs[1][0],
            )
            result["rows"].append(row)
            (out / "prefill_probe.json").write_text(json.dumps(result, indent=2) + "\n")
            print("PREFILL_PROBE_ROW", row, flush=True)
        result.update(
            passed=True,
            trace_ids={k: str(v) for k, v in traces.items()},
            counters=dict(gen.counters - counts),
            seconds=time.monotonic() - setup_started,
        )
        (out / "prefill_probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print("OPTIMIZED_FULL_PROBE_PASS", flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
