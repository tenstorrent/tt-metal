# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full advertised context, streamed HF oracle, full and non-aligned TT prefill."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.output:
        record(args.output)
    if args.oracle_only and not args.oracle_cache:
        raise ValueError("--oracle-only requires --oracle-cache")
    torch.set_num_threads(8)
    torch.manual_seed(154)
    c = config()
    n = args.length
    sd = weights(args.layer)
    x = torch.randn(1, n, 2048, dtype=torch.bfloat16) * 0.05
    identity = {
        "layer": args.layer,
        "length": n,
        "seed": 154,
        "config": c.to_dict(),
        "versions": {v: str(importlib.metadata.version(v)) for v in ("torch", "transformers")},
        "reference_source": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("reference.py", "long_context.py")
        },
        "weight_sha256": {
            name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in sd.items()
        },
    }
    oracle_path = Path(args.oracle_cache) if args.oracle_cache else None
    if oracle_path is not None and oracle_path.exists():
        saved = torch.load(oracle_path, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, "HF oracle cache identity mismatch"
        expected = saved["expected"]
        assert expected.shape == x.shape and expected.dtype == x.dtype
        print("HF_ORACLE_CACHE_VERIFIED", oracle_path, flush=True)
    else:
        hf = layer(sd, args.layer)
        cache = DynamicCache(config=c)
        expected = torch.empty_like(x)
        start = time.monotonic()
        for off in range(0, n, 1024):
            expected[:, off : off + 1024] = hf_forward(hf, x[:, off : off + 1024], cache, off)
            if off % 16384 == 0:
                print("HF_PROGRESS", off, n, time.monotonic() - start, flush=True)
        print("HF_DONE", time.monotonic() - start, flush=True)
        del hf, cache
        if oracle_path is not None:
            oracle_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = oracle_path.with_suffix(".pending")
            torch.save({"identity": identity, "expected": expected}, temporary)
            temporary.replace(oracle_path)
    if args.oracle_only:
        assert "ttnn" not in sys.modules, "host-only oracle unexpectedly imported TTNN"
        summary = {
            "mode": "host-only HF oracle; not a TTNN correctness result",
            "layer": args.layer,
            "length": n,
            "cache": str(oracle_path),
            "identity": identity,
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "finite": bool(torch.isfinite(expected).all()),
            "ttnn_imported": False,
        }
        assert summary["finite"]
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
        print("HF_ORACLE_ONLY_PASS", n, oracle_path, flush=True)
        return

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    result = {
        "layer": args.layer,
        "advertised": c.max_position_embeddings,
        "tested": n,
        "hf_reference": "real weights; HF layer with DynamicCache; 1024-token continuation chunks",
        "results": [],
    }
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=n, page_order=torch.randperm((n + 31) // 32))
        )
        keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")

        def upload(x):
            return ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        for length in (n - 1, n):
            for k in keys:
                ttnn.copy(ttnn.zeros_like(state[k]), state[k])
            tx = upload(x[:, :length])
            plan = dec.prepare_prefill(seq_len=length)
            t0 = time.monotonic()
            with device_only():
                out = dec.prefill_forward(tx, state=state, plan=plan)
            got = ttnn.to_torch(out).reshape(1, length, 2048)
            score = pcc(expected[:, :length], got)
            # Check windows at both ends and every chunk transition, not only aggregate PCC.
            windows = [0, 31, 255, 256, 512, length - 33, length - 1]
            perrow = {str(i): pcc(expected[:, i], got[:, i]) for i in windows}
            rec = {
                "mode": "prefill",
                "length": length,
                "pcc": score,
                "selected_row_pcc": perrow,
                "host_wall_seconds_including_readback": time.monotonic() - t0,
            }
            result["results"].append(rec)
            print("LONG_PREFILL", rec, flush=True)
            assert score >= 0.995 and min(perrow.values()) >= 0.995, rec
            del out, got, tx, plan
            if length == n - 1:
                td = upload(x[:, -1:])
                pos = dec.prepare_positions(start_pos=n - 1, seq_len=1) if dec.kind == "full_attention" else None
                snapshot = {k: ttnn.clone(state[k]) for k in keys}

                def restore():
                    for k in keys:
                        ttnn.copy(snapshot[k], state[k])

                with device_only():
                    warm = dec.decode_forward(td, state=state, positions=pos)
                del warm
                restore()
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    out = dec.decode_forward(td, state=state, positions=pos)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                from ttnn.tools import trace_allocation_tracker

                trace_allocation_tracker.acknowledge_corruptible(out)
                first = None
                for j in range(3):
                    restore()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    got = ttnn.to_torch(out).reshape(1, 1, 2048)
                    score = pcc(expected[:, -1:], got)
                    assert score >= 0.995, ("last_decode", score)
                    if first is None:
                        first = got.clone()
                    else:
                        assert torch.equal(first, got)
                result["results"].append(
                    {
                        "mode": "traced_decode",
                        "position": n - 1,
                        "context": n,
                        "pcc": score,
                        "repeats": 3,
                        "deterministic": True,
                    }
                )
                print("LONG_DECODE", result["results"][-1], flush=True)
                ttnn.release_trace(mesh, trace)
                del snapshot, out, td, pos
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--length", type=int, default=262144)
    p.add_argument("--output", required=True)
    p.add_argument("--oracle-cache", help="Optional local HF reference cache, validated against weights/config/source")
    p.add_argument("--oracle-only", action="store_true", help="Prepare/validate the HF oracle without importing TTNN")
    run(p.parse_args())
