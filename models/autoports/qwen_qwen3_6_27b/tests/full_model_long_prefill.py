# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Maximum-context public-wrapper prefill followed by one eager decode."""

from __future__ import annotations

import argparse
import json
import resource
import threading
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=192511)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--stack-chunk-size", type=int, default=None)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--compare-ordinary", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=300.0,
        help="print host-side elapsed time and peak RSS while a long device call is in flight; 0 disables",
    )
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=args.max_context,
            batch=1,
            num_layers=args.num_layers,
            layer_indices=args.layer_indices,
        )
        tokens = (torch.arange(args.sequence, dtype=torch.int64) % generator.model.vocab_size).reshape(1, -1)

        def run_with_heartbeat(label, operation):
            stop = threading.Event()
            started = time.monotonic()

            def report():
                while not stop.wait(args.heartbeat_seconds):
                    elapsed = time.monotonic() - started
                    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                    print(
                        f"LONG_PREFILL_HEARTBEAT stage={label} elapsed_seconds={elapsed:.1f} "
                        f"peak_rss_gib={peak_rss_gib:.3f}",
                        flush=True,
                    )

            thread = None
            if args.heartbeat_seconds > 0:
                thread = threading.Thread(target=report, name="long-prefill-heartbeat", daemon=True)
                thread.start()
            print(f"LONG_PREFILL_START stage={label} sequence={args.sequence}", flush=True)
            try:
                return operation()
            finally:
                stop.set()
                if thread is not None:
                    thread.join()
                elapsed = time.monotonic() - started
                peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                print(
                    f"LONG_PREFILL_END stage={label} elapsed_seconds={elapsed:.1f} " f"peak_rss_gib={peak_rss_gib:.3f}",
                    flush=True,
                )

        def run_once(stack_chunk_size):
            generator.reset()
            generator.model.PREFILL_STACK_CHUNK_SIZE = stack_chunk_size
            prefill = run_with_heartbeat(
                "prefill",
                lambda: generator.prefill_forward(
                    tokens,
                    page_table=generator._page_table,
                    kv_cache=generator.kv_cache,
                    prompt_lens=[args.sequence],
                ),
            )
            token = int(torch.argmax(prefill[0, 0]).item())
            decode = generator.decode_forward(
                torch.tensor([[token]], dtype=torch.int64),
                torch.tensor([args.sequence], dtype=torch.int64),
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
            )
            cache = []
            if args.compare_ordinary:
                for layer in generator.model.layers:
                    for name in sorted(layer.caches):
                        cache.extend(
                            ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(layer.caches[name])
                        )
            return prefill, token, decode, cache

        if args.compare_ordinary:
            ordinary = run_once(args.sequence + 1)
            streamed = run_once(args.stack_chunk_size or 64)

            def pcc(lhs, rhs):
                lhs, rhs = lhs.float().reshape(-1), rhs.float().reshape(-1)
                if torch.equal(lhs, rhs):
                    return 1.0
                return float(torch.corrcoef(torch.stack((lhs, rhs)))[0, 1])

            prefill_pcc = pcc(ordinary[0], streamed[0])
            decode_pcc = pcc(ordinary[2], streamed[2])
            cache_pcc = [pcc(lhs, rhs) for lhs, rhs in zip(ordinary[3], streamed[3])]
            result = {
                "sequence": args.sequence,
                "ordinary_first_token": ordinary[1],
                "streamed_first_token": streamed[1],
                "prefill_pcc": prefill_pcc,
                "decode_pcc": decode_pcc,
                "minimum_cache_pcc": min(cache_pcc),
                "cache_tensors_compared": len(cache_pcc),
                "prefill_top1_equal": ordinary[1] == streamed[1],
            }
            if not (
                result["prefill_top1_equal"]
                and prefill_pcc >= 0.999
                and decode_pcc >= 0.999
                and result["minimum_cache_pcc"] >= 0.999
            ):
                raise AssertionError(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return

        if args.stack_chunk_size is not None:
            generator.model.PREFILL_STACK_CHUNK_SIZE = args.stack_chunk_size
        if args.skip_decode:
            generator.reset()
            prefill_logits = run_with_heartbeat(
                "prefill",
                lambda: generator.prefill_forward(
                    tokens,
                    page_table=generator._page_table,
                    kv_cache=generator.kv_cache,
                    prompt_lens=[args.sequence],
                ),
            )
            first_token = int(torch.argmax(prefill_logits[0, 0]).item())
            decode_logits = None
        else:
            prefill_logits, first_token, decode_logits, _ = run_once(generator.model.PREFILL_STACK_CHUNK_SIZE)
        result = {
            "sequence": args.sequence,
            "prefill_shape": list(prefill_logits.shape),
            "prefill_finite": bool(torch.isfinite(prefill_logits).all()),
            "first_token": first_token,
            "decode_shape": None if decode_logits is None else list(decode_logits.shape),
            "decode_finite": None if decode_logits is None else bool(torch.isfinite(decode_logits).all()),
            "prefill_request_state_cleared": all(
                layer._sequence_masks is None
                and layer._conv_state_selector_chunks is None
                and layer._sequence_mask is None
                and layer._conv_state_selectors is None
                and layer._prefill_chunk_start is None
                for layer in generator.model.layers
            ),
        }
        if not all(
            (
                result["prefill_finite"],
                result["decode_finite"] is not False,
                result["prefill_request_state_cleared"],
            )
        ):
            raise AssertionError(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
