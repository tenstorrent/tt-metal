# SPDX-License-Identifier: Apache-2.0
"""Source-shape byte lower bound and same-capture host/device reconciliation."""

import csv
import json

from .multichip_sweep import OUT


def main():
    # One BFP4 32x32 tile has 512 mantissa + 64 exponent bytes.
    bfp4 = lambda elements: elements // 1024 * 576
    weights = dict(
        qkv=bfp4(2560 * 1792),
        wo=bfp4(1536 * 2560),
        router_replicated=bfp4(2560 * 1024),
        shared_gate_up_down=bfp4(2560 * 256 + 128 * 2560),
        six_active_experts=bfp4(6 * (2560 * 256 + 128 * 2560)),
        norm_tiles=4 * 80 * 2048 + 2 * 4 * 2048,
        router_bias_tiles=32 * 4096,
    )
    rows = []
    for layer, tokens in ((0, 128), (4, 128), (0, 8193)):
        start = max(0, tokens - 512) if layer == 0 else 0
        useful_pages = tokens // 32 - start // 32 + 1
        # rt_args_common.hpp:get_workload_for_core rounds the accessed interval
        # to the selected SDPA K chunk, not just to a 32-token cache page.
        k_chunk = 256 if layer == 0 and tokens > 8192 else 128
        read_start = start // k_chunk * k_chunk
        read_end = (tokens + 1 + k_chunk - 1) // k_chunk * k_chunk
        pages = (read_end - read_start) // 32
        # Both K/V, one local head, four tiles per 32-token page, BFP8=1088 bytes/tile.
        kv_bytes = 2 * pages * 4 * 1088
        per_device_bytes = sum(weights.values()) + kv_bytes
        folder = OUT / f"profile_final_{layer}_{tokens}"
        device = []
        for rank in range(4):
            with (folder / f"decode_device{rank}_report.csv").open() as f:
                ops = list(csv.DictReader(f))
            kernel = sum(float(r["Device Time"] or 0) for r in ops)
            gaps = sum(float(r["Op-to-Op Gap"] or 0) for r in ops)
            device.append(dict(device=rank, kernel_us=kernel, inter_op_gap_us=gaps, span_us=kernel + gaps))
        profiled = json.loads((OUT / f"profiled_final_{layer}_{tokens}.json").read_text())
        final = json.loads((OUT / f"final_{layer}_{tokens}.json").read_text())
        span = max(d["span_us"] for d in device)
        host = profiled["latency_ms"]["decode"] * 1000
        rows.append(
            dict(
                layer=layer,
                current_position=tokens,
                useful_kv_tokens=tokens - start + 1,
                useful_kv_pages=useful_pages,
                touched_kv_pages=pages,
                sdpa_k_chunk=k_chunk,
                chunk_aligned_read_interval=[read_start, read_end],
                local_kv_read_bytes=kv_bytes,
                weight_bytes_per_device=weights,
                minimum_bytes_per_device=per_device_bytes,
                aggregate_bytes=4 * per_device_bytes,
                aggregate_bandwidth_bytes_per_second=4 * 512e9,
                roofline_us=per_device_bytes / 512e9 * 1e6,
                same_capture_device=device,
                same_capture_device_span_us=span,
                same_capture_host_us=host,
                same_capture_host_minus_device_us=host - span,
                warmed_100_replay_host_us=final["latency_ms"]["decode"] * 1000,
            )
        )
    (OUT / "decode_accounting.json").write_text(
        json.dumps(
            dict(
                method="Minimum one read of stored active weight tiles and chunk-rounded unique local KV; includes SDPA chunk overfetch, excludes activation traffic, duplicate head reads and kernel rereads. 512 GB/s/device matches tt-perf-report Blackhole ceiling. Device span=sum kernel durations+in-window gaps, max across ranks. Host and device same-capture values use the one-replay profiler run; uninstrumented 100-replay timing is separate.",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
