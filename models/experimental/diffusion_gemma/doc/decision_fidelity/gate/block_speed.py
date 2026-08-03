#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Output speed and phase breakdown for a served run, from its ``DG_VLLM_METRIC`` telemetry.

**Do not read speed off vLLM's own ``Avg generation throughput`` line.** DG commits a whole 256-token
block at once, so vLLM sees 256 tokens arrive in one step and reports instantaneous spikes -- 921,
972, 1100 tok/s were all in one real log whose sustained rate was 44.7. The honest number is committed
tokens divided by block latency, which is what this computes.

Also drops the smoke stage. The runner smoke-tests a couple of questions against the same server, so a
log can hold 200 ``prefill_block0`` events for a 198-question eval; leaving those in mixes two configs
into one average.

Usage::

    block_speed.py <server.log>
    block_speed.py <server.log> --expect 198        # drop leading requests beyond this count
    block_speed.py <server.log> --json              # machine-readable, for diffing two runs
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

METRIC = re.compile(r"DG_VLLM_METRIC (\{.*\})\s*$")


def read(path: Path):
    """The metric events, plus the interleaving order needed to find request boundaries."""
    b0, dec, rel, rej, build, order = [], [], [], [], None, []
    with path.open(errors="replace") as fh:
        for line in fh:
            m = METRIC.search(line.rstrip())
            if not m:
                continue
            try:
                ev = json.loads(m.group(1))
            except ValueError:
                continue
            event = ev.get("event")
            if event == "prefill_block0":
                b0.append(ev)
                order.append(("b0", len(b0) - 1))
            elif event == "decode_block":
                dec.append(ev)
                order.append(("dec", len(dec) - 1))
            elif event == "request_release":
                rel.append(ev)
            elif event == "prefill_rejected":
                # An unwarmed prefill length ends ONE request with an empty answer while the server
                # stays up (tt/generator_vllm.py). Since DG_UPFRONT_STRICT_PREFILL_LENS was deleted
                # 2026-08-03 there is no engine-fatal arm left, so this event is the only evidence
                # that a sample was silently lost -- a speed number computed over the survivors is
                # not the configuration it claims to measure.
                rej.append(ev)
            elif event == "model_build":
                build = ev
    return b0, dec, rel, rej, build, order


def drop_leading(b0, dec, order, keep: int):
    """Keep only the last `keep` requests -- i.e. drop the smoke stage that ran first."""
    starts = [i for i, (kind, _j) in enumerate(order) if kind == "b0"]
    extra = len(starts) - keep
    if extra <= 0:
        return b0, dec, 0
    tail = order[starts[extra] :]
    return ([b0[j] for k, j in tail if k == "b0"], [dec[j] for k, j in tail if k == "dec"], extra)


def measure(b0: list, dec: list) -> dict:
    blocks = b0 + dec
    if not blocks:
        sys.exit("no block events in that log -- is it a served run's server.log?")
    tok = sum(b["committed_tokens"] for b in blocks)
    lat = sum(b["block_latency_s"] for b in blocks)
    per = sorted(b["committed_tokens"] / b["block_latency_s"] for b in blocks if b["block_latency_s"] > 0)
    out = {
        "requests": len(b0),
        "blocks": len(blocks),
        "committed_tokens": tok,
        "block_time_s": round(lat, 1),
        "output_tok_s": round(tok / lat, 1) if lat else None,
        "per_block_tok_s_p50": round(st.median(per), 1),
        "per_block_tok_s_mean": round(st.mean(per), 1),
        "per_block_tok_s_p10": round(per[len(per) // 10], 1),
        "per_block_tok_s_p90": round(per[9 * len(per) // 10], 1),
    }
    if dec:
        bl = [b["block_latency_s"] for b in dec]
        tl = sum(bl)
        ds = [b["denoise_steps"] for b in dec]
        out.update(
            decode_latency_s_p50=round(st.median(bl), 2),
            decode_latency_s_mean=round(st.mean(bl), 2),
            decode_latency_s_max=round(max(bl), 2),
            denoise_pct=round(sum(b["denoise_latency_s"] for b in dec) / tl * 100, 1),
            commit_pct=round(sum(b["commit_latency_s"] for b in dec) / tl * 100, 1),
            denoise_steps_p50=st.median(ds),
            denoise_steps_mean=round(st.mean(ds), 1),
            denoise_steps_max=max(ds),
            halted_blocks=sum(1 for b in dec if b.get("halted")),
            decode_blocks=len(dec),
        )
    if b0:
        out.update(
            prefill_s_p50=round(st.median([b["prefill_s"] for b in b0]), 2),
            ttft_s_p50=round(st.median([b["ttft_s"] for b in b0]), 2),
            ttft_s_mean=round(st.mean([b["ttft_s"] for b in b0]), 2),
            ttft_s_max=round(max(b["ttft_s"] for b in b0), 2),
        )
    return out


def show(m: dict) -> None:
    print(f"  requests {m['requests']}   blocks {m['blocks']}   committed tokens {m['committed_tokens']:,}")
    print(f"  block time {m['block_time_s']:,.0f} s = {m['block_time_s']/3600:.2f} h")
    print(f"  OUTPUT SPEED: {m['output_tok_s']} tok/s aggregate")
    print(
        f"    per block: p50 {m['per_block_tok_s_p50']}  mean {m['per_block_tok_s_mean']}  "
        f"p10 {m['per_block_tok_s_p10']}  p90 {m['per_block_tok_s_p90']}"
    )
    if "decode_latency_s_p50" in m:
        print(
            f"  decode block latency s: p50 {m['decode_latency_s_p50']}  mean {m['decode_latency_s_mean']}  "
            f"max {m['decode_latency_s_max']}"
        )
        print(f"  time split: denoise {m['denoise_pct']}%  commit {m['commit_pct']}%")
        print(
            f"  denoise steps/block: p50 {m['denoise_steps_p50']:.0f}  mean {m['denoise_steps_mean']}  "
            f"max {m['denoise_steps_max']}   halted {m['halted_blocks']}/{m['decode_blocks']}"
        )
    if "ttft_s_p50" in m:
        print(
            f"  prefill_s p50 {m['prefill_s_p50']} | ttft_s p50 {m['ttft_s_p50']}  "
            f"mean {m['ttft_s_mean']}  max {m['ttft_s_max']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server_log", type=Path)
    ap.add_argument("--expect", type=int, default=None, help="question count; leading extras are the smoke stage")
    ap.add_argument("--json", action="store_true", help="emit the metrics as JSON")
    args = ap.parse_args()

    b0, dec, rel, rej, build, order = read(args.server_log)
    if build:
        print(f"model_build_s {build['model_build_s']:.0f}   trace_region {build.get('trace_region_size_env')}")
    print(f"prefill_block0 events in log: {len(b0)}")
    if rej:
        lens = sorted({r.get("cache_len") for r in rej})
        print(
            f"  !! {len(rej)} REJECTED prefill(s) at aligned length(s) {lens} -- those requests "
            f"returned an empty answer. Warm them via DG_UPFRONT_PREFILL_WARMUP_LENS and re-run; "
            f"the numbers below cover only the requests that were served."
        )
    if args.expect and len(b0) != args.expect:
        # Not an error: the smoke stage explains a small excess. A SHORTFALL is the false-green signal.
        verb = "excess (smoke stage?)" if len(b0) > args.expect else "SHORTFALL -- engine may have died mid-run"
        print(f"  !! expected {args.expect}: {abs(len(b0) - args.expect)} {verb}")

    scoped_b0, scoped_dec, dropped = (b0, dec, 0)
    if args.expect:
        scoped_b0, scoped_dec, dropped = drop_leading(b0, dec, order, args.expect)

    m = measure(scoped_b0, scoped_dec)
    if args.json:
        print(json.dumps(m, indent=2))
        return 0
    if dropped:
        print(f"(dropped the first {dropped} request(s) as the smoke stage)")
    print()
    show(m)
    if rel:
        bpr = [r["blocks_emitted"] for r in rel]
        print(f"\n  blocks per request: p50 {st.median(bpr):.0f}  mean {st.mean(bpr):.1f}  max {max(bpr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
