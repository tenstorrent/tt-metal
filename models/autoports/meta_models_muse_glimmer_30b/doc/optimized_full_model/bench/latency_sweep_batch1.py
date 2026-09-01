# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 latency sweep across ISL at a fixed OSL, reporting E2EL.

Why this exists alongside the TTI serving sweep:

* The TTI sweep's largest ISL is 65536 - **half** the advertised 131072 context -
  and every point past ISL 128 uses OSL 128.  Nothing in the bring-up measured
  the advertised context, and nothing measured the OSL that matters for agentic
  coding.  The top point here is **ISL 130560 with OSL 512**, so
  ``ISL + OSL == 131072`` exactly: the advertised context, saturated.
* It reports **E2EL** as a first-class column.  E2EL is what a caller actually
  waits for, and the release report surfaced only TTFT/TPOT/throughput.

Method.  Two free-running generations per point:

* ``max_new_tokens=1`` gives **TTFT** (prefill + first sampled token).
* ``max_new_tokens=OSL`` gives **E2EL**, from which ``TPOT = (E2EL - TTFT) /
  (OSL - 1)``.

Both use the fast path.  Per-token ITL is deliberately *not* sampled: the only
per-step hook available (``next_input``) disables the free-running device loop
and restages a token every step, which would change the very number being
measured.  So ITL is reported as derived-uniform (== TPOT) and labelled as such
rather than presented as a measured distribution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from loguru import logger

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)

HF_ADVERTISED_CONTEXT = 131072
DEFAULT_OSL = 512
#: Top point saturates the advertised context exactly at OSL 512.
DEFAULT_ISLS = (128, 1024, 4096, 8192, 16384, 32768, 65536, 130560)


def synthetic_prompt(tokenizer, length: int) -> list[int]:
    """A prompt of exactly ``length`` tokens.

    Content is irrelevant to latency; length is not.  Built by repeating a mid-vocab
    token so no special/EOS id can end generation early and truncate a measurement.
    """
    filler = 1000
    return [filler] * length


def measure(gen, prompt_ids: list[int], osl: int) -> dict:
    # Warm THIS prefill shape before timing anything.  tt-metal JIT-compiles per
    # shape, so the first call at a new ISL pays compilation; without this the
    # TTFT measurement absorbs it (observed: 10.7 s at ISL 1024 against 223 ms
    # warm) and TPOT, derived as (E2EL - TTFT)/(n-1), is dragged negative-ward
    # into nonsense (4.33 ms/token, 231 t/s/u).  E2EL was unaffected because its
    # call already ran second, which is exactly how the bug hides.
    gen.reset()
    gen.generate(prompt_ids, 1)

    gen.reset()
    t0 = time.perf_counter()
    first = gen.generate(prompt_ids, 1)
    ttft = time.perf_counter() - t0

    gen.reset()
    t0 = time.perf_counter()
    full = gen.generate(prompt_ids, osl, stop_on_eos=False)
    e2el = time.perf_counter() - t0

    produced = len(full)
    tpot = (e2el - ttft) / max(produced - 1, 1)
    return {
        "isl": len(prompt_ids),
        "osl": osl,
        "tokens_produced": produced,
        "ttft_ms": ttft * 1000.0,
        "e2el_ms": e2el * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "itl_ms_derived_uniform": tpot * 1000.0,
        "tokens_per_second_per_user": (1.0 / tpot) if tpot > 0 else 0.0,
        "decode_tokens_per_second": (produced - 1) / max(e2el - ttft, 1e-9),
        "first_token_id": int(first[0]) if first else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osl", type=int, default=DEFAULT_OSL)
    parser.add_argument("--isl", type=int, nargs="*", default=list(DEFAULT_ISLS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    for isl in args.isl:
        if isl + args.osl > HF_ADVERTISED_CONTEXT:
            raise SystemExit(f"ISL {isl} + OSL {args.osl} exceeds the advertised context {HF_ADVERTISED_CONTEXT}")

    mesh = open_generator_mesh()
    rows: list[dict] = []
    try:
        logger.info("building generator (max_seq_len = full advertised context) ...")
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=HF_ADVERTISED_CONTEXT)

        # Warm traces at the smallest point so the first measured row is not paying capture.
        logger.info("warming traces ...")
        gen.generate(synthetic_prompt(gen.tokenizer, 128), 8)
        gen.reset()

        header = (
            f"{'ISL':>8} {'OSL':>5} | {'TTFT ms':>10} {'TPOT ms':>9} | "
            f"{'E2EL ms':>11} | {'t/s/u':>7} {'dec tok/s':>10}"
        )
        print("\n" + header, flush=True)
        print("-" * len(header), flush=True)

        for isl in args.isl:
            prompt = synthetic_prompt(gen.tokenizer, isl)
            row = measure(gen, prompt, args.osl)
            rows.append(row)
            print(
                f"{row['isl']:>8,} {row['osl']:>5} | {row['ttft_ms']:>10,.1f} {row['tpot_ms']:>9.2f} | "
                f"{row['e2el_ms']:>11,.1f} | {row['tokens_per_second_per_user']:>7.2f} "
                f"{row['decode_tokens_per_second']:>10.2f}",
                flush=True,
            )

        payload = {
            "batch_size": 1,
            "osl": args.osl,
            "hf_advertised_context": HF_ADVERTISED_CONTEXT,
            "itl_note": "itl_ms_derived_uniform == tpot_ms; per-step sampling would disable the free-running device loop",
            "rows": rows,
        }
        out = Path(args.out) if args.out else Path(__file__).with_name("latency_sweep_batch1.json")
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}", flush=True)
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
