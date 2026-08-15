# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``prefill_trace_max_entries > 1`` -- the configuration this stage recommends to serving.

Round 16 of the stage review pointed out that every test, probe and evidence run had used the
default of **1**, including the partial-failure fix in ``_release_prefill_traces`` whose
docstring singles out ``> 1`` as its reason.  A recommendation nothing has ever run is not a
recommendation, so this runs it: two buckets resident at once, each replayed after the other,
against an eager arm on the same host and the same build.

Arm B (eager) establishes the reference; arm A (``prefill_trace=True``,
``prefill_trace_max_entries=2``) must match it token for token, with both buckets resident.

Usage::

    python doc/optimized_full_model/bench/prefill_trace_multibucket_probe.py
"""
import pathlib
import sys

REPO = pathlib.Path("/home/ttuser/dev/muse-glimmer/tt-metal")
sys.path.insert(0, str(REPO))
MODEL = REPO / "models/autoports/meta_models_muse_glimmer_30b"

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

REDUCED_LAYERS = [0, 3]
REDUCED_MAX_SEQ = 4096


def prompt(n, seed):
    import random

    rng = random.Random(seed)
    return [rng.randrange(1, 200000) for _ in range(n)]


def run(gen, p128, p256):
    gen.reset()
    a = gen.generate(prompt_token_ids=p128, max_new_tokens=3, enable_trace=True)
    gen.reset()
    b = gen.generate(prompt_token_ids=p256, max_new_tokens=3, enable_trace=True)
    gen.reset()
    c = gen.generate(prompt_token_ids=p128, max_new_tokens=3, enable_trace=True)
    return a, b, c


def main():
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    p128, p256 = prompt(128, 41), prompt(256, 41)
    try:
        eager = build_generator(MODEL, mesh, max_seq_len=REDUCED_MAX_SEQ, layer_indices=REDUCED_LAYERS, reuse=False)
        try:
            e1, e2, e3 = run(eager, p128, p256)
            print(f"B eager      : first={e1} other={e2} repeat={e3}")
            print(f"B repeat matches first: {e1 == e3}")
        finally:
            eager.teardown()
            eager.model.deallocate()
            clear_generator_cache()

        traced = build_generator(
            MODEL,
            mesh,
            max_seq_len=REDUCED_MAX_SEQ,
            layer_indices=REDUCED_LAYERS,
            reuse=False,
            prefill_trace=True,
            prefill_trace_max_entries=2,
        )
        try:
            t1, t2, t3 = run(traced, p128, p256)
            print(f"A two-bucket : first={t1} other={t2} repeat={t3}")
            print(f"A buckets    : {sorted(traced._prefill_traces)}")
            print(f"A repeat matches first: {t1 == t3}")
            print(f"A matches eager: first={t1 == e1} other={t2 == e2} repeat={t3 == e3}")
        finally:
            traced.teardown()
            traced.model.deallocate()
            clear_generator_cache()
    finally:
        close_multichip_mesh(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
