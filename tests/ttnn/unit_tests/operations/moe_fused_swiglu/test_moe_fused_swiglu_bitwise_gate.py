# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""CROSS-REVISION bitwise gate, for any change that claims to be SEMANTICALLY EMPTY.

Written for the rewrite — deleting dead ablations, restructuring the host, extracting the shared
transport — where each step removed code the shipped configuration never executes or moved code
that it does. It is kept because that claim recurs: the honest gate for it is not PCC and not
timing, it is that the op returns the SAME BITS before and after.

The digest is NOT checked in. It is recorded from whatever revision you are comparing against, so
a stale one cannot quietly become the baseline.

Timing cannot serve as the gate: this op's per-cell run-to-run band is ~1 %, and session drift
is the same size, so a 1 % regression and a clean run are indistinguishable in one A/B. Bits
have no band.

`test_moe_fused_swiglu_determinism.py` proves the op agrees with ITSELF within one process.
This proves it agrees with a PREVIOUS REVISION across processes, which needs the digest to
outlive the process:

    # on the base revision
    MOE_BITWISE_MODE=record scripts/run_safe_pytest.sh --run-all <this file>
    # after the change
    MOE_BITWISE_MODE=check  scripts/run_safe_pytest.sh --run-all <this file>

`check` is the default, and it FAILS if the digest file is missing rather than silently
recording one — a gate that quietly re-baselines itself is not a gate.

WHAT IS HASHED. Only rows `[0, ceil(count/32)*32)`. Rows past that are UNDEFINED by the op's
contract (never written), so they hold whatever the freshly allocated DRAM buffer contained and
differ run to run for reasons that have nothing to do with the code. Same slice the determinism
test compares, for the same reason.

Inputs come from the determinism harness's `_build`, so the bits certified here are the bits at
the call site perf is measured at — ND-sharded weights, seed 42, the 100.0 phantom-row sentinel.
"""

import hashlib
import json
import os
import pathlib

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

from .test_moe_fused_swiglu_determinism import (
    LOCAL_EXPERT_ID,
    _build,
    defined_region,
    written_rows,
)

MODE = os.environ.get("MOE_BITWISE_MODE", "check")

DIGEST_PATH = pathlib.Path(os.environ.get("MOE_BITWISE_FILE", "generated/moe_fused_swiglu_bitwise_digest.json"))

#: One representative per distinct kernel path x activation layout x M regime, plus the two
#: shapes whose WRITTEN extent differs from `count` (the non-tile-aligned seam, count == capacity)
#: and a zero-count dispatch. Deliberately the same spread as the determinism guard set.
CASES = [
    (7168, 5120, 128, "bf16_rm"),
    (7168, 5120, 256, "bf16_rm"),
    (7168, 5120, 512, "bf16_rm"),
    (7168, 5120, 128, "bfp8_tile"),
    (7168, 5120, 256, "bfp8_tile"),
    (7168, 5120, 512, "bfp8_tile"),
    (7168, 1024, 255, "bf16_rm"),  # ragged: count not tile-aligned
    (7168, 1024, 257, "bfp8_tile"),  # ragged the other side of the tile
    (6144, 5120, 256, "bf16_rm"),  # the other emb
    (6144, 1024, 1024, "bfp8_tile"),  # count == capacity
    (7168, 1024, 0, "bf16_rm"),  # zero-count: no rows written at all
    # PADDED TAILS — m_t not a power of two, so `m_tiles_eff` rounds the tail block UP and the
    # gate/up matmul runs over FEWER rows than the block's page count (`m_tiles_real`). Every case
    # above happens to have m_t == m_eff, so none of them can tell that path from the old one: 128
    # -> 4, 256 -> 8, 255 -> 8, 257 -> 8+1, 512/1024 -> whole blocks. Without these the gate would
    # certify the change by not exercising it.
    (7168, 1024, 96, "bf16_rm"),  # m_t 3 -> m_eff 4
    (7168, 1024, 160, "bf16_rm"),  # m_t 5 -> m_eff 8, the widest single-block gap
    (7168, 1024, 160, "bfp8_tile"),  # same gap on the tiled path
    (7168, 1024, 224, "bf16_rm"),  # m_t 7 -> m_eff 8
    (7168, 1024, 150, "bf16_rm"),  # ragged AND padded: m_t 5, written extent 160
    (7168, 1024, 352, "bf16_rm"),  # two blocks, tail m_t 3 -> m_eff 4
    (6144, 1024, 480, "bfp8_tile"),  # two blocks, tail m_t 7 -> m_eff 8, other emb
]


def case_id(emb, capacity, count, fmt):
    return f"{emb},{capacity},{count},{fmt}"


def output_digest(out, count):
    """SHA-256 over the raw bytes of the DEFINED region, plus its shape and dtype.

    `to_torch` dequantizes bfp8 to float32 by a fixed rule, so identical device bits give
    identical host bytes; the shape and dtype go into the hash so a silently changed output
    format cannot collide with the old digest.
    """
    rows = written_rows(count)
    if rows == 0:
        # Nothing is written. Pin that the op still produced a correctly shaped buffer.
        return f"empty:{tuple(out.shape)}:{out.dtype}"
    host = ttnn.to_torch(defined_region(out, rows)).contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(host.shape)}|{host.dtype}|".encode())
    h.update(host.numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("emb, capacity, count, input_format", CASES)
def test_bitwise(device, emb, capacity, count, input_format, request):
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    assert list(out.shape) == [1, 1, capacity, emb]

    key = case_id(emb, capacity, count, input_format)
    digest = output_digest(out, count)
    print(f"[bitwise] {key} -> {digest}")

    store = json.loads(DIGEST_PATH.read_text()) if DIGEST_PATH.exists() else {}

    if MODE == "record":
        store[key] = digest
        DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_PATH.write_text(json.dumps(store, indent=2, sort_keys=True))
        return

    assert DIGEST_PATH.exists(), (
        f"no digest at {DIGEST_PATH} — run with MOE_BITWISE_MODE=record on the base revision "
        f"first. Refusing to silently re-baseline."
    )
    assert key in store, f"{key} not in the recorded digest ({sorted(store)}); re-record the base revision"
    assert store[key] == digest, (
        f"BITWISE REGRESSION on {key}\n  recorded: {store[key]}\n  now:      {digest}\n"
        f"The change was supposed to be semantically empty and is not. If the change was MEANT to\n"
        f"move the numbers, re-record rather than widening this gate."
    )
