# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""H2D producer for the M3 prefill stress test: the stock prefill_producer (connect, push, barrier, SHUTDOWN) with
the schedule picked by STRESS_PROFILE (see traffic.py):

  agentx | paced | cold   schedules from traffic.py
  custom                  the stock run_schedule (PREFILL_PRODUCER_* knobs); with STRESS_CUSTOM_DEPTH=lo,hi every
                          fresh request starts at a random 32-aligned depth in [lo, hi) over an unwritten prefix

Ends with one "[stress] END {json}" line that summarize.py reads.
"""

import json
import os
import random
import sys
import time

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traffic  # noqa: E402

import models.demos.common.prefill.runners.prefill_producer as pp  # noqa: E402

PROFILE = os.environ.get("STRESS_PROFILE", "paced")


def _log(msg):
    logger.info(f"[stress] {msg}")


def _end(st):
    _log("END " + json.dumps(st, default=float))


def stress_schedule(cfg, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    push_ms = []

    def push(*args):
        push_ms.append(push_fn(*args))

    common = dict(
        push=push, chunk=pp.CHUNK_SIZE, max_seq=pp.MAX_SEQ_LEN, num_users=cfg.num_users, seed=cfg.seed,
        knobs=traffic.Knobs.from_env(), now_fn=now_fn, sleep_fn=sleep_fn, log=_log,
    )  # fmt: skip
    if PROFILE == "cold":
        st, resident = traffic.cold_schedule(**common)
        completed = st["requests"]
    else:
        st, resident = traffic.agentx_schedule(duration_s=cfg.duration_s, paced=PROFILE == "paced", **common)
        completed = st["turns"]
    _end(dict(profile=PROFILE, **st))
    return pp.RunStats(
        resident={s: pp._SlotFill(real_len=e) for s, e in resident.items()},
        total_pushes=st["pushes"],
        push_ms=push_ms,
        completed=completed,
        wall_s=st["wall_s"],
    )


def _install_custom():
    depth = os.environ.get("STRESS_CUSTOM_DEPTH")
    stock_schedule = pp.run_schedule

    def custom_schedule(cfg, **kw):
        stats = stock_schedule(cfg, **kw)
        _end(dict(profile="custom", pushes=stats.total_pushes, requests=stats.completed, wall_s=stats.wall_s))
        return stats

    pp.run_schedule = custom_schedule
    if not depth:
        return
    lo, hi = (int(x) for x in depth.split(","))
    stock_new_request = pp._new_request

    def deep_new_request(slot, req_id, cfg, rng: random.Random, *, prefix_len=0, turn_idx=0):
        if prefix_len == 0 or prefix_len > hi:
            prefix_len, turn_idx = rng.randrange(lo // 32, hi // 32) * 32, 0
        stock_new_request(slot, req_id, cfg, rng, prefix_len=prefix_len, turn_idx=turn_idx)

    pp._new_request = deep_new_request


if __name__ == "__main__":
    if PROFILE == "custom":
        _install_custom()
    elif PROFILE in ("agentx", "paced", "cold"):
        pp.run_schedule = stress_schedule  # pp.main() looks it up as a module global
    else:
        sys.exit(f"unknown STRESS_PROFILE={PROFILE!r}")
    pp.main()
