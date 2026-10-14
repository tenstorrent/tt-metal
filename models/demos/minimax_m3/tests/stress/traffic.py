# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Traffic schedules for the M3 prefill stress test. Pure Python (no ttnn), so they can be previewed on a host:

    python3 traffic.py --profile paced --sim-s 7200        # simulated clock, prints the END stats

Every schedule calls ``push(slot, chunk_idx, start, end, isl)`` once per chunk, in the stock prefill_producer
push_fn signature, and returns ``(stats, resident)`` (resident = slot -> last written end).

agentx: a pool of multi-turn sessions, one KV slot each. A session starts deep (lognormal STRESS_DEPTH over an
  unwritten KV prefix: the depth is reached without prefilling it) or, with STRESS_FRESH_PROB, at 0. Each turn
  appends lognormal STRESS_TURN new tokens from the 32-floored previous end, so most turns are 1-2 ragged chunks
  that start mid-slab. Saturated: ready turns are pushed as fast as the H2D service takes them; sessions "decode"
  for STRESS_THINK between turns; active sessions are redrawn in STRESS_CONC every STRESS_CONC_PERIOD_S.
paced: the same sessions at the production duty cycle. Concurrency walks STRESS_CONC_PHASES (STRESS_PHASE_S each,
  shuffled per round). Bursts of whole turns (1 + Exp(conc / STRESS_BURST_DIV)), 1..STRESS_INFLIGHT_MAX turns
  interleaved, separated by lognormal STRESS_GAP idle gaps; with STRESS_LONG_IDLE_PROB a gap is a long idle in
  STRESS_LONG_IDLE_S, followed by one tiny (1-64 token) request, a 1-5 s pause, then the burst.
cold: right after runner start, STRESS_COLD=tiny,normal requests (tiny = 1-64 tokens, normal = 1-4 chunks with a
  ragged end) at 0 or a deep 32-aligned offset, STRESS_COLD_GAP_MS apart. The first request is always tiny.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field

KV_ALIGN = 32  # resume offsets are floored to the KV page


def _floats(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",")]


@dataclass
class Knobs:
    conc: tuple = (56, 90)
    conc_period_s: float = 600
    conc_phases: list = field(default_factory=lambda: [2, 8, 30, 90])
    phase_s: float = 900
    depth: tuple = (300000, 0.55, 32768, 746000)  # median, sigma, min, max
    fresh_prob: float = 0.1
    fresh_med: float = 12000
    turn: tuple = (1600, 1.0, 30000)  # median, sigma, max
    think: tuple = (2, 60)  # median s, max s
    cont_prob: float = 0.95
    ctx_cap: int = 760000
    gap: tuple = (3, 1.0, 120)  # median s, sigma, max s
    long_idle_prob: float = 0.006
    long_idle_s: tuple = (120, 600)
    burst_div: float = 4
    inflight_max: int = 4
    stats_every_s: float = 60
    cold: tuple = (6, 3)  # tiny, normal
    cold_gap_ms: tuple = (0, 1500)

    @classmethod
    def from_env(cls):
        k = cls()
        e = os.environ.get
        k.conc = tuple(int(x) for x in _floats("STRESS_CONC", "56,90"))
        k.conc_period_s = float(e("STRESS_CONC_PERIOD_S", k.conc_period_s))
        k.conc_phases = [int(x) for x in _floats("STRESS_CONC_PHASES", "2,8,30,90")]
        k.phase_s = float(e("STRESS_PHASE_S", k.phase_s))
        k.depth = tuple(_floats("STRESS_DEPTH", "300000,0.55,32768,746000"))
        k.fresh_prob = float(e("STRESS_FRESH_PROB", k.fresh_prob))
        k.fresh_med = float(e("STRESS_FRESH_MED", k.fresh_med))
        k.turn = tuple(_floats("STRESS_TURN", "1600,1.0,30000"))
        k.think = tuple(_floats("STRESS_THINK", "2,60"))
        k.cont_prob = float(e("STRESS_CONT_PROB", k.cont_prob))
        k.ctx_cap = int(float(e("STRESS_CTX_CAP", k.ctx_cap)))
        k.gap = tuple(_floats("STRESS_GAP", "3,1.0,120"))
        k.long_idle_prob = float(e("STRESS_LONG_IDLE_PROB", k.long_idle_prob))
        k.long_idle_s = tuple(_floats("STRESS_LONG_IDLE_S", "120,600"))
        k.burst_div = float(e("STRESS_BURST_DIV", k.burst_div))
        k.inflight_max = int(e("STRESS_INFLIGHT_MAX", k.inflight_max))
        k.stats_every_s = float(e("STRESS_STATS_EVERY_S", k.stats_every_s))
        k.cold = tuple(int(x) for x in _floats("STRESS_COLD", "6,3"))
        k.cold_gap_ms = tuple(_floats("STRESS_COLD_GAP_MS", "0,1500"))
        return k


def _lognorm(rng, med, sigma, lo, hi):
    return int(min(hi, max(lo, rng.lognormvariate(math.log(med), sigma))))


class _Session:
    __slots__ = ("slot", "pos", "turn", "end", "next_start", "ready_at")

    def __init__(self, slot, pos):
        self.slot, self.pos, self.turn = slot, pos, 0
        self.end = self.next_start = 0
        self.ready_at = 0.0


def agentx_schedule(*, push, chunk, max_seq, num_users, duration_s, seed, paced, knobs, now_fn, sleep_fn, log):
    k = knobs
    rng = random.Random(seed)
    turn_med, turn_sigma, turn_max = k.turn[0], k.turn[1], int(k.turn[2])
    turn_max = min(turn_max, max_seq // 4)
    cap = min(k.ctx_cap, max_seq - turn_max - chunk)
    assert cap > 0, f"max_seq {max_seq} too small for turns up to {turn_max} tokens"
    d_med, d_sigma = k.depth[0], k.depth[1]
    d_max = min(int(k.depth[3]), cap)
    d_min = min(int(k.depth[2]), d_max)
    free = list(range(num_users))
    rng.shuffle(free)
    active, resident = [], {}
    st = dict(
        pushes=0, turns=0, sessions=0, sessions_done=0, mid_start=0, ragged_end=0, fresh=0,
        gaps=0, long_idles=0, idle_s=0.0, tiny=0,
    )  # fmt: skip
    depths = []
    burst_left, tiny_pending, inflight = 0, False, []
    phase_order = []
    start = now_fn()
    conc, conc_at = rng.randint(*k.conc), start
    if paced:
        conc_at = start - k.phase_s - 1  # pick the first phase right away
    last_stats = start

    def new_turn(s, now):
        if s.turn == 0 and s.pos == 0:
            n = _lognorm(rng, k.fresh_med, 0.8, 1, turn_max)
        else:
            n = _lognorm(rng, turn_med, turn_sigma, 1, turn_max)
        s.end, s.next_start, s.ready_at = s.pos + n, s.pos, now
        st["turns"] += 1

    def start_session(now):
        slot = free.pop()
        if rng.random() < k.fresh_prob:
            pos = 0
            st["fresh"] += 1
        else:
            pos = _lognorm(rng, d_med, d_sigma, d_min, d_max) // KV_ALIGN * KV_ALIGN
        depths.append(pos)
        s = _Session(slot, pos)
        st["sessions"] += 1
        new_turn(s, now)
        active.append(s)

    while now_fn() - start < duration_s:
        now = now_fn()
        if paced and now - conc_at > k.phase_s:
            if not phase_order:
                phase_order = k.conc_phases[:]
                rng.shuffle(phase_order)
            conc, conc_at = phase_order.pop(), now
            log(f"PHASE conc={conc} for {k.phase_s:.0f}s")
        if paced and burst_left <= 0 and not inflight:
            long_idle = rng.random() < k.long_idle_prob
            if long_idle:
                idle = rng.uniform(*k.long_idle_s)
            else:
                idle = min(k.gap[2], max(0.2, rng.lognormvariate(math.log(k.gap[0]), k.gap[1])))
            idle = max(0.0, min(idle, duration_s - (now - start)))
            st["gaps"] += 1
            st["long_idles"] += long_idle
            st["idle_s"] += idle
            if long_idle or idle > 30:
                log(f"IDLE {idle:.0f}s{' (long; tiny request next)' if long_idle else ''}")
            sleep_fn(idle)
            burst_left = 1 + int(rng.expovariate(1.0 / max(1.0, conc / k.burst_div)))
            tiny_pending = long_idle
            continue
        if not paced and now - conc_at > k.conc_period_s:
            conc, conc_at = rng.randint(*k.conc), now
        while len(active) < min(conc, num_users) and free:
            start_session(now)
        ready = [s for s in active if s.ready_at <= now]
        if not ready:
            sleep_fn(max(0.01, min(s.ready_at for s in active) - now))
            continue
        if paced:
            inflight = [x for x in inflight if x in active]
            if tiny_pending and not inflight:
                s = rng.choice([x for x in active if x.next_start == x.pos] or ready)
                s.end = s.next_start + rng.randint(1, 64)
                log(f"TINY slot={s.slot} [{s.next_start},{s.end}) after long idle")
                st["tiny"] += 1
                inflight = [s]
            elif not tiny_pending:
                want = min(rng.randint(1, k.inflight_max), burst_left) if not inflight else len(inflight)
                pool = [x for x in active if x not in inflight]
                while len(inflight) < want and pool:
                    inflight.append(pool.pop(rng.randrange(len(pool))))
            s = rng.choice(inflight)
        else:
            s = rng.choice(ready)
        a = s.next_start
        e = min(a + chunk, s.end)
        st["mid_start"] += a % chunk != 0
        st["ragged_end"] += e - a < chunk
        push(s.slot, (a - s.pos) // chunk, a, e, s.end)
        st["pushes"] += 1
        resident[s.slot] = e
        s.next_start = e
        if e >= s.end:  # turn done
            s.pos = s.end // KV_ALIGN * KV_ALIGN
            s.turn += 1
            if paced:
                inflight.remove(s)
                if tiny_pending:
                    tiny_pending = False
                    sleep_fn(rng.uniform(1.0, 5.0))
                else:
                    burst_left -= 1
                    if burst_left > len(inflight) and rng.random() < 0.5:
                        pool = [x for x in active if x not in inflight and x is not s]
                        if pool:
                            inflight.append(rng.choice(pool))
            if s.pos < cap and rng.random() < k.cont_prob and not (paced and len(active) > conc):
                new_turn(s, now_fn())
                if not paced:
                    s.ready_at += min(k.think[1], rng.lognormvariate(math.log(k.think[0]), 0.8))
            else:
                active.remove(s)
                free.insert(rng.randrange(len(free) + 1), s.slot)
                st["sessions_done"] += 1
        if now - last_stats > k.stats_every_s:
            last_stats = now
            pos = sorted(x.pos for x in active)
            log(
                f"t={now - start:.0f}s conc={conc} active={len(active)} ready={len(ready)} inflight={len(inflight)} "
                + " ".join(f"{key}={v:.0f}" if isinstance(v, float) else f"{key}={v}" for key, v in st.items())
                + f" ctx p10/50/90/max={pos[len(pos) // 10]}/{pos[len(pos) // 2]}/{pos[9 * len(pos) // 10]}/{pos[-1]}"
            )
    depths.sort()
    st["start_depth_p50"] = depths[len(depths) // 2] if depths else 0
    st["wall_s"] = now_fn() - start
    return st, resident


def cold_schedule(*, push, chunk, max_seq, num_users, seed, knobs, now_fn, sleep_fn, log):
    rng = random.Random(seed)
    deep_hi = max(1, (max_seq - 5 * chunk) // KV_ALIGN)
    kinds = ["tiny"] * knobs.cold[0] + ["normal"] * knobs.cold[1]
    rng.shuffle(kinds)
    kinds = ["tiny"] + kinds
    slots = rng.sample(range(num_users), min(num_users, len(kinds)))
    resident, st = {}, dict(pushes=0, requests=len(kinds), tiny=kinds.count("tiny"), mid_start=0, ragged_end=0)
    start = now_fn()
    for i, kind in enumerate(kinds):
        slot = slots[i % len(slots)]
        a = 0 if rng.random() < 0.5 else rng.randrange(deep_hi) * KV_ALIGN
        real = rng.randint(1, 64) if kind == "tiny" else rng.randint(1, 4) * chunk - rng.randint(0, chunk - 1)
        end = a + real
        log(f"request {i} {kind} slot={slot} [{a},{end})")
        s = a
        while s < end:
            e = min(s + chunk, end)
            push(slot, (s - a) // chunk, s, e, end)
            st["pushes"] += 1
            st["mid_start"] += s % chunk != 0
            st["ragged_end"] += e - s < chunk
            resident[slot] = e
            s = e
        sleep_fn(rng.uniform(*knobs.cold_gap_ms) / 1000.0)
    st["wall_s"] = now_fn() - start
    return st, resident


def _simulate(argv=None):
    ap = argparse.ArgumentParser(description="Preview a schedule on a simulated clock (no device).")
    ap.add_argument("--profile", choices=["agentx", "paced", "cold"], default="paced")
    ap.add_argument("--sim-s", type=float, default=3600)
    ap.add_argument("--chunk-s", type=float, default=0.025, help="simulated device time per chunk")
    ap.add_argument("--users", type=int, default=90)
    ap.add_argument("--max-seq-len", type=int, default=1048576)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=59542)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    clock = [0.0]

    def push(*_):
        clock[0] += a.chunk_s

    def sleep(s):
        clock[0] += s

    log = (lambda m: None) if a.quiet else (lambda m: print(f"[{clock[0]:8.0f}s] {m}"))
    common = dict(
        push=push, chunk=a.chunk_size, max_seq=a.max_seq_len, num_users=a.users, seed=a.seed,
        knobs=Knobs.from_env(), now_fn=lambda: clock[0], sleep_fn=sleep, log=log,
    )  # fmt: skip
    if a.profile == "cold":
        st, _ = cold_schedule(**common)
    else:
        st, _ = agentx_schedule(duration_s=a.sim_s, paced=a.profile == "paced", **common)
    print("END " + json.dumps(st))


if __name__ == "__main__":
    _simulate()
