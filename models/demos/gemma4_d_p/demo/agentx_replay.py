# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay an AgentX slice through the batching demo's scheduler on silicon (EXPERIMENT, not for upstream).

    AGENTX_SLICE=slice.json AGENTX_OUT=replay.json pytest models/demos/gemma4_d_p/demo/agentx_replay.py -s

The slice (models/demos/gemma4_d_p/agentx_sim/agentx_slice.py build) lists requests as (arrival s, input tokens, cached
prefix tokens). Every mode plays the same arrivals open loop: each request is submitted at its time and resumes at
its cached prefix rounded down to the mode's chunk width (KV below it is whatever the slot holds: timing only, no
accuracy). A warm fill first writes real KV into every slot to its full capacity, so attention never reads zeros.

Env: AGENTX_MODES (comma list of demo modes, default all four), DEMO_SLOTS (4), DEMO_CAPACITY (default: the slice's
largest padded request, rounded up to 8192), AGENTX_NO_FILL=1 (skip the warm fill).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger

from models.demos.gemma4_d_p.demo import batching_server as bs
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric


def _submit_at(scheduler, n_tokens, cached, label, mode):
    """Scheduler.submit, but the request starts at its cached prefix (aligned to the mode's width)."""
    width = bs.SERIAL_CHUNKS.get(mode, bs.CHUNK)
    n_tokens = -(-int(n_tokens) // bs.CHUNK) * bs.CHUNK
    rid = next(scheduler.ids)
    stream = scheduler.stream
    offset = (rid * 37_000) % (len(stream) - n_tokens)
    request = bs.Request(
        rid=rid,
        label=label,
        tokens=stream[offset : offset + n_tokens].clone(),
        arrived=time.perf_counter(),
        color=bs.COLORS[rid % len(bs.COLORS)],
    )
    request.done = (min(int(cached), n_tokens - 1) // width) * width
    scheduler.incoming.put(request)
    return request


def _play(scheduler, reqs, mode):
    t0, out = time.perf_counter(), []
    for i, r in enumerate(reqs):
        time.sleep(max(0.0, t0 + r["t"] - time.perf_counter()))
        out.append(_submit_at(scheduler, r["in"], r["cached"], f"#{i}", mode))
    for request in out:
        request.event.wait()
    assert not any(r.error for r in out), [r.error for r in out if r.error][:3]
    return t0, out


def _drive(scheduler, counters, reqs, modes, fill_tokens):
    results = {}
    try:
        if fill_tokens:
            scheduler.mode = bs.BATCHED
            logger.info(f"[agentx] warm fill: {scheduler.num_slots} x {fill_tokens} tokens")
            t = time.perf_counter()
            fill = [scheduler.submit(fill_tokens, f"fill{s}") for s in range(scheduler.num_slots)]
            for request in fill:
                request.event.wait()
            logger.info(f"[agentx] warm fill done in {time.perf_counter() - t:.1f} s")
        for mode in modes:
            scheduler.mode = mode
            busy0, steps0, replay0 = scheduler.busy_s, counters["steps"], counters["replay_ms"]
            t0, played = _play(scheduler, reqs, mode)
            res = {
                str(i): {"started": r.started - t0, "finished": r.finished - t0, "arrived": r.arrived - t0}
                for i, r in enumerate(played)
            }
            results[mode] = {
                "requests": res,
                "busy_s": scheduler.busy_s - busy0,
                "steps": counters["steps"] - steps0,
                "replay_s": (counters["replay_ms"] - replay0) / 1e3,
                "makespan_s": max(v["finished"] for v in res.values()),
            }
            m = results[mode]
            logger.info(
                f"[agentx] {mode}: {len(played)} requests, makespan {m['makespan_s']:.2f} s, busy {m['busy_s']:.2f} s "
                f"(replay {m['replay_s']:.2f} s), {m['steps']} steps"
            )
    finally:
        scheduler.stop.set()
    return results


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": bs.TRACE_REGION_SIZE})
def test_agentx_replay(mesh_device, reset_seeds, monkeypatch):
    sl = json.load(open(os.environ["AGENTX_SLICE"]))
    reqs = sl["requests"]
    modes = [m.strip() for m in os.environ.get("AGENTX_MODES", ",".join(bs.MODES)).split(",")]
    assert all(m in bs.MODES for m in modes), modes
    num_slots = int(os.environ.get("DEMO_SLOTS", "4"))
    biggest = max(-(-r["in"] // bs.CHUNK) * bs.CHUNK for r in reqs)
    capacity = int(os.environ.get("DEMO_CAPACITY", str(-(-biggest // bs.WIDE) * bs.WIDE)))
    scheduler, runner = bs.start_scheduler(mesh_device, monkeypatch, num_slots, capacity)
    if len(scheduler.stream) <= capacity:  # tile the text so every window fits
        reps = capacity // len(scheduler.stream) + 2
        scheduler.stream = scheduler.stream.repeat(reps)
    counters = {"steps": 0, "replay_ms": 0.0}
    step = runner.step

    def counted(lanes):
        ms = step(lanes)
        counters["steps"] += 1
        counters["replay_ms"] += ms
        return ms

    runner.step = counted
    fill = 0 if os.environ.get("AGENTX_NO_FILL") == "1" else capacity
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            driver = pool.submit(_drive, scheduler, counters, reqs, modes, fill)
            scheduler.run()
            results = driver.result()
    finally:
        runner.release()
    out = os.environ.get("AGENTX_OUT", "agentx_replay.json")
    json.dump(
        {"slice": os.environ["AGENTX_SLICE"], "slots": num_slots, "capacity": capacity, "modes": results},
        open(out, "w"),
        indent=1,
    )
    logger.info(f"[agentx] wrote {out}")
