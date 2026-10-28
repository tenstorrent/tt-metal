# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replay an AgentX slice through batching_scheduler.py on silicon, to check the AgentX simulator's predictions.

    AGENTX_SLICE=slice.json AGENTX_OUT=replay.json pytest models/demos/gemma4_d_p/demo/agentx_replay.py -s

The slice lists requests as (arrival s, input tokens, cached prefix tokens). Every mode plays the same arrivals open
loop; each request resumes at its cached prefix rounded down to the mode's chunk width, or with AGENTX_ALIGNS also
"resume", to the model's resume granularity (results keyed "<mode> (resume)"). This measures timing only: the
KV below that point is whatever the slot holds. A warm fill writes real KV into every slot first, so attention never
reads zeros.

Env: AGENTX_MODES (comma list of demo modes, default all four), AGENTX_ALIGNS (chunk, resume or both; default chunk),
DEMO_SLOTS (4), DEMO_CAPACITY (default: the slice's
largest request, rounded up to 8192), AGENTX_NO_FILL=1 (skip the warm fill).

SP4 x 2 study: GEMMA4_TEST_SP4=1 AGENTX_HALF=k replays every other request from the k-th (0 or 1) on one CP4 x TP4
half, as one of the two prefill servers of a split Galaxy would see them (agentx_slice.py predict --split k/2).
"""

import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger

from models.demos.gemma4_d_p.demo import batching_scheduler as bs
from models.demos.gemma4_d_p.demo.batched_steps import TRACE_REGION_SIZE
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric, sp4_half


def _drive(scheduler, reqs, modes, fill_tokens, aligns=("chunk",)):
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
        for align, mode in itertools.product(aligns, modes):
            scheduler.mode, scheduler.align = mode, align
            key = mode if align == "chunk" else f"{mode} (resume)"
            busy0, steps0, replay0 = scheduler.busy_s, scheduler.steps, scheduler.replay_ms
            t0, played = scheduler.play([(r["t"], r["in"], f"#{i}", r["cached"]) for i, r in enumerate(reqs)])
            res = {
                str(i): {"started": r.started - t0, "finished": r.finished - t0, "arrived": r.arrived - t0}
                for i, r in enumerate(played)
            }
            results[key] = {
                "requests": res,
                "busy_s": scheduler.busy_s - busy0,
                "steps": scheduler.steps - steps0,
                "replay_s": (scheduler.replay_ms - replay0) / 1e3,
                "makespan_s": max(v["finished"] for v in res.values()),
            }
            m = results[key]
            logger.info(
                f"[agentx] {key}: {len(played)} requests, makespan {m['makespan_s']:.2f} s, busy {m['busy_s']:.2f} s "
                f"(replay {m['replay_s']:.2f} s), {m['steps']} steps"
            )
    finally:
        scheduler.stop.set()
    return results


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_agentx_replay(mesh_device, reset_seeds):
    if "AGENTX_SLICE" not in os.environ:
        pytest.skip("set AGENTX_SLICE to a slice from the AgentX simulator's agentx_slice.py build")
    with open(os.environ["AGENTX_SLICE"]) as f:
        sl = json.load(f)
    reqs = sl["requests"]
    mesh_device = sp4_half(mesh_device)
    if (half := os.environ.get("AGENTX_HALF")) is not None:
        reqs = reqs[int(half) :: 2]
    modes = [m.strip() for m in os.environ.get("AGENTX_MODES", ",".join(bs.MODES)).split(",")]
    assert all(m in bs.MODES for m in modes), modes
    aligns = [a.strip() for a in os.environ.get("AGENTX_ALIGNS", "chunk").split(",")]
    assert all(a in ("chunk", "resume") for a in aligns), aligns
    num_slots = int(os.environ.get("DEMO_SLOTS", "4"))
    biggest = max(r["in"] for r in reqs)
    capacity = int(os.environ.get("DEMO_CAPACITY", str(-(-biggest // bs.WIDE) * bs.WIDE)))
    scheduler, runner = bs.start_scheduler(mesh_device, num_slots, capacity, resume="resume" in aligns)
    if len(scheduler.stream) <= capacity:  # tile the text so every window fits
        reps = capacity // len(scheduler.stream) + 2
        scheduler.stream = scheduler.stream.repeat(reps)
    fill = 0 if os.environ.get("AGENTX_NO_FILL") == "1" else capacity
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            driver = pool.submit(_drive, scheduler, reqs, modes, fill, aligns)
            scheduler.run()
            results = driver.result()
    finally:
        runner.release()
    out = os.environ.get("AGENTX_OUT", "agentx_replay.json")
    with open(out, "w") as f:
        json.dump(
            {"slice": os.environ["AGENTX_SLICE"], "slots": num_slots, "capacity": capacity, "modes": results},
            f,
            indent=1,
        )
    logger.info(f"[agentx] wrote {out}")
