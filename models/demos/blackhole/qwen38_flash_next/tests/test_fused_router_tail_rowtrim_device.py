# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_RTF_ROWTRIM: router_tail_sc (core A's division and typecast on 2 SFPU iterations per face) with the
flag off and on, same logits (tie-heavy rows included, the kinds of the sort-core test), scores bits and all ten indices on
every device, then the traced slope per call off vs on.  The program cache stays off for the bitwise part so each call
rebuilds the program with the flag it was given.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_router_tail_sc_device as sc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail_sc

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
PER_KIND = int(os.environ.get("RTF_ROWS", "40"))
RESULTS = os.environ.get("RTF_RESULTS", "/tmp/rtf_rowtrim_results.jsonl")
KINDS = (
    "normal2",
    "normal4",
    "levels",
    "top_ties",
    "few_ties",
    "pairs",
    "constant",
    "neg_levels",
    "neg_top_ties",
    "tiny",
)
PARAMS = {**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}


def run(flag, mesh, logits):
    os.environ[router_tail_sc.ROWTRIM_ENV] = "1" if flag else "0"
    out = sc._outputs(mesh)
    router_tail_sc.router_tail_sc_into(logits, *out)
    ttnn.synchronize_device(mesh)
    got = [
        (ttnn.to_torch(s).view(torch.int16).clone(), ttnn.to_torch(i).clone())
        for s, i in zip(*(ttnn.get_device_tensors(t) for t in out))
    ]
    for t in out:
        ttnn.deallocate(t)
    return got


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
def test_rowtrim_bitwise_and_slope(mesh_device):
    mesh = mesh_device
    generator = torch.Generator().manual_seed(912)
    router_tail_sc.prepare(mesh)
    total = 0
    try:
        for kind in KINDS:
            for n, values in enumerate(sc._rows(kind, 12 if kind == "constant" else PER_KIND, generator)):
                logits = sc._upload(mesh, values)
                off, on = run(False, mesh, logits), run(True, mesh, logits)
                for d, (a, b) in enumerate(zip(off, on)):
                    assert torch.equal(a[0], b[0]), f"{kind} row {n} dev {d}: scores differ"
                    assert torch.equal(a[1], b[1]), f"{kind} row {n} dev {d}: indices differ"
                total += 1
                ttnn.deallocate(logits)
        logger.info(f"flag off/on bit-identical on {total} rows x 8 devices")
        logits = sc._upload(mesh, sc._rows("normal2", 1, generator)[0])
        times = {"off": [], "on": []}
        mesh.enable_program_cache()
        for rep in range(3):
            for name, flag in (("off", False), ("on", True)):
                os.environ[router_tail_sc.ROWTRIM_ENV] = "1" if flag else "0"
                mesh.clear_program_cache()
                out = sc._outputs(mesh)
                router_tail_sc.router_tail_sc_into(logits, *out)  # compile and cache before the trace capture
                ttnn.synchronize_device(mesh)
                times[name].append(sc._traced_us(mesh, lambda: router_tail_sc.router_tail_sc_into(logits, *out)))
                logger.info(f"rep {rep} {name}: slope/call {times[name][-1][0]:.2f} us")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(rows_checked=total, us=times)) + "\n")
    finally:
        os.environ.pop(router_tail_sc.ROWTRIM_ENV, None)
