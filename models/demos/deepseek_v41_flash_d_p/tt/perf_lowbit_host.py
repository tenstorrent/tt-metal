# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.12 (LOWBIT_HOST): host time a call of FIX.14's low-bit writes on the run's mesh, split into its parts. Not a gate:
numbers for BREADCRUMBS.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_lowbit_host.py -s

Each line: the median host time of N warm calls (dispatch is asynchronous; the device is synchronised before and after
each timed loop), for
  - ckv_bfp4 window blocks as tt/attention._put_blocks calls them (chunk 5120 at row0 128, carry 128 at row0 0) and the
    one-call form (carry + chunk, ``prefix``);
  - bfp_round as tt/indexer.py calls it (index query [1, 64, 160, 128], keys [1, 1, 1760, 128]: a new output each
    call) and with a reused output (``out``);
  - the parts: the registered op vs the plain function (ttnn's operation wrapper), allocate_tensor_on_device +
    deallocate, run_prepared alone.
"""

from __future__ import annotations

import importlib
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

ck = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")  # (the package exports the function)
WARM, N = 5, 40
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _time(mesh, fn):
    for _ in range(WARM):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.synchronize_device(mesh)
    return statistics.median(ts)


@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_lowbit_host(mesh_device):
    mesh = mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)
    g = torch.Generator().manual_seed(0)

    def up(shape):
        return ttnn.from_torch(
            torch.randn(*shape, generator=g).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
        )

    chunk, carry = up([1, 1, 5120, 512]), up([1, 1, 128, 512])
    blocks = ttnn.zeros(
        [1, 1, (128 + 5120) // 32, ck.BLOCK_BYTES],
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
    )
    res = {}

    def put(t, row0, f=ttnn.bringup.ckv_bfp4, **kw):
        return lambda: f(t, blocks, row0, dtype=ttnn.bfloat8_b, blocks=True, memory_config=DRAM, prepared=True, **kw)

    res["ckv_bfp4 chunk 5120 (op)"] = _time(mesh, put(chunk, 128))
    res["ckv_bfp4 carry 128 (op)"] = _time(mesh, put(carry, 0))
    res["ckv_bfp4 chunk 5120 (plain function)"] = _time(mesh, put(chunk, 128, ck.ckv_bfp4))
    res["ckv_bfp4 carry 128 (plain function)"] = _time(mesh, put(carry, 0, ck.ckv_bfp4))
    res["ckv_bfp4 chunk 5120 compact (op)"] = _time(mesh, put(chunk, 128, compact=True))
    res["ckv_bfp4 carry 128 compact (op)"] = _time(mesh, put(carry, 0, compact=True))
    res["ckv_bfp4 carry+chunk one call (op)"] = _time(mesh, put(chunk, 0, prefix=carry))
    res["ckv_bfp4 carry+chunk one call compact (op)"] = _time(mesh, put(chunk, 0, prefix=carry, compact=True))

    q, k = up([1, 64, 160, 128]), up([1, 1, 1760, 128])
    for name, x in (("q [1,64,160,128]", q), ("k [1,1,1760,128]", k)):

        def new(f=ttnn.bringup.bfp_round, x=x):
            ttnn.deallocate(f(x, ttnn.bfloat4_b, memory_config=DRAM, prepared=True))

        res[f"bfp_round {name} new out (op)"] = _time(mesh, new)
        res[f"bfp_round {name} new out (plain function)"] = _time(mesh, lambda x=x: new(ck.bfp_round, x))
        res[f"bfp_round {name} new out compact (op)"] = _time(
            mesh,
            lambda x=x: ttnn.deallocate(
                ttnn.bringup.bfp_round(x, ttnn.bfloat4_b, memory_config=DRAM, prepared=True, compact=True)
            ),
        )
        out = ttnn.bringup.bfp_round(x, ttnn.bfloat4_b, memory_config=DRAM)
        res[f"bfp_round {name} reused out (op)"] = _time(
            mesh, lambda x=x, out=out: ttnn.bringup.bfp_round(x, out=out, prepared=True)
        )
        res[f"bfp_round {name} reused out compact (op)"] = _time(
            mesh, lambda x=x, out=out: ttnn.bringup.bfp_round(x, out=out, prepared=True, compact=True)
        )
        res[f"bfp_round {name} reused out (plain function)"] = _time(
            mesh, lambda x=x, out=out: ck.bfp_round(x, out=out, prepared=True)
        )

        def alloc(x=x):
            ttnn.deallocate(
                ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, mesh, DRAM)
            )

        res[f"allocate+deallocate {name}"] = _time(mesh, alloc)
        pp = ck._prepared_round(x, out, 0, [x, out])
        assert pp is not None, "bfp_round did not prepare"
        res[f"run_prepared alone {name}"] = _time(
            mesh, lambda x=x, out=out, pp=pp: ttnn.bringup.run_prepared(pp, [x, out], [0])
        )
        res[f"_prepared_round lookup {name}"] = _time(
            mesh, lambda x=x, out=out: ck._prepared_round(x, out, 0, [x, out])
        )
    pp = ck._prepared(chunk, blocks, None, 128, [chunk, blocks], ttnn.bfloat8_b, True)
    assert pp is not None
    res["run_prepared alone ckv_bfp4 chunk"] = _time(
        mesh, lambda: ttnn.bringup.run_prepared(pp, [chunk, blocks], [128])
    )
    res["_prepared lookup ckv_bfp4 chunk"] = _time(
        mesh, lambda: ck._prepared(chunk, blocks, None, 128, [chunk, blocks], ttnn.bfloat8_b, True)
    )

    # bytes: two default calls vs one compact call with the carry as prefix; bfp_round default vs compact
    def host(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).view(torch.uint8)

    ttnn.bringup.ckv_bfp4(carry, blocks, 0, dtype=ttnn.bfloat8_b, blocks=True, prepared=True)
    ttnn.bringup.ckv_bfp4(chunk, blocks, 128, dtype=ttnn.bfloat8_b, blocks=True, prepared=True)
    ref = host(blocks)
    # the defined bytes of a page (the gaps between its sections are not written by the packer: stale L1, K / reference)
    mask = (
        importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.tests.reference")
        .encode_blocks(torch.zeros(32, 512))[1]
        .reshape(1, -1)
    )
    variants = {
        "prefix": [dict(t=chunk, row0=0, prefix=carry)],
        "compact": [dict(t=carry, row0=0, compact=True), dict(t=chunk, row0=128, compact=True)],
        "prefix+compact": [dict(t=chunk, row0=0, prefix=carry, compact=True)],
    }
    bad = []
    for name, calls in variants.items():
        b2 = ttnn.zeros(
            list(blocks.shape), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM
        )
        for c in calls:
            c = dict(c)
            ttnn.bringup.ckv_bfp4(c.pop("t"), b2, c.pop("row0"), dtype=ttnn.bfloat8_b, blocks=True, prepared=True, **c)
        h = host(b2).reshape(-1, ck.BLOCK_BYTES)
        diff = ((h != ref.reshape(-1, ck.BLOCK_BYTES)) & mask).any(-1).nonzero().flatten().tolist()
        print(f"LOWBIT_HOST bytes {name}: pages differing in the defined bytes {diff} ({len(diff)})", flush=True)
        bad += diff
    assert not bad, "one compact call != two default calls"
    for x in (q, k):
        a_ = ttnn.bringup.bfp_round(x, ttnn.bfloat4_b, prepared=True)
        b_ = ttnn.bringup.bfp_round(x, ttnn.bfloat4_b, prepared=True, compact=True)
        assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(a_)[0]), ttnn.to_torch(ttnn.get_device_tensors(b_)[0]))
    print("LOWBIT_HOST bytes: equal", flush=True)
    for k_, v in res.items():
        print(f"LOWBIT_HOST {k_}: {v * 1e3:.1f} us", flush=True)
