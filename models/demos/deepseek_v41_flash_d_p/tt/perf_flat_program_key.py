# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.5 (owner order 2026-10-27): flat_routed_expert_with_program's program-cache key, EXPERTS_PROGRAM_KEY patch
(the op's patch_addresses: no address in the key, every one patched on a hit) vs addresses (the op before: every
baked address in the key). Not a gate: numbers for BREADCRUMBS.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_flat_program_key.py -s

One layer's routed experts (real weights, layer LAYER, the run's chunk, random inputs as perf_experts.test_weighted),
per mode: (1) STEADY: the same call repeated (the buffers land at the same addresses: both modes hit); (2) MOVING: a
DRAM tensor about the gathered x's size allocated before every call shifts the gathered x (and the other per-call DRAM buffers) to a new
address. Per call: the program-cache entries it added and the op's host time (the Python call of
flat_routed_expert_with_program, dispatch is asynchronous). Outputs of every call are compared bit for bit with the
first call of the addresses mode (the op before).
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt import experts as E
from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_device
from models.demos.deepseek_v41_flash_d_p.tt.perf_experts import _host, _weighted_params

LAYER = 0
N_STEADY = 6
N_MOVING = 6


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_program_key(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, cfg.dim, generator=g)
    sc = torch.rand(chunk, cfg.n_experts, generator=g)
    top = sc.topk(cfg.topk_experts, dim=-1)
    dense = torch.zeros(chunk, cfg.n_experts)
    dense.scatter_(1, top.indices, 1.5 * top.values / top.values.sum(-1, keepdim=True))
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    mod = E.build_experts(mesh_device, ck, cfg, LAYER, chunk)
    xd = row_split_to_device(mesh_device, x, dtype=xdt)
    rd = row_split_to_device(mesh_device, dense)

    op = ttnn.bringup.flat_routed_expert_with_program
    host_s = []

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = op(*a, **k)
        host_s.append(time.perf_counter() - t0)
        return r

    monkeypatch.setattr(ttnn.bringup, "flat_routed_expert_with_program", timed)
    ref = None
    shift = []  # (MOVING: the tensors that push the next call's buffers to new addresses)
    env = settings.env_name("EXPERTS_PROGRAM_KEY")
    try:
        for mode in ("addresses", "patch"):
            monkeypatch.setenv(env, mode)
            settings.refresh()  # (HOST_GLUE=on, P.21: the model froze the settings at load)
            for phase, n in (("steady", N_STEADY), ("moving", N_MOVING)):
                rows = []
                for i in range(n):
                    if phase == "moving":
                        shift.append(
                            ttnn.allocate_tensor_on_device(
                                ttnn.Shape([chunk + 32 * i, cfg.dim]),  # (about the gathered x's size: it must move)
                                ttnn.bfloat16,
                                ttnn.ROW_MAJOR_LAYOUT,
                                mesh_device,
                                ttnn.DRAM_MEMORY_CONFIG,
                            )
                        )
                    ttnn.synchronize_device(mesh_device)
                    n0 = mesh_device.num_program_cache_entries()
                    k0 = len(host_s)
                    od = mod(xd, rd)
                    ttnn.synchronize_device(mesh_device)
                    new = mesh_device.num_program_cache_entries() - n0
                    o = _host(mesh_device, od)
                    ttnn.deallocate(od)
                    if ref is None:
                        ref = o
                    same = bool(torch.equal(o, ref))
                    op_ms = 1e3 * sum(host_s[k0:])
                    rows.append((new, op_ms, same))
                    print(
                        f"FIX5_KEY mode={mode} {phase} call {i}: +{new} program-cache entries, op host "
                        f"{op_ms:.3f} ms ({len(host_s) - k0} op calls), bit-identical to the first call: {same}"
                    )
                    assert same, f"{mode} {phase} call {i}: output differs from the address-keyed op's"
                hits = [ms for new, ms, _ in rows[1:] if new == 0]
                miss = [ms for new, ms, _ in rows if new > 0]
                print(
                    f"FIX5_KEY_SUMMARY mode={mode} {phase}: entries added per call {[r[0] for r in rows]}; op host on "
                    f"hits median {statistics.median(hits) if hits else float('nan'):.3f} ms ({len(hits)} calls), on "
                    f"misses median {statistics.median(miss) if miss else float('nan'):.3f} ms ({len(miss)} calls)"
                )
                if mode == "patch" and phase == "moving":
                    assert all(r[0] == 0 for r in rows), "patch mode: a moved buffer must not build a program"
    finally:
        monkeypatch.delenv(env, raising=False)
        for t in shift:
            ttnn.deallocate(t)
        for t in (xd, rd):
            ttnn.deallocate(t)
        mod.free()
