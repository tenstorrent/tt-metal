# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""kernels/restore.cpp (model/restore.py): the restore of the fused step's L1 contents at rest as ONE program, against
the host-write registry it is made from. No step runs here: the kernel's result is bytes.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k solo       (one chip)
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k all         (32 chips)

A Stack of two layers (the dense FFN and a MoE layer: every kind of host-written state) with random parameters on
the rung's chips. Then, on the board:
  1. the host writes random bytes into every registered tensor on the rung's chips (the registry switched off: its
     entries keep the contents at rest), and hw.verify finds every tensor different
  2. the restore program is launched once
  3. hw.verify(ranges_only=True): every registered tensor equal to what the host last wrote; the arena (scratch) zero
     in its three status ranges and, outside them, still the random bytes (the program left it alone)
  4. 20 launches back to back: the time of one
The semaphores' values cannot be read from the host: they are covered where a step follows the restore
(tests/test_coexist.py).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.model import reserve as RS
from models.experimental.hy4_preview.fused.model.program import is_mock
from models.experimental.hy4_preview.fused.model.restore import Restore
from models.experimental.hy4_preview.fused.model.tests.test_mock import build_random, galaxy

RUNGS = {"solo": [(0, 0)], "all": None}


def garbage(mesh, hw, chips, seed=7):
    """Random bytes from the host into every registered tensor (but the per-step ones) on ``chips`` (linear indices;
    the other chips get the contents at rest again). Returns the bytes written into the scratch tensors."""
    g = torch.Generator().manual_seed(seed)
    copy = hw.orig["copy_host_to_device_tensor"]
    kept = {}
    n = mesh.get_num_devices()
    for e in hw.live().values():
        if e.per_step:
            continue
        rest = ttnn.get_device_tensors(e.host)
        like = ttnn.to_torch(rest[0]).contiguous()
        noise = torch.randint(1, 255, (like.view(torch.uint8).numel(),), generator=g, dtype=torch.uint8)
        value = noise.view(like.dtype).reshape(like.shape)
        if value.dtype == torch.bfloat16:
            value = value.float()
        bad = ttnn.from_torch(value, dtype=e.tensor.dtype, layout=e.tensor.layout)
        if e.tensor.dtype == ttnn.bfloat16 or e.tensor.dtype == ttnn.float32:
            # (a float tensor's noise through the host conversion: what lands is what to_torch gives back)
            noise = ttnn.to_torch(bad).contiguous().view(torch.uint8).flatten()
        shards = [bad if c in chips else rest[c if len(rest) > 1 else 0] for c in range(n)]
        copy(ttnn.from_host_shards(shards, mesh.shape), e.tensor)
        if e.scratch is not None:
            kept[e.address] = noise
    ttnn.synchronize_device(mesh)
    return kept


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS))
def test_restore(mesh_device, rung):
    mesh = mesh_device
    coords = RUNGS[rung]
    st = build_random(mesh, 2, **({} if coords is None else {"chips": coords}))
    hw = st.hw
    hw.mark_per_step(st.a.rope, st.a.streams)
    r = Restore(mesh, hw, chips=coords)
    logger.info(
        f"HY4_MK11 restore [{rung}]: {hw.summary()}; {r.summary()}; unregistered L1 tensors of the Stack: {sorted(hw.unregistered(RS.device_tensors(st)))}"
    )
    try:
        chips = list(range(mesh.get_num_devices())) if coords is None else [st.topo.chip(c) for c in coords]
        if is_mock():
            garbage(mesh, hw, chips[:2])  # (the host side of the test: nothing lands on the mock)
            hw.verify(mesh, chips=chips[:1])
            hw.verify(mesh, ranges_only=True, chips=chips[:1])
            r.launch()
            return
        kept = garbage(mesh, hw, chips)
        before = hw.verify(mesh, chips=chips)
        entries = [e for e in hw.live().values() if not e.per_step]
        differ = {(address, c) for _, c, _, address in before}
        logger.info(
            f"HY4_MK11 restore [{rung}]: after the host's random bytes {len({a for a, _ in differ})} of {len(entries)} registered "
            f"tensors differ from their contents at rest on the rung's chips ({len(differ)} tensor-chip pairs of {len(entries) * len(chips)})"
        )
        assert len(differ) == len(entries) * len(chips), "the random bytes did not land everywhere"
        t0 = time.perf_counter()
        r.launch()
        ttnn.synchronize_device(mesh)
        first_ms = (time.perf_counter() - t0) * 1e3
        bad = hw.verify(mesh, ranges_only=True, chips=chips)
        logger.info(
            f"HY4_MK11 restore [{rung}]: after ONE launch ({first_ms:.1f} ms with its compile): registered tensors that are not at rest: {len(bad)} {bad[:6]}"
        )
        assert not bad, bad[:8]
        # the scratch tensor outside its zero ranges: untouched
        for e in entries:
            if e.scratch is None:
                continue
            cores = RS.cores_of(e.tensor)
            per = e.bytes() // len(cores)
            want = kept[e.address].clone()
            for core, offset, n in e.scratch[0]:
                start = cores.index(tuple(core)) * per + offset
                want[start : start + n] = 0
            for chip in chips:
                got = RS.chip_bytes(e.tensor, chip)[1]
                same = int((got == want).sum())
                logger.info(
                    f"HY4_MK11 restore [{rung}]: scratch {e.where} chip {chip}: {same} of {want.numel()} bytes as expected (random outside the zero ranges)"
                )
                assert same == want.numel()
                if chip >= 2:
                    break
        other = [c for c in range(mesh.get_num_devices()) if c not in chips]
        if other:
            off = hw.verify(mesh, chips=other[:3])
            logger.info(
                f"HY4_MK11 restore [{rung}]: chips outside the rung (the first three), against their contents at rest: {len(off)} differ"
            )
            assert not off, off[:4]
        t0 = time.perf_counter()
        for _ in range(20):
            r.launch()
        ttnn.synchronize_device(mesh)
        each = (time.perf_counter() - t0) * 1e3 / 20
        logger.info(
            f"HY4_MK11 restore [{rung}]: 20 launches back to back from the host: {each * 1e3:.0f} us each (untraced: with the host's dispatch)"
        )
        assert not hw.verify(mesh, ranges_only=True, chips=chips)
    finally:
        r.free()
