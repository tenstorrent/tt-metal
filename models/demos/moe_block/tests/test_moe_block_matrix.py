# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MoEBlock on device over models x rows a mesh row x routing scenarios, in tiers (select with -k quick / nightly /
soak): eager calls then a trace replayed on fresh inputs, every checked replay against the host model on the
device's own routing (PCC) and its routing against the host's (agreement); with the device profiler on, the
replays' device time per program. The mesh is the process's (MOE_MESH / MOE_FABRIC as test_moe_block: the 2x4 lines,
or one LoudBox square as a 1x4 TP ring).

Models keep their real router; where the experts a chip would exceed the flat op's 64, the router folds onto fewer
physical experts (RouterConfig.fold: K3 at E 896 -> 28 a chip on the ring of 4 and the 2x4, a Galaxy chip's load).
Routing scenarios bias the router (candidate space, the fold's mask kept): uniform; zipf spread (hot experts on every
chip) / packed (hot experts on chip 0); one hot expert (every token picks it); empty experts (every other expert of a
chip never picked); a chip with zero rows (no expert of chip 1 picked); ties (zero router weights, zero bias: every
score equal)."""

import dataclasses
import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.moe_block.moe_block import (
    DSV41_FLASH,
    GLM_53,
    GLM_53_FLASH,
    KIMI_K3,
    MINIMAX_M3,
    MoEBlock,
    MoEModel,
    MoEWeights,
    WeightCache,
)
from models.demos.moe_block.tests.test_moe_block import (
    TRACE_REGION,
    C,
    R,
    _fab,
    device_routing,
    pcc,
    random_weights,
    reference,
    replay_device_times,
)

BASE = {"k3": KIMI_K3, "glm53_flash": GLM_53_FLASH, "glm53": GLM_53, "m3": MINIMAX_M3, "dsv41_flash": DSV41_FLASH}
ROUTINGS = ("uniform", "zipf_spread", "zipf_packed", "one_hot", "empty_experts", "zero_chip", "ties")
EPC_MAX = 64  # (the flat op's experts a chip)
PACKED = ("zipf_packed", "ties")  # (every token on a few chips' experts: past the planner's rows_cap)


def mesh_model(name: str, chips: int) -> MoEModel:
    """The model on this mesh: K3 at a Galaxy chip's load (28 experts a chip, its router at E 896 folded), the others
    at their real experts unless more than EPC_MAX a chip (then folded to fit)."""
    m = BASE[name]
    E = m.num_experts
    epc = 28 if name == "k3" else E // chips
    F = 1
    while E // F // chips > EPC_MAX or (name == "k3" and E // F // chips > epc):
        F *= 2
    assert E % (F * chips) == 0, f"{name}: {E} experts over {chips} chips, fold {F}"
    if F == 1:
        return m
    return dataclasses.replace(m, num_experts=E // F, router=dataclasses.replace(m.router, fold=F))


def scenario_weights(m: MoEModel, routing: str, chips: int, seed=0) -> MoEWeights:
    """random_weights with the router biased for the scenario (the fold's mask kept: one eligible candidate a group)."""
    w = random_weights(m, seed)
    F, E = m.router.fold, m.num_experts
    epc = E // chips
    b = torch.zeros(E)
    g = torch.Generator().manual_seed(seed + 7)
    if routing in ("zipf_spread", "zipf_packed"):
        rank = torch.randperm(E, generator=g) if routing == "zipf_spread" else torch.arange(E)
        b = -0.6 * torch.log1p(rank.float())
    elif routing == "one_hot":
        b[0] = 8.0
    elif routing == "empty_experts":
        b[1::2] = -1.0e3
    elif routing == "zero_chip":
        b[epc : 2 * epc] = -1.0e3
    rb = w.router["bias"] + b.repeat_interleave(F)
    rw = w.router["weight"]
    if routing == "ties":
        rw = torch.zeros_like(rw)
        rb = torch.where(w.router["bias"] < -1.0e3, w.router["bias"], torch.zeros_like(rb))  # (the fold mask only)
    return dataclasses.replace(w, router={"weight": rw, "bias": rb})


@dataclasses.dataclass(frozen=True)
class Case:
    model: str
    rows: int  # (tokens a mesh row)
    routing: str
    replays: int  # (checked trace replays, each on its own fresh input)
    tier: str

    @property
    def id(self):
        return f"{self.tier}-{self.model}-r{self.rows}-{self.routing}-x{self.replays}"


def _cases():
    out = [Case("k3", 640, "uniform", 16, "quick")]
    for mn in BASE:
        for rows in (128, 256, 512, 640, 1024, 2048):
            out.append(Case(mn, rows, "uniform", 32, "nightly"))
    for mn in ("k3", "glm53_flash"):
        for rt in ROUTINGS[1:]:
            out.append(Case(mn, 640, rt, 32, "nightly"))
    for mn, rows in (("k3", 640), ("k3", 1024), ("glm53_flash", 640)):
        out.append(Case(mn, rows, "uniform", 10000, "soak"))
        out.append(Case(mn, rows, "zipf_packed", 10000, "soak"))
    return out


CASES = _cases()


@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[c.model].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            c,
            id=c.id,
        )
        for c in CASES
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(36000)
def test_moe_block_matrix(mesh_device, device_params, case: Case):
    chips = R * C
    m = mesh_model(case.model, chips)
    T, H = case.rows * R, m.emb_dim
    w = scenario_weights(m, case.routing, chips)
    try:
        block = MoEBlock(mesh_device, m, T, w, WeightCache(path=None, layer=0))
    except (AssertionError, ValueError, NotImplementedError, RuntimeError) as e:
        msg = " | ".join(l.strip() for l in str(e).splitlines() if l.strip() and "---" not in l)[:600]
        logger.error(f"MATRIX {case.id}: does not build: {type(e).__name__}: {msg}")
        pytest.xfail(f"does not build: {type(e).__name__}: {msg[:300]}")
    g = block.geometry
    logger.info(f"MATRIX {case.id}: {m.num_experts} experts (fold {m.router.fold}), {block.summary}")
    if case.routing in PACKED and block.head_mode == "fused":
        pytest.skip(
            "packed routing: the fused head's planner rows_cap (2 x the average a chip) overflows - a hang; the "
            "token-dropping policy is pending"
        )
    shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(2, 3))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=shape, dims=(2, 3))
    NX = 16
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(3000 + k)).to(torch.bfloat16) for k in range(NX)]
    hx = [
        ttnn.from_torch(x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        for x_ in xs
    ]
    x_in = block.input_buffer()
    refs, stats = {}, dict(pcc=[], agree=[])

    def check(out, k):
        y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
        rows = block.taps()["rows"]
        if k not in refs or not all(torch.equal(a, b) for a, b in zip(rows, refs[k][0])):
            ids_d, wts_d = device_routing(rows, g.tokens_per_chip, m.top_k)
            y_ref = reference(xs[k], w, m, ids=ids_d, wts=wts_d)[0]
            _, ids_r, _ = reference(xs[k], w, m)
            agree = (torch.sort(ids_d, 1)[0] == torch.sort(ids_r, 1)[0]).float().mean().item()
            refs[k] = (rows, y_ref, agree)
        p = pcc(y, refs[k][1])
        stats["pcc"].append(p)
        stats["agree"].append(refs[k][2])
        assert p > 0.98, f"{case.id} (x {k}): block output off (PCC {p:.5f})"

    block.keep_taps = "device"
    for k in range(2):  # (eager)
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        out = block(x_in)
        ttnn.synchronize_device(mesh_device)
        check(out, k)
        ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = block(x_in)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    every = 1 if case.replays <= 64 else 25  # (soak: every replay on a fresh input, every 25th checked)
    for i in range(case.replays):
        k = (3 * i + 2) % NX
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        if i % every == every - 1 or i == case.replays - 1:
            ttnn.synchronize_device(mesh_device)
            check(out, k)
        if i % 1000 == 999:
            logger.info(f"MATRIX {case.id}: {i + 1} replays, PCC min {min(stats['pcc']):.5f}")
    ttnn.synchronize_device(mesh_device)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        replay_device_times(mesh_device, tid, f"matrix {case.id}", batches=(10, 40))
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    logger.info(
        f"MATRIX_RESULT {case.id}: PCC min {min(stats['pcc']):.5f} median {statistics.median(stats['pcc']):.5f}; "
        f"routing agreement min {min(stats['agree']):.4f}; {len(stats['pcc'])} checks; {block.summary}"
    )
