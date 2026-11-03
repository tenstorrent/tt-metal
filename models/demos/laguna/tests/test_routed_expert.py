# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna's routed-expert generic_op (tt/routed_expert.py) against a torch SiLU-gated FFN per expert.

Laguna-S-2.1 shapes on a 1 x D mesh (D = min(4, devices)): 64 local experts per chip, hidden 3072, moe
intermediate 1024, top-k 10, BFP4 stacked weights [1, 64, 3072, 2048] (gate | up) and [1, 64, 1024, 3072] built the
way MultichipDecoder builds them (bf4 TILE DRAM, gate|up concat on device, then colpage.py's column pages), and the
row-major bf16 dispatch buffer [T * 10 + 31 * 64, 3072] with counts / region offsets / the local -> global index
table laid out like Laguna's token dispatch (LagunaRoutingSetup, one-chip dispatch groups). Every chip gets its own
routing, so the chips run different counts through the same program.

The op computes in bf8 activations and LoFi (exact for bf4 x bf8 operands); the reference uses the device's own
dequantized weights and the bf16 x in fp32, so the gap is the bf8 rounding of x, the intermediates and the output.
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.laguna.tt import colpage
from models.demos.laguna.tt.routed_expert import routed_expert_plan, routed_expert_stacked

# Laguna-S-2.1 on p150x4 (multichip_decoder.TOKEN_DISPATCH_GEOMETRIES)
LOCAL_EXPERTS = 64
HIDDEN = 3072
MOE_INTERMEDIATE = 1024
TOP_K = 10
TILE = 32
PCC_MIN = 0.99


def open_mesh():
    """1 x D mesh, D = min(4, devices); no fabric (the op is chip-local)."""
    num_devices = min(4, ttnn.get_num_devices())
    return ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices))


def build_weights(mesh, *, column_pages: bool, seed: int = 0):
    """Stacked BFP4 expert weights as MultichipDecoder builds them (replicated: every chip holds the same 64 experts
    here). Returns (gate_up, down, host) where host = (gate [E, H, I], up [E, H, I], down [E, I, H]) bf16 holds the
    device's dequantized values (read back before the column-page relayout, which moves tiles without changing them)."""
    g = torch.Generator().manual_seed(seed)
    E, H, I = LOCAL_EXPERTS, HIDDEN, MOE_INTERMEDIATE

    def make(shape, fan_in):
        return (torch.randn(shape, generator=g) / fan_in**0.5).to(torch.bfloat16)

    replicate = ttnn.ReplicateTensorToMesh(mesh)

    def upload(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    gate = upload(make((1, E, H, I), H))
    up = upload(make((1, E, H, I), H))
    down = upload(make((1, E, I, H), I))
    gate_up = ttnn.concat([gate, up], dim=-1)  # _pack_gate_up: [1, E, H, 2I]
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    def chip0(t):
        # bf4 values are exact in bf16; [E, K, N]
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0], dtype=torch.bfloat16).reshape(list(t.padded_shape)[-3:])

    gu_host, down_host = chip0(gate_up), chip0(down)
    host = (gu_host[..., :I], gu_host[..., I:], down_host)
    if column_pages:  # _pack_gate_up's TT_LAGUNA_COLPAGE step (the default)
        gate_up_cp, down_cp = colpage.to_column_pages(gate_up), colpage.to_column_pages(down)
        ttnn.deallocate(gate_up)
        ttnn.deallocate(down)
        gate_up, down = gate_up_cp, down_cp
    return gate_up, down, host


def global_expert_idx_table(mesh):
    """MultichipDecoder._setup_token_dispatch's index table: per chip [64] uint32 row-major (local slot -> global id).
    Returns (device tensor, host [D, 64])."""
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper

    D = mesh.get_num_devices()
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=LOCAL_EXPERTS, dispatch_group_size=1, num_dispatch_groups=D
    )
    tt = ttnn.from_torch(
        table, mesh_mapper=get_ep_mesh_mapper(mesh), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, dtype=ttnn.uint32
    )
    tt = ttnn.squeeze(ttnn.squeeze(tt, 0), 0)
    return tt, table.reshape(D, LOCAL_EXPERTS).long()


def build_dispatch(mesh, seq_len: int, idx_host, *, seed: int):
    """Per chip: top-k routing of seq_len tokens over the D * 64 global experts (an always-picked hot expert whose count
    reaches seq_len, four never-picked experts, the rest random, so counts are mostly not multiples of 32), the counts
    and region offsets LagunaRoutingSetup produces ([1, D * 64] uint32 row-major, non-local experts 0) and the row-major
    bf16 dispatch buffer [seq_len * 10 + 31 * 64, 3072] with each local expert's tokens at its region (other rows
    hold noise, as a real dispatch buffer holds stale data). Returns device tensors and the host tensors."""
    D = mesh.get_num_devices()
    E_glob = LOCAL_EXPERTS * D
    rows = seq_len * TOP_K + (TILE - 1) * LOCAL_EXPERTS  # MultichipDecoder's max_dispatch_rows
    g = torch.Generator().manual_seed(1000 + seed)
    counts = torch.zeros(D, E_glob, dtype=torch.int64)
    region = torch.zeros(D, E_glob, dtype=torch.int64)
    xbuf = torch.empty(D, rows, HIDDEN, dtype=torch.bfloat16)
    for d in range(D):
        xbuf[d] = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
        local = idx_host[d]
        probs = torch.ones(E_glob)
        probs[local[[3, 17, 40, 63]]] = 0.0  # zero-count experts
        probs[local[0]] = 1e6  # hot expert: picked by (nearly) every token
        probs[local[5]] = 20.0  # warm expert
        indices = torch.multinomial(probs.expand(seq_len, E_glob).contiguous(), TOP_K, replacement=False, generator=g)
        hist = torch.bincount(indices.flatten(), minlength=E_glob)
        is_local = torch.zeros(E_glob, dtype=torch.bool)
        is_local[local] = True
        hist[~is_local] = 0
        counts[d] = hist
        padded = (hist + TILE - 1) // TILE * TILE
        for g0 in range(0, E_glob, LOCAL_EXPERTS):
            grp = padded[g0 : g0 + LOCAL_EXPERTS]
            region[d, g0 : g0 + LOCAL_EXPERTS] = torch.cumsum(grp, 0) - grp
        tokens = torch.randn(seq_len, HIDDEN, generator=g).to(torch.bfloat16)
        for gid in local.tolist():
            routed = (indices == gid).any(dim=1).nonzero().flatten()
            s, n = int(region[d, gid]), int(counts[d, gid])
            assert n == routed.numel() and s + n <= rows
            xbuf[d, s : s + n] = tokens[routed]
    shard0 = ttnn.ShardTensorToMesh(mesh, dim=0)

    def vec(t):
        return ttnn.from_torch(
            t.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=shard0
        )

    x = ttnn.from_torch(
        xbuf.reshape(D * rows, HIDDEN),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard0,
    )
    return dict(x=x, counts=vec(counts), region=vec(region)), dict(x=xbuf, counts=counts, region=region)


def chip_to_torch(t, d):
    """Chip d's [R, K] of a TILE bf8 output, as bf16 (bf8 values are exact in bf16)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[d], dtype=torch.bfloat16)


def expert_rows(host, d, gid):
    s, n = int(host["region"][d, gid]), int(host["counts"][d, gid])
    return s, n


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / ((a.norm() * b.norm()).clamp_min(1e-30)))


@pytest.mark.parametrize("seq_len", [128, 1024, 4096, 8192])  # row groups of 1, 1, 2 and 5 grid rows
def test_routed_expert_matches_torch(seq_len):
    mesh = open_mesh()
    try:
        D = mesh.get_num_devices()
        gate_up, down, (wg, wu, wd) = build_weights(mesh, column_pages=True)
        idx, idx_host = global_expert_idx_table(mesh)
        for seed in (0, 1):  # the second call reuses the cached program with new buffer addresses
            dev, host = build_dispatch(mesh, seq_len, idx_host, seed=seed)
            plan = routed_expert_plan(dev["x"], dev["region"], dev["counts"], idx, gate_up, down, seq_len)
            out = routed_expert_stacked(dev["x"], dev["region"], dev["counts"], idx, gate_up, down, seq_len)
            for d in range(D):
                got_chip = chip_to_torch(out, d)
                got_all, ref_all = [], []
                # the hot expert (count ~ seq_len: every chunk), the warm one, three random ones, and a zero-count one
                for e in (0, 5, 3, 21, 42, 58):
                    gid = int(idx_host[d, e])
                    s, n = expert_rows(host, d, gid)
                    if e == 3:
                        assert n == 0, f"chip {d}: expert {e} was meant to have no tokens"
                        continue
                    if n == 0:
                        continue
                    xe = host["x"][d, s : s + n].float()
                    ref = (torch.nn.functional.silu(xe @ wg[e].float()) * (xe @ wu[e].float())) @ wd[e].float()
                    got = got_chip[s : s + n].float()
                    assert torch.isfinite(got).all(), f"chip {d} expert {e}: non-finite output"
                    got_all.append(got)
                    ref_all.append(ref)
                p = pcc(torch.cat(ref_all), torch.cat(got_all))
                print(
                    f"seq_len {seq_len} seed {seed} chip {d}: row_group {plan['row_group']} per_core_M "
                    f"{plan['per_core_M']} in0_block_w_gu {plan['in0_block_w_gu']} hot count "
                    f"{int(host['counts'][d, int(idx_host[d, 0])])} pcc {p:.6f}"
                )
                assert p >= PCC_MIN, f"seq_len {seq_len} seed {seed} chip {d}: PCC {p:.6f} < {PCC_MIN}"
            for t in (out, dev["x"], dev["counts"], dev["region"]):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)
