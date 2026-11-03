# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""custom_ops/expert_gather_small vs the torch golden, single chip: the 26B (4,1) shapes (T 128 / 256, E 128, 32
local experts, top-8, H 2816) with random top-k routing, an empty expert, a full-region expert (every token on one
expert), the drop-free worst case (every token's 8 experts all local) and the 32 cores on a forced 8x8 grid; exact
buffer rows, counts, offsets, W."""

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.expert_gather_small import ExpertGatherSmall

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]


def _routing(T, E, K, mode, idx):
    """dense [T, E] bf16: K experts per token with random positive weights; ``mode`` shapes the selection."""
    r = torch.zeros(T, E)
    for t in range(T):
        if mode == "all_local":  # every token's K experts on this chip (the drop-free worst case)
            sel = torch.tensor(idx)[torch.randperm(len(idx))[:K]]
        elif mode == "one_expert":  # every token on local expert 0 (+ K-1 random others)
            others = torch.randperm(E)[: K - 1]
            sel = torch.unique(torch.cat([torch.tensor([idx[0]]), others]))[:K]
        else:
            sel = torch.randperm(E)[:K]
        r[t, sel] = torch.rand(len(sel)) + 0.1
    if mode == "empty_expert":  # local expert 1 never selected
        r[:, idx[1]] = 0
    return r.to(torch.bfloat16)


@pytest.mark.parametrize(
    "T, E, E_loc, K, H, mode",
    [
        (64, 64, 1, 2, 256, "random"),  # one core: no count exchange (localizes a hang to scan / copy / W)
        (64, 64, 2, 2, 256, "random"),  # the smallest cross-core exchange (leader + one sender)
        (64, 64, 4, 2, 256, "random"),  # the unit-test-sized shape (Wormhole-friendly), 4-core exchange
        (128, 128, 32, 8, 2816, "random"),
        (256, 128, 32, 8, 2816, "random"),
        (128, 128, 32, 8, 2816, "empty_expert"),
        (128, 128, 32, 8, 2816, "one_expert"),
        (256, 128, 32, 8, 2816, "all_local"),
        (128, 128, 32, 8, 2816, "random_wh8x8"),  # the 32 cores laid out on a forced 8x8 (Wormhole) grid
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_expert_gather_small(mesh_device, device_params, T, E, E_loc, K, H, mode, monkeypatch):
    torch.manual_seed(1)
    if mode.endswith("_wh8x8"):
        monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
        mode = mode[: -len("_wh8x8")]
    idx = list(range(32, 32 + E_loc)) if E >= 64 else list(range(E_loc))  # this chip's experts: a middle slab
    if mode == "empty_expert" and E_loc < 2:
        pytest.skip("needs two local experts")
    hidden = (torch.randn(1, 1, T, H) * 2).to(torch.bfloat16)
    routing = _routing(T, E, K, mode, idx)
    gb, gc, go, gw = ExpertGatherSmall.golden(hidden.reshape(T, H).float(), routing.float(), idx, T, K, E)
    op = ExpertGatherSmall(T, E, E_loc, K, H)
    th = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tr = ttnn.from_torch(routing.reshape(1, 1, T, E), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ti = ttnn.from_torch(
        torch.tensor(idx, dtype=torch.int32), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    buffer, counts, offsets, w = op.op(th, tr, ti)
    c, o = ttnn.to_torch(counts).flatten().long(), ttnn.to_torch(offsets).flatten().long()
    assert torch.equal(c, gc), f"counts differ: {c[idx].tolist()} vs {gc[idx].tolist()}"
    assert torch.equal(o, go), "offsets differ"
    b = ttnn.to_torch(buffer).float()
    wt = ttnn.to_torch(w).float().reshape(T, op.M)
    if not torch.equal(wt, gw):
        bad = torch.nonzero(wt != gw)[:12].tolist()
        print(
            f"W mismatches {int((wt != gw).sum())}: first (t, m) {bad}; op nonzeros {int((wt != 0).sum())} golden {int((gw != 0).sum())}"
        )
        raise AssertionError(f"W differs: {int((wt != gw).sum())} entries")
    used = torch.zeros(op.M, dtype=torch.bool)
    for g in idx:
        used[go[g] : go[g] + gc[g]] = True
    if not torch.equal(b[used], gb[used]):
        # diagnose: for every gathered row, which token's row the buffer actually holds (-1 = no token's row)
        hid = hidden.reshape(T, H).float()
        report = []
        for g in idx:
            for i in range(int(gc[g])):
                m = int(go[g]) + i
                hits = torch.nonzero((hid == b[m]).all(-1)).flatten().tolist()
                want = torch.nonzero((hid == gb[m]).all(-1)).flatten().tolist()
                if not torch.equal(b[m], gb[m]):
                    bm = b[m]
                    seg_ok = sum(int(torch.equal(bm[j : j + 32], gb[m][j : j + 32])) for j in range(0, H, 32))
                    report.append(
                        f"expert {g} row {i} (buffer row {m}): holds token {hits[:1] or [-1]}, expected token {want[:1]}; "
                        f"got {[round(v, 3) for v in bm[:6].tolist()]} (zeros {int((bm == 0).sum())}/{H}, nan {int(bm.isnan().sum())}) "
                        f"want {[round(v, 3) for v in gb[m][:6].tolist()]}; correct 32-col segments {seg_ok}/{H // 32}"
                    )
        print("\n".join(report[:24]))
        raise AssertionError(f"buffer rows differ: {len(report)} of {int(used.sum())} gathered rows (first 24 above)")
    assert int((wt != 0).sum()) == int(gc[idx].sum()), "W carries one weight per gathered row"
    print(
        f"[T{T} E{E} E_loc{E_loc} K{K} H{H} {mode}] rows gathered {int(gc[idx].sum())} of M {op.M}; counts {gc[idx].tolist()[:8]}..."
    )
    # cache hit on fresh tensors
    th2 = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    b2, c2, o2, w2 = op.op(th2, tr, ti)
    assert torch.equal(ttnn.to_torch(w2).float(), ttnn.to_torch(w).float()) and torch.equal(
        ttnn.to_torch(c2), ttnn.to_torch(counts)
    )


def test_local_expert_bound_host(expect_error):
    """The kernel's fixed scratch regions hold at most MAX_LOCAL_EXPERTS experts (static_asserts in the kernel:
    idx table 256 B, offset table 512 B whose entry E_loc = the end of the last region would land on counts[0]);
    supports() refuses above it so no build reaches the kernel's assert."""
    assert ExpertGatherSmall.MAX_LOCAL_EXPERTS == 64
    assert ExpertGatherSmall.supports(128, 64, 8, 2816) and ExpertGatherSmall.supports(128, 32, 8, 2816)
    assert not ExpertGatherSmall.supports(128, 65, 8, 2816) and not ExpertGatherSmall.supports(128, 128, 8, 2816)
    assert not ExpertGatherSmall.supports(256, 128, 8, 2816)
    assert not ExpertGatherSmall.supports(128, 32, 129, 2816) and ExpertGatherSmall.supports(128, 32, 128, 2816)
    with expect_error(AssertionError, "unsupported expert layout"):
        ExpertGatherSmall(128, 128, 128, 8, 2816)
    assert ExpertGatherSmall(128, 128, 64, 8, 2816).M == 128 * 8 + 32 * 63
