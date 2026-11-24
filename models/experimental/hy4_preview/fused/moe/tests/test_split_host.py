# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the "N chips per group" split of fused/moe/prepare.py (tp8g4: 4 groups of 8 chips, group g owns the
routed experts [64 g, 64 g + 64), each chip a 1/8 slice of each of its group's experts, the shared expert on one
group).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

test_group_partials_sum_to_reference   a small MoE: the sum over the 32 chips of the partial golden of what the
                                       chip runs (prepare.chip_union: the hub's union restricted to its group,
                                       plus the plain slot on the shared expert's group) on the chip's weights
                                       (prepare.chip_weights) is reference/moe.py's routed + shared output; a
                                       group nobody picked contributes exact zeros; picks all in one group.
test_files_round_trip                  the converted files of a checkpoint layer (skipped when the split is not
                                       converted): every chip's two files have one shape, a chip's stream is the
                                       stream of ITS group's experts at ITS slice (bit for bit against the
                                       checkpoint through layout.gate_up_stream / down_stream), the two entries
                                       behind the routed ones are the shared expert's two terms on the shared
                                       group and zeros elsewhere.
Log lines start with HY4_TP8."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.reference.moe import Hy4MoE, MoEConfig

SPLIT = "tp8g4"


def test_split_names():
    assert prepare.split_groups("tp32") is None and prepare.split_groups("ep") is None
    assert prepare.split_groups(SPLIT) == (8, 4)
    assert prepare.split_chips(SPLIT) == 32 and prepare.split_chips("tp32") == 32
    share = prepare.split_share(SPLIT)
    assert (share.P, share.S, share.pb, share.piece_bytes) == (8, 1, 1, 221184)
    # row-major on the 8x4 mesh: the group is the mesh column, the slice the mesh row
    assert [prepare.chip_group(SPLIT, c) for c in (0, 1, 3, 4, 31)] == [(0, 0), (1, 0), (3, 0), (0, 1), (3, 7)]
    assert [prepare.group_experts(SPLIT, g) for g in range(4)] == [(0, 64), (64, 64), (128, 64), (192, 64)]
    # every (expert, intermediate channel) is held by exactly one chip
    held = torch.zeros(256, 8, dtype=torch.int32)
    for chip in range(32):
        group, part = prepare.chip_group(SPLIT, chip)
        base, count = prepare.group_experts(SPLIT, group)
        held[base : base + count, part] += 1
    assert bool((held == 1).all())


def _small(seed=0):
    cfg = MoEConfig(hidden_size=96, num_experts=16, top_k=4, intermediate_size=64)
    torch.manual_seed(seed)
    ref = Hy4MoE(cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn(p.shape) * (0.3 if p.dim() > 1 else 0.05))
    return cfg, ref


@pytest.mark.parametrize("shared_group", [0, 3])
@pytest.mark.parametrize("picks", ["router", "one_group", "avoid_group"])
def test_group_partials_sum_to_reference(picks, shared_group):
    cfg, ref = _small()
    E, I = cfg.num_experts, cfg.intermediate_size
    x = torch.randn(8, cfg.hidden_size) * 3  # large enough for the clamp to bite
    with torch.no_grad():
        _, idx, w = L.route_golden(x, ref.router_weight, ref.router_bias, cfg.top_k, cfg.routed_scaling_factor)
        if picks == "one_group":  # every row's picks inside group 1: experts 4..7
            idx = torch.tensor([[4, 5, 6, 7]] * 8)
        elif picks == "avoid_group":  # nobody picks group 2 (experts 8..11)
            idx = torch.tensor([[0, 1, 4, 12], [2, 3, 5, 13], [0, 6, 7, 14], [1, 2, 15, 12]] * 2)
        dense = ref.dense_weights(w, idx)
        want = ref.routed(x, dense) + ref.shared(x)
        sh = ref.shared
        gate_up = torch.cat((ref.gate_up_proj, torch.cat((sh.gate_proj, sh.up_proj), dim=0)[None]), dim=0)
        down = torch.cat((ref.down_proj, sh.down_proj[None]), dim=0)
        total = torch.zeros_like(want)
        idle, slots = [], []
        for chip in range(32):
            gate, up, dn = prepare.chip_weights(gate_up, down, SPLIT, chip, routed=E, shared_group=shared_group)
            assert gate.shape == (E // 4 + 1, I // 8, cfg.hidden_size) and dn.shape == (
                E // 4 + 1,
                cfg.hidden_size,
                I // 8,
            )
            ids, cw, n_plain = prepare.chip_union(SPLIT, chip, idx, w, routed=E, shared_group=shared_group)
            group, _ = prepare.chip_group(SPLIT, chip)
            assert n_plain == (1 if group == shared_group else 0)
            if group != shared_group:  # the entries behind the routed ones are never read there: zeros
                assert not gate[E // 4 :].any() and not up[E // 4 :].any() and not dn[E // 4 :].any()
            part = L.partial_golden(x, ids, cw, gate, up, dn, limit=cfg.swiglu_limit, n_plain=n_plain)
            if not ids:
                idle.append(chip)
                assert not part.any()
            slots.append(len(ids))
            total += part
        assert torch.allclose(total, want, atol=2e-4), (total - want).abs().max()
        # the layout's own numbers: a group's 8 chips run the same list; the lists differ between groups
        by_group = [slots[g] for g in range(4)]
        assert all(slots[c] == by_group[c % 4] for c in range(32))
        if picks == "one_group":
            assert by_group == [4 * (g == 1) + (g == shared_group) for g in range(4)]
            assert len(idle) == 8 * sum(1 for g in range(4) if g not in (1, shared_group))
        if picks == "avoid_group":
            assert by_group[2] == (2 == shared_group)
    logger.info(
        f"HY4_TP8 host {picks} shared on group {shared_group}: slots per group {by_group}, chips with an empty union {len(idle)}, "
        f"sum of the 32 partials vs reference routed + shared max_abs={(total - want).abs().max():.2e}"
    )


@pytest.mark.parametrize("layer", [1])
def test_files_round_trip(layer):
    import ttnn

    out = Path(prepare.DEFAULT_OUT)
    if not (out / f"layer{layer}" / SPLIT / "manifest.json").is_file():
        pytest.skip(f"layer {layer} is not converted to {SPLIT} under {out}")
    if not os.environ.get("HY4_HF_MODEL"):
        pytest.skip("needs the checkpoint (HY4_HF_MODEL)")
    share = prepare.split_share(SPLIT)
    gate_up, down = prepare.layer_experts(layer, shared=True)  # entries 256, 257: the shared expert's two terms
    assert gate_up.shape[0] == 258
    values = lambda path: ttnn.to_torch(ttnn.load_tensor(str(path))).float()[0, 0]
    shapes = set()
    for chip in range(32):
        for path in prepare.stream_paths(out, layer, SPLIT, chip):
            assert path.is_file(), path
            shapes.add((path.name.split(".")[1], path.stat().st_size))
    assert len(shapes) == 2, f"the chips' files differ in size: {shapes}"
    entries = 64 + 2
    rows_gu = entries * share.gate_up_stride // L.BFP4_TILE_BYTES * L.TILE
    rows_dn = entries * share.down_stride // L.BFP4_TILE_BYTES * L.TILE
    for chip in (0, 6, 7, 31):  # groups 0, 2, 3, 3; slices 0, 1, 1, 7
        group, part = prepare.chip_group(SPLIT, chip)
        base, count = prepare.group_experts(SPLIT, group)
        p_gu, p_dn = prepare.stream_paths(out, layer, SPLIT, chip)
        gu, dn = values(p_gu), values(p_dn)
        assert gu.shape == (rows_gu, 256) and dn.shape == (rows_dn, 256), (gu.shape, dn.shape)
        # the same stream built here from the checkpoint: this group's experts, this slice, the shared expert behind
        c0, c1 = part * 256, part * 256 + 256
        sel = list(range(base, base + count)) + ([256, 257] if group == prepare.GROUP_SHARED else [])
        gate = gate_up[sel, c0:c1].float()
        up = gate_up[sel, 2048 + c0 : 2048 + c1].float()
        dw = down[sel][:, :, c0:c1].float()
        pad = entries - len(sel)
        if pad:
            gate = torch.cat((gate, torch.zeros(pad, *gate.shape[1:])), dim=0)
            up = torch.cat((up, torch.zeros(pad, *up.shape[1:])), dim=0)
            dw = torch.cat((dw, torch.zeros(pad, *dw.shape[1:])), dim=0)
        want_gu = ttnn.to_torch(prepare.host_stream(L.gate_up_stream(gate, up, share))).float()[0, 0]
        want_dn = ttnn.to_torch(prepare.host_stream(L.down_stream(dw, share))).float()[0, 0]
        d_gu, d_dn = int((gu != want_gu).sum()), int((dn != want_dn).sum())
        # one tile by hand, from the checkpoint's own indices: local entry 5 = expert base + 5, pair 3 of the slice
        e, bank, k = 5, 3, 7  # (P 8, S 1: pair p = bank; K tile k)
        tile0 = e * share.piece_k * 2 + k * 2
        rows, cols = slice(tile0 * 32, tile0 * 32 + 32), slice(bank * 32, bank * 32 + 32)
        src = gate_up[base + e, c0 + bank * 32 : c0 + bank * 32 + 32, k * 32 : k * 32 + 32].float().T
        tile_err = (gu[rows, cols] - src).abs().max().item() / src.abs().max().item()
        tail = gu[64 * share.gate_up_stride // L.BFP4_TILE_BYTES * L.TILE :]
        logger.info(
            f"HY4_TP8 files layer {layer} chip {chip} (group {group}, slice {part}): gate|up {tuple(gu.shape)}, down {tuple(dn.shape)}; "
            f"elements that differ from the stream built from the checkpoint: {d_gu}, {d_dn}; one tile of expert {base + e} "
            f"against the checkpoint's values: max rel err {tile_err:.3f} (BFP4); entries 64, 65 {'hold the shared expert' if tail.any() else 'are zero'}"
        )
        assert d_gu == 0 and d_dn == 0
        assert tile_err < 0.15
        assert bool(tail.any()) == (group == prepare.GROUP_SHARED)
