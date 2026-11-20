# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The log-sum-exp merge on REAL partials: fused/sparse_mla's attention runs on every chip of the mesh over that
chip's share of each user's picks (packed FP8 rows read by id from DRAM, 64 heads, 8 users), its output tensor is
the merge's input, and every chip's ctx of its own heads is compared with

  * the float64 merge of the device's own partials (the merge alone), and
  * the reference's softmax over ALL picked rows of all chips and the sink (sparse_mla layout.reference_ctx,
    which is reference/attention_decode.py absorbed_stages' ctx): attention and merge together.

Every chip holds the same storage (the rows of one stripe); what differs per chip is which rows the selection
gives it, so the union the reference attends over is the chips' lists one after another. Some (chip, user) pairs
get no rows, one user gets none at all.

    pytest <this file> -x          (after test_lse_merge.py has passed on the board)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.merges import lse_merge as lm
from models.experimental.hy4_preview.fused.sparse_mla import layout as lo
from models.experimental.hy4_preview.fused.sparse_mla.op import SparseMla
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MOCK = cl.is_mock()
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), (8, 4))], indirect=True, ids=["8x4-torus"]
)


def chip_lists(cfg, chips, seed):
    """Per chip and user the rows that chip attends over: user u's 2048 picks dealt over the chips unevenly
    (some chips get none, one gets many), the last user gets none anywhere."""
    g = torch.Generator().manual_seed(seed)
    lists = [[torch.zeros(0, dtype=torch.int64) for _ in range(cfg.users)] for _ in range(chips)]
    for u in range(cfg.users - 1):
        weights = torch.rand(chips, generator=g) ** 3
        weights[torch.randperm(chips, generator=g)[: chips // 8]] = 0
        owner = torch.multinomial(weights, cfg.topk, replacement=True, generator=g)
        for c in range(chips):
            n = int((owner == c).sum())
            lists[c][u] = torch.randperm(cfg.rows_per_slot, generator=g)[:n]
    return lists


@galaxy
def test_lse_real_partials(mesh_device):
    cfg = lo.SparseMlaConfig(rows_per_slot=4096, num_layers=1)
    chips = mesh_device.get_num_devices()
    latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(mesh_device, cfg, cfg.users, latent, rope)
    rows = rows[: cfg.users]  # every chip holds the same rows: the first chip's read-back
    lists = chip_lists(cfg, chips, seed=11)
    table = torch.stack([lo.selection_table(cfg, lists[c]) for c in range(chips)])[:, None]
    shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    sel = ttnn.from_torch(
        table,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=shard,
    )
    g = torch.Generator().manual_seed(7)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=g)
    sink = torch.randn(cfg.heads, generator=g)
    op = SparseMla(cfg, mesh_device)
    q = hs.to_dev(op.queries_to_pages(q_lat), mesh_device, ttnn.bfloat16)
    out = ttnn.from_torch(
        torch.zeros(chips, 1, cfg.users * cfg.head_groups, op.out_page_values),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=shard,
    )
    assert op.out_page_values == lm.PAGE_VALUES
    # One program for every chip: the buffers of a mesh tensor have one address on all chips.
    first = lambda t: ttnn.get_device_tensors(t)[0]
    program = op.attend_program(first(kv), first(sel), first(q), first(out), 0)
    # The attention first, the merge's persistent L1 buffers after it: the attention's circular buffers reach
    # 1,257,792 of the 1,427,520 bytes of L1 on its cores, which leaves 170 KB for L1 buffers while it runs,
    # and the merge's landings take 310 KB (lse_merge.py "L1").
    if MOCK:
        ttnn.experimental.prepare_generic_op([kv, sel, q, out], program)
        lm.LseMerge(mesh_device, partials=out).compile()
        return
    ttnn.generic_op([kv, sel, q, out], program)
    ttnn.synchronize_device(mesh_device)
    merge = lm.LseMerge(mesh_device, partials=out)
    partials = [op.pages_to_partials(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(out)]
    for c, (_, _, _, status) in enumerate(partials):
        assert (status[:, :, 1] == 0).all(), f"chip {c}: a wait of the attention expired: {status[:, :, 1].tolist()}"
    o, m, s = (torch.stack([p[i] for p in partials]) for i in range(3))
    merge.set_sink(sink)
    merge()
    got = merge.read()
    assert bool((got["status"] == 0x100).all()), [hex(int(x)) for x in got["status"].flatten()]
    own = merge.golden(o, m, s, sink)  # the merge of the device's partials
    ctx = got["ctx"].double()
    worst = {"merge": 1.0, "reference": 1.0, "merge_abs": 0.0, "reference_abs": 0.0}
    for u in range(cfg.users):
        union = torch.cat([rows[u][lists[c][u]].float() for c in range(chips)])
        if union.shape[0] == 0:
            assert float(ctx[:, u].abs().max()) == 0.0, f"user {u} has no rows: ctx must be 0"
            continue
        ref = lo.reference_ctx(q_lat[u].to(torch.bfloat16).float(), union, sink, cfg)  # [64, 512]
        for k in range(chips):
            heads = merge.heads_of(k // merge.C, k % merge.C)
            worst["merge"] = min(worst["merge"], pcc(ctx[k, u], own[k, u]))
            worst["reference"] = min(worst["reference"], pcc(ctx[k, u], ref[heads]))
            worst["merge_abs"] = max(worst["merge_abs"], (ctx[k, u] - own[k, u]).abs().max().item())
            worst["reference_abs"] = max(worst["reference_abs"], (ctx[k, u] - ref[heads]).abs().max().item())
        logger.info(
            f"HY4_MK6 lse real user {u}: {union.shape[0]} rows over {sum(1 for c in range(chips) if lists[c][u].numel())} "
            f"chips (most on one chip {max(lists[c][u].numel() for c in range(chips))})"
        )
    logger.info(
        f"HY4_MK6 lse real partials, 32 chips x 8 users x 64 heads: ctx against the float64 merge of the device's "
        f"partials: worst pcc {worst['merge']:.7f}, max abs {worst['merge_abs']:.3e}; against the reference softmax "
        f"over all rows and the sink: worst pcc {worst['reference']:.7f}, max abs {worst['reference_abs']:.3e}"
    )
    assert worst["merge"] >= 0.9999 and worst["reference"] >= 0.99
