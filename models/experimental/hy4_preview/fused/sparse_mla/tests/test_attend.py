# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse row read + latent attention with all 64 heads for 8 query rows (8 users) on one chip.

The chip's partial (O, max, sum) per user and head is compared with the torch partial of the same rows
(layout.partial), and, merged with the sink (layout.merge, the contract of the cross-chip merge), with
reference/attention_decode.py's absorbed core. Rows come from a packed storage built with the prefill stack's
own pack ops: random latents, and latents and queries produced by the reference attention from a checkpoint
layer's weights.

Mock Galaxy (HY4_MOCK=1): the program is compiled and its CBs / kernel sizes checked, nothing is compared."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.sparse_mla import layout as lo
from models.experimental.hy4_preview.fused.sparse_mla.op import STATUS_MAGIC, SparseMla
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.tests.common import pcc

FIDELITY = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}

COUNTS = {
    "one_user_64": [64, 0, 0, 0, 0, 0, 0, 0],
    "one_user_2048": [0, 0, 0, 2048, 0, 0, 0, 0],
    "all_1": [1] * 8,
    "all_37": [37] * 8,
    "all_64": [64] * 8,
    "all_512": [512] * 8,
    "all_2048": [2048] * 8,
    "all_0": [0] * 8,
    "mixed": [1, 37, 512, 2048, 0, 64, 129, 300],
}

# One device open for the module: every test takes the plain `device` fixture.
pytestmark = pytest.mark.use_module_device


def run_attend(device, cfg, kv, rows, lists, q_lat, layer, fidelity, op=None):
    """One launch. Returns the device partials (O, m, s) and the per-core status words."""
    op = op or SparseMla(cfg, device)
    sel = hs.make_selection(device, cfg, lists)
    q = hs.to_dev(op.queries_to_pages(q_lat), device, ttnn.bfloat16)
    out = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.head_groups, op.out_page_values), device, ttnn.bfloat16)
    if hs.MOCK:
        ttnn.experimental.prepare_generic_op(
            [kv, sel, q, out] + op.io_extra(), op.attend_program(kv, sel, q, out, layer, fidelity=fidelity)
        )
        return None
    op.attend(kv, sel, q, out, layer, fidelity=fidelity)
    return op.pages_to_partials(ttnn.to_torch(out))


def check_partials(cfg, got, rows, lists, q_lat, sink, layer, tag, floor, op=None):
    o, m, s, status = got
    first_user = next((u for u, ids in enumerate(lists) if ids.numel()), None)
    assert (status[:, :, 0] == STATUS_MAGIC).all(), f"{tag}: a root core did not write its status page"
    assert (status[:, :, 1] == 0).all(), f"{tag}: a wait expired on a core: {status[:, :, 1].tolist()}"
    assert (status[:, :, 2] == 3 * cfg.users).all(), f"{tag}: receive slots published {status[:, :, 2].tolist()}"
    worst = {"o": 1.0, "ctx": 1.0}
    for u, ids in enumerate(lists):
        sel_rows = rows[cfg.slot(u, layer)][ids].float()
        q = q_lat[u].to(torch.bfloat16).float()
        ro, rm, rs = lo.partial(q, sel_rows, cfg)
        if ids.numel() == 0:
            assert torch.equal(o[u], torch.zeros_like(o[u])) and torch.equal(s[u], torch.zeros_like(s[u]))
            assert torch.allclose(m[u].double(), rm), f"{tag}: user {u} empty partial max {m[u][:4].tolist()}"
            continue
        if u == first_user and op is not None:
            tile = op.last_ms[u, 0]
            logger.info(
                f"HY4_MK3 attend {tag} user {u} head 0: reference max {rm[0].item():.5f} sum {rs[0].item():.5f}; raw "
                f"max/sum tile row 0, columns 0..7 and 16..19: {[round(v, 4) for v in tile[0, :8].tolist()]} "
                f"{[round(v, 4) for v in tile[0, 16:20].tolist()]}"
            )
        m_err = (m[u].double() - rm).abs().max().item()
        s_err = ((s[u].double() - rs).abs() / rs).max().item()
        o_pcc = pcc(o[u], ro)
        o_err = (o[u].double() - ro).abs().max().item()
        ctx = lo.merge([(o[u], m[u], s[u])], sink, cfg)
        ref = lo.reference_ctx(q, sel_rows, sink, cfg)
        c_pcc = pcc(ctx, ref)
        c_err = (ctx - ref).abs().max().item()
        worst["o"], worst["ctx"] = min(worst["o"], o_pcc), min(worst["ctx"], c_pcc)
        logger.info(
            f"HY4_MK3 attend {tag} user {u} rows {ids.numel()}: max err {m_err:.3e} sum rel err {s_err:.3e} "
            f"O pcc {o_pcc:.6f} max abs {o_err:.3e} | ctx (merged with the sink) pcc {c_pcc:.6f} max abs {c_err:.3e} "
            f"(ref max abs {ref.abs().max().item():.3e})"
        )
        assert o_pcc >= floor and c_pcc >= floor, f"{tag}: user {u} O pcc {o_pcc:.6f} ctx pcc {c_pcc:.6f} < {floor}"
    logger.info(f"HY4_MK3 attend {tag}: worst O pcc {worst['o']:.6f} worst ctx pcc {worst['ctx']:.6f}")


@pytest.mark.parametrize("fidelity", ["lofi"])
@pytest.mark.parametrize("counts", list(COUNTS))
def test_attend_random(device, counts, fidelity):
    cfg = lo.SparseMlaConfig(rows_per_slot=4096, num_layers=2)
    layer = 1
    slots = cfg.users * cfg.num_layers
    latent, rope = hs.random_latents(slots, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(device, cfg, slots, latent, rope)
    lists = [hs.spread_ids(cfg, n, seed=20 + u) for u, n in enumerate(COUNTS[counts])]
    gen = torch.Generator().manual_seed(7)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    op = SparseMla(cfg, device)
    got = run_attend(device, cfg, kv, rows, lists, q_lat, layer, FIDELITY[fidelity], op)
    if got is not None:
        check_partials(cfg, got, rows, lists, q_lat, sink, layer, f"random {counts} {fidelity}", floor=0.99, op=op)


# 16 token rows on the 8 S-blocks (PLAN.md of layer_attn, section 19): rows u and u + 8 queue on the same S-blocks
ROWS16 = {
    "all_64": [64] * 16,
    "live4": [64, 0, 0, 64, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0],  # 4 live rows, each alone on its S-block
    "pairs": [64, 0, 300, 1, 0, 128, 37, 2048, 200, 64, 0, 0, 5, 129, 2048, 1],  # every way two rows meet on a root
    "mixed": [1, 37, 512, 2048, 0, 64, 129, 300, 64, 0, 128, 1, 2048, 0, 37, 200],
}


@pytest.mark.parametrize("fidelity", ["lofi"])
@pytest.mark.parametrize("counts", list(ROWS16))
def test_attend_rows16(device, counts, fidelity):
    """16 users on one chip: every user's partial against the torch partial of its rows, as with 8."""
    cfg = lo.SparseMlaConfig(rows_per_slot=4096, num_layers=1, users=16)
    layer = 0
    latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(device, cfg, cfg.users, latent, rope)
    lists = [hs.spread_ids(cfg, n, seed=20 + u) for u, n in enumerate(ROWS16[counts])]
    gen = torch.Generator().manual_seed(7)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    op = SparseMla(cfg, device)
    got = run_attend(device, cfg, kv, rows, lists, q_lat, layer, FIDELITY[fidelity], op)
    if got is not None:
        check_partials(cfg, got, rows, lists, q_lat, sink, layer, f"rows16 {counts} {fidelity}", floor=0.99, op=op)


# what a selection that went wrong could leave in the count row: one above the capacity, the largest count the
# 16-bit copy for the compute kernel holds, the first it does not, and all ones; then four ordinary users
CLAMP_COUNTS = [2049, 65535, 65536, 0xFFFFFFFF, 64, 0, 300, 2048]


@pytest.mark.parametrize("fidelity", ["lofi"])
def test_count_clamp(device, fidelity):
    """Every kernel that reads the count row clamps it to topk (kernels/sm_dm.hpp): a table whose counts exceed the
    capacity ends like the table with those counts at 2048, bit for bit, with no wait expired; before the clamp a
    count of 65,536 or more gave the row reader and the compute kernel different numbers of chunks."""
    cfg = lo.SparseMlaConfig(rows_per_slot=4096, num_layers=2)
    layer = 1
    slots = cfg.users * cfg.num_layers
    latent, rope = hs.random_latents(slots, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(device, cfg, slots, latent, rope)
    lists = [hs.spread_ids(cfg, min(n, cfg.topk), seed=20 + u) for u, n in enumerate(CLAMP_COUNTS)]
    good = lo.selection_table(cfg, lists)
    bad = good.clone()
    over = torch.tensor(CLAMP_COUNTS, dtype=torch.int64)
    bad[0, : cfg.users] = torch.where(over >= 1 << 31, over - (1 << 32), over).to(torch.int32)
    gen = torch.Generator().manual_seed(7)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    op = SparseMla(cfg, device)
    q = hs.to_dev(op.queries_to_pages(q_lat), device, ttnn.bfloat16)
    results = {}
    for name, table in (("good", good), ("bad", bad), ("good again", good)):
        sel = hs.to_dev(table[None, None], device, ttnn.uint32)
        out = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.head_groups, op.out_page_values), device, ttnn.bfloat16)
        if hs.MOCK:
            ttnn.experimental.prepare_generic_op(
                [kv, sel, q, out] + op.io_extra(),
                op.attend_program(kv, sel, q, out, layer, fidelity=FIDELITY[fidelity]),
            )
            return
        back = ttnn.to_torch(sel).to(torch.int64).reshape(1 + cfg.users, cfg.topk)[0, : cfg.users] & 0xFFFFFFFF
        assert back.tolist() == (
            CLAMP_COUNTS if name == "bad" else [min(n, cfg.topk) for n in CLAMP_COUNTS]
        ), back.tolist()
        op.attend(kv, sel, q, out, layer, fidelity=FIDELITY[fidelity])
        results[name] = op.pages_to_partials(ttnn.to_torch(out))
        o, m, s_, status = results[name]
        assert (status[:, :, 0] == STATUS_MAGIC).all(), f"count row {name}: a root core did not write its status page"
        assert (status[:, :, 1] == 0).all(), f"count row {name}: a wait expired on a core: {status[:, :, 1].tolist()}"
        assert (
            status[:, :, 2] == 3 * cfg.users
        ).all(), f"count row {name}: receive slots published {status[:, :, 2].tolist()}"
        logger.info(
            f"HY4_MK3 count clamp, count row {name} {back.tolist()}: every root core wrote its status page, no wait expired"
        )
    check_partials(cfg, results["good"], rows, lists, q_lat, sink, layer, f"count clamp good {fidelity}", floor=0.99)
    for name in ("bad", "good again"):
        same = [torch.equal(a, b) for a, b in zip(results[name][:3], results["good"][:3])]
        logger.info(
            f"HY4_MK3 count clamp: O, max, sum of the run with the count row '{name}' bit-equal to the first run: {same}"
        )
        assert all(same), f"count row {name}: the partials differ from the run with the counts at {cfg.topk}"


def _checkpoint_reference(layer):
    """reference/attention_decode.py with layer ``layer``'s attention weights from the checkpoint."""
    from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention
    from models.experimental.hy4_preview.reference.attention_decode import Hy4AttentionDecode
    from models.experimental.hy4_preview.tests.common import hy4_tensors

    prefix = f"model.layers.{layer}.self_attn."
    names = [prefix + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES]
    attention = Hy4Attention(AttentionConfig()).eval()
    with torch.no_grad():
        for name, value in zip(Hy4Attention.PARAMETER_NAMES, hy4_tensors(names)):
            getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
    return Hy4AttentionDecode(attention, None)


@pytest.mark.parametrize("fidelity", ["lofi"])
@pytest.mark.parametrize("layer", [10])
def test_attend_checkpoint(device, layer, fidelity):
    """Cache rows and queries as the model makes them: the reference attention of a checkpoint layer applied to
    random pre-normed inputs, the latents packed by the prefill stack's pack, the queries absorbed by the
    reference (latent_queries). The device result, merged with the layer's sink, against absorbed_stages."""
    from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import KvStripe
    from models.experimental.hy4_preview.reference.attention import rope_cos_sin
    from models.experimental.hy4_preview.tt import frames

    cfg = lo.SparseMlaConfig(rows_per_slot=1024, num_layers=1)  # a 32768-token cache
    stripe, chip = KvStripe(sp=8, tp=4), 3
    with torch.no_grad():
        ref = _checkpoint_reference(layer)
        acfg = ref.cfg
        gen = torch.Generator().manual_seed(layer)
        positions = stripe.local_positions(chip, cfg.rows_per_slot)  # the positions this chip's rows hold
        cos, sin = rope_cos_sin(positions, acfg)
        latents, tokens, xs = [], [], []
        for u in range(cfg.users):
            x = torch.randn(cfg.rows_per_slot, acfg.hidden_size, generator=gen)
            _, k_pass, k_rope = ref.attention.latent(x, cos, sin)
            latents.append(torch.cat([k_pass, k_rope], dim=-1))
            xq = torch.randn(1, acfg.hidden_size, generator=gen)
            xs.append(xq)
            tokens.append(ref.new_token(xq, cfg.rows_per_slot * stripe.num_chips + 17 * u))
        held = torch.stack([frames.tail_to_device(l, acfg.qk_rope_head_dim) for l in latents])  # the cache's frame
        q_ref = torch.stack([ref.latent_queries(t) for t in tokens])  # [users, heads, 576], HF frame
        q_lat = frames.tail_to_device(q_ref, acfg.qk_rope_head_dim)
        sink = ref.attention.learnable_sink_param.detach().float()

    kv, rows = hs.make_storage(device, cfg, cfg.users, held[..., : cfg.latent], held[..., cfg.latent :])
    lists = [hs.spread_ids(cfg, min(n, cfg.rows_per_slot), seed=40 + u) for u, n in enumerate(COUNTS["mixed"])]
    op = SparseMla(cfg, device)
    got = run_attend(device, cfg, kv, rows, lists, q_lat, 0, FIDELITY[fidelity], op)
    if got is None:
        return
    check_partials(cfg, got, rows, lists, q_lat, sink, 0, f"checkpoint layer {layer} {fidelity}", floor=0.99, op=op)

    o, m, s, _ = got
    worst_stored, worst_exact = 1.0, 1.0
    with torch.no_grad():
        for u, ids in enumerate(lists):
            if ids.numel() == 0:
                continue
            ctx = lo.merge([(o[u], m[u], s[u])], sink, cfg)
            stored = frames.tail_from_device(rows[u][ids].float(), acfg.qk_rope_head_dim)  # what the chip read
            want = ref.absorbed_stages(xs[u], tokens[u], stored)["ctx"].double()
            exact = ref.absorbed_stages(xs[u], tokens[u], latents[u][ids])["ctx"].double()  # before FP8
            p_stored, p_exact = pcc(ctx, want), pcc(ctx, exact)
            worst_stored, worst_exact = min(worst_stored, p_stored), min(worst_exact, p_exact)
            logger.info(
                f"HY4_MK3 attend checkpoint layer {layer} {fidelity} user {u} rows {ids.numel()}: ctx against "
                f"absorbed_stages on the stored rows pcc {p_stored:.6f} max abs {(ctx - want).abs().max().item():.3e} "
                f"(ref max abs {want.abs().max().item():.3e}); on the unquantised latents pcc {p_exact:.6f}; "
                f"FP8 storage alone pcc {pcc(want, exact):.6f}"
            )
    logger.info(
        f"HY4_MK3 attend checkpoint layer {layer} {fidelity}: worst ctx pcc {worst_stored:.6f} against the reference "
        f"on the stored rows, {worst_exact:.6f} on the unquantised latents"
    )
    assert worst_stored >= 0.99


@pytest.mark.parametrize("fidelity", ["lofi"])
def test_attend_merge_across_chips(device, fidelity):
    """The partial contract end to end: a user's 2048 picks split over four "chips" (four launches on this chip,
    each with its own share of the rows; one chip gets none for some users), the four device partials and the
    sink merged on the host by log-sum-exp (layout.merge) against the reference core over all picks."""
    cfg = lo.SparseMlaConfig(rows_per_slot=4096, num_layers=1)
    latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
    kv, rows = hs.make_storage(device, cfg, cfg.users, latent, rope)
    gen = torch.Generator().manual_seed(9)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    picks = [hs.spread_ids(cfg, cfg.topk, seed=60 + u) for u in range(cfg.users)]
    # user u's picks: 5/8, 2/8, 1/8 on chips 0..2 in an order that depends on u, and nothing on one chip
    cuts = [0, 1280, 1792, 2048, 2048]
    op = SparseMla(cfg, device)
    partials = [[] for _ in range(cfg.users)]
    for chip in range(4):
        lists = []
        for u in range(cfg.users):
            k = (chip + u) % 4
            lists.append(picks[u][cuts[k] : cuts[k + 1]])
        got = run_attend(device, cfg, kv, rows, lists, q_lat, 0, FIDELITY[fidelity], op)
        if got is None:
            return
        o, m, s, status = got
        assert (status[:, :, 1] == 0).all()
        for u in range(cfg.users):
            partials[u].append((o[u], m[u], s[u]))
    worst = 1.0
    for u in range(cfg.users):
        q = q_lat[u].to(torch.bfloat16).float()
        ctx = lo.merge(partials[u], sink, cfg)
        ref = lo.reference_ctx(q, rows[u][picks[u]].float(), sink, cfg)
        score = pcc(ctx, ref)
        worst = min(worst, score)
        logger.info(
            f"HY4_MK3 attend merge across 4 chips {fidelity} user {u}: ctx pcc {score:.6f} max abs "
            f"{(ctx - ref).abs().max().item():.3e} (ref max abs {ref.abs().max().item():.3e})"
        )
    logger.info(f"HY4_MK3 attend merge across 4 chips {fidelity}: worst ctx pcc {worst:.6f}")
    assert worst >= 0.99
