# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention half of a decoder layer on the 8x4 mesh (fused/layer_attn/program.py: P1 then P2) with checkpoint
weights, against the HF golden of a 128-token prompt (reference/hf_harness.py) and the reference decode step
(reference/attention_decode.py).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k "accuracy and L1"
    pytest <this file> -x -k accuracy          (HY4_ATTN_FIDELITY=custom_mm for the other setting)
    pytest <this file> -x -k latency

accuracy: 8 users = the golden prompt read up to positions 120 .. 127: user u's cache holds the reference's latent
rows of positions 0 .. P_u (its own new row included: the host appends it, the append kernel is not built), packed
by the prefill stack's own ops into its packed-row storage; the selection is every cached position (a context
shorter than the top-k), as sparse_mla's table; the rows are striped over the 32 chips in blocks of 4 positions here
(the model's 128 would put the whole 128-token prompt on one chip), so every chip holds up to 4 rows a user and the
cross-chip merge has 32 real partials. The golden's streams go into the root's site; P1 and P2 run; checked per
layer, both programs' words at rest (every number is logged; lines start with HY4_MK8A):
  x          the root's BF16 x against the golden's attn_in; bit-equal on every chip
  queries    every chip's absorbed queries of its 2 heads against the reference's latent_queries
  ctx        the merged attention output of all 64 heads in latent space (2 heads from each chip) against the
             reference's absorbed core on the STORED rows (what the chips read) and on the unquantised latents
  F          the sublayer output on the root's site cores against the reference on the stored rows, against the
             golden's attn_out (HF), and against exact arithmetic on the stored projection weights for the device's
             own ctx and gate (what the post group and the reduce add)
latency: P1 and P2 as ONE trace, 20 replays and one synchronize, 8 rows; random projection weights and rows; every
user's 2048 picks are spread at random over the 32 chips (about 64 rows per chip and user), ids spread over a slab of
1024 rows per user and chip; stage times from the cores' stamps. On a shared-indexer layer the context length enters
only through which rows are read, so this is the program of the AgentX contexts and of 8 x 1M alike.
"""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import (
    gain,
    galaxy,
    golden_layer,
    rel_rms,
    site_params,
    stored_share,
)
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p2 import Case2, stage_times, two_stage_sum
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import _checkpoint_reference
from models.experimental.hy4_preview.reference.attention import rope_cos_sin
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

FID = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
# "truncate" / "nearest": the new token's cache row is written by P1's append on its owner chip (PLAN.md section 17)
# with that rounding of the e4m3 pack; unset: by the host before the step, through the prefill stack's pack
APPEND = os.environ.get("HY4_ATTN_APPEND")
POSITIONS = list(range(120, 128))  # one per user: the golden prompt read up to that position
BLOCK = 4  # stripe block of this test: chip(p) = (p // 4) % 32, row(p) = p % 4 on that chip
CHIPS = 32


def chip_of(p):
    return (p // BLOCK) % CHIPS


def row_of(p):
    return (p // (BLOCK * CHIPS)) * BLOCK + p % BLOCK


class LayerCase:
    def __init__(self, mesh_device, layer, timing=False, latent_scale=None, host=None):
        self.mesh, self.layer = mesh_device, layer
        self.hf = hf = golden_layer(layer)
        self.ref = ref = _checkpoint_reference(layer)
        self.acfg = acfg = ref.cfg
        attention, gamma, _ = PP.checkpoint_layer(layer)
        self.gamma = gamma
        self.site = site_params(layer)
        self.lw = L.layer_weights(attention, None, None)  # a shared-indexer form of the layer: no index weights
        # (with the append the store is addressed through frame tables: a slot is a whole frame of 256 rows)
        self.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=256 if APPEND else 32, num_layers=1)
        users = len(POSITIONS)
        # the reference's cache rows and new tokens
        self.x_new, self.tokens, self.latents = [], [], []
        with torch.no_grad():
            for p in POSITIONS:
                x_all = hf["attn_in"][: p + 1].float()
                cache = ref.prefill_cache(x_all[:p])
                token = ref.new_token(x_all[p : p + 1], p)
                cache.append(token.latent)
                self.x_new.append(x_all[p : p + 1])
                self.tokens.append(token)
                self.latents.append(cache.latent.float())  # [p + 1, 576], HF frame
        # the packed storage of every chip, and its selection table (every cached position)
        lat = torch.zeros(CHIPS, users, cfg.rows_per_slot, cfg.latent)
        rope = torch.zeros(CHIPS, users, cfg.rows_per_slot, cfg.rope)
        tables = []
        for c in range(CHIPS):
            lists = []
            for u, p_u in enumerate(POSITIONS):
                mine = [p for p in range(p_u + 1) if chip_of(p) == c]
                held = frames.tail_to_device(self.latents[u][mine], acfg.qk_rope_head_dim)
                if APPEND and p_u in mine:  # the new token's own row: the device writes it in P1
                    held[mine.index(p_u)] = 0.0
                rows = torch.tensor([row_of(p) for p in mine], dtype=torch.long)
                lat[c, u, rows] = held[:, : cfg.latent]
                rope[c, u, rows] = held[:, cfg.latent :]
                lists.append(rows)
            tables.append(SL.selection_table(cfg, lists))
        self.per_chip_rows = [
            [len([p for p in range(p_u + 1) if chip_of(p) == c]) for p_u in POSITIONS] for c in range(CHIPS)
        ]
        # (the layer's L1 tensors first: the pack ops below leave the L1 allocator where a later arena does not fit)
        shared = {} if host is None else dict(host=host)
        if (
            host is None and os.environ.get("HY4_ATTN_SHARED") == "1"
        ):  # the 8-row programs on the 16-row layout (PLAN.md 19.11)
            self.host = A.AttnLayer(mesh_device, rows=16, fidelity="recommended")
            shared = dict(host=self.host)
        self.a = a = A.AttnLayer(
            mesh_device, fidelity=FID, timing=timing, append=APPEND, latent_scale=latent_scale, **shared
        )
        a.set_site([self.site], 0, gamma)
        a.set_weights(self.lw)
        mesh_bf16 = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_lat = mesh_bf16(lat.reshape(CHIPS * users, 1, cfg.rows_per_slot, cfg.latent))
        tt_rope = mesh_bf16(rope.reshape(CHIPS * users, 1, cfg.rows_per_slot, cfg.rope))
        fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(tt_lat, round_scale_to_power_of_two=True)
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(fp8, scales, tt_rope)
        self.kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh_device, cfg))
        for t in (tt_lat, tt_rope, fp8, scales, packed):
            ttnn.deallocate(t)
        shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
        self.sel = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh_device.shape), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        a.set_cache(cfg, self.kv, self.sel, 0, keep_out=True)
        if APPEND:  # the append's positions in the real stripe (128 positions per chip and turn) of this test's rows
            a.set_positions([chip_of(p) * 128 + row_of(p) for p in POSITIONS])
        self.sink = ref.attention.learnable_sink_param.detach().float().flatten()
        a.set_sink(self.sink)
        cos, sin = rope_cos_sin(torch.tensor(POSITIONS), acfg)
        self.cos, self.sin = frames.rope_tables(cos, sin)
        a.set_rope(self.cos, self.sin)
        self.streams = hf["streams_in"][POSITIONS].float()
        a.set_streams(self.streams)
        self._shares = {}
        ttnn.synchronize_device(mesh_device)

    def stored_latents(self):
        """Every user's cached rows as the chips hold them (the packed storage read back through the prefill stack's
        unpack), in position order, HF frame: [P_u + 1, 576] per user."""
        cfg = self.cfg
        physical = ttnn.to_torch(self.kv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
        rows = (
            reconstruct_scaled_fp8_kv_cache(physical, geometry)[:, 0]
            .float()
            .reshape(CHIPS, cfg.users, cfg.rows_per_slot, cfg.width)
        )
        out = []
        for u, p_u in enumerate(POSITIONS):
            held = torch.stack([rows[chip_of(p), u, row_of(p)] for p in range(p_u + 1)])
            out.append(frames.tail_from_device(held, self.acfg.qk_rope_head_dim))
        return out

    def share(self, chip):
        if chip not in self._shares:
            # (the shares as the streams hold them: with the read-side scalar folded in, if one is set)
            self._shares[chip] = stored_share(self.a, self.a.chip_share(self.a.scaled(self.lw), chip))
        return self._shares[chip]


def check_layer(case, label):
    a, ref, hf = case.a, case.ref, case.hf
    topo = a.topo
    failed = []
    assert a.streams_from is case.lw, "the layer's streams were packed from other weights than this check's reference"
    status = a.status()
    if status:
        failed.append(f"words not at rest: {status}")
    root = a.read_root()
    x = root["x"]
    gold_x = hf["attn_in"][POSITIONS].float()
    images = {c: a.read_arena(c) for c in topo.chips}
    p1 = {c: a.read_p1(c, images[c]) for c in topo.chips}
    p2 = {c: a.read_p2(c, images[c]) for c in topo.chips}
    # (P2 has run: the pre hub core is an attention core there and its partial took the first 17 tiles of the x
    # buffer's place; the rest of x is still as it arrived)
    keep = 17 * L.TILE
    x_ok = sum(torch.equal(v["x"][:, keep:], x[:, keep:]) for v in p1.values())
    logger.info(
        f"HY4_MK8A {label} x: vs the golden attn_in pcc={pcc(gold_x, x):.7f} max_abs={(gold_x - x).abs().max():.3e}; "
        f"columns {keep}.. bit-equal on {x_ok} of {len(p1)} chips"
    )
    if x_ok != len(p1):
        failed.append(f"x bit-equal on {x_ok} chips")
    stored = case.stored_latents()
    users = len(POSITIONS)
    with torch.no_grad():
        on_stored = [ref.absorbed_stages(case.x_new[u], case.tokens[u], stored[u]) for u in range(users)]
        exact = [ref.absorbed_stages(case.x_new[u], case.tokens[u], case.latents[u]) for u in range(users)]
    if APPEND:  # the rows P1 appended (the last of each user), as the store holds them, against the reference's
        new_dev = torch.stack([stored[u][-1] for u in range(users)])
        new_ref = torch.stack([case.latents[u][-1] for u in range(users)])
        kept = a.read_append(topo.chips[0])["rows"]  # BF16, device frame
        kept_ref = frames.tail_to_device(new_ref, case.acfg.qk_rope_head_dim)
        logger.info(
            f"HY4_MK8A {label} [append {APPEND}] the 8 new rows: BF16 rows before the cast vs the reference's latent pcc "
            f"{pcc(kept_ref, kept):.6f} gain {gain(kept_ref, kept):.4f} (k_pass gain {gain(kept_ref[:, :512], kept[:, :512]):.4f}); as "
            f"stored vs the reference: k_pass pcc {pcc(new_ref[:, :512], new_dev[:, :512]):.6f} gain "
            f"{gain(new_ref[:, :512], new_dev[:, :512]):.4f}, k_rope pcc {pcc(new_ref[:, 512:], new_dev[:, 512:]):.6f} gain "
            f"{gain(new_ref[:, 512:], new_dev[:, 512:]):.4f}"
        )
        if pcc(new_ref, new_dev) < 0.999:
            failed.append(f"the appended rows: pcc {pcc(new_ref, new_dev):.5f} against the reference")
    fp8 = min(pcc(stored[u], case.latents[u]) for u in range(users))
    fp8_gain = [gain(case.latents[u], stored[u]) for u in range(users)]
    # queries: [chip][2 heads][8 users][576] in the device frame
    q_ref = torch.stack(
        [frames.tail_to_device(on_stored[u]["q_lat"].float(), case.acfg.qk_rope_head_dim) for u in range(users)]
    )  # [u, 64, 576]
    q_pcc, q_gain = [], []
    ctx_dev = torch.zeros(users, 64, L.KV_LORA)
    gate_ok = []
    partial_from_ctx = {}
    for coord in topo.chips:
        chip = topo.chip(coord)
        heads = L.heads_of(chip)
        want = q_ref[:, heads].permute(1, 0, 2)
        q_pcc.append(pcc(want, p1[coord]["q_lat"]))
        q_gain.append(gain(want, p1[coord]["q_lat"]))
        ctx_dev[:, heads] = p2[coord]["ctx"].permute(1, 0, 2)
        gate_ok.append(torch.equal(p2[coord]["gate"], p1[coord]["gate"]))
        if any(p1[coord]["status"][k][1:6] != [0] * 5 for k in ("pre", "query")) or any(p2[coord]["status"][1:6]):
            failed.append(f"chip {coord}: a projection group reported a status")
        partial_from_ctx[coord], _ = L.post_golden(p2[coord]["ctx"], p2[coord]["gate"], case.share(chip))
    logger.info(
        f"HY4_MK8A {label} [{FID}] queries vs the reference's latent queries: pcc min {min(q_pcc):.6f} median "
        f"{statistics.median(q_pcc):.6f}, gain {min(q_gain):.4f} to {max(q_gain):.4f}; the gate reached the post hub on "
        f"{sum(gate_ok)} of {len(gate_ok)} chips"
    )
    ctx_stored = torch.stack([on_stored[u]["ctx"].float() for u in range(users)])
    ctx_exact = torch.stack([exact[u]["ctx"].float() for u in range(users)])
    per_user = [pcc(ctx_stored[u], ctx_dev[u]) for u in range(users)]
    logger.info(
        f"HY4_MK8A {label} [{FID}] ctx (64 heads x 512, merged over 32 chips and the sink): vs the reference on the "
        f"stored rows pcc {pcc(ctx_stored, ctx_dev):.6f} (per user min {min(per_user):.6f}), gain "
        f"{gain(ctx_stored, ctx_dev):.4f}, rel rms {rel_rms(ctx_stored, ctx_dev):.3e}; vs the reference on the "
        f"unquantised latents pcc {pcc(ctx_exact, ctx_dev):.6f}; FP8 storage alone (rows) pcc {fp8:.6f} gain {min(fp8_gain):.4f} to {max(fp8_gain):.4f}, (ctx) "
        f"{pcc(ctx_exact, ctx_stored):.6f} gain {gain(ctx_exact, ctx_stored):.4f}; rows per chip and user {min(min(r) for r in case.per_chip_rows)} to "
        f"{max(max(r) for r in case.per_chip_rows)}"
    )
    # where ctx's gain against the reference comes from: every chip's partial against the torch partial of the same
    # stored rows for the device's own queries (O, max, sum), the device's merge against the host merge of the
    # device partials, and the host merge of the torch partials against the reference
    cfg = case.cfg
    q_dev = torch.zeros(users, 64, L.LATENT)
    for coord in topo.chips:
        q_dev[:, L.heads_of(topo.chip(coord))] = p1[coord]["q_lat"].permute(1, 0, 2)
    sink = a.sink_peer.double() * cfg.scale
    o_ref, o_dev, s_ratio, m_err, n_ref, n_dev = [], [], [], [], [], []
    host_of_dev, host_of_torch = torch.zeros(users, 64, L.KV_LORA), torch.zeros(users, 64, L.KV_LORA)
    for u, p_u in enumerate(POSITIONS):
        held = frames.tail_to_device(stored[u], case.acfg.qk_rope_head_dim)  # [P_u + 1, 576], the cache's frame
        dev_parts, torch_parts = [], []
        for coord in topo.chips:
            c = topo.chip(coord)
            mine = [p for p in range(p_u + 1) if chip_of(p) == c]
            to, tm, ts = SL.partial(q_dev[u], held[mine], cfg)
            do, dm, ds = p2[coord]["o"][u].double(), p2[coord]["max"][u].double(), p2[coord]["sum"][u].double()
            dev_parts.append((do, dm, ds))
            torch_parts.append((to, tm, ts))
            if mine:
                o_ref.append(to), o_dev.append(do)
                n_ref.append(to / ts[:, None]), n_dev.append(do / ds[:, None])
                s_ratio.append((ds / ts).mean().item())
                m_err.append((dm - tm).abs().max().item())
        host_of_dev[u] = SL.merge(dev_parts, sink, cfg).float()
        host_of_torch[u] = SL.merge(torch_parts, sink, cfg).float()
    cat = lambda xs: torch.cat([x.flatten() for x in xs])
    logger.info(
        f"HY4_MK8A {label} [{FID}] where the gain is. One chip's partial against the torch partial of the same stored "
        f"rows and the device's queries ({len(o_ref)} chip-user partials): O gain {gain(cat(o_ref), cat(o_dev)):.4f} pcc "
        f"{pcc(cat(o_ref).float(), cat(o_dev).float()):.6f}; O / sum (the chip's own normalised output) gain "
        f"{gain(cat(n_ref), cat(n_dev)):.4f}; sum, device / torch: mean {statistics.mean(s_ratio):.4f} (min "
        f"{min(s_ratio):.4f}, max {max(s_ratio):.4f}); max: abs err at most {max(m_err):.3e}. The merges: device ctx "
        f"against the host merge of the DEVICE partials gain {gain(host_of_dev, ctx_dev):.4f} pcc "
        f"{pcc(host_of_dev, ctx_dev):.6f}. The host merge of the TORCH partials against the reference on the stored rows "
        f"gain {gain(ctx_stored, host_of_torch):.4f} pcc {pcc(ctx_stored, host_of_torch):.6f}; device ctx against that host "
        f"merge gain {gain(host_of_torch, ctx_dev):.4f}"
    )
    f = root["f"]
    out_stored = torch.cat([on_stored[u]["out"].float() for u in range(users)])
    out_exact = torch.cat([exact[u]["out"].float() for u in range(users)])
    gold = hf["attn_out"][POSITIONS].float()
    model = two_stage_sum(a, {c: p2[c]["out"] for c in topo.chips})
    wrong = int((model != f).sum())
    from_ctx = sum(partial_from_ctx.values())
    res = {}
    for name, want in (
        ("the reference on the stored rows", out_stored),
        ("the reference on the unquantised latents", out_exact),
        ("the golden attn_out (HF)", gold),
        ("exact arithmetic on the stored weights for the device's ctx and gate", from_ctx),
    ):
        res[name] = (pcc(want, f), gain(want, f), rel_rms(want, f))
        logger.info(
            f"HY4_MK8A {label} [{FID}] F vs {name}: pcc={res[name][0]:.6f} gain={res[name][1]:.4f} rel_rms={res[name][2]:.3e}"
        )
    logger.info(
        f"HY4_MK8A {label} [{FID}] F vs the two-stage sum of the 32 device partials: {wrong} of {f.numel()} elements "
        f"differ; the reference itself vs HF's attn_out pcc {pcc(gold, out_exact):.6f}"
    )
    if wrong:
        failed.append(f"F differs from the two-stage sum in {wrong} elements")
    if pcc(ctx_stored, ctx_dev) < 0.99:
        failed.append(f"ctx pcc {pcc(ctx_stored, ctx_dev):.5f}")
    if res["the reference on the stored rows"][0] < 0.98:
        failed.append(f"F pcc {res['the reference on the stored rows'][0]:.5f}")
    assert not failed, f"{label}: " + "; ".join(failed[:10])
    return res


@galaxy
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 4, 5], ids=[f"L{i}" for i in range(6)])
def test_accuracy(mesh_device, layer):
    case = LayerCase(mesh_device, layer)
    a = case.a
    try:
        a.run_p1()
        a.run_p2()
        if A.is_mock():
            return
        ttnn.synchronize_device(mesh_device)
        check_layer(case, f"layer {layer}")
    finally:
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
        a.release()


@galaxy
def test_shared_step(mesh_device):
    """The 8-row step (P1 + P2, layer 1) as a program set of a 16-row layer (PLAN.md 19.11), in turn with the 16-row
    P2, as serving switches sets from pass to pass: 8-row step | 16-row P2 with 9 live | 8-row step | 16-row P2 with
    16 live | 8-row step. No host write goes between the launches except the inputs the 16-row P2 has no P1 for yet
    (its queries and gates: an arena write; its mask); the words are never written, and they must be at rest after
    every launch. Then, in the same process, the same step on a layer of its own: the 8-row step's F must be the
    same bit for bit every time, on the shared layout and alone (the same kernels on the same data). (The shared
    pair is built first: it needs the L1 address space unfragmented.)"""
    from models.experimental.hy4_preview.fused.layer_attn.tests.test_p2 import Case2, check_p2

    layer = 1
    l1 = lambda: ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank
    rel = lambda x, y: float((x.double() - y.double()).pow(2).sum().sqrt() / y.double().pow(2).sum().sqrt())
    case16 = Case2(mesh_device, rows=16, mask=0x01FF, sink_args=True, p2_alone=True)
    with16 = l1()
    case8 = LayerCase(mesh_device, layer, host=case16.a)
    a8, a16 = case8.a, case16.a
    both = l1()
    shared_f = []
    for turn, mask in enumerate((0x01FF, 0xFFFF, None)):
        a8.run_p1()
        a8.run_p2()
        if not A.is_mock():
            ttnn.synchronize_device(mesh_device)
            status = a8.status()
            assert not status, f"turn {turn}: words not at rest after the 8-row step: {status}"
            shared_f.append(a8.read_root()["f"].clone())
            differ = int((shared_f[-1] != shared_f[0]).sum())
            logger.info(
                f"HY4_MK8A shared step turn {turn}: the 8-row step's F against turn 0's: {differ} of {shared_f[0].numel()} "
                f"elements differ, relative rms difference {rel(shared_f[-1], shared_f[0]):.3e}, largest {float((shared_f[-1] - shared_f[0]).abs().max()):.3e}"
            )
        if mask is None:
            break
        case16.mask = mask
        case16.launches = turn + 1
        case16.prefill()
        ttnn.synchronize_device(mesh_device)
        a16.run_p2()
        if not A.is_mock():
            ttnn.synchronize_device(mesh_device)
            check_p2(case16, "all", f"shared step turn {turn}: 16-row P2 mask {mask:#06x}")
    if not A.is_mock():
        check_layer(case8, f"shared step: layer {layer} on the shared layout, last turn")
    for case in (case8, case16):
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
    a8.release()
    a16.release()
    released = l1()

    case = LayerCase(mesh_device, layer)
    a = case.a
    logger.info(
        f"HY4_MK8A shared step: persistent L1 per core: {with16} B with the 16-row layer, {both} with the 8-row set "
        f"beside it, {released} after their release, {l1()} with the 8-row layer of its own"
    )
    own_f = []
    for run in range(3):  # (the step's own run-to-run difference: the merges take the partials in arrival order)
        a.run_p1()
        a.run_p2()
        if A.is_mock():
            break
        ttnn.synchronize_device(mesh_device)
        status = a.status()
        assert not status, f"own layer, run {run}: words not at rest: {status}"
        own_f.append(a.read_root()["f"].clone())
    if not A.is_mock():
        check_layer(case, f"shared step: layer {layer} on a layer of its own")
        own = max(rel(own_f[i], own_f[0]) for i in (1, 2))
        between = max(rel(f, own_f[0]) for f in shared_f)
        turns = max(rel(f, shared_f[0]) for f in shared_f[1:])
        logger.info(
            f"HY4_MK8A shared step: relative rms difference of F: between runs of the step on a layer of its own {own:.3e} "
            f"({[int((own_f[i] != own_f[0]).sum()) for i in (1, 2)]} elements differ); between the turns on the shared "
            f"layout {turns:.3e}; shared layout against own layer {between:.3e}"
        )
        # the step on the shared layout must be the step on its own layer up to what two runs of the latter differ by
        assert between <= max(4 * own, 1e-6) and turns <= max(4 * own, 1e-6), (own, turns, between)
    ttnn.deallocate(case.kv)
    ttnn.deallocate(case.sel)
    a.release()


LATENT_SCALE = 1.0430  # the read-side scalar for the truncating cast's bias (PLAN.md 12a, 17; the 78-layer control)


@galaxy
@pytest.mark.parametrize("layer", [0, 1, 2, 3, 4, 5], ids=[f"L{i}" for i in range(6)])
def test_latent_scale(mesh_device, layer):
    """The half without and with `latent_scale` in the same job: F against HF's attn_out. Also what the scaled
    weights' own storage does to the scalar: the streams hold stored(c w), compared with c w and with c stored(w)."""
    out = {}
    for scale in (None, LATENT_SCALE):
        case = LayerCase(mesh_device, layer, latent_scale=scale)
        a = case.a
        try:
            a.run_p1()
            a.run_p2()
            if A.is_mock():
                continue
            ttnn.synchronize_device(mesh_device)
            out[scale] = check_layer(case, f"layer {layer}{'' if scale is None else f' latent_scale {scale}'}")
            if scale is not None:
                w0, w1 = stored_share(a, a.chip_share(case.lw, 0)), case.share(0)
                rows = []
                for key in ("w_k", "w_v"):
                    if key not in w0:
                        continue
                    cat = lambda v: torch.cat([m.flatten() for m in v]) if isinstance(v, list) else v.flatten()
                    plain, folded = cat(w0[key]).double(), cat(w1[key]).double()
                    exact = cat(a.chip_share(case.lw, 0)[key]).double() * scale
                    rows.append(
                        f"{key}: stored(c w) against c w gain {gain(exact, folded):.5f} rel rms {rel_rms(exact, folded):.2e}; "
                        f"stored(w) against w rel rms {rel_rms(exact / scale, plain):.2e}; stored(c w) / stored(w) = "
                        f"{gain(plain, folded):.5f}"
                    )
                logger.info(
                    f"HY4_MK8A layer {layer} latent_scale {scale}: chip 0's shares as the streams hold them: {'; '.join(rows) or sorted(w0)}"
                )
        finally:
            ttnn.deallocate(case.kv)
            ttnn.deallocate(case.sel)
            a.release()
    if out:
        hf_name = "the golden attn_out (HF)"
        logger.info(
            f"HY4_MK8A layer {layer} latent_scale: F against HF's attn_out without the scalar pcc {out[None][hf_name][0]:.6f} gain "
            f"{out[None][hf_name][1]:.4f}; with {LATENT_SCALE}: pcc {out[LATENT_SCALE][hf_name][0]:.6f} gain {out[LATENT_SCALE][hf_name][1]:.4f}"
        )


def spread_case(mesh_device, **kw):
    """test_latency's case: random projection weights and rows, every user's 2,048 picks spread at random over the 32
    chips (Binomial(2048, 1 / 32) rows per chip and user), layer 1's site on the golden's streams."""
    case = Case2(mesh_device, counts=[64] * 8, keep_out=False, **kw)
    a = case.a
    g = torch.Generator().manual_seed(11)
    owner = torch.randint(0, CHIPS, (case.cfg.users, case.cfg.topk), generator=g)
    per_chip = [[int((owner[u] == c).sum()) for u in range(case.cfg.users)] for c in range(CHIPS)]
    tables_ = [
        SL.selection_table(case.cfg, [hs.spread_ids(case.cfg, n, seed=100 * c + u) for u, n in enumerate(per_chip[c])])
        for c in range(CHIPS)
    ]
    shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables_]
    ttnn.deallocate(case.sel)
    case.sel = ttnn.to_device(
        ttnn.from_host_shards(shards, mesh_device.shape), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a.set_cache(case.cfg, case.kv, case.sel, 0, keep_out=False)
    a.set_streams(golden_layer(1)["streams_in"][slice(0, 8)].float())
    from models.experimental.hy4_preview.fused.projections.tests.harness import tables

    _, _, cos, sin = tables()
    a.set_rope(cos, sin)
    ttnn.synchronize_device(mesh_device)
    return case


@galaxy
@pytest.mark.parametrize("order", ["arrival", "fixed"])
def test_replays(mesh_device, order):
    """Is a traced step the same from replay to replay (PLAN.md section 19.14)? P1 + P2 as ONE trace on 32 chips, 8
    rows, 2,048 picks a row spread over the chips, replayed REPLAYS times with a blocking replay and a read of the
    root's F after each: how many distinct F there are, which token rows differ and by how much. "arrival": X4's
    merge cores take their partners' partials as they land (the default); "fixed": in index order
    (AttnLayer(fixed_order=True)), where every replay must give the first one's F bit for bit; if one does not, the
    chips' ctx rows of that replay are compared with the first replay's to say where the order still enters."""
    replays = int(os.environ.get("HY4_ATTN_REPLAYS", "50"))
    case = spread_case(mesh_device, fixed_order=order == "fixed")
    a = case.a

    def step():
        a.run_p1()
        a.run_p2()

    step()
    if A.is_mock():
        return
    ttnn.synchronize_device(mesh_device)
    assert not a.status(), f"words not at rest after a step: {a.status()}"
    merge = [k for k in a._p2_chip((0, 1), "all")[0].kernels if str(k.kernel_source).endswith("attn_merge_dm.cpp")]
    defines = [tuple(d) for d in merge[0].defines if "FIXED" in str(d[0])]
    assert bool(defines) == (order == "fixed"), defines
    trace_id, _ = tracing.capture(mesh_device, step)
    results, first_ctx, where = [], None, None
    try:
        for i in range(replays):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            f = a.read_root()["f"].clone()
            results.append(f)
            if i == 0:
                first_ctx = {c: a.read_p2(c)["ctx"].clone() for c in a.topo.chips}
            elif where is None and order == "fixed" and not torch.equal(f, results[0]):
                ctx = {c: a.read_p2(c)["ctx"] for c in a.topo.chips}
                where = (i, sorted(c for c in a.topo.chips if not torch.equal(ctx[c], first_ctx[c])))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    status = a.status()
    distinct = []
    for f in results:
        if not any(torch.equal(f, d) for d in distinct):
            distinct.append(f)
    differ = [i for i, f in enumerate(results) if not torch.equal(f, results[0])]
    rows = sorted({int(u) for i in differ for u in (results[i] != results[0]).any(dim=1).nonzero().flatten()})
    worst = min((pcc(results[0][u], results[i][u]) for i in differ for u in rows), default=1.0)
    rel = max((rel_rms(results[0], results[i]) for i in differ), default=0.0)
    logger.info(
        f"HY4_MK8A layer replays [{FID}, {order} order, merge kernel defines {defines}]: {replays} replays of P1 + P2 as one "
        f"trace, 2,048 picks a row: {len(distinct)} distinct F; {len(differ)} replays differ from the first, on token rows "
        f"{rows}; worst row pcc against the first replay's {worst:.6f}, largest relative rms difference of F {rel:.3e}; words "
        f"after the replays: {status or 'at rest'}"
        + (
            f"; first differing replay {where[0]}: chips whose ctx rows differ from the first replay's: {where[1]}"
            if where
            else ""
        )
    )
    assert not status, f"words not at rest after the replays: {status}"
    if order == "fixed":
        assert len(distinct) == 1, f"{len(distinct)} distinct F in the fixed order: {where}"


@galaxy
def test_latency(mesh_device):
    """P1 + P2 as one trace on 32 chips, 8 rows: per call with 1 and 8 steps per trace, without and with stamps."""
    out = {}
    counts = [64] * 8
    for timing in (False, True):
        # (HY4_ATTN_APPEND: with P1's append of the 8 new rows, each on another owner chip)
        kw = dict(append=APPEND, append_keep=False) if APPEND else {}
        case = Case2(mesh_device, counts=counts, timing=timing, keep_out=False, **kw)
        a = case.a
        # every user's 2048 picks spread at random over the 32 chips: Binomial(2048, 1 / 32) rows per chip and user
        g = torch.Generator().manual_seed(11)
        owner = torch.randint(0, CHIPS, (case.cfg.users, case.cfg.topk), generator=g)
        per_chip = [[int((owner[u] == c).sum()) for u in range(case.cfg.users)] for c in range(CHIPS)]
        tables = [
            SL.selection_table(
                case.cfg, [hs.spread_ids(case.cfg, n, seed=100 * c + u) for u, n in enumerate(per_chip[c])]
            )
            for c in range(CHIPS)
        ]
        shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
        ttnn.deallocate(case.sel)
        case.sel = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh_device.shape), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        a.set_cache(case.cfg, case.kv, case.sel, 0, keep_out=False)
        if APPEND:
            a.set_positions([4 * u * 128 + 5 for u in range(case.cfg.users)])  # owners: chips 0, 4, .. 28, stripe row 5
        flat = [n for row in per_chip for n in row]
        logger.info(
            f"HY4_MK8A layer latency: selected rows per chip and user {min(flat)} to {max(flat)}, mean {sum(flat) / len(flat):.1f}"
        )
        hf = golden_layer(1)
        a.set_streams(hf["streams_in"][slice(0, 8)].float())
        from models.experimental.hy4_preview.fused.projections.tests.harness import tables

        _, _, cos, sin = tables()
        a.set_rope(cos, sin)
        ttnn.synchronize_device(mesh_device)

        def step():
            a.run_p1()
            a.run_p2()

        step()
        if A.is_mock():
            ttnn.deallocate(case.kv)
            ttnn.deallocate(case.sel)
            a.release()
            continue
        ttnn.synchronize_device(mesh_device)
        status = a.status()
        assert not status, f"words not at rest after a step: {status}"
        for calls in (1, 8):
            r = tracing.traced_ms(mesh_device, step, calls=calls)
            out[(timing, calls)] = r["replay_ms"] * 1e3
            logger.info(
                f"HY4_MK8A layer latency [{FID}] timing={timing} {calls} step(s) per trace: {r['replay_ms'] * 1e3:.1f} us per "
                f"step (rounds {[round(v * 1e3, 1) for v in r['rounds_ms']]}) = {8 / (78 * r['replay_ms'] * 1e-3):.0f} tokens/s "
                f"for 8 rows if all 78 layers were this half alone"
            )
        status = a.status()
        logger.info(f"HY4_MK8A layer latency [{FID}] timing={timing}: words after the traces: {status or 'at rest'}")
        assert not status, f"words not at rest after the traces: {status}"
        if timing:
            stage_times(case, f"layer latency [{FID}] P2")
        ttnn.deallocate(case.kv)
        ttnn.deallocate(case.sel)
        a.release()
    if out:
        logger.info(
            f"HY4_MK8A layer latency [{FID}{', append ' + APPEND if APPEND else ''}]: the attention half for 8 rows, traced: {out[(False, 8)]:.1f} us per step with 8 "
            f"steps per trace, {out[(False, 1)]:.1f} as a trace of its own (with stamps {out[(True, 8)]:.1f})"
        )
