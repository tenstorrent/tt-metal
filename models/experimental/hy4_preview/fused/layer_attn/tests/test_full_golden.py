# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The attention half of a full-indexer layer with CHECKPOINT weights at a context above the top-k, the selection
made on the chips with nothing from the host, against the reference's indexer (PLAN.md section 21.3):

    P1      a full layer's: the step's latent row and index key appended, the scan's query and multipliers written
    scan | merge (TopkMergeHub, rounds of 6, 5, 5 bits) | cut | P2

on the corrected goldens' 6-layer decode case of a 3,000-token prompt (6 decode steps at positions 2,999 .. 3,004;
the query at position p sees p + 1 keys and keeps 2,048 of them). Run with the pairing and the goldens together:

    HY4_ROPE_PAIRING=interleaved HY4_GOLDEN_DIR=generated/hy4_preview/golden_interleaved \\
        pytest <this file> -x -k L1          (HY4_MOCK=1 first)

One full layer a test (0, 1, 5), given the golden's own input of that layer (its streams): 8 token rows = the 6 decode
steps, and steps 0 and 1 once more on two more user slots. Every user's stores hold the golden's cache of that layer,
positions 0 .. 3,004 striped over the 32 chips as the model stripes them (128 positions a chip and turn): the latent
rows through the prefill stack's pack, the index keys through tt/frames.py index_to_device as BFP8 tiles; the row of
the user's OWN position is zero in both stores before the step (P1 writes it), rows of later positions are there and
must not be seen.
Checked (lines start with HY4_MK8A):
  words      every status and counter at rest, the merge's flags
  query      Scan.qt / the step pages the same on 32 chips; the device's query and head weights against the
             reference's (Hy4Indexer on the golden's q_resid and attn_in) in the device's frame
  selection  every chip's table against the host's cut of the chips' own survivors (exact ids); the picked POSITIONS
             of every token row against the reference's selection: the golden's recorded picks, and the reference
             indexer's scores recomputed on the golden's inputs and cached keys (reference/indexer.py
             compare_selection: differences among near-ties of the cut, and real ones); whether the row's own position
             is picked on both sides; the two repeated rows' picks equal to their originals'
  F          the sublayer output against the golden's attn_out (HF) per row
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as isc
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.layer_attn.tests.test_p1 import ALL, gain, galaxy, site_params
from models.experimental.hy4_preview.fused.merges import topk_merge as tm
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin, rope_pairing
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig, compare_selection
from models.experimental.hy4_preview.tests.common import pcc
from models.experimental.hy4_preview.tt import frames

FID = os.environ.get("HY4_ATTN_FIDELITY", "recommended")
LATENT_SCALE = float(os.environ.get("HY4_LATENT_SCALE", "1.0430"))  # (0: none)
PROMPT, LAYERS = 3000, 6
CHIPS = 32
KEY_ROWS = 4096  # stripe rows a slot of the key store (a stripe the scan's own tests run)
LAT_ROWS = 256  # stripe rows a slot of the latent store: one frame (3,005 positions put at most 128 on a chip)
BITS = (6, 5, 5)
STEP_OF = [0, 1, 2, 3, 4, 5, 0, 1]  # token row -> the golden's decode step


def chip_of(p):
    return (p // 128) % CHIPS


def row_of(p):
    return (p // 4096) * 128 + p % 128


def position_of(chip, row):
    return (row // 128 * CHIPS + chip) * 128 + row % 128


class GoldenCase:
    def __init__(self, mesh_device, layer):
        golden = hf_harness.find_decode_golden(LAYERS, PROMPT)
        if golden is None:
            pytest.skip(f"no {LAYERS}-layer decode golden of a {PROMPT}-token prompt under {hf_harness.GOLDEN_ROOT}")
        assert golden["meta"]["indexer_types"][layer] == "full", layer
        self.mesh, self.layer, self.meta = mesh_device, layer, golden["meta"]
        self.hf = hf = golden["layers"][layer]
        cache = golden["cache"][layer]
        users = L.ROWS
        self.positions = [golden["meta"]["positions"][j] for j in STEP_OF]
        self.contexts = [p + 1 for p in self.positions]
        attention, gamma, indexer = PP.checkpoint_layer(layer)
        self.lw = L.layer_weights(attention, None, indexer)
        self.indexer = Hy4Indexer(IndexerConfig()).eval().load_weights(indexer)
        self.keys_ref = cache["index_key"].float()  # [S, 128], HF frame
        S = self.keys_ref.shape[0]
        assert S > max(self.positions)
        # (the layer's L1 tensors first: the pack ops below leave the L1 allocator where a later arena does not fit)
        self.a = a = A.AttnLayer(
            mesh_device, fidelity=FID, full=True, append="truncate", index_query=True, latent_scale=LATENT_SCALE or None
        )
        a.set_site([site_params(layer)], 0, gamma)
        a.set_weights(self.lw)
        acfg = AttentionConfig()
        # the latent store: the golden's cache rows, the own position's zero
        self.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=LAT_ROWS, num_layers=1)
        held = frames.tail_to_device(cache["latent"].float(), acfg.qk_rope_head_dim)  # [S, 576], the cache's frame
        lat = torch.zeros(CHIPS, users, cfg.rows_per_slot, cfg.latent)
        rope = torch.zeros(CHIPS, users, cfg.rows_per_slot, cfg.rope)
        p = torch.arange(S)
        chips, rows = (p // 128) % CHIPS, (p // 4096) * 128 + p % 128
        lat[chips, :, rows] = held[:, None, : cfg.latent]
        rope[chips, :, rows] = held[:, None, cfg.latent :]
        for u, own in enumerate(self.positions):
            lat[chip_of(own), u, row_of(own)] = 0.0
            rope[chip_of(own), u, row_of(own)] = 0.0
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
        # the key store: the golden's index keys in the stored frame, the own position's zero
        dev_keys = frames.index_to_device(self.keys_ref, L.ROPE)  # [S, 128]
        shards = []
        for chip in range(CHIPS):
            mine = p[chips == chip]
            keys = torch.zeros(users, KEY_ROWS, isc.KEY_DIM)
            keys[:, rows[mine]] = dev_keys[mine][None]
            for u, own in enumerate(self.positions):
                if chip_of(own) == chip:
                    keys[u, row_of(own)] = 0.0
            shards.append(
                ttnn.from_torch(
                    keys.reshape(users, 1, KEY_ROWS, isc.KEY_DIM), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
                )
            )
        self.keys = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh_device.shape),
            mesh_device,
            memory_config=isc.key_memory_config(mesh_device),
        )
        self.scan_cfg = isc.ScanConfig(
            users=users, stripe_rows=KEY_ROWS, num_chips=CHIPS, chip_block=128, mode="tree", query_rows=True
        )
        self.scan = isc.Scan(mesh_device, self.scan_cfg, self.keys, list(range(users)), coords=list(ALL))
        self.scan.set_contexts(self.contexts)
        self.cut = isc.Cut(self.scan)
        self.hub = tm.TopkMergeHub(
            mesh_device, coords=list(ALL), users=users, bits=BITS, survivors=self.scan.out, cuts=self.cut.cuts
        )
        a.set_cache(cfg, self.kv, self.cut.table, 0, keep_out=True)
        a.set_keys(self.keys, 0, 1)
        a.set_scan(self.scan)
        a.set_sink(attention["learnable_sink_param"].detach().float().flatten())
        self.cos_hf, self.sin_hf = rope_cos_sin(torch.tensor(self.positions), acfg)
        cos, sin = frames.rope_tables(self.cos_hf, self.sin_hf)
        a.set_rope(cos, sin)
        a.set_positions(self.positions)
        a.set_streams(hf["streams_in"][STEP_OF].float())
        ttnn.synchronize_device(mesh_device)

    def step(self):
        self.a.run_p1()
        self.scan()
        self.hub()
        self.cut()
        self.a.run_p2()

    def compile(self):
        self.a.run_p1()
        self.scan.compile()
        self.hub.compile()
        self.cut.compile()
        self.a.run_p2()


def check_golden(case, label):
    a, scan, hf = case.a, case.scan, case.hf
    users = L.ROWS
    failed = []
    status = a.status()
    if status:
        failed.append(f"words not at rest: {status}")
    got = case.hub.read()
    if bool((got["flags"] & tm.FLAG_TIMEOUT).any()):
        failed.append("the merge: a wait ran out")
    # the query the scan read
    qts = [
        ttnn.to_torch(ttnn.get_device_tensors(scan.qt)[d]).float().reshape(users, isc.HEADS, isc.KEY_DIM)
        for d in range(CHIPS)
    ]
    pages = [
        (ttnn.to_torch(ttnn.get_device_tensors(scan.step)[d]).to(torch.int64) & 0xFFFFFFFF).reshape(8, 32)
        for d in range(CHIPS)
    ]
    same = sum(torch.equal(q, qts[0]) and torch.equal(p, pages[0]) for q, p in zip(qts, pages))
    idx = a.read_idx(ALL[0])
    dev_q, head_w = idx["landed"].permute(1, 0, 2).contiguous(), idx["head_w"]
    if same != CHIPS or not torch.equal(qts[0], dev_q):
        failed.append(
            f"Scan.qt: the same on {same} of {CHIPS} chips; equal to the assembly of the landed tiles: {torch.equal(qts[0], dev_q)}"
        )
    x_gold = hf["attn_in"][STEP_OF].float()
    q_resid = hf["q_resid"][STEP_OF].float()
    with torch.no_grad():
        ref_q = case.indexer.queries(q_resid, case.cos_hf, case.sin_hf)  # [8, 32, 128], HF frame
        ref_w = case.indexer.head_weights(x_gold)
    ref_q_dev = frames.index_to_device(ref_q, L.ROPE)
    rows_pcc = [pcc(ref_q_dev[u], dev_q[u]) for u in range(users)]
    x = a.read_root()["x"]
    logger.info(
        f"HY4_MK8A {label} [{FID}, pairing {rope_pairing()}]: status {status or 'at rest'}; merge flags "
        f"{sorted(set(got['flags'].flatten().tolist()))}; x vs the golden attn_in pcc {pcc(x_gold, x):.6f}; Scan.qt and the "
        f"step pages the same on {same} of {CHIPS} chips; the device's index query vs the reference's (Hy4Indexer.queries on "
        f"the golden's q_resid, index_to_device) pcc per token row {min(rows_pcc):.5f} to {max(rows_pcc):.5f}; head weights "
        f"vs the reference's pcc {pcc(ref_w.float(), head_w):.5f} gain {gain(ref_w, head_w):.4f}"
    )
    # the tables against the host's cut of the chips' own survivors
    survivors = [scan.read(d) for d in range(CHIPS)]
    words = torch.stack([sv["word"] for sv in survivors])
    gold = tm.golden(words, isc.TOPK)
    wrong, total = 0, torch.zeros(users, dtype=torch.int64)
    picked = [[] for _ in range(users)]
    for d in range(CHIPS):
        counts, rows = case.cut.read(d)
        total += counts
        for u in range(users):
            keep = int(gold["keep"][d, u])
            if int(counts[u]) != keep or not torch.equal(rows[u, :keep], survivors[d]["row"][u, :keep]):
                wrong += 1
            picked[u].append(position_of(d, rows[u, : int(counts[u])]))
    picked = [torch.cat(p) for p in picked]
    logger.info(
        f"HY4_MK8A {label}: tables differing from the host's cut of the chips' own survivors: {wrong} of {CHIPS * users}; "
        f"picks per token row {total.tolist()}"
    )
    if wrong or total.tolist() != [min(isc.TOPK, c) for c in case.contexts]:
        failed.append(f"{wrong} tables differ from the host's cut; picks {total.tolist()}")
    # the picked positions against the reference's selection
    recorded = hf["topk_indices"][STEP_OF].long()
    lines = []
    for u, p_u in enumerate(case.positions):
        n = p_u + 1
        if picked[u].numel() and (int(picked[u].max()) > p_u or picked[u].unique().numel() != picked[u].numel()):
            failed.append(
                f"token row {u}: a position beyond its own ({int(picked[u].max())} > {p_u}) or a position twice"
            )
            continue
        mine = torch.zeros(1, n, dtype=torch.bool)
        mine[0, picked[u]] = True
        with torch.no_grad():
            scores = case.indexer.scores(ref_q[u : u + 1], ref_w[u : u + 1], case.keys_ref[:n])  # [1, n]
        ref = torch.zeros(1, n, dtype=torch.bool)
        ref[0, scores[0].topk(isc.TOPK).indices] = True
        rec = torch.zeros(1, n, dtype=torch.bool)
        rec[0, recorded[u][recorded[u] <= p_u]] = True
        kth = float(scores[0].topk(isc.TOPK).values[-1])
        spread = float(scores[0].topk(isc.TOPK).values[0]) - kth
        # (the device ranks BF16 scores, not centred: a score of the cut's magnitude m has a step of m / 256 to m / 128)
        tol = max(abs(kth), spread) / 64
        c_ref = compare_selection(scores, ref, mine, isc.TOPK, tol)
        c_rec = compare_selection(scores, rec, mine, isc.TOPK, tol)
        c_self = compare_selection(scores, rec, ref, isc.TOPK, tol)
        shared_ref, shared_rec = int((ref & mine).sum()), int((rec & mine).sum())
        lines.append((shared_ref, shared_rec, c_ref["real"], c_rec["real"]))
        logger.info(
            f"HY4_MK8A {label} token row {u} (position {p_u}): {int(mine.sum())} picks; shared with the reference indexer's "
            f"top-{isc.TOPK} on the golden's inputs {shared_ref} (differences among near-ties of the cut {c_ref['near_tie']}, "
            f"real {c_ref['real']}; tolerance {tol:.3e} at a cut of {kth:+.4f}, {spread:.3f} below the best); shared with the "
            f"golden's recorded picks {shared_rec} (near-ties {c_rec['near_tie']}, real {c_rec['real']}); the recomputed "
            f"reference against the recorded picks: {c_self['missing'] + c_self['extra']} differences; own position picked: "
            f"device {bool(mine[0, p_u])}, reference {bool(ref[0, p_u])}, recorded {bool(rec[0, p_u])}"
        )
        if shared_ref < 0.9 * isc.TOPK:
            failed.append(f"token row {u}: only {shared_ref} picks shared with the reference")
    for u in (6, 7):
        if not torch.equal(picked[u].sort().values, picked[STEP_OF[u]].sort().values):
            failed.append(
                f"token row {u}: its picks differ from token row {STEP_OF[u]}'s (the same step on another slot)"
            )
    logger.info(
        f"HY4_MK8A {label}: picks shared with the reference indexer, per token row: {[l[0] for l in lines]} of {isc.TOPK}; with "
        f"the golden's recorded picks {[l[1] for l in lines]}; real differences {[l[2] for l in lines]} / {[l[3] for l in lines]}; "
        f"the repeated rows' picks equal their originals': {not any('another slot' in f for f in failed)}"
    )
    # F against HF
    f = a.read_root()["f"]
    gold_f = hf["attn_out"][STEP_OF].float()
    per_row = [pcc(gold_f[u], f[u]) for u in range(users)]
    logger.info(
        f"HY4_MK8A {label} [{FID}, latent_scale {LATENT_SCALE or None}]: F vs the golden attn_out (HF): pcc {pcc(gold_f, f):.6f} "
        f"(per token row {min(per_row):.6f} to {max(per_row):.6f}), gain {gain(gold_f, f):.4f}"
    )
    if pcc(gold_f, f) < 0.98:
        failed.append(f"F pcc {pcc(gold_f, f):.5f} against HF")
    assert not failed, f"{label}: " + "; ".join(failed[:8])


@galaxy
@pytest.mark.parametrize("layer", [0, 1, 5], ids=["L0", "L1", "L5"])
def test_golden(mesh_device, layer):
    case = GoldenCase(mesh_device, layer)
    if A.is_mock():
        case.compile()
        return
    case.step()
    ttnn.synchronize_device(mesh_device)
    check_golden(case, f"full golden P{PROMPT} layer {layer}")
    # the same step again, back to back: the stores now hold the own positions' rows the first run wrote
    case.step()
    ttnn.synchronize_device(mesh_device)
    check_golden(case, f"full golden P{PROMPT} layer {layer}, the step again")
