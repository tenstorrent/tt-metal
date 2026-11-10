# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: the 16-row verify pass (TargetVerifier) matches 16 plain decode steps (tokens, taps, KV rows)."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.muse_glimmer_30b.tt.attention import kv_page_table
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import DEFAULT_CKPT_DIR, MuseGlimmerModelArgs

CKPT = os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not available"
)

_mesh_params = mesh_test_params()

S = 100
ROWS = 16
TAP_LAYER = 1
KV_LAYERS = (0, 3)  # 0 is sliding, 3 is global


def _replicated(t, mesh, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_kv_rows(model, layer, table_row, positions):
    """-> {'k': [n_dev, P, kv_local, hd], 'v': ...} float32; each device holds its own KV head(s)."""
    bs = model.args.block_size
    out = {}
    for name, cache in zip(("k", "v"), model.kv_caches[layer]):
        devs = []
        for d in ttnn.get_device_tensors(cache):
            t = ttnn.to_torch(d).float()  # [n_blocks, kv_local, block, hd]
            devs.append(torch.stack([t[int(table_row[p // bs]), :, p % bs, :] for p in positions]))
        out[name] = torch.stack(devs)
    return out


def _zero_kv(model, layers):
    for li in layers:
        for cache in model.kv_caches[li]:
            h = ttnn.from_torch(
                torch.zeros(list(cache.shape)),
                dtype=cache.dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(h, cache)


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_matches_plain_decode(mesh_device, tp):
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier, set_greedy_sampling
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    assert [args.is_sliding(i) for i in range(4)] == [True, True, True, False]
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)
    sliding_row = kv_page_table(args, 1, sliding=True)[0]

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    positions = list(range(S, S + ROWS))
    set_greedy_sampling(model)

    def kv_rows():
        return {
            li: _read_kv_rows(model, li, sliding_row if args.is_sliding(li) else pt[0], positions) for li in KV_LAYERS
        }

    # ---- path A: 16 plain decode steps
    model.prefill_forward(prompt, 0, pt_tt)
    buf_a = _replicated(torch.zeros(1, 1, 32, args.hidden_size), sub, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    dev_in = model.allocate_decode_inputs(1, nb)
    arg_a, tap_a = [], []
    for i in range(ROWS):
        host = model.prepare_decode_inputs_host(torch.tensor([T[i]]), torch.tensor([S + i]), pt)
        model.copy_decode_inputs(host, dev_in)
        logits = model.decode_forward(*dev_in, taps={TAP_LAYER: buf_a})
        tok = model.sample_on_device(logits)
        ttnn.deallocate(logits)
        arg_a.append(int(model.tokens_to_torch(tok, 1)[0]))
        ttnn.deallocate(tok)
        tap_a.append(ttnn.to_torch(ttnn.get_device_tensors(buf_a)[0])[0, 0, 0].float())
    tap_a = torch.stack(tap_a)
    kv_a = kv_rows()

    # ---- path B: fresh prefill, one verify pass. Zero the compared caches first so a missed write shows up.
    _zero_kv(model, KV_LAYERS)
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt)
    arg_b = verifier.run(T, S)
    tap_b = ttnn.to_torch(ttnn.get_device_tensors(verifier.tap_list()[0])[0])[0, 0, :ROWS].float()
    kv_b = kv_rows()

    # ---- compare
    matches = 0
    for i in range(ROWS):
        if arg_a[i] == arg_b[i]:
            matches += 1
        else:
            logger.info(f"token mismatch row {i}: plain={arg_a[i]} verify={arg_b[i]}")
    logger.info(f"token agreement {matches}/{ROWS}")
    pccs = []
    for i in range(ROWS):
        ok, msg = comp_pcc(tap_a[i], tap_b[i], 0.99)
        logger.info(f"tap row {i}: {msg}")
        pccs.append(ok)
    kv_ok = []
    for li in KV_LAYERS:
        for name in ("k", "v"):
            ok, msg = comp_pcc(kv_a[li][name], kv_b[li][name], 0.999)
            logger.info(f"kv layer {li} {name}: {msg}")
            kv_ok.append(ok)
    assert matches >= 15, f"only {matches}/{ROWS} tokens agree"
    assert all(pccs), "tap PCC below 0.99 on some row"
    assert all(kv_ok), "KV PCC below 0.999"


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_topk(mesh_device, tp):
    """Sample mode returns the per-device-merged top-32 of the target logits for the 16 verify rows."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt)

    verifier.prepare_single(T, S)
    logits = model.decode_forward(*verifier.decode_inputs, verify=verifier, taps=verifier.taps)
    ref = model.logits_to_torch(logits, ROWS)
    ttnn.deallocate(logits)
    ref_vals, ref_ids = torch.topk(ref, 33)  # the 33rd value exposes a tie across the top-32 boundary
    ref_next = ref_vals[:, 32]
    ref_vals, ref_ids = ref_vals[:, :32], ref_ids[:, :32]

    vals, ids = verifier.run(T, S, mode="topk")
    greedy = verifier.run(T, S)
    assert vals.shape == (ROWS, 32) and ids.shape == (ROWS, 32)

    val_bad, id_bad, id_checked, top_bad, top_ties = 0, 0, 0, 0, 0
    for r in range(ROWS):
        if not torch.equal(vals[r].float(), ref_vals[r].float()):
            val_bad += 1
            logger.info(f"row {r}: values differ, max abs {(vals[r].float() - ref_vals[r].float()).abs().max()}")
        uniq = {v: c for v, c in zip(*[x.tolist() for x in torch.unique(ref_vals[r], return_counts=True)])}
        for j in range(32):
            if uniq[ref_vals[r, j].item()] == 1 and not (j == 31 and ref_vals[r, 31] == ref_next[r]):
                id_checked += 1
                if int(ids[r, j]) != int(ref_ids[r, j]):
                    id_bad += 1
                    logger.info(f"row {r} col {j}: id {int(ids[r, j])} != ref {int(ref_ids[r, j])}")
        if ref_vals[r, 0] == ref_vals[r, 1]:
            top_ties += 1
        elif int(ids[r, 0]) != int(greedy[r]):
            top_bad += 1
            logger.info(f"row {r}: top id {int(ids[r, 0])} != greedy {greedy[r]}")
    logger.info(
        f"rows with value mismatch {val_bad}/{ROWS}; unique-value id mismatches {id_bad}/{id_checked}; "
        f"top-vs-greedy mismatches {top_bad} (exact ties skipped: {top_ties})"
    )
    assert val_bad == 0, "top-32 values differ from the reference"
    assert id_bad == 0, "top-32 ids differ from the reference"
    assert top_bad == 0, "top-1 id differs from the greedy token"


@pytest.mark.timeout(1800)
@_mesh_params
def test_verify_logits(mesh_device, tp):
    """Logits mode returns the full target logits of the 16 verify rows, equal to the plain decode_forward logits."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=1, max_seq_len=8192)
    model = MuseGlimmerModel(args, sub)
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = _replicated(pt, sub, ttnn.int32)

    prompt = torch.randint(1000, 100000, (S,), generator=torch.Generator().manual_seed(0))
    T = torch.randint(1000, 100000, (ROWS,), generator=torch.Generator().manual_seed(1)).tolist()
    model.prefill_forward(prompt, 0, pt_tt)
    verifier = TargetVerifier(model, [TAP_LAYER], pt)

    verifier.prepare_single(T, S)
    logits = model.decode_forward(*verifier.decode_inputs, verify=verifier, taps=verifier.taps)
    ref = model.logits_to_torch(logits, ROWS)
    ttnn.deallocate(logits)

    lg = verifier.run(T, S, mode="logits")
    greedy = verifier.run(T, S)
    assert lg.shape == ref.shape == (ROWS, args.vocab_size)
    assert torch.equal(lg, ref), f"logits differ from the reference, max abs {(lg - ref).abs().max()}"
    am = lg.argmax(-1).tolist()
    bad, ties = 0, 0
    for r in range(ROWS):
        top = torch.topk(lg[r], 2).values
        if top[0] == top[1]:
            ties += 1
        elif am[r] != int(greedy[r]):
            bad += 1
            logger.info(f"row {r}: logits argmax {am[r]} != greedy {greedy[r]}")
    logger.info(f"logits-vs-greedy mismatches {bad} (exact ties skipped: {ties})")
    assert bad == 0, "logits argmax differs from the greedy token"


KV_BLOCK = 64
KV_HD = 128
KV_NKV = 1


def _kv_case(mesh, U, R, pt_rows, nblocks, seed):
    """Random bf16 cache pair and decode-heads K/V [1, 32, nkv, hd] (row u * R + p = user u, row p), a compact page table.
    -> (caches ttnn, host caches, heads ttnn, host heads, page_table ttnn)"""
    g = torch.Generator().manual_seed(seed)
    host_caches = [torch.randn(nblocks, KV_NKV, KV_BLOCK, KV_HD, generator=g).bfloat16() for _ in range(2)]
    host_heads = [torch.randn(1, 32, KV_NKV, KV_HD, generator=g).bfloat16() for _ in range(2)]
    caches = [_replicated(c, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT) for c in host_caches]
    heads = [_replicated(h, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT) for h in host_heads]
    return caches, host_caches, heads, host_heads, _replicated(pt_rows, mesh, ttnn.int32)


def _check_caches(caches, golden, what):
    for name, c, gold in zip("kv", caches, golden):
        for d, t in enumerate(ttnn.get_device_tensors(c)):
            got = ttnn.to_torch(t)
            bad = (got != gold).any(dim=-1).nonzero().tolist()  # [block, head, pos]
            assert (
                not bad
            ), f"{what}: {name} cache of device {d} differs from the golden at (block, head, pos) {bad[:40]}"


@pytest.mark.timeout(600)
@_mesh_params
def test_write_kv_rows_multi_user(mesh_device, tp):
    """write_kv_rows(batch=U): R write tensors [U], compact page table [U, nb], source row u * R + p = user u's row p.
    Equals the op golden for several users, partial / inactive users, a ring table and a tile-row crossing, and equals
    the default (batch-1) call for one user. write_kv_rows frees a DRAM heads input (to_memory_config aliases), so the
    test passes clones."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.gemma4.custom_ops.paged_update_rows import PagedUpdateRows
    from models.demos.muse_glimmer_30b.tt.dflash.kv_rows import write_kv_rows

    nb = 8

    def run(pos, pt, nblocks, seed, what):
        """pos: [U][R] cache positions (-1 skips)."""
        U, R = len(pos), len(pos[0])
        caches, host_caches, heads, host_heads, pt_tt = _kv_case(sub, U, R, pt, nblocks, seed)
        w = [
            _replicated(torch.tensor([pos[u][p] for u in range(U)], dtype=torch.int32), sub, ttnn.int32)
            for p in range(R)
        ]
        write_kv_rows(
            caches[0],
            caches[1],
            ttnn.clone(heads[0]),
            ttnn.clone(heads[1]),
            w,
            pt_tt,
            num_kv_heads=KV_NKV,
            block_size=KV_BLOCK,
            batch=U,
        )
        idx = torch.tensor(pos, dtype=torch.int64).T.contiguous()  # [R, U]
        gold = [
            PagedUpdateRows.golden(c, h.permute(0, 2, 1, 3), idx, pt, num_kv_heads=KV_NKV, block_size=KV_BLOCK)
            for c, h in zip(host_caches, host_heads)
        ]
        _check_caches(caches, gold, what)
        logger.info(f"{what}: ok")
        return caches, host_caches, heads, w, pt_tt

    def compact(U):
        return torch.stack([u * nb + torch.arange(nb) for u in range(U)]).to(torch.int32)

    # a. 2 users x 16 rows
    run([list(range(100, 116)), list(range(300, 316))], compact(2), 2 * nb, 1, "a: U=2 R=16")

    # b. 4 users x 8 rows, one user with valid=5
    pos_b = [
        [10 + i for i in range(8)],
        [200 + i for i in range(8)],
        [64 * 3 + 20 + i for i in range(8)],
        [500 + i for i in range(8)],
    ]
    pos_b[2] = pos_b[2][:5] + [-1] * 3
    run(pos_b, compact(4), 4 * nb, 2, "b: U=4 R=8 valid=5")

    # c. 8 users x 4 rows, some users inactive
    pos_c = [[37 * u + 5 + i for i in range(4)] for u in range(8)]
    for u in (1, 4, 7):
        pos_c[u] = [-1] * 4
    run(pos_c, compact(8), 8 * nb, 3, "c: U=8 R=4 inactive users")

    # d. ring table: row u = u * n + arange(nb) % n; positions wrap past the ring
    ring = 3
    ptr = torch.stack([u * ring + torch.arange(nb) % ring for u in range(2)]).to(torch.int32)
    run(
        [[3 * KV_BLOCK - 8 + i for i in range(16)], [5 * KV_BLOCK - 8 + i for i in range(16)]],
        ptr,
        2 * ring,
        4,
        "d: ring wrap",
    )

    # e. U=1, R=16 equals the batch-1 call (the old call) exactly
    pt1 = compact(1)
    pos_e = [3 * KV_BLOCK - 5 + i for i in range(16)]
    caches_n, host_caches, heads, w, pt_tt = run([pos_e], pt1, nb, 5, "e: U=1 R=16 vs golden")
    caches_o = [_replicated(c, sub, ttnn.bfloat16, ttnn.TILE_LAYOUT) for c in host_caches]
    w1 = [_replicated(torch.tensor([p], dtype=torch.int32), sub, ttnn.int32) for p in pos_e]
    write_kv_rows(
        caches_o[0],
        caches_o[1],
        ttnn.clone(heads[0]),
        ttnn.clone(heads[1]),
        w1,
        pt_tt,
        num_kv_heads=KV_NKV,
        block_size=KV_BLOCK,
    )
    for name, a, b in zip("kv", caches_n, caches_o):
        for d, (ta, tb) in enumerate(zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b))):
            assert torch.equal(ttnn.to_torch(ta), ttnn.to_torch(tb)), f"U=1 vs old call: {name} cache, device {d}"

    # f. positions crossing a 32-position tile-row boundary (28..43), second user crossing the block boundary too
    run([list(range(28, 44)), list(range(KV_BLOCK - 4, KV_BLOCK + 12))], compact(2), 2 * nb, 6, "f: tile-row crossing")


USERS = 8
PROMPT_LEN = [90 + 7 * u for u in range(USERS)]  # a different length per slot


@pytest.mark.timeout(3600)
@_mesh_params
def test_verify_multi_user(mesh_device, tp):
    """One multi-user verify pass (row groups of a RowLayout) gives, per group, the logits and argmax tokens of a
    single-user verify of that slot alone.

    Fairness: for each layout the 8 prompts are prefilled again (clean caches), then the multi-user pass runs FIRST.
    The single-user passes run after it; they rewrite the same positions with the same values (same tokens, same
    positions) and write only extra rows past the group's rows, which causal SDPA masks for the compared rows. The
    K/V rows of each group are also compared between the two passes."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.layout import RowLayout
    from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier, set_greedy_sampling
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sub, num_layers=4, max_batch_size=USERS, max_seq_len=8192)
    model = MuseGlimmerModel(args, sub)
    pt = kv_page_table(args, USERS)
    pt_tt = _replicated(pt, sub, ttnn.int32)
    sliding = kv_page_table(args, USERS, sliding=True)
    set_greedy_sampling(model)
    prompts = [
        torch.randint(1000, 100000, (PROMPT_LEN[u],), generator=torch.Generator().manual_seed(10 + u))
        for u in range(USERS)
    ]
    verifier = TargetVerifier(model, [TAP_LAYER], pt)

    def prefill_all():
        _zero_kv(model, range(4))
        for u in range(USERS):
            model.prefill_forward(prompts[u], u, pt_tt)

    def kv_of(slot, R):
        pos = list(range(PROMPT_LEN[slot], PROMPT_LEN[slot] + R))
        return {
            li: _read_kv_rows(model, li, sliding[slot] if args.is_sliding(li) else pt[slot], pos) for li in KV_LAYERS
        }

    def logits_of(layout, tokens, positions):
        verifier.set_layout(layout)
        verifier.prepare(tokens, positions)
        out = verifier.forward("logits")
        lg = verifier.read_logits(out)
        ttnn.deallocate(out)
        return lg

    layouts = [
        RowLayout(16, (5, 2)),
        RowLayout(8, (4, 0, 7, 3)),
        RowLayout(4, (6, 1, 3, 0, 7, 5, 2, None)),
    ]
    for li, lay in enumerate(layouts):
        R = lay.R
        gen = torch.Generator().manual_seed(100 + li)
        tokens, positions, group_tokens = [0] * 32, [-1] * 32, {}
        for g, slot in enumerate(lay.slots):
            if slot is None:
                continue
            group_tokens[slot] = torch.randint(1000, 100000, (R,), generator=gen).tolist()
            for p in range(R):
                tokens[g * R + p] = group_tokens[slot][p]
                positions[g * R + p] = PROMPT_LEN[slot] + p

        prefill_all()
        multi = logits_of(lay, tokens, positions)
        kv_multi = {slot: kv_of(slot, R) for slot in group_tokens}

        for g, slot in enumerate(lay.slots):
            if slot is None:
                continue
            tk = group_tokens[slot] + [group_tokens[slot][-1]] * (16 - R)
            single = logits_of(
                RowLayout(16, (slot,)),
                tk + [0] * 16,
                [PROMPT_LEN[slot] + i for i in range(16)] + [-1] * 16,
            )
            ref = single[:R]
            got = multi[g * R : (g + 1) * R]
            ok, msg = comp_pcc(ref, got, 0.999)
            logger.info(f"layout R={R} groups={lay.groups} group {g} slot {slot}: logits {msg}")
            assert ok, f"layout {lay}: group {g} slot {slot} logits PCC below 0.999"
            assert ref.argmax(-1).tolist() == got.argmax(-1).tolist(), f"layout {lay}: group {g} argmax differs"
            kv_single = kv_of(slot, R)
            for layer in KV_LAYERS:
                for name in ("k", "v"):
                    ok, msg = comp_pcc(kv_single[layer][name], kv_multi[slot][layer][name], 0.999)
                    assert ok, f"layout {lay}: slot {slot} layer {layer} {name} KV differs: {msg}"
    verifier.close()
