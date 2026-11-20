# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Index-key scan + chip-local top-2048 (fused/indexer_scan/indexer_scan.py) on one chip.

On the mock Galaxy (HY4_MOCK=1) every case compiles its kernels and checks CB / L1 / kernel sizes; nothing runs.
On the board a case builds a key cache laid out as the prefill stack's (BFP8, ND-sharded, slot = user * 2 + 1 of
2 slots per user), reads the keys back (so the reference sees the BFP8-rounded values) and checks, per user:

* scores (mode "scores") against reference/indexer.py Hy4Indexer.scores on the same keys: max |error| relative to
  the largest sum of |terms| (the measure tests/test_indexer.py uses), and against exact arithmetic on the
  operands the device was given; every row the query may not see holds NEG;
* the top-2048 (mode "tree", and "local" per bank) against the top-k of the device's OWN scores, ordered as the
  fused words are (BF16 score, then stamp): the same rows, in the same order, zero differences;
* the top-2048 against the reference's selection on the reference's scores: disagreements only on positions whose
  reference score is within 2x the measured score error of the cut (reference/indexer.py compare_selection).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k test_compile
    pytest <this file> -k "test_scan and u1-r4096-random"     (smallest first)
    pytest <this file> -k test_scan
    pytest <this file> -k test_latency

Result lines start with HY4_SCAN; the latency table also goes to $HY4_SCAN_RESULTS (a JSON file).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as isc
from models.experimental.hy4_preview.reference.attention import rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig, compare_selection
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt import frames

HY4 = IndexerConfig()
TRACE_REGION_BYTES = 16 << 20
RESULTS = Path(os.environ.get("HY4_SCAN_RESULTS", "generated/hy4_preview/indexer_scan_latency.json"))
NUM_CHIPS, CHIP_BLOCK = 32, 128
CHIP = int(os.environ.get("HY4_SCAN_CHIP", "5"))  # the chip of the 32 whose stripe the single test chip holds
SLOTS_PER_USER = 2  # the cache holds 2 slots per user; the scanned layer is the second (slot = user * 2 + 1)
FIDELITY = os.environ.get("HY4_SCAN_FIDELITY", "hifi4")
# The largest score error allowed, as a share of the largest sum of |terms| (tests/test_indexer.py's budget).
SCORE_ERROR_BUDGET = 0.02

# Set when a program did not come back: the board then needs a reset, and nothing else may touch it. The file
# carries that across this piece's queued jobs.
BOARD = {"down": None}
BOARD_DOWN_FILE = Path(os.environ.get("HY4_SCAN_BOARD_DOWN", "generated/hy4_preview/indexer_scan_BOARD_DOWN"))


def board_only(fn):
    """Skip on the mock (nothing executes there) and after a program of this session timed out; record a timeout."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if isc.is_mock() and not os.environ.get("HY4_SCAN_DRY"):
            pytest.skip("nothing executes on the mock Galaxy")
        if not isc.is_mock() and BOARD_DOWN_FILE.is_file():
            BOARD["down"] = BOARD_DOWN_FILE.read_text()[:200]
        if not isc.is_mock() and BOARD_DOWN_FILE.with_name("indexer_scan_HOLD").is_file():
            pytest.skip("held: a queued job waits for a fix (indexer_scan_HOLD exists)")
        if BOARD["down"]:
            pytest.skip(f"an earlier program did not come back ({BOARD['down']}): no more device work")
        try:
            return fn(*args, **kwargs)
        except AssertionError:
            raise
        except Exception as e:
            # Anything that is not a failed check is treated as a program that did not come back (an operation
            # timeout raises "device is unrecoverable"): no later test of this piece touches the board.
            text = f"{type(e).__name__}: {e}"
            BOARD["down"] = text[:200]
            logger.info(f"HY4_SCAN BOARD DOWN: {text[:300]}")
            if not isc.is_mock():
                BOARD_DOWN_FILE.write_text(f"{fn.__name__}: {text[:300]}")
            raise

    return wrapper


one_chip = pytest.mark.parametrize(
    "device_params, mesh_device", [({"trace_region_size": TRACE_REGION_BYTES}, (1, 1))], indirect=True, ids=["1x1"]
)


# The AgentX decode batch (fused/README.md): 8 users, context at decode time, cache capacity 1M
# per user = stripes of 32,768 rows per chip of which 1,408 to 16,768 are visible.
AGENTX_CONTEXTS = [45609, 71284, 92682, 115666, 144011, 175480, 215110, 535958]
AGENTX_ROWS = 32768


def stripe_positions(rows):
    return isc.row_position(torch.arange(rows), CHIP, NUM_CHIPS, CHIP_BLOCK)


def contexts_for(users, rows, kind):
    """Context lengths (global positions) per user. "full": the whole stripe is visible. "ragged": each user ends
    somewhere else inside the stripe: mid-block, on a tile, in the first block, nothing on this chip, everything."""
    whole = int(stripe_positions(rows)[-1]) + 1
    if kind == "full":
        return [whole] * users
    if kind == "agentx":
        return AGENTX_CONTEXTS[:users]
    at_row = lambda r: int(stripe_positions(rows)[r]) + 1  # row r is the last visible one
    picks = [
        at_row(rows - 1 - 77),  # ends inside the last 32-row block of a bank, part of a block visible
        at_row(rows // 2 + 1023),  # ends on a block boundary
        at_row(2048 + 5) if rows > 4096 else at_row(300),  # little more than the top-k
        at_row(17),  # fewer rows than one block: one bank sees 18 rows, seven see nothing
        CHIP * CHIP_BLOCK,  # nothing on this chip yet
        whole,
        at_row(rows // 4 * 3 + 500),
        at_row(1100),  # fewer visible rows than the top-k
    ]
    return picks[:users]


def random_case(users, rows, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(users * SLOTS_PER_USER, rows, isc.KEY_DIM, generator=g).bfloat16().float()
    q = torch.randn(users, isc.HEADS, isc.KEY_DIM, generator=g).bfloat16().float()
    w = torch.randn(users, isc.HEADS, generator=g) * HY4.weight_scale
    return keys, q, w


def checkpoint_case(users, rows, seed, layer=1):
    """Keys, queries and head weights computed with the checkpoint's indexer weights of ``layer`` from random
    hidden states (as tests/test_indexer.py test_indexer_sc1_real_weights does), at the positions the chip's
    stripe holds, in the cache's frame."""
    cfg = HY4
    prefix = f"model.layers.{layer}.self_attn."
    names = list(Hy4Indexer.CHECKPOINT_NAMES.values())
    tensors = hy4_tensors(
        [prefix + "indexer." + n for n in names] + [prefix + "q_a_proj.weight", prefix + "q_a_layernorm.weight"]
    )
    ref = Hy4Indexer(cfg).eval().load_weights(dict(zip(names, tensors)))
    q_a_proj, q_a_norm = tensors[-2], tensors[-1]
    g = torch.Generator().manual_seed(seed)
    positions = stripe_positions(rows)
    cos, sin = rope_cos_sin(positions, cfg)
    rope = cfg.qk_rope_head_dim
    keys = torch.zeros(users * SLOTS_PER_USER, rows, isc.KEY_DIM)
    q, w = torch.zeros(users, isc.HEADS, isc.KEY_DIM), torch.zeros(users, isc.HEADS)
    with torch.no_grad():
        for user in range(users):
            for start in range(0, rows, 4096):
                x = torch.randn(min(4096, rows - start), cfg.hidden_size, generator=g).bfloat16().float()
                sl = slice(start, start + x.shape[0])
                keys[user * SLOTS_PER_USER + 1, sl] = frames.index_to_device(ref.keys(x, cos[sl], sin[sl]), rope)
            # the query: a new token at the end of the stripe
            x = torch.randn(1, cfg.hidden_size, generator=g).bfloat16().float()
            q_resid = rms_norm(torch.nn.functional.linear(x, q_a_proj), q_a_norm, 1e-6)
            q[user] = frames.index_to_device(ref.queries(q_resid, cos[-1:], sin[-1:]), rope)[0]
            w[user] = ref.head_weights(x)[0]
    # the other slot of every user holds other keys, so a wrong slot offset cannot pass
    keys[0::SLOTS_PER_USER] = torch.randn(users, rows, isc.KEY_DIM, generator=g)
    return keys.bfloat16().float(), q.bfloat16().float(), w


# ScanConfig fields applied to every scan a test builds, e.g. '{"selectors": "l1"}' (a JSON object)
OVERRIDES = json.loads(os.environ.get("HY4_SCAN_OVERRIDES", "{}"))


def build(mesh_device, users, rows, mode, keys_tensor, fidelity=FIDELITY, **kwargs):
    kwargs = {**OVERRIDES, **kwargs}
    fidelity = kwargs.pop("fidelity", fidelity)
    cfg = isc.ScanConfig(
        users=users,
        stripe_rows=rows,
        num_chips=NUM_CHIPS,
        chip_block=CHIP_BLOCK,
        mode=mode,
        fidelity=fidelity,
        **kwargs,
    )
    slots = [u * SLOTS_PER_USER + 1 for u in range(users)]
    return isc.Scan(mesh_device, cfg, keys_tensor, slots, chip_index=lambda coord: CHIP)


def held_keys(keys_tensor, users):
    """The keys the cache holds for the scanned slots (BFP8-rounded), [users, rows, 128]."""
    if DRY:
        return HOST_KEYS["keys"][1::SLOTS_PER_USER][:users]
    held = ttnn.to_torch(ttnn.get_device_tensors(keys_tensor)[0]).float()
    return held[1::SLOTS_PER_USER, 0][:users]


HOST_KEYS = {}


def fused_order_topk(scores, stamps, visible, k):
    """Rows of the k best visible entries as the device orders them: BF16 score, then the fused word."""
    bits = scores.bfloat16().view(torch.int16).to(torch.int64) & 0xFFFF
    words = ((bits << 16) | stamps).to(torch.int32).view(torch.float32)
    words = torch.where(visible, words.double(), torch.tensor(float("-inf"), dtype=torch.double))
    order = torch.argsort(words, descending=True, stable=True)
    return order[: min(k, int(visible.sum()))]


# ---------------------------------------------------------------------------------------------------------------
# A host model of what the kernels write, used only for a dry run of this file's own logic on the mock Galaxy
# (HY4_SCAN_DRY=1): scores in the layout "scores" stores them, survivors as raw tiles in rank layout.
# ---------------------------------------------------------------------------------------------------------------
DRY = bool(os.environ.get("HY4_SCAN_DRY"))
if DRY:  # the mock cannot capture a trace

    def _dry_traced_ms(mesh_device, fn, repeats=20, rounds=3, calls=1):
        fn()
        return {"replay_ms": 0.1, "rounds_ms": [0.1] * rounds}

    tracing.traced_ms = _dry_traced_ms
    tracing.idle_sync_ms = lambda mesh_device, repeats=20: 0.1


def model_scores_out(scan, exact, visible):
    cfg = scan.cfg
    rows = cfg.stripe_rows
    r = torch.arange(rows)
    bank, bank_block, pos = (r // 32) % isc.BANKS, r // 32 // isc.BANKS, r % 32
    tile, block = bank_block // 32, bank_block % 32
    out = torch.zeros(cfg.users, isc.BANKS, cfg.score_tiles, 32, 32)
    value = torch.where(visible, exact.float().bfloat16().float(), torch.tensor(isc.NEG))
    out[:, bank, tile, pos, block] = value
    return out.reshape(cfg.users * isc.BANKS, cfg.score_tiles, 32, 32)


def model_survivors_out(scan, scores, visible):
    cfg = scan.cfg
    rows = cfg.stripe_rows
    stamps = scan.row_stamp(torch.arange(rows))
    out = torch.zeros(cfg.users, isc.BANKS, 2, 32, 32, dtype=torch.int64)
    rr = torch.arange(32)[:, None].expand(32, 32).reshape(-1)
    cc = torch.arange(32)[None, :].expand(32, 32).reshape(-1)
    for u in range(cfg.users):
        for b in range(isc.BANKS):
            if cfg.mode == "tree" and b:
                continue
            pool = visible[u] if cfg.mode == "tree" else visible[u] & ((torch.arange(rows) // 32) % isc.BANKS == b)
            order = fused_order_topk(scores[u], stamps, pool, isc.TOPK)
            bits = scores[u][order].bfloat16().view(torch.int16).to(torch.int64) & 0xFFFF
            words = torch.full((isc.TOPK,), 0xFF80 << 16, dtype=torch.int64)
            words[: len(order)] = (bits << 16) | stamps[order]
            raw = torch.zeros(2048, dtype=torch.int64)
            raw[isc.rank_word(torch.arange(isc.TOPK))] = words
            for t in range(2):
                tile = torch.zeros(32, 32, dtype=torch.int64)
                tile[rr, cc] = raw[t * 1024 + isc.tile_word(rr, cc)]
                out[u, b, t] = tile
    out = torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)
    return out.reshape(cfg.users * isc.BANKS, 2, 32, 32)


CASES = {
    # id: (users, stripe rows, keys, contexts)
    "u1-r4096-random-full": (1, 4096, "random", "full"),
    "u1-r4096-random-ragged": (1, 4096, "random", "ragged"),
    "u1-r32768-random-full": (1, 32768, "random", "full"),
    "u1-r32768-random-ragged": (1, 32768, "random", "ragged"),
    "u8-r4096-random-ragged": (8, 4096, "random", "ragged"),
    "u8-r32768-random-full": (8, 32768, "random", "full"),
    "u8-r32768-random-ragged": (8, 32768, "random", "ragged"),
    "u8-r32768-random-agentx": (8, AGENTX_ROWS, "random", "agentx"),
    "u1-r32768-checkpoint-full": (1, 32768, "checkpoint", "full"),
    "u8-r4096-checkpoint-ragged": (8, 4096, "checkpoint", "ragged"),
}


@one_chip
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("mode", ["scores", "local", "tree"])
def test_compile(mesh_device, case, mode):
    """Kernels compile and the program fits (CBs, L1, kernel config) for every shape the board tests run."""
    users, rows, _, _ = CASES[case]
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"HY4_SCAN compile {case} {mode}: worker grid {grid.x}x{grid.y}")
    keys = isc.keys_to_device(mesh_device, torch.zeros(users * SLOTS_PER_USER, rows, isc.KEY_DIM))
    scan = build(mesh_device, users, rows, mode, keys)
    scan.compile()


def check_case(mesh_device, case, modes=("scores", "local", "tree"), centred=False):
    """``centred``: every user's scores are centred on (the BF16 value nearest to) the reference's cut."""
    users, rows, kind, ctx_kind = CASES[case]
    keys, q, w = (random_case if kind == "random" else checkpoint_case)(users, rows, seed=len(case) + rows)
    contexts = contexts_for(users, rows, ctx_kind)
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    HOST_KEYS["keys"] = keys
    held = held_keys(keys_tensor, users)
    qt, s = isc.device_query(q, w)
    visible = torch.stack(
        [torch.arange(rows) < isc.visible_rows(c, CHIP, NUM_CHIPS, CHIP_BLOCK, rows) for c in contexts]
    )
    logger.info(f"HY4_SCAN {case}: contexts {contexts} -> visible rows {visible.sum(-1).tolist()} of {rows}")

    ref = torch.stack([Hy4Indexer.scores(q[u : u + 1], w[u : u + 1], held[u])[0] for u in range(users)])
    size = torch.stack([Hy4Indexer.scores(q[u : u + 1], w[u : u + 1].abs(), held[u])[0] for u in range(users)])
    exact = torch.stack([isc.golden_scores(qt[u], s[u], held[u]) for u in range(users)])
    moved = (keys[1::SLOTS_PER_USER][:users] - held).abs().max().item()
    logger.info(f"HY4_SCAN {case}: BFP8 moves a key value by at most {moved:.3e}")

    # the reference's cut per user: its 2048th best visible score (0 when it sees fewer)
    cut = torch.zeros(users)
    for u in range(users):
        if int(visible[u].sum()) > isc.TOPK:
            cut[u] = ref[u][visible[u]].topk(isc.TOPK).values[-1]
    centres = cut.bfloat16().float() if centred else torch.zeros(users)
    if centred:
        case = case + "-centred"
        logger.info(f"HY4_SCAN {case}: centres {centres.tolist()}")

    out = {}
    scans = {}
    score_problem = None
    for mode in modes:
        scan = scans[mode] = build(mesh_device, users, rows, mode, keys_tensor)
        scan.set_query(qt, s)
        scan.set_contexts(contexts, centres)
        scan()
        ttnn.synchronize_device(mesh_device)
        if DRY:
            # a device with some error
            model = exact.float() + 1e-3 * torch.sin(torch.arange(rows) * 0.37) - centres[:, None]
            if mode == "scores":
                out[mode] = scan.decode_scores(model_scores_out(scan, model, visible))
            else:
                out[mode] = scan.decode(model_survivors_out(scan, out["scores"], visible))
        else:
            out[mode] = scan.read_scores() if mode == "scores" else scan.read()

    dev = out["scores"]  # the device's own scores: BF16(score - centre)
    uncentred = dev + centres[:, None]
    eps = eps_cut = 0.0
    if visible.any():
        eps = (uncentred - ref)[visible].abs().max().item()
        eps_exact = (uncentred.double() - exact)[visible].abs().max().item()
        near_cut = visible & ((ref - cut[:, None]).abs() < 0.1) & (visible.sum(-1, keepdim=True) > isc.TOPK)
        eps_cut = (uncentred - ref)[near_cut].abs().max().item() if near_cut.any() else 0.0
        scale = size[visible].max().item()
        spread = ref[visible].std().item()
        logger.info(
            f"HY4_SCAN {case} scores ({FIDELITY}): max|err| vs reference {eps:.3e} = {eps / scale:.2e} of the "
            f"largest sum of |terms| ({scale:.3f}); vs exact arithmetic on the device operands {eps_exact:.3e}; "
            f"pcc {pcc(ref[visible], uncentred[visible]):.7f}; score std {spread:.3f}; within 0.1 of the "
            f"reference's cut ({int(near_cut.sum())} rows) max|err| {eps_cut:.3e}"
        )
        score_problem = (
            None if eps <= SCORE_ERROR_BUDGET * scale else f"score error {eps:.3e} against terms of {scale:.3f}"
        )
        if score_problem:
            for u in range(users):
                n = int(visible[u].sum())
                logger.info(
                    f"HY4_SCAN {case} PROBLEM user {u} scores: device {dev[u, :6].tolist()} reference {ref[u, :6].tolist()} "
                    f"rows off by more than 5% of the scale: {int(((uncentred[u] - ref[u])[visible[u]].abs() > 0.05 * scale).sum())} of {n}"
                )
    # every row the query may not see, inside the score tiles that were produced, holds NEG
    for u in range(users):
        n = int(visible[u].sum())
        per_bank = torch.tensor([int(visible[u].reshape(-1, isc.BANKS, 32)[:, b].sum()) for b in range(isc.BANKS)])
        bank_of_row = (torch.arange(rows) // 32) % isc.BANKS
        index_in_bank = (torch.arange(rows) // 32 // isc.BANKS) * 32 + torch.arange(rows) % 32
        produced = index_in_bank < torch.clamp(-(-per_bank // 1024), min=1)[bank_of_row] * 1024
        masked = produced & ~visible[u]
        not_neg = int((dev[u][masked] != isc.NEG).sum())
        if not_neg:
            logger.info(
                f"HY4_SCAN {case} PROBLEM user {u}: {not_neg} of {int(masked.sum())} rows past the context ({n}) are not NEG: "
                f"{dev[u][masked][dev[u][masked] != isc.NEG][:8].tolist()}"
            )
            score_problem = score_problem or f"user {u}: rows past the context are not NEG"

    if "tree" not in out:
        assert score_problem is None, f"{case}: {score_problem}"
        return scans
    scan = scans["tree"]
    stamps = scan.row_stamp(torch.arange(rows))
    totals = {"own_diff": 0, "order_diff": 0, "local_diff": 0, "near_tie": 0, "real": 0, "picks": 0}
    problems = [score_problem] if score_problem else []
    for u in range(users):
        n = int(visible[u].sum())
        want = fused_order_topk(dev[u], stamps, visible[u], isc.TOPK)
        # the chip's top-2048: the same rows in the same order as the top-k of the device's own scores
        top = {k: v[u] for k, v in out["tree"].items()}
        got = top["row"][top["valid"]]
        if int(top["valid"].sum()) != min(n, isc.TOPK):
            problems.append(f"user {u}: {int(top['valid'].sum())} valid entries, {min(n, isc.TOPK)} expected")
        if not bool(top["valid"][: len(got)].all()):
            problems.append(f"user {u}: valid entries are not a prefix of the rank order")
        own_diff = len(set(got.tolist()) ^ set(want.tolist()))
        order_ok = got.tolist() == want.tolist()
        in_range = got < rows
        value_ok = bool(in_range.all()) and bool((top["score"][top["valid"]] == dev[u][got].bfloat16().float()).all())
        totals["own_diff"] += own_diff
        totals["order_diff"] += 0 if order_ok else 1
        if own_diff or not value_ok or not order_ok:
            sc = top["score"]
            problems.append(
                f"user {u}: {own_diff} rows differ from the device's own top-k (values match their rows: {value_ok}; "
                f"same order: {order_ok}; scores non-increasing in rank: {bool((sc[1:] <= sc[:-1]).all())}); "
                f"first entries got {list(zip(got[:6].tolist(), sc[:6].tolist()))} want "
                f"{list(zip(want[:6].tolist(), dev[u][want[:6]].tolist()))}; words {[hex(x) for x in top['word'][:6].tolist()]}"
            )
        # every bank's own top-2048
        if "local" in out:
            for b in range(isc.BANKS):
                in_bank = visible[u] & ((torch.arange(rows) // 32) % isc.BANKS == b)
                want_b = fused_order_topk(dev[u], stamps, in_bank, isc.TOPK)
                got_b = out["local"]["row"][u, b][out["local"]["valid"][u, b]]
                if got_b.tolist() != want_b.tolist():
                    totals["local_diff"] += 1
                    sc = out["local"]["score"][u, b]
                    same_set = set(got_b.tolist()) == set(want_b.tolist())
                    problems.append(
                        f"user {u} bank {b}: local top-k differs (same set: {same_set}; {len(got_b)} valid, "
                        f"{len(want_b)} expected; non-increasing: {bool((sc[1:] <= sc[:-1]).all())}); got "
                        f"{list(zip(got_b[:5].tolist(), sc[:5].tolist()))} want "
                        f"{list(zip(want_b[:5].tolist(), dev[u][want_b[:5]].tolist()))}; "
                        f"words {[hex(x) for x in out['local']['word'][u, b][:5].tolist()]}"
                    )
        # against the reference's selection
        if n and bool(in_range.all()):
            allowed = visible[u][None]
            masked_ref = ref[u][None].masked_fill(~allowed, float("-inf"))
            ref_sel = torch.zeros(1, rows, dtype=torch.bool)
            ref_sel[0, masked_ref.topk(min(isc.TOPK, rows), dim=-1).indices[0]] = True
            ref_sel &= allowed
            dev_sel = torch.zeros(1, rows, dtype=torch.bool)
            dev_sel[0, got] = True
            r = compare_selection(masked_ref, ref_sel, dev_sel, isc.TOPK, tol=2 * (eps_cut if centred else eps))
            totals["near_tie"] += r["near_tie"]
            totals["real"] += r["real"]
            totals["picks"] += int(ref_sel.sum())
            if r["real"] or r["size_diff"]:
                problems.append(f"user {u}: selection differs from the reference: {r}")
    logger.info(
        f"HY4_SCAN {case} top-2048: rows differing from the top-k of the device's own scores {totals['own_diff']} "
        f"(users with another order {totals['order_diff']}, banks whose local top-k differs {totals['local_diff']}); "
        f"vs the reference's selection: near-ties {totals['near_tie']} of {totals['picks']} picks "
        f"(within 2 x {eps_cut if centred else eps:.3e} of the cut), real errors {totals['real']}"
    )
    for line in problems[:24]:
        logger.info(f"HY4_SCAN {case} PROBLEM {line}")
    assert not problems, f"{case}: {len(problems)} problems, first: {problems[0]}"
    return scans


@one_chip
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@board_only
def test_scan(mesh_device, case):
    check_case(mesh_device, case)


@one_chip
@pytest.mark.parametrize("case", ["u1-r32768-random-full", "u8-r4096-random-ragged", "u1-r32768-checkpoint-full"])
@board_only
def test_centred(mesh_device, case):
    """The same checks with every user's scores centred on its cut: BF16 then resolves the order around the cut,
    and the selection's near-ties against the reference shrink to the arithmetic's own error."""
    check_case(mesh_device, case, modes=("scores", "tree"), centred=True)


@one_chip
@pytest.mark.parametrize("case", ["u1-r4096-random-ragged"])
@board_only
def test_scores_only(mesh_device, case):
    """The first thing to run on the board: scoring alone, one user, the smallest stripe."""
    check_scores_only(mesh_device, case)


def check_scores_only(mesh_device, case):
    users, rows, kind, ctx_kind = CASES[case]
    keys, q, w = random_case(users, rows, seed=3)
    contexts = contexts_for(users, rows, ctx_kind)
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    HOST_KEYS["keys"] = keys
    held = held_keys(keys_tensor, users)
    qt, s = isc.device_query(q, w)
    scan = build(mesh_device, users, rows, "scores", keys_tensor)
    scan.set_query(qt, s)
    scan.set_contexts(contexts)
    scan()
    ttnn.synchronize_device(mesh_device)
    dev = scan.read_scores()
    for u in range(users):
        n = isc.visible_rows(contexts[u], CHIP, NUM_CHIPS, CHIP_BLOCK, rows)
        ref = Hy4Indexer.scores(q[u : u + 1], w[u : u + 1], held[u])[0]
        err = (dev[u, :n] - ref[:n]).abs().max().item() if n else 0.0
        logger.info(
            f"HY4_SCAN scores-only {case} user {u}: {n} visible rows, max|err| {err:.3e}, "
            f"device {dev[u, :4].tolist()} reference {ref[:4].tolist()}"
        )
        assert err < 0.05, f"user {u}: scores are off by {err}"


@one_chip
@pytest.mark.parametrize("users, rows", [(1, 4096), (8, 32768)], ids=["u1-r4096", "u8-r32768"])
def test_compile_cut_and_ablations(mesh_device, users, rows):
    """The cut program and the measurement variants of the scan compile and fit."""
    keys = isc.keys_to_device(mesh_device, torch.zeros(users * SLOTS_PER_USER, rows, isc.KEY_DIM))
    scan = build(mesh_device, users, rows, "tree", keys)
    scan.compile()
    isc.Cut(scan).compile()
    for mode, ablate in (("scores", 2), ("tree", 1), ("tree", 3)):
        build(mesh_device, users, rows, mode, keys, ablate=ablate).compile()
    # the earlier forms of the reader: awaited reads, selectors from DRAM
    build(mesh_device, users, rows, "tree", keys, stream_depth=0, placement="grid", reader_noc=1).compile()
    build(mesh_device, users, rows, "tree", keys, stream_depth=0, selectors="dram").compile()
    build(mesh_device, users, rows, "tree", keys, selectors="dram").compile()
    for packet, depth in ((3, 4), (2, 6), (1, 8)):
        scan = build(
            mesh_device,
            users,
            rows,
            "tree",
            keys,
            selectors="l1",
            stream_depth=depth,
            blocks_per_read=packet,
            placement="rows",
            reader_noc=0,
        )
        logger.info(
            f"HY4_SCAN compile pipelined packet {packet} depth {depth}: key CB {scan.cfg.k_blocks} blocks; user 0 "
            f"cores {[scan.core_of(0, b) for b in range(isc.BANKS)]} vcs {[scan.vc_of(0, b) for b in range(isc.BANKS)]}"
        )
        scan.compile()
    for placement, reader_noc in (("bank_rows", 1), ("bank_rows", 0), ("grid", 0)):
        scan = build(mesh_device, users, rows, "tree", keys, placement=placement, reader_noc=reader_noc)
        logger.info(
            f"HY4_SCAN compile {placement} noc{reader_noc}: user 0 cores {[scan.core_of(0, b) for b in range(isc.BANKS)]}"
        )
        scan.compile()


def check_cut(mesh_device, scan, label):
    """Cuts by count, by threshold with every tie, by threshold with some ties, below everything, above
    everything: counts and row lists against the scan's own output read on the host."""
    users = scan.cfg.users
    top = scan.read()
    cut = isc.Cut(scan)
    g = torch.Generator().manual_seed(11)
    for round_ in range(3):
        words, want = [], []
        for u in range(users):
            score, row, valid = top["score"][u], top["row"][u], top["valid"][u]
            n = int(valid.sum())
            kind = (u + round_) % 5
            if kind == 0 or n == 0:  # the merge piece knows the count
                keep = int(torch.randint(0, n + 1, (1,), generator=g))
                words.append(isc.cut_by_count(keep))
            elif kind == 1:  # "how many of your scores are >= t", t one of the chip's own scores
                t = float(score[int(torch.randint(0, n, (1,), generator=g))])
                keep = int((score[:n] >= t).sum())
                words.append(isc.cut_by_threshold(t))
            elif kind == 2:  # some of the entries tied at t
                t = float(score[int(torch.randint(0, n, (1,), generator=g))])
                tied = int((score[:n] == t).sum())
                ties = int(torch.randint(0, tied + 1, (1,), generator=g))
                keep = int((score[:n] > t).sum()) + ties
                words.append(isc.cut_by_threshold(t, ties))
            elif kind == 3:  # a cut below every score of this chip: all it sees
                keep = n
                words.append(isc.cut_by_threshold(-(2.0**60)))
            else:  # a cut above every score of this chip: nothing
                keep = 0
                words.append(isc.cut_by_threshold(2.0**60))
            want.append(row[:keep])
        cut.set_cuts(words)
        cut()
        ttnn.synchronize_device(mesh_device)
        counts, rows_ = cut.read()
        for u in range(users):
            assert int(counts[u]) == len(
                want[u]
            ), f"{label} round {round_} user {u}: count {int(counts[u])} != {len(want[u])}"
            assert (
                rows_[u, : len(want[u])].tolist() == want[u].tolist()
            ), f"{label} round {round_} user {u}: rows differ"
        logger.info(f"HY4_SCAN cut {label} round {round_}: counts {counts.tolist()} and row lists match")
    return cut


@one_chip
@pytest.mark.parametrize("case", ["u8-r4096-random-ragged", "u1-r32768-random-ragged"])
@board_only
def test_cut(mesh_device, case):
    scans = check_case(mesh_device, case, modes=("scores", "tree"))
    check_cut(mesh_device, scans["tree"], case)


@one_chip
@board_only
def test_trace_follows_inputs(mesh_device):
    """One captured trace, replayed after the contexts and the query were rewritten in place: every replay's
    result is that of its own inputs (the context lengths and the query are device tensors, not program args)."""
    users, rows = 8, 4096
    keys, q, w = random_case(users, rows, seed=5)
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    scan = build(mesh_device, users, rows, "tree", keys_tensor)
    plain = build(mesh_device, users, rows, "tree", keys_tensor)
    qt, s = isc.device_query(q, w)
    scan.set_query(qt, s)
    scan.set_contexts(contexts_for(users, rows, "full"))
    scan()
    ttnn.synchronize_device(mesh_device)
    trace_id, _ = tracing.capture(mesh_device, lambda: scan())
    try:
        g = torch.Generator().manual_seed(9)
        for step, kind in enumerate(["ragged", "full", "ragged"]):
            contexts = contexts_for(users, rows, kind)
            q2 = torch.randn(users, isc.HEADS, isc.KEY_DIM, generator=g).bfloat16().float()
            qt2, s2 = isc.device_query(q2, w)
            for target in (scan, plain):
                target.set_query(qt2, s2)
                target.set_contexts(contexts)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            replayed = scan.read()
            plain()
            ttnn.synchronize_device(mesh_device)
            direct = plain.read()
            same = bool((replayed["word"] == direct["word"]).all())
            logger.info(f"HY4_SCAN trace step {step} ({kind}): replay equals the untraced run: {same}")
            assert same, f"step {step}: the replay did not follow its inputs"
    finally:
        ttnn.release_trace(mesh_device, trace_id)


LATENCY = [(1, 4096), (8, 4096), (1, 32768), (8, 32768)]
# what each measured variant leaves out, to tell the DRAM stream, the FPU and the top-k apart
VARIANTS = {
    "all": ("tree", 0),  # the piece: DRAM stream + scoring + top-2048 + tree
    "no_topk": ("scores", 0),  # DRAM stream + scoring (the score tiles are stored instead)
    "stream_only": ("scores", 2),  # DRAM stream alone, keys consumed unscored
    "no_stream": ("tree", 1),  # scoring + top-2048 + tree on keys that are not read
    "topk_only": ("tree", 3),  # top-2048 + tree alone
}


@one_chip
@board_only
def test_latency(mesh_device):
    """Traced latency (20 replays back to back, one synchronize), 1 and 8 users at 4,096 and 32,768 positions per
    chip, every user at the end of its stripe; the variants above; the cut program."""
    only = os.environ.get("HY4_SCAN_LATENCY")  # e.g. "8x32768" or "8x32768,1x4096"
    variants = os.environ.get("HY4_SCAN_VARIANTS", ",".join(VARIANTS)).split(",")
    # a JSON list of ScanConfig overrides, one measured configuration each, e.g.
    # [{}, {"fidelity": "lofi"}, {"placement": "bank_rows", "reader_noc": 0}]
    sweep = json.loads(os.environ.get("HY4_SCAN_SWEEP", "[{}]"))
    idle = tracing.idle_sync_ms(mesh_device)
    table = []
    for users, rows in LATENCY:
        if only and f"{users}x{rows}" not in only.split(","):
            continue
        keys, q, w = random_case(users, rows, seed=7)
        keys_tensor = isc.keys_to_device(mesh_device, keys)
        qt, s = isc.device_query(q, w)
        key_bytes = users * rows * 136
        for entry in sweep:
            # "_sizes" / "_variants" restrict a sweep entry to some sizes / variants; the rest are ScanConfig fields
            if "_sizes" in entry and f"{users}x{rows}" not in entry["_sizes"]:
                continue
            overrides = {k: v for k, v in entry.items() if not k.startswith("_")}
            row = {"users": users, "rows": rows, "key_bytes": key_bytes, "fidelity": FIDELITY, **OVERRIDES, **overrides}
            tree = None
            for name in entry.get("_variants", variants):
                mode, ablate = VARIANTS[name]
                scan = build(mesh_device, users, rows, mode, keys_tensor, **{"ablate": ablate, **overrides})
                scan.set_query(qt, s)
                scan.set_contexts(contexts_for(users, rows, "full"))
                t = tracing.traced_ms(mesh_device, lambda: scan())
                row[name + "_us"] = round(1e3 * t["replay_ms"], 1)
                row[name + "_rounds_us"] = [round(1e3 * x, 1) for x in t["rounds_ms"]]
                if name == "all":
                    tree = scan
            if tree is not None and not overrides and "cut" in os.environ.get("HY4_SCAN_EXTRAS", "cut"):
                cut = isc.Cut(tree)
                top = tree.read()
                cut.set_cuts([isc.cut_by_count(64)] * users)
                row["cut64_us"] = round(1e3 * tracing.traced_ms(mesh_device, lambda: cut())["replay_ms"], 1)
                cut.set_cuts([isc.cut_by_threshold(float(top["score"][u, 1023])) for u in range(users)])
                row["cut1024_us"] = round(1e3 * tracing.traced_ms(mesh_device, lambda: cut())["replay_ms"], 1)
            for name in ("all", "stream_only"):
                if name + "_us" in row:
                    row[name + "_gb_per_s"] = round(key_bytes / (row[name + "_us"] * 1e-6) / 1e9, 1)
            table.append(row)
            logger.info("HY4_SCAN latency " + json.dumps(row))
    logger.info(f"HY4_SCAN latency: idle synchronize {1e3 * idle:.1f} us")
    if table:
        previous = json.loads(RESULTS.read_text()) if RESULTS.is_file() else []
        RESULTS.write_text(json.dumps(previous + table, indent=1))


@one_chip
def test_probe_geometry(mesh_device):
    """Where the DRAM banks sit relative to the worker grid (logged, nothing asserted beyond the bank count)."""
    grid = mesh_device.compute_with_storage_grid_size()
    assert mesh_device.dram_grid_size().x == isc.BANKS
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        best = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(noc, ttnn.MeshCoordinate(0, 0))
        logger.info(
            f"HY4_SCAN probe {noc}: bank -> closest logical worker {[(b, (c.x, c.y)) for b, c in sorted(best.items())]}"
        )
    rows = []
    for y in range(grid.y):
        rows.append(
            [
                (lambda v: f"{v.x},{v.y}")(mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)))
                for x in range(grid.x)
            ]
        )
    logger.info(
        f"HY4_SCAN probe grid {grid.x}x{grid.y}; virtual coordinates of logical row 0: {rows[0]}; of column 0: {[r[0] for r in rows]}"
    )


@one_chip
@pytest.mark.parametrize("kind", ["random", "checkpoint"])
@board_only
def test_fidelity(mesh_device, kind):
    """Score error and selection near-ties at every math fidelity, one user, 32,768 positions, the same inputs."""
    users, rows = 1, 32768
    keys, q, w = (random_case if kind == "random" else checkpoint_case)(users, rows, seed=21)
    contexts = contexts_for(users, rows, "full")
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    HOST_KEYS["keys"] = keys
    held = held_keys(keys_tensor, users)
    ref = Hy4Indexer.scores(q[:1], w[:1], held[0])[0]
    scale = Hy4Indexer.scores(q[:1], w[:1].abs(), held[0])[0].max().item()
    masked_ref = ref[None]
    ref_sel = torch.zeros(1, rows, dtype=torch.bool)
    ref_sel[0, ref.topk(isc.TOPK).indices] = True
    for fold in (True, False):
        qt, s = isc.device_query(q, w, fold=fold)
        exact = isc.golden_scores(qt[0], s[0], held[0])
        for fidelity in ("lofi", "hifi2", "hifi3", "hifi4"):
            got = {}
            for mode in ("scores", "tree"):
                scan = build(mesh_device, users, rows, mode, keys_tensor, fidelity=fidelity)
                scan.set_query(qt, s)
                scan.set_contexts(contexts)
                scan()
                ttnn.synchronize_device(mesh_device)
                got[mode] = scan.read_scores() if mode == "scores" else scan.read()
            dev = got["scores"][0]
            eps = (dev - ref).abs().max().item()
            eps_exact = (dev.double() - exact).abs().max().item()
            rms = (dev - ref).pow(2).mean().sqrt().item()
            dev_sel = torch.zeros(1, rows, dtype=torch.bool)
            dev_sel[0, got["tree"]["row"][0][got["tree"]["valid"][0]]] = True
            r = compare_selection(masked_ref, ref_sel, dev_sel, isc.TOPK, tol=2 * eps)
            logger.info(
                f"HY4_SCAN fidelity {kind} fold={fold} {fidelity}: max|err| {eps:.3e} = {eps / scale:.2e} of the largest "
                f"sum of |terms| ({scale:.3f}), rms {rms:.3e}, vs exact arithmetic on the device operands "
                f"{eps_exact:.3e}; selection vs reference: missing {r['missing']} extra {r['extra']} near-ties "
                f"{r['near_tie']} real {r['real']}"
            )


def test_host_arithmetic():
    """The layout arithmetic the kernels and the host share, against tt/kv_layout.py and against itself. No device."""
    from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

    layout = BlockCyclicLayout(num_chips=NUM_CHIPS, block=CHIP_BLOCK)
    for rows in (4096, 32768):
        for chip in (0, 5, 31):
            positions = isc.row_position(torch.arange(rows), chip, NUM_CHIPS, CHIP_BLOCK)
            assert bool((layout.chip(positions) == chip).all()) and bool(
                (layout.slot(positions) == torch.arange(rows)).all()
            )
            for context in (
                0,
                1,
                chip * CHIP_BLOCK,
                chip * CHIP_BLOCK + 1,
                4095,
                4096,
                70001,
                rows * NUM_CHIPS - 77,
                rows * NUM_CHIPS,
            ):
                want = int((positions < context).sum())
                assert isc.visible_rows(context, chip, NUM_CHIPS, CHIP_BLOCK, rows) == want, (rows, chip, context)
                assert bool((positions[:want] < context).all()), "the visible rows are a prefix of the stripe"
    # a tile page <-> to_torch's logical tile
    tiles = torch.arange(2 * 1024).reshape(2, 32, 32)
    raw = isc.raw_words(tiles)
    r, c = torch.arange(32)[:, None].expand(32, 32), torch.arange(32)[None, :].expand(32, 32)
    assert bool((raw[0][isc.tile_word(r, c)] == tiles[0]).all())
    # rank order <-> word offset: a bijection on the survivor's 2048 words
    assert sorted(isc.rank_word(torch.arange(isc.TOPK)).tolist()) == list(range(isc.TOPK))

    for rows in (4096, 32768, 65536):
        cfg = isc.ScanConfig(users=1, stripe_rows=rows)
        scan = isc.Scan.__new__(isc.Scan)
        scan.cfg = cfg
        row = torch.arange(rows)
        stamp = scan.row_stamp(row)
        assert int(stamp.max()) < 1 << 16 and len(set(stamp.tolist())) == rows, "stamps are unique 16-bit values"
        assert bool((scan.stamp_row(stamp) == row).all())
        # a bank's stream order is the stamp order: [chunk | block | tile | position] grows with (tile, block, pos)
        bank0 = row[(row // 32) % isc.BANKS == 0]
        assert bool((scan.stamp_row(scan.row_stamp(bank0)) == bank0).all())
    # the cut words
    assert isc.cut_by_count(64) == (1 << 31) | 64
    assert isc.cut_by_threshold(1.5, 3) == (3 << 16) | 0x3FC0
    assert isc.cut_by_threshold(isc.NEG) & 0xFFFF == 0xF180


@one_chip
@pytest.mark.parametrize("chip", [0, 16], ids=["chip0", "chip16"])
@board_only
def test_agentx(mesh_device, chip, monkeypatch):
    """Scores, exact ids (per bank and per chip), the reference's selection and the cut at the AgentX shape, on
    the first chip of the 32 and on one in the middle (where one user's context ends inside a 32-row block)."""
    monkeypatch.setattr(sys.modules[__name__], "CHIP", chip)
    case = "u8-r32768-random-agentx"
    rows = [isc.visible_rows(c, chip, NUM_CHIPS, CHIP_BLOCK, AGENTX_ROWS) for c in AGENTX_CONTEXTS]
    logger.info(f"HY4_SCAN agentx chip {chip}: visible rows {rows}, {sum(rows)} of {8 * AGENTX_ROWS}")
    scans = check_case(mesh_device, case)
    check_cut(mesh_device, scans["tree"], f"{case} chip {chip}")


# name: (contexts by user, ScanConfig overrides, variants)
def agentx_configs():
    c = AGENTX_CONTEXTS
    every = list(VARIANTS)
    some = ["all", "stream_only", "no_stream"]
    return {
        "shape": (c, {}, every),  # the batch as it is: user u = the u-th shortest, the longest on the farthest cores
        "longest_first": (c[::-1], {}, some),  # the longest user on the cores nearest each bank's DRAM endpoint
        "grid": (c, {"placement": "grid"}, every),  # one 8 x 8 rectangle of cores: the cheapest launch
        "grid_longest_first": (c[::-1], {"placement": "grid"}, some),
        "grid_hifi3": (c, {"placement": "grid", "fidelity": "hifi3"}, ["all", "no_stream"]),
        "grid_long_only": ([0] * 7 + c[7:], {"placement": "grid"}, some),
        "long_only": ([0] * 7 + c[7:], {}, some),  # what the longest user costs with the other seven absent
        "long_only_first": (c[7:] + [0] * 7, {}, some),
        "short_only": (c[:7] + [0], {}, some),  # the seven short users with the longest absent
        "one_user_long": (c[7:], {"users": 1}, some),  # a one-user program for the longest user
        "hifi3": (c, {"fidelity": "hifi3"}, ["all", "no_stream"]),
    }


# the launch floor: every kernel returns at once
FLOORS = {
    "floor": {"ablate": 4},
    "floor_no_semaphores": {"ablate": 12},
    "floor_grid": {"ablate": 4, "placement": "grid"},
    "floor_1user_rows": {"ablate": 4, "users": 1},
    "floor_1user_grid": {"ablate": 4, "users": 1, "placement": "grid"},
}
AGENTX_CALLS = (1, 8)


@one_chip
@board_only
def test_agentx_latency(mesh_device, monkeypatch):
    """Traced latency at the AgentX shape (20 replays, one synchronize; 1 and 8 calls per trace), with the stream /
    scoring / top-k split, variations that show what the uneven stripes cost, and the program's launch floor."""
    chips = [int(x) for x in os.environ.get("HY4_SCAN_AGENTX_CHIPS", "0,16").split(",")]
    only = os.environ.get("HY4_SCAN_AGENTX_CONFIGS")
    keys, q, w = random_case(8, AGENTX_ROWS, seed=13)
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    qt, s = isc.device_query(q, w)
    table = []

    def measure(row, contexts, overrides, mode, ablate):
        users = overrides.get("users", 8)
        kwargs = {k: v for k, v in overrides.items() if k != "users"}
        kwargs["ablate"] = kwargs.get("ablate", 0) | ablate
        scan = build(mesh_device, users, AGENTX_ROWS, mode, keys_tensor, **kwargs)
        scan.set_query(qt[:users], s[:users])
        scan.set_contexts(contexts)
        return {
            calls: round(1e3 * tracing.traced_ms(mesh_device, lambda: scan(), calls=calls)["replay_ms"], 1)
            for calls in AGENTX_CALLS
        }

    for chip in chips:
        monkeypatch.setattr(sys.modules[__name__], "CHIP", chip)
        for name, (contexts, overrides, variants) in agentx_configs().items():
            if (only and name not in only.split(",")) or (chip != chips[0] and name != "shape"):
                continue
            rows = [isc.visible_rows(c, chip, NUM_CHIPS, CHIP_BLOCK, AGENTX_ROWS) for c in contexts]
            row = {"shape": "agentx", "config": name, "chip": chip, "visible_rows": rows, "key_bytes": sum(rows) * 136}
            for variant in variants if chip == chips[0] else ["all", "stream_only"]:
                mode, ablate = VARIANTS[variant]
                row[variant + "_us"] = measure(row, contexts, overrides, mode, ablate)
            for variant in ("all", "stream_only"):
                if variant + "_us" in row:
                    row[variant + "_gb_per_s"] = round(row["key_bytes"] / (row[variant + "_us"][1] * 1e-6) / 1e9, 1)
            table.append(row)
            logger.info("HY4_SCAN agentx latency " + json.dumps(row))
    monkeypatch.setattr(sys.modules[__name__], "CHIP", chips[0])
    for name, overrides in FLOORS.items():
        if only and "floor" not in only.split(","):
            continue
        users = overrides.get("users", 8)
        probe = build(
            mesh_device, users, AGENTX_ROWS, "tree", keys_tensor, **{k: v for k, v in overrides.items() if k != "users"}
        )
        row = {
            "shape": "agentx",
            "config": name,
            "users": users,
            "core_ranges": len(isc._core_set(probe.cores).ranges()),
        }
        del probe
        row["all_us"] = measure(row, AGENTX_CONTEXTS[-users:], overrides, "tree", 0)
        table.append(row)
        logger.info("HY4_SCAN agentx latency " + json.dumps(row))
    if table:
        previous = json.loads(RESULTS.read_text()) if RESULTS.is_file() else []
        RESULTS.write_text(json.dumps(previous + table, indent=1))


@one_chip
@pytest.mark.parametrize("case", ["u8-r4096-random-ragged", "u8-r32768-random-agentx", "u1-r32768-checkpoint-full"])
@board_only
def test_query_rows(mesh_device, case):
    """ScanConfig(query_rows=True): the query's tiles as [32 heads x 32 dims], transposed by the unpacker
    (scan_compute.cpp under HY4_SCAN_QT_ROWS; what a query assembled on the chips from per-head pieces is,
    layer_attn/PLAN.md section 21). The same inputs through both forms in one process: the score tiles (mode
    "scores": every BF16 score of the visible rows) and the chip's top-2048 (mode "tree": the fused words, score and
    stamp) must be bit-equal."""
    users, rows, kind, ctx_kind = CASES[case]
    keys, q, w = (random_case if kind == "random" else checkpoint_case)(users, rows, seed=len(case) + rows)
    contexts = contexts_for(users, rows, ctx_kind)
    keys_tensor = isc.keys_to_device(mesh_device, keys)
    visible = torch.stack(
        [torch.arange(rows) < isc.visible_rows(c, CHIP, NUM_CHIPS, CHIP_BLOCK, rows) for c in contexts]
    )
    qt, s = isc.device_query(q, w)
    got = {}
    for query_rows in (False, True):
        for mode in ("scores", "tree"):
            scan = build(mesh_device, users, rows, mode, keys_tensor, query_rows=query_rows)
            scan.set_query(qt, s)
            scan.set_contexts(contexts, torch.zeros(users))
            scan()
            ttnn.synchronize_device(mesh_device)
            got[(query_rows, mode)] = scan.read_scores() if mode == "scores" else scan.read()
    a, b = got[(False, "scores")], got[(True, "scores")]
    differ = int((a.bfloat16().view(torch.int16) != b.bfloat16().view(torch.int16))[visible].sum())
    words = int((got[(False, "tree")]["word"] != got[(True, "tree")]["word"]).sum())
    valid = int(got[(False, "tree")]["valid"].sum())
    logger.info(
        f"HY4_SCAN {case} query as rows against query as columns: {differ} of {int(visible.sum())} visible scores "
        f"differ in their BF16 bits (largest difference {float((a - b)[visible].abs().max()) if int(visible.sum()) else 0.0:.3e}); "
        f"{words} of {got[(False, 'tree')]['word'].numel()} words of the top-2048 differ ({valid} valid picks)"
    )
    assert differ == 0 and words == 0, (case, differ, words)
