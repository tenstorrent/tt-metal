# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the single-core kernels of the MTP step program (tt/kernels/step_accept.cpp, launched by
spec_accept.StepAccept) and of the chain program (tt/kernels/chain_inputs.cpp, launched by spec_chain.ChainInputs).

* Their arithmetic, tt/kernels/spec_step_words.hpp compiled with the host C++ compiler (tests/spec_step_words_driver.cpp),
  writes the words of the op graphs it replaced (spec_accept.graph and spec_chain.chain_graph, run in torch with the
  float32 rules of the device), and of the host path, over every acceptance count 0 to K at B = 1 and 8 (and other
  shapes): greedy and drawn rows, padding, declined and kept rows, M-RoPE deltas, RoPE rows clamped at the table end;
  the chain's ctrl pages of every count, with start blocks that change at a block boundary and held rows
  (GdnBlockState.ctrl_candidates); the sampler seeds of seeded, unseeded and non-drawing users
  (spec_sample.chained_sampling_plan); rows that stay and rows that continue another predecessor row (a slot_remap).
  Over any small-integer words (flags outside {0, 1}, counts outside [0, K], source rows outside [0, B)) the words
  still equal the graphs' after the float32 -> uint32 typecast rule (a negative value is 0).
* The kernels' L1 layouts equal the CB sizes the launchers reserve.
* The launchers run one generic_op on one core over the engine's own buffers (no ttnn op of the replaced graphs),
  pass the shapes, constants and buffer addresses the kernels read, stage the chain candidates as int32 words only
  when they change, and refuse a live tensor whose accessor differs from the stand-in the program was compiled on.
"""

import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks_host import NUM_BLOCKS
from models.demos.blackhole.qwen36.tests.test_spec_accept_host import ROWS, VOCAB, _lanes, _step_buffer
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _block_state, _chain_case, _ctrl_words
from models.demos.blackhole.qwen36.tt.kv_cache_spec import SPEC_STEP_HOLD
from models.demos.blackhole.qwen36.tt.spec_accept import (
    RS_CHUNK_MAX,
    SLOT,
    StepAccept,
    TorchOps,
    graph,
    greedy_counts,
    rs_chunk_bytes,
    step_accept_reference,
    step_kernel_l1_bytes,
)
from models.demos.blackhole.qwen36.tt.spec_chain import (
    CH_DELTA,
    CH_DRAWN,
    CH_P0_PREV,
    CH_SRC,
    CHAIN_OVERLAY_ROWS,
    ChainInputs,
    ChainTorchOps,
    candidate_words,
    chain_graph,
    chain_kernel_l1_bytes,
    chain_overlay,
    chain_reference,
)
from models.demos.blackhole.qwen36.tt.spec_sample import NO_CORE, chained_sampling_plan, sampling_plan
from models.demos.blackhole.qwen36.tt.spec_step import (
    STEP_DECLINED,
    STEP_DRAWN,
    STEP_KEEP,
    STEP_NUM_VALID,
    STEP_P0,
    STEP_ROPE_BASE,
    VERIFY_STEP_ROWS,
    VERIFY_TOKENS,
    draft_anchor,
    drafter_pack_rows,
    pack_width,
    rope_rows,
)

_TESTS = Path(__file__).resolve().parent
_KERNELS = _TESTS.parent / "tt" / "kernels"
_REPO = _TESTS.parents[4]
USER_ROWS = list(range(STEP_NUM_VALID, VERIFY_STEP_ROWS))


class _Words:
    """The compiled host driver of spec_step_words.hpp: one line of input words per case, one line of output words."""

    def __init__(self, exe):
        self.exe = exe

    def run(self, lines):
        out = subprocess.run([str(self.exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
        return [[int(w) for w in line.split()] for line in out.stdout.splitlines()]


@pytest.fixture(scope="module")
def words(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no host C++ compiler for tt/kernels/spec_step_words.hpp")
    exe = tmp_path_factory.mktemp("spec_step_words") / "driver"
    cmd = [cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{_REPO / 'tt_metal/hw/inc'}", f"-I{_KERNELS}"]
    subprocess.run([*cmd, str(_TESTS / "spec_step_words_driver.cpp"), "-o", str(exe)], check=True)
    return _Words(exe)


def _line(mode, *parts):
    out = [mode]
    for p in parts:
        out += [str(int(v)) for v in torch.as_tensor(p).reshape(-1).tolist()]
    return " ".join(out)


def _as_u32(x):
    """ttnn.typecast float32 -> uint32 of integer values below 2**24: a negative value is 0."""
    return torch.as_tensor(x).to(torch.int64).clamp(min=0)


# ------------------------------------------------------------------------------------------------ step program


def _accept_line(ids, draws, step, B, K, W, top):
    R = B * (K + 1)
    head = [B, K, W, top, int(draws is not None)]
    parts = [torch.as_tensor(ids).reshape(-1)[:R]]
    if draws is not None:
        parts.append(torch.as_tensor(draws).reshape(-1)[:R])
    parts += [step[VERIFY_TOKENS, :R], step[STEP_NUM_VALID:VERIFY_STEP_ROWS, :B]]
    return _line("accept", head, *parts)


def _accept_words(words, cases, B, K, W):
    """[(pack int64 [2K + 5, W], real bool [R])] of each (ids, draws, step, top) case."""
    R = B * (K + 1)
    out = words.run([_accept_line(ids, draws, step, B, K, W, top) for ids, draws, step, top in cases])
    n = drafter_pack_rows(K) * W
    return [(torch.tensor(o[:n]).reshape(-1, W), torch.tensor(o[n : n + R]) == 1) for o in out]


def _accept_case(rng, B, K, sampled, table_rows=ROWS):
    """A verify whose users take random counts 0 .. K: (ids [B, T], draws [B, T] or None, tokens [B, T], acceptance
    rows, counts). A drawn user walks its draws. Padding rows, declined and kept rows, M-RoPE deltas, and column-0
    positions near the end of the RoPE table (clamped RoPE rows) occur."""
    T = K + 1
    ids = torch.tensor([[rng.randrange(VOCAB) for _ in range(T)] for _ in range(B)])
    draws = torch.tensor([[rng.randrange(VOCAB) for _ in range(T)] for _ in range(B)]) if sampled else None
    tokens = torch.tensor([[rng.randrange(VOCAB) for _ in range(T)] for _ in range(B)])
    acc = {r: [0] * B for r in USER_ROWS}
    counts = [0] * B
    for u in range(B):
        if B > 1 and rng.random() < 0.15:  # a padding row
            acc[STEP_DECLINED][u] = 1
            continue
        drawn = int(sampled and rng.random() < 0.5)
        m = rng.randrange(T)
        nv = rng.randrange(m, T)
        walked = draws[u] if drawn else ids[u]
        tokens[u, 1 : m + 1] = walked[:m]
        if m < nv:
            tokens[u, m + 1] = (int(walked[m]) + 1) % VOCAB
        if rng.random() < 0.3:
            tokens[u, nv + 1 :] = walked[nv : T - 1]  # columns past the valid drafts that would match
        p0 = rng.choice([rng.randrange(40, table_rows - 3 * T), table_rows - rng.randrange(1, 2 * T)])
        delta = rng.choice([0, -rng.randrange(1, 30), rng.randrange(1, 5)])
        declined = rng.random() < 0.2
        acc[STEP_NUM_VALID][u], acc[STEP_DRAWN][u], acc[STEP_P0][u] = nv, drawn, p0
        acc[STEP_ROPE_BASE][u] = int(rope_rows([p0], delta, table_rows)[0])
        acc[STEP_DECLINED][u], acc[STEP_KEEP][u] = int(declined), int(declined and rng.random() < 0.5)
        counts[u] = m
    return ids, draws, tokens, acc, counts


@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("B,K", [(1, 3), (8, 3), (2, 3), (4, 3), (1, 1), (4, 7), (2, 11), (16, 1)])
def test_the_step_kernel_writes_the_words_of_the_op_graph(words, B, K, sampled):
    """accept_words == spec_accept.graph (the replaced op graph in torch) == step_accept_reference (the plugin's walk
    and the host's drafter pack and reseed table), over every count 0 .. K."""
    T, R, W, nb, scratch = K + 1, B * (K + 1), pack_width(B * (K + 1)), 7, 999
    rng = random.Random(1000 * B + 10 * K + sampled)
    cases, wants, seen = [], [], set()
    for _ in range(80):
        ids, draws, tokens, acc, counts = _accept_case(rng, B, K, sampled)
        step = _step_buffer(tokens.reshape(-1), list(range(R)), list(range(R)), acc, B, T)
        pt = torch.randint(1, 900, (B, nb), dtype=torch.int32)
        pt_rows = pt.repeat_interleave(T, dim=0)
        dl = None if draws is None else _lanes(draws)
        o = TorchOps(_lanes(ids), step, pt_rows, B, K, scratch, ROWS, draws=dl)
        pack, rs = graph(o, B, K, W)
        ref = step_accept_reference(ids.reshape(-1), tokens.reshape(-1), acc, pt, scratch, K, ROWS, draws=draws)
        assert ref[1].tolist() == counts
        cases.append((_lanes(ids), dl, step, ROWS - 1))
        wants.append((pack, rs, ref, pt_rows))
        seen |= set(counts)
    assert seen == set(range(K + 1)), "every acceptance count"
    for (pack, real), (want_pack, want_rs, ref, pt_rows) in zip(_accept_words(words, cases, B, K, W), wants):
        assert torch.equal(pack, want_pack.to(torch.int64)), "drafter pack"
        assert torch.equal(pack, ref[2].to(torch.int64)), "drafter pack of the host path"
        rs = torch.where(real.reshape(R, 1), pt_rows, torch.full_like(pt_rows, scratch))
        assert torch.equal(rs, want_rs.to(torch.int32)) and torch.equal(rs, ref[3]), "reseed table"


@pytest.mark.parametrize("B,K", [(1, 3), (8, 3), (4, 7)])
def test_the_step_kernel_writes_the_op_graph_words_for_any_small_integer_words(words, B, K):
    """Out-of-contract words (flags 0 .. 3, counts and valid drafts outside [0, K], draws that make negative ids,
    a page table holding -1): the kernel keeps every expression of the graph, so its words equal the graph's after the
    float32 -> uint32 typecast rule."""
    T, R, W, nb = K + 1, B * (K + 1), pack_width(B * (K + 1)), 5
    rng = random.Random(77 + B + K)
    cases, wants = [], []
    for _ in range(300):
        ids = torch.randint(0, 4, (B * T,))
        draws = torch.randint(0, 4, (B * T,)) if rng.random() < 0.7 else None
        tokens = torch.randint(0, 4, (B * T,))
        acc = {
            STEP_NUM_VALID: [rng.randrange(0, K + 3) for _ in range(B)],
            STEP_DRAWN: [rng.randrange(4) for _ in range(B)],
            STEP_P0: [rng.randrange(1 << 20) for _ in range(B)],
            STEP_ROPE_BASE: [rng.randrange(1 << 20) for _ in range(B)],
            STEP_DECLINED: [rng.randrange(4) for _ in range(B)],
            STEP_KEEP: [rng.randrange(4) for _ in range(B)],
        }
        step = _step_buffer(tokens, list(range(R)), list(range(R)), acc, B, T)
        top = rng.randrange(1, 1 << 20)
        scratch = rng.randrange(1 << 20)
        pt_rows = torch.randint(-1, 1 << 20, (R, nb), dtype=torch.int32)
        dl = None if draws is None else _lanes(draws)
        o = TorchOps(_lanes(ids), step, pt_rows, B, K, scratch, top + 1, draws=dl)
        pack, rs = graph(o, B, K, W)
        cases.append((_lanes(ids), dl, step, top))
        wants.append((_as_u32(pack), rs.to(torch.int64), pt_rows, scratch))
    for (pack, real), (want_pack, want_rs, pt_rows, scratch) in zip(_accept_words(words, cases, B, K, W), wants):
        assert torch.equal(pack, want_pack)
        rs = torch.where(real.reshape(R, 1), pt_rows, torch.full_like(pt_rows, scratch)).to(torch.int64)
        assert torch.equal(rs, want_rs)


def test_the_greedy_walk_of_the_step_kernel_stops_where_the_plugin_walk_stops(words):
    ids = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]])
    tokens = torch.tensor([[1, 5, 6, 9], [1, 5, 0, 7], [1, 5, 6, 7]])
    acc = {r: [0] * 3 for r in USER_ROWS}
    acc[STEP_NUM_VALID] = [3, 3, 1]
    step = _step_buffer(tokens.reshape(-1), [0] * 12, [0] * 12, acc, 3, 4)
    ((pack, _),) = _accept_words(words, [(_lanes(ids), None, step, ROWS - 1)], 3, 3, pack_width(12))
    m = (pack[draft_anchor(3), :3] - torch.arange(3) * 4).tolist()
    assert m == greedy_counts(ids, tokens, [3, 3, 1]).tolist() == [2, 1, 1]


# ----------------------------------------------------------------------------------------------- chain program


def _chain_line(B, K, nv, groups, step_words, ctrl_words, top, ids, draws, pack, overlay, drafts, cand, seed_cand):
    R = B * (K + 1)
    head = [B, K, nv, groups, step_words, ctrl_words, top, int(seed_cand is not None)]
    parts = [torch.as_tensor(ids).reshape(-1)[:R]]
    if seed_cand is not None:
        parts.append(torch.as_tensor(draws).reshape(-1)[:R])
    parts += [torch.as_tensor(pack)[draft_anchor(K), :B], torch.as_tensor(overlay)[:, :B]]
    parts += [torch.stack([torch.as_tensor(d).reshape(-1)[:B] for d in drafts]), candidate_words(cand)]
    if seed_cand is not None:
        parts.append(candidate_words(seed_cand))
    return _line("chain", head, *parts)


def _chain_words(words, lines, B, K, groups, step_words, ctrl_words, sampler):
    """[(step [9, step_words], [ctrl [1, ctrl_words]] per group, seeds [R] or None)] of each case line."""
    R, n_step = B * (K + 1), VERIFY_STEP_ROWS * step_words
    out = []
    for o in words.run(lines):
        o = torch.tensor(o)
        ctrl = o[n_step : n_step + groups * ctrl_words].reshape(groups, 1, ctrl_words)
        seeds = o[n_step + groups * ctrl_words :] if sampler else None
        assert seeds is None or seeds.numel() == R
        out.append((o[:n_step].reshape(VERIFY_STEP_ROWS, step_words), list(ctrl), seeds))
    return out


def _case_line(case, B, K, groups, nv, ctrl_words, sampled, seed_cand=None, overlay=None):
    W = pack_width(B * (K + 1))
    seed_cand = case.seed_cand if seed_cand is None else seed_cand
    overlay = case.overlay if overlay is None else overlay
    return _chain_line(
        B, K, nv, groups, W, ctrl_words, ROWS - 1, case.ids, case.draws, case.pack, overlay, case.drafts, case.cand,
        seed_cand if sampled else None,
    )  # fmt: skip


@pytest.mark.parametrize("moved", [False, True], ids=["rows-stay", "rows-move"])
@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("B,K", [(1, 3), (8, 3), (2, 3), (4, 3), (4, 7), (2, 11)])
def test_the_chain_kernel_writes_the_words_of_the_op_graph(words, B, K, sampled, moved):
    """chain_words == spec_chain.chain_graph (the replaced op graph in torch) == the block, ctrl pages and seeds the
    host stages for the chained verify, over every count 0 .. K of the predecessor; drawn predecessor users continue
    from their draws, held rows keep SPEC_STEP_HOLD. With moved rows (a random slot_remap per case) each row continues
    the predecessor row its overlay CH_SRC word names."""
    groups, nv = 3, 6
    words_c = _ctrl_words(B, nv)
    rng = random.Random(B * 13 + K + 1000 * sampled + 7 * moved)
    cases = [_chain_case(rng, B, K, groups, nv, sampled, moved) for _ in range(80)]
    if moved and B > 1:
        assert any(c.src != list(range(B)) for c in cases), "a row that moves"
    if K <= 7:  # _chain_case reaches all K + 1 counts of K = 11 too rarely
        assert set().union(*(set(c.m) for c in cases)) == set(range(K + 1)), "every count of the predecessor"
    lines = [_case_line(c, B, K, groups, nv, words_c, sampled) for c in cases]
    held = 0
    for case, (step, ctrls, seeds) in zip(
        cases, _chain_words(words, lines, B, K, groups, pack_width(B * (K + 1)), words_c, sampled)
    ):
        ref = chain_reference(
            case.ids, case.pack, case.overlay, case.cand, case.drafts, B, K, ROWS, words_c, nv, groups,
            draws=case.draws, seed_cand=case.seed_cand,
        )  # fmt: skip
        assert torch.equal(step, ref[0].to(torch.int64)) and torch.equal(step, case.step.to(torch.int64)), "step"
        for got, want, host in zip(ctrls, ref[1], case.ctrls):
            assert torch.equal(got, want.to(torch.int64)) and torch.equal(got, host.to(torch.int64)), "ctrl page"
            held += int((got == SPEC_STEP_HOLD).sum())
        assert (seeds is None) == (ref[2] is None)
        if seeds is not None:
            assert torch.equal(seeds, ref[2]) and torch.equal(seeds, case.seeds), "seeds"
    assert held > 0, "held rows write SPEC_STEP_HOLD"


@pytest.mark.parametrize("B", [1, 8])
def test_the_chain_kernel_seeds_are_the_unchained_seeds_of_the_device_count(words, B):
    """With the seed candidates of spec_sample.chained_sampling_plan, the kernel gives a seeded user the seeds an
    unchained verify at the device count's column-0 position draws with, an unseeded user its fresh seeds at every
    count, and a user that draws nothing seed 0 with the plan's user id NO_CORE; also on rows that continue another
    predecessor row."""
    K, groups, nv = 3, 3, 6
    T, R = K + 1, B * (K + 1)
    words_c = _ctrl_words(B, nv)
    rng = random.Random(B + 5)
    counter = iter(range(1, 1 << 20))
    kinds_seen, counts = set(), set()
    for i in range(80):
        case = _chain_case(rng, B, K, groups, nv, True, moved=i % 2 == 1)
        kinds = [rng.choice(["seeded", "unseeded", "greedy"]) for _ in range(B)]
        sampling = SimpleNamespace(
            temperature=[0.0 if k == "greedy" else 0.8 for k in kinds],
            top_k=[20] * B,
            top_p=[0.95] * B,
            seed=[rng.randrange(1 << 40) if k == "seeded" else None for k in kinds],
        )
        p0_prev = [p if a else -1 for p, a in zip(case.p0_prev, case.active)]
        plan, seed_cand = chained_sampling_plan(sampling, p0_prev, T, fresh_seed=lambda: next(counter))
        assert plan.seeds is None
        drawing = [plan.sampled[u * T] for u in range(B)]
        overlay = case.overlay.clone()
        overlay[CH_DRAWN, :B] = torch.tensor([int(d) for d in drawing])
        line = _case_line(case, B, K, groups, nv, words_c, True, seed_cand=seed_cand, overlay=overlay)
        ((step, _, seeds),) = _chain_words(words, [line], B, K, groups, pack_width(R), words_c, True)
        assert step[STEP_DRAWN, :B].tolist() == [int(d) for d in drawing]
        positions = [p0_prev[u] + case.m[u] + 1 if case.active[u] else -1 for u in range(B)]
        unchained = sampling_plan(sampling, positions, T, fresh_seed=lambda: 0)
        for u in range(B):
            got = seeds[u * T : (u + 1) * T].tolist()
            if not drawing[u]:
                assert got == [0] * T and plan.user_ids[u * T : (u + 1) * T] == [NO_CORE] * T
                continue
            assert plan.user_ids[u * T : (u + 1) * T] == list(range(u * T, (u + 1) * T))
            kinds_seen.add(kinds[u])
            if kinds[u] == "seeded":
                assert got == unchained.seeds[u * T : (u + 1) * T], "the unchained seeds of that position"
            else:
                fresh = seed_cand[u::B].to(torch.int64)
                assert all(got == fresh[c].tolist() for c in range(T)) and 0 not in got, "fresh, the same per count"
            counts.add(case.m[u])
    assert kinds_seen == {"seeded", "unseeded"} and counts == set(range(K + 1))


@pytest.fixture
def host_tables(monkeypatch):
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone(), dtype=kw.get("dtype")))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: None)


@pytest.mark.parametrize("bmax", [1, 8])
def test_the_chain_kernel_writes_the_ctrl_pages_of_the_start_blocks_of_each_count(words, host_tables, bmax):
    """Ctrl candidates of GdnBlockState.ctrl_candidates for rows whose start block changes with the count (a 1024-token
    block boundary inside the verify window) and a held padding row: the kernel's ctrl page of each table group is
    the verify_tables page of the start blocks of each row's count."""
    st, plugin = _block_state(bmax=bmax)
    B, T = st.bmax, st.verify_tokens
    K, nv = T - 1, st.dims.nv
    rng = random.Random(bmax)
    groups = []
    for names in plugin.group_names:
        windows = torch.randint(1, NUM_BLOCKS, (B, T), dtype=torch.int32)
        by_count = torch.empty(B, T, dtype=torch.int32)
        for r in range(B):
            first, second = rng.sample(range(1, NUM_BLOCKS), 2)
            cross = rng.randrange(T + 1)  # counts from `cross` on start in the next block
            by_count[r] = torch.tensor([first if c < cross else second for c in range(T)])
        if B > 1:
            windows[B - 1] = 0  # a padding row: held at every count
            by_count[B - 1] = 0
        groups.append(
            SimpleNamespace(
                layer_names=names, windows=windows, start_blocks=by_count[:, -1], start_blocks_by_count=by_count
            )
        )
    state = SimpleNamespace(groups=tuple(groups))
    st.set_verify_blocks(state, stage_ctrl=False)
    cand = st.ctrl_candidates(state)
    ctrl_words = int(st._verify_ctrl[0].host.shape[-1])
    n_groups = len(st._verify_ctrl)
    W = pack_width(B * T)
    lines, ms = [], []
    for _ in range(30):
        m = [rng.randrange(T) for _ in range(B)]
        pack = torch.zeros(drafter_pack_rows(K), W, dtype=torch.int64)
        pack[draft_anchor(K), :B] = torch.arange(B) * T + torch.tensor(m)
        overlay = chain_overlay([100] * B, [1] * B, [0] * B, [K] * B, [0] * B, [0] * B)
        drafts = [torch.zeros(B, dtype=torch.int64) for _ in range(K)]
        lines.append(
            _chain_line(B, K, nv, n_groups, W, ctrl_words, ROWS - 1, _lanes(torch.zeros(B * T)), None, pack, overlay,
                        drafts, cand, None)
        )  # fmt: skip
        ms.append(m)
    crossing = 0
    for m, (_, ctrls, _) in zip(ms, _chain_words(words, lines, B, K, n_groups, W, ctrl_words, False)):
        for g, grp in enumerate(state.groups):
            k = st._verify_order[1][g]
            starts = grp.start_blocks_by_count[torch.arange(B), torch.tensor(m)]
            want, _ = st.layout.verify_tables(starts, grp.windows, NUM_BLOCKS, B)
            assert torch.equal(ctrls[k], want.to(torch.int64)), f"ctrl page of table group {k}"
            crossing += int((grp.start_blocks_by_count[:, 0] != grp.start_blocks_by_count[:, -1]).sum())
    assert crossing > 0, "start blocks that change with the count"
    if B > 1:
        assert bool((cand[B - 1 :: B] == SPEC_STEP_HOLD).all())


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("B,K", [(1, 3), (8, 3), (2, 7)])
def test_the_chain_kernel_writes_the_op_graph_words_for_any_small_integer_words(words, B, K, sampled):
    """Out-of-contract words: anchors whose count falls outside [0, T) (the graph's one-hot picks nothing), flags
    -2 .. 3, negative column-0 positions and M-RoPE deltas, valid drafts outside [0, K], negative candidates, source
    rows outside [0, B) (the graph's one-hot gather reads 0) and repeated. The kernel's words equal chain_graph's after
    the float32 typecast rules (uint32: a negative value is 0)."""
    groups, nv = 2, 3
    T, R, W = K + 1, B * (K + 1), pack_width(B * (K + 1))
    words_c = _ctrl_words(B, nv)
    rng = random.Random(B * 7 + K + sampled)
    lines, wants = [], []
    for _ in range(200):
        ids = _lanes(torch.randint(0, 1 << 20, (R,)))
        draws = _lanes(torch.randint(0, 1 << 20, (R,))) if sampled else None
        pack = torch.zeros(drafter_pack_rows(K), W, dtype=torch.int64)
        pack[draft_anchor(K), :B] = torch.randint(0, 2 * R, (B,))
        overlay = torch.zeros(CHAIN_OVERLAY_ROWS, pack_width(B), dtype=torch.int32)
        overlay[:, :B] = torch.randint(-2, 4, (CHAIN_OVERLAY_ROWS, B), dtype=torch.int32)
        overlay[CH_P0_PREV, :B] = torch.randint(-100, 5000, (B,), dtype=torch.int32)
        overlay[CH_DELTA, :B] = torch.randint(-60, 60, (B,), dtype=torch.int32)
        overlay[CH_SRC, :B] = torch.randint(-2, B + 2, (B,), dtype=torch.int32)
        drafts = [torch.randint(0, 1 << 20, (B,)) for _ in range(K)]
        cand = torch.randint(-1, 1 << 20, (R, 2 * nv * groups)).to(torch.float32)
        seed_cand = torch.randint(0, 1 << 24, (R, T)).to(torch.float32) if sampled else None
        top = rng.randrange(1, 6000)
        lines.append(_chain_line(B, K, nv, groups, W, words_c, top, ids, draws, pack, overlay, drafts, cand, seed_cand))
        o = ChainTorchOps(ids, pack, overlay, cand, drafts, B, K, top + 1, words_c, nv, draws, seed_cand)
        step, ctrls, seeds = chain_graph(o, B, K, W, groups, nv, words_c)
        wants.append((_as_u32(step), [c.to(torch.int64) for c in ctrls], None if seeds is None else _as_u32(seeds)))
    for (step, ctrls, seeds), (want_step, want_ctrls, want_seeds) in zip(
        _chain_words(words, lines, B, K, groups, W, words_c, sampled), wants
    ):
        assert torch.equal(step, want_step)
        assert all(torch.equal(got, want) for got, want in zip(ctrls, want_ctrls))
        assert (seeds is None) == (want_seeds is None)
        if seeds is not None:
            assert torch.equal(seeds, want_seeds.reshape(-1))


def test_rows_continue_their_own_predecessor_row_unless_a_source_is_given():
    """chain_overlay's CH_SRC row is range(B) without ``src``: the words of rows that stay."""
    B = 5
    rows = ([7] * B, [1] * B, [0] * B, [3] * B, [0] * B, [0] * B)
    assert chain_overlay(*rows)[CH_SRC, :B].tolist() == list(range(B))
    assert chain_overlay(*rows, src=[4, 0, 1, 2, 3])[CH_SRC, :B].tolist() == [4, 0, 1, 2, 3]


# ----------------------------------------------------------------------------------------------- L1 layouts


@pytest.mark.parametrize("B,K", [(1, 1), (1, 3), (8, 3), (4, 7), (2, 11), (16, 1)])
def test_the_kernel_cbs_hold_the_kernels_l1_layouts(words, B, K):
    R = B * (K + 1)
    W = pack_width(R)
    lines, want = [], []
    for nb in (1, 9, 256, 257, 4097):
        chunk = rs_chunk_bytes(nb)
        assert chunk % SLOT == 0 and SLOT <= chunk <= RS_CHUNK_MAX and (chunk >= 4 * nb or chunk == RS_CHUNK_MAX)
        lines.append(_line("accept_layout", [B, K, W, chunk]))
        want.append(step_kernel_l1_bytes(B, K, W, chunk))
    for nv, groups, sampler in ((6, 3, False), (6, 3, True), (3, 1, True)):
        ctrl_words = _ctrl_words(B, nv)
        lines.append(_line("chain_layout", [B, K, nv, groups, W, ctrl_words, int(sampler)]))
        want.append(chain_kernel_l1_bytes(B, K, nv, groups, W, ctrl_words, sampler))
    assert [o[0] for o in words.run(lines)] == want
    assert max(want) < 64 * 1024, "a few tens of KB of L1 on one core"


# --------------------------------------------------------------------------------------------- the launchers


class _Tensor(SimpleNamespace):
    def buffer_address(self):
        return self.addr

    def buffer_aligned_page_size(self):
        return self.page


class _Kernel(SimpleNamespace):
    SourceType = SimpleNamespace(FILE_PATH="FILE_PATH")


_REPLACED_OPS = (
    "add", "subtract", "multiply", "minimum", "rsub", "eq", "lt", "cumprod", "sum", "concat", "pad", "slice",
    "reshape", "transpose", "typecast", "to_layout", "copy", "relu",
)  # fmt: skip


@pytest.fixture
def device(monkeypatch):
    """ttnn stand-ins that record the tensors the launchers allocate, the copies they stage and the programs they run;
    every op of the replaced op graphs raises."""
    log = SimpleNamespace(programs=[], copies=[], addr=iter(range(0x10000, 1 << 30, 0x1000)))

    def tensor(shape, dtype=None, acc=("dram",), page=16, host=None):
        return _Tensor(shape=tuple(shape), dtype=dtype, addr=next(log.addr), acc=list(acc), page=page, host=host)

    def from_torch(t, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        return tensor(torch.as_tensor(t).shape, dtype, acc=("dram", str(dtype)), host=torch.as_tensor(t).clone())

    def refuse(name):
        def op(*args, **kwargs):
            raise AssertionError(f"ttnn.{name}: an op of the replaced op graph ran")

        return op

    for name in _REPLACED_OPS:
        monkeypatch.setattr(ttnn, name, refuse(name))
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: log.copies.append((host, dev)))
    monkeypatch.setattr(ttnn, "TensorAccessorArgs", lambda t: SimpleNamespace(get_compile_time_args=lambda: t.acc))
    monkeypatch.setattr(ttnn, "generic_op", lambda io, program: log.programs.append((list(io), program)))
    monkeypatch.setattr(ttnn, "KernelDescriptor", _Kernel)
    for name in ("ProgramDescriptor", "CBDescriptor", "CBFormatDescriptor", "CoreRangeSet"):
        monkeypatch.setattr(ttnn, name, lambda *a, **kw: SimpleNamespace(args=a, **kw))
    monkeypatch.setattr(ttnn, "CoreRange", lambda a, b: (a, b))
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y))
    log.tensor = tensor
    return log


def _one_kernel(log, source):
    ((io, program),) = log.programs
    (kernel,) = program.kernels
    assert Path(kernel.kernel_source) == _KERNELS / source and Path(kernel.kernel_source).is_file()
    assert kernel.core_ranges.args == ([((0, 0), (0, 0))],), "one core"
    (cb,) = program.cbs
    assert cb.total_size == cb.format_descriptors[0].page_size and program.semaphores == []
    return io, kernel, cb


@pytest.mark.parametrize("draws", [False, True])
def test_the_step_trace_is_one_kernel_over_the_engine_buffers(device, draws, expect_error):
    B, K, nb, scratch = 8, 3, 9, 999
    R, W = B * (K + 1), pack_width(B * (K + 1))
    step, pack = device.tensor((9, W)), device.tensor((drafter_pack_rows(K), W))
    rs, dr = device.tensor((R, nb)), device.tensor((1, 1, 1, R)) if draws else None
    acc = StepAccept(None, B, K, nb, ROWS, scratch, step, pack, rs, draws=dr)
    ids = device.tensor((1, 1, 1, 32), acc=acc.standin.acc)
    acc.bind_ids(ids)
    acc.body()
    io, kernel, cb = _one_kernel(device, "step_accept.cpp")
    assert io == [t for t in (ids, dr, step, acc.pt_rows, pack, rs) if t is not None]
    chunk = rs_chunk_bytes(nb)
    assert kernel.compile_time_args[:10] == [B, K, W, nb, ROWS - 1, scratch, int(draws), chunk, 0, acc.cb_bytes]
    order = (ids, dr if draws else ids, step, acc.pt_rows, pack, rs)
    assert kernel.compile_time_args[10:] == [a for t in order for a in t.acc]
    assert kernel.common_runtime_args == [t.addr for t in order]
    assert cb.total_size == acc.cb_bytes == step_kernel_l1_bytes(B, K, W, chunk) + SLOT
    with expect_error(ValueError, "stand-in"):
        acc.bind_ids(device.tensor((1, 1, 1, 32), acc=("l1",)))
    with expect_error(ValueError, "id lanes"):
        StepAccept(None, 9, 3, nb, ROWS, scratch, step, pack, rs)


@pytest.mark.parametrize("sampled", [False, True])
def test_the_chain_program_is_one_kernel_that_stages_int32_candidates_when_they_change(device, sampled, expect_error):
    """ChainInputs: one kernel over the verify ids, the sampler's draws, the drafter pack, the overlay, the
    candidates and the drafts, writing the step buffer, every ctrl page and the sampler's seeds as one row; the ctrl
    and seed candidates go to the device as the int32 words the op graph wrote, and only when they change."""
    B, K, nv, groups = 8, 3, 6, 3
    T, R, W, words_c = K + 1, B * (K + 1), pack_width(B * (K + 1)), _ctrl_words(B, nv)
    step, pack = device.tensor((9, W)), device.tensor((drafter_pack_rows(K), W))
    ctrls = [device.tensor((1, words_c)) for _ in range(groups)]
    sampler = None
    if sampled:
        sampler = SimpleNamespace(draws=device.tensor((1, 1, 1, R)), seeds=device.tensor((R,)))
    chain = ChainInputs(None, B, K, ROWS, step, pack, ctrls, nv, sampler=sampler)
    assert (chain.seed_cand is None) == (not sampled) and chain.cand.host.dtype == torch.int32
    ids = device.tensor((1, 1, 1, 32), acc=chain.standin.acc)
    drafts = [device.tensor((1, 1, B, 1), acc=chain.draft_standins[0].acc, page=32) for _ in range(K)]
    chain.bind(ids, drafts)
    chain.body()
    io, kernel, cb = _one_kernel(device, "chain_inputs.cpp")
    draws, seed_cand, seeds = (sampler.draws, chain.seed_cand, sampler.seeds) if sampled else (None, None, None)
    inputs = (ids, draws, pack, chain.overlay, chain.cand, seed_cand, *drafts)
    assert io == [t for t in (*inputs, step, *ctrls, seeds) if t is not None]
    assert kernel.compile_time_args[:11] == [
        B,
        K,
        nv,
        groups,
        W,
        words_c,
        ROWS - 1,
        int(sampled),
        32,
        0,
        chain.cb_bytes,
    ]
    ph = lambda t: ids if t is None else t
    order = (ids, ph(draws), pack, chain.overlay, chain.cand, ph(seed_cand), drafts[0], step, ctrls[0], ph(seeds))
    assert kernel.compile_time_args[11:] == [a for t in order for a in t.acc]
    addrs = [t.addr for t in (ids, ph(draws), pack, chain.overlay, chain.cand, ph(seed_cand), step, ph(seeds))]
    assert kernel.common_runtime_args == addrs + [d.addr for d in drafts] + [c.addr for c in ctrls]
    assert cb.total_size == chain.cb_bytes == SLOT + chain_kernel_l1_bytes(B, K, nv, groups, W, words_c, sampled)

    overlay = chain_overlay([5] * B, [1] * B, [0] * B, [K] * B, [0] * B, [0] * B)
    cand = torch.randint(-1, 1 << 20, (T * B, 2 * nv * groups)).to(torch.float32)
    seed_cand = torch.randint(1, 1 << 20, (T * B, T)).to(torch.float32)
    device.copies.clear()
    chain.stage(overlay, cand, seed_cand)
    chain.stage(overlay, cand, seed_cand)
    chain.stage(overlay, cand + 1, seed_cand)
    devs = [dev for _, dev in device.copies]
    want = [chain.overlay, chain.cand] + ([chain.seed_cand] if sampled else []) + [chain.overlay, chain.overlay]
    assert devs == want + [chain.cand]
    for host, dev in device.copies:
        if dev is not chain.overlay:
            assert host.dtype == ttnn.int32 and host.host.dtype == torch.int32
    staged_cand = [h.host for h, d in device.copies if d is chain.cand]
    assert torch.equal(staged_cand[0], cand.to(torch.int32)) and torch.equal(staged_cand[1], (cand + 1).to(torch.int32))

    with expect_error(ValueError, "stand-in"):
        chain.bind(ids, [*drafts[:-1], device.tensor((1, 1, B, 1), acc=("l1",))])
    with expect_error(ValueError, "draft id tensors"):
        chain.bind(ids, drafts[:-1])
    big = [device.tensor((1, 1, B, 1), acc=chain.draft_standins[0].acc, page=128) for _ in range(K)]
    chain.bind(ids, big)
    with expect_error(ValueError, "slot"):
        chain.body()


def test_the_candidate_words_are_the_op_graphs_typecast_of_float32_candidates():
    """candidate_words: the float32 value the op graph held, truncated as its float32 -> int32 typecast; a held row's
    -1 and integers up to 2**24 pass unchanged."""
    host = torch.tensor([[-1.0, 0.0, 7.0, float(1 << 24)], [3.0, 16777215.0, 2.0, 1.0]], dtype=torch.float64)
    assert candidate_words(host).tolist() == [[-1, 0, 7, 1 << 24], [3, 16777215, 2, 1]]
    assert candidate_words(host).dtype == torch.int32
