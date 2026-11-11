# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 prefill-side prefix reuse H1 end to end on a HOST fake device (``fake_v41.py``; no chips, no real ttnn): the real
``V41PrefillRuntime`` chunk path, the real ``kv_export.py`` (all three export layouts) and ``tt/v41/prefix_reuse.py``.

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse

* every reuse scenario's WHOLE export (all 7 tensors, rows past the prompt included) and the model's own state equal a cold
  prefill of the same prompt (run after an unrelated prompt, flag off = today's runner);
* the score mask is re-opened without any zero block built after the island capture (design risk R2), also when the
  first-traced-chunk block differs from a one-chunk piece;
* with ``V41_PREFILL_PREFIX_REUSE`` unset the runtime issues the SAME device work, call for call, as the base commit's
  runtime.py + kv_export.py (``git show``; skipped without git);
* each piece of the resume is needed: removing it makes a scenario fail (mutations).
"""

from __future__ import annotations

import contextlib
import importlib
import os
import random
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_v41 as F  # noqa: E402 -- installs the fake ttnn before anything imports it
import pytest  # noqa: E402
import torch  # noqa: E402

from models.demos.deepseek_v3_d_p.tt.v41 import kv_export as KX  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v41 import prefix_reuse as PR  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v41 import runtime as RT  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config  # noqa: E402

# the layer-role fields of DeepSeek-V4.1-Flash-1017's inference/config.json (2026-11-11), used when the checkpoint is not readable
_ROLES_1017 = dict(
    n_layers=40,
    compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20,
    kv_source_layers=(2, 8, 14, 20),
    index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
    candidate_source_layer=20,
    engram_layer_ids=(1, 14),
    vocab_size=129280,
    window_size=128,
)


def _config():
    d = os.environ.get("V41_PREFIX_MODEL_DIR", "/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash-1017")
    try:
        return V41Config.load(d)  # the checkpoint's own inference/config.json: the real layer roles
    except OSError:
        kw = {f: 0 for f in V41Config.__dataclass_fields__ if f != "raw"}
        kw.update(_ROLES_1017)
        return V41Config(**kw)


CFG = _config()
assert {k: getattr(CFG, k) for k in _ROLES_1017} == _ROLES_1017, "the checkpoint's roles changed: update _ROLES_1017"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
BASE = os.environ.get("V41_PREFIX_BASE_COMMIT", "03b370b312b")  # sdawle/dsv41p_1017_tt-d-gen, the branch point
C = 512  # chunk (a multiple of 64 x sp = 512)
MAXSEQ = 8 * C
MARKED, LAST = 1 << 29, 1 << 30
LAYOUTS = {
    "plain": {"V41_EXPORT_CP2": "0"},
    "cp2": {"V41_EXPORT_CP2": "1", "V41_EXPORT_IDX_SHARD": "8"},  # index split block 64 rows (512 in production)
    "cp2_idx0": {"V41_EXPORT_CP2": "1", "V41_EXPORT_CP2_IDX": "0"},
}
assert not hasattr(sys.modules["ttnn"], "open_mesh_device"), "the REAL ttnn got imported"


def _env(**kv):
    @contextlib.contextmanager
    def cm():
        old = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return cm()


def _base_env(layout: str, flag: bool, ckpts: int = 8) -> dict:
    env = {
        "V41_EXPORT_CP2": None,
        "V41_EXPORT_IDX_SHARD": None,
        "V41_EXPORT_CP2_IDX": None,
        "V41_PREFILL_ISLANDS": "1",
        "V41_PREFILL_WARM_ALL": None,
        "V41_PREFILL_WARM_RAGGED": None,
        "PREFILL_REQUEST_SIDECAR_DIR": None,
        PR.ENV_FLAG: "1" if flag else None,
        PR.ENV_CKPTS: str(ckpts),
    }
    env.update(LAYOUTS[layout])
    return env


class System:
    """One runner: the runtime (built like the runner does: compile, on_services_ready) over a fake device and export."""

    def __init__(self, rt_mod=RT, kx_mod=KX, *, users=1, topk=64, flag=False):
        F.SYS.reset()
        rt = object.__new__(rt_mod.V41PrefillRuntime)
        rt.mesh_device, rt.cfg = F.MESH, CFG
        rt.config = SimpleNamespace(is_first_rank=True, is_last_rank=True, use_trace=False, num_users=users)
        rt.chunk_size, rt.max_seq_len, rt.num_users = C, MAXSEQ, users
        rt.pf = F.FakePrefill(CFG, chunk=C, max_seq_len=MAXSEQ, num_users=users, topk=topk)
        rt.engram = F.FakeEngram(users, MAXSEQ)
        rt.vit, rt.num_layers, rt._request_id = None, int(CFG.n_layers), 0
        self.acks = []
        rt._ack = lambda L: self.acks.append((rt._request_id, L))
        rt._prefix = None  # what V41PrefillRuntime.__init__ sets (the base commit's runtime never reads it)
        self.rt, self.kv = rt, F.make_export(kx_mod, CFG, max_seq_len=MAXSEQ, num_users=users)
        self.request = 0
        self.last_did = []
        rt.compile(self.kv)
        rt.on_services_ready(self.kv)
        assert (getattr(rt, "_prefix", None) is not None) == flag

    def prompt(self, toks, *, slot=0, marks=True, cut=None) -> list:
        """Every chunk of one prompt as the engine sends them (padded to C); returns S / C(+silent) per chunk."""
        S = len(toks)
        parts = [(s, min(S, s + C)) for s in range(0, S, C)]
        parts = parts[:cut] if cut else parts
        did = []
        for k, (s, e) in enumerate(parts):
            word = slot | ((MARKED | (LAST if k == len(parts) - 1 else 0)) if marks else 0)
            n0, a0 = len(F.SYS.trace), len(self.acks)
            self.rt.prefill_chunk(
                list(toks[s:e]) + [1] * (C - (e - s)),
                self.kv,
                slot_id=word,
                actual_start=s,
                actual_end=e,
                request_id=self.request,
            )
            self.request += 1
            assert len(self.acks) - a0 == CFG.n_layers, "every chunk acks every contract layer"
            n = sum(1 for op in F.SYS.trace[n0:] if op[0] == "pf._chunk")
            did.append("S" if n == 0 else ("C" if n == 1 else f"C+{n - 1}silent"))
        self.last_did = did
        return did

    def state(self, slot, S, live=True):
        return F.export_bytes(self.kv), F.model_state(self.rt.pf, slot, S, live)


def toks(n, seed):
    r = random.Random(seed)
    return [r.randrange(1000, 120000) for _ in range(n)]


X = toks(2 * C, 7)  # an unrelated prompt: dirties the caches before every cold reference


@contextlib.contextmanager
def _isolated_sys():
    saved = (F.SYS.trace, F.SYS.next_id, F.SYS.captured, F.SYS.late_allocs)
    try:
        yield
    finally:
        F.SYS.trace, F.SYS.next_id, F.SYS.captured, F.SYS.late_allocs = saved


_REFS: dict = {}


def cold_ref(layout, tokens, *, topk=64, users=1, slot=0):
    """The cold reference: today's runner (flag off), an unrelated prompt, then ``tokens``."""
    key = (layout, tuple(tokens), topk, users, slot)
    if key not in _REFS:
        with _isolated_sys(), _env(**_base_env(layout, flag=False)):
            s = System(users=users, topk=topk)
            s.prompt(X, slot=slot)
            s.prompt(tokens, slot=slot)
            _REFS[key] = (s.state(slot, len(tokens)), F.model_state(s.rt.pf, slot, len(tokens), live=False))
    return _REFS[key]


def assert_cold(sys_, layout, tokens, *, topk=64, users=1, slot=0, what=""):
    """The WHOLE export equals a cold prefill's; so do the model's rows [0, S); and -- when the prompt's last chunk was
    computed (the device's live state is the prompt's end) -- its counters and carries too."""
    (ref_exp, ref_live), ref_rows = cold_ref(layout, tokens, topk=topk, users=users, slot=slot)
    exp, live = sys_.state(slot, len(tokens))
    bad = [k for k in ref_exp if not F.same(exp[k], ref_exp[k])]
    assert not bad, f"{what}: export tensors {bad} differ from a cold prefill's"
    assert F.same(F.model_state(sys_.rt.pf, slot, len(tokens), live=False), ref_rows), f"{what}: model rows differ"
    if sys_.last_did and sys_.last_did[-1] != "S":
        assert F.same(live, ref_live), f"{what}: the model's counters / carries differ from a cold prefill's"


A = toks(3 * C + 384, 1)  # ragged last chunk; 384 % 256 = 128: its window block is the other half of the ring
A2 = A + toks(2 * C + 128, 2)  # the next turn: resumes at 3C
B = A[: C + 256] + toks(2 * C - 128, 3)  # a sibling sharing chunk 0 (and 256 tokens of chunk 1): resumes at C
AP = A[: 2 * C]  # a strict prefix ending ON a boundary: skipped through, its last chunk's carries from a checkpoint
LONG = toks(7 * C + 256, 4)
SHORT_SIB = LONG[:C] + toks(C + 128, 5)  # far shorter than LONG: resuming at C must clear LONG's rows past it


# ------------------------------------------------------------------------------------------------------- scenarios
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_reuse_sequence_equals_cold(layout):
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(X)
        assert s.prompt(A) == ["C"] * 4
        assert_cold(s, layout, A, what="A computed")
        assert s.prompt(B) == ["S", "C", "C", "C"], "chunk 0 shared"
        assert_cold(s, layout, B, what="B after A (resume at C)")
        assert s.prompt(A) == ["S", "C", "C", "C"]
        assert_cold(s, layout, A, what="A after B")
        assert s.prompt(A2) == ["S", "S", "S", "C", "C", "C"]
        assert_cold(s, layout, A2, what="A2 extends A (resume at 3C)")
        assert s.prompt(A2) == ["S"] * 6
        assert_cold(s, layout, A2, what="A2 again (every chunk skipped, live carries)")
        assert s.prompt(AP) == ["S", "S"]
        assert_cold(
            s, layout, AP, what="AP: strict prefix on a boundary (carries from the checkpoint, rows past cleared)"
        )
        assert s.prompt(A) == ["S", "S", "C", "C"]
        assert_cold(s, layout, A, what="A after AP (resume at 2C, never shipping the cleared rows)")
        assert s.prompt(LONG) == ["C"] * 8
        assert s.prompt(SHORT_SIB) == ["S", "C", "C"]
        assert_cold(s, layout, SHORT_SIB, what="short sibling after LONG (rows [3C, 8C) cleared on resume)")
        assert F.SYS.late_allocs == [], f"zero blocks built after the island capture: {F.SYS.late_allocs}"
        st = s.rt._prefix.ret[0].stats
        assert st.skipped_chunks == 16 and st.resumed_prompts == 7, s.rt._prefix.ret[0].summary()


@pytest.mark.parametrize("marks", [True, False])
@pytest.mark.parametrize("tail", [384, 256])
def test_full_repeat_of_a_partial_last_chunk_equals_cold(marks, tail):
    """Device check run1 (11-11, S.17): A2 sent again (41600 = 5 x 8192 + 640, every chunk matching) exported different
    window rings than cold. A partial last chunk ending at E = 128 (mod 256) leaves TWO carries in a cold export's
    256-row ring (its own at rows [0, 128), the previous chunk boundary's at [128, 256)); skipping it rebuilt only one.
    tail 384 (E = 128 mod 256) is that case; tail 256 (E = 0 mod 256) is the other half."""
    layout = "cp2"
    P = A[: 3 * C] + toks(tail, 21)
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(X, marks=marks)
        s.prompt(P, marks=marks)
        did = s.prompt(P, marks=marks)
        assert did == ["S"] * 3 + ["C"], did  # the partial chunk computed, resumed at 3C (skip_partial_last=False)
        assert_cold(s, layout, P, what=f"P ({len(P)} tokens) sent again")
        assert s.prompt(P, marks=marks) == ["S"] * 3 + ["C"]
        assert_cold(s, layout, P, what="P a third time")
        assert s.rt._prefix.ret[0].stats.silent_chunks == 0


def test_unmarked_headers_equal_cold_without_a_strict_prefix():
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(X, marks=False)
        s.prompt(A, marks=False)
        assert s.prompt(A2, marks=False) == ["S", "S", "S", "C", "C", "C"]  # every full chunk's end carries are held
        assert_cold(s, layout, A2, what="A2 unmarked")
        assert s.prompt(B, marks=False) == ["S", "C", "C", "C"]
        assert_cold(s, layout, B, what="B unmarked")


def test_small_pool_rebuilds_the_carries_silently():
    layout = "cp2"
    with _env(**_base_env(layout, flag=True, ckpts=2)):
        s = System(flag=True)
        s.prompt(LONG)
        # 2 slots keep {4C, 7C} after LONG (the pool's thinning): a resume at 3C has no checkpoint at or below it but 0
        assert sorted(s.rt._prefix.ret[0].pool.by_pos) == [4 * C, 7 * C]
        Q = LONG[: 3 * C] + toks(2 * C + 256, 9)
        did = s.prompt(Q)
        assert did == ["S"] * 3 + ["C+3silent", "C", "C"], did
        assert_cold(s, layout, Q, what="resume at 4C rebuilt from an older checkpoint")
        assert s.rt._prefix.ret[0].stats.silent_chunks > 0
        assert F.SYS.late_allocs == []


def test_mask_pieces_when_the_first_traced_block_is_wider_than_a_chunk():
    """top-k x ratio > chunk: the A3 alloc's first-open block is 2 chunks of entries, the resume's pieces are 1; the piece
    must be built at compile time, before the capture (R2), never at the resume."""
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True, topk=512)
        s.prompt(A)
        assert s.prompt(A2) == ["S", "S", "S", "C", "C", "C"]
        assert_cold(s, layout, A2, topk=512, what="A2, top-k 512")
        assert F.SYS.late_allocs == [], F.SYS.late_allocs


def test_image_span_ids_are_never_reused():
    layout = "cp2"
    img = A[:C] + [CFG.vocab_size + 3] * 128 + A[C + 128 :]  # virtual image-span ids in chunk 1
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        assert s.prompt(img) == ["C"] * 4
        assert s.prompt(img) == ["C"] * 4, "the same ids can carry a different image: computed whole"
        assert_cold(s, layout, img, what="image prompt again")
        assert s.prompt(A) == ["C"] * 4  # nothing of the image prompt was retained
        assert_cold(s, layout, A, what="A after the image prompt")


def test_two_slots_keep_separate_retention():
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True, users=2)
        s.prompt(A, slot=0)
        s.prompt(B, slot=1)
        assert s.prompt(A2, slot=0) == ["S", "S", "S", "C", "C", "C"]
        exp, st = s.state(0, len(A2))
    with _isolated_sys(), _env(**_base_env(layout, flag=False)):
        r = System(users=2)
        r.prompt(A, slot=0)
        r.prompt(B, slot=1)
        r.prompt(A2, slot=0)
        ref_exp, ref_st = r.state(0, len(A2))
    assert all(F.same(exp[k], ref_exp[k]) for k in ref_exp), "two-slot export differs from today's runner"
    assert F.same(st, ref_st)


def test_failed_chunk_invalidates_the_slot():
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(A)
        real_chunk = s.rt.pf._chunk

        def boom(*a, **k):
            raise RuntimeError("device fault")

        s.rt.pf._chunk = boom
        with pytest.raises(RuntimeError):  # allow-pytest.raises: no ttnn conftest
            s.prompt(A2)  # chunks 0..2 skipped, chunk 3 fails
        s.rt.pf._chunk = real_chunk
        assert s.rt._prefix.ret[0].length == 0
        assert s.prompt(A2) == ["C"] * 6
        assert_cold(s, layout, A2, what="A2 after a failed chunk")


def _empty_chunk(s, *, slot=0, marks=True, tensor=False):
    """tt-d-gen prefill_wait_for_arm_us: a request declined before compute sends ONE empty chunk [0, 0) (marked last when
    mark_last_chunk is on). Returns (the device calls it issued, its acks, its input)."""
    word = slot | ((MARKED | LAST) if marks else 0)
    inp = F.FakeT(torch.ones(8, 1, C // 8, dtype=torch.int64), "u32", "RM", F.MESH) if tensor else [1] * C
    n0, a0 = len(F.SYS.trace), len(s.acks)
    s.rt.prefill_chunk(inp, s.kv, slot_id=word, actual_start=0, actual_end=0, request_id=s.request)
    s.request += 1
    return [op[0] for op in F.SYS.trace[n0:]], len(s.acks) - a0, inp


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("marks", [False, True])
def test_empty_chunk_then_a_prompt_equals_cold(flag, marks):
    """The declined request's empty chunk computes and exports nothing, acks every layer, and the next prompt is
    exactly that prompt cold (also as the very first chunk after startup)."""
    layout = "cp2"
    with _env(**_base_env(layout, flag=flag)):
        s = System(flag=flag)
        ops, acks, _ = _empty_chunk(s, marks=marks)  # first thing after startup
        assert ops == [] and acks == CFG.n_layers, (ops, acks)
        assert s.prompt(A, marks=marks) == ["C"] * 4
        assert_cold(s, layout, A, what="A after an empty chunk at startup")
        s.prompt(X, marks=marks)
        ops, acks, _ = _empty_chunk(s, marks=marks)
        assert ops == [] and acks == CFG.n_layers, (ops, acks)
        s.prompt(B, marks=marks)
        assert_cold(s, layout, B, what="B after X and an empty chunk")


def test_empty_chunk_keeps_the_retained_prefix():
    """With prefix reuse, empty chunks between (and around) prompts leave the retained prefix and checkpoints usable."""
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(A)
        before = (s.rt._prefix.ret[0].length, sorted(s.rt._prefix.ret[0].pool.by_pos))
        for _ in range(3):
            assert _empty_chunk(s)[:2] == ([], CFG.n_layers)
        assert (s.rt._prefix.ret[0].length, sorted(s.rt._prefix.ret[0].pool.by_pos)) == before
        assert s.prompt(A2) == ["S", "S", "S", "C", "C", "C"]
        assert_cold(s, layout, A2, what="A2 after empty chunks")
        _empty_chunk(s, marks=False)
        assert s.prompt(A2) == ["S"] * 6
        assert_cold(s, layout, A2, what="A2 again after an unmarked empty chunk")
        assert F.SYS.late_allocs == []


def test_empty_chunk_frees_its_h2d_tensor_without_reading_it():
    with _env(**_base_env("cp2", flag=True)):
        s = System(flag=True)
        ops, acks, inp = _empty_chunk(s, tensor=True)
        assert ops == ["deallocate"] and acks == CFG.n_layers and inp.dead, ops


def test_marked_words_with_the_flag_off_behave_as_plain_words():
    """split_slot_word always runs: tt-d-gen's mark_last_chunk bits no longer trip the slot assert (runtime.py:201)."""
    layout = "cp2"
    out = []
    for marks in (False, True):
        with _isolated_sys(), _env(**_base_env(layout, flag=False)):
            s = System()
            s.prompt(A, marks=marks)
            s.prompt(A2, marks=marks)
            out.append((F.export_bytes(s.kv), F.model_state(s.rt.pf, 0, len(A2)), list(F.SYS.trace)))
    assert F.same(out[0][0], out[1][0]) and F.same(out[0][1], out[1][1]) and out[0][2] == out[1][2]


# -------------------------------------------------------------------------------------------- flag off == base commit
def _base_modules(tmp_path):
    try:
        srcs = {
            n: subprocess.check_output(
                ["git", "-C", ROOT, "show", f"{BASE}:models/demos/deepseek_v3_d_p/tt/v41/{n}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for n in ("runtime.py", "kv_export.py")
        }
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"base commit {BASE} not readable with git ({e})")
    pkg = tmp_path / "v41_base"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "prefill.py").write_text("class V41Prefill:\n    pass\n")
    for n, src in srcs.items():
        (pkg / n).write_text(src)
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module("v41_base.runtime"), importlib.import_module("v41_base.kv_export")
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("users", [1, 2])
def test_flag_off_issues_the_base_commits_device_work(tmp_path, layout, users):
    rt0, kx0 = _base_modules(tmp_path)
    runs = []
    for rt_mod, kx_mod in ((rt0, kx0), (RT, KX)):
        with _isolated_sys(), _env(**_base_env(layout, flag=False)):
            s = System(rt_mod, kx_mod, users=users)
            for slot, p in [(0, X), (0, A), (users - 1, B), (0, A2), (0, A2), (0, AP), (users - 1, LONG)]:
                s.prompt(p, slot=slot, marks=False)  # the base runtime asserts on a marked slot word
            runs.append((list(F.SYS.trace), F.export_bytes(s.kv), F.model_state(s.rt.pf, 0, len(AP)), list(s.acks)))
    (t0, e0, m0, a0), (t1, e1, m1, a1) = runs
    assert len(t0) == len(t1) and t0 == t1, next(
        (f"first difference at call {i}: {x} != {y}" for i, (x, y) in enumerate(zip(t0, t1)) if x != y), "length"
    )
    assert F.same(e0, e1) and F.same(m0, m1) and a0 == a1


# ------------------------------------------------------------------------------------------------------- mutations
def _no_zero_rows(*a, **k):
    return 0


MUTATIONS = {
    "no rows cleared on resume / on a skipped last chunk": ("zero_export_rows", _no_zero_rows),
    "no window rings re-written": ("export_windows", lambda *a, **k: None),
    "score mask not re-opened": ("open_score_masks", lambda pf, pos: 0),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_is_caught(name, monkeypatch):
    attr, fn = MUTATIONS[name]
    monkeypatch.setattr(PR, attr, fn)
    layout = "cp2"
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        failures = []
        for p in (A, B, A, A2, AP, A, LONG, SHORT_SIB):
            s.prompt(p)
            try:
                assert_cold(s, layout, p)
            except AssertionError as e:
                failures.append(str(e)[:80])
        failures += [f"late alloc {k}" for k in F.SYS.late_allocs]
    assert failures, f"mutation '{name}' went unnoticed"


def test_counters_not_set_is_caught(monkeypatch):
    def half_resume(pf, pos):
        pf.reset()  # counters left at 0: the model's own contiguity check refuses the next chunk

    monkeypatch.setattr(PR, "resume_at", half_resume)
    with _env(**_base_env("cp2", flag=True)):
        s = System(flag=True)
        s.prompt(A)
        with pytest.raises(AssertionError):  # allow-pytest.raises: no ttnn conftest
            s.prompt(A2)


def test_skipping_the_partial_last_chunk_is_caught(monkeypatch):
    """The 11-11 device failure (S.17) on the host: with the old rule (skip a matching partial last chunk) the full repeat
    of a prompt ending at E = 128 (mod 256) exports different window rings than cold."""
    real = PR.PrefixRetention
    monkeypatch.setattr(PR, "PrefixRetention", lambda **kw: real(**{**kw, "skip_partial_last": True}))
    layout = "cp2"
    P = A[: 3 * C] + toks(384, 21)
    with _env(**_base_env(layout, flag=True)):
        s = System(flag=True)
        s.prompt(X)
        s.prompt(P)
        assert s.prompt(P) == ["S"] * 4
        with pytest.raises(AssertionError, match=r"\['swa', 'win'\]"):  # allow-pytest.raises: no ttnn conftest
            assert_cold(s, layout, P)
