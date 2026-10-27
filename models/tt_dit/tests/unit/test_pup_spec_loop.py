# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only (gate G2 of the pup spec-decode design): ``run_spec_greedy`` driven by a fake verifier reproduces the
plain ``presence_greedy`` loop of ``Flux3KleinUpsampler.generate_ids`` token for token, for every drafter, K in
{1, 3, 7}, both penalties, stop ids inside accepted drafts, budget truncation, empty drafts and forced top-N fallback.

The fake engine computes every logit row as a deterministic function of the token prefix (bf16 values, frequent
exact ties), checks the position bookkeeping of the engine contract (``tokens[0]`` = pending at ``start_pos``,
``accepted_prev``, padding columns, ``propose`` exactly once after every seed / verify with consistent positions),
returns garbage for rows past ``num_valid`` and derives its top-N per device the way the device does (arbitrary tie
choice)."""

import zlib
from dataclasses import asdict, fields

import pytest
import torch

from models.tt_dit.pipelines.flux3_klein.pup_spec_accept import reference_topn, run_spec_greedy
from models.tt_dit.pipelines.flux3_klein.pup_spec_drafters import make_drafter
from models.tt_dit.pipelines.flux3_klein.upsampler import (
    Flux3KleinUpsampler,
    PupInputs,
    PupSpecConfig,
    PupSpecTimings,
    PupTimings,
    presence_greedy,
    spec_config_from_env,
)

torch.set_num_threads(1)

V, N_DEV = 128, 8
STOPS = (126, 127)


class FakeModel:
    """Logits as a pure function of the prefix (the tokens through the row's input position)."""

    def __init__(self, seed: int, tie_heavy: bool, stop_after: int | None = None, copy_bias: float = 10.0):
        self.seed, self.tie_heavy, self.stop_after, self.copy_bias = seed, tie_heavy, stop_after, copy_bias
        self.prompt_len = None
        self.calls = 0

    def row(self, prefix) -> torch.Tensor:
        self.calls += 1
        prefix = [int(t) for t in prefix]
        key = zlib.crc32(torch.tensor(prefix + [self.seed], dtype=torch.int32).numpy().tobytes())
        g = torch.Generator().manual_seed(key)
        if self.tie_heavy:
            x = torch.randint(-8, 9, (V,), generator=g).float() * 0.5
        else:
            x = torch.randn(V, generator=g).to(torch.bfloat16).float() * 2.0
        # a "copy" structure (the speed300 regime): continue the most recent earlier occurrence of the last two
        # tokens when there is one (prompt lookup then drafts it), else a function of those two tokens
        a, b = (prefix[-2] if len(prefix) > 1 else 0), prefix[-1]
        nxt = None
        for i in range(len(prefix) - 3, 0, -1):
            if prefix[i - 1] == a and prefix[i] == b:
                nxt = prefix[i + 1]
                break
        if nxt is None or nxt in STOPS:
            nxt = zlib.crc32(bytes([a % 256, b % 256, self.seed % 256])) % (V - len(STOPS))
        x[nxt] += self.copy_bias
        x[list(STOPS)] -= 6.0
        if self.prompt_len is not None:
            n = len(prefix) - self.prompt_len
            if self.stop_after is not None and n >= self.stop_after:
                x[STOPS[n % 2]] += 40.0
            else:
                x[STOPS[0]] += 0.05 * n  # slowly rising, so free runs end "stop" eventually
        return x.to(torch.bfloat16).float()


def plain_greedy(model, prompt, penalty, max_new):
    """The non-spec host loop of ``generate_ids`` (presence_greedy per step, stop never appended)."""
    token = presence_greedy(model.row(prompt), [], penalty)
    generated = []
    finish = "stop" if token in STOPS else "length"
    if token not in STOPS and max_new > 0:
        generated.append(token)
        while len(generated) < max_new:
            token = presence_greedy(model.row(prompt + generated), generated, penalty)
            if token in STOPS:
                finish = "stop"
                break
            generated.append(token)
    return generated, finish


class FakeEngine:
    def __init__(self, model, prompt, K, top_n, *, use_mtp, mtp_noise=0.3, empty_every=0, seed=0):
        self.model, self.K, self.top_n, self.use_mtp = model, K, top_n, use_mtp
        self.mtp_noise, self.empty_every = mtp_noise, empty_every
        self.n_dev, self.vocab = N_DEV, V
        self.seq = list(prompt)  # tokens whose KV / state is committed
        self.block = None  # the last seed / verify block (tokens, num_valid)
        self.need_propose = False
        self.rows = None
        self.g = torch.Generator().manual_seed(seed)
        self.verifies = self.proposes = self.row_reads = 0
        self.stop_in_accepted_draft = False
        self.last_timing = {}

    def seed(self, first_token):
        assert self.block is None, "seed twice"
        self.block = ([int(first_token)], 0)
        self.need_propose = self.use_mtp
        return self.model.row(self.seq + [int(first_token)])

    def verify(self, tokens, start_pos, accepted_prev, num_valid):
        tokens = [int(t) for t in tokens]
        assert self.block is not None, "verify before seed"
        assert not self.need_propose, "propose was not called after the last seed / verify"
        assert len(tokens) == self.K + 1 and 0 <= num_valid <= self.K
        assert all(t == tokens[0] for t in tokens[num_valid + 1 :]), "padding columns must repeat pending"
        prev_tokens, prev_valid = self.block
        assert 1 <= accepted_prev <= prev_valid + 1, (accepted_prev, prev_valid)
        self.seq += prev_tokens[:accepted_prev]
        assert start_pos == len(self.seq), (start_pos, len(self.seq))
        rows = []
        for j in range(self.K + 1):
            if j <= num_valid:
                rows.append(self.model.row(self.seq + tokens[: j + 1]))
            else:  # garbage: rows past num_valid must never be used
                rows.append(torch.randn(V, generator=self.g).to(torch.bfloat16).float() * 50.0)
        self.rows = torch.stack(rows)
        self.block = (tokens, num_valid)
        self.need_propose = self.use_mtp
        self.verifies += 1
        self.last_timing = {"verify_device_ms": 1.0}
        return reference_topn(self.rows, N_DEV, self.top_n, generator=self.g)

    def read_rows(self, rows):
        self.row_reads += len(rows)
        return self.rows[list(rows)].clone()

    def propose(self, committed, first_pos):
        if not self.use_mtp:
            raise RuntimeError("propose without MTP")
        assert self.need_propose, "propose called twice"
        self.need_propose = False
        self.proposes += 1
        self.last_timing["propose_ms"] = 2.0
        prev_tokens, prev_valid = self.block
        m = len(committed) - 1
        assert 0 <= m <= prev_valid and list(committed[:-1]) == prev_tokens[1 : 1 + m], (committed, prev_tokens)
        assert first_pos == len(self.seq) + 1, (first_pos, len(self.seq))
        if any(t in STOPS for t in committed[:-1]):
            self.stop_in_accepted_draft = True
        if self.empty_every and self.proposes % self.empty_every == 0:
            return []  # the engine's "window past capacity" answer
        ctx = self.seq + prev_tokens[:1] + list(committed)
        drafts = []
        for _ in range(self.K):  # unpenalised greedy continuation with noise (an imperfect drafter)
            if float(torch.rand(1, generator=self.g)) < self.mtp_noise:
                d = int(torch.randint(0, V, (1,), generator=self.g))
            else:
                d = int(torch.argmax(self.model.row(ctx)))
            drafts.append(d)
            ctx.append(d)
        return drafts


def _prompt(seed, n=None, repeat=True):
    g = torch.Generator().manual_seed(1000 + seed)
    n = n or int(torch.randint(20, 60, (1,), generator=g))
    p = torch.randint(0, V - len(STOPS), (n,), generator=g).tolist()
    if repeat:  # a repeated span (copy regime)
        p = p + p[: n // 2]
    return p


def _run(
    drafter_kind,
    K,
    penalty,
    tie_heavy,
    seed,
    max_new=200,
    top_n=4,
    stop_after=None,
    mtp_noise=0.3,
    empty_every=0,
    repeat=True,
):
    model = FakeModel(seed, tie_heavy, stop_after)
    prompt = _prompt(seed, repeat=repeat)
    model.prompt_len = len(prompt)
    ref, ref_finish = plain_greedy(model, prompt, penalty, max_new)
    eng = FakeEngine(
        model, prompt, K, top_n, use_mtp=drafter_kind == "mtp", mtp_noise=mtp_noise, empty_every=empty_every, seed=seed
    )
    drafter = make_drafter(drafter_kind, k=K, engine=eng)
    first = presence_greedy(model.row(prompt), [], penalty)
    out, finish, stats = run_spec_greedy(
        eng,
        drafter,
        prompt_ids=prompt,
        first_token=first,
        prompt_len=len(prompt),
        penalty=penalty,
        stop_ids=STOPS,
        max_new_tokens=max_new,
    )
    assert out == ref, (drafter_kind, K, penalty, tie_heavy, seed, len(out), len(ref))
    assert finish == ref_finish
    assert stats.tokens == len(out)
    return eng, stats, ref


@pytest.mark.parametrize("drafter_kind", ["mtp", "ngram", "none"])
@pytest.mark.parametrize("K", [1, 3, 7])
@pytest.mark.parametrize("penalty", [1.5, 0.0])
@pytest.mark.parametrize("tie_heavy", [True, False])
def test_spec_loop_equals_plain_greedy(drafter_kind, K, penalty, tie_heavy):
    accepted = 0
    for seed in range(6):
        eng, stats, ref = _run(drafter_kind, K, penalty, tie_heavy, seed)
        accepted += stats.accepted
        if drafter_kind == "none":
            assert stats.accepted == 0 and stats.drafted == 0
    if drafter_kind != "none":
        assert accepted > 0


@pytest.mark.parametrize("max_new", [0, 1, 2, 3, 5, 17])
@pytest.mark.parametrize("drafter_kind", ["mtp", "ngram", "none"])
def test_budget_truncation(max_new, drafter_kind):
    for seed in range(4):
        eng, stats, ref = _run(drafter_kind, 7, 1.5, True, seed, max_new=max_new, mtp_noise=0.0)
        assert len(ref) <= max_new


@pytest.mark.parametrize("K", [3, 7])
def test_stop_inside_accepted_drafts(K):
    hit = False
    for seed in range(12):
        for stop_after in (3, 6, 10):
            eng, stats, ref = _run("mtp", K, 1.5, False, seed, stop_after=stop_after, mtp_noise=0.0)
            # the engine saw a block whose drafts held the stop at an accepted position (the run ended there)
            prev_tokens, prev_valid = eng.block
            if any(t in STOPS for t in prev_tokens[1 : prev_valid + 1]):
                hit = True
    assert hit


def test_empty_ngram_drafts_and_empty_mtp_windows():
    _, stats, _ = _run("ngram", 3, 1.5, False, 1, repeat=False)
    assert stats.empty_draft_steps > 0
    _, stats, _ = _run("mtp", 3, 1.5, False, 2, empty_every=2)
    assert stats.empty_draft_steps > 0


@pytest.mark.parametrize("drafter_kind", ["mtp", "ngram"])
def test_forced_fallback_with_top1(drafter_kind):
    reads = 0
    for seed in range(6):
        eng, stats, _ = _run(drafter_kind, 3, 1.5, True, seed, top_n=1)
        assert stats.fallback_rows == eng.row_reads
        reads += eng.row_reads
    assert reads > 0


# --- the upsampler wiring (no device) -----------------------------------------------------------------------------------


def test_spec_config_default_off_and_parsing(expect_error):
    assert spec_config_from_env({}) is None
    assert spec_config_from_env({"PUP_SPEC": "0", "PUP_SPEC_DRAFTER": "ngram"}) is None
    cfg = spec_config_from_env({"PUP_SPEC": "1"})
    assert cfg == PupSpecConfig()
    assert (cfg.drafter, cfg.k, cfg.top_n) == ("mtp", 3, 32)
    cfg = spec_config_from_env({"PUP_SPEC": "1", "PUP_SPEC_DRAFTER": "NGram", "PUP_SPEC_K": "7", "PUP_SPEC_TOPN": "64"})
    assert (cfg.drafter, cfg.k, cfg.top_n) == ("ngram", 7, 64)
    for bad in (
        {"PUP_SPEC_DRAFTER": "eagle"},
        {"PUP_SPEC_K": "0"},
        {"PUP_SPEC_K": "32"},
        {"PUP_SPEC_TOPN": "16"},
        {"PUP_SPEC_NGRAM_NMIN": "5"},
        {"PUP_SPEC_NGRAM_SOURCE": "gen"},
    ):
        with expect_error(ValueError, "PUP_SPEC_"):
            spec_config_from_env({"PUP_SPEC": "1", **bad})


def test_spec_timings_extend_pup_timings_without_changing_them():
    base = [f.name for f in fields(PupTimings)]
    spec = [f.name for f in fields(PupSpecTimings)]
    assert spec[: len(base)] == base and all(n.startswith("spec_") for n in spec[len(base) :])
    assert asdict(PupTimings()) == {k: v for k, v in asdict(PupSpecTimings()).items() if k in base}


class _SpecEngineForUpsampler(FakeEngine):
    """The FakeEngine plus the request-level surface ``_generate_spec`` uses."""

    def __init__(self, model, prompt, K, top_n, use_mtp):
        super().__init__(model, prompt, K, top_n, use_mtp=use_mtp)
        self.ready, self.captures, self.ended, self.prefills = False, 0, 0, 0
        self.prompt = list(prompt)

    def prepare_and_capture(self):
        self.ready = True
        self.captures += 1
        return {"seconds": 0.0, "trace_bytes_per_bank": 0}

    def set_request_penalty(self, presence_penalty):
        self.request_penalty = float(presence_penalty)  # the fake drafter ignores it (acceptance only)
        return False

    def prefill(self, input_ids, vision_tokens, *, pad_id):
        assert input_ids.tolist() == [self.prompt] and vision_tokens is None
        self.prefills += 1
        return self.model.row(self.prompt)

    def end_request(self):
        self.ended += 1


@pytest.mark.parametrize("drafter_kind", ["mtp", "ngram", "none"])
def test_generate_ids_spec_branch_equals_plain_loop(drafter_kind):
    seed, K = 3, 3
    model = FakeModel(seed, True)
    prompt = _prompt(seed)
    model.prompt_len = len(prompt)
    ref, ref_finish = plain_greedy(model, prompt, 1.5, 3584)

    up = object.__new__(Flux3KleinUpsampler)
    up.max_seq_len, up.load_s, up.stop_ids = 8192, 0.0, STOPS
    up._generator, up._prefill_trace_captured = None, False
    up._spec_cfg = PupSpecConfig(drafter=drafter_kind, k=K, top_n=4)
    up._spec = _SpecEngineForUpsampler(model, prompt, K, 4, use_mtp=drafter_kind == "mtp")
    up.tokenizer = type("Tok", (), {"pad_token_id": 0})()
    up.args = type("Args", (), {"image_token_id": 120, "video_token_id": 121})()
    up._dram = lambda: "fake"
    inputs = PupInputs(
        task="t2i",
        system_key="t2i",
        locked=None,
        text="",
        input_ids=torch.tensor([prompt]),
        pixel_values=None,
        image_grid_thw=None,
        num_image_tokens=0,
        presence_penalty=1.5,
    )
    ids, timings = up.generate_ids(inputs)
    assert ids == ref and timings.finish == ref_finish and timings.generated_tokens == len(ref)
    assert isinstance(timings, PupSpecTimings) and timings.spec_drafter == drafter_kind and timings.spec_k == K
    assert timings.spec_iterations == up._spec.verifies and timings.trace_captured
    assert up._spec.captures == 1 and up._spec.prefills == 1 and up._spec.ended == 1
    assert sum(timings.spec_accept_hist.values()) == timings.spec_iterations
    assert up.last_spec_stats["iterations"] == timings.spec_iterations
    assert timings.spec_engine_timing["verify_device_ms"] == 1.0  # once per verify, never double counted
    if drafter_kind == "mtp":
        assert 0 < timings.spec_engine_timing["propose_ms"] <= 2.0
    # a second request reuses the capture
    up._spec = _SpecEngineForUpsampler(model, prompt, K, 4, use_mtp=drafter_kind == "mtp")
    up._spec.ready = True
    ids2, t2 = up.generate_ids(inputs)
    assert ids2 == ref and up._spec.captures == 0 and not t2.trace_captured
