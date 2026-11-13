# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: Qwen36MTPContractForCausalLM with the GDN state in vLLM KV cache blocks.

A verify writes the state after each query token, recurrent state and convolution history, into the block of
mamba_state.windows, and the next step starts from the block of the count the accept walk committed. Every test runs
the class next to Qwen36ForCausalLM (ordinary execution: one token per step, gdn_decode_step on the blocks) on the
same truncated model and block ids, feeds both the same tokens, and requires for every committed count:

* the verify's target logits at the committed positions equal the ordinary decode's at the same positions (PCC and
  argmax, as the two use different attention and matmul shapes),
* the GDN state in the window block of the committed count equals the ordinary state after the same tokens.

The committed counts are forced by the test (the drafts are the ordinary greedy tokens up to the position that should
be rejected), so every count from 1 to K + 1, a rejection at every draft position, zero and partial valid drafts, and an
EOS or length cut (a count below the matched drafts) occur. The block ids follow the plugin's rule
(tests/spec_block_helpers.py) in mamba_cache_mode "none" and "align".

    MESH_DEVICE=P150x8 HF_MODEL=<Qwen3.8-27B dir> pytest models/demos/blackhole/qwen36/tests/test_mtp_gdn_blocks.py
"""

import dataclasses
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tests.spec_block_helpers import SpecBlocks
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, model_path
from models.demos.blackhole.qwen36.tests.test_gdn_block_state import (
    BMAX,
    CTX,
    N_LAYERS,
    _build,
    _decode_flags,
    _device_tensors,
    _gdn_layers,
    _parametrize,
    _prompts,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

LOGIT_PCC = 0.99


class Pair:
    """Qwen36MTPContractForCausalLM (K drafts) and Qwen36ForCausalLM on two copies of the truncated model, with one
    block layout, the same attention block ids per row and state blocks from the plugin's rule."""

    def __init__(
        self, mesh_device, monkeypatch, K, mode, pool_rows=BMAX, n_layers=N_LAYERS, bmax=BMAX, spec_sample=False
    ):
        stub_vllm_if_missing(monkeypatch)
        from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, Qwen36MTPContractForCausalLM

        monkeypatch.setattr(Qwen36MTPContractForCausalLM, "_SPEC_K", K)
        monkeypatch.setattr(Qwen36MTPContractForCausalLM, "_SUPPORTED_K", (K,))
        self.K, self.mode, self.bmax = K, mode, bmax
        # A deeper model drifts further between the verify's and the ordinary decode's attention kernels.
        self.state_pcc = 0.999 if n_layers <= N_LAYERS else 0.99
        self.worst_state_pcc = 1.0
        build = lambda mtp: Qwen36Model.from_pretrained(
            mesh_device, max_batch_size=bmax, max_seq_len=CTX, n_layers=n_layers, hf_model=model_path(), enable_mtp=mtp
        )
        self.spec_model = build(True)
        self.ref_model = build(False) if (n_layers, bmax) != (N_LAYERS, BMAX) else _build(mesh_device)
        self.spec = Qwen36MTPContractForCausalLM([self.spec_model], [self.spec_model.args], mesh_device)
        self.ref = Qwen36ForCausalLM([self.ref_model], [self.ref_model.args], mesh_device)
        probe = PluginLayout.for_model(self.spec_model, num_blocks=1)
        self.bs = probe.block_size
        self.bpu = -(-(CTX + 2 * (K + 1)) // self.bs)  # attention blocks per row
        state_blocks = probe.num_groups * (CTX // self.bs + 2 * (K + 1)) * pool_rows
        layout = PluginLayout.for_model(self.spec_model, num_blocks=1 + self.bmax * self.bpu + state_blocks)
        self.layout = layout
        self.kv_spec = self.spec.allocate_kv_cache_per_layer(layout.per_layer_specs)
        self.kv_ref = self.ref.allocate_kv_cache_per_layer(layout.per_layer_specs)
        free = list(range(1 + self.bmax * self.bpu, layout.num_blocks))
        self.spec_blocks = SpecBlocks(layout, K, mode, free=free)
        self.ref_blocks = SpecBlocks(layout, 0, mode, free=free)
        self.vocab = self.spec_model.args.vocab_size
        self.used = 2 * (2 * self.spec_model.args.gdn_dk + self.spec_model.args.gdn_dv) // 32
        self.group_of = {i: g for g, layers in enumerate(layout.gdn_groups) for i in layers}
        # The plugin's two-phase warmup of both classes, interleaved so that both allocate everything before either
        # captures a trace: a buffer allocated after a capture and alive at its replay can be overwritten.
        warm = []
        for cls, kv in ((self.ref, self.kv_ref), (self.spec, self.kv_spec)):
            prefill_kw = dict(kv_cache=kv, can_sample_on_device=False)
            sample = spec_sample and cls is self.spec  # the MTP class's verify sampler (decode_only device sampling)
            decode_kw = dict(kv_cache=kv, max_batch_size=self.bmax, num_blocks=self.bpu, can_sample_on_device=sample)
            warm.append((cls, prefill_kw, decode_kw))
        for enable_trace in (False, True):
            for cls, prefill_kw, decode_kw in warm:
                cls.warmup_model_prefill(enable_trace=enable_trace, **prefill_kw)
                cls.warmup_model_decode(enable_trace=enable_trace, **decode_kw)
        self.rows = {}  # row -> request dict

    def pages(self, row):
        return torch.arange(1 + row * self.bpu, 1 + (row + 1) * self.bpu, dtype=torch.int32)

    def table(self, rows):
        pt = torch.zeros(self.bmax, self.bpu, dtype=torch.int32)
        for r in rows:
            pt[r] = self.rows[r]["pages"]
        return pt

    def prefill(self, row, prompt, start=0, end=None, hit=None, pages=None):
        """Prefill tokens [start, end) of ``prompt`` into ``row`` on both classes (a new request when start is 0 or
        ``hit`` is given). Returns the ordinary class's logits at the last token."""
        end = len(prompt) if end is None else end
        if start == 0 or hit is not None:
            for b in (self.spec_blocks, self.ref_blocks):
                b.admit(row)
            self.rows[row] = dict(prompt=list(prompt), pages=self.pages(row) if pages is None else pages, n=end)
        req = self.rows[row]
        req["n"] = end
        final = () if end == len(prompt) else (row,)
        tokens = torch.tensor([prompt[:end]], dtype=torch.int64)
        out = []
        for cls, blocks, kv in ((self.spec, self.spec_blocks, self.kv_spec), (self.ref, self.ref_blocks, self.kv_ref)):
            state = blocks.state({row: blocks.prefill(row, start, end, hit=hit)}, self.bmax, intermediate=final)
            logits, _ = cls.prefill_forward(
                tokens,
                req["pages"].reshape(1, -1),
                kv,
                [end],
                start_pos=[start],
                empty_slots=[row],
                mamba_state=SliceRows(state, row),
            )
            out.append(logits[0, 0, : self.vocab].float())
        assert compute_pcc(out[0], out[1]) > LOGIT_PCC, "prefill logits of the two classes differ"
        return out[1]

    def ref_decode(self, row, tok):
        """One traced ordinary decode step of the request in ``row``, run in batch row 0 (the ordinary decode keeps
        its live rows in front). Returns its logits; the state after the step is recorded."""
        req = self.rows[row]
        t = torch.zeros(self.bmax, 1, dtype=torch.int32)
        p = torch.full((self.bmax,), -1, dtype=torch.int32)
        pt = torch.zeros(self.bmax, self.bpu, dtype=torch.int32)
        t[0, 0], p[0], pt[0] = tok, req["ref_n"], req["pages"]
        sel = self.ref_blocks.decode(row, req["ref_n"], 1)
        out = self.ref.decode_forward(
            t,
            p,
            page_table=pt,
            kv_cache=self.kv_ref,
            enable_trace=True,
            read_from_device=True,
            mamba_state=self.ref_blocks.state({0: sel}, self.bmax),
            **_decode_flags(),
        )[0]
        req["ref_n"] += 1
        req["ref_state"][req["ref_n"]] = self.state(self.ref_model, [w[0] for w in sel[0]])
        return out[0, -1, : self.vocab].float()

    def state(self, model, blocks):
        return {i: block_state(model, i, blocks[g]) for i, g in self.group_of.items()}

    def reference(self, row, first_token, n_tokens):
        """Ordinary greedy run of n_tokens from the row's prefill: tokens x_0 (first_token) .. x_n, the logits after
        consuming x_i (predicting x_{i+1}) and the state after each token count."""
        req = self.rows[row]
        req["ref_n"], req["ref_state"] = req["n"], {}
        xs, logits = [first_token], []
        for _ in range(n_tokens):
            lg = self.ref_decode(row, xs[-1])
            logits.append(lg)
            xs.append(int(lg.argmax()))
        req["xs"], req["ref_logits"], req["c"] = xs, logits, 0
        return xs

    def verify(self, plan, spec_mode="logits", sampling=None):
        """One verify of every row in ``plan``: row -> (valid drafts v, committed count a). Drafts are the ordinary
        greedy tokens, with draft a (if a <= v) replaced by a wrong token, so the verify's argmax rejects there; the
        test commits a tokens either way (a below the matched drafts stands for an EOS or length cut). With ``sampling``
        (spec_mode "argmax_ids", a device-sampled step), a sampled row's ids are the device sampler's draws: every
        draw whose input column follows the ordinary tokens is within the row's top-k of the ordinary logits there."""
        T = self.K + 1
        tokens = torch.zeros(self.bmax, T, dtype=torch.int32)
        pos = torch.full((self.bmax, T), -1, dtype=torch.int32)
        nv = torch.zeros(self.bmax, dtype=torch.int32)
        acc = torch.ones(self.bmax, dtype=torch.int32)
        sel = {}
        for r, (v, a) in plan.items():
            req = self.rows[r]
            c, n = req["c"], req["n"]
            row = req["xs"][c : c + 1 + v]
            if a <= v:
                row[a] = (row[a] + 1) % self.vocab
            tokens[r, : 1 + v] = torch.tensor(row, dtype=torch.int32)
            tokens[r, 1 + v :] = -1
            pos[r, : 1 + v] = torch.arange(n, n + 1 + v, dtype=torch.int32)
            nv[r], acc[r] = v, req.get("acc", 1)
            sel[r] = self.spec_blocks.decode(r, n, 1 + v)
        res = self.spec.decode_forward(
            tokens=tokens,
            start_pos=pos,
            page_table=self.table(plan),
            kv_cache=self.kv_spec,
            enable_trace=True,
            read_from_device=False,
            num_valid_drafts=nv,
            accepted_counts=acc,
            spec_mode=spec_mode,
            mamba_state=self.spec_blocks.state(sel, self.bmax),
            **({} if sampling is None else {"sampling_params": sampling}),
            **_decode_flags(),
        )
        if spec_mode == "argmax_ids":
            # read_from_device=False leaves an argmax_ids verify unread, as the plugin reads it.
            res = dataclasses.replace(res, argmax_ids=self.spec.read_decode_output(res.argmax_ids))
        if sampling is not None:
            self.check_draws(plan, res, sampling)
        for r, (v, a) in plan.items():
            req = self.rows[r]
            c, n = req["c"], req["n"]
            for t in range(a if spec_mode == "logits" else 0):
                got, want = res.logits[r, t, : self.vocab].float(), req["ref_logits"][c + t]
                pcc = compute_pcc(got, want)
                assert pcc > LOGIT_PCC, f"row {r} position {n + t}: verify logits PCC {pcc:.5f} against ordinary decode"
                assert int(got.argmax()) == int(want.argmax()) or self._near_tie(want), f"row {r} pos {n + t}: argmax"
            self.spec_blocks.commit(r, a)
            got = self.state(self.spec_model, [w[a - 1] for w in sel[r][0]])
            want = req["ref_state"][n + a]
            self.check_state(got, want, f"row {r} after {n + a} tokens")
            req["c"], req["n"], req["acc"] = c + a, n + a, a
            checked = "logits" if spec_mode == "logits" else "ids"
            logger.info(f"row {r}: {v} valid drafts, committed {a} at position {n}: {checked} and GDN state match")

    def check_draws(self, plan, res, sampling):
        """A sampled row's ids at the columns whose input tokens are the ordinary tokens (before its wrong draft) are
        within its top-k of the ordinary logits there; a greedy row's are the ordinary argmax (or a near-tie)."""
        for r, (v, a) in plan.items():
            req, k = self.rows[r], int(sampling.top_k[r])
            for t in range(min(a, v + 1)):
                want, got = req["ref_logits"][req["c"] + t], int(res.argmax_ids[r, t])
                if float(sampling.temperature[r]) > 0 and k > 1:
                    top = set(torch.topk(want, k).indices.tolist())
                    assert got in top or self._near_tie(want), f"row {r} column {t}: draw {got} outside the top-{k}"
                else:
                    assert got == int(want.argmax()) or self._near_tie(want), f"row {r} column {t}: id {got}"

    def check_state(self, got, want, label):
        """Recurrent state and history of every GDN layer against ordinary decode, PCC per device."""
        u = self.used
        for i in _gdn_layers(self.spec_model):
            for d, ((rec_g, hist_g), (rec_w, hist_w)) in enumerate(zip(got[i], want[i])):
                for what, a, b in (
                    ("recurrent state", rec_g, rec_w),
                    ("history", hist_g[:, 1:, :u], hist_w[:, 1:, :u]),
                ):
                    pcc = compute_pcc(a.float(), b.float())
                    self.worst_state_pcc = min(self.worst_state_pcc, pcc)
                    assert pcc >= self.state_pcc, f"{label}: layer {i} {what} PCC {pcc} on device {d}"

    def propose(self, committed):
        """The drafter after a step, as the plugin calls it: committed = {row: number of tokens the step committed}.
        Its drafts are not used; the verifies that follow still compare against ordinary decode."""
        T = self.K + 1
        toks = torch.full((self.bmax, T), -1, dtype=torch.int32)
        pos = torch.full((self.bmax, T), -1, dtype=torch.int32)
        acc = torch.ones(self.bmax, dtype=torch.int32)
        for r, a in committed.items():
            req = self.rows[r]
            c, n = req["c"], req["n"]  # after the commit: x_c sits at position n
            toks[r, :a] = torch.tensor(req["xs"][c - a + 1 : c + 1], dtype=torch.int32)
            pos[r, :a] = torch.arange(n - a + 1, n + 1, dtype=torch.int32)
            acc[r] = a
        out = self.spec.propose_draft_tokens(self.K, toks, pos, acc)
        assert tuple(out.draft_token_ids.shape) == (self.bmax, self.K)

    def narrow(self, rows):
        """One ordinary [B, 1] decode step of the MTP class (a verify without drafts) for ``rows``, whose last steps
        each committed one token; logits and the window-0 state are compared with ordinary decode like verify."""
        t = torch.zeros(self.bmax, 1, dtype=torch.int32)
        p = torch.full((self.bmax,), -1, dtype=torch.int32)
        sel = {}
        for r in rows:
            req = self.rows[r]
            assert req.get("acc", 1) == 1, "the plugin sends a verify after a multi-token commit"
            t[r, 0], p[r] = req["xs"][req["c"]], req["n"]
            sel[r] = self.spec_blocks.decode(r, req["n"], 1)
        out = self.spec.decode_forward(
            tokens=t,
            start_pos=p,
            page_table=self.table(rows),
            kv_cache=self.kv_spec,
            enable_trace=True,
            read_from_device=False,
            mamba_state=self.spec_blocks.state(sel, self.bmax),
            **_decode_flags(),
        )
        for r in rows:
            req = self.rows[r]
            c, n = req["c"], req["n"]
            got, want = out[r, -1, : self.vocab].float(), req["ref_logits"][c]
            pcc = compute_pcc(got, want)
            assert pcc > LOGIT_PCC, f"row {r} position {n}: narrow logits PCC {pcc:.5f} against ordinary decode"
            self.spec_blocks.commit(r, 1)
            got = self.state(self.spec_model, [w[0] for w in sel[r][0]])
            self.check_state(got, req["ref_state"][n + 1], f"row {r} after {n + 1}")
            req["c"], req["n"], req["acc"] = c + 1, n + 1, 1
            logger.info(f"row {r}: narrow step at position {n}: logits and GDN state match")

    @staticmethod
    def _near_tie(logits):
        top = logits.topk(2).values
        return float(top[0] - top[1]) <= 0.25

    def release(self, row):
        self.spec.release_request(row)
        for b in (self.spec_blocks, self.ref_blocks):
            b.release(row)
        self.rows.pop(row)


def block_state(model, layer, block):
    """Per device: (recurrent [Nv, Dk, Dv] float32, packed history [Nv, 4, 32, 32] bfloat16) of ``block``, read through
    GdnBlockState's narrow views of that block only (read_block_state of test_gdn_block_state reads whole buffers)."""
    sb, a = model.layers[layer].attention.state_blocks, model.args
    k, v, h = model.gdn_blocks._block_views((sb.rec_k, sb.rec_v, sb.hist), block)
    ks, hs = _device_tensors(k), _device_tensors(h)
    vs = _device_tensors(v) if v is not None else [None] * len(ks)
    out = []
    for kd, vd, hd in zip(ks, vs, hs):
        rec = kd.reshape(-1, a.gdn_dk, a.gdn_dv)
        if vd is not None:
            rec = torch.cat([rec, vd.reshape(-1, a.gdn_dk, a.gdn_dv)])
        out.append((rec, hd.reshape(a.gdn_nv_tp, 4, 32, 32)))
    return out


class SliceRows:
    """mamba_state of a one-row prefill call: the plugin's prefill rows are the call's rows, so row ``row`` of the
    BMAX-row state is row 0 here."""

    def __init__(self, state, row):
        self.groups = tuple(
            g.__class__(
                layer_names=g.layer_names, windows=g.windows[row : row + 1], start_blocks=g.start_blocks[row : row + 1]
            )
            for g in state.groups
        )
        self.checkpoint_positions = None
        mask = state.intermediate_prefill_mask
        self.intermediate_prefill_mask = None if mask is None else mask[row : row + 1]


def _schedule(K):
    """(valid drafts, committed count) per step: every count 1 .. K + 1 with all drafts valid (a rejection at every
    draft position, and a full accept), zero valid drafts, partial valid drafts, and a count below the matched drafts
    (an EOS or length cut)."""
    full = [(K, a) for a in range(1, K + 2)]
    partial = [(0, 1)] + [(v, a) for v in range(1, K) for a in (1, v + 1)]
    return full + partial + [(K, 1), (K, K)]


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("K", [1, 3])
def test_every_committed_count_matches_ordinary_decode(mesh_device, K, mode, monkeypatch, reset_seeds, ensure_gc):
    """Two rows (rows 1 and 3 live, rows 0 and 2 padding) through the whole schedule. In mode "align" row 1's prompt
    ends two tokens before a block boundary, so its verifies cross the boundary at every committed count."""
    pair = Pair(mesh_device, monkeypatch, K, mode)
    steps = _schedule(K)
    total = sum(a for _, a in steps) + K + 2
    prompts = _prompts(2, 2 * pair.bs, seed=K)
    lens = {1: pair.bs - 2, 3: 700}
    for r, prompt in zip((1, 3), prompts):
        first = int(pair.prefill(r, prompt[: lens[r]].tolist()).argmax())
        pair.reference(r, first, total)
    for v, a in steps:
        pair.verify({1: (v, a), 3: ((v + 1) % (K + 1), min(a, (v + 1) % (K + 1) + 1))})


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
@pytest.mark.parametrize("K, bmax", [(7, 4), (11, 2)])
def test_long_verifies_match_ordinary_decode(mesh_device, K, bmax, monkeypatch, reset_seeds, ensure_gc):
    """K = 7 and 11 (verify rows bmax x (K + 1) <= 32), mode "none": every committed count from 1 to K + 1 on one row
    matches ordinary decode."""
    pair = Pair(mesh_device, monkeypatch, K, "none", bmax=bmax)
    steps = [(K, a) for a in range(1, K + 2)] + [(0, 1), (K // 2, K // 2 + 1)]
    prompt = _prompts(1, 900, seed=K)[0].tolist()
    first = int(pair.prefill(0, prompt).argmax())
    pair.reference(0, first, sum(a for _, a in steps) + K + 2)
    for v, a in steps:
        pair.verify({0: (v, a)})


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
def test_long_prompt_with_narrow_steps(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """K = 3, mode "none": a 4300-token prompt (two traced 2048-token chunks and a tail) in row 0 and a 130-token one
    in row 2, then ordinary [B, 1] steps (verifies without drafts) mixed with verifies, the drafter running after every
    step as the plugin calls it: every step matches ordinary decode. QWEN36_PAIR_LAYERS sets the layer count."""
    K = 3
    pair = Pair(mesh_device, monkeypatch, K, "none", n_layers=int(os.environ.get("QWEN36_PAIR_LAYERS", N_LAYERS)))
    prompts = _prompts(2, 4300, seed=31)
    first = {
        0: int(pair.prefill(0, prompts[0].tolist()).argmax()),
        2: int(pair.prefill(2, prompts[1][:130].tolist()).argmax()),
    }
    for r in (0, 2):
        pair.reference(r, first[r], 40)
    for step in ["n", (K, K + 1), (K, 1), "n", (K, 2), (2, 1), "n", "n", (K, 3), (0, 1), "n"]:
        if step == "n":
            pair.narrow([0, 2])
        else:
            pair.verify({0: step, 2: step})
        pair.propose({r: 1 if step == "n" else step[1] for r in (0, 2)})
    logger.info(f"worst GDN state PCC against ordinary decode: {pair.worst_state_pcc:.6f}")


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
def test_prefix_hit_and_chunked_prefill_then_verify(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Mode "align", K = 3. Request A prefills 2 * block + 300 tokens in two chunks (the first ends on a block
    boundary and leaves the state after 2 * block tokens in its block); request B shares A's first 2 * block tokens,
    hits that block and its attention blocks, and prefills only its own tail. Both then speculate: every verify
    matches the ordinary class, which ran the same chunks and the same hit."""
    K = 3
    pair = Pair(mesh_device, monkeypatch, K, "align")
    bs = pair.bs
    shared = _prompts(1, 2 * bs, seed=11)[0].tolist()
    tails = _prompts(2, 400, seed=12)
    prompt_a = shared + tails[0][:300].tolist()
    prompt_b = shared + tails[1][:357].tolist()
    pair.prefill(0, prompt_a, 0, 2 * bs)
    hit = [w[0] for w in pair.spec_blocks.last[0]]
    first_a = int(pair.prefill(0, prompt_a, 2 * bs).argmax())
    pages_b = torch.cat([pair.pages(0)[:2], pair.pages(1)[2:]])
    first_b = int(pair.prefill(1, prompt_b, 2 * bs, hit=hit, pages=pages_b).argmax())
    for r, first in ((0, first_a), (1, first_b)):
        pair.reference(r, first, 4 * (K + 1))
    for v, a in [(K, K + 1), (K, 2), (2, 1), (K, 3)]:
        pair.verify({0: (v, a), 1: (v, max(1, a - 1))})


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
def test_abort_and_small_pool_reuse(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Mode "none", K = 3, a pool with state blocks for two requests. Request A is aborted after two verifies and its
    state blocks go back to the pool; request C takes row 0 and A's blocks (holding A's last states), prefills and
    speculates. Request B in row 2 runs throughout. Every verify of B and C matches the ordinary class."""
    K = 3
    pair = Pair(mesh_device, monkeypatch, K, "none", pool_rows=1)
    prompts = _prompts(3, 600, seed=21)
    first = {r: int(pair.prefill(r, p[:500].tolist()).argmax()) for r, p in ((0, prompts[0]), (2, prompts[1]))}
    for r in (0, 2):
        pair.reference(r, first[r], 6 * (K + 1))
    pair.verify({0: (K, K + 1), 2: (K, 2)})
    pair.verify({0: (K, 1), 2: (1, 2)})
    reused = sorted({b for cols in pair.spec_blocks.cols[0] for b in cols.values()})
    pair.release(0)
    first_c = int(pair.prefill(0, prompts[2][:333].tolist()).argmax())
    pair.reference(0, first_c, 4 * (K + 1))
    for v, a in [(K, 3), (0, 1), (K, K + 1)]:
        pair.verify({0: (v, a), 2: (v, 1)})
    taken = sorted({b for cols in pair.spec_blocks.cols[0] for b in cols.values()})
    assert taken == reused, f"C holds blocks {taken}, A held {reused}"


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
def test_narrow_step_cost(mesh_device, monkeypatch, reset_seeds, ensure_gc):
    """Wall time of one [B, 1] decode step of the MTP class (a traced verify of K + 1 columns whose drafts are all
    padding) against Qwen36ForCausalLM's traced decode step (gdn_decode_step), one live row, both after the plugin's
    warmup, host logits read back in both. QWEN36_COST_LAYERS sets the layer count (default: the truncated model).
    Not a pass / fail gate."""
    n_layers = int(os.environ.get("QWEN36_COST_LAYERS", N_LAYERS))
    K, steps = 3, 24
    pair = Pair(mesh_device, monkeypatch, K, "none", n_layers=n_layers)
    prompt = _prompts(1, 1000, seed=5)[0].tolist()
    tok = int(pair.prefill(0, prompt).argmax())
    req = pair.rows[0]
    req["ref_n"], req["ref_state"] = req["n"], {}
    # One class after the other: the two models' traces never interleave.
    times, logits = {"mtp": [], "ordinary": []}, {"mtp": [], "ordinary": []}
    toks = [tok]
    for name, cls, blocks, kv in (
        ("ordinary", pair.ref, pair.ref_blocks, pair.kv_ref),
        ("mtp", pair.spec, pair.spec_blocks, pair.kv_spec),
    ):
        for s in range(steps):
            n = req["n"] + s
            t = torch.zeros(BMAX, 1, dtype=torch.int32)
            p = torch.full((BMAX,), -1, dtype=torch.int32)
            t[0, 0], p[0] = toks[s], n
            state = blocks.state({0: blocks.decode(0, n, 1)}, BMAX)
            t0 = time.perf_counter()
            out = cls.decode_forward(
                tokens=t,
                start_pos=p,
                page_table=pair.table([0]),
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                mamba_state=state,
                **_decode_flags(),
            )
            times[name].append(time.perf_counter() - t0)
            out = out[0] if isinstance(out, tuple) else out
            logits[name].append(out[0, -1, : pair.vocab].float())
            if name == "ordinary":
                toks.append(int(logits[name][-1].argmax()))
    pccs = [compute_pcc(a, b) for a, b in zip(logits["mtp"], logits["ordinary"])]
    same = [int(a.argmax()) == int(b.argmax()) for a, b in zip(logits["mtp"], logits["ordinary"])]
    logger.info(f"NARROW STEP logits PCC per step (MTP vs ordinary): {[round(x, 5) for x in pccs]}")
    logger.info(f"NARROW STEP same argmax per step: {same}")
    med = {k: sorted(v[4:])[len(v[4:]) // 2] * 1e3 for k, v in times.items()}
    logger.info(
        f"NARROW STEP COST ({n_layers} layers, K={K}): MTP class {med['mtp']:.2f} ms, "
        f"Qwen36ForCausalLM {med['ordinary']:.2f} ms per [B, 1] step (median of {steps - 4})"
    )
    assert min(pccs) > 0.95, "a [B, 1] step of the MTP class is far from the ordinary decode step"


@torch.no_grad()
@pytest.mark.timeout(3600)
@_parametrize
@pytest.mark.parametrize("K", [3])
def test_device_sampled_verify_draws_from_the_target(mesh_device, K, monkeypatch, reset_seeds, ensure_gc):
    """A device-sampled step (``sampling_params``, spec_ids_are_sampled): row 1 samples (temperature 1, top-k 20,
    top-p 0.95), row 3 is greedy. Through the whole schedule the sampled row's ids are draws within its top-k of the
    ordinary logits, the greedy row's ids stay the argmax, and the GDN state of every committed count matches."""
    pair = Pair(mesh_device, monkeypatch, K, "none", spec_sample=True)
    steps = _schedule(K)
    total = sum(a for _, a in steps) + K + 2
    for r, prompt in zip((1, 3), _prompts(2, 1200, seed=K)):
        first = int(pair.prefill(r, prompt.tolist()).argmax())
        pair.reference(r, first, total)
    sampling = SimpleNamespace(
        temperature=[0.0, 1.0, 0.0, 0.0], top_k=[1, 20, 1, 1], top_p=[1.0, 0.95, 1.0, 1.0], seed=[None] * 4
    )
    for v, a in steps:
        pair.verify({1: (v, a), 3: (v, a)}, "argmax_ids", sampling)
