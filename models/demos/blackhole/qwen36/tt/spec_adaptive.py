# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_ADAPTIVE_K: per-step draft depth for the MTP spec engine (tt/spec_engine.py MTPSpecEngine).

The verify packs B users x (K+1) candidate rows into one 32-row tile, so the base engine (B = 8 users, K = 3) leaves the
tile half empty whenever 4 or fewer users are live. With the flag on the engine also prepares and captures a WIDE family
(QWEN36_SPEC_ADAPTIVE_K_WIDE "b:Kw", default 4 users x 5 drafts = 24 rows) and runs every step whose live engine rows all
sit below b in it: its own verify traces (b users x T = Kw+1, row buckets b, b/2, ..), drafter traces (Kw chained draft
steps), fused greedy / sampled accept devices and chain devices, on the SAME model state:

  * the GDN ring and conv-window pair are allocated once for the longest T (TPGatedDeltaNet.prepare_spec_verify T_alloc)
    with the ring laid out at the full state width B (gdn_spec_step max_users = B), so a user's accepted state block
    (mi*B + u) and its carry rows mean the same under either T; a short-T verify after a long-T one reads the carry rows
    of the long mi through gdn_spec_step(mi_max = T_alloc - 1). The wide family's ctrl pages are laid out at the state
    width too (Qwen36Model.prepare_verify_trace state_users, ChainDevice state_B);
  * paged KV, MTP KV, rec_state / conv taps, the prefill traces, the join seed traces and every per-request host set
    (_active, _fresh, _parked, _phys, ...) are shared: they are row-indexed state, not draft-length state.

Each family's draft-length state (the engine attributes in FAM_ENGINE_ATTRS and the model's ``_vfy_*`` attributes) is
swapped onto the objects only while a step / handle of that family is handled on the engine thread (``_ak_in``); at
rest the base family is live. The wide family runs with self.B = b over engine rows 0..b-1 (``_ak_rows_mode``: inside
it the runner slots ARE those engine rows; the wrapper scatters the runner block to them and gathers results back).

Runner contract (vLLM plugin, docs/SPEC_DECODE_CONTRACT.md): the block is [B, 1 + K_api], K_api = max(K, Kw) =
num_speculative_tokens, and SpecPlan.k_by_rows ((b, Kw), (B, K)) caps num_valid_drafts by live rows. A base-family step
gets columns 0..K (a fused step truncates a longer draft to K: the verified drafts stay a prefix of the proposal, which
the contract allows; a non-fused step never carries more than K drafts at more than b rows). Results go back padded to
1 + K_api columns (PLACEHOLDER past a row's count, drafts 0 past num_valid). A family switch breaks the chain: the pending
step is finished on the host in its own family and the switching step runs host-staged.

Cost model (32K KV, r5 stack; ttsim-perf per-op times anchored on the B = 8, K = 3 silicon step, within 1.3% of the
production-loop harness at every measured (B, K)): step(B, K) = verify(B, K + 1) + 0.35 + K * d1(B) + 1.03 ms with
d1 = 1.29 / 1.35 / 1.44 ms per chained draft step at B = 1 / 2 / 4, and the verify growing only ~0.45 ms per extra
candidate (its weights dominate; a T > 4 verify reads each user's KV once, QWEN36_SPEC_ADAPTIVE_K_SDPA). So each extra
draft costs ~6.5% of the K = 3 step at <= 4 users and pays only while P(all drafts so far accepted) stays above that
share of the committed tokens. On real text the per-depth acceptance is ~0.7 (32K harness: K = 4 / 5 +1..5%, K = 7
-1..5% vs K = 3 at 1 / 2 / 4 users); hence the 4:5 default.
"""

import contextlib
import time
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.tp_common import (
    engine_spec_buckets,
    spec_adaptive_k_compact_enabled,
    spec_adaptive_k_enabled,
    spec_adaptive_k_hold,
    spec_adaptive_k_wide,
    spec_buckets_flag,
    spec_fused_enabled,
    spec_hostset_trace_enabled,
)

_MISSING = object()
PLACEHOLDER = -1

# Engine attributes that belong to one draft-length family (allocated / captured per family, or describing the family's
# last step). Everything else on the engine is row-indexed request state or shared model state.
FAM_ENGINE_ATTRS = (
    "B",
    "K",
    "T",
    "_buckets",
    "_dr_bks",
    "_dr_traces",
    "_dr_trace",
    "_dr_outs",
    "_dr_standins",
    "_dr_bufs",
    "_dr_tok0",
    "_dr_pos",
    "_dr_cos",
    "_dr_sin",
    "_dr_rs_tok",
    "_dr_rs_pos",
    "_dr_rs_pt",
    "_dr_rs_cos",
    "_dr_rs_sin",
    "_dr_hostset",
    "_dr_pack",
    "_dr_eye",
    "_dr_rs_cache",
    "_dr_rs_dirty",
    "_hp_bufs",
    "_hp_buf",
    "_anchor_sel",
    "_eager_sel",
    "_mtp_pt",
    "_fa",
    "_fa_trace",
    "_fa_pt_host",
    "_fs",
    "_fs_trace",
    "_fs_standin",
    "_ch_g",
    "_ch_s",
    "_ch_trace_g",
    "_ch_trace_s",
    "_ch_standins",
    "_bk_cb",
    "_vb",
    "_vb_fell",
    "_bk_streak",
    "_bk_smax",
    "_bk_cand",
    "_join_take",
    "_join_keep",
    "_addr_snap",
    "_vfeed",
    "_fused_cache",
    "_topk_logits",
    "_last_pt",
    "_row_active",
    "_chain_prev",
    "_chain_in",
    "_chain_cand",
    "_chain_ev",
    "_chain_why",
    "_chain_dropped",
)


class KFamily:
    """One draft-length family: B users (engine rows 0..B-1) x K drafts. ``eng`` / ``model`` hold its engine attributes and
    the model's ``_vfy_*`` attributes while another family is live on the objects."""

    def __init__(self, name, B, K):
        self.name, self.B, self.K, self.T = name, int(B), int(K), int(K) + 1
        self.eng = None
        self.model = None
        self.steps = 0

    def __repr__(self):
        return f"KFamily({self.name}: B={self.B} K={self.K})"


class AdaptiveKState:
    """The engine's adaptive-K bookkeeping (engine._ak; None = QWEN36_SPEC_ADAPTIVE_K off)."""

    def __init__(self, base, wide):
        self.base, self.wide = base, wide
        self.cur = base  # the family whose attributes are live on the engine / model
        self.last = base  # family of the last verify step (a plain propose and fetch_verify_logits run in it)
        self.depth = 0  # > 0 while a wrapper runs the engine's own code (public methods do not re-dispatch then)
        self.streak = 0  # consecutive chained steps that wanted the wide family while the base family ran
        self.ready = False  # the wide family is captured
        self.switch = {"to_wide": 0, "to_base": 0}
        self.truncated = 0  # base-family fused steps whose rows carried more than K drafts (cut to K)


# ------------------------------------------------------------------------------------------------------------------- #
# Pure host helpers (no device; tests/test_spec_adaptive_k_host.py)
# ------------------------------------------------------------------------------------------------------------------- #
def ak_phys_after(phys, finishing_rows, slot_remap):
    """The runner slot -> engine row map after this step's finishing release (no change to the map) and slot_remap
    (spec_engine.slot_map_after_remap: runner row i takes the request that sat at row remap[i])."""
    if slot_remap is None:
        return list(phys)
    idx = [int(x) for x in slot_remap]
    return [phys[j] for j in idx]


def ak_step_rows(start_pos, phys):
    """Engine rows of the block's active rows (start_pos[s, 0] >= 0), sorted."""
    pos0 = torch.as_tensor(start_pos).reshape(len(phys), -1)[:, 0].tolist()
    return sorted(int(phys[s]) for s in range(len(phys)) if int(pos0[s]) >= 0)


def ak_wide_fits(rows, b):
    """True when every engine row of the step sits below the wide family's width ``b`` (and the step has a row)."""
    return bool(rows) and max(rows) < int(b)


def ak_cols(x, width, fill):
    """[B, w, ...] -> [B, width, ...]: the first ``width`` columns, padded with ``fill`` (int) when w < width."""
    x = torch.as_tensor(x)
    w = int(x.shape[1])
    if w == width:
        return x
    if w > width:
        return x[:, :width].contiguous()
    pad = torch.full((x.shape[0], width - w, *x.shape[2:]), fill, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def ak_rows(x, n, fill, col0=None):
    """[b, ...] -> [n, ...]: the first ``n`` rows, padded with ``fill`` when b < n. ``col0`` (a 2-D id block): the padded
    rows' column 0 instead. The runner treats every row it did not send as a padding row that commits count 1 and its
    column 0 (the plugin's AcceptResult rejects PLACEHOLDER inside a committed prefix), as the B-row verify returns it."""
    x = torch.as_tensor(x)
    b = int(x.shape[0])
    if b == n:
        return x
    if b > n:
        return x[:n].contiguous()
    pad = torch.full((n - b, *x.shape[1:]), fill, dtype=x.dtype)
    if col0 is not None and x.dim() == 2:
        pad[:, 0] = col0
    return torch.cat([x, pad], dim=0)


def ak_block_to_k(tokens, positions, num_valid, K):
    """A runner block [B, 1 + K_api] cut to the family's [B, 1 + K]: the first 1 + K columns, num_valid clamped to K.
    Returns (tokens, positions, num_valid, truncated rows). Cutting a row's drafts keeps them a prefix of the proposal."""
    nv = torch.as_tensor(num_valid).to(torch.int64).reshape(-1)
    cut = nv > int(K)
    return (
        ak_cols(tokens, K + 1, PLACEHOLDER),
        ak_cols(positions, K + 1, PLACEHOLDER),
        torch.where(cut, torch.full_like(nv, int(K)), nv).to(torch.as_tensor(num_valid).dtype),
        [int(u) for u in torch.nonzero(cut).reshape(-1).tolist()],
    )


def ak_uniforms_to_k(uniforms, K, K_api):
    """The plugin's pre-drawn table [B, K_api + 2] (col 0 bonus, 1..K_api accepts, K_api + 1 correction) as the table of a
    K-draft step: [bonus, accepts 1..K, correction] (spec_accept.predraw_uniforms; the dropped accept columns belong to
    drafts the step does not verify)."""
    u = torch.as_tensor(uniforms)
    if int(K) == int(K_api):
        return u
    return torch.cat([u[:, : K + 1], u[:, K_api + 1 : K_api + 2]], dim=1).contiguous()


def ak_mask_to_k(mask, K):
    """The plugin's grammar mask [b, 1 + K_api, W] (spec_masks; column c = the decision after token column c) as the mask of a
    K-draft step: columns 0..K (the dropped columns decide tokens after drafts the step does not verify)."""
    m = torch.as_tensor(mask)
    if int(m.shape[1]) == int(K) + 1:
        return m
    return m[:, : int(K) + 1].contiguous()


def ak_choose(st, rows, spec_mode, narrow, chained, max_nv, compact_ok=True):
    """(family, compact) of a step (pure: reads and updates only ``st.streak``). The wide family when the step has at most
    b live rows (b = its width); ``compact`` when they do not all sit below b yet (the engine moves them down first,
    AdaptiveKMixin._ak_compact). A step whose rows carry more drafts than the base K in a non-fused mode MUST run wide (the
    runner walks every draft column it sent; a fused step's longer rows can be cut to K instead). Otherwise a chained step
    switches base -> wide only after QWEN36_SPEC_ADAPTIVE_K_HOLD consecutive wide-fitting steps (the switch breaks the
    chain once), and a step that needs a compaction waits as well."""
    base, wide = st.base, st.wide
    n = len(rows)
    if not st.ready or narrow or n == 0 or n > wide.B:
        st.streak = 0
        return base, False
    fits = ak_wide_fits(rows, wide.B)
    if not fits and not compact_ok:
        st.streak = 0
        return base, False
    must = spec_mode != "fused_sample" and int(max_nv) > base.K
    if must or (fits and (st.last is wide or not chained)):
        st.streak = 0
        return wide, not fits
    st.streak += 1
    if st.streak >= spec_adaptive_k_hold():
        st.streak = 0
        return wide, not fits
    return base, False


def ak_compact_perm(live_rows, B):
    """A slot_remap-style permutation of engine rows that moves the ``live_rows`` to rows 0..n-1 (in order) and keeps the
    other rows' relative order: new row i takes old row idx[i]. None when they are already there."""
    live = sorted(int(u) for u in live_rows)
    if live == list(range(len(live))):
        return None
    rest = [u for u in range(B) if u not in set(live)]
    return live + rest


# ------------------------------------------------------------------------------------------------------------------- #
# Engine mixin
# ------------------------------------------------------------------------------------------------------------------- #
class AdaptiveKMixin:
    """MTPSpecEngine's QWEN36_SPEC_ADAPTIVE_K methods (every one is inert with the flag off: ``_ak`` is None)."""

    _ak = None

    # The GDN state width, the GDN ring / window T and the runner block's draft columns: stored outside the family
    # attributes (a family swap changes B / K / T, never these). An engine built without __init__ (host-test fakes) reads
    # its own B / T / K.
    @property
    def _B_state(self):
        v = self.__dict__.get("_akv_B_state")
        return self.B if v is None else v

    @_B_state.setter
    def _B_state(self, v):
        self.__dict__["_akv_B_state"] = int(v)

    @property
    def _T_state(self):
        v = self.__dict__.get("_akv_T_state")
        return self.T if v is None else v

    @_T_state.setter
    def _T_state(self, v):
        self.__dict__["_akv_T_state"] = int(v)

    @property
    def K_api(self):
        v = self.__dict__.get("_akv_K_api")
        return self.K if v is None else v

    @K_api.setter
    def K_api(self, v):
        self.__dict__["_akv_K_api"] = int(v)

    # ---- setup
    def _ak_init(self):
        """Called at the end of MTPSpecEngine.__init__: the base family is the engine as constructed (B, K); the wide
        family (b, Kw) is prepared and captured after it. ``K_api`` = the runner block's draft columns, ``_B_state`` /
        ``_T_state`` = the GDN state width / the ring and window T (both the base values with the flag off)."""
        self._ak = None
        self.K_api = self.K
        self._B_state, self._T_state = self.B, self.T
        if not spec_adaptive_k_enabled():
            return
        if self.B < 2:  # nothing narrower to widen
            logger.warning(f"[spec_adaptive] QWEN36_SPEC_ADAPTIVE_K=1 ignored: engine width B={self.B}")
            return
        bw, kw = spec_adaptive_k_wide(self.B)
        if kw <= self.K:
            raise ValueError(f"QWEN36_SPEC_ADAPTIVE_K_WIDE K={kw} must exceed the base K={self.K}")
        base, wide = KFamily("base", self.B, self.K), KFamily("wide", bw, kw)
        wide.eng = self._ak_fresh_attrs(bw, kw)
        self._ak = AdaptiveKState(base, wide)
        self.K_api = kw
        self._T_state = kw + 1
        logger.info(
            f"[spec_adaptive] QWEN36_SPEC_ADAPTIVE_K=1: base {self.B} users x K={self.K}, wide {bw} users x K={kw} "
            f"(block [B, {1 + kw}], GDN ring / window T_alloc={kw + 1})"
        )

    def _ak_fresh_attrs(self, B, K):
        """The family attributes of a not yet prepared family: what MTPSpecEngine.__init__ sets (class defaults otherwise)."""
        raw = spec_buckets_flag()
        d = {k: _MISSING for k in FAM_ENGINE_ATTRS}
        d.update(
            B=int(B),
            K=int(K),
            T=int(K) + 1,
            _buckets=(int(B),) if raw in ("", "8") else tuple(engine_spec_buckets(int(B))),
            _dr_traces={},
            _dr_outs=[],
            _dr_bufs=[],
            _dr_pos=[],
            _dr_cos=[],
            _dr_sin=[],
            _dr_trace=None,
            _dr_tok0=None,
            _dr_rs_tok=None,
            _dr_rs_pos=None,
            _dr_rs_pt=None,
            _dr_rs_cos=None,
            _dr_rs_sin=None,
            _dr_hostset=False,
            _dr_pack=None,
            _dr_eye=None,
            _dr_rs_cache=None,
            _hp_buf=None,
            _anchor_sel=None,
            _eager_sel=None,
            _mtp_pt=None,
            _join_take=None,
            _join_keep=None,
            _fa=None,
            _fa_trace=None,
            _fused_cache=None,
            _fa_pt_host=None,
            _fs=None,
            _fs_trace=None,
            _ch_g=None,
            _ch_s=None,
            _ch_trace_g=None,
            _ch_trace_s=None,
            _ch_standins=(),
            _vfeed=None,
            _topk_logits=None,
            _last_pt=None,
            _row_active=None,
        )
        return d

    # ---- family swap (engine thread only)
    def _ak_use(self, fam):
        st = self._ak
        if st.cur is fam:
            return
        cur = st.cur
        cur.eng = {k: self.__dict__.get(k, _MISSING) for k in FAM_ENGINE_ATTRS}
        cur.model = self.model.vfy_family_stash()
        for k in FAM_ENGINE_ATTRS:
            v = (fam.eng or {}).get(k, _MISSING)
            if v is _MISSING:
                self.__dict__.pop(k, None)
            else:
                self.__dict__[k] = v
        self.model.vfy_family_restore(fam.model or {})
        st.cur = fam

    @contextlib.contextmanager
    def _ak_in(self, fam):
        """Run with ``fam``'s attributes live (and the engine's own public methods not re-dispatched)."""
        st = self._ak
        if st is None:
            yield
            return
        prev = st.cur
        self._ak_use(fam)
        st.depth += 1
        try:
            yield
        finally:
            st.depth -= 1
            self._ak_use(prev)

    @contextlib.contextmanager
    def _ak_rows_mode(self):
        """Inside the wide family: the runner slots are the engine rows 0..b-1 (identity slot map)."""
        saved = (self._phys, self._phys_perm)
        self._phys, self._phys_perm = list(range(self.B)), None
        try:
            yield
        finally:
            self._phys, self._phys_perm = saved

    @contextlib.contextmanager
    def _ak_handle_ctx(self, h):
        """The family (and slot mode) a deferred handle was made in."""
        st = self._ak
        fam = getattr(h, "ak_fam", None) or st.base
        with self._ak_in(fam):
            if fam is st.wide:
                with self._ak_rows_mode():
                    yield
            else:
                yield

    def _ak_finish_any(self, h):
        """_def_finish_any of ``h`` in its own family (flag off: the plain call)."""
        if self._ak is None:
            return self._def_finish_any(h)
        with self._ak_handle_ctx(h):
            return self._def_finish_any(h)

    def _ak_inside(self):
        """True while a wrapper (or a family context) runs engine code: public methods then run as they always did."""
        return self._ak is None or self._ak.depth > 0

    # ---- prepare / capture of the wide family (after the base family's)
    def _ak_prepare_wide(self):
        """prepare_decode's tail with the flag on: allocate and compile every program of the wide family (verify trace +
        row buckets, drafter + buckets, fused greedy / sampled accept, chain) before any capture. The shared state
        (GDN ring / window at T_alloc, join selectors, seed traces' buffers, MTP KV) was prepared by the base family."""
        st = self._ak
        if st is None:
            return
        model = self.model
        if not (self.recorded_drafter and spec_hostset_trace_enabled()):
            raise RuntimeError("QWEN36_SPEC_ADAPTIVE_K needs the recorded packed drafter (QWEN36_SPEC_HOSTSET, no EAGER_DRAFTER)")
        if not (getattr(model, "_vfy_fused", False) and getattr(model, "_vfy_sharded", False)):
            raise RuntimeError(
                "QWEN36_SPEC_ADAPTIVE_K needs the fused gdn_spec_step verify (QWEN36_VERIFY_SPEC_STEP=1) and the sharded argmax"
            )
        if not self.fast_join:
            raise RuntimeError("QWEN36_SPEC_ADAPTIVE_K needs QWEN36_SPEC_FAST_JOIN=1 (the eager seed / join run B-row traces)")
        t0 = time.perf_counter()
        W = st.wide
        with self._ak_in(W):
            B, T = self.B, self.T
            pt = self._prep_pt[:B].contiguous()
            model.prepare_verify_trace(pt, T, warm_positions=[0] * B, state_users=self._B_state)
            self._dr_bks = self._bucket_plan()
            if self._dr_bks:
                model.prepare_verify_buckets(self._dr_bks)
            self._mtp_pt = self._alloc_host_replicated(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self._anchor_warmup(self._dim_frac, self._feed_dtype)
            self._reseed_warmup(B * T, self._dim_frac, self._feed_dtype)
            self._draft_warmup([0] * B, self._hp_buf, [0] * B)
            self._alloc_drafter_inputs()
            z = ttnn.zeros(
                [1, 1, B * T, self._dim_frac],
                device=self.mesh,
                dtype=self._feed_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            feed = model.spec_feed_rows(z)
            if feed is not z:
                ttnn.deallocate(z)
            for o in self._drafter_body(feed):
                ttnn.deallocate(o)
            if self._dr_bks:
                self._dr_standins = tuple(
                    ttnn.zeros(
                        [1, 1, B, 1],
                        device=self.mesh,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(self.K)
                )
                for b in self._dr_bks:
                    self._drafter_body(feed, b, list(self._dr_standins))
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(feed)
            if spec_fused_enabled():
                self._prepare_fused()
            ttnn.synchronize_device(self.mesh)
        logger.info(
            f"[spec_adaptive] wide family prepared ({W.B} users x K={W.K}, buckets {W.eng.get('_buckets')}) in "
            f"{time.perf_counter() - t0:.1f}s"
        )

    def _ak_capture_wide(self, snap):
        """capture_decode's tail with the flag on (inside its guarded scope, after every base-family capture): the wide
        family's verify trace + buckets, drafter + buckets, accept / sampled / chain traces."""
        st = self._ak
        if st is None:
            return
        W = st.wide
        model = self.model
        with self._ak_in(W):
            model.record_verify_trace()
            self._log_trace_memory(f"V{self.B}w")
            if self._dr_bks:
                model.record_verify_buckets(self._dr_bks)
            feed = model._vfy_rows_out
            self._dr_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._dr_outs = self._drafter_body(feed)
            ttnn.end_trace_capture(self.mesh, self._dr_trace, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self._log_trace_memory(f"D{self.B}w")
            for b in self._dr_bks:
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._drafter_body(feed, b, self._dr_outs)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self._dr_traces[b] = tid
                ttnn.synchronize_device(self.mesh)
                self._log_trace_memory(f"D{b}w")
            for t in self._dr_standins:
                ttnn.deallocate(t)
            self._dr_standins = ()
            if self._fa is not None:
                self._capture_fused()
            self._zero_shared_outputs()
            self._addr_snap = self._shared_addrs()
        self._check_shared_addrs(snap, "after the adaptive-K wide family")
        st.ready = True
        logger.info(f"[spec_adaptive] wide family captured ({W.B} users x K={W.K})")

    def _ak_shutdown_wide(self):
        """shutdown's part for the wide family: its traces and family buffers (the shared state goes with the base family)."""
        st = self._ak
        if st is None or st.wide.eng is None and st.cur is not st.wide:
            return
        model = self.model
        with self._ak_in(st.wide):
            model.release_verify_trace()
            model._vfy_prepared_key = None
            for name in ("_ch_trace_g", "_ch_trace_s", "_fa_trace", "_fs_trace", "_dr_trace"):
                tid = getattr(self, name, None)
                if tid is not None:
                    ttnn.release_trace(self.mesh, tid)
                    setattr(self, name, None)
            for tid in (self._dr_traces or {}).values():
                ttnn.release_trace(self.mesh, tid)
            self._dr_traces = {}
            for t in (*self._dr_outs, *self._dr_bufs, *(self._hp_bufs or {}).values(), self._hp_buf, self._mtp_pt):
                if t is not None:
                    ttnn.deallocate(t)
            for t in (self._anchor_sel, self._eager_sel):
                if t is not None:
                    ttnn.deallocate(t)
            self._fa = self._fs = self._ch_g = self._ch_s = None
            self._dr_outs, self._dr_bufs, self._hp_bufs = [], [], None
            self._hp_buf = self._mtp_pt = self._anchor_sel = self._eager_sel = None
        st.wide.eng = self._ak_fresh_attrs(st.wide.B, st.wide.K)
        st.wide.model = None
        st.ready = False

    # ---- handle metadata (the reader thread reads these, never the live attributes)
    def _ak_tag_handle(self, h, fam, outer_perm):
        """Mark a FusedHandle made inside ``fam``: its geometry, the devices / model view its read-time checks use, and how
        its results map back to the runner block (rows -> engine rows -> outer slot map; columns -> 1 + K_api)."""
        h.ak_fam = fam
        h.ak_api = dict(perm=outer_perm, B=self._B_state, W=1 + self.K_api)
        if getattr(h, "_readout", None) is not None and getattr(h, "_data", None) is not None:
            h._readout_core = h._readout
            h._readout = self._def_readout(h, h._data)

    def _ak_handle_view(self, h):
        """Attach the family geometry, the sampled accept device and a model view to a handle (engine thread, at enqueue),
        so its reader-thread checks never read the swappable attributes."""
        m = self.model
        h.geo = (self.B, self.K, self.T)
        h.fs = self._fs
        h.mview = types.SimpleNamespace(
            _vfy_T=m._vfy_T,
            _vfy_B=m._vfy_B,
            _vfy_Nv=getattr(m, "_vfy_Nv", None),
            _vfy_fused=getattr(m, "_vfy_fused", False),
            _vfy_gdn=getattr(m, "_vfy_gdn", None),
            _slot_rope_delta=getattr(m, "_slot_rope_delta", None),
            rope=m.rope,
        )
        if "_vfy_state_B" in m.__dict__:
            h.mview._vfy_state_B = m._vfy_state_B

    def _ak_api_rows(self, h, x, fill, col0=None):
        """A family-geometry engine-row tensor of handle ``h`` -> the runner block's geometry: rows padded to the state width
        (the wide family covers engine rows 0..b-1; ``col0``: a padded id row's column 0, see ak_rows), columns padded to
        1 + K_api, mapped to runner slots."""
        api = h.ak_api
        x = torch.as_tensor(x)
        x = ak_rows(x, api["B"], fill, col0)
        if x.dim() == 2:
            x = ak_cols(x, api["W"], fill)
        return self._perm_slots(x, api["perm"])

    # ---- the runner-facing wrappers
    def _ak_decode_forward(
        self,
        tokens,
        start_pos,
        *,
        num_valid_drafts,
        accepted_counts,
        page_table,
        spec_mode="argmax_ids",
        slot_remap=None,
        narrow_col0=False,
        spec_sampling=None,
        defer_read=False,
        chain_rows=None,
        chain_tip=None,
        finishing_slots=None,
    ):
        """decode_forward with the flag on (see the module doc). The block is [B, 1 + K_api] in runner slots."""
        from models.demos.blackhole.qwen36.tt.spec_engine import FusedHandle, VerifyResult, slot_map_inverse

        st = self._ak
        Bs, Ka = self._B_state, self.K_api
        tok = torch.as_tensor(tokens)
        pos = torch.as_tensor(start_pos)
        if tuple(tok.shape) != (Bs, 1 + Ka) or tuple(pos.shape) != (Bs, 1 + Ka):
            raise ValueError(
                f"adaptive-K block must be [{Bs}, {1 + Ka}], got tokens {tuple(tok.shape)} start_pos {tuple(pos.shape)}"
            )
        nv = torch.as_tensor(num_valid_drafts)
        acc = torch.as_tensor(accepted_counts)
        cr = None if chain_rows is None else torch.as_tensor(chain_rows)
        chained = cr is not None and bool(cr.bool().any())
        prev = self._def_last
        # the step's engine rows (after this step's finishing release and slot remap) -> its family
        phys = ak_phys_after(self._phys, None, slot_remap if self.host_slotmap else None)
        rows = ak_step_rows(pos, phys)
        max_nv = int(nv.max()) if nv.numel() else 0
        compact_ok = self.host_slotmap and spec_adaptive_k_compact_enabled()  # QWEN36_SPEC_ADAPTIVE_K_COMPACT
        fam, compact = ak_choose(st, rows, spec_mode, bool(narrow_col0), chained, max_nv, compact_ok)
        prev_fam = (getattr(prev, "ak_fam", None) or st.base) if prev is not None else None
        cross = chained and prev is not None and (prev_fam is not fam or compact)
        fin = None
        if prev is not None and (cross or compact or (chained and not prev.device)):
            # finished BEFORE the slot map moves (a host-ready step proposes through the map of its own step): a chain that
            # cannot continue across families (or across a row compaction) runs this step host-staged from its commit
            fin = self._ak_finish_any(prev)
        self._apply_finishing(finishing_slots)
        if slot_remap is not None:
            self.remap_slots(slot_remap)
        if cross:
            tok, pos, nv, acc = self._ak_materialize(prev, fin, cr.bool(), tok, pos, nv, acc)
            cr, chain_tip = None, None
            self._def_count_break("adaptive_k_switch")
        if compact:  # the step's rows move to engine rows 0..n-1 (after the materialize: it reads the old rows)
            self._ak_compact(ak_step_rows(pos, self._phys))
        outer_perm = self._phys_perm
        if fam is not st.last:
            st.switch["to_wide" if fam is st.wide else "to_base"] += 1
        st.last = fam
        fam.steps += 1
        kw = dict(spec_mode=spec_mode, narrow_col0=narrow_col0, defer_read=defer_read)
        if fam is st.base:
            res = self._ak_run_base(tok, pos, nv, acc, page_table, spec_sampling, cr, chain_tip, kw)
        else:
            res = self._ak_run_wide(tok, pos, nv, acc, page_table, spec_sampling, cr, chain_tip, kw)
        # results in the runner block's geometry
        if res.handle is not None and isinstance(res.handle, FusedHandle):
            self._ak_tag_handle(res.handle, fam, outer_perm)
            return res
        return self._ak_result_api(res, fam, outer_perm, slot_map_inverse, VerifyResult)

    def _ak_compact(self, step_rows):
        """Move the step's engine rows to rows 0..n-1 (the other live rows after them, then the free ones): the device slot
        remap of QWEN36_SPEC_SLOTMAP=0 (every GDN layer's ring slots, window row, rec_state and conv taps; the per-row host
        state, the M-RoPE deltas and the prefix-cache pins move with them), with the runner slot map following, so every
        runner slot keeps its request. Nothing may be in flight (the caller finished the pending step)."""
        st = self._ak
        B = self._B_state
        live = sorted((set(self._active) | set(self._chunk_reserved_rows())) - set(step_rows))
        order = sorted(int(u) for u in step_rows) + live
        idx = order + [u for u in range(B) if u not in set(order)]
        if idx == list(range(B)):
            return
        t0 = time.perf_counter()
        gens = list(self._row_gens())
        with self._ak_in(st.base):
            saved = self.host_slotmap
            self.host_slotmap = False
            try:
                self.remap_slots(idx)  # new engine row i <- old engine row idx[i]
            finally:
                self.host_slotmap = saved
        new_of = {old: new for new, old in enumerate(idx)}
        self._set_slot_map([new_of[int(r)] for r in self._phys])
        self._row_gen = [gens[j] for j in idx]
        st.compactions = getattr(st, "compactions", 0) + 1
        logger.info(
            f"[spec_adaptive] compacted engine rows {sorted(step_rows)} -> 0..{len(step_rows) - 1} "
            f"(remap {idx}) in {1e3 * (time.perf_counter() - t0):.1f} ms"
        )

    def _ak_materialize(self, prev, fin, cr, tok, pos, nv, acc):
        """The chain rows of a step that switches family, as host rows of the runner block (post-remap runner slots): column 0
        the pending step's bonus token, then its drafts (the fewer of the runner's hint and the drafts it made), positions
        from its exact p0 + count, accepted count = its count."""
        tok, pos, nv, acc = tok.clone(), pos.clone(), nv.clone(), acc.clone()
        committed, counts, p0 = fin["committed"], fin["counts"], fin["p0"]
        drafts, dnv = fin["drafts"], fin["nv"]
        gen = self._row_gens()
        for s in torch.nonzero(cr).reshape(-1).tolist():
            u = int(self._phys[s])
            if prev.gen is not None and prev.gen[u] != gen[u]:
                continue  # the row went to another request since the pending step (its output was dropped)
            if u >= int(counts.shape[0]):
                raise RuntimeError(f"adaptive-K switch: chain row {s} (engine row {u}) is outside the pending step's rows")
            c = int(counts[u])
            n = min(int(nv[s]), int(dnv[u]) if u < int(dnv.shape[0]) else 0)
            t0 = int(committed[u, c - 1])
            row = [t0] + [int(x) for x in drafts[u, :n]]
            p = int(p0[u]) + c
            tok[s] = PLACEHOLDER
            tok[s, : len(row)] = torch.tensor(row, dtype=tok.dtype)
            pos[s] = PLACEHOLDER
            pos[s, : len(row)] = torch.arange(p, p + len(row), dtype=pos.dtype)
            nv[s], acc[s] = n, c
        return tok, pos, nv, acc

    def _ak_run_base(self, tok, pos, nv, acc, page_table, spec_sampling, cr, chain_tip, kw):
        """A base-family step: the block cut to 1 + K columns (a fused step's longer rows truncated to K drafts)."""
        st = self._ak
        K, Ka = st.base.K, self.K_api
        tok_b, pos_b, nv_b, cut = ak_block_to_k(tok, pos, nv, K)
        if cut:
            if kw["spec_mode"] != "fused_sample":
                raise RuntimeError(
                    f"adaptive-K: a {kw['spec_mode']!r} step carries rows with more than K={K} drafts {cut} on the base "
                    "family (its live rows do not fit the wide family): the runner walks every draft column"
                )
            st.truncated += 1
        ss = spec_sampling
        if ss is not None:
            ss = dict(ss)
            ss["uniforms"] = ak_uniforms_to_k(ss["uniforms"], K, Ka)
            if ss.get("mask") is not None:  # device grammar mask [b, 1 + K_api, W]: the base step's 1 + K decision columns
                ss["mask"] = ak_mask_to_k(ss["mask"], K)
        with self._ak_in(st.base):
            return self.decode_forward(
                tok_b,
                pos_b,
                num_valid_drafts=nv_b,
                accepted_counts=acc,
                page_table=page_table,
                **({} if ss is None else {"spec_sampling": ss}),
                **({} if cr is None else {"chain_rows": cr, "chain_tip": chain_tip}),
                **kw,
            )

    def _ak_run_wide(self, tok, pos, nv, acc, page_table, spec_sampling, cr, chain_tip, kw):
        """A wide-family step: the runner block gathered to engine rows 0..b-1 (every active row is one of them) and run with
        the wide family live in rows mode."""
        st = self._ak
        b = st.wide.B
        g = lambda x: None if x is None else self._to_rows(torch.as_tensor(x))[0][:b].contiguous()  # noqa: E731
        tok_w, pos_w, nv_w, acc_w, pt_w = (g(x) for x in (tok, pos, nv, acc, page_table))
        ss = None if spec_sampling is None else {k: g(v) for k, v in spec_sampling.items()}
        cr_w = g(cr)
        if cr_w is not None and not bool(cr_w.bool().any()):
            cr_w = None
        with self._ak_in(st.wide), self._ak_rows_mode():
            return self.decode_forward(
                tok_w,
                pos_w,
                num_valid_drafts=nv_w,
                accepted_counts=acc_w,
                page_table=pt_w,
                **({} if ss is None else {"spec_sampling": ss}),
                **({} if cr_w is None else {"chain_rows": cr_w.bool(), "chain_tip": chain_tip}),
                **kw,
            )

    def _ak_result_api(self, res, fam, outer_perm, slot_map_inverse, VerifyResult):
        """A synchronous VerifyResult of ``fam`` in the runner block's geometry ([B, 1 + K_api] ids / committed, [B, 1 + K_api,
        ...] logits / top-k; padding columns PLACEHOLDER / zero; the wide family's rows scattered back to runner slots)."""
        st = self._ak
        Bs, W = self._B_state, 1 + self.K_api
        wide = fam is st.wide

        def api(x, fill, cols=True, col0=None):
            if x is None:
                return None
            x = torch.as_tensor(x)
            if wide:  # engine rows 0..b-1 -> every engine row -> runner slots (padding rows: count 1, a real column 0)
                x = ak_rows(x, Bs, fill, col0)
            if cols and x.dim() >= 2 and int(x.shape[1]) > 1:
                x = ak_cols(x, W, fill)
            if wide:
                x = self._perm_slots(x, outer_perm)
            return x

        out = VerifyResult(
            res.spec_mode,
            api(res.argmax_ids, PLACEHOLDER, col0=0),
            logits=api(res.logits, 0),
            topk_values=api(res.topk_values, 0),
            topk_ids=api(res.topk_ids, 0),
            accepted_token_ids=api(res.accepted_token_ids, PLACEHOLDER, col0=0),
            accepted_counts=api(res.accepted_counts, 1, cols=False),
        )
        if out.accepted_token_ids is not None:
            out.accepted_token_ids = out.accepted_token_ids.to(torch.int32)
            out.accepted_counts = out.accepted_counts.to(torch.int32)
        if out.argmax_ids is not None:
            out.argmax_ids = out.argmax_ids.to(torch.int32)
        if res.fetch_logits is not None:
            inner = res.fetch_logits

            def fetch():
                with self._ak_in(fam):
                    if wide:
                        with self._ak_rows_mode():
                            lg = inner()
                    else:
                        lg = inner()
                return api(lg, 0)

            out.fetch_logits = fetch
        return out

    def _ak_propose(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table=None, hidden=None
    ):
        """propose_draft_tokens with the flag on: [B, 1 + K_api] in, DraftResult [B, K_api] out, run in the family of the
        verify it follows (a deferred handle: the family it was made in)."""
        from models.demos.blackhole.qwen36.tt.spec_engine import DraftResult, FusedHandle

        st = self._ak
        Bs, Ka = self._B_state, self.K_api
        if int(num_drafts) != Ka:
            raise ValueError(f"num_drafts {num_drafts} != the adaptive-K block's K_api {Ka}")
        if isinstance(hidden, FusedHandle):
            # the step's own bookkeeping, in the family it was made in (a host-ready step proposes from its own commit)
            fam = getattr(hidden, "ak_fam", None) or st.base
            self._ak_finish_any(hidden)
            api = getattr(hidden, "ak_api", None) or dict(perm=self._phys_perm, B=Bs, W=1 + Ka)
            return self._ak_drafts_api(hidden.draft_result, fam, api["perm"] if fam is st.wide else None, DraftResult)
        fam = st.last
        K = fam.K
        ctok = ak_cols(committed_tokens, K + 1, PLACEHOLDER)
        cpos = ak_cols(committed_positions, K + 1, PLACEHOLDER)
        acc = torch.as_tensor(accepted_counts)
        if fam is st.base:
            with self._ak_in(fam):
                dr = self.propose_draft_tokens(K, ctok, cpos, acc, page_table, None)
            return self._ak_drafts_api(dr, fam, None, DraftResult)
        b = fam.B
        outer = self._phys_perm
        g = lambda x: None if x is None else self._to_rows(torch.as_tensor(x))[0][:b].contiguous()  # noqa: E731
        with self._ak_in(fam), self._ak_rows_mode():
            dr = self.propose_draft_tokens(K, g(ctok), g(cpos), g(acc), g(page_table), None)
        return self._ak_drafts_api(dr, fam, outer, DraftResult)

    def _ak_drafts_api(self, dr, fam, perm, DraftResult):
        """A family DraftResult (wide: engine rows 0..b-1; base: runner slots) -> [B, K_api] drafts (0 past num_valid) and
        [B] num_valid in runner slots."""
        Bs, Ka = self._B_state, self.K_api
        d, n = torch.as_tensor(dr.draft_token_ids), torch.as_tensor(dr.num_valid)
        if fam is self._ak.wide:
            d, n = ak_rows(d, Bs, 0), ak_rows(n, Bs, 0)
            d, n = self._perm_slots(d, perm), self._perm_slots(n, perm)
        return DraftResult(ak_cols(d, Ka, 0).to(torch.int32), n.to(torch.int32))

    def _ak_note_text(self):
        """Tail of the periodic fused line (flag on only)."""
        st = self._ak
        if st is None:
            return ""
        return (
            f"; adaptive-K steps base {st.base.steps} wide {st.wide.steps}, switches to wide {st.switch['to_wide']} "
            f"to base {st.switch['to_base']}, compactions {getattr(st, 'compactions', 0)}, truncated base steps "
            f"{st.truncated}"
        )
