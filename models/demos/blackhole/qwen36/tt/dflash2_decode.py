# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash / DFlash2 block-diffusion drafter on the SpeculativeDecoder substrate.

Same loop, same verify (traced verify at T = block), same greedy accept, same GDN slot commit — only
the DRAFT changes: instead of K sequential MTP decode steps, ONE block forward proposes K = block-1
tokens from the target's residual taps (layers per the drafter's config, e.g. [1,16,31,46,61]).

Iteration shape (anchor p; the base has consumed through p):

    pending = argmax(base logits at p)                  # token at p+1, known
    draft   : drafter.draft(pending, C=p+1)             # block [pending, MASK*K] at p+1..p+K+1
              -> drafts[j] = candidate for p+2+j        # context = taps of positions 0..p
    verify  : ONE traced chunk over [pending, d_0..d_K-1] at p+1..p+K+1 (unchanged)
    accept  : d_j vs argmax(verify logits at p+1+j); commit [pending] + matching prefix (unchanged)
    extend  : the verify's tap rows for the committed positions -> the drafter's context KV

Context plumbing:
  * prompt   : the eager masked prefill captures the taps per chunk (Qwen36Model._dflash_tap);
               _warm_mtp_chunk (called per chunk per user by generate) fills that user's draft KV.
  * seed     : the eager verify-style seed (model.seed_spec_step) captures one tap row per user;
               _after_seed extends each user's context by its seed slot Tp[u].
  * loop     : the verify TRACE copies its taps into fixed buffers (model._dflash_tap_bufs, B*T rows);
               the reseed hook extends each user's slots p_u+1..p_u+m_u+1 from its rows after accept.
Every hook that the MTP drafter used for its own KV maintenance is repointed, so generate() runs
unmodified. B > 1 (multi-user): the drafter runs every user's block in ONE forward of B*block
user-major rows (row u*block + j), mirroring the verify bucket, so the verify trace's tap rows
feed the extend directly; each user has its own context blocks in the drafter's paged KV.
Two drafter implementations (QWEN36_DFLASH_TP, default 1):
  * DFlash2DrafterTP (tt/dflash2_tp.py): weights TP-sharded, taps consumed on device, draft + extend
    traced (captured lazily on the first loop iteration, after the verify trace).
  * DFlash2Drafter   (tt/dflash2.py): replicated, eager, host tap round-trips (the validated oracle
    form; QWEN36_DFLASH_TP=0).
"""
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, DFlash2Drafter, load_config
from models.demos.blackhole.qwen36.tt.spec_decode import SpeculativeDecoder
from models.tt_transformers.tt.common import get_block_size


def _load_embed_host(ckpt_dir):
    """The target's token embedding, bf16 on host (the drafter's block/noise embedding)."""
    from safetensors import safe_open

    for shard in sorted(os.listdir(ckpt_dir)):
        if not shard.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt") as f:
            for key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
                if key in f.keys():
                    return f.get_tensor(key).to(torch.bfloat16)
    raise KeyError("embed_tokens.weight not found in the checkpoint shards")


def drafter_weights_dir(weights_dir=None):
    return weights_dir or os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)


def default_draft_len(weights_dir=None):
    """K the checkpoint drafts (block_size - 1), or the QWEN36_DFLASH_BLOCK override - 1."""
    cfg_block = load_config(drafter_weights_dir(weights_dir))["block"]
    b = os.environ.get("QWEN36_DFLASH_BLOCK")
    # Never above the checkpoint's own block (a block-16 v1 default of 12 clamps to 8 for DFlash2).
    return (min(int(b), cfg_block) if b else cfg_block) - 1


def use_tp_drafter():
    return os.environ.get("QWEN36_DFLASH_TP", "1") != "0"


_DEBUG = os.environ.get("QWEN36_DFLASH_DEBUG", "0") == "1"
# QWEN36_DFLASH_DRAFT_TRACE=0: run the TP drafter's draft/extend EAGERLY (no draft/extend trace capture) -- an
# A/B switch for trace-interaction problems; ~2x slower drafts.
_DRAFT_TRACED = os.environ.get("QWEN36_DFLASH_DRAFT_TRACE", "1") != "0"


def _dbg(msg):
    """QWEN36_DFLASH_DEBUG=1: fence the device and log each drafter hook (localizes a device hang)."""
    if _DEBUG:
        logger.info(f"[dflash2-dbg] {msg}")


def get_drafter(model, weights_dir=None, block=None):
    """One drafter per model (weights loaded once; the draft KV is per generate)."""
    weights_dir = drafter_weights_dir(weights_dir)
    tp = use_tp_drafter()
    d = getattr(model, "_dflash2_drafter", None)
    if d is not None and (d.weights_dir != weights_dir or (block is not None and d.block != block) or d.is_tp != tp):
        # Different checkpoint / block / implementation: release the old weights (~0.9-3.6 GB/device)
        # before loading the new ones.
        d.free_weights()
        model._dflash2_drafter = d = None
    if d is None:
        logger.info(f"[dflash2] loading drafter weights from {weights_dir} (block={block or 'config'}, tp={tp})")
        embed = _load_embed_host(model.args.CKPT_DIR)
        if tp:
            import hashlib

            from models.demos.blackhole.qwen36.tt.dflash2_tp import DFlash2DrafterTP

            cache = os.environ.get("TT_CACHE_PATH")
            real = os.path.realpath(weights_dir)
            tag = hashlib.sha1(real.encode()).hexdigest()[:10]
            cache_dir = f"{cache}/dflash/{os.path.basename(real)}-{tag}" if cache else None
            d = DFlash2DrafterTP(
                model.mesh_device,
                weights_dir,
                embed,
                model.tt_ccl,
                model.args.ccl_topology(),
                model.lm_head_weight,
                lm_vocab_sharded=model._lmhead_vocab_sharded,
                block=block,
                cache_dir=cache_dir,
            )
        else:
            d = DFlash2Drafter(model.mesh_device, weights_dir, embed_host=embed, lm_head_fn=model._lm_head, block=block)
        d.weights_dir = weights_dir
        d.is_tp = tp
        logger.info(
            f"[dflash2] drafter {d.cfg['arch']} tp={tp}: block={d.block} K={d.K} taps={d.taps} causal={d.causal} "
            f"windows={d.windows} conv={d.cfg['has_conv']} selector={d.has_selector}"
        )
        model._dflash2_drafter = d
    return d


class DFlash2Decoder(SpeculativeDecoder):
    """SpeculativeDecoder with the DFlash block-diffusion drafter (lossless: same verify/accept).

    B users share one draft forward (B*block rows, user-major) and one verify replay (B*T rows);
    the drafter keeps a private paged context KV with one block range per user (the same [B, nb]
    page tables as the target). K = block-1 per user, so B*(K+1) <= 32 binds K to the batch: K=7
    (block 8) up to B=4, K=3 (block 4) at B=8.
    """

    def __init__(self, model, page_tables, draft_len=None, stop_tokens=None, weights_dir=None, sampling=None):
        # The block drafter yields tokens, not a draft distribution: greedy only (the substrate's rejection
        # sampler needs q(x) per draft position).
        assert sampling is None, "DFlash2Decoder is greedy-only (sampling=None)"
        K = default_draft_len(weights_dir) if draft_len is None else int(draft_len)
        self._weights_dir = drafter_weights_dir(weights_dir)
        self.drafter = get_drafter(model, self._weights_dir, block=K + 1)
        assert self.drafter.K == K
        self.tp = self.drafter.is_tp
        assert (
            self.tp
        ), "DFlash2Decoder runs the TP drafter (QWEN36_DFLASH_TP=1); the replicated oracle is B=1 test-only"
        # The shared model is armed for DFlash taps only inside generate() (and disarmed in its
        # finally), so a constructed-but-idle decoder leaves other decoders/prefills untouched.
        super().__init__(model, page_tables, draft_len=K, stop_tokens=stop_tokens)
        self._pt_torch = self.page_tables  # [B, nb] (normalized by the base class)
        self.ctx_len = [0] * self.B  # per user: positions whose taps the draft KV holds (asserts)
        self._armed = False

    # ------------------------------------------------------------------ prompt / seed context
    def _take_prompt_taps(self):
        taps = self.model.take_dflash_eager_taps()
        assert taps is not None, (
            "no eager taps: the masked-bucket prefill replayed a trace instead of running eagerly "
            "(QWEN36_PREFILL_BUCKET_TRACE captured?) -- the DFlash drafter needs the eager masked prefill"
        )
        return taps

    def _user_of(self, page_table_torch):
        """Which user a per-user [1, nb] page-table row belongs to (the prefill hook gets the row, not u)."""
        row = torch.as_tensor(page_table_torch).reshape(-1).to(torch.int32)
        for u in range(self.B):
            if torch.equal(self.page_tables[u], row):
                return u
        raise AssertionError(f"page-table row {row[:4].tolist()}... is not one of this decoder's {self.B} users")

    def _warm_mtp_chunk(self, feed, chunk_start, valid_len, prompt_ids, page_table_torch):
        """Per prefill chunk of one user: this chunk's taps -> that user's draft context KV at chunk_start.."""
        u = self._user_of(page_table_torch)
        taps = self._take_prompt_taps()
        _dbg(f"fill_context user={u} start={chunk_start} valid={valid_len} rows={taps[0].shape[-2]}")
        self.drafter.fill_context(taps, chunk_start, user=u)
        for t in taps:
            ttnn.deallocate(t)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg("fill_context done")
        self.ctx_len[u] = chunk_start + valid_len

    def _warm_mtp_last(self, last_hidden, first_tok, slot, page_table_tt):
        pass  # no MTP KV to warm

    def _after_seed(self, first, positions):
        """The seed consumed first[u] at positions[u] (= Tp[u]) through the verify-style seed body,
        which captured one tap row per user; those rows -> each user's draft context slot Tp[u].
        Eager and pre-capture: also compiles the R-row extend body."""
        taps = self._take_prompt_taps()  # [1,1,B,dim/tp] per tap layer, row u = user u
        _dbg(f"seed extend slots={positions}")
        self.drafter.extend_seed_rows(taps, list(positions))
        for t in taps:
            ttnn.deallocate(t)
        for u, pu in enumerate(positions):
            assert self.ctx_len[u] == pu, f"user {u}: prompt context covers {self.ctx_len[u]}, seed at {pu}"
            self.ctx_len[u] = pu + 1
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------ pre-capture warmups
    def _draft_warmup(self, pending, Hp, p):
        """Compile every draft program BEFORE the verify trace is captured. Its block KV write at
        p_u+1..p_u+K+1 is exactly what the first real draft repeats, so it is inert."""
        _dbg(f"draft warmup C={[pu + 1 for pu in p]}")
        self.drafter.draft([int(t) for t in pending], [pu + 1 for pu in p], traced=False)
        ttnn.synchronize_device(self.mesh)
        _dbg("draft warmup done")

    def _reseed_warmup(self, rows, dim_frac, dtype):
        """Compile the extend body with every row routed to the scratch block (no real slot touched)."""
        self.drafter.extend_context([0] * self.B, [0] * self.B, traced=False)
        ttnn.synchronize_device(self.mesh)

    # ------------------------------------------------------------------ loop hooks
    def _draft(self, pending, anchor_hidden, p):
        for u in range(self.B):
            # A LIVE user's context covers exactly 0..p (== p+1 rows). A FROZEN user keeps its p while
            # the extend still wrote its last committed rows, so its context runs past p+1: harmless
            # (its block rows at p+1..p+block overwrite those slots and nothing beyond them is read).
            assert self.ctx_len[u] >= p[u] + 1, f"user {u}: draft at p={p[u]} but context covers only {self.ctx_len[u]}"
        if not self._armed and _DRAFT_TRACED:
            # First loop draft: the verify trace is captured and every drafter program has run
            # eagerly, so the drafter may now capture its own draft/extend traces.
            self.drafter.arm_traces()
            self._armed = True
        _dbg(f"draft C={[pu + 1 for pu in p]}")
        out = self.drafter.draft([int(t) for t in pending], [pu + 1 for pu in p], traced=_DRAFT_TRACED)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg(f"draft done {out}")
        return out

    def _reseed_mtp_batched(self, prev_p, vfeed, committed, scratch_only=False):
        """After accept: the verify trace's tap rows for each user's committed positions
        (prev_p[u]+1 .. prev_p[u]+len(committed[u]), i.e. [pending] + accepted drafts) -> that user's
        draft context KV. Rows past the committed prefix go to the scratch block."""
        if scratch_only:
            return
        slot0 = [int(pp) + 1 for pp in prev_p]
        n = [len(c) for c in committed]
        _dbg(f"extend slot0={slot0} n={n}")
        self.drafter.extend_context(slot0, n, traced=_DRAFT_TRACED)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg("extend done")
        for u in range(self.B):
            self.ctx_len[u] = slot0[u] + n[u]

    _reseed_mtp = _reseed_mtp_batched

    # ------------------------------------------------------------------ generate
    def generate(self, prompts, max_new_tokens):
        model = self.model
        bs = get_block_size(model._paged_kv_caches)
        T = self.K + 1
        rows = self.B * T
        try:
            # Another DFlash2Decoder on this model may have replaced (and freed) our drafter via
            # get_drafter: re-resolve, and re-pin the model's tap layers to OURS.
            self.drafter = get_drafter(model, self._weights_dir, block=T)
            assert self.drafter.K == self.K and self.drafter.is_tp == self.tp
            model._dflash_tap = True
            model._dflash_tap_layers = tuple(self.drafter.taps)
            # Every drafter buffer is allocated HERE, before the verify trace is (re)captured below, so
            # no replay can land on it; force a fresh verify capture per generate for the same reason.
            self._vfy_captured = False
            if model._dflash_tap_bufs is not None and (
                len(model._dflash_tap_bufs) != len(model._dflash_tap_layers)
                or model._dflash_tap_bufs[0].shape[-2] != rows
            ):
                model.free_dflash_tap_bufs()
            if model._dflash_tap_bufs is None:
                model._dflash_tap_bufs = model.alloc_dflash_tap_bufs(rows)
            self.drafter.alloc(self._pt_torch, bs, model._dflash_tap_bufs)
            self.ctx_len = [0] * self.B
            self._armed = False
            return super().generate(prompts, max_new_tokens)
        finally:
            ttnn.synchronize_device(model.mesh_device)  # a non-blocking replay may be in flight (exception path)
            self.drafter.free()
            # The verify trace captured above bakes the tap-buf addresses freed below: release it so
            # nothing can ever replay copies into freed DRAM (the base generate already released it on
            # the normal path; this covers the exception path). A later decoder re-captures.
            model.release_verify_trace()
            self._vfy_captured = False
            # Leave the shared model as the native-MTP path expects it: no tap clones on later
            # prefills/seeds, no tap copies baked into a later (native) verify trace.
            model._free_dflash_eager_taps()
            model.free_dflash_tap_bufs()
            model._dflash_tap = False
