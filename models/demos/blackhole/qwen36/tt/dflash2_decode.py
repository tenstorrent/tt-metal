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
               _warm_mtp_chunk (called per chunk by generate) fills the draft KV for that chunk.
  * seed     : the eager seed verify (position T) captures the same way; _seed extends slot T.
  * loop     : the verify TRACE copies its taps into fixed buffers (model._dflash_tap_bufs);
               the reseed hook extends slots p+1..p+m+1 from them after each commit.
Every hook that the MTP drafter used for its own KV maintenance is repointed, so generate() runs
unmodified. Two drafter implementations (QWEN36_DFLASH_TP, default 1):
  * DFlash2DrafterTP (tt/dflash2_tp.py): weights TP-sharded, taps consumed on device, draft + extend
    traced (captured lazily on the first loop iteration, after the verify trace).
  * DFlash2Drafter   (tt/dflash2.py): replicated, eager, host tap round-trips (the validated oracle
    form; QWEN36_DFLASH_TP=0).
"""
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, DFlash2Drafter, load_config, taps_to_host
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
    """SpeculativeDecoder with the DFlash block-diffusion drafter (lossless: same verify/accept)."""

    def __init__(self, model, page_table_torch, draft_len=None, stop_tokens=None, weights_dir=None):
        K = default_draft_len(weights_dir) if draft_len is None else int(draft_len)
        self._weights_dir = drafter_weights_dir(weights_dir)
        self.drafter = get_drafter(model, self._weights_dir, block=K + 1)
        assert self.drafter.K == K
        self.tp = self.drafter.is_tp
        # The shared model is armed for DFlash taps only inside generate() (and disarmed in its
        # finally), so a constructed-but-idle decoder leaves other decoders/prefills untouched.
        super().__init__(model, page_table_torch, draft_len=K, stop_tokens=stop_tokens)
        self._pt_torch = page_table_torch
        self.ctx_len = 0  # positions whose taps the draft KV holds (debug/asserts)
        self._armed = False

    # ------------------------------------------------------------------ prompt / seed context
    def _take_prompt_taps(self):
        taps = self.model.take_dflash_eager_taps()
        assert taps is not None, (
            "no eager prompt taps: the masked-bucket prefill replayed a trace instead of running eagerly "
            "(QWEN36_PREFILL_BUCKET_TRACE captured?) — the DFlash drafter needs the eager masked prefill"
        )
        return taps

    def _warm_mtp_chunk(self, feed, chunk_start, valid_len, prompt_ids):
        """Per prefill chunk: this chunk's taps -> draft context KV at chunk_start.. (bucket rows)."""
        taps = self._take_prompt_taps()
        _dbg(f"fill_context start={chunk_start} valid={valid_len} rows={taps[0].shape[-2]}")
        if self.tp:
            self.drafter.fill_context(taps, chunk_start)
        else:
            self.drafter.fill_context(taps_to_host(self.mesh, taps), chunk_start)
        for t in taps:
            ttnn.deallocate(t)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg("fill_context done")
        self.ctx_len = chunk_start + valid_len

    def _warm_mtp_last(self, last_hidden, first_tok, slot):
        pass  # no MTP KV to warm

    def _seed(self, first, p):
        """Base seed at position p+1 (=T), then its tap -> draft context slot T. Eager: also compiles
        the extend programs (identical body/shapes to the loop's) before the verify trace exists."""
        _dbg(f"seed p={p}")
        Lp, Hp = super()._seed(first, p)
        _dbg("seed base done; extending slot")
        taps = self._take_prompt_taps()  # bucket-128 taps; row 0 is position p+1
        if self.tp:
            self.drafter.extend_from_taps(taps, p + 1, 1)
        else:
            self.drafter.extend_context(taps_to_host(self.mesh, taps, rows=self.drafter.block), p + 1, 1)
        for t in taps:
            ttnn.deallocate(t)
        assert self.ctx_len == p + 1, f"prompt context covers {self.ctx_len}, seed at {p + 1}"
        self.ctx_len = p + 2
        return Lp, Hp

    # ------------------------------------------------------------------ pre-capture warmups
    def _draft_warmup(self, pending, Hp, p):
        """Compile every draft program BEFORE the verify trace is captured. Its block KV write at
        p+1..p+K+1 is exactly what the first real draft repeats, so it is inert."""
        _dbg(f"draft warmup C={p + 1}")
        if self.tp:
            self.drafter.draft(int(pending), p + 1, traced=False)
        else:
            self.drafter.draft(int(pending), p + 1)
        ttnn.synchronize_device(self.mesh)
        _dbg("draft warmup done")

    def _reseed_warmup(self, T, dim_frac, dtype):
        pass  # the seed's extend already compiled the extend programs (same shapes)

    # ------------------------------------------------------------------ loop hooks
    def _draft(self, pending_tok, anchor_hidden, p):
        assert self.ctx_len == p + 1, f"draft at p={p} but context covers {self.ctx_len}"
        if self.tp and not self._armed:
            # First loop draft: the verify + commit traces are captured and every drafter program has
            # run eagerly, so the drafter may now capture its own draft/extend traces.
            self.drafter.arm_traces()
            self._armed = True
        _dbg(f"draft C={p + 1}")
        out = self.drafter.draft(int(pending_tok), p + 1)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg(f"draft done {out}")
        return out

    def _reseed_mtp_batched(self, slot0, vhidden, tokens, scratch_only=False):
        """After commit: the verify trace's taps for rows 0..m (positions slot0..slot0+m, i.e.
        [pending] + accepted drafts) -> draft context KV. Rows past m go to the scratch block."""
        if scratch_only:
            return
        n = len(tokens) + 1
        _dbg(f"extend slot0={slot0} n={n}")
        if self.tp:
            self.drafter.extend_context(slot0, n)
        else:
            self.drafter.extend_context(self.model.dflash_taps(), slot0, n)
        if _DEBUG:
            ttnn.synchronize_device(self.mesh)
        _dbg("extend done")
        self.ctx_len = slot0 + n

    _reseed_mtp = _reseed_mtp_batched

    # ------------------------------------------------------------------ generate
    def generate(self, prompt_ids, max_new_tokens):
        model = self.model
        bs = get_block_size(model._paged_kv_caches)
        T = self.K + 1
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
                len(model._dflash_tap_bufs) != len(model._dflash_tap_layers) or model._dflash_tap_bufs[0].shape[-2] != T
            ):
                model.free_dflash_tap_bufs()
            if model._dflash_tap_bufs is None:
                model._dflash_tap_bufs = model.alloc_dflash_tap_bufs(T)
            if self.tp:
                self.drafter.alloc(self._pt_torch, bs, model._dflash_tap_bufs)
            else:
                self.drafter.alloc_kv(self._pt_torch, bs)
            self.ctx_len = 0
            self._armed = False
            return super().generate(prompt_ids, max_new_tokens)
        finally:
            ttnn.synchronize_device(model.mesh_device)  # a non-blocking replay may be in flight (exception path)
            if self.tp:
                self.drafter.free()
            else:
                self.drafter.free_kv()
            # The verify trace captured above (and the commit traces cut against it) bake the tap-buf
            # addresses freed below: release them so nothing can ever replay copies into freed DRAM.
            # A later decoder re-captures (SpeculativeDecoder.generate checks the model's trace).
            model.release_commit_traces()
            tid = getattr(model, "_vfy_trace_id", None)
            if tid is not None:
                ttnn.release_trace(model.mesh_device, tid)
                model._vfy_trace_id = None
            self._vfy_captured = False
            # Leave the shared model as the native-MTP path expects it: no tap clones on later
            # prefills, no tap copies baked into a later (native) verify trace.
            model._free_dflash_eager_taps()
            model.free_dflash_tap_bufs()
            model._dflash_tap = False
