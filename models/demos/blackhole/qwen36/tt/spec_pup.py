# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode for the FLUX 3 Klein prompt upsampler ("pup", a Qwen3.5-9B fine-tune) at B=1 on a TP
submesh: a thin wrapper over MTPSpecEngine for one request at a time (overview: the FLUX 3 Klein pipeline README,
models/tt_dit/pipelines/flux3_klein/README.md, "Upsampler: speculative decode").

The engine side only: no sampling decisions are made here. Every verify returns, per row, the per-device top-N
(value, global id) of the bit-exact bf16 verify logits plus a bound every other token's logit is <= (``TopN``); the
caller applies its presence penalty, certifies its argmax against the bound and falls back to ``read_rows`` when it
cannot. The seed returns the full row.

* The prompt is prefilled by the pup's own TP forward (Qwen36Model._prefill_paged_tp: vision splice + M-RoPE), so
  the prefill row and the prompt KV / GDN state are exactly the non-spec path's; the engine adopts its last-layer
  hidden for the MTP prompt warm (MTPSpecEngine.adopt_prefill).
* Drafts come either from the MTP head of the BASE Qwen3.5-9B checkpoint (``attach_mtp_head``: the pup ships no
  mtp.* tensors; bf16 everywhere) through ``propose``, or from the caller (n-gram, ``use_mtp=False``: no MTP work).
* KV plan (``spec_kv_plan``): block 0 of the base and MTP caches is the null block the padded verify columns and
  the warm/capture passes write; the request's table is arange(1, nb + 1) and never names it.

Lifecycle (one process, B = 1): attach_mtp_head (optional) -> model.allocate_kv_caches([plan.num_blocks, ...]) ->
PupSpecEngine(...) -> prepare_and_capture() -> per request: prefill -> seed -> (propose ->) verify ... ->
end_request. model.free_kv_caches() shuts the engine down (ready -> False); prepare_and_capture() again after the
caches are re-allocated. Everything here is opt-in: nothing in qwen36 imports this module.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, spec_table_width, table_capacity

# The 15 tensors of a single-layer Qwen3.5 MTP head (weight_mapping.load_mtp_tensors).
MTP_TOP_KEYS = ("mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight")
MTP_LAYER_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
MTP_CACHE_SUBDIR = "mtp_qwen35_9b_base"
_BLOCK_FLOAT = (ttnn.bfloat8_b, ttnn.bfloat4_b)


# --------------------------------------------------------------------------------------------- #
# KV plan (pure)
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpecKVPlan:
    block_size: int  # 64
    table_blocks: int  # nb = ceil((max_seq_len + draft_len + 1) / block_size)  (129 for 8192, K <= 63)
    num_blocks: int  # nb + 1: base KV cache blocks to allocate; block 0 is the null block
    page_table: torch.Tensor  # int32 [1, nb] = arange(1, nb + 1); never names block 0
    capacity: int  # nb * block_size (token positions usable by prompt + generation + K lookahead)


def spec_kv_plan(max_seq_len: int, draft_len: int, block_size: int = 64) -> SpecKVPlan:
    """The pup's spec KV layout (pure, no device). The verify writes K+1 rows past the last committed position and
    its padded columns (and the warm / capture passes) write the null block 0, so the base cache gets one block
    more than the table and the table starts at block 1. The MTP cache that allocate_kv_caches derives from the
    same shape has nb + 2 blocks: the last one (nb + 1) is the reseed scratch, past every block the table names."""
    if int(max_seq_len) < 1 or int(draft_len) < 0 or int(block_size) < 1:
        raise ValueError(f"bad plan args max_seq_len={max_seq_len} draft_len={draft_len} block_size={block_size}")
    nb = spec_table_width(max_seq_len, draft_len, block_size)
    return SpecKVPlan(
        block_size=int(block_size),
        table_blocks=nb,
        num_blocks=nb + 1,
        page_table=torch.arange(1, nb + 1, dtype=torch.int32).unsqueeze(0),
        capacity=table_capacity(nb, block_size),
    )


# --------------------------------------------------------------------------------------------- #
# MTP head from the base checkpoint
# --------------------------------------------------------------------------------------------- #
def _read_safetensors_shapes(snapshot: Path, keys) -> dict:
    """{key: (dtype str, shape tuple)} from the safetensors headers only (no tensor data)."""
    import struct

    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {k: "model.safetensors" for k in keys}
    by_file = {}
    for k in keys:
        if k in weight_map:
            by_file.setdefault(weight_map[k], []).append(k)
    out = {}
    for fname, ks in by_file.items():
        path = snapshot / fname
        if not path.is_file():
            continue
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in ks:
            if k in hdr:
                out[k] = (hdr[k]["dtype"], tuple(int(d) for d in hdr[k]["shape"]))
    return out


def check_mtp_tensors(tensors: dict, *, dim: int, ref_layer_shapes: dict) -> None:
    """Raise ValueError unless ``tensors`` holds the 15 mtp.* tensors of a single-layer head of hidden size ``dim``:
    fc [dim, 2*dim], the three norms [dim], and mtp.layers.0.<suffix> shaped like ``ref_layer_shapes[suffix]``
    (the checkpoint's own first full-attention layer). Pure (torch only)."""
    missing = [k for k in MTP_TOP_KEYS if k not in tensors]
    missing += [f"mtp.layers.0.{s}" for s in MTP_LAYER_SUFFIXES if f"mtp.layers.0.{s}" not in tensors]
    if missing:
        raise ValueError(f"MTP head tensors missing: {missing}")
    extra = sorted(set(tensors) - set(MTP_TOP_KEYS) - {f"mtp.layers.0.{s}" for s in MTP_LAYER_SUFFIXES})
    if extra:
        raise ValueError(f"unexpected MTP tensors (only a single-layer head is supported): {extra[:8]}")
    want = {"mtp.fc.weight": (dim, 2 * dim)}
    for k in MTP_TOP_KEYS[1:]:
        want[k] = (dim,)
    for s in MTP_LAYER_SUFFIXES:
        if s not in ref_layer_shapes:
            raise ValueError(f"no reference shape for mtp.layers.0.{s}")
        want[f"mtp.layers.0.{s}"] = tuple(ref_layer_shapes[s])
    bad = {k: (tuple(tensors[k].shape), w) for k, w in want.items() if tuple(tensors[k].shape) != tuple(w)}
    if bad:
        raise ValueError(f"MTP tensor shapes (got, want): {bad}")
    for k, t in tensors.items():
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{k}: dtype {t.dtype}; the MTP head is loaded in bf16 (from bf16 or fp32 only)")


def _block_float_tensors(obj, prefix="mtp", depth=0, seen=None):
    """Names of every ttnn.Tensor reachable from ``obj``'s attributes (dicts / lists / models.* objects, depth <= 4)
    whose dtype is block float (bfloat8_b / bfloat4_b)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 4:
        return []
    seen.add(id(obj))
    out = []
    if isinstance(obj, ttnn.Tensor):
        try:
            return [prefix] if obj.dtype in _BLOCK_FLOAT else []
        except Exception:  # deallocated / host-only placeholder
            return []
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif type(obj).__module__.startswith("models.") and hasattr(obj, "__dict__"):
        items = list(vars(obj).items())
    else:
        return []
    for k, v in items:
        if k in ("parent", "embd", "lm_head_weight", "_lm_head", "tt_ccl", "args", "device", "mesh_device", "mesh"):
            continue  # the parent's (bf16-checked elsewhere) or not weights
        out += _block_float_tensors(v, f"{prefix}.{k}", depth + 1, seen)
    return out


def attach_mtp_head(model, base_snapshot, *, cache_dir=None) -> None:
    """Build model.mtp (Qwen36MTP, bf16 everywhere, asserts it) from the base checkpoint's 15 mtp.* tensors.
    Precondition: model.mtp is None and model.allocate_kv_caches has NOT run yet (it then allocates the MTP cache,
    num_blocks + 1 blocks, the last one the reseed scratch). Raises ValueError on missing/mis-shaped tensors.

    The head reuses the pup's embedding, LM head and final norm wiring (Qwen36MTP: mtp_use_dedicated_embeddings is
    False in both configs). Its device weights are cached under ``cache_dir`` (default
    $TT_CACHE_PATH/mtp_qwen35_9b_base_<dtype>_mesh<shape>, or next to the base snapshot without TT_CACHE_PATH): a
    directory of its own, since the tensor cache key does not encode the source checkpoint."""
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
    from models.demos.blackhole.qwen36.tt.weight_mapping import load_mtp_tensors

    if getattr(model, "mtp", None) is not None:
        raise ValueError("model already has an MTP head")
    if getattr(model, "_paged_kv_caches", None) is not None:
        raise ValueError("attach_mtp_head must run before allocate_kv_caches (it allocates the MTP cache too)")
    args = model.args
    if not getattr(args, "exact_profile", False):
        raise ValueError("attach_mtp_head is the exact (bf16) profile only: set QWEN36_EXACT_PROFILE=1")
    if int(getattr(args, "mtp_num_hidden_layers", 1) or 1) != 1 or getattr(args, "mtp_use_dedicated_embeddings", False):
        raise ValueError("only a single-layer MTP head sharing the main embedding / LM head is supported")
    snap = Path(base_snapshot)
    cfg = json.loads((snap / "config.json").read_text())
    tcfg = cfg.get("text_config", cfg)
    for name, ours in (("hidden_size", args.dim), ("vocab_size", args.vocab_size)):
        if int(tcfg.get(name, -1)) != int(ours):
            raise ValueError(f"base checkpoint {name} {tcfg.get(name)} != model {ours}")
    if int(tcfg.get("mtp_num_hidden_layers", 0) or 0) != 1 or tcfg.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("base checkpoint config does not declare a single shared-embedding MTP layer")

    tensors = load_mtp_tensors(snap)
    L = next((i for i, t in enumerate(args.attention_type_list) if t == "full_attention"), None)
    if L is None:
        raise ValueError("model has no full-attention layer to host the MTP decoder layer")
    ref_keys = [f"model.language_model.layers.{L}.{s}" for s in MTP_LAYER_SUFFIXES]
    ref_keys += [f"model.layers.{L}.{s}" for s in MTP_LAYER_SUFFIXES]
    hdr = _read_safetensors_shapes(snap, ref_keys)
    ref = {}
    for s in MTP_LAYER_SUFFIXES:
        for pre in ("model.language_model.", "model."):
            k = f"{pre}layers.{L}.{s}"
            if k in hdr:
                ref[s] = hdr[k][1]
                break
    check_mtp_tensors(tensors, dim=int(args.dim), ref_layer_shapes=ref)
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}

    if cache_dir is None:
        shape = "x".join(str(int(d)) for d in model.mesh_device.shape)
        sub = f"{MTP_CACHE_SUBDIR}_bf16_mesh{shape}"
        root = os.environ.get("TT_CACHE_PATH")
        cache_dir = (Path(root) if root else snap) / sub
    cache_dir = Path(cache_dir)
    (cache_dir / "mtp").mkdir(parents=True, exist_ok=True)  # Qwen36MTP caches under <tensor_cache_path>/mtp
    t0 = time.time()
    logger.info(f"[spec_pup] attaching the base MTP head from {snap} (cache {cache_dir})")
    mtp = Qwen36MTP(model.mesh_device, args, tensors, parent=model, tensor_cache_path=cache_dir, tt_ccl=model.tt_ccl)
    bad = _block_float_tensors(mtp)
    if bad:
        raise RuntimeError(f"MTP head has block-float device weights (bf16 only): {bad[:8]}")
    model.mtp = mtp
    model._mtp_kv_cache = None
    logger.info(f"[spec_pup] MTP head attached in {time.time() - t0:.1f}s (host layer {mtp.mtp_host_layer})")


# --------------------------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------------------------- #
TopN = tuple  # (vals: float32 [R, C], ids: int64 [R, C], thresh: float32 [R]); R = K+1, C = n_dev * top_n.
# Contract: vals[j, c] is the bit-exact bf16 logit of token ids[j, c] in verify row j (cast to fp32);
# every token id NOT in ids[j] has logit <= thresh[j]; ids are unique per row and in [0, vocab).


class PupSpecEngine:
    """B=1 greedy spec-decode engine for the pup (module docstring). Device work only; acceptance is the caller's."""

    def __init__(
        self, model, plan: SpecKVPlan, *, draft_len: int, use_mtp: bool, top_n: int = 32, draft_penalty: float = 0.0
    ) -> None:
        """No device work. use_mtp requires model.mtp. Registers as model._spec_engine's owner so that
        model.free_kv_caches() shuts the inner MTPSpecEngine down (ready -> False). draft_penalty > 0 (use_mtp):
        the MTP drafter picks under that presence penalty (set_request_penalty turns it on per request)."""
        if int(model.args.max_batch_size) != 1:
            raise ValueError(f"PupSpecEngine is B=1; model max_batch_size is {model.args.max_batch_size}")
        if model.num_devices < 2:
            raise ValueError("PupSpecEngine needs a TP mesh (the spec verify is the TP path)")
        if use_mtp and getattr(model, "mtp", None) is None:
            raise ValueError("use_mtp needs model.mtp (attach_mtp_head before allocate_kv_caches)")
        K = int(draft_len)
        if K < 1 or K + 1 > ttnn.TILE_SIZE:
            raise ValueError(f"draft_len {K} outside [1, {ttnn.TILE_SIZE - 1}]")
        if spec_table_width(int(model.args.max_seq_len), K, plan.block_size) > plan.table_blocks:
            raise ValueError(f"plan has {plan.table_blocks} blocks; max_seq_len + K + 1 needs more")
        if int(top_n) not in (32, 64):
            raise ValueError(f"top_n {top_n} not in (32, 64): each device's candidates are whole 32-wide tiles")
        if model.vocab_size % model.num_devices != 0:
            raise ValueError("the top-N verify needs a vocab-sharded LM head (vocab % num_devices == 0)")
        self.model = model
        self.plan = plan
        self.K = K
        self.T = K + 1
        self.top_n = int(top_n)
        self.n_dev = int(model.num_devices)
        self.vocab = int(model.vocab_size)
        self.use_mtp = bool(use_mtp)
        self.timing = os.environ.get("PUP_SPEC_TIMING") == "1"
        self.last_timing: dict = {}
        self.last_prepare: dict = {}
        self._Tp = None
        self._pad_rows = int(getattr(model.args, "gdn_chunk_size", 128))
        self.inner = MTPSpecEngine(
            model,
            1,
            K,
            use_mtp=self.use_mtp,
            external_prefill=True,
            verify_output="topn",
            top_n=self.top_n,
            require_fused_decode=False,
            mtp_prefill_chunk=int(os.environ.get("PUP_SPEC_MTP_PREFILL_CHUNK", "2048")),
        )
        self.inner._timing = self.timing
        if self.use_mtp and float(draft_penalty) > 0:
            self.inner.draft_penalty = float(draft_penalty)
        model._spec_engine = self.inner

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """Traces captured against the live KV caches."""
        m = self.model
        return (
            self.inner.captured
            and getattr(m, "_spec_engine", None) is self.inner
            and getattr(m, "_vfy_trace_id", None) is not None
            and m._paged_kv_caches is not None
        )

    def _check_caches(self):
        m, plan = self.model, self.plan
        if m._paged_kv_caches is None:
            raise RuntimeError("allocate_kv_caches before prepare_and_capture")
        nblk = int(m._paged_kv_caches[0][0].shape[0])
        bs = int(m._paged_kv_caches[0][0].shape[-2])
        if nblk < plan.num_blocks or bs != plan.block_size:
            raise RuntimeError(
                f"base KV cache {nblk} blocks x {bs}; the plan needs {plan.num_blocks} x {plan.block_size}"
            )
        if self.use_mtp:
            mb = int(m.mtp.attention.paged_k.shape[0])
            if mb - 1 <= int(plan.page_table.max()):
                raise RuntimeError(
                    f"MTP cache has {mb} blocks: its scratch block must be past block {int(plan.page_table.max())}"
                )

    def prepare_and_capture(self) -> dict:
        """Compile everything and capture the verify trace (+ drafter trace when use_mtp). Idempotent (no-op when ready).
        Preconditions: allocate_kv_caches done with plan.num_blocks; no request prefill since (it clobbers GDN state and
        writes KV only into null block 0). Returns {"seconds": float, "trace_bytes_per_bank": int}."""
        if self.ready:
            return dict(self.last_prepare)
        self._check_caches()
        m = self.model
        t0 = time.time()
        if self.inner.captured or self.inner.prefill_prepared:
            self.inner.shutdown()  # stale (caches re-allocated underneath it)
        m._spec_engine = self.inner
        null = torch.zeros(1, self.plan.table_blocks, dtype=torch.int32)  # warm + capture write the null block 0
        self.inner.prepare(null)
        self.inner.capture()
        ttnn.synchronize_device(m.mesh_device)
        trace = ttnn.get_memory_view(m.mesh_device, ttnn.BufferType.TRACE)
        dram = ttnn.get_memory_view(m.mesh_device, ttnn.BufferType.DRAM)
        out = {"seconds": time.time() - t0, "trace_bytes_per_bank": int(trace.total_bytes_allocated_per_bank)}
        self.last_prepare = out
        logger.info(
            f"[spec_pup] prepared + captured K={self.K} T={self.T} top_n={self.top_n} use_mtp={self.use_mtp} "
            f"nb={self.plan.table_blocks} in {out['seconds']:.1f}s; TRACE {out['trace_bytes_per_bank']} B per bank; "
            f"DRAM {dram.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank"
        )
        return out

    def set_request_penalty(self, presence_penalty: float) -> bool:
        """Before prefill: does the drafter apply its presence penalty to this request? On only when the request's
        penalty equals the captured one (a different nonzero value drafts unpenalised: acceptance only). Returns it."""
        pen = float(getattr(self.inner, "draft_penalty", 0.0))
        on = pen > 0 and float(presence_penalty) == pen
        self.inner._pen_active = on
        return on

    # ------------------------------------------------------------------ #
    def prefill(self, input_ids: torch.Tensor, vision_tokens, *, pad_id: int) -> torch.Tensor:
        """input_ids int64 [1, Tp]; vision_tokens = the device tensor from Flux3KleinUpsampler._vision_tokens or None
        (NOT freed here). Same computation as Flux3KleinUpsampler._prefill's TP branch on plan.page_table; use_mtp also
        warms the MTP prompt KV and the anchor row. Returns fp32 [vocab] logits at Tp-1. Raises RuntimeError on NaN."""
        if not self.ready:
            raise RuntimeError("prefill before prepare_and_capture (or after the caches were freed)")
        m = self.model
        ids = torch.as_tensor(input_ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, Tp], got {tuple(ids.shape)}")
        Tp = int(ids.shape[1])
        if Tp < 1 or Tp + self.T > self.plan.capacity:
            raise ValueError(f"prompt of {Tp} tokens does not fit the plan capacity {self.plan.capacity} (K={self.K})")
        prompt = ids[0, :Tp].to(torch.int64).tolist()
        pad = (-Tp) % self._pad_rows
        if pad:
            ids = torch.nn.functional.pad(ids, (0, pad), value=int(pad_id))
        self.inner.release(0)  # a new request on slot 0 (host bookkeeping only)
        self.inner.reset_draft_penalty()  # forget the last request's generated ids (no-op when off)
        t0 = time.perf_counter()
        out = m._prefill_paged_tp(
            ids, self.plan.page_table, valid_len=Tp, vision_tokens=vision_tokens, return_hidden=self.use_mtp
        )
        logits, hidden = out if self.use_mtp else (out, None)
        ttnn.synchronize_device(m.mesh_device)
        # the lm_head all-gathers the vocab row: every chip holds the full row; read ONE replica (as the upsampler)
        row = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).reshape(-1)[: self.vocab].float()
        ttnn.deallocate(logits)
        try:
            if torch.isnan(row).any():
                raise RuntimeError("NaN in the prefill logits")
            self.inner.adopt_prefill(0, prompt, self.plan.page_table, hidden)
        finally:
            if hidden is not None:
                ttnn.deallocate(hidden)
        self._Tp = Tp
        if self.timing:
            ttnn.synchronize_device(m.mesh_device)
            self.last_timing = {"prefill_ms": (time.perf_counter() - t0) * 1e3}
        return row

    def seed(self, first_token: int) -> torch.Tensor:
        """first_token at position Tp (the prompt length of the last prefill). Returns the full fp32 [vocab] row
        predicting position Tp+1. Writes KV at Tp, seeds the GDN ring; use_mtp: writes MTP slot Tp-1."""
        if self._Tp is None or 0 not in self.inner._fresh:
            raise RuntimeError("seed needs a prefill first (and only once per prefill)")
        t0 = time.perf_counter()
        tok = int(first_token)
        _, logits = self.inner.verify([[tok] * self.T], [self._Tp], [1], self.plan.page_table, read_logits=True)
        row = logits[0, 0, : self.vocab].float()
        if torch.isnan(row).any():
            raise RuntimeError("NaN in the seed logits")
        if self.timing:
            self.last_timing = {"seed_ms": (time.perf_counter() - t0) * 1e3}
        return row

    def verify(self, tokens: Sequence[int], start_pos: int, accepted_prev: int, num_valid: int) -> TopN:
        """tokens: exactly K+1 ids, [pending, d_0, ..., d_{K-1}] (columns > num_valid are padding: pass `pending`);
        tokens[j] is processed at start_pos + j; row j predicts position start_pos + j + 1.
        accepted_prev in [1, K+1] = number of tokens the previous seed()/verify() committed (1 after the seed).
        num_valid in [0, K]. Rows > num_valid are garbage."""
        toks = [int(t) for t in tokens]
        if len(toks) != self.T:
            raise ValueError(f"verify takes exactly K+1 = {self.T} tokens, got {len(toks)}")
        if any(t < 0 or t >= self.vocab for t in toks):
            raise ValueError(f"token ids outside [0, {self.vocab}): {toks}")
        if not 1 <= int(accepted_prev) <= self.T:
            raise ValueError(f"accepted_prev {accepted_prev} outside [1, {self.T}]")
        if not 0 <= int(num_valid) <= self.K:
            raise ValueError(f"num_valid {num_valid} outside [0, {self.K}]")
        if self._Tp is None or 0 in self.inner._fresh:
            raise RuntimeError("verify before seed")
        if int(start_pos) + self.T > self.plan.capacity:
            raise ValueError(f"verify window {start_pos}..{int(start_pos) + self.K} past the plan capacity")
        t0 = time.perf_counter()
        topn, _ = self.inner.verify(
            [toks], [int(start_pos)], [int(accepted_prev)], self.plan.page_table, num_valid_drafts=[int(num_valid)]
        )
        vals, ids, thresh = topn
        if torch.isnan(vals[: int(num_valid) + 1]).any():
            raise RuntimeError("NaN in the verify logits")
        if self.timing:
            t = {"verify_ms": (time.perf_counter() - t0) * 1e3}
            if self.inner.last_verify_timing is not None:
                dev_s, read_s = self.inner.last_verify_timing
                t.update(verify_device_ms=dev_s * 1e3, verify_read_ms=read_s * 1e3)
            self.last_timing = t
        return vals, ids, thresh

    def read_rows(self, rows: Sequence[int]) -> torch.Tensor:
        """fp32 [len(rows), vocab] full logit rows of the LAST verify (fallback); values bit-identical to TopN vals."""
        t0 = time.perf_counter()
        out = self.inner.read_verify_rows(rows)
        if self.timing:
            self.last_timing["read_rows_ms"] = (
                self.last_timing.get("read_rows_ms", 0.0) + (time.perf_counter() - t0) * 1e3
            )
        return out

    def propose(self, committed: Sequence[int], first_pos: int) -> list[int]:
        """use_mtp only (else raises). committed = tokens the last seed()/verify() committed, in order
        (accepted drafts + bonus, 1..K+1 ids), committed[0] at position first_pos. Returns K draft ids, or [] when the
        draft window would pass plan.capacity. Must be called exactly once after every seed()/verify() that does not
        end the request (MTP KV upkeep)."""
        if not self.use_mtp:
            raise RuntimeError("propose needs the MTP drafter (use_mtp=False: the caller drafts)")
        c = [int(t) for t in committed]
        n = len(c)
        if not 1 <= n <= self.T:
            raise ValueError(f"committed has {n} tokens; want 1..{self.T}")
        first = int(first_pos)
        p = first + n - 2  # anchor slot: the position before the last committed token (the bonus / pending)
        if p + 1 + self.K >= self.plan.capacity:
            return []  # draft window (p .. p+K-1) and next verify window would pass the plan
        t0 = time.perf_counter()
        ctok = [c + [c[-1]] * (self.T - n)]
        cpos = [[first + i for i in range(self.T)]]
        drafts = self.inner.propose(ctok, cpos, [n], self.plan.page_table)
        out = [int(x) for x in drafts.reshape(-1)[: self.K].tolist()]
        if self.timing:
            self.last_timing["propose_ms"] = (time.perf_counter() - t0) * 1e3
        return out

    def end_request(self) -> None:
        """Host bookkeeping (inner.release(0)); no device work."""
        self.inner.release(0)
        self._Tp = None

    def shutdown(self) -> None:
        """Release traces + engine buffers; ready -> False."""
        if self.inner.captured or self.inner.prefill_prepared:
            self.inner.shutdown()
        self._Tp = None
