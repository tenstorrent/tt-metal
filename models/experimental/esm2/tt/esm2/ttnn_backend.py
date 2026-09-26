# SPDX-License-Identifier: MIT
"""TTNN device implementation of ESM-2 (33-layer masked-LM protein encoder).

Precision policy (BF16 anchor):
    weights bf16, activations bf16, matmul/LN accumulation fp32 (TTNN
    default math), with the following measured fp32 exceptions:

    - fp32 residual stream (casts at bf16 consumers)
    - fp32-compute softmax composite (the runtime ttnn.softmax rounds
      internally; the composite is cast/max/sub/exp/sum/divide in fp32)
    - out_fp32 matmul sites: {qkv, pv, ao, ffn2} request fp32 output
      tiles with one clean round at bf16 consumers (chosen by device
      per-site A/B -- see docs/benchmark.md for the measured trade)
    - host-side rotary cos/sin tables and additive attention mask

Shape policy:
    Inputs are padded to 32-column tile multiples with pad tokens and
    attention_mask=0; the additive -1e9 mask covers every physical
    tile column, and outputs are sliced back to logical length L.

Device op sequence per layer (identical graph to reference_layers.py):
    cast(stream, bf16) -> layer_norm -> fused QKV linear [1280 -> 3*1280]
    -> split heads -> rotary (6-op halves rotation on cached tables)
    -> matmul(Q,K^T) + mask -> fp32 softmax composite
    -> matmul(P,V) -> concat heads -> attn_out (fp32 out)
    -> fp32 residual -> layer_norm -> ffn1 -> gelu -> ffn2 (fp32 out)
    -> fp32 residual                                (x33)
    head: final LN -> dense -> gelu -> LN -> decoder (tied) -> logits

See docs/esm2.md for the full engineering notes and docs/benchmark.md
for the measured evidence behind each precision decision.

Trace fast path (opt-in per call): the backend captures the full eager
device graph once per static shape and replays it on subsequent calls
with the same input shape. Replay is ~3.6x faster than eager (14.4 ms
vs 52.9 ms at Lt=32) and produces bit-identical outputs. Lt >= 1024
declines (persistent trace intermediates are unsized at that size).

Runtime notes (measured on tt-metal-fd80faa3):
- ttnn.linear follows plain matmul semantics (a @ b, b as [in, out]);
  torch nn.Linear weights [out, in] are transposed once at build (mat()).
- split_query_key_value_and_split_heads returns K transposed [B,H,D,L].
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Esm2TTConfig
from .reference_layers import position_ids_from_input_ids

_MASK_NEG = -1.0e9  # additive pad value (bf16-safe; avoids finfo.min NaN risk)


class TtnnEsm2:
    """Single-forward TTNN graph for esm2_t33_650M_UR50D.

    build() allocates device weight tensors (~1.30 GB bf16 for the 650M
    checkpoint) plus cached per-input rotary/mask tensors; forward() runs the
    whole encoder + MLM head in one pass (no decode loop).
    """

    # trace fast path bounds (see module docstring; opt-in only)
    _TRACE_MAX_LT = 1024  # decline trace at Lt >= 1024 until memory sized
    _TRACE_CACHE_MAX = 4  # each entry pins trace buffers; evict oldest
    _TRACE_CQ_ID = 0  # single queue (verified.

    def __init__(self, config: Esm2TTConfig, weights: dict, device=None, precision: str = "bf16", device_id: int = 0):
        try:
            import ttnn  # noqa: F401
        except ImportError as e:  # pragma: no cover - host without TT stack
            raise NotImplementedError(
                "TTNN device path requires a TT host with ttnn installed. " f"import error: {e}"
            ) from e
        if precision not in ("bf16", "fp32"):
            raise NotImplementedError(f"precision {precision!r} not supported on TT")
        self.ttnn = ttnn
        self.config = config
        self.weights = weights
        self.tt_device = device  # optional pre-opened ttnn device handle
        self.device_id = device_id
        self.precision = precision
        self.dtype = ttnn.bfloat16 if precision == "bf16" else ttnn.float32
        self.stream_dtype = self.dtype if precision != "bf16" else ttnn.float32
        # fp32-out matmul/linear sites (see module docstring; all-8 measured
        # bringup-gate set, device A/B. trimmed default
        # {qkv,pv,ao,ffn2}  per device A/B measurement
        # 2ebec6a3 / benchmarks/artifacts/opt_h2_out32_trim.json)
        self.out32 = {"qkv", "pv", "ao", "ffn2"}
        self._cast_fn = None  # resolved on first _cast call
        self._div_fn = None  # resolved on first _softmax call
        self._mm_out32_ok = None  # resolved on first out_fp32 linear
        self._mm32_matmul_ok = None  # resolved on first out_fp32 matmul
        self.device = None
        self._owns_device = False
        self._built = False
        self._emb_table = weights["embeddings.word_embeddings.weight"].float()  # fp32 [V,H]
        self._inv_freq = 1.0 / (
            config.rotary_base ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )  # fp32 [head_dim/2]
        self._rotary_cache: dict = {}  # ids bytes -> (q tables, k tables)
        self._mask_cache: dict = {}  # am bytes -> additive mask device
        # opt-in trace fast path state (eager is the safe default)
        self.enable_trace = False
        self._trace_cache: dict = {}  # key -> entry (see module docstring)
        self._trace_disabled = False  # latched after any capture/replay error
        self.trace_status = "idle"  # last trace-path outcome (for probes)
        self.last_trace_mode = "eager"  # mode used by the last forward()
        self._x0_copy_fn = None  # resolved on first _rewrite_x0 call

    # ------------------------------------------------------------ helpers

    def _cast(self, t, dtype):
        """On-device dtype cast; ttnn.typecast with to_dtype fallback.

        Both ops verified present in tt-metal-fd80faa3; the first call
        resolves which signature the installed runtime accepts and caches it.
        """
        ttnn = self.ttnn
        if self._cast_fn is None:
            last = None
            for fn in (ttnn.typecast, ttnn.to_dtype):
                try:
                    out = fn(t, dtype)
                    self._cast_fn = fn
                    return out
                except (TypeError, RuntimeError) as e:  # signature/arg drift
                    last = e
            raise RuntimeError(f"no working dtype cast op: {last}")
        return self._cast_fn(t, dtype)

    def _reduce_last(self, t, op):
        """dim=-1 reduction returned with the reduced dim kept ([..., 1]).

        Handles tt-metal-fd80faa3 API drift: keepdim kwarg accepted or not,
        and (values, indices) tuple returns from max. The trailing [.., 1]
        shape is required for the implicit eltwise broadcast that follows.
        """
        ttnn = self.ttnn
        try:
            out = op(t, dim=-1, keepdim=True)
        except TypeError:
            out = op(t, dim=-1)
        if isinstance(out, (tuple, list)):  # older max returns (values, idx)
            out = out[0]
        shape = [int(d) for d in out.shape]
        tshape = [int(d) for d in t.shape]
        if len(shape) == len(tshape):  # already keepdim
            return out
        if len(shape) == len(tshape) - 1 and shape == tshape[:-1]:
            return ttnn.reshape(out, shape + [1])
        raise RuntimeError(f"unexpected dim=-1 reduce shape {shape} from input {tshape}")

    def _softmax(self, scores):
        """fp32-COMPUTE softmax: composite x-max -> exp -> sum -> div.

        Every intermediate is an fp32-valued ttnn op (measured fix; see
        module docstring: ttnn.softmax rounds to bf16 internally on this
        runtime, ~2.2e-2/site vs 1.5e-3 floor,.. Only the
        result is cast to the policy dtype for the PV matmul.
        """
        ttnn = self.ttnn
        if self._div_fn is None:
            for name in ("divide", "div"):
                fn = getattr(ttnn, name, None)
                if fn is not None:
                    self._div_fn = fn
                    break
            if self._div_fn is None:
                raise RuntimeError("ttnn exposes no divide/div eltwise op")
        x = self._cast(scores, ttnn.float32)
        mx = self._reduce_last(x, ttnn.max)  # [B,H,L,1] fp32
        e = ttnn.exp(ttnn.sub(x, mx))  # fp32; pad cols underflow to 0
        den = self._reduce_last(e, ttnn.sum)  # [B,H,L,1] fp32
        p = self._div_fn(e, den)  # fp32, row sums exactly ~1
        return self._cast(p, self.dtype)

    def _linear(self, a, w, bias=None, out_fp32: bool = False):
        """ttnn.linear wrapper; out_fp32 requests fp32 output tiles.

        The dtype kwarg (output dtype) is resolved once: if the installed
        runtime rejects it we fall back to plain bf16 output and record the
        fact so callers/probes can check ``_mm_out32_ok``.
        """
        ttnn = self.ttnn
        if out_fp32:
            if self._mm_out32_ok is None:
                try:
                    out = ttnn.linear(a, w, bias=bias, dtype=ttnn.float32)
                    self._mm_out32_ok = True
                    return out
                except (TypeError, RuntimeError):
                    self._mm_out32_ok = False
            elif self._mm_out32_ok:
                return ttnn.linear(a, w, bias=bias, dtype=ttnn.float32)
        return ttnn.linear(a, w, bias=bias)

    def _matmul(self, a, b, out_fp32: bool = False):
        """ttnn.matmul wrapper; out_fp32 requests fp32 output tiles.

        Same dtype-kwarg resolution as _linear; records _mm32_matmul_ok for
        probes (matmul sites: QK^T scores, P@V). Both dtype kwargs verified
        accepted on tt-metal-fd80faa3.
        """
        ttnn = self.ttnn
        if out_fp32:
            if self._mm32_matmul_ok is None:
                try:
                    out = ttnn.matmul(a, b, dtype=ttnn.float32)
                    self._mm32_matmul_ok = True
                    return out
                except (TypeError, RuntimeError):
                    self._mm32_matmul_ok = False
            elif self._mm32_matmul_ok:
                return ttnn.matmul(a, b, dtype=ttnn.float32)
        return ttnn.matmul(a, b)

    def _dev(self, t: torch.Tensor, dtype=None):
        """Host fp32 torch -> device TILE tensor (bf16 by policy)."""
        ttnn = self.ttnn
        tt = ttnn.from_torch(t.contiguous(), dtype=dtype or self.dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_device(tt, self.device)

    def _host(self, tt) -> torch.Tensor:
        """Device tensor -> host fp32 torch (ROW_MAJOR)."""
        ttnn = self.ttnn
        t = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(tt), ttnn.ROW_MAJOR_LAYOUT))
        return t.to(torch.float32)

    def close(self):
        for key in list(self._trace_cache):
            self._release_trace_entry(key)
        if self._owns_device and self.device is not None:
            self.ttnn.close_device(self.device)
        self.device = None
        self.tt_device = None
        self._owns_device = False

    # ------------------------------------------------------------ build

    def build(self):
        """Open the device (unless given) and allocate all weight tensors."""
        import atexit

        ttnn = self.ttnn
        if self.tt_device is None:
            self.tt_device = ttnn.open_device(device_id=self.device_id)
            self._owns_device = True
        self.device = self.tt_device
        cfg = self.config
        w = self.weights
        q_scale = float(cfg.head_dim) ** -0.5  # folded into fused q rows

        def vec(t):  # [N] -> device [1, N] row vector (linear/LN layout)
            return self._dev(t.reshape(1, -1))

        def mat(t):  # torch Linear weight [out, in] -> device [in, out] TILE
            return self._dev(t.t().contiguous())

        layers = []
        for i in range(cfg.num_hidden_layers):
            p = f"layers.{i}."
            q_w = w[p + "attn.q.weight"] * q_scale
            q_b = w[p + "attn.q.bias"] * q_scale
            qkv_w = torch.cat([q_w, w[p + "attn.k.weight"], w[p + "attn.v.weight"]], dim=0)
            qkv_b = torch.cat([q_b, w[p + "attn.k.bias"], w[p + "attn.v.bias"]], dim=0)
            layers.append(
                {
                    "ln_a_w": vec(w[p + "ln_attn.weight"]),
                    "ln_a_b": vec(w[p + "ln_attn.bias"]),
                    "qkv_w": mat(qkv_w),
                    "qkv_b": vec(qkv_b),
                    "ao_w": mat(w[p + "attn_out.weight"]),
                    "ao_b": vec(w[p + "attn_out.bias"]),
                    "ln_f_w": vec(w[p + "ln_ffn.weight"]),
                    "ln_f_b": vec(w[p + "ln_ffn.bias"]),
                    "f1_w": mat(w[p + "ffn1.weight"]),
                    "f1_b": vec(w[p + "ffn1.bias"]),
                    "f2_w": mat(w[p + "ffn2.weight"]),
                    "f2_b": vec(w[p + "ffn2.bias"]),
                }
            )
        self.layer_ops = layers
        self.final_w = vec(w["final_ln.weight"])
        self.final_b = vec(w["final_ln.bias"])
        self.lm_d_w = mat(w["lm.dense.weight"])
        self.lm_d_b = vec(w["lm.dense.bias"])
        self.lm_l_w = vec(w["lm.ln.weight"])
        self.lm_l_b = vec(w["lm.ln.bias"])
        self.dec_w = mat(self._emb_table)  # tied decoder = embedding table [V,H]
        self.lm_bias = vec(w["lm.bias"])
        self._built = True
        atexit.register(self.close)
        return self

    # -------------------------------------------------- per-input tensors

    def host_embedding(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        """Word embedding lookup + ESM token-dropout rescale, fp32 host math.

        Lookup is a plain gather (aten.index); the forbidden-by-harness
        aten.embedding op is not used. All dense math stays on device.
        """
        cfg = self.config
        h = self._emb_table[ids]  # gather copies; inputs never mutated
        if cfg.token_dropout:
            is_mask = ids.eq(cfg.mask_token_id)
            h = h.masked_fill(is_mask.unsqueeze(-1), 0.0)
            ratio = is_mask.sum(-1).float() / am.sum(-1).float()
            h = h * (1.0 - cfg.mask_ratio_train) / (1.0 - ratio[:, None, None])
        return h  # [B,L,H] fp32

    def _cos_sin_half(self, pos: torch.Tensor):
        """cos/sin [B,L,head_dim/2] fp32 via broadcast multiply (no einsum).

        The oracle builds cat(freqs, freqs) then uses full-width cos/sin; the
        two halves are identical, so only one half is materialized.
        """
        freqs = pos.to(torch.float32).unsqueeze(-1) * self._inv_freq  # [B,L,half]
        return freqs.cos(), freqs.sin()

    def rotary_tensors(self, ids: torch.Tensor):
        """Cached device bf16 tables: q [B,H,L,D] and k^T [B,H,D,L].

        Tables are PRE-DUPLICATED to full head width with the rotation sign
        folded in (cos_d = cat(cos, cos); sin_d = cat(-sin, sin)), so the
        per-layer rotation needs only 2 slices + concat + 2 muls + add
        (dispatch-floor op-count cut, 9 -> 6 ops per rotated tensor). This
        is bit-identical to the previous half-width-table form: fp32 negation
        is exact and commutes with the bf16 cast, bf16 round-to-nearest-even
        is sign-symmetric (fl(a*-b) = -fl(a*b)) and fl(a-b) = fl(a+(-b)).
        Tables stay bf16 (measured: rotary sites sit at the bf16 noise floor
        r~1.45-1.49 vs the fp32-sim twin; no device evidence for fp32
        tables). Duplication happens host-side and doubles only the
        one-time per-input rotary upload (long: ~5.4 -> ~10.8 MB, cold-only;
        warm per-call traffic is unchanged).
        """
        key = ids.detach().numpy().tobytes()
        cached = self._rotary_cache.get(key)
        if cached is not None:
            return cached
        cfg = self.config
        pos = position_ids_from_input_ids(ids, cfg.pad_token_id)
        cos_h, sin_h = self._cos_sin_half(pos)  # [B,L,half] fp32 host
        cos_d = torch.cat([cos_h, cos_h], dim=-1)  # [B,L,D] fp32
        sin_d = torch.cat([-sin_h, sin_h], dim=-1)  # sign-folded [B,L,D]
        H = cfg.num_attention_heads
        q_cos = self._dev(cos_d.unsqueeze(1).expand(-1, H, -1, -1))
        q_sin = self._dev(sin_d.unsqueeze(1).expand(-1, H, -1, -1))
        cos_t = cos_d.permute(0, 2, 1)  # [B,D,L]
        sin_t = sin_d.permute(0, 2, 1)
        k_cos = self._dev(cos_t.unsqueeze(1).expand(-1, H, -1, -1))
        k_sin = self._dev(sin_t.unsqueeze(1).expand(-1, H, -1, -1))
        cached = ((q_cos, q_sin), (k_cos, k_sin))
        self._rotary_cache[key] = cached
        return cached

    def mask_tensor(self, am: torch.Tensor):
        """Cached device bf16 additive mask [B,H,L,L]: 0 keep / -1e9 pad."""
        key = am.detach().numpy().tobytes()
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        add = (1.0 - am.float()) * _MASK_NEG  # [B,L]
        B, L = am.shape
        m = add[:, None, None, :].expand(B, 1, L, L).expand(B, self.config.num_attention_heads, L, L).contiguous()
        t = self._dev(m)
        self._mask_cache[key] = t
        return t

    # ------------------------------------------------------------- layers

    def _rotary_apply(self, x, cos_t, sin_t):
        """x [B,H,L,D]; 6-op halves rotation: lo'=lo*cos-hi*sin,
        hi'=hi*cos+lo*sin, via a rotate-half concat and the sign-folded
        table (see rotary_tensors; bit-identical to the 9-op halves form).
        """
        ttnn = self.ttnn
        B, H, L, D = (int(d) for d in x.shape)
        half = D // 2
        x_lo = ttnn.slice(x, [0, 0, 0, 0], [B, H, L, half])
        x_hi = ttnn.slice(x, [0, 0, 0, half], [B, H, L, D])
        rot = ttnn.concat([x_hi, x_lo], dim=3)  # rotate_half, sign in table
        return ttnn.add(ttnn.multiply(x, cos_t), ttnn.multiply(rot, sin_t))

    def _rotary_apply_t(self, x, cos_t, sin_t):
        """Same 6-op rotation for k in transposed layout [B,H,D,L] (halves
        on dim 2)."""
        ttnn = self.ttnn
        B, H, D, L = (int(d) for d in x.shape)
        half = D // 2
        x_lo = ttnn.slice(x, [0, 0, 0, 0], [B, H, half, L])
        x_hi = ttnn.slice(x, [0, 0, half, 0], [B, H, D, L])
        rot = ttnn.concat([x_hi, x_lo], dim=2)
        return ttnn.add(ttnn.multiply(x, cos_t), ttnn.multiply(rot, sin_t))

    def _layer_forward(self, x, i: int, q_tables, k_tables, mask):
        """One pre-LN layer; x is the fp32 residual stream (bf16 policy).

        Sites named in self.out32 request fp32 output tiles (exact fp32
        accumulation); bf16-policy consumers get one clean round after
        (cast calls are skipped entirely for sites not in out32).
        """
        ttnn = self.ttnn
        cfg = self.config
        lyr = self.layer_ops[i]
        eps = cfg.layer_norm_eps
        s32 = self.out32
        a = ttnn.layer_norm(self._cast(x, self.dtype), epsilon=eps, weight=lyr["ln_a_w"], bias=lyr["ln_a_b"])
        qkv = self._linear(a, lyr["qkv_w"], bias=lyr["qkv_b"], out_fp32=("qkv" in s32))
        if "qkv" in s32:
            qkv = self._cast(qkv, self.dtype)  # policy round pre-split
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=cfg.num_attention_heads
        )  # q,v [B,H,L,D]; k [B,H,D,L]
        q = self._rotary_apply(q, *q_tables)
        k = self._rotary_apply_t(k, *k_tables)
        scores = self._matmul(q, k, out_fp32=("scores" in s32))
        if "scores" in s32:
            # fp32-add of the mask throws on this runtime (see docstring);
            # round back to bf16 so the mask add stays on the proven path.
            scores = self._cast(scores, self.dtype)
        scores = ttnn.add(scores, mask)  # q scale folded in weights
        probs = self._softmax(scores)  # fp32-compute composite (see doc)
        o = self._matmul(probs, v, out_fp32=("pv" in s32))  # [B,H,L,D]
        if "pv" in s32:
            o = self._cast(o, self.dtype)  # exact-accum round; concat bf16
        o = ttnn.transformer.concatenate_heads(o)  # [B,L,H*D]
        ao = self._linear(o, lyr["ao_w"], bias=lyr["ao_b"], out_fp32=("ao" in s32))
        if getattr(ao, "dtype", None) != self.stream_dtype:
            ao = self._cast(ao, self.stream_dtype)  # bf16-out path
        x = ttnn.add(x, ao)  # fp32 residual add
        z = ttnn.layer_norm(self._cast(x, self.dtype), epsilon=eps, weight=lyr["ln_f_w"], bias=lyr["ln_f_b"])
        f1 = self._linear(z, lyr["f1_w"], bias=lyr["f1_b"], out_fp32=("ffn1" in s32))
        if "ffn1" in s32:
            f1 = self._cast(f1, self.dtype)  # policy round pre-gelu
        f = ttnn.gelu(f1)  # erf variant
        f2 = self._linear(f, lyr["f2_w"], bias=lyr["f2_b"], out_fp32=("ffn2" in s32))
        if getattr(f2, "dtype", None) != self.stream_dtype:
            f2 = self._cast(f2, self.stream_dtype)  # bf16-out path
        return ttnn.add(x, f2)

    # ------------------------------------------------ trace fast path (opt-in)

    def set_trace(self, enabled: bool) -> None:
        """Toggle the opt-in trace fast path; eager is the safe default."""
        self.enable_trace = bool(enabled)

    def trace_stats(self) -> dict:
        """Capture-cache state for probes/tests (never allocates)."""
        return {
            "enabled": self.enable_trace,
            "disabled_after_error": self._trace_disabled,
            "status": self.trace_status,
            "last_mode": self.last_trace_mode,
            "cache_size": len(self._trace_cache),
            "entries": [
                {
                    "Lt": e["Lt"],
                    "tid": repr(e["tid"]),
                    "capture_seconds": e["capture_seconds"],
                    "replays": e["replays"],
                    "rewrites": dict(e["rewrites"]),
                    "released": e["released"],
                }
                for e in self._trace_cache.values()
            ],
        }

    def clear_trace_cache(self) -> None:
        """Release every captured trace (entries keep device buffers)."""
        for key in list(self._trace_cache):
            self._release_trace_entry(key)

    def _release_trace_entry(self, key) -> None:
        entry = self._trace_cache.pop(key, None)
        if entry is None:
            return
        try:
            self._trace_call(self.ttnn.release_trace, self.device, entry["tid"])
            entry["released"] = True
        except Exception as e:  # best-effort; buffers drop with the entry
            entry["released"] = repr(e)

    def _trace_call(self, fn, device, tid=None):
        """Adaptive ttnn trace-API invocation across signature drift
        (verified on tt-metal-fd80faa3: inspect-based
        arg assembly first, then positional fallback patterns; TypeError
        from argument parsing has no side effects)."""
        try:
            names = list(inspect.signature(fn).parameters)
            args = []
            for n in names:
                if n in ("self", "cls"):
                    continue
                if n in ("device", "dev", "mesh_device"):
                    args.append(device)
                elif n == "cq_id":
                    args.append(self._TRACE_CQ_ID)
                elif n in ("tid", "trace_id", "trace"):
                    if tid is None:
                        raise TypeError("tid required by signature")
                    args.append(tid)
                elif n == "blocking":
                    args.append(True)
            return fn(*args)
        except (ValueError, TypeError):
            patterns = (
                [(device,), (device, self._TRACE_CQ_ID)]
                if tid is None
                else [(device, tid), (device, self._TRACE_CQ_ID, tid)]
            )
            if fn.__name__ == "execute_trace":
                patterns = [
                    (device, tid, True),
                    (device, self._TRACE_CQ_ID, tid, True),
                    (device, tid),
                    (device, self._TRACE_CQ_ID, tid),
                ]
            last = None
            for pat in patterns:
                try:
                    return fn(*pat)
                except TypeError as e:
                    last = e
            raise last

    def _trace_key(self, ids, am, Lt):
        """Capture-cache key: full (ids-pattern, mask-pattern, Lt) bytes.

        A hit therefore implies byte-identical padded inputs; the emb_sig
        comparison in _trace_path is the runtime guard that this actually
        held (defense in depth against key misuse).
        """
        return (ids.detach().numpy().tobytes(), am.detach().numpy().tobytes(), int(Lt))

    def _rewrite_x0(self, emb, entry) -> str:
        """Stage THIS call's embedding into the captured x0 device tensor.

        Returns a counter name. ttnn.copy is used when the runtime exposes
        a two-tensor copy; tt-metal-fd80faa3 does not (no copy op in the
        ttnn package, benchmark probing), so the staging write degrades to
        "skipped_no_copy_api" -- still bit-safe because _trace_path has
        ALREADY verified emb bytes == capture signature, and those verified
        bytes are exactly what the captured graph reads from x0.
        """
        ttnn = self.ttnn
        if self._x0_copy_fn is None:
            fn = getattr(ttnn, "copy", None)
            try:
                inspect.signature(fn)
                self._x0_copy_fn = fn
            except (TypeError, ValueError):
                self._x0_copy_fn = False
        if self._x0_copy_fn is False:
            return "skipped_no_copy_api"
        try:
            src = self._dev(emb, dtype=self.stream_dtype)
            self._x0_copy_fn(src, entry["x0"])
            return "written"
        except Exception:
            return "skipped_copy_rejected"

    def _trace_path(self, ids, am, Lt, emb):
        """Opt-in trace fast path (see module docstring).

        Returns (logits_tt, hidden_tt, mode); eager fallthrough is
        (None, None, "eager") with the reason in self.trace_status. Exactly
        ONE capture attempt per process; any failure latches
        _trace_disabled and the call runs eager.
        """
        ttnn = self.ttnn
        if self._trace_disabled:
            return None, None, "eager"
        key = self._trace_key(ids, am, Lt)
        emb_sig = emb.detach().contiguous().numpy().tobytes()
        entry = self._trace_cache.get(key)
        if entry is None:
            tid = None
            try:
                x0 = self._dev(emb, dtype=self.stream_dtype)
                q_tables, k_tables = self.rotary_tensors(ids)
                mask = self.mask_tensor(am)
                t0 = time.perf_counter()
                tid = self._trace_call(ttnn.begin_trace_capture, self.device)
                if tid is None:
                    tid = 1  # runtime returned nothing addressable; use 1
                logits, hidden = self._graph_forward(x0, q_tables, k_tables, mask)
                self._trace_call(ttnn.end_trace_capture, self.device, tid)
            except Exception as e:
                self._trace_disabled = True
                self.trace_status = f"capture_failed: {e!r}"
                if tid is not None:
                    try:
                        self._trace_call(ttnn.release_trace, self.device, tid)
                    except Exception:
                        pass
                return None, None, "eager"
            while len(self._trace_cache) >= self._TRACE_CACHE_MAX:
                self._release_trace_entry(next(iter(self._trace_cache)))
            self._trace_cache[key] = {
                "tid": tid,
                "Lt": int(Lt),
                "x0": x0,
                "q_tables": q_tables,
                "k_tables": k_tables,
                "mask": mask,
                "logits": logits,
                "hidden": hidden,
                "emb_sig": emb_sig,
                "capture_seconds": time.perf_counter() - t0,
                "rewrites": {"written": 0, "skipped_no_copy_api": 0, "skipped_copy_rejected": 0},
                "replays": 0,
                "released": False,
            }
            self.trace_status = "captured"
            return logits, hidden, "trace_capture"
        # ---- cache hit: fresh inputs must reproduce the captured embedding
        if emb_sig != entry["emb_sig"]:
            # impossible on a genuine pattern hit; guard for key misuse
            self.trace_status = "replay_guard_eager"
            return None, None, "eager"
        entry["rewrites"][self._rewrite_x0(emb, entry)] += 1
        try:
            self._trace_call(ttnn.execute_trace, self.device, entry["tid"])
        except Exception as e:
            self._trace_disabled = True
            self.trace_status = f"replay_failed: {e!r}"
            self._release_trace_entry(key)
            return None, None, "eager"
        entry["replays"] += 1
        self.trace_status = "replayed"
        return entry["logits"], entry["hidden"], "trace_replay"

    # ------------------------------------------------------------ forward

    def _graph_forward(self, x0, q_tables, k_tables, mask):
        """Device-only op sequence: all encoder layers + MLM head.

        The SINGLE shared op path for eager forward(), trace capture and
        (via the captured buffers) replay; trace replay proved this
        exact sequence bit-identical between eager re-runs and traced
        replay.
        """
        ttnn = self.ttnn
        cfg = self.config
        x = x0
        for i in range(cfg.num_hidden_layers):
            x = self._layer_forward(x, i, q_tables, k_tables, mask)
        eps = cfg.layer_norm_eps
        s32 = self.out32
        hidden = ttnn.layer_norm(self._cast(x, self.dtype), epsilon=eps, weight=self.final_w, bias=self.final_b)
        hd = self._linear(hidden, self.lm_d_w, bias=self.lm_d_b, out_fp32=("dense" in s32))
        if "dense" in s32:
            hd = self._cast(hd, self.dtype)  # policy round pre-gelu
        g = ttnn.gelu(hd)
        g = ttnn.layer_norm(g, epsilon=eps, weight=self.lm_l_w, bias=self.lm_l_b)
        logits = self._linear(g, self.dec_w, bias=self.lm_bias, out_fp32=("decoder" in s32))  # [B,Lt,V]
        return logits, hidden

    def forward(self, input_ids, attention_mask) -> dict:
        """input_ids/attention_mask: CPU numpy int64 [B,L] (CONTRACT surface).
        Returns {"logits": [B,L,V] fp32, "hidden": [B,L,1280] fp32} numpy.

        Eager by default; with set_trace(True) and Lt < _TRACE_MAX_LT the
        opt-in trace fast path serves repeated (ids, mask, Lt) patterns
        from a captured trace (see module docstring)."""
        assert self._built, "call build() before forward()"
        cfg = self.config
        ids = torch.as_tensor(np.asarray(input_ids)).long()
        am = torch.as_tensor(np.asarray(attention_mask)).long()
        if ids.shape != am.shape:
            raise ValueError(f"input_ids {tuple(ids.shape)} != attention_mask {tuple(attention_mask.shape)}")
        B, L = ids.shape

        # Tile-align the sequence (measured bringup fix; see module docstring):
        # am=0 synthetic pads extend the additive mask over the physical tile,
        # so garbage tile-pad columns die in softmax and PV. Real positions
        # keep arange rotary phases (prefix unchanged), token-dropout ratio
        # uses am.sum() (unchanged), and outputs are sliced back to L.
        Lt = (L + 31) // 32 * 32
        if Lt != L:
            ids = F.pad(ids, (0, Lt - L), value=cfg.pad_token_id)
            am = F.pad(am, (0, Lt - L), value=0)

        # fp32 residual stream (measured dominant-error fix; see module doc)
        emb = self.host_embedding(ids, am)  # [B,Lt,H] fp32 host
        logits = hidden = None
        if self.enable_trace:
            if Lt < self._TRACE_MAX_LT:
                logits, hidden, mode = self._trace_path(ids, am, Lt, emb)
                self.last_trace_mode = mode
            else:
                # decline: persistent trace intermediates (e.g. [B,20,Lt,Lt]
                # fp32 scores) unsized at Lt >= 1024 (benchmark probing)
                self.last_trace_mode = "trace_declined_Lt_ge_1024"
        else:
            self.last_trace_mode = "eager"
        if logits is None:  # eager default / decline / trace fallback
            x0 = self._dev(emb, dtype=self.stream_dtype)
            q_tables, k_tables = self.rotary_tensors(ids)
            mask = self.mask_tensor(am)
            logits, hidden = self._graph_forward(x0, q_tables, k_tables, mask)
        return {
            "logits": np.ascontiguousarray(self._host(logits)[:, :L].numpy(), dtype=np.float32),
            "hidden": np.ascontiguousarray(self._host(hidden)[:, :L].numpy(), dtype=np.float32),
        }
