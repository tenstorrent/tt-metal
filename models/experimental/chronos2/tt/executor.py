# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# TTNN executor for amazon/chronos-2 (encoder-only T5-style forecaster).
#
# Design notes (measurements in docs/validation.md):
#   * RMSNorm and the attention softmax are composed from elementwise ops; the
#     softmax runs in FP32. Both were more accurate than ttnn.rms_norm and
#     ttnn.softmax at these masked, non-tile-aligned sequence lengths.
#   * Eager execution is dispatch-bound (~500 small ops, latency flat in the
#     sequence length), so the graph is captured into a metal trace per static
#     (batch, context patches, output patches) shape and replayed.
#   * Only one trace is alive at a time. Allocating a new shape's persistent
#     buffers while another trace existed hung the device on a later replay,
#     so a new shape releases the live trace before allocating.
#   * The op count is reduced at weight-upload time with exact FP32 folds:
#       - RMSNorm weight * sqrt(D) is folded into the following linear(s); the
#         device computes x * rsqrt(sum(x^2) + D * eps).
#       - q|k|v are one linear pair: Y1 = h @ [Wq|Wk|Wv] and
#         Y2 = h @ [rot(Wq)|rot(Wk)|0] with rotate_half folded into the weight
#         columns, so RoPE is Y1 * C3 + Y2 * S3 with host tables
#         C3 = [cos|cos|1] and S3 = [sin|sin|0].
#       - Group attention over independent series reduces exactly to one
#         linear, W_o @ W_v (softmax over a single key is 1).
#       - Heads are split/merged with ttnn.transformer ops; ReLU is fused into
#         ttnn.linear.
#   * bf16 keeps the numerically sensitive parts in FP32: HiFi4 with FP32
#     accumulation on every matmul, an FP32 residual stream (RMSNorm runs in
#     FP32 and is cast to bf16 only as a matmul input), and an FP32 final norm
#     and quantile head. Plain bf16 lost accuracy on short, high-dynamic-range
#     contexts; this costs about 11 % latency.
#   * bfp8_b uploads only the linear weight matrices as BFP8_B (after the FP32
#     folds) and otherwise runs the plain bf16 graph. It halves weight memory
#     but does not reduce latency, because replay is dispatch-bound.
# The host precomputes the RoPE tables and the additive attention mask
# (architecture constants, no learned values); the final quantile-major
# rearrange is pure data movement.

from __future__ import annotations

import warnings

import numpy as np

from .model_config import PRECISION_DTYPES, Chronos2Config
from .weights import fuse_group_attention, rotate_half_columns

# Device bytes per 32x32 tile (BFP8_B: 1024 mantissa bytes + 64 shared exponents).
_TILE_BYTES = {"float32": 4096, "bfloat16": 2048, "bfloat8_b": 1088}

# Precisions that run the FP32-residual policy (HiFi4 matmuls, fp32 residual
# stream, fp32 head); see the header.
_ACCURATE_BF16 = ("bf16",)


class TTChronos2Executor:
    """Runs learned compute on TT. One instance per (config, precision, device).

    All learned arithmetic (residual-block embeddings, 12 encoder blocks with
    RoPE time-attention / fused group-attention linear / ReLU FFN, final
    RMSNorm, quantile head) runs as TTNN ops on the caller-owned device.
    The host only: prepares FP32 features (Preprocessor), folds weights once
    at upload (norm scales, rotate_half, W_o@W_v), precomputes RoPE cos/sin
    tables and the additive attention mask (architecture constants, no
    learned values), and performs the final quantile-major rearrange (pure
    data movement) plus FP32 unscale in Backend.forecast.

    With ``use_trace`` the device graph for the current static (B,
    n_ctx_patches, n_out) shape is captured into a metal trace over persistent
    input buffers and replayed while calls keep that shape. Per-call inputs
    (features, mask) are copied into those buffers; per-shape constants (REG
    token, RoPE tables) are written once at capture. Only one trace is alive
    at a time: a new shape first releases the live trace and frees its
    buffers, so no device buffer is ever allocated while a trace exists.
    """

    def __init__(
        self, cfg: Chronos2Config, weights: dict[str, np.ndarray], device, precision: str, use_trace: bool = False
    ):
        import ttnn

        if str(cfg.dense_act_fn).lower() != "relu":
            raise NotImplementedError(f"dense_act_fn '{cfg.dense_act_fn}' not implemented (only relu)")
        if cfg.d_kv % 2:
            raise NotImplementedError(f"RoPE needs an even d_kv, got {cfg.d_kv}")
        self.ttnn = ttnn
        self.cfg = cfg
        self.device = device
        self.precision = precision
        act_name, w_name = PRECISION_DTYPES[precision]
        self.dtype = getattr(ttnn, act_name)  # activations, biases, tables
        self.wdtype = getattr(ttnn, w_name)  # linear weight matrices
        self._dtype_names = {self.dtype: act_name, self.wdtype: w_name, ttnn.float32: "float32"}
        # bf16 accuracy policy: HiFi4 compute config, fp32 residual stream, fp32 head
        self._accurate = precision in _ACCURATE_BF16
        self._kcfg = None
        if self._accurate:
            kcls = getattr(ttnn, "BlackholeComputeKernelConfig", None) or ttnn.WormholeComputeKernelConfig
            self._kcfg = kcls(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        self.weight_bytes = 0  # device bytes of uploaded weights + biases
        self._eps_sum = float(cfg.d_model) * cfg.layer_norm_epsilon  # eps on sum(x^2), see _rms
        self._reg_np = np.ascontiguousarray(weights["shared.weight"][1])  # REG = vocab row 1
        self.use_trace = bool(use_trace)
        self._trace = None  # {"key", "tid", "ins", "out"} of the live trace
        self.trace_stats = {"captures": 0, "replays": 0, "releases": 0, "eager": 0, "capture_error": None}
        self._upload(weights)

    # -- device plumbing ----------------------------------------------------- #

    def _host_tile(self, arr: np.ndarray):
        import torch

        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        return self.ttnn.from_torch(t, dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT)

    def _to_device(self, arr: np.ndarray, dtype=None):
        ttnn = self.ttnn
        dtype = self.dtype if dtype is None else dtype
        # owned, writable C-order copy: B=1 broadcast views are "contiguous"
        # and read-only, which torch.from_numpy warns about
        t = np.array(arr, dtype=np.float32, copy=True, order="C")
        import torch

        tt = torch.from_numpy(t)
        try:
            return ttnn.from_torch(tt, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        except TypeError:
            return ttnn.to_device(ttnn.from_torch(tt, dtype=dtype, layout=ttnn.TILE_LAYOUT), self.device)

    def _param(self, arr: np.ndarray, is_matrix: bool, dtype=None):
        """Upload a learned parameter; matrices use the weight dtype, biases the
        activation dtype, unless ``dtype`` overrides. Accumulates tile-padded
        device bytes."""
        if dtype is None:
            dtype = self.wdtype if is_matrix else self.dtype
        a = np.atleast_2d(arr)
        tiles = int(np.prod(a.shape[:-2], dtype=np.int64)) * (-(-a.shape[-2] // 32)) * (-(-a.shape[-1] // 32))
        self.weight_bytes += tiles * _TILE_BYTES[self._dtype_names[dtype]]
        return self._to_device(arr, dtype)

    def _to_host(self, x) -> np.ndarray:
        """Device -> host FP32. Non-fp32 tensors are typecast on device first
        (torch bf16 tensors have no .numpy())."""
        ttnn = self.ttnn
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return ttnn.to_torch(x).numpy()

    def _linear(self, x, w_t, b=None, relu: bool = False, out32: bool = False, plain: bool = False):
        """ttnn.linear with the precision's compute config (``plain`` skips it).
        ``out32`` returns FP32: emitted directly when there is no bias,
        typecast otherwise."""
        ttnn = self.ttnn
        kw = {}
        if self._kcfg is not None and not plain:
            kw["compute_kernel_config"] = self._kcfg
        if b is not None:
            kw["bias"] = b
        if relu:
            kw["activation"] = "relu"
        if out32 and b is None:
            kw["dtype"] = ttnn.float32
        y = ttnn.linear(x, w_t, **kw)
        if out32 and y.dtype != ttnn.float32:
            y = ttnn.typecast(y, ttnn.float32)
        return y

    def _matmul(self, a, b):
        if self._kcfg is not None:
            return self.ttnn.matmul(a, b, compute_kernel_config=self._kcfg)
        return self.ttnn.matmul(a, b)

    def _rms(self, x):
        """Unweighted RMSNorm scaled by 1/sqrt(D): x * rsqrt(sum(x^2) + D*eps).

        The checkpoint's norm weight times sqrt(D) is folded into the rows of
        the consuming linear(s) at upload, so the product equals
        w * x * rsqrt(mean(x^2) + eps) exactly in real arithmetic.
        """
        ttnn = self.ttnn
        s = ttnn.add(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), self._eps_sum)
        return ttnn.multiply(x, ttnn.rsqrt(s))

    def _rms_in(self, x):
        """RMSNorm as a matmul input: computed at x's dtype (fp32 residual
        stream under the bf16 policy), cast to the activation dtype."""
        y = self._rms(x)
        return self.ttnn.typecast(y, self.dtype) if y.dtype != self.dtype else y

    def _softmax_fp32(self, scores):
        """Manual stable softmax computed in FP32."""
        ttnn = self.ttnn
        sf = ttnn.typecast(scores, ttnn.float32) if scores.dtype != ttnn.float32 else scores
        m = ttnn.max(sf, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(sf, m))
        s = ttnn.sum(e, dim=-1, keepdim=True)
        return ttnn.multiply(e, ttnn.reciprocal(s))

    # -- weight upload ------------------------------------------------------ #

    def _upload(self, w: dict[str, np.ndarray]) -> None:
        """Fold (FP32 host math) and upload; each norm scale goes into the
        rows of the linear(s) that consume the normalised activations."""
        cfg = self.cfg
        H, dk = cfg.num_heads, cfg.d_kv
        root_d = np.float32(np.sqrt(cfg.d_model))
        mat = lambda a: self._param(a, True)  # noqa: E731
        vec = lambda a: self._param(a, False)  # noqa: E731

        def wt(key: str) -> np.ndarray:
            return np.ascontiguousarray(w[f"{key}.weight"].T)  # [K, N] for ttnn.linear

        def norm_scale(key: str) -> np.ndarray:
            return (w[f"{key}.weight"] * root_d)[:, None].astype(np.float32)  # [K, 1]

        def residual_params(prefix: str, scale: np.ndarray | None = None, dtype=None) -> dict:
            s = np.float32(1.0) if scale is None else scale
            m = mat if dtype is None else (lambda a: self._param(a, True, dtype))
            v = vec if dtype is None else (lambda a: self._param(a, False, dtype))
            return {
                "Wh": m(s * wt(f"{prefix}.hidden_layer")),
                "bh": v(w[f"{prefix}.hidden_layer.bias"]),
                "Wo": m(wt(f"{prefix}.output_layer")),
                "bo": v(w[f"{prefix}.output_layer.bias"]),
                "Wr": m(s * wt(f"{prefix}.residual_layer")),
                "br": v(w[f"{prefix}.residual_layer.bias"]),
            }

        self.inp_emb = residual_params("input_patch_embedding")
        # final RMSNorm feeds only the quantile head → fold into its first linears;
        # the bf16 policy keeps the head in FP32 (weights and biases)
        head_dtype = self.ttnn.float32 if self._accurate else None
        self.out_emb = residual_params("output_patch_embedding", norm_scale("encoder.final_layer_norm"), head_dtype)
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"encoder.block.{i}"
            a = f"{p}.layer.0.self_attention"
            s0 = norm_scale(f"{p}.layer.0.layer_norm")
            s1 = norm_scale(f"{p}.layer.1.layer_norm")
            s2 = norm_scale(f"{p}.layer.2.layer_norm")
            wq, wk, wv = (s0 * wt(f"{a}.{n}") for n in ("q", "k", "v"))
            w1 = np.concatenate([wq, wk, wv], axis=1)
            w2 = np.concatenate(
                [rotate_half_columns(wq, H, dk), rotate_half_columns(wk, H, dk), np.zeros_like(wv)], axis=1
            )
            fused = fuse_group_attention(w, f"{p}.layer.1")
            self.layers.append(
                {
                    "W1": mat(w1),  # h @ [Wq|Wk|Wv]
                    "W2": mat(w2),  # h @ [rot(Wq)|rot(Wk)|0]
                    "Wo": mat(wt(f"{a}.o")),
                    "Wg": mat(s1 * np.ascontiguousarray(fused.T)),
                    "wi": mat(s2 * wt(f"{p}.layer.2.mlp.wi")),
                    "wo": mat(wt(f"{p}.layer.2.mlp.wo")),
                }
            )

    # -- device submodules --------------------------------------------------- #

    def _residual_block(self, x, prm: dict, out32: bool = False, plain: bool = False):
        """act(x@Wh+bh)@Wo+bo + x@Wr+br  (relu, with biases)."""
        h = self._linear(x, prm["Wh"], prm["bh"], relu=True, plain=plain)
        out = self._linear(h, prm["Wo"], prm["bo"], out32=out32, plain=plain)
        res = self._linear(x, prm["Wr"], prm["br"], out32=out32, plain=plain)
        return self.ttnn.add(out, res)

    def _attention(self, lyr: dict, x, c3, s3, addmask):
        """Time self-attention: RMSNorm → fused q|k|v with RoPE folded in
        (no score scaling) → FP32 softmax with additive mask → o-projection.
        Heads are [B,H,S,dk] from ttnn.transformer split/concatenate_heads."""
        ttnn = self.ttnn
        h = self._rms_in(x)
        qkv = ttnn.add(
            ttnn.multiply(self._linear(h, lyr["W1"]), c3), ttnn.multiply(self._linear(h, lyr["W2"]), s3)
        )  # [B,S,3*inner], RoPE applied
        q, k_t, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.cfg.num_heads, transpose_key=True
        )
        scores = ttnn.add(self._matmul(q, k_t), addmask)  # [B,H,S,S]; scale=1.0 by design
        probs = self._softmax_fp32(scores)
        if probs.dtype != self.dtype:
            probs = ttnn.typecast(probs, self.dtype)
        att = ttnn.transformer.concatenate_heads(self._matmul(probs, v))  # [B,S,inner]
        return ttnn.add(x, self._linear(att, lyr["Wo"], out32=self._accurate))

    def _block(self, i: int, x, c3, s3, addmask):
        ttnn = self.ttnn
        lyr = self.layers[i]
        r32 = self._accurate
        x = self._attention(lyr, x, c3, s3, addmask)
        x = ttnn.add(x, self._linear(self._rms_in(x), lyr["Wg"], out32=r32))  # fused group attn
        h = self._linear(self._rms_in(x), lyr["wi"], relu=True)  # ReLU FFN
        return ttnn.add(x, self._linear(h, lyr["wo"], out32=r32))

    # -- host inputs ------------------------------------------------------------ #

    def _shape(self, prepared: dict) -> tuple[int, int, int, int]:
        B, n_ctx, _ = prepared["ctx_features"].shape
        n_out = int(prepared["num_output_patches"])
        S = n_ctx + (1 if self.cfg.use_reg_token else 0) + n_out
        return int(B), int(n_ctx), n_out, S

    def _shape_consts(self, B: int, S: int) -> dict:
        """Per-shape constants (architecture tables; no learned values):
        RoPE tables in the fused q|k|v layout, C3=[cos|cos|1], S3=[sin|sin|0]."""
        cfg = self.cfg
        H, dk = cfg.num_heads, cfg.d_kv
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, dk, 2, dtype=np.float64) / dk))
        freqs = np.outer(np.arange(S, dtype=np.float64), inv)
        emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32)  # [S, dk]
        cos = np.tile(np.cos(emb), (1, H))  # [S, inner], head-major columns
        sin = np.tile(np.sin(emb), (1, H))
        c3 = np.concatenate([cos, cos, np.ones_like(cos)], axis=-1)
        s3 = np.concatenate([sin, sin, np.zeros_like(sin)], axis=-1)
        out = {
            "c3": np.ascontiguousarray(np.broadcast_to(c3[None], (B, *c3.shape)), dtype=np.float32),
            "s3": np.ascontiguousarray(np.broadcast_to(s3[None], (B, *s3.shape)), dtype=np.float32),
        }
        if cfg.use_reg_token:
            out["reg"] = np.tile(self._reg_np[None, None, :], (B, 1, 1))
        return out

    def _call_inputs(self, prepared: dict, B: int, S: int) -> dict:
        """Per-call inputs: features and the additive attention mask [B,H,S,S]."""
        H = self.cfg.num_heads
        amask = np.ascontiguousarray(prepared["attn_mask"], dtype=np.float32)
        neg_min = np.finfo(np.float32).min
        am = np.broadcast_to(((1.0 - amask) * neg_min)[:, None, None, :], (B, H, S, S))
        return {
            "ctx": np.ascontiguousarray(prepared["ctx_features"], dtype=np.float32),
            "fut": np.ascontiguousarray(prepared["fut_features"], dtype=np.float32),
            "am": np.ascontiguousarray(am, dtype=np.float32),
        }

    # -- forward --------------------------------------------------------------- #

    def _forward(self, ins: dict, B: int, S: int, n_out: int):
        """Device graph from device inputs; returns the FP32 head [B, n_out, Q*T]."""
        ttnn = self.ttnn
        cfg = self.cfg
        r32 = self._accurate
        # --- token embeddings: emb(ctx) | REG | emb(fut) ---
        parts = [self._residual_block(ins["ctx"], self.inp_emb, out32=r32)]
        if cfg.use_reg_token:
            parts.append(ttnn.typecast(ins["reg"], ttnn.float32) if r32 else ins["reg"])
        parts.append(self._residual_block(ins["fut"], self.inp_emb, out32=r32))
        tokens = ttnn.concat(parts, dim=1) if len(parts) > 1 else parts[0]

        # --- encoder ---
        for i in range(cfg.num_layers):
            tokens = self._block(i, tokens, ins["c3"], ins["s3"], ins["am"])
        # final norm weight folded into out_emb; FP32 head under the bf16 policy
        tokens = self._rms(tokens) if r32 else self._rms_in(tokens)

        # --- quantile head on the last n_out tokens ---
        head_in = ttnn.slice(tokens, [0, S - n_out, 0], [B, S, cfg.d_model])
        head = self._residual_block(head_in, self.out_emb, plain=r32)
        if head.dtype != ttnn.float32:
            head = ttnn.typecast(head, ttnn.float32)
        return head

    def _run_eager(self, prepared: dict, B: int, S: int, n_out: int) -> np.ndarray:
        host = {**self._shape_consts(B, S), **self._call_inputs(prepared, B, S)}
        ins = {k: self._to_device(v) for k, v in host.items()}
        self.trace_stats["eager"] += 1
        return self._to_host(self._forward(ins, B, S, n_out))

    def _capture(self, key, host: dict, B: int, S: int, n_out: int) -> dict:
        """Allocate persistent inputs and capture; must run with no live trace."""
        ttnn = self.ttnn
        ins = {k: self._to_device(v) for k, v in host.items()}
        self._forward(ins, B, S, n_out)  # compile + program cache outside the trace
        ttnn.synchronize_device(self.device)
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            out = self._forward(ins, B, S, n_out)
        finally:
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.trace_stats["captures"] += 1
        return {"key": key, "tid": tid, "ins": ins, "out": out}

    def release_trace(self) -> None:
        """Release the live trace and deallocate its persistent buffers."""
        entry, self._trace = self._trace, None
        if entry is None:
            return
        ttnn = self.ttnn
        ttnn.synchronize_device(self.device)
        ttnn.release_trace(self.device, entry["tid"])
        for t in list(entry["ins"].values()) + [entry["out"]]:
            ttnn.deallocate(t)
        self.trace_stats["releases"] += 1

    def _run_traced(self, prepared: dict, B: int, n_ctx: int, n_out: int, S: int) -> np.ndarray:
        ttnn = self.ttnn
        key = (B, n_ctx, n_out)
        call = self._call_inputs(prepared, B, S)
        if self._trace is None or self._trace["key"] != key:
            self.release_trace()
            self._trace = self._capture(key, {**self._shape_consts(B, S), **call}, B, S, n_out)
        entry = self._trace
        for name, arr in call.items():
            ttnn.copy_host_to_device_tensor(self._host_tile(arr), entry["ins"][name])
        ttnn.execute_trace(self.device, entry["tid"], cq_id=0, blocking=False)
        self.trace_stats["replays"] += 1
        return ttnn.to_torch(entry["out"]).float().numpy()

    def run(self, prepared: dict) -> np.ndarray:
        """[B, Q, n_out*output_patch_size] FP32 quantile predictions (scaled space)."""
        cfg = self.cfg
        B, n_ctx, n_out, S = self._shape(prepared)
        out = None
        if self.use_trace:
            try:
                out = self._run_traced(prepared, B, n_ctx, n_out, S)
            except Exception as e:  # trace region exhausted / unsupported → eager from now on
                self.use_trace = False
                self.trace_stats["capture_error"] = repr(e)
                warnings.warn(f"chronos2: metal trace disabled, falling back to eager: {e!r}")
                self._trace = None
        if out is None:
            out = self._run_eager(prepared, B, S, n_out)

        # --- host rearrange (pure data movement): 'b p (q t) -> b q (p t)' ---
        q_n, t_p = cfg.num_quantiles, cfg.output_patch_size
        pred = out.astype(np.float32).reshape(B, n_out, q_n, t_p).transpose(0, 2, 1, 3).reshape(B, q_n, n_out * t_p)
        return np.ascontiguousarray(pred, dtype=np.float32)
