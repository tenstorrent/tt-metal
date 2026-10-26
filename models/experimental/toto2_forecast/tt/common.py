# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Toto 2.0: precision policies, uploads, per-token linears and norms.

Per-token activations are flat, ``(1, 1, tokens, D)``, so every linear folds the whole batch into M. Rows of
``S_pad`` tokens (a whole number of tiles) are viewed as ``(R, 1, S_pad, D)`` around time attention; both views
share one buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.toto2_forecast.tt.preprocessing import TILE, pad_to_tile  # noqa: F401


@dataclass(frozen=True)
class Toto2Precision:
    """Precision policies (CONTRACT section 1; PLAN section 9 gate metrics). Per role:

    | role                         | accuracy                          | performance                         |
    |------------------------------|-----------------------------------|-------------------------------------|
    | QKV, out_proj, fc1, fc2      | bf16 W, HiFi2, fp32 dest acc      | bfp8 W, HiFi2, fp32 dest acc        |
    | SDPA (time and variate)      | HiFi2, fp32 dest acc              | HiFi2, fp32 dest acc                |
    | patch embedding (3 matmuls)  | fp32 tokens, bf16 W, HiFi4, fp32 acc | fp32 tokens, bf16 W, HiFi4, fp32 acc |
    | head (3 matmuls)             | bf16 W, HiFi2, fp32 acc, fp32 out | bf16 W, HiFi2, fp32 acc, fp32 out   |
    | RMSNorm                      | HiFi4, fp32 dest acc              | HiFi4, fp32 dest acc                |
    | RoPE (rotary_embedding_llama)| HiFi4, bf16 dest                  | HiFi4, bf16 dest                    |
    Activations are bf16 throughout (the head output fp32). Performance is P6 = P1 + bf16 head (D4 ruling 2); the round-0 LoFi
    performance is the sweep variant P0.

    Measured (2.5B-FT, paper; ttq jobs 474, 482, 486, 501, 518):
    - fp32 vs bf16 dest accumulation, accuracy, realistic paper_mix: active per-row PCC p01 0.9990 vs 0.9758 and
      spread_ratio p99 1.05 vs 2.60 (bands up to 2.6x the reference's); device 8184 vs ~7465 ms (+9.6 %).
    - performance, paper_mix: fp32 acc gives active p01 0.9944 / spread_ratio p99 1.36 vs 0.9705 / 2.79 with bf16
      acc; bf16 FFN weights instead of bfp8 change nothing (0.9676 / 2.84), so the per-row error is accumulation.
    - the patch embedding at HiFi2 drops mantissa bits of the bf16 tokens: rows whose within-patch variation sits at
      the token ulp (strong trends) scored per-row PCC 0.21 / 0.22 / 0.67, 0.996 / 0.99 / 0.99 at HiFi4.
    - the patch tokens are uploaded in fp32 (asinh values up to ~4 have a bf16 step of 0.016): on long_context_mix
      spread_ratio p99 1.45 -> 1.12 and active per-row PCC p01 0.9912 -> 0.9991 (job 540).
    - the head writes fp32: a near-constant forecast (intermittent series, head std 6e-4 around -0.22) is below the
      bf16 ulp; fp32 output raised its PCC 0.804 -> 0.981.
    - SDPA stays at HiFi2 in performance (LoFi cost 22m at context 8192 0.9905 -> 0.9947 normalized).
    """

    name: str = "accuracy"
    bf8_weights: bool = False
    lofi: bool = False
    sdpa_lofi: bool = False
    fp32_acc: bool = False
    hifi4: bool = False  # HiFi4 matmuls
    embed_hifi4: bool = False  # the patch embedding at HiFi4 with fp32 accumulation
    head_hifi4: bool = False  # the quantile head at HiFi4 with fp32 accumulation
    fp32_tokens: bool = False  # patch tokens uploaded in fp32 (the embedding's first matmuls read them)
    fp32_head_out: bool = False  # the head writes its output in fp32
    ffn_bf16_weights: bool = False  # bf16 fc1 / fc2 weights under bf8_weights (a per-role exception)
    ffn_lofi: bool = False  # LoFi on the FFN matmuls (fc1 / fc2) only
    head_bf16_hifi2: bool = False  # the quantile head with bf16 weights at HiFi2 under bf8_weights / lofi
    out_bf16_weights: bool = False  # bf16 wo / w_down under bf8_weights (the residual-epilogue known winner)

    @classmethod
    def accuracy(cls) -> "Toto2Precision":
        return cls(fp32_acc=True, embed_hifi4=True, fp32_tokens=True, fp32_head_out=True)

    @classmethod
    def performance(cls) -> "Toto2Precision":
        return cls(
            name="performance",
            bf8_weights=True,
            lofi=False,  # P1, D2 (G1): HiFi2 on every matmul and SDPA; CONTRACT section 2's LoFi is amended
            head_bf16_hifi2=True,  # P6, D4 ruling 2: bf16 weights / HiFi2 on the quantile head
            fp32_acc=True,
            embed_hifi4=True,
            fp32_tokens=True,
            fp32_head_out=True,
        )

    # Performance-policy sweep variants (D1 item 3): P0 = the round-0 performance (LoFi); P1 = performance (D2); P1 HiFi2 everywhere; P2 LoFi on the FFN
    # only; P3 = P1 with bf16 FFN weights; P4 = P0 with a bf16 / HiFi2 head; P5 = P2 with a LoFi head;
    # P6 = P1 with a bf16 / HiFi2 head; P7 = P1 with bf16 wo / w_down.
    VARIANTS = {
        "P0": dict(lofi=True),
        "P1": dict(lofi=False),
        "P2": dict(lofi=False, ffn_lofi=True),
        "P3": dict(lofi=False, ffn_bf16_weights=True),
        "P4": dict(head_bf16_hifi2=True),
        "P5": dict(lofi=False, ffn_lofi=True, head_lofi=True),
        # D3 ruling 2: P1 variants scored for the 313m band floors.
        "P6": dict(lofi=False, head_bf16_hifi2=True),
        "P7": dict(lofi=False, out_bf16_weights=True),
    }
    head_lofi: bool = False  # LoFi on the head under lofi=False (P5)

    @classmethod
    def named(cls, name: str) -> "Toto2Precision":
        """``accuracy``, ``performance`` or ``performance_P<n>`` (``VARIANTS``)."""
        from dataclasses import replace

        if name.startswith("performance_"):
            return replace(cls.performance(), name=name, **cls.VARIANTS[name.split("_", 1)[1]])
        return {"accuracy": cls.accuracy, "performance": cls.performance}[name]()

    def weight_dtype(self, role: str = ""):
        import ttnn

        # roles: "ffn" (fc1), "ffn_out" (fc2 = w_down), "out" (attention wo)
        if role.startswith("ffn") and self.ffn_bf16_weights:
            return ttnn.bfloat16
        if role in ("out", "ffn_out") and self.out_bf16_weights:
            return ttnn.bfloat16
        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16

    def _config(self, math_fidelity, fp32_dest_acc_en: bool = False):
        import ttnn

        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def matmul_config(self):
        import ttnn

        fidelity = ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2
        if self.hifi4:
            fidelity = ttnn.MathFidelity.HiFi4
        return self._config(fidelity, fp32_dest_acc_en=self.fp32_acc)

    def for_role(self, role: str) -> "Toto2Precision":
        """The policy of one module: ``embed`` / ``head`` take HiFi4 + fp32 accumulation when asked."""
        from dataclasses import replace

        if (role == "embed" and self.embed_hifi4) or (role == "head" and self.head_hifi4):
            return replace(self, bf8_weights=False, lofi=False, hifi4=True, fp32_acc=True)
        if role == "head" and self.head_bf16_hifi2:
            return replace(self, bf8_weights=False, lofi=False)
        if (role == "ffn" and self.ffn_lofi) or (role == "head" and self.head_lofi):
            return replace(self, lofi=True)
        return self

    def sdpa_config(self):
        import ttnn

        fidelity = ttnn.MathFidelity.LoFi if self.sdpa_lofi else ttnn.MathFidelity.HiFi2
        return self._config(fidelity, fp32_dest_acc_en=self.fp32_acc)

    def norm_config(self):
        import ttnn

        return self._config(ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def rope_config(self):
        import ttnn

        # Below HiFi4 the +-1 rotation matmul drops input mantissa bits.
        return self._config(ttnn.MathFidelity.HiFi4)


def upload(device, t: torch.Tensor, *, dtype=None, layout=None):
    """Host tensor -> device DRAM (bf16 tiles by default)."""
    import ttnn

    dtype = ttnn.bfloat16 if dtype is None else dtype
    host_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(
        t.detach().to(host_dtype).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def upload_linear(device, weight: torch.Tensor, dtype, *, pad_in: int | None = None, pad_out: int | None = None):
    """``nn.Linear`` weight ``(out, in)`` -> device ``(in, out)``, zero-padded to ``pad_in`` x ``pad_out``."""
    w = weight.t()
    k, n = w.shape
    w = torch.nn.functional.pad(w, (0, (pad_out or n) - n, 0, (pad_in or k) - k))
    return upload(device, w, dtype=dtype)


def upload_bias(device, bias: torch.Tensor | None, *, pad_out: int | None = None):
    if bias is None:
        return None
    b = torch.nn.functional.pad(bias, (0, (pad_out or bias.shape[0]) - bias.shape[0]))
    return upload(device, b.reshape(1, -1))


def linear(x, weight, precision: Toto2Precision, *, bias=None, activation=None, dtype=None):
    """``x @ weight (+ bias)``, then ``activation`` ("silu"); output ``dtype`` (default: the input's)."""
    import ttnn

    return ttnn.linear(
        x,
        weight,
        bias=bias,
        activation=activation,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.matmul_config(),
    )


def rms_norm(x, eps: float, precision: Toto2Precision):
    """Weightless RMSNorm, statistics accumulated in fp32."""
    import ttnn

    return ttnn.rms_norm(
        x, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=precision.norm_config()
    )


def scaled_residual(f, x, beta: float):
    """``f + beta * x``; consumes ``f``, borrows ``x``."""
    import ttnn

    if beta == 1.0:  # folded (R2-1)
        out = ttnn.add(f, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(f)
        return out
    scaled = ttnn.multiply(x, beta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(f, scaled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(scaled)
    ttnn.deallocate(f)
    return out


def take_slots(x, rows: int, seq: int, start: int, count: int):
    """Flat ``(1, 1, rows * seq, D)`` -> ``(1, 1, rows * count, D)``: token slots ``[start, start + count)`` of every
    row; ``x`` is borrowed."""
    import ttnn

    d = x.shape[-1]
    picked = ttnn.slice(ttnn.reshape(x, (rows, seq, d)), (0, start, 0), (rows, start + count, d))
    return ttnn.reshape(picked, (1, 1, rows * count, d))
