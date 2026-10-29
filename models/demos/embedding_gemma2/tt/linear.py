# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-tower projections: measured program configs (E5a `mm_sweep`), precision from the policy.

Weight dtype, compute fidelity and fp32 accumulation per weight group come from the precision
policy (`tt/precision.py`, `doc/datatype_sweep/`): text modules pass a `MatmulPolicy` as `policy=`.
Without one (vision tower callers) the call uses the E5a default: HiFi2, fp32 accumulation
(`EG2_MM_FP32`), bf16 output.

Program configs (doc/optimized_text/README.md, "Matmul search"):
    wide roles   (N >= 768 or K == 512: QKV, MLP gate/up, PLE, head) 2D multicast on the
                 full grid, in0_block_w 8 (2x faster than the default config at M = 512).
    deep roles   (K >= 1024 and N == 512: attention out, MLP down) at M >= 2048:
                 `ttnn.experimental.minimal_matmul` (fp32 accumulate); below that the 2D config.

`linear(x, w)` takes `x` [B, 1, S, K] (S a multiple of 32) and folds the batch into M, so one
program config covers every batch size. The program config is derived from the shapes only and
is cached, so the call adds no device op.

Tuned configs per geometry (E9.3, doc/prefill/03-matmul/): callers name their role (`role=`, e.g.
"up", "gate", "down"); `TUNED_CONFIGS[grid][(role, S)]` gives an `MMSpec` that replaces the rule above
for that role and bucket S on that compute grid only, at batch 1 (batched calls keep the rule). Grids without an
entry (the P150a's 13 x 10) keep the rule above unchanged. `EG2_MM_TUNED=0` disables the table
(same-session control). `MM_OVERRIDES[(role, M)]` (in-process, empty by default) takes precedence on
every grid: the in-model config search (tt-ops e9/03-matmul/scripts/mm_sweep.py) sets it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import ttnn
from models.demos.embedding_gemma2.tt.config import compute_grid

# Default (no policy) fp32 dest accumulation for the 2D-multicast roles; minimal_matmul without a
# policy always accumulates in fp32. Text calls take both from their MatmulPolicy.
FP32_ACC = os.environ.get("EG2_MM_FP32", "1") == "1"


def compute_kernel_config_matmul(fp32_acc: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )


_CKC = {False: None, True: None}


def _ckc(fp32_acc: bool):
    if _CKC[fp32_acc] is None:
        _CKC[fp32_acc] = compute_kernel_config_matmul(fp32_acc)
    return _CKC[fp32_acc]


def _activation(name: str | None):
    if name is None:
        return None
    if name == "gelu_tanh":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
    raise ValueError(f"unsupported fused activation {name}")


def _largest_divisor(n: int, limit: int) -> int:
    return max(d for d in range(1, limit + 1) if n % d == 0)


# Static circular-buffer budget per core for the 2D config (Blackhole L1 is 1.5 MB; SDPA/other
# program buffers and the allocator need the rest). Above it `linear` uses minimal_matmul.
_CB_BUDGET_BYTES = 900 * 1024
_TILE_BYTES = {"bf16": 2048, "bfp8": 1088}


def _cb_bytes(per_core_M: int, per_core_N: int, in0_block_w: int, fp32_acc: bool) -> int:
    in0 = 2 * per_core_M * in0_block_w * _TILE_BYTES["bf16"]
    in1 = 2 * in0_block_w * per_core_N * _TILE_BYTES["bfp8"]
    out = per_core_M * per_core_N * _TILE_BYTES["bf16"]
    interm = per_core_M * per_core_N * 4096 if fp32_acc else 0
    return in0 + in1 + out + interm


@lru_cache(maxsize=None)
def mcast2d_config(M: int, K: int, N: int, gx: int, gy: int, activation: str | None, fp32_acc: bool = False):
    """2D multicast config, or None when no in0_block_w keeps the CBs inside the L1 budget."""
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    gy = min(gy, Mt)
    per_core_M = -(-Mt // gy)
    per_core_N = -(-Nt // gx)
    in0_block_w = _largest_divisor(Kt, 8)
    while _cb_bytes(per_core_M, per_core_N, in0_block_w, fp32_acc) > _CB_BUDGET_BYTES:
        if in0_block_w == 1:
            return None
        in0_block_w = _largest_divisor(Kt, in0_block_w - 1)
    max_sb = 4 if fp32_acc else 8
    sbw = _largest_divisor(per_core_N, max_sb)
    sbh = max(d for d in range(1, max_sb + 1) if per_core_M % d == 0 and d * sbw <= max_sb)
    act = _activation(activation)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=act,
    )


@dataclass(frozen=True)
class MMSpec:
    """One matmul config (tile units). family "2d": `MatmulMultiCoreReuseMultiCastProgramConfig` on
    `grid` with `in0_block_w`, `sb` (out_subblock_h, out_subblock_w), `per_core` (per_core_M,
    per_core_N), `out_block` (out_block_h, out_block_w); None fields are derived as in
    `mcast2d_config`. `chunk`: fold M into chunks of this many rows (x viewed as [1, M / chunk,
    chunk, K], fuse_batch=False), so per_core_M is per chunk. family "1d":
    `MatmulMultiCoreReuseMultiCast1DProgramConfig` (in0 multicast, N split over the grid's cores,
    per_core_M = all of M) with `in0_block_w`, `sb`, `per_core`. family "minimal":
    `ttnn.experimental.minimal_matmul` with `MinimalMatmulConfig(blocks = (M, K, N block), sb,
    grid)`, or its defaults when `blocks` is None. `out_mem` "L1" / "DRAM" overrides the caller's
    output memory. `gelu` (gated_product only): "mm" fuses gelu_tanh into the matmul, "mul" runs it
    as the multiply's pre-activation. `transpose`: 2d transpose_mcast."""

    family: str
    grid: tuple[int, int] | None = None
    in0_block_w: int | None = None
    sb: tuple[int, int] | None = None
    per_core: tuple[int, int] | None = None
    out_block: tuple[int, int] | None = None
    chunk: int | None = None
    blocks: tuple[int, int, int] | None = None
    out_mem: str | None = None
    gelu: str | None = None
    transpose: bool = False  # 2d: transpose_mcast (M along the grid columns)


def _tuned_12x10() -> dict[tuple[str, int], MMSpec]:
    """E9.3 kept configs for the 12 x 10 grid (doc/prefill/03-matmul/README.md, final config table).
    Each entry won full-forward burst wall time alone in the model (search tables) and is confirmed in
    the per-bucket `final` sweep; the tokens are the contract buckets at batch 1."""
    bw16 = MMSpec("2d", in0_block_w=16)  # K = 512 in one block: one in0 / in1 pass per output block
    m_4_8_2 = MMSpec("minimal", blocks=(4, 8, 2), sb=(2, 2))
    m_4_8_4 = MMSpec("minimal", blocks=(4, 8, 4), sb=(2, 2))
    gelu_mul = MMSpec("minimal", gelu="mul")  # plain minimal_matmul, gelu_tanh in the multiply
    t: dict[tuple[str, int], MMSpec] = {}
    # 128 tokens: 1D in0-multicast for the wide projections (24 / 48 cores in the 2D rule)
    t[("qkv_sliding", 128)] = MMSpec("1d", grid=(8, 8))
    t[("qkv_full", 128)] = MMSpec("1d")
    t[("ple_wide", 128)] = MMSpec("1d")
    t[("ple_gate", 128)] = bw16
    # 256: in0_block_w 16 on the gated roles, gelu fused in the MLP gate matmul. 512: the rule is kept
    t[("gate", 256)] = MMSpec("2d", in0_block_w=16, gelu="mm")
    t[("ple_gate", 256)] = bw16
    # 1024: in0_block_w 16 on the N = 2048 roles; gelu fused in the gate matmul with it
    for r in ("up", "qkv_sliding"):
        t[(r, 1024)] = bw16
    t[("gate", 1024)] = MMSpec("2d", in0_block_w=16, gelu="mm")
    # 2048 / 4096: in0_block_w 16 on every K = 512 2D role, gelu in the multiply, smaller minimal blocks
    for m in (2048, 4096):
        for r in ("up", "qkv_sliding", "qkv_full"):
            t[(r, m)] = bw16
        t[("gate", m)] = MMSpec("2d", in0_block_w=16, gelu="mul")
        t[("ple_proj", m)] = MMSpec("2d")
    t[("down", 2048)] = m_4_8_2
    t[("o_sliding", 2048)] = MMSpec("2d")
    t[("ple_gate", 2048)] = MMSpec("2d", gelu="mul")
    t[("down", 4096)] = m_4_8_4
    t[("o_sliding", 4096)] = m_4_8_4
    t[("ple_gate", 4096)] = MMSpec("2d", in0_block_w=16)
    # >= 8192: both gated roles on plain minimal_matmul with gelu_tanh in the multiply (the fused
    # activation doubles minimal_matmul's time; the 2D config does not fit L1 at these M)
    for m in (8192, 16384, 32768, 65536):
        t[("ple_gate", m)] = gelu_mul
        if m < 65536:  # 65536 (power-limited clock): no gate candidate beat the session drift
            t[("gate", m)] = gelu_mul
    for m in (8192, 16384):  # at 32768 / 65536 their burst gain is inside the session drift: not kept
        t[("down", m)] = m_4_8_2
        t[("ple_layer", m)] = m_4_8_4
    t[("o_sliding", 8192)] = m_4_8_4
    t[("up", 8192)] = MMSpec("2d", chunk=4096, in0_block_w=8)
    t[("up", 16384)] = MMSpec("2d", chunk=4096, in0_block_w=8)
    return t


# (role, tokens) -> MMSpec per compute grid (columns, rows). The P150a's 13 x 10 has no entry.
TUNED_CONFIGS: dict[tuple[int, int], dict[tuple[str, int], MMSpec]] = {(12, 10): _tuned_12x10()}
TUNED_ON = os.environ.get("EG2_MM_TUNED", "1") == "1"
MM_OVERRIDES: dict[tuple[str, int], MMSpec] = {}


def lookup_spec(role: str | None, tokens: int, grid, batch: int = 1) -> MMSpec | None:
    """Spec for (role, tokens) on `grid`. Batch 1 only: the contract is batch 1, and batched calls keep
    the rule (stage-2 configs) whatever their token count."""
    if role is None or batch != 1:
        return None
    spec = MM_OVERRIDES.get((role, tokens))
    if spec is None and TUNED_ON:
        spec = TUNED_CONFIGS.get((grid.x, grid.y), {}).get((role, tokens))
    return spec


@lru_cache(maxsize=None)
def spec_program_config(spec: MMSpec, M: int, K: int, N: int, gx: int, gy: int, activation, fp32_acc: bool):
    """Program config for `spec` at [M, K] x [K, N] (M per chunk for a chunked 2D spec)."""
    if spec.family == "minimal":
        if spec.blocks is None:
            return None
        mb, kb, nb = spec.blocks
        sbh, sbw = spec.sb or (2, 2)
        g = spec.grid or (gx, gy)
        return ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sbh,
            subblock_w=sbw,
            compute_with_storage_grid_size=ttnn.CoreCoord(*g),
        )
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    max_sb = 4 if fp32_acc else 8
    if spec.family == "1d":  # in0 multicast, N split over grid[0] x grid[1] cores, all of M per core
        sgx, sgy = spec.grid or (gx, gy)
        per_core_M, per_core_N = spec.per_core or (Mt, -(-Nt // (sgx * sgy)))
        sbw = _largest_divisor(per_core_N, max_sb)
        sbh = max(d for d in range(1, max_sb + 1) if per_core_M % d == 0 and d * sbw <= max_sb)
        sbh, sbw = spec.sb or (sbh, sbw)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(sgx, sgy),
            in0_block_w=spec.in0_block_w or _largest_divisor(Kt, 8),
            out_subblock_h=sbh,
            out_subblock_w=sbw,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=_activation(activation),
            mcast_in0=True,
        )
    if spec.family != "2d":
        raise ValueError(f"unknown matmul family {spec.family}")
    sgx, sgy = spec.grid or (gx, gy)
    if spec.per_core is not None:
        per_core_M, per_core_N = spec.per_core
    elif spec.transpose:  # M along the grid columns, N along the rows
        per_core_M = -(-Mt // min(sgx, Mt))
        per_core_N = -(-Nt // sgy)
    else:
        per_core_M = -(-Mt // min(sgy, Mt))
        per_core_N = -(-Nt // sgx)
    in0_block_w = spec.in0_block_w or _largest_divisor(Kt, 8)
    if spec.sb is not None:
        sbh, sbw = spec.sb
    else:
        ob_h, ob_w = spec.out_block or (per_core_M, per_core_N)
        sbw = _largest_divisor(ob_w, max_sb)
        sbh = max(d for d in range(1, max_sb + 1) if ob_h % d == 0 and d * sbw <= max_sb)
    ob_h, ob_w = spec.out_block or (per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(sgx, sgy),
        in0_block_w=in0_block_w,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        out_block_h=ob_h,
        out_block_w=ob_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=spec.transpose,
        fused_activation=_activation(activation),
        fuse_batch=spec.chunk is None,
    )


# Where gelu_tanh runs for gelu(x @ w_gate) * other: as an SFPU pre-activation of the multiply's
# first input up to this many tokens (faster at 1x128 / 1x512 by 0.05-0.1 ms), in the matmul
# packer above it (faster at 1x2048 / 8x512 by 0.15-0.4 ms). E5a c5 A/B, burst.
GELU_IN_MUL_MAX_TOKENS = int(os.environ.get("EG2_GELU_IN_MUL_MAX_TOKENS", "1024"))


def gated_product(
    x: ttnn.Tensor,
    w_gate: ttnn.Tensor,
    other: ttnn.Tensor,
    *,
    memory_config,
    policy=None,
    dtype=ttnn.bfloat16,
    role: str | None = None,
) -> ttnn.Tensor:
    """gelu_tanh(x @ w_gate) * other, x [B, 1, S, K]. `policy` / `dtype` / `role` as in `linear`;
    a tuned spec's `gelu` field picks where the gelu runs."""
    tokens = x.shape[0] * x.shape[2]
    spec = lookup_spec(role, tokens, compute_grid(x.device()), batch=x.shape[0] * x.shape[1])
    gelu_in_mul = tokens <= GELU_IN_MUL_MAX_TOKENS
    if spec is not None and spec.gelu is not None:
        gelu_in_mul = spec.gelu == "mul"
    if gelu_in_mul:
        gate = linear(x, w_gate, memory_config=memory_config, policy=policy, dtype=dtype, role=role)
        out = ttnn.multiply(
            gate,
            other,
            memory_config=memory_config,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
        )
    else:
        gate = linear(
            x, w_gate, activation="gelu_tanh", memory_config=memory_config, policy=policy, dtype=dtype, role=role
        )
        out = ttnn.multiply(gate, other, memory_config=memory_config)
    ttnn.deallocate(gate)
    return out


def linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    *,
    activation: str | None = None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    policy=None,
    dtype=ttnn.bfloat16,
    role: str | None = None,
) -> ttnn.Tensor:
    """x [B, 1, S, K] @ w [1, 1, K, N] -> [B, 1, S, N] `dtype` TILE (optionally gelu_tanh fused).

    `policy` (a `precision.MatmulPolicy`) sets the compute fidelity and fp32 accumulation; the
    weight dtype is the one `w` was built with (`policy.dtype` at construction). None: HiFi2,
    fp32 accumulation `FP32_ACC` (2D config) / always (minimal_matmul). `role` names the projection
    for the tuned-config table (module doc)."""
    if policy is None:
        fp32_acc, ckc_2d, ckc_minimal = FP32_ACC, _ckc(FP32_ACC), _ckc(True)
    else:
        fp32_acc = policy.fp32_acc
        ckc_2d = ckc_minimal = policy.compute_kernel_config
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    grid = compute_grid(x.device())
    spec = lookup_spec(role, M, grid, batch=B * shape[1])
    if spec is not None:
        return _linear_spec(x, w, spec, activation, memory_config, fp32_acc, ckc_2d, ckc_minimal, dtype, grid)
    x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
    pc = None
    # minimal_matmul wins for narrow outputs at large M and for everything at M >= 8192 (E5a
    # mm_grid sweep); the 2D config wins elsewhere.
    # minimal_matmul's fused activation is slow (PLE gate + gelu at M=4096: 218 us vs 40 us without).
    if activation is not None or not ((N <= 768 and M >= 2048) or M >= 8192):
        pc = mcast2d_config(M, K, N, grid.x, grid.y, activation, fp32_acc)
    if pc is None:
        out = ttnn.experimental.minimal_matmul(
            x2,
            w,
            fused_activation=_activation(activation),
            compute_kernel_config=ckc_minimal,
            memory_config=memory_config,
            dtype=dtype,
        )
    else:
        out = ttnn.linear(
            x2,
            w,
            program_config=pc,
            compute_kernel_config=ckc_2d,
            memory_config=memory_config,
            dtype=dtype,
        )
    if x2 is not x:
        out = ttnn.reshape(out, (B, shape[1], S, N))
    return out


_OUT_MEM = {"L1": ttnn.L1_MEMORY_CONFIG, "DRAM": ttnn.DRAM_MEMORY_CONFIG}


def _linear_spec(x, w, spec: MMSpec, activation, memory_config, fp32_acc, ckc_2d, ckc_minimal, dtype, grid):
    """`linear` with a tuned / searched `MMSpec` (module doc)."""
    shape = x.shape
    B, S, K = shape[0], shape[-2], shape[-1]
    N = w.shape[-1]
    M = B * shape[1] * S
    mem = _OUT_MEM[spec.out_mem] if spec.out_mem else memory_config
    if spec.family == "minimal":
        cfg = spec_program_config(spec, M, K, N, grid.x, grid.y, activation, fp32_acc)
        x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
        out = ttnn.experimental.minimal_matmul(
            x2,
            w,
            fused_activation=_activation(activation),
            config=cfg,
            compute_kernel_config=ckc_minimal,
            memory_config=mem,
            dtype=dtype,
        )
    else:
        chunk = spec.chunk if spec.chunk is not None and spec.chunk < M else None
        if chunk is not None:
            if M % chunk:
                raise ValueError(f"chunk {chunk} does not divide M {M}")
            x2 = ttnn.reshape(x, (1, M // chunk, chunk, K))
            pc = spec_program_config(spec, chunk, K, N, grid.x, grid.y, activation, fp32_acc)
        else:
            x2 = ttnn.reshape(x, (1, 1, M, K)) if B * shape[1] != 1 else x
            pc = spec_program_config(
                MMSpec(**{**spec.__dict__, "chunk": None}) if spec.chunk else spec,
                M,
                K,
                N,
                grid.x,
                grid.y,
                activation,
                fp32_acc,
            )
        out = ttnn.linear(x2, w, program_config=pc, compute_kernel_config=ckc_2d, memory_config=mem, dtype=dtype)
    if tuple(out.shape) != (B, shape[1], S, N):
        out = ttnn.reshape(out, (B, shape[1], S, N))
    return out
