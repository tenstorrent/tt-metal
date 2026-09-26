# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The matmul-class ops of one MiniMax-H3 transformer block, per device, as the model runs them on the
Wormhole 4x8 Galaxy (TP = 4 / SP = 8): one `OpSpec` per op with its shape, fusion, blocking, grid, sweep-harness
use case, Tracy op code and the 2026-09-17 baseline. Shared by `transformer_roofline.py` (host-only: this module
imports neither torch nor ttnn at import time), `transformer_op_mesh_bench.py`, `transformer_op_single_device_bench.py`
and `agmm_compute_zones.py`. Not a test; pytest leaves it alone.

    to_qkv  ColParallelLinear(5376, 3*7168, chunks=3)  AGMM, output split into q|k|v by the writer (copy epilogue)
    to_out  ColParallelLinear(7168, 5376)              AGMM, fused addcmul epilogue: a + scalar * (x @ w) * b
    ff1     ColParallelLinear(5376, 2*14336) SwiGLU    AGMM, fused SwiGLU epilogue on the packed gate|up weight
    ff2     RowParallelLinear(14336, 5376)             minimal_matmul on the full 8x9 grid + reduce-scatter

AGMM = `ttnn.experimental.all_gather_minimal_matmul_async`: the TP all-gather of the K-sharded activation fused
with the matmul (`models/tt_dit/layers/linear.py`, ColParallelLinear.forward). K is the gathered K the kernel
multiplies over; K_local = K / TP is what each device holds. N is the per-device weight width (packed gate|up for
ff1, the pre-reduce-scatter width for ff2); N_out is the per-device output width.

Shapes at 15 s / 768P / 16:9: hidden 5376, inner 7168 (56 heads x 128), ffn 14336, 13664 rows per device
(Tracy block breakdown, 2026-09-17). Blockings, grids and `measured_us_wh_15s` are the WORMHOLE galaxy baseline at
this M: to_qkv / to_out from `AGMM_BLOCK_SIZES` (`agmm_config.py`, subblock fixed at 2x2 by `get_matmul_config`),
ff1 and ff2 from the swept `grid_88_configs` / `grid_89_configs` entries (`models/tt_dit/utils/matmul.py`), which
win over the model's `default_block_size`. On another architecture the model resolves different grids and possibly
different blockings; the AGMM sweep (`test_h3_agmm_sweep`) records the blocking it resolves at run time in its
sidecar instead of reading these fields.

Torch-side helpers (`prepare_weight`, `make_extra_inputs`, `golden`, `output_parts`) import torch lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

M_15S_768P_16_9 = 13664  # rows per device at SP=8 (Tracy block breakdown, 2026-09-17)
TP = 4


@dataclass(frozen=True)
class OpSpec:
    name: str
    family: str  # "agmm" (all-gather + matmul on the 8x8 worker grid) | "mm+rs" (matmul on the full grid, then RS)
    model_call: str
    K: int  # K the kernel multiplies over: gathered for an AGMM, per-device for ff2
    K_local: int  # K held per device before the gather (== K for ff2)
    N: int  # per-device weight width
    N_out: int  # per-device output width
    fusion: str  # short label for tables and figures
    blocks: tuple[int, int, int, int, int]  # M_block, K_block, N_block, subblock_h, subblock_w the model runs on WH
    grid: tuple[int, int]  # WH worker grid (8x8 AGMM, 8x9 full); BH is 12x9 / 11x10, resolved at run time by the sweep
    sweep_use_case: str  # `USE_CASE_CONFIGS` key in models/tt_dit/utils/sweep_mm_block_sizes.py
    sweep_is_agmm: bool
    op_code: str  # Tracy OP CODE
    measured_us_wh_15s: float  # WH 2026-09-17 baseline at M=13664, HiFi2, shipped blocking (sweep_mm_block_sizes.py)
    color: str
    marker: str
    chunks: int = 1
    fuse_swiglu: bool = False
    addcmul_scalar: float | None = None
    bias: bool = False
    rs_op_code: str | None = None
    M: int = M_15S_768P_16_9
    tp: int = TP

    # --- names the roofline tool has always used ---
    @property
    def kind(self) -> str:
        return self.family

    @property
    def is_agmm(self) -> bool:
        return self.family == "agmm"

    @property
    def cores(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def has_fusion(self) -> bool:
        return self.chunks > 1 or self.fuse_swiglu or self.addcmul_scalar is not None

    def flops(self, M: int | None = None) -> float:
        return 2.0 * (self.M if M is None else M) * self.K * self.N

    def sweep_id(self, M: int | None = None) -> str:
        """Shape id of the sweep harness (`SHAPE_IDS` / the `mm_sweep_<config>_<id>` profiler directory)."""
        M = self.M if M is None else M
        return f"{M}_{self.K}_{self.N}_{self.grid[0]}x{self.grid[1]}_{'agmm' if self.sweep_is_agmm else 'mm'}_{self.sweep_use_case}"

    def blocks_str(self) -> str:
        return ",".join(str(v) for v in self.blocks)

    @classmethod
    def adhoc(cls, name: str, K: int, N: int, fusion: str, family: str = "agmm", marker: str = "o") -> "OpSpec":
        """A shape-only spec for rooflining something that is not one of the block's ops."""
        return cls(
            name=name,
            family=family,
            model_call="",
            K=K,
            K_local=K // TP if family == "agmm" else K,
            N=N,
            N_out=N,
            fusion=fusion,
            blocks=(8, 8, 8, 2, 2),
            grid=(8, 8) if family == "agmm" else (8, 9),
            sweep_use_case="",
            sweep_is_agmm=family == "agmm",
            op_code="",
            measured_us_wh_15s=float("nan"),
            color="#8a8983",
            marker=marker,
        )


# Colours: the dataviz categorical slots after the three resource colours the roofline uses, so an op colour
# is never mistaken for a resource colour.
TO_QKV = OpSpec(
    name="to_qkv",
    family="agmm",
    model_call="ColParallelLinear(5376, 3*7168, chunks=3, bias=False)  attention_minimax_h3.py",
    K=5376,
    K_local=1344,
    N=5376,  # 3 chunks x 1792 (q, k, v heads of this device)
    N_out=5376,
    fusion="chunks=3",
    chunks=3,
    blocks=(8, 7, 12, 2, 2),  # AGMM_BLOCK_SIZES[(5376, 5376)], agmm_config.py
    grid=(8, 8),
    sweep_use_case="qkv",
    sweep_is_agmm=True,
    op_code="AllGatherMinimalMatmulAsyncOp",
    measured_us_wh_15s=10401.8,  # blocking sweep 2026-09-17, shipped (8,7,12)
    color="#eda100",
    marker="o",
)
TO_OUT = OpSpec(
    name="to_out",
    family="agmm",
    model_call="ColParallelLinear(7168, 5376, bias=False) + addcmul(residual, ., gate)  attention_minimax_h3.py",
    K=7168,
    K_local=1792,
    N=1344,
    N_out=1344,
    fusion="addcmul",
    addcmul_scalar=1.0,
    blocks=(8, 8, 6, 2, 2),  # AGMM_BLOCK_SIZES[(7168, 1344)]
    grid=(8, 8),
    sweep_use_case="to_out",
    sweep_is_agmm=True,
    op_code="AllGatherMinimalMatmulAsyncOp",
    # Shipped (8,8,6) with the real addcmul epilogue: harness "to_out" use case, 2026-09-21 (5,294 / 5,311 us in two
    # runs; includes the harness's bias add, ~50 us, which the model does not have). The 2026-09-17 blocking-sweep
    # figure for to_out, 4332.8 us, was swept with the "plain" use case: no epilogue, math_approx_mode=False.
    measured_us_wh_15s=5294.0,
    color="#e87ba4",
    marker="s",
)
FF1 = OpSpec(
    name="ff1",
    family="agmm",
    model_call="ParallelFeedForward.ff1 = ColParallelLinear(5376, 2*14336, activation swiglu, bias=False)",
    K=5376,
    K_local=1344,
    N=7168,  # packed gate|up
    N_out=3584,
    fusion="SwiGLU",
    fuse_swiglu=True,
    blocks=(8, 7, 10, 2, 2),  # grid_88_configs[(13664, 5376, 7168)], matmul.py (sweep rank 1/320)
    grid=(8, 8),
    sweep_use_case="ff1_swiglu",
    sweep_is_agmm=True,
    op_code="AllGatherMinimalMatmulAsyncOp",
    measured_us_wh_15s=15709.9,
    color="#008300",
    marker="^",
)
FF2 = OpSpec(
    name="ff2",
    family="mm+rs",
    model_call="ParallelFeedForward.ff2 = RowParallelLinear(14336, 5376): minimal_matmul (8x9) + ccl reduce_scatter",
    K=3584,
    K_local=3584,
    N=5376,  # pre-reduce-scatter
    N_out=1344,
    fusion="MM + reduce-scatter",
    blocks=(8, 7, 10, 2, 2),  # grid_89_configs[(13664, 3584, 5376)], matmul.py (sweep rank 2/322)
    grid=(8, 9),
    sweep_use_case="ff2",
    sweep_is_agmm=False,
    op_code="MinimalMatmulDeviceOperation",
    rs_op_code="ReduceScatterMinimalAsyncDeviceOperation",
    measured_us_wh_15s=6770.7,  # the matmul alone; the reduce-scatter is a separate ~2.8 ms op
    color="#8a8983",
    marker="D",
)

# to_out in row-parallel form (experiment, 2026-09-25): each device multiplies its own K_local = 1792 head
# columns by a K-sharded [1792, 5376] weight and the ring reduce-scatters the [M, 5376] partial to [M, 1344],
# with the gated residual applied after the reduce -- the ff2 shape of work, half the FLOPs. Same FLOPs as the
# shipped AGMM form (M x 1792 x 5376 == M x 7168 x 1344) but 21 N tiles per core instead of 6, so each in0
# byte feeds 3.5x the MACs. Bench-only: not in ALL_OPS, so the roofline figures and tables ignore it.
TO_OUT_MMRS = OpSpec(
    name="to_out_mmrs",
    family="mm+rs",
    model_call="to_out as RowParallelLinear(7168, 5376): minimal_matmul on K_local + reduce-scatter + addcmul",
    K=1792,
    K_local=1792,
    N=5376,  # pre-reduce-scatter
    N_out=1344,
    fusion="MM + reduce-scatter + addcmul",
    blocks=(6, 7, 8, 2, 2),  # ff2's fused entry as the seed: M 61 tiles/core on 7 rows, N 21/core on 8 columns
    grid=(8, 9),
    sweep_use_case="to_out_mmrs",
    sweep_is_agmm=False,
    op_code="MinimalMatmulStridedReduceScatterAsyncDeviceOperation",
    measured_us_wh_15s=0.0,  # unmeasured; see the bench
    color="#e87ba4",
    marker="v",
)

AGMM_OPS = [TO_QKV, TO_OUT, FF1]
ALL_OPS = [TO_QKV, TO_OUT, FF1, FF2]
OPS_BY_NAME = {s.name: s for s in ALL_OPS} | {TO_OUT_MMRS.name: TO_OUT_MMRS}
OP_FAMILIES = {"agmm": AGMM_OPS, "all": ALL_OPS}

MEASURED_US_WH_15S = {s.name: s.measured_us_wh_15s for s in ALL_OPS}
OP_COLOR = {s.name: s.color for s in ALL_OPS}
# sweep-harness use case -> op name; "plain" is the use case the 2026-09-17 to_out rows were swept with
SWEEP_USE_CASE_TO_OP = {s.sweep_use_case: s.name for s in ALL_OPS} | {"plain": "to_out"}


def select_ops(arg: str) -> list[OpSpec]:
    """`--ops agmm|all|<comma list of op names and/or families>`, e.g. `agmm,ff2`."""
    out: list[OpSpec] = []
    for name in arg.split(","):
        name = name.strip()
        picked = OP_FAMILIES.get(name) or ([OPS_BY_NAME[name]] if name in OPS_BY_NAME else None)
        if picked is None:
            raise SystemExit(f"unknown op {name!r}; choose from {', '.join(OPS_BY_NAME)}, agmm or all")
        out += [s for s in picked if s not in out]
    return out


# ----------------------------------------------------------------------------------------------
# torch-side helpers for the benches (lazy imports keep the module host-only for the roofline)
# ----------------------------------------------------------------------------------------------


def prepare_weight(spec: OpSpec, w, fused: bool = True):
    """Host weight as the kernel wants it: ff1's packed gate|up weight is tile-pair interleaved for the fused
    SwiGLU kernel (`prepare_for_fused_swiglu`); everything else is used as is."""
    if spec.fuse_swiglu and fused:
        from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

        return prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True)
    return w


def make_extra_inputs(spec: OpSpec, M: int, fused: bool = True, gate_broadcast: bool = False) -> dict:
    """Extra host tensors (bf16) the fused op needs: to_out's addcmul operands a [M, N] and b [M, N] (the model's
    per-token gate; `gate_broadcast` gives the kernel's [1, N] row-broadcast path instead)."""
    import torch

    if spec.addcmul_scalar is not None and fused:
        a = (torch.randn(M, spec.N) * 0.5).to(torch.bfloat16)
        b = torch.randn(1 if gate_broadcast else M, spec.N).to(torch.bfloat16)
        return {"a": a, "b": b}
    return {}


def golden(spec: OpSpec, x, w, extras: dict, fused: bool = True):
    """fp32 torch reference of the fused op on the given (bf16-rounded) inputs; `w` is the UNPREPARED weight."""
    import torch

    y = x.float() @ w.float()
    if not fused:
        return y
    if spec.fuse_swiglu:
        g, u = torch.chunk(y, 2, dim=-1)
        return torch.nn.functional.silu(g) * u
    if spec.addcmul_scalar is not None:
        return extras["a"].float() + spec.addcmul_scalar * y * extras["b"].float()
    return y  # chunks: the concatenation of the chunk outputs equals the unsplit product


def output_parts(spec: OpSpec, outs, fused: bool = True) -> list:
    """The op's output as a list of tensors whose torch conversions concatenate along -1 into the golden's
    shape: the AGMM and `minimal_matmul_split` return one tensor per chunk, the other ops one tensor."""
    if isinstance(outs, (list, tuple)):
        return list(outs)
    return [outs]
