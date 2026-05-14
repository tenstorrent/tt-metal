# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-shape matmul micro-benchmarks for the dots.ocr pipeline.

Each test reproduces a single matmul observed in the dots.ocr Tracy report
(``perf13054.txt``) using the *exact same* dtype / fidelity / program_config
the production pipeline runs on Wormhole. Use these to A/B program-config
tweaks for one shape at a time without re-running the full pipeline.

Test shapes are derived directly from the aggregated Tracy histogram
(per-device shapes; the report merges N300 device data so each row is the
work performed on a single Wormhole chip):

| ID                       |     M |     K |     N | Fidelity | in0  | in1  | out  |
|--------------------------|------:|------:|------:|----------|------|------|------|
| ``vision_qkv``           | 12288 |  1536 |  4608 | LoFi     | BF16 | BFP8 | BFP8 |
| ``vision_o_proj``        | 12288 |  1536 |  1536 | LoFi     | BFP8 | BFP8 | BFP8 |
| ``vision_mlp_fc1``       | 12288 |  1536 |  4224 | LoFi     | BF16 | BFP4 | BFP8 |
| ``vision_mlp_fc2``       | 12288 |  4224 |  1536 | LoFi     | BFP8 | BFP4 | BFP8 |
| ``vision_patch_embed``   | 11232 |   608 |   608 | LoFi     | BF16 | BFP8 | BF16 |
| ``vision_proj_pre``      |  3072 |  6144 |   768 | LoFi     | BF16 | BFP8 | BF16 |
| ``vision_proj_post``     |  3072 |  6144 |  6144 | LoFi     | BF16 | BFP8 | BF16 |
| ``text_prefill_qkv``     |  2816 |   768 |  2048 | HiFi2    | BF16 | BFP8 | BF16 |
| ``text_prefill_mlp_w13`` |  2816 |   768 | 17920 | HiFi2    | BF16 | BFP8 | BFP8 |
| ``text_prefill_mlp_w2``  |  2816 |  4480 |  1536 | HiFi2    | BFP8 | BFP8 | BFP8 |
| ``text_prefill_o_proj``  |  2816 |  1536 |   768 | HiFi2    | BF16 | BFP8 | BF16 |
| ``text_decode_qkv``      |    32 |   768 |  2048 | HiFi2    | BF16 | BFP8 | BF16 |
| ``text_decode_mlp_w13``  |    32 |   768 | 17920 | HiFi2    | BF16 | BFP8 | BFP8 |
| ``text_decode_mlp_w2``   |    32 |  4480 |  1536 | HiFi2    | BFP8 | BFP8 | BFP8 |
| ``text_decode_o_proj``   |    32 |  1536 |   768 | HiFi2    | BF16 | BFP8 | BF16 |
| ``lm_head_decode_chunk`` |    32 |  1536 | 11136 | HiFi2    | BF16 | BFP8 | BFP8 |
| ``lm_head_decode_tail``  |    32 |  1536 |  9152 | HiFi2    | BF16 | BFP8 | BFP8 |

Run a single shape::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_perf.py::test_dots_ocr_matmul_perf -k vision_qkv -s

Run all shapes::

    pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_perf.py -s

Run with Tracy for per-op device-time::

    python -m tracy -r -p -v --no-device-data -m \
        "pytest models/experimental/tt_symbiote/tests/test_dots_ocr_matmul_perf.py -s"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    _vision_matmul_compute_config,
    _vision_matmul_program_config,
)
from models.experimental.tt_symbiote.modules.linear import (
    _dp_decode_matmul_program_config,
    _dp_prefill_matmul_program_config,
)


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------


_PCC_TARGETS = {
    "BFP4": 0.965,  # K-side BFP4 (vision MLP) is the noise floor
    "BFP8": 0.985,
    "BF16": 0.997,
}


@dataclass(frozen=True)
class MatmulCase:
    """One matmul shape + production dtype/fidelity/program_config family."""

    name: str
    m: int
    k: int
    n: int
    in0_dtype: ttnn.DataType
    in1_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    # Program config family. Values:
    #   "vision_2d"  -> _vision_matmul_program_config (2D mcast, vision tower)
    #   "text_2d"    -> _dp_prefill_matmul_program_config (2D mcast, text prefill)
    #   "text_1d"    -> _dp_decode_matmul_program_config (1D mcast, text decode)
    #   "auto"       -> let ttnn pick (matches what the production code does
    #                   when the helper above bails out / returns None)
    program_config_kind: str
    in0_kind: str  # "BF16" | "BFP8"
    in1_kind: str  # "BF16" | "BFP8" | "BFP4"
    out_kind: str  # "BF16" | "BFP8"
    expected_device_us: int  # observed in perf13054.txt for ranking only

    @property
    def pcc_target(self) -> float:
        # PCC is bounded by the lossiest dtype in the chain.
        worst = self.in1_kind if self.in1_kind in _PCC_TARGETS else self.out_kind
        # K-side BFP4 dominates output noise -> use BFP4 floor whenever
        # *any* input is BFP4.
        if self.in1_kind == "BFP4" or self.in0_kind == "BFP4":
            worst = "BFP4"
        return _PCC_TARGETS.get(worst, 0.965)


_BF16, _BFP8, _BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
_LOFI, _HIFI2 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2


_CASES = [
    # ---- Vision tower (M = 12288 = 384 tiles per device) -----------------
    MatmulCase(
        name="vision_qkv",
        m=12288,
        k=1536,
        n=4608,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        program_config_kind="vision_2d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=1900,
    ),
    MatmulCase(
        name="vision_o_proj",
        m=12288,
        k=1536,
        n=1536,
        in0_dtype=_BFP8,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        program_config_kind="vision_2d",
        in0_kind="BFP8",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=665,
    ),
    MatmulCase(
        name="vision_mlp_fc1",
        m=12288,
        k=1536,
        n=4224,
        in0_dtype=_BF16,
        in1_dtype=_BFP4,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        program_config_kind="vision_2d",
        in0_kind="BF16",
        in1_kind="BFP4",
        out_kind="BFP8",
        expected_device_us=2370,
    ),
    MatmulCase(
        name="vision_mlp_fc2",
        m=12288,
        k=4224,
        n=1536,
        in0_dtype=_BFP8,
        in1_dtype=_BFP4,
        out_dtype=_BFP8,
        math_fidelity=_LOFI,
        program_config_kind="vision_2d",
        in0_kind="BFP8",
        in1_kind="BFP4",
        out_kind="BFP8",
        expected_device_us=1410,
    ),
    # ---- Vision patch / projector ---------------------------------------
    # K=608 / N=608 don't tile cleanly to grid_y=8 -> _vision_matmul_program_config
    # returns None and the production path uses the auto-config; mirror that.
    MatmulCase(
        name="vision_patch_embed",
        m=11232,
        k=608,
        n=608,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_LOFI,
        program_config_kind="auto",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=961,
    ),
    MatmulCase(
        name="vision_proj_pre",
        m=3072,
        k=6144,
        n=768,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_LOFI,
        program_config_kind="auto",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=1500,
    ),
    MatmulCase(
        name="vision_proj_post",
        m=3072,
        k=6144,
        n=6144,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_LOFI,
        program_config_kind="auto",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=4000,
    ),
    # ---- Text decoder prefill (M = 2816 = 88 tiles per device) ----------
    MatmulCase(
        name="text_prefill_qkv",
        m=2816,
        k=768,
        n=2048,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_HIFI2,
        program_config_kind="text_2d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=350,
    ),
    MatmulCase(
        name="text_prefill_mlp_w13",
        m=2816,
        k=768,
        n=17920,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_2d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=1400,
    ),
    MatmulCase(
        name="text_prefill_mlp_w2",
        m=2816,
        k=4480,
        n=1536,
        in0_dtype=_BFP8,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_2d",
        in0_kind="BFP8",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=900,
    ),
    MatmulCase(
        name="text_prefill_o_proj",
        m=2816,
        k=1536,
        n=768,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_HIFI2,
        program_config_kind="text_2d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=300,
    ),
    # ---- Text decoder decode (M = 32 = 1 tile, per-token) ----------------
    MatmulCase(
        name="text_decode_qkv",
        m=32,
        k=768,
        n=2048,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=20,
    ),
    MatmulCase(
        name="text_decode_mlp_w13",
        m=32,
        k=768,
        n=17920,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=120,
    ),
    MatmulCase(
        name="text_decode_mlp_w2",
        m=32,
        k=4480,
        n=1536,
        in0_dtype=_BFP8,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BFP8",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=60,
    ),
    MatmulCase(
        name="text_decode_o_proj",
        m=32,
        k=1536,
        n=768,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BF16,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BF16",
        expected_device_us=20,
    ),
    # ---- LM head decode (split into 11136-col chunks + a 9152 tail) ------
    MatmulCase(
        name="lm_head_decode_chunk",
        m=32,
        k=1536,
        n=11136,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=80,
    ),
    MatmulCase(
        name="lm_head_decode_tail",
        m=32,
        k=1536,
        n=9152,
        in0_dtype=_BF16,
        in1_dtype=_BFP8,
        out_dtype=_BFP8,
        math_fidelity=_HIFI2,
        program_config_kind="text_1d",
        in0_kind="BF16",
        in1_kind="BFP8",
        out_kind="BFP8",
        expected_device_us=70,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_program_config(case: MatmulCase, device):
    """Return the production program_config for ``case``, or ``None``.

    Mirrors ``TTNNLinear*`` callers in the dots.ocr pipeline: when the
    helper returns ``None`` the production code passes ``program_config=None``
    to ``ttnn.matmul`` and lets ttnn auto-pick.
    """
    kind = case.program_config_kind
    if kind == "vision_2d":
        return _vision_matmul_program_config(device, case.m, case.k, case.n)
    input_shape = (1, 1, case.m, case.k)
    weight_shape = (case.k, case.n)
    if kind == "text_2d":
        return _dp_prefill_matmul_program_config(device, input_shape, weight_shape)
    if kind == "text_1d":
        return _dp_decode_matmul_program_config(device, input_shape, weight_shape)
    if kind == "auto":
        return None
    raise ValueError(f"Unknown program_config_kind: {kind}")


def _resolve_compute_kernel_config(case: MatmulCase, device):
    """Match ``_vision_matmul_compute_config`` / ``TTNNLinearLLama*`` settings.

    Both the vision tower and the text-decoder linears in dots.ocr use
    ``packer_l1_acc=True`` + ``math_approx_mode=True`` + ``fp32_dest_acc_en=False``;
    only the math_fidelity differs (LoFi for vision, HiFi2 for text). Use the
    vision helper for both -- it already encodes that exact tuple.
    """
    return _vision_matmul_compute_config(device, math_fidelity=case.math_fidelity)


def _allocate_inputs(case: MatmulCase, device):
    torch.manual_seed(0xD075)
    a_torch = torch.randn(1, 1, case.m, case.k, dtype=torch.bfloat16) * 0.1
    b_torch = torch.randn(case.k, case.n, dtype=torch.bfloat16) * 0.1

    a_tt = ttnn.from_torch(
        a_torch,
        dtype=case.in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_torch,
        dtype=case.in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return a_torch, b_torch, a_tt, b_tt


def _summarize(case: MatmulCase, device_time_us: float, pcc: float, pc_obj: Optional[object]) -> None:
    flops = 2 * case.m * case.k * case.n
    achieved_tflops = flops / max(device_time_us, 1e-6) / 1e6
    pc_name = type(pc_obj).__name__ if pc_obj is not None else "auto"
    logger.info(
        f"[{case.name:<24}] M={case.m:>5} K={case.k:>5} N={case.n:>5} "
        f"{case.math_fidelity.name:<5} {case.in0_kind}x{case.in1_kind}=>{case.out_kind} "
        f"| pc={pc_name} | dev={device_time_us:>7.1f} us "
        f"| TFLOPs={achieved_tflops:>5.1f} | pcc={pcc:.4f} "
        f"(target {case.pcc_target:.3f}, perf-log {case.expected_device_us:>5} us)"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
@pytest.mark.parametrize("num_iters", [10])
def test_dots_ocr_matmul_perf(device, case: MatmulCase, num_iters: int):
    """Run one dots.ocr-shape matmul and report device time + PCC.

    The device-time number is the mean wall-clock of ``num_iters`` back-to-back
    ``ttnn.matmul`` calls with a ``synchronize_device`` at the end. For a more
    precise per-call number wrap with Tracy (see module docstring).

    The PCC target is set per dtype-chain. K-side BFP4 (vision MLP) is the
    noisiest case at ~0.965; pure BF16/BFP8 paths land at ~0.997.
    """
    program_config = _resolve_program_config(case, device)
    compute_kernel_config = _resolve_compute_kernel_config(case, device)
    a_torch, b_torch, a_tt, b_tt = _allocate_inputs(case, device)

    out_tt = ttnn.matmul(
        a_tt,
        b_tt,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=case.out_dtype,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.synchronize_device(device)
    ttnn.deallocate(out_tt)

    start = time.perf_counter()
    last_out = None
    for _ in range(num_iters):
        if last_out is not None:
            ttnn.deallocate(last_out)
        last_out = ttnn.matmul(
            a_tt,
            b_tt,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=case.out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    ttnn.synchronize_device(device)
    elapsed_us = (time.perf_counter() - start) * 1e6
    avg_us = elapsed_us / max(num_iters, 1)

    out_torch = ttnn.to_torch(last_out).reshape(case.m, case.n).to(torch.float32)
    ref = a_torch.reshape(case.m, case.k).to(torch.float32) @ b_torch.to(torch.float32)

    pcc_passed, pcc_value = comp_pcc(ref, out_torch, case.pcc_target)
    pcc_value = float(pcc_value)

    _summarize(case, avg_us, pcc_value, program_config)

    ttnn.deallocate(last_out)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)

    assert pcc_passed, (
        f"{case.name} PCC {pcc_value:.4f} below target {case.pcc_target:.3f} "
        f"(M={case.m}, K={case.k}, N={case.n}, "
        f"{case.in0_kind}x{case.in1_kind}=>{case.out_kind}, {case.math_fidelity.name})"
    )
