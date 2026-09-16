# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ttpoly.precision — the bit-exact silicon model, single owner of arithmetic.

  bf16        — BF16/FP16 quantization, FTZ, BF16 grid
  fma         — bit-exact SFPU fused multiply-add (fma_bh / fma_wh)
  reciprocal  — modeled SFPU reciprocal (NEW) with declared ULP budget
  eval        — THE single coefficients->output evaluator
"""
