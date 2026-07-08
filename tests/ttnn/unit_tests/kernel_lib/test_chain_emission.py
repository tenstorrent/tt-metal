# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
G5 — emission / codegen asserts for eltwise_chain (inspect the compiled artifact, not values).

Coverage spec: ttnn/cpp/ttnn/kernel_lib/docs/eltwise_helper_test_coverage.html (group G5).

Some helper behavior is invisible to PCC. The one with a real precedent is INLINING: a CPS lambda
once defeated always_inline and pushed the Tensix MATH body out of line, producing NaN
(project_eltwise_chain_no_inline_lambda). EM-03 guards that structurally — it disassembles the
trisc1 (MATH) kernel and asserts the chain body is fully inlined (no out-of-line chain-helper
symbols). This runs after the kernel JITs; the ELF is on disk even on a cache hit.

EM-01/02/04 (reconfig-elision instruction counts, no-engine-init, block-clamp at asm) need finer
asm diffing and are tracked in TEST_DECISION_LOG.md as follow-up — they are more fragile than a
symbol-table scan, so they are deliberately not shipped here as flaky asserts.
"""

import glob
import os
import subprocess

import torch
import pytest
import ttnn
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from loguru import logger

NM = "runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
# A multi-element chain (CopyTile -> Exp -> PackTile) — enough elements that a non-inlined
# exec body would show up as its own symbol.
KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_single_call.cpp"
KERNEL_STEM = "hoist_single_call"
# Demangled symbol fragments that should NEVER appear as out-of-line functions if the chain inlines.
OUT_OF_LINE_MARKERS = ("compute_kernel_lib", "eltwise_chain", "CopyTile", "PackTile", "::exec")


def _jit_compile(device, n=4):
    """Build + run the chain kernel so its trisc artifacts exist on disk."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    _, tt_in = lib.make_input(shape, dt, device, seed=1001)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_in, tt_out], program)


def _newest_trisc_elf(kernel_stem, trisc):
    paths = glob.glob(f"built/tt-metal-cache*/kernels/{kernel_stem}/*/{trisc}/{trisc}.elf")
    assert paths, f"no {trisc} ELF found for {kernel_stem} — did it compile?"
    return max(paths, key=os.path.getmtime)


def _demangled_text_symbols(elf):
    out = subprocess.run([NM, "-C", elf], capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if " t " in f" {ln} " or " T " in f" {ln} "]


def test_em03_chain_body_is_inlined(device):
    """EM-03: the MATH (trisc1) kernel must contain NO out-of-line chain-helper symbols — the chain
    body is fully inlined into the kernel entry. A regression that broke always_inline (the NaN bug)
    would surface an out-of-line exec function here."""
    _jit_compile(device)
    elf = _newest_trisc_elf(KERNEL_STEM, "trisc1")
    syms = _demangled_text_symbols(elf)
    offenders = [s for s in syms if any(m in s for m in OUT_OF_LINE_MARKERS)]
    logger.info(f"EM-03 trisc1 text symbols={len(syms)} | offenders={len(offenders)} | {elf}")
    assert (
        not offenders
    ), "chain body is NOT fully inlined — out-of-line helper symbols found (the no-inline NaN class):\n" + "\n".join(
        offenders
    )


def test_em03_pack_body_is_inlined(device):
    """Same inlining guard on the PACK (trisc2) kernel — the PackTile path must inline too."""
    _jit_compile(device)
    elf = _newest_trisc_elf(KERNEL_STEM, "trisc2")
    syms = _demangled_text_symbols(elf)
    offenders = [s for s in syms if any(m in s for m in OUT_OF_LINE_MARKERS)]
    logger.info(f"EM-03 trisc2 text symbols={len(syms)} | offenders={len(offenders)}")
    assert not offenders, "pack body not fully inlined:\n" + "\n".join(offenders)
