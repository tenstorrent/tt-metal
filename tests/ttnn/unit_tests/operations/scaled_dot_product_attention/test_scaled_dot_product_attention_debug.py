# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug reproduction for the SDPA (Flash Attention) compute-kernel hang.

Authored by ttnn-expert-debugger. This is NOT the acceptance spec — it is a
minimal, deterministic harness for the single-tile (1,1,32,32) hang.

Run with per-thread device prints + watcher (the auto tt-triage is broken on
this branch — version mismatch — so use DEVICE_PRINT + watcher waypoints):

  TT_METAL_DPRINT_CORES=0,0 \
  TT_METAL_DPRINT_RISCVS=BR,NC,TR0,TR1,TR2 \
  TT_METAL_DEVICE_PRINT=1 \
  scripts/run_safe_pytest.sh --dev \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_debug.py::test_single_tile_hang_repro

Compute-core (0,0) waypoint decode at the hang (watcher.log, virtual 1,2):
  fields = BRISC, NCRISC, TRISC0(UNPACK), TRISC1(MATH), TRISC2(PACK)
    BRISC  = CWFW  -> writer waiting on cb_out (expected; output never produced)
    NCRISC = W     -> reader finished all reads
    TR0    = UPAD  -> UNPACK parked at end of an llk_unpack_A (datacopy), NOT in
                       the matmul AB unpack -> stuck entering the QK matmul
    TR1    = MWDD  -> MATH got DEST, raced ahead of UNPACK/PACK
    TR2    = K     -> PACK spinning with no LLK waypoint = the true blocker

Two distinct bugs were isolated (see the ttnn-expert-debugger RESULT commit):

  Bug #1 (FIXED in ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl): eltwise_chain
  never emitted the PackTile pack-format reconfig in its per-tile path, so the
  phase-0 scale-Q chain packed bf16 into cb_q while the packer was still set to
  cb_scores (fp32) from the boot mm_init -> page overrun + DEST desync -> PACK
  hung in phase 0. The fix emits emit_pre_element_transitions() in
  elem_apply_pack for PackTile elements.

  Bug #2 (ROOT-CAUSED + chain->matmul FIXED): a matmul_block FOLLOWING an
  eltwise_chain hung. DEVICE_PRINT inside matmul_block_helpers.inl pinned the
  hang to the OUTPUT pack_reconfig_data_format (gated on fp32_dest_acc_en /
  packer_l1_acc): PACK reached 'mm:before pack_reconfig' and never 'mm:after'.
  That reconfig ends in TTI_STALLWAIT(STALL_CFG, PACK|THCON)
  (cpack_common.h::reconfig_packer_data_format). Issued AFTER the matmul's
  tile_regs_wait (which sets up an fp32 DEST read), the STALLWAIT never drains
  when the matmul follows a foreign-op pack -> packer deadlock.

  Bisection (minimal boot + ONE eltwise_chain + ONE matmul_block harness, all
  on [1x1x32x32]) — NONE of the prescribed in-kernel resets cleared it:
    reduce_uninit<true>, copy_tile_init, reconfig+mm_block_init_short(None),
    no-op tile_regs cycle, full mm_init mid-kernel, pack_reconfig-after-chain,
    llk_pack_dest_init, dst_full_sync_en=True  -> ALL HANG.
  What DID clear it:
    fp32_dest_acc_en=False (reconfig skipped)            -> WORKS
    skip matmul's pack_reconfig (fp32 on)                -> WORKS
    issue matmul pack_reconfig BEFORE tile_regs_acquire  -> WORKS (the FIX)

  FIX (ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.inl): move the output
  pack_reconfig_data_format to BEFORE tile_regs_acquire (packer idle) instead of
  after tile_regs_wait. Clears the eltwise_chain -> matmul_block hang (phase
  0 -> A; verified P:phA + M:phA print on the real kernel).

  Bug #3 (RESIDUAL, separate boundary): with Bug #2 fixed the kernel now hangs at
  matmul -> reduce (phase A -> C). The reduce helper issues
  pack_reconfig_data_format(output) (reduce_helpers_compute.inl, default
  reconfig_mode INPUT_AND_OUTPUT) after the matmul's pack; in fp32 DEST mode the
  same STALLWAIT deadlocks. This is a matmul-exit / reduce-entry issue, NOT
  eltwise_chain -> matmul_block. Production SDPA sidesteps the whole class by
  using bf16 (Float16_b) intermediate CBs with fp32 DEST accumulation
  (sdpa_program_factory.cpp: "need to disable fp32 cbs", Issue #13364).
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _to_device(t, device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)], ids=lambda s: "x".join(map(str, s)))
def test_single_tile_hang_repro(shape, device):
    """Deterministic all-ones single-tile reproduction of the compute-kernel hang.

    All-ones makes every intermediate hand-calculable: QK = D = 32 per element
    (before scale), rowmax = 32*scale, exp(scores-max) = 1.0 everywhere,
    rowsum = 32, softmax weights = 1/32, output = mean(V) = 1.0. If/when the hang
    is fixed this test should PCC-match a trivial reference.
    """
    B, H, S, D = shape
    Q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    K = torch.ones(B, H, S, D, dtype=torch.bfloat16)
    V = torch.ones(B, H, S, D, dtype=torch.bfloat16)

    ttnn_out = scaled_dot_product_attention(
        _to_device(Q, device),
        _to_device(K, device),
        _to_device(V, device),
    )
    actual = ttnn.to_torch(ttnn_out).to(torch.float32)

    # All-ones self-attention output is mean(V) == 1.0 in every position.
    expected = torch.ones(B, H, S, D, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
