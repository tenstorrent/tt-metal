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

  Bug #2 (OPEN): a matmul_block immediately FOLLOWING an eltwise_chain still
  hangs (phase 0 chain -> phase A QK matmul; and with phase 0 removed, phase E
  exp chain -> phase F matmul-reduce -> phase H PV matmul hangs at H). The
  `reduce`-with-matmul path survives a preceding chain, but `matmul_block` does
  not. A full mm_block_init (pack-sync re-init) before the matmul did NOT fix it,
  so the residual is a deeper chain-exit state (DEST bank parity / datacopy
  unpacker MOP) that matmul_block's init does not reconcile.
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
