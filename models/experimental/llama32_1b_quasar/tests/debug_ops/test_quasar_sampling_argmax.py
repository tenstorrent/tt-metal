# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test: the greedy sampling / argmax path consuming the lm_head's DRAM logits.

After the lm_head concat L1->DRAM fix (_install_quasar_concat_l1_overflow_to_dram), the full-vocab logits
[1, 1, 32, VOCAB=128256] live in DRAM. The greedy decode path then turns them into token ids
(sampling_1d.py::_sample_argmax):

    x_untilized = ttnn.untilize(logits, use_multicore=True)
    tt_out_tok  = ttnn.argmax(x_untilized, dim=-1, keepdim=False)

The e2e has not exercised this on Quasar yet (it died at the lm_head concat, before sampling), so this isolates
the untilize + argmax over a DRAM logits tensor: it verifies both ops accept the DRAM input on Quasar (no
L1/sharded assert, no Gen1-only fallback) AND return the correct argmax token per row. (The masking / all-gather
/ vocab-slice around _sample_argmax are multi-device / vocab-padding concerns; single-device greedy reduces to
untilize + argmax, which is what we test.)

This test also guards the argmax Quasar DFB-format fix: argmax hard-forces its output dtype to UINT32, and its
multicore factory derived the intermediate/output DFB data format straight from that -> DataFormat::UInt32,
which Quasar's is_data_format_supported rejects (it supports Int32 / RawUInt32, not UInt32). The factory now
remaps that DFB format to the byte-identical RawUInt32 on Quasar (the output tensor dtype stays UINT32), so the
raw 4-byte index storage validates. Without that fix this test FATALs at program creation
("DFB 'dst' has data format 'UInt32' which is not supported on architecture quasar").

Each of the 32 rows gets one clearly-dominant logit at a known random column, so the argmax is unambiguous in
bf16 and the expected token ids are exact. Inputs built bf16 via RM upload + quasar.tilize (from_torch(TILE)
hangs on the sim); VOCAB is the real 128256.

Run (Quasar sim, 2-node emulator, SLOW dispatch):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_sampling_argmax.py
"""

import pytest
import torch
from loguru import logger

import ttnn

VOCAB = 128256  # llama-3.2-1B (padded) vocab
ROWS = 32  # decode batch padded to a tile


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


@pytest.mark.timeout(3600)
def test_argmax_dram_logits(mesh_device):
    """untilize + argmax over full-vocab DRAM logits (the greedy sampling path). Validates the ops run on Quasar
    with a DRAM input and return the correct token id per row."""
    torch.manual_seed(0)
    logits = torch.randn(1, 1, ROWS, VOCAB, dtype=torch.bfloat16) * 0.1
    # Inject one clearly-dominant logit per row at a known column so argmax is unambiguous in bf16.
    winners = torch.randint(0, VOCAB, (ROWS,))
    for r in range(ROWS):
        logits[0, 0, r, int(winners[r])] = 10.0
    logits_t = _tile_bf16_dram(logits, mesh_device)  # DRAM TILE [1,1,ROWS,VOCAB]

    x_unt = ttnn.untilize(logits_t, use_multicore=True)  # mirror _sample_argmax
    tok = ttnn.argmax(x_unt, dim=-1, keepdim=False)  # [1,1,ROWS] token ids
    ttnn.synchronize_device(mesh_device)
    got = ttnn.to_torch(tok).reshape(-1).to(torch.int64)  # [ROWS]

    ref = winners.to(torch.int64)
    match = (got == ref).float().mean().item()
    logger.info(
        f"[argmax-dram] match={match * 100:.1f}% got[:4]={got[:4].tolist()} ref[:4]={ref[:4].tolist()} "
        f"shape={tuple(ttnn.to_torch(tok).shape)}"
    )
    assert got.numel() == ROWS, f"expected {ROWS} tokens, got {got.numel()} (shape {tuple(got.shape)})"
    assert (
        match == 1.0
    ), f"argmax over DRAM logits mismatched: {match * 100:.1f}% (got {got.tolist()} vs {ref.tolist()})"
