# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the one-shot parallel prefill (ttnn.fill_cache) of TTNNGPTTracedDecoder.

Validation for both device tests: prefill positions 0..P-1 in one batched pass, then decode past P
from the traced step. Because attention is causal those latents must match the CPU reference's
one-pass latents at the same positions, which holds only if the cache was seeded correctly.

  * one shape, deep  — prefill half the reference sequence and decode the whole second half, held
    to the tight gate against the standard golden.
  * every shape      — the same idea at all 16 tile counts, a short decode window each. One length
    cannot catch a length-dependent fault: fill_cache splits n_head*tiles blocks over the cores as
    consecutive runs and hands each core only its first cache address, so at some tile counts a run
    crosses a head boundary and those positions are seeded with zeros. The damage is bit-identical
    on every repeat, so only sweeping the length sees it (BUG-7 in the bringup notes).
  * pad arithmetic   — no device: _prefill_tiles must never return a straddling shape.

Both device tests run the serving order (prefill after capture), so they also cover the shape the
request path actually uses. The reference is computed live (see tests/reference_helpers.py).

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_prefill_pcc.py
"""

import pytest
import torch
import ttnn

# Eager GPT2 import, before any XTTS checkpoint load — see tests/reference_helpers.py.
from transformers import GPT2Model  # noqa: F401

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_gpt_ref import build_reference, make_golden_input, reference_forward
from models.experimental.xtts_v2.tests.reference_helpers import gpt_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder, _prefill_tiles

TARGET_PCC = 0.999  # one shape against the standard golden
# The sweep's bar sits lower: bf16 error grows with context and varies by position, and it uses a
# long synthetic input. Still far above a straddled prefill, which lands in the 0.6-0.97 range.
TARGET_PCC_LENGTHS = 0.998
TRACE_REGION = 50_000_000
N_HEAD, GRID_CORES = 16, 64  # test_prefill_pad_is_safe is pure arithmetic, so it states its shape
MAX_TILES = 16  # 512 rows: 32 conditioning + 404 text + START, the model's longest prompt
DECODE_WINDOW = 4  # positions decoded per length; one latent alone is a noisy PCC
LENGTHS = tuple(32 * t for t in range(1, MAX_TILES + 1))


def _decoder(device, max_seq):
    return TTNNGPTTracedDecoder(device, preprocess_gpt_parameters(device, dtype=ttnn.bfloat16), max_seq=max_seq)


def _step(dec, device, row, pos):
    """One traced decode step, host [1,1,1024] row in -> host latent out."""
    emb = ttnn.from_torch(
        row.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=dec.mesh_mapper
    )
    return ttnn.to_torch(dec.step_device(emb, pos)).to(torch.float32)


def run_prefill_pcc(device):
    ref = gpt_reference()
    inputs_embeds, golden = ref["inputs_embeds"], ref["latents"]
    S = inputs_embeds.shape[1]
    split = S // 2  # prefill the first half in one shot, decode the second half

    dec = _decoder(device, ((S + 31) // 32) * 32)
    dec.reset_caches()
    dec.prefill(inputs_embeds[:, :split, :].contiguous())
    dec.capture()  # capture AFTER prefill; leaves the prefilled cache intact

    lat = [_step(dec, device, inputs_embeds[:, t : t + 1, :], t) for t in range(split, S)]
    decoded = torch.cat(lat, dim=1)  # [1, S-split, 1024]
    passed, pcc_msg = comp_pcc(golden[:, split:, :], decoded, pcc=TARGET_PCC)
    print(f"  split={split} decoded {tuple(decoded.shape)} vs golden[{split}:]  pcc: {pcc_msg}")
    return passed, pcc_msg


def run_prefill_lengths_pcc(device, verbose=True):
    S = LENGTHS[-1] + DECODE_WINDOW  # room to decode past the longest prefill
    inputs_embeds = make_golden_input(n_text=105, n_mel=S - 105)
    _, golden = reference_forward(*build_reference(), inputs_embeds)

    dec = _decoder(device, S)
    dec.reset_caches()
    dec.prefill(inputs_embeds[:, : LENGTHS[0], :].contiguous())  # compile before capture
    dec.capture()
    grid = device.compute_with_storage_grid_size()

    scored = []
    for P in LENGTHS:
        dec.reset_caches()
        dec.prefill(inputs_embeds[:, :P, :].contiguous())
        end = P + DECODE_WINDOW
        lat = [_step(dec, device, inputs_embeds[:, t : t + 1, :], t) for t in range(P, end)]
        passed, pcc_msg = comp_pcc(golden[:, P:end, :], torch.cat(lat, dim=1), pcc=TARGET_PCC_LENGTHS)
        if verbose:
            pad = 32 * _prefill_tiles(P, dec.config.n_head, grid.x * grid.y)
            print(f"  P={P:4d} ({P // 32:2d} tiles, padded to {pad:4d})  {pcc_msg}")
        scored.append((P, passed, pcc_msg))

    failed = [(P, m) for P, ok, m in scored if not ok]
    return not failed, f"{len(scored)} lengths, {len(failed)} below {TARGET_PCC_LENGTHS}: {failed}"


def test_prefill_pad_is_safe():
    """_prefill_tiles must never hand fill_cache a shape where a core's run can straddle a head."""
    for P in range(1, 32 * MAX_TILES + 1):
        t = _prefill_tiles(P, N_HEAD, GRID_CORES)
        assert 32 * t >= P, f"P={P} padded down to {32 * t}"
        blocks = N_HEAD * t
        assert blocks <= GRID_CORES or blocks % GRID_CORES == 0, f"P={P} -> {t} tiles straddles"


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_prefill_pcc(device):
    passed, pcc_msg = run_prefill_pcc(device)
    assert passed, f"parallel-prefill decode PCC below {TARGET_PCC}: {pcc_msg}"


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_prefill_lengths_pcc(device):
    passed, msg = run_prefill_lengths_pcc(device)
    assert passed, f"prefill drifted from the reference at some prompt length: {msg}"


if __name__ == "__main__":
    import sys

    test_prefill_pad_is_safe()
    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        dev.enable_program_cache()
        results = [run_prefill_pcc(dev), run_prefill_lengths_pcc(dev)]
    finally:
        ttnn.close_device(dev)
    ok = all(r[0] for r in results)
    print(("PASSED " if ok else "FAILED ") + "; ".join(str(r[1]) for r in results))
    sys.exit(0 if ok else 1)
