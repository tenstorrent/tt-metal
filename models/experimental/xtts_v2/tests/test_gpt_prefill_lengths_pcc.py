# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the one-shot prefill across every prompt length the model can reach.

test_gpt_prefill_pcc validates fill_cache at ONE length. That is not enough: fill_cache splits
n_head*tiles blocks over the cores as consecutive runs and hands each core only its first cache
address, so at some tile counts a run crosses a head boundary and those positions are seeded with
zeros instead of K/V. The damage is length-dependent and bit-identical on every repeat, so neither
a determinism check nor a fixed-shape PCC test sees it — only sweeping the length does.

Validation: for each tile count, prefill positions 0..P-1 in one shot, decode the next DECODE_WINDOW
positions from the traced step, and compare those latents against the CPU reference's parallel
forward. Because attention is causal they must match, which holds only if the cache was seeded
correctly. Runs the serving order (prefill after capture), so it also covers the shape the request
path actually uses.
"""

import pytest
import torch
import ttnn

# Eager GPT2 import, before any XTTS checkpoint load — see tests/reference_helpers.py.
from transformers import GPT2Model  # noqa: F401

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_gpt_ref import build_reference, make_golden_input, reference_forward
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder, _prefill_tiles

# This gate is about SHAPE coverage; test_gpt_prefill_pcc holds the tight numeric line at one
# length. bf16 error here grows with context and varies by position, so the bar sits below
# that — still far above a straddled prefill, which lands in the 0.8-0.96 range.
TARGET_PCC = 0.998
TRACE_REGION = 50_000_000
N_HEAD, GRID_CORES = 16, 64  # test_prefill_pad_is_safe is pure arithmetic, so it states its shape
MAX_TILES = 16  # 512 rows: 32 conditioning + 404 text + START, the model's longest prompt
DECODE_WINDOW = 4  # positions decoded per length; one latent alone is a noisy PCC
LENGTHS = tuple(32 * t for t in range(1, MAX_TILES + 1))


def run_prefill_lengths_pcc(device, verbose=True):
    S = LENGTHS[-1] + DECODE_WINDOW  # room to decode past the longest prefill
    inputs_embeds = make_golden_input(n_text=105, n_mel=S - 105)
    _, golden = reference_forward(*build_reference(), inputs_embeds)

    dec = TTNNGPTTracedDecoder(device, preprocess_gpt_parameters(device, dtype=ttnn.bfloat16), max_seq=S)
    g = device.compute_with_storage_grid_size()
    dec.reset_caches()
    dec.prefill(inputs_embeds[:, : LENGTHS[0], :].contiguous())  # compile before capture
    dec.capture()

    scored = []
    for P in LENGTHS:
        dec.reset_caches()
        dec.prefill(inputs_embeds[:, :P, :].contiguous())
        lat = []
        for t in range(P, P + DECODE_WINDOW):
            emb = ttnn.from_torch(
                inputs_embeds[:, t : t + 1, :].contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=dec.mesh_mapper,
            )
            lat.append(ttnn.to_torch(dec.step_device(emb, t)).to(torch.float32))
        end = P + DECODE_WINDOW
        passed, pcc_msg = comp_pcc(golden[:, P:end, :], torch.cat(lat, dim=1), pcc=TARGET_PCC)
        if verbose:
            pad = 32 * _prefill_tiles(P, dec.config.n_head, g.x * g.y)
            print(f"  P={P:4d} ({P // 32:2d} tiles, padded to {pad:4d})  {pcc_msg}")
        scored.append((P, passed, pcc_msg))

    failed = [(P, m) for P, ok, m in scored if not ok]
    return not failed, f"{len(scored)} lengths, {len(failed)} below {TARGET_PCC}: {failed}"


def test_prefill_pad_is_safe():
    """_prefill_tiles must never hand fill_cache a shape where a core's run can straddle a head."""
    for P in range(1, 32 * MAX_TILES + 1):
        t = _prefill_tiles(P, N_HEAD, GRID_CORES)
        assert 32 * t >= P, f"P={P} padded down to {32 * t}"
        blocks = N_HEAD * t
        assert blocks <= GRID_CORES or blocks % GRID_CORES == 0, f"P={P} -> {t} tiles straddles"


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_prefill_lengths_pcc(device):
    passed, msg = run_prefill_lengths_pcc(device)
    assert passed, f"prefill drifted from the reference at some prompt length: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        dev.enable_program_cache()
        ok, msg = run_prefill_lengths_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
