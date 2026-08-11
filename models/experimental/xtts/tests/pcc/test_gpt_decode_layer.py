# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 GPT DECODE forward on a SINGLE decoder block.

The companion to ``test_tt_gpt_prefill.py``: that one isolates PREFILL on one block
(``num_layers=1``) and gates on the KV cache it seeds; this one isolates the other
forward — ``forward_decode``, one token at a time over a FIXED-size KV cache — on the
same single block.

Decode is the harder of the two to get right and the one the model spends all its time
in: it reads the whole fixed cache under an additive position mask, writes the new K/V
in place at the current row, and runs a width-sharded LayerNorm the prefill path never
touches. Running it at ``num_layers=1`` means a regression shows up as a numeric miss on
*this* block rather than being diluted by 30 layers of averaging.

Design — OPEN LOOP, so a step's error cannot cascade into the next:

  * Each step feeds a fresh random hidden ``[1, 1, hidden]`` (NOT the previous step's
    output), starting from a zeroed cache at position 0. The reference is fed the same
    hiddens.
  * TT runs ``stack.forward_decode(x_i, kv, cache_pos(i), write_idx=i)`` — the production
    decode path, so the mask construction and the O(1) ``update_cache`` write are the real
    ones and not re-implemented here.
  * The reference re-runs the whole prefix ``x_0..x_i`` through the causal stack and takes
    the last position, which is by definition what an exact decode step must equal.

This walks the cache from empty to ``N_STEPS`` entries, so it covers decode SDPA at the
LOW cache depths where the mask is mostly -inf — the regime the fully-populated
end-to-end tests never isolate.

Gate: the MINIMUM per-step PCC over all steps (a mean would hide a single bad step).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/pcc/test_gpt_decode_layer.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_gpt_stack import reference_gpt_stack
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

NUM_DECODE_LAYERS = 1  # a SINGLE GPT decoder block (not the full 30-layer stack)
MAX_SEQ = 64  # fixed cache length (tile-aligned); decode attends over all of it, masked
N_STEPS = 10  # cache depths 0..9 — the low-depth regime


@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_decode_layer(device, xtts_state_dict, pcc, reset_seeds):
    sd = xtts_state_dict

    # One fresh hidden per decode step (open loop — never the previous step's output).
    steps = [torch.randn(1, 1, HIDDEN_SIZE) * 0.1 for _ in range(N_STEPS)]

    # Reference: block 0 + ln_f, full causal recompute of the prefix at every step.
    reference = reference_gpt_stack(sd, num_layers=NUM_DECODE_LAYERS)

    # TT: the model owns the fixed KV cache and the position tensors the decode forward needs;
    # build it at depth 1 and drive its stack's decode forward directly.
    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_DECODE_LAYERS)
    tt_model.alloc_static_kv(MAX_SEQ)  # zeroed [1, heads, MAX_SEQ, head_dim] per layer

    pccs = []
    for i, x in enumerate(steps):
        with torch.no_grad():
            ref_step = reference(torch.cat(steps[: i + 1], dim=1))[:, -1:, :]  # [1, 1, hidden]

        # forward_decode consumes (deallocates) its input, so upload a fresh tensor each step.
        x_tt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_step = tt_model.stack.forward_decode(x_tt, tt_model._static_kv, tt_model.cache_pos(i), write_idx=i)
        tt_step = ttnn.to_torch(tt_step).float().reshape(1, 1, HIDDEN_SIZE)

        _, step_pcc = comp_pcc(ref_step, tt_step, pcc)
        pccs.append(float(step_pcc))
        logger.info(f"decode step {i} (cache depth {i}): PCC {step_pcc}")

    logger.info(comp_allclose(ref_step, tt_step))  # last step, as a magnitude sanity check
    logger.info(f"single-layer decode PCC over {N_STEPS} steps: min={min(pccs):.6f} mean={sum(pccs)/len(pccs):.6f}")

    worst = pccs.index(min(pccs))
    assert min(pccs) >= pcc, f"single-layer decode PCC below {pcc} at step {worst} (cache depth {worst}): {min(pccs)}"
