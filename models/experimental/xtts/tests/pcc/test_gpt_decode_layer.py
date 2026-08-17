# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_gpt_stack import reference_gpt_stack
from models.experimental.xtts.tt.xtts_gpt_model import TtXttsGptModel

NUM_DECODE_LAYERS = 1
MAX_SEQ = 64
N_STEPS = 10


@pytest.mark.parametrize("pcc", [0.99])
def test_tt_gpt_decode_layer(device, xtts_state_dict, pcc, reset_seeds):
    """Compare single-layer TTNN GPT decode steps to the PyTorch reference via PCC."""
    sd = xtts_state_dict

    steps = [torch.randn(1, 1, HIDDEN_SIZE) * 0.1 for _ in range(N_STEPS)]

    reference = reference_gpt_stack(sd, num_layers=NUM_DECODE_LAYERS)

    tt_model = TtXttsGptModel(sd, device, num_layers=NUM_DECODE_LAYERS)
    tt_model.alloc_static_kv(MAX_SEQ)

    pccs = []
    for i, x in enumerate(steps):
        with torch.no_grad():
            ref_step = reference(torch.cat(steps[: i + 1], dim=1))[:, -1:, :]

        x_tt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_step = tt_model.stack.forward_decode(x_tt, tt_model._static_kv, tt_model.cache_pos(i), write_idx=i)
        tt_step = ttnn.to_torch(tt_step).float().reshape(1, 1, HIDDEN_SIZE)

        _, step_pcc = comp_pcc(ref_step, tt_step, pcc)
        pccs.append(float(step_pcc))
        logger.info(f"decode step {i} (cache depth {i}): PCC {step_pcc}")

    logger.info(comp_allclose(ref_step, tt_step))
    logger.info(f"single-layer decode PCC over {N_STEPS} steps: min={min(pccs):.6f} mean={sum(pccs)/len(pccs):.6f}")

    worst = pccs.index(min(pccs))
    assert min(pccs) >= pcc, f"single-layer decode PCC below {pcc} at step {worst} (cache depth {worst}): {min(pccs)}"
