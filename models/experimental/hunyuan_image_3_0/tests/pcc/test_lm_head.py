# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device PCC test for the text-gen LM head (tt/lm_head.py) vs a torch Linear, on the
# real checkpoint `lm_head.weight` ([133120, 4096], untied). Validates both the
# full-sequence and last-token-only projection paths.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_lm_head.py -v -s

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead

H = 4096
PCC_THR = 0.99


@pytest.fixture(scope="module")
def lm_head_state():
    index = MODEL_DIR / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"checkpoint not found at {MODEL_DIR}")
    return load_tensors(MODEL_DIR, ["lm_head.weight"])


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("last_token_only", [False, True])
def test_lm_head_pcc(device, lm_head_state, last_token_only):
    torch.manual_seed(0)
    S = 32
    w = lm_head_state["lm_head.weight"].float()  # [V, H]
    V = w.shape[0]
    hidden = torch.randn(1, S, H)

    ref = lm_head_logits(hidden, w)  # ref/lm_head.py golden -> [1, S, V]
    if last_token_only:
        ref = ref[:, -1:, :]

    head = HunyuanTtLMHead(device, lm_head_state)
    assert head.vocab_size == V
    h_tt = ttnn.from_torch(
        hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = head(h_tt, last_token_only=last_token_only)
    out = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)
    ttnn.deallocate(h_tt)

    assert tuple(out.shape) == tuple(ref.shape), f"{tuple(out.shape)} != {tuple(ref.shape)}"
    passing, pcc = comp_pcc(ref, out, PCC_THR)
    logger.info(f"lm_head last_only={last_token_only} V={V} PCC={pcc} (passing={passing})")
    assert passing, f"PCC {pcc} < {PCC_THR}"
    # greedy argmax token must agree (the only thing sampling actually consumes)
    assert int(ref[0, -1].argmax()) == int(out[0, -1].argmax()), "argmax token mismatch"
