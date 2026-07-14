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
#
# Production (slow, last-token @ S=4160):
#   HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_lm_head.py -k lm_head_production -v -s

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from pcc_common import PCC_STRICT, PRODUCTION_SEQ

H = 4096
PCC_THR = 0.99


def assert_argmax_agree(ref_row: torch.Tensor, out_row: torch.Tensor) -> None:
    """Greedy next-token must agree, tolerating bf16-indistinguishable ties.

    The device head runs the [H, V] projection in bf16, whose ~8-bit mantissa gives
    a ULP of ~|x|·2⁻⁸ at the top-logit magnitude. When the fp32 top-1/top-2 gap is
    smaller than that, the two tokens are *effectively tied* and bf16 rounding may
    reorder them — no bf16 matmul can resolve it, so requiring the exact same index
    is spurious. Accept the TT token iff its *reference* logit is within a couple of
    bf16 ULPs of the reference max (i.e. it is one of the tied leaders).
    """
    ref_arg = int(ref_row.argmax())
    out_arg = int(out_row.argmax())
    if ref_arg == out_arg:
        return
    ref_top = ref_row.max().item()
    ref_at_out = ref_row[out_arg].item()
    tie_tol = abs(ref_top) * 2**-7  # ≈ 2 bf16 ULPs at the top-logit magnitude
    assert ref_top - ref_at_out <= tie_tol, (
        f"argmax token mismatch beyond bf16 tie tolerance: ref {ref_arg} ({ref_top:.4f}) "
        f"vs out {out_arg} (its ref logit {ref_at_out:.4f}, gap {ref_top - ref_at_out:.4f} > tol {tie_tol:.4f})"
    )


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
    assert_argmax_agree(ref[0, -1], out[0, -1])


@pytest.mark.slow
def test_lm_head_production_pcc(device, lm_head_state):
    """Standalone lm_head last-token logits at production prefill length (S=4160)."""
    torch.manual_seed(0)
    S = PRODUCTION_SEQ
    w = lm_head_state["lm_head.weight"].float()
    V = w.shape[0]
    hidden = torch.randn(1, S, H)

    ref = lm_head_logits(hidden, w)[:, -1:, :]

    head = HunyuanTtLMHead(device, lm_head_state)
    assert head.vocab_size == V
    h_tt = ttnn.from_torch(
        hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = head(h_tt, last_token_only=True)
    out = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)
    ttnn.deallocate(h_tt)

    assert tuple(out.shape) == (1, 1, V), f"{tuple(out.shape)} != (1, 1, {V})"
    passing, pcc = comp_pcc(ref, out, PCC_STRICT)
    logger.info(f"lm_head production last-token S={S} V={V} PCC={pcc} (passing={passing})")
    assert passing, f"PCC {pcc} < {PCC_STRICT}"
    assert_argmax_agree(ref[0, -1], out[0, -1])
