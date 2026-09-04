# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/lm_head.py` vs an fp32 torch reference. Owned by P9 item 9 (every `tt/` module owns a test).

The LM head is one column-parallel matmul, so it is a **stage** in recipe §2.2's sense and carries
the 3x stage budget rather than a block budget. It exists so `G-MODEL` can gate top-1 token
agreement, and this file gates the two things `G-MODEL` would otherwise have to take on trust: the
matmul's own precision, and that the vocab axis is not permuted.

* **Input distribution:** `x` standard normal `[1, 1, 32, 4096]` — the last-token tile, which is the
  only shape the head is ever fed on the prefill path. Weight `randn * 0.02`, the scale both
  templates use.
* **Reference dtype policy:** fp32 weight, fp32 input, fp32 matmul. The floor quantises **only** the
  input and the weight (recipe §2.2) — nothing internal, since a matmul has no intermediates this
  test can see.
* **Computed noise floor:** the fp32 reference against the same matmul with both operands rounded
  to the device dtypes (`x` bf16, weight bf8_b or bf16).
* **Negative control:** the vocab rows rolled by one. The PCC must collapse **and** top-1 must
  change — this is what proves the gate is sensitive to *which* row is which logit, which PCC alone
  would barely notice (§2.5).

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_lm_head_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.config import MeshConfig
from models.demos.llama31_8b_d_p.tt.lm_head import LMHead

ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
WEIGHT_SCALE = 0.02
MAX_STAGE_ERR_RATIO = 3.0  # recipe §2.2's stage budget; the head is one matmul
PCC_THRESHOLD = 0.999

_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat16]
_DTYPE_IDS = {ttnn.bfloat8_b: "bf8_b", ttnn.bfloat16: "bf16"}

# The real vocab: 128256 / 8 = 16032 = 501 * 32, so no padding is needed at the deployment TP and
# none is implemented (`tt/lm_head.py`). Kept at the real value here because the tile-alignment
# claim is what the module asserts.
VOCAB = llama_config_dims()["vocab_size"]


def _to_device(t, mesh_device, dtype=ACTIVATION_DTYPE):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("weight_dtype", _DTYPES, ids=lambda d: _DTYPE_IDS[d])
def test_lm_head_vs_ref(mesh_device, weight_dtype, reset_seeds):
    """One column-parallel matmul: PCC >= 0.999 and <= 3x its computed floor."""
    hf = llama_config_dims()
    hidden = hf["hidden_size"]
    x = torch.randn(1, 1, ttnn.TILE_SIZE, hidden)
    weight = torch.randn(VOCAB, hidden) * WEIGHT_SCALE  # HF `[vocab, hidden]`

    ref = x @ weight.transpose(-1, -2)
    # Floor: both operands rounded to what the device holds, in the orientation it holds them.
    w_dev = quantize_like_device(weight.transpose(-1, -2)[None, None], weight_dtype)[0, 0]
    floor_out = quantize_like_device(x, ACTIVATION_DTYPE) @ w_dev
    _, floor = comp_pcc(ref, floor_out, 0.0)

    head = LMHead(
        mesh_device,
        hf,
        {"weight": weight},
        mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        weight_dtype=weight_dtype,
    )
    out = _from_device(head(_to_device(x, mesh_device)))

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    ratio = err_ratio(float(pcc), float(floor))
    top1_ref = int(ref[0, 0, -1].argmax())
    top1_dev = int(out[0, 0, -1].argmax())
    logger.info(
        f"[LM-HEAD] {_DTYPE_IDS[weight_dtype]}: PCC={float(pcc):.7f} floor={float(floor):.7f} "
        f"ratio={ratio:.2f}x (budget {MAX_STAGE_ERR_RATIO}x); top-1 ref={top1_ref} dev={top1_dev}"
    )
    assert passing, f"below threshold {PCC_THRESHOLD}: {pcc}"
    assert ratio <= MAX_STAGE_ERR_RATIO, f"the head is {ratio:.2f}x off its floor (stage budget 3x)"
    assert top1_dev == top1_ref, f"top-1 disagrees on the last row: ref {top1_ref}, device {top1_dev}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_lm_head_vocab_row_map_negative_control(mesh_device, reset_seeds):
    """**The negative control:** vocab rows rolled by one. PCC must collapse and top-1 must move."""
    hf = llama_config_dims()
    hidden = hf["hidden_size"]
    x = torch.randn(1, 1, ttnn.TILE_SIZE, hidden)
    weight = torch.randn(VOCAB, hidden) * WEIGHT_SCALE

    ref = x @ weight.transpose(-1, -2)
    head = LMHead(mesh_device, hf, {"weight": torch.roll(weight, 1, dims=0)}, weight_dtype=ttnn.bfloat16)
    out = _from_device(head(_to_device(x, mesh_device)))
    _, pcc = comp_pcc(ref, out, 0.0)

    top1_ref = int(ref[0, 0, -1].argmax())
    top1_dev = int(out[0, 0, -1].argmax())
    logger.info(f"[LM-HEAD] control: vocab rolled by one -> PCC={float(pcc):.5f}, top-1 {top1_ref} -> {top1_dev}")
    assert float(pcc) < PCC_THRESHOLD, f"a rolled vocab axis still scores {float(pcc)} — the gate is blind to it"
    assert top1_dev != top1_ref, "a rolled vocab axis left top-1 unchanged — the top-1 check is blind"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_lm_head_refuses_to_build_weightless(mesh_device, expect_error):
    """No `state_dict` and no cache path must fail loud, not project through a `None` weight."""
    with expect_error(ValueError, "tensor_cache_path"):
        LMHead(mesh_device, llama_config_dims(), {})


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_lm_head_refuses_a_transposed_weight(mesh_device, expect_error, reset_seeds):
    """`[hidden, vocab]` instead of HF's `[vocab, hidden]` must be refused at load.

    Both are legal tensors and the matmul would run on neither silently — but the assert names the
    expected orientation, which is what stops the "transpose applied twice" bug `G-WEIGHTS` also
    covers.
    """
    hf = llama_config_dims()
    with expect_error(AssertionError, "tie_word_embeddings is false"):
        LMHead(mesh_device, hf, {"weight": torch.randn(hf["hidden_size"], VOCAB) * WEIGHT_SCALE})
