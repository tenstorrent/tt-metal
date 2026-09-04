# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/lm_head.py` vs the torch reference, `(1,1)` mesh, TP=1, no CCL. Gate `G-CLEAN` item 9.

**Why this file exists.** Same finding as `test_embedding_vs_ref.py`: P9's test-inventory sweep found
`tt/lm_head.py` had no test naming it, only transitive coverage through `tt/model.py` in
`tests/unit/test_model_vs_ref.py`. There the LM head is scored by **top-1 agreement** — a single
`argmax` over 128256 logits — which is a strong end-to-end statement and a very coarse one: the head
can be measurably wrong everywhere except at the winning token and still pass. `DEC-122`.

**Threshold, and why it is not the plain 0.999 an ungated eye would pick.** The head is one
`bfloat8_b` matmul, so the right reference point is the *storage* noise floor: the same fp32 matmul
run on weights first quantised to bf8_b through ttnn's own quantiser. The gate is
`PCC >= 0.999` **and** `<= 3x` that floor, which is the P5 module convention
(`BRINGUP_RECIPE.md` Appendix E.2 — gate on the gap to the noise floor, never on another
implementation's absolute PCC).

`vocab_size` is reduced to 8192 (`dataclasses.replace`) so the `[vocab, hidden]` weight is 128 MiB
in fp32 rather than 2.0 GiB, and — deliberately — it is **not** equal to `hidden_size` (4096), so the
`[out, in] -> [in, out]` transpose the module does at load time cannot be wrong-but-square.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_lm_head_vs_ref.py -x -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, err_ratio, quantize_like_device
from models.demos.llama31_8b_d_p.tt.lm_head import LMHead

PCC_THRESHOLD = 0.999
MAX_ERR_RATIO = 3.0  # BRINGUP_RECIPE.md Appendix E.2 — the single-op module convention.

TEST_VOCAB = 8192  # != hidden_size (4096), tile-aligned, divisible by every TP this package allows.
WEIGHT_SCALE = 0.02  # ~ the checkpoint's own lm_head scale; keeps logits off bf8_b's saturation.


def _small_vocab_config(hf_config):
    return replace(hf_config, vocab_size=TEST_VOCAB)


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 512], ids=["s32", "s512"])
@torch.no_grad()
def test_lm_head_vs_ref(mesh_device, seq_len, reset_seeds):
    """`[1,1,S,4096]` -> `[1,1,S,8192]`: PCC >= 0.999 and within 3x the bf8_b storage floor."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    hidden = hf_config.hidden_size

    # HF layout is [out, in] = [vocab, hidden]; the module transposes once, at load.
    weight = torch.randn(TEST_VOCAB, hidden, dtype=torch.float32) * WEIGHT_SCALE
    x = torch.randn(1, 1, seq_len, hidden, dtype=torch.float32)

    # --- reference: fp32 linear, no bias (Llama-3.1 has none)
    ref = torch.nn.functional.linear(x, weight)

    # --- the floor: the same fp32 math on exactly the values the device stores
    w_q = quantize_like_device(weight.transpose(0, 1).unsqueeze(0).unsqueeze(0), ttnn.bfloat8_b)[0, 0]
    x_q = quantize_like_device(x, ttnn.bfloat16)
    floor = torch.matmul(x_q, w_q)
    _, floor_pcc = comp_pcc(ref, floor, 0.0)

    # --- device
    head = LMHead(mesh_device, hf_config, {"weight": weight}, mesh_config=objs["mesh_config"])
    assert tuple(head.weight.shape)[-2:] == (hidden, TEST_VOCAB), (
        f"the on-device weight is {tuple(head.weight.shape)}; the module must store [.., hidden, vocab] "
        f"= [.., {hidden}, {TEST_VOCAB}] after transposing HF's [out, in]"
    )

    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.to_torch(head(tt_x), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()

    assert out.shape == ref.shape == (1, 1, seq_len, TEST_VOCAB), f"got {tuple(out.shape)}"

    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    ratio = err_ratio(pcc, floor_pcc)
    logger.info(comp_allclose(ref, out))
    logger.info(
        f"[lm_head] seq_len={seq_len}: PCC = {pcc} (threshold {PCC_THRESHOLD}), "
        f"bf8_b storage floor {floor_pcc}, err_ratio {ratio:.2f}x (ceiling {MAX_ERR_RATIO}x)"
    )
    assert passing, f"[lm_head] seq_len={seq_len} PCC {pcc} < {PCC_THRESHOLD}"
    assert ratio <= MAX_ERR_RATIO, (
        f"[lm_head] seq_len={seq_len} sits {ratio:.2f}x off its own bf8_b storage floor "
        f"({pcc} vs {floor_pcc}); one matmul should not cost more than {MAX_ERR_RATIO}x"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_lm_head_negative_control_untransposed_weight(mesh_device, reset_seeds):
    """Feeding the weight already in `[hidden, vocab]` must NOT reproduce the reference.

    The module transposes at load, so pre-transposing it is a double transpose. With `vocab` and
    `hidden` deliberately different this is caught by shape; the point of asserting it is that the
    transpose is load-bearing and not an accident of a square weight.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    pre_transposed = torch.randn(hf_config.hidden_size, TEST_VOCAB, dtype=torch.float32) * WEIGHT_SCALE

    try:
        LMHead(mesh_device, hf_config, {"weight": pre_transposed}, mesh_config=objs["mesh_config"])
    except AssertionError as exc:
        logger.info(f"[lm_head] pre-transposed weight refused, as required: {str(exc)[:160]}")
        return
    raise AssertionError(
        "LMHead accepted a [hidden, vocab] weight. HF's layout is [out, in] = [vocab, hidden] and "
        "the module transposes it; accepting the transposed form means the shape assert is missing."
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
def test_lm_head_refusals(mesh_device, expect_error, reset_seeds):
    """The constructor's four refusals, each matched on its message."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = _small_vocab_config(objs["hf_config"])
    weight = torch.randn(TEST_VOCAB, hf_config.hidden_size, dtype=torch.float32) * WEIGHT_SCALE

    # 1. a tied-embedding config: the weights would have to come from the embedding table instead.
    with expect_error(AssertionError, "tie_word_embeddings"):
        LMHead(
            mesh_device,
            replace(hf_config, tie_word_embeddings=True),
            {"weight": weight},
            mesh_config=objs["mesh_config"],
        )

    # 2. the whole state dict instead of the stripped sub-dict (the Meta-rename signature, DEC-039).
    with expect_error(AssertionError, "stripped sub-dict"):
        LMHead(mesh_device, hf_config, {"lm_head.weight": weight}, mesh_config=objs["mesh_config"])

    # 3. a bias, which Llama-3.1 does not have.
    with expect_error(AssertionError, "carries a bias"):
        LMHead(
            mesh_device,
            hf_config,
            {"weight": weight, "bias": torch.zeros(TEST_VOCAB)},
            mesh_config=objs["mesh_config"],
        )

    # 4. cache-only mode with no cache to read from (DEC-038).
    with expect_error(AssertionError, "no tensor_cache_path"):
        LMHead(mesh_device, hf_config, {}, mesh_config=objs["mesh_config"])
