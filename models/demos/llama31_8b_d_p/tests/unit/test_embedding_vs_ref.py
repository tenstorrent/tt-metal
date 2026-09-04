# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/embedding.py` vs a torch reference. Owned by P9 item 9 (every `tt/` module owns a test).

The embedding is a **pure gather**, so this file gates it the way recipe §2.5 says a mapping claim
must be gated — on **bit-equality**, not PCC. Correlation barely notices a permuted or shifted
token->row map (§2.5 measured a rotated head->column map still scoring PCC 0.99890), and a
one-row-off embedding table is exactly that class of bug.

* **Input distribution:** uniform random token ids over the whole vocab (`randint(0, vocab)`), plus
  the three boundary ids `{0, 1, vocab-1}` pinned into the sequence so an off-by-one at either end
  cannot hide. The table is `randn * 0.02`.
* **Reference dtype policy:** the reference gathers from the **bf16-quantised** table, because a
  gather performs no arithmetic — the device's only error source is storing the table in bf16
  (`DEC-022`), so the reference and the device must agree **exactly**, not approximately. The fp32
  table is also scored, to record what that storage costs.
* **Computed noise floor:** PCC of the fp32-table gather against the bf16-table gather. The device
  is required to sit *at* it (ratio 1.00x) rather than within a budget of it.
* **Negative control:** the same ids shifted by one row. Bit-equality must fail **and** the PCC
  against the correct reference must collapse.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_embedding_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.config import MeshConfig
from models.demos.llama31_8b_d_p.tt.embedding import Embedding

SEQ_LENS = [128, 512]
# A small stand-in vocab keeps the gather test cheap; the real 128256x4096 table is exercised by
# `G-WEIGHTS` (bit-exact against the checkpoint) and by `G-MODEL` (the full stack). What this file
# tests is the token -> row map, which is vocab-size-independent as long as the boundary ids are hit.
TEST_VOCAB = 2048


def _hf(vocab=TEST_VOCAB):
    hf = dict(llama_config_dims())
    hf["vocab_size"] = vocab
    return hf


def _token_ids(seq_len, vocab):
    """Uniform ids with the three boundary ids pinned in: 0, 1 and `vocab - 1`."""
    ids = torch.randint(0, vocab, (1, seq_len))
    ids[0, 0], ids[0, 1], ids[0, -1] = 0, 1, vocab - 1
    return ids


def _to_device_ids(ids, mesh_device):
    return ttnn.from_torch(
        ids.reshape(1, 1, 1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_embedding_vs_ref(mesh_device, seq_len, reset_seeds):
    """The gather must be **bit-exact** against the bf16-table reference."""
    hf = _hf()
    table = torch.randn(hf["vocab_size"], hf["hidden_size"]) * 0.02
    ids = _token_ids(seq_len, hf["vocab_size"])

    ref_fp32 = table[ids[0]].reshape(1, 1, seq_len, -1)
    table_bf16 = quantize_like_device(table[None, None], ttnn.bfloat16)[0, 0]
    ref_bf16 = table_bf16[ids[0]].reshape(1, 1, seq_len, -1)
    _, floor = comp_pcc(ref_fp32, ref_bf16, 0.0)

    emb = Embedding(
        mesh_device, hf, {"weight": table}, mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])
    )
    tt_ids = _to_device_ids(ids, mesh_device)
    out = _from_device(emb(tt_ids))

    _, pcc = comp_pcc(ref_fp32, out, 0.0)
    max_delta = (out - ref_bf16).abs().max().item()
    logger.info(
        f"[EMBED] s{seq_len}: PCC vs fp32 table={float(pcc):.7f} floor={float(floor):.7f} "
        f"ratio={err_ratio(float(pcc), float(floor)):.2f}x; max|delta| vs the bf16-table "
        f"reference={max_delta:.3e} (must be exactly 0)"
    )
    assert torch.equal(out, ref_bf16), (
        f"the gather is not bit-exact against the bf16-quantised table (max|delta| {max_delta:.3e}); "
        f"a gather does no arithmetic, so any delta is a wrong token->row map or a wrong dtype"
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_embedding_row_map_negative_control(mesh_device, reset_seeds):
    """**The negative control:** ids shifted by one row. Bit-equality must fail and PCC must collapse."""
    hf = _hf()
    seq_len = 128
    table = torch.randn(hf["vocab_size"], hf["hidden_size"]) * 0.02
    ids = _token_ids(seq_len, hf["vocab_size"])
    shifted = (ids + 1) % hf["vocab_size"]

    table_bf16 = quantize_like_device(table[None, None], ttnn.bfloat16)[0, 0]
    ref_bf16 = table_bf16[ids[0]].reshape(1, 1, seq_len, -1)

    emb = Embedding(mesh_device, hf, {"weight": table})
    out = _from_device(emb(_to_device_ids(shifted, mesh_device)))
    _, pcc = comp_pcc(ref_bf16, out, 0.0)

    logger.info(
        f"[EMBED] control: ids shifted by one row -> PCC={float(pcc):.5f}, bit-equal={torch.equal(out, ref_bf16)}"
    )
    assert not torch.equal(
        out, ref_bf16
    ), "a one-row token id shift produced a bit-identical gather — the probe is blind"
    assert (
        float(pcc) < 0.9
    ), f"a one-row token id shift still scores {float(pcc)} — the gate is not sensitive to the row map"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_embedding_refuses_to_build_weightless(mesh_device, expect_error):
    """No `state_dict` and no cache path must fail loud, not embed against a `None` table."""
    with expect_error(ValueError, "tensor_cache_path"):
        Embedding(mesh_device, _hf(), {})


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_embedding_refuses_a_wrong_shaped_table(mesh_device, expect_error, reset_seeds):
    """A table whose shape does not match `(vocab_size, hidden_size)` must be refused at load.

    The failure this prevents is the transposed-table one: `[hidden, vocab]` is a legal tensor and
    `ttnn.embedding` would gather 4096 rows of width 128256 without complaint.
    """
    hf = _hf()
    transposed = torch.randn(hf["hidden_size"], hf["vocab_size"]) * 0.02
    with expect_error(AssertionError, "expected"):
        Embedding(mesh_device, hf, {"weight": transposed})
