# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""``engram_checkpoint`` against the checkpoint (no device): sliced packing equals whole-table packing, the
prepacked table dequantizes like the reference, and the small Engram weights have the module's shapes."""

import numpy as np
import pytest
import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tt.v41 import engram_checkpoint as ec
from models.demos.deepseek_v3_d_p.tt.v41.engram import PACKED_WIDTH, V41EngramTable, pack_rows
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint

ROWS = 4096


@pytest.fixture(scope="module")
def ckpt():
    c = resolve_checkpoint()
    if c is None or ec.engram_names(1)["embed"] not in c:
        pytest.skip("V4.1 checkpoint (shard 47) not downloaded")
    return c


def _slice(ckpt, layer, start, end):
    names = ec.engram_names(layer)
    with safe_open(ckpt.root / ckpt.weight_map[names["embed"]], framework="pt", device="cpu") as h:
        return h.get_slice(names["embed"])[start:end], h.get_slice(names["embed_scale"])[start:end]


@pytest.mark.parametrize("layer", list(C.ENGRAM_LAYER_IDS))
def test_sliced_rows_pack_like_the_whole(ckpt, layer):
    weight, scale = _slice(ckpt, layer, 0, ROWS)
    assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
    assert weight.shape == (ROWS, C.ENGRAM_HEAD_DIM) and scale.shape == (ROWS, C.ENGRAM_HEAD_DIM // 32)
    whole = V41EngramTable(weight, scale)
    halves = torch.cat([pack_rows(weight[:1000], scale[:1000]), pack_rows(weight[1000:], scale[1000:])])
    assert torch.equal(halves, whole.rows)
    # through an int16 file, as pack_table_to_file writes and load_packed_table reads
    arr = halves.view(torch.int16).numpy()
    back = torch.from_numpy(np.array(arr)).view(ROWS, PACKED_WIDTH).view(torch.uint16)
    prepacked = V41EngramTable.from_packed(back, C.ENGRAM_HEAD_DIM)
    ids = torch.tensor([0, 1, 31, 32, 1000, 1001, ROWS - 1])
    assert torch.equal(prepacked.lookup(ids), whole.lookup(ids))
    # the reference dequant: fp32(e4m3) * fp32(2^(e8m0-127)) per 32, to bf16
    ref = (
        (weight[ids].float().unflatten(-1, (-1, 32)) * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    )
    assert torch.equal(prepacked.lookup(ids), ref)
    assert torch.isfinite(ref.float()).all() and ref.float().abs().max() > 0


def test_engram_weights_shapes(ckpt):
    w = ec.load_engram_weights(ckpt, 1)
    assert w["wkv"].shape == (
        (C.HC_MULT + 1) * C.EMB_SIZE,
        (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS * C.ENGRAM_HEAD_DIM,
    )
    assert w["wkv"].dtype == torch.bfloat16
    assert w["q_weight"].shape == w["k_weight"].shape == (C.HC_MULT, C.EMB_SIZE)
    assert (
        ec.packed_table_path(1).name.startswith("layer_1.")
        and str(C.ENGRAM_NUM_EMBEDDINGS[0]) in ec.packed_table_path(1).name
    )
