# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pin the contract of BgeM3Model.forward(no_padding=...).

no_padding=True skips the dense [B, 1, S, S] attention mask. That is correct
only when no row holds a pad token, and the caller owns that statement. These
tests fix three facts so that a later change cannot move the default or widen
the skip without a failure:

  1. Unpadded input: no_padding=True returns the default result.
  2. Padded input: the default masks, so it returns the explicit keep-mask result.
  3. Padded input: no_padding=True returns a different result for the padded row,
     so the flag does remove the mask. The earlier measurement gave cos 0.4147.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import require_single_device
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
SEQ_LEN = 512
PADDED_ROW = 1
PADDED_VALID = 64

# Unpadded rows match to float rounding; the measured value is 1.00016.
SAME_COS = 0.9999
# A padded row under no_padding=True measured cos 0.4147 against the masked row.
DIFFERENT_COS = 0.9


@pytest.fixture(scope="module")
def model_path(model_location_generator):
    return str(model_location_generator(MODEL_NAME, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))


def _row_cos(a, b, row):
    return F.cosine_similarity(a[row].flatten().float(), b[row].flatten().float(), dim=0).item()


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _run(model, device, input_ids, **kwargs):
    out = model.forward(_to_device(input_ids, device), **kwargs)
    host = ttnn.to_torch(out)
    ttnn.deallocate(out)
    return host


@pytest.mark.parametrize("batch_size", [8, 16, 32], ids=["batch8", "batch16", "batch32"])
def test_no_padding_contract(device, model_path, batch_size, reset_seeds):
    require_single_device(device)
    model_args, model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=batch_size,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        hf_model_name=model_path,
    )
    pad = model_args.pad_token_id

    # Token ids 5 and up avoid the special ids, so only the padded tail is a pad token.
    full = torch.randint(5, 1000, (batch_size, SEQ_LEN), dtype=torch.int32)
    padded = full.clone()
    padded[PADDED_ROW, PADDED_VALID:] = pad
    keep = torch.ones((batch_size, SEQ_LEN), dtype=torch.int32)
    keep[PADDED_ROW, PADDED_VALID:] = 0

    # 1. Unpadded input: the skip changes nothing.
    default_full = _run(model, device, full)
    skip_full = _run(model, device, full, no_padding=True)
    for row in range(batch_size):
        cos = _row_cos(default_full, skip_full, row)
        assert cos >= SAME_COS, f"unpadded row {row}: no_padding=True gives cos {cos:.6f}"

    # 2. Padded input: the default masks, the same as an explicit keep-mask.
    default_padded = _run(model, device, padded)
    keep_padded = _run(model, device, padded, attention_mask=_to_device(keep, device))
    cos = _row_cos(default_padded, keep_padded, PADDED_ROW)
    assert cos >= SAME_COS, f"padded row: default gives cos {cos:.6f} against the keep-mask"

    # 3. Padded input: no_padding=True removes the mask, so the padded row moves.
    skip_padded = _run(model, device, padded, no_padding=True)
    cos = _row_cos(default_padded, skip_padded, PADDED_ROW)
    assert cos < DIFFERENT_COS, f"padded row: no_padding=True gives cos {cos:.6f}, so the mask was not removed"
