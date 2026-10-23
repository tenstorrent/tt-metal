# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The MTP union's ids survive the bf16 byte block the pipeline carries them in, bit for bit."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.id_transport import ID_ROWS, pack_ids, unpack_ids

VOCAB = 154880
WINDOW, N_MTP, WIDTH = 640, 32, 1536


def _ids(t: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
def test_mtp_id_transport_round_trip(mesh_device):
    torch.manual_seed(0)
    n_dev = mesh_device.get_num_devices()
    chunk = torch.randint(0, VOCAB, (n_dev, 1, WINDOW), dtype=torch.int64)
    chunk[0, 0, :8] = torch.tensor([0, 1, 255, 256, 65535, 65536, VOCAB - 1, 70000])
    mtp = torch.randint(0, VOCAB, (n_dev, 1, N_MTP), dtype=torch.int64)
    mtp[:, 0, 7:] = MTP_PAD_TOKEN_ID  # the lookahead tail is pad
    tt_chunk, tt_mtp = _ids(chunk, mesh_device), _ids(mtp, mesh_device)

    block = pack_ids(tt_chunk, tt_mtp, vocab_size=VOCAB, width=WIDTH)
    assert list(block.shape) == [1, 1, ID_ROWS, WIDTH] and block.dtype == ttnn.bfloat16
    got_chunk, got_mtp = unpack_ids(block, num_chunk_ids=WINDOW, num_mtp_ids=N_MTP)

    want_chunk = chunk.clamp(max=VOCAB - 1)
    want_mtp = mtp.clamp(max=VOCAB - 1)
    for dev, (gc, gm) in enumerate(zip(ttnn.get_device_tensors(got_chunk), ttnn.get_device_tensors(got_mtp))):
        gc = ttnn.to_torch(gc).to(torch.int64).reshape(-1)
        gm = ttnn.to_torch(gm).to(torch.int64).reshape(-1)
        assert torch.equal(gc, want_chunk[dev].reshape(-1)), f"device {dev}: chunk ids differ"
        assert torch.equal(gm, want_mtp[dev].reshape(-1)), f"device {dev}: lookahead ids differ"
