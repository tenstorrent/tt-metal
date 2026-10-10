# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The batched decode's device sampling tail (``TtV41BatchDecoder.sample``) on the mesh, against torch: argmax rows,
top-k rows (the sample lies in the row's top-k) and the gathered slot order."""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import ROWS, DeviceSampler
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.pool import V41DecodePool


@pytest.mark.timeout(int(os.environ.get("V41_SAMPLING_TIMEOUT", 1800)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=0), indirect=True)
def test_v41_device_sampling(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    vocab = C.VOCAB_SIZE
    per_chip = vocab // tp
    torch.manual_seed(0)
    # every rank its own logits [ROWS, vocab]; chip (r, j) holds columns [j * per_chip, (j + 1) * per_chip)
    logits = torch.randn(sp, ROWS, vocab) * 4
    # a clear maximum per row (the candidates are compared in bf16: two of 129,280 Gaussians can tie there)
    peak = torch.randint(0, vocab, (sp, ROWS))
    logits.view(sp * ROWS, vocab)[torch.arange(sp * ROWS), peak.flatten()] = 40.0
    host = logits.view(sp, ROWS, tp, per_chip).permute(0, 2, 1, 3).reshape(sp, tp, 1, 1, ROWS, per_chip)
    t = ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )
    t = ttnn.reshape(t, (1, 1, ROWS, per_chip))
    ccl = V41Collectives(mesh_device)
    pool_like = V41DecodePool.__new__(V41DecodePool)  # only gather_sp is needed
    V41DecodePool._init_ccl(pool_like, mesh_device)
    sampler = DeviceSampler(mesh_device, C, ccl, pool_like.gather_sp)
    # argmax on every row
    k = torch.ones(sp * ROWS, dtype=torch.int32)
    p = torch.zeros(sp * ROWS)
    temp = torch.ones(sp * ROWS)
    inputs = sampler.host_inputs(k, p, temp)
    buffers = {name: ttnn.to_device(v, mesh_device) for name, v in inputs.items()}
    ids = sampler(ttnn.clone(t), buffers)  # the sampler consumes its logits
    got = DeviceSampler.to_host(ids)
    want = logits.view(sp * ROWS, vocab).argmax(dim=-1)
    bad = (got != want).nonzero().flatten().tolist()
    if bad:
        # the candidates of the first bad rows: is the peak among them, what does the op see
        values, indices = sampler.candidates(ttnn.clone(t))
        for slot in bad[:4]:
            r, u = divmod(slot, ROWS)
            v = ttnn.to_torch(ttnn.get_device_tensors(values)[r * tp])[0, 0, u].float()
            i = ttnn.to_torch(ttnn.get_device_tensors(indices)[r * tp])[0, 0, u].to(torch.int64)
            where = (i == int(want[slot])).nonzero().flatten().tolist()
            print(
                f"slot {slot}: peak {int(want[slot])} among the candidates at {where}; candidate max "
                f"{v.max().item():.2f} at {int(i[v.argmax()])}; got {int(got[slot])} (candidate "
                f"{(i == int(got[slot])).nonzero().flatten().tolist()}); first 8 {i[:8].tolist()} {v[:8].tolist()}"
            )
    assert not bad, f"argmax rows {bad[:8]}: got {got[bad[:8]].tolist()} want {want[bad[:8]].tolist()}"
    # top-8 sampling at temperature 1 on every row of flat logits (the top-8 within a few tenths of each other),
    # 16 draws: every draw within the row's top-8 (ties in bf16 allowed), and not always the same one
    k[:] = 8
    p[:] = 1.0  # no nucleus truncation (p 0 keeps the top candidate only: the argmax convention)
    inputs = sampler.host_inputs(k, p, temp)
    for name, v in inputs.items():
        ttnn.copy_host_to_device_tensor(v, buffers[name])
    flat = torch.randn(sp, ROWS, vocab)
    host = flat.view(sp, ROWS, tp, per_chip).permute(0, 2, 1, 3).reshape(sp, tp, 1, 1, ROWS, per_chip)
    t2 = ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )
    t2 = ttnn.reshape(t2, (1, 1, ROWS, per_chip))
    # the row's top-12 (the candidates are compared in bf16: the 8th and 9th of 129,280 normals are within it)
    top8 = flat.view(sp * ROWS, vocab).topk(12, dim=-1).indices
    draws = []
    for _ in range(16):
        ids = sampler(ttnn.clone(t2), buffers)
        draws.append(DeviceSampler.to_host(ids))
    draws = torch.stack(draws)  # [16, slots]
    inside = (draws.unsqueeze(-1) == top8.unsqueeze(0)).any(-1)
    assert inside.all(), f"draws outside the top-12: {(~inside).sum().item()} of {inside.numel()}"
    assert (draws != draws[0]).any(), "16 draws at temperature 1 over 8 candidates never differed"
