# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct flat recurrence checks, including B1 and zero Q/K, against PyTorch."""
import argparse
import json
from pathlib import Path

import torch
import ttnn

from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_flat_recurrent import flat_recurrent
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", default="1,2,13,14,16,17,26,27,31,32")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(7901)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    records = []

    def device(value, dtype):
        return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    try:
        for batch in map(int, args.batches.split(",")):
            state = torch.randn(batch * 8, 128, 128) * 0.01
            state_tt = device(state, ttnn.float32)
            for step in range(args.steps):
                conv = (torch.randn(1, 1, batch, 2048) * 0.2).bfloat16()
                # Exercise finite rsqrt at zero and a range of state-decay factors.
                if step % 4 == 0:
                    conv[..., :1024] = 0
                g = -torch.rand(batch, 1, 8) * (0.01 if step % 2 else 3.0)
                beta = torch.rand(batch, 1, 8).bfloat16()
                flat = conv.float().reshape(batch, 2048)
                q = flat[:, :512].reshape(batch, 4, 128).repeat_interleave(2, 1).reshape(-1, 128)
                k = flat[:, 512:1024].reshape(batch, 4, 128).repeat_interleave(2, 1).reshape(-1, 128)
                v = flat[:, 1024:].reshape(-1, 128)
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                decayed = state * g.reshape(-1, 1, 1).exp()
                delta = (v - torch.einsum("hk,hkv->hv", k, decayed)) * beta.float().reshape(-1, 1)
                state = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
                expected = torch.einsum("hk,hkv->hv", q, state).unsqueeze(1)
                out = flat_recurrent(device(conv, ttnn.bfloat16), device(g, ttnn.float32),
                                     device(beta, ttnn.bfloat16), state_tt)
                output_pcc, state_pcc, max_errors = [], [], []
                for output_rank, state_rank in zip(ttnn.get_device_tensors(out), ttnn.get_device_tensors(state_tt)):
                    actual = ttnn.to_torch(output_rank).float()
                    actual_state = ttnn.to_torch(state_rank).float()
                    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
                    opcc = 1.0 if not expected.count_nonzero() and not actual.count_nonzero() else pcc(expected, actual)
                    spcc = pcc(state, actual_state)
                    assert opcc >= .995 and spcc >= .995, (batch, step, opcc, spcc)
                    output_pcc.append(opcc)
                    state_pcc.append(spcc)
                    max_errors.append(float((actual - expected).abs().max()))
                records.append(dict(batch=batch, step=step, output_pcc=output_pcc,
                                    state_pcc=state_pcc, max_output_error=max_errors))
            print("FLAT_RECURRENT_BATCH_PASS", batch, flush=True)
        Path(args.output).write_text(json.dumps(dict(records=records, devices=[0,1,2,3],
                                                    complete=True, options=vars(args)), indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
