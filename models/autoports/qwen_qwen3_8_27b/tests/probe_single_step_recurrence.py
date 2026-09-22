# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded recurrence component comparison; does not replace the autoport model.

Inputs are pre-laid-out for each implementation. Thus an apparent one-step gain
would still need to pay GQA/layout conversion costs in a real model caller.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(19)
    b, h, hv, d = args.batch, 4, 12, 128
    q = torch.randn(b, 1, h, d).bfloat16().float()
    k = torch.randn(b, 1, h, d).bfloat16().float()
    v = torch.randn(b, 1, hv, d).bfloat16().float()
    beta = torch.rand(b, 1, hv).bfloat16().float()
    g = -torch.rand(b, 1, hv) * 0.1
    initial = torch.randn(b, hv, d, d) * 0.05
    qq, kk = [x.repeat_interleave(hv // h, dim=2) for x in (q, k)]
    qn, kn = [x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6) for x in (qq, kk)]
    decayed = initial * torch.exp(g[:, 0, :, None, None])
    delta = beta[:, 0, :, None] * (v[:, 0] - torch.einsum("bhk,bhkv->bhv", kn[:, 0], decayed))
    ref_state = decayed + kn[:, 0, :, :, None] * delta[:, :, None, :]
    ref_output = torch.einsum("bhk,bhkv->bhv", qn[:, 0] * d**-0.5, ref_state)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    trace = None
    report = dict(batch=b, rows=[])
    try:

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )

        def read(x):
            return ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()

        def flat(x):
            return torch.nn.functional.pad(x.reshape(b, 1, -1), (0, 0, 0, 31))

        native_inputs = [upload(flat(x)) for x in (q, k, v)]
        native_g, native_beta = [upload(torch.nn.functional.pad(x, (0, 0, 0, 31)), ttnn.float32) for x in (g, beta)]
        one_inputs = [upload(x) for x in (qq, kk, v, beta)] + [upload(g, ttnn.float32)]
        initial_tt = upload(initial, ttnn.float32)
        eye = upload(torch.eye(32).reshape(1, 1, 32, 32), ttnn.float32)
        tril = upload(torch.ones(32, 32).tril().reshape(1, 1, 32, 32), ttnn.float32)
        ones = upload(torch.ones(1, 1, 32, 32), ttnn.float32)
        masks_host = torch.zeros(1, 1, 32, 96)
        masks_host[:, :, :16, :16] = 1
        masks_host[:, :, 16:, 48:64] = 1
        masks_host[:, :, 16:, 64:80] = 1
        masks = upload(masks_host, ttnn.float32)

        def native():
            grid = mesh.compute_with_storage_grid_size()
            scan_batch = grid.x * grid.y // hv
            outputs, states = [], []
            for start in range(0, b, scan_batch):
                end = min(start + scan_batch, b)
                out, state = ttnn.transformer.chunk_gated_delta_rule(
                    *[x[start:end] for x in native_inputs],
                    native_g[start:end],
                    native_beta[start:end],
                    initial_state=initial_tt[start:end],
                    output_final_state=True,
                    output_head_major=True,
                    chunk_size=32,
                    eye=eye,
                    tril=tril,
                    ones=ones,
                    masks=masks,
                )
                outputs.append(out)
                states.append(state)
            return tuple(parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=0) for parts in (outputs, states))

        def single():
            return recurrent_gated_delta_rule_decode_ttnn(
                *one_inputs, initial_state=initial_tt, device=mesh, high_precision=True
            )

        def single_with_layout():
            shaped = []
            for flat_input, heads in zip(native_inputs, (h, h, hv)):
                live = ttnn.to_layout(flat_input[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
                live = ttnn.reshape(live, [b, 1, heads, d])
                live = ttnn.to_layout(live, ttnn.TILE_LAYOUT)
                shaped.append(ttnn.repeat_interleave(live, hv // heads, dim=2) if heads != hv else live)
            out, state = recurrent_gated_delta_rule_decode_ttnn(
                *shaped,
                native_beta[:, :1, :],
                native_g[:, :1, :],
                initial_state=initial_tt,
                device=mesh,
                high_precision=True,
            )
            out = ttnn.permute(out, (0, 2, 1, 3))
            out = ttnn.pad(out, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
            return out, state

        for name, forward in (
            ("chunk32", native),
            ("single_step", single),
            ("single_step_with_layout", single_with_layout),
        ):
            out, state = forward()
            stable_out, stable_state = ttnn.clone(out), ttnn.clone(state)
            ttnn.copy(out, stable_out)
            ttnn.copy(state, stable_state)
            del out, state
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out, state = forward()
            ttnn.copy(out, stable_out)
            ttnn.copy(state, stable_state)
            del out, state
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            times = []
            for _ in range(5):
                begin = time.perf_counter()
                for _ in range(32):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - begin) / 32)
            actual = read(stable_out)
            actual = actual[:, 0] if name == "single_step" else actual.reshape(b, hv, 32, d)[:, :, 0]
            actual_state = read(stable_state)
            pcc = lambda a, z: torch.corrcoef(torch.stack([a.flatten(), z.flatten()]))[0, 1].item()
            report["rows"].append(
                dict(
                    mode=name,
                    seconds=times,
                    output_pcc=pcc(actual, ref_output),
                    state_pcc=pcc(actual_state, ref_state),
                    output_max_error=(actual - ref_output).abs().max().item(),
                    state_max_error=(actual_state - ref_state).abs().max().item(),
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("SINGLE_STEP_RECURRENCE", json.dumps(report["rows"][-1]), flush=True)
            ttnn.release_trace(mesh, trace)
            trace = None
            del stable_out, stable_state
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
