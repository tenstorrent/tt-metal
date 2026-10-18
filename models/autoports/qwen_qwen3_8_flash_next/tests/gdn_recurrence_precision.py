# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminate native recurrence precision from the unchanged GDN frontend.

Run only under the parent's serialized device lease. CPU preprocessing uses
real layer-0 weights and the exact first384 long-test inputs. Both TT paths
consume identical uploaded FP32 q/k/v/g/beta; host conversions occur outside
passes. This is a diagnostic, not an alternate accepted decoder implementation.
"""
import json
import runpy
from pathlib import Path

import torch

import ttnn

ROOT = Path(__file__).resolve().parents[1]


def composite(q, k, v, g, beta, state, compute):
    """Device-only FP32 recurrence using SFPU arithmetic/accurate reduction."""
    b, length, h, d = q.shape
    q = ttnn.permute(q, (0, 2, 1, 3))
    k = ttnn.permute(k, (0, 2, 1, 3))
    v = ttnn.permute(v, (0, 2, 1, 3))
    decay = ttnn.exp(ttnn.permute(g, (0, 2, 1)))
    beta = ttnn.permute(beta, (0, 2, 1))
    outputs = []
    for token in range(length):

        def row(x):
            return ttnn.slice(x, [0, 0, token, 0], [b, h, token + 1, d])

        qt, kt, vt = row(q), row(k), row(v)

        def scalar(x):
            value = ttnn.slice(x, [0, 0, token], [b, h, token + 1])
            return ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), [b, h, 1, 1]), ttnn.TILE_LAYOUT
            )

        state = ttnn.multiply(state, scalar(decay))
        kcol = ttnn.transpose(kt, -1, -2)
        read = ttnn.sum(
            ttnn.multiply(state, kcol),
            dim=-2,
            keepdim=True,
            compute_kernel_config=compute,
            fast_and_approximate_mode=False,
        )
        delta = ttnn.multiply(ttnn.subtract(vt, read), scalar(beta))
        state = ttnn.add(state, ttnn.multiply(kcol, delta))
        out = ttnn.sum(
            ttnn.multiply(state, ttnn.transpose(qt, -1, -2)),
            dim=-2,
            keepdim=True,
            compute_kernel_config=compute,
            fast_and_approximate_mode=False,
        )
        outputs.append(out)
    return ttnn.permute(ttnn.concat(outputs, dim=2), (0, 2, 1, 3)), state


def main():
    data = runpy.run_path(str(ROOT / "doc/functional_decoder/gdn_debug/probe_cpu.py"))
    q, k, v, g, beta = (data[name].detach().contiguous() for name in ("q", "k", "v", "g", "beta"))
    beta = beta.float()
    state = torch.zeros(1, 48, 128, 128)
    expected = []
    with torch.no_grad():
        for token in range(q.shape[1]):
            state = state * g[:, token].exp()[..., None, None]
            kt = k[:, token]
            residual = (v[:, token] - (state * kt[..., None]).sum(-2)) * beta[:, token, ..., None]
            state = state + kt[..., None] * residual[..., None, :]
            expected.append((state * q[:, token, ..., None]).sum(-2))
    expected = torch.stack(expected, dim=1)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        qd, kd, vd, gd, bd = [upload(t) for t in (q, k, v, g, beta)]
        initial = upload(torch.zeros_like(state))
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        for label, operation in (
            (
                "native",
                lambda: ttnn.transformer.fused_recurrent_gated_delta_rule(
                    ttnn.multiply(qd, 128**0.5),
                    kd,
                    vd,
                    gd,
                    bd,
                    initial_state=initial,
                    output_final_state=True,
                    compute_kernel_config=compute,
                ),
            ),
            ("composite_fp32", lambda: composite(qd, kd, vd, gd, bd, initial, compute)),
        ):
            output, final = operation()
            actual = ttnn.to_torch(output)
            for position in (0, 128, 256):
                score = data["pcc"](expected[:, position : position + 128], actual[:, position : position + 128])
                print(json.dumps({"precision_probe": label, "position": position, "output_pcc": score}), flush=True)
            print(
                json.dumps({"precision_probe": label, "state_pcc": data["pcc"](state, ttnn.to_torch(final))}),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
