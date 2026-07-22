# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: can ttnn.swiglu replace the slice+typecast+mul in the fused gate/up path?

GLM4 fused path today: sparse_matmul([w1|w3]) -> slice gate -> slice up ->
typecast(gate) -> typecast(up) -> mul(SiLU(gate), up).  The 2 slices + 2 materializing
typecasts are the overhead cancelling the single-matmul fusion win.

ttnn.swiglu(x) = SiLU(x[second_half]) * x[first_half]. GLM4 needs SiLU(w1)*w3, so the
fused tensor must be ordered [w3 | w1] (up first, gate second). This probe validates that
swiglu on a [.,.,.,3072] tensor ordered [w3|w1] equals the reference SiLU(w1)*w3, on the
real 6D sparse output shape, single device (no weights).

Run: ./python_env/bin/python models/experimental/glm4_moe/experiments/decode_matmul_sweep/probe_swiglu_fuse.py
"""
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

MOE_INTER = 1536
E_LOCAL = 3
BLOCK = 32


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        # Mimic the 6D sparse_matmul fused output shape: [1,1,1,E,BLOCK,2*MOE_INTER].
        w1 = torch.rand((1, 1, 1, E_LOCAL, BLOCK, MOE_INTER)).bfloat16()  # gate
        w3 = torch.rand((1, 1, 1, E_LOCAL, BLOCK, MOE_INTER)).bfloat16()  # up

        # Reference: SiLU(w1) * w3
        ref = torch.nn.functional.silu(w1.float()) * w3.float()

        # swiglu convention: SiLU(second half) * first half -> order must be [w3 | w1]
        fused_w3w1 = torch.cat([w3, w1], dim=-1)  # [.., 3072]
        x = ttnn.from_torch(fused_w3w1, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        try:
            out = ttnn.swiglu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
            t = ttnn.to_torch(out).float()
            eq, msg = comp_pcc(t, ref, 0.99)
            print(
                f"[swiglu 6D, order w3|w1] out.shape={tuple(out.shape)}  PCC vs SiLU(w1)*w3: {msg}  -> {'PASS' if eq else 'FAIL'}"
            )
        except Exception as e:
            print(f"[swiglu 6D] FAILED: {str(e).splitlines()[0][:200]}")
            # Fallback: try 4D (collapse E into batch) in case 6D is unsupported.
            fused4d = fused_w3w1.reshape(E_LOCAL, 1, BLOCK, 2 * MOE_INTER)
            x4 = ttnn.from_torch(fused4d, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            out4 = ttnn.swiglu(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
            t4 = ttnn.to_torch(out4).float().reshape(1, 1, 1, E_LOCAL, BLOCK, MOE_INTER)
            eq, msg = comp_pcc(t4, ref, 0.99)
            print(f"[swiglu 4D reshape] out.shape={tuple(out4.shape)}  PCC: {msg}  -> {'PASS' if eq else 'FAIL'}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
