#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-card probe for the two native Gated DeltaNet ops, against the torch reference.

Run before wiring the GDN module, so a shape/dtype/semantics mismatch surfaces on its own rather
than inside a layer. Not part of the test suite — the real coverage is ``tests/unit/test_gdn_*``
on the target mesh.

    python3 models/demos/qwen_3_8_27b_d_p/scripts/probe_gdn_op.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn  # noqa: E402
from models.common.utility_functions import comp_pcc  # noqa: E402
from models.demos.qwen_3_8_27b_d_p.reference.modeling import torch_chunk_gated_delta_rule  # noqa: E402

T = 512
HV, H, DK, DV = 12, 4, 128, 128
G = HV // H
CONV_DIM = H * DK * 2 + HV * DV  # 512 + 512 + 1536
KERNEL = 4

_results: list[tuple[str, bool, float]] = []


def check(name: str, ref: torch.Tensor, got: torch.Tensor, bar: float = 0.99) -> None:
    ok, pcc = comp_pcc(ref.float(), got.float(), bar)
    _results.append((name, bool(ok), float(pcc)))
    print(f"  {'OK ' if ok else 'BAD'} {name}: {pcc}")


def ref_scan(q, k, v, g, beta, initial_state=None, chunk=32):
    q4 = q.reshape(1, -1, H, DK).repeat_interleave(G, dim=2).to(torch.float16)
    k4 = k.reshape(1, -1, H, DK).repeat_interleave(G, dim=2).to(torch.float16)
    v4 = v.reshape(1, -1, HV, DV).to(torch.float16)
    return torch_chunk_gated_delta_rule(
        q4,
        k4,
        v4,
        g=g,
        beta=beta.to(torch.float16),
        chunk_size=chunk,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def main() -> int:
    torch.manual_seed(0)
    q = torch.randn(1, T, H * DK, dtype=torch.bfloat16)
    k = torch.randn(1, T, H * DK, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV * DV, dtype=torch.bfloat16)
    beta = torch.rand(1, T, HV, dtype=torch.float32)
    g = -torch.rand(1, T, HV, dtype=torch.float32) * 0.5

    ref_o, ref_s = ref_scan(q, k, v, g, beta, chunk=32)
    ref_o64, ref_s64 = ref_scan(q, k, v, g, beta, chunk=64)
    print("reference self-consistency (the device runs chunk 32, the HF default is 64):")
    check("ref_chunk32_vs_64_o", ref_o64, ref_o, 0.999)
    check("ref_chunk32_vs_64_state", ref_s64, ref_s, 0.999)

    # Chunked carry, host side: two halves threaded through the state must equal one shot.
    half = T // 2
    o_a, s_a = ref_scan(q[:, :half], k[:, :half], v[:, :half], g[:, :half], beta[:, :half])
    o_b, s_b = ref_scan(q[:, half:], k[:, half:], v[:, half:], g[:, half:], beta[:, half:], initial_state=s_a)

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        to = lambda t, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
            t, dtype=dt, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        print("\nchunk_gated_delta_rule (flat q/k/v, chunk 32, in-kernel L2 norm + GQA expand):")
        o_tt, s_tt = ttnn.transformer.chunk_gated_delta_rule(
            to(q, ttnn.bfloat16),
            to(k, ttnn.bfloat16),
            to(v, ttnn.bfloat16),
            to(g, ttnn.float32),
            to(beta, ttnn.float32),
            initial_state=None,
            output_final_state=True,
            chunk_size=32,
        )
        check("scan_out", ref_o, ttnn.to_torch(o_tt).reshape(1, T, HV, DV))
        check("scan_state", ref_s, ttnn.to_torch(s_tt).reshape(1, HV, DK, DV))

        print("\ninitial_state carry (the mechanism chunked prefill rests on for the 48 GDN layers):")
        o0, s0 = ttnn.transformer.chunk_gated_delta_rule(
            to(q[:, :half], ttnn.bfloat16),
            to(k[:, :half], ttnn.bfloat16),
            to(v[:, :half], ttnn.bfloat16),
            to(g[:, :half], ttnn.float32),
            to(beta[:, :half], ttnn.float32),
            initial_state=None,
            output_final_state=True,
            chunk_size=32,
        )
        o1, s1 = ttnn.transformer.chunk_gated_delta_rule(
            to(q[:, half:], ttnn.bfloat16),
            to(k[:, half:], ttnn.bfloat16),
            to(v[:, half:], ttnn.bfloat16),
            to(g[:, half:], ttnn.float32),
            to(beta[:, half:], ttnn.float32),
            initial_state=s0,
            output_final_state=True,
            chunk_size=32,
        )
        chunked = torch.cat(
            [ttnn.to_torch(o0).reshape(1, half, HV, DV), ttnn.to_torch(o1).reshape(1, half, HV, DV)], dim=1
        )
        check("carry_out_vs_oneshot", ref_o, chunked)
        check("carry_state_vs_oneshot", ref_s, ttnn.to_torch(s1).reshape(1, HV, DK, DV))
        check("carry_out_vs_ref_chunked", torch.cat([o_a, o_b], dim=1), chunked)

        print("\nqkv_causal_conv1d_silu (4-tap causal depthwise conv + SiLU + q/k/v split):")
        x = torch.randn(1, T, CONV_DIM, dtype=torch.bfloat16)
        w = torch.randn(CONV_DIM, 1, KERNEL, dtype=torch.bfloat16) * 0.5
        conv = torch.nn.Conv1d(CONV_DIM, CONV_DIM, KERNEL, groups=CONV_DIM, bias=False, padding=KERNEL - 1)
        with torch.no_grad():
            conv.weight.copy_(w)
        conv = conv.to(torch.bfloat16)
        with torch.no_grad():
            xt = x.transpose(1, 2)
            ref_conv = F.silu(conv(xt)[:, :, :T]).transpose(1, 2)

        taps = [to(w[:, 0, j].reshape(1, 1, 1, CONV_DIM), ttnn.bfloat16) for j in range(KERNEL)]
        qc, kc, vc = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            to(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            to(torch.zeros(1, KERNEL - 1, CONV_DIM, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            H * DK,
            H * DK,
            HV * DV,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=512),
        )
        got_conv = torch.cat([ttnn.to_torch(t).reshape(1, T, -1) for t in (qc, kc, vc)], dim=-1)
        check("causal_conv_silu", ref_conv, got_conv)

        print("\nqkv_causal_conv1d_silu with a carried 3-token history:")
        hist = torch.randn(1, KERNEL - 1, CONV_DIM, dtype=torch.bfloat16)
        with torch.no_grad():
            joined = torch.cat([hist, x], dim=1).transpose(1, 2)
            ref_hist = F.silu(conv(joined)[:, :, : joined.shape[-1]])[:, :, -T:].transpose(1, 2)
        qc2, kc2, vc2 = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            to(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            to(hist, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            H * DK,
            H * DK,
            HV * DV,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=512),
        )
        got_hist = torch.cat([ttnn.to_torch(t).reshape(1, T, -1) for t in (qc2, kc2, vc2)], dim=-1)
        check("causal_conv_silu_with_history", ref_hist, got_hist)
    finally:
        ttnn.close_mesh_device(device)

    bad = [r for r in _results if not r[1]]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
