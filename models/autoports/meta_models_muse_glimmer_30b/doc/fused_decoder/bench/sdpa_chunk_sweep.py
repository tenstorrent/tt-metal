# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep prefill-SDPA q/k chunk sizes for both Muse-Glimmer layer kinds.

The functional decoder pinned ``q_chunk == k_chunk == 128`` because
``q_chunk == 2*k_chunk`` is numerically broken with ``sliding_window_size``
(functional-stage limitation 1).  That left the chunk *size* untuned.  This
sweep measures every ``q == k`` chunk against a PyTorch masked-softmax
reference (correctness) and against wall time (speed), for the sliding
(windowed) and full (plain causal) kinds at several prefill-chunk lengths.
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
WINDOW = 2048
SCALE = 0.342063
CHUNKS = (128, 256, 320, 384, 512)
ITERS = 5
ROUNDS = 3


def torch_reference(q, k, v, seq_len, window):
    keys = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
    values = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
    index = torch.arange(seq_len)
    mask = index[:, None] >= index[None, :]
    if window:
        mask = mask & (index[None, :] > index[:, None] - window)
    scores = (q.float() @ keys.transpose(-1, -2)) * SCALE
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ values


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # 2080 / 4128 / 8224 are the lengths the functional stage's
        # sdpa_sliding_window_chunk_repro.py showed are mis-masked when
        # q_chunk == 2 * k_chunk, so the chosen chunk size has to be re-pinned
        # at exactly those lengths, not only at the round ones.
        # Round lengths, the three lengths the functional stage's
        # sdpa_sliding_window_chunk_repro.py showed are mis-masked when
        # q_chunk == 2 * k_chunk (2080 / 4128 / 8224), and a spread of
        # in-between prompt lengths, because the chunk size is one global
        # constant that every prefill chunk uses.
        for seq_len in (1024, 2048, 2080, 3008, 4096, 4128, 6144, 8192, 8224):
            torch.manual_seed(1)
            q = torch.randn(1, NUM_Q_HEADS, seq_len, HEAD_DIM).to(torch.bfloat16)
            k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM).to(torch.bfloat16)
            v = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM).to(torch.bfloat16)
            tq, tk, tv = (
                ttnn.from_torch(
                    t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                for t in (q, k, v)
            )
            for kind, window in (("sliding", WINDOW), ("full", None)):
                ref = torch_reference(q, k, v, seq_len, window)
                for chunk in CHUNKS:
                    if chunk > seq_len:
                        continue
                    pc = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                        q_chunk_size=chunk,
                        k_chunk_size=chunk,
                        exp_approx_mode=False,
                    )
                    kwargs = dict(is_causal=True, scale=SCALE, program_config=pc, compute_kernel_config=ck)
                    if window:
                        kwargs["sliding_window_size"] = window
                    try:
                        out = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, **kwargs)
                        pcc = comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]
                        ttnn.deallocate(out)
                        rounds = []
                        for _ in range(ROUNDS):
                            ttnn.synchronize_device(mesh)
                            t0 = time.perf_counter()
                            for _ in range(ITERS):
                                o = ttnn.transformer.scaled_dot_product_attention(tq, tk, tv, **kwargs)
                                ttnn.deallocate(o)
                            ttnn.synchronize_device(mesh)
                            rounds.append((time.perf_counter() - t0) / ITERS * 1e3)
                        print(
                            f"SWEEP seq={seq_len:6d} kind={kind:8s} chunk={chunk:5d}  min {min(rounds):8.3f} ms"
                            f"  (rounds {'/'.join(f'{r:.3f}' for r in rounds)})  PCC={pcc}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"SWEEP seq={seq_len:6d} kind={kind:8s} chunk={chunk:5d}  FAILED "
                            f"{type(exc).__name__}: {str(exc)[:160]}"
                        )
            for t in (tq, tk, tv):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
