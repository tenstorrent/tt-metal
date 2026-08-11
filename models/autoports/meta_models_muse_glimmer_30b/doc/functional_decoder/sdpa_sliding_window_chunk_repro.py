# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone reproducer: sliding-window prefill SDPA is wrong when q_chunk == 2*k_chunk.

``ttnn.transformer.scaled_dot_product_attention(..., is_causal=True,
sliding_window_size=W)`` returns visibly wrong output (PCC ~0.97 against a
plain PyTorch masked-softmax reference) for several sequence lengths a little
past the window when the SDPA program config uses ``q_chunk_size == 2 *
k_chunk_size``.  Every configuration with ``q_chunk_size == k_chunk_size`` is
accurate (~0.9998) on the same inputs, as are the same q/k chunk pairs at other
sequence lengths, so this is a work-plan/masking bug rather than precision.

Observed on Blackhole (11x10 grid), tt-metal @ this branch — this is exactly the
output of the committed run in ``logs/sdpa_sliding_window_chunk_repro.log``:

    S=2080  q256/k128=0.97796  q128/k64=0.97319  q128/k128=0.99987  q256/k256=0.99988
    S=2304  q256/k128=0.99983  q128/k64=0.99983  q128/k128=0.99982  q256/k256=0.99982
    S=4128  q256/k128=0.97495  q128/k64=0.96852  q128/k128=0.99980  q256/k256=0.99981
    S=8224  q256/k128=0.97613  q128/k64=0.97084  q128/k128=0.99985  q256/k256=0.99986

The Muse-Glimmer functional decoder works around it by always using
``q_chunk_size == k_chunk_size`` in its prefill SDPA program config.

Run with::

    python models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder/sdpa_sliding_window_chunk_repro.py
"""

from __future__ import annotations

import torch

import ttnn

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
WINDOW = 2048
SCALE = 0.342063
SEQ_LENS = (2080, 2304, 4128, 8224)
CHUNK_PAIRS = ((256, 128), (128, 64), (128, 128), (256, 256))


def torch_reference(q, k, v, seq_len):
    keys = k.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
    values = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
    index = torch.arange(seq_len)
    mask = (index[:, None] >= index[None, :]) & (index[None, :] > index[:, None] - WINDOW)
    scores = (q.float() @ keys.transpose(-1, -2)) * SCALE
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ values


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def to_device(tensor):
            return ttnn.from_torch(
                tensor,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        torch.manual_seed(0)
        for seq_len in SEQ_LENS:
            q = (torch.randn(1, NUM_Q_HEADS, seq_len, HEAD_DIM) / 3).to(torch.bfloat16)
            k = (torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM) / 3).to(torch.bfloat16)
            v = (torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM) / 3).to(torch.bfloat16)
            expected = torch_reference(q, k, v, seq_len)
            results = []
            for q_chunk, k_chunk in CHUNK_PAIRS:
                out = ttnn.transformer.scaled_dot_product_attention(
                    to_device(q),
                    to_device(k),
                    to_device(v),
                    is_causal=True,
                    scale=SCALE,
                    sliding_window_size=WINDOW,
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                        q_chunk_size=q_chunk,
                        k_chunk_size=k_chunk,
                        exp_approx_mode=False,
                    ),
                    compute_kernel_config=compute_kernel_config,
                )
                actual = ttnn.to_torch(out).float()
                pcc = float(torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1])
                results.append(f"q{q_chunk}/k{k_chunk}={pcc:.5f}")
            print(f"S={seq_len}  " + "  ".join(results), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
