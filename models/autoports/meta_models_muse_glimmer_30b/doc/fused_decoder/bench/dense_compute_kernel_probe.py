# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: the prefill dense kernel's compute-kernel config, end to end.

``ttnn.experimental.minimal_matmul``'s own default compute-kernel config is more
accurate than ``ttnn.linear``'s HiFi2 / no-fp32-accumulate default for BF16, and
the fused decoder deliberately does *not* take it: precision policy belongs to
the optimized-decoder stage, and pinning the baseline's fidelity keeps the
fusing stage's before/after a pure topology comparison.

This measures both, end to end through ``FusedDecoder.prefill_forward``, for
both layer kinds at 2049 and 8192 tokens.  The reported time includes the host
upload of the activation each iteration (the tensors are rebuilt per call so
nothing is reused across the two configs), so the *delta* between the two rows
is the signal, not the absolute.

Output: ``doc/fused_decoder/logs/dense_compute_kernel_probe.log``.
"""

import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder, reference_layer_indices
from models.common.utility_functions import comp_pcc

PAGE = 64
MAXSEQ = 16384
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.SetDefaultDevice(mesh)


def pt(b=1, seed=7):
    bps = (MAXSEQ + PAGE - 1) // PAGE
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(
        torch.randperm(b * bps, generator=g).reshape(b, bps).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def dev(h):
    return ttnn.from_torch(
        h.reshape(1, 1, h.shape[0] * h.shape[1], h.shape[2]),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


idxs = reference_layer_indices(R.hf_config())
for kind, li in idxs.items():
    sd = R.synthetic_state_dict(li)
    ref_layer = R.reference_layer(li, sd)
    dec = FusedDecoder.from_state_dict(
        sd,
        hf_config=R.hf_config(),
        layer_idx=li,
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=MAXSEQ,
        page_block_size=PAGE,
        prefill_chunk_size=8192,
    )
    for label, ck in (("hifi2(shipped)", dec.dense_compute_kernel_config), ("op-default", None)):
        dec.dense_compute_kernel_config = ck
        dec.mlp.compute_kernel_config = ck
        for S in (2049, 8192):
            hid = R.synthetic_hidden_states(1, S)
            exp, _ = R.reference_prefill(ref_layer, li, hid)
            table = pt()
            x = dev(hid)
            o = dec.prefill_forward(x, page_table=table, user_id=0)
            pcc = comp_pcc(exp.float(), ttnn.to_torch(o).reshape(1, S, -1).float(), 0.99)[1]
            ttnn.deallocate(o)
            best = 1e9
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(3):
                    xi = dev(hid)
                    ttnn.deallocate(dec.prefill_forward(xi, page_table=table, user_id=0))
                    ttnn.deallocate(xi)
                ttnn.synchronize_device(mesh)
                best = min(best, (time.perf_counter() - t0) / 3 * 1e3)
            print(f"CK {kind:8s} {label:14s} S={S:5d}  {best:8.2f} ms  PCC={pcc}", flush=True)
            ttnn.deallocate(table)
    del dec
ttnn.close_mesh_device(mesh)
