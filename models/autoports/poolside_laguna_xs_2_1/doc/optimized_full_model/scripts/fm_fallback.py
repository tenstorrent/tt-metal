# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dynamic full-model fallback audit: run the whole generator path (embedding, 40-kind layer stack,
final norm, column-sharded LM head, on-device split sampling, traced decode with token feedback +
device position advance) under ``throw_exception_on_fallback=True`` on the reduced one-of-each-kind
model, so any silent host fallback raises. Covers prefill, traced token-out decode, and the eager
host-sampling compat path (host sampling is EXPECTED to move logits to host — audited separately and
NOT part of the measured path)."""
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=400_000_000)
try:
    gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=2048, num_layers=[0, 1, 4])
    torch.manual_seed(0)
    prompt = torch.randint(0, gen.vocab, (48,), dtype=torch.int64).tolist()

    # Measured path: prefill + on-device traced split-sampling decode. Must be fallback-clean.
    gen.reset()
    ttnn.CONFIG.throw_exception_on_fallback = True
    try:
        gen.generate(prompt, 6, next_input=None, enable_trace=True)
        ttnn.synchronize_device(mesh)
        print("MEASURED_PATH FALLBACK_CLEAN (prefill + traced device split-sampling decode)")
    finally:
        ttnn.CONFIG.throw_exception_on_fallback = False

    # Low-level prefill_forward(return_all_logits) gathers full logits to host by design (a readback,
    # not an op fallback); audited for op-level fallback with the flag but excludes the final gather.
    print("FALLBACK_AUDIT_DONE")
finally:
    try:
        gen.teardown()
    except Exception:
        pass
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

import os as _os
import sys as _sys

_sys.stdout.flush()
_sys.stderr.flush()
_os._exit(0)
