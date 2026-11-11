# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Eager forward_device time and PCC vs the reference at the paper shape with grouped series."""

import sys
import time

import torch
import ttnn

from models.experimental.chronos_forecast.tests.perf.test_paper_forward import (
    CONTEXT,
    NUM_OUTPUT_PATCHES,
    _load_reference,
)
from models.experimental.chronos_forecast.tt.model import TtChronos
from models.experimental.chronos_forecast.tt.program_configs import TtChronosPrecision


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


batch = int(sys.argv[1])
group_spec = sys.argv[2]  # "4" -> uniform groups of 4, "mixed" -> sizes cycling 1..7
precision_name = sys.argv[3] if len(sys.argv) > 3 else "default"
l1 = len(sys.argv) > 4 and sys.argv[4] == "l1"
check = batch <= 256

dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
dev.enable_program_cache()
try:
    precision = TtChronosPrecision.performance() if precision_name == "performance" else TtChronosPrecision()
    reference, _ = _load_reference()
    model = TtChronos.from_torch_model(
        dev, reference, precision, l1_chunk_tokens=precision.l1_chunk_tokens() if l1 else None
    )
    torch.manual_seed(0)
    if group_spec == "mixed":
        sizes, total = [], 0
        while total < batch:
            sizes.append(min(1 + len(sizes) % 7, batch - total))
            total += sizes[-1]
        group_ids = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        group_ids = group_ids[torch.randperm(batch)]
    else:
        group_ids = torch.arange(batch) // int(group_spec)
    context = torch.cumsum(torch.randn(batch, CONTEXT), dim=-1)
    t0 = time.perf_counter()
    prepared = model.prepare_inputs(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES)
    prep_s = time.perf_counter() - t0
    inputs = model.upload_inputs(prepared)
    for _ in range(2):
        out = model.forward_device(inputs)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    from tracy import signpost

    ttnn.ReadDeviceProfiler(dev)
    signpost("chronos_device_forward_start")
    t0 = time.perf_counter()
    out = model.forward_device(inputs)
    ttnn.synchronize_device(dev)
    dev_s = time.perf_counter() - t0
    signpost("chronos_device_forward_stop")
    ttnn.ReadDeviceProfiler(dev)
    got = model.postprocess_output(
        out, prepared.loc_scale, num_output_patches=NUM_OUTPUT_PATCHES, output_rows=prepared.output_rows
    )
    ttnn.deallocate(out)
    model.deallocate_inputs(inputs)
    p = float("nan")
    if check:
        with torch.no_grad():
            exp = reference(context=context, group_ids=group_ids, num_output_patches=NUM_OUTPUT_PATCHES).quantile_preds
        p = pcc(exp, got)
    print(
        f"[GROUPED] batch={batch} groups={group_spec} precision={precision_name} l1={l1} prep_s={prep_s:.3f} forward_s={dev_s:.4f} pcc={p:.6f}"
    )
finally:
    ttnn.close_mesh_device(dev)
