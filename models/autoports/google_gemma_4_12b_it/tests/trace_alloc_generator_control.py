# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Positive control on the generator itself (run with TT_METAL_TRACE_ALLOC_TRACKING=1).

The reduced generator (layers 0 and 5) is prepared and captured, a traced decode step passes, then one program that
``prepare()`` never compiled runs (a decoder-layer prefill with a non-bucket row count). Its program-cache buffers
are allocated after the captures, so the next traced decode must be refused by the tracker. This shows that the
clean tracking runs of the generator are meaningful (warmup coverage is what keeps them clean).
"""

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_12b_it.tt.model import Gemma4Model
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

mesh = open_qb2_mesh(trace_region_size=0)
try:
    model = Gemma4Model(mesh, layer_indices=[0, 5], max_context=16384, num_blocks=300, log=lambda *_: None)
    gen = Gemma4Generator(model, decode_buckets=(1,))
    row = gen._alloc_row(64)
    gen.decode_forward(torch.tensor([2]), torch.tensor([0]), page_table=row[None])
    print("CONTROL: traced decode after prepare accepted (no unsafe allocation)")
    table = model.device_ints(row[None])
    x = ttnn.from_torch(
        torch.zeros(1, 1, 100, 3840),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )
    ttnn.deallocate(model.layers[0].prefill_forward(x, page_table=table, start_pos=0, user_id=0))  # 100 rows: unwarmed
    ttnn.deallocate(x)
    ttnn.deallocate(table)
    try:
        gen.decode_forward(torch.tensor([3]), torch.tensor([1]), page_table=row[None])
        ttnn.synchronize_device(mesh)
        print("CONTROL: replay accepted after a post-capture compile (checker inactive)")
    except RuntimeError as e:
        print("CONTROL: checker active, replay after a post-capture compile refused:", str(e).splitlines()[0][:200])
    gen.teardown()
finally:
    close_qb2_mesh(mesh)
