# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Opt-in startup warmup for DFlash prefill tap capture and drafter seeding.

The drafter seed runs at its fixed SEED_ROWS shape, so its kernels compile once per variant, not per prompt
length: taps shorter than SEED_ROWS (zero fill) vs a full window, a partial last tile (validity select) or
none, and a window starting at row 0 vs a later tile (offset slices and the ring roll).

The tap capture (Gemma4Model.capture_tap) is per row count: a hidden-sharded prefill output (L > 128) is
gathered to full width, and all_gather_async takes the row-tile count as a compile-time argument
(all_gather_async_default_program_factory.cpp, input/output_tensor_Ht). Every window row count from 160 to
SEED_ROWS is its own kernel: 61 classes, about 1 s each on a cold kernel cache. warmup_tap_seed compiles all
of them, plus the seed variants, at startup instead of on users' first prompts. The drafter's context ring
is reset afterwards; the target model and its KV cache are not touched.
"""

import time

import torch

import ttnn


def warmup_tap_seed(model, drafter, lengths=None, row_classes=True):
    """Compile the tap-capture and seed programs.

    lengths: prompt lengths run through capture_tap + seed; default (20, 64, 100, SEED_ROWS + 52): replicated-width
    prefill outputs (<= 128 rows) with a lone partial tile and with whole tiles, and a full window starting past
    row 0 with a partial tile. row_classes: also gather every hidden-sharded window row count (160..SEED_ROWS).
    Returns {length or "row_classes": seconds}. Call after model and drafter construction, before trace capture.
    """
    hidden, mesh = model.config.hidden_size, model.mesh
    lengths = lengths or (20, 64, 100, drafter.SEED_ROWS + 52)

    def layer_output(rows, width):
        # A prefill layer output: [1,1,L,5376] replicated for L <= 128, else [1,1,L,1344] hidden-sharded.
        return ttnn.from_torch(
            torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if width == hidden else ttnn.ShardTensorToMesh(mesh, dim=3),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    spent = {}
    for length in lengths:
        t = time.perf_counter()
        start, end = drafter.tap_window(length)
        x = layer_output(length, model.layers[0].prefill_input_width(length))
        taps = {layer: model.capture_tap(x, start, end) for layer in drafter.tap_layers}
        drafter.seed(taps, start, length - start)
        ttnn.synchronize_device(mesh)
        spent[length] = time.perf_counter() - t
        del taps, x
    if row_classes:
        t = time.perf_counter()
        first = next(r for r in range(32, drafter.SEED_ROWS + 1, 32) if model.layers[0].prefill_input_width(r) != hidden)
        x = layer_output(drafter.SEED_ROWS, model.layers[0].prefill_input_width(drafter.SEED_ROWS))
        for rows in range(first, drafter.SEED_ROWS + 1, 32):
            # A window [0, rows) of a prompt of `rows` tokens: capture_tap's clone + gather at this row count.
            part = x[:, :, :rows, :]
            model.capture_tap(part, 0, rows)
        ttnn.synchronize_device(mesh)
        spent["row_classes"] = time.perf_counter() - t
        del x
    drafter.reset()
    return spent
