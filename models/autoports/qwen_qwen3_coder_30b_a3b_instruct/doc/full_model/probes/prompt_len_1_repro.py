# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproduction: a whole-tensor ``ttnn.slice`` aliases its input.

This is the defect that made a **one-token prompt** segfault the full model.
``prefill_forward`` retains the last prompt row with ``ttnn.slice`` and then
deallocates the chunk it came from. At every prompt length but one that is a
real copy. At ``prompt_len == 1`` the requested slice covers the entire tensor,
``ttnn.slice`` returns a view -- as a **different Python object**, so an ``is``
guard does not catch it -- and the deallocate frees the buffer the retained row
points at. Nothing raises; the next op that reads it takes SIGSEGV.

Run with ``--fixed`` to see the ``ttnn.clone`` the model now uses instead.

    python .../probes/prompt_len_1_repro.py            # prints the aliasing
    python .../probes/prompt_len_1_repro.py --fixed    # prints the fix working
"""

from __future__ import annotations

import argparse

import torch

import ttnn

H = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        for seq_len in (1, 2):
            host = torch.randn(1, 1, seq_len, H)
            hidden = ttnn.from_torch(
                host,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            row = seq_len - 1
            if args.fixed and seq_len == 1:
                piece = ttnn.clone(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                piece = ttnn.slice(hidden, [0, 0, row, 0], [1, 1, row + 1, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            same_object = piece is hidden
            same_buffer = piece.buffer_address() == hidden.buffer_address()
            print(
                f"seq_len={seq_len} row={row}: piece is hidden -> {same_object}, "
                f"same buffer address -> {same_buffer}",
                flush=True,
            )
            if same_buffer and not args.fixed:
                print("  ^^ deallocating `hidden` here leaves `piece` dangling; the next read segfaults", flush=True)
            ttnn.deallocate(hidden, True)
            # Reading `piece` after the deallocate is the crash in the unfixed case.
            value = ttnn.to_torch(piece)
            print(f"  read back ok, finite={bool(torch.isfinite(value).all())}", flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
