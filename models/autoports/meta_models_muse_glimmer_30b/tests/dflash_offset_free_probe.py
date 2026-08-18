# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is the offset-free verify *wrong*, or merely numerically different?

F24 held ``offset_free_verify`` back because generated tokens diverge from greedy at
exactly one 32-row window in.  That was graded with token equality -- which **F2
already established is not a valid correctness gate for this port**: the target's own
argmax depends on forward width in bf16, so two arithmetically-equivalent graphs
legitimately disagree on argmax and then cascade.

The offset-free path pins the SDPA chunk to ``TILE_SIZE`` where the shipped path derives
it from ``start_pos`` (64 at offset 64, 32 at offset 96).  A different chunk size is a
different reduction order, i.e. exactly the bf16 sensitivity F2 describes -- and the one
number that would fall if the graph were actually broken, acceptance, went *up* (2.783
against 2.723).

So compare the two graphs directly, on the same window, with a metric that survives bf16:
PCC of the hidden state plus argmax agreement over the block rows.  No generation loop, so
no cascade, and each window is judged on its own.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import DFlashDrafter
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_runner import DFlashRunner
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    build_generator,
    close_generator_mesh,
    open_generator_mesh,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a32 = a32 - a32.mean()
    b32 = b32 - b32.mean()
    denom = a32.norm() * b32.norm()
    return float((a32 @ b32) / denom) if float(denom) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offsets", type=int, nargs="*", default=[64, 96, 128, 160])
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args()

    mesh = open_generator_mesh()
    try:
        gen = build_generator(".", mesh, max_batch_size=1, max_seq_len=4096, page_block_size=32)
        model = gen.model
        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=R.draft_config(),
            mesh_device=mesh,
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat16,
        )
        runner = DFlashRunner(gen, drafter, trace_verify=False, offset_free_verify=True)

        rows = args.rows
        top = max(args.offsets) + rows
        torch.manual_seed(0)
        # Deterministic pseudo-text: real ids, so the K/V cache holds plausible values.
        ids = torch.randint(low=1000, high=20000, size=(top + rows,), dtype=torch.int32).tolist()

        # Same order generate() uses: the staging buffers must exist before _stage
        # writes into them, or the page-table key is simply absent.
        gen._invalidate_traces_if_cache_moved()
        gen._allocate_device_inputs()
        table = gen._coerce_page_table(None)
        gen._stage(page_table=table)
        slot_row = model.page_table_row(table, 0)
        tt_page_table = model.page_table_row_to_device(slot_row)
        runner._ensure_verify_inputs(tt_page_table, host_page_row=slot_row)

        tap_layers = runner._tap_layers()
        print(f"{'offset':>8} {'PCC':>10} {'max|Δ|':>10} {'argmax agree':>14}")
        for aligned_start in args.offsets:
            window = ids[aligned_start : aligned_start + rows]
            host = torch.zeros((1, rows), dtype=torch.int32)
            host[0, :] = torch.tensor(window, dtype=torch.int32)

            def stage_tokens() -> None:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        host,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=ttnn.uint32,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    ),
                    runner._verify_tokens,
                )

            def forward(runtime_offset) -> torch.Tensor:
                model.arm_hidden_state_taps(tap_layers)
                model.release_sliding_tails()
                stage_tokens()
                embedded = model.embed_prefill(runner._verify_tokens)
                hidden = model.prefill_forward(
                    embedded,
                    page_table=tt_page_table,
                    user_id=0,
                    start_pos=aligned_start,
                    runtime_offset=runtime_offset,
                )
                ttnn.synchronize_device(mesh)
                out = ttnn.to_torch(hidden, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].clone()
                ttnn.deallocate(hidden)
                for tensor in model.take_hidden_state_taps().values():
                    ttnn.deallocate(tensor)
                return out

            shipped = forward(None)
            runner._stage_offset(aligned_start)
            free = forward(runner._offset_inputs)

            a = shipped.reshape(-1, shipped.shape[-1])[:rows]
            b = free.reshape(-1, free.shape[-1])[:rows]
            agree = int((a.argmax(dim=-1) == b.argmax(dim=-1)).sum())
            print(
                f"{aligned_start:>8} {pcc(a, b):>10.6f} "
                f"{float((a.to(torch.float32) - b.to(torch.float32)).abs().max()):>10.5f} "
                f"{agree:>10}/{rows}"
            )
        runner.release_verify_traces()
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
