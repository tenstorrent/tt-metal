# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signposted perf windows for Tracy / ``tt-perf-report``.

Standalone rather than a pytest node so a profiled window never drags a fixture's
extra device work into the capture, and so the profiling script does not depend on
the correctness suite's parametrisation.  One window per invocation
(``$tt-device-usage``: one device job at a time).

    python -m tracy -r -p -v .../bench/perf_windows.py --window prefill --seq-len 8192 --kind sliding
    python -m tracy -r -p -v .../bench/perf_windows.py --window decode  --context 2048 --kind full

Signposts are ``PERF_PREFILL`` / ``PERF_PREFILL_END`` and ``PERF_DECODE`` /
``PERF_DECODE_END``, matching every earlier stage in this model so the tables are
directly comparable.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_L1_SMALL_SIZE,
    DEFAULT_MESH_SHAPE,
    FABRIC_CONFIG,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - profiler not installed

    def signpost(header: str) -> None:
        del header


PAGE_BLOCK = 64
SHORT_MAX_SEQ = 16384
FULL_CONTEXT = 131072


def replicated(mesh, tensor: torch.Tensor, *, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def page_table(mesh, batch: int, max_seq_len: int, seed: int) -> ttnn.Tensor:
    blocks = (max_seq_len + PAGE_BLOCK - 1) // PAGE_BLOCK
    perm = torch.randperm(batch * blocks, generator=torch.Generator().manual_seed(seed))
    return replicated(mesh, perm.reshape(batch, blocks).to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def hidden(mesh, batch: int, seq_len: int, seed: int) -> ttnn.Tensor:
    tensor = R.synthetic_hidden_states(batch, seq_len, seed=seed)
    return replicated(mesh, tensor.reshape(1, 1, batch * seq_len, -1), dtype=ttnn.bfloat16)


def positions(mesh, values: torch.Tensor):
    current = replicated(mesh, values.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rope = replicated(mesh, values.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return current, rope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=("prefill", "decode"))
    parser.add_argument("--kind", default="sliding", choices=("sliding", "full"))
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    parser.add_argument("--single-chip", action="store_true", help="profile the OptimizedDecoder baseline instead")
    args = parser.parse_args()

    mesh_shape = tuple(int(v) for v in args.mesh.split("x"))
    single = args.single_chip or mesh_shape == (1, 1)
    max_seq_len = SHORT_MAX_SEQ if max(args.context + 1, args.seq_len) <= SHORT_MAX_SEQ else FULL_CONTEXT
    layer_idx = reference_layer_indices(R.hf_config())[args.kind]
    state_dict = R.synthetic_state_dict(layer_idx)

    # The single-chip baseline is opened without fabric: there is no link to
    # configure on a 1x1 mesh and the baseline has to be measured in its own regime.
    mesh = open_multichip_mesh(
        mesh_shape,
        trace_region_size=90112 * 12,
        l1_small_size=0 if single else DEFAULT_L1_SMALL_SIZE,
        fabric_config=None if single else FABRIC_CONFIG,
    )
    try:
        builder = OptimizedDecoder if single else MultichipDecoder
        decoder = builder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=max(args.batch, 1),
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK,
        )
        table = page_table(mesh, max(args.batch, 1), max_seq_len, seed=3)

        if args.window == "prefill":
            tt_hidden = hidden(mesh, 1, args.seq_len, seed=42)
            for _ in range(2):
                ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=table, user_id=0))
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            out = decoder.prefill_forward(tt_hidden, page_table=table, user_id=0)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            print(f"PERF prefill kind={args.kind} seq_len={args.seq_len} out={tuple(out.shape)}")
            ttnn.deallocate(out)
        else:
            warm_prompt = min(2048, args.context)
            ttnn.deallocate(decoder.prefill_forward(hidden(mesh, 1, warm_prompt, seed=43), page_table=table, user_id=0))
            token = hidden(mesh, 1, args.batch, seed=44)
            current, rope = positions(mesh, torch.full((args.batch,), args.context))
            warm = decoder.decode_forward(token, current_pos=current, page_table=table, rope_pos_ids=rope)
            ttnn.deallocate(warm)
            ttnn.synchronize_device(mesh)
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            held = decoder.decode_forward(token, current_pos=current, page_table=table, rope_pos_ids=rope)
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            for _ in range(4):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            for _ in range(args.iters):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            print(f"PERF decode kind={args.kind} context={args.context} iters={args.iters} out={tuple(held.shape)}")
            ttnn.release_trace(mesh, trace_id)
            ttnn.deallocate(held)
    finally:
        close_multichip_mesh(mesh, fabric_config=None if single else FABRIC_CONFIG)


if __name__ == "__main__":
    main()
