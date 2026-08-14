# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace regression for canonical padded feedback with B1/B2 active rows."""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import SamplingArgs
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=8_000_000)
    sampler = None
    try:
        vocab = 256
        config = SamplingArgs(vocab, vocab, args.batch, force_argmax_active_rows=args.batch)
        sampler = SamplingGenerator(args=config, mesh_device=mesh, tt_ccl=get_tt_ccl(mesh))
        sampler.reset_sampling_params(format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32))

        logits_host = torch.full((1, 1, 32, vocab), -100.0, dtype=torch.bfloat16)
        expected = []
        for row in range(args.batch):
            token = 17 + row * 101
            logits_host[0, 0, row, token] = 20.0 + row
            expected.append(token)
        logits = ttnn.from_torch(
            logits_host,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        feedback = ttnn.from_torch(
            torch.full((1, 1, 1, 32), 123, dtype=torch.uint32),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sampler.sample(logits, enable_trace=True, tt_out_tok=feedback)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.full((1, 1, 1, 32), 123, dtype=torch.uint32), dtype=ttnn.uint32),
            feedback,
        )
        sampler.sample(logits, enable_trace=True, tt_out_tok=feedback)
        ttnn.synchronize_device(mesh)
        per_device = [ttnn.to_torch(tensor).reshape(-1) for tensor in ttnn.get_device_tensors(feedback)]
        for rank, values in enumerate(per_device):
            actual = [int(value) for value in values[: args.batch]]
            assert actual == expected, (rank, actual, expected)
            # The row-major argmax writer emits one aligned 32-byte page, so
            # padding words after the fixed-slot prefix are unspecified.  The
            # model slices and consumes exactly ``batch`` prefix tokens.
        print(f"GREEDY_ACTIVE_ROWS_OK batch={args.batch} expected={expected}", flush=True)
    finally:
        if sampler is not None:
            sampler.reset_trace()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
