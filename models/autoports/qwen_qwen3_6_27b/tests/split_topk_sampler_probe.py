# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused TP4 probe for two-stage split TopK and invalid-vocab masking."""

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import SamplingArgs
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params


def main() -> None:
    vocab_size = 248_064
    padded_vocab_size = 248_320
    expected = [0, 32_767, 32_768, vocab_size - 1]
    logits_host = torch.full((1, 1, 32, padded_vocab_size), -100.0, dtype=torch.bfloat16)
    # Invalid padded IDs remain zero, deliberately higher than ordinary valid
    # logits.  The explicit sampler mask must prevent them from winning.
    logits_host[..., vocab_size:] = 0.0
    for row, token in enumerate(expected):
        logits_host[0, 0, row, token] = 20.0 + row

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16_000_000)
    sampler = None
    try:
        config = SamplingArgs(vocab_size, padded_vocab_size, 4, force_argmax_active_rows=4)
        config.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = False
        sampler = SamplingGenerator(args=config, mesh_device=mesh, tt_ccl=get_tt_ccl(mesh))
        sampler.reset_sampling_params(format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32))
        logits = ttnn.from_torch(
            logits_host,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        feedback = ttnn.from_torch(
            torch.zeros((1, 1, 1, 32), dtype=torch.uint32),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sampler.sample(logits, enable_trace=True, tt_out_tok=feedback)
        sampler.sample(logits, enable_trace=True, tt_out_tok=feedback)
        ttnn.synchronize_device(mesh)
        for rank, device_tensor in enumerate(ttnn.get_device_tensors(feedback)):
            actual = [int(x) for x in ttnn.to_torch(device_tensor).reshape(-1)[:4]]
            assert actual == expected, (rank, actual, expected)
            assert all(token < vocab_size for token in actual), (rank, actual)
        print("SPLIT_TOPK_SAMPLER_OK", expected, flush=True)
    finally:
        if sampler is not None:
            sampler.reset_trace()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
