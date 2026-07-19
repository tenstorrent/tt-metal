# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal P300 Sampling1D watcher reproducer, independent of model code."""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def main() -> None:
    mesh_device = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    try:
        tt_ccl = get_tt_ccl(mesh_device)
        num_links = int(os.getenv("SAMPLER_LINKS", "2"))
        sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=131072,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=num_links,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=num_links,
                ag_topology=ttnn.Topology.Ring,
                pad_to_power_of_2=True,
            )
        )
        vocab_size = 131072
        hf_vocab_size = 128256
        expected = torch.tensor([(row * 4093 + 17) % hf_vocab_size for row in range(32)], dtype=torch.long)
        logits = torch.full((1, 1, 32, vocab_size), -10.0, dtype=torch.bfloat16)
        logits[..., hf_vocab_size:] = torch.finfo(torch.bfloat16).min
        for row, token in enumerate(expected.tolist()):
            logits[0, 0, row, token] = 20.0
        logits_tt = ttnn.from_torch(
            logits,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=(1, 4)),
        )
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output = ttnn.from_torch(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        sampler.load_device_buffers()
        greedy_one = sampler.greedy_decode_forward(logits_tt)
        greedy_two = sampler.greedy_decode_forward(logits_tt)
        greedy_output = sampler.greedy_decode_forward(logits_tt, tt_out_tok=output)
        gathered_values, gathered_indices = sampler._topk(ttnn.typecast(logits_tt, dtype=ttnn.bfloat16))
        first_values = ttnn.to_torch(ttnn.get_device_tensors(gathered_values)[0]).float()[0, 0]
        first_indices = ttnn.to_torch(ttnn.get_device_tensors(gathered_indices)[0]).long()[0, 0]
        offsets = torch.arange(4).reshape(1, 4, 1) * (vocab_size // 4)
        global_indices = (first_indices.reshape(32, 4, 32) + offsets).reshape(32, 128)
        winners = first_values.argmax(dim=-1)
        candidate_tokens = global_indices.gather(1, winners.reshape(-1, 1)).reshape(-1)
        force, _ = sampler.decode_forward(logits_tt)
        ttnn.synchronize_device(mesh_device)
        greedy_one_host = ttnn.to_torch(ttnn.get_device_tensors(greedy_one)[0]).reshape(-1)[:32].long()
        greedy_two_host = ttnn.to_torch(ttnn.get_device_tensors(greedy_two)[0]).reshape(-1)[:32].long()
        greedy_output_host = ttnn.to_torch(ttnn.get_device_tensors(greedy_output)[0]).reshape(-1)[:32].long()
        force_host = ttnn.to_torch(ttnn.get_device_tensors(force)[0]).reshape(-1)[:32].long()
        print(f"expected={expected.tolist()}")
        print(f"candidate_tokens={candidate_tokens.tolist()}")
        print(f"greedy_one={greedy_one_host.tolist()}")
        print(f"greedy_two={greedy_two_host.tolist()}")
        print(f"greedy_caller_output={greedy_output_host.tolist()}")
        print(f"force={force_host.tolist()}")
        print(
            "row0_segments="
            f"{[(first_values[0, rank * 32 : (rank + 1) * 32].max().item(), first_indices[0, rank * 32].item()) for rank in range(4)]}"
        )
        assert torch.equal(candidate_tokens, expected)
        assert torch.equal(greedy_one_host, expected)
        assert torch.equal(greedy_two_host, expected)
        assert torch.equal(greedy_output_host, expected)
        assert torch.equal(force_host, expected)
        print("SAMPLER_WATCHER_REPRO_OK")
    finally:
        close_readiness_mesh_device(mesh_device, "FABRIC_1D_RING")


if __name__ == "__main__":
    main()
