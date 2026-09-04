# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Muse-Glimmer's full-width P150x2 top-k sampling geometry."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import close_multichip_mesh, open_multichip_mesh
from models.common.sampling.tt_sampling import TTSampling

VOCAB = 202048
PADDED_VOCAB = 202112
EXPECTED_TOKENS = (50000, 150000)


def candidate_maxima(sampler: TTSampling, logits: ttnn.Tensor) -> list[int]:
    """Global index paired with each row's largest gathered candidate."""
    values = ttnn.clone(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if sampler.pad_to_power_of_2 and values.shape[-1] & (values.shape[-1] - 1):
        width = 1 << (int(values.shape[-1]) - 1).bit_length()
        values = ttnn.pad(values, [(0, 0), (0, 0), (0, 0), (0, width - values.shape[-1])], value=float("-inf"))
    local_values, local_indices = sampler._topk_multicore_split(values)
    gathered_values = sampler._perform_all_gather(
        local_values,
        dim=3,
        cluster_axis=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_links=sampler.num_gather_links,
    )
    gathered_indices = sampler._perform_all_gather(
        local_indices,
        dim=3,
        cluster_axis=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_links=sampler.num_gather_links,
        dtype=ttnn.uint16,
    )
    indices_i32 = ttnn.typecast(gathered_indices, dtype=ttnn.int32)
    global_indices = ttnn.add(
        sampler.tt_indices_device_offsets,
        indices_i32,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    indices_rm = ttnn.untilize(global_indices, use_multicore=True)

    host_values = ttnn.to_torch(ttnn.get_device_tensors(gathered_values)[0]).reshape(32, -1)
    host_indices = ttnn.to_torch(ttnn.get_device_tensors(indices_rm)[0]).reshape(32, -1)
    positions = host_values.float().argmax(dim=-1)
    maxima = host_indices[torch.arange(32), positions].to(torch.int64).tolist()

    for tensor in (
        local_values,
        local_indices,
        gathered_values,
        gathered_indices,
        indices_i32,
        global_indices,
        indices_rm,
    ):
        ttnn.deallocate(tensor)
    ttnn.deallocate(values)
    return maxima


def main() -> int:
    mesh = open_multichip_mesh(mesh_shape=(1, 2), trace_region_size=0)
    try:
        args = SimpleNamespace(
            vocab_size=VOCAB,
            padded_vocab_size=PADDED_VOCAB,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=[1, 2],
            sampling_dp=1,
            pad_logits_to_power_of_2=False,
            topk_split_to_power_of_2=True,
            model_config={
                "GALAXY_NUM_LINKS": 2,
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": False,
                    "num_links": 2,
                    "topology": ttnn.Topology.Linear,
                },
            },
        )
        sampler = TTSampling(mesh_device=mesh, tt_ccl=None, args=args)
        if sampler.topk_pieces != 4 or sampler.candidates_per_device != 128:
            raise RuntimeError(
                f"unexpected P150x2 geometry: pieces={sampler.topk_pieces}, "
                f"candidates/device={sampler.candidates_per_device}"
            )

        for expected in EXPECTED_TOKENS:
            host_logits = torch.full((1, 1, 32, PADDED_VOCAB), -4.0, dtype=torch.bfloat16)
            host_logits[:, :, :, expected] = 4.0
            logits = ttnn.from_torch(
                host_logits,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            maxima = candidate_maxima(sampler, logits)
            print(f"CANDIDATE expected={expected} maxima={maxima}", flush=True)
            sampled, _ = sampler(logits)
            for device, tensor in enumerate(ttnn.get_device_tensors(sampled)):
                got = ttnn.to_torch(tensor).reshape(-1)[:32]
                if not got.eq(expected).all():
                    raise RuntimeError(
                        f"P150x2 sampling returned {got.tolist()} on device {device}, expected {expected}"
                    )
            ttnn.deallocate(sampled)
            ttnn.deallocate(logits)

        print(
            f"PROBE_PASS rows=32 tokens={EXPECTED_TOKENS} pieces={sampler.topk_pieces} "
            f"candidates_per_device={sampler.candidates_per_device}",
            flush=True,
        )
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
