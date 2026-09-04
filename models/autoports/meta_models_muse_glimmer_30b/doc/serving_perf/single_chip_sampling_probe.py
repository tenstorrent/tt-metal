# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Muse-Glimmer's full-width single-chip top-k sampling geometry."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.common.sampling.tt_sampling import TTSampling

VOCAB = 202048
EXPECTED_TOKEN = 150000


def main() -> int:
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        args = SimpleNamespace(
            vocab_size=VOCAB,
            padded_vocab_size=VOCAB,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=[1, 1],
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
        host_logits = torch.full((1, 1, 32, VOCAB), -4.0, dtype=torch.bfloat16)
        host_logits[:, :, :, EXPECTED_TOKEN] = 4.0
        logits = ttnn.from_torch(
            host_logits,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sampled, _ = sampler(logits)
        got = ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[:32]
        if not got.eq(EXPECTED_TOKEN).all():
            raise RuntimeError(f"single-chip sampling returned {got.tolist()}, expected {EXPECTED_TOKEN}")
        print(f"PROBE_PASS rows=32 token={EXPECTED_TOKEN}", flush=True)
        ttnn.deallocate(sampled)
        ttnn.deallocate(logits)
        return 0
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
