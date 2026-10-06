# SPDX-License-Identifier: Apache-2.0
"""Exact integer prefix offsets via a BF16 triangular matrix and FP32 sums."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from . import multichip_checks


class PrefixDecoder(MultichipDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.prefix_matrix = ttnn.from_torch(
            torch.triu(torch.ones(384, 384, dtype=torch.bfloat16)),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        self.old_cumsum = ttnn.cumsum

        def prefix(x, dim):
            assert dim == 3 and x.shape[-1] == 384
            return ttnn.matmul(
                ttnn.typecast(x, ttnn.bfloat16),
                self.prefix_matrix,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.prefix = prefix
        # Real per-expert aligned counts cannot exceed4096. All multiples32 in
        # that interval are exactly BF16; FP32 output preserves integer sums.
        gen = torch.Generator().manual_seed(719)
        counts = torch.randint(0, 129, (1, 1, 32, 384), generator=gen, dtype=torch.int32) * 32
        counts[0, 0, 0] = 4096
        counts[0, 0, 1] = 0
        x = ttnn.from_torch(
            counts.float(),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        y = prefix(x, 3)
        expected = counts.cumsum(-1).float()
        for device_y in ttnn.get_device_tensors(y):
            assert torch.equal(ttnn.to_torch(device_y), expected)
        ttnn.cumsum = prefix
        return self


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    a.baseline = False
    a.tag = "prefix_matmul"
    os.environ["MC_POLICY"] = "{}"
    multichip_checks.MultichipDecoder = PrefixDecoder
    original = ttnn.cumsum
    try:
        multichip_checks.run(a)
    finally:
        ttnn.cumsum = original
    path = multichip_checks.OUT / f"{a.tag}_{a.layer}_{a.tokens}.json"
    value = json.loads(path.read_text())
    value["candidate_adapter"] = dict(
        path=__file__,
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exact_prefix_test_rows=32,
        all_four_devices_exact=True,
        decoder_sha256=hashlib.sha256((multichip_checks.ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
    )
    path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
