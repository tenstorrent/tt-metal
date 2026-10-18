# SPDX-License-Identifier: Apache-2.0
"""Exhaustively prove the QSA tail-count rewrite over its public context range."""

import hashlib
import json
from pathlib import Path

import torch

import ttnn


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        positions = torch.arange(262144, dtype=torch.float32).reshape(1, 1, -1, 1)
        pos = ttnn.from_torch(positions, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        complete = ttnn.floor(ttnn.multiply(ttnn.add(pos, 1.0), 0.25))
        control = ttnn.subtract(ttnn.add(pos, 1.0), ttnn.multiply(complete, 4.0))
        next_pos = ttnn.add(pos, 1.0)
        new_complete = ttnn.floor(ttnn.multiply(next_pos, 0.25))
        candidate = ttnn.mac(new_complete, -4.0, next_pos)
        expected = torch.remainder(positions + 1, 4)
        complete_host, new_host = ttnn.to_torch(complete), ttnn.to_torch(new_complete)
        control_host, candidate_host = ttnn.to_torch(control), ttnn.to_torch(candidate)
        assert torch.equal(complete_host, new_host)
        assert torch.equal(control_host, candidate_host)
        assert torch.equal(candidate_host, expected)
        result = dict(
            positions=262144,
            minimum=0,
            maximum=262143,
            dtype="FP32",
            block_count_bitwise=True,
            tail_count_bitwise=True,
            integer_oracle_bitwise=True,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        output = Path(__file__).resolve().parents[1] / "doc/fused_decoder/position_arithmetic_exhaustive.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("POSITION_ARITHMETIC_EXHAUSTIVE_PASS", json.dumps(result))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
