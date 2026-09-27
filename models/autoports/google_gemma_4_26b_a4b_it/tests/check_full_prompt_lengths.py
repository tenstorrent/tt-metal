# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logical prompt boundaries through the real reduced public generator."""
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/full_model")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    rows = []
    try:
        gen = build_generator(None, mesh, max_seq_len=8192, layer_indices=(0, 5))
        for length in (1, 31, 32, 33, 1023, 1024, 1025, 4097):
            tokens = gen.generate([2] + [100] * (length - 1), 3, stop_on_eos=False)
            assert len(tokens) == 3 and all(0 <= t < gen.model.config.vocab_size for t in tokens)
            pos = ttnn.to_torch(ttnn.get_device_tensors(gen.cache_positions)[0]).item()
            assert pos == length + 2
            rows.append(dict(prompt_length=length, returned_tokens=len(tokens), final_position=pos, passed=True))
            (root / "prompt_lengths.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(rows[-1], flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
