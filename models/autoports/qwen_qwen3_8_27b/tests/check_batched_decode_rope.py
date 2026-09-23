# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device check: independent request positions survive batched decode RoPE."""

from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder


def main():
    torch.manual_seed(17)
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)

        def upload(value):
            return ttnn.from_torch(
                value,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=mapper,
            )

        for batch in (1, 2, 5, 8, 16):
            for heads in (1, 6):
                x = torch.randn(1, batch, heads, 256).bfloat16()
                positions = torch.tensor(
                    [0, 7, 31, 32, 127, 4095, 8192, 32767, 3, 19, 63, 128, 511, 2047, 16383, 65535]
                )[:batch]
                phase = positions[:, None] * (10000.0 ** (-torch.arange(0, 64, 2) / 64))
                phase = torch.cat([phase, phase], dim=-1).reshape(batch, 1, 64)
                tx, cos, sin = upload(x), upload(phase.cos()), upload(phase.sin())
                outputs = []
                for enabled in (False, True):
                    layer = SimpleNamespace(policy={"batched_decode_rope": enabled})
                    output = OptimizedDecoder._rope_decode(layer, tx, cos, sin)
                    outputs.append(ttnn.to_torch(output, mesh_composer=composer))
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                # The non-rotary channels must be untouched on every device.
                torch.testing.assert_close(outputs[1][..., 64:], x[..., 64:].repeat(4, 1, 1, 1), rtol=0, atol=0)
                print(f"PASS batch={batch} heads={heads}: exact parity at independent positions", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
