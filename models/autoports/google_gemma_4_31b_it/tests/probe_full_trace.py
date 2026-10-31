"""Short split-feedback probe, isolated from all-layer accuracy gates."""

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.generator import Gemma4Generator


def main():
    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), max_seq_len=256)
        print("PREPARE", flush=True)
        gen.prepare_decode()
        print("CAPTURED", flush=True)
        gen._copy(torch.tensor([[32]], dtype=torch.int32), gen.positions)
        for step in range(3):
            result = gen._replay()
            print(
                "STEP", step, result, "POSITION", ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]), flush=True
            )
        print("GENERATE", gen.generate([2] * 33, 4), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
