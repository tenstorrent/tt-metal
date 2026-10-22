import sys

import torch
import ttnn


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    torch.manual_seed(0)
    verbose = "-v" in sys.argv
    shapes = [(1, 1, 32, 64), (1, 1, 64, 64), (1, 2, 64, 64)]
    if "--big" in sys.argv:
        shapes = [(1, 4, 256, 128)]
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        T = torch.zeros(32, 32)
        for i in range(16):
            T[2 * i, 2 * i + 1] = 1.0
            T[2 * i + 1, 2 * i] = -1.0
        for B, H, S, D in shapes:
            x = torch.randn(B, H, S, D)
            for mode in ["identity", "sin_only", "full"]:
                if mode == "identity":
                    cos, sin = torch.ones(1, 1, S, D), torch.zeros(1, 1, S, D)
                elif mode == "sin_only":
                    cos, sin = torch.zeros(1, 1, S, D), torch.ones(1, 1, S, D)
                else:
                    cos, sin = torch.randn(1, 1, S, D), torch.randn(1, 1, S, D)
                blk = torch.block_diag(*[T] * (D // 32))
                ref = x * cos + (x @ blk) * sin

                def mk(t):
                    return ttnn.from_torch(
                        t,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                out = ttnn.experimental.rotary_embedding_llama(
                    mk(x), mk(cos), mk(sin), mk(T.reshape(1, 1, 32, 32)), is_decode_mode=False
                )
                got = ttnn.to_torch(out).float()[:B, :H, :S, :D]
                print(
                    f"{(B, H, S, D)} {mode:9s} pcc={pcc(got, ref):.5f} max_abs={(got - ref).abs().max().item():.4f}",
                    flush=True,
                )
                if verbose:
                    print("got", got[0, 0, :2, :8], "\nref", ref[0, 0, :2, :8], "\nx", x[0, 0, :2, :8])
    finally:
        ttnn.close_mesh_device(dev)


main()
