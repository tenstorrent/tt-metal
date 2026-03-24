import torch


def frobenius_error_tiled(golden, calculated, tile_h=16, tile_w=16):
    H, W = golden.shape
    total = 0.0

    for i in range(0, H, tile_h):
        for j in range(0, W, tile_w):
            g_tile = golden[i : i + tile_h, j : j + tile_w]
            c_tile = calculated[i : i + tile_h, j : j + tile_w]

            diff = g_tile - c_tile
            total += torch.sum(diff * diff).item()

    return total**0.5


def frobenius_error_direct(golden, calculated):
    return torch.norm(golden - calculated, p="fro").item()


def main():
    # -------------------------
    # User controls
    # -------------------------
    H = 7
    W = 7
    tile_h = 3
    tile_w = 3
    seed = 0
    # -------------------------

    torch.manual_seed(seed)

    golden = torch.randn(H, W)
    calculated = golden + 0.01 * torch.randn(H, W)  # add small noise

    tiled_error = frobenius_error_tiled(golden, calculated, tile_h, tile_w)
    direct_error = frobenius_error_direct(golden, calculated)

    print("Matrix size:", (H, W))
    print("Tile size:", (tile_h, tile_w))
    print()

    print("Golden matrix:")
    print(golden)
    print()

    print("Calculated matrix:")
    print(calculated)
    print()

    print("Frobenius error (tiled):", tiled_error)
    print("Frobenius error (direct):", direct_error)
    print("Difference:", abs(tiled_error - direct_error))


if __name__ == "__main__":
    main()
