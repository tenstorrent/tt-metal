"""Host-side preparation of per-channel snake parameters for the fused depthwise conv (Step 2a).

The kernel-side half of Step 2a needs alpha and 1/(beta+eps) delivered as tiles laid out exactly like
the conv's output tiles, so it can apply

    y = x + inv_beta * sin(alpha * x)^2

with plain `mul_binary_tile` and no broadcast. This module builds those tiles and is verified on CPU,
so the remaining work is CB plumbing rather than arithmetic.

Two things it encodes that are easy to get wrong:

* **inv_beta is precomputed here**, not on device. `SnakeBeta._prepare_torch_state` already folds eps
  into beta, so this takes the reciprocal of the already-offset beta -- do not add eps twice.
* **Replication down all 32 rows.** A per-channel vector is a single row; the kernel multiplies whole
  tiles, so each value must repeat down the tile's height. For C > 32 that is ceil(C/32) tiles per
  vector, ordered along the channel axis to match how output tiles walk it.
"""

import torch

TILE = 32


def snake_param_tiles(alpha: torch.Tensor, beta_plus_eps: torch.Tensor) -> torch.Tensor:
    """``(alpha, beta+eps)`` per channel -> ``(2, n_channel_tiles, TILE, TILE)`` ready for a CB.

    `beta_plus_eps` is what `SnakeBeta` stores: eps is already folded in, so this only reciprocates.
    Channels are zero-padded up to a tile boundary; alpha pads with 0 and inv_beta with 0, which makes
    the padded lanes compute ``0 + 0 * sin(0)^2 == 0`` rather than a NaN from dividing by zero.
    """
    alpha = alpha.reshape(-1).float()
    beta = beta_plus_eps.reshape(-1).float()
    assert alpha.shape == beta.shape, f"alpha {tuple(alpha.shape)} != beta {tuple(beta.shape)}"
    c = alpha.numel()
    n_tiles = (c + TILE - 1) // TILE
    padded = n_tiles * TILE

    inv_beta = torch.zeros(padded, dtype=torch.float32)
    inv_beta[:c] = 1.0 / beta
    a_pad = torch.zeros(padded, dtype=torch.float32)
    a_pad[:c] = alpha

    # (n_tiles, TILE, TILE): each tile is one channel-group, its value repeated down all rows.
    a_tiles = a_pad.reshape(n_tiles, 1, TILE).expand(n_tiles, TILE, TILE).contiguous()
    b_tiles = inv_beta.reshape(n_tiles, 1, TILE).expand(n_tiles, TILE, TILE).contiguous()
    return torch.stack([a_tiles, b_tiles])


def snake_reference(x: torch.Tensor, alpha: torch.Tensor, beta_plus_eps: torch.Tensor) -> torch.Tensor:
    """What the fused kernel must reproduce, in float64."""
    a = alpha.reshape(1, 1, -1).double()
    b = beta_plus_eps.reshape(1, 1, -1).double()
    xd = x.double()
    return xd + (1.0 / b) * torch.sin(a * xd) ** 2


def _selftest() -> None:
    torch.manual_seed(0)
    for c in (8, 16, 32, 168, 512):
        alpha = torch.rand(c) + 0.5
        beta = torch.rand(c) + 0.5
        params = snake_param_tiles(alpha, beta)
        n_tiles = (c + TILE - 1) // TILE
        assert params.shape == (2, n_tiles, TILE, TILE), params.shape
        # every row of a tile carries the same per-channel value
        flat_a = params[0].permute(0, 2, 1).reshape(-1, TILE)[:, 0][:c]
        assert torch.allclose(flat_a, alpha, atol=0), "alpha not replicated down tile rows"
        flat_b = params[1].permute(0, 2, 1).reshape(-1, TILE)[:, 0][:c]
        assert torch.allclose(flat_b, 1.0 / beta, atol=1e-7), "inv_beta wrong"
        # padded lanes must be inert, not NaN
        if c % TILE:
            assert torch.isfinite(params).all(), "padding produced non-finite values"
            assert params[0].reshape(-1)[-1] == 0.0 and params[1].reshape(-1)[-1] == 0.0

        # the reference the kernel is measured against
        x = torch.randn(2, 64, c) * 0.3
        ref = snake_reference(x, alpha, beta)
        naive = (
            x.double()
            + (1.0 / beta.reshape(1, 1, -1).double()) * torch.sin(alpha.reshape(1, 1, -1).double() * x.double()) ** 2
        )
        assert torch.equal(ref, naive)
        print(f"  C={c:<4} tiles={n_tiles:<3} ok")


if __name__ == "__main__":
    print("snake_param_tiles self-test")
    _selftest()
    print("all shapes and values verified on CPU")
