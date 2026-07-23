# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FIBO-owned conv3d blocking configs.

These were previously inlined into the shared ``utils/conv3d.py`` ``_BLOCKINGS`` /
``_DEFAULT_BLOCKINGS`` tables. They are registered into the shared tables at import of the FIBO
VAE via :func:`register_fibo_conv3d_configs`, so the shared file carries no FIBO data.
"""
from ...utils.conv3d import register_conv3d_configs

# Exact-match entries: (h_factor, w_factor, C_in, C_out, kernel, T, H, W) -> blocking.
_FIBO_EXACT_BLOCKINGS = {
    # BH 2x2, FIBO image decode (1024x1024, latent T=1, full-T uncached). h_factor=2, w_factor=2.
    (2, 2, 1024, 1024, (3, 3, 3), 3, 64, 64): (64, 256, 1, 4, 8),
    (2, 2, 1024, 1024, (3, 3, 3), 3, 128, 128): (64, 256, 1, 2, 16),
    (2, 2, 512, 512, (3, 3, 3), 3, 256, 256): (64, 256, 1, 8, 4),
    (2, 2, 256, 256, (3, 3, 3), 3, 512, 512): (64, 256, 1, 8, 4),
    (2, 2, 64, 1024, (3, 3, 3), 3, 64, 64): (32, 512, 1, 8, 4),
    (2, 2, 1024, 512, (3, 3, 3), 3, 256, 256): (64, 256, 1, 8, 4),
    (2, 2, 512, 256, (3, 3, 3), 3, 512, 512): (64, 256, 1, 8, 4),
    (2, 2, 256, 12, (3, 3, 3), 3, 512, 512): (128, 32, 1, 16, 2),
    (2, 2, 1024, 1024, (1, 3, 3), 1, 128, 128): (256, 128, 1, 4, 8),
    (2, 2, 1024, 1024, (1, 3, 3), 1, 256, 256): (256, 128, 1, 4, 8),
    (2, 2, 512, 512, (1, 3, 3), 1, 512, 512): (256, 128, 1, 8, 4),
    (2, 2, 1024, 2048, (3, 1, 1), 2, 64, 64): (256, 512, 1, 2, 16),
    (2, 2, 1024, 2048, (3, 1, 1), 2, 128, 128): (256, 512, 1, 4, 8),
    # BH Galaxy 4x8, FIBO image decode. h_factor=8, w_factor=4.
    (8, 4, 1024, 1024, (3, 3, 3), 3, 16, 32): (64, 256, 1, 8, 4),
    (8, 4, 1024, 1024, (3, 3, 3), 3, 32, 64): (64, 256, 1, 8, 4),
    (8, 4, 512, 512, (3, 3, 3), 3, 64, 128): (64, 256, 1, 4, 8),
    (8, 4, 256, 256, (3, 3, 3), 3, 128, 256): (64, 256, 1, 16, 2),
    (8, 4, 64, 1024, (3, 3, 3), 3, 16, 32): (64, 256, 1, 8, 4),
    (8, 4, 1024, 512, (3, 3, 3), 3, 64, 128): (64, 256, 1, 8, 4),
    (8, 4, 512, 256, (3, 3, 3), 3, 128, 256): (64, 256, 1, 16, 2),
    (8, 4, 256, 12, (3, 3, 3), 3, 128, 256): (128, 32, 1, 16, 2),
    (8, 4, 1024, 1024, (1, 3, 3), 1, 32, 64): (256, 128, 1, 4, 8),
    (8, 4, 1024, 1024, (1, 3, 3), 1, 64, 128): (256, 128, 1, 2, 16),
    (8, 4, 512, 512, (1, 3, 3), 1, 128, 256): (128, 256, 1, 2, 16),
    (8, 4, 1024, 2048, (3, 1, 1), 2, 16, 32): (256, 256, 1, 2, 16),
    (8, 4, 1024, 2048, (3, 1, 1), 2, 32, 64): (256, 512, 1, 16, 2),
}

# Fallback channel-keyed defaults: (C_in, C_out, kernel) -> blocking.
_FIBO_DEFAULT_BLOCKINGS = {
    (64, 1024, (3, 3, 3)): (64, 32, 1, 1, 1),
    (1024, 1024, (3, 3, 3)): (256, 32, 1, 1, 1),
    (1024, 1024, (1, 3, 3)): (256, 32, 1, 1, 1),
    (1024, 2048, (3, 1, 1)): (256, 32, 1, 1, 1),
    (1024, 512, (3, 3, 3)): (256, 32, 1, 1, 1),
    (512, 512, (3, 3, 3)): (256, 32, 1, 1, 1),
    (512, 512, (1, 3, 3)): (256, 32, 1, 1, 1),
    (512, 256, (3, 3, 3)): (256, 32, 1, 1, 1),
    (256, 256, (3, 3, 3)): (256, 32, 1, 8, 8),
    (256, 12, (3, 3, 3)): (256, 32, 1, 8, 8),
}

_registered = False


def register_fibo_conv3d_configs() -> None:
    """Idempotently inject FIBO conv3d blockings into the shared conv3d tables."""
    global _registered
    if _registered:
        return
    register_conv3d_configs(_FIBO_EXACT_BLOCKINGS)
    register_conv3d_configs(_FIBO_DEFAULT_BLOCKINGS)
    _registered = True
