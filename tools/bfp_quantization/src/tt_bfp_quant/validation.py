# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Optional checks at the actual TTNN host-packing boundary."""
import torch

from .quantize import _matrix, _splits, search_packed


def validate_repacking(weight, bits=4, *, layout="linear", output_splits=None, native=False):
    """Require ordinary rounding to leave prepared values numerically unchanged.

    native=True also invokes the installed TTNN host packer (no device opened).
    It validates the specified standard tile layout, not model-specific fusion
    or sharding done later. For packed layout, pass one physical shard at a time.
    """
    if layout not in ("linear", "packed"):
        raise ValueError("layout must be linear or packed")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    value = _matrix(weight, "weight")
    if layout == "packed":
        if output_splits is not None:
            raise ValueError("packed input must already be one physical shard")
        shards = [value]
    else:
        shards = [s.T.contiguous() for s in value.split(_splits(value.shape[0], output_splits), dim=0)]
    if native:
        import ttnn
    for shard in shards:
        ordinary, _ = search_packed(shard, bits, (0,), backend="numpy")
        if not torch.equal(ordinary, shard):
            raise ValueError("Prepared values change under ordinary BFP repacking in this layout")
        if native:
            packed = ttnn.from_torch(
                shard, dtype=ttnn.bfloat4_b if bits == 4 else ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )
            actual = ttnn.to_torch(packed).float()
            if not torch.equal(actual, shard):
                raise ValueError("Installed TTNN host packer changes prepared values")
    return {
        "numerical_repacking_exact": True,
        "native_ttnn_checked": native,
        "shards": len(shards),
        "bits": bits,
        "layout": layout,
    }
