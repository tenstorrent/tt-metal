# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host conversion examples; TTNN is optional for the rest of the package."""
import torch
import ttnn

from tt_bfp_quant import search_linear, search_packed, to_bf16_exact, validate_repacking


def prepare_linear_shards(weight_out_in, output_splits, bits=4):
    """Return host TT tensors; feed them into your existing per-device loader."""
    q, _ = search_linear(weight_out_in, bits=bits, output_splits=output_splits)
    validate_repacking(q, bits=bits, output_splits=output_splits, native=True)
    dtype = ttnn.bfloat4_b if bits == 4 else ttnn.bfloat8_b
    return [
        ttnn.from_torch(to_bf16_exact(shard).T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        for shard in q.split(output_splits, dim=0)
    ]


def prepare_already_packed_shard(final_weight_kn, bits=8):
    """Call AFTER the model's output concatenation/permutation/shard split."""
    q, info = search_packed(final_weight_kn, bits=bits)
    validate_repacking(q, bits=bits, layout="packed", native=True)
    dtype = ttnn.bfloat4_b if bits == 4 else ttnn.bfloat8_b
    return ttnn.from_torch(to_bf16_exact(q), dtype=dtype, layout=ttnn.TILE_LAYOUT), info


if __name__ == "__main__":
    torch.manual_seed(1)
    host_shards = prepare_linear_shards(torch.randn(80, 65), [40, 40])
    print("Validated and packed", len(host_shards), "host shards; no device opened")
