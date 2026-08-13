# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bring-up smoke for the multichip decoder: build, prefill, decode, PCC vs HF.

Deliberately standalone (no pytest) so a failure prints the first broken op with
its shapes instead of a collection error.  Usage::

    python .../bench/smoke.py --kind sliding --seq-len 2049
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_MESH_SHAPE,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.common.utility_functions import comp_pcc

PAGE_BLOCK_SIZE = 64


def page_table(mesh, batch: int, max_seq_len: int, *, seed: int = 7) -> ttnn.Tensor:
    blocks = (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
    permutation = torch.randperm(batch * blocks, generator=torch.Generator().manual_seed(seed))
    return ttnn.from_torch(
        permutation.reshape(batch, blocks).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def hidden_to_mesh(mesh, hidden: torch.Tensor) -> ttnn.Tensor:
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def positions_to_mesh(mesh, positions: torch.Tensor):
    current_pos = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rope_pos_ids = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return current_pos, rope_pos_ids


def first_device(tensor: ttnn.Tensor) -> torch.Tensor:
    """Read device 0's copy of a replicated tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="sliding", choices=("sliding", "full"))
    parser.add_argument("--seq-len", type=int, default=2049)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()
    mesh_shape = tuple(int(v) for v in args.mesh.split("x"))

    layer_idx = R.reference_layer_indices(R.hf_config())[args.kind] if hasattr(R, "reference_layer_indices") else None
    if layer_idx is None:
        from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices

        layer_idx = reference_layer_indices(R.hf_config())[args.kind]
    state_dict = R.synthetic_state_dict(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)

    mesh = open_multichip_mesh(mesh_shape, trace_region_size=0)
    print(
        f"MESH {mesh.shape} devices={mesh.get_num_devices()} grid={mesh.compute_with_storage_grid_size()} "
        f"dram_grid={mesh.dram_grid_size()}"
    )
    try:
        decoder = MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
        )
        plan = decoder.plan
        print(
            f"PLAN tp={plan.tp} local_heads={plan.local_heads} local_kv={plan.local_kv_heads} "
            f"kv_replicated={plan.kv_replicated} local_qkv_width={plan.local_qkv_width} "
            f"local_intermediate={plan.local_intermediate}"
        )
        for name, tensor in (
            ("wqkv", decoder.wqkv),
            ("attn_gate", decoder.w_attn_gate),
            ("o_proj", decoder.wo),
            ("mlp_gate", decoder.mlp.gate),
            ("mlp_down", decoder.mlp.down),
            ("k_cache", decoder.k_cache),
        ):
            print(f"WEIGHT {name:10s} per-device shape={tuple(tensor.shape)} dtype={tensor.dtype}")

        pt = page_table(mesh, 1, args.max_seq_len)
        hidden = R.synthetic_hidden_states(1, args.seq_len, seed=101 + args.seq_len)
        expected, cache = R.reference_prefill(layer, layer_idx, hidden)
        tt_out = decoder.prefill_forward(hidden_to_mesh(mesh, hidden), page_table=pt, user_id=0)
        actual = first_device(tt_out).reshape(1, args.seq_len, -1)
        passed, message = comp_pcc(expected.float(), actual.float(), 0.99)
        print(f"PREFILL[{args.kind}] seq_len={args.seq_len} {message} -> {'PASS' if passed else 'FAIL'}")
        # Every device must return the same replicated residual.
        for device in range(1, mesh.get_num_devices()):
            other = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[device]).reshape(1, args.seq_len, -1)
            same = torch.equal(actual, other)
            print(f"PREFILL replica device={device} bit-identical={same}")
        ttnn.deallocate(tt_out)

        token = R.synthetic_hidden_states(1, 1, seed=999)
        expected_decode = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([args.seq_len])
        )
        current_pos, rope_pos_ids = positions_to_mesh(mesh, torch.tensor([args.seq_len]))
        tt_dec = decoder.decode_forward(
            hidden_to_mesh(mesh, token), current_pos=current_pos, page_table=pt, rope_pos_ids=rope_pos_ids
        )
        actual_decode = first_device(tt_dec).reshape(1, 1, -1)
        passed, message = comp_pcc(expected_decode.float(), actual_decode.float(), 0.99)
        print(f"DECODE[{args.kind}] pos={args.seq_len} {message} -> {'PASS' if passed else 'FAIL'}")
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
