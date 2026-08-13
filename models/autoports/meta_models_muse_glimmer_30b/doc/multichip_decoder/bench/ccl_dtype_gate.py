# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the BFP8 **decode** collective payload against the bar it must clear.

The prefill collective reduces in BFP8 and the decode collective does not, and
the reason is an inherited number: the single-chip stage's worst real-weight
check is a *decode* step at 0.995079 against the functional bar of 0.995, i.e.
7.9e-5 of headroom, while the measured BFP8 decode cost is ~1.8e-4.

``$optimize`` OPT-012 forbids rejecting a faster reduced-precision candidate on
synthetic-weight evidence, so this runs the candidate on the **released
checkpoint**, over the same eight-step decode surface the single-chip stage's
worst case came from, and reports whether it clears 0.995 -- rather than
inferring it from a single 4097-token A/B.

    python .../bench/ccl_dtype_gate.py --out logs/real_weight_ccl_dtype_gate.log
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_MESH_SHAPE,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.common.utility_functions import comp_pcc

PAGE_BLOCK = 64
MAX_SEQ = 16384
BAR = 0.995


def replicated(mesh, tensor, *, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def first_device(tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def run(mesh, kind: str, layer_idx: int, state_dict, layer, *, decode_ccl_dtype, prompt_len: int, steps: int):
    decoder = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=R.hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=MAX_SEQ,
        page_block_size=PAGE_BLOCK,
        decode_ccl_dtype=decode_ccl_dtype,
    )
    blocks = (MAX_SEQ + PAGE_BLOCK - 1) // PAGE_BLOCK
    rows = torch.randperm(blocks, generator=torch.Generator().manual_seed(5150)).reshape(1, blocks).to(torch.int32)
    page_table = replicated(mesh, rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    hidden = R.synthetic_hidden_states(1, prompt_len, seed=90210)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    ttnn.deallocate(
        decoder.prefill_forward(
            replicated(mesh, hidden.reshape(1, 1, prompt_len, -1), dtype=ttnn.bfloat16),
            page_table=page_table,
            user_id=0,
        )
    )

    worst = (1.0, "")
    for step in range(steps):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=90300 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        positions = torch.tensor([position])
        current_pos = replicated(mesh, positions.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rope_pos_ids = replicated(
            mesh, positions.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tt_out = decoder.decode_forward(
            replicated(mesh, token.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        _, message = comp_pcc(expected.float(), first_device(tt_out).reshape(1, 1, -1).float(), BAR)
        value = float(str(message).strip().split(":")[-1])
        label = f"decode[{kind}] step={step} pos={position}"
        print(
            f"GATE payload={'BFP8' if decode_ccl_dtype is not None else 'BF16'} {label:34s} {value:.6f} "
            f"{'PASS' if value >= BAR else 'FAIL'}",
            flush=True,
        )
        if value < worst[0]:
            worst = (value, label)
        ttnn.deallocate(tt_out)
        ttnn.deallocate(current_pos)
        ttnn.deallocate(rope_pos_ids)
    ttnn.deallocate(page_table)
    del decoder
    return worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()

    mesh = open_multichip_mesh(tuple(int(v) for v in args.mesh.split("x")), trace_region_size=0)
    try:
        for kind in ("sliding", "full"):
            layer_idx = reference_layer_indices(R.hf_config())[kind]
            state_dict = R.real_state_dict(layer_idx)
            layer = R.reference_layer(layer_idx, state_dict)
            for payload in (None, ttnn.bfloat8_b):
                worst = run(
                    mesh,
                    kind,
                    layer_idx,
                    state_dict,
                    layer,
                    decode_ccl_dtype=payload,
                    prompt_len=args.prompt_len,
                    steps=args.steps,
                )
                name = "BFP8" if payload is not None else "BF16"
                print(
                    f"GATE-WORST payload={name} kind={kind:8s} worst={worst[0]:.6f} on {worst[1]} "
                    f"bar={BAR} margin={worst[0] - BAR:+.2e}",
                    flush=True,
                )
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
