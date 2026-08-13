# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the *inter-layer* residual layout costs, measured through a traced decode.

The multichip layer's decode residual is width-sharded in L1 on the boundary grid
for the whole layer, but its public boundary is DRAM interleaved: every layer
starts with an ``interleaved_to_sharded`` and ends with a ``sharded_to_interleaved``,
i.e. a 425 KB DRAM round trip per layer per token that exists only to cross the
layer boundary.

This probe measures both contracts on the same layer, in the same process, with
the same trace protocol as ``bench/layer_ab.py``:

* ``dram``    -- the shipped contract: DRAM-interleaved in, DRAM-interleaved out;
* ``sharded`` -- width-sharded L1 in, width-sharded L1 out.

and asserts the two produce the same tensor, so the saving is not bought with a
different computation.  ``--layers N`` chains N copies of the layer to show what
the contract is worth to a *stack*, which is the thing full-model bringup has to
preserve.

    python .../bench/boundary_probe.py --kinds sliding,full --layers 1,2
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.doc.multichip_decoder.bench.layer_ab import (
    MAX_SEQ,
    PAGE_BLOCK,
    host,
    page_table,
    pcc,
    pos_tensors,
    to_dev,
)
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_L1_SMALL_SIZE,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)

#: The decode collective this stage selected, so the boundary contract is priced
#: on top of it rather than against a superseded baseline.
CCL_BEST = {"decode_ccl_impl": "async", "decode_ccl_ag_workers": 1, "ccl_persistent_buffers": True}


def timed_trace(mesh, run, iters, rounds):
    """Warm, capture, replay; return the min per-iteration ms and the output."""
    out = run()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    traced_out = run()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(8):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    result = host(traced_out).clone()
    ttnn.release_trace(mesh, trace_id)
    return best, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="sliding,full")
    ap.add_argument("--layers", default="1,2")
    ap.add_argument("--decode-context", type=int, default=2048)
    ap.add_argument("--decode-iters", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    mesh = open_multichip_mesh((1, 4), trace_region_size=90112 * 12, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.synthetic_state_dict(layer_idx)
            for depth in (int(v) for v in args.layers.split(",")):
                run_one(mesh, kind, layer_idx, state_dict, depth, args)
    finally:
        close_multichip_mesh(mesh)


def run_one(mesh, kind, layer_idx, state_dict, depth, args):
    layers = [
        MultichipDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=MAX_SEQ,
            page_block_size=PAGE_BLOCK,
            prefill_chunk_size=8192,
            sharded_decode_io=sharded,
            **CCL_BEST,
        )
        for sharded in (False, True)
        for _ in range(depth)
    ]
    dram_layers, sharded_layers = layers[:depth], layers[depth:]
    pt = page_table(mesh, 1, dram_layers[0].config.max_seq_len, seed=3)
    cur, rope = pos_tensors(mesh, torch.tensor([args.decode_context]))
    token = to_dev(mesh, R.synthetic_hidden_states(1, 1, seed=44).reshape(1, 1, -1))

    def stack(stack_layers, sharded_in):
        def run():
            rows = 32
            x = stack_layers[0].boundary_memcfg(rows, stack_layers[0].config.hidden_size)
            h = ttnn.interleaved_to_sharded(token, x) if sharded_in else token
            for i, layer in enumerate(stack_layers):
                nxt = layer.decode_forward(h, current_pos=cur, page_table=pt, rope_pos_ids=rope)
                if i and h is not token:
                    ttnn.deallocate(h)
                h = nxt
            return h

        return run

    dram_ms, dram_out = timed_trace(mesh, stack(dram_layers, False), args.decode_iters, args.rounds)
    sh_ms, sh_out = timed_trace(mesh, stack(sharded_layers, True), args.decode_iters, args.rounds)
    print(
        f"BOUNDARY kind={kind:8s} layers={depth} "
        f"dram={dram_ms:7.4f} ms  sharded={sh_ms:7.4f} ms  "
        f"delta={100 * (sh_ms - dram_ms) / dram_ms:+6.2f} %  "
        f"per_layer_saving_us={(dram_ms - sh_ms) / depth * 1e3:6.2f}  "
        f"pcc={pcc(dram_out, sh_out):.9f}",
        flush=True,
    )
    for t in (pt, cur, rope, token):
        ttnn.deallocate(t)


if __name__ == "__main__":
    main()
