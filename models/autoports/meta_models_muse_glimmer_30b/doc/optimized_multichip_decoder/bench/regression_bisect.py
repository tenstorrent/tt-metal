# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which optimized knob breaks ``full``, batch 4, 12345 tokens?

``test_multichip_matches_single_chip[12345-4-full]`` dropped to PCC 0.727 on the
first decode step against the single-chip baseline, while the same shape on
``sliding`` and the same layer at batch 1 both stayed at 0.9998.  This reproduces
that case on the 1x4 mesh alone -- no 1x1 mesh, no HF reference -- and scores
every optimized knob against the pre-stage configuration, so the culprit is a
column in a table rather than a guess.

    python .../bench/regression_bisect.py --kinds full,sliding --batch 4 --seq-len 12345
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.doc.multichip_decoder.bench.layer_ab import (
    host,
    page_table,
    pcc,
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

PAGE_BLOCK = 64
MAX_SEQ = 16384
DECODE_STEPS = 4

#: Pre-stage configuration first; then one knob at a time on top of it.
CONFIGS = {
    "before (wrapper, dram io)": dict(
        decode_ccl_impl="wrapper", prefill_ccl_impl="wrapper", ccl_persistent_buffers=False, sharded_decode_io=False
    ),
    "+sharded_io": dict(
        decode_ccl_impl="wrapper", prefill_ccl_impl="wrapper", ccl_persistent_buffers=False, sharded_decode_io=True
    ),
    "+decode async (no persist)": dict(
        decode_ccl_impl="async", prefill_ccl_impl="wrapper", ccl_persistent_buffers=False, sharded_decode_io=False
    ),
    "+decode async +persist": dict(
        decode_ccl_impl="async", prefill_ccl_impl="wrapper", ccl_persistent_buffers=True, sharded_decode_io=False
    ),
    "+prefill async (no persist)": dict(
        decode_ccl_impl="wrapper", prefill_ccl_impl="async", ccl_persistent_buffers=False, sharded_decode_io=False
    ),
    "+prefill async +persist": dict(
        decode_ccl_impl="wrapper", prefill_ccl_impl="async", ccl_persistent_buffers=True, sharded_decode_io=False
    ),
    "default (all on)": {},
    # The case the fix has to cover: a caller that never prefills through this
    # layer, so the decode-shape persistent buffers are first used by decode
    # step 0 itself.  vLLM and any generator with a separate prefill path look
    # like this.
}

#: Configurations that must not run prefill before decode.  These are scored
#: against each other rather than against the prefilled base: with no prefill the
#: KV cache is empty, so the decode output legitimately differs from every
#: prefilled arm.  What the pair pins is that persistent buffers change nothing
#: for a decode-only caller.
DECODE_ONLY = {
    "decode-only, no persist (control)": dict(
        decode_ccl_impl="async", prefill_ccl_impl="wrapper", ccl_persistent_buffers=False, sharded_decode_io=False
    ),
    "decode-only, persist": dict(
        decode_ccl_impl="async", prefill_ccl_impl="wrapper", ccl_persistent_buffers=True, sharded_decode_io=False
    ),
}


def run(mesh, kind, layer_idx, state_dict, cfg, args, prefill=True):
    dec = MultichipDecoder.from_state_dict(
        state_dict,
        hf_config=R.hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_batch_size=args.batch,
        max_seq_len=MAX_SEQ,
        page_block_size=PAGE_BLOCK,
        prefill_chunk_size=8192,
        **cfg,
    )
    pt = page_table(mesh, args.batch, MAX_SEQ, seed=1717)
    out = {}
    hidden = R.synthetic_hidden_states(1, args.seq_len, seed=4242 + args.seq_len)
    for user in range(0 if prefill else args.batch, args.batch):
        tt_out = dec.prefill_forward(to_dev(mesh, hidden), page_table=pt, user_id=user)
        if user == 0:
            out["prefill"] = host(tt_out).reshape(1, args.seq_len, -1).clone()
        ttnn.deallocate(tt_out)
    token = R.synthetic_hidden_states(1, args.batch, seed=4343).reshape(1, args.batch, -1)
    tt_token = to_dev(mesh, token)
    for step in range(DECODE_STEPS):
        position = torch.tensor([args.seq_len + step + u for u in range(args.batch)])
        cur = ttnn.from_torch(
            position.to(torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        rope = ttnn.from_torch(
            position.reshape(1, -1).to(torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        dec_out = dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope)
        out[f"decode{step}"] = host(dec_out).reshape(1, args.batch, -1).clone()
        for t in (dec_out, cur, rope):
            ttnn.deallocate(t)
    for t in (tt_token, pt):
        ttnn.deallocate(t)
    del dec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="full,sliding")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=12345)
    args = ap.parse_args()

    mesh = open_multichip_mesh((1, 4), trace_region_size=0, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.synthetic_state_dict(layer_idx)
            base = None
            for name, cfg in list(CONFIGS.items()) + list(DECODE_ONLY.items()):
                try:
                    result = run(mesh, kind, layer_idx, state_dict, cfg, args, prefill=name not in DECODE_ONLY)
                except Exception as exc:  # noqa: BLE001
                    print(f"BISECT {kind:8s} {name:30s} FAILED {str(exc).splitlines()[0][:200]}", flush=True)
                    continue
                if base is None or (name in DECODE_ONLY and "decode-only base" not in globals()):
                    if name in DECODE_ONLY:
                        globals()["decode-only base"] = result
                    else:
                        base = result
                against = globals().get("decode-only base", base) if name in DECODE_ONLY else base
                scores = {k: pcc(against[k], result[k]) for k in against if k in result}
                worst = min(scores, key=scores.get)
                print(
                    f"BISECT {kind:8s} {name:30s} worst={scores[worst]:.6f} on {worst:9s} "
                    f"| " + "  ".join(f"{k}={v:.5f}" for k, v in scores.items()),
                    flush=True,
                )
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
