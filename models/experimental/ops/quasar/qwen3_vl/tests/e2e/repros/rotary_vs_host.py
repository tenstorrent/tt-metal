# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: rotary_embedding_llama on the captured cases vs the host rope fallback (certified against WH).

usage: rotary_vs_host.py [case-id prefixes, run in this order in one process]
"""
import sys

import ttnn
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.op_overrides import FALLBACKS
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.pcc import pcc
from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G
from models.experimental.ops.quasar.tests.qwen3_vl_ops.test_rotary_embedding_llama import CASES


def main():
    want = sys.argv[1:]
    ref_fn = FALLBACKS["ttnn.experimental.rotary_embedding_llama"].torch_fn
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        # Cases run in the order given (prefix match), so e.g. `02_ 00_` checks decode rope after prefill rope.
        order = [c for w in want for c in CASES if c["id"].startswith(w)] if want else CASES
        for case in order:
            try:
                args, kwargs, _ = G.build_inputs_for_case(case, mesh)
                host = [to_host(a) if isinstance(a, ttnn.Tensor) else a for a in args]
                out = ttnn.experimental.rotary_embedding_llama(*args, **kwargs)
                got, ref = to_host(out), ref_fn(host, kwargs).float()
                got = got[tuple(slice(0, s) for s in ref.shape)]
                print(
                    f"{case['id']:28s} pcc={pcc(got, ref):.5f} max_abs={(got - ref).abs().max().item():.4f}", flush=True
                )
            except Exception as e:
                print(f"{case['id']:28s} {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
