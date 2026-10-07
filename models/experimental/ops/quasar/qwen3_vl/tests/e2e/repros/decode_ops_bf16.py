# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: replay captured decode-op cases in bf16 (bfp8 tensors converted), optionally through an experimental.quasar op.

usage: decode_ops_bf16.py paged_update_cache | sdpa_decode
"""
import importlib
import json
import sys

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

CASES = {
    "paged_update_cache": ("test_paged_update_cache", lambda: ttnn.experimental.paged_update_cache),
    "sdpa_decode": (
        "test_paged_scaled_dot_product_attention_decode",
        lambda: ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode,
    ),
}


def bf16(case):
    return json.loads(json.dumps(case).replace('"BFLOAT8_B"', '"BFLOAT16"').replace('"BFLOAT4_B"', '"BFLOAT16"'))


def main():
    module, op = CASES[sys.argv[1]]
    m = importlib.import_module(f"models.experimental.ops.quasar.tests.qwen3_vl_ops.{module}")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        for case in m.CASES:
            try:
                G.run_case(op(), bf16(case), mesh)
                print(f"{sys.argv[1]} {case['id']} PASS", flush=True)
            except Exception as e:  # report every case, including pytest skips raised by run_case
                lines = [ln.strip() for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("---")]
                print(f"{sys.argv[1]} {case['id']} {type(e).__name__}: {' | '.join(lines[:2])[:220]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
