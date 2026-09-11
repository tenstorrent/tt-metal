# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated native chunk GDN recurrence experiment using the real-weight probe."""

import json
import os
import sys
from pathlib import Path

import probe
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles


def native_recurrence(
    query, key, value, beta, decay, *, initial_state, groups, sequence, value_dim, batch, value_heads
):
    if os.environ.get("SCAN_CAPTURE_INPUTS") and len(native_recurrence.captured) < int(
        os.environ.get("SCAN_CAPTURE_CHUNKS", "128")
    ):
        native_recurrence.captured.append([probe.host_tensor(t) for t in (query, key, value, beta, decay)])

    def token_heads(t):
        t = ttnn.reshape(t, (batch, value_heads, sequence, value_dim))
        return ttnn.permute(t, (0, 2, 1, 3))

    def gates(t):
        t = ttnn.reshape(t, (batch, value_heads, sequence))
        return ttnn.permute(ttnn.typecast(t, ttnn.float32), (0, 2, 1))

    g = ttnn.log(gates(decay))
    out, state = ttnn.transformer.chunk_gated_delta_rule(
        token_heads(query),
        token_heads(key),
        token_heads(value),
        g,
        gates(beta),
        scale=1.0,
        initial_state=ttnn.typecast(initial_state, ttnn.float32),
        output_final_state=True,
        chunk_size=32,
        use_qk_l2norm=False,
        output_head_major=True,
        eye=native_recurrence.constants[0],
        tril=native_recurrence.constants[1],
        ones=native_recurrence.constants[2],
        masks=native_recurrence.constants[3],
    )
    return ttnn.reshape(out, (groups, sequence, 1, value_dim)), state


original_configure = probe.configure
original_build = probe.build_generator


def configure(candidate):
    original_configure(candidate)
    probe.md._sequential_recurrence = native_recurrence


def build(**kwargs):
    gen = original_build(**kwargs)
    native_recurrence.constants = build_fused_const_tiles(kwargs["mesh_device"])
    return gen


if __name__ == "__main__":
    native_recurrence.captured = []
    probe.configure = configure
    probe.build_generator = build
    probe.main()
    if native_recurrence.captured:
        torch.save(native_recurrence.captured, os.environ["SCAN_CAPTURE_INPUTS"])
    result_path = Path(sys.argv[sys.argv.index("--result") + 1])
    result = json.loads(result_path.read_text())
    result["candidate"] = "native_chunk"
    result["recurrence_implementation"] = "ttnn.transformer.chunk_gated_delta_rule, internal chunk32"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
