# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One measured real-weight iteration for Stage 02 tt-perf-report artifacts."""

from __future__ import annotations

import argparse

import torch
from tracy import signpost

import models.autoports.qwen_qwen3_32b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    EMITTED_PREFILL_SEQUENCE,
    REPRESENTATIVE_LAYER,
)
from models.autoports.qwen_qwen3_32b.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("prefill", "decode"), required=True)
    args = parser.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64_000_000)
    try:
        config = functional_test._config()
        decoder = FusedDecoder.from_state_dict(
            functional_test._real_state(),
            hf_config=config,
            layer_idx=REPRESENTATIVE_LAYER,
            mesh_device=mesh,
        )
        generator = torch.Generator().manual_seed(5501)
        prefill_input = functional_test._tt_tensor(
            torch.randn(
                (1, EMITTED_BATCH, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ),
            mesh,
        )
        decode_input = functional_test._tt_tensor(
            torch.randn(
                (1, EMITTED_BATCH, 1, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ),
            mesh,
        )
        key_cache, value_cache = functional_test._empty_caches(config, mesh)
        warm = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        if args.path == "prefill":
            signpost("FUSED_PREFILL")
            output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
            ttnn.synchronize_device(mesh)
            signpost("FUSED_PREFILL_END")
            output.deallocate(True)
            return

        warm = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        try:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            signpost("FUSED_TRACED_DECODE")
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost("FUSED_TRACED_DECODE_END")
        finally:
            ttnn.release_trace(mesh, trace_id)
        output.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
