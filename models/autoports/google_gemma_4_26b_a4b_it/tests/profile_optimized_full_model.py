# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced full path profile: one real layer per kind, real terminal and feedback."""

import torch
from tracy import signpost

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=8192, layer_indices=(0, 5))
        prompt = [2] + [100] * 4095
        gen.generate(prompt, 3, stop_on_eos=False)
        # Release traces before eager prefill profiling to preserve allocation rules.
        gen._release_trace()
        ids = gen.model.upload(torch.tensor(prompt).reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL")
        logits = gen.model.prefill_forward(ids, page_table=gen.table, kv_cache=gen.cache)
        gen.sampler.sample(gen._sampler_logits(logits), tt_out_tok=gen.tokens, enable_trace=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        gen.generate(prompt, 3, stop_on_eos=False)
        gen._release_trace()
        gen._prepare_output_buffer(127)
        gen._capture()
        gen._capture_output_buffer()
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE")
        gen._replay()
        ttnn.execute_trace(mesh, gen.output_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
