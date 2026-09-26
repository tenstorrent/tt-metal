# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-layer controls for cache precision and native paged SDPA."""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--cache-bfp8", action="store_true")
    p.add_argument("--native-sdpa", action="store_true")
    p.add_argument("--sdpa-full-sync", action="store_true")
    p.add_argument("--precise-bf16-query", action="store_true")
    p.add_argument("--precise-bf16-output", action="store_true")
    p.add_argument("--qkv-weight-dtype")
    p.add_argument("--output-weight-dtype")
    args, rest = p.parse_known_args()
    factory = OptimizedDecoder.from_state_dict.__func__
    from_torch = ttnn.from_torch
    fill = ttnn.experimental.paged_fill_cache

    def upload(value, *a, **kw):
        if args.cache_bfp8 and value.ndim == 4 and value.shape[0] > 1 and value.shape[2] == 32:
            kw["dtype"] = ttnn.bfloat8_b
        return from_torch(value, *a, **kw)

    def fill_cache(cache, value, *a, **kw):
        return fill(cache, ttnn.typecast(value, cache.dtype), *a, **kw)

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        attn = decoder.layer.self_attn
        source = attn.source
        if args.qkv_weight_dtype:
            projection = source.weights.wqkv
            if hasattr(projection, "weights"):
                projection.weights = tuple(
                    ttnn.typecast(weight, getattr(ttnn, args.qkv_weight_dtype)) for weight in projection.weights
                )
                projection = projection.source
            projection.weight = ttnn.typecast(projection.weight, getattr(ttnn, args.qkv_weight_dtype))
            projection.rows = ttnn.transpose(ttnn.typecast(projection.weight, ttnn.float32), -2, -1)
        if args.output_weight_dtype:
            source.weights = replace(
                source.weights, o_proj=ttnn.typecast(source.weights.o_proj, getattr(ttnn, args.output_weight_dtype))
            )
        cast = attn.cache_cast
        if args.precise_bf16_query:
            precise = attn.decode_sdpa

            def rounded_query(q, *a, **kw):
                return precise(ttnn.typecast(ttnn.typecast(q, ttnn.bfloat16), ttnn.float32), *a, **kw)

            attn.decode_sdpa = rounded_query
        if args.precise_bf16_output:
            precise_output = attn.decode_sdpa

            def rounded_output(*a, **kw):
                return ttnn.typecast(ttnn.typecast(precise_output(*a, **kw), ttnn.bfloat16), ttnn.float32)

            attn.decode_sdpa = rounded_output
        if args.cache_bfp8:
            attn.cache_cast = lambda value, dtype, memory: cast(value, ttnn.bfloat16, memory)
        if args.native_sdpa:
            cfg = attn.config
            compute = (
                ttnn.init_device_compute_kernel_config(
                    kw["mesh_device"].arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                    dst_full_sync_en=True,
                )
                if args.sdpa_full_sync
                else attn.compute
            )
            program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
            )

            def sdpa(q, k, v, *, cur_pos_tensor, page_table_tensor, **unused):
                return ttnn.typecast(
                    ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        ttnn.to_memory_config(ttnn.typecast(q, ttnn.bfloat16), ttnn.DRAM_MEMORY_CONFIG),
                        k,
                        v,
                        cur_pos_tensor=cur_pos_tensor,
                        page_table_tensor=page_table_tensor,
                        scale=1.0,
                        sliding_window_size=cfg.sliding_window if cfg.is_sliding else None,
                        program_config=program,
                        compute_kernel_config=compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ttnn.float32,
                )

            attn.decode_sdpa = sdpa
        return decoder

    sys.argv = [sys.argv[0], *rest]
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)))
            stack.enter_context(patch.object(ttnn, "from_torch", upload))
            stack.enter_context(patch.object(ttnn.experimental, "paged_fill_cache", fill_cache))
            run_optimized_decoder.main()
    finally:
        out = Path(rest[rest.index("--output") + 1])
        if out.exists():
            data = json.loads(out.read_text())
            data["attention_probe"] = vars(args)
            data["attention_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
