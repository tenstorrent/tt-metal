# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context execution with a sampled, exact HF layer oracle.

All K/V positions participate. Only the compared query rows are sampled to
avoid a quadratic full-context HF attention matrix; model shapes, weights,
and the TT prefill invocation are not reduced. Sampling is reported explicitly.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_long_reference import (
    load_input_fixture,
    load_reference,
    sampled_reference,
)
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--input-fixture", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--decoder", choices=("functional", "fused"), default="functional")
    parser.add_argument("--fusion")
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()
    decoder_options = {}
    if args.decoder == "fused":
        decoder_options = {
            k: v for k, v in {"fusion": args.fusion, "group_size": args.group_size}.items() if v is not None
        }
    decoder_class = FunctionalDecoder
    if args.decoder == "fused":
        from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder

        decoder_class = FusedDecoder
    torch.manual_seed(123)
    torch.set_num_threads(args.threads)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    config._attn_implementation = "sdpa"
    hf = load_layer(config, args.layer, True)
    kind = config.layer_types[args.layer]
    length = args.length
    input_fixture = None
    if args.input_fixture:
        x, input_fixture = load_input_fixture(args.input_fixture, config, args.layer, length)
    else:
        x = torch.randn(1, length, config.hidden_size).bfloat16().float()
    rope = Gemma4TextRotaryEmbedding(config)
    extent = (length + 1023) // 1024 * 1024
    cos, sin = rope(x, torch.arange(extent)[None], layer_type=kind)
    if args.reference_file:
        ref, samples = load_reference(args.reference_file, args.layer, length, input_fixture)
    else:
        ref, samples = sampled_reference(hf, x, config, args.layer, cos, sin)
    print("HF_SAMPLED_REFERENCE_READY", len(samples), flush=True)
    if args.reference_only:
        torch.save(
            {
                "reference": ref,
                "samples": samples,
                "length": length,
                "layer": args.layer,
                "revision": REVISION,
                "input_fixture": input_fixture,
            },
            args.output,
        )
        return
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = decoder_class.from_state_dict(
            hf.state_dict(), hf_config=config, layer_idx=args.layer, mesh_device=mesh, **decoder_options
        )

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        block = 32
        pages = (extent + block - 1) // block
        table = torch.randperm(pages).int()[None]
        pt = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cfg = decoder.layer.self_attn.config
        cache = [
            device(
                torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim),
                getattr(decoder, "kv_cache_dtype", ttnn.bfloat16),
            )
            for _ in range(2)
        ]
        xt = device(x[None])
        rt = tuple(device(t[None]) for t in (cos, sin))
        print("TT_PREFILL_START", length, flush=True)
        with device_only():
            out = decoder.prefill_forward(xt, rope_mats=rt, page_table=pt, kv_cache=cache)
        actual = ttnn.to_torch(out).squeeze(0).float()[:, samples]
        passed, pcc = comp_pcc(ref, actual, 0.995)
        result = dict(
            layer_type=kind,
            length=length,
            phase="prefill",
            real_weights=True,
            input_fixture=input_fixture,
            pcc=float(pcc),
            passed=bool(passed),
            compared_query_rows=samples,
            scope="subset",
            cache_positions=length,
            runtime_prefill_audit="passed",
        )
        # Per-row diagnostics distinguish a length-growing attention error from
        # a few activation-sensitive rows without saving model tensors.
        row_checks = []
        for index, position in enumerate(samples):
            ok, value = comp_pcc(ref[:, index : index + 1], actual[:, index : index + 1], 0.995)
            row_checks.append(dict(position=position, pcc=float(value), passed=bool(ok)))
        result["sampled_row_diagnostics"] = row_checks
        # Regression for long-prefill endpoint drift hidden by aggregate PCC.
        # These are the same final positions compared individually by decode.
        tail_positions = sorted({max(0, length - 2), length - 1})
        tail_checks = [row_checks[samples.index(position)] for position in tail_positions]
        result["prefill_aggregate_passed"] = bool(passed)
        result["prefill_tail_checks"] = tail_checks
        passed = passed and all(check["passed"] for check in tail_checks)
        if input_fixture is not None:
            # Recorded text must pass each sampled row; aggregate PCC can hide
            # activation-sensitive expert projection errors.
            sampled_passed = all(check["passed"] for check in row_checks)
            result["prefill_sampled_rows_passed"] = sampled_passed
            passed = passed and sampled_passed
        result["passed"] = bool(passed)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        dt = device(x[:, -1:][None])
        decode_rope_layout = getattr(decoder, "decode_rope_layout", ttnn.TILE_LAYOUT)
        dr = tuple(device(t.squeeze(0), layout=decode_rope_layout) for t in (cos, sin))
        p = torch.zeros(1, 32, dtype=torch.int32)
        p[0, 0] = length - 1
        cp = torch.tensor([length - 1], dtype=torch.int32)
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def decode():
            with device_only():
                return decoder.decode_forward(
                    dt, rope_mats=dr, current_pos=rp, cache_pos=dp, page_table=pt, kv_cache=cache
                )

        warm = decode()
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        decode_results = []
        for position in [length - 1, max(0, length - 2), length - 1]:
            p[0, 0], cp[0] = position, position
            for tensor, dest, dtype, layout in [
                (x[:, position : position + 1][None], dt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (p, rp, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (cp, dp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            ]:
                host = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
                ttnn.copy_host_to_device_tensor(host, dest)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(traced).squeeze(0).float()
            oracle = ref[:, samples.index(position) : samples.index(position) + 1]
            ok, value = comp_pcc(oracle, got, 0.995)
            decode_results.append(dict(position=position, pcc=float(value), passed=bool(ok)))
        ttnn.release_trace(mesh, trace)
        result["decode"] = decode_results
        result["runtime_decode_audit"] = "passed"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("DECODE", decode_results, flush=True)
        assert passed and all(r["passed"] for r in decode_results), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
