# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight fusion controls using the existing decoder parity harnesses.

This diagnostic patches only the chosen boundary in a loaded functional layer.
It saves bounded device comparisons, reads them before the harness closes its
device, and leaves the harness's HF, trace and runtime-audit checks in place.
Extra comparison operations make its timings unsuitable for performance claims.
"""

import argparse
import importlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.precision_ops import rms_norm, rotary
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        choices=("norm_fp32", "norm_external_weight", "rope_native_fp32", "rope_hf_fp32", "sdpa_exact"),
    )
    parser.add_argument("--norm-site", choices=("all", "input", "post", "head", "router"), default="all")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--runner", choices=("run_decoder", "request_reuse", "batched"), default="run_decoder")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.candidate == "sdpa_exact" and args.phase == "prefill":
        parser.error("Prefill already uses native SDPA; sdpa_exact replaces decode attention")
    if any(flag in remaining for flag in ("--profile", "--timing", "--decoder")):
        parser.error("This numerical probe requires the functional decoder and does not measure performance")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    harness_output = args.output.with_name(args.output.stem + ".harness.json")
    if harness_output.exists():
        parser.error(f"Refusing to reuse existing harness evidence: {harness_output}")
    report = dict(
        candidate=args.candidate,
        norm_site=args.norm_site,
        phase=args.phase,
        layer=args.layer,
        runner=args.runner,
        real_weights=True,
        threshold=0.995,
        same_input=[],
        harness_output=str(harness_output),
        performance_evidence=False,
    )
    pending = {}
    observed = set()
    factory = FunctionalDecoder.from_state_dict.__func__
    close_device = ttnn.close_mesh_device
    old_argv = sys.argv

    def capture(name, reference, actual):
        # Each boundary/phase is sampled once. Clones survive downstream releases.
        if name not in observed:
            observed.add(name)
            pending[name] = (ttnn.clone(reference), ttnn.clone(actual))

    def close(mesh):
        try:
            for name, (reference, actual) in pending.items():
                expected, got = (ttnn.to_torch(t).float() for t in (reference, actual))
                passed, pcc = comp_pcc(expected, got, report["threshold"])
                error = got - expected
                row = dict(
                    boundary=name,
                    shape=list(expected.shape),
                    pcc=float(pcc),
                    max_abs_error=float(error.abs().max()),
                    relative_l2=float(
                        torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-30)
                    ),
                    passed=bool(passed),
                )
                report["same_input"].append(row)
                print(row, flush=True)
            pending.clear()
        finally:
            close_device(mesh)

    with ExitStack() as stack:

        def load(cls, *a, **kw):
            layer = factory(cls, *a, **kw)
            attention = layer.layer.self_attn
            cfg = attention.config
            compute = attention.compute

            def norm(value, epsilon, weight=None, *, site):
                if site == "decoder":
                    site = "input" if weight is layer.input_norm_weight else "post"
                if args.norm_site not in ("all", site):
                    return rms_norm(value, epsilon, weight)
                phase = "decode" if (value.shape[1] == 1 if site == "head" else value.shape[-2] == 1) else "prefill"
                if args.phase not in ("all", phase):
                    return rms_norm(value, epsilon, weight)
                label = site
                if site == "head":
                    label += "_q" if weight is attention.q_weight else "_k" if weight is attention.k_weight else "_v"
                name = f"{label}_{phase}"
                value = ttnn.typecast(value, ttnn.float32)
                fused_weight = weight if args.candidate == "norm_fp32" else None
                result = ttnn.rms_norm(value, epsilon=epsilon, weight=fused_weight, compute_kernel_config=compute)
                if args.candidate == "norm_external_weight" and weight is not None:
                    result = ttnn.mul(result, weight)
                if name not in observed:
                    capture(name, rms_norm(value, epsilon, weight), result)
                return result

            def rope(value, cos, sin, *, decode=False):
                if args.phase not in ("all", "decode" if decode else "prefill"):
                    return rotary(value, cos, sin, decode=decode)
                # Use prefill geometry even for decode: heads occupy sequence
                # rows, and their selected position is explicitly repeated.
                # This avoids the separate sharded decode HF kernel.
                if decode:
                    repeats = (1, 1, value.shape[-2], 1)
                    selected = tuple(ttnn.repeat(t[:, :, :1, :], repeats) for t in (cos, sin))
                else:
                    selected = tuple(t[:, :, : value.shape[-2], :] for t in (cos, sin))
                selected = tuple(ttnn.typecast(t, ttnn.float32) for t in selected)
                interleaved = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                operation = (
                    ttnn.experimental.rotary_embedding_hf
                    if args.candidate == "rope_hf_fp32"
                    else ttnn.experimental.rotary_embedding
                )
                result = operation(interleaved, *selected, compute_kernel_config=compute)
                if result.shape != value.shape:
                    result = ttnn.reshape(result, value.shape, value.padded_shape)
                name = f"rope_{'decode' if decode else 'prefill'}_heads{value.shape[2] if decode else value.shape[1]}"
                if name not in observed:
                    capture(name, rotary(value, cos, sin, decode=decode), result)
                return result

            if args.candidate.startswith("norm_"):
                for suffix, site in (
                    ("functional_decoder", "decoder"),
                    ("decode_attention", "head"),
                    ("routing_precision", "router"),
                ):
                    module = importlib.import_module(f"models.autoports.google_gemma_4_26b_a4b_it.tt.{suffix}")

                    def replacement(value, epsilon, weight=None, site=site):
                        return norm(value, epsilon, weight, site=site)

                    stack.enter_context(patch.object(module, "rms_norm", replacement))
            elif args.candidate.startswith("rope_"):
                module = importlib.import_module("models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention")
                stack.enter_context(patch.object(module, "rotary", rope))
            else:
                precise = attention.decode_sdpa
                device_grid = attention.mesh_device.compute_with_storage_grid_size()
                grid = ttnn.CoreCoord(min(8, device_grid.x), min(4 if cfg.head_dim >= 512 else 8, device_grid.y))
                program = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False
                )

                def sdpa(q, k, v, *, cur_pos_tensor, page_table_tensor, **unused):
                    rounded_q = ttnn.to_memory_config(ttnn.typecast(q, ttnn.bfloat16), ttnn.DRAM_MEMORY_CONFIG)
                    result = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        rounded_q,
                        k,
                        v,
                        cur_pos_tensor=cur_pos_tensor,
                        page_table_tensor=page_table_tensor,
                        scale=1.0,
                        sliding_window_size=cfg.sliding_window if cfg.is_sliding else None,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=program,
                        compute_kernel_config=compute,
                    )
                    result = ttnn.typecast(result, ttnn.float32)
                    if "sdpa_decode" not in observed:
                        positions = dict(cur_pos_tensor=cur_pos_tensor, page_table_tensor=page_table_tensor)
                        baseline = precise(q, k, v, **positions)
                        rounded_baseline = precise(rounded_q, k, v, **positions)
                        capture("sdpa_decode", baseline, result)
                        capture("sdpa_q_rounding", baseline, rounded_baseline)
                        capture("sdpa_same_bf16_q", rounded_baseline, result)
                    return result

                attention.decode_sdpa = sdpa
            return layer

        stack.enter_context(patch.object(FunctionalDecoder, "from_state_dict", classmethod(load)))
        stack.enter_context(patch.object(ttnn, "close_mesh_device", close))
        runner = importlib.import_module(f"models.autoports.google_gemma_4_26b_a4b_it.tests.{args.runner}")
        sys.argv = [sys.argv[0], "--layer", str(args.layer), "--output", str(harness_output), *remaining]
        if args.runner == "run_decoder":
            sys.argv.extend(("--real", "--decode"))
        try:
            runner.main()
            report["integrated_harness_passed"] = True
            if not report["same_input"]:
                raise AssertionError("No candidate boundary was exercised")
            assert all(row["passed"] for row in report["same_input"]), report["same_input"]
            report["passed"] = True
        except BaseException as error:
            report["passed"] = False
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            sys.argv = old_argv
            if harness_output.exists():
                report["harness"] = json.loads(harness_output.read_text())
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
