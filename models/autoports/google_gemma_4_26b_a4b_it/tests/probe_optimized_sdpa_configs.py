# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate attention compute policies through the audited parity harness.

Decode overrides require an already selected NativePagedAttention backend.
Here --prefill-attention-fidelity applies only to attention, not expert prefill. All
compute configurations are constructed after decoder setup, before forward.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import NativePagedAttention, OptimizedDecoder


def compute_description(config):
    if config is None:
        return None
    return {
        "math_fidelity": str(config.math_fidelity),
        "math_approx_mode": config.math_approx_mode,
        "fp32_dest_acc_en": config.fp32_dest_acc_en,
        "packer_l1_acc": config.packer_l1_acc,
        "dst_full_sync_en": config.dst_full_sync_en,
    }


def program_description(config):
    if config is None:
        return None
    grid = config.compute_with_storage_grid_size
    return {
        "compute_with_storage_grid_size": [grid.x, grid.y],
        "q_chunk_size": config.q_chunk_size,
        "k_chunk_size": config.k_chunk_size,
        "exp_approx_mode": config.exp_approx_mode,
        "max_cores_per_head_batch": config.max_cores_per_head_batch,
        "sub_core_grids": str(config.sub_core_grids),
    }


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--decode-attention-fidelity", choices=("HiFi4", "HiFi2", "LoFi"))
    parser.add_argument("--decode-attention-grid", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--prefill-attention-fidelity", choices=("HiFi4", "HiFi2", "LoFi"))
    args, rest = parser.parse_known_args()
    if args.decode_attention_grid is not None and tuple(args.decode_attention_grid) not in ((8, 8), (11, 10), (8, 4)):
        parser.error("--decode-attention-grid must be 8 8, 11 10, or 8 4")
    output = Path(rest[rest.index("--output") + 1])
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {"prefill_calls": {}}
    prefill_compute = None

    def compute(mesh, fidelity):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=True,
        )

    def create(cls, *a, **kw):
        nonlocal prefill_compute
        decoder = factory(cls, *a, **kw)
        attention = decoder.layer.self_attn
        native = attention.decode_sdpa
        if args.decode_attention_fidelity or args.decode_attention_grid:
            if not isinstance(native, NativePagedAttention):
                raise ValueError("Decode SDPA overrides require native_sdpa=True in the decoder policy")
            if args.decode_attention_fidelity:
                native.compute = compute(kw["mesh_device"], args.decode_attention_fidelity)
            if args.decode_attention_grid:
                native.program.compute_with_storage_grid_size = tuple(args.decode_attention_grid)
        runtime["decode"] = {"backend": type(native).__name__}
        if isinstance(native, NativePagedAttention):
            runtime["decode"].update(
                compute=compute_description(native.compute), program=program_description(native.program)
            )
        if args.prefill_attention_fidelity:
            prefill_compute = compute(kw["mesh_device"], args.prefill_attention_fidelity)
            runtime["prefill_override"] = compute_description(prefill_compute)
        return decoder

    def prefill_wrapper(name, operation):
        def call(*a, **kw):
            if prefill_compute is None:
                raise RuntimeError("Prefill compute override must be initialized during decoder construction")
            kw["compute_kernel_config"] = prefill_compute
            if name not in runtime["prefill_calls"]:
                runtime["prefill_calls"][name] = {
                    "compute": compute_description(prefill_compute),
                    "program": program_description(kw.get("program_config")),
                }
            return operation(*a, **kw)

        return call

    original_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)))
            if args.prefill_attention_fidelity:
                for name in ("scaled_dot_product_attention", "chunked_scaled_dot_product_attention"):
                    original = getattr(ttnn.transformer, name)
                    stack.enter_context(patch.object(ttnn.transformer, name, prefill_wrapper(name, original)))
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = original_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["sdpa_config_probe"] = vars(args)
        report["sdpa_config_probe_runtime"] = runtime
        report["sdpa_config_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if failure is not None:
            report["sdpa_config_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
