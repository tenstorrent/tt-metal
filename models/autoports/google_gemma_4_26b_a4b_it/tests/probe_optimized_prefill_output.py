# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-only output GEMM geometry/fidelity controls with unchanged decode."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention import concat_heads
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


class PrefillOutputProbe:
    def __init__(self, attention, mesh, max_rows, args, runtime):
        self.original = attention.project
        self.attention = attention
        self.args = args
        self.runtime = runtime
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, args.prefill_output_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=False,
        )
        self.programs = {}
        weight = attention.source.weights.o_proj
        k, n = weight.shape[-2], weight.shape[-1]
        gx, gy = args.prefill_output_grid
        available = mesh.compute_with_storage_grid_size()
        if not (0 < gx <= available.x and 0 < gy <= available.y):
            raise ValueError("Prefill output grid must fit the device")
        block = args.prefill_output_block
        if block <= 0 or k // 32 % block:
            raise ValueError("Prefill output K block must divide tiled K")
        if args.prefill_output_mode == "explicit":
            for rows in range(32, max_rows + 1, 32):
                per_m, per_n = math.ceil(rows / 32 / gy), math.ceil(n / 32 / gx)
                candidates = [(h, w) for h in (1, 2, 4) for w in (1, 2, 4) if h * w <= 4]
                h, w = max(
                    ((h, w) for h, w in candidates if per_m % h == 0 and per_n % w == 0),
                    key=lambda pair: (pair[0] * pair[1], pair[1]),
                )
                self.programs[rows] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=block,
                    out_subblock_h=h,
                    out_subblock_w=w,
                    out_block_h=per_m,
                    out_block_w=per_n,
                    per_core_M=per_m,
                    per_core_N=per_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
        runtime.update(
            mode=args.prefill_output_mode,
            decode="unchanged production project()",
            attention="unchanged production SDPA/cache path",
            weight_shape=list(weight.shape),
            weight_dtype=str(weight.dtype),
            weight_memory=str(weight.memory_config()),
            output_dtype="float32",
            output_memory="DRAM interleaved",
            compute=dict(fidelity=args.prefill_output_fidelity, fp32_dest_acc_en=True, packer_l1_acc=False),
            programs={str(rows): str(program) for rows, program in self.programs.items()},
        )

    def __call__(self, attention, decode):
        if decode:
            return self.original(attention, decode)
        source = self.attention.source
        cfg = source.config
        combined = concat_heads(
            attention,
            is_decode_mode=False,
            num_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            mesh_device=source.mesh_device,
        )
        if self.args.prefill_output_input_memory == "l1":
            combined = ttnn.to_memory_config(combined, ttnn.L1_MEMORY_CONFIG)
        rows = combined.padded_shape[-2]
        if self.args.prefill_output_mode == "explicit" and rows not in self.programs:
            raise ValueError(f"No setup-time prefill output program for {rows} rows")
        output = ttnn.linear(
            combined,
            source.weights.o_proj,
            dtype=ttnn.float32,
            program_config=self.programs.get(rows),
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if "observed_prefill" not in self.runtime:
            self.runtime["observed_prefill"] = dict(
                input_shape=list(combined.shape),
                input_dtype=str(combined.dtype),
                input_memory=str(combined.memory_config()),
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
                selected_program=str(self.programs.get(rows)),
            )
        return output


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prefill-output-mode", choices=("explicit", "auto"), default="explicit")
    parser.add_argument("--prefill-output-grid", type=int, nargs=2, default=(11, 8))
    parser.add_argument("--prefill-output-block", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--prefill-output-fidelity", choices=("HiFi4", "HiFi2", "LoFi"), default="HiFi4")
    parser.add_argument("--prefill-output-input-memory", choices=("inherit", "l1"), default="inherit")
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prefill output evidence: {output}")
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        decoder.layer.self_attn.project = PrefillOutputProbe(
            decoder.layer.self_attn, kw["mesh_device"], decoder.chunk_size, args, runtime
        )
        decoder.precision_policy["prefill_output_probe"] = runtime
        return decoder

    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = previous_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["prefill_output_probe"] = vars(args)
        report["prefill_output_runtime"] = runtime
        report["prefill_output_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["runtime_sha256"] = hashlib.sha256(
            Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
        ).hexdigest()
        if failure is not None:
            report["prefill_output_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
