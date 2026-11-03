# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run established validation/benchmark unchanged through the selected default constructor."""

import argparse
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import ttnn

from ..tt.precision import runtime_summary

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def describe(value):
    if isinstance(value, ttnn.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    return str(value)


@contextmanager
def observe_operations(rows):
    """Observe eager preparation only; no tensor readback, dispatch changes or timing instrumentation."""
    counts = Counter()
    operations = [(ttnn, name) for name in ("linear", "sparse_matmul", "rms_norm", "typecast", "add", "mul")] + [
        (ttnn.experimental, name)
        for name in (
            "minimal_matmul",
            "all_gather_async",
            "all_reduce_async",
            "reduce_scatter_minimal_async",
            "paged_fill_cache",
            "paged_fused_update_cache",
            "fast_reduce_nc",
            "deepseek_moe_fast_reduce_nc_fused",
        )
    ]
    operations += [
        (ttnn.transformer, "chunked_scaled_dot_product_attention"),
        (ttnn.transformer, "paged_scaled_dot_product_attention_decode"),
    ]
    with ExitStack() as stack:
        for namespace, name in operations:
            original = getattr(namespace, name)

            def observed(*args, _original=original, _name=name, **kwargs):
                frame = sys._getframe(1)
                caller = frame.f_code.co_filename
                output = _original(*args, **kwargs)
                if str(ROOT / "tt") in caller:
                    owner = frame.f_locals.get("self")
                    config = kwargs.get("compute_kernel_config")
                    row = dict(
                        operation=_name,
                        caller=str(Path(caller).relative_to(ROOT)),
                        function=frame.f_code.co_name,
                        line=frame.f_lineno,
                        layer=getattr(owner, "layer_idx", None),
                        role={k: frame.f_locals[k] for k in ("name", "role") if isinstance(frame.f_locals.get(k), str)},
                        inputs=[describe(v) for v in args if isinstance(v, ttnn.Tensor)],
                        tensor_kwargs={k: describe(v) for k, v in kwargs.items() if isinstance(v, ttnn.Tensor)},
                        requested_dtype=str(kwargs.get("dtype")),
                        math_fidelity=str(config.math_fidelity) if config is not None else None,
                        compute_flags=(
                            {
                                name: getattr(config, name)
                                for name in ("math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
                            }
                            if config is not None
                            else None
                        ),
                        output=describe(output),
                    )
                    counts[json.dumps(row, sort_keys=True)] += 1
                return output

            stack.enter_context(patch.object(namespace, name, observed))
        yield
    rows.extend(dict(**json.loads(key), count=count) for key, count in counts.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("performance", "validation", "prefill-trace", "contract"), required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    args.proof = args.proof.resolve()
    args.proof.parent.mkdir(parents=True, exist_ok=True)
    assert not os.environ.get("K2_PRECISION_CONFIG"), "Default-path evidence must not use a precision override"
    expected = json.loads((DOC / "selected_precision_config.json").read_text())
    if args.mode == "performance":
        from . import optimized_full_model_performance as target
    elif args.mode == "validation":
        from . import full_model_validation as target
    elif args.mode == "contract":
        from . import full_model_probe as target
    else:
        from . import optimized_prefill_trace as target
    build = target.build_generator
    proof = dict(
        status="running",
        command=shlex.join([sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:]]),
        hardware="4x Blackhole p300c",
        mesh=[1, 4],
        environment={
            name: os.environ.get(name)
            for name in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
        },
        construction="normal build_generator with no precision argument/environment override",
        operation_observation="First normal generator preparation only; timed trace replay is uninstrumented",
        selected_policy=expected,
        creation_source_sha256=hashlib.sha256(
            Path("ttnn/cpp/ttnn/operations/creation/creation.cpp").read_bytes()
        ).hexdigest(),
        runtime_library_sha256=hashlib.sha256(Path("build_Release/lib/_ttnncpp.so").read_bytes()).hexdigest(),
        operations=[],
    )

    def save():
        operations = json.dumps(proof["operations"], separators=(",", ":")).encode()
        archive = args.proof.with_suffix(".operations.json.gz")
        archive.write_bytes(gzip.compress(operations, mtime=0))
        compact = {k: v for k, v in proof.items() if k != "operations"}
        compact["operations"] = dict(
            artifact=str(archive.relative_to(ROOT)),
            rows=len(proof["operations"]),
            uncompressed_sha256=hashlib.sha256(operations).hexdigest(),
        )
        args.proof.write_text(json.dumps(compact, indent=2) + "\n")

    def observed_build(*a, **kw):
        assert "precision_config" not in kw
        gen = build(*a, **kw)
        assert gen.model.precision == expected
        proof["runtime"] = runtime_summary(gen.model)
        proof["cache_dtypes"] = (
            sorted({str(t.dtype) for pair in gen.kv_cache for t in pair}) if gen.kv_cache is not None else None
        )
        original = gen.prepare
        prepared = False

        def prepare(*a, **kw):
            nonlocal prepared
            if prepared:
                return original(*a, **kw)
            proof["cache_dtypes"] = sorted({str(t.dtype) for pair in gen.kv_cache for t in pair})
            sample = gen._sample

            def observed_sample(logits):
                proof["sampling_input_dtype"] = str(logits.dtype)
                assert logits.dtype == getattr(ttnn, expected["sampling_dtype"])
                result = sample(logits)
                proof["sampled_token_dtype"] = str(gen.tokens.dtype)
                return result

            with observe_operations(proof["operations"]), patch.object(gen, "_sample", observed_sample):
                result = original(*a, **kw)
            prepared = True
            save()
            return result

        gen.prepare = prepare
        save()
        return gen

    save()
    try:
        with (
            patch.object(target, "build_generator", observed_build),
            patch.object(sys, "argv", [target.__name__, *forwarded]),
        ):
            target.main()
    except Exception:
        proof["status"] = "runtime_fail"
        proof["error"] = traceback.format_exc()
        save()
        raise
    proof["status"] = "pass"
    save()


if __name__ == "__main__":
    main()
