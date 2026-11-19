# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic q32/k32 bounded-prefill experiment; not a readiness gate or policy selection."""

import hashlib
import importlib.util
import json
import sys
import traceback
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch


def attention_chunks(original, kind, length, start):
    if kind != "full_attention":
        yield from original(kind, length, start)
        return
    if length <= 0 or start < 0:
        raise ValueError("Prefill needs a positive length and nonnegative start")
    offset = 0
    while offset < length:
        absolute, remaining = start + offset, length - offset
        rows = min(256, remaining // 32 * 32) if absolute % 32 == 0 and remaining >= 32 else 1
        assert rows == 1 or (absolute % 32 == 0 and rows % 32 == 0)
        yield offset, rows, absolute
        offset += rows


def main():
    here = Path(__file__).resolve()
    base_path = here.with_name("datatype_resume_variants.py")
    spec = importlib.util.spec_from_file_location("datatype_resume_variants_base", base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    args = base.parse_args()
    if not set(args.resolved_case_plan).issubset({"grouped_singleton", "no_sr_grouped_singleton"}):
        raise ValueError(
            "This experiment requires grouped singleton with optional no-SR; baseline parity is inapplicable"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = args.output_dir / "attention_q32_k32_diagnostic.json"
    assert not sidecar.exists() and not (args.output_dir / "results.json").exists(), "Refusing to replace evidence"

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt import prefill

    repo = here.parents[4]
    sources = [
        here,
        base_path,
        Path(prefill.__file__),
        base.ROOT / "tt/prefill_transport.py",
        base.ROOT / "tt/multichip_decoder.py",
        repo / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp",
        repo / "ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp",
        repo / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp",
        repo / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp",
    ]
    report = dict(
        status="running",
        command=sys.argv,
        case_plan=args.resolved_case_plan,
        diagnostic_only=True,
        gate_modified=False,
        production_runtime_changed=False,
        scope="q32/k32 SDPA only inside BoundedPrefill._attention, including warmup; normal decode unchanged",
        planner="Full attention: singleton until absolute32 alignment, then <=256 real rows in multiples of32; no padding",
        legality=dict(
            chunked="sdpa_device_operation.cpp:389-403 requires start divisible by q and k; both are32 here",
            decode="sdpa_decode.cpp requires power-of-two/multiple32; paged validator:455-459 permits32",
            cache="Existing32-token pages and existing _positions/page_window; no intra-page bulk fills",
            hardware_verified=False,
        ),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        counters=dict(attention_scopes=0, chunked_overrides=0, decode_overrides=0, fill_checks=0, outside_unchanged=0),
        native_calls=[],
        cache_fills=[],
    )

    def save():
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def tensor_meta(tensor):
        return dict(
            shape=list(tensor.shape),
            padded_shape=list(tensor.padded_shape),
            dtype=str(tensor.dtype),
            layout=str(tensor.layout),
            memory_config=str(tensor.memory_config()),
        )

    scope = ContextVar("diagnostic_bounded_attention", default=None)
    original_attention = prefill.BoundedPrefill._attention
    original_chunks = prefill.attention_chunks
    original_fill = ttnn.experimental.paged_fill_cache
    original_page_window = prefill.PrefillTransport.page_window

    def scoped_attention(executor, layer, state, source, offset, rows, absolute, public_length):
        assert scope.get() is None, "Unexpected nested attention scope"
        context = dict(
            executor=executor,
            state=state,
            kind=layer.kind,
            layer_idx=layer.layer_idx,
            offset=offset,
            rows=rows,
            absolute=absolute,
            public_length=public_length,
            preparing=not executor.prepared,
            native_calls=0,
            fills=0,
            page_windows=0,
        )
        if layer.kind == "full_attention":
            assert state["k"].shape[2] == state["v"].shape[2] == 32
            assert absolute + rows <= state["context_length"]
            assert rows == 1 or (absolute % 32 == 0 and rows % 32 == 0 and rows <= 256)
        token = scope.set(context)
        report["counters"]["attention_scopes"] += 1
        try:
            result = original_attention(executor, layer, state, source, offset, rows, absolute, public_length)
            assert context["native_calls"] == int(layer.kind == "full_attention")
            assert context["fills"] == (2 if layer.kind == "full_attention" and rows > 1 else 0)
            assert context["page_windows"] == int(layer.kind == "full_attention" and rows > 1)
            return result
        finally:
            scope.reset(token)
            if report["counters"]["attention_scopes"] % 128 == 0:
                save()

    def native_wrapper(original, operation):
        def wrapped(*operands, **kwargs):
            context = scope.get()
            if context is None:
                report["counters"]["outside_unchanged"] += 1
                return original(*operands, **kwargs)
            assert context["kind"] == "full_attention"
            executor, state = context["executor"], context["state"]
            assert operands[1] is state["k"] and operands[2] is state["v"]
            old = kwargs["program_config"]
            assert old.q_chunk_size == 32 and old.k_chunk_size == 256
            if operation == "chunked":
                assert context["rows"] > 1 and context["absolute"] % 32 == 0
                assert operands[3] is state["page_table"]
                assert kwargs.get("chunk_start_idx_tensor") is executor.position
                assert "chunk_start_idx" not in kwargs
                position_argument = "chunk_start_idx_tensor"
            else:
                assert context["rows"] == 1
                assert kwargs["page_table_tensor"] is state["page_table"]
                assert kwargs["cur_pos_tensor"] is executor.position
                position_argument = "cur_pos_tensor"
            kwargs["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=32
            )
            report["counters"][operation + "_overrides"] += 1
            context["native_calls"] += 1
            report["native_calls"].append(
                dict(
                    operation=operation,
                    **{
                        key: context[key]
                        for key in ("layer_idx", "offset", "rows", "absolute", "public_length", "preparing")
                    },
                    position_argument=position_argument,
                    position_value_source="_attention absolute -> existing _positions host-to-device refresh; no device readback",
                    q=tensor_meta(operands[0]),
                    k=tensor_meta(operands[1]),
                    v=tensor_meta(operands[2]),
                    page_table=tensor_meta(state["page_table"]),
                    program_config=dict(grid=[8, 8], q_chunk_size=32, k_chunk_size=32),
                    original_program_config=dict(q_chunk_size=old.q_chunk_size, k_chunk_size=old.k_chunk_size),
                    scale=kwargs.get("scale"),
                    compute_kernel_config=str(kwargs.get("compute_kernel_config")),
                    memory_config=str(kwargs.get("memory_config", "native default")),
                    kwarg_names=sorted(kwargs),
                )
            )
            return original(*operands, **kwargs)

        return wrapped

    def checked_page_window(transport, source, destination, first_page):
        context = scope.get()
        if context is not None:
            assert context["kind"] == "full_attention" and context["rows"] > 1
            assert source is context["state"]["page_table"]
            assert destination is context["executor"].fill_pages
            assert first_page * 32 == context["absolute"]
            context["page_windows"] += 1
        return original_page_window(transport, source, destination, first_page)

    def checked_fill(destination, source, pages, **kwargs):
        context = scope.get()
        if context is not None:
            assert context["kind"] == "full_attention" and context["rows"] > 1
            assert context["absolute"] % 32 == 0
            assert destination is context["state"]["k"] or destination is context["state"]["v"]
            assert destination.shape[2] == 32 and source.shape[2] == context["rows"]
            assert pages is context["executor"].fill_pages
            assert source.dtype == destination.dtype and kwargs == {"batch_idx": 0}
            context["fills"] += 1
            report["counters"]["fill_checks"] += 1
            report["cache_fills"].append(
                dict(
                    layer_idx=context["layer_idx"],
                    absolute=context["absolute"],
                    rows=context["rows"],
                    first_page=context["absolute"] // 32,
                    page_count=context["rows"] // 32,
                    destination="k" if destination is context["state"]["k"] else "v",
                    source=tensor_meta(source),
                    pages=tensor_meta(pages),
                    kwargs=kwargs,
                )
            )
        return original_fill(destination, source, pages, **kwargs)

    save()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(prefill.BoundedPrefill, "_attention", scoped_attention))
            stack.enter_context(
                patch.object(prefill, "attention_chunks", lambda *a: attention_chunks(original_chunks, *a))
            )
            stack.enter_context(patch.object(ttnn.experimental, "paged_fill_cache", checked_fill))
            stack.enter_context(patch.object(prefill.PrefillTransport, "page_window", checked_page_window))
            for name, operation in (
                ("chunked_scaled_dot_product_attention", "chunked"),
                ("paged_scaled_dot_product_attention_decode", "decode"),
            ):
                stack.enter_context(
                    patch.object(ttnn.transformer, name, native_wrapper(getattr(ttnn.transformer, name), operation))
                )
            base.main()
        assert report["counters"]["chunked_overrides"] > 0 and report["counters"]["decode_overrides"] > 0
        assert report["counters"]["fill_checks"] == 2 * report["counters"]["chunked_overrides"]
        report["status"] = "complete"
        report["legality"]["hardware_verified"] = True
    except BaseException:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
