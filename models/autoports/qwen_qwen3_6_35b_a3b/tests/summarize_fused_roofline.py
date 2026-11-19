# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile useful-byte roofline, device kernel time and warmed traced latency.

CPU-only accounting for batch-one, position-256 decode on one 8-bank Blackhole.
The useful-byte model is deliberately separate from implementation traffic:
charging repeated intermediates to the former would reward an inefficient graph.
This script neither measures DRAM traffic nor establishes the optimization stop rule.
"""

import ast
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = Path(os.environ.get("QWEN_FUSED_ARTIFACT_DIR", ROOT / "doc/fused_decoder"))
DRAM_BANDWIDTH_GB_S = 512


def read(path):
    return json.loads(path.read_text())


def convolution_history_allocation(source):
    """Resolve the known BF16 cache allocator, without executing decoder code.

    Payload includes standard TILE padding, not allocator/bank overhead. Unknown
    helper wiring, shape, dtype or layout requires a new accounting review.
    """
    tree = ast.parse(source)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    allocate = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "new_recurrent_state")
    helpers = [n for n in allocate.body if isinstance(n, ast.FunctionDef) and n.name == "zeros"]
    assert len(helpers) == 1, "Re-audit convolution cache allocation helper"
    helper = helpers[0]
    assert [a.arg for a in helper.args.args] == ["shape", "dtype", "layout"]
    assert len(helper.args.defaults) == 1 and ast.unparse(helper.args.defaults[0]) == "ttnn.TILE_LAYOUT"
    assert not helper.args.posonlyargs and not helper.args.kwonlyargs
    assert helper.args.vararg is None and helper.args.kwarg is None
    assert len(helper.body) == 1 and isinstance(helper.body[0], ast.Return)
    upload = helper.body[0].value
    assert isinstance(upload, ast.Call) and ast.unparse(upload.func) == "ttnn.from_torch"
    assert len(upload.args) == 1 and ast.unparse(upload.args[0]) == "torch.zeros(shape)"
    options = {kw.arg: ast.unparse(kw.value) for kw in upload.keywords}
    assert len(options) == len(upload.keywords) and options == {
        "device": "self.mesh_device",
        "dtype": "dtype",
        "layout": "layout",
        "memory_config": "DRAM",
    }, "Re-audit convolution cache upload options"
    dram = [
        n.value for n in tree.body if isinstance(n, ast.Assign) and any(ast.unparse(t) == "DRAM" for t in n.targets)
    ]
    assert len(dram) == 1 and ast.unparse(dram[0]) == "ttnn.DRAM_MEMORY_CONFIG"
    body = [n for n in allocate.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
    assert len(body) == 3 and isinstance(body[0], ast.Import) and ast.unparse(body[0]) == "import torch"
    assert body[1] is helper and isinstance(body[2], ast.Return) and isinstance(body[2].value, ast.Dict)
    entries = body[2].value
    values = [v for k, v in zip(entries.keys, entries.values) if isinstance(k, ast.Constant) and k.value == "conv"]
    assert len(values) == 1 and isinstance(values[0], ast.Call)
    call = values[0]
    assert ast.unparse(call.func) == "zeros" and len(call.args) == 3 and not call.keywords
    shape = ast.literal_eval(call.args[0])
    dtype, layout = ast.unparse(call.args[1]), ast.unparse(call.args[2])
    assert shape == (1, 3, 8192), "Re-audit convolution cache shape"
    assert dtype == "ttnn.bfloat16", "Re-audit convolution cache dtype"
    assert layout in ("ttnn.TILE_LAYOUT", "ttnn.ROW_MAJOR_LAYOUT"), "Re-audit convolution cache layout"
    padded = list(shape)
    if layout == "ttnn.TILE_LAYOUT":
        padded[-2:] = [(n + 31) // 32 * 32 for n in padded[-2:]]
    return dict(
        logical_shape=list(shape),
        padded_shape=padded,
        dtype="bfloat16",
        layout=layout.removeprefix("ttnn."),
        memory_config="DRAM_INTERLEAVED",
        allocation_bytes=math.prod(padded) * 2,
        allocation_line=call.lineno,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        scope="Tensor payload including layout padding; excludes allocator/bank overhead and conversion traffic",
    )


def decode_output_norm_read(source):
    """Resolve the consumed decode gamma, excluding the prefill-only norm path.

    Support only the audited explicit v4 and native true-head v5 tails. Unknown
    consumer wiring, upload dtype, gamma shape or memory placement fails closed.
    The charge is one logical gamma read, never internal buffers or padding.
    """
    tree = ast.parse(source)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    methods = {n.name: n for n in decoder.body if isinstance(n, ast.FunctionDef)}
    setup, attention = methods["from_state_dict"], methods["_linear_attention"]

    def assignments(scope, target):
        return [
            n for n in ast.walk(scope) if isinstance(n, ast.Assign) and any(ast.unparse(t) == target for t in n.targets)
        ]

    uploads = [n for n in setup.body if isinstance(n, ast.FunctionDef) and n.name == "upload"]
    assert len(uploads) == 1, "Re-audit output-norm gamma upload helper"
    helper = uploads[0]
    assert [a.arg for a in helper.args.args] == ["x", "dtype"]
    assert len(helper.args.defaults) == 1 and ast.unparse(helper.args.defaults[0]) == "ttnn.bfloat16"
    assert not helper.args.posonlyargs and not helper.args.kwonlyargs
    assert helper.args.vararg is None and helper.args.kwarg is None
    assert len(helper.body) == 1 and isinstance(helper.body[0], ast.Return)
    upload = helper.body[0].value
    assert isinstance(upload, ast.Call) and ast.unparse(upload.func) == "ttnn.from_torch"
    assert len(upload.args) == 1 and ast.unparse(upload.args[0]) == "x.contiguous()"
    options = {kw.arg: ast.unparse(kw.value) for kw in upload.keywords}
    assert len(options) == len(upload.keywords) and options == {
        "device": "mesh_device",
        "dtype": "dtype",
        "layout": "ttnn.TILE_LAYOUT",
        "memory_config": "DRAM",
    }, "Re-audit output-norm gamma upload options"
    dram = assignments(tree, "DRAM")
    assert len(dram) == 1 and ast.unparse(dram[0].value) == "ttnn.DRAM_MEMORY_CONFIG"
    expected = assignments(setup, "expected")
    assert len(expected) == 1 and isinstance(expected[0].value, ast.Call)
    assert ast.unparse(expected[0].value.func) == "dict" and not expected[0].value.args
    dimensions = {kw.arg: ast.literal_eval(kw.value) for kw in expected[0].value.keywords}
    assert dimensions["linear_value_head_dim"] == 128, "Re-audit logical gamma width"

    # Prefill returns early. Its native BF16 gamma is not a decode read in v4.
    branches = [
        (i, n)
        for i, n in enumerate(attention.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not decode"
        and any(isinstance(child, ast.Return) for child in n.body)
    ]
    assert len(branches) == 1 and not branches[0][1].orelse, "Re-audit prefill/decode norm split"
    tail = ast.Module(body=attention.body[branches[0][0] + 1 :], type_ignores=[])
    native_calls = [
        n
        for n in ast.walk(tail)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "ttnn.experimental.kda.sigmoid_gated_rms_norm"
    ]
    gamma_loads = [
        n
        for n in ast.walk(tail)
        if isinstance(n, (ast.Attribute, ast.Subscript))
        and isinstance(n.ctx, ast.Load)
        and ast.unparse(n) in ("self.output_norm_bf16", "self.w['linear_attn.norm.weight']")
    ]
    assert len(gamma_loads) == 1, "Re-audit number of consumed decode gamma operands"
    operand = ast.unparse(gamma_loads[0])
    assert not assignments(decoder, "self.w['linear_attn.norm.weight']"), "Re-audit direct gamma overwrite"
    if native_calls:
        assert len(native_calls) == 1 and operand == "self.output_norm_bf16"
        consumer = native_calls[0]
        assert [ast.unparse(a) for a in consumer.args] == ["head_first", "gate_rows", operand, "32"]
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in consumer.keywords}
        assert kwargs.get("gate_column_offset") == "2048"
        assert kwargs.get("output_dtype") in ("ttnn.bfloat16", "ttnn.float32")
        assert kwargs.get("epsilon") == "self.config.rms_norm_eps"
        gamma = assignments(decoder, operand)
        assert len(gamma) == 1
        call = gamma[0].value
        assert isinstance(call, ast.Call) and ast.unparse(call.func) == "upload"
        assert [ast.unparse(a) for a in call.args] == ["state_dict['linear_attn.norm.weight'].reshape(128)"]
        assert not call.keywords, "Re-audit native gamma dtype override"
        dtype, width, path = "bfloat16", 2, "native_sigmoid_gated_rms_norm"
    else:
        assert operand == "self.w['linear_attn.norm.weight']"
        weighted = assignments(tail, "weighted")
        assert len(weighted) == 1 and isinstance(weighted[0].value, ast.Call)
        consumer = weighted[0].value
        assert ast.unparse(consumer.func) == "ttnn.multiply"
        assert [ast.unparse(a) for a in consumer.args] == ["normalized", operand]
        norms = [n for n in ast.walk(tail) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ttnn.rms_norm"]
        assert len(norms) == 1 and len(norms[0].args) == 1 and all(kw.arg != "weight" for kw in norms[0].keywords)
        gamma = assignments(setup, "self.w[name]")
        gamma = [
            n
            for n in gamma
            if isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "upload"
            and [ast.unparse(a) for a in n.value.args] == ["v.reshape(1, 1, -1)", "ttnn.float32"]
        ]
        assert len(gamma) == 1 and not gamma[0].value.keywords, "Re-audit explicit FP32 gamma upload"
        norm_branch = [n for n in ast.walk(setup) if isinstance(n, ast.If) and gamma[0] in n.body]
        assert len(norm_branch) == 1
        assert ast.unparse(norm_branch[0].test) == "name.endswith('norm.weight') or name.endswith('layernorm.weight')"
        values = assignments(norm_branch[0], "v")
        assert len(values) == 1 and norm_branch[0].body == [values[0], gamma[0]], "Re-audit gamma setup branch"
        assert len(values) == 1 and ast.unparse(values[0].value) == (
            "value.float() if name == 'linear_attn.norm.weight' else value.float() + 1"
        )
        loops = [n for n in ast.walk(setup) if isinstance(n, ast.For) and norm_branch[0] in list(ast.walk(n))]
        assert len(loops) == 1 and ast.unparse(loops[0].target) == "(name, value)"
        assert ast.unparse(loops[0].iter) == "state_dict.items()"
        dtype, width, path = "float32", 4, "explicit_unweighted_rms_norm_then_gamma_multiply"
    return dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        path=path,
        operand=operand,
        dtype=dtype,
        logical_elements=128,
        logical_read_bytes=128 * width,
        weight_ledger_key="output_norm_bf16" if width == 2 else "output_norm_fp32",
        memory_config="DRAM_INTERLEAVED",
        consumer_line=consumer.lineno,
        upload_line=gamma[0].lineno,
        scope="One logical decode gamma read; excludes prefill-only copies, TILE padding and all intermediate traffic",
    )


def singleton_projection_placement(source):
    """Resolve the actual singleton multiply placement; fail closed on new wiring.

    This is a narrow source-contract check, not a Python dataflow interpreter.
    Inherited placement is accepted only for the known DRAM upload helper.
    """
    tree = ast.parse(source)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    linear = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "linear")
    branches = [
        n
        for n in linear.body
        if isinstance(n, ast.If)
        and "x.shape[-2] == 1" in ast.unparse(n.test)
        and "attention_packed" in ast.unparse(n.test)
    ]
    assert len(branches) == 1, "Re-audit singleton projection branch"
    products = [
        n.value
        for n in branches[0].body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "products" for t in n.targets)
    ]
    assert len(products) == 1 and isinstance(products[0], ast.Call)
    call = products[0]
    assert ast.unparse(call.func) == "ttnn.multiply", "Re-audit projection product producer"
    memory = next((kw.value for kw in call.keywords if kw.arg == "memory_config"), None)
    expression = ast.unparse(memory) if memory is not None else "inherited from weight"
    known = {
        "ttnn.L1_MEMORY_CONFIG": "L1_INTERLEAVED",
        "ttnn.DRAM_MEMORY_CONFIG": "DRAM_INTERLEAVED",
        "DRAM": "DRAM_INTERLEAVED",
    }
    if memory is None:
        setup = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == "from_state_dict")
        upload = next(n for n in ast.walk(setup) if isinstance(n, ast.FunctionDef) and n.name == "upload")
        upload_call = next(
            n for n in ast.walk(upload) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ttnn.from_torch"
        )
        upload_memory = next(kw.value for kw in upload_call.keywords if kw.arg == "memory_config")
        assert known.get(ast.unparse(upload_memory)) == "DRAM_INTERLEAVED"
        weight = next(
            n.value
            for n in branches[0].body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "weight" for t in n.targets)
        )
        assert isinstance(weight, ast.IfExp)
        assert {ast.unparse(weight.body), ast.unparse(weight.orelse)} == {
            "self.reduce_attention_weight",
            "self.reduce_prefill_attention_weight",
        }
        for attribute in ("reduce_attention_weight", "reduce_prefill_attention_weight"):
            assignment = next(
                n
                for n in ast.walk(setup)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == attribute for t in n.targets)
            )
            assert isinstance(assignment.value, ast.Call) and ast.unparse(assignment.value.func) == "upload"
        buffer = "DRAM_INTERLEAVED"
    else:
        assert expression in known, f"Re-audit product memory config: {expression}"
        buffer = known[expression]
    return dict(
        buffer=buffer,
        expression=expression,
        multiply_line=call.lineno,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def linear_decode_traffic_contract(source):
    """Fail closed if compact-table selection or fused cache-boundary wiring changes."""
    tree = ast.parse(source)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    methods = {n.name: n for n in decoder.body if isinstance(n, ast.FunctionDef)}
    setup, attention, recurrence = (
        methods[name] for name in ("from_state_dict", "_linear_attention", "_recurrent_step")
    )

    def assigned(scope, target):
        return [
            n for n in ast.walk(scope) if isinstance(n, ast.Assign) and any(ast.unparse(t) == target for t in n.targets)
        ]

    compact = assigned(setup, "compact")
    assert len(compact) == 1 and isinstance(compact[0].value, ast.Call)
    assert ast.unparse(compact[0].value.func) == "torch.zeros"
    compact_shape = ast.literal_eval(compact[0].value.args[0])
    assert compact_shape == (32, 16384), "Re-audit compact decay table shape"
    branch = [n for n in ast.walk(setup) if isinstance(n, ast.If) and any(c is compact[0] for c in n.body)]
    assert len(branch) == 1 and ast.unparse(branch[0].test) == "window + 127 <= 8192"
    upload = [n.value for n in assigned(setup, "self.compact_decay_table") if isinstance(n.value, ast.Call)]
    assert len(upload) == 1 and ast.unparse(upload[0].func) == "ttnn.from_torch"
    assert ast.unparse(upload[0].args[0]) == "compact"
    options = {kw.arg: ast.unparse(kw.value) for kw in upload[0].keywords}
    assert options["dtype"] == "ttnn.float32" and options["layout"] == "ttnn.ROW_MAJOR_LAYOUT"
    assert options["memory_config"] == "DRAM"
    original = assigned(setup, "(self.gate_table, self.decay_table)")
    assert len(original) == 1 and isinstance(original[0].value, ast.ListComp)
    assert ast.unparse(original[0].value.generators[0].iter) == "(gate_values, decay_values)"
    original_options = {kw.arg: ast.unparse(kw.value) for kw in original[0].value.elt.keywords}
    assert original_options["dtype"] == "ttnn.float32" and original_options["layout"] == "ttnn.ROW_MAJOR_LAYOUT"
    assert original_options["memory_config"] == "DRAM"
    code_aranges = [
        n
        for n in ast.walk(assigned(setup, "codes")[0])
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.arange"
    ]
    assert len(code_aranges) == 1 and ast.literal_eval(code_aranges[0].args[0]) == 65536
    chosen = [n for n in assigned(attention, "decay_value") if isinstance(n.value, ast.Call)]
    assert len(chosen) == 1 and ast.unparse(chosen[0].value) == (
        "lookup(self.compact_decay_table if self.compact_decay_table is not None else self.decay_table)"
    )
    lookup = next(n for n in ast.walk(attention) if isinstance(n, ast.FunctionDef) and n.name == "lookup")
    gathers = [n for n in ast.walk(lookup) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ttnn.gather"]
    assert (
        len(gathers) == 1 and ast.unparse(gathers[0].args[0]) == "table" and ast.literal_eval(gathers[0].args[1]) == 1
    )

    # The stage's layer0 CPU checkpoint proof establishes that the setup guard selects the compact path.
    proof_path = ROOT / "doc/fused_decoder/endurance/gate_lut_plateau_cpu.json"
    proof = read(proof_path)
    assert proof["layer"] == 0 and proof["revision"] == "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
    decay = proof["tables"]["decay"]
    assert decay["all_code_bit_mismatches"] == 0 and decay["all_code_elements"] == 32 * 65536
    assert decay["finite_window_per_sign"] + decay["nan_entries_per_sign"] <= 8192
    assert decay["padded_entries_per_head"] == compact_shape[1]

    activations = assigned(recurrence, "state_activations")
    assert (
        len(activations) == 1 and ast.unparse(activations[0].value) == "self.round_widen_chain if round_state else []"
    )
    assert not assigned(recurrence, "state"), "Re-audit materialized recurrent cache-boundary casts"
    state_uses = [n for n in ast.walk(recurrence) if isinstance(n, ast.Name) and n.id == "state"]
    decayed = assigned(recurrence, "decayed")
    assert len(state_uses) == len(decayed) == 2
    for statement in decayed:
        call = statement.value
        assert isinstance(call, ast.Call) and ast.unparse(call.func) == "ttnn.multiply"
        assert ast.unparse(call.args[0]) == "state"
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert kwargs["input_tensor_a_activations"] == "state_activations"
    initial = assigned(attention, "initial")
    assert len(initial) == 1 and ast.unparse(initial[0].value) == "state['recurrent']"
    chain = assigned(setup, "self.round_widen_chain")
    assert len(chain) == 1 and ast.unparse(chain[0].value) == "[round_op, widen_op]"
    return dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        compact_table_shape=list(compact_shape),
        compact_table_bytes=math.prod(compact_shape) * 4,
        retained_prefill_decay_table_bytes=32 * 65536 * 4,
        selection="Layer0 checkpoint proof fits setup guard; other layers can retain full-table fallback",
        table_proof_path=str(proof_path.relative_to(ROOT)),
        table_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        compact_upload_line=upload[0].lineno,
        decode_lookup_line=chosen[0].lineno,
        recurrence_boundary="BF16 round/widen fused into both state-decay multiply input activation paths",
        recurrence_external_boundary_cast_bytes=0,
        recurrence_state_decay_multiply_lines=[n.lineno for n in decayed],
        convolution_history_allocation=convolution_history_allocation(source),
        output_norm_read=decode_output_norm_read(source),
    )


def artifact_bytes(path):
    """Hash the original CSV bytes, including when evidence was archived."""
    if path.exists():
        return path.read_bytes()
    archive = path.with_suffix(path.suffix + ".gz")
    if archive.exists():
        return gzip.decompress(archive.read_bytes())
    parts = sorted(path.parent.glob(archive.name + ".part*"))
    assert parts, f"Missing evidence: {path}"
    return gzip.decompress(b"".join(p.read_bytes() for p in parts))


def verify_device_report(device, idx):
    """Bind the summary's latency to its actual CSV, not just nearby provenance."""
    expected = f"tracy/layer_{idx}/decode_perf_report.csv"
    assert device["report"] == expected
    assert device["provenance"] == f"tracy/layer_{idx}/correctness.json.provenance.json"
    raw = artifact_bytes(DOC / expected)
    assert hashlib.sha256(raw).hexdigest() == device["report_sha256"], "Stale or changed profiler CSV"
    assert device["measured_passes"] == 16 and device["length"] == 1
    timed = []
    for row in csv.DictReader(raw.decode().splitlines()):
        duration = float(row.get("Device Time [μs]") or 0)
        if duration > 0:
            assert math.isfinite(duration)
            assert float(row["Device"]) == 0
            assert float(row.get("Overlap [μs]") or 0) == 0, "Overlapping ops need timeline accounting"
            timed.append(duration)
    assert timed and len(timed) == device["timed_operation_rows"]
    assert math.isclose(sum(timed) / 16, device["latency_us"], rel_tol=1e-12)
    return expected, hashlib.sha256(raw).hexdigest()


def byte_model(kind, projection_placement=None, linear_traffic=None):
    """Logical once-per-token lower bound and a NON-exhaustive traffic ledger.

    Shapes/dtypes describe FusedDecoder's default projection_k=512 path. Resident
    prefill/fallback copies and all unselected experts are excluded from decode.
    All quantities are decimal bytes; logical elements are not tile allocations.
    """
    linear = kind == "linear_attention"
    if kind not in ("linear_attention", "full_attention"):
        raise ValueError(kind)
    weight_bytes = {
        "attention_projection_logical_bf16": 2048 * (12352 if linear else 9216) * 2,
        "attention_output_bf16": 4096 * 2048 * 2,
        "eight_unique_expert_gate_up_bf16": 8 * 2048 * 1024 * 2,
        "eight_unique_expert_down_bf16": 8 * 512 * 2048 * 2,
        "shared_gate_up_scalar_gate_logical_bf16": 2048 * (512 + 512 + 1) * 2,
        "shared_down_bf16": 512 * 2048 * 2,
        "router_fp32": 2048 * 256 * 4,
        "layer_norm_weights_fp32": 2 * 2048 * 4,
    }
    # This ledger is not summed into the roofline numerator. It contains both
    # full tensor sizes and incremental overhead, explicitly distinguished.
    execution = {
        "shared_projection_padding_extra_weight_bytes": 2048 * 31 * 2,
        "coverage": "Selected known costs only; not a total or measured DRAM traffic",
    }
    if linear:
        if projection_placement is None:
            projection_placement = singleton_projection_placement((ROOT / "tt/fused_decoder.py").read_text())
        if linear_traffic is None:
            linear_traffic = linear_decode_traffic_contract((ROOT / "tt/fused_decoder.py").read_text())
        weight_bytes.update(
            convolution_taps_fp32=4 * 8192 * 4,
            selected_decay_lookup_entries_fp32=32 * 4,
        )
        gamma = linear_traffic["output_norm_read"]
        weight_bytes[gamma["weight_ledger_key"]] = gamma["logical_read_bytes"]
        cache_bytes = {
            "recurrent_state_read_write_fp32": 2 * 32 * 128 * 128 * 4,
            "convolution_history_read_write_logical_bf16": 2 * 3 * 8192 * 2,
        }
        # K/V/A/B remain ordered; Q/Z are a separate native BF16 matmul.
        # 6208 + 6144 == 12352 useful columns. Padding belongs only to the
        # execution ledger, so splitting the graph cannot inflate useful bytes.
        ordered_logical_columns, ordered_padded_columns, native_columns = 6208, 7168, 6144
        projection_elements = 2048 * ordered_padded_columns
        execution.update(
            ordered_projection_logical_columns=ordered_logical_columns,
            ordered_projection_padded_columns=ordered_padded_columns,
            native_qz_projection_columns=native_columns,
            ordered_projection_padded_bf16_weight_bytes=projection_elements * 2,
            ordered_projection_padding_extra_weight_bytes=2048 * 960 * 2,
            native_qz_projection_logical_bf16_weight_bytes=2048 * native_columns * 2,
            projection_fp32_product_tensor_bytes=projection_elements * 4,
            projection_product_one_write_one_read_bytes=2 * projection_elements * 4,
            projection_product_placement=projection_placement,
            projection_product_l1_write_read_bytes=(
                2 * projection_elements * 4 if projection_placement["buffer"] == "L1_INTERLEAVED" else 0
            ),
            projection_product_dram_write_read_bytes=(
                2 * projection_elements * 4 if projection_placement["buffer"] == "DRAM_INTERLEAVED" else 0
            ),
            projection_product_traffic_assumption=(
                "Only the ordered K/V/A/B branch materializes this FP32 product tensor: "
                "one multiply write and one fast_reduce_nc read in the AST-derived product buffer. "
                "L1 placement routes those bytes through L1/NoC, not DRAM; native Q/Z has no such tensor. "
                "Excludes broadcast rereads, partial reductions, slicing and block additions; "
                "these are operation-derived bytes, not measured DRAM traffic."
            ),
            projection_partial_fp32_logical_bytes=4 * ordered_padded_columns * 4,
            linear_decode_traffic_contract=linear_traffic,
            decay_lut_resident_fp32_bytes=(
                linear_traffic["compact_table_bytes"] + linear_traffic["retained_prefill_decay_table_bytes"]
            ),
            decay_lut_selected_decode_table_fp32_bytes=linear_traffic["compact_table_bytes"],
            decay_lut_retained_prefill_table_fp32_bytes=linear_traffic["retained_prefill_decay_table_bytes"],
            decay_lut_selected_logical_bytes=32 * 4,
            decay_lut_native_full_row_decode_read_bytes=linear_traffic["compact_table_bytes"],
            decay_lut_traffic_note=(
                "Only 32 selected FP32 values (128 bytes) enter the useful-byte numerator. "
                "Current native RM gather stages one complete compact row per head: 2 MiB total table reads, "
                "derived from kernel behavior, not measured DRAM transactions. The original 8 MiB decay "
                "table remains resident for prefill/fallback and is not read by selected layer0 decode. "
                "The separate original gate_table is prefill-only."
            ),
            convolution_history_allocation_bytes=linear_traffic["convolution_history_allocation"]["allocation_bytes"],
            convolution_history_layout=linear_traffic["convolution_history_allocation"]["layout"],
            convolution_history_dtype=linear_traffic["convolution_history_allocation"]["dtype"],
            convolution_history_logical_shape=linear_traffic["convolution_history_allocation"]["logical_shape"],
            convolution_history_padded_shape=linear_traffic["convolution_history_allocation"]["padded_shape"],
            convolution_taps_tile_allocation_bytes=4 * 32 * 8192 * 4,
            recurrence_cache_boundary_cast_logical_read_write_bytes=(
                linear_traffic["recurrence_external_boundary_cast_bytes"]
            ),
            recurrence_note=(
                "The FP32→BF16→FP32 public-cache boundary is fused into the state-decay multiply's "
                "input activation chain. It materializes no standalone cast tensors, so the former 6 MiB "
                "external cast read/write charge is zero. On-core activation buffers/work are not zero. "
                "A direct recurrent-cache writer can also move the subsequent Q-contraction state read "
                "from L1 to DRAM while removing the final copy; no synthetic traffic total is inferred here. "
                "Other recurrence multiply/reduce/update intermediates remain outside the useful-byte bound."
            ),
            excluded_resident_copies=(
                "BF16 ordered prefill projection copy; interleaved native Q/Z copy when decode uses DRAM shards; "
                "prefill gate_table; grouped prefill expert banks; unselected experts"
            ),
            projection_resident_copy_note="There is no FP32 precise-attention-weight copy in this implementation.",
        )
    else:
        weight_bytes["qk_norm_weights_fp32"] = 2 * 256 * 4
        cache_bytes = {
            "logical_kv_reads_bf16": (256 + 1) * 2 * 2 * 256 * 2,
            "one_token_kv_writes_bf16": 2 * 2 * 256 * 2,
        }
        execution.update(
            sdpa_chunk_rounded_kv_span_bytes=512 * 2 * 2 * 256 * 2,
            kv_allocation_bytes=512 * 2 * 2 * 256 * 2,
            kv_note=(
                "257 live positions, K chunk 256, allocated context 512. The chunk-rounded "
                "span is not proof that every byte is fetched once; SDPA head/core reuse "
                "and masked reads require kernel/profiler evidence."
            ),
        )
    return weight_bytes, cache_bytes, execution


def main():
    source = (ROOT / "tt/fused_decoder.py").read_text()
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    placement = singleton_projection_placement(source)
    linear_traffic = linear_decode_traffic_contract(source)
    profile_path = DOC / "perf_summary.json"
    profile = read(profile_path)["measurements"]
    evidence_hashes = {"perf_summary.json": hashlib.sha256(profile_path.read_bytes()).hexdigest()}
    rows = []
    for idx, kind in ((0, "linear_attention"), (3, "full_attention")):
        host = read(DOC / f"final_{idx}.json")
        device = next(r for r in profile if r["layer"] == idx and r["phase"] == "decode")
        for path in (DOC / f"final_{idx}.json", DOC / f"tracy/layer_{idx}/correctness.json"):
            provenance_path = Path(str(path) + ".provenance.json")
            provenance = read(provenance_path)
            assert provenance["source_sha256"]["tt/fused_decoder.py"] == source_sha
            assert provenance["regime"] == "serial"
            assert not provenance["env"].get("TT_METAL_WATCHER")
            for evidence in (path, provenance_path):
                evidence_hashes[str(evidence.relative_to(DOC))] = hashlib.sha256(evidence.read_bytes()).hexdigest()
        report, digest = verify_device_report(device, idx)
        evidence_hashes[report] = digest
        assert host["length"] == device["decode_position"] == 256
        assert host["context"] == device["allocated_context"] == 512
        assert not host["latency_tracking_enabled"]
        weight_bytes, cache_bytes, execution = byte_model(kind, placement, linear_traffic)
        useful_bytes = sum(weight_bytes.values()) + sum(cache_bytes.values())
        # tt-perf-report 1.5 Blackhole 8-bank specification: 512 decimal GB/s.
        roofline_ms = useful_bytes / (DRAM_BANDWIDTH_GB_S * 10**9) * 1000
        device_ms = device["latency_us"] / 1000
        host_ms = host["decode_traced_warmed_ms"]
        rows.append(
            dict(
                layer=idx,
                kind=kind,
                decode_position=256,
                batch=1,
                weight_bytes=weight_bytes,
                cache_bytes=cache_bytes,
                minimum_useful_bytes_per_token=useful_bytes,
                dram_bandwidth_gb_s=DRAM_BANDWIDTH_GB_S,
                roofline_ms_per_token_estimate=roofline_ms,
                decode_ms_per_token_device=device_ms,
                decode_ms_per_token_e2e=host_ms,
                device_to_host_gap_ms=host_ms - device_ms,
                useful_byte_roofline_fraction_device=roofline_ms / device_ms,
                useful_byte_roofline_fraction_e2e=roofline_ms / host_ms,
                implementation_traffic_and_footprint=execution,
                e2e_reaches_70_percent_useful_byte_roofline=0.7 <= roofline_ms / host_ms <= 1.0,
                source_sha256=source_sha,
                device_evidence="perf_summary.json",
                device_report_evidence=report,
                device_report_sha256=digest,
                host_evidence=f"final_{idx}.json",
            )
        )
    result = dict(
        method="Logical once-per-token weights/cache lower bound divided by aggregate single-chip DRAM bandwidth",
        accounting_version="qz_ordered_l1_compact_decay_source_bound_conv_cache_gamma_v7",
        source_contract=(
            "Default linear decode projection_k=512 uses 6208 logical K/V/A/B columns padded to7168 "
            "as BF16 ordered weights and FP32 multiply/fast_reduce_nc products with source-bound placement, plus6144 native "
            "BF16 Q/Z columns. Useful projection bytes still charge exactly12352 logical columns. "
            "No FP32 projection-weight copy. FP32 four-tap convolution; exact compact 2 MiB decode decay LUT "
            "with original 8 MiB retained for prefill; cache-boundary casts fused into multiply input activations; "
            "explicit BF16-boundary QK normalization and FP32 recurrence. No learned L2 "
            "weight, projection bias or separate negative-exp-A tensor is read at decode. "
            f"Decode output norm uses {linear_traffic['output_norm_read']['path']} and reads "
            f"128 {linear_traffic['output_norm_read']['dtype']} gamma values "
            f"({linear_traffic['output_norm_read']['logical_read_bytes']} logical bytes); "
            "prefill-only gamma copies and native/explicit norm intermediates do not inflate the useful-byte bound."
        ),
        regime="Serialized physical chip0; identical source and workload; separate uninstrumented and profiler runs",
        accounting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence_sha256=evidence_hashes,
        bandwidth_assumption=(
            "tt-perf-report perf_report.py ArchitectureSpec(name='blackhole'): 8 banks, 512 decimal GB/s per chip. "
            "This is an assumed aggregate bandwidth ceiling, not measured sustainable bandwidth; "
            "do not double it for the two-chip P300 card or reduce it to the achieved kernel bandwidth."
        ),
        exclusions=[
            "Intermediate activation rereads/writes and metadata are not used to inflate the lower bound",
            "Only eight unique selected experts are charged, not all 256 resident weights",
            "L1/DRAM padding and replicated weight reads are execution overhead, not useful bytes",
            "Profiler drain gaps are measurement overhead and are not normal host dispatch gaps",
        ],
        limitations=[
            "This is an optimistic bandwidth-only estimate, not a compute/communication roofline",
            "Once-per-token logical DRAM reads assume weights are streamed and ignore any cross-token retention",
            "Tensor allocations, tile footprints and operation read/write counts are not measured DRAM bytes",
            "Implementation traffic entries overlap and are non-exhaustive; never sum them into the useful-byte numerator",
            "Small-op launch/data movement costs and exact-precision state arithmetic remain material",
            "Device time is the sum of non-overlapping operation durations; host time includes normal gaps",
        ],
        stop_rule=(
            "The 70% check uses warmed traced E2E, not profiler-only device time, and also "
            "requires the last two kept gains <3%. This report does not verify those gains "
            "or timebox exhaustion. Eight elapsed wall-clock hours alone do not establish "
            "eight hours of device work; use an explicit device-session interval ledger."
        ),
        measurements=rows,
    )
    (DOC / "roofline.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(
            row["kind"],
            "roofline/device/e2e ms:",
            row["roofline_ms_per_token_estimate"],
            row["decode_ms_per_token_device"],
            row["decode_ms_per_token_e2e"],
        )


if __name__ == "__main__":
    main()
