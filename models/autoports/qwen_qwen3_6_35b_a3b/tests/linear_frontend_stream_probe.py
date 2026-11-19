# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: stream the real decoder frontend, stopping before recurrence.

Both paths start with canonical prefix convolution history and then maintain
their own history. No recurrence, output projection, MLP, or acceptance claim.
--oracle-only validates the CPU reconstruction without importing TTNN.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, SNAPSHOT, config

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
REPORT_ABS_ERROR_THRESHOLD = 1e-20  # Reporting label only, never a correctness tolerance.


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gate_table_layout_name(setup_source):
    """Read explicit layout on the actual LUT assignment, not unrelated setup tensors."""
    layouts = set()
    for node in ast.walk(ast.parse(textwrap.dedent(setup_source))):
        if not isinstance(node, ast.Assign):
            continue
        targets = {item.attr for target in node.targets for item in ast.walk(target) if isinstance(item, ast.Attribute)}
        if not targets.intersection({"gate_table", "decay_table"}):
            continue
        for item in ast.walk(node.value):
            if isinstance(item, ast.keyword) and item.arg == "layout":
                assert isinstance(
                    item.value, ast.Attribute
                ), "LUT layout is dynamic; update diagnostic setup explicitly"
                layouts.add(item.value.attr)
    assert len(layouts) <= 1, f"Mixed LUT layouts require separate diagnostic setup: {layouts}"
    name = next(iter(layouts), "TILE_LAYOUT")  # Older upload(...) helper default.
    assert name in ("TILE_LAYOUT", "ROW_MAJOR_LAYOUT"), f"Unsupported LUT layout: {name}"
    return name


def pack_reduce_weight(setup_source, attribute, padded):
    """Replay only the actual setup's padded.reshape(...).permute(...) packing."""
    nodes = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(setup_source)))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == attribute for target in node.targets)
    ]
    assert len(nodes) == 1, f"Expected one explicit setup assignment for {attribute}"
    upload_call = nodes[0].value
    assert (
        isinstance(upload_call, ast.Call) and isinstance(upload_call.func, ast.Name) and upload_call.func.id == "upload"
    )
    assert (
        len(upload_call.args) == 2
        and isinstance(upload_call.args[1], ast.Attribute)
        and upload_call.args[1].attr == "bfloat16"
    )

    def apply(node):
        if isinstance(node, ast.Name) and node.id == "padded":
            return padded
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        assert (
            node.func.attr in ("reshape", "permute") and not node.keywords
        ), "Unknown ordered-weight packing; update probe"
        arguments = [ast.literal_eval(arg) for arg in node.args]
        return getattr(apply(node.func.value), node.func.attr)(*arguments)

    return apply(upload_call.args[0])


def split_projection_weights(weights):
    qkv = weights["linear_attn.in_proj_qkv.weight"]
    state = torch.cat(
        (qkv[2048:], weights["linear_attn.in_proj_a.weight"], weights["linear_attn.in_proj_b.weight"]), dim=0
    ).T
    output = torch.cat((qkv[:2048], weights["linear_attn.in_proj_z.weight"]), dim=0).T
    assert state.shape == (2048, 6208) and output.shape == (2048, 6144)
    return state, output


def assemble_split_projection(state, output):
    assert state.shape[-1] == 6208 and output.shape[-1] == 6144
    return torch.cat((output[..., :2048], state[..., :6144], output[..., 2048:], state[..., 6144:]), dim=-1)


def load_weights(layer):
    names = ["input_layernorm.weight", "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias"]
    names += [f"linear_attn.in_proj_{name}.weight" for name in ("qkv", "z", "a", "b")]
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in names:
        full = f"model.language_model.layers.{layer}.{name}"
        with safe_open(SNAPSHOT / index[full], framework="pt", device="cpu") as f:
            result[name] = f.get_tensor(full)
    return result


def cpu_norm(x, weights, eps):
    wide = x.float()
    normalized = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
    return (normalized * (1 + weights["input_layernorm.weight"].float())).bfloat16()


def cpu_frontend(token, history4, weights, eps):
    norm = cpu_norm(token, weights, eps)
    projections = {
        name: F.linear(norm, weights[f"linear_attn.in_proj_{name}.weight"]) for name in ("qkv", "z", "a", "b")
    }
    # Match HF torch_causal_conv1d_update exactly: cache4 + token -> two outputs,
    # keep the final output. A four-row convolution can choose a different kernel.
    joined = torch.cat([history4, projections["qkv"].transpose(1, 2)], dim=-1)
    pre = F.conv1d(joined, weights["linear_attn.conv1d.weight"], groups=8192)[:, :, -1:]
    conv = F.silu(pre).transpose(1, 2).contiguous()
    history4 = joined[:, :, -4:].contiguous()
    q = conv[..., :2048].reshape(1, 1, 16, 128).repeat_interleave(2, dim=2)
    k = conv[..., 2048:4096].reshape(1, 1, 16, 128).repeat_interleave(2, dim=2)

    def l2(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    result = dict(
        input_norm=norm,
        conv_preactivation=pre.transpose(1, 2).contiguous(),
        conv_silu=conv,
        q=l2(q),
        k=l2(k),
        v=conv[..., 4096:].reshape(1, 1, 32, 128),
        g=-weights["linear_attn.A_log"].float().exp()
        * F.softplus(projections["a"].float() + weights["linear_attn.dt_bias"]),
        beta=projections["b"].sigmoid(),
        conv_history=history4[:, :, -3:].transpose(1, 2).contiguous(),
    )
    result.update({f"projection_{key}": value for key, value in projections.items()})
    result.update(
        projection_q=projections["qkv"][..., :2048],
        projection_k=projections["qkv"][..., 2048:4096],
        projection_v=projections["qkv"][..., 4096:],
        projection_state_packed=torch.cat((projections["qkv"][..., 2048:], projections["a"], projections["b"]), dim=-1),
        projection_output_packed=torch.cat((projections["qkv"][..., :2048], projections["z"]), dim=-1),
    )
    result["decay_value"] = result["g"].exp()
    return result, history4


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def update(summary, stage, position, expected, actual, limit):
    actual = actual.reshape(expected.shape).to(expected.dtype)
    integer = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    ec = expected.contiguous().view(integer).reshape(-1)
    ac = actual.contiguous().view(integer).reshape(-1)
    mask = ec != ac
    e, a = expected.float().reshape(-1), actual.float().reshape(-1)
    finite = torch.isfinite(e) & torch.isfinite(a)
    count = int(mask.sum())
    error = (e.double() - a.double()).abs()
    above_threshold = mask & finite & (error > REPORT_ABS_ERROR_THRESHOLD)
    row = summary.setdefault(
        stage,
        dict(
            dtype=str(expected.dtype),
            tokens=0,
            elements=0,
            code_differences=0,
            bf16_value_differences=0,
            nonfinite_pairs=0,
            affected_tokens=0,
            max_abs_error=0.0,
            max_abs_error_position=None,
            max_abs_error_examples=[],
            mismatch_position_ranges=[],
            first_mismatches=[],
            diagnostic_abs_error_threshold=REPORT_ABS_ERROR_THRESHOLD,
            above_threshold_code_differences=0,
            above_threshold_affected_tokens=0,
            above_threshold_position_ranges=[],
            first_above_threshold_mismatches=[],
        ),
    )

    def examples(indices):
        width = 4 if integer == torch.int16 else 8
        bitmask = (1 << (4 * width)) - 1
        return [
            dict(
                index=i,
                reference=number(e[i]),
                actual=number(a[i]),
                absolute_error=number(error[i]),
                reference_code=f"0x{int(ec[i]) & bitmask:0{width}x}",
                actual_code=f"0x{int(ac[i]) & bitmask:0{width}x}",
            )
            for i in indices
        ]

    def append_range(ranges):
        if ranges and ranges[-1][1] == position - 1:
            ranges[-1][1] = position
        else:
            ranges.append([position, position])

    row["tokens"] += 1
    row["elements"] += e.numel()
    row["code_differences"] += count
    row["bf16_value_differences"] += int((expected.bfloat16() != actual.bfloat16()).sum())
    row["nonfinite_pairs"] += int((~finite).sum())
    if finite.any():
        maximum = float(error[finite].max())
        if maximum > row["max_abs_error"]:
            row["max_abs_error"], row["max_abs_error_position"] = maximum, position
            ranked = torch.where(finite, error, -1).argsort(descending=True)[:4].tolist()
            row["max_abs_error_examples"] = examples(ranked)
    if above_threshold.any():
        meaningful_count = int(above_threshold.sum())
        row["above_threshold_code_differences"] += meaningful_count
        row["above_threshold_affected_tokens"] += 1
        append_range(row["above_threshold_position_ranges"])
        if len(row["first_above_threshold_mismatches"]) < limit:
            ranked = (
                torch.where(above_threshold, error, -1).argsort(descending=True)[: min(4, meaningful_count)].tolist()
            )
            row["first_above_threshold_mismatches"].append(
                dict(position=position, code_differences=meaningful_count, examples=examples(ranked))
            )
    if count:
        row["affected_tokens"] += 1
        append_range(row["mismatch_position_ranges"])
        valid_mismatch = mask & finite
        if valid_mismatch.any():
            for label, values in (("reference", e), ("actual", a)):
                lo, hi = float(values[valid_mismatch].min()), float(values[valid_mismatch].max())
                key = f"mismatching_{label}_range"
                old = row.get(key, [lo, hi])
                row[key] = [min(old[0], lo), max(old[1], hi)]
        if len(row["first_mismatches"]) < limit:
            row["first_mismatches"].append(
                dict(
                    position=position,
                    code_differences=count,
                    examples=examples(mask.nonzero().reshape(-1)[:4].tolist()),
                )
            )
    return count


def make_device_frontend(ttnn, mesh, weights, cfg, seed3):
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

    # Only attributes used by the actual current decode frontend are initialized.
    # In particular, there are no experts, output projection, or recurrent weights.
    dec = FusedDecoder()
    dec.config, dec.mesh_device, dec.kind, dec.layer_idx = cfg, mesh, "linear_attention", 0
    dec.compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    round_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
    widen_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
    dec.round_widen_chain = [round_op, widen_op]
    dec.conv_post_chain = [round_op, widen_op, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), round_op]
    dec.beta_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), round_op, widen_op]
    dec.rsqrt_round_chain = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0), round_op, widen_op]

    def upload(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    packed = torch.cat([weights[f"linear_attn.in_proj_{name}.weight"] for name in ("qkv", "z", "a", "b")], 0).T
    linear_source = inspect.getsource(FusedDecoder.linear)
    setup_source = inspect.getsource(FusedDecoder.from_state_dict)
    frontend_source = inspect.getsource(FusedDecoder._linear_attention)
    split_projection = '"output_only_packed"' in frontend_source
    ordered_packed, output_packed = split_projection_weights(weights) if split_projection else (packed, None)
    if "attention_k_indices" in linear_source:
        order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(2048)
        dec.attention_k_indices = upload(order.to(torch.int32).reshape(1, 1, 2048), ttnn.uint16)
        if "precise_attention_weight" in linear_source:
            dec.precise_attention_weight = upload(ordered_packed[order])
        padded = F.pad(ordered_packed[order], (0, 960))
        for attribute in ("reduce_attention_weight", "reduce_prefill_attention_weight"):
            if attribute in linear_source:
                setattr(dec, attribute, upload(pack_reduce_weight(setup_source, attribute, padded), ttnn.bfloat16))
    else:
        dec.precise_attention_weight = upload(packed)
    # Current LUT candidate adds these setup-only tables; older arithmetic candidates
    # do not reference them and keep the original lightweight setup.
    if "gate_table" in frontend_source or "decay_table" in frontend_source:
        gate_layout_name = gate_table_layout_name(inspect.getsource(FusedDecoder.from_state_dict))
        gate_layout = getattr(ttnn, gate_layout_name)
        codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
        gate_values = -weights["linear_attn.A_log"].float().exp().reshape(32, 1) * F.softplus(
            codes.reshape(1, -1) + weights["linear_attn.dt_bias"].float().reshape(32, 1)
        )
        if "gate_table" in frontend_source:
            dec.gate_table = upload(gate_values, layout=gate_layout)
        if "decay_table" in frontend_source:
            dec.decay_table = upload(gate_values.exp(), layout=gate_layout)
    dec.w = {
        "attention_packed": upload(ordered_packed, ttnn.bfloat16),
        "input_layernorm.weight": upload((1 + weights["input_layernorm.weight"].float()).reshape(1, 1, -1)),
        "linear_attn.A_log": upload(-weights["linear_attn.A_log"].float().exp().reshape(1, 1, -1)),
        "linear_attn.dt_bias": upload(weights["linear_attn.dt_bias"].reshape(1, 1, -1)),
    }
    if split_projection:
        dec.w["output_only_packed"] = upload(output_packed, ttnn.bfloat16)
    dec.decode_linear, dec.attention_bias = {}, None
    dec.taps_fp32 = [upload(weights["linear_attn.conv1d.weight"][:, 0, i].reshape(1, 1, -1)) for i in range(4)]
    conv_layout = (
        ttnn.ROW_MAJOR_LAYOUT
        if 'ttnn.to_layout(state["conv"]' in inspect.getsource(FusedDecoder._linear_attention)
        else ttnn.TILE_LAYOUT
    )
    state = dict(conv=upload(seed3, ttnn.bfloat16, conv_layout), recurrent=upload(torch.zeros(1, 32, 128, 128)))
    capture = {}
    actual_linear = dec.linear

    class FrontendComplete(Exception):
        pass

    def stop(q, k, v, g, beta, incoming, **kwargs):
        capture.update(q=q, k=k, v=v, beta=beta)
        if g is not None:
            capture["g"] = g
        if kwargs.get("decay_value") is not None:
            capture["decay_value"] = kwargs["decay_value"]
        raise FrontendComplete

    def linear(x, key, *args, **kwargs):
        out = actual_linear(x, key, *args, **kwargs)
        if key == "attention_packed":
            if split_projection:
                assert out.shape[-1] == 6208 and out.dtype == ttnn.bfloat16
                capture["state_projected_bf16"] = out
            else:
                capture["packed"] = out
                if out.dtype == ttnn.bfloat16:
                    capture["packed_bf16"] = out
        elif key == "output_only_packed":
            assert split_projection and out.shape[-1] == 6144 and out.dtype == ttnn.bfloat16
            capture["output_projected_bf16"] = out
        return out

    dec._recurrent_step, dec.linear = stop, linear

    def run(token):
        # No torch calculations, uploads, or CPU oracle values inside this call.
        capture.clear()
        norm = dec.norm(token, "input_layernorm.weight")
        capture["input_norm"] = norm
        original_silu, original_cast, original_chain = ttnn.silu, ttnn.typecast, ttnn.unary_chain
        original_concat = ttnn.concat

        def silu(x, *args, **kwargs):
            out = original_silu(x, *args, **kwargs)
            capture["conv_preactivation"], capture["conv_fp32"] = x, out
            return out

        def cast(x, dtype, *args, **kwargs):
            out = original_cast(x, dtype, *args, **kwargs)
            if x is capture.get("conv_fp32") and dtype == ttnn.bfloat16:
                capture["conv_silu"] = out
            if x is capture.get("packed") and dtype == ttnn.bfloat16:
                capture["packed_bf16"] = out
            return out

        def chain(x, ops_chain, *args, **kwargs):
            out = original_chain(x, ops_chain, *args, **kwargs)
            if ops_chain is dec.conv_post_chain:
                assert "conv_silu" not in capture, "multiple convolution activation producers in one decode"
                capture["conv_accum_fp32"], capture["conv_silu"] = x, out
            return out

        def concat(tensors, *args, **kwargs):
            out = original_concat(tensors, *args, **kwargs)
            if split_projection and out.shape[-1] == 12352:
                assert "state_projected_bf16" in capture and "output_projected_bf16" in capture
                assert len(tensors) == 4 and [tensor.shape[-1] for tensor in tensors] == [2048, 6144, 4096, 64]
                assert "packed_bf16" not in capture, "multiple full projected tensor producers"
                capture["packed"], capture["packed_bf16"] = out, out
            return out

        ttnn.silu, ttnn.typecast, ttnn.unary_chain = silu, cast, chain
        ttnn.concat = concat
        try:
            try:
                dec._linear_attention(norm, state, decode=True, cache_boundary=False)
            except FrontendComplete:
                pass
            else:
                raise AssertionError("frontend capture did not stop before recurrence")
        finally:
            ttnn.silu, ttnn.typecast, ttnn.unary_chain = original_silu, original_cast, original_chain
            ttnn.concat = original_concat
        # Diagnostic materialization happens only after the actual frontend stopped.
        # It never feeds the decoder or its convolution history. The chain's real
        # input/output are captured above; its internal BF16 boundary has no tensor.
        if "conv_accum_fp32" in capture:
            assert "conv_preactivation" not in capture, "both fused and standalone convolution activation executed"
            capture["conv_preactivation"] = original_cast(capture["conv_accum_fp32"], ttnn.bfloat16)
            preactivation_origin = "diagnostic BF16 cast of actual unary_chain input, after frontend stop"
        else:
            preactivation_origin = "actual standalone SiLU input"
        capture["conv_history"] = state["conv"]
        required = {
            "input_norm",
            "packed_bf16",
            "conv_preactivation",
            "conv_silu",
            "q",
            "k",
            "v",
            "beta",
            "conv_history",
        }
        if split_projection:
            required.update(("state_projected_bf16", "output_projected_bf16"))
        assert required <= capture.keys(), f"frontend changed; missing captures: {required-capture.keys()}"
        assert "g" in capture or "decay_value" in capture, "frontend produced neither g nor decay_value"
        assert capture["packed_bf16"].dtype == capture["conv_silu"].dtype == ttnn.bfloat16
        capture["_diagnostic_metadata"] = dict(
            conv_preactivation_origin=preactivation_origin,
            packed_projection_output_dtype=str(capture["packed"].dtype),
            projection_mode="ordered K/V/A/B plus native Q/Z" if split_projection else "single packed projection",
            packed_origin="actual final concat; independently checked against producer outputs on CPU"
            if split_projection
            else "actual packed linear output/cast",
            g_produced="g" in capture,
            decay_value_produced="decay_value" in capture,
            missing_g_reason=None if "g" in capture else "decode uses materialized decay_value; g was not produced",
            gate_table_layout=(
                gate_layout_name if "gate_table" in frontend_source or "decay_table" in frontend_source else None
            ),
        )
        return capture

    return run, upload


def main(args):
    torch.set_num_threads(8)
    if args.steps < 1 or args.first_mismatches < 1 or args.flush_every < 1:
        raise ValueError("steps, first-mismatches and flush-every must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    cfg, weights = config(), load_weights(0)
    assert cfg.layer_types[0] == "linear_attention"
    saved = torch.load(args.stream, map_location="cpu", weights_only=False)
    prefix, tokens = saved["prefix"], saved["tokens"]
    assert prefix.shape[0] == tokens.shape[0] == 1 and prefix.shape[1] >= 4
    assert tokens.shape[-1] == prefix.shape[-1] == 2048
    if args.steps > tokens.shape[1]:
        raise ValueError("requested more decode tokens than saved stream contains")
    # Canonical HF prefill projects the prefix as one matrix, then caches its last four rows.
    seed = F.linear(cpu_norm(prefix, weights, cfg.rms_norm_eps), weights["linear_attn.in_proj_qkv.weight"])
    history4 = seed[:, -4:].transpose(1, 2).contiguous()
    seed3 = history4[:, :, -3:].transpose(1, 2).contiguous()
    source = ROOT / "tt/fused_decoder.py"
    source_hash = digest(source)
    Path(str(output) + ".fused_source.py.snapshot").write_bytes(source.read_bytes())
    result = dict(
        purpose="Diagnostic only; frontend stopped before recurrence; not acceptance or latency evidence",
        execution="CPU-only" if args.oracle_only else "serial physical chip 0",
        checkpoint_revision=REVISION,
        stream_sha256=digest(args.stream),
        fused_source_sha256=source_hash,
        prefix_length=prefix.shape[1],
        requested_decode_tokens=args.steps,
        completed_decode_tokens=0,
        initialization="Canonical HF prefix conv history seeded once; independent CPU/device histories thereafter",
        diagnostic_abs_error_threshold=REPORT_ABS_ERROR_THRESHOLD,
        threshold_note="Reporting label only; all exact-code difference totals remain unfiltered",
        interpretation="Exact-code differences are diagnostic observations, not runtime-bug or model-acceptance verdicts. Native Q/Z projections may differ from canonical HF; their differences remain fully counted.",
        stages={},
        cpu_capture_validation={},
        first_global_mismatch=None,
        capture_contract=None,
    )

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    mesh = None
    if not args.oracle_only:
        import ttnn

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        if mesh is not None:
            frontend, upload = make_device_frontend(ttnn, mesh, weights, cfg, seed3)
        for i in range(args.steps):
            position = prefix.shape[1] + i
            token = tokens[:, i : i + 1].contiguous()
            expected, history4 = cpu_frontend(token, history4, weights, cfg.rms_norm_eps)
            probe_path = Path(f"{args.validate_capture_prefix}.pos{position}.pt")
            if args.validate_capture_prefix and probe_path.exists():
                probe = torch.load(probe_path, map_location="cpu", weights_only=False)
                assert torch.equal(token, probe["token"]), f"capture stream mismatch at {position}"
                mappings = dict(
                    input_norm="input_norm",
                    projection_qkv="projection_qkv",
                    projection_a="projection_a",
                    projection_b="projection_b",
                    q="delta_q",
                    k="delta_k",
                    v="delta_v",
                    g="delta_g",
                    beta="delta_beta",
                )
                for stage, key in mappings.items():
                    if key in probe["hf_values"]:
                        differences = update(
                            result["cpu_capture_validation"],
                            stage,
                            position,
                            probe["hf_values"][key],
                            expected[stage],
                            args.first_mismatches,
                        )
                        assert differences == 0, f"CPU oracle disagrees with saved HF {stage} at {position}"
            if mesh is not None:
                device_token = upload(token, ttnn.bfloat16)
                captured = frontend(device_token)
                contract = captured["_diagnostic_metadata"]
                if result["capture_contract"] is None:
                    result["capture_contract"] = contract
                assert (
                    result["capture_contract"] == contract
                ), "frontend producer/capture contract changed within stream"
                actual = {
                    key: ttnn.to_torch(value).clone()
                    for key, value in captured.items()
                    if key
                    not in (
                        "packed",
                        "packed_bf16",
                        "conv_fp32",
                        "conv_accum_fp32",
                        "_diagnostic_metadata",
                        "state_projected_bf16",
                        "output_projected_bf16",
                    )
                }
                packed = ttnn.to_torch(captured["packed_bf16"]).clone()
                if "state_projected_bf16" in captured:
                    state_projected = ttnn.to_torch(captured["state_projected_bf16"]).clone()
                    output_projected = ttnn.to_torch(captured["output_projected_bf16"]).clone()
                    assembled = assemble_split_projection(state_projected, output_projected)
                    assert torch.equal(
                        assembled.contiguous().view(torch.int16), packed.contiguous().view(torch.int16)
                    ), "actual projected concat differs from actual producer-output reassembly"
                    actual["projection_state_packed"] = state_projected
                    actual["projection_output_packed"] = output_projected
                for name, start, end in (
                    ("qkv", 0, 8192),
                    ("z", 8192, 12288),
                    ("a", 12288, 12320),
                    ("b", 12320, 12352),
                ):
                    actual[f"projection_{name}"] = packed[..., start:end]
                for name, start, end in (("q", 0, 2048), ("k", 2048, 4096), ("v", 4096, 8192)):
                    actual[f"projection_{name}"] = packed[..., start:end]
                for name in ("q", "k"):
                    actual[name] = actual[name].repeat_interleave(2, dim=2)
                producer_order = (
                    ("projection_state_packed", "projection_output_packed")
                    if "state_projected_bf16" in captured
                    else ()
                )
                order = (
                    ("input_norm",)
                    + producer_order
                    + (
                        "projection_qkv",
                        "projection_q",
                        "projection_k",
                        "projection_v",
                        "projection_z",
                        "projection_a",
                        "projection_b",
                        "conv_history",
                        "conv_preactivation",
                        "conv_silu",
                        "q",
                        "k",
                        "v",
                        "g",
                        "beta",
                        "decay_value",
                    )
                )
                required_stages = set(order) - {"g", "decay_value"}
                assert (
                    required_stages <= actual.keys()
                ), f"missing executed frontend producer: {required_stages-actual.keys()}"
                assert actual.keys() <= set(order), f"uncompared frontend captures: {actual.keys()-set(order)}"
                assert ("g" in actual) == contract["g_produced"]
                assert ("decay_value" in actual) == contract["decay_value_produced"]
                for stage in order:
                    if stage in ("g", "decay_value") and stage not in actual:
                        continue  # Only absent producers are omitted; every materialized value is compared.
                    different = update(
                        result["stages"], stage, position, expected[stage], actual[stage], args.first_mismatches
                    )
                    if different and result["first_global_mismatch"] is None:
                        result["first_global_mismatch"] = dict(
                            position=position, stage=stage, code_differences=different
                        )
            result["completed_decode_tokens"] = i + 1
            if (i + 1) % args.flush_every == 0:
                save()
                print(json.dumps(dict(completed=i + 1, first_mismatch=result["first_global_mismatch"])), flush=True)
        assert digest(source) == source_hash, "Fused source changed during diagnostic; rerun with a fixed candidate"
        save()
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(
        json.dumps(
            dict(
                output=str(output),
                completed=result["completed_decode_tokens"],
                first_mismatch=result["first_global_mismatch"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=str(DOC / "oracle_cache/stress_8192_0.pt"))
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--first-mismatches", type=int, default=20)
    parser.add_argument("--flush-every", type=int, default=128)
    parser.add_argument("--validate-capture-prefix", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--output", default=str(DOC / "endurance/linear_frontend_stream_probe_qz_split.json"))
    main(parser.parse_args())
