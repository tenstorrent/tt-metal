# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured saved-stream diagnostic; immediate host snapshots prevent alias corruption."""

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "doc/fused_decoder/endurance"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(x) for x in value)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return value


def compare(reference, actual):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import pcc

    assert reference.numel() == actual.numel(), (reference.shape, actual.shape)
    actual = actual.reshape(reference.shape)
    a, b = reference.float(), actual.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    return {
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "shape": list(reference.shape),
        "pcc": pcc(a, b),
        "max_abs": float((a[finite] - b[finite]).abs().max()) if finite.any() else None,
        "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
        "value_differences": int((a != b).sum()),
        "bf16_code_differences": int((a.bfloat16().view(torch.int16) != b.bfloat16().view(torch.int16)).sum()),
        "nonfinite_elements": int((~finite).sum()),
    }


def decoder_identity(args):
    source = (Path(args.snapshot) if args.snapshot else ROOT / "tt/fused_decoder.py").resolve()
    assert source.is_file(), source
    tree = ast.parse(source.read_text(), filename=str(source))
    assert any(isinstance(node, ast.ClassDef) and node.name == "FusedDecoder" for node in tree.body), source
    return dict(
        measured_decoder_source=str(source),
        measured_decoder_source_sha256=digest(source),
        candidate_selection="Explicit diagnostic snapshot; not production selection"
        if args.snapshot
        else "Current production source; diagnostic execution only",
        acceptance_eligible=False,
    )


def load_decoder(args, identity):
    source = identity["measured_decoder_source"]
    assert digest(source) == identity["measured_decoder_source_sha256"], "Selected source changed before import"
    if args.snapshot:
        loader = importlib.machinery.SourceFileLoader("layer_stream_candidate_" + digest(source)[:16], source)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        loader.exec_module(module)
        decoder = module.FusedDecoder
    else:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

        decoder = FusedDecoder
    assert Path(inspect.getfile(decoder)).resolve() == Path(source)
    assert digest(source) == identity["measured_decoder_source_sha256"], "Selected source changed during import"
    return decoder


def load_stream(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

    identity = decoder_identity(args)
    cached = torch.load(args.oracle_cache, map_location="cpu", weights_only=False)
    key = cached["key"]
    params = key["parameters"]
    assert params["layer"] == args.layer and params["seed"] == args.seed, params
    assert key["revision"] == reference.REVISION
    assert key["torch_version"] == torch.__version__
    for name, expected in key["source_hashes"].items():
        assert digest(name) == expected, f"Frozen oracle source changed: {name}"
    for name, length in (
        ("prefix", params["prompt_length"]),
        ("tokens", params["decode_length"]),
        ("continuation", params["continuation_length"]),
    ):
        assert tuple(cached[name].shape) == (1, length, 2048), name
        assert cached[name].dtype == torch.bfloat16
    failure = json.loads(Path(args.failure_json).read_text())
    assert failure["layer"] == args.layer and failure["seed"] == args.seed
    assert failure["cpu_oracle_key"] == key
    worst = min(failure["results"], key=lambda r: r["contribution_pcc"])
    positions = sorted(set(args.positions + ([worst["position"]] if args.include_worst else [])))
    assert all(params["prompt_length"] <= pos < params["prompt_length"] + params["decode_length"] for pos in positions)
    metadata = {
        **identity,
        "classification": "diagnostic_only_not_acceptance_or_timing",
        "layer": args.layer,
        "seed": args.seed,
        "positions": positions,
        "worst_recorded_row": worst,
        "include_worst": args.include_worst,
        "decode_steps_to_last_capture": max(positions) - params["prompt_length"] + 1,
        "oracle": str(Path(args.oracle_cache).resolve()),
        "oracle_sha256": digest(args.oracle_cache),
        "oracle_key": key,
        "stream_sha256": {name: tensor_digest(cached[name]) for name in ("prefix", "tokens", "continuation")},
        "failure_json": str(Path(args.failure_json).resolve()),
        "failure_json_sha256": digest(args.failure_json),
        "source_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                ROOT / "tt/fused_decoder.py",
                Path(reference.__file__),
                Path(identity["measured_decoder_source"]),
            )
        },
        "stream_policy": "Load saved BF16 tensors verbatim; no RNG generation or truncated-stream regeneration.",
        "capture_policy": "Immediate synchronous TT-to-CPU clone at producer/input boundary on selected tokens only; no live TT aliases retained.",
        "execution": "cpu_validation_only" if args.cpu_only else "eager_serial_diagnostic_with_synchronizing_observers",
        "device_closed": None,
        "rows": [],
    }
    return cached, metadata


def run(args):
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output)
    assert (
        not output.exists() and not Path(str(output) + ".provenance.json").exists()
    ), "Use a fresh diagnostic output path"
    output.parent.mkdir(parents=True, exist_ok=True)
    cached, report = load_stream(args)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.cpu_only:
        assert "ttnn" not in sys.modules, "CPU validation imported TTNN"
        original = torch.arange(8).reshape(2, 4)
        preserved = cpu_copy({"input": original})
        original.zero_()
        assert preserved["input"][1, 3].item() == 7, "CPU snapshot aliases its producer"
        print(json.dumps({k: report[k] for k in ("positions", "worst_recorded_row", "stream_sha256", "execution")}))
        return

    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights

    FusedDecoder = load_decoder(args, report)

    record(output)
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance.update({key: report[key] for key in decoder_identity(args)})
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    c, sd = config(), weights(args.layer)
    hf = layer(sd, args.layer)
    assert hf.layer_type == "linear_attention"
    report["hf_source_sha256"] = {inspect.getfile(type(hf)): digest(inspect.getfile(type(hf)))}
    hf_values, tt_values = {}, {}
    recording = False

    def hf_hook(name):
        def capture(module, inputs, result):
            if recording:
                hf_values[name] = cpu_copy(result)

        return capture

    for name, module in (
        ("input_norm", hf.input_layernorm),
        ("attention", hf.linear_attn),
        ("post_norm", hf.post_attention_layernorm),
        ("router", hf.mlp.gate),
        ("projection_a", hf.linear_attn.in_proj_a),
        ("projection_b", hf.linear_attn.in_proj_b),
        ("projection_qkv", hf.linear_attn.in_proj_qkv),
        ("projection_z", hf.linear_attn.in_proj_z),
        ("gated_core", hf.linear_attn.norm),
    ):
        module.register_forward_hook(hf_hook(name))

    def hf_norm_input(name):
        def capture(module, inputs):
            if recording:
                hf_values[name] = cpu_copy(inputs[0])

        return capture

    hf.input_layernorm.register_forward_pre_hook(hf_norm_input("input_norm_input"))
    hf.post_attention_layernorm.register_forward_pre_hook(hf_norm_input("post_norm_input"))
    old_hf_delta = hf.linear_attn.recurrent_gated_delta_rule
    old_hf_conv = hf.linear_attn.causal_conv1d_update
    assert old_hf_delta.__name__ == "torch_recurrent_gated_delta_rule", old_hf_delta

    def hf_delta(q, k, v, **kw):
        if recording:
            assert kw["use_qk_l2norm_in_kernel"]
            hf_values.update(
                cpu_copy(
                    dict(
                        delta_q=l2norm(q),
                        delta_k=l2norm(k),
                        delta_v=v,
                        delta_g=kw["g"],
                        delta_decay=kw["g"].float().exp(),
                        delta_beta=kw["beta"],
                        delta_state_input=kw["initial_state"],
                    )
                )
            )
        result = old_hf_delta(q, k, v, **kw)
        if recording:
            hf_values.update(cpu_copy(dict(core=result[0], new_state=result[1])))
            # Explicitly labeled CPU reconstruction from the captured canonical inputs.
            # Actual returned core/state above remain the reference, not this reconstruction.
            initial = hf_values["delta_state_input"].float()
            kk = hf_values["delta_k"].transpose(1, 2).float()[:, :, 0]
            qq = hf_values["delta_q"].transpose(1, 2).float()[:, :, 0] * (128**-0.5)
            vv = hf_values["delta_v"].transpose(1, 2).float()[:, :, 0]
            decay = hf_values["delta_decay"].transpose(1, 2)[:, :, 0, None, None]
            beta = hf_values["delta_beta"].transpose(1, 2).float()[:, :, 0, None]
            decayed = initial * decay
            memory = (decayed * kk.unsqueeze(-1)).sum(dim=-2)
            delta = (vv - memory) * beta
            new = decayed + kk.unsqueeze(-1) * delta.unsqueeze(-2)
            hf_values["canonical_reconstruction"] = dict(
                decayed_state=decayed,
                memory=memory,
                value_minus_memory=vv - memory,
                delta_update=delta,
                new_state=new,
                core_fp32=(new * qq.unsqueeze(-1)).sum(dim=-2),
            )
            assert torch.equal(new, result[1]), "Canonical recurrence reconstruction changed"
        return result

    def hf_conv(*a, **kw):
        if recording:
            hf_values["conv_input_state"] = cpu_copy(a[1])
        result = old_hf_conv(*a, **kw)
        if recording:
            hf_values["conv_output"] = cpu_copy(result.transpose(1, 2))
        return result

    hf.linear_attn.recurrent_gated_delta_rule = hf_delta
    hf.linear_attn.causal_conv1d_update = hf_conv
    hc = DynamicCache(config=c)
    prefix, tokens = cached["prefix"], cached["tokens"]
    ref_prefix = hf_forward(hf, prefix, hc, 0)
    assert torch.equal(ref_prefix, cached["ref_p"]), "Fresh HF prefix differs from frozen oracle"
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    report["device_closed"] = False
    try:
        dec = FusedDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = dec.new_recurrent_state()

        def upload(x):
            return ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def save(name, value):
            # Reading here synchronizes before any later in-place write or buffer reuse.
            tt_values[name] = ttnn.to_torch(value).detach().cpu().clone()

        prefix_out = dec.prefill_forward(upload(prefix), state=state, plan=dec.prepare_prefill(seq_len=prefix.shape[1]))
        report["prefill"] = compare(ref_prefix, ttnn.to_torch(prefix_out).reshape(prefix.shape))
        del prefix_out
        old_delta, old_sum = dec._recurrent_step, dec._recurrent_sum
        old_norm, old_attention, old_linear = dec.norm, dec._linear_attention, dec.linear
        old_chain = ttnn.unary_chain
        old_rms, old_multiply = ttnn.rms_norm, ttnn.multiply
        norm_producers = {}
        old_inline_selector = getattr(dec, "inline_selector", None)
        if old_inline_selector is not None:

            def inline_selector(logits):
                result = old_inline_selector(logits)
                if recording:
                    save("router_scores", result[0])
                    save("router_ids", result[1])
                return result

            dec.inline_selector = inline_selector

        def delta(q, k, v, g, beta, incoming, **kw):
            if not recording:
                return old_delta(q, k, v, g, beta, incoming, **kw)
            for name, value in dict(
                delta_q=q, delta_k=k, delta_v=v, delta_beta=beta, delta_state_input=incoming
            ).items():
                save(name, value)
            if g is not None:
                save("delta_g", g)
            if kw.get("decay_value") is not None:
                save("delta_decay", kw["decay_value"])
            tt_values["round_state"] = kw.get("round_state", False)
            tt_values["output_state_is_input"] = kw.get("output_state") is incoming
            original_ops = {name: getattr(ttnn, name) for name in ("multiply", "subtract", "add", "addcmul")}
            sum_count = 0

            def reduction(products):
                nonlocal sum_count
                sum_count += 1
                name = "memory" if sum_count == 1 else "reduced_output"
                save(name + "_products", products)
                out = old_sum(products)
                save(name, out)
                return out

            def observe(name):
                def invoke(*a, **kwargs):
                    out = original_ops[name](*a, **kwargs)
                    if name == "multiply" and a[0] is incoming:
                        save("decayed_state", out)
                    elif name == "subtract":
                        save("value_minus_memory", out)
                    elif name == "multiply" and tuple(out.shape) == (1, 32, 1, 128):
                        save("delta_update", out)
                    elif name in ("add", "addcmul") and tuple(out.shape) == (1, 32, 128, 128):
                        save("new_state", out)
                    return out

                return invoke

            try:
                dec._recurrent_sum = reduction
                for name in original_ops:
                    setattr(ttnn, name, observe(name))
                result = old_delta(q, k, v, g, beta, incoming, **kw)
                save("core", result[0])
                save("new_state", result[1])
                assert sum_count == 2
            finally:
                dec._recurrent_sum = old_sum
                for name, op in original_ops.items():
                    setattr(ttnn, name, op)
            return result

        def norm(x, key):
            if recording and key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                save("input_norm_input" if key == "input_layernorm.weight" else "post_norm_input", x)
            out = old_norm(x, key)
            if recording and key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                save("input_norm" if key == "input_layernorm.weight" else "post_norm", out)
            return out

        def attention(x, cache, **kw):
            if recording:
                save("conv_input_state", cache["conv"])
            norm_producers.clear()
            try:
                out = old_attention(x, cache, **kw)
            finally:
                norm_producers.clear()
            if recording:
                save("attention", out)
            return out

        def linear(x, key, *a, **kw):
            if recording and key == "linear_attn.out_proj.weight":
                save("gated_core", x)
            out = old_linear(x, key, *a, **kw)
            if recording:
                name = {
                    "mlp.gate.weight": "router",
                    "attention_packed": "projection_packed",
                    "output_only_packed": "projection_output_only",
                }.get(key)
                if name:
                    save(name, out)
            return out

        def chain(x, ops, *a, **kw):
            if recording and ops is dec.conv_post_chain:
                save("conv_accum_fp32", x)
            out = old_chain(x, ops, *a, **kw)
            if recording and ops is dec.conv_post_chain:
                save("conv_output", out)
            return out

        def rms(x, *a, **kw):
            target = recording and tuple(x.shape) == (1, 32, 128) and kw.get("weight") is None
            if target:
                save("gated_unweighted_rms_input", x)
            out = old_rms(x, *a, **kw)
            if target:
                save("gated_unweighted_rms_output", out)
                norm_producers["normalized"] = out
            return out

        def multiply(*a, **kw):
            normalized = bool(a) and a[0] is norm_producers.get("normalized")
            weighted = bool(a) and a[0] is norm_producers.get("weighted")
            out = old_multiply(*a, **kw)
            if recording and normalized:
                save("gated_weighted_output", out)
                norm_producers["weighted"] = out
            elif recording and weighted:
                save("gated_silu_product", out)
            return out

        dec._recurrent_step, dec.norm, dec._linear_attention, dec.linear = delta, norm, attention, linear
        ttnn.unary_chain, ttnn.rms_norm, ttnn.multiply = chain, rms, multiply
        try:
            for index in range(max(report["positions"]) - prefix.shape[1] + 1):
                position = prefix.shape[1] + index
                recording = position in report["positions"]
                hf_values.clear()
                tt_values.clear()
                x = tokens[:, index : index + 1]
                ref = hf_forward(hf, x, hc, position)
                device_out = dec.decode_forward(upload(x), state=state)
                if not recording:
                    del device_out
                    if index % 512 == 0:
                        print("STREAM_PROGRESS", position, flush=True)
                    continue
                assert torch.equal(ref, cached["refs"][index]), f"Fresh HF differs from saved oracle at {position}"
                got = ttnn.to_torch(device_out).reshape(x.shape).clone()
                del device_out
                save("state", state["recurrent"])
                save("conv_state", state["conv"])
                hf_values["state"] = cpu_copy(hc.layers[args.layer].recurrent_states)
                hf_values["conv_state"] = cpu_copy(hc.layers[args.layer].conv_states)
                packed, output_packed = tt_values["projection_packed"], tt_values["projection_output_only"]
                tt_values.update(
                    projection_qkv=torch.cat((output_packed[..., :2048], packed[..., :6144]), -1),
                    projection_z=output_packed[..., 2048:],
                    projection_a=packed[..., 6144:6176],
                    projection_b=packed[..., 6176:6208],
                )
                comparisons = {}
                required = (
                    "input_norm_input",
                    "post_norm_input",
                    "input_norm",
                    "projection_qkv",
                    "projection_z",
                    "projection_a",
                    "projection_b",
                    "conv_output",
                    "delta_q",
                    "delta_k",
                    "delta_v",
                    "delta_decay",
                    "delta_beta",
                    "delta_state_input",
                    "decayed_state",
                    "memory",
                    "value_minus_memory",
                    "delta_update",
                    "new_state",
                    "core",
                    "gated_core",
                    "attention",
                    "post_norm",
                    "router",
                )
                assert all(name in tt_values for name in required), set(required) - tt_values.keys()
                for name in required:
                    if name == "router":
                        continue
                    h = hf_values.get(name, hf_values["canonical_reconstruction"].get(name))
                    t = tt_values[name]
                    if name in ("delta_q", "delta_k"):
                        assert t.shape[2] * 2 == h.shape[2]
                        t = t.repeat_interleave(2, dim=2)
                    comparisons[name] = compare(h, t)
                if "delta_g" in tt_values:
                    comparisons["delta_g"] = compare(hf_values["delta_g"], tt_values["delta_g"])
                comparisons["core_fp32_before_hf_rounding"] = compare(
                    hf_values["canonical_reconstruction"]["core_fp32"], tt_values["core"]
                )
                comparisons["core_after_bf16_boundary"] = compare(hf_values["core"], tt_values["core"].bfloat16())
                for name in ("delta_state_input", "state"):
                    h, t = hf_values[name], tt_values[name]
                    assert h.dtype == torch.bfloat16 and t.dtype == torch.float32, (name, h.dtype, t.dtype)
                    comparisons[name + "_raw_fp32_storage"] = compare(h, t)
                    comparisons[name + "_effective_bf16_public_boundary"] = compare(h, t.bfloat16().float())
                for name in ("conv_input_state", "conv_state"):
                    comparisons[name] = compare(hf_values[name][..., -3:].transpose(1, 2), tt_values[name])
                hf_logits, hf_scores, hf_ids = hf_values["router"]
                tt_logits = tt_values["router"].reshape(hf_logits.shape)
                comparisons["router_logits"] = compare(hf_logits, tt_logits)
                cpu_logits = torch.nn.functional.linear(tt_values["post_norm"].reshape(1, 2048), sd["mlp.gate.weight"])
                if old_inline_selector is not None:
                    native_ids = tt_values["router_ids"].flatten().clone()
                    native_scores = tt_values["router_scores"].flatten().clone()
                else:
                    native_ids = ttnn.to_torch(dec.router_out_ids).reshape(32, 32)[0, :8].clone()
                    native_scores = ttnn.to_torch(dec.router_out).reshape(32, 32)[0, :8].clone()
                tt_values["router_ids"] = native_ids
                tt_values["router_scores"] = native_scores
                # Direct forward calls bypass hooks, preserving all actual HF captures.
                with torch.no_grad():
                    canonical_gate = type(hf.linear_attn.norm).forward(
                        hf.linear_attn.norm,
                        tt_values["core"].reshape(-1, 128).bfloat16(),
                        tt_values["projection_z"].reshape(-1, 128),
                    )
                    residual_from_tt_attention = x + tt_values["attention"].reshape(x.shape)
                    canonical_post_assuming_residual = type(hf.post_attention_layernorm).forward(
                        hf.post_attention_layernorm,
                        residual_from_tt_attention,
                    )
                    canonical_post = type(hf.post_attention_layernorm).forward(
                        hf.post_attention_layernorm,
                        tt_values["post_norm_input"].reshape(x.shape),
                    )
                tt_values["cpu_canonical_gated_core_on_tt_inputs"] = cpu_copy(canonical_gate)
                tt_values["cpu_bf16_residual_from_tt_attention"] = cpu_copy(residual_from_tt_attention)
                tt_values["cpu_canonical_postnorm_on_tt_residual"] = cpu_copy(canonical_post_assuming_residual)
                tt_values["cpu_canonical_postnorm_on_actual_tt_input"] = cpu_copy(canonical_post)
                tt_values["cpu_canonical_router_on_tt_postnorm"] = cpu_copy(cpu_logits)
                comparisons["gated_core_local_arithmetic"] = compare(canonical_gate, tt_values["gated_core"])
                comparisons["post_norm_input_vs_cpu_bf16_residual"] = compare(
                    residual_from_tt_attention, tt_values["post_norm_input"]
                )
                comparisons["post_norm_local_arithmetic_assuming_bf16_residual"] = compare(
                    canonical_post_assuming_residual, tt_values["post_norm"]
                )
                comparisons["post_norm_local_arithmetic"] = compare(canonical_post, tt_values["post_norm"])
                if "gated_unweighted_rms_output" in tt_values:
                    rms_input = tt_values["gated_unweighted_rms_input"]
                    wide = rms_input.float()
                    canonical_normalized = (
                        wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + c.rms_norm_eps)
                    ).to(rms_input.dtype)
                    comparisons["gated_unweighted_rms_local_arithmetic"] = compare(
                        canonical_normalized, tt_values["gated_unweighted_rms_output"]
                    )
                    tt_values["cpu_canonical_unweighted_rms_on_tt_input"] = cpu_copy(canonical_normalized)
                    if "gated_weighted_output" in tt_values:
                        actual_normalized = tt_values["gated_unweighted_rms_output"]
                        canonical_weighted = actual_normalized * sd["linear_attn.norm.weight"].reshape(1, 1, 128)
                        comparisons["gated_gamma_product_local_arithmetic"] = compare(
                            canonical_weighted, tt_values["gated_weighted_output"]
                        )
                        if "gated_silu_product" in tt_values:
                            canonical_product = (
                                tt_values["gated_weighted_output"].float()
                                * torch.nn.functional.silu(tt_values["projection_z"].reshape(1, 32, 128).float())
                            ).bfloat16()
                            comparisons["gated_silu_product_local_arithmetic"] = compare(
                                canonical_product, tt_values["gated_silu_product"]
                            )
                            comparisons["gated_head_merge"] = compare(
                                tt_values["gated_silu_product"], tt_values["gated_core"]
                            )
                comparisons["router_projection_local_arithmetic"] = compare(cpu_logits, tt_logits)

                def route(logits):
                    probabilities = logits.float().softmax(-1)
                    scores, ids = probabilities.topk(8, dim=-1)
                    scores = (scores / scores.sum(-1, keepdim=True)).to(logits.dtype)
                    sorted_logits = logits.float().sort(descending=True).values.reshape(-1)
                    return dict(
                        ids=ids.flatten().tolist(),
                        scores=scores.flatten().tolist(),
                        eighth_ninth_logit_gap=float(sorted_logits[7] - sorted_logits[8]),
                    )

                def top(logits):
                    values, ids = torch.topk(logits.float().reshape(1, -1), 16, dim=-1)
                    return dict(ids=ids.flatten().tolist(), logits=values.flatten().tolist())

                row = dict(
                    position=position,
                    pcc=pcc(ref, got),
                    contribution_pcc=pcc(ref - x, got - x),
                    oracle_output_bit_equal=True,
                    intermediates=comparisons,
                    hf_router=top(hf_logits),
                    tt_router=top(tt_logits),
                    cpu_on_tt_postnorm=top(cpu_logits),
                    hf_ids=hf_ids.flatten().tolist(),
                    native_ids=native_ids.tolist(),
                    native_scores=native_scores.tolist(),
                    router_backend="inline_device_selector"
                    if old_inline_selector is not None
                    else "generalized_moe_gate",
                    hf_scores=hf_scores.flatten().tolist(),
                    routing_boundary={
                        "canonical_from_hf_logits": route(hf_logits),
                        "canonical_from_tt_logits": route(tt_logits),
                        "canonical_from_cpu_logits_on_tt_postnorm": route(cpu_logits),
                    },
                    round_state=tt_values["round_state"],
                    output_state_is_input=tt_values["output_state_is_input"],
                )
                path = Path(str(output) + f".pos{position}.pt")
                torch.save(
                    dict(
                        hf_values=hf_values,
                        tt_values=tt_values,
                        ref=ref.clone(),
                        got=got,
                        token=x.clone(),
                        layer=args.layer,
                        position=position,
                        seed=args.seed,
                        stream_sha256=report["stream_sha256"],
                        source_sha256=report["source_sha256"],
                        capture_policy=report["capture_policy"],
                        measured_decoder_source=report["measured_decoder_source"],
                        measured_decoder_source_sha256=report["measured_decoder_source_sha256"],
                    ),
                    path,
                )
                row["tensor_capture"] = str(path)
                row["tensor_capture_sha256"] = digest(path)
                report["rows"].append(row)
                output.write_text(json.dumps(report, indent=2) + "\n")
                print("STREAM_CAPTURE", position, row["contribution_pcc"], flush=True)
        finally:
            ttnn.unary_chain, ttnn.rms_norm, ttnn.multiply = old_chain, old_rms, old_multiply
    finally:
        ttnn.close_mesh_device(mesh)
        report["device_closed"] = True
        report["source_unchanged"] = all(digest(name) == value for name, value in report["source_sha256"].items())
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, help="Explicit isolated FusedDecoder source; default uses current production module"
    )
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--positions", type=int, nargs="+", default=[137, 165, 181])
    parser.add_argument(
        "--include-worst",
        action="store_true",
        help="Also replay through the worst row in --failure-json; default stops at the last requested position",
    )
    parser.add_argument(
        "--oracle-cache", default=str(ROOT / "doc/fused_decoder/oracle_cache/v5_layer34_seed154_8192.pt")
    )
    parser.add_argument("--failure-json", default=str(EVIDENCE / "v5_representative_layer34_stress_seed154.json"))
    parser.add_argument("--output", default=str(EVIDENCE / "v5_layer34_stream_probe.json"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Validate saved stream/source identity without TTNN or device imports"
    )
    run(parser.parse_args())
