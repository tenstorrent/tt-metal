# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Instrumented layer0 batch32 failures, preserving probe_batch's RNG order.

Requests3/23 have independent HF/device recurrent states. All32 requests are
generated in the original order: prompt randn, decode randn, randint IDs, HF
prefill, HF decode. Token IDs consume RNG even for layer0. Model construction
and rotary setup also match probe_batch. This is a diagnostic with deliberate
HF injections/readbacks, not runtime-audit, tracing or performance evidence.
No runtime source is modified. --cpu-only never imports TTNN.

Optional gdn_nonlinears_cpu_{qkv,a,b,z,out,all} variants replace only the
named GDN projection with literal CPU F.linear on that call's actual device
input and the unchanged HF BF16 weight. They retain precise GR/router and
FP32-then-BF16 GDN nonlinear controls. These deliberate host fallbacks are
causal diagnostics, never acceptance evidence or candidate runtime fixes.
--gdn-linear-controls records same-input native projection errors without
injecting CPU outputs. Existing commands do not enable this extra work.
Suffixes on gdn_nonlinears_cpu_all select additional causal boundaries:
_exact_attn_mixed, _same_norm, _exact_hf_norm, _recurrence, _norm_recurrence.
Same-norm and recurrence controls consume the actual operands in that variant;
exact-HF-norm instead injects the original HF trajectory and is labeled as such.
gdn_cpu_full replaces GDN with the literal HF module on the actual mixed input
and actual prefix state. All these substitutions remain diagnostic-only.
"""

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_decoder_intermediates import metrics


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def hf_read_parts(module, hidden):
    normalized = module.hc_norm(hidden)
    down = module.input_mix_weight_down(normalized)
    scaled = down / module.hc_count
    activated = F.silu(scaled)
    up = module.input_mix_weight_up(activated)
    mixing = torch.sigmoid(up)
    product = mixing * normalized
    mixed = product.unflatten(-1, (module.hc_count, module.hidden_size)).mean(-2)
    injection_logits = module.block_inject_weight(normalized)
    injection_scaled = injection_logits / module.hc_count
    injection_sigmoid = torch.sigmoid(injection_scaled)
    injection = 2 * injection_sigmoid
    return dict(
        normalized=normalized,
        down=down,
        scaled=scaled,
        activated=activated,
        up=up,
        mixing=mixing,
        product=product,
        mixed=mixed,
        injection_logits=injection_logits,
        injection_scaled=injection_scaled,
        injection_sigmoid=injection_sigmoid,
        injection=injection,
    )


@torch.inference_mode()
def fixture(snapshot, selected, *, synthetic=False):
    torch.set_num_threads(8)
    torch.manual_seed(1981)
    if synthetic:
        from models.autoports.qwen_qwen3_8_flash_next.tests.synthetic_weights import layer_weights, target_config

        config, weights = target_config(), layer_weights(0)
    else:
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
        config._attn_implementation = "eager"
        weights = load_weights(snapshot, 0, "")
    with torch.device("meta"):
        reference = hf.Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    # This constructor is present in probe_batch even for linear_attention.
    hf.Qwen4ExpTextRotaryEmbedding(config)
    captured, enabled = {}, False

    def read_hook(kind):
        def hook(module, inputs, output):
            if not enabled:
                return
            captured[kind + "_mixed"] = output[0].clone()
            captured[kind + "_injection"] = output[2].clone()
            captured[kind + "_read_parts"] = {k: v.clone() for k, v in hf_read_parts(module, inputs[0]).items()}
            assert torch.equal(captured[kind + "_read_parts"]["mixed"], output[0])
            assert torch.equal(captured[kind + "_read_parts"]["injection"], output[2])
            if kind == "mlp":
                captured["post_attention"] = inputs[0].clone()

        return hook

    def output_hook(name):
        def hook(_module, _inputs, output):
            if enabled:
                captured[name] = output.clone()

        return hook

    def norm_hook(_module, inputs, output):
        if enabled:
            captured["gdn_norm_input"] = inputs[0].clone().reshape(1, -1, 6144)
            captured["gdn_norm_gate"] = inputs[1].clone().reshape(1, -1, 6144)
            captured["gdn_norm_output"] = output.clone().reshape(1, -1, 6144)

    def router_hook(_module, _inputs, output):
        if enabled:
            for name, value in zip(("router_logits", "router_scores", "router_indices"), output):
                captured[name] = value.clone()

    hooks = [
        reference.attn_hyper_connection.register_forward_hook(read_hook("attn")),
        reference.linear_attn.register_forward_hook(output_hook("gdn_output")),
        reference.linear_attn.norm.register_forward_hook(norm_hook),
        reference.mlp_hyper_connection.register_forward_hook(read_hook("mlp")),
        reference.mlp.register_forward_hook(output_hook("moe_output")),
        reference.mlp.gate.register_forward_hook(router_hook),
    ]
    requests, sequence_hashes = {}, []
    for row in range(1 if synthetic else 32):
        count = 1 + row % 3
        if synthetic:
            # Exactly test_synthetic_decoder's layer0/S1 seed, BF16 sampling,
            # and literal prefill1 -> three separate decode calls.
            hidden = torch.empty(1, 4, 10240, dtype=torch.bfloat16).normal_(
                mean=0, std=0.02, generator=torch.Generator().manual_seed(1901)
            )
            prompt, decode = hidden[:, :1], hidden[:, 1:]
            ids = torch.empty((1, 0), dtype=torch.int64)
            phase_inputs = [("prefill", prompt), *[(f"decode{i}", hidden[:, i + 1 : i + 2]) for i in range(3)]]
        else:
            prompt = (torch.randn(1, count, 10240) * 0.02).bfloat16()
            decode = (torch.randn(1, 1, 10240) * 0.02).bfloat16()
            ids = torch.randint(0, config.vocab_size, (1, count + 1))
            phase_inputs = (("prefill", prompt), ("decode", decode))
        sequence_hashes.append(dict(request=row, prompt=digest(prompt), decode=digest(decode), token_ids=digest(ids)))
        enabled = row in selected
        cache = DynamicCache(config=config)
        phases = {}
        for phase, hidden in phase_inputs:
            captured = {}
            rng = torch.get_rng_state().clone()
            output = reference(hidden, position_embeddings=(None, None), past_key_values=cache)
            assert torch.equal(rng, torch.get_rng_state()), "HF forward unexpectedly consumes RNG"
            if enabled:
                phases[phase] = {
                    **captured,
                    "input": hidden,
                    "output": output.clone(),
                    "conv_state": cache.get_conv_state(0).clone(),
                    "recurrent_state": cache.get_recurrent_state(0).clone(),
                }
        if enabled:
            requests[row] = {"phases": phases, "token_ids": ids}
    for hook in hooks:
        hook.remove()
    return config, reference, weights, requests, sequence_hashes


def cpu_summary(requests):
    result = {}
    for request, item in requests.items():
        result[request] = {"token_ids": item["token_ids"].tolist(), "phases": {}}
        for phase, values in item["phases"].items():
            logits = values["router_logits"].float()
            ordered, indices = logits.topk(12, dim=-1)
            result[request]["phases"][phase] = {
                "tokens": values["input"].shape[1],
                "input_sha256": digest(values["input"]),
                "output_sha256": digest(values["output"]),
                "hf_experts": values["router_indices"].tolist(),
                "top12_logit_experts": indices.tolist(),
                "top12_logits": ordered.tolist(),
                "logit_boundary_gap": (ordered[:, 9] - ordered[:, 10]).tolist(),
            }
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", default="3,23")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Exact offline layer0/S1 seed1901 CI fixture")
    parser.add_argument(
        "--precise-router", action="store_true", help="Hold the test-local SFPU router fixed in every variant"
    )
    parser.add_argument(
        "--variants",
        default="baseline,fp32_gr_nonlinears,fp32_gr_product_mean,fp32_gr_all,exact_attn_mixed,exact_attn_read_both,exact_gdn_output,exact_post_attention,exact_mlp_mixed,exact_hf_prefix_state,exact_attn_and_prefix",
    )
    parser.add_argument("--phases", help="Optional contiguous phase prefix; default runs all phases")
    parser.add_argument(
        "--gdn-core-fixture-dir", type=Path, help="Optional external diagnostic recurrence operand cache"
    )
    parser.add_argument(
        "--gdn-linear-controls",
        action="store_true",
        help="Diagnostic CPU F.linear comparisons on actual device operands for all five GDN projections",
    )
    parser.add_argument(
        "--gdn-sfpu-recurrence", action="store_true", help="Test model-local all-FP32 SFPU dot products"
    )
    parser.add_argument("--gdn-projection-block", type=int, choices=(32, 64, 256), default=256)
    args = parser.parse_args()
    if args.snapshot is None and not args.synthetic:
        parser.error("--snapshot is required for the real-weight fixture")
    selected = (0,) if args.synthetic else tuple(map(int, args.requests.split(",")))
    assert selected and all(0 <= request < 32 for request in selected)
    variants = tuple(args.variants.split(","))
    assert variants[0] == "baseline", "Baseline must run first to preserve an independent TT prefix"
    cpu_projection_variants = {"gdn_nonlinears_cpu_" + name: name for name in ("qkv", "a", "b", "z", "out", "all")}
    for names in (("qkv", "out"), ("qkv", "z"), ("qkv", "z", "out"), ("qkv", "a")):
        cpu_projection_variants["gdn_nonlinears_cpu_" + "_".join(names)] = "+".join(names)
    cpu_projection_variants["gdn_cpu_qkv_prefill_only"] = "qkv"
    cpu_projection_variants["gdn_cpu_qkv_decode_only"] = "qkv"
    for suffix in (
        "exact_attn_mixed",
        "same_norm",
        "exact_hf_norm",
        "recurrence",
        "norm_recurrence",
        "recurrence_exact_attn_mixed",
    ):
        cpu_projection_variants["gdn_nonlinears_cpu_all_" + suffix] = "all"
    recurrence_only_variants = {"gdn_cpu_recurrence", "gdn_cpu_recurrence_exact_attn_mixed"}
    full_cpu_variants = {"gdn_cpu_full", "gdn_cpu_full_exact_attn_mixed"}
    exact_prefix_variants = {
        "exact_hf_prefix_state",
        "exact_attn_and_prefix",
        "exact_hf_conv_prefix",
        "exact_hf_recurrent_prefix",
    }
    for variant in variants:
        if (
            "_cpu_" in variant
            and variant not in cpu_projection_variants.keys() | recurrence_only_variants | full_cpu_variants
        ):
            parser.error(f"Unknown CPU projection diagnostic variant: {variant}")
    fp32pack_variants = {"gdn_fp32pack_" + name: name for name in ("qkv", "a", "b", "z", "out", "all")}
    precise_gdn_variants = {"gdn_precise_qkv_out": {"qkv", "out"}, "gdn_precise_all": {"qkv", "a", "b", "z", "out"}}
    split_gdn_variants = {"gdn_split32_qkv_out": 32, "gdn_split128_qkv_out": 128}
    linear_controls_enabled = args.gdn_linear_controls or any(
        v in cpu_projection_variants or v in fp32pack_variants or v in precise_gdn_variants or v in split_gdn_variants
        for v in variants
    )
    config, reference, weights, requests, sequence_hashes = fixture(args.snapshot, selected, synthetic=args.synthetic)
    report = {
        "scope": __doc__,
        "seed": 1901 if args.synthetic else 1981,
        "layer": 0,
        "batch_fixture": 1 if args.synthetic else 32,
        "weights": "deterministic synthetic CI fixture" if args.synthetic else "real checkpoint",
        "all_variants_precise_router": args.precise_router,
        "runtime_gr_projections": "precise SFPU down/up; native injection",
        "runtime_router_projection": "precise SFPU",
        "gdn_linear_controls_enabled": linear_controls_enabled,
        "gdn_sfpu_recurrence": args.gdn_sfpu_recurrence,
        "gdn_projection_control_block": args.gdn_projection_block,
        "gdn_projection_runtime": "QKV/output SFPU tree, BF16-resident32-output blocks; others native",
        "cpu_projection_injections_are_diagnostic_only": True,
        "cpu_threads": 8,
        "snapshot": str(args.snapshot),
        "hf_source_sha256": hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),
        "runtime_components_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "tt/decoder_components.py").read_bytes()
        ).hexdigest(),
        "rng_forward_unchanged": True,
        "all32_fixture_hashes": sequence_hashes,
        "cpu_reference": cpu_summary(requests),
        "device_executed": not args.cpu_only,
        "requests": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("BATCH_INTERMEDIATE_HF_READY", selected, flush=True)
    if args.cpu_only:
        assert "ttnn" not in sys.modules
        return
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt import decoder_components as components
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import (
        GatedDeltaNet,
        GatedResidual,
        SparseMoe,
        WeightLoader,
        linear,
        rms,
    )
    from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import aten_router_topk

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        attn_mix = GatedResidual(loader, "attn_hyper_connection.", config)
        mlp_mix = GatedResidual(loader, "mlp_hyper_connection.", config)
        gdn, moe = GatedDeltaNet(loader, config), SparseMoe(loader, config, expert_dtype=ttnn.bfloat16)
        gdn_projection_weights = {
            name: (getattr(gdn, name), weights["linear_attn." + checkpoint + ".weight"])
            for name, checkpoint in (
                ("qkv", "in_proj_qkv"),
                ("a", "in_proj_a"),
                ("b", "in_proj_b"),
                ("z", "in_proj_z"),
                ("out", "out_proj"),
            )
        }

        # Native duplicates are diagnostic-only, allocated only for explicit
        # FPU packing/split counterfactuals after the runtime precision repair.
        native_gdn_matrices = {name: matrix for name, (matrix, _) in gdn_projection_weights.items()}
        if any(v in fp32pack_variants or v in split_gdn_variants for v in variants):
            native_gdn_matrices["qkv"] = loader.matrix("linear_attn.in_proj_qkv.weight")
            native_gdn_matrices["out"] = loader.matrix("linear_attn.out_proj.weight")

        precise_router = moe.precise_router
        native_router = loader.matrix("mlp.gate.weight", ttnn.float32)  # Diagnostic counterfactual only.

        class PreciseGRProjection:
            """Test-local balanced SFPU dot products at the actual GR shapes."""

            def __init__(self, name):
                weight = weights[name]
                self.output_width, self.input_width = weight.shape
                assert (self.output_width, self.input_width) in (
                    (320, 10240),
                    (10240, 320),
                    (4, 10240),
                    (10240, 2560),
                    (6144, 2560),
                    (48, 2560),
                    (2560, 6144),
                )
                self.block = 32 if self.input_width == 10240 else 256
                if name.startswith("linear_attn."):
                    self.block = args.gdn_projection_block
                self.weights = []
                for start in range(0, self.output_width, self.block):
                    part = weight[start : start + self.block]
                    if part.shape[0] < self.block:
                        part = F.pad(part, (0, 0, 0, self.block - part.shape[0]))
                    self.weights.append(loader.tensor(part.reshape(1, 1, self.block, self.input_width), ttnn.float32))

            def __call__(self, value):
                count = value.shape[-2]
                assert tuple(value.shape) == (1, 1, count, self.input_width)
                expanded = ttnn.reshape(ttnn.typecast(value, ttnn.float32), (1, count, 1, self.input_width))
                columns = []
                for weight in self.weights:
                    products = ttnn.multiply(expanded, weight)
                    width, tails = self.input_width, []
                    while width > 1:
                        half = width // 2
                        if width % 2:
                            tails.append(products[..., -1:])
                        products = ttnn.add(products[..., :half], products[..., half : 2 * half])
                        width = half
                    for tail in tails:
                        products = ttnn.add(products, tail)
                    columns.append(ttnn.reshape(products, (1, 1, count, self.block)))
                result = ttnn.concat(columns, dim=-1) if len(columns) > 1 else columns[0]
                return ttnn.typecast(result[..., : self.output_width], ttnn.bfloat16)

        precise_gdn_names = set().union(*(precise_gdn_variants.get(v, set()) for v in variants))
        precise_gdn = {
            name: PreciseGRProjection("linear_attn." + checkpoint + ".weight")
            for name, checkpoint in (
                ("qkv", "in_proj_qkv"),
                ("a", "in_proj_a"),
                ("b", "in_proj_b"),
                ("z", "in_proj_z"),
                ("out", "out_proj"),
            )
            if name in precise_gdn_names
        }
        split_gdn_weights = {
            (size, name): [matrix[offset : offset + size, :] for offset in range(0, matrix.shape[0], size)]
            for size in {split_gdn_variants[v] for v in variants if v in split_gdn_variants}
            for name, matrix in native_gdn_matrices.items()
            if name in ("qkv", "out")
        }

        gr_projections = {}
        if any("precise_mlp_gr" in variant or "precise_both_gr" in variant for variant in variants):
            for label, prefix in (("attn", "attn_hyper_connection."), ("mlp", "mlp_hyper_connection.")):
                gr_projections[label] = {
                    name: PreciseGRProjection(prefix + checkpoint_name + ".weight")
                    for name, checkpoint_name in (
                        ("down", "input_mix_weight_down"),
                        ("up", "input_mix_weight_up"),
                        ("inject", "block_inject_weight"),
                    )
                }

        def read_parts(component, value, policy):
            label = "attn" if component is attn_mix else "mlp"
            precise_gr = "precise_both_gr" in policy or (label == "mlp" and "precise_mlp_gr" in policy)
            count = value.shape[-2]
            grouped = ttnn.permute(ttnn.reshape(value, (1, count, component.streams, component.width)), (0, 2, 1, 3))
            normalized_grouped = rms(grouped, component.weight, component.eps)
            normalized = ttnn.reshape(
                ttnn.permute(normalized_grouped, (0, 2, 1, 3)), (1, 1, count, component.streams * component.width)
            )
            # Current runtime evaluates nonlinear/reduction arithmetic in
            # FP32 and rounds back at each HF BF16 boundary. Keep the old
            # path only as an explicitly named diagnostic negative control.
            precise_nonlinears = policy != "legacy_gr_native"
            precise_mean = policy != "legacy_gr_native"
            precise_arithmetic = policy in ("fp32_gr_product_mean", "fp32_gr_all")

            def multiply(a, b):
                if not precise_arithmetic:
                    return ttnn.multiply(a, b)
                lhs = ttnn.typecast(a, ttnn.float32)
                rhs = ttnn.typecast(b, ttnn.float32) if isinstance(b, ttnn.Tensor) else b
                return ttnn.typecast(ttnn.multiply(lhs, rhs), ttnn.bfloat16)

            def activation(function, value):
                return (
                    ttnn.typecast(function(ttnn.typecast(value, ttnn.float32)), ttnn.bfloat16)
                    if precise_nonlinears
                    else function(value)
                )

            down = gr_projections[label]["down"](normalized) if precise_gr else component.precise_down(normalized)
            scaled = multiply(down, 1 / component.streams)
            activated = activation(ttnn.silu, scaled)
            up = gr_projections[label]["up"](activated) if precise_gr else component.precise_up(activated)
            mixing = activation(ttnn.sigmoid, up)
            # Preserve the runtime's grouped product layout to avoid changing
            # its kernel specialization in the baseline diagnostic.
            mixing_grouped = ttnn.permute(
                ttnn.reshape(mixing, (1, count, component.streams, component.width)), (0, 2, 1, 3)
            )
            product_grouped = multiply(mixing_grouped, normalized_grouped)
            mean_input = ttnn.typecast(product_grouped, ttnn.float32) if precise_mean else product_grouped
            mixed = ttnn.mean(mean_input, dim=1, keepdim=True)
            if precise_mean:
                mixed = ttnn.typecast(mixed, ttnn.bfloat16)
            product = ttnn.reshape(
                ttnn.permute(product_grouped, (0, 2, 1, 3)), (1, 1, count, component.streams * component.width)
            )
            injection_logits = (
                gr_projections[label]["inject"](normalized)
                if precise_gr and policy.endswith("_injection")
                else linear(normalized, component.inject, component.compute)
            )
            injection_scaled = multiply(injection_logits, 1 / component.streams)
            injection_sigmoid = activation(ttnn.sigmoid, injection_scaled)
            injection_flat = multiply(injection_sigmoid, 2)
            injection = ttnn.permute(ttnn.reshape(injection_flat, (1, 1, count, component.streams)), (0, 3, 2, 1))
            parts = dict(
                normalized=normalized,
                down=down,
                scaled=scaled,
                activated=activated,
                up=up,
                mixing=mixing,
                product=product,
                mixed=mixed,
                injection_logits=injection_logits,
                injection_scaled=injection_scaled,
                injection_sigmoid=injection_sigmoid,
                injection=injection_flat,
            )
            return mixed, injection, parts

        def host(value, expected):
            return ttnn.to_torch(value).reshape(expected.shape)

        def exact(value, injection=False):
            return loader.tensor(value.unsqueeze(-1).permute(0, 2, 1, 3) if injection else value.unsqueeze(0))

        def router_report(value, expected, precise=False):
            logits = (
                ttnn.typecast(precise_router(value), ttnn.float32)
                if precise
                else linear(ttnn.typecast(value, ttnn.float32), native_router, loader.compute)
            )
            expected_sets = expected["router_indices"].sort(-1).values
            result = {}
            for rounding in (False, True):
                routed = ttnn.typecast(ttnn.typecast(logits, ttnn.bfloat16), ttnn.float32) if rounding else logits
                _, indices = aten_router_topk(routed)
                _, native_indices = ttnn.topk(routed, k=10, dim=-1, sorted=True)
                actual_logits = ttnn.to_torch(routed).reshape(-1, 512)
                actual_indices = ttnn.to_torch(indices).reshape(-1, 10).long()
                native_indices = ttnn.to_torch(native_indices).reshape(-1, 10).long()
                cpu_indices = actual_logits.softmax(-1).topk(10, dim=-1).indices
                changed = torch.where((actual_indices.sort(-1).values != expected_sets).any(-1))[0].tolist()
                details = []
                for row in range(actual_indices.shape[0]):
                    experts = torch.unique(torch.cat((expected["router_indices"][row], actual_indices[row]))).long()
                    details.append(
                        dict(
                            row=row,
                            hf_indices=expected["router_indices"][row].tolist(),
                            tt_indices=actual_indices[row].tolist(),
                            expert_ids=experts.tolist(),
                            hf_logits=expected["router_logits"][row, experts].float().tolist(),
                            tt_logits=actual_logits[row, experts].tolist(),
                        )
                    )
                result["bf16_rounded" if rounding else "runtime_fp32"] = dict(
                    logits=metrics(expected["router_logits"], actual_logits),
                    changed_rows=changed,
                    details=details,
                    all_logits=actual_logits.tolist(),
                    cpu_topk_on_same_logits=cpu_indices.tolist(),
                    cpu_topk_changed_rows=torch.where((cpu_indices.sort(-1).values != expected_sets).any(-1))[
                        0
                    ].tolist(),
                    cpu_vs_tt_topk_changed_rows=torch.where(
                        (cpu_indices.sort(-1).values != actual_indices.sort(-1).values).any(-1)
                    )[0].tolist(),
                    native_tt_topk_indices=native_indices.tolist(),
                    selection_backend="model-local ATen-equivalent topk",
                    projection_backend="precise_SFPU_BF16_output" if precise else "native_FP32_matmul",
                )
            # An exact logit injection separates top-k tie policy from matmul
            # and all upstream differences. This is a diagnostic boundary.
            exact_logits = loader.tensor(expected["router_logits"].float().reshape(1, 1, -1, 512), ttnn.float32)
            _, exact_indices = ttnn.topk(exact_logits, k=10, dim=-1, sorted=True)
            exact_indices = ttnn.to_torch(exact_indices).reshape(-1, 10).long()
            result["tt_topk_on_exact_hf_logits"] = {
                "indices": exact_indices.tolist(),
                "changed_rows": torch.where((exact_indices.sort(-1).values != expected_sets).any(-1))[0].tolist(),
            }
            hf_on_tt = reference.mlp.gate(host(value, expected["mlp_mixed"]).reshape(-1, 2560))
            result["hf_router_on_tt_input_changed_rows"] = torch.where(
                (hf_on_tt[2].sort(-1).values != expected_sets).any(-1)
            )[0].tolist()
            result["hf_router_on_tt_input"] = {
                "indices": hf_on_tt[2].tolist(),
                "logits": hf_on_tt[0].float().tolist(),
                "logits_metrics": metrics(expected["router_logits"], hf_on_tt[0]),
            }
            return result

        for request, item in requests.items():
            phases, prefixes = item["phases"], {}
            request_report = report["requests"][request] = {}
            phase_names = list(phases)
            if args.phases is not None:
                requested_phases = args.phases.split(",")
                if requested_phases != phase_names[: len(requested_phases)]:
                    raise ValueError("--phases must select a contiguous prefix of the literal HF phase sequence")
                phase_names = requested_phases
            for phase_index, phase in enumerate(phase_names):
                previous_phase = phase_names[phase_index - 1] if phase_index else None
                expected = phases[phase]
                phase_report = request_report[phase] = {}
                value = exact(expected["input"])
                exact_values = {
                    name: exact(expected[name], name.endswith("injection"))
                    for name in ("attn_mixed", "attn_injection", "gdn_output", "post_attention", "mlp_mixed")
                }
                for variant in variants:
                    if phase == "prefill" and variant in exact_prefix_variants:
                        continue
                    if phase == "prefill":
                        conv = ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                        state = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
                        prefix_policy = "empty"
                    elif variant in exact_prefix_variants:
                        conv, state = (ttnn.clone(tensor) for tensor in prefixes[(previous_phase, "baseline")])
                        if variant != "exact_hf_recurrent_prefix":
                            conv = loader.tensor(
                                phases[previous_phase]["conv_state"][:, :, -3:].transpose(1, 2).unsqueeze(0)
                            )
                        if variant != "exact_hf_conv_prefix":
                            state = loader.tensor(phases[previous_phase]["recurrent_state"], ttnn.float32)
                        prefix_policy = variant
                    else:
                        prefix_policy = (
                            variant
                            if variant.startswith(("fp32_gr_", "precise_both_gr")) or "gdn_" in variant
                            else "baseline"
                        )
                        conv, state = (ttnn.clone(tensor) for tensor in prefixes[(previous_phase, prefix_policy)])
                    mixed, injection, attn_parts = read_parts(attn_mix, value, variant)
                    read_runtime_exact = None
                    if variant == "baseline":
                        runtime_mixed, runtime_injection = attn_mix.read(value)
                        read_runtime_exact = torch.equal(
                            ttnn.to_torch(mixed), ttnn.to_torch(runtime_mixed)
                        ) and torch.equal(ttnn.to_torch(injection), ttnn.to_torch(runtime_injection))
                        assert read_runtime_exact, "Instrumented read differs from actual runtime read"
                    if (
                        variant in ("exact_attn_mixed", "exact_attn_read_both", "exact_attn_and_prefix")
                        or "exact_attn_mixed" in variant
                    ):
                        mixed = exact_values["attn_mixed"]
                    if variant == "exact_attn_read_both":
                        injection = exact_values["attn_injection"]
                    internals = {}
                    nonlinear_controls = {}
                    projection_controls = {}
                    injected_projection = cpu_projection_variants.get(variant)
                    if (variant == "gdn_cpu_qkv_prefill_only" and phase != "prefill") or (
                        variant == "gdn_cpu_qkv_decode_only" and phase == "prefill"
                    ):
                        injected_projection = None
                    deep_controls = {}
                    deep_diagnostic = injected_projection == "all" or variant in recurrence_only_variants
                    original_linear, original_recurrence = components.linear, components.fp32_recurrent_gated_delta_rule
                    if args.gdn_sfpu_recurrence:
                        original_recurrence = partial(original_recurrence, precise_dot_products=True)
                    original_silu, original_sigmoid = ttnn.silu, ttnn.sigmoid

                    def nonlinear_control(name, function, value, *arguments, **kwargs):
                        use_precise = "gdn_nonlinears" in variant or f"gdn_{name}" in variant
                        if value.dtype != ttnn.bfloat16:
                            return function(value, *arguments, **kwargs)
                        result = (
                            ttnn.typecast(
                                function(ttnn.typecast(value, ttnn.float32), *arguments, **kwargs), ttnn.bfloat16
                            )
                            if use_precise
                            else function(value, *arguments, **kwargs)
                        )
                        # Exact same-operand CPU control, deliberately outside
                        # acceptance/runtime-audit scope of this diagnostic.
                        host_value = ttnn.to_torch(value)
                        target = F.silu(host_value) if name == "silu" else torch.sigmoid(host_value)
                        nonlinear_controls[name] = {
                            "fp32_then_bf16": use_precise,
                            "same_operand_hf": metrics(target, ttnn.to_torch(result)),
                        }
                        return result

                    def silu_control(value, *arguments, **kwargs):
                        return nonlinear_control("silu", original_silu, value, *arguments, **kwargs)

                    def sigmoid_control(value, *arguments, **kwargs):
                        return nonlinear_control("sigmoid", original_sigmoid, value, *arguments, **kwargs)

                    def capture_linear(x, weight, compute):
                        if weight is gdn.out and deep_diagnostic:
                            norm_target = reference.linear_attn.norm(
                                internals["gdn_norm_input"].reshape(-1, gdn.dvalue),
                                internals["gdn_norm_gate"].reshape(-1, gdn.dvalue),
                            ).reshape(tuple(x.shape))
                            deep_controls["norm_same_operand_hf"] = metrics(norm_target, ttnn.to_torch(x))
                            if variant.endswith(("same_norm", "norm_recurrence")):
                                x = loader.tensor(norm_target)
                                deep_controls["norm_injection"] = "HF norm on actual recurrence output and actual z"
                            elif variant.endswith("exact_hf_norm"):
                                x = loader.tensor(expected["gdn_norm_output"].reshape(tuple(x.shape)))
                                deep_controls["norm_injection"] = "literal HF trajectory; NOT same operands"
                        result = original_linear(x, weight, compute)
                        if linear_controls_enabled:
                            matches = [
                                (name, hf_weight)
                                for name, (device_weight, hf_weight) in gdn_projection_weights.items()
                                if weight is device_weight
                            ]
                            assert len(matches) == 1, "Unexpected GDN linear projection"
                            name, hf_weight = matches[0]
                            # Preserve the literal HF operand dtype and BF16
                            # projection output boundary; do not substitute
                            # the original HF trajectory's input here.
                            operand = ttnn.to_torch(x).contiguous()
                            assert operand.dtype == hf_weight.dtype == torch.bfloat16
                            target = F.linear(operand, hf_weight)
                            native_host = ttnn.to_torch(result)
                            assert tuple(target.shape) == tuple(native_host.shape)
                            inject = injected_projection == "all" or name in (injected_projection or "").split("+")
                            control = {
                                "operand_source": "actual device input to this projection in this variant",
                                "operand_sha256": digest(operand),
                                "hf_weight_dtype": str(hf_weight.dtype),
                                "native_same_operand_hf": metrics(target, native_host),
                                "cpu_output_injected": inject,
                            }
                            if fp32pack_variants.get(variant) in (name, "all"):
                                result = ttnn.typecast(
                                    ttnn.linear(
                                        x,
                                        native_gdn_matrices[name],
                                        dtype=ttnn.float32,
                                        compute_kernel_config=compute,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    ),
                                    ttnn.bfloat16,
                                )
                                control["fp32_output_then_bf16_same_operand_hf"] = metrics(
                                    target, ttnn.to_torch(result)
                                )
                                control["device_fp32_pack_candidate_selected"] = True
                            if name in precise_gdn_variants.get(variant, set()):
                                result = precise_gdn[name](x)
                                control["sfpu_tree_same_operand_hf"] = metrics(target, ttnn.to_torch(result))
                                control["device_sfpu_tree_selected"] = True
                            if variant in split_gdn_variants and name in ("qkv", "out"):
                                size = split_gdn_variants[variant]
                                partials = [
                                    ttnn.linear(
                                        x[..., offset : offset + size],
                                        part,
                                        dtype=ttnn.float32,
                                        compute_kernel_config=compute,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    )
                                    for offset, part in zip(range(0, x.shape[-1], size), split_gdn_weights[size, name])
                                ]
                                while len(partials) > 1:
                                    paired = [
                                        ttnn.add(partials[i], partials[i + 1]) for i in range(0, len(partials) - 1, 2)
                                    ]
                                    partials = paired + (partials[-1:] if len(partials) % 2 else [])
                                result = ttnn.typecast(partials[0], ttnn.bfloat16)
                                control["split_k_tree_same_operand_hf"] = metrics(target, ttnn.to_torch(result))
                                control["device_split_k_size"] = size
                            if inject:
                                result = loader.tensor(target, ttnn.bfloat16)
                                roundtrip = ttnn.to_torch(result)
                                assert torch.equal(target, roundtrip), "CPU injection changed BF16 output bits"
                                control["injection_roundtrip_exact"] = True
                            projection_controls[name] = control
                        if weight is gdn.z:
                            internals["gdn_norm_gate"] = ttnn.to_torch(result).reshape(1, -1, 6144)
                        elif weight is gdn.out:
                            internals["gdn_norm_output"] = ttnn.to_torch(x).reshape(1, -1, 6144)
                        return result

                    def capture_recurrence(*inputs, **kwargs):
                        if deep_diagnostic:
                            q, k, v, decay, beta = [ttnn.to_torch(t).clone() for t in inputs]
                            initial = ttnn.to_torch(kwargs["initial_state"]).clone()
                            repeat = v.shape[2] // q.shape[2]
                            q, k = (t.repeat_interleave(repeat, dim=2) for t in (q, k))
                            cpu_output, cpu_state = hf.torch_recurrent_gated_delta_rule(
                                q,
                                k,
                                v,
                                decay,
                                beta,
                                initial_state=initial,
                                output_final_state=True,
                                use_qk_l2norm_in_kernel=False,
                            )
                        result, final = original_recurrence(*inputs, **kwargs)
                        if deep_diagnostic:
                            deep_controls["recurrence_same_operand_hf"] = {
                                "output": metrics(cpu_output, ttnn.to_torch(result)),
                                "state": metrics(cpu_state, ttnn.to_torch(final)),
                                "scope": "HF recurrent rule on actual normalized q/k, v, decay, beta, initial state; no renormalization",
                            }
                            if args.gdn_core_fixture_dir is not None:
                                args.gdn_core_fixture_dir.mkdir(parents=True, exist_ok=True)
                                fixture_path = args.gdn_core_fixture_dir / f"request{request}_{phase}_{variant}.pt"
                                torch.save(
                                    {
                                        "query": q,
                                        "key": k,
                                        "value": v,
                                        "decay": decay,
                                        "beta": beta,
                                        "initial_state": initial,
                                        "hf_output": cpu_output,
                                        "hf_state": cpu_state,
                                        "tt_output": ttnn.to_torch(result),
                                        "tt_state": ttnn.to_torch(final),
                                        "runtime_components_sha256": report["runtime_components_sha256"],
                                        "hf_source_sha256": report["hf_source_sha256"],
                                        "normalized_qk": True,
                                        "query_scaling_already_applied": False,
                                    },
                                    fixture_path,
                                )
                                deep_controls["recurrence_fixture"] = str(fixture_path)
                            if "recurrence" in variant:
                                result = loader.tensor(cpu_output, ttnn.float32)
                                final = loader.tensor(cpu_state, ttnn.float32)
                                deep_controls["recurrence_cpu_injected"] = True
                        internals["gdn_norm_input"] = ttnn.to_torch(result).bfloat16().reshape(1, -1, 6144)
                        return result, final

                    cpu_full = None
                    if variant in full_cpu_variants:
                        cpu_cache = DynamicCache(config=config)
                        if previous_phase is not None:
                            prior_conv = ttnn.to_torch(conv).squeeze(0).transpose(1, 2)
                            cpu_cache.update_conv_state(F.pad(prior_conv, (1, 0)), 0, conv_kernel_size=4)
                            cpu_cache.update_recurrent_state(ttnn.to_torch(state).clone(), 0)
                        cpu_full = reference.linear_attn(
                            ttnn.to_torch(mixed).reshape(1, -1, config.hidden_size), cache_params=cpu_cache
                        )
                    with (
                        mock.patch.object(components, "linear", capture_linear),
                        mock.patch.object(components, "fp32_recurrent_gated_delta_rule", capture_recurrence),
                        mock.patch.object(ttnn, "silu", silu_control),
                        mock.patch.object(ttnn, "sigmoid", sigmoid_control),
                    ):
                        block = gdn(mixed, conv_state=conv, recurrent_state=state)
                    if cpu_full is not None:
                        deep_controls["full_gdn_same_operand_hf"] = metrics(cpu_full, ttnn.to_torch(block))
                        deep_controls["full_gdn_cpu_injected"] = True
                        block = loader.tensor(cpu_full.unsqueeze(0))
                        ttnn.copy(
                            loader.tensor(cpu_cache.get_conv_state(0)[:, :, -3:].transpose(1, 2).unsqueeze(0)), conv
                        )
                        ttnn.copy(loader.tensor(cpu_cache.get_recurrent_state(0), ttnn.float32), state)
                    prefixes[(phase, variant)] = (ttnn.clone(conv), ttnn.clone(state))
                    if variant == "exact_gdn_output":
                        block = exact_values["gdn_output"]
                    updated = attn_mix.write(value, block, injection)
                    if variant.startswith("exact_post_attention"):
                        updated = exact_values["post_attention"]
                    router_input, mlp_injection, mlp_parts = read_parts(mlp_mix, updated, variant)
                    if variant in ("exact_mlp_mixed", "exact_mlp_mixed_precise_router"):
                        router_input = exact_values["mlp_mixed"]
                    use_precise_router = variant != "native_router_control"

                    def chosen_router_logits(x):
                        if use_precise_router:
                            return ttnn.typecast(precise_router(x), ttnn.float32)
                        native = original_linear(ttnn.typecast(x, ttnn.float32), native_router, loader.compute)
                        return ttnn.typecast(ttnn.typecast(native, ttnn.bfloat16), ttnn.float32)

                    with mock.patch.object(moe, "router_logits", chosen_router_logits):
                        moe_output = moe(router_input)
                    final = mlp_mix.write(updated, moe_output, mlp_injection)
                    counterparts = dict(
                        attn_mixed=mixed,
                        attn_injection=injection,
                        gdn_output=block,
                        post_attention=updated,
                        mlp_mixed=router_input,
                        mlp_injection=mlp_injection,
                        moe_output=moe_output,
                        output=final,
                    )
                    result = {
                        "prefix_policy": prefix_policy,
                        "instrumented_read_matches_runtime": read_runtime_exact,
                        "components": {
                            name: metrics(
                                expected[name],
                                ttnn.to_torch(tensor).permute(0, 2, 1, 3)
                                if name.endswith("injection")
                                else host(tensor, expected[name]),
                            )
                            for name, tensor in counterparts.items()
                        },
                        "output_contribution": metrics(
                            expected["output"].float() - expected["input"].float(),
                            host(final, expected["output"]).float() - expected["input"].float(),
                        ),
                        "gdn_internals": {name: metrics(expected[name], tensor) for name, tensor in internals.items()},
                        "gdn_updated_states": {
                            "conv": metrics(
                                expected["conv_state"][:, :, -3:].transpose(1, 2).unsqueeze(0), ttnn.to_torch(conv)
                            ),
                            "recurrent": metrics(expected["recurrent_state"], ttnn.to_torch(state)),
                        },
                        "gdn_nonlinear_same_operand_controls": nonlinear_controls,
                        "gdn_projection_same_operand_controls": projection_controls,
                        "diagnostic_cpu_projection_injection": injected_projection,
                        "gdn_deep_diagnostic_controls": deep_controls,
                        "read_parts": {
                            kind: {
                                name: metrics(
                                    expected[kind + "_read_parts"][name],
                                    host(tensor, expected[kind + "_read_parts"][name]),
                                )
                                for name, tensor in parts.items()
                            }
                            for kind, parts in (("attn", attn_parts), ("mlp", mlp_parts))
                        },
                        "router": router_report(router_input, expected, use_precise_router),
                    }
                    phase_report[variant] = result
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "request": request,
                                "phase": phase,
                                "variant": variant,
                                "contribution_pcc": result["output_contribution"]["minimum_row_pcc"],
                                "router_changed_rows": result["router"]["runtime_fp32"]["changed_rows"],
                            }
                        ),
                        flush=True,
                    )
                    ttnn.deallocate(conv)
                    ttnn.deallocate(state)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
