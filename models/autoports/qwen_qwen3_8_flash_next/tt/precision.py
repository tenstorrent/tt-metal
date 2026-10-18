# SPDX-License-Identifier: Apache-2.0
"""Constructible precision policy shared by full-model consumers.

Fixed native contracts are validated, never silently ignored. Explicit candidate
paths override the required stage selection; ``baseline`` restores stage7.
"""
import copy
import json
from pathlib import Path
import ttnn

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {
    "config_id": "baseline",
    "weight_groups": {
        "gdn_input": "bfloat8_b", "gdn_output": "bfloat8_b",
        "qsa_prefill": "bfloat8_b", "qsa_decode": "bfloat4_b", "qsa_output": "bfloat8_b",
        "shared_expert": "bfloat16", "hc": "bfloat8_b", "ple_projection": "bfloat8_b",
        "lm_head": "bfloat16", "routed_expert_gate_up": "bfloat4_b", "routed_expert_down": "bfloat4_b",
        "ple_tables": "bfloat4_b", "embedding": "bfloat16", "router": "bfloat16",
        "norms": "mixed_bfloat16_float32", "terminal_mixer": "bfloat16",
    },
    "compute_fidelities": {
        "gdn_input": "LoFi", "gdn_output": "LoFi", "qsa_prefill": "HiFi2", "qsa_decode": "LoFi",
        "qsa_output": "LoFi", "shared_expert": "HiFi2", "hc": "LoFi", "ple_projection": "HiFi2",
        "lm_head": "HiFi4", "routed_expert_gate_up": "LoFi", "routed_expert_down": "LoFi",
        "router": "HiFi2", "norms": "HiFi4", "terminal_mixer": "HiFi4",
        "qsa_attention": "HiFi4", "gdn_recurrence": "HiFi4",
    },
    "layer_exceptions": {},
    "qsa_execution": {
        "singleton_packed_fidelity_group": "qsa_decode",
        "batched_decode_weight_group": "qsa_prefill",
    },
    "activation_dtype": "bfloat16",
    "residual_dtype": "bfloat16",
    "ccl_dtype": "mixed",
    "kv_cache_dtype": "fp8_e4m3",
    "state_dtypes": {"raw_index": "bfloat16", "recurrent": "float32", "convolution": "bfloat16"},
    "logits_dtype": "bfloat16",
    "sampling": {"logits": "bfloat16", "scores": "bfloat16", "token_ids": "uint32", "max_top_k": 32},
    "routing": {"active_experts_per_token": 10, "experts_total": 512, "dispatch": "active_only"},
    "max_context": 262144,
}
VARIABLE_GROUPS = {"gdn_input", "gdn_output", "qsa_prefill", "qsa_decode", "qsa_output", "shared_expert", "hc", "ple_projection", "lm_head"}

def load_policy(value=None):
    if value is None:
        value = ROOT / "doc/datatype_sweep/selected_precision_config.json"
    if isinstance(value, dict):
        policy = copy.deepcopy(value)
    elif str(value) == "baseline":
        policy = copy.deepcopy(BASELINE)
    else:
        policy = json.loads(Path(value).read_text())
    if set(policy) != set(BASELINE):
        raise ValueError("Precision policy must contain exactly the complete schema")
    for kind in ("weight_groups", "compute_fidelities"):
        if policy[kind].keys() != BASELINE[kind].keys():
            raise ValueError(f"Incomplete {kind}")
        for group, setting in policy[kind].items():
            allowed = ("bfloat16", "bfloat8_b", "bfloat4_b") if kind == "weight_groups" else ("LoFi", "HiFi2", "HiFi4")
            if kind == "compute_fidelities" and group in ("routed_expert_gate_up", "routed_expert_down"):
                if setting not in ("LoFi", "HiFi2"):
                    raise ValueError("Routed experts support LoFi/HiFi2")
            elif group in VARIABLE_GROUPS:
                if setting not in allowed:
                    raise ValueError((kind, group, setting))
            elif setting != BASELINE[kind][group]:
                raise ValueError(f"Fixed native contract: {kind}.{group}={BASELINE[kind][group]}")
    if policy["compute_fidelities"]["routed_expert_gate_up"] != policy["compute_fidelities"]["routed_expert_down"]:
        raise ValueError("Native routed FFN has one shared fidelity, including gate product")
    for key in ("activation_dtype", "residual_dtype", "state_dtypes", "logits_dtype", "sampling", "routing", "max_context", "qsa_execution"):
        if policy[key] != BASELINE[key]:
            raise ValueError(f"Unsupported native contract {key}: {policy[key]}")
    if policy["ccl_dtype"] not in ("mixed", "bfloat8_b"):
        raise ValueError("CCL supports mixed baseline or BF8 activation transfers")
    if policy["kv_cache_dtype"] not in ("fp8_e4m3", "bfloat16"):
        raise ValueError("Packed sparse_sdpa accepts BF16 or FP8_E4M3, not tiled BFP8")
    for layer, groups in policy["layer_exceptions"].items():
        if not isinstance(layer, str) or not layer.isdecimal() or str(int(layer)) != layer:
            raise ValueError("Layer exception keys must be canonical decimal strings")
        if not 0 <= int(layer) < 48 or set(groups) - {"weight_groups", "compute_fidelities"}:
            raise ValueError("Invalid layer exception")
        for kind, overrides in groups.items():
            for group, setting in overrides.items():
                if group not in VARIABLE_GROUPS or group == "lm_head":
                    raise ValueError("Invalid layer group")
                probe = copy.deepcopy(policy)
                probe["layer_exceptions"] = {}
                probe[kind][group] = setting
                load_policy(probe)
    return policy

def for_layer(policy, index):
    result = copy.deepcopy(policy)
    for kind, overrides in policy["layer_exceptions"].get(str(index), {}).items():
        result[kind].update(overrides)
    return result

def compute(mesh, name):
    return ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, name),
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)

def cast_attr(obj, attr, dtype):
    value = getattr(obj, attr)
    dtype = getattr(ttnn, dtype)
    if value is not None and value.dtype != dtype:
        setattr(obj, attr, ttnn.typecast(value, dtype))

def apply_layer(layer, policy):
    """Apply before preparation/capture; no global class monkeypatching."""
    w, f, mesh = policy["weight_groups"], policy["compute_fidelities"], layer.mesh
    a = layer.attention
    if layer.kind == "linear_attention":
        cast_attr(a, "packed_weight", w["gdn_input"])
        a.projection_compute = compute(mesh, f["gdn_input"])
        cast_attr(a, "out_weight", w["gdn_output"])
        group = "gdn_output"
    else:
        # Restore decode precision from the immutable BF8 source. Widening the
        # baseline BF4 decode tensor would retain its quantization loss and
        # make BF8 candidate accuracy a mislabeled BF4 comparison.
        if a.decode_projection is not None and w["qsa_decode"] != "bfloat4_b":
            a.decode_projection = ttnn.typecast(a.packed_projection, getattr(ttnn, w["qsa_decode"]))
        # The shared packed representation also serves B>1 decode. This fixed
        # execution rule is explicit in the artifact and consumed here.
        cast_attr(a, "packed_projection", w[policy["qsa_execution"]["batched_decode_weight_group"]])
        a.prefill_projection_compute = compute(mesh, f["qsa_prefill"])
        a.projection_compute = compute(mesh, f["qsa_decode"])
        a.packed_singleton_compute = compute(mesh, f[policy["qsa_execution"]["singleton_packed_fidelity_group"]])
        cast_attr(a, "wo", w["qsa_output"])
        group = "qsa_output"
        desired = getattr(ttnn, policy["kv_cache_dtype"])
        if a.cache_dtype != desired:
            a.cache_dtype = desired
            del a.packed_cache
            a.packed_cache = ttnn.empty((1, 1, a.pool_tokens, 512), dtype=desired, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    projection = a.output_projection
    projection.row_weight = a.out_weight if layer.kind == "linear_attention" else a.wo
    cast_attr(projection, "column_weight", w[group])
    projection.compute = compute(mesh, f[group])
    for hc in (layer.attn_hyper, layer.mlp_hyper):
        for name in ("read_weight", "up"):
            cast_attr(hc, name, w["hc"])
        hc.compute = compute(mesh, f["hc"])
    for name in ("shared_gu", "shared_down"):
        cast_attr(layer.mlp, name, w["shared_expert"])
    layer.mlp.shared_compute = compute(mesh, f["shared_expert"])
    # Router must retain its independent sensitive policy.
    layer.mlp.router_compute = compute(mesh, f["router"])
    layer.mlp.routed_compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, f["routed_expert_gate_up"]),
        math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True, dst_full_sync_en=False)
    if layer.ple is not None:
        cast_attr(layer.ple, "projection", w["ple_projection"])
        layer.ple.projection_compute = compute(mesh, f["ple_projection"])
    layer.ops.precision_ccl = policy["ccl_dtype"]
    layer.precision_policy = policy

def runtime_summary(model):
    """Read constructed tensors and compute objects, not just the requested JSON."""
    def row(obj, weight, kernel):
        value = getattr(obj, weight)
        cfg = getattr(obj, kernel)
        return {"dtype": str(value.dtype) if value is not None else None, "compute": {key: str(getattr(cfg, key)) for key in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")}}
    layers = []
    for index, l in zip(model.layer_indices, model.layers):
        a = l.attention
        r = {"layer": index, "kind": l.kind, "ccl": l.ops.precision_ccl,
             "norms": {"attention_hc": str(l.attn_hyper.norm.dtype), "mlp_hc": str(l.mlp_hyper.norm.dtype),
                       "fidelity": str(l.ops.compute.math_fidelity)},
             "shared": row(l.mlp, "shared_gu", "shared_compute"), "shared_down": row(l.mlp, "shared_down", "shared_compute"), "router": row(l.mlp, "router", "router_compute"),
             "hc": row(l.attn_hyper, "read_weight", "compute"),
             "hc_up": row(l.attn_hyper, "up", "compute"),
             "mlp_hc": row(l.mlp_hyper, "read_weight", "compute"),
             "mlp_hc_up": row(l.mlp_hyper, "up", "compute"),
             "output": row(a.output_projection, "column_weight", "compute"),
             "output_row": row(a.output_projection, "row_weight", "compute"),
             "experts": {"gate": str(l.mlp.gates[0].dtype), "down": str(l.mlp.downs[0].dtype), "fidelity": str(l.mlp.routed_compute.math_fidelity),
                         "gate_dtypes": sorted({str(t.dtype) for t in l.mlp.gates}),
                         "up_dtypes": sorted({str(t.dtype) for t in l.mlp.ups}),
                         "down_dtypes": sorted({str(t.dtype) for t in l.mlp.downs}),
                         "local_experts": len(l.mlp.gates), "fp32_dest_acc_en": l.mlp.routed_compute.fp32_dest_acc_en}}
        if l.kind == "linear_attention":
            r["input"] = row(a, "packed_weight", "projection_compute")
            r["recurrence_fidelity"] = str(a.compute.math_fidelity)
        else:
            r.update(prefill=row(a, "packed_projection", "prefill_projection_compute"), decode=row(a, "decode_projection", "projection_compute"), kv=str(a.packed_cache.dtype))
            r["attention_fidelity"] = str(a.sparse_compute.math_fidelity)
            r["packed_singleton_fidelity"] = str(a.packed_singleton_compute.math_fidelity)
        if l.ple is not None:
            r["ple"] = row(l.ple, "projection", "projection_compute")
            r["ple_table_dtypes"] = sorted({str(t.dtype) for t in l.ple.tables})
            r["norms"]["ple"] = {name: str(getattr(l.ple,name).dtype) for name in ("norm_key", "norm_query", "norm_conv")}
        if l.kind == "linear_attention":
            assert a.recurrent_state.dtype == getattr(ttnn, model.precision_policy["state_dtypes"]["recurrent"])
            assert a.conv_state.dtype == getattr(ttnn, model.precision_policy["state_dtypes"]["convolution"])
            r["state"] = {"recurrent": str(a.recurrent_state.dtype), "convolution": str(a.conv_state.dtype)}
        else:
            assert a.raw_key_cache.dtype == getattr(ttnn, model.precision_policy["state_dtypes"]["raw_index"])
            r["raw_index"] = str(a.raw_key_cache.dtype)
        layers.append(r)
    return {"policy": model.precision_policy, "layers": layers, "head": row(model, "lm_head", "head_compute"),
            "routing": {"active_experts_per_token": model.cfg.num_experts_per_tok, "experts_total": model.cfg.num_experts},
            "embedding": str(model.embedding.dtype), "context": model.context,
            "residual_dtype": str(model.residual_dtype), "logits_dtype": str(model.logits_dtype),
            "terminal_mixer": {name: str(getattr(model.terminal, name).dtype) for name in ("norm", "down", "up")},
            "terminal_fidelity": str(model.ops.compute.math_fidelity),
            "sampling_max_top_k": model.precision_policy["sampling"]["max_top_k"]}
