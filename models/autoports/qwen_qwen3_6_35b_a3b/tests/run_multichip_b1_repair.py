# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental setup-only B1 repair wrapper; use the ordinary runner arguments.

For example add --mesh-build-kwargs
'{"functional_repair":{"topk_width":1024,"full_input_shard":true}}'.
Options are independent. The runner retains its quality gates, argv, real weights,
and measurement/provenance paths. This file does not promote any candidate.
Native linear math intentionally changes accumulation; measured PCC must decide
whether that candidate is admissible. No failure is caught or silently retried.
The four promoted B1 controls default to false in this historical experiment
wrapper, so its independent candidate flags retain their original baseline.
Explicit top-level decode_* kwargs can opt back into the promoted helpers.
``functional_repair.exact_recurrence`` composes the two-stage FP32 SFPU probe
with independent native norm/projection experiments; it does not change gates.
"""

import copy
import runpy
import sys
import types
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder


class _TopkProxy:
    """Private function-global proxy, never a mutation of the TTNN module."""

    def __init__(self, width=None, selector=None, round_chain=None):
        self.width = width
        self.selector = selector
        self.round_chain = round_chain

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def topk(self, logits, *args, **kwargs):
        if logits.shape[-2] == 1:
            if logits.shape[-1] != 256 or logits.dtype != ttnn.float32:
                raise ValueError("B1 topk repair requires 256 FP32 router logits")
            if self.round_chain is not None:
                logits = ttnn.unary_chain(logits, self.round_chain)
            if self.selector is not None:
                if args or kwargs != {"k": 8, "dim": -1, "sorted": True}:
                    raise ValueError("FP32 selector implements the exact sorted 256-to-8 MoE call")
                return self.selector(logits)
            if self.width is not None:
                logits = ttnn.pad(logits, [(0, 0)] * (len(logits.shape) - 1) + [(0, self.width - 256)], float("-inf"))
        # All 256 real logits precede -inf padding, preserving the expert ID
        # domain for finite router logits; padding is never renumbered as an ID.
        return ttnn.topk(logits, *args, **kwargs)


def _install_topk(dec, width=None, fp32_selector=False, router_round_bf16=False):
    # The private topk proxy owns this experiment, including baseline TTNN calls.
    dec.decode_fp32_selector = False
    dec.precision_policy["decode_fp32_selector"] = False
    selector = None
    if fp32_selector:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_topk_probe import MeshFP32TopK

        selector = MeshFP32TopK(dec.mesh_device)
        dec.functional_repair_fp32_selector = selector
    method = dec._moe.__func__
    namespace = dict(
        method.__globals__, ttnn=_TopkProxy(width, selector, dec.round_widen_chain if router_round_bf16 else None)
    )
    replacement = types.FunctionType(
        method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__
    )
    replacement.__kwdefaults__ = method.__kwdefaults__
    dec._moe = types.MethodType(replacement, dec)


def _install_norm(dec, *, full_input_shard, native_linear_norm, full_input_projection_shard=False):
    original = dec._normalized_input
    if dec.residual_layout != "replicated" or not hasattr(dec, "direct_ar_memory"):
        raise ValueError("norm repair requires replicated residuals and the direct all-reduce memory contract")
    if (
        full_input_projection_shard
        and dec.kind == "full_attention"
        and "attention_packed" not in dec.mesh_decode_weights
    ):
        raise ValueError("projection-sharded norm requires the DRAM decode projection configuration")

    def normalized(self, x, key):
        if x.shape[-2] == 1:
            if (
                (full_input_shard or full_input_projection_shard)
                and self.kind == "full_attention"
                and key == "input_layernorm.weight"
            ):
                memory = (
                    self.mesh_decode_weights["attention_packed"][1]
                    if full_input_projection_shard
                    else self.direct_ar_memory
                )
                x = ttnn.to_memory_config(x, memory)
            if native_linear_norm and self.kind == "linear_attention":
                x = ttnn.to_memory_config(x, self.direct_ar_memory)
                result = ttnn.rms_norm(
                    x,
                    weight=self.w[key],
                    memory_config=self.direct_ar_memory,
                    epsilon=self.config.rms_norm_eps,
                    compute_kernel_config=self.compute,
                )
                return ttnn.typecast(result, ttnn.bfloat16, memory_config=self.direct_ar_memory)
        return original(x, key)

    dec._normalized_input = types.MethodType(normalized, dec)


def _raw_projection_weights(sd, packed):
    q, k, v = sd["linear_attn.in_proj_qkv.weight"].split((2048, 2048, 4096), dim=0)
    z = sd["linear_attn.in_proj_z.weight"]
    a, b = (sd[f"linear_attn.in_proj_{name}.weight"] for name in ("a", "b"))
    state, output = [], []
    for rank in range(4):
        qs, vs, hs = (
            slice(rank * 512, (rank + 1) * 512),
            slice(rank * 1024, (rank + 1) * 1024),
            slice(rank * 8, (rank + 1) * 8),
        )
        state.append(
            torch.cat(
                (
                    k[qs],
                    v[vs],
                    torch.nn.functional.pad(a[hs], (0, 0, 0, 24)),
                    torch.nn.functional.pad(b[hs], (0, 0, 0, 24)),
                ),
                dim=0,
            ).T
        )
        output.append(torch.cat((q[qs], z[vs]), dim=0).T)
    if packed:
        return {"attention_combined": torch.cat([torch.cat((s, o), dim=1) for s, o in zip(state, output)], dim=1)}
    return {"attention_packed": torch.cat(state, dim=1), "output_only_packed": torch.cat(output, dim=1)}


def _install_projection(dec, sd):
    dec.functional_repair_projection_weights = {
        key: ttnn.from_torch(
            value.contiguous(),
            device=dec.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(dec.mesh_device, dim=1),
        )
        for key, value in _raw_projection_weights(sd, dec.pack_linear).items()
    }
    original = dec.linear

    def linear(
        self, x, key, dtype=ttnn.bfloat16, *, projection_k=512, ordered_out=False, terminal_scan=False, **kwargs
    ):
        selected = x.shape[-2] == 1 and projection_k == 512
        if selected and key in self.functional_repair_projection_weights:
            if dtype != ttnn.bfloat16:
                raise ValueError("native B1 projection repair requires BF16 projection output")
            return ttnn.linear(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG),
                self.functional_repair_projection_weights[key],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if selected and key == "linear_attn.out_proj.weight":
            return ttnn.linear(
                ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG),
                self.w[key],
                dtype=ttnn.float32,
                compute_kernel_config=self.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return original(
            x, key, dtype, projection_k=projection_k, ordered_out=ordered_out, terminal_scan=terminal_scan, **kwargs
        )

    dec.linear = types.MethodType(linear, dec)


def _validate(spec):
    if not isinstance(spec, dict):
        raise ValueError("functional_repair must be an object")
    known = {
        "topk_width",
        "fp32_selector",
        "full_input_shard",
        "full_input_projection_shard",
        "native_linear_norm",
        "native_linear_projection",
        "compensated_norm_epilogue",
        "silu_variant",
        "native_dram_projection",
        "native_dram_geometry",
        "native_dram_fidelity",
        "router_round_bf16",
        "baseline_router_selector",
        "exact_recurrence",
    }
    if set(spec) - known:
        raise ValueError(f"unknown functional_repair keys: {sorted(set(spec) - known)}")
    if "topk_width" in spec and (type(spec["topk_width"]) is not int or spec["topk_width"] not in (1024, 2048)):
        raise ValueError("topk_width must be 1024 or 2048")
    if "topk_width" in spec and spec.get("fp32_selector"):
        raise ValueError("topk_width and fp32_selector are alternative selectors")
    if spec.get("full_input_shard") and spec.get("full_input_projection_shard"):
        raise ValueError("full_input_shard and full_input_projection_shard are alternative norm layouts")
    if "silu_variant" in spec and spec["silu_variant"] not in ("once_per_page", "tile_workers"):
        raise ValueError("invalid silu_variant")
    if "native_dram_geometry" in spec and not isinstance(spec["native_dram_geometry"], dict):
        raise ValueError("native_dram_geometry must be an object")
    if "native_dram_fidelity" in spec and spec["native_dram_fidelity"] not in ("HiFi4", "HiFi2", "LoFi"):
        raise ValueError("invalid native_dram_fidelity")
    for key in known - {"topk_width", "silu_variant", "native_dram_geometry", "native_dram_fidelity"}:
        if key in spec and type(spec[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")


def main():
    original_descriptor = MultichipDecoder.__dict__["from_state_dict"]
    original = original_descriptor.__func__
    invocation = list(sys.argv)

    def construct(cls, state_dict, *, functional_repair=None, **kwargs):
        spec = {} if functional_repair is None else copy.deepcopy(functional_repair)
        _validate(spec)
        for key in (
            "decode_fp32_selector",
            "decode_compensated_norm",
            "decode_tile_silu",
            "decode_full_input_shard",
        ):
            kwargs.setdefault(key, False)
        dec = original(cls, state_dict, **kwargs)
        if "topk_width" in spec or spec.get("fp32_selector") or spec.get("router_round_bf16"):
            _install_topk(
                dec, spec.get("topk_width"), spec.get("fp32_selector", False), spec.get("router_round_bf16", False)
            )
        if spec.get("full_input_shard") or spec.get("full_input_projection_shard") or spec.get("native_linear_norm"):
            _install_norm(
                dec,
                full_input_shard=spec.get("full_input_shard", False),
                native_linear_norm=spec.get("native_linear_norm", False),
                full_input_projection_shard=spec.get("full_input_projection_shard", False),
            )
        if spec.get("native_linear_projection") and dec.kind == "linear_attention":
            _install_projection(dec, state_dict)
        if spec.get("native_dram_projection") and dec.kind == "linear_attention":
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_native_projection_probe import install

            install(
                dec,
                state_dict,
                geometry=spec.get("native_dram_geometry"),
                fidelity=spec.get("native_dram_fidelity", "HiFi4"),
            )
        if spec.get("compensated_norm_epilogue"):
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_norm_epilogue_probe import install

            install(dec)
        if spec.get("silu_variant"):
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_silu_probe import install

            install(dec, variant=spec["silu_variant"])
        if spec.get("baseline_router_selector"):
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_baseline_router_probe import install

            install(dec)
        if spec.get("exact_recurrence"):
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_exact_recurrence_probe import install

            install(dec)
        dec.precision_policy["functional_repair"] = spec
        dec.precision_policy["functional_repair_provenance"] = {
            "experimental": True,
            "wrapper": str(Path(__file__).resolve()),
            "argv": invocation,
            "scope": "setup-bound instance methods; original quality gates unchanged",
        }
        return dec

    MultichipDecoder.from_state_dict = classmethod(construct)
    try:
        runpy.run_module("models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder", run_name="__main__")
    finally:
        MultichipDecoder.from_state_dict = original_descriptor


if __name__ == "__main__":
    main()
