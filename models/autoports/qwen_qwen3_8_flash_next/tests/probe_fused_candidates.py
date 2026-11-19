# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-process functional/fused comparison with shared real checkpoint weights.

Each variant gets identical restored request state and inputs. Traces are released
before any variant change; no weights/state are allocated while a trace is live.
Only output storage created by capture is acknowledged: it is fully overwritten
on replay, read before other work, and dies with that trace.
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import measure, runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference as load_ple_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import token_rows
from models.autoports.qwen_qwen3_8_flash_next.tests.run_fused_probe import POLICIES
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    FusedChunkedQSA,
    FusedDecoder,
    FusedGatedDeltaNet,
    FusedGatedResidual,
    FusedPLE,
    FusedPreciseProjection,
    FusedSparseMoe,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding
from models.autoports.qwen_qwen3_8_flash_next.tt.precise_gdn_projection import PreciseGDNProjection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=(0, 1, 3), default=0)
    p.add_argument("--seq-len", type=int, default=33)
    p.add_argument("--capacity", type=int, default=262144)
    p.add_argument("--expert-dtype", default="bfloat16", choices=("bfloat16", "bfloat8_b"))
    p.add_argument(
        "--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide")
    )
    p.add_argument("--variants", default="baseline,norm,activation,combined")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--clean-timing", action="store_true")
    p.add_argument("--single-policy", choices=tuple(POLICIES))
    a = p.parse_args()
    if a.single_policy:
        assert a.variants == "baseline", "Single-model timing cannot retain comparison models"
    tracked = trace_allocation_tracker.TRACE_ALLOC_TRACKING
    if a.clean_timing:
        assert not tracked and not trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS
    else:
        assert tracked and os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    audit = nullcontext if a.clean_timing else runtime_audit
    torch.set_num_threads(8)
    torch.manual_seed(2071 + a.layer)
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    if a.layer == 1:
        reference, weights = load_ple_reference(a.snapshot, cfg)
        del reference
    else:
        weights = load_weights(a.snapshot, a.layer, "")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(
        layer=a.layer,
        seq_len=a.seq_len,
        regime="serial chip0",
        allocation_tracking=bool(tracked),
        clean_timing=a.clean_timing,
        rows=[],
        passed=False,
        reference_kind="same-candidate eager output/state" if a.single_policy else "functional decoder",
        single_policy=a.single_policy,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    trace = None
    try:
        packed = (
            PackedEmbedding.from_checkpoint(a.snapshot, cache_dir=a.cache_dir, mesh_device=mesh)
            if a.layer == 1
            else None
        )
        factory = FunctionalDecoder if not a.single_policy else FusedDecoder
        base = factory.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, a.expert_dtype),
            **(POLICIES[a.single_policy] if a.single_policy else {}),
        )
        # Direct packed setup needs only these small projection matrices, not
        # the multi-gigabyte expert checkpoint retained by the initial loader.
        setup_names = (
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.out_proj.weight",
            "mlp.gate.weight",
            "attn_hyper_connection.input_mix_weight_down.weight",
            "attn_hyper_connection.input_mix_weight_up.weight",
            "mlp_hyper_connection.input_mix_weight_down.weight",
            "mlp_hyper_connection.input_mix_weight_up.weight",
        )
        if any(POLICIES.get(v, {}).get("pack_experts", False) for v in a.variants.split(",") if v != "baseline"):
            setup_names += ("mlp.experts.gate_up_proj",)
        setup_weights = {name: value for name, value in weights.items() if name.endswith(setup_names)}
        del weights
        state = base.create_state(capacity=a.capacity)
        names = (
            ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
            if a.layer == 3
            else (("conv", "recurrent", "ple_conv", "token_history") if a.layer == 1 else ("conv", "recurrent"))
        )
        owner = state.attention if a.layer == 3 else state
        buffers = {n: getattr(owner, n) for n in names}

        def snapshot():
            return {n: ttnn.to_torch(t).clone() for n, t in buffers.items()}

        def prepare(saved):
            return {
                n: ttnn.from_torch(v.contiguous(), dtype=buffers[n].dtype, layout=buffers[n].layout)
                for n, v in saved.items()
            }

        def restore(saved):
            for n, v in saved.items():
                ttnn.copy_host_to_device_tensor(v, buffers[n])

        def upload(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v, device=mesh, dtype=dtype, layout=layout)

        host_prompt = (torch.randn(1, 1, a.seq_len, 10240) * 0.02).bfloat16()
        host_tokens = [(torch.randn(1, 1, 1, 10240) * 0.02).bfloat16() for _ in range(2)]
        prompt = upload(host_prompt)
        token = upload(host_tokens[0])
        prepared_tokens = [ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT) for x in host_tokens]
        prompt_ids = (
            upload(token_rows(torch.randint(cfg.vocab_size, (1, a.seq_len))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if a.layer == 1
            else None
        )
        token_ids = (
            upload(token_rows(torch.randint(cfg.vocab_size, (1, 1))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if a.layer == 1
            else None
        )
        initial = prepare(snapshot())
        reference = {}
        models = {"baseline": base}
        for variant in a.variants.split(","):
            if variant == "baseline":
                continue
            if variant.startswith(("best_", "qkv_tree_")) or variant == "final":
                prepared = FusedDecoder.__new__(FusedDecoder)
                prepared.__dict__.update(base.__dict__)
                for name in ("mlp", "attn_mix", "mlp_mix", "attn", "ple", "chunked_attn"):
                    setattr(prepared, name, copy(getattr(base, name)))
                if prepared.chunked_attn is not None:
                    prepared.chunked_attn.attention = prepared.attn
                # Direct packed setup takes ownership of replaced projector
                # objects. Keep the functional reference's objects intact.
                for component in (prepared.attn_mix, prepared.mlp_mix):
                    component.precise_down = copy(component.precise_down)
                    component.precise_up = copy(component.precise_up)
                prepared.mlp.precise_router = copy(prepared.mlp.precise_router)
                for field in ("qkv", "out"):
                    value = getattr(prepared.attn, field, None)
                    if isinstance(value, PreciseGDNProjection):
                        setattr(prepared.attn, field, copy(value))
                with patch.object(FunctionalDecoder, "from_state_dict", return_value=prepared):
                    models[variant] = FusedDecoder.from_state_dict(setup_weights, **POLICIES[variant])
                if variant.startswith("qkv_tree_"):
                    assert isinstance(models[variant].attn.qkv, FusedPreciseProjection)
                    assert models[variant].attn.qkv.balanced_tree
                continue
            model = FusedDecoder.__new__(FusedDecoder)
            model.__dict__.update(base.__dict__)
            moe = FusedSparseMoe.__new__(FusedSparseMoe)
            moe.__dict__.update(base.mlp.__dict__)
            moe.fuse_expert_activation = variant in (
                "activation",
                "combined",
                "expert",
                "qsa_concat",
                "qsa_gate",
                "qsa_both",
            )
            moe.fuse_shared_activation = variant in (
                "activation",
                "combined",
                "shared",
                "qsa_concat",
                "qsa_gate",
                "qsa_both",
            )
            if variant.startswith("packed"):
                moe.fuse_expert_activation = True
                moe.fuse_shared_activation = True
                moe.pack_projections(
                    shared=variant in ("packed_shared", "packed_both"),
                    experts=variant in ("packed_experts", "packed_both"),
                )
            if variant == "fused_reduce":
                moe.fuse_expert_activation = True
                moe.fuse_shared_activation = True
                moe.pack_projections(shared=True)
                moe.prepare_fused_reduce(mesh)
            if variant in ("gdn_packed", "gdn_conv", "gdn_layout", "gdn_conv_bf16"):
                attn = FusedGatedDeltaNet.__new__(FusedGatedDeltaNet)
                attn.__dict__.update(model.attn.__dict__)
                attn.zab = ttnn.concat([attn.z, attn.a, attn.b], dim=-1)
                del attn.z, attn.a, attn.b
                attn.fuse_recurrence_layout = variant == "gdn_layout"
                if variant in ("gdn_conv", "gdn_conv_bf16"):
                    attn.prepare_conv(mesh)
                    if variant == "gdn_conv_bf16":
                        attn.conv_compute = ttnn.init_device_compute_kernel_config(
                            mesh.arch(),
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=False,
                            packer_l1_acc=False,
                        )
                model.attn = attn
            moe.matmul_expert_activation = variant == "matmul_activation"
            moe.matmul_shared_activation = variant in ("matmul_activation", "matmul_shared")
            if variant in ("ple_activation", "ple_norm"):
                assert model.ple is not None
                ple = FusedPLE.__new__(FusedPLE)
                ple.__dict__.update(model.ple.__dict__)
                ple.merge_norm = variant == "ple_norm"
                model.ple = ple
            model.mlp = moe
            if variant.startswith("sfpu_"):
                if variant in ("sfpu_gr", "sfpu_all"):
                    for name in ("attn_mix", "mlp_mix"):
                        gr = copy(getattr(model, name))
                        gr.precise_down = FusedPreciseProjection(gr.precise_down, gr.compute)
                        gr.precise_up = FusedPreciseProjection(gr.precise_up, gr.compute)
                        setattr(model, name, gr)
                if variant in ("sfpu_router", "sfpu_all"):
                    moe.precise_router = FusedPreciseProjection(moe.precise_router, moe.compute)
                if variant in ("sfpu_gdn", "sfpu_all") and model.kind == "linear_attention":
                    model.attn = copy(model.attn)
                    model.attn.qkv = FusedPreciseProjection(model.attn.qkv, model.attn.compute)
                    model.attn.out = FusedPreciseProjection(model.attn.out, model.attn.compute)
            if variant in ("norm", "combined", "merged_norm", "norm_rhs"):
                for n in ("attn_mix", "mlp_mix"):
                    gr = FusedGatedResidual.__new__(FusedGatedResidual)
                    gr.__dict__.update(getattr(base, n).__dict__)
                    gr.merge_rsqrt = variant == "merged_norm"
                    gr.merge_rhs = variant == "norm_rhs"
                    setattr(model, n, gr)
            if variant.startswith("qsa_"):
                assert model.chunked_attn is not None
                attention = FusedChunkedQSA.__new__(FusedChunkedQSA)
                attention.__dict__.update(model.chunked_attn.__dict__)
                attention.fuse_gate = variant in ("qsa_gate", "qsa_both")
                attention.fuse_concat = variant in ("qsa_concat", "qsa_both")
                if variant in ("qsa_packed", "qsa_split"):
                    attention.packed_qkv = ttnn.concat([model.attn.q, model.attn.k, model.attn.v], dim=-1)
                    attention.split_heads = variant == "qsa_split"
                attention.fuse_rope = variant in ("qsa_rope", "qsa_decode_rope")
                attention.decode_rope = variant == "qsa_decode_rope"
                attention.pool_mean = variant == "qsa_poolmean"
                attention.decode_concat = variant == "qsa_decode_concat"
                if variant == "qsa_decode_heads":
                    queries = [model.attn.q[..., i * 512 : i * 512 + 256] for i in range(24)]
                    gates = [model.attn.q[..., i * 512 + 256 : (i + 1) * 512] for i in range(24)]
                    attention.decode_packed_qkv = ttnn.concat([*queries, model.attn.k, model.attn.v, *gates], dim=-1)
                model.chunked_attn = attention
            models[variant] = model
        if a.clean_timing:
            models["baseline_tail"] = base
        for variant, model in models.items():
            restore(initial)
            with audit():
                out = model.prefill_forward(prompt, state=state, token_ids=prompt_ids)
            actual = ttnn.to_torch(out)
            del out
            prefix_raw = snapshot()
            prefix = prepare(prefix_raw)
            if variant == "baseline":
                reference["prefill"] = actual
                reference["prefix"] = prefix_raw
            prefill_pcc = measure(reference["prefill"], actual, host_prompt)
            prefill_exact = torch.equal(reference["prefill"], actual)
            state_pcc = {
                n: (
                    float(
                        torch.corrcoef(torch.stack([reference["prefix"][n].float().flatten(), v.float().flatten()]))[
                            0, 1
                        ]
                    )
                    if v.numel() > 1 and v.is_floating_point()
                    else None
                )
                for n, v in prefix_raw.items()
            }
            prefill_samples = []
            for _ in range(a.prefill_repeats):
                restore(initial)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                with audit():
                    out = model.prefill_forward(prompt, state=state, token_ids=prompt_ids)
                ttnn.synchronize_device(mesh)
                prefill_samples.append((time.perf_counter() - start) * 1000)
                del out
            prefill_ms = float(torch.tensor(prefill_samples).median())
            eager = []
            for i, host in enumerate(prepared_tokens):
                restore(prefix)
                ttnn.copy_host_to_device_tensor(host, token)
                with audit():
                    out = model.decode_forward(token, state=state, token_ids=token_ids)
                eager.append((ttnn.to_torch(out).clone(), snapshot()))
                del out
            if variant == "baseline":
                reference["decode"] = eager
            restore(prefix)
            ttnn.copy_host_to_device_tensor(prepared_tokens[0], token)
            begin_op = int(ttnn._ttnn.get_device_operation_id())
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with audit():
                out = model.decode_forward(token, state=state, token_ids=token_ids)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            captured_ops = int(ttnn._ttnn.get_device_operation_id()) - begin_op
            if tracked:
                trace_allocation_tracker.acknowledge_corruptible(out)
            repeats = []
            for j in range(a.repeats):
                i = j % 2
                restore(prefix)
                ttnn.copy_host_to_device_tensor(prepared_tokens[i], token)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                with audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - start) * 1000
                actual = ttnn.to_torch(out)
                observed = snapshot()
                repeats.append(
                    dict(
                        ms=ms,
                        input=i,
                        exact_output=torch.equal(actual, eager[i][0]),
                        exact_state=all(torch.equal(observed[n], eager[i][1][n]) for n in names),
                        pcc=measure(reference["decode"][i][0], actual, host_tokens[i]),
                    )
                )
            ttnn.release_trace(mesh, trace)
            trace = None
            del out
            row = dict(
                variant=variant,
                captured_device_operations=captured_ops,
                runtime_class=type(model).__name__,
                moe_class=type(model.mlp).__name__,
                selected_policy=POLICIES.get(variant),
                qkv_projection=(
                    dict(
                        runtime_class=type(model.attn.qkv).__name__,
                        balanced_tree=getattr(model.attn.qkv, "balanced_tree", None),
                        weight_blocks=[list(w.shape) for w in model.attn.qkv.weights],
                    )
                    if model.kind == "linear_attention"
                    else None
                ),
                prefill_pcc=prefill_pcc,
                prefill_exact_vs_functional=prefill_exact,
                prefix_state_pcc=state_pcc,
                warmed_prefill_host_ms=prefill_ms,
                warmed_prefill_host_samples_ms=prefill_samples,
                replays=repeats,
                traced_decode_median_ms=float(torch.tensor([r["ms"] for r in repeats[2:]]).median()),
            )
            if a.single_policy:
                row["variant"] = a.single_policy
                row["selected_policy"] = POLICIES[a.single_policy]
                row["prefill_exact_vs_same_candidate"] = row.pop("prefill_exact_vs_functional")
            row["passed"] = prefill_pcc["passed"] and all(
                r["exact_output"] and r["exact_state"] and r["pcc"]["passed"] for r in repeats
            )
            report["rows"].append(row)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(variant, row["passed"], row["traced_decode_median_ms"], flush=True)
        report["passed"] = all(r["passed"] for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
