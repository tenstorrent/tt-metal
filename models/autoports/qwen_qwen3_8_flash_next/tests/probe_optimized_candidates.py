# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-process optimized program search and matched fused/default measurements.

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
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import measure, runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference as load_ple_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import token_rows
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder

POLICIES = {"baseline": {}}
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, choices=(0, 1, 3), default=0)
    p.add_argument("--seq-len", type=int, default=33)
    p.add_argument("--capacity", type=int, default=262144)
    p.add_argument("--expert-dtype", default="bfloat16", choices=("bfloat16", "bfloat8_b", "bfloat4_b"))
    p.add_argument(
        "--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide")
    )
    p.add_argument("--variants", default="baseline")
    p.add_argument("--policies", type=Path, required=True)
    p.add_argument("--build-kwargs", type=Path)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--clean-timing", action="store_true")
    p.add_argument("--factory", choices=("optimized", "fused"), default="optimized")
    p.add_argument("--split-matmul-activation", action="store_true")
    p.add_argument("--save-outputs", type=Path, help="Save exact inputs/outputs for independent CPU HF diagnosis")
    a = p.parse_args()
    POLICIES.update(json.loads(a.policies.read_text()))
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
        reference_kind="same-construction conservative sparse/SDPA program control",
        factory=a.factory,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    trace = None
    try:
        packed = (
            PackedEmbedding.from_checkpoint(a.snapshot, cache_dir=a.cache_dir, mesh_device=mesh)
            if a.layer == 1
            else None
        )
        factory = OptimizedDecoder if a.factory == "optimized" else FusedDecoder
        base = factory.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, a.expert_dtype),
            **(json.loads(a.build_kwargs.read_text()) if a.build_kwargs else {}),
        )
        del weights
        if a.split_matmul_activation:
            assert a.factory == "optimized" and not hasattr(base.mlp, "expert_packed")
            base.mlp.matmul_expert_activation = True
        report["split_matmul_activation"] = a.split_matmul_activation
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
        saved_outputs = dict(prompt=host_prompt, tokens=host_tokens, variants={})
        default_policy = dict(base.expert_policy) if a.factory == "optimized" else {}
        variants = ["baseline", *[v for v in a.variants.split(",") if v != "baseline"], "baseline_tail"]
        for variant in variants:
            model = base
            if a.factory == "optimized":
                model.configure_experts(**(default_policy if variant == "default" else POLICIES.get(variant, {})))
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
            prefill_pcc["whole_output_global_pcc"] = float(
                torch.corrcoef(torch.stack([reference["prefill"].float().flatten(), actual.float().flatten()]))[0, 1]
            )
            if a.save_outputs:
                saved_outputs["variants"][variant] = dict(prefill=actual.clone())
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
            if a.save_outputs:
                saved_outputs["variants"][variant]["decode"] = [entry[0] for entry in eager]
                a.save_outputs.parent.mkdir(parents=True, exist_ok=True)
                torch.save(saved_outputs, a.save_outputs)
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
                selected_policy=default_policy if variant == "default" else POLICIES.get(variant),
                resolved_gate_program=str(
                    model.mlp.program.get_decode_gate_up_config(
                        1, 1280 if hasattr(model.mlp, "expert_packed") else 640, k=2560
                    )
                ),
                resolved_down_program=str(model.mlp.program.get_decode_down_config(1, 2560, k=640)),
                expert_compute=str(getattr(model.mlp, "expert_compute", model.mlp.compute)),
                expert_dtype=str(model.mlp.down.dtype),
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
                prefill_exact_vs_baseline=prefill_exact,
                prefix_state_pcc=state_pcc,
                warmed_prefill_host_ms=prefill_ms,
                warmed_prefill_host_samples_ms=prefill_samples,
                replays=repeats,
                traced_decode_median_ms=float(torch.tensor([r["ms"] for r in repeats[2:]]).median()),
            )
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
