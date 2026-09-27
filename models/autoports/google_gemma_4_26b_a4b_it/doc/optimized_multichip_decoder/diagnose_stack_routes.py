# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture TP1/TP4 decode boundaries and compare HF routing on their exact inputs.

Uses the unchanged stack harness and accepts its arguments. Host reads happen
after blocking trace replay, never inside forward/capture. Retaining intermediate
handles changes allocation lifetimes. Collective outputs are copied on device
before the next layer reuses the pooled buffers. This is diagnostic evidence;
acceptance must also pass without this wrapper. No HF attention/prefill execution.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder import probe_stack_precision
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_multichip_stack as stack


class CallProbe:
    def __init__(self, source, callback):
        self._probe_source = source
        self._probe_callback = callback

    def __getattr__(self, name):
        return getattr(self._probe_source, name)

    def __setattr__(self, name, value):
        if name.startswith("_probe_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._probe_source, name, value)

    def __call__(self, *args, **kwargs):
        output = self._probe_source(*args, **kwargs)
        if args[0].shape[-2] == 1:
            self._probe_callback(args, kwargs, output)
        return output


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--capture-positions", type=int, nargs="+", default=[4103, 4117])
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining, "--length", str(args.length), "--output", str(args.output)]
    captures, cpu_routers, projection_owners = {}, {}, []
    replay_counts, snapshots = {}, {}
    original_linear = ttnn.linear
    original_execute = ttnn.execute_trace

    def retain(tp, layer, label, tensor):
        if tp == 4 and label in ("attention_output", "shared", "routed"):
            # These may alias another layer's persistent collective output.
            tensor = ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        captures[(tp, layer, label)] = tensor

    def wrap(decoder, state, config, tp):
        layer_idx = decoder.layer_idx
        router = decoder.layer.moe.router
        projection_owners.append((router.projection_weight, tp, layer_idx))
        if layer_idx not in cpu_routers:
            cpu = Gemma4TextRouter(config)
            cpu.load_state_dict(
                {key.removeprefix("router."): value for key, value in state.items() if key.startswith("router.")}
            )
            cpu_routers[layer_idx] = cpu.eval()

        def save(label, tensor):
            retain(tp, layer_idx, label, tensor)

        forward = decoder._forward

        def forward_probe(hidden, **kwargs):
            output = forward(hidden, **kwargs)
            if hidden.shape[-2] == 1:
                save("layer_input", hidden)
                save("layer_output", output)
            return output

        decoder._forward = forward_probe

        def route_probe(inputs, kwargs, output):
            save("attention_residual", inputs[0])
            if kwargs.get("normalized") is not None:
                save("normalized_residual", kwargs["normalized"])
            save("routes", output)
            save("route_indices", router.decode_indices())

        def experts_probe(inputs, kwargs, output):
            save("expert_input", inputs[0])
            save("routed_local" if tp == 4 else "routed", output)

        def shared_probe(inputs, kwargs, output):
            save("shared_input", inputs[0])
            save("shared_local" if tp == 4 else "shared", output)

        def attention_probe(inputs, kwargs, output):
            save("attention_input", inputs[0])
            save("attention_output", output)

        decoder.layer.moe.router = CallProbe(router, route_probe)
        decoder.layer.moe.experts = CallProbe(decoder.layer.moe.experts, experts_probe)
        decoder.layer.shared_mlp = CallProbe(decoder.layer.shared_mlp, shared_probe)
        decoder.layer.self_attn = CallProbe(decoder.layer.self_attn, attention_probe)
        if tp == 4:
            tail = decoder._fused_tail

            def tail_probe(residual, shared, routed, decode):
                if residual.shape[-2] == 1:
                    save("shared", shared)
                    save("routed", routed)
                return tail(residual, shared, routed, decode)

            decoder._fused_tail = tail_probe
        return decoder

    for tp, cls in ((1, stack.OptimizedDecoder), (4, stack.MultichipDecoder)):
        factory = cls.from_state_dict

        def controlled(cls, state, _tp=tp, _factory=factory, **kwargs):
            decoder = _factory(state, **kwargs)
            config = getattr(kwargs["hf_config"], "text_config", kwargs["hf_config"])
            return wrap(decoder, state, config, _tp)

        cls.from_state_dict = classmethod(controlled)

    def linear_probe(value, weight, *positional, **kwargs):
        result = original_linear(value, weight, *positional, **kwargs)
        if value.shape[-2] == 1:
            for candidate, tp, layer in projection_owners:
                if weight is candidate:
                    retain(tp, layer, "router_scaled_input", value)
                    retain(tp, layer, "router_scores_fp32", result)
                    break
        return result

    def execute_probe(mesh, *positional, **kwargs):
        result = original_execute(mesh, *positional, **kwargs)
        tp = int(tuple(mesh.shape)[1])
        count = replay_counts.get(tp, 0)
        replay_counts[tp] = count + 1
        position = args.length + count // 2
        if count % 2 == 0 and position in args.capture_positions:
            if not kwargs.get("blocking", False):
                raise ValueError("Capture requires blocking trace replay")
            for (captured_tp, layer, label), tensor in captures.items():
                if captured_tp == tp:
                    snapshots[(position, tp, layer, label)] = [
                        ttnn.to_torch(part).float().clone() for part in ttnn.get_device_tensors(tensor)
                    ]
            print("ROUTE_BOUNDARIES_CAPTURED", tp, position, flush=True)
        return result

    ttnn.linear, ttnn.execute_trace = linear_probe, execute_probe
    try:
        # The controls are optional; with none selected this runs the defaults.
        probe_stack_precision.main()
    finally:
        ttnn.linear, ttnn.execute_trace = original_linear, original_execute
        torch.save(snapshots, args.output.with_suffix(".routes.tensors.pt"))
        rows, comparisons = [], []

        def metric(left, right):
            left, right = left.flatten().double(), right.flatten().double()
            return dict(
                pcc=float(torch.corrcoef(torch.stack((left, right)))[0, 1]),
                max_abs=float((left - right).abs().max()),
            )

        with torch.no_grad():
            for position, tp, layer in sorted({key[:3] for key in snapshots}):
                values = {key[3]: parts for key, parts in snapshots.items() if key[:3] == (position, tp, layer)}
                if "attention_residual" not in values:
                    continue
                cpu = cpu_routers[layer]
                residual = values["attention_residual"][0].reshape(1, -1)
                scores = cpu.proj(cpu.norm(residual) * cpu.scale * cpu.scalar_root_size)
                _, weights, hf_ids = cpu(residual)
                device_ids = values["route_indices"][0].flatten().long()
                device_set, hf_set = set(device_ids.tolist()), set(hf_ids.flatten().tolist())
                top9 = scores.topk(9, dim=-1)
                hf_routes = torch.zeros_like(scores).scatter(-1, hf_ids, weights)
                row = dict(
                    position=position,
                    tp=tp,
                    layer=layer,
                    device_indices=device_ids.tolist(),
                    hf_indices_same_residual=hf_ids.flatten().tolist(),
                    device_vs_hf_symmetric_difference=sorted(device_set ^ hf_set),
                    hf_rank8_rank9_margin=float(top9.values[0, 7] - top9.values[0, 8]),
                    route_weights_same_residual=metric(values["routes"][0], hf_routes),
                    replica_mismatches={
                        name: [int((part != parts[0]).sum()) for part in parts]
                        for name, parts in values.items()
                        if not name.endswith("_local")
                    },
                )
                if "router_scores_fp32" in values:
                    device_scores = values["router_scores_fp32"][0].reshape(1, -1)
                    raw_ids = device_scores.topk(8, dim=-1).indices.flatten()
                    centered = device_scores - device_scores.max(dim=-1, keepdim=True).values
                    rounded = centered.bfloat16().float()
                    rounded9 = rounded.topk(9, dim=-1)
                    row.update(
                        device_score_vs_hf=metric(device_scores, scores),
                        raw_score_top8=raw_ids.tolist(),
                        centered_bf16_top8=rounded9.indices[0, :8].tolist(),
                        centered_bf16_rank8_rank9_margin=float(rounded9.values[0, 7] - rounded9.values[0, 8]),
                        gate_vs_raw_score_symmetric_difference=sorted(device_set ^ set(raw_ids.tolist())),
                    )
                rows.append(row)
            for position, layer, label in sorted({(key[0], key[2], key[3]) for key in snapshots}):
                a, b = (snapshots.get((position, tp, layer, label)) for tp in (1, 4))
                if a is not None and b is not None and a[0].shape == b[0].shape:
                    comparisons.append(dict(position=position, layer=layer, boundary=label, **metric(a[0], b[0])))
        report = dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            allocation_lifetimes_perturbed=True,
            collective_boundary_snapshots="device copies inside the diagnostic trace",
            hf_reference_scope="router only, independently on each captured TP residual",
            capture_positions=args.capture_positions,
            routes=rows,
            tp1_tp4_boundary_comparisons=comparisons,
        )
        args.output.with_suffix(".routes.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
