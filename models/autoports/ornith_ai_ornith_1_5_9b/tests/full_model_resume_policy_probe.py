# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: isolate partition effects using identical real layer inputs.

Run externally under timeout and exclusive device ownership. This is an eager,
readback-heavy probe, never a production implementation or performance result.
"""

import argparse
import bz2
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16-prefill", action="store_true", help="Diagnostic-only norm/CCL control")
    parser.add_argument("--length", type=int, default=513)
    parser.add_argument("--starts", type=int, nargs="+", default=[32, 68, 512])
    parser.add_argument("--text-file", default="models/ttt_compat/tests/tale-of-two-cities.txt.bz2")
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel

    if args.bf16_prefill:
        import inspect
        import textwrap

        from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as decoder

        for name in ("norm", "reduce"):
            source = textwrap.dedent(inspect.getsource(getattr(decoder.MultichipDecoder, name)))
            source = source.replace("ttnn.bfloat8_b", "ttnn.bfloat16")
            namespace = dict(decoder.__dict__)
            exec(  # noqa: S102 -- local source control only
                compile(source, f"<diagnostic_bf16_{name}>", "exec"), namespace
            )
            setattr(decoder.MultichipDecoder, name, namespace[name])

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    report = {
        "diagnostic_only": True,
        "bf16_prefill_control": args.bf16_prefill,
        "length": args.length,
        "starts": args.starts,
    }
    try:
        model = OrnithModel(mesh, layer_indices=[0])
        layer = model.layers[0]
        assert layer.kind == "linear_attention"
        tokenizer = AutoTokenizer.from_pretrained(model.checkpoint.path)
        with bz2.open(args.text_file, "rt") as handle:
            text = handle.read()
        ids = tokenizer.encode(text, add_special_tokens=False)[: args.length]
        assert len(ids) == args.length
        tokens = model.upload(torch.tensor([ids]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden = model.embed(tokens)
        cache = model.allocate_cache(batch=1, context=1024)
        state = cache[0][0]
        records = {}
        original_linear = layer.linear
        original_reduce = layer.reduce
        original_gdn = ttnn.transformer.chunk_gated_delta_rule
        original_minimal = ttnn.experimental.minimal_matmul

        def read(tensor, dim):
            return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim)).float()

        def metrics(reference, candidate):
            assert reference.shape == candidate.shape, (reference.shape, candidate.shape)
            a, b = reference.reshape(-1).double(), candidate.reshape(-1).double()
            ac, bc = a - a.mean(), b - b.mean()
            denom = ac.norm() * bc.norm()
            return {
                "shape": list(reference.shape),
                "bit_equal": torch.equal(reference, candidate),
                "max_abs": float((a - b).abs().max()),
                "rms_error": float((a - b).square().mean().sqrt()),
                "pcc": float(torch.dot(ac, bc) / denom) if denom else None,
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            }

        def linear(x, name):
            out = original_linear(x, name)
            if name == "packed_input":
                records["projection"] = (x, out)
            return out

        def reduce(x):
            out = original_reduce(x)
            if "reduction" not in records:
                records["reduction"] = (x, out)
            return out

        def minimal(x, weight, *positional, **kwargs):
            result = original_minimal(x, weight, *positional, **kwargs)
            if weight is layer.weights["packed_mlp.weight"]:
                records["packed_mlp"] = (x, result)
            return result

        def gdn(*inputs, **kwargs):
            # The original initial-state object is subsequently overwritten by the
            # layer. Keep its exact bytes independently for the controlled reruns.
            initial = ttnn.clone(kwargs["initial_state"])
            result = original_gdn(*inputs, **kwargs)
            records["gdn"] = (inputs, dict(kwargs, initial_state=initial), result)
            return result

        layer.linear, layer.reduce = linear, reduce
        ttnn.transformer.chunk_gated_delta_rule = gdn
        ttnn.experimental.minimal_matmul = minimal
        try:
            baseline = layer.prefill_forward(hidden, state=state, start_position=0)
            # Synchronize via readback before restoring hooks and retaining tensors.
            read(baseline, 3)
        finally:
            layer.linear, layer.reduce = original_linear, original_reduce
            ttnn.transformer.chunk_gated_delta_rule = original_gdn
            ttnn.experimental.minimal_matmul = original_minimal

        report["projection_same_input"] = []
        report["mlp_same_input"] = []
        mlp_input, mlp_output = records["packed_mlp"]
        mlp_ref = read(mlp_output, 3)
        report["reduction_same_input"] = []
        report["norm_partition"] = []
        projected_input, projected_output = records["projection"]
        reduction_input, reduction_output = records["reduction"]
        p_ref, r_ref = read(projected_output, 3), read(reduction_output, 3)
        norm_ref = read(layer.norm(hidden, "input_layernorm"), 3)
        for start in args.starts:
            if not 0 < start < args.length:
                continue
            for begin, end in [(0, start), (start, args.length)]:
                interval = {"begin": begin, "end": end}
                x = projected_input[:, :, begin:end, :]
                result = original_linear(x, "packed_input")
                report["projection_same_input"].append(
                    dict(interval, input_dtype=str(x.dtype), **metrics(p_ref[:, :, begin:end], read(result, 3)))
                )
                x = mlp_input[:, :, begin:end, :]
                if end - begin == 1:
                    gate = original_linear(x, "mlp.gate_proj")
                    up = original_linear(x, "mlp.up_proj")
                    result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                elif end - begin <= 128:
                    result = layer.short_prefill_mlp(x, end - begin)
                else:
                    result = original_minimal(
                        x,
                        layer.weights["packed_mlp.weight"],
                        fuse_swiglu=True,
                        config=layer.prefill_program,
                        compute_kernel_config=layer.mlp_compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                    )
                report["mlp_same_input"].append(
                    dict(interval, input_dtype=str(x.dtype), **metrics(mlp_ref[:, :, begin:end], read(result, 3)))
                )
                x = reduction_input[:, :, begin:end, :]
                if end - begin == 1:
                    # The actual native decode projection supplies width-sharded
                    # rows; slicing a prefill tensor alone does not reproduce it.
                    x = ttnn.to_memory_config(x, layer.ar_mem)
                result = original_reduce(x)
                report["reduction_same_input"].append(
                    dict(interval, **metrics(r_ref[:, :, begin:end], read(result, 3)))
                )
                result = layer.norm(hidden[:, :, begin:end, :], "input_layernorm")
                result_bf8 = ttnn.typecast(result, ttnn.bfloat8_b)
                report["norm_partition"].append(
                    dict(
                        interval,
                        native=metrics(norm_ref[:, :, begin:end], read(result, 3)),
                        canonical_bf8=metrics(norm_ref[:, :, begin:end], read(result_bf8, 3)),
                    )
                )

        inputs, gdn_kwargs, (out_ref, state_ref) = records["gdn"]
        out_ref_host, state_ref_host = read(out_ref, 0), read(state_ref, 1)
        report["gdn_same_input"] = []
        report["gdn_dtypes"] = {
            "inputs": [str(x.dtype) for x in inputs],
            "initial_state": str(gdn_kwargs["initial_state"].dtype),
            "final_state": str(state_ref.dtype),
        }

        def segment(x, begin, end):
            value = x[:, begin:end, :]
            pad = (-(end - begin)) % 32
            return ttnn.pad(value, [(0, 0), (0, pad), (0, 0)], 0.0) if pad else value

        for start in args.starts:
            if not 0 < start < args.length:
                continue
            parts = []
            running_state = gdn_kwargs["initial_state"]
            for begin, end in [(0, start), (start, args.length)]:
                values = [segment(x, begin, end) for x in inputs]
                out, running_state = original_gdn(*values, **dict(gdn_kwargs, initial_state=running_state))
                parts.append(read(out, 0)[:, : end - begin, :])
            joined = torch.cat(parts, dim=1)
            # FP32 host->device roundtrip is intentionally NOT involved: this
            # checks the actual device final-state/initial-state boundary.
            report["gdn_same_input"].append(
                {
                    "start": start,
                    "output": metrics(out_ref_host[:, : args.length, :], joined),
                    "state": metrics(state_ref_host, read(running_state, 1)),
                    "first_changed_row": next(
                        (i for i in range(args.length) if not torch.equal(out_ref_host[:, i], joined[:, i])), None
                    ),
                }
            )
        report["interpretation"] = (
            "Projection uses identical already-normalized tensors, so differences isolate shape-selected "
            "projection execution. GDN uses identical Q/K/V/g/beta, so differences isolate chunk boundaries, "
            "kernel selection, and FP32 state handoff, excluding projection/norm/CCL/cache differences. "
            "Aligned32 and unaligned68 distinguish state export from regrouped32-token recurrence math."
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print("RESUME_POLICY_PROBE", args.output, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
