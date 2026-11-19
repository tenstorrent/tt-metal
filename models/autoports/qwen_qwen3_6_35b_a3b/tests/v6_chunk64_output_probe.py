# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic full chunk64 outputs and final state on identical HF operands.

Reuses unchanged canonical-sum/double-single state helpers. All hardware belongs
to the parent. CPU setup verifies every FP32/BF16 output and cache code.
"""

import argparse
import ast
import copy
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_double_single_state_probe as combined

base = combined.base
ordered = combined.ordered
helper = combined.helper


class OutputChunk64Graph(ordered.OrderedChunk64StateGraph):
    """Normalized FP32 [B,H,S,D] inputs; returns BF16 output, state, FP32 output.

    Outputs use [B,S,H,Dv]; state remains FP32 [B,H,Dk,Dv]. The state method is
    the unchanged combined diagnostic. Internal observers retain device handles
    only; user observers are optional diagnostic instrumentation.
    """

    def __call__(self, query, key, value, g, beta, initial, observer=None):
        t = self.t
        n = key.shape[-2]
        assert n > 0 and tuple(query.shape) == tuple(key.shape)
        assert query.dtype == key.dtype == value.dtype == t.float32
        query = self.mul(query, 1 / (query.shape[-1] ** 0.5))
        output_key = key
        pad = (-n) % 64
        if pad:
            query, output_key = [
                t.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=self.mem) for x in (query, output_key)
            ]
        outputs = []
        matrices = {}
        carry = {"state": initial}

        def collect(name, tensor):
            prefix, stage = name.rsplit("/", 1)
            if observer:
                observer(name, tensor)
            if stage in ("cumulative_g", "decay_mask", "v_new"):
                matrices[stage] = tensor
            if stage == "state":
                start = len(outputs) * 64
                q = query[..., start : start + 64, :]
                k = output_key[..., start : start + 64, :]
                # HF's decay_mask is lower triangular including its diagonal.
                # Multiplication therefore masks QK exactly as in the HF source.
                attn = self.mul(self.mm(q, t.transpose(k, -1, -2)), matrices["decay_mask"])
                inter = self.mm(self.mul(q, self.exp(matrices["cumulative_g"])), carry["state"])
                output = self.add(inter, self.mm(attn, matrices["v_new"]))
                outputs.append(output)
                if observer:
                    for label, item in (
                        ("query_scaled", q),
                        ("output_attn", attn),
                        ("attn_inter", inter),
                        ("chunk_output_fp32", output),
                    ):
                        observer(prefix + "/" + label, item)
                carry["state"] = tensor
                matrices.clear()

        # The state graph allocates each new state; no in-place mutation aliases
        # carry['state']. Computing outputs afterward preserves its arithmetic.
        state = self.state_call(key, value, g, beta, initial, observer=collect)
        padded_output = t.concat(outputs, dim=-2, memory_config=self.mem) if len(outputs) > 1 else outputs[0]
        shape = tuple(padded_output.shape)
        cropped = t.slice(padded_output, (0, 0, 0, 0), (shape[0], shape[1], n, shape[3]), memory_config=self.mem)
        output_fp32 = t.permute(cropped, (0, 2, 1, 3), memory_config=self.mem)
        output = t.typecast(output_fp32, t.bfloat16, memory_config=self.mem)
        return output, state, output_fp32


def attach_state_call():
    combined.attach_call()
    OutputChunk64Graph.state_call = ordered.OrderedChunk64StateGraph.__call__
    return copy.deepcopy(combined.attach_call.proof)


def cpu_outputs(query, operands, initial, state_matrices):
    """HF output equations, sharing proven CPU state matrices unchanged."""
    n = query.shape[-2]
    query = query * (1 / (query.shape[-1] ** 0.5))
    query, key = [torch.nn.functional.pad(x, (0, 0, 0, (-n) % 64)) for x in (query, operands["key"])]
    outputs = []
    saved = {}
    state = initial
    for start in range(0, n, 64):
        prefix = f"chunk{start // 64}/"
        q, k = query[..., start : start + 64, :], key[..., start : start + 64, :]
        cumulative, decay, v_new = [state_matrices[prefix + name] for name in ("cumulative_g", "decay_mask", "v_new")]
        attn = q @ k.transpose(-1, -2) * decay
        inter = (q * cumulative.exp()) @ state
        output = inter + attn @ v_new
        outputs.append(output)
        for name, tensor in (
            ("query_scaled", q),
            ("output_attn", attn),
            ("attn_inter", inter),
            ("chunk_output_fp32", output),
        ):
            saved[prefix + name] = tensor.clone()
        state = state_matrices[prefix + "state"]
    return torch.cat(outputs, dim=-2)[..., :n, :].transpose(1, 2).contiguous(), state, saved


def cpu_setup(args):
    result, payload = base.cpu_setup(args)
    raw = payload["raw_operands"]
    normalized_q = base.l2norm(raw["q"], dim=-1, eps=1e-6)
    assert base.tensor_sha(normalized_q) == result["normalized_operand_sha256"]["query"]
    result.update(
        classification="same-operand full chunk64 output/state diagnostic; not decoder acceptance",
        script_sha256=base.sha(__file__),
        cumsum="double_single",
        sum_order=args.sum_order,
        output_shape="[B,logical_S,H,Dv]",
        output_dtype="BF16; FP32 pre-cast outputs also compared",
        output_equation="(q*exp(cumulative_g))@state_before + ((q@k.T)*lower_decay_mask)@v_new; q scaled by1/sqrt(Dk)",
        hf_output_function_sha256=base.sha(inspect.getfile(base.torch_chunk_gated_delta_rule)),
    )
    for row in result["cases"]:
        n = row["length"]
        item = payload["cases"][n]
        q = normalized_q[:, :n]
        query = q.transpose(1, 2).contiguous().float()
        normalized_k = item["operands"]["key"].transpose(1, 2).contiguous()
        with torch.no_grad():
            expected, expected_state = base.torch_chunk_gated_delta_rule(
                q,
                normalized_k.bfloat16(),
                raw["v"][:, :n],
                g=raw["g"][:, :n],
                beta=raw["beta"][:, :n],
                initial_state=payload["initial"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            expected_fp32, fp32_state = base.torch_chunk_gated_delta_rule(
                q.float(),
                normalized_k,
                raw["v"][:, :n].float(),
                g=raw["g"][:, :n],
                beta=raw["beta"][:, :n],
                initial_state=payload["initial"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            got, state, output_matrices = cpu_outputs(query, item["operands"], payload["initial"], item["matrices"])
        assert torch.equal(expected_state, item["canonical"]) and torch.equal(fp32_state, expected_state)
        checks = dict(
            output_fp32=base.code_compare(expected_fp32, got),
            output_bf16=base.code_compare(expected, got.bfloat16()),
            state_fp32=base.code_compare(expected_state, state),
        )
        assert all(v["fp32_code_differences"] == 0 and v["bf16_code_differences"] == 0 for v in checks.values()), checks
        item.update(query=query, canonical_output=expected, canonical_output_fp32=expected_fp32)
        item["matrices"].update(output_matrices)
        row["cpu_full_output_state_transcription_vs_hf"] = checks
    return result, payload


def dependency_metadata(output):
    paths = [
        Path(__file__),
        Path(base.__file__),
        Path(ordered.__file__),
        Path(helper.__file__),
        Path(combined.__file__),
    ]
    dependencies = []
    for i, path in enumerate(paths):
        snapshot = Path(str(output) + f".dependency{i}.{path.name}.snapshot")
        snapshot.write_bytes(path.read_bytes())
        dependencies.append(
            dict(
                path=str(path.resolve()),
                sha256=base.sha(path),
                snapshot=str(snapshot),
                snapshot_sha256=base.sha(snapshot),
            )
        )
    runtime = ast.parse(Path(__file__).read_text())
    cls = next(n for n in runtime.body if isinstance(n, ast.ClassDef) and n.name == "OutputChunk64Graph")
    calls = [ast.unparse(n.func) for n in ast.walk(cls) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    return dict(
        dependencies=dependencies,
        unchanged_state_call_ast_proof=attach_state_call(),
        output_runtime_ast_audit="No Torch/from_torch/to_torch; metadata shapes converted to tuple before slicing",
        acceptance_eligible=False,
    )


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), "Preserve previous evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = dependency_metadata(args.output)
    result, payload = cpu_setup(args)
    result.update(identity, status="cpu_setup_pass")
    matrices_path = Path(str(args.output) + ".cpu_matrices.pt")
    torch.save(payload, matrices_path)
    result.update(cpu_matrix_snapshot=str(matrices_path), cpu_matrix_snapshot_sha256=base.sha(matrices_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    provenance = Path(str(args.output) + ".provenance.json")
    data = json.loads(provenance.read_text())
    data.update(identity)
    provenance.write_text(json.dumps(data, indent=2) + "\n")
    save()
    if args.oracle_only:
        assert "ttnn" not in sys.modules
        print("CPU_CHUNK64_OUTPUT_STATE_PASS", args.initial_state, args.lengths)
        return
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(key) in (None, "", "0"), key
    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False)

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        graph = OutputChunk64Graph(ttnn, mesh, upload, variant=args.variant, cumsum=args.cumsum)
        graph.sum_order = args.sum_order
        device_payload = {}
        for row in result["cases"]:
            n = row["length"]
            item = payload["cases"][n]
            observed = {}

            def observe(name, tensor):
                actual = ttnn.to_torch(tensor).clone()
                assert torch.isfinite(actual).all(), name
                observed[name] = actual
                row["device_intermediates"][name] = base.code_compare(item["matrices"][name], actual)
                save()

            inputs = {key: upload(value) for key, value in item["operands"].items()}
            output, state, fp32_output = graph(
                upload(item["query"]), **inputs, initial=upload(payload["initial"]), observer=observe
            )
            out, st, f32 = [ttnn.to_torch(tensor).clone() for tensor in (output, state, fp32_output)]
            assert tuple(out.shape) == tuple(item["canonical_output"].shape) and out.dtype == torch.bfloat16
            assert all(torch.isfinite(tensor).all() for tensor in (out, st, f32))
            row["device_output_bf16_vs_hf"] = base.code_compare(item["canonical_output"], out)
            row["device_output_fp32_vs_hf"] = base.code_compare(item["canonical_output_fp32"], f32)
            row["device_state_vs_hf"] = base.code_compare(item["canonical"], st)
            row["device_state_vs_cpu_sequential"] = base.code_compare(item["sequential"], st)
            device_payload[n] = dict(output=out, state=st, output_fp32=f32, intermediates=observed)
            save()
        tensors_path = Path(str(args.output) + ".device_matrices.pt")
        torch.save(device_payload, tensors_path)
        result.update(
            status="completed_diagnostic",
            device_matrix_snapshot=str(tensors_path),
            device_matrix_snapshot_sha256=base.sha(tensors_path),
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        assert all(base.sha(Path(d["path"])) == d["sha256"] for d in identity["dependencies"])
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 127, 65, 1])
    p.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    p.add_argument("--variant", choices=["native", "sfpu_reduce", "sfpu_fma"], default="sfpu_fma")
    p.add_argument("--sum-order", choices=["native", "canonical"], default="canonical")
    p.add_argument(
        "--cumsum",
        choices=["plain", "compensated"],
        default="compensated",
        help="Compatibility only; double-single replaces it",
    )
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
