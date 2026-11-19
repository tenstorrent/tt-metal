# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build-time-only static integration; default audits without emitting a decoder.

No TTNN/Torch imports or hardware. --emit requires an existing full-output device
result and is intended only after the parent has approved that experiment.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
INPUTS = {
    E
    / "v6_inline_router_silu_chunk64_double_single.py.snapshot": "dcaa36081fbce0169fb625a7ff2438643b0180a6a34379d96694330b6cdb668c",
    ROOT / "tests/v6_chunk64_output_probe.py": "4c17aca9c01f3e19ed60b99a5f4880bf7a83053713778316c19cba781737b5d6",
    ROOT / "tests/v6_mkl_exp_ttnn_probe.py": "8b262be19cf72ca69d553a65de656398a3efa8388f5e3ed45e9a1df5113787fa",
    ROOT / "tests/v6_mkl_exp_scaled_ttnn_probe.py": "c55f1249fae3671bdf98496d582346a947f5d27163ac2bc8a43b224346738748",
}
REFERENCES = {
    ROOT / "tests/v6_double_single_cumsum_probe.py": "fc5b300fa4e1fea9023cd6eb8e6123dea394b1cdb89a6fdf2c16e96a537cb419",
    ROOT
    / "tests/v6_chunk64_ordered_state_probe.py": "9e4e4d864ad235a0e1bb318d79f77c2a47e22aa3a06635b082c3d027b95d97cc",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def statements(source):
    return ast.parse(source).body


def expr(source):
    return ast.parse(source, mode="eval").body


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


def build(norm_mode, cached_singleton):
    for path, expected in (INPUTS | REFERENCES).items():
        assert verify_source(path, expected), path
    trees = [ast.parse(path.read_text()) for path in INPUTS]
    tree, output_tree, exp_tree, _ = trees
    before = copy.deepcopy(tree)
    decoder, state = named(tree, "FusedDecoder"), named(tree, "Chunk64StateGraph")
    state_call_before = digest(named(state, "__call__"))
    state_module_hash = digest(ast.Module(body=[named(state, "__call__")], type_ignores=[]))
    assert state_module_hash == "c9f581041918cb82627b78c5658a53472bab15f5c24f620a76e211f9335935d7"
    cumsum_tree = ast.parse((ROOT / "tests/v6_double_single_cumsum_probe.py").read_text())
    assert digest(named(tree, "ttnn_double_single_chunk_cumsum")) == digest(
        named(cumsum_tree, "ttnn_double_single_chunk_cumsum")
    )
    ordered_tree = ast.parse((ROOT / "tests/v6_chunk64_ordered_state_probe.py").read_text())
    assert digest(named(state, "triangular_sum")) == digest(
        named(named(ordered_tree, "OrderedChunk64StateGraph"), "triangular_sum")
    )

    # Materialize scaled exp with only the two proven arithmetic substitutions.
    table = copy.deepcopy(named(exp_tree, "mathematical_table"))
    table.name = "chunk64_mathematical_exp_table"
    table.body[1:1] = statements("import torch\nimport struct\nfrom decimal import Decimal, localcontext")
    exp_class = copy.deepcopy(named(exp_tree, "MKLNegativeExp"))
    exp_class.name = "Chunk64MKLNegativeExp"
    exp_class = Rename({"mathematical_table": table.name, "SAFE_ABS": "CHUNK64_EXP_SAFE_ABS"}).visit(exp_class)
    exp_init = named(exp_class, "__init__")
    exp_init.body.insert(0, ast.Import(names=[ast.alias(name="torch")]))
    exp_init.body += statements(
        "self.exponent_bias = t.from_torch(torch.tensor([[[[16 << 23]]]], dtype=torch.int64), "
        "dtype=t.uint32, layout=t.TILE_LAYOUT, device=mesh, memory_config=memory_config)"
    )
    replacements = 0
    for node in ast.walk(named(exp_class, "__call__")):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id == "exponent_bits":
            replacement = expr("t.add(ORIGINAL, self.exponent_bias, dtype=t.uint32, memory_config=mem)")
            replacement.args[0] = node.value
            node.value = replacement
            replacements += 1
        elif node.targets[0].id == "value":
            assert ast.unparse(node.value) == "mul(reduced, exponent)"
            node.value = expr("mul(mul(reduced, 0.0000152587890625), exponent)")
            replacements += 1
    assert replacements == 2
    exp_runtime = copy.deepcopy(named(exp_class, "__call__"))
    Rename({"CHUNK64_EXP_SAFE_ABS": "SAFE_ABS"}).visit(exp_runtime)
    assert digest(exp_runtime) == "1e6725c17c029178886b21457e3fac45ad8f5e516cd96c656abfd9c214f2c205"
    named(state, "__init__").body += statements(
        "self.mkl_negative_exp = Chunk64MKLNegativeExp(ttnn, mesh, upload, self.mem)"
    )
    named(state, "exp").body = statements("return self.mkl_negative_exp(a)")

    output_class = copy.deepcopy(named(output_tree, "OutputChunk64Graph"))
    output_class.name = "Chunk64OutputGraph"
    output_class.bases = [ast.Name(id="Chunk64StateGraph", ctx=ast.Load())]
    output_class.body.insert(1, statements("state_call = Chunk64StateGraph.__call__")[0])
    output_call_hash = digest(named(output_class, "__call__"))
    assert output_call_hash == digest(named(named(output_tree, "OutputChunk64Graph"), "__call__"))
    tree.body += statements("CHUNK64_EXP_SAFE_ABS = 87.33654022216797\nROW_SNAPSHOTS = (1, 2, 31, 63)")
    tree.body += [table, exp_class, output_class]
    Rename({"Chunk64StateGraph": "Chunk64OutputGraph"}).visit(named(decoder, "from_state_dict"))

    linear = named(decoder, "_linear_attention")
    # Public projection policy applies to both packs, including singleton tails.
    packed = next(
        n for n in linear.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "output_projected"
    )
    packed.value.keywords.append(ast.keyword(arg="projection_k", value=ast.Name(id="projection_k", ctx=ast.Load())))
    branch = next(
        n
        for n in linear.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "decode"
        and any(isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "initial" for x in n.body)
    )
    branch.orelse = statements(
        """
qq = ttnn.permute(ttnn.repeat_interleave(q, 2, dim=2), (0, 2, 1, 3))
kk = ttnn.permute(ttnn.repeat_interleave(k, 2, dim=2), (0, 2, 1, 3))
vv = ttnn.permute(ttnn.typecast(reshape(v, (1, s, 32, 128)), ttnn.float32), (0, 2, 1, 3))
bb = ttnn.permute(reshape(beta, (1, s, 32, 1)), (0, 2, 1, 3))
gg = ttnn.permute(reshape(g, (1, s, 32, 1)), (0, 2, 1, 3))
o, new, _ = self.chunk64_state(qq, kk, vv, gg, bb, state["recurrent"])
"""
    )
    norm_branch = next(
        n
        for n in linear.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not decode"
        and any(isinstance(x, ast.Return) for x in n.body)
    )
    if norm_mode == "native_padded":
        norm_branch.body = statements(
            """
m = ((s + 31) // 32) * 32
head_first = reshape(ttnn.permute(o, (0, 2, 1, 3)), (32, s, 128))
norm_gate = output_projected
if m != s:
    head_first = ttnn.pad(head_first, [(0, 0), (0, m-s), (0, 0)], 0.0, memory_config=DRAM)
    norm_gate = ttnn.pad(norm_gate, [(0, 0), (0, m-s), (0, 0)], 0.0, memory_config=DRAM)
gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(head_first, norm_gate, self.output_norm_bf16, 32,
    epsilon=self.config.rms_norm_eps, output_dtype=ttnn.bfloat16, gate_column_offset=2048,
    compute_kernel_config=self.compute)
gated = gated[:, :s, :]
merged = ttnn.multiply(gated, projection_z)
return self.linear(merged, "linear_attn.out_proj.weight")
"""
        )
    else:
        assert norm_mode == "explicit_silu"
        # Same per-head boundaries as repaired decode; no singleton concat-heads.
        norm_branch.body = statements(
            """
core = reshape(ttnn.typecast(o, ttnn.bfloat16), (1, s*32, 128))
wide = ttnn.typecast(core, ttnn.float32, memory_config=DRAM)
squares = ttnn.multiply(wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM)
sums = ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=DRAM)
variance = ttnn.multiply(sums, 1/128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM)
shifted = ttnn.add(variance, self.config.rms_norm_eps, memory_config=DRAM)
inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=DRAM)
normalized_fp32 = ttnn.multiply(wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=DRAM)
normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=DRAM)
weighted = ttnn.multiply(normalized, self.w["linear_attn.norm.weight"], dtype=ttnn.float32,
    fast_and_approximate_mode=False, activations=self.round_widen_chain, memory_config=DRAM)
silu_indices = ttnn.bitcast(projection_z, ttnn.uint16)
silu_indices = ttnn.reshape(ttnn.to_layout(silu_indices, ttnn.ROW_MAJOR_LAYOUT), (1, s*4096))
gate_rows = ttnn.gather(self.silu_table, 1, silu_indices)
gate_rows = ttnn.to_layout(ttnn.reshape(gate_rows, (1, s*32, 128)), ttnn.TILE_LAYOUT)
gated = ttnn.multiply(weighted, gate_rows, dtype=ttnn.bfloat16, fast_and_approximate_mode=False, memory_config=DRAM)
return self.linear(reshape(gated, (1, s, 4096)), "linear_attn.out_proj.weight")
"""
        )

    planner = named(decoder, "prepare_prefill")
    planner.body[0] = ast.Expr(
        value=ast.Constant(
            value=(
                "Prepare a host plan outside measured execution. Linear prefill uses "
                "chunks64 relative to this public call, including a logical final tail. "
                "Full-attention retains its aligned interior and singleton boundaries. "
                "All positive logical lengths and valid starts remain supported."
            )
        )
    )
    # Retain original full-attention validation/planner unchanged after early return.
    planner.body[1:1] = statements(
        """
if self.kind == "linear_attention":
    if not 0 <= start_pos < start_pos + seq_len <= self.config.max_position_embeddings:
        raise ValueError("logical sequence is outside HF context")
    if chunk_size <= 0:
        raise ValueError("physical chunk bound must be positive")
    return [(offset, min(64, seq_len-offset), start_pos+offset, None) for offset in range(0, seq_len, 64)]
"""
    )
    prefill = named(decoder, "prefill_forward")
    replacements = 0
    policy = "n == 1 and self.kind == 'full_attention'"
    if cached_singleton == "recurrent":
        policy = "n == 1 and (self.kind == 'full_attention' or hidden_states.shape[1] == 1)"
    for node in ast.walk(prefill):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._forward":
            for keyword in node.keywords:
                if keyword.arg == "decode":
                    assert ast.unparse(keyword.value) == "n == 1"
                    keyword.value = expr(policy)
                    replacements += 1
    assert replacements == 1
    assert digest(named(state, "__call__")) == state_call_before

    changed = {}
    for old_cls in [n for n in before.body if isinstance(n, ast.ClassDef)]:
        new_cls = named(tree, old_cls.name)
        for old in [n for n in old_cls.body if isinstance(n, ast.FunctionDef)]:
            new = named(new_cls, old.name)
            if digest(old) != digest(new):
                changed[f"{old_cls.name}.{old.name}"] = dict(before=digest(old), after=digest(new))
    allowed = {
        "FusedDecoder.from_state_dict",
        "FusedDecoder._linear_attention",
        "FusedDecoder.prepare_prefill",
        "FusedDecoder.prefill_forward",
        "Chunk64StateGraph.__init__",
        "Chunk64StateGraph.exp",
    }
    assert set(changed) == allowed, changed
    ast.fix_missing_locations(tree)
    source = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(tree)
        + "\n"
    )
    compile(source, "<static-chunk64-candidate>", "exec")
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(tree))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(c in ("exec", "compile", "eval") or c.endswith(".to_torch") for c in calls)
    runtime_classes = [state, output_class, exp_class]
    for cls in runtime_classes:
        for method in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"]:
            for node in ast.walk(method):
                if isinstance(node, ast.Call):
                    call = ast.unparse(node.func)
                    assert not call.startswith("torch.") and not call.endswith((".from_torch", ".to_torch")), call
    return source, dict(
        inputs={str(p): raw_sha256(p) for p in INPUTS},
        inputs_historical_sha256={str(p): h for p, h in INPUTS.items()},
        unchanged_helper_references={str(p): raw_sha256(p) for p in REFERENCES},
        unchanged_helper_references_historical_sha256={str(p): h for p, h in REFERENCES.items()},
        changed_methods=changed,
        embedded_output_call_ast_sha256=output_call_hash,
        unchanged_state_call_ast_sha256=state_call_before,
        state_call_module_ast_sha256=state_module_hash,
        embedded_scaled_exp_runtime_ast_sha256=digest(exp_runtime),
        output_norm=norm_mode,
        cached_public_singleton=cached_singleton,
        singleton_moe_policy="unchanged baseline shape-selected router/expert branches",
        planner="64 relative to public call; final positive logical tail; full-attention unchanged",
        static_runtime_audit="No diagnostic imports, runtime exec/compile/eval, or helper Torch/host-conversion calls",
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        acceptance_eligible=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm", choices=["native_padded", "explicit_silu"], default="native_padded")
    parser.add_argument("--cached-singleton", choices=["chunk", "recurrent"], default="recurrent")
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--component-result", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), "Preserve evidence"
    source, report = build(args.norm, args.cached_singleton)
    report.update(builder_source=str(Path(__file__).resolve()), builder_sha256=sha(__file__), emitted=False)
    if args.emit:
        assert args.component_result, "Require completed parent-owned full-output result before emitting"
        result = json.loads(args.component_result.read_text())
        assert result["status"] == "completed_diagnostic" and result["device_closed"]
        assert all("device_output_bf16_vs_hf" in row for row in result["cases"])
        assert not args.emit.exists(), "Preserve snapshots"
        args.emit.write_text(source)
        report.update(
            emitted=True,
            snapshot=str(args.emit),
            component_result=str(args.component_result),
            component_result_sha256=sha(args.component_result),
        )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("STATIC_BUILD_PASS", args.norm, report["candidate_sha256"], "emitted", report["emitted"])


if __name__ == "__main__":
    main()
