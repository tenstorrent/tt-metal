# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Execute extracted norm/layout/consumer methods without importing TTNN."""

import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL.parents[2]
OUT = MODEL / "doc/datatype_sweep/norm_guard_host_check.json"
paths = {
    name: MODEL / "tt" / name
    for name in ("optimized_decoder.py", "multichip_decoder.py", "prepared_prefill.py", "generator.py")
}
sources = {name: path.read_text() for name, path in paths.items()}
trees = {name: ast.parse(source, filename=str(paths[name])) for name, source in sources.items()}


def class_node(file, name):
    return next(node for node in trees[file].body if isinstance(node, ast.ClassDef) and node.name == name)


def assignment(cls, name):
    return next(
        node.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )


base_class = class_node("optimized_decoder.py", "OptimizedDecoder")
multichip_class = class_node("multichip_decoder.py", "MultichipDecoder")
base_policy = ast.literal_eval(assignment(base_class, "POLICY"))
multichip_policy = dict(base_policy)
for keyword in assignment(multichip_class, "POLICY").keywords:
    if keyword.arg is None:
        multichip_policy.update(ast.literal_eval(keyword.value))
    else:
        multichip_policy[keyword.arg] = ast.literal_eval(keyword.value)
kind_policies = ast.literal_eval(assignment(multichip_class, "KIND_POLICY"))
prepared_policy = ast.literal_eval(assignment(class_node("prepared_prefill.py", "PreparedPrefill"), "POLICY"))
buckets = ast.literal_eval(assignment(class_node("generator.py", "MuseGlimmerGenerator"), "BUCKETS"))
method_names = ("_width_mem", "_sharded_prefill_norm", "_norm")
methods = [
    copy.deepcopy(next(node for node in base_class.body if isinstance(node, ast.FunctionDef) and node.name == name))
    for name in method_names
]


@dataclass(eq=True)
class Memory:
    layout: str
    buffer: str
    shard_spec: object = None


@dataclass(eq=True)
class Shard:
    cores: object
    shape: object
    orientation: str


@dataclass
class Tensor:
    padded_shape: tuple
    dtype: str = "bfloat16"
    memory: object = None


def core_set(count, grid, row_wise=True):
    assert row_wise
    assert 0 < count <= grid.x * grid.y
    return [(i % grid.x, i // grid.x) for i in range(count)]


def create_sharded_memory_config(shape, grid, strategy, orientation, use_height_and_width_as_shard_shape):
    assert use_height_and_width_as_shard_shape
    return Memory(strategy, "L1", Shard(core_set(grid.x * grid.y, grid), list(shape), orientation))


def rms_norm(x, **kwargs):
    assert x.memory == kwargs["memory_config"]
    return kwargs


fake_ttnn = SimpleNamespace(
    bfloat16="bfloat16",
    TensorMemoryLayout=SimpleNamespace(WIDTH_SHARDED="WIDTH"),
    BufferType=SimpleNamespace(L1="L1"),
    ShardStrategy=SimpleNamespace(WIDTH="WIDTH", BLOCK="BLOCK"),
    ShardOrientation=SimpleNamespace(ROW_MAJOR="ROW_MAJOR"),
    CoreCoord=lambda x, y: SimpleNamespace(x=x, y=y),
    CoreGrid=lambda x, y: SimpleNamespace(x=x, y=y),
    num_cores_to_corerangeset=core_set,
    MemoryConfig=Memory,
    ShardSpec=Shard,
    create_sharded_memory_config=create_sharded_memory_config,
    LayerNormShardedMultiCoreProgramConfig=lambda **kwargs: kwargs,
    DRAM_MEMORY_CONFIG=Memory("INTERLEAVED", "DRAM"),
    L1_MEMORY_CONFIG=Memory("INTERLEAVED", "L1"),
    to_memory_config=lambda x, mem: Tensor(x.padded_shape, x.dtype, mem),
    typecast=lambda x, dtype: Tensor(x.padded_shape, dtype, x.memory),
    rms_norm=rms_norm,
)


class RemoveGuard(ast.NodeTransformer):
    count = 0

    def visit_If(self, node):
        if ast.unparse(node.test) == "height // gy <= 128":
            assert not node.orelse
            self.count += 1
            return node.body
        return self.generic_visit(node)


def build(old=False):
    selected = copy.deepcopy(methods)
    if old:
        remover = RemoveGuard()
        selected = [remover.visit(method) for method in selected]
        assert remover.count == 1
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"ttnn": fake_ttnn, "math": math}
    exec(
        compile(
            module, str(paths["optimized_decoder.py"]) + ("::without_guard" if old else "::source_methods"), "exec"
        ),
        namespace,
    )
    return type("ExtractedNorm", (), {name: namespace[name] for name in method_names})


New, Old = build(), build(old=True)
norm_names = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")


def instance(cls, kind, policy=None, decode=False):
    obj = cls()
    obj.policy = dict(multichip_policy, **kind_policies[kind])
    obj.policy.update(policy or {})
    obj._decode = decode
    obj.grid = tuple(obj.policy["grid"])
    obj.sharded_norm = True
    obj.residual_mem = obj._width_mem(6656, obj.grid, exact=True)
    obj.config = SimpleNamespace(rms_norm_eps=1e-6)
    obj.ck = "default_compute_config"
    obj.norm_configs = {
        phase: {name: f"{kind}:{phase}:{name}" for name in norm_names} for phase in ("prefill", "decode")
    }
    obj.norm_weights = {name: f"row_major_gamma:{name}" for name in norm_names}
    obj.weights = {name: f"interleaved_gamma:{name}" for name in norm_names}
    return obj


def run(cls, kind, height, policy=None, decode=False, name="input_layernorm", dtype="bfloat16"):
    obj = instance(cls, kind, policy, decode)
    result = obj._norm(Tensor((1, 1, height, 6656), dtype), name)
    mem = result["memory_config"]
    shard = mem.shard_spec
    program = result.get("program_config")
    row = {"layout": mem.layout, "buffer": mem.buffer}
    if shard:
        grid = list(program["compute_with_storage_grid_size"])
        assert all(0 <= x < grid[0] and 0 <= y < grid[1] for x, y in shard.cores)
        assert program["block_h"] * 32 == shard.shape[0]
        assert program["block_w"] * 32 == shard.shape[1]
        assert program["block_w"] % program["subblock_w"] == 0
        assert shard.orientation == "ROW_MAJOR"
        row.update(
            grid=grid,
            cores=len(shard.cores),
            shard_shape=list(shard.shape),
            modeled_bf16_shard_bytes=math.prod(shard.shape) * 2,
            program=program,
        )
    else:
        assert program is None
    return row, {key: value for key, value in result.items() if key not in ("memory_config", "program_config")}


checks = 0
expected_changed = [352, 416, 544, 608, 704, 736, 800, 832, 864, 928, 960, 992]
rows = []
for kind in kind_policies:
    changed = []
    for height in range(32, 1057, 32):
        for name in norm_names:
            current, current_math = run(New, kind, height, name=name)
            previous, previous_math = run(Old, kind, height, name=name)
            assert current_math == previous_math
            differs = current != previous
            assert differs == (height in expected_changed)
            if differs:
                assert previous["layout"] == "BLOCK" and current["layout"] == "WIDTH"
                assert previous["shard_shape"][0] > 128
                assert current["grid"] == [11, 10]
                assert current["cores"] == 104 and current["shard_shape"] == [height, 64]
            elif current["layout"] == "BLOCK":
                assert current["shard_shape"][0] <= 128
            checks += 1
            if kind == "sliding_attention" and name == "input_layernorm":
                rows.append(
                    {
                        "padded_height": height,
                        "changed": differs,
                        "old": {key: value for key, value in previous.items() if key != "program"},
                        "new": {key: value for key, value in current.items() if key != "program"},
                    }
                )
        if differs:
            changed.append(height)
    assert changed == expected_changed

boundaries = {}
for height in (640, 896, 1024):
    current, _ = run(New, "sliding_attention", height)
    assert current["layout"] == "BLOCK" and current["shard_shape"][0] == 128
    boundaries[str(height)] = current
    checks += 1

bucket_results = []
for kind in kind_policies:
    for bucket in buckets:
        policy = prepared_policy if bucket <= 64 else None
        physical_height = 32 if bucket == 64 else bucket
        current = run(New, kind, physical_height, policy)
        previous = run(Old, kind, physical_height, policy)
        assert current == previous
        if bucket <= 64:
            assert current[0]["layout"] == "WIDTH" and current[0]["grid"] == [8, 8]
        elif bucket <= 1024:
            assert current[0]["layout"] == "BLOCK"
        else:
            assert current[0] == {"layout": "INTERLEAVED", "buffer": "DRAM"}
        checks += 1
        if kind == "sliding_attention":
            bucket_results.append(
                {"bucket": bucket, "norm_physical_height": physical_height, "unchanged": True, "geometry": current[0]}
            )

branch_checks = []
for kind in kind_policies:
    for label, height, kwargs in (
        ("decode", 32, {"decode": True}),
        ("unnamed_qk", 992, {"name": None}),
        ("sharded_norm_disabled", 992, {"policy": {"prefill_norm_sharded": False}}),
        ("explicit_width_default_grid", 992, {"policy": {"prefill_norm_block": False}}),
        ("explicit_width_custom_grid", 992, {"policy": {"prefill_norm_block": False, "prefill_norm_grid": [8, 8]}}),
        ("short_prepared_width", 32, {"policy": prepared_policy}),
    ):
        assert run(New, kind, height, **kwargs) == run(Old, kind, height, **kwargs)
        branch_checks.append({"kind": kind, "branch": label, "unchanged": True})
        checks += 1
    for height in (352, 640, 992, 1024):
        assert run(New, kind, height, dtype="bfloat8_b") == run(New, kind, height)
        checks += 1
    for height in expected_changed:
        custom, _ = run(New, kind, height, policy={"prefill_norm_grid": [8, 8]})
        assert custom["layout"] == "WIDTH" and custom["grid"] == [8, 8]
        assert custom["cores"] == 52 and custom["shard_shape"] == [height, 128]
        checks += 1


# Execute the production wrapper and base class together so zero-argument
# super(), actual slicing, and concat control flow participate in the checks.
# The stubs carry row identities, making row loss, duplication, and reordering
# observable without reproducing the guard's decision in the test harness.
@dataclass
class RowTensor:
    shape: tuple
    dtype: str
    memory: object
    row_ids: tuple

    @property
    def padded_shape(self):
        return self.shape[:-2] + (math.ceil(self.shape[-2] / 32) * 32, self.shape[-1])

    def is_sharded(self):
        return self.memory.shard_spec is not None


norm_calls, matmul_calls = [], []


def move_rows(x, memory):
    return RowTensor(x.shape, x.dtype, memory, x.row_ids)


def slice_rows(x, starts, stops):
    assert starts[:2] == [0, 0] and stops[:2] == list(x.shape[:2])
    assert starts[-1] == 0 and stops[-1] == x.shape[-1]
    assert 0 <= starts[-2] < stops[-2] <= x.shape[-2]
    return RowTensor(
        tuple(stop - start for start, stop in zip(starts, stops)),
        x.dtype,
        x.memory,
        x.row_ids[starts[-2] : stops[-2]],
    )


def concat_rows(parts, dim, memory_config):
    assert dim == 2 and parts
    assert all(p.dtype == parts[0].dtype and p.shape[-1] == parts[0].shape[-1] for p in parts)
    shape = parts[0].shape[:2] + (sum(p.shape[2] for p in parts), parts[0].shape[-1])
    return RowTensor(shape, parts[0].dtype, memory_config, sum((p.row_ids for p in parts), ()))


def norm_rows(x, **kwargs):
    assert x.memory == kwargs["memory_config"]
    norm_calls.append(dict(kwargs, shape=x.shape, padded_shape=x.padded_shape, row_ids=x.row_ids))
    return move_rows(x, kwargs["memory_config"])


def matmul_rows(x, weight, **kwargs):
    matmul_calls.append(dict(kwargs, input=x))
    width = weight.output_width // (2 if kwargs.get("fuse_swiglu") else 1)
    return RowTensor(x.shape[:-1] + (width,), kwargs["dtype"], kwargs["memory_config"], x.row_ids)


wrapper_ttnn = SimpleNamespace(**vars(fake_ttnn))
wrapper_ttnn.bfloat8_b = "bfloat8_b"
wrapper_ttnn.to_memory_config = move_rows
wrapper_ttnn.typecast = lambda x, dtype: RowTensor(x.shape, dtype, x.memory, x.row_ids)
wrapper_ttnn.slice = slice_rows
wrapper_ttnn.concat = concat_rows
wrapper_ttnn.rms_norm = norm_rows
wrapper_ttnn.MinimalMatmulConfig = lambda **kwargs: kwargs
wrapper_ttnn.experimental = SimpleNamespace(minimal_matmul=matmul_rows)


def method(cls, name):
    return copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))


class RemoveChunkGuard(ast.NodeTransformer):
    count = 0

    def visit_If(self, node):
        expected = ast.parse(
            "name and self._sharded_prefill_norm(x) and self.policy.get('prefill_norm_block')", mode="eval"
        )
        if ast.dump(node.test) == ast.dump(expected.body):
            assert not node.orelse
            self.count += 1
            return None
        return self.generic_visit(node)


def build_wrapper(old=False):
    wrapper_method = method(multichip_class, "_norm")
    if old:
        remover = RemoveChunkGuard()
        wrapper_method = remover.visit(wrapper_method)
        assert remover.count == 1
    classes = [
        ast.ClassDef(
            name="ExtractedBase",
            bases=[],
            keywords=[],
            body=[method(base_class, name) for name in (*method_names, "_prefill_input_l1", "_linear")],
            decorator_list=[],
        ),
        ast.ClassDef(
            name="ExtractedWrapper",
            bases=[ast.Name(id="ExtractedBase", ctx=ast.Load())],
            keywords=[],
            body=[wrapper_method, method(multichip_class, "_mlp"), method(multichip_class, "_minimal_config")],
            decorator_list=[],
        ),
    ]
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    namespace = {"ttnn": wrapper_ttnn, "math": math}
    exec(compile(module, str(paths["multichip_decoder.py"]) + "::extracted_wrapper", "exec"), namespace)
    return namespace["ExtractedWrapper"]


Wrapper, PreviousWrapper = build_wrapper(), build_wrapper(old=True)


def run_wrapper(cls, kind, height, name="input_layernorm", policy=None, decode=False):
    obj = instance(cls, kind, policy, decode)
    obj.sharded_residual = False
    obj.dram_sharded = True
    # Both production gamma representations describe the same learned vector.
    obj.weights.update({name: name for name in norm_names})
    obj.norm_weights.update({name: name for name in norm_names})
    x = RowTensor((1, 1, height, 6656), "bfloat16", wrapper_ttnn.DRAM_MEMORY_CONFIG, tuple(range(height)))
    norm_calls.clear()
    result = obj._norm(x, name, eps=0.125)
    calls = list(norm_calls)
    assert result.shape == x.shape and result.row_ids == x.row_ids and result.dtype == x.dtype
    assert sum((call["row_ids"] for call in calls), ()) == x.row_ids
    phase = "decode" if decode else "prefill"
    for call in calls:
        assert call["epsilon"] == 0.125 and call["weight"] == name
        assert call["compute_kernel_config"] == obj.norm_configs.get(phase, {}).get(name, obj.ck)
    return obj, result, calls


def scratch(call, fp32=True):
    """Independent C++ buffer accounting; input/output borrow tensor memory.

    sharded_layernorm_factory_helpers.cpp:510-575,627-738. The default
    non-distributed RMSNorm has gamma, no beta/pre-add/mask, and no Welford.
    """
    mem = call["memory_config"]
    if mem.shard_spec is None:
        return {"rows": call["shape"][-2], "layout": "DRAM", "scratch": "existing interleaved path; not modeled"}
    pc = call["program_config"]
    height, width = pc["block_h"], pc["block_w"]
    blocks = len(mem.shard_spec.cores) if mem.layout == "WIDTH" else 8
    tile = 4096 if fp32 else 2048
    parts = {
        "scalers": 4096 + tile,
        "row_statistics": 3 * height * tile,
        "external": math.ceil(208 / width) * tile,
        "ex2pe": math.ceil(height / blocks) * tile,
        "gamma": width * 2048,
        "x_xmm": 2 * height * width * tile,
    }
    return {
        "rows": call["shape"][-2],
        "physical_rows": call["padded_shape"][-2],
        "layout": mem.layout,
        "shard_shape": list(mem.shard_spec.shape),
        "fp32_scratch": fp32,
        "modeled_payload_bytes": sum(parts.values()),
        "modeled_endpoint": 111744 + sum(parts.values()),
        "modeled_input_bytes": height * width * 2048,
        "modeled_buffer_bytes": parts,
    }


wrapper_checks = 0
affected = [544, 608, 704, 736, 800, 832, 864, 928, 960, 992]
wrapper_rows, bounded_children, downstream_rows = [], [], []
for kind in kind_policies:
    observed = []
    for height in range(32, 1057, 32):
        for name in norm_names:
            obj, result, calls = run_wrapper(Wrapper, kind, height, name)
            _, previous, previous_calls = run_wrapper(PreviousWrapper, kind, height, name)
            changed = len(calls) != len(previous_calls)
            assert changed == (height in affected)
            if not changed:
                assert result == previous and calls == previous_calls
            else:
                assert len(calls) == 2 and calls[0]["shape"][-2] == 512
                assert result.memory == wrapper_ttnn.DRAM_MEMORY_CONFIG
                assert all(call["padded_shape"][-2] <= 512 for call in calls)
            wrapper_checks += 1
            if kind == "sliding_attention" and name == "input_layernorm" and changed:
                children = [scratch(call) for call in calls]
                bounded_children.extend(children)
                wrapper_rows.append({"height": height, "children": children, "output_memory": "DRAM"})
            if changed and name == "input_layernorm":
                # Execute the actual consumer methods, including their memory
                # conversion decisions. Only the matmul operation is stubbed.
                obj.prefill_weights = {
                    "qkv": SimpleNamespace(output_width=2304),
                    "mlp.down_proj": SimpleNamespace(output_width=6656),
                }
                obj.minimal_gate_up = SimpleNamespace(output_width=9984)
                obj.mm_ck = {name: name for name in ("qkv", "mlp.gate_up", "mlp.down_proj")}
                obj._active_projection = "qkv"
                matmul_calls.clear()
                obj._linear(result, "qkv")
                obj._mlp(result)
                assert len(matmul_calls) == 3
                for call in matmul_calls[:2]:
                    consumer = call["input"]
                    expected_buffer = "L1" if kind == "full_attention" else "DRAM"
                    assert consumer.memory == Memory("INTERLEAVED", expected_buffer)
                    assert consumer.shape == result.shape and consumer.row_ids == result.row_ids
                    assert call["config"]["M_block_size"] == 4
                bank_bytes = math.ceil((height // 32) * 208 / 110) * 2048 if kind == "full_attention" else 0
                assert bank_bytes <= (height // 32) * 2 * 2048
                downstream_rows.append(
                    {
                        "kind": kind,
                        "height": height,
                        "qkv_mlp_input_memory": expected_buffer,
                        "modeled_input_bytes_per_bank": bank_bytes,
                    }
                )
                wrapper_checks += 1
        if changed:
            observed.append(height)
    assert observed == affected
    for bucket in buckets:
        physical = 32 if bucket == 64 else bucket
        options = prepared_policy if bucket <= 64 else None
        current = run_wrapper(Wrapper, kind, physical, policy=options)[1:]
        previous = run_wrapper(PreviousWrapper, kind, physical, policy=options)[1:]
        assert current == previous
        wrapper_checks += 1
    for label, height, kwargs in (
        ("decode", 32, {"decode": True}),
        ("unnamed_qk", 992, {"name": None}),
        ("disabled_sharding", 992, {"policy": {"prefill_norm_sharded": False}}),
        ("explicit_width", 992, {"policy": {"prefill_norm_block": False}}),
        ("smaller_existing_chunk", 992, {"policy": {"prefill_norm_chunk": 256}}),
    ):
        assert (
            run_wrapper(Wrapper, kind, height, **kwargs)[1:] == run_wrapper(PreviousWrapper, kind, height, **kwargs)[1:]
        )
        wrapper_checks += 1
    # Logical lengths exercise a partial final tile through the actual slices.
    for logical in (513, 543, 577, 769, 897, 927, 961, 991):
        _, result, calls = run_wrapper(Wrapper, kind, logical)
        assert len(calls) == 2 and result.shape[-2] == logical
        assert result.padded_shape[-2] == math.ceil(logical / 32) * 32
        wrapper_checks += 1

sharded_children = [child for child in bounded_children if child["layout"] != "DRAM"]
assert max(child["physical_rows"] for child in sharded_children if child["layout"] == "WIDTH") == 416
assert max(child["modeled_endpoint"] for child in sharded_children) == 926848
assert max(child["modeled_endpoint"] + child["modeled_input_bytes"] for child in sharded_children) == 1045632
_, _, failed_calls = run_wrapper(PreviousWrapper, "sliding_attention", 992)
failed_scratch = scratch(failed_calls[0])
assert failed_scratch["modeled_endpoint"] == 1442944
assert failed_scratch["modeled_payload_bytes"] == 1331200
assert 1078400 + max(row["modeled_input_bytes_per_bank"] for row in downstream_rows) == 1199232

accounting_paths = [
    ROOT / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp",
    ROOT / "ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp",
    ROOT / "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_descriptor.cpp",
    ROOT / "tt_metal/impl/buffers/buffer.cpp",
]

output = {
    "status": "pass",
    "scope": "host geometry only; no TTNN imports, device operations, numerical execution, or performance measurement",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "command": [sys.executable, *sys.argv],
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "method": "AST-extracted current _width_mem, _sharded_prefill_norm, and _norm executed with metadata-only TTNN stubs. Comparison removes exactly the height // gy <= 128 guard from a copied AST to recover prior selection. Policies and generator buckets read using standard-library AST evaluation.",
    "sources": {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(sources[name].encode()).hexdigest()}
        for name, path in paths.items()
    },
    "case_count": checks + wrapper_checks,
    "base_layout_case_count": checks,
    "wrapper_case_count": wrapper_checks,
    "padded_heights": {"first": 32, "last": 1056, "step": 32, "count": 33},
    "layer_kinds": list(kind_policies),
    "named_norms": list(norm_names),
    "changed_heights": expected_changed,
    "exact_128_row_boundaries": boundaries,
    "all_height_results": rows,
    "generator_buckets": bucket_results,
    "unaffected_branches": branch_checks,
    "wrapper_guard": {
        "method": "Actual MultichipDecoder._norm AST inherits extracted OptimizedDecoder methods. Metadata tensors track every row through source slice/concat calls. Previous-wrapper comparison removes only the new chunk-policy conditional. Actual _linear and _mlp methods execute their input-memory decisions with matmul stubbed.",
        "affected_physical_heights": affected,
        "children": wrapper_rows,
        "max_width_child_rows": 416,
        "max_modeled_fp32_child_endpoint": 926848,
        "max_modeled_fp32_child_endpoint_plus_input": 1045632,
        "failed_992_width_accounting": failed_scratch,
        "downstream_consumers": downstream_rows,
        "max_modeled_full_input_bytes_per_bank": 120832,
        "modeled_baseline_full_qkv_endpoint_plus_input": 1199232,
        "accounting_sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in accounting_paths
        },
        "preserved_contracts": [
            "Each input row normalized exactly once, in order",
            "Logical and padded shapes, dtype, epsilon, gamma identity, and compute configuration preserved",
            "All generator buckets unchanged",
            "Decode, unnamed norm, disabled sharding, explicit WIDTH, and existing smaller chunk limit unchanged",
        ],
    },
    "additional_passed_checks": [
        "Program grid contains every shard core",
        "Program block dimensions match shard dimensions and subblock divides width",
        "Default fallback retains WIDTH grid 11x10, 104 cores, width64",
        "Custom WIDTH grid 8x8 remains 52 cores, width128",
        "Input BF16 conversion leaves geometry unchanged",
        "Epsilon, gamma identity, and compute-config identity unchanged",
    ],
    "limits": [
        "TTNN stubs validate Python geometry selection, not the TTNN runtime, shader reduction order, actual allocation addresses, or numerical accuracy",
        "Byte figures are modeled, not measured: wrapper evidence includes non-distributed sharded RMSNorm scratch with FP32 as an upper bound; interleaved 32-row tail scratch is not modeled",
        "Consumer staging assumes 110 L1 banks. Scratch plus input figures exclude other live tensors and allocator fragmentation",
        "Generator 64 bucket uses its existing 32+32 decomposition; this harness checks its norm geometry only",
        "Full stage gates, device eager tests, trace checks, and accuracy/performance evidence remain separate",
    ],
}
OUT.write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": output["status"],
            "case_count": output["case_count"],
            "changed_heights": expected_changed,
            "wrapper_affected_heights": affected,
            "output": str(OUT),
        }
    )
)
