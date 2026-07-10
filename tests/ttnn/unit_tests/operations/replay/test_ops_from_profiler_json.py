# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Replay profiler-derived ops from a JSON dump.

Loads ops_input_params_*.json (deduped TILE-layout ops), builds random device
tensors matching each op's input shapes/dtypes/memory, invokes the matching
ttnn API, and compares against a torch / identity reference.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

TILE_HEIGHT = 16
TILE_WIDTH = 32
tile = ttnn.Tile((TILE_HEIGHT, TILE_WIDTH))

# Layout / memory moves that do not change tensor values.
IDENTITY_OP_CODES = {
    "InterleavedToShardedDeviceOperation",
    "ShardedToInterleavedDeviceOperation",
    "CopyDeviceOperation",
}

pytestmark = pytest.mark.use_module_device

JSON_PATH = Path(__file__).parent / "data" / "ops_input_params_md_m32_2026_07_09_06_09_28.json"

DTYPE_MAP = {
    "FLOAT32": ttnn.float32,
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "UINT32": ttnn.uint32,
    "INT32": ttnn.int32,
}

TORCH_DTYPE_MAP = {
    "FLOAT32": torch.float32,
    "BFLOAT16": torch.bfloat16,
    "BFLOAT8_B": torch.bfloat16,
    "BFLOAT4_B": torch.bfloat16,
    "UINT32": torch.int32,
    "INT32": torch.int32,
}

LAYOUT_MAP = {
    "TILE": ttnn.TILE_LAYOUT,
    "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
}

BINARY_OP_MAP = {
    "MUL": ttnn.mul,
    "ADD": ttnn.add,
    "SUB": ttnn.sub,
    "DIV": ttnn.div,
    "POWER": ttnn.pow,
}

UNARY_OP_MAP = {
    "FILL": None,  # handled specially
    "COS": ttnn.cos,
    "SIN": ttnn.sin,
    "RECIP": ttnn.reciprocal,
    "GELU": ttnn.gelu,
    "RELU": ttnn.relu,
    "EXP": ttnn.exp,
    "SQRT": ttnn.sqrt,
}


def load_ops():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["ops"]


OPS = load_ops()


def _parse_enum_tail(value: str) -> str:
    """'BinaryOpType::MUL' / 'DataType::FLOAT32' -> 'MUL' / 'FLOAT32'."""
    if value is None:
        return ""
    s = str(value).strip()
    if "::" in s:
        s = s.split("::")[-1]
    return s


def parse_dtype(name: str):
    # key = _parse_enum_tail(name)
    # if key not in DTYPE_MAP:
    #     raise ValueError(f"Unsupported dtype: {name}")
    # return DTYPE_MAP[key]
    # Force bfloat16 for all replay tensors/ops (ignore profiled dtype).
    del name  # profiled dtype intentionally unused
    return ttnn.bfloat16


def parse_layout(name: str):
    key = _parse_enum_tail(name) if name else "TILE"
    return LAYOUT_MAP.get(key, ttnn.TILE_LAYOUT)


def parse_shape_list(shape_str: str):
    """Parse 'Shape([1; 864; 1])' or 'Shape([0; 0; 0; 0])' into a list of ints."""
    if not shape_str or shape_str == "std::nullopt":
        return None
    nums = re.findall(r"-?\d+", shape_str)
    return [int(n) for n in nums]


def tile_padded_shape(logical_shape, tile_h: int = TILE_HEIGHT, tile_w: int = TILE_WIDTH):
    """Pad last two dims of a logical shape up to tile height/width multiples."""
    shape = list(logical_shape)
    if len(shape) >= 2:
        shape[-2] = ((shape[-2] + tile_h - 1) // tile_h) * tile_h
        shape[-1] = ((shape[-1] + tile_w - 1) // tile_w) * tile_w
    return shape


def parse_core_range_set(grid_blob: str) -> ttnn.CoreRangeSet:
    """Parse shard grid fragment with one or more {start/end} ranges."""
    ranges = []
    for m in re.finditer(
        r'"start"\s*:\s*\{\s*"x"\s*:\s*(\d+)\s*;\s*"y"\s*:\s*(\d+)\s*\}\s*;\s*'
        r'"end"\s*:\s*\{\s*"x"\s*:\s*(\d+)\s*;\s*"y"\s*:\s*(\d+)\s*\}',
        grid_blob,
    ):
        x0, y0, x1, y1 = map(int, m.groups())
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)))
    if not ranges:
        raise ValueError(f"Could not parse CoreRangeSet from: {grid_blob[:200]}")
    return ttnn.CoreRangeSet(set(ranges))


def parse_memory_config(mem_str: str | None, memory_tag: str | None = None):
    """
    Build a ttnn.MemoryConfig from a profiler MemoryConfig(...) string, or fall
    back to INTERLEAVED DRAM/L1 inferred from the DEV_* memory tag.
    """
    if mem_str and mem_str != "std::nullopt" and "MemoryConfig(" in mem_str:
        layout_m = re.search(r"TensorMemoryLayout::(\w+)", mem_str)
        buffer_m = re.search(r"BufferType::(\w+)", mem_str)
        layout_name = layout_m.group(1) if layout_m else "INTERLEAVED"
        buffer_name = buffer_m.group(1) if buffer_m else "DRAM"

        memory_layout = getattr(ttnn.TensorMemoryLayout, layout_name)
        buffer_type = getattr(ttnn.BufferType, buffer_name)

        if memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
            return ttnn.MemoryConfig(memory_layout, buffer_type)

        # Sharded: require shard_spec
        shape_m = re.search(r"shape=\[(\d+)\s*;\s*(\d+)\]", mem_str)
        orient_m = re.search(r"orientation=ShardOrientation::(\w+)", mem_str)
        if not shape_m:
            raise ValueError(f"Sharded MemoryConfig missing shape: {mem_str[:240]}")
        shard_shape = [int(shape_m.group(1)), int(shape_m.group(2))]
        orientation = getattr(ttnn.ShardOrientation, orient_m.group(1) if orient_m else "ROW_MAJOR")
        # Extract the grid=[{...}] region
        grid_m = re.search(r"grid=\[(.*)\]\s*;\s*shape=", mem_str, re.DOTALL)
        if not grid_m:
            raise ValueError(f"Sharded MemoryConfig missing grid: {mem_str[:240]}")
        core_grid = parse_core_range_set(grid_m.group(1))
        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, orientation)
        return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)

    # Fallback from DEV_1_DRAM_INTERLEAVED / DEV_1_L1_WIDTH_SHARDED style tags
    tag = (memory_tag or "").upper()
    if "L1" in tag:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def infer_concat_dim(inputs, output) -> int:
    """Infer concat axis from padded shapes (profiler dim can be ambiguous)."""
    out_shape = output["shape_padded"]
    for dim in range(len(out_shape)):
        if all(inp["shape_padded"][dim] == out_shape[dim] for inp in inputs):
            continue
        total = sum(inp["shape_padded"][dim] for inp in inputs)
        if total == out_shape[dim]:
            return dim
    # Fall back to last differing axis
    for dim in range(len(out_shape) - 1, -1, -1):
        if any(inp["shape_padded"][dim] != out_shape[dim] for inp in inputs):
            return dim
    return -1


def parse_unary_op(op_chain: str):
    """Extract UnaryOpType name and optional float params from op_chain string."""
    m = re.search(r"UnaryOpType::(\w+)", op_chain or "")
    if not m:
        raise ValueError(f"Could not parse UnaryOpType from: {op_chain}")
    op_name = m.group(1)
    params = []
    pm = re.search(r"param=\{([^}]*)\}", op_chain or "")
    if pm and pm.group(1).strip():
        params = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", pm.group(1))]
    return op_name, params


def parse_layernorm_program_config(pc_str: str):
    if not pc_str or pc_str == "std::nullopt":
        return None
    grid_m = re.search(r"compute_with_storage_grid_size=(\d+)-(\d+)", pc_str)
    sub_w = int(re.search(r"subblock_w=(\d+)", pc_str).group(1))
    block_h = int(re.search(r"block_h=(\d+)", pc_str).group(1))
    block_w = int(re.search(r"block_w=(\d+)", pc_str).group(1))
    inplace = bool(int(re.search(r"inplace=(\d+)", pc_str).group(1)))
    grid = ttnn.CoreCoord(int(grid_m.group(1)), int(grid_m.group(2)))
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid,
        subblock_w=sub_w,
        block_h=block_h,
        block_w=block_w,
        inplace=inplace,
    )


def parse_sdpa_program_config(pc_str: str):
    if not pc_str or pc_str == "std::nullopt":
        return None
    grid_m = re.search(r"compute_with_storage_grid_size=(\d+)-(\d+)", pc_str)
    q_chunk = int(re.search(r"q_chunk_size=(\d+)", pc_str).group(1))
    k_chunk = int(re.search(r"k_chunk_size=(\d+)", pc_str).group(1))
    exp_approx = bool(int(re.search(r"exp_approx_mode=(\d+)", pc_str).group(1)))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid_m.group(1)), int(grid_m.group(2))),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
    )


def make_random_torch(shape, datatype: str = "BFLOAT16") -> torch.Tensor:
    # Always generate bf16 host data; profiled datatype is ignored.
    del datatype
    return torch.randint(-10, 10, shape, dtype=torch.float32).to(torch.bfloat16) * 0.1


def create_tt_tensor(spec: dict, device, mem_config_override=None):
    """Create a random TILE tensor matching a JSON input/output tensor spec.

    Uses shape_logical so tile padding / broadcast semantics match the profiled
    op (e.g. reshape volume, ROW_B_COL_A binary broadcast). ttnn.from_torch
    applies TILE padding automatically. Always uses the module tile and bfloat16.

    Returns (torch_tensor, ttnn_tensor).
    """
    shape = list(spec.get("shape_logical") or spec["shape_padded"])
    layout = parse_layout(spec.get("layout", "TILE"))
    mem_config = mem_config_override or parse_memory_config(None, spec.get("memory"))

    torch_tensor = make_random_torch(shape)
    tt_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=layout,
        tile=tile,
        device=device,
        memory_config=mem_config,
    )
    return torch_tensor, tt_tensor


def _shard_covers_tensor(mem_config, shape_padded) -> bool:
    """True if sharded mem_config tiles the tensor's HxW exactly."""
    if not mem_config.is_sharded():
        return False
    ss = mem_config.shard_spec
    h, w = shape_padded[-2], shape_padded[-1]
    shard_h, shard_w = ss.shape[0], ss.shape[1]
    if h % shard_h != 0 or w % shard_w != 0:
        return False
    num_cores = ss.grid.num_cores()
    layout = mem_config.memory_layout
    if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return shard_h == h and num_cores * shard_w == w
    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return shard_w == w and num_cores * shard_h == h
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return (h // shard_h) * (w // shard_w) == num_cores
    return False


def _matmul_decode_input_configs(op: dict, device):
    """
    Build WIDTH-sharded configs for matmul_decode A[M,K] and B[K,N].

    Matches the profiled denoise MLP layout:
      A on 2 cores (shard [M, K/2])
      B width-sharded with the largest tile-aligned shard_w that fits the grid
      (e.g. [K, 64] x 64 cores for N=4096, [K, 32] x 32 cores for N=1024).
    """
    a_shape = op["inputs"][0]["shape_padded"]
    b_shape = op["inputs"][1]["shape_padded"]
    m, k = a_shape[-2], a_shape[-1]
    k_b, n = b_shape[-2], b_shape[-1]
    assert k == k_b, f"matmul_decode K mismatch: {k} vs {k_b}"

    grid_size = device.compute_with_storage_grid_size()
    max_cores = grid_size.x * grid_size.y

    num_a_cores = 2 if k % (2 * 32) == 0 else 1
    shard_w = None
    for candidate in (32, 64, 128, 256, 512):
        if n % candidate == 0 and (n // candidate) <= max_cores:
            shard_w = candidate
            break
    if shard_w is None:
        raise ValueError(f"Cannot width-shard B N={n} onto {max_cores} cores")
    num_b_cores = n // shard_w

    a_grid = ttnn.num_cores_to_corerangeset(num_a_cores, grid_size, True)
    b_grid = ttnn.num_cores_to_corerangeset(num_b_cores, grid_size, True)
    a_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(a_grid, [m, k // num_a_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )
    b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(b_grid, [k, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return [a_mem, b_mem]


def create_inputs(op: dict, device):
    """
    Create device tensors for an op. For sharded inputs, reuse an attribute
    MemoryConfig only when it exactly covers that tensor's HxW; otherwise
    derive a simple width/block shard from the tensor shape.

    Returns (torch_inputs, tt_inputs).
    """
    attrs = op.get("attributes") or {}

    if op["op_code"] == "MatmulDecodeDeviceOperation":
        mem_configs = _matmul_decode_input_configs(op, device)
        torch_inputs, tt_inputs = [], []
        for spec, mc in zip(op["inputs"], mem_configs):
            torch_t, tt_t = create_tt_tensor(spec, device, mem_config_override=mc)
            torch_inputs.append(torch_t)
            tt_inputs.append(tt_t)
        return torch_inputs, tt_inputs

    torch_inputs, tt_inputs = [], []
    for spec in op["inputs"]:
        mem_tag = spec.get("memory", "")
        mem_config = None

        if "SHARDED" in mem_tag.upper():
            for key in ("output_mem_config", "memory_config", "output_memory_config"):
                raw = attrs.get(key)
                if not (raw and "SHARDED" in str(raw).upper()):
                    continue
                try:
                    candidate = parse_memory_config(raw)
                    if _shard_covers_tensor(candidate, spec["shape_padded"]):
                        mem_config = candidate
                        break
                except Exception:
                    pass
            if mem_config is None:
                mem_config = _make_default_sharded_config(spec, device, mem_tag)
        else:
            mem_config = parse_memory_config(None, mem_tag)

        torch_t, tt_t = create_tt_tensor(spec, device, mem_config_override=mem_config)
        torch_inputs.append(torch_t)
        tt_inputs.append(tt_t)
    return torch_inputs, tt_inputs


def _make_default_sharded_config(spec: dict, device, mem_tag: str):
    """Fallback width/block shard covering the full tensor on as few cores as needed."""
    h, w = spec["shape_padded"][-2], spec["shape_padded"][-1]
    grid_size = device.compute_with_storage_grid_size()
    max_cores = grid_size.x * grid_size.y

    if "BLOCK" in mem_tag.upper():
        # Prefer a 1-row block shard along width (matches RMSNorm profile: 8x1, shard 32x128)
        shard_h = h
        # Pick a tile-aligned shard_w that fits on the grid
        for n_cores in range(1, min(max_cores, w // 32) + 1):
            if w % (n_cores * 32) == 0:
                shard_w = w // n_cores
                core_grid = ttnn.num_cores_to_corerangeset(n_cores, grid_size, True)
                return ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(core_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
                )
        raise ValueError(f"Could not build BLOCK shard for shape {[h, w]}")

    # WIDTH sharded
    for n_cores in range(1, min(max_cores, w // 32) + 1):
        if w % (n_cores * 32) == 0:
            shard_w = w // n_cores
            core_grid = ttnn.num_cores_to_corerangeset(n_cores, grid_size, True)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core_grid, [h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
            )
    raise ValueError(f"Could not build WIDTH shard for shape {[h, w]}")


def output_memory_config(op: dict):
    attrs = op.get("attributes") or {}
    for key in ("output_mem_config", "memory_config", "output_memory_config"):
        if key in attrs and attrs[key] not in (None, "std::nullopt"):
            try:
                return parse_memory_config(attrs[key])
            except Exception:
                pass
    if op.get("outputs"):
        return parse_memory_config(None, op["outputs"][0].get("memory"))
    return ttnn.DRAM_MEMORY_CONFIG


def run_op(op: dict, inputs: list, device):
    """Dispatch to the correct ttnn op and return the result."""
    op_code = op["op_code"]
    attrs = op.get("attributes") or {}
    out_mem = output_memory_config(op)

    if op_code == "BinaryNgDeviceOperation":
        binary_type = _parse_enum_tail(attrs.get("binary_op_type", ""))
        fn = BINARY_OP_MAP.get(binary_type)
        if fn is None:
            raise ValueError(f"Unsupported binary_op_type: {binary_type}")
        scalar = attrs.get("scalar", "std::nullopt")
        if scalar not in (None, "std::nullopt", "") and len(inputs) == 1:
            return fn(inputs[0], float(scalar), memory_config=out_mem)
        return fn(inputs[0], inputs[1], memory_config=out_mem)

    if op_code == "UnaryDeviceOperation":
        unary_name, params = parse_unary_op(attrs.get("op_chain", ""))
        if unary_name == "FILL":
            value = params[0] if params else 0.0
            return ttnn.fill(inputs[0], value, memory_config=out_mem)
        fn = UNARY_OP_MAP.get(unary_name)
        if fn is None:
            raise ValueError(f"Unsupported UnaryOpType: {unary_name}")
        if unary_name == "GELU":
            approx = bool(params and int(params[0]) == 1)
            return fn(inputs[0], fast_and_approximate_mode=approx, memory_config=out_mem)
        return fn(inputs[0], memory_config=out_mem)

    if op_code == "ReshapeViewDeviceOperation":
        # Prefer the profiled output logical shape (rank-matched to the tensor
        # view in the report). Fall back to the attribute Shape([...]).
        if op.get("outputs"):
            logical = list(op["outputs"][0]["shape_logical"])
        else:
            logical = parse_shape_list(attrs.get("logical_output_shape", ""))
        return ttnn.reshape(inputs[0], logical)

    if op_code == "ConcatDeviceOperation":
        dim = infer_concat_dim(op["inputs"], op["outputs"][0])
        return ttnn.concat(inputs, dim=dim, memory_config=out_mem)

    if op_code == "TypecastDeviceOperation":
        out_dtype = parse_dtype(attrs.get("output_dtype", op["outputs"][0]["datatype"]))
        return ttnn.typecast(inputs[0], out_dtype, memory_config=out_mem)

    if op_code == "InterleavedToShardedDeviceOperation":
        # Prefer parsed output shard config; fall back to default from output spec
        try:
            sharded_mem = parse_memory_config(attrs.get("output_mem_config"))
        except Exception:
            sharded_mem = _make_default_sharded_config(op["outputs"][0], device, op["outputs"][0]["memory"])
        keep_l1 = str(attrs.get("keep_l1_aligned", "false")).lower() == "true"
        out_dtype = parse_dtype(attrs.get("output_dtype", op["outputs"][0]["datatype"]))
        return ttnn.interleaved_to_sharded(inputs[0], sharded_mem, out_dtype, keep_l1_aligned=keep_l1)

    if op_code == "ShardedToInterleavedDeviceOperation":
        return ttnn.sharded_to_interleaved(inputs[0], out_mem)

    if op_code == "CopyDeviceOperation":
        # Profiler Copy DRAM->L1 is a memory-config move
        return ttnn.to_memory_config(inputs[0], out_mem)

    if op_code == "LayerNormDeviceOperation":
        eps = float(attrs.get("eps", "1e-06"))
        program_config = parse_layernorm_program_config(attrs.get("program_config", ""))
        weight = inputs[1] if len(inputs) > 1 else None
        bias = inputs[2] if len(inputs) > 2 else None
        norm_type = _parse_enum_tail(attrs.get("norm_type", "LAYERNORM"))
        kwargs = dict(epsilon=eps, weight=weight, memory_config=out_mem, program_config=program_config)
        if norm_type == "RMSNORM":
            # RMSNorm may still carry a bias tensor in the trace; pass if present
            if bias is not None:
                kwargs["bias"] = bias
            return ttnn.rms_norm(inputs[0], **kwargs)
        kwargs["bias"] = bias
        return ttnn.layer_norm(inputs[0], **kwargs)

    if op_code == "MatmulDeviceOperation":
        out_dtype = attrs.get("output_dtype")
        dtype = parse_dtype(out_dtype) if out_dtype and out_dtype != "std::nullopt" else None
        return ttnn.matmul(inputs[0], inputs[1], memory_config=out_mem, dtype=dtype)

    if op_code == "MatmulDecodeDeviceOperation":
        if not hasattr(ttnn, "matmul_decode"):
            pytest.skip("ttnn.matmul_decode not available in this build")
        partial = str(attrs.get("partial_width_sharded", "false")).lower() == "true"
        out_dtype = attrs.get("output_dtype")
        dtype = parse_dtype(out_dtype) if out_dtype and out_dtype != "std::nullopt" else None
        return ttnn.matmul_decode(inputs[0], inputs[1], partial_width_sharded=partial, dtype=dtype)

    if op_code == "NlpCreateHeadsDeviceOperation":
        num_q = int(attrs["num_q_heads"])
        num_kv = int(attrs.get("num_kv_heads", "1"))
        transpose_k = str(attrs.get("transpose_k_heads", "false")).lower() == "true"
        return ttnn.experimental.nlp_create_qkv_heads(
            inputs[0],
            num_heads=num_q,
            num_kv_heads=num_kv,
            transpose_k_heads=transpose_k,
            memory_config=out_mem,
        )

    if op_code == "NLPConcatHeadsDeviceOperation":
        return ttnn.experimental.nlp_concat_heads(inputs[0], memory_config=out_mem)

    if op_code == "SliceDeviceOperation":
        starts = parse_shape_list(attrs["slice_start"])
        ends = parse_shape_list(attrs["slice_end"])
        steps = parse_shape_list(attrs.get("step", "")) or [1] * len(starts)
        return ttnn.slice(inputs[0], starts, ends, steps, memory_config=out_mem)

    if op_code == "RotaryEmbeddingDeviceOperation":
        return ttnn.experimental.rotary_embedding(inputs[0], inputs[1], inputs[2], memory_config=out_mem)

    if op_code == "SDPAOperation":
        scale = float(attrs.get("scale", 1.0 / math.sqrt(inputs[0].shape[-1])))
        is_causal = str(attrs.get("is_causal", "true")).lower() == "true"
        program_config = parse_sdpa_program_config(attrs.get("program_config", ""))
        return ttnn.transformer.scaled_dot_product_attention(
            inputs[0],
            inputs[1],
            inputs[2],
            is_causal=is_causal,
            scale=scale,
            program_config=program_config,
            memory_config=out_mem,
        )

    raise ValueError(f"No runner registered for op_code={op_code}")


def _to_torch_outputs(result):
    results = result if isinstance(result, (tuple, list)) else (result,)
    return [ttnn.to_torch(out).float() for out in results]


def _pcc_threshold(op: dict) -> float:
    """Looser PCC for low-precision / approximate kernels."""
    dtypes = {inp.get("datatype") for inp in op.get("inputs", [])}
    dtypes |= {out.get("datatype") for out in op.get("outputs", [])}
    if "BFLOAT8_B" in dtypes or "BFLOAT4_B" in dtypes:
        return 0.98
    if op["op_code"] in ("SDPAOperation", "RotaryEmbeddingDeviceOperation", "MatmulDecodeDeviceOperation"):
        return 0.99
    if op["op_code"] == "UnaryDeviceOperation":
        unary_name, params = parse_unary_op((op.get("attributes") or {}).get("op_chain", ""))
        if unary_name == "GELU" and params and int(params[0]) == 1:
            return 0.99
    return 0.999


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_reference(op: dict, torch_inputs: list[torch.Tensor]):
    """Torch / identity golden for the profiled op. Returns a list of torch tensors."""
    op_code = op["op_code"]
    attrs = op.get("attributes") or {}

    if op_code in IDENTITY_OP_CODES:
        return [torch_inputs[0].float()]

    if op_code == "BinaryNgDeviceOperation":
        binary_type = _parse_enum_tail(attrs.get("binary_op_type", ""))
        scalar = attrs.get("scalar", "std::nullopt")
        a = torch_inputs[0].float()
        if scalar not in (None, "std::nullopt", "") and len(torch_inputs) == 1:
            b = float(scalar)
        else:
            b = torch_inputs[1].float()
        if binary_type == "MUL":
            return [a * b]
        if binary_type == "ADD":
            return [a + b]
        if binary_type == "SUB":
            return [a - b]
        if binary_type == "DIV":
            return [a / b]
        if binary_type == "POWER":
            return [torch.pow(a, b)]
        raise ValueError(f"Unsupported binary_op_type for reference: {binary_type}")

    if op_code == "UnaryDeviceOperation":
        unary_name, params = parse_unary_op(attrs.get("op_chain", ""))
        a = torch_inputs[0].float()
        if unary_name == "FILL":
            return [torch.full_like(a, params[0] if params else 0.0)]
        if unary_name == "COS":
            return [torch.cos(a)]
        if unary_name == "SIN":
            return [torch.sin(a)]
        if unary_name == "RECIP":
            return [torch.reciprocal(a)]
        if unary_name == "GELU":
            return [torch.nn.functional.gelu(a)]
        if unary_name == "RELU":
            return [torch.relu(a)]
        if unary_name == "EXP":
            return [torch.exp(a)]
        if unary_name == "SQRT":
            return [torch.sqrt(a)]
        raise ValueError(f"Unsupported UnaryOpType for reference: {unary_name}")

    if op_code == "ReshapeViewDeviceOperation":
        logical = (
            list(op["outputs"][0]["shape_logical"])
            if op.get("outputs")
            else parse_shape_list(attrs.get("logical_output_shape", ""))
        )
        return [torch_inputs[0].float().reshape(logical)]

    if op_code == "ConcatDeviceOperation":
        dim = infer_concat_dim(op["inputs"], op["outputs"][0])
        return [torch.cat([t.float() for t in torch_inputs], dim=dim)]

    if op_code == "TypecastDeviceOperation":
        # Compare in float; typecast itself is a dtype change of the same values.
        return [torch_inputs[0].float()]

    if op_code == "LayerNormDeviceOperation":
        eps = float(attrs.get("eps", "1e-06"))
        x = torch_inputs[0].float()
        weight = torch_inputs[1].float() if len(torch_inputs) > 1 else None
        bias = torch_inputs[2].float() if len(torch_inputs) > 2 else None
        norm_type = _parse_enum_tail(attrs.get("norm_type", "LAYERNORM"))
        if norm_type == "RMSNORM":
            # Match ttnn rms_norm: normalize over the last dim.
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            out = x / rms
            if weight is not None:
                out = out * weight
            if bias is not None:
                out = out + bias
            return [out]
        out = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)
        return [out]

    if op_code in ("MatmulDeviceOperation", "MatmulDecodeDeviceOperation"):
        return [torch.matmul(torch_inputs[0].float(), torch_inputs[1].float())]

    if op_code == "NlpCreateHeadsDeviceOperation":
        num_q = int(attrs["num_q_heads"])
        num_kv = int(attrs.get("num_kv_heads", "1"))
        transpose_k = str(attrs.get("transpose_k_heads", "false")).lower() == "true"
        head_dim = int(attrs["head_dim"])
        x = torch_inputs[0].float()
        q, k, v = torch.split(x, [num_q * head_dim, num_kv * head_dim, num_kv * head_dim], dim=-1)
        batch, _, seq, _ = x.shape
        q = q.reshape(batch, seq, num_q, head_dim).transpose(1, 2)
        k = k.reshape(batch, seq, num_kv, head_dim).transpose(1, 2)
        v = v.reshape(batch, seq, num_kv, head_dim).transpose(1, 2)
        if transpose_k:
            k = k.transpose(-2, -1)
        return [q, k, v]

    if op_code == "NLPConcatHeadsDeviceOperation":
        x = torch_inputs[0].float()
        batch, num_heads, seq, head_dim = x.shape
        return [x.transpose(1, 2).reshape(batch, 1, seq, num_heads * head_dim)]

    if op_code == "SliceDeviceOperation":
        starts = parse_shape_list(attrs["slice_start"])
        ends = parse_shape_list(attrs["slice_end"])
        steps = parse_shape_list(attrs.get("step", "")) or [1] * len(starts)
        slices = tuple(slice(s, e, st) for s, e, st in zip(starts, ends, steps))
        return [torch_inputs[0].float()[slices]]

    if op_code == "RotaryEmbeddingDeviceOperation":
        x = torch_inputs[0].float()
        cos = torch_inputs[1].float()
        sin = torch_inputs[2].float()
        return [(x * cos) + (_rotate_half(x) * sin)]

    if op_code == "SDPAOperation":
        q, k, v = (t.float() for t in torch_inputs)
        scale = float(attrs.get("scale", 1.0 / math.sqrt(q.shape[-1])))
        is_causal = str(attrs.get("is_causal", "true")).lower() == "true"
        # Repeat KV heads to match Q heads when using GQA / MQA.
        nh, nkv = q.shape[1], k.shape[1]
        if nh != nkv:
            reps = nh // nkv
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        return [torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)]

    raise ValueError(f"No reference registered for op_code={op_code}")


def compare_with_reference(op: dict, torch_inputs: list[torch.Tensor], tt_inputs: list, result):
    """Compare device outputs against the torch / identity reference."""
    actuals = _to_torch_outputs(result)

    if op["op_code"] in IDENTITY_OP_CODES:
        # Compare against the device input round-trip so dtype quantization
        # (e.g. bfloat8_b) is already accounted for — these ops must not alter data.
        expected = ttnn.to_torch(tt_inputs[0]).float()
        actual = actuals[0]
        assert list(actual.shape) == list(
            expected.shape
        ), f"{op['op_code']}: shape {list(actual.shape)} != input {list(expected.shape)}"
        if any(inp.get("datatype") in ("BFLOAT8_B", "BFLOAT4_B") for inp in op["inputs"]):
            assert_with_pcc(expected, actual, pcc=0.999)
        else:
            assert_equal(expected, actual)
        return

    expecteds = compute_reference(op, torch_inputs)
    assert len(actuals) == len(
        expecteds
    ), f"{op['op_code']}: expected {len(expecteds)} reference outputs, got {len(actuals)}"

    pcc = _pcc_threshold(op)
    for i, (actual, expected) in enumerate(zip(actuals, expecteds)):
        assert list(actual.shape) == list(
            expected.shape
        ), f"{op['op_code']} out{i}: shape {list(actual.shape)} != ref {list(expected.shape)}"
        passed, message = assert_with_pcc(expected, actual, pcc=pcc)
        print(message)


def _op_id(idx: int, op: dict) -> str:
    shapes = [tuple(i.get("shape_logical") or i["shape_padded"]) for i in op["inputs"]]
    return f"{idx:02d}-{op['op_code']}-{shapes}"


@pytest.mark.parametrize("op_index", list(range(len(OPS))), ids=[_op_id(i, OPS[i]) for i in range(len(OPS))])
def test_profiler_op_replay(device, op_index):
    """Create random TILE inputs from the JSON, run the matching ttnn op, and check vs reference."""
    torch.manual_seed(0)
    op = OPS[op_index]

    # All inputs must be TILE (JSON was pre-filtered, re-check here)
    assert all(inp.get("layout") == "TILE" for inp in op["inputs"])

    torch_inputs, tt_inputs = create_inputs(op, device)
    result = run_op(op, tt_inputs, device)

    # Basic sanity: result exists and shapes roughly match profiled outputs
    assert result is not None
    results = result if isinstance(result, (tuple, list)) else (result,)
    expected_outputs = op.get("outputs") or []
    if expected_outputs:
        assert len(results) == len(
            expected_outputs
        ), f"{op['op_code']}: expected {len(expected_outputs)} outputs, got {len(results)}"
        for out_tt, out_spec in zip(results, expected_outputs):
            got_logical = list(out_tt.shape)
            exp_logical = list(out_spec["shape_logical"])
            assert got_logical == exp_logical, f"{op['op_code']}: logical shape {got_logical} != expected {exp_logical}"
            # Profiled shape_padded assumes the capture tile (often 32x32). Recompute
            # from logical shape using this test's tile so 16x32 vs 32x32 padding matches.
            got_padded = list(out_tt.padded_shape)
            exp_padded = tile_padded_shape(exp_logical)
            assert got_padded == exp_padded, f"{op['op_code']}: padded_shape {got_padded} != expected {exp_padded}"

    compare_with_reference(op, torch_inputs, tt_inputs, result)
