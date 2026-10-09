"""Static shape tracer for the SDXL UNet ttnn model.

Monkeypatches the ttnn ops the model calls with shape/memory-config propagating fakes,
runs TtUNet2DConditionModel.forward and TtEulerDiscreteScheduler with random weights,
and dumps every op call (inputs/outputs specs + configs) to JSON.

Usage: python trace_sdxl.py <out.json> [--res 1024]
"""
import functools
import inspect
import json
import math
import os
import sys

import torch

import ttnn

RECORDS = []
CTX = []
MODEL_DIR = os.path.join("models", "demos", "stable_diffusion_xl_base", "tt")

# ----------------------------------------------------------------------------- serialization


def enum_name(v):
    s = str(v)
    return s.split(".")[-1].split("::")[-1]


def mem_to_dict(mc):
    if mc is None:
        return None
    d = {"layout": enum_name(mc.memory_layout), "buffer": enum_name(mc.buffer_type), "shard": None}
    ss = mc.shard_spec
    if ss is not None:
        d["shard"] = {
            "grid": [[r.start.x, r.start.y, r.end.x, r.end.y] for r in ss.grid.ranges()],
            "shape": list(ss.shape),
            "orientation": enum_name(ss.orientation),
        }
    return d


CONFIG_TYPES = (
    "Conv2dConfig",
    "MatmulMultiCoreReuseMultiCastProgramConfig",
    "MatmulMultiCoreReuseMultiCast1DProgramConfig",
    "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
    "MatmulMultiCoreReuseProgramConfig",
    "SDPAProgramConfig",
    "LayerNormShardedMultiCoreProgramConfig",
    "LayerNormDefaultProgramConfig",
    "WormholeComputeKernelConfig",
    "BlackholeComputeKernelConfig",
    "GrayskullComputeKernelConfig",
    "Conv2dSliceConfig",
)


def ser(v, depth=0):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, FT):
        return v.spec()
    if isinstance(v, (list, tuple)):
        return [ser(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): ser(x, depth + 1) for k, x in v.items()}
    if isinstance(v, ttnn.Shape):
        return list(v)
    if isinstance(v, ttnn.MemoryConfig):
        return mem_to_dict(v)
    if isinstance(v, torch.Tensor):
        return {"torch_shape": list(v.shape), "torch_dtype": str(v.dtype)}
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, ttnn.CoreGrid):
        return {"CoreGrid": [v.x, v.y]}
    if isinstance(v, ttnn.CoreCoord):
        return {"CoreCoord": [v.x, v.y]}
    if isinstance(v, ttnn.CoreRangeSet):
        return {"CoreRangeSet": [[r.start.x, r.start.y, r.end.x, r.end.y] for r in v.ranges()]}
    if isinstance(v, ttnn.CoreRange):
        return {"CoreRange": [v.start.x, v.start.y, v.end.x, v.end.y]}
    if isinstance(v, ttnn.UnaryWithParam):
        return {"UnaryWithParam": enum_name(v.op_type), "params": list(v.params)}
    tn = type(v).__name__
    if tn in CONFIG_TYPES:
        d = {"type": tn}
        for a in dir(v):
            if a.startswith("_") or a in ("from_json", "to_json"):
                continue
            try:
                x = getattr(v, a)
            except Exception:
                continue
            if callable(x):
                continue
            d[a] = ser(x, depth + 1)
        return d
    if tn in (
        "DataType",
        "Layout",
        "TensorMemoryLayout",
        "BufferType",
        "MathFidelity",
        "UnaryOpType",
        "ShardOrientation",
        "ShardStrategy",
    ):
        return enum_name(v)
    if hasattr(v, "x") and hasattr(v, "y"):
        return {"xy": [v.x, v.y]}
    return repr(v)


# ----------------------------------------------------------------------------- fake tensor


class FT:
    def __init__(self, shape, dtype, layout, mem, device=None, torch_ref=None):
        self.shape = ttnn.Shape([int(x) for x in shape])
        self.dtype = dtype
        self.layout = layout
        self._mem = mem
        self._device = device
        self.torch_ref = torch_ref

    # --- ttnn.Tensor API used by the model
    def memory_config(self):
        return self._mem

    def is_sharded(self):
        return (
            self._mem is not None
            and self._mem.shard_spec is not None
            or (self._mem is not None and "SHARDED" in enum_name(self._mem.memory_layout))
        )

    @property
    def padded_shape(self):
        return self.shape

    def device(self):
        return self._device

    def cpu(self):
        return FT(self.shape, self.dtype, self.layout, None, None)

    def buffer_address(self):
        return 0

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shp = list(self.shape)
        out = []
        for i, s in enumerate(idx):
            if isinstance(s, slice):
                out.append(len(range(*s.indices(shp[i]))))
            else:
                out.append(1)
        out += shp[len(idx) :]
        res = FT(out, self.dtype, self.layout, self._mem, self._device)
        rec("slice", {"input": self}, {"index": repr(idx)}, [res])
        return res

    def __mul__(self, other):
        return patched_multiply(self, other)

    def __add__(self, other):
        return patched_add(self, other)

    def spec(self):
        return {
            "shape": list(self.shape),
            "dtype": enum_name(self.dtype),
            "layout": enum_name(self.layout),
            "mem": mem_to_dict(self._mem),
            "on_device": self._device is not None,
        }

    def __repr__(self):
        return f"FT{self.spec()}"


class FakeDevice:
    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 8)

    def arch(self):
        return "quasar"

    def __repr__(self):
        return "FakeDevice"


DEVICE = FakeDevice()

# ----------------------------------------------------------------------------- recording


def site():
    for fr in inspect.stack()[2:]:
        fn = fr.filename
        if MODEL_DIR in fn:
            return f"{os.path.basename(fn)}:{fr.function}:{fr.lineno}"
    return "?"


def rec(op, tensors, params, outputs):
    RECORDS.append(
        {
            "idx": len(RECORDS),
            "op": op,
            "module": CTX[-1] if CTX else "",
            "site": site(),
            "inputs": {k: (v.spec() if isinstance(v, FT) else ser(v)) for k, v in tensors.items() if v is not None},
            "params": {k: ser(v) for k, v in params.items()},
            "outputs": [o.spec() if isinstance(o, FT) else ser(o) for o in outputs],
        }
    )


# ----------------------------------------------------------------------------- shape helpers


def broadcast(a, b):
    a = list(a)
    b = list(b)
    n = max(len(a), len(b))
    a = [1] * (n - len(a)) + a
    b = [1] * (n - len(b)) + b
    return [max(x, y) for x, y in zip(a, b)]


def grid_dims(ranges):
    xs = max(r[2] for r in ranges) - min(r[0] for r in ranges) + 1
    ys = max(r[3] for r in ranges) - min(r[1] for r in ranges) + 1
    return xs, ys


def rup(v, m):
    return int(math.ceil(v / m) * m)


def sharded_mem(layout, nx, ny, shape, buffer=ttnn.BufferType.L1, x0=0, y0=0):
    """Build an L1 sharded MemoryConfig approximating what an op would produce."""
    h = int(math.prod(list(shape)[:-1]))
    w = int(shape[-1])
    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard = [rup(math.ceil(h / (nx * ny)), 32), w]
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard = [rup(h, 32), rup(math.ceil(w / (nx * ny)), 32)]
    else:
        shard = [rup(math.ceil(h / ny), 32), rup(math.ceil(w / nx), 32)]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + nx - 1, y0 + ny - 1))})
    return ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


def resolve_mem(requested, fallback_fn):
    """Return a concrete MemoryConfig: requested if it is complete, synthesized otherwise."""
    if requested is None:
        return ttnn.DRAM_MEMORY_CONFIG
    if "SHARDED" in enum_name(requested.memory_layout) and requested.shard_spec is None:
        return fallback_fn(requested)
    return requested


# ----------------------------------------------------------------------------- patched ops


def patched_from_torch(
    t, dtype=None, *, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, memory_config=None, mesh_mapper=None, **kw
):
    if dtype is None:
        dtype = ttnn.bfloat16 if t.dtype == torch.bfloat16 else ttnn.float32
    mem = (memory_config or ttnn.DRAM_MEMORY_CONFIG) if device is not None else None
    return FT(t.shape, dtype, layout, mem, device, torch_ref=t)


def patched_to_torch(t, *a, **k):
    return torch.zeros(list(t.shape))


def noop(*a, **k):
    return None


def patched_to_device(t, device, memory_config=None, **k):
    mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = FT(t.shape, t.dtype, t.layout, mem, device, t.torch_ref)
    rec("to_device", {"input": t}, {"memory_config": memory_config}, [out])
    return out


def patched_to_layout(t, layout, dtype=None, memory_config=None, device=None, **k):
    out = FT(t.shape, dtype or t.dtype, layout, memory_config or t._mem, t._device or device)
    rec("to_layout", {"input": t}, {"layout": layout, "dtype": dtype, "memory_config": memory_config}, [out])
    return out


def patched_to_memory_config(t, memory_config, dtype=None, **k):
    out = FT(t.shape, dtype or t.dtype, t.layout, memory_config, t._device)
    rec("to_memory_config", {"input": t}, {"memory_config": memory_config, "dtype": dtype}, [out])
    return out


def patched_sharded_to_interleaved(t, memory_config=None, output_dtype=None, **k):
    mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = FT(t.shape, output_dtype or t.dtype, t.layout, mem, t._device)
    rec("sharded_to_interleaved", {"input": t}, {"memory_config": memory_config, "output_dtype": output_dtype}, [out])
    return out


def patched_move(t, memory_config=None, **k):
    out = FT(t.shape, t.dtype, t.layout, memory_config or t._mem, t._device)
    rec("move", {"input": t}, {"memory_config": memory_config}, [out])
    return out


def patched_permute(t, dims, memory_config=None, **k):
    shp = list(t.shape)
    out = FT([shp[d] for d in dims], t.dtype, t.layout, memory_config or t._mem, t._device)
    rec("permute", {"input": t}, {"dims": list(dims), "memory_config": memory_config}, [out])
    return out


def patched_reshape(t, shape, memory_config=None, **k):
    shape = list(shape)
    total = math.prod(list(t.shape))
    if -1 in shape:
        known = math.prod([s for s in shape if s != -1])
        shape[shape.index(-1)] = total // known
    out = FT(shape, t.dtype, t.layout, memory_config or t._mem, t._device)
    rec("reshape", {"input": t}, {"shape": shape, "memory_config": memory_config}, [out])
    return out


def patched_unsqueeze(t, dim, **k):
    shp = list(t.shape)
    if dim < 0:
        dim = len(shp) + dim + 1
    shp.insert(dim, 1)
    out = FT(shp, t.dtype, t.layout, t._mem, t._device)
    rec("unsqueeze", {"input": t}, {"dim": dim}, [out])
    return out


def patched_squeeze(t, dim, **k):
    shp = list(t.shape)
    if dim < 0:
        dim = len(shp) + dim
    if shp[dim] == 1:
        shp.pop(dim)
    out = FT(shp, t.dtype, t.layout, t._mem, t._device)
    rec("squeeze", {"input": t}, {"dim": dim}, [out])
    return out


def patched_concat(tensors, dim, memory_config=None, **k):
    shp = list(tensors[0].shape)
    if dim < 0:
        dim += len(shp)
    shp[dim] = sum(int(t.shape[dim]) for t in tensors)
    out = FT(shp, tensors[0].dtype, tensors[0].layout, memory_config or tensors[0]._mem, tensors[0]._device)
    rec("concat", {f"input_{i}": t for i, t in enumerate(tensors)}, {"dim": dim, "memory_config": memory_config}, [out])
    return out


def make_binary(name, inplace=False):
    def f(a, b, *args, memory_config=None, output_tensor=None, dtype=None, **kw):
        bshape = list(b.shape) if isinstance(b, FT) else []
        out_shape = broadcast(a.shape, bshape)
        mem = memory_config or a._mem
        out = a if inplace else FT(out_shape, dtype or a.dtype, a.layout, mem, a._device)
        tens = {"input_a": a}
        params = {"memory_config": memory_config, **{k: v for k, v in kw.items()}}
        if isinstance(b, FT):
            tens["input_b"] = b
        else:
            params["scalar_b"] = b
        rec(name, tens, params, [out])
        return out

    return f


patched_add = make_binary("add")
patched_multiply = make_binary("multiply")


def make_unary(name):
    def f(a, *args, memory_config=None, output_tensor=None, **kw):
        out = (
            a
            if output_tensor is a and output_tensor is not None
            else FT(a.shape, a.dtype, a.layout, memory_config or a._mem, a._device)
        )
        rec(name, {"input": a}, {"memory_config": memory_config, "inplace": output_tensor is not None, **kw}, [out])
        return out

    return f


def matmul_out_mem(a, w, memory_config, program_config, out_shape):
    def synth(req):
        pc = program_config
        tn = type(pc).__name__ if pc is not None else ""
        if "MultiCast1D" in tn:
            g = pc.compute_with_storage_grid_size
            if pc.mcast_in0:
                return sharded_mem(ttnn.TensorMemoryLayout.WIDTH_SHARDED, g.x, g.y, out_shape, req.buffer_type)
            return sharded_mem(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, g.x, g.y, out_shape, req.buffer_type)
        if "MultiCastProgramConfig" in tn:
            g = pc.compute_with_storage_grid_size
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
            shard = [pc.per_core_M * 32, pc.per_core_N * 32]
            return ttnn.MemoryConfig(
                req.memory_layout, req.buffer_type, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
            )
        return sharded_mem(req.memory_layout, 8, 8, out_shape, req.buffer_type)

    return resolve_mem(memory_config, synth)


def make_matmul(name):
    def f(
        a,
        w,
        *args,
        bias=None,
        program_config=None,
        memory_config=None,
        compute_kernel_config=None,
        activation=None,
        dtype=None,
        **kw,
    ):
        out_shape = list(a.shape)[:-1] + [int(w.shape[-1])]
        mem = matmul_out_mem(a, w, memory_config, program_config, out_shape)
        out = FT(out_shape, dtype or a.dtype, ttnn.TILE_LAYOUT, mem, a._device)
        rec(
            name,
            {"input": a, "weight": w, "bias": bias},
            {
                "program_config": program_config,
                "memory_config": memory_config,
                "compute_kernel_config": compute_kernel_config,
                "activation": activation,
                "dtype": dtype,
                **kw,
            },
            [out],
        )
        return out

    return f


def patched_conv2d(
    *,
    input_tensor,
    weight_tensor,
    in_channels,
    out_channels,
    device,
    bias_tensor=None,
    kernel_size,
    stride,
    padding,
    dilation,
    batch_size,
    input_height,
    input_width,
    conv_config=None,
    compute_config=None,
    slice_config=None,
    groups=1,
    memory_config=None,
    return_output_dim=False,
    return_weights_and_bias=False,
    dtype=None,
    **kw,
):
    kh, kw_ = kernel_size
    ph, pw = padding if len(padding) == 2 else (padding[0], padding[2])
    oh = (input_height + 2 * ph - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (input_width + 2 * pw - dilation[1] * (kw_ - 1) - 1) // stride[1] + 1
    out_shape = [1, 1, batch_size * oh * ow, out_channels]
    layout = conv_config.shard_layout if conv_config is not None else None
    if layout is None:
        mem = ttnn.DRAM_MEMORY_CONFIG
    else:
        nx, ny = 8, 8
        x0 = y0 = 0
        cg = conv_config.core_grid if conv_config.override_output_sharding_config else None
        if cg is not None:
            ranges = [[r.start.x, r.start.y, r.end.x, r.end.y] for r in cg.ranges()]
            nx, ny = grid_dims(ranges)
        mem = sharded_mem(layout, nx, ny, out_shape)
    out = FT(out_shape, dtype or input_tensor.dtype, ttnn.TILE_LAYOUT, mem, input_tensor._device)
    wt = FT(weight_tensor.shape, weight_tensor.dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, device)
    bt = (
        FT(bias_tensor.shape, bias_tensor.dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, device)
        if bias_tensor is not None
        else None
    )
    rec(
        "conv2d",
        {"input": input_tensor, "weight": weight_tensor, "bias": bias_tensor},
        {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": list(kernel_size),
            "stride": list(stride),
            "padding": list(padding),
            "dilation": list(dilation),
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "groups": groups,
            "output_height": oh,
            "output_width": ow,
            "conv_config": conv_config,
            "compute_config": compute_config,
            "slice_config": repr(slice_config),
            "memory_config": memory_config,
            "dtype": dtype,
            "note": "output shard spec is synthesized (conv picks the real one)",
        },
        [out],
    )
    res = [out]
    if return_output_dim:
        res.append([oh, ow])
    if return_weights_and_bias:
        res.append([wt, bt])
    return res if len(res) > 1 else out


def patched_group_norm(
    x,
    *,
    num_groups,
    input_mask=None,
    negative_mask=None,
    weight=None,
    bias=None,
    epsilon=1e-5,
    memory_config=None,
    core_grid=None,
    num_out_blocks=None,
    inplace=False,
    dtype=None,
    **kw,
):
    out = x if inplace else FT(x.shape, dtype or x.dtype, x.layout, memory_config or x._mem, x._device)
    rec(
        "group_norm",
        {"input": x, "input_mask": input_mask, "negative_mask": negative_mask, "weight": weight, "bias": bias},
        {
            "num_groups": num_groups,
            "epsilon": epsilon,
            "memory_config": memory_config,
            "core_grid": core_grid,
            "num_out_blocks": num_out_blocks,
            "inplace": inplace,
            "dtype": dtype,
            **kw,
        },
        [out],
    )
    return out


def patched_layer_norm(
    x,
    *,
    weight=None,
    bias=None,
    epsilon=1e-5,
    memory_config=None,
    program_config=None,
    compute_kernel_config=None,
    residual_input_tensor=None,
    **kw,
):
    out = FT(x.shape, x.dtype, x.layout, memory_config or x._mem, x._device)
    rec(
        "layer_norm",
        {"input": x, "weight": weight, "bias": bias},
        {
            "epsilon": epsilon,
            "memory_config": memory_config,
            "program_config": program_config,
            "compute_kernel_config": compute_kernel_config,
            **kw,
        },
        [out],
    )
    return out


def patched_nlp_create_qkv_heads(x, *, num_heads, num_kv_heads=None, transpose_k_heads=True, memory_config=None, **kw):
    shp = list(x.shape)
    b, s, d = (shp[0], shp[-2], shp[-1])
    mem = memory_config or x._mem
    if num_kv_heads == 0:
        hd = d // num_heads
        q = FT([b, num_heads, s, hd], x.dtype, x.layout, mem, x._device)
        outs = [q, None, None]
    else:
        nkv = num_kv_heads if num_kv_heads else num_heads
        hd = d // (num_heads + 2 * nkv)
        q = FT([b, num_heads, s, hd], x.dtype, x.layout, mem, x._device)
        kk = FT([b, nkv, hd, s] if transpose_k_heads else [b, nkv, s, hd], x.dtype, x.layout, mem, x._device)
        v = FT([b, nkv, s, hd], x.dtype, x.layout, mem, x._device)
        outs = [q, kk, v]
    rec(
        "nlp_create_qkv_heads",
        {"input": x},
        {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "transpose_k_heads": transpose_k_heads,
            "memory_config": memory_config,
            **kw,
        },
        outs,
    )
    return tuple(outs)


def patched_nlp_concat_heads(x, *, memory_config=None, **kw):
    b, h, s, d = list(x.shape)
    out = FT([b, 1, s, h * d], x.dtype, x.layout, memory_config or x._mem, x._device)
    rec("nlp_concat_heads", {"input": x}, {"memory_config": memory_config, **kw}, [out])
    return out


def patched_sdpa(
    q,
    k,
    v,
    *,
    is_causal=True,
    attn_mask=None,
    program_config=None,
    compute_kernel_config=None,
    memory_config=None,
    scale=None,
    **kw,
):
    out = FT(q.shape, q.dtype, q.layout, memory_config or q._mem, q._device)
    rec(
        "scaled_dot_product_attention",
        {"query": q, "key": k, "value": v, "attn_mask": attn_mask},
        {
            "is_causal": is_causal,
            "program_config": program_config,
            "compute_kernel_config": compute_kernel_config,
            "memory_config": memory_config,
            "scale": scale,
            **kw,
        },
        [out],
    )
    return out


def patched_upsample(x, scale_factor, *, mode="nearest", memory_config=None, **kw):
    sh, sw = (scale_factor, scale_factor) if isinstance(scale_factor, int) else scale_factor
    b, h, w, c = list(x.shape)
    out_shape = [b, h * sh, w * sw, c]
    mem = memory_config
    if mem is None and x._mem is not None and x._mem.shard_spec is not None:
        ranges = [[r.start.x, r.start.y, r.end.x, r.end.y] for r in x._mem.shard_spec.grid.ranges()]
        nx, ny = grid_dims(ranges)
        mem = sharded_mem(x._mem.memory_layout, nx, ny, out_shape, x._mem.buffer_type)
    out = FT(out_shape, x.dtype, x.layout, mem or x._mem, x._device)
    rec("upsample", {"input": x}, {"scale_factor": [sh, sw], "mode": mode, "memory_config": memory_config, **kw}, [out])
    return out


def patched_allocate_tensor_on_device(shape, dtype, layout, device, memory_config=None, **kw):
    out = FT(list(shape), dtype, layout, memory_config or ttnn.DRAM_MEMORY_CONFIG, device)
    rec(
        "allocate_tensor_on_device",
        {},
        {"shape": list(shape), "dtype": dtype, "layout": layout, "memory_config": memory_config},
        [out],
    )
    return out


def patched_copy_host_to_device_tensor(host, dev, **kw):
    rec("copy_host_to_device_tensor", {"host": host, "device_tensor": dev}, {}, [dev])
    return dev


def install():
    P = {
        "from_torch": patched_from_torch,
        "to_torch": patched_to_torch,
        "deallocate": noop,
        "ReadDeviceProfiler": noop,
        "synchronize_device": noop,
        "to_device": patched_to_device,
        "to_layout": patched_to_layout,
        "to_memory_config": patched_to_memory_config,
        "sharded_to_interleaved": patched_sharded_to_interleaved,
        "move": patched_move,
        "permute": patched_permute,
        "reshape": patched_reshape,
        "unsqueeze": patched_unsqueeze,
        "squeeze": patched_squeeze,
        "concat": patched_concat,
        "add": patched_add,
        "add_": make_binary("add_", inplace=True),
        "multiply": patched_multiply,
        "mul_": make_binary("mul_", inplace=True),
        "div": make_binary("div"),
        "silu": make_unary("silu"),
        "sin": make_unary("sin"),
        "cos": make_unary("cos"),
        "reciprocal": make_unary("reciprocal"),
        "linear": make_matmul("linear"),
        "matmul": make_matmul("matmul"),
        "conv2d": patched_conv2d,
        "group_norm": patched_group_norm,
        "layer_norm": patched_layer_norm,
        "upsample": patched_upsample,
        "allocate_tensor_on_device": patched_allocate_tensor_on_device,
        "copy_host_to_device_tensor": patched_copy_host_to_device_tensor,
    }
    for k, v in P.items():
        setattr(ttnn, k, v)
    ttnn.experimental.nlp_create_qkv_heads = patched_nlp_create_qkv_heads
    ttnn.experimental.nlp_concat_heads = patched_nlp_concat_heads
    ttnn.transformer.scaled_dot_product_attention = patched_sdpa


# ----------------------------------------------------------------------------- module context


def wrap_module_classes(modules):
    from models.common.lightweightmodule import LightweightModule

    for m in modules:
        for name, cls in vars(m).items():
            if not (inspect.isclass(cls) and issubclass(cls, LightweightModule) and cls.__module__ == m.__name__):
                continue
            if getattr(cls, "_traced", False):
                continue
            cls._traced = True
            orig_init = cls.__init__

            @functools.wraps(orig_init)
            def init(self, *a, _orig=orig_init, _cls=cls, **k):
                path = a[1] if len(a) > 1 and isinstance(a[1], str) else k.get("module_path", _cls.__name__)
                self._trace_path = f"{path}<{_cls.__name__}>"
                CTX.append(self._trace_path + ".__init__")
                try:
                    return _orig(self, *a, **k)
                finally:
                    CTX.pop()

            cls.__init__ = init
            for meth in ("forward", "step", "scale_model_input", "set_timesteps", "interpolate"):
                if meth in vars(cls):
                    orig = getattr(cls, meth)

                    def wrapped(self, *a, _orig=orig, _meth=meth, **k):
                        CTX.append(
                            getattr(self, "_trace_path", type(self).__name__)
                            + ("" if _meth == "forward" else f".{_meth}")
                        )
                        try:
                            return _orig(self, *a, **k)
                        finally:
                            CTX.pop()

                    setattr(cls, meth, wrapped)


# ----------------------------------------------------------------------------- main


def main():
    out_path = sys.argv[1]
    res = 1024
    install()

    import models.demos.stable_diffusion_xl_base.tt.tt_attention as m1
    import models.demos.stable_diffusion_xl_base.tt.tt_crossattndownblock2d as m2
    import models.demos.stable_diffusion_xl_base.tt.tt_crossattnmidblock2d as m3
    import models.demos.stable_diffusion_xl_base.tt.tt_crossattnupblock2d as m4
    import models.demos.stable_diffusion_xl_base.tt.tt_downblock2d as m5
    import models.demos.stable_diffusion_xl_base.tt.tt_downsample2d as m6
    import models.demos.stable_diffusion_xl_base.tt.tt_embedding as m7
    import models.demos.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler as m8
    import models.demos.stable_diffusion_xl_base.tt.tt_feedforward as m9
    import models.demos.stable_diffusion_xl_base.tt.tt_geglu as m10
    import models.demos.stable_diffusion_xl_base.tt.tt_resnetblock2d as m11
    import models.demos.stable_diffusion_xl_base.tt.tt_timesteps as m12
    import models.demos.stable_diffusion_xl_base.tt.tt_transformerblock as m13
    import models.demos.stable_diffusion_xl_base.tt.tt_transformermodel as m14
    import models.demos.stable_diffusion_xl_base.tt.tt_unet as m15
    import models.demos.stable_diffusion_xl_base.tt.tt_upblock2d as m16
    import models.demos.stable_diffusion_xl_base.tt.tt_upsample2d as m17
    from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations

    wrap_module_classes([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17])

    from diffusers import UNet2DConditionModel

    cfg = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_config(cfg)
    unet.eval()
    state_dict = unet.state_dict()

    from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024 import (
        ModelOptimisations1024x1024,
    )

    model_config = ModelOptimisations1024x1024()
    print("model_config:", type(model_config).__name__)
    CTX.append("__init__")
    tt_unet = m15.TtUNet2DConditionModel(DEVICE, state_dict, "unet", model_config=model_config, debug_mode=False)
    CTX.pop()
    n_init = len(RECORDS)
    print("records during init:", n_init)

    # inputs exactly as tests/pcc/test_module_tt_unet.py::prepare_ttnn_tensors
    lat = res // 8
    B, C, H, W = 1, 4, lat, lat
    CTX.append("test_inputs")
    timestep = ttnn.from_torch(
        torch.zeros((1,)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    encoder = ttnn.from_torch(
        torch.zeros((1, 77, 2048)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    text_embeds = ttnn.from_torch(
        torch.zeros((1, 1280)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    time_ids = ttnn.from_torch(
        torch.zeros((6,)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sample = ttnn.from_torch(
        torch.zeros((B, C, H, W)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sample = ttnn.permute(sample, (0, 2, 3, 1))
    sample = ttnn.reshape(sample, (B, 1, H * W, C))
    CTX.pop()

    CTX.append("unet")
    out, out_shape = tt_unet.forward(
        sample,
        [B, C, H, W],
        timestep=timestep,
        encoder_hidden_states=encoder,
        time_ids=time_ids,
        text_embeds=text_embeds,
    )
    CTX.pop()
    print("unet output:", out, out_shape, "records:", len(RECORDS))

    # scheduler, as in tests/pcc/test_euler_discrete_scheduler.py
    CTX.append("scheduler")
    sch = m8.TtEulerDiscreteScheduler(
        DEVICE,
        1000,
        0.00085,
        0.012,
        "scaled_linear",
        None,
        "epsilon",
        "linear",
        False,
        False,
        False,
        None,
        None,
        "leading",
        "discrete",
        1,
        False,
        "zero",
    )
    sch.set_timesteps(num_inference_steps=20)
    latents = ttnn.from_torch(
        torch.zeros((1, 1, lat * lat, 4)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaled = sch.scale_model_input(latents, None)
    noise_pred = ttnn.from_torch(
        torch.zeros((1, 1, lat * lat, 4)),
        dtype=ttnn.bfloat16,
        device=DEVICE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sch.step(noise_pred, None, latents)
    CTX.pop()

    with open(out_path, "w") as f:
        json.dump({"resolution": res, "model_config": type(model_config).__name__, "records": RECORDS}, f, indent=1)
    print("wrote", out_path, len(RECORDS), "records")


# ----------------------------------------------------------------------------- group norm host helpers (device-querying in ttnn)


def _max_tile_span(W, group_size, tile_width=32):
    pos = 0
    span = 0
    while pos < W:
        end = pos + group_size
        span = max(span, (end - 1) // tile_width - pos // tile_width + 1)
        pos = end
    return span


def fake_gn_mask(C, G, num_cores, dtype=ttnn.bfloat8_b, **kw):
    block_wt = _max_tile_span(C, C // G)
    return FT([1, G, 32, block_wt * 32], dtype, ttnn.TILE_LAYOUT, None, None)


def fake_dram_gn_params(
    torch_params, num_channels, num_groups, device, core_grid=None, return_mask=True, dtype=ttnn.bfloat16, **kw
):
    ncols = core_grid.x if core_grid is not None else 1
    outs = []
    for t in torch_params:
        rm = ttnn.create_group_norm_weight_bias_rm(t, num_channels, ncols)
        outs.append(
            ttnn.from_torch(
                rm, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
    if return_mask:
        mask = fake_gn_mask(num_channels, num_groups, ncols, dtype)
        mask = ttnn.to_device(mask, device)
        return outs, mask
    return outs


FakeDevice.shape = (1, 1)

_install0 = install


def install():
    _install0()
    ttnn.create_group_norm_input_mask = fake_gn_mask
    ttnn.create_group_norm_input_negative_mask = fake_gn_mask
    ttnn.dram_group_norm_params_from_torch = fake_dram_gn_params


if __name__ == "__main__":
    main()
