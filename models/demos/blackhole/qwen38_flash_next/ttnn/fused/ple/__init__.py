# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``ple``: the parallel-lookup-embedding layer's device body (``Qwen38TTNNPLE.forward_prepared``, layer 1, once per
step: 53 programs + 6 collectives) as fused programs.  Built in stages; this module holds the stage builders and the
model hook grows with them.

Stage 3, ``gate``: everything of ``_gate`` after the key/query norm all_gathers -- typecasts, the fp32 product, the
fp32 sum (the accurate SFPU fold), the scalar chain (2560^-0.5 scale, abs, clamp 1e-6, sqrt, sign, product, sigmoid),
the value row repeated over the branch rows and gated (bf16) -- as one program on one core, each op the chain's LLK
in the chain's order (docs in kernels/gate_compute.cpp).  Chain: 12 programs -> 1."""

from __future__ import annotations

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "ple"
TP_SIZE, TP_AXIS = 4, 1
HIDDEN = 2560
LOCAL_HIDDEN = HIDDEN // TP_SIZE  # 640
BRANCHES = 4
GLOBAL_TILES = HIDDEN // fp.TILE  # 80
LOCAL_TILES = LOCAL_HIDDEN // fp.TILE  # 20
GATE_SCALE = HIDDEN**-0.5
BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE_BF16, TILE_FP32 = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
NORM_COMPUTE = fp.kernel_source(NAME, "norm_gamma_rows_compute.cpp")
STATS_TILES = TP_SIZE  # gathered stats tiles per row-block
EPS = 1e-6
CONV_COMPUTE = fp.kernel_source(NAME, "conv_compute.cpp")
CONV_READER = fp.kernel_source(NAME, "conv_reader.cpp")
CONV_WRITER = fp.kernel_source(NAME, "conv_writer.cpp")
CONV_STATE_LENGTH = 9
CONV_TAPS = 4
MAC_FORM = 0  # 0: mac(row, tap, acc) = row * tap + acc (the golden); 1: tap * acc + row (the factory's literal binding)
GATE_COMPUTE = fp.kernel_source(NAME, "gate_compute.cpp")
GATE_READER = fp.kernel_source(NAME, "gate_reader.cpp")
GATE_WRITER = fp.kernel_source(NAME, "gate_writer.cpp")
_CONSTANTS: dict[int, tuple] = {}


def constants(mesh):
    """Per mesh, uploaded once: the fp32 tile filled with 2560^-0.5 (binary_ng's scalar operand as its reader fills it)
    and the bf16 zero tile (the repeat's padding rows)."""

    key = id(mesh)
    if key not in _CONSTANTS:
        scale = torch.full((1, 1, fp.TILE, fp.TILE), GATE_SCALE, dtype=torch.float32)
        zero = torch.zeros(1, 1, fp.TILE, fp.TILE)
        _CONSTANTS[key] = (
            ttnn.from_torch(
                scale, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            ttnn.from_torch(
                zero, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        )
    return _CONSTANTS[key]


def _expect(tensor, shape, dtype, label):
    got = tuple(int(v) for v in tensor.shape)
    if got != shape or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{label} must be {dtype} TILE {shape}, got {tensor.dtype} {tensor.layout} {got}")


def _one_core(mesh):
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def stats(x):
    """``rms_norm_pre_all_gather`` of one ``[1,1,4,640]`` bf16 row-block -> its stats tile ``[1,1,4,32]`` bf16 (F2's
    STATS kernel on one core; the chain's reshape to (1,1,4,32) is this shape)."""

    from .. import gr_read as gr

    _expect(x, (1, 1, BRANCHES, LOCAL_HIDDEN), BF16, "PLE group-norm input")
    mesh = x.device()
    out = fp.allocate((1, 1, BRANCHES, fp.TILE), BF16, ttnn.TILE_LAYOUT, mesh)
    core, one = _one_core(mesh)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, LOCAL_TILES, one),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(2, FP32, TILE_FP32, LOCAL_TILES, one),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, one),
    ]
    reader = gr._reader(
        one, [(x, 0)], [(gr.CONST_SCALER, 1)], [(core, ([gr._stream(x, 1, LOCAL_TILES, 0, 1, 0, 4)], [gr._bits(1.0)]))]
    )
    compute = fp.compute_kernel(gr.STATS, one, [LOCAL_TILES], fp32_dest=True)
    writer = gr._writer(one, [(out, 16)], [(core, [(1, 0, 1, 1)])])
    meta = fp.program_meta(  # the row-block in, the stats tile out; square and sum per element
        NAME, "stats", BRANCHES, reads=(x,), writes=(out,), flops=2 * BRANCHES * LOCAL_HIDDEN, cores=1
    )
    return fp.run_program([x, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def normalize(x, gathered_stats, weight):
    """``rms_norm_post_all_gather(x, gathered, eps, weight=)`` of one row-block: F2's unit path, then the chain's
    FUSE_GAMMA FPU multiply (the weight's row 0 over every row).  ``weight`` fp32 TILE ``[1,1,4,640]``."""

    from .. import gr_read as gr

    _expect(x, (1, 1, BRANCHES, LOCAL_HIDDEN), BF16, "PLE group-norm input")
    _expect(gathered_stats, (1, 1, BRANCHES, fp.TILE * STATS_TILES), BF16, "PLE gathered stats")
    _expect(weight, (1, 1, BRANCHES, LOCAL_HIDDEN), FP32, "PLE norm weight")
    mesh = x.device()
    out = fp.allocate((1, 1, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = gr.avg_scaler("chain")
    core, one = _one_core(mesh)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, LOCAL_TILES, one),
        fp.cb_descriptor(1, BF16, TILE_BF16, STATS_TILES, one),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, one),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, one),
        fp.cb_descriptor(4, FP32, TILE_FP32, LOCAL_TILES, one),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(7, FP32, TILE_FP32, LOCAL_TILES, one),  # x_normed: fp32 under the op's fp32 dest (FUSE_GAMMA)
        fp.cb_descriptor(16, BF16, TILE_BF16, LOCAL_TILES, one),
    ]
    reader = gr._reader(
        one,
        [(gathered_stats, 1), (x, 0), (weight, 4)],
        [(gr.CONST_SCALER, 2), (gr.CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        gr._stream(gathered_stats, 1, STATS_TILES, 0, 1, 0, STATS_TILES),
                        gr._stream(x, 1, LOCAL_TILES, 0, 1, 0, 4),
                        gr._stream(weight, 1, LOCAL_TILES, 0, 1, 0, 4),
                    ],
                    [scaler_bits, gr._bits(EPS)],
                ),
            )
        ],
    )
    compute = fp.compute_kernel(NORM_COMPUTE, one, [LOCAL_TILES, STATS_TILES, 4, 16], fp32_dest=True)
    writer = gr._writer(one, [(out, 16)], [(core, [(LOCAL_TILES, 0, 1, 4)])])
    meta = fp.program_meta(  # the row-block, the gathered stats and the weight in, the normalized block out
        NAME,
        "normalize",
        BRANCHES,
        reads=(x, gathered_stats, weight),
        writes=(out,),
        flops=4 * BRANCHES * LOCAL_HIDDEN,
        cores=1,
    )
    return fp.run_program(
        [x, gathered_stats, weight, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def group_norm_composed(x, gathered_stats, weight, compute_config, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The chain's post-gather op of ``_distributed_group_norm``."""

    return ttnn.rms_norm_post_all_gather(
        x,
        gathered_stats,
        epsilon=EPS,
        weight=weight,
        memory_config=memory_config,
        compute_kernel_config=compute_config,
        dtype=BF16,
    )


def stats_composed(x, compute_config, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    s = ttnn.rms_norm_pre_all_gather(x, dtype=BF16, memory_config=memory_config, compute_kernel_config=compute_config)
    return ttnn.reshape(s, (1, 1, BRANCHES, fp.TILE))


def gate(key_global, query_global, value, *, debug: bool = False):
    """``_gate`` after its all_gathers: ``(key_global, query_global)`` bf16 TILE ``[1,1,4,2560]`` (replicated), ``value``
    bf16 TILE ``[1,1,1,640]`` -> gated bf16 ``[1,1,4,640]``.  ``debug`` also returns the fp32 gate tile (the scaled
    sum, column 0 rows 0-3) and the fp32 coefficient tile as ``[1,1,32,32]`` tensors."""

    _expect(key_global, (1, 1, BRANCHES, HIDDEN), BF16, "PLE key global norm")
    _expect(query_global, (1, 1, BRANCHES, HIDDEN), BF16, "PLE query global norm")
    _expect(value, (1, 1, 1, LOCAL_HIDDEN), BF16, "PLE value projection")
    mesh = key_global.device()
    scale, zero = constants(mesh)
    out = fp.allocate((1, 1, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    dbg = (
        [fp.allocate((1, 1, fp.TILE, fp.TILE), FP32, ttnn.TILE_LAYOUT, mesh) for _ in range(2)] if debug else [out, out]
    )
    core = ttnn.CoreCoord(0, 0)
    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, GLOBAL_TILES, one),
        fp.cb_descriptor(1, BF16, TILE_BF16, GLOBAL_TILES, one),
        fp.cb_descriptor(2, FP32, TILE_FP32, GLOBAL_TILES, one),
        fp.cb_descriptor(3, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(4, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(7, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(8, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(9, BF16, TILE_BF16, LOCAL_TILES, one),
        fp.cb_descriptor(16, BF16, TILE_BF16, LOCAL_TILES, one),
    ]
    if debug:
        cbs += [fp.cb_descriptor(17, FP32, TILE_FP32, 1, one), fp.cb_descriptor(18, FP32, TILE_FP32, 1, one)]
    reader = fp.reader_kernel(
        GATE_READER,
        one,
        [GLOBAL_TILES, LOCAL_TILES]
        + sum((fp.accessor_args(t) for t in (key_global, query_global, value, scale, zero)), []),
        [(core, [t.buffer_address() for t in (key_global, query_global, value, scale, zero)] + [0, 0])],
    )
    compute = fp.compute_kernel(
        GATE_COMPUTE,
        one,
        [GLOBAL_TILES, LOCAL_TILES, int(debug)],
        fp32_dest=True,
        unpack_to_dest_fp32=(2, 4, 5, 6, 7, 8),
    )
    writer = fp.writer_kernel(
        GATE_WRITER,
        one,
        [LOCAL_TILES, int(debug)] + sum((fp.accessor_args(t) for t in (out, dbg[0], dbg[1])), []),
        [(core, [out.buffer_address(), dbg[0].buffer_address(), dbg[1].buffer_address(), 0])],
    )
    io = [key_global, query_global, value, scale, zero, out] + (dbg if debug else [])
    # the two global norms, the value row and the constants in, the gated block out; the fp32 product and sum over
    # the 4 x 2560 gate inputs, the scalar chain, the value repeated and gated
    meta = fp.program_meta(
        NAME,
        "gate",
        BRANCHES,
        reads=(key_global, query_global, value, scale, zero),
        writes=(out, *(dbg if debug else [])),
        flops=4 * BRANCHES * HIDDEN + 8 * BRANCHES + BRANCHES * LOCAL_HIDDEN,
        cores=1,
    )
    fp.run_program(io, fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)
    return (out, dbg[0], dbg[1]) if debug else out


def gate_composed(key_global, query_global, value, compute_config, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The chain's ops of ``_gate`` after the all_gathers (ttnn/ple.py), as written."""

    key_fp32 = ttnn.typecast(key_global, FP32, memory_config=ttnn.L1_MEMORY_CONFIG)
    query_fp32 = ttnn.typecast(query_global, FP32, memory_config=ttnn.L1_MEMORY_CONFIG)
    products = ttnn.multiply(key_fp32, query_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
    unscaled_gate = ttnn.sum(
        products, dim=3, keepdim=True, memory_config=memory_config, compute_kernel_config=compute_config
    )
    gate_t = ttnn.multiply(unscaled_gate, GATE_SCALE, memory_config=memory_config)
    magnitude = ttnn.abs(gate_t, memory_config=memory_config)
    bounded = ttnn.clamp(magnitude, min=1.0e-6, memory_config=memory_config)
    root = ttnn.sqrt(bounded, memory_config=memory_config)
    direction = ttnn.sign(gate_t, memory_config=memory_config)
    transformed = ttnn.multiply(root, direction, memory_config=memory_config)
    coefficient = ttnn.sigmoid(transformed, memory_config=memory_config)
    value_branches = ttnn.repeat(value, (1, 1, BRANCHES, 1), memory_config=memory_config)
    gated = ttnn.multiply(value_branches, coefficient, memory_config=memory_config)
    for t in (
        key_fp32,
        query_fp32,
        products,
        unscaled_gate,
        magnitude,
        bounded,
        root,
        direction,
        transformed,
        value_branches,
    ):
        ttnn.deallocate(t)
    return gated, gate_t, coefficient


def conv(
    state_rows,
    normalized,
    taps,
    gated,
    *,
    mac_form: int = MAC_FORM,
    shift: bool = True,
    debug_stage: int = 0,
    inject_residual=None,
):
    """``_convolve`` + the delta add of one row-block: ``state_rows`` = the nine bf16 conv rows (fixed addresses),
    ``normalized`` the norm_conv output, ``taps`` the four bf16 tap tensors, ``gated`` the gate output -> the delta
    bf16 ``[1,1,4,640]``; with ``shift`` the reader performs the chain's nine in-place state copies while the compute
    runs (its inputs are already in the CBs)."""

    if len(state_rows) != CONV_STATE_LENGTH or len(taps) != CONV_TAPS:
        raise ValueError("PLE conv takes nine state rows and four taps")
    for label, t in (
        ("normalized", normalized),
        ("gated", gated),
        *((f"state row {i}", r) for i, r in enumerate(state_rows)),
        *((f"tap {i}", w) for i, w in enumerate(taps)),
    ):
        _expect(t, (1, 1, BRANCHES, LOCAL_HIDDEN), BF16, f"PLE conv {label}")
    mesh = normalized.device()
    inject = inject_residual is not None
    if inject:
        _expect(inject_residual, (1, BRANCHES, 1, LOCAL_HIDDEN), BF16, "PLE layer residual")
    out = fp.allocate((1, 1, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    injected = fp.allocate((1, BRANCHES, 1, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh) if inject else out
    work = fp.split_work(LOCAL_TILES, mesh)  # one column tile per core, as the chain's elementwise ops
    cores = fp.core_rectangle(work, mesh)
    per_core = work[0].count
    if inject and per_core != 1:
        raise ValueError("the injected form needs one column tile per core")
    inputs = [state_rows[0], state_rows[3], state_rows[6], normalized, *taps, gated]
    others = [state_rows[i] for i in (1, 2, 4, 5, 7, 8)]  # the shift's other rows (reader only)
    residual_in = [inject_residual] if inject else [gated]  # the sixteenth accessor slot (unused without INJECT)
    cbs = [fp.cb_descriptor(i, BF16, TILE_BF16, per_core, cores) for i in range(9)]
    cbs += [fp.cb_descriptor(i, BF16, TILE_BF16, 1, cores) for i in (9, 10, 11)]
    cbs += [
        fp.cb_descriptor(16, BF16, TILE_BF16, per_core, cores),
        fp.cb_descriptor(17, BF16, TILE_BF16, per_core, cores),
    ]
    if inject:
        cbs += [fp.cb_descriptor(i, BF16, TILE_BF16, BRANCHES, cores) for i in (18, 19, 20)]
    reader_tensors = inputs + others + residual_in
    reader = fp.reader_kernel(
        CONV_READER,
        cores,
        [per_core, int(shift), int(inject)] + sum((fp.accessor_args(t) for t in reader_tensors), []),
        [(w.core, [t.buffer_address() for t in reader_tensors] + [w.start, w.start]) for w in work],
    )
    compute = fp.compute_kernel(
        CONV_COMPUTE, cores, [per_core, int(mac_form), int(debug_stage), int(inject)], fp32_dest=False
    )
    writer = fp.writer_kernel(
        CONV_WRITER,
        cores,
        [per_core, int(inject)] + fp.accessor_args(out) + fp.accessor_args(injected),
        [(w.core, [out.buffer_address(), w.start, injected.buffer_address()]) for w in work],
    )
    io = list(inputs) + others + ([inject_residual, out, injected] if inject else [out])  # each buffer once
    # the four conv rows, the gate output (and the layer residual) in, the taps once (each core its column tile of
    # every tap), the delta (or the
    # injected residual) out, and with ``shift`` the nine state copies (eight rows read, nine written); per element
    # the multiply, three macs, the silu and the add (and the residual add)
    meta = fp.program_meta(
        NAME,
        "conv_inject" if inject else "conv",
        BRANCHES,
        reads=(state_rows[0], state_rows[3], state_rows[6], normalized, gated, *((inject_residual,) if inject else ())),
        writes=(injected,) if inject else (out,),
        dram_bytes=sum(fp.tensor_bytes(w) for w in taps) + int(shift) * (8 + 9) * fp.tensor_bytes(normalized),
        flops=(9 + int(inject)) * BRANCHES * LOCAL_HIDDEN,
        cores=len(work),
    )
    fp.run_program(io, fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)
    if inject:
        ttnn.deallocate(out)
        return injected
    return out


def conv_stages_composed(state_rows, normalized, taps):
    """The chain's intermediates: the tap-0 product, the accumulator after the three macs, the silu (no shift)."""

    rows = (state_rows[0], state_rows[3], state_rows[6], normalized)
    product = ttnn.multiply(rows[0], taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
    acc = product
    for row, tap in zip(rows[1:], taps[1:]):
        acc = ttnn.mac(row, tap, acc)
    return product, acc, ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def conv_composed(state_rows, normalized, taps, gated, *, shift: bool = True, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The chain's ops of ``_convolve`` and the delta add (ttnn/ple.py), as written, the state shift included."""

    rows = (state_rows[0], state_rows[3], state_rows[6], normalized)
    convolution = ttnn.multiply(rows[0], taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
    for row, tap in zip(rows[1:], taps[1:]):
        convolution = ttnn.mac(row, tap, convolution)
    convolution = ttnn.silu(convolution, memory_config=memory_config)
    if shift:
        for index in range(CONV_STATE_LENGTH - 1):
            ttnn.copy(state_rows[index + 1], state_rows[index])
        ttnn.copy(normalized, state_rows[-1])
    return ttnn.add(gated, convolution, memory_config=memory_config)


def _group_norm(module, x, weight):
    """``_distributed_group_norm`` on the fused programs around the chain's stats all_gather."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    s = stats(x)
    s.update_tensor_topology(x.tensor_topology())
    module.mesh_contract.validate_tensor(s, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    gathered = ttnn.all_gather(s, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s)
    module.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
    out = normalize(x, gathered, weight)
    ttnn.deallocate(gathered)
    out.update_tensor_topology(x.tensor_topology())
    module._validate_residual(out, label="PLE group-norm output")
    return out


def _body(module, residual_rows, prepared, state, *, inject_residual=None):
    """The fused stages of ``forward_prepared`` on the PLE-layout residual ``[1,1,4,640]``; with ``inject_residual`` (the
    layer's branch-major residual) the conv program also performs the layer's ``add(residual, delta)`` and returns the
    injected residual instead of the delta."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    embedding_tile = ttnn.to_layout(prepared.embedding_sharded, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    embedding_tile.update_tensor_topology(prepared.embedding_sharded.tensor_topology())
    module.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    key, value = module._project(embedding_tile)
    key_norm = _group_norm(module, key, module.weights.norm_key)
    query_norm = _group_norm(module, residual_rows, module.weights.norm_query)
    ttnn.deallocate(key)
    key_global = ttnn.all_gather(key_norm, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    query_global = ttnn.all_gather(query_norm, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(key_norm)
    ttnn.deallocate(query_norm)
    for tensor in (key_global, query_global):
        module.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    gated = gate(key_global, query_global, value)
    ttnn.deallocate(key_global)
    ttnn.deallocate(query_global)
    ttnn.deallocate(value)
    gated.update_tensor_topology(residual_rows.tensor_topology())
    module._validate_residual(gated, label="PLE gated value")  # the residual's shape [1,1,4,640]
    normalized = _group_norm(module, gated, module.weights.norm_conv)
    output = conv(state.conv, normalized, module.weights.conv_taps, gated, shift=True, inject_residual=inject_residual)
    ttnn.deallocate(normalized)
    ttnn.deallocate(gated)
    output.update_tensor_topology(residual_rows.tensor_topology())
    return output


def ple_fused(module, residual, prepared, state):
    """``Qwen38TTNNPLE.forward_prepared`` on the fused programs: the chain's validations, tilize and ``_project`` (the
    embedding gather and the two linears), then the fused group norms, gate and convolution around the chain's
    collectives; the state shift happens in the conv program's writer."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import ple as pm
    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    module._validate_residual(residual, label="PLE residual input")
    state.validate()
    if not isinstance(prepared, pm.Qwen38TTNNPLEPreparedInput) or not prepared.active:
        raise TypeError("PLE prepared input must be a live Qwen38TTNNPLEPreparedInput")
    if not module._same_host_context(state.token_context, prepared.source_token_context):
        raise RuntimeError("prepared PLE input does not match the state's host token context")
    module._validate_prepared_row(prepared.embedding_sharded, label="prepared PLE row")
    output = _body(module, residual, prepared, state)
    module._validate_residual(output, label="PLE residual delta")
    state.token_context = prepared.next_token_context
    state.validate()
    return pm.Qwen38TTNNPLEResult(output, state)


def ple_layer_fused(layer, residual, state, *, token_id=None, prepared_ple=None, release_input: bool = True):
    """``Qwen38TTNNDecoderLayer._apply_ple`` on the fused body: the chain's input permute stays; the output permute and
    the ``add(residual, delta)`` happen inside the conv program.  The eager host-token branch (no prepared row) and the
    layers without a PLE take the chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import ple as pm

    if layer.ple is None or prepared_ple is None:
        return type(layer)._apply_ple(
            layer, residual, state, token_id=token_id, prepared_ple=prepared_ple, release_input=release_input
        )
    module, ple_state = layer.ple, state.ple
    branch_rows = ttnn.permute(residual, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    module._validate_residual(branch_rows, label="PLE residual input")
    ple_state.validate()
    if not isinstance(prepared_ple, pm.Qwen38TTNNPLEPreparedInput) or not prepared_ple.active:
        raise TypeError("PLE prepared input must be a live Qwen38TTNNPLEPreparedInput")
    if not module._same_host_context(ple_state.token_context, prepared_ple.source_token_context):
        raise RuntimeError("prepared PLE input does not match the state's host token context")
    module._validate_prepared_row(prepared_ple.embedding_sharded, label="prepared PLE row")
    injected = _body(module, branch_rows, prepared_ple, ple_state, inject_residual=residual)
    ttnn.deallocate(branch_rows)
    if release_input:
        ttnn.deallocate(residual)
    layer._validate_residual(injected, label="PLE-injected residual")
    ple_state.token_context = prepared_ple.next_token_context
    ple_state.validate()
    return injected, ple_state


def ple_composed(module, residual, prepared, state):
    return type(module).forward_prepared(module, residual, prepared, state)


# ---------------------------------------------------------------------------------------------- the lanes (B row-blocks)
# forward_prepared_lanes runs the same stages on [1, B, 4, 640] tensors (lane u = dim-1 index u, its own 20-tile
# row-block); every stage is per row-block or per element, so the lane forms run the 1-row programs' kernels with one
# core per lane (stats, normalize, gate: the lane's pages) or per column block over the B x 20 tiles (conv: a block
# never spans two lanes, so its tap tiles are the block's columns).  The layer's inject_lanes keeps its permutes and
# the residual add (a lane's delta rows land in every branch tile: not a per-core write).

LANE_COLUMNS = 8


def _lane_core_set(lanes: int):
    cores = [ttnn.CoreCoord(u % LANE_COLUMNS, u // LANE_COLUMNS) for u in range(lanes)]
    by_row: dict[int, list[int]] = {}
    for core in cores:
        by_row.setdefault(core.y, []).append(core.x)
    return cores, ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(min(xs), y), ttnn.CoreCoord(max(xs), y)) for y, xs in sorted(by_row.items())]
    )


def _expect_lanes(tensor, lanes: int, width: int, dtype, label: str) -> None:
    got = tuple(int(v) for v in tensor.shape)
    if got != (1, lanes, BRANCHES, width) or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"{label} must be {dtype} TILE {(1, lanes, BRANCHES, width)}, got {tensor.dtype} {tensor.layout} {got}"
        )


def stats_lanes(x, lanes: int):
    """``rms_norm_pre_all_gather`` of the B row-blocks ``[1,B,4,640]`` -> their stats tiles ``[1,B,4,32]`` (the chain's
    reshape to (1, B, 4, 32) is this shape); one core per lane."""

    from .. import gr_read as gr

    _expect_lanes(x, lanes, LOCAL_HIDDEN, BF16, "PLE lanes group-norm input")
    mesh = x.device()
    out = fp.allocate((1, lanes, BRANCHES, fp.TILE), BF16, ttnn.TILE_LAYOUT, mesh)
    cores, core_set = _lane_core_set(lanes)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, LOCAL_TILES, core_set),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(2, FP32, TILE_FP32, LOCAL_TILES, core_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, core_set),
    ]
    reader = gr._reader(
        core_set,
        [(x, 0)],
        [(gr.CONST_SCALER, 1)],
        [
            (c, ([gr._stream(x, 1, LOCAL_TILES, u * LOCAL_TILES, 1, 0, 4)], [gr._bits(1.0)]))
            for u, c in enumerate(cores)
        ],
    )
    compute = fp.compute_kernel(gr.STATS, core_set, [LOCAL_TILES], fp32_dest=True)
    writer = gr._writer(core_set, [(out, 16)], [(c, [(1, u, 1, 1)]) for u, c in enumerate(cores)])
    meta = fp.program_meta(
        NAME, "stats_lanes", lanes, reads=(x,), writes=(out,), flops=2 * lanes * BRANCHES * LOCAL_HIDDEN, cores=lanes
    )
    return fp.run_program([x, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def normalize_lanes(x, gathered_stats, weight, lanes: int):
    """``rms_norm_post_all_gather(x, gathered, eps, weight=)`` of the B row-blocks: ``gathered_stats`` ``[1,B,4,128]``
    (lane u's four devices' stats tiles), ``weight`` fp32 TILE ``[1,1,4,640]`` (every lane); one core per lane."""

    from .. import gr_read as gr

    _expect_lanes(x, lanes, LOCAL_HIDDEN, BF16, "PLE lanes group-norm input")
    _expect_lanes(gathered_stats, lanes, fp.TILE * STATS_TILES, BF16, "PLE lanes gathered stats")
    _expect(weight, (1, 1, BRANCHES, LOCAL_HIDDEN), FP32, "PLE norm weight")
    mesh = x.device()
    out = fp.allocate((1, lanes, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = gr.avg_scaler("chain")
    cores, core_set = _lane_core_set(lanes)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, LOCAL_TILES, core_set),
        fp.cb_descriptor(1, BF16, TILE_BF16, STATS_TILES, core_set),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, core_set),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, core_set),
        fp.cb_descriptor(4, FP32, TILE_FP32, LOCAL_TILES, core_set),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(7, FP32, TILE_FP32, LOCAL_TILES, core_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, LOCAL_TILES, core_set),
    ]
    reader = gr._reader(
        core_set,
        [(gathered_stats, 1), (x, 0), (weight, 4)],
        [(gr.CONST_SCALER, 2), (gr.CONST_COL_SCALAR, 3)],
        [
            (
                c,
                (
                    [
                        gr._stream(gathered_stats, 1, STATS_TILES, u * STATS_TILES, 1, 0, STATS_TILES),
                        gr._stream(x, 1, LOCAL_TILES, u * LOCAL_TILES, 1, 0, 4),
                        gr._stream(weight, 1, LOCAL_TILES, 0, 1, 0, 4),
                    ],
                    [scaler_bits, gr._bits(EPS)],
                ),
            )
            for u, c in enumerate(cores)
        ],
    )
    compute = fp.compute_kernel(NORM_COMPUTE, core_set, [LOCAL_TILES, STATS_TILES, 4, 16], fp32_dest=True)
    writer = gr._writer(
        core_set, [(out, 16)], [(c, [(LOCAL_TILES, u * LOCAL_TILES, 1, 4)]) for u, c in enumerate(cores)]
    )
    meta = fp.program_meta(  # the weight once per lane core
        NAME,
        "normalize_lanes",
        lanes,
        reads=(x, gathered_stats),
        writes=(out,),
        dram_bytes=lanes * fp.tensor_bytes(weight),
        flops=4 * lanes * BRANCHES * LOCAL_HIDDEN,
        cores=lanes,
    )
    return fp.run_program(
        [x, gathered_stats, weight, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def gate_lanes(key_global, query_global, value, lanes: int):
    """``_gate_rows`` after its all_gathers on B lanes: ``key_global`` / ``query_global`` bf16 TILE ``[1,B,4,2560]``,
    ``value`` bf16 TILE ``[1,B,1,640]`` -> gated bf16 ``[1,B,4,640]``; one core per lane (the 1-row gate program on
    the lane's pages)."""

    _expect_lanes(key_global, lanes, HIDDEN, BF16, "PLE lanes key global norm")
    _expect_lanes(query_global, lanes, HIDDEN, BF16, "PLE lanes query global norm")
    got = tuple(int(v) for v in value.shape)
    if got != (1, lanes, 1, LOCAL_HIDDEN) or value.dtype != BF16 or value.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"PLE lanes value projection must be bf16 TILE {(1, lanes, 1, LOCAL_HIDDEN)}, got {got}")
    mesh = key_global.device()
    scale, zero = constants(mesh)
    out = fp.allocate((1, lanes, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    cores, core_set = _lane_core_set(lanes)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, GLOBAL_TILES, core_set),
        fp.cb_descriptor(1, BF16, TILE_BF16, GLOBAL_TILES, core_set),
        fp.cb_descriptor(2, FP32, TILE_FP32, GLOBAL_TILES, core_set),
        fp.cb_descriptor(3, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(4, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(7, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(8, FP32, TILE_FP32, 1, core_set),
        fp.cb_descriptor(9, BF16, TILE_BF16, LOCAL_TILES, core_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, LOCAL_TILES, core_set),
    ]
    tensors = (key_global, query_global, value, scale, zero)
    reader = fp.reader_kernel(
        GATE_READER,
        core_set,
        [GLOBAL_TILES, LOCAL_TILES] + sum((fp.accessor_args(t) for t in tensors), []),
        [(c, [t.buffer_address() for t in tensors] + [u * GLOBAL_TILES, u * LOCAL_TILES]) for u, c in enumerate(cores)],
    )
    compute = fp.compute_kernel(
        GATE_COMPUTE, core_set, [GLOBAL_TILES, LOCAL_TILES, 0], fp32_dest=True, unpack_to_dest_fp32=(2, 4, 5, 6, 7, 8)
    )
    writer = fp.writer_kernel(
        GATE_WRITER,
        core_set,
        [LOCAL_TILES, 0] + sum((fp.accessor_args(t) for t in (out, out, out)), []),
        [
            (c, [out.buffer_address(), out.buffer_address(), out.buffer_address(), u * LOCAL_TILES])
            for u, c in enumerate(cores)
        ],
    )
    meta = fp.program_meta(  # the constants once per lane core
        NAME,
        "gate_lanes",
        lanes,
        reads=(key_global, query_global, value),
        writes=(out,),
        dram_bytes=lanes * (fp.tensor_bytes(scale) + fp.tensor_bytes(zero)),
        flops=lanes * (4 * BRANCHES * HIDDEN + 8 * BRANCHES + BRANCHES * LOCAL_HIDDEN),
        cores=lanes,
    )
    fp.run_program([*tensors, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)
    return out


def conv_lanes(state_rows, normalized, taps, gated, lanes: int, *, mac_form: int = MAC_FORM, shift: bool = True):
    """``_convolve_lanes`` + the delta add on the B row-blocks: ``state_rows`` = the nine bf16 ``[1,B,4,640]`` slots
    (fixed addresses, shifted in place by the reader), ``normalized`` / ``gated`` ``[1,B,4,640]``, ``taps`` the four
    bf16 ``[1,1,4,640]`` tap tensors -> the delta ``[1,B,4,640]``.  The B x 20 tiles in column blocks of T tiles (T
    divides 20: a block stays inside one lane, its tap tiles are the block's columns), one block per core."""

    if len(state_rows) != CONV_STATE_LENGTH or len(taps) != CONV_TAPS:
        raise ValueError("PLE conv takes nine state rows and four taps")
    for label, t in (
        ("normalized", normalized),
        ("gated", gated),
        *((f"state row {i}", r) for i, r in enumerate(state_rows)),
    ):
        _expect_lanes(t, lanes, LOCAL_HIDDEN, BF16, f"PLE lanes conv {label}")
    for i, w in enumerate(taps):
        _expect(w, (1, 1, BRANCHES, LOCAL_HIDDEN), BF16, f"PLE conv tap {i}")
    mesh = normalized.device()
    out = fp.allocate((1, lanes, BRANCHES, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    grid = mesh.compute_with_storage_grid_size()
    tiles = lanes * LOCAL_TILES
    per_core = next(T for T in (1, 2, 4, 5, 10, 20) if tiles // T <= grid.x * grid.y)
    work = fp.split_work(tiles // per_core, mesh)  # one block of per_core consecutive tiles per core
    cores = fp.core_rectangle(work, mesh)
    inputs = [state_rows[0], state_rows[3], state_rows[6], normalized, *taps, gated]
    others = [state_rows[i] for i in (1, 2, 4, 5, 7, 8)]
    reader_tensors = inputs + others + [gated]  # the sixteenth accessor slot (the residual, unused without INJECT)
    cbs = [fp.cb_descriptor(i, BF16, TILE_BF16, per_core, cores) for i in range(9)]
    cbs += [fp.cb_descriptor(i, BF16, TILE_BF16, 1, cores) for i in (9, 10, 11)]
    cbs += [
        fp.cb_descriptor(16, BF16, TILE_BF16, per_core, cores),
        fp.cb_descriptor(17, BF16, TILE_BF16, per_core, cores),
    ]
    reader = fp.reader_kernel(
        CONV_READER,
        cores,
        [per_core, int(shift), 0] + sum((fp.accessor_args(t) for t in reader_tensors), []),
        [
            (
                w.core,
                [t.buffer_address() for t in reader_tensors] + [w.start * per_core, (w.start * per_core) % LOCAL_TILES],
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(CONV_COMPUTE, cores, [per_core, int(mac_form), 0, 0], fp32_dest=False)
    writer = fp.writer_kernel(
        CONV_WRITER,
        cores,
        [per_core, 0] + fp.accessor_args(out) + fp.accessor_args(out),
        [(w.core, [out.buffer_address(), w.start * per_core, out.buffer_address()]) for w in work],
    )
    io = list(inputs) + others + [out]  # each buffer once
    meta = fp.program_meta(  # as ``conv`` on the B row-blocks; the taps once per lane block
        NAME,
        "conv_lanes",
        lanes,
        reads=(state_rows[0], state_rows[3], state_rows[6], normalized, gated),
        writes=(out,),
        dram_bytes=lanes * sum(fp.tensor_bytes(w) for w in taps) + int(shift) * (8 + 9) * fp.tensor_bytes(normalized),
        flops=9 * lanes * BRANCHES * LOCAL_HIDDEN,
        cores=len(work),
    )
    fp.run_program(io, fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)
    return out


def _group_norm_lanes(module, x, weight, lanes: int):
    """``_distributed_group_norm_rows`` on the fused lane programs around the chain's stats all_gather."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    s = stats_lanes(x, lanes)
    s.update_tensor_topology(x.tensor_topology())
    module.mesh_contract.validate_tensor(s, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    gathered = ttnn.all_gather(s, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(s)
    module.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
    out = normalize_lanes(x, gathered, weight, lanes)
    ttnn.deallocate(gathered)
    out.update_tensor_topology(x.tensor_topology())
    module._validate_rows_residual(out, lanes, label="PLE lanes group-norm output")
    return out


def ple_lanes_fused(module, residual_lanes, prepared, lanes_state):
    """``Qwen38TTNNPLE.forward_prepared_lanes`` on the fused lane programs: the chain's validations, tilize and
    ``_project_rows`` (the embedding gather and the two linears), then the fused group norms, gate and convolution
    around the chain's collectives (the state shift happens in the conv program's reader); the contexts advance."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import ple as pm
    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    lanes_state.validate()
    lanes = lanes_state.lanes
    module._validate_rows_residual(residual_lanes, lanes, label="PLE lanes residual input")
    if not isinstance(prepared, pm.Qwen38TTNNPLELanesPreparedInput) or not prepared.active:
        raise TypeError("PLE lanes input must be a live Qwen38TTNNPLELanesPreparedInput")
    if prepared.source_contexts != lanes_state.token_contexts:
        raise RuntimeError(
            f"prepared PLE lanes were looked up from contexts {prepared.source_contexts}, the state holds "
            f"{lanes_state.token_contexts}"
        )
    module._validate_prepared_rows(prepared.embedding_rows, lanes, label="prepared PLE lanes")
    embedding_tile = ttnn.to_layout(prepared.embedding_rows, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    embedding_tile.update_tensor_topology(prepared.embedding_rows.tensor_topology())
    module.mesh_contract.validate_tensor(embedding_tile, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    key, value = module._project_rows(embedding_tile, lanes)
    key_norm = _group_norm_lanes(module, key, module.weights.norm_key, lanes)
    query_norm = _group_norm_lanes(module, residual_lanes, module.weights.norm_query, lanes)
    ttnn.deallocate(key)
    key_global = ttnn.all_gather(key_norm, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    query_global = ttnn.all_gather(query_norm, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(key_norm)
    ttnn.deallocate(query_norm)
    for tensor in (key_global, query_global):
        module.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    gated = gate_lanes(key_global, query_global, value, lanes)
    ttnn.deallocate(key_global)
    ttnn.deallocate(query_global)
    ttnn.deallocate(value)
    gated.update_tensor_topology(residual_lanes.tensor_topology())
    module._validate_rows_residual(gated, lanes, label="PLE lanes gated value")
    normalized = _group_norm_lanes(module, gated, module.weights.norm_conv, lanes)
    output = conv_lanes(lanes_state.conv, normalized, module.weights.conv_taps, gated, lanes, shift=True)
    ttnn.deallocate(normalized)
    ttnn.deallocate(gated)
    output.update_tensor_topology(residual_lanes.tensor_topology())
    module._validate_rows_residual(output, lanes, label="PLE lanes residual delta")
    lanes_state.token_contexts = prepared.next_contexts
    lanes_state.validate()
    return output


def ple_lanes_composed(module, residual_lanes, prepared, lanes_state):
    return type(module).forward_prepared_lanes(module, residual_lanes, prepared, lanes_state)


register(
    FusedKernel(
        name=NAME,
        replaces="Qwen38TTNNPLE.forward_prepared (layer 1, once per step): the group norms (3 x [pre, reshape, post]), the "
        "gate chain (12 programs), the convolution (multiply, 3 mac, silu), the delta add and the 10 state copies -> stats x3, "
        "normalize x3, gate, conv = 8 programs; tilize, the embedding gather, the two linears and the 6 collectives stay; "
        "forward_prepared_lanes runs the same 8 programs on the B row-blocks (one core per lane, the conv per column block)",
        tolerance=BITWISE,
        fused=ple_fused,
        composed=ple_composed,
        gate=None,
    )
)
