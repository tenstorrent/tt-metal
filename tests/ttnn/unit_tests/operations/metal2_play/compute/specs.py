# SPDX-License-Identifier: Apache-2.0
"""Metal 2.0 ProgramSpec builders for the compute/DFB-memory probes.

Every builder returns (spec, run_args, tensor_arg_map) and everything runs on a single core so
the probes exercise the *expression* of the programming model, not work distribution.
"""

import struct
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"

CORE = ttnn.CoreCoord(0, 0)
ONE_CORE = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])

TILE_BF16 = 32 * 32 * 2
TILE_F32 = 32 * 32 * 4


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _wu(kernels):
    return [ttnn.WorkUnitSpec(name="main", kernels=kernels, target_nodes=ONE_CORE)]


def _rta(name_to_value):
    return {k: {CORE: v} for k, v in name_to_value.items()}


def reader_kspec(uid="reader", *, tensor="src", num_tiles_rta=True):
    return ttnn.KernelSpec(
        unique_id=uid,
        source=str(KERNELS / "reader_tiles.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of("in_tiles", "in_tiles")],
        tensor_bindings=[ttnn.TensorBinding(tensor, "src")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )


def writer_kspec(uid="writer", *, dfb="out_tiles", tensor="dst"):
    return ttnn.KernelSpec(
        unique_id=uid,
        source=str(KERNELS / "writer_tiles.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of(dfb, "out_tiles")],
        tensor_bindings=[ttnn.TensorBinding(tensor, "dst")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )


def dfb(name, entry_size, num_entries, dtype, **kw):
    return ttnn.DataflowBufferSpec(
        unique_id=name, entry_size=entry_size, num_entries=num_entries, data_format=dtype, **kw
    )


# ---------------------------------------------------------------- PROBE A: reduce helper


def build_reduce(a, out, Ht, Wt):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "reduce_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(fpu_math_fidelity=ttnn.MathFidelity.HiFi4),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.consumer_of("scaler", "scaler"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        compile_time_args={"Ht": Ht, "Wt": Wt},
    )
    reader = ttnn.KernelSpec(
        unique_id="reader",
        source=str(KERNELS / "reduce_reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[
            ttnn.producer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("scaler", "scaler"),
        ],
        tensor_bindings=[ttnn.TensorBinding("src", "src")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    n_in = Ht * Wt
    spec = ttnn.ProgramSpec(
        name="probe_reduce",
        kernels=[reader, compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, n_in, a.dtype),
            dfb("scaler", TILE_BF16, 1, a.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_in})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": Ht})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE B: eltwise chain helper


def build_chain(a, out, Ht, Wt):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "chain_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        compile_time_args={"Ht": Ht, "Wt": Wt},
    )
    n = Ht * Wt
    spec = ttnn.ProgramSpec(
        name="probe_chain",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, 2, a.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE C: scratchpad on compute


def build_scratchpad(a, out, n_tiles, pad_bytes, kernel_file="scratch_compute.cpp"):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / kernel_file),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        scratchpad_bindings=[ttnn.ScratchpadBinding("scale_table", "scale_table")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_scratchpad",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, 2, a.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        scratchpads=[ttnn.ScratchpadSpec(unique_id="scale_table", size_per_node=pad_bytes)],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE D: aliased DFBs


def build_alias(a, out, n_tiles, *, aliased, entries=None, in_dtype="use_tensor"):
    """1-tile-at-a-time copy. When `aliased`, in_tiles and out_tiles share one L1 region."""
    entries = entries if entries is not None else n_tiles
    in_dtype = a.dtype if in_dtype == "use_tensor" else in_dtype
    in_opts = ttnn.DFBAdvancedOptions(alias_with=["out_tiles"]) if aliased else ttnn.DFBAdvancedOptions()
    out_opts = ttnn.DFBAdvancedOptions(alias_with=["in_tiles"]) if aliased else ttnn.DFBAdvancedOptions()

    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "copy_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    dfbs = [
        dfb("in_tiles", TILE_BF16, entries, in_dtype, advanced_options=in_opts),
        dfb("out_tiles", TILE_BF16, entries, out.dtype, advanced_options=out_opts),
    ]
    spec = ttnn.ProgramSpec(
        name="probe_alias",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


def build_alias_stress(a, out, n_tiles, *, n_scratch_dfbs, entries_each, aliased):
    """`n_scratch_dfbs` same-size DFBs on top of the working pair, optionally all in one alias
    clique. Aliased => one region's worth of L1; not aliased => n_scratch_dfbs regions.

    Each scratch DFB is bound producer=reader / consumer=writer and never touched at runtime, so
    it contributes L1 footprint only.
    """
    clique = [f"pad{i}" for i in range(n_scratch_dfbs)]

    def opts(me):
        if not aliased or n_scratch_dfbs < 2:
            return ttnn.DFBAdvancedOptions()
        return ttnn.DFBAdvancedOptions(alias_with=[n for n in clique if n != me])

    reader = ttnn.KernelSpec(
        unique_id="reader",
        source=str(KERNELS / "reader_tiles.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of("in_tiles", "in_tiles")] + [ttnn.producer_of(n, n) for n in clique],
        tensor_bindings=[ttnn.TensorBinding("src", "src")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = ttnn.KernelSpec(
        unique_id="writer",
        source=str(KERNELS / "writer_tiles.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of("out_tiles", "out_tiles")] + [ttnn.consumer_of(n, n) for n in clique],
        tensor_bindings=[ttnn.TensorBinding("dst", "dst")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "copy_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    dfbs = [
        dfb("in_tiles", TILE_BF16, 2, a.dtype),
        dfb("out_tiles", TILE_BF16, 2, out.dtype),
    ] + [dfb(n, TILE_BF16, entries_each, a.dtype, advanced_options=opts(n)) for n in clique]

    spec = ttnn.ProgramSpec(
        name="probe_alias_stress",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE E: fp32 dest + unpack_modes


def build_fp32(a, out, n_tiles, *, enable_32_bit_dest, unpack_mode):
    unpack_modes = {} if unpack_mode is None else {"in_tiles": unpack_mode}
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "copy_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(
            enable_32_bit_dest=enable_32_bit_dest,
            unpack_modes=unpack_modes,
        ),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_fp32",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_F32, 2, a.dtype),
            dfb("out_tiles", TILE_F32, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE F: compute + tensor access


def build_compute_tensor_access(a, out, scale_t, n_tiles, *, kernel_file):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / kernel_file),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        tensor_bindings=[ttnn.TensorBinding("scale", "scale")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_compute_tensor_access",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, 2, a.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
            ttnn.TensorParameter(unique_id="scale", spec=scale_t.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1, "scale": 2}


# ---------------------------------------------------------------- PROBE F3: borrowed-memory DFB


def build_borrowed(resident_in, out, n_tiles):
    """No reader NoC transfer at all: the input DFB IS the L1-resident input tensor."""
    producer = ttnn.KernelSpec(
        unique_id="producer",
        source=str(KERNELS / "borrowed_producer.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of("in_tiles", "in_tiles")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "copy_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_borrowed",
        kernels=[producer, compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, n_tiles, resident_in.dtype, borrowed_from="src"),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=resident_in.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["producer", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="producer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE G: self-loop DFB


def build_selfloop(a, resident_out, n_tiles):
    """The compute kernel packs straight into the L1-resident output tensor. No writer kernel."""
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "selfloop_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            # Same accessor name, both roles -- the sanctioned self-loop pair.
            ttnn.producer_of("resident_out", "resident_out"),
            ttnn.consumer_of("resident_out", "resident_out"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_selfloop",
        kernels=[reader_kspec(), compute],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, 2, a.dtype),
            dfb("resident_out", TILE_BF16, n_tiles, resident_out.dtype, borrowed_from="dst"),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=resident_out.spec),
        ],
        work_units=_wu(["reader", "compute"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE H: TT_KERNEL on compute


def build_ttkernel(a, out, n_tiles, *, do_scale, scale):
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "ttk_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        compile_time_args={"do_scale": int(do_scale), "scale_bits": f32_bits(scale)},
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_ttkernel",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", TILE_BF16, 2, a.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}


# ---------------------------------------------------------------- PROBE I: matmul_block helper


def build_matmul(a, b, out, kernel_file="matmul_compute.cpp"):
    reader = ttnn.KernelSpec(
        unique_id="reader",
        source=str(KERNELS / "matmul_reader.cpp"),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of("in0", "in0"), ttnn.producer_of("in1", "in1")],
        tensor_bindings=[ttnn.TensorBinding("a", "a"), ttnn.TensorBinding("b", "b")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / kernel_file),
        hw_config=ttnn.ComputeGen1Config(fpu_math_fidelity=ttnn.MathFidelity.HiFi2),
        dfb_bindings=[
            ttnn.consumer_of("in0", "in0"),
            ttnn.consumer_of("in1", "in1"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
    )
    spec = ttnn.ProgramSpec(
        name="probe_matmul",
        kernels=[reader, compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in0", TILE_BF16, 2, a.dtype),
            dfb("in1", TILE_BF16, 2, b.dtype),
            dfb("out_tiles", TILE_BF16, 2, out.dtype),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="a", spec=a.spec),
            ttnn.TensorParameter(unique_id="b", spec=b.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": 1})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": 1})),
        ]
    )
    return spec, run, {"a": 0, "b": 1, "dst": 2}


# ---------------------------------------------------------------- PROBE M: DFB format plumbing


def build_mixed_format(a, out, n_tiles, *, in_entry, out_entry, in_fmt, out_fmt):
    """The compute kernel source is format-agnostic; formats come only from the DFB specs."""
    compute = ttnn.KernelSpec(
        unique_id="compute",
        source=str(KERNELS / "copy_compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        dfb_bindings=[
            ttnn.consumer_of("in_tiles", "in_tiles"),
            ttnn.producer_of("out_tiles", "out_tiles"),
        ],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    spec = ttnn.ProgramSpec(
        name="probe_mixed_format",
        kernels=[reader_kspec(), compute, writer_kspec()],
        dataflow_buffers=[
            dfb("in_tiles", in_entry, 2, in_fmt),
            dfb("out_tiles", out_entry, 2, out_fmt),
        ],
        tensor_parameters=[
            ttnn.TensorParameter(unique_id="src", spec=a.spec),
            ttnn.TensorParameter(unique_id="dst", spec=out.spec),
        ],
        work_units=_wu(["reader", "compute", "writer"]),
    )
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel="reader", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="compute", runtime_arg_values=_rta({"num_tiles": n_tiles})),
            ttnn.KernelRunArgs(kernel="writer", runtime_arg_values=_rta({"num_tiles": n_tiles})),
        ]
    )
    return spec, run, {"src": 0, "dst": 1}
