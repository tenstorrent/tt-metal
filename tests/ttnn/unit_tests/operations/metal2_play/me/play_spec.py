# Metal 2.0 playground: optional DFBs, local L1 copies, self-loop, validator violations.
from pathlib import Path

import ttnn

K = Path(__file__).parent / "kernels"

DFB_A, DFB_B, DFB_OUT, DFB_OUT2, DFB_STAGE = "in_a", "in_b", "out", "out2", "stage"
K_READER, K_COMPUTE, K_WRITER = "reader", "compute", "writer"
TP_A, TP_B, TP_OUT, TP_OUT2 = "a", "b", "out", "out2"


def _split(num_tiles, grid):
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:num_tiles]
    n = max(len(cores), 1)
    base, rem = divmod(num_tiles, n)
    out, start = [], 0
    for i, c in enumerate(cores):
        cnt = base + (1 if i < rem else 0)
        out.append((c, cnt, start))
        start += cnt
    return out


def build(
    a,
    b,
    out,
    out2=None,
    *,
    gate=None,  # None | "ifdef" | "always"
    stage_local=False,
    local_route="corelocalmem",  # "corelocalmem" | "unicast_self"
    ttk_reader=False,
    ttk_unbound_token=False,
    ttk_tiles_per_iter=1,
    # deliberate-violation knobs
    break_producer_only=False,
    break_unbound_tensor_param=False,
    break_two_accessor_names=False,
    break_self_loop_missing=False,
    stage_depth=2,
):
    tile = out.buffer_page_size()
    ntiles = out.buffer_num_pages()
    assignment = _split(ntiles, a.device().compute_with_storage_grid_size())
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in assignment])

    fused = gate is not None
    defines = {}
    if gate == "ifdef":
        defines["GATE_IFDEF"] = "1"
    elif gate == "always":
        defines["GATE_ALWAYS"] = "1"
    w_defines = dict(defines)
    if stage_local:
        w_defines["STAGE_LOCAL"] = "1"
        if local_route == "unicast_self":
            w_defines["LOCAL_VIA_UNICAST_SELF"] = "1"
        elif local_route == "self_read":
            w_defines["LOCAL_VIA_SELF_READ"] = "1"

    dfbs = [
        ttnn.DataflowBufferSpec(unique_id=DFB_A, entry_size=tile, num_entries=2, data_format=a.dtype),
        ttnn.DataflowBufferSpec(unique_id=DFB_B, entry_size=tile, num_entries=2, data_format=b.dtype),
        ttnn.DataflowBufferSpec(unique_id=DFB_OUT, entry_size=tile, num_entries=2, data_format=out.dtype),
    ]
    if fused:
        dfbs.append(ttnn.DataflowBufferSpec(unique_id=DFB_OUT2, entry_size=tile, num_entries=2, data_format=out.dtype))
    if stage_local:
        dfbs.append(
            ttnn.DataflowBufferSpec(
                unique_id=DFB_STAGE, entry_size=tile, num_entries=stage_depth, data_format=out.dtype
            )
        )

    tps = [
        ttnn.TensorParameter(unique_id=TP_A, spec=a.spec),
        ttnn.TensorParameter(unique_id=TP_B, spec=b.spec),
        ttnn.TensorParameter(unique_id=TP_OUT, spec=out.spec),
    ]
    if fused:
        tps.append(ttnn.TensorParameter(unique_id=TP_OUT2, spec=out2.spec))
    if break_unbound_tensor_param:
        # declared but no kernel binds it
        tps.append(ttnn.TensorParameter(unique_id="orphan", spec=a.spec))

    r_dfbs = [ttnn.producer_of(DFB_A, "in_a"), ttnn.producer_of(DFB_B, "in_b")]
    if break_two_accessor_names:
        r_dfbs.append(ttnn.producer_of(DFB_A, "in_a_again"))

    r_defines = {"TTK_UNBOUND_TOKEN": "1"} if ttk_unbound_token else {}
    reader = ttnn.KernelSpec(
        unique_id=K_READER,
        source=str(K / ("reader_ttk.cpp" if ttk_reader else "reader.cpp")),
        hw_config=ttnn.create_reader_dm_config(),
        compiler_options=ttnn.CompilerOptions(defines=r_defines),
        dfb_bindings=r_dfbs,
        tensor_bindings=[ttnn.TensorBinding(TP_A, "a"), ttnn.TensorBinding(TP_B, "b")],
        compile_time_args=({"tiles_per_iter": ttk_tiles_per_iter, "touch_optional": 0} if ttk_reader else {}),
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )

    c_dfbs = [
        ttnn.consumer_of(DFB_A, "in_a"),
        ttnn.consumer_of(DFB_B, "in_b"),
        ttnn.producer_of(DFB_OUT, "out"),
    ]
    if fused:
        c_dfbs.append(ttnn.producer_of(DFB_OUT2, "out2"))
    c_cta = {"emit_second": 1} if gate == "always" else {}
    compute = ttnn.KernelSpec(
        unique_id=K_COMPUTE,
        source=str(K / "compute.cpp"),
        hw_config=ttnn.ComputeGen1Config(),
        compiler_options=ttnn.CompilerOptions(defines=defines),
        dfb_bindings=c_dfbs,
        compile_time_args=c_cta,
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )

    w_dfbs = [ttnn.consumer_of(DFB_OUT, "out")]
    if fused:
        w_dfbs.append(ttnn.consumer_of(DFB_OUT2, "out2"))
    if stage_local:
        # self-loop: the writer is both producer and consumer of the staging DFB
        w_dfbs.append(ttnn.producer_of(DFB_STAGE, "stage"))
        if not break_self_loop_missing:
            w_dfbs.append(ttnn.consumer_of(DFB_STAGE, "stage"))
    w_tb = [ttnn.TensorBinding(TP_OUT, "out")]
    if fused:
        w_tb.append(ttnn.TensorBinding(TP_OUT2, "out2"))
    writer = ttnn.KernelSpec(
        unique_id=K_WRITER,
        source=str(K / "writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        compiler_options=ttnn.CompilerOptions(defines=w_defines),
        dfb_bindings=w_dfbs,
        tensor_bindings=w_tb,
        compile_time_args=c_cta,
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )

    if break_producer_only:
        # drop every consumer binding of DFB_OUT -> producer-only DFB
        writer.dfb_bindings = [x for x in w_dfbs if x.dfb_spec_name != DFB_OUT]

    spec = ttnn.ProgramSpec(
        name="play_optional",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tps,
        work_units=[ttnn.WorkUnitSpec(name="main", kernels=[K_READER, K_COMPUTE, K_WRITER], target_nodes=cores)],
    )

    nt = {c: n for c, n, _ in assignment}
    sid = {c: s for c, _, s in assignment}
    dm = {"num_tiles": nt, "start_id": sid}
    run = ttnn.ProgramRunArgs(
        kernel_run_args=[
            ttnn.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm),
            ttnn.KernelRunArgs(kernel=K_COMPUTE, runtime_arg_values={"num_tiles": nt}),
            ttnn.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm),
        ]
    )
    return spec, run


def run_op(a, b, out, out2=None, **kw):
    spec, run = build(a, b, out, out2, **kw)
    io = [a, b, out]
    mapping = {TP_A: 0, TP_B: 1, TP_OUT: 2}
    if out2 is not None and kw.get("gate") is not None:
        io.append(out2)
        mapping[TP_OUT2] = 3
    return ttnn.generic_op(io, spec, run, mapping)
