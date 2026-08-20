"""Two core groups with different per-group CTA + varargs, from Python ProgramSpec.

This is the shape EVERY op we generate uses: split_work_to_cores yields core_group_1 and
core_group_2 with different per-core tile counts. Under ProgramDescriptor that is two
KernelDescriptors. Here: two KernelSpecs of one source, two WorkUnitSpecs, disjoint nodes.
"""

import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).parent))

K = Path(__file__).parent / "kernels"
LOG = Path(__file__).parent.parent / "VIOLATION_MESSAGES.txt"


def _record(tag, msg, limit=2500):
    with LOG.open("a") as f:
        f.write("\n" + "=" * 78 + "\n### " + tag + "\n" + "=" * 78 + "\n" + msg[:limit] + "\n")


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _mk(device, shape, seed=0):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    return ta, tb, a, b, out


def _split_two_groups(num_tiles, grid):
    """Deliberately ragged: emulate split_work_to_cores' two-group output."""
    ncores = min(grid.x * grid.y, num_tiles)
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:ncores]
    base, rem = divmod(num_tiles, ncores)
    g1 = cores[:rem]  # get base+1
    g2 = cores[rem:]  # get base
    return g1, base + 1, g2, base


def _build_two_group(a, b, out, *, use_varargs=False):
    tile = out.buffer_page_size()
    ntiles = out.buffer_num_pages()
    grid = a.device().compute_with_storage_grid_size()
    g1, n1, g2, n2 = _split_two_groups(ntiles, grid)
    assert g1 and g2, f"need a ragged split; got {len(g1)}/{len(g2)}"

    all_cores = g1 + g2
    crs = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs])

    dfbs = [
        ttnn.DataflowBufferSpec(unique_id=n, entry_size=tile, num_entries=2, data_format=a.dtype)
        for n in ("in_a", "in_b", "out")
    ]
    tps = [
        ttnn.TensorParameter(unique_id="a", spec=a.spec),
        ttnn.TensorParameter(unique_id="b", spec=b.spec),
        ttnn.TensorParameter(unique_id="out", spec=out.spec),
    ]

    adv = ttnn.KernelAdvancedOptions(num_runtime_varargs=2) if use_varargs else ttnn.KernelAdvancedOptions()
    reader = ttnn.KernelSpec(
        unique_id="reader",
        source=str(K / ("reader_va.cpp" if use_varargs else "reader.cpp")),
        hw_config=ttnn.create_reader_dm_config(),
        dfb_bindings=[ttnn.producer_of("in_a", "in_a"), ttnn.producer_of("in_b", "in_b")],
        tensor_bindings=[ttnn.TensorBinding("a", "a"), ttnn.TensorBinding("b", "b")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=[] if use_varargs else ["num_tiles", "start_id"]),
        advanced_options=adv,
    )
    writer = ttnn.KernelSpec(
        unique_id="writer",
        source=str(K / "writer.cpp"),
        hw_config=ttnn.create_writer_dm_config(),
        dfb_bindings=[ttnn.consumer_of("out", "out")],
        tensor_bindings=[ttnn.TensorBinding("out", "out")],
        runtime_arg_schema=ttnn.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]),
    )

    def compute(uid, tiles_per_core):
        return ttnn.KernelSpec(
            unique_id=uid,
            source=str(K / "compute_grouped.cpp"),  # SAME source, different CTA
            hw_config=ttnn.ComputeGen1Config(),
            dfb_bindings=[
                ttnn.consumer_of("in_a", "in_a"),
                ttnn.consumer_of("in_b", "in_b"),
                ttnn.producer_of("out", "out"),
            ],
            compile_time_args={"tiles_per_core": tiles_per_core},
        )

    spec = ttnn.ProgramSpec(
        name="two_group",
        kernels=[reader, writer, compute("compute_g1", n1), compute("compute_g2", n2)],
        dataflow_buffers=dfbs,
        tensor_parameters=tps,
        work_units=[
            # reader + writer belong to BOTH work units; each compute to exactly one.
            ttnn.WorkUnitSpec(name="g1", kernels=["reader", "writer", "compute_g1"], target_nodes=crs(g1)),
            ttnn.WorkUnitSpec(name="g2", kernels=["reader", "writer", "compute_g2"], target_nodes=crs(g2)),
        ],
    )

    counts, starts, start = {}, {}, 0
    for c in g1:
        counts[c], starts[c] = n1, start
        start += n1
    for c in g2:
        counts[c], starts[c] = n2, start
        start += n2

    kra = [ttnn.KernelRunArgs(kernel="writer", runtime_arg_values={"num_tiles": counts, "start_id": starts})]
    if use_varargs:
        kra.append(
            ttnn.KernelRunArgs(
                kernel="reader",
                advanced_options=ttnn.AdvancedKernelRunArgs(
                    runtime_varargs={c: [counts[c], starts[c]] for c in all_cores}
                ),
            )
        )
    else:
        kra.append(ttnn.KernelRunArgs(kernel="reader", runtime_arg_values={"num_tiles": counts, "start_id": starts}))
    return spec, ttnn.ProgramRunArgs(kernel_run_args=kra), (len(g1), n1, len(g2), n2)


def test_two_core_groups_same_source_different_cta(device):
    """The split_work_to_cores shape: does it survive into Metal 2.0 from Python?"""
    ta, tb, a, b, out = _mk(device, (1, 1, 32, 32 * 200), seed=40)  # 100 tiles -> ragged
    spec, run, info = _build_two_group(a, b, out)
    print(f"\n[GROUPS] g1={info[0]} cores x {info[1]} tiles, g2={info[2]} cores x {info[3]} tiles")
    ttnn.generic_op([a, b, out], spec, run, {"a": 0, "b": 1, "out": 2})
    got = ttnn.to_torch(out).float()
    assert torch.allclose(got, (ta * tb).float(), atol=0.15), (got - (ta * tb).float()).abs().max()


def test_overlapping_work_units_same_role_is_rejected(device):
    """Two computes bound to the same DFB role over OVERLAPPING nodes must be caught."""
    _, _, a, b, out = _mk(device, (1, 1, 32, 32 * 200), seed=41)
    spec, run, _ = _build_two_group(a, b, out)
    # make g2's work unit cover g1's nodes too -> two computes on one node
    wus = list(spec.work_units)
    union = ttnn.CoreRangeSet(
        [ttnn.CoreRange(c, c) for c in [ttnn.CoreCoord(x, y) for y in range(2) for x in range(4)]]
    )
    wus[1] = ttnn.WorkUnitSpec(name="g2", kernels=["reader", "writer", "compute_g2"], target_nodes=union)
    spec.work_units = wus
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        Exception
    ) as ei:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, b, out], spec, run, {"a": 0, "b": 1, "out": 2})
    _record("[TWO COMPUTE KERNELSPECS OVERLAPPING ON ONE NODE]", str(ei.value))
    print(f"\n[OVERLAP] rejected: {str(ei.value).splitlines()[:5]}")


def test_runtime_varargs_from_python(device):
    """num_runtime_varargs on the schema + runtime_varargs on the run args."""
    ta, tb, a, b, out = _mk(device, (1, 1, 32, 32 * 200), seed=42)
    spec, run, _ = _build_two_group(a, b, out, use_varargs=True)
    ttnn.generic_op([a, b, out], spec, run, {"a": 0, "b": 1, "out": 2})
    got = ttnn.to_torch(out).float()
    assert torch.allclose(got, (ta * tb).float(), atol=0.15), (got - (ta * tb).float()).abs().max()


def test_cta_change_vs_rta_change_cache_cost(device):
    """A CTA value is baked into kernel_args_generated.h -> new hash -> recompile.
    An RTA value is not. Measures the real cost of choosing CTA over RTA."""
    results = {}
    base = device.num_program_cache_entries()

    # (a) two shapes whose ragged split yields DIFFERENT per-group CTAs
    for n_tiles in (200, 300, 401):
        ta, tb, a, b, out = _mk(device, (1, 1, 32, 32 * n_tiles), seed=50 + n_tiles)
        spec, run, info = _build_two_group(a, b, out)
        ttnn.generic_op([a, b, out], spec, run, {"a": 0, "b": 1, "out": 2})
        results[f"cta n={n_tiles} (g1 {info[1]}t / g2 {info[3]}t)"] = device.num_program_cache_entries()

    after_cta = device.num_program_cache_entries()

    # (b) repeat the SAME shape with fresh tensors -> only addresses change
    for rep in range(3):
        ta, tb, a, b, out = _mk(device, (1, 1, 32, 32 * 200), seed=60 + rep)
        spec, run, _ = _build_two_group(a, b, out)
        ttnn.generic_op([a, b, out], spec, run, {"a": 0, "b": 1, "out": 2})
    after_reps = device.num_program_cache_entries()

    print(f"\n[CACHE] baseline entries          = {base}")
    for k, v in results.items():
        print(f"[CACHE] after {k:<34} = {v}")
    print(f"[CACHE] after 3 reps of an EXISTING shape = {after_reps} (delta {after_reps - after_cta})")
    assert after_reps == after_cta, "re-running a known shape must not add cache entries"
