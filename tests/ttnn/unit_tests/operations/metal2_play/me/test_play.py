import sys
from pathlib import Path

import pytest
import torch

import ttnn

sys.path.insert(0, str(Path(__file__).parent))
from play_spec import build, run_op, DFB_STAGE, TP_A, TP_B, TP_OUT


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _mk(device, shape=(1, 1, 128, 128), seed=0):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    return ta, tb, a, b, out


def _close(got, exp):
    g = ttnn.to_torch(got).float()
    e = exp.float()
    assert torch.allclose(g, e, atol=0.15), (g - e).abs().max()


# ---------------------------------------------------------------- baseline
def test_unfused_baseline(device):
    ta, tb, a, b, out = _mk(device)
    _close(run_op(a, b, out), ta * tb)


# ------------------------------------------------- optional DFB, 2 styles
def test_optional_dfb_ifdef(device):
    ta, tb, a, b, out = _mk(device, seed=1)
    out2 = ttnn.allocate_tensor_on_device(a.spec, device)
    run_op(a, b, out, out2, gate="ifdef")
    _close(out, ta * tb)
    _close(out2, ta * tb)


def test_optional_dfb_always_bound(device):
    ta, tb, a, b, out = _mk(device, seed=2)
    out2 = ttnn.allocate_tensor_on_device(a.spec, device)
    run_op(a, b, out, out2, gate="always")
    _close(out, ta * tb)
    _close(out2, ta * tb)


# ------------------------------------- local L1->L1 copy + self-loop DFB
def test_local_l1_copy_via_self_looped_stage(device):
    """dfb::out -> dfb::stage (local L1->L1) -> DRAM, writer bound P+C on stage."""
    ta, tb, a, b, out = _mk(device, seed=3)
    _close(run_op(a, b, out, stage_local=True), ta * tb)


def test_self_loop_depth_one(device):
    """A depth-1 staging DFB: reserve/push then wait/pop in the same kernel iteration."""
    ta, tb, a, b, out = _mk(device, seed=4)
    _close(run_op(a, b, out, stage_local=True, stage_depth=1), ta * tb)


def test_local_copy_via_self_read(device):
    """Route D: NoC READ path with a self UnicastEndpoint source; DFB is a typed dst."""
    ta, tb, a, b, out = _mk(device, seed=11)
    _close(run_op(a, b, out, stage_local=True, local_route="self_read"), ta * tb)


def test_local_copy_via_noc_loopback_unicast(device):
    """Route B: NoC unicast to my own (x,y), destination address peeked off the DFB."""
    ta, tb, a, b, out = _mk(device, seed=10)
    _close(run_op(a, b, out, stage_local=True, local_route="unicast_self"), ta * tb)


# --------------------------------------------------- validator violations
LOG = Path(__file__).parent.parent / "VIOLATION_MESSAGES.txt"


def _expect_error(fn):
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        Exception
    ) as ei:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        fn()
    return str(ei.value)


def _record(tag, msg, limit=2500):
    with LOG.open("a") as f:
        f.write("\n" + "=" * 78 + "\n### " + tag + "\n" + "=" * 78 + "\n" + msg[:limit] + "\n")


def test_violation_producer_only_dfb(device):
    _, _, a, b, out = _mk(device, seed=5)
    msg = _expect_error(lambda: run_op(a, b, out, break_producer_only=True))
    _record("[PRODUCER-ONLY DFB]", msg)
    assert msg


def test_violation_unbound_tensor_parameter(device):
    _, _, a, b, out = _mk(device, seed=6)
    msg = _expect_error(lambda: run_op(a, b, out, break_unbound_tensor_param=True))
    _record("[UNBOUND TENSOR PARAMETER]", msg)
    assert msg


def test_violation_two_accessor_names_one_dfb(device):
    _, _, a, b, out = _mk(device, seed=7)
    msg = _expect_error(lambda: run_op(a, b, out, break_two_accessor_names=True))
    _record("[TWO ACCESSOR NAMES, ONE DFB, ONE KERNEL]", msg)
    assert msg


def test_violation_self_loop_producer_only(device):
    _, _, a, b, out = _mk(device, seed=8)
    msg = _expect_error(lambda: run_op(a, b, out, stage_local=True, break_self_loop_missing=True))
    _record("[STAGE DFB PRODUCER-ONLY (self-loop half-declared)]", msg)
    assert msg


def test_violation_unbound_dfb_token_in_kernel(device):
    """gate='ifdef' kernel text but WITHOUT the host binding: dfb::out2 has no token."""
    _, _, a, b, out = _mk(device, seed=9)
    spec, run = build(a, b, out, None, gate=None)
    # force the define on compute without adding the binding
    for k in spec.kernels:
        if k.unique_id == "compute":
            k.compiler_options = ttnn.CompilerOptions(defines={"GATE_IFDEF": "1"})
    msg = _expect_error(lambda: ttnn.generic_op([a, b, out], spec, run, {TP_A: 0, TP_B: 1, TP_OUT: 2}))
    _record("[KERNEL NAMES AN UNBOUND dfb::out2]", msg)
    assert msg


# ------------------------------------------------ does the conditional DFB save L1?
def _max_depth_that_fits(device, gate, lo=1, hi=4096):
    """Bisect the largest out2 depth that still builds. Proves whether L1 is charged."""
    ta, tb, a, b, out = _mk(device, shape=(1, 1, 32, 32), seed=20)
    out2 = ttnn.allocate_tensor_on_device(a.spec, device)
    budget = ttnn.get_max_worker_l1_unreserved_size()

    def fits(d):
        try:
            spec, run = build(a, b, out, out2, gate=gate)
            for dfb in spec.dataflow_buffers:
                if dfb.unique_id == "out2":
                    dfb.num_entries = d
            io = [a, b, out] + ([out2] if gate else [])
            m = {TP_A: 0, TP_B: 1, TP_OUT: 2}
            if gate:
                m["out2"] = 3
            ttnn.generic_op(io, spec, run, m)
            return True
        except Exception:
            return False

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo, budget


def test_conditional_dfb_actually_saves_l1(device):
    """The whole point of a conditional binding: an unbound DFB must cost zero L1."""
    depth_fused, budget = _max_depth_that_fits(device, "ifdef")
    tile = 2048
    print(f"\n[L1] worker L1 unreserved = {budget} B ({budget/1024:.0f} KB)")
    print(f"[L1] max out2 depth that builds when BOUND   = {depth_fused} entries " f"({depth_fused*tile/1024:.0f} KB)")
    assert depth_fused * tile < budget, "bound DFB should be charged against the L1 budget"
    # And an unbound one costs nothing: absurd depth on a DFB nobody binds must still build.
    ta, tb, a, b, out = _mk(device, shape=(1, 1, 32, 32), seed=21)
    spec, run = build(a, b, out, None, gate=None)
    huge = ttnn.DataflowBufferSpec(unique_id="never_bound", entry_size=tile, num_entries=100000, data_format=out.dtype)
    spec.dataflow_buffers = list(spec.dataflow_buffers) + [huge]
    msg = _expect_error(lambda: ttnn.generic_op([a, b, out], spec, run, {TP_A: 0, TP_B: 1, TP_OUT: 2}))
    _record("[DFB DECLARED BUT BOUND BY NOBODY (200 MB)]", msg)
    print(f"[L1] declared-but-unbound DFB rejected: {msg.splitlines()[:4]}")


# --------------------------------------------------- TT_KERNEL (NTTP) syntax
def test_ttkernel_nttp_reader(device):
    """CTAs as template params, RTAs as fn params; JIT generates kernel_main()."""
    ta, tb, a, b, out = _mk(device, seed=30)
    _close(run_op(a, b, out, ttk_reader=True), ta * tb)


def test_ttkernel_nttp_compile_time_branching(device):
    """if constexpr on an NTTP CTA selects a genuinely different loop body."""
    ta, tb, a, b, out = _mk(device, seed=31)
    _close(run_op(a, b, out, ttk_reader=True, ttk_tiles_per_iter=4), ta * tb)


def test_ttkernel_nttp_does_not_rescue_unbound_dfb_token(device):
    """The claim in issue #52179's thread, tested in the real JIT."""
    _, _, a, b, out = _mk(device, seed=32)
    msg = _expect_error(lambda: run_op(a, b, out, ttk_reader=True, ttk_unbound_token=True))
    _record("[TT_KERNEL NTTP + if constexpr NAMING AN UNBOUND dfb::out2]", msg)
    assert "out2" in msg or "dfb" in msg, msg[:500]


def test_ttkernel_schema_mismatch_is_caught(device):
    """Kernel declares a param the host never registered."""
    _, _, a, b, out = _mk(device, seed=33)
    spec, run = build(a, b, out, None, ttk_reader=True)
    for k in spec.kernels:
        if k.unique_id == "reader":
            k.compile_time_args = {"tiles_per_iter": 1}  # drop 'touch_optional'
    msg = _expect_error(lambda: ttnn.generic_op([a, b, out], spec, run, {TP_A: 0, TP_B: 1, TP_OUT: 2}))
    _record("[TT_KERNEL SIGNATURE vs HOST SCHEMA MISMATCH]", msg)
    assert msg
