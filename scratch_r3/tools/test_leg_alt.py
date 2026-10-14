# #58817 legacy port, in-process alternation: one legacy bmm source (variant "alt") compiled per LEG_VARIANT (0 and 4
# neither opt-in, 1 both, 2 the row MOP only, 3 Auto TTSync only); the scratch host adds the define from
# TT_MM_LEG_VARIANT when it builds the program, and the program cache is off so every launch builds for the variant set
# at that moment. Per round each variant runs ALT_CALLS launches behind a signpost "ALT|case|variant|round"; the order
# rotates and reverses every round. alt_reduce.py pairs the variants by round.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
_os.environ["TT_MM_LEGACY_1D"] = "1"
import pytest
import ttnn

from test_leg_prof import BF, CASES, agmm_setup

ROUNDS = int(_os.environ.get("ALT_ROUNDS", "11"))
CALLS = int(_os.environ.get("ALT_CALLS", "8"))
VARIANTS = [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.disable_and_clear_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_leg_alt(device, case):
    from tracy import signpost

    name, dim, wdt, fid, approx = case
    a, b, pc, ckc, out_mc = agmm_setup(device, dim, wdt, fid, approx)
    for r in range(ROUNDS):
        order = VARIANTS[r % 5 :] + VARIANTS[: r % 5]
        if r % 2:
            order = order[::-1]
        for v in order:
            _os.environ["TT_MM_LEG_VARIANT"] = str(v)
            ttnn.synchronize_device(device)
            signpost(header=f"ALT|{name}|{v}|{r}")
            for _ in range(CALLS):
                out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc, memory_config=out_mc, dtype=BF)
                out.deallocate()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    _os.environ.pop("TT_MM_LEG_VARIANT", None)
    a.deallocate()
    b.deallocate()
