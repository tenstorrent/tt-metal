# #58817 fork Auto TTSync audit, in-process alternation: the fork bmm source compiled per FORK_VARIANT (0 and 4 neither
# opt-in, 1 the head's rules, 2 the row MOP only, 3 Auto TTSync only); the scratch host adds the define from
# TT_MM_FORK_VARIANT when it builds the program, and the program cache is off. Cases: fork_decode_cases.py.
# Under BITID_OUT (bits runs) one round of one launch per variant, outputs saved per variant.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

import fork_decode_cases as fdc

BITS = _os.environ.get("BITID_OUT")
ROUNDS = 1 if BITS else int(_os.environ.get("ALT_ROUNDS", "11"))
CALLS = 1 if BITS else int(_os.environ.get("ALT_CALLS", "8"))
VARIANTS = [int(x) for x in _os.environ.get("ALT_VARIANTS", "0,1,2,3,4").split(",")]
GROUPS = [m[0] for m in fdc.TT_MODELS] + ["q36_27b_p150x4"]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.disable_and_clear_program_cache()
    yield dev
    ttnn.close_device(dev)


def _note(line):
    p = _os.environ.get("FORK_CASES_OUT")
    if p:
        with open(p, "a") as f:
            f.write(line + "\n")


@pytest.mark.parametrize("group", GROUPS)
def test_fork_decode(device, group):
    from tracy import signpost

    if group.startswith("q36"):
        cases = fdc.qwen36_cases(device)
    else:
        prefix, hf, nd, prec = next(m for m in fdc.TT_MODELS if m[0] == group)
        cases = fdc.tt_model_cases(device, prefix, hf, nd, prec)
    for name, build, info in cases:
        _note(f"{name}: k {info[0]} n {info[1]} | {info[2]}")
        x, w, kw = build()
        for r in range(ROUNDS):
            k = r % len(VARIANTS)
            order = VARIANTS[k:] + VARIANTS[:k]
            if r % 2:
                order = order[::-1]
            for v in order:
                _os.environ["TT_MM_FORK_VARIANT"] = str(v)
                ttnn.synchronize_device(device)
                signpost(header=f"ALT|{name}|{v}|{r}")
                for _ in range(CALLS):
                    out = ttnn.linear(x, w, **kw)
                    if BITS:
                        t = ttnn.to_torch(out).contiguous()
                        torch.save(t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32), _os.path.join(BITS, f"fd_{name}_v{v}.pt"))
                    out.deallocate()
            ttnn.synchronize_device(device)
            if not BITS:
                ttnn.ReadDeviceProfiler(device)
        _os.environ.pop("TT_MM_FORK_VARIANT", None)
        x.deallocate()
        w.deallocate()
