# k2_soak: QWEN38_SDPA_KSPLIT (key_splits=8, rows 32 / 17 / 9 = 3 / 2 / 1 turns), 20 calls per trace, 1000 replays; q and the
# index rows are rewritten every 10 replays.  Temporary: delete with models/.../tests/soak_common.py.
import importlib.util
import os

import pytest
import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ks = _load("ksplit_turns_for_soak", os.path.join(HERE, "test_sparse_sdpa_ksplit_turns.py"))
sc = _load("soak_common_for_soak", os.path.join(ROOT, "models/demos/blackhole/qwen38_flash_next/tests/soak_common.py"))
N = 8


@pytest.fixture(scope="module")
def soak_device():
    dev = ttnn.open_device(device_id=0, trace_region_size=1 << 27)
    yield dev
    ttnn.close_device(dev)


def vary(rows, nvs, seed):
    """q [1,32,rows,512] (only the second half non-zero, as make_lanes) and idx [1,1,rows,TOPK] (-1 tail) for seed."""
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(1, 32, rows, 2 * ks.HEAD_DIM)
    q[..., ks.HEAD_DIM :] = torch.randn(1, 32, rows, ks.HEAD_DIM, generator=g)
    idx = torch.full((1, 1, rows, ks.TOPK), -1, dtype=torch.int32)
    for t in range(rows):
        idx[0, 0, t, : nvs[t]] = torch.randperm(ks.T, generator=g)[: nvs[t]].to(torch.int32)
    return q.to(torch.bfloat16), idx


@pytest.mark.parametrize("rows,kind", [(32, "mixed"), (32, "full"), (17, "mixed"), (9, "mixed")], ids=lambda v: str(v))
def test_soak_ksplit_n8(soak_device, rows, kind):
    dev = soak_device
    nvs = (2055,) * 32 if kind == "full" else ks.NV32
    c = ks.Lanes(dev, nvs, seed=4321)
    nv = nvs[:rows]
    variants = [vary(rows, nv, 600 + v) for v in range(sc.VARIANTS)]
    tq = ttnn.from_torch(variants[0][0], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    tidx = ttnn.from_torch(variants[0][1], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    hosts = [
        (
            sc.host_tensor(q, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            sc.host_tensor(i, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        )
        for q, i in variants
    ]
    box = {}

    def call(keep):
        out = c.call(tq, tidx, N, "model")
        if keep:
            box["out"] = out
        else:
            ttnn.deallocate(out)

    def prepare(i):
        hq, hi = hosts[i % sc.VARIANTS]
        sc.copy_in(hq, tq)
        sc.copy_in(hi, tidx)

    def observe():
        return sc.dies(box["out"])

    sc.soak(dev, f"ksplit_n8_{kind}", rows, call, prepare, observe)
