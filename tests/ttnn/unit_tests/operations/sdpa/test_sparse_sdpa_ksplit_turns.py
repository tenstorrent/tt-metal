# Op-level tests for the key-split SparseSDPA with units > cores (ttnn.transformer.sparse_sdpa key_splits=N, S*N may
# exceed the grid: whole groups run in turns, epoch handshake between turns).  Adapted from
# qwen38_work/runs/plan_nov04/i1_sdpa/test_sparse_sdpa_ksplit.py.  Device tests; each fits `timeout 600`.
# A lane's output must depend on (N, that lane's data) only: bitwise equal for rows 1, 8, 32 at fixed N.
# Not covered: fp32 partials (QWEN38_SDPA_KSPLIT_FP32 is not implemented, see CODE_REPORT.md).
import os
import time

import pytest
import torch

import ttnn

T = 32768
TOPK = 2080
HEAD_DIM = 256
RESULTS = os.environ.get("SDPA_RESULTS", "/home/ttuser/atupe/qwen38_work/runs/plan_nov05/k1_ksplit/turns_results.txt")
# 32 lanes, mixed nv: boundary values (1, 31, 32, 33), partial chunks, full 2055, and repeats at different lanes.
NV32 = (
    1, 31, 32, 33, 128, 2055, 500, 64, 2055, 17, 1000, 2055, 3, 65, 96, 1500,
    2055, 1, 2055, 700, 33, 2000, 250, 2055, 5, 1024, 2055, 31, 2055, 129, 2054, 2055,
)  # fmt: skip
assert len(NV32) == 32


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=1 << 24)
    yield dev
    ttnn.close_device(dev)


def make_cfg(name, device):
    if name == "model":  # what qsa.py uses: HiFi4, exact exp, fp32 dest
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def make_lanes(nvs, seed=4321):
    """q [1,32,L,512], kv [1,1,T,512], idx [1,1,L,TOPK] int32 (-1 tail) for L = len(nvs) lanes (one nv per lane)."""
    L = len(nvs)
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(1, 32, L, 2 * HEAD_DIM)
    q[..., HEAD_DIM:] = torch.randn(1, 32, L, HEAD_DIM, generator=g)
    q = q.to(torch.bfloat16)
    kv = torch.randn(1, 1, T, 2 * HEAD_DIM, generator=g).to(torch.bfloat16)
    idx = torch.full((1, 1, L, TOPK), -1, dtype=torch.int32)
    for t, nv in enumerate(nvs):
        idx[0, 0, t, :nv] = torch.randperm(T, generator=g)[:nv].to(torch.int32)
    return q, kv, idx


def reference(q, kv, idx):
    L = q.shape[2]
    out = torch.zeros(1, 32, L, HEAD_DIM)
    for t in range(L):
        rows = idx[0, 0, t][idx[0, 0, t] >= 0].long()
        k = kv[0, 0, rows].float()
        s = (q[0, :, t, :].float() @ k.T) * (HEAD_DIM**-0.5)
        out[0, :, t, :] = torch.softmax(s, dim=-1) @ k[:, :HEAD_DIM]
    return out


class Lanes:
    """One resident case (kv uploaded once); lane subsets are uploaded per call."""

    def __init__(self, device, nvs, seed=4321):
        self.device = device
        self.q, self.kv, self.idx = make_lanes(nvs, seed)
        self.tkv = ttnn.from_torch(self.kv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self._ref = None

    def ref(self):
        if self._ref is None:
            self._ref = reference(self.q, self.kv, self.idx)
        return self._ref

    def upload(self, lanes):
        lanes = list(lanes)
        q = self.q[:, :, lanes].contiguous()
        idx = self.idx[:, :, lanes].contiguous()
        d = self.device
        return (
            ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=d),
            ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d),
        )

    def call(self, tq, tidx, N, cfg_name):
        kw = {"key_splits": N} if N > 1 else {}
        return ttnn.transformer.sparse_sdpa(
            tq,
            self.tkv,
            tidx,
            HEAD_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=HEAD_DIM**-0.5,
            k_chunk_size=32,
            compute_kernel_config=make_cfg(cfg_name, self.device),
            **kw,
        )

    def run(self, lanes, N, cfg_name="model"):
        tq, tidx = self.upload(lanes)
        out = self.call(tq, tidx, N, cfg_name)
        ttnn.synchronize_device(self.device)
        return ttnn.to_torch(out)  # [1,32,len(lanes),HEAD_DIM]


_CASES = {}


def lanes_case(device, nvs, seed=4321):
    key = (tuple(nvs), seed)
    if key not in _CASES:
        _CASES.clear()  # keep one case resident
        _CASES[key] = Lanes(device, nvs, seed)
    return _CASES[key]


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def log(line):
    print(line)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


# (a) bitwise row invariance ------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg_name", ["model", "hifi2"])
@pytest.mark.parametrize("N", [4, 8, 32])
def test_ksplit_row_invariant(device, N, cfg_name):
    c = lanes_case(device, NV32)
    out32 = c.run(range(32), N, cfg_name)  # rows 32: N=4 2 turns, N=8 3 turns, N=32 11 turns
    out8 = c.run(range(8), N, cfg_name)  # rows 8: N=4/8 one turn, N=32 3 turns
    bad = []
    if not same(out8, out32[:, :, :8]):
        bad.append("rows8 vs rows32 lanes 0-7")
    for lane in (0, 5, 17, 31):  # rows 1 (also lanes from the middle / last turn)
        o1 = c.run([lane], N, cfg_name)
        if not same(o1, out32[:, :, lane : lane + 1]):
            bad.append(f"rows1 lane {lane} vs rows32")
        if lane < 8 and not same(o1, out8[:, :, lane : lane + 1]):
            bad.append(f"rows1 lane {lane} vs rows8")
    log(f"row invariance N={N} {cfg_name}: {'BITWISE EQUAL' if not bad else 'DIFFERENT ' + str(bad)}")
    assert not bad
    ref = c.ref()
    log(f"  rows32 N={N} {cfg_name} vs torch fp32: PCC={pcc(out32, ref):.6f}")


# (b) 11 turns, repeated launches of the cached program ----------------------------------------------------------
def test_ksplit_n32_rows32_repeat(device):
    c = lanes_case(device, NV32)
    tq, tidx = c.upload(range(32))
    first = None
    for i in range(20):
        out = c.call(tq, tidx, 32, "model")
        o = ttnn.to_torch(out)
        ttnn.deallocate(out)
        if first is None:
            first = o
        assert same(o, first), f"launch {i} differs from launch 0 (stale epoch / semaphore state?)"
    log("N=32 rows=32 (11 turns) x20 cached launches: BITWISE EQUAL")
    assert same(c.run([17], 32), first[:, :, 17:18])


# (c) skewed nv: fast senders race ahead of slow parents -----------------------------------------------------------
SKEWS = {
    "alt": tuple(1 if t % 2 == 0 else 2055 for t in range(32)),
    "half": tuple(1 if t < 16 else 2055 for t in range(32)),
    "rare_slow": tuple(2055 if t % 8 == 3 else 1 for t in range(32)),
}


@pytest.mark.parametrize("name", list(SKEWS))
@pytest.mark.parametrize("N", [8, 16])
def test_ksplit_skewed_nv(device, name, N):
    c = lanes_case(device, SKEWS[name], seed=99)
    out32 = c.run(range(32), N)
    for _ in range(3):
        assert same(c.run(range(32), N), out32), "rows32 launch differs"
    for lane in (0, 3, 16, 19, 31):
        assert same(c.run([lane], N), out32[:, :, lane : lane + 1]), f"lane {lane} differs from rows=1"
    log(f"skewed nv {name} N={N}: BITWISE EQUAL (rows32 x4, lanes 0,3,16,19,31 vs rows 1)")


# (d) accuracy: PCC vs N=1 and vs torch fp32 ------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_ksplit_accuracy(device, N):
    c = lanes_case(device, NV32)
    ref = c.ref()
    base = c.run(range(32), 1)
    out = c.run(range(32), N)
    p1, pr, pb = pcc(out, base), pcc(out, ref), pcc(base, ref)
    d = float((out.float() - base.float()).abs().max())
    log(f"N={N} rows=32 fp32partials=0: PCC vs N=1 {p1:.7f} maxabs {d:.5f}; vs torch fp32 {pr:.6f} (N=1: {pb:.6f})")
    assert p1 > 0.9995
    assert pr > pb - 2e-4


# (e) traced timing ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_ksplit_timing(device, rows):
    nv2055 = lanes_case(device, (2055,) * 32, seed=5)  # full-length lanes: the worst case
    tq, tidx = nv2055.upload(range(rows))
    for N in (1, 4, 8, 16, 32):
        out = nv2055.call(tq, tidx, N, "model")  # compile
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = nv2055.call(tq, tidx, N, "model")
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(device, tid)
        log(f"traced rows={rows} N={N} fp32partials=0 nv=2055: {us:.1f} us/call")
        ttnn.deallocate(out)


# (f) traced timing, short contexts (ISL 128-512: nv keys = 64..512, i.e. 2..16 key chunks of 32) --------------------
@pytest.mark.parametrize("nv", [64, 130, 260, 512])
@pytest.mark.parametrize("rows", [8, 32])
def test_ksplit_timing_short(device, rows, nv):
    case = lanes_case(device, (nv,) * 32, seed=5)  # all lanes the same short nv (n_active = ceil(nv / 32) chunks)
    tq, tidx = case.upload(range(rows))
    for N in (1, 8):
        out = case.call(tq, tidx, N, "model")  # compile
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = case.call(tq, tidx, N, "model")
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) / n * 1e6
        ttnn.release_trace(device, tid)
        log(f"traced rows={rows} N={N} fp32partials=0 nv={nv}: {us:.1f} us/call")
        ttnn.deallocate(out)


# (g) short lanes (n_active <= SPARSE_KSPLIT_MIN_CHUNKS=8 -> n_eff=1): KSPLIT N=8 is bitwise the N=1 path ----------------
@pytest.mark.parametrize("nv", [1, 64, 130, 256])
@pytest.mark.parametrize("rows", [8, 32])
def test_ksplit_short_equals_n1(device, rows, nv):
    case = lanes_case(device, (nv,) * 32, seed=5)
    o1 = case.run(range(rows), 1)
    o8 = case.run(range(rows), 8)
    bad = [i for i in range(rows) if not same(o1[:, :, i : i + 1], o8[:, :, i : i + 1])]
    log(f"short bitwise rows={rows} nv={nv}: N=8 vs N=1 {'EQUAL' if not bad else 'DIFFERENT lanes ' + str(bad)}")
    assert not bad
