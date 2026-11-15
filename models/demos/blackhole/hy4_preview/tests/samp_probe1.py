"""device-sampling probe 1 (4x8 mesh): (A) integer RNG ops (xor / shifts / multiply wrap on uint32), (B) shared TTSampling on the Hy4 layout (sharded vocab 8 x 15104, sampling_dp=4):
greedy / top-k 20 / top-k 32 vs the exact law on real rows, slot independence of a seeded draw, per-call time (eager + traced)."""
import json
import os
import statistics
import time
import types

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

V, COLS, ROWS = 120832, 8, 4


def res(**kw):
    print("RES " + json.dumps(kw, default=str), flush=True)


def mix_np(x):  # lowbias32
    x = x.astype("uint64") & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def xs_np(x):  # xorshift32
    x = x.astype("uint64") & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def probe_int(md):
    import numpy as np

    rng = np.random.default_rng(1)
    vals = rng.integers(0, 2**32, size=(1, 1, 32, 32), dtype=np.uint64)
    t = torch.from_numpy(vals.astype(np.int64)).to(torch.int64)

    def up(dt):
        # uint32 via int64 -> torch has no uint32 math: go through int32 view
        x = torch.from_numpy(vals.astype(np.uint32).view(np.int32).copy())
        return ttnn.from_torch(
            x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
        )

    def dn(y):
        z = ttnn.to_torch(ttnn.get_device_tensors(y)[0])
        return z.numpy().astype(np.int64) & 0xFFFFFFFF

    def attempt(name, fn, ref):
        try:
            out = dn(fn())
            ok = bool((out == ref).all())
            res(event="int_op", op=name, ok=ok, mismatches=int((out != ref).sum()))
        except Exception as e:  # noqa: BLE001
            res(event="int_op", op=name, ok=False, error=str(e)[:200])

    for dt_name, dt in (("uint32", ttnn.uint32), ("int32", ttnn.int32)):
        a = up(dt)
        v = vals.astype(np.int64)
        attempt(
            f"{dt_name} xor const",
            lambda: ttnn.bitwise_xor(a, 0x5BD1E995 if dt_name == "uint32" else 0x1BD1E995),
            v ^ (0x5BD1E995 if dt_name == "uint32" else 0x1BD1E995),
        )
        attempt(f"{dt_name} left_shift 13", lambda: ttnn.bitwise_left_shift(a, 13), (v << 13) & 0xFFFFFFFF)
        attempt(f"{dt_name} logical_right_shift 17", lambda: ttnn.logical_right_shift(a, 17), v >> 17)
        attempt(
            f"{dt_name} xor tensor",
            lambda: ttnn.bitwise_xor(a, ttnn.bitwise_left_shift(a, 13)),
            (v ^ ((v << 13) & 0xFFFFFFFF)),
        )
        attempt(
            f"{dt_name} mul const 0x7feb352d (wrap)",
            lambda: ttnn.multiply(a, 0x7FEB352D),
            (v * 0x7FEB352D) & 0xFFFFFFFF,
        )
        attempt(
            f"{dt_name} add const 0x9e3779b9 (wrap)",
            lambda: ttnn.add(a, 0x9E3779B9 if dt_name == "uint32" else 0x1E3779B9),
            (v + (0x9E3779B9 if dt_name == "uint32" else 0x1E3779B9)) & 0xFFFFFFFF,
        )
    a = up(ttnn.uint32)

    def xs():
        x = a
        x = ttnn.bitwise_xor(x, ttnn.bitwise_left_shift(x, 13))
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 17))
        x = ttnn.bitwise_xor(x, ttnn.bitwise_left_shift(x, 5))
        return x

    attempt("xorshift32 (xor/shift only)", xs, xs_np(vals.copy()).astype(np.int64))

    def mx():
        x = a
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))
        x = ttnn.multiply(x, 0x7FEB352D)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 15))
        x = ttnn.multiply(x, 0x846CA68B)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))
        return x

    attempt("lowbias32 full hash", mx, mix_np(vals.copy()).astype(np.int64))


def exact_p(l, T, k=0):
    s = l.double() / T
    s = s - s.max()
    e = s.exp()
    if k > 0:
        thr = torch.topk(s, k).values[-1]
        e = e * (s >= thr)
    return e / e.sum()


def tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def probe_tt(md):
    from models.common.sampling import SamplingParams
    from models.common.sampling.generator import format_sampling_params
    from models.common.sampling.tt_sampling import TTSampling
    from models.demos.gpt_oss.tt.ccl import CCLManager

    d = torch.load("/lmhead/final_ref.pt", map_location="cpu")
    L = d["sub_logits"]
    ent = []
    for i in range(0, len(L), 7):
        s = L[i].double() / 0.9
        p = torch.softmax(s, 0)
        ent.append((-(p * (p + 1e-30).log()).sum().item(), i))
    ent.sort()
    rows = {"peaked": ent[len(ent) // 2][1], "mid": ent[int(len(ent) * 0.93)][1], "flat": ent[-1][1]}
    res(event="rows", rows=rows, entropy={k: round([e for e, i in ent if i == v][0], 3) for k, v in rows.items()})
    ccl = CCLManager(md, num_links=1, topology=ttnn.Topology.Ring)
    args = types.SimpleNamespace(
        vocab_size=V,
        padded_vocab_size=V,
        cluster_shape=(ROWS, COLS),
        max_batch_size=32,
        max_top_k=32,
        sampling_dp=ROWS,
        sampling_all_gather_axis=1,
        sub_core_grids=None,
    )
    smp = TTSampling(md, ccl, args)
    B = 32 * ROWS
    mapper = ttnn.ShardTensor2dMesh(md, dims=(0, 3), mesh_shape=(ROWS, COLS))

    def logits_tensor(row_rows):  # row_rows: 4 real logits rows [V] (one per mesh row), replicated to the 32 tile rows
        x = torch.stack([r.float().reshape(1, 1, V).repeat(1, 32, 1) for r in row_rows]).reshape(ROWS, 1, 32, V)
        return ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def set_params(T, k, p=1.0):
        sp = format_sampling_params(SamplingParams(temperature=[T] * B, top_k=[k] * B, top_p=[p] * B), B)
        smp.reset_params(torch.tensor(sp.top_k), torch.tensor(sp.top_p), torch.tensor(sp.temperature))

    def set_seeds(vals):
        h = ttnn.from_torch(
            torch.as_tensor(vals, dtype=torch.int64).to(torch.int32).reshape(-1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )
        ttnn.copy_host_to_device_tensor(h, smp.seeds_tt_tensor)

    def read_tok(o):
        t = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=(ROWS, COLS)))
        return t.reshape(ROWS, COLS, -1)[:, 0, :32].long()  # [rows, 32] (first column's copy)

    for name, ri in rows.items():
        lt = logits_tensor([L[ri]] * ROWS)
        for T, k in ((0.9, 20), (0.9, 32), (0.9, 0), (1.0, 1)):
            set_params(T, k)
            toks = []
            g = torch.Generator().manual_seed(5)
            t0 = time.perf_counter()
            for it in range(48):
                set_seeds(torch.randint(1, 2**31 - 1, (B,), generator=g))
                o, _ = smp(lt)
                toks.append(read_tok(o))
            dt = (time.perf_counter() - t0) / 48 * 1e3
            tk = torch.cat([t.flatten() for t in toks])
            cnt = torch.bincount(tk, minlength=V).double()
            emp = cnt / cnt.sum()
            ex = exact_p(L[ri], T, k if 0 < k <= 32 else 32)  # the shared law's SET for k off is the top-32
            exfull = exact_p(L[ri], T, 0 if k == 0 else k)
            ref = torch.multinomial(exfull, tk.numel(), replacement=True, generator=g)
            noise = tv(exfull, torch.bincount(ref, minlength=V).double() / tk.numel())
            greedy_ok = bool((tk == L[ri].argmax()).all()) if k == 1 else None
            res(
                event="tt_dist",
                row=name,
                T=T,
                k=k,
                n=int(tk.numel()),
                tv_vs_exact_law=round(tv(exfull, emp), 4),
                tv_vs_top32_law=round(tv(ex, emp), 4),
                noise_tv_exact_sampler=round(noise, 4),
                greedy_all_equal_fp32_argmax=greedy_ok,
                eager_ms_per_call_incl_seed_upload=round(dt, 2),
            )
    # slot independence: same seed in all 128 users, same logits
    lt = logits_tensor([L[rows["mid"]]] * ROWS)
    set_params(0.9, 20)
    set_seeds([12345] * B)
    o, _ = smp(lt)
    t1 = read_tok(o)
    res(event="seed_slot_independence", distinct_tokens_with_one_seed_in_128_slots=int(len(set(t1.flatten().tolist()))))
    set_seeds([12345] * B)
    o, _ = smp(lt)
    res(event="seed_repeat", same=bool((read_tok(o) == t1).all()))
    # trace timing of the sampler body (params already set; seeds fixed)
    set_seeds(list(range(1, B + 1)))
    o, _ = smp(lt)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    o, _ = smp(lt)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    v = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        v.append((time.perf_counter() - t0) / 20 * 1e3)
    res(event="tt_trace_ms", med=round(statistics.median(v), 3), min=round(min(v), 3))


def main():
    md = open_mesh(trace=200_000_000)
    try:
        for fn in (probe_int, probe_tt):
            if os.environ.get("ONLY") and os.environ["ONLY"] != fn.__name__:
                continue
            try:
                fn(md)
            except Exception as e:  # noqa: BLE001
                import traceback

                res(event="probe_error", fn=fn.__name__, error=repr(e)[:300], tb=traceback.format_exc()[-1500:])
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
