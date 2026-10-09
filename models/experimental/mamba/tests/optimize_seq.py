#!/usr/bin/env python3
"""Ten sequential optimization attempts for each op in the std-vs-PyTorch table.

Each attempt is measured on its own. A faster attempt replaces the current best
only when PCC against the PyTorch reference stays at least 0.99. The timed
region matches std_vs_pytorch.py: expanded tensors and weights are prepared
before the timer.
"""

from __future__ import annotations

import importlib.util
import json
import time
import traceback
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "optimize_seq.jsonl"
PCC_MIN = 0.99
SMALL_ITERS = 10
HEAVY_ITERS = 3

TABLE = {
    "std repeat + multiply (B*dt)": (1.0007, 0.126),
    "custom repeat_and_interleave_eltwise_mul": (1.0007, 0.080),
    "std reshape + sum": (1.0000, 0.310),
    "custom hc_sum_reduce": (1.0000, 0.100),
    "std mul/add scan y vs pytorch": (1.0000, 23.806),
    "std mixer out vs pytorch slow_forward": (0.9999, 18.228),
}


def load_std():
    spec = importlib.util.spec_from_file_location("std_vs_pytorch", HERE / "std_vs_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pcc(actual, expected) -> float:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 1.0
    return float((a * b).sum() / denom)


def on_device(tensor, device, dtype=ttnn.bfloat16, memory=None):
    return ttnn.from_torch(
        tensor.contiguous().float(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
    )


def move(tensor, memory):
    if tensor.memory_config() == memory:
        return tensor
    return ttnn.to_memory_config(tensor, memory)


def kernel_config(device, fidelity, approx=False, fp32=False):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32,
    )


def short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".split("\n")[0]
    return text[:240]


def measure(device, fn, expected, iters):
    out = fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    ttnn.synchronize_device(device)
    score = pcc(ttnn.to_torch(out).float(), expected.float())
    return (time.perf_counter() - start) / iters * 1e3, score


def run_attempts(device, op, attempts, expected, iters):
    rows = []
    best = None
    print(f"\n== {op}  ({len(attempts)} attempts)", flush=True)
    for index, (name, fn) in enumerate(attempts, 1):
        row = {"op": op, "attempt": index, "name": name}
        try:
            ms, score = measure(device, fn, expected, iters)
            row["ms"] = ms
            row["pcc"] = score
            keep = score >= PCC_MIN and (best is None or ms < best["ms"])
            row["kept"] = keep
            if keep:
                best = row
            mark = "KEEP" if keep else ("ok" if score >= PCC_MIN else "drop")
            print(f"  {index:2} {mark:4} {name:42} pcc={score:.4f}  {ms:8.3f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad attempt must not stop the sequence
            row["error"] = short_error(exc)
            row["kept"] = False
            print(f"  {index:2} FAIL {name:42} {row['error']}", flush=True)
            traceback.print_exc()
        with RESULTS.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        rows.append(row)
    return best, rows


def mul_kwargs(memory, dtype, approx, output=None):
    kwargs = {
        "memory_config": memory,
        "dtype": dtype,
        "fast_and_approximate_mode": approx,
    }
    if output is not None:
        kwargs["output_tensor"] = output
    return kwargs


def std_repeat_attempts(device, a_rep, b_rep):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    bf16, bf8 = ttnn.bfloat16, ttnn.bfloat8_b
    a_l1 = move(a_rep, l1)
    b_l1 = move(b_rep, l1)
    slot_dram = ttnn.multiply(a_rep, b_rep, memory_config=dram)
    slot_l1 = ttnn.multiply(a_rep, b_rep, memory_config=l1)
    return [
        ("DRAM bf16", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(dram, bf16, False))),
        ("L1 output", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(l1, bf16, False))),
        ("fast approx", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(dram, bf16, True))),
        ("bf8 output", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(dram, bf8, False))),
        ("bf8 output + approx", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(dram, bf8, True))),
        ("L1 inputs", lambda: ttnn.multiply(a_l1, b_l1, **mul_kwargs(dram, bf16, False))),
        ("L1 inputs and output", lambda: ttnn.multiply(a_l1, b_l1, **mul_kwargs(l1, bf16, False))),
        ("reuse DRAM output", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(dram, bf16, False, slot_dram))),
        ("reuse L1 output", lambda: ttnn.multiply(a_rep, b_rep, **mul_kwargs(l1, bf16, False, slot_l1))),
        ("L1 reuse + approx", lambda: ttnn.multiply(a_l1, b_l1, **mul_kwargs(l1, bf16, True, slot_l1))),
    ]


def custom_repeat_attempts(device, at, bt):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    bf16, bf8 = ttnn.bfloat16, ttnn.bfloat8_b
    fids = [
        ("HiFi2", ttnn.MathFidelity.HiFi2),
        ("LoFi", ttnn.MathFidelity.LoFi),
        ("HiFi4", ttnn.MathFidelity.HiFi4),
        ("HiFi3", ttnn.MathFidelity.HiFi3),
    ]
    attempts = []
    for mem_name, mem in (("DRAM", dram), ("L1", l1)):
        for dtype_name, dtype in (("bf16", bf16), ("bf8", bf8)):
            for fid_name, fid in fids:
                if len(attempts) == 10:
                    return attempts
                attempts.append(
                    (
                        f"{mem_name} {dtype_name} {fid_name}",
                        lambda mem=mem, dtype=dtype, fid=fid: ttnn.experimental.repeat_and_interleave_eltwise_mul(
                            at, bt, memory_config=mem, dtype=dtype, math_fidelity=fid
                        ),
                    )
                )
    return attempts


def std_sum_attempts(device, xt, e):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    shaped5 = ttnn.reshape(xt, (1, 1, 32, e, 32))
    shaped4 = ttnn.reshape(xt, (1, 32, e, 32))
    wide = ttnn.reshape(xt, (1, 1, 32 * e, 32))
    xt_l1 = move(xt, l1)
    shaped5_l1 = ttnn.reshape(xt_l1, (1, 1, 32, e, 32))
    slot = ttnn.sum(shaped5, dim=-1, memory_config=dram)

    def summed(tensor, memory, output=None):
        kwargs = {"dim": -1, "memory_config": memory}
        if output is not None:
            kwargs["output_tensor"] = output
        return ttnn.sum(tensor, **kwargs)

    return [
        ("5D sum DRAM", lambda: summed(shaped5, dram)),
        ("5D sum L1", lambda: summed(shaped5, l1)),
        ("4D sum DRAM", lambda: summed(shaped4, dram)),
        ("4D sum L1", lambda: summed(ttnn.reshape(xt, (1, 32, e, 32)), l1)),
        ("wide 4D sum DRAM", lambda: summed(wide, dram)),
        ("bf8 input 5D DRAM", lambda: summed(ttnn.reshape(ttnn.typecast(xt, ttnn.bfloat8_b), (1, 1, 32, e, 32)), dram)),
        ("reuse DRAM output", lambda: summed(shaped5, dram, slot)),
        ("L1 input 5D", lambda: summed(shaped5_l1, dram)),
        ("L1 input and output", lambda: summed(shaped5_l1, l1)),
        ("wide sum L1", lambda: summed(ttnn.reshape(xt_l1, (1, 1, 32 * e, 32)), l1)),
    ]


def custom_sum_attempts(device, xt):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    pairs = [
        ("DRAM bf16 HiFi2", dram, ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
        ("L1 bf16 HiFi2", l1, ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
        ("DRAM bf16 LoFi", dram, ttnn.bfloat16, ttnn.MathFidelity.LoFi),
        ("DRAM bf16 HiFi4", dram, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),
        ("DRAM bf8 HiFi2", dram, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        ("L1 bf8 HiFi2", l1, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        ("L1 bf16 LoFi", l1, ttnn.bfloat16, ttnn.MathFidelity.LoFi),
        ("DRAM bf8 LoFi", dram, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        ("DRAM bf16 HiFi3", dram, ttnn.bfloat16, ttnn.MathFidelity.HiFi3),
        ("L1 bf8 LoFi", l1, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
    ]
    return [
        (
            name,
            lambda mem=mem, dtype=dtype, fid=fid: ttnn.experimental.hc_sum_reduce(
                xt, memory_config=mem, dtype=dtype, math_fidelity=fid
            ),
        )
        for name, mem, dtype, fid in pairs
    ]


def scan_once(a_steps, bu_steps, c_steps, h, e, mem, dtype, approx):
    ys = []
    for i in range(len(a_steps)):
        h = ttnn.multiply(a_steps[i], h, memory_config=mem, dtype=dtype, fast_and_approximate_mode=approx)
        h = ttnn.add(bu_steps[i], h, memory_config=mem, dtype=dtype)
        proj = ttnn.multiply(h, c_steps[i], memory_config=mem, dtype=dtype, fast_and_approximate_mode=approx)
        ys.append(ttnn.sum(proj, dim=-1, memory_config=mem))
    return ttnn.concat([ttnn.reshape(y, (1, 1, 1, e)) for y in ys], dim=2), h


def persistent_scan(a_steps, bu_steps, c_steps, zeros, e, mem, dtype, approx):
    h_a = ttnn.add(zeros, zeros, memory_config=mem, dtype=dtype)
    h_b = ttnn.add(zeros, zeros, memory_config=mem, dtype=dtype)
    proj = ttnn.add(zeros, zeros, memory_config=mem, dtype=dtype)

    def fn():
        ttnn.add(zeros, zeros, memory_config=mem, dtype=dtype, output_tensor=h_a)
        ys = []
        for i in range(len(a_steps)):
            ttnn.multiply(
                a_steps[i],
                h_a,
                memory_config=mem,
                dtype=dtype,
                fast_and_approximate_mode=approx,
                output_tensor=h_b,
            )
            ttnn.add(bu_steps[i], h_b, memory_config=mem, dtype=dtype, output_tensor=h_a)
            ttnn.multiply(
                h_a,
                c_steps[i],
                memory_config=mem,
                dtype=dtype,
                fast_and_approximate_mode=approx,
                output_tensor=proj,
            )
            ys.append(ttnn.sum(proj, dim=-1, memory_config=mem))
        return ttnn.concat([ttnn.reshape(y, (1, 1, 1, e)) for y in ys], dim=2)

    return fn


def relocate(steps, memory, dtype):
    moved = []
    for tensor in steps:
        current = tensor if tensor.dtype == dtype else ttnn.typecast(tensor, dtype)
        moved.append(move(current, memory))
    return moved


def std_scan_attempts(device, a_steps, bu_steps, c_steps, zeros, e):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    bf16, bf8 = ttnn.bfloat16, ttnn.bfloat8_b
    a_l1 = relocate(a_steps, l1, bf16)
    bu_l1 = relocate(bu_steps, l1, bf16)
    c_l1 = relocate(c_steps, l1, bf16)
    zeros_l1 = move(zeros, l1)
    a8 = relocate(a_steps, dram, bf8)
    bu8 = relocate(bu_steps, dram, bf8)
    c8 = relocate(c_steps, dram, bf8)
    zeros8 = ttnn.typecast(zeros, bf8)
    return [
        ("DRAM bf16", lambda: scan_once(a_steps, bu_steps, c_steps, zeros, e, dram, bf16, False)[0]),
        ("L1 bf16", lambda: scan_once(a_l1, bu_l1, c_l1, zeros_l1, e, l1, bf16, False)[0]),
        ("DRAM approx", lambda: scan_once(a_steps, bu_steps, c_steps, zeros, e, dram, bf16, True)[0]),
        ("DRAM bf8", lambda: scan_once(a8, bu8, c8, zeros8, e, dram, bf8, False)[0]),
        ("reuse DRAM buffers", persistent_scan(a_steps, bu_steps, c_steps, zeros, e, dram, bf16, False)),
        ("reuse L1 buffers", persistent_scan(a_l1, bu_l1, c_l1, zeros_l1, e, l1, bf16, False)),
        ("L1 state, DRAM y", lambda: scan_once(a_l1, bu_l1, c_l1, zeros_l1, e, dram, bf16, False)[0]),
        ("reuse DRAM + approx", persistent_scan(a_steps, bu_steps, c_steps, zeros, e, dram, bf16, True)),
        ("reuse DRAM bf8", persistent_scan(a8, bu8, c8, zeros8, e, dram, bf8, False)),
        ("reuse L1 + approx", persistent_scan(a_l1, bu_l1, c_l1, zeros_l1, e, l1, bf16, True)),
    ]


def out_proj_config(device):
    grid = device.compute_with_storage_grid_size()
    cols = 8 if grid.x >= 8 else grid.x
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(cols, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=1,
        per_core_N=80 // cols,
        transpose_mcast=False,
        fused_activation=None,
    )


def finish_mixer(y, x, d, gate, w_out, bias, e, seq, mem, dtype, cfg=None, program=None):
    y = ttnn.add(
        y,
        ttnn.multiply(x, d, memory_config=mem, dtype=dtype),
        memory_config=mem,
        dtype=dtype,
    )
    y = ttnn.multiply(y, gate, memory_config=mem, dtype=dtype)
    y = ttnn.reshape(y, (1, seq, e))
    kwargs = {"bias": bias, "memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": dtype}
    if cfg is not None:
        kwargs["compute_kernel_config"] = cfg
    if program is not None:
        kwargs["program_config"] = program
    return ttnn.linear(y, w_out, **kwargs)


def std_mixer_attempts(device, scan_fns, x, d, gate, w_out, w8, bias, e, seq):
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    bf16, bf8 = ttnn.bfloat16, ttnn.bfloat8_b
    lofi = kernel_config(device, ttnn.MathFidelity.LoFi)
    hifi2 = kernel_config(device, ttnn.MathFidelity.HiFi2)
    hifi4 = kernel_config(device, ttnn.MathFidelity.HiFi4)
    fp32 = kernel_config(device, ttnn.MathFidelity.HiFi2, fp32=True)
    program = out_proj_config(device)
    x_l1, d_l1, gate_l1 = move(x, l1), move(d, l1), move(gate, l1)
    return [
        (
            "DRAM scan + linear HiFi2",
            lambda: finish_mixer(scan_fns[0](), x, d, gate, w_out, bias, e, seq, dram, bf16, hifi2),
        ),
        (
            "L1 scan + linear HiFi2",
            lambda: finish_mixer(scan_fns[1](), x, d, gate, w_out, bias, e, seq, l1, bf16, hifi2),
        ),
        (
            "reuse scan + linear HiFi2",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w_out, bias, e, seq, dram, bf16, hifi2),
        ),
        (
            "reuse scan + linear LoFi",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w_out, bias, e, seq, dram, bf16, lofi),
        ),
        (
            "reuse scan + linear HiFi4",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w_out, bias, e, seq, dram, bf16, hifi4),
        ),
        (
            "reuse scan + bf8 weights LoFi",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w8, bias, e, seq, dram, bf8, lofi),
        ),
        (
            "reuse scan + linear fp32 acc",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w_out, bias, e, seq, dram, bf16, fp32),
        ),
        (
            "L1 residual + linear HiFi2",
            lambda: finish_mixer(scan_fns[1](), x_l1, d_l1, gate_l1, w_out, bias, e, seq, l1, bf16, hifi2),
        ),
        (
            "reuse scan + program config",
            lambda: finish_mixer(scan_fns[2](), x, d, gate, w_out, bias, e, seq, dram, bf16, hifi2, program),
        ),
        (
            "L1 reuse scan + bf8 LoFi",
            lambda: finish_mixer(scan_fns[3](), x_l1, d_l1, gate_l1, w8, bias, e, seq, l1, bf8, lofi),
        ),
    ]


def pre_scan(std, device, pieces, e, state):
    discrete_a = std.pad_n(pieces["discrete_A"], state)
    delta_b_u = std.pad_n(pieces["delta_b_u"], state)
    a_steps = [on_device(discrete_a[:, :, i, :].reshape(1, 1, e, 32), device) for i in range(std.SEQ)]
    bu_steps = [on_device(delta_b_u[:, :, i, :].reshape(1, 1, e, 32), device) for i in range(std.SEQ)]
    c_steps = [
        on_device(std.pad_n(pieces["C"][:, i, :], state).reshape(1, 1, 1, 32).repeat(1, 1, e, 1), device)
        for i in range(std.SEQ)
    ]
    zeros = on_device(torch.zeros(1, 1, e, 32), device)
    return a_steps, bu_steps, c_steps, zeros


def pytorch_pre(pieces, e, state):
    hidden = torch.zeros(1, e, state)
    ys = []
    for i in range(pieces["discrete_A"].shape[2]):
        hidden = pieces["discrete_A"][:, :, i, :] * hidden + pieces["delta_b_u"][:, :, i, :]
        ys.append((hidden * pieces["C"][:, i, :].unsqueeze(1)).sum(-1))
    return torch.stack(ys, dim=-1)


def compare(bests):
    print("\n== comparison", flush=True)
    print(
        f"{'op':48} {'table ms':>10} {'best ms':>10} {'delta':>8} {'table pcc':>10} {'best pcc':>10}  attempt",
        flush=True,
    )
    for op, (table_pcc, table_ms) in TABLE.items():
        best = bests.get(op)
        if best is None:
            print(f"{op:48} {table_ms:10.3f} {'-':>10} {'-':>8} {table_pcc:10.4f} {'-':>10}  none passed", flush=True)
            continue
        delta = best["ms"] - table_ms
        print(
            f"{op:48} {table_ms:10.3f} {best['ms']:10.3f} {delta:+8.3f} {table_pcc:10.4f} {best['pcc']:10.4f}  {best['name']}",
            flush=True,
        )


def main():
    std = load_std()
    torch.manual_seed(0)
    if RESULTS.exists():
        RESULTS.unlink()
    mixer = std.load_mixer()
    hidden = torch.randn(1, std.SEQ, mixer.hidden_size)
    with torch.no_grad():
        reference = mixer.slow_forward(hidden)
        pieces = std.torch_pieces(mixer, hidden)
    e = mixer.intermediate_size
    state = mixer.ssm_state_size
    b = std.pad_n(pieces["B"][:, 0, :], state).reshape(1, 1, 1, 32).repeat(1, 1, 32, 1)
    dt = pieces["dt"][:, :, 0].reshape(1, 1, 1, e).repeat(1, 1, 32, 1)
    torch_bd = b.repeat(1, 1, 1, e) * dt.repeat_interleave(32, dim=-1)
    packed = torch.randn(1, 1, 32, e * 32)
    expected_sum = packed.reshape(1, 1, 32, e, 32).sum(-1)
    pre = pytorch_pre(pieces, e, state)

    device = ttnn.CreateDevice(0, l1_small_size=16384)
    bests = {}
    try:
        at = on_device(b, device)
        bt = on_device(dt, device)
        a_rep = ttnn.repeat(at, ttnn.Shape([1, 1, 1, e]))
        b_rep = ttnn.repeat_interleave(bt, 32, dim=3)
        bests["std repeat + multiply (B*dt)"], _ = run_attempts(
            device, "std repeat + multiply (B*dt)", std_repeat_attempts(device, a_rep, b_rep), torch_bd, SMALL_ITERS
        )
        bests["custom repeat_and_interleave_eltwise_mul"], _ = run_attempts(
            device,
            "custom repeat_and_interleave_eltwise_mul",
            custom_repeat_attempts(device, at, bt),
            torch_bd,
            SMALL_ITERS,
        )

        xt = on_device(packed, device)
        bests["std reshape + sum"], _ = run_attempts(
            device, "std reshape + sum", std_sum_attempts(device, xt, e), expected_sum, SMALL_ITERS
        )
        bests["custom hc_sum_reduce"], _ = run_attempts(
            device, "custom hc_sum_reduce", custom_sum_attempts(device, xt), expected_sum, SMALL_ITERS
        )

        a_steps, bu_steps, c_steps, zeros = pre_scan(std, device, pieces, e, state)
        dram = ttnn.DRAM_MEMORY_CONFIG
        l1 = ttnn.L1_MEMORY_CONFIG
        scan_attempts = std_scan_attempts(device, a_steps, bu_steps, c_steps, zeros, e)
        bests["std mul/add scan y vs pytorch"], _ = run_attempts(
            device,
            "std mul/add scan y vs pytorch",
            scan_attempts,
            pre.transpose(1, 2).reshape(1, 1, std.SEQ, e),
            HEAVY_ITERS,
        )

        d = on_device(mixer.D.detach().reshape(1, 1, 1, e), device)
        x = on_device(pieces["x"].detach().transpose(1, 2).reshape(1, 1, std.SEQ, e), device)
        gate = on_device(
            torch.nn.functional.silu(pieces["gate"].detach()).transpose(1, 2).reshape(1, 1, std.SEQ, e), device
        )
        w_out = on_device(mixer.out_proj.weight.detach().t().contiguous(), device)
        w8 = on_device(mixer.out_proj.weight.detach().t().contiguous(), device, dtype=ttnn.bfloat8_b)
        bias = None
        a_l1 = relocate(a_steps, l1, ttnn.bfloat16)
        bu_l1 = relocate(bu_steps, l1, ttnn.bfloat16)
        c_l1 = relocate(c_steps, l1, ttnn.bfloat16)
        zeros_l1 = move(zeros, l1)
        scan_fns = [
            lambda: scan_once(a_steps, bu_steps, c_steps, zeros, e, dram, ttnn.bfloat16, False)[0],
            lambda: scan_once(a_l1, bu_l1, c_l1, zeros_l1, e, l1, ttnn.bfloat16, False)[0],
            persistent_scan(a_steps, bu_steps, c_steps, zeros, e, dram, ttnn.bfloat16, False),
            persistent_scan(a_l1, bu_l1, c_l1, zeros_l1, e, l1, ttnn.bfloat16, False),
        ]
        bests["std mixer out vs pytorch slow_forward"], _ = run_attempts(
            device,
            "std mixer out vs pytorch slow_forward",
            std_mixer_attempts(device, scan_fns, x, d, gate, w_out, w8, bias, e, std.SEQ),
            reference,
            HEAVY_ITERS,
        )
    finally:
        ttnn.close_device(device)
    compare(bests)


if __name__ == "__main__":
    main()
