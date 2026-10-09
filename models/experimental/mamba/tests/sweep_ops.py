#!/usr/bin/env python3
"""Dtype and math-fidelity sweep of the ops in a Mamba-2.8B forward pass.

Shapes are the ones the Wormhole demo actually launches (inner width 5120,
state padded to 32, decode outer 32, prefill outer 128). Only dtype and
math fidelity change. A config is marked pass when PCC against the float32
PyTorch reference is at least 0.99.

This folder is gitignored. Run from the tt-metal checkout whose Python
package matches the built _ttnn.so:

    TT_METAL_HOME=/home/andy/tt-metal python sweep_ops.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
PCC_MIN = 0.99
ITERS = 3

D_MODEL = 2560
D_INNER = 5120
D_STATE = 32
DT_RANK = 160
VOCAB = 50280
CONV_K = 4
# The demo splits the depthwise conv in half so one launch fits.
CONV_CHANNELS = D_INNER // 2
PREFILL_LEN = 128
DECODE_OUTER = 32

DTYPES = (
    ("bfloat16", ttnn.bfloat16),
    ("bfloat8_b", ttnn.bfloat8_b),
    ("bfloat4_b", ttnn.bfloat4_b),
)
FIDS = (
    ("LoFi", ttnn.MathFidelity.LoFi),
    ("HiFi2", ttnn.MathFidelity.HiFi2),
    ("HiFi3", ttnn.MathFidelity.HiFi3),
    ("HiFi4", ttnn.MathFidelity.HiFi4),
)


def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 1.0 if torch.allclose(actual.float(), expected.float()) else 0.0
    return float((a * b).sum() / denom)


def to_dev(tensor: torch.Tensor, dtype, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_cpu(tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor)


def kernel_config(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )


def time_calls(device, fn, iters=ITERS) -> tuple[float, object]:
    # PCC uses this first output. Later calls are latency only, because a few
    # ops (prefix scan) write the hidden state in place.
    out = fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / iters, out


def record(row: dict) -> None:
    row["pass"] = bool(row.get("pcc") is not None and row["pcc"] >= PCC_MIN and not row.get("error"))
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    status = "PASS" if row["pass"] else "FAIL"
    pcc_s = f"{row['pcc']:.4f}" if isinstance(row.get("pcc"), float) else "-"
    ms = f"{row['ms']:.3f}" if isinstance(row.get("ms"), float) else "-"
    err = f"  {row['error']}" if row.get("error") else ""
    print(
        f"{status:4} {row['op']:28} {row['mode']:8} {row['dtype']:12} {row['fidelity']:6} "
        f"pcc={pcc_s:7} {ms:>10} ms{err}",
        flush=True,
    )


def run_case(device, op, mode, dtype_name, dtype, fid_name, fid, build):
    row = {
        "op": op,
        "mode": mode,
        "dtype": dtype_name,
        "fidelity": fid_name,
        "pcc": None,
        "ms": None,
        "error": None,
    }
    made = []
    try:
        torch.manual_seed(0)
        fn, expected, owned = build(device, dtype, fid)
        made.extend(owned)

        def wrapped():
            out = fn()
            made.append(out)
            return out

        seconds, out = time_calls(device, wrapped)
        actual = to_cpu(out).float()
        row["ms"] = seconds * 1e3
        row["pcc"] = pcc(actual, expected)
    except Exception as exc:  # noqa: BLE001 - one bad config must not stop the sweep
        row["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:400]
    finally:
        for tensor in made:
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass
    record(row)


def linear_case(name, outer, k, n):
    def build(device, dtype, fid):
        x = torch.randn(1, 1, outer, k)
        w = torch.randn(k, n)
        expected = torch.matmul(x.float(), w.float())
        cfg = kernel_config(device, fid)
        xt = to_dev(x, dtype, device)
        wt = to_dev(w, dtype, device)

        def fn():
            return ttnn.linear(
                xt,
                wt,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=cfg,
            )

        return fn, expected, [xt, wt]

    return name, build


def eltwise_case(name, shape, torch_fn, tt_fn, scale=1.0):
    def build(device, dtype, fid):
        x = torch.randn(*shape) * scale
        expected = torch_fn(x.float())
        xt = to_dev(x, dtype, device)
        cfg = kernel_config(device, fid)

        def fn():
            try:
                return tt_fn(xt, dtype, cfg)
            except TypeError:
                return tt_fn(xt, dtype, None)

        return fn, expected, [xt]

    return name, build


def binary_case(name, shape, torch_fn, tt_fn):
    def build(device, dtype, fid):
        a = torch.randn(*shape)
        b = torch.randn(*shape)
        expected = torch_fn(a.float(), b.float())
        at = to_dev(a, dtype, device)
        bt = to_dev(b, dtype, device)

        def fn():
            return tt_fn(at, bt, dtype)

        return fn, expected, [at, bt]

    return name, build


def repeat_ref(a, b):
    hidden = D_INNER
    latent = D_STATE
    aw, bw = a.shape[-1], b.shape[-1]
    if aw == latent and bw == hidden:
        return a.repeat(1, 1, 1, hidden) * b.repeat_interleave(latent, dim=-1)
    if aw == latent * hidden and bw == hidden:
        return a * b.repeat_interleave(latent, dim=-1)
    if aw == latent and bw == latent * hidden:
        return a.repeat(1, 1, 1, hidden) * b
    raise AssertionError(f"bad repeat shapes {aw} {bw}")


def repeat_case(name, outer, aw, bw):
    def build(device, dtype, fid):
        a = torch.randn(1, 1, outer, aw)
        b = torch.randn(1, 1, outer, bw)
        expected = repeat_ref(a, b)
        at = to_dev(a, dtype, device)
        bt = to_dev(b, dtype, device)

        def fn():
            return ttnn.experimental.repeat_and_interleave_eltwise_mul(
                at,
                bt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype,
                math_fidelity=fid,
            )

        return fn, expected, [at, bt]

    return name, build


def sum_reduce_case(name, outer):
    def build(device, dtype, fid):
        x = torch.randn(1, 1, outer, D_INNER * D_STATE)
        expected = x.float().reshape(1, 1, outer, D_INNER, D_STATE).sum(-1)
        xt = to_dev(x, dtype, device)

        def fn():
            return ttnn.experimental.hc_sum_reduce(
                xt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype,
                math_fidelity=fid,
            )

        return fn, expected, [xt]

    return name, build


def prefix_scan_case():
    """Prefill only. Sequence length must be a multiple of 32 and the op is sharded."""

    def build(device, dtype, fid):
        length = PREFILL_LEN
        width = D_INNER * D_STATE
        num_cores = 64
        grid = device.compute_with_storage_grid_size()
        if grid.x * grid.y < num_cores:
            raise RuntimeError(f"need {num_cores} cores, device has {grid.x * grid.y}")
        a = torch.randn(1, 1, length, width)
        bx = torch.randn(1, 1, length, width)
        h0 = torch.randn(1, 1, 1, width)
        hidden = torch.zeros(1, 1, length, width)
        prev = h0[0, 0, 0]
        for i in range(length):
            prev = a[0, 0, i] * prev + bx[0, 0, i]
            hidden[0, 0, i] = prev
        shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
        shard = ttnn.ShardSpec(shard_grid, [length, width // num_cores], ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)
        h_shard = ttnn.ShardSpec(shard_grid, [1, width // num_cores], ttnn.ShardOrientation.ROW_MAJOR)
        h_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, h_shard)
        at = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        bt = ttnn.from_torch(bx, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        ht = ttnn.from_torch(h0, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=h_mem)

        def fn():
            return ttnn.experimental.prefix_scan(
                at,
                bt,
                ht,
                memory_config=mem,
                dtype=dtype,
                math_fidelity=fid,
            )

        return fn, hidden, [at, bt, ht]

    return "prefix_scan", build


def embedding_case(outer):
    def build(device, dtype, fid):
        ids = torch.randint(0, VOCAB, (1, 1, outer, 1), dtype=torch.int32)
        table = torch.randn(VOCAB, D_MODEL)
        expected = F.embedding(ids.long().reshape(-1), table.float()).reshape(1, 1, outer, D_MODEL)
        idt = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        wt = to_dev(table, dtype, device, layout=ttnn.ROW_MAJOR_LAYOUT)

        def fn():
            return ttnn.embedding(idt, wt, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return fn, expected, [idt, wt]

    return "embedding", build


def rms_case(outer):
    def build(device, dtype, fid):
        x = torch.randn(1, 1, outer, D_MODEL)
        w = torch.randn(1, D_MODEL)
        var = x.float().pow(2).mean(-1, keepdim=True)
        expected = x.float() * torch.rsqrt(var + 1e-5) * w.float()
        xt = to_dev(x, dtype, device)
        wt = to_dev(w, dtype, device)
        cfg = kernel_config(device, fid)

        def fn():
            return ttnn.rms_norm(
                xt,
                epsilon=1e-5,
                weight=wt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=cfg,
            )

        return fn, expected, [xt, wt]

    return "rms_norm", build


def conv_case():
    """One of the two depthwise conv launches the prefill block uses."""

    def build(device, dtype, fid):
        length = PREFILL_LEN
        c = CONV_CHANNELS
        x = torch.randn(1, length, 1, c)
        w = torch.randn(c, 1, CONV_K)
        ref = F.conv1d(x.float().reshape(1, length, c).permute(0, 2, 1), w.float(), groups=c)
        expected = ref.permute(0, 2, 1)
        xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        wt = ttnn.from_torch(w, dtype=ttnn.float32)
        cfg = kernel_config(device, fid)
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=dtype if dtype != ttnn.bfloat8_b else ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
        )

        def fn():
            out, _length, _wb = ttnn.conv1d(
                input_tensor=xt,
                weight_tensor=wt,
                device=device,
                in_channels=c,
                out_channels=c,
                batch_size=1,
                input_length=length,
                kernel_size=CONV_K,
                stride=1,
                padding=0,
                groups=c,
                dtype=dtype,
                conv_config=conv_config,
                compute_config=cfg,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            return out

        return fn, expected, [xt]

    return "conv1d_depthwise", build


def cases_for(mode, outer):
    """Ops in a standard Mamba block, at this mode's outer dimension."""
    found = [
        embedding_case(outer),
        rms_case(outer),
        linear_case("linear_in_proj", outer, D_MODEL, D_INNER),
        linear_case("linear_dt", outer, D_INNER, DT_RANK),
        linear_case("linear_B", outer, D_INNER, D_STATE),
        linear_case("linear_C", outer, D_INNER, D_STATE),
        linear_case("linear_dt_proj", outer, DT_RANK, D_INNER),
        linear_case("linear_out_proj", outer, D_INNER, D_MODEL),
        linear_case("linear_lm_head", outer, D_MODEL, VOCAB),
        eltwise_case(
            "silu",
            (1, 1, outer, D_INNER),
            torch.nn.functional.silu,
            lambda x, dtype, cfg: ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ),
        eltwise_case(
            "softplus",
            (1, 1, outer, D_INNER),
            lambda t: F.softplus(t, beta=1.0, threshold=20.0),
            lambda x, dtype, cfg: ttnn.softplus(x, beta=1.0, threshold=20.0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ),
        eltwise_case(
            "exp",
            (1, 1, outer, D_INNER * D_STATE),
            torch.exp,
            lambda x, dtype, cfg: ttnn.exp(x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            scale=0.05,
        ),
        binary_case(
            "multiply",
            (1, 1, outer, D_INNER * D_STATE),
            torch.mul,
            lambda a, b, dtype: ttnn.multiply(a, b, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ),
        binary_case(
            "add",
            (1, 1, outer, D_INNER * D_STATE),
            torch.add,
            lambda a, b, dtype: ttnn.add(a, b, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ),
        repeat_case("ssm_mul_A_dt", outer, D_INNER * D_STATE, D_INNER),
        repeat_case("ssm_mul_B_dt", outer, D_STATE, D_INNER),
        repeat_case("ssm_mul_C_h", outer, D_STATE, D_INNER * D_STATE),
        sum_reduce_case("hc_sum_reduce", outer),
    ]
    if mode == "prefill":
        found.append(prefix_scan_case())
        found.append(conv_case())
    return found


def main():
    smoke = "--smoke" in sys.argv
    if RESULTS.exists() and "--append" not in sys.argv and not smoke:
        RESULTS.unlink()
    device = ttnn.open_device(device_id=0)
    print(f"arch={device.arch()} grid={device.compute_with_storage_grid_size()}", flush=True)
    try:
        modes = (("decode", DECODE_OUTER), ("prefill", PREFILL_LEN))
        for mode, outer in modes:
            for name, build in cases_for(mode, outer):
                fid_list = FIDS
                # Embedding has no math-fidelity input. Time it once per dtype.
                if name == "embedding":
                    fid_list = (("n/a", ttnn.MathFidelity.HiFi4),)
                for dtype_name, dtype in DTYPES:
                    for fid_name, fid in fid_list:
                        run_case(device, name, mode, dtype_name, dtype, fid_name, fid, build)
                        if smoke:
                            print("smoke done", flush=True)
                            return
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
