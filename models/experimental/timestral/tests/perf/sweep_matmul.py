# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Matmul config sweep for every Timestral linear role at the paper shape (one chip).

    python models/experimental/timestral/tests/perf/sweep_matmul.py <out_dir> [--precision accuracy|performance]
        [--roles qkv,down,...] [--rows 98304]

Each role runs ``(1, 1, rows, K) @ (K, N)`` (+ bias, + activation as in the model) under:

    default        ttnn.linear auto config; the activation runs as a separate unary op
    core_grid      ttnn.linear with an explicit core grid (fuses the activation)
    1d_full_n      1D config, weights multicast, M split over all cores, each core's whole N in one
                   out block, so activations are read once
    2d             2D multicast, M over grid rows, N over grid columns
    minimal        ttnn.experimental.minimal_matmul (default, tt_dit heuristic and manual blockings)

Device time per call is the median wall time of synchronized calls (each call is milliseconds, so
the sync overhead is negligible). PCC is against torch on the same bf16 inputs. Writes
``matmul_sweep_<precision>.json`` and ``.md`` to ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

TILE = 32
TIMED_RUNS = 5


@dataclass(frozen=True)
class Role:
    name: str
    k: int
    n: int
    activation: str | None
    bias: bool
    count: int  # calls per full forward

    @property
    def label(self) -> str:
        extra = "".join([" +bias" if self.bias else "", f" +{self.activation}" if self.activation else ""])
        return f"{self.k}x{self.n}{extra}"


ROLES = [
    Role("qkv", 768, 2304, None, False, 24),
    Role("gate_sigmoid", 768, 768, "sigmoid", False, 30),
    Role("proj768", 768, 768, None, False, 36),
    Role("up_silu", 768, 2048, "silu", False, 24),
    Role("mlp_gate", 768, 2048, None, False, 24),
    Role("down", 2048, 768, None, False, 24),
    Role("embed_hidden", 64, 3072, "silu", True, 1),
    Role("embed_out", 3072, 768, None, True, 1),
    Role("embed_res", 64, 768, None, True, 1),
]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _subblock(block_h: int, block_w: int, max_tiles: int = 8) -> tuple[int, int]:
    best = (1, 1)
    for h in _divisors(block_h):
        for w in _divisors(block_w):
            if h * w <= max_tiles and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _unary(name):
    import ttnn

    if name is None:
        return None
    return ttnn.UnaryWithParam({"silu": ttnn.UnaryOpType.SILU, "sigmoid": ttnn.UnaryOpType.SIGMOID}[name])


def _torch_activation(name, y):
    if name is None:
        return y
    return torch.nn.functional.silu(y) if name == "silu" else torch.sigmoid(y)


def _candidates(role: Role, m_tiles: int, grid, precision) -> list[tuple[str, dict]]:
    """``(name, spec)`` pairs; ``spec['kind']`` picks the call in ``_run``."""
    import ttnn

    k_tiles, n_tiles = role.k // TILE, role.n // TILE
    gx, gy = grid.x, grid.y
    out: list[tuple[str, dict]] = [("default", {"kind": "linear"})]
    out.append(("core_grid", {"kind": "linear", "core_grid": ttnn.CoreGrid(y=gy, x=gx)}))

    per_core_m = math.ceil(m_tiles / (gx * gy))
    for obh in [d for d in _divisors(per_core_m) if d <= 14]:
        for ibw in [d for d in _divisors(k_tiles) if d <= 8]:
            if 2 * obh * ibw + 2 * ibw * n_tiles + 2 * obh * n_tiles > 640:
                continue
            sh, sw = _subblock(obh, n_tiles)
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=ibw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=obh,
                out_block_w=n_tiles,
                per_core_M=per_core_m,
                per_core_N=n_tiles,
                fuse_batch=True,
                fused_activation=_unary(role.activation),
                mcast_in0=False,
            )
            out.append((f"1d_full_n[obh={obh},ibw={ibw},sb={sh}x{sw}]", {"kind": "linear", "program_config": cfg}))

    per_core_n = math.ceil(n_tiles / gx)
    cols = math.ceil(n_tiles / per_core_n)
    per_core_m2 = math.ceil(m_tiles / gy)
    for obh in [d for d in _divisors(per_core_m2) if 4 <= d <= 16]:
        for ibw in [d for d in _divisors(k_tiles) if d <= 12]:
            if 2 * obh * ibw + 2 * ibw * per_core_n + 2 * obh * per_core_n > 640:
                continue
            sh, sw = _subblock(obh, per_core_n)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(cols, gy),
                in0_block_w=ibw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                out_block_h=obh,
                out_block_w=per_core_n,
                per_core_M=per_core_m2,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=_unary(role.activation),
                fuse_batch=True,
            )
            out.append((f"2d[{cols}x{gy},obh={obh},ibw={ibw},sb={sh}x{sw}]", {"kind": "linear", "program_config": cfg}))

    out.append(("minimal[default]", {"kind": "minimal", "config": None}))
    try:
        from models.tt_dit.utils.matmul import get_matmul_config

        rows = m_tiles * TILE
        heuristic = get_matmul_config(rows, role.k, role.n, ttnn.CoreCoord(gx, gy), use_heuristic=True)
        out.append(("minimal[tt_dit heuristic]", {"kind": "minimal", "config": heuristic}))
    except Exception as e:  # noqa: BLE001 - heuristic is optional
        print(f"  tt_dit heuristic unavailable: {e}")
    for mb, kb, nb in [(8, 8, 8), (16, 8, 8), (8, 4, 8), (8, 8, 4), (16, 4, 4), (4, 8, 8)]:
        if k_tiles % kb:
            continue
        nb_eff = min(nb, n_tiles)
        sh, sw = _subblock(mb, nb_eff, 8)
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb_eff,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        )
        out.append((f"minimal[{mb},{kb},{nb_eff},sb={sh}x{sw}]", {"kind": "minimal", "config": cfg}))
    return out


def _run(spec: dict, x, w, b, role: Role, precision):
    import ttnn

    kernel = precision.compute_kernel_config()
    if spec["kind"] == "minimal":
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            bias_tensor=b,
            fused_activation=_unary(role.activation),
            config=spec["config"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )
    kwargs = {k: v for k, v in spec.items() if k in ("core_grid", "program_config")}
    activation = None if "program_config" in kwargs else role.activation
    return ttnn.linear(
        x,
        w,
        bias=b,
        activation=activation,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=kernel,
        **kwargs,
    )


def sweep(device, roles: list[Role], rows: int, precision) -> list[dict]:
    import ttnn

    from models.experimental.timestral.tt.common import upload, upload_bias, upload_linear
    from tests.ttnn.utils_for_testing import comp_pcc

    grid = device.compute_with_storage_grid_size()
    m_tiles = rows // TILE
    results = []
    for role in roles:
        torch.manual_seed(0)
        x = torch.randn(rows, role.k).to(torch.bfloat16).float()
        weight = (torch.randn(role.n, role.k) / math.sqrt(role.k)).to(torch.bfloat16).float()
        bias = (0.1 * torch.randn(role.n)).to(torch.bfloat16).float() if role.bias else None
        y = x @ weight.t() + (0 if bias is None else bias)
        expected = _torch_activation(role.activation, y)
        x_tt = upload(device, x.reshape(1, 1, rows, role.k))
        w_tt = upload_linear(device, weight, precision.weight_dtype())
        b_tt = upload_bias(device, bias)
        print(f"== {role.name} {role.label} (x{role.count})")
        for name, spec in _candidates(role, m_tiles, grid, precision):
            entry = {"role": role.name, "shape": role.label, "count": role.count, "candidate": name}
            try:
                out = _run(spec, x_tt, w_tt, b_tt, role, precision)
                ttnn.synchronize_device(device)
                got = ttnn.to_torch(out).float().reshape(rows, role.n)
                ttnn.deallocate(out)
                entry["pcc"] = comp_pcc(expected, got, 0.0)[1]
                times = []
                for _ in range(TIMED_RUNS):
                    ttnn.synchronize_device(device)
                    start = time.perf_counter()
                    out = _run(spec, x_tt, w_tt, b_tt, role, precision)
                    ttnn.synchronize_device(device)
                    times.append(time.perf_counter() - start)
                    ttnn.deallocate(out)
                entry["ms"] = 1e3 * statistics.median(times)
                print(f"  {name:48s} {entry['ms']:8.3f} ms  pcc {entry['pcc']:.5f}")
            except Exception as e:  # noqa: BLE001 - illegal configs are recorded, not fatal
                entry["error"] = str(e).strip().splitlines()[0][:200]
                print(f"  {name:48s} FAILED: {entry['error']}")
            results.append(entry)
        for t in (x_tt, w_tt, b_tt):
            if t is not None:
                ttnn.deallocate(t)
    return results


def _report(results: list[dict], precision_name: str) -> str:
    lines = [
        f"# Matmul sweep ({precision_name})",
        "",
        "| role | shape | x | default ms | best candidate | best ms | speedup | full-model saving ms | PCC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for role in dict.fromkeys(r["role"] for r in results):
        rows = [r for r in results if r["role"] == role and "ms" in r and r["pcc"] >= 0.999]
        default = next((r for r in rows if r["candidate"] == "default"), None)
        if not rows or default is None:
            continue
        best = min(rows, key=lambda r: r["ms"])
        saving = (default["ms"] - best["ms"]) * best["count"]
        lines.append(
            f"| {role} | {best['shape']} | {best['count']} | {default['ms']:.3f} | `{best['candidate']}` | "
            f"{best['ms']:.3f} | {default['ms'] / best['ms']:.2f}x | {saving:.1f} | {best['pcc']:.5f} |"
        )
    lines += ["", "Default ms for activation roles includes the separate unary op.", ""]
    for role in dict.fromkeys(r["role"] for r in results):
        lines += [f"## {role}", "", "| candidate | ms | PCC / error |", "|---|---|---|"]
        for r in sorted((r for r in results if r["role"] == role), key=lambda r: r.get("ms", 1e9)):
            status = f"{r['pcc']:.5f}" if "pcc" in r else f"FAILED: {r['error']}"
            lines.append(f"| `{r['candidate']}` | {r.get('ms', float('nan')):.3f} | {status} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    import ttnn

    from models.experimental.timestral.tt.common import TtTimestralPrecision

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--precision", choices=["accuracy", "performance"], default="accuracy")
    parser.add_argument("--roles", default=",".join(r.name for r in ROLES))
    parser.add_argument("--rows", type=int, default=1024 * 96)
    args = parser.parse_args()
    precision = TtTimestralPrecision.performance() if args.precision == "performance" else TtTimestralPrecision()
    roles = [r for r in ROLES if r.name in args.roles.split(",")]

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        results = sweep(device, roles, args.rows, precision)
    finally:
        ttnn.close_mesh_device(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"matmul_sweep_{args.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
    report = _report(results, args.precision)
    (args.out_dir / f"matmul_sweep_{args.precision}.md").write_text(report)
    print(report.split("\n## ")[0])


if __name__ == "__main__":
    main()
