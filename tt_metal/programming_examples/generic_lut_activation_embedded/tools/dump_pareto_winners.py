#!/usr/bin/env python3
"""Generate raw input/output dumps for Pareto winner manifest rows."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


LEGACY_TT_METAL_ROOT = os.environ.get("LEGACY_TT_METAL_ROOT")
LEGACY_FIT_ROOT = os.environ.get("LEGACY_TT_POLY_FIT_ROOT")


def repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def repo_python(repo: Path) -> str:
    python = repo / "python_env" / "bin" / "python"
    return str(python if python.exists() else Path(sys.executable))


def resolve_manifest_path(path: str, repo: Path, fit: Path) -> Path:
    if not path:
        return Path()
    p = Path(path)
    if LEGACY_TT_METAL_ROOT:
        try:
            rel = p.relative_to(Path(LEGACY_TT_METAL_ROOT))
            return repo / rel
        except ValueError:
            pass
    if LEGACY_FIT_ROOT:
        try:
            rel = p.relative_to(Path(LEGACY_FIT_ROOT))
            return fit / rel
        except ValueError:
            pass
    return p


def activation_domain(fit: Path, activation: str) -> tuple[str, str]:
    with (fit / "activations" / f"{activation}.json").open() as f:
        domain = json.load(f).get("domain") or {}
    lo = domain.get("min")
    hi = domain.get("max")
    if lo is None or hi is None:
        raise RuntimeError(f"missing domain for {activation}")
    return str(lo), str(hi)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def write_ttnn_dump(repo: Path, fit: Path, row: dict[str, str], dump_csv: Path, tiles: int) -> None:
    activation = row["activation"]
    dtype = row["dtype"]
    lo, hi = activation_domain(fit, activation)
    dump_csv.parent.mkdir(parents=True, exist_ok=True)

    native_py = f"""
import numpy as np
import sys
import torch
import ttnn

act, prec, lo, hi, out_csv = {activation!r}, {dtype!r}, float({lo!r}), float({hi!r}), {str(dump_csv)!r}
dev = ttnn.open_device(device_id=0)
try:
    fn = getattr(ttnn, act)
    is_bf16 = prec == "bf16"
    dt_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32
    dt_t = torch.bfloat16 if is_bf16 else torch.float32
    if is_bf16:
        bits = np.arange(65536, dtype=np.uint16)
        vals = np.frombuffer((bits.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
        mask = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
        x = torch.from_numpy(np.sort(vals[mask])).bfloat16()
    else:
        x = torch.linspace(lo, hi, 262144, dtype=torch.float32)
    n = len(x)
    if n:
        pad = ((n + 1023) // 1024) * 1024
        xp = torch.zeros(pad, dtype=dt_t)
        xp[:n] = x
        xt = ttnn.from_torch(xp.reshape(1, 1, 1, -1), device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
        hw = ttnn.to_torch(fn(xt)).squeeze().float().numpy()[:n].astype(np.float32)
        xn = x.float().numpy().astype(np.float32)
    else:
        xn = np.array([], dtype=np.float32)
        hw = np.array([], dtype=np.float32)
    np.savez_compressed(out_csv, input=xn, output=hw)
finally:
    ttnn.close_device(dev)
"""
    env = os.environ.copy()
    subprocess.run(
        [repo_python(repo), "-c", native_py],
        cwd=repo,
        env=env,
        check=True,
    )


def run_embedded_dump(
    repo: Path,
    fit: Path,
    row: dict[str, str],
    coeff_csv: Path,
    dump_csv: Path,
    tiles: int,
    skip_build: bool,
) -> None:
    activation = row["activation"]
    dtype = row["dtype"]
    lo, hi = activation_domain(fit, activation)
    run_csv = repo / "tt_metal/programming_examples/generic_lut_activation_embedded/run_csv.sh"
    dump_csv.parent.mkdir(parents=True, exist_ok=True)
    dump_flag = "--dump-npz" if dump_csv.suffix == ".npz" else "--dump-csv"
    cmd = [
        str(run_csv),
        str(coeff_csv),
        "--activation",
        activation,
        "--precision",
        dtype,
        "--tiles",
        str(tiles),
        "--runs",
        "1",
        "--range-min",
        lo,
        "--range-max",
        hi,
        dump_flag,
        str(dump_csv),
    ]
    if skip_build:
        cmd.append("--skip-build")
    env = os.environ.copy()
    env["TT_POLY_FIT_DIR"] = str(fit)
    subprocess.run(cmd, cwd=repo, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tt-poly-fit-dir", type=Path, default=None)
    parser.add_argument("--tiles", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo = repo_root()
    fit = (
        (args.tt_poly_fit_dir or Path(os.environ.get("TT_POLY_FIT_DIR", repo.parent / "tt-polynomial-fitter")))
        .expanduser()
        .resolve()
    )
    manifest = args.manifest.expanduser().resolve()
    rows = read_manifest(manifest)
    if args.limit:
        rows = rows[: args.limit]

    done = skipped = failed = 0
    for idx, row in enumerate(rows, 1):
        dump_csv = resolve_manifest_path(row.get("dump_csv", ""), repo, fit)
        if not dump_csv:
            skipped += 1
            continue
        if args.skip_existing and nonempty(dump_csv):
            skipped += 1
            continue

        try:
            if row.get("method") == "ttnn":
                write_ttnn_dump(repo, fit, row, dump_csv, args.tiles)
            else:
                coeff_csv = resolve_manifest_path(row.get("coeff_csv", ""), repo, fit)
                if not coeff_csv.exists():
                    raise FileNotFoundError(coeff_csv)
                run_embedded_dump(repo, fit, row, coeff_csv, dump_csv, args.tiles, args.skip_build and done > 0)
            done += 1
            print(f"[{idx}/{len(rows)}] wrote {dump_csv}")
        except Exception as exc:
            failed += 1
            print(
                f"[{idx}/{len(rows)}] FAIL {row.get('activation')} {row.get('dtype')} {row.get('role')}: {exc}",
                file=sys.stderr,
            )

    print(f"# dump_pareto_winners done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
