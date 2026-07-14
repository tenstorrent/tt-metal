#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regenerate conv3d_blockings.py from encoder sweep JSON results."""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "sweep_results"
BLOCKINGS_PY = ROOT / "tt" / "vae" / "conv3d_blockings.py"

_ENC_DIR_RE = re.compile(r"^enc_(?P<h>\d+)(?:x(?P<w>\d+))?$")


def _enc_dir_pixel_size(dir_name: str) -> tuple[int, int] | None:
    m = _ENC_DIR_RE.match(dir_name)
    if not m:
        return None
    h = int(m.group("h"))
    w = int(m.group("w")) if m.group("w") else h
    return h, w


def _load_results() -> list[dict]:
    rows: list[dict] = []
    seen_json: set[pathlib.Path] = set()
    for enc_dir in sorted(SWEEP_DIR.glob("enc_*")):
        if not enc_dir.is_dir():
            continue
        pixel = _enc_dir_pixel_size(enc_dir.name)
        for p in sorted(enc_dir.glob("*.json")):
            rp = p.resolve()
            if rp in seen_json:
                continue
            seen_json.add(rp)
            data = json.loads(p.read_text())
            blk = data.get("best_blocking")
            if not blk:
                continue
            rows.append(
                {
                    "path": p,
                    "enc_dir": enc_dir.name,
                    "pixel_h": pixel[0] if pixel else None,
                    "pixel_w": pixel[1] if pixel else None,
                    "name": p.stem.rsplit("_", 4)[0] if "_T" in p.stem else p.stem,
                    "h_factor": data["h_factor"],
                    "w_factor": data["w_factor"],
                    "c_in": data["C_in"],
                    "c_out": data["C_out"],
                    "kernel": tuple(data["kernel"]),
                    "t": data["T"],
                    "h": data["H"],
                    "w": data["W"],
                    "blocking": tuple(blk),
                    "us": data.get("best_us"),
                }
            )
    return rows


def _dedupe_exact(rows: list[dict]) -> list[dict]:
    """Keep one row per exact blocking key; prefer lower ``best_us``."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["h_factor"], r["w_factor"], r["c_in"], r["c_out"], r["kernel"], r["t"], r["h"], r["w"])
        prev = best.get(key)
        if prev is None or (r.get("us") or float("inf")) < (prev.get("us") or float("inf")):
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["c_in"], r["c_out"], r["t"], r["h"], r["w"], str(r["path"])))


def _format_exact(rows: list[dict]) -> str:
    lines = ["HUNYUAN_EXACT_BLOCKINGS: dict = {"]
    for r in rows:
        k = r["kernel"]
        key = (
            f"({r['h_factor']}, {r['w_factor']}, {r['c_in']}, {r['c_out']}, "
            f"({k[0]}, {k[1]}, {k[2]}), {r['t']}, {r['h']}, {r['w']})"
        )
        val = r["blocking"]
        us = f"  # {r['us']:.0f} us" if r.get("us") else ""
        tag = f" [{r['enc_dir']}]" if r.get("enc_dir") else ""
        comment = f"  # {r['path'].name}{tag}{us}"
        lines.append(comment)
        lines.append(f"    {key}: {val},")
    lines.append("}")
    return "\n".join(lines)


def _is_chunk_strip(path: pathlib.Path) -> bool:
    """Chunk JSON names embed ``_H{strip}_`` before ``_{Cin}x{Cout}_``."""
    return re.search(r"_H\d+_\d+x", path.stem) is not None


def _format_channel(rows: list[dict], decoder_blockings: dict) -> str:
    channel: dict[tuple, tuple] = {}
    # Prefer 1024² full-shape winners for channel fallbacks.
    pref = [r for r in rows if not _is_chunk_strip(r["path"]) and r.get("pixel_h") == r.get("pixel_w") == 1024]
    src = pref if pref else [r for r in rows if not _is_chunk_strip(r["path"])]
    for r in src:
        channel[(r["c_in"], r["c_out"], r["kernel"])] = r["blocking"]

    lines = ["HUNYUAN_CHANNEL_BLOCKINGS: dict = {"]
    enc_keys = sorted(channel.keys())
    if enc_keys:
        lines.append("    # --- encoder (from full-shape sweeps, 1024² preferred) ---")
        for key in enc_keys:
            lines.append(f"    {key}: {channel[key]},")
    lines.append("    # --- decoder (unchanged placeholders until decoder sweep) ---")
    for key, val in sorted(decoder_blockings.items()):
        if key not in channel:
            lines.append(f"    {key}: {val},")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    rows = _dedupe_exact(_load_results())
    if not rows:
        print("No sweep JSON with best_blocking found under sweep_results/enc_*", file=sys.stderr)
        return 1

    decoder_only = {
        (128, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (256, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (512, 1024, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 2048, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 4096, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 8192, (3, 3, 3)): (64, 32, 1, 2, 2),
        (128, 3, (3, 3, 3)): (32, 32, 1, 2, 2),
        (1024, 1024, (1, 1, 1)): (256, 32, 1, 1, 1),
    }

    dirs = sorted({r["enc_dir"] for r in rows if r.get("enc_dir")})
    header = f'''# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan VAE conv3d blocking tables (exact + channel fallbacks).

Auto-generated from sweep JSON via scripts/apply_encoder_sweep_blockings.py.
Do not edit HUNYUAN_EXACT_BLOCKINGS by hand — re-run the sweep + apply script.

Exact entries: (h_factor, w_factor, C_in, C_out, kernel, T, H_in, W_in)
-> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).

Swept enc dirs: {", ".join(dirs)}
"""

from __future__ import annotations

from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, _ntuple

# ===================================================================
# BH 2×2 replicated encoder — multi-resolution presets
# ===================================================================
'''
    body = _format_exact(rows)
    channel = _format_channel(rows, decoder_blockings=decoder_only)
    footer = '''

def register_hunyuan_conv3d_blockings() -> None:
    """Merge Hunyuan blocking tables into the global conv3d lookup."""
    _BLOCKINGS.update(HUNYUAN_EXACT_BLOCKINGS)
    _DEFAULT_BLOCKINGS.update(
        {(c_in, c_out, _ntuple(ks, 3)): tuple(v) for (c_in, c_out, ks), v in HUNYUAN_CHANNEL_BLOCKINGS.items()}
    )
'''
    BLOCKINGS_PY.write_text(header + body + "\n\n" + channel + footer)
    print(f"Wrote {len(rows)} exact entries from {len(dirs)} enc dir(s) to {BLOCKINGS_PY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
