#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Merge decoder conv3d sweep JSON into tt/vae/conv3d_blockings.py.

Reads:
  - models/experimental/hunyuan_image_3_0/sweep_results/dec_*/
  - sweep_results_hunyuan_bh_2x2/*.json (legacy manual sweeps)

Updates ``HUNYUAN_DECODER_EXACT_BLOCKINGS`` and decoder rows in
``HUNYUAN_CHANNEL_BLOCKINGS``. Preserves encoder ``HUNYUAN_EXACT_BLOCKINGS`` unchanged.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SWEEP_DIR = ROOT / "sweep_results"
LEGACY_DIR = REPO / "sweep_results_hunyuan_bh_2x2"
BLOCKINGS_PY = ROOT / "tt" / "vae" / "conv3d_blockings.py"

# Faster sweep winners that fail full-decode PCC — keep the verified blocking instead.
_PCC_SAFE_DECODER_OVERRIDES: dict[tuple, tuple] = {
    # ~10.5 ms isolated sweep vs (64,256,1,4,8); full decode PCC 0.762 with [128,64,1,16,16].
    (1, 1, 1024, 8192, (3, 3, 3), 1, 34, 34): (64, 256, 1, 4, 8),
}

_PCC_SAFE_CHANNEL_OVERRIDES: dict[tuple, tuple] = {
    (1024, 8192, (3, 3, 3)): (64, 256, 1, 4, 8),
    # H514 isolated sweep winner regresses on full decode; H130 strip blocking is safe.
    (128, 3, (3, 3, 3)): (128, 32, 2, 8, 16),
}

# Full-tensor sweep entries that regress in full decode (perfvae19: conv_out H514 -> ~138 ms/op).
# Runtime always H-chunks; last strips fall back to channel blockings from H130/H386 winners.
_DECODER_EXACT_DENYLIST: set[tuple] = {
    (1, 1, 128, 3, (3, 3, 3), 4, 514, 514),
    (1, 1, 128, 128, (3, 3, 3), 4, 514, 514),
    (1, 1, 256, 256, (3, 3, 3), 4, 258, 258),
}

_DEC_DIR_RE = re.compile(r"^dec_(?P<h>\d+)(?:x(?P<w>\d+))?$")


def _dec_dir_latent_size(dir_name: str) -> tuple[int, int] | None:
    m = _DEC_DIR_RE.match(dir_name)
    if not m:
        return None
    h = int(m.group("h"))
    w = int(m.group("w")) if m.group("w") else h
    return h, w


def _is_chunk_strip(path: pathlib.Path) -> bool:
    return re.search(r"_H\d+_\d+x", path.stem) is not None


def _load_json(path: pathlib.Path, *, dec_dir: str | None, latent: tuple[int, int] | None) -> dict | None:
    data = json.loads(path.read_text())
    blk = data.get("best_blocking")
    if not blk:
        ok = [r for r in (data.get("top_20") or []) if r.get("status") == "ok" and r.get("blocking")]
        if ok:
            blk = ok[0]["blocking"]
    if not blk:
        return None
    return {
        "path": path,
        "dec_dir": dec_dir,
        "latent_h": latent[0] if latent else None,
        "latent_w": latent[1] if latent else None,
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


def _load_results() -> list[dict]:
    rows: list[dict] = []
    seen: set[pathlib.Path] = set()

    for dec_dir in sorted(SWEEP_DIR.glob("dec_*")):
        if not dec_dir.is_dir():
            continue
        latent = _dec_dir_latent_size(dec_dir.name)
        for p in sorted(dec_dir.glob("*.json")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            row = _load_json(p, dec_dir=dec_dir.name, latent=latent)
            if row:
                rows.append(row)

    if LEGACY_DIR.is_dir():
        for p in sorted(LEGACY_DIR.glob("*.json")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            row = _load_json(p, dec_dir="legacy_bh_2x2", latent=(64, 64))
            if row:
                rows.append(row)

    return rows


def _dedupe_exact(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["h_factor"], r["w_factor"], r["c_in"], r["c_out"], r["kernel"], r["t"], r["h"], r["w"])
        if key in _DECODER_EXACT_DENYLIST:
            continue
        prev = best.get(key)
        if prev is None or (r.get("us") or float("inf")) < (prev.get("us") or float("inf")):
            best[key] = r
    out = sorted(best.values(), key=lambda r: (r["c_in"], r["c_out"], r["t"], r["h"], r["w"], str(r["path"])))
    for r in out:
        key = (r["h_factor"], r["w_factor"], r["c_in"], r["c_out"], r["kernel"], r["t"], r["h"], r["w"])
        if key in _PCC_SAFE_DECODER_OVERRIDES:
            r["blocking"] = _PCC_SAFE_DECODER_OVERRIDES[key]
    return out


def _format_decoder_exact(rows: list[dict]) -> str:
    lines = ["HUNYUAN_DECODER_EXACT_BLOCKINGS: dict = {"]
    for r in rows:
        k = r["kernel"]
        key = (
            f"({r['h_factor']}, {r['w_factor']}, {r['c_in']}, {r['c_out']}, "
            f"({k[0]}, {k[1]}, {k[2]}), {r['t']}, {r['h']}, {r['w']})"
        )
        us = f"  # {r['us']:.0f} us" if r.get("us") else ""
        tag = f" [{r['dec_dir']}]" if r.get("dec_dir") else ""
        lines.append(f"    # {r['path'].name}{tag}{us}")
        lines.append(f"    {key}: {r['blocking']},")
    lines.append("}")
    return "\n".join(lines)


def _decoder_channel_defaults() -> dict[tuple, tuple]:
    return {
        (128, 3, (3, 3, 3)): (32, 32, 1, 2, 2),
        (128, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (256, 512, (3, 3, 3)): (64, 32, 1, 2, 2),
        (512, 1024, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 1024, (1, 1, 1)): (256, 32, 1, 1, 1),
        (1024, 2048, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 4096, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 8192, (3, 3, 3)): (64, 32, 1, 2, 2),
    }


def _format_channel_blockings(enc_section: str, dec_rows: list[dict]) -> str:
    channel: dict[tuple, tuple] = {}
    pref = [
        r
        for r in dec_rows
        if not _is_chunk_strip(r["path"])
        and r.get("latent_h") == 64
        and r.get("latent_w") == 64
        and (r["h_factor"], r["w_factor"], r["c_in"], r["c_out"], r["kernel"], r["t"], r["h"], r["w"])
        not in _DECODER_EXACT_DENYLIST
    ]
    for r in pref:
        channel[(r["c_in"], r["c_out"], r["kernel"])] = r["blocking"]

    defaults = _decoder_channel_defaults()
    for key, val in defaults.items():
        channel.setdefault(key, val)
    for key, val in _PCC_SAFE_CHANNEL_OVERRIDES.items():
        channel[key] = val

    lines = ["HUNYUAN_CHANNEL_BLOCKINGS: dict = {"]
    lines.append(enc_section.rstrip())
    lines.append("    # --- decoder (from dec_64x64 full-shape sweeps + legacy fallbacks) ---")
    dec_keys = sorted(k for k in channel if k not in _parse_encoder_channel_keys(enc_section))
    for key in dec_keys:
        lines.append(f"    {key}: {channel[key]},")
    lines.append("}")
    return "\n".join(lines)


def _parse_encoder_channel_keys(enc_section: str) -> set[tuple]:
    keys: set[tuple] = set()
    for m in re.finditer(r"\((\d+), (\d+), \((\d+), (\d+), (\d+)\)\):", enc_section):
        keys.add((int(m.group(1)), int(m.group(2)), (int(m.group(3)), int(m.group(4)), int(m.group(5)))))
    return keys


def _extract_section(text: str, name: str, next_name: str | None) -> str:
    start = text.index(f"{name}:")
    if next_name is None:
        return text[start:]
    end = text.index(next_name, start)
    return text[start:end]


def main() -> int:
    rows = _dedupe_exact(_load_results())
    if not rows:
        print("No decoder sweep JSON with best_blocking found.", file=sys.stderr)
        return 1

    text = BLOCKINGS_PY.read_text()
    enc_exact = _extract_section(text, "HUNYUAN_EXACT_BLOCKINGS", "HUNYUAN_CHANNEL_BLOCKINGS")
    enc_channel_match = re.search(
        r"HUNYUAN_CHANNEL_BLOCKINGS: dict = \{(.*?)\n    # --- decoder",
        text,
        re.DOTALL,
    )
    enc_channel_section = enc_channel_match.group(1) if enc_channel_match else ""

    dec_dirs = sorted({r["dec_dir"] for r in rows if r.get("dec_dir")})
    if LEGACY_DIR.is_dir():
        dec_dirs = sorted(set(dec_dirs) | {"legacy_bh_2x2"})
    header = f'''# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hunyuan VAE conv3d blocking tables (exact + channel fallbacks).

Auto-generated from sweep JSON via scripts/apply_decoder_sweep_blockings.py
(encoder exact table preserved; re-run apply_encoder_sweep_blockings.py to refresh enc).

Exact entries: (h_factor, w_factor, C_in, C_out, kernel, T, H_in, W_in)
-> (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).

Swept dec dirs: {", ".join(dec_dirs)}
"""

from __future__ import annotations

from models.tt_dit.utils.conv3d import _BLOCKINGS, _DEFAULT_BLOCKINGS, _ntuple

# ===================================================================
# BH 2×2 replicated encoder — multi-resolution presets
# ===================================================================
'''
    body_enc = enc_exact.strip() + "\n\n"
    channel = _format_channel_blockings(enc_channel_section, rows)
    dec_exact = _format_decoder_exact(rows)
    footer = f'''
# ===================================================================
# BH 2×2 spatial-sharded decoder — hardware-swept exact blockings
# ===================================================================
{dec_exact}


def register_hunyuan_conv3d_blockings() -> None:
    """Merge Hunyuan blocking tables into the global conv3d lookup."""
    _BLOCKINGS.update(HUNYUAN_EXACT_BLOCKINGS)
    _BLOCKINGS.update(HUNYUAN_DECODER_EXACT_BLOCKINGS)
    _DEFAULT_BLOCKINGS.update(
        {{(c_in, c_out, _ntuple(ks, 3)): tuple(v) for (c_in, c_out, ks), v in HUNYUAN_CHANNEL_BLOCKINGS.items()}}
    )
'''
    BLOCKINGS_PY.write_text(header + body_enc + channel + footer)
    print(f"Wrote {len(rows)} decoder exact entries from {len(dec_dirs)} dir(s) to {BLOCKINGS_PY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
