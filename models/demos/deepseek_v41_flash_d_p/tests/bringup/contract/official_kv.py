# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The reference's window KV at its own shipped precision, every position of the contract golden's dumped layers
(FIX.19, owner 2026-11-01): the margin of the contract's window-KV check, measured on the SAME positions it checks.

The contract compares a dumped layer's window KV over every position of a range [lo, hi) with the full-precision
golden (numerics native). Its limit is calib x how far the reference at its own precision (numerics official, dtype
reference.shipped_dtype, fed as the golden is: every chunk from the start) lands from the golden on those rows and that
channel (testing/e2e_limits.py convention: 1 - calib^2 x max(1 - pcc, 1 - pcc of one bf16 rounding); never below
the server's own prefill-golden PCC). The golden's drift-<rung>.json keeps only the final ring (the last 128
positions), so this file holds the per-position rows:

    <golden dir>-official/window_kv.safetensors   key "kv_L<i>" [seq, 512] bf16 (the official golden's store dtype)
    <golden dir>-official/window_kv.json          golden content_hash, numerics, dtype, layers, seconds

beside the golden (the ``<rung dir>-official`` place reference/golden.py names for official-precision outputs; the
golden itself is not touched). Made once per golden (CPU only, no device):

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.official_kv

A file made from another golden (content_hash) is refused, never used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

NAME = "window_kv"


def paths(golden_dir) -> tuple[Path, Path]:
    d = Path(str(golden_dir) + "-official")
    return d / f"{NAME}.safetensors", d / f"{NAME}.json"


def load(golden) -> dict[int, torch.Tensor]:
    """{layer: [seq, 512] official window KV} for the golden ``golden`` (reference.golden.Golden)."""
    from safetensors.torch import load_file

    st, meta = paths(golden.dir)
    if not st.exists():
        raise FileNotFoundError(
            f"{st}: no official per-position window KV for this golden (the contract's window-KV margin, FIX.19); "
            "make it: PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.official_kv"
        )
    m = json.loads(meta.read_text())
    if m.get("golden_content_hash") != golden.pinned_hash():
        raise ValueError(f"{st} was made from golden {m.get('golden_content_hash')}, not {golden.pinned_hash()}")
    return {int(k.split("_L")[1]): v for k, v in load_file(str(st)).items()}


def generate(layers: list[int] | None = None) -> Path:
    """Runs the official reference (layers 0..max(layers) resident) over every chunk of the contract golden's tokens
    and keeps the window KV ("L<i>.kv", the step output, cloned before act_quant's in-place rounding of a later step
    could touch it) of ``layers`` (default: the golden's dumped layers)."""
    from safetensors.torch import save_file

    from models.demos.common.bringup.reference.golden import Golden, load_spec, reference_at, store_dtype
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

    s = load_spec()
    g = Golden.for_rung(s, R.GOLDEN_RUNG)
    if g.manifest.get("numerics") != "native":
        raise SystemExit(f"{g.dir}: numerics {g.manifest.get('numerics')!r}, want the native golden")
    dumped = [i for i in g.layers if all(g.has_layer(c, i) for c in range(g.n_chunks))]
    layers = sorted(layers or dumped)
    dtype = getattr(torch, str(s.get("reference.shipped_dtype")))
    keep = tuple(s.get("golden.keep_fp32", []))
    t0 = time.time()
    ref = reference_at(s, "official", layers=list(range(max(layers) + 1)), dtype=dtype)
    state, tokens = ref.new_state(g.seq), g.tokens()
    rows = {i: [] for i in layers}
    times = []
    for c in range(g.n_chunks):
        s0, t1 = c * g.chunk, time.time()

        def rec(name, t):
            if name.startswith("L") and name.endswith(".kv"):
                li = int(name[1:].split(".", 1)[0])
                if li in rows:
                    rows[li].append(store_dtype(t.detach().reshape(-1, t.shape[-1]).clone(), keep, "kv"))

        ref.forward_chunk(tokens[s0 : s0 + g.chunk], s0, state, rec)
        times.append(round(time.time() - t1, 1))
        print(f"[official_kv] chunk {c}: {times[-1]} s", flush=True)
    out = {}
    for i in layers:
        t = torch.cat(rows[i])
        want = torch.cat([g.layer(c, i)["kv"] for c in range(g.n_chunks)])
        assert t.shape == want.shape, (i, t.shape, want.shape)
        out[f"kv_L{i:02d}"] = t.contiguous()
    st, meta = paths(g.dir)
    st.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(st))
    meta.write_text(
        json.dumps(
            {
                "golden": str(g.dir),
                "golden_content_hash": g.pinned_hash(),
                "rung": R.GOLDEN_RUNG,
                "numerics": "official",
                "dtype": str(dtype),
                "layers": layers,
                "resident_layers": max(layers) + 1,
                "chunk_seconds": times,
                "total_seconds": round(time.time() - t0, 1),
                "made": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=1,
        )
    )
    print(f"[official_kv] wrote {st} ({round(time.time() - t0, 1)} s)")
    return st


if __name__ == "__main__":
    generate()
