# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek's reference at its own shipped precision (numerics official, dtype reference.shipped_dtype: bf16 with its
fp8 / fp4 quantizers), the final norm of every position of a ladder rung's last chunk (FIX.20, owner 2026-11-01): the
margin of the logit-KL check (tests/bringup/test_logit_kl.py), fed exactly as the ladder feeds the device
(reference/drift.py:replay: every chunk from position 0, or the golden's state prefix loaded for a
``prefix_from_golden`` rung).

    <golden dir>-official/final_norm-<rung>.safetensors   key "final_norm" [chunk, hidden] fp32 (bf16 values)
    <golden dir>-official/final_norm-<rung>.json          golden content_hash, rung, first chunk, sha256, seconds

beside the golden (the ``<rung dir>-official`` place reference/golden.py names for official-precision outputs; the
golden itself is not touched; the rung is in the name because rung ``last`` and ``s56320`` share one golden but not one
feed). Made once per golden and rung (CPU only, no device):

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.official_final_norm <rung>

A file made from another golden (content_hash) or whose own sha256 does not match is refused, never used.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch


def paths(golden_dir, rung: str) -> tuple[Path, Path]:
    d = Path(str(golden_dir) + "-official")
    return d / f"final_norm-{rung}.safetensors", d / f"final_norm-{rung}.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(golden, rung: str) -> tuple[torch.Tensor, dict]:
    """([chunk, hidden] official final norm of the rung's last chunk, its meta) for ``golden`` (Golden)."""
    from safetensors.torch import load_file

    st, meta = paths(golden.dir, rung)
    if not st.exists():
        raise FileNotFoundError(
            f"{st}: no official final norm for rung {rung} (FIX.20); make it: PYTHONPATH=$PWD python -m "
            f"models.demos.deepseek_v41_flash_d_p.tests.bringup.official_final_norm {rung}"
        )
    m = json.loads(meta.read_text())
    if m.get("golden_content_hash") != golden.pinned_hash():
        raise ValueError(f"{st} was made from golden {m.get('golden_content_hash')}, not {golden.pinned_hash()}")
    if m.get("sha256") != _sha(st):
        raise ValueError(f"{st}: content hash does not match its {meta.name}")
    return load_file(str(st))["final_norm"], m


def generate(rung: str) -> Path:
    from safetensors.torch import save_file

    from models.demos.common.bringup.reference.drift import replay
    from models.demos.common.bringup.reference.golden import Golden, load_spec, reference_at

    s = load_spec()
    r = s.rung(rung)
    g = Golden.for_rung(s, rung)
    if g.manifest.get("numerics") != "native":
        raise SystemExit(f"{g.dir}: numerics {g.manifest.get('numerics')!r}, want the native golden")
    first = g.n_chunks - 1 if r.get("prefix_from_golden") else 0
    dtype = getattr(torch, str(s.get("reference.shipped_dtype")))
    t0 = time.time()
    ref = reference_at(s, "official", layers=None, dtype=dtype)
    t_load = time.time() - t0
    rep = replay(s, ref, g, list(range(s.num_layers)), first)
    fn = rep["final_norm"].float().reshape(g.chunk, -1).contiguous()
    secs = round(time.time() - t0, 1)
    st, meta = paths(g.dir, rung)
    st.parent.mkdir(parents=True, exist_ok=True)
    save_file({"final_norm": fn}, str(st))
    meta.write_text(
        json.dumps(
            {
                "golden": str(g.dir),
                "golden_content_hash": g.pinned_hash(),
                "rung": rung,
                "first_chunk": first,
                "positions": [(g.n_chunks - 1) * g.chunk, g.n_chunks * g.chunk],
                "numerics": "official",
                "dtype": str(dtype),
                "sha256": _sha(st),
                "load_seconds": round(t_load, 1),
                "total_seconds": secs,
                "made": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=1,
        )
    )
    print(f"[official_final_norm] wrote {st} ({secs} s)", flush=True)
    return st


if __name__ == "__main__":
    for name in sys.argv[1:] or ["s10240", "last"]:
        generate(name)
