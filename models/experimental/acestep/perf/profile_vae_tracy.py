# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""VAE-only Tracy device-profiler: REAL device-kernel time of the Oobleck VAE decode.

The VAE is 99% of the pipeline device time (Tracy, run_profile_e2e). This harness isolates JUST the
VAE decode so we can iterate on it fast: build the decoder (dtype + external conv blockings applied
via tt.vae_conv_config), warm up (compile + program cache + conv weight cache), then decode one
latent under signpost("start")/("stop"). The orchestrator runs it under the Tracy device profiler and
sums device-kernel duration over the measured region (and the Conv3d share), printing METRIC lines.

    # worker (under profiler, launched by the orchestrator):
    pytest models/experimental/acestep/perf/profile_vae_tracy.py::test_profile_vae -q -s
    # orchestrator (what measure.sh calls):
    python models/experimental/acestep/perf/profile_vae_tracy.py

Env knobs (so the SAME code profiles different configs without edits — external, not hardcoded):
  ACESTEP_VAE_DTYPE = float32 (default) | bfloat16   -> VAE compute dtype
  ACESTEP_VAE_T     = latent frames (default 256 ≈ 10.24 s, the profiling reference length)

No trace capture here (kept separate from Tracy, per the trace/tracy exclusivity rule).
"""

import os

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.weight_utils import have_pipeline

SUBDIR = "acestep_vae_only"
VAE_T = int(os.environ.get("ACESTEP_VAE_T", "256"))


def _dtype():
    from models.experimental.acestep.tt.vae_conv_config import vae_default_dtype

    return vae_default_dtype()


try:
    from tracy import signpost
except ImportError:

    def signpost(_):
        pass


@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step VAE not downloaded")
def test_profile_vae(device):
    from models.experimental.acestep.tt.model_config import build_vae_decoder

    dec, cfg = build_vae_decoder(device, dtype=_dtype())

    torch.manual_seed(0)
    lat = torch.randn(1, VAE_T, cfg.decoder_input_channels)  # [B, T, 64] ROW_MAJOR
    lat_tt = ttnn.from_torch(lat, device=device, dtype=_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT)

    # ---- warmup (compile + program cache + conv weight cache), EXCLUDED from the measured region ----
    _ = dec.forward(lat_tt)
    ttnn.synchronize_device(device)

    # ---- measured region: one VAE decode ----
    signpost("start")
    out = dec.forward(lat_tt)
    ttnn.synchronize_device(device)
    signpost("stop")
    assert out is not None


def _orchestrate():
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    cmd = "pytest models/experimental/acestep/perf/profile_vae_tracy.py::test_profile_vae -q -s"
    run_device_profiler(
        cmd,
        SUBDIR,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=50000,
    )

    df = pd.read_csv(get_latest_ops_log_filename(SUBDIR))
    dur = next(c for c in df.columns if "DEVICE KERNEL DURATION" in c.upper())
    sp = df[df["OP TYPE"] == "signpost"]["OP CODE"]
    s = sp[sp == "start"].index[0]
    e = sp[sp == "stop"].index[0]
    seg = df.iloc[s + 1 : e].copy()
    seg[dur] = pd.to_numeric(seg[dur], errors="coerce")
    seg = seg.dropna(subset=[dur])

    total_us = seg[dur].sum() / 1e3
    conv = seg[seg["OP CODE"] == "Conv3dDeviceOperation"]
    conv_us = conv[dur].sum() / 1e3

    # Per-op breakdown (top cost centers) for next-iteration signal.
    bd = seg.groupby("OP CODE")[dur].agg(total_ns="sum", count="count").sort_values("total_ns", ascending=False)
    print("\n--- VAE decode device-kernel breakdown (measured region) ---")
    for op, row in bd.head(10).iterrows():
        print(f"  {op:32s} {row.total_ns/1e3:10.1f} us  x{int(row['count'])}")

    print(f"\nMETRIC vae_us={total_us:.1f}")
    print(f"METRIC vae_conv_us={conv_us:.1f}")
    print(f"METRIC vae_conv_pct={(conv_us / total_us * 100 if total_us else 0):.1f}")


if __name__ == "__main__":
    _orchestrate()
