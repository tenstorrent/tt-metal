# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy device-profiler harness for the ACE-Step DiT denoise step.

Runs one warmed DiT forward at T'=128 wrapped in Tracy signposts ("caching" -> warmup/compile,
"performance" -> the measured pass). Launch it under the Tracy device profiler to get REAL per-op
device-kernel durations (not host wall-clock, which conflates dispatch overhead and is noisy):

    python models/experimental/acestep/perf/profile_dit_tracy.py            # runs the profiled pass
    # or, to capture + post-process device kernel durations:
    python -c "from tracy.process_model_log import run_device_profiler, post_process_ops_log; \
      run_device_profiler('-m \\'python models/experimental/acestep/perf/profile_dit_tracy.py\\'', \
      'acestep_dit', device_analysis_types=['device_kernel_duration'], is_command_binary_exe=True); \
      r=post_process_ops_log('acestep_dit', sum_vals=False, has_signposts=True); \
      import pandas as pd; \
      print(r[['OP CODE','DEVICE KERNEL DURATION [ns]']].groupby('OP CODE').sum().sort_values('DEVICE KERNEL DURATION [ns]',ascending=False).head(25))"

The signpost("performance") marker lets post_process_ops_log(has_signposts=True) isolate the
measured region from the compile/cache warmup. Grouping by OP CODE shows which device kernels
(matmul, sdpa, nlp_create_qkv_heads, transpose, mul/add, rmsnorm...) actually cost the most.
"""

import torch
import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

T = 256  # T'=128 target (10.24 s)


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device, with_encoders=False)
        hch = pipe.args.audio_acoustic_hidden_dim
        torch.manual_seed(0)
        xt = ttnn.from_torch(torch.randn(1, 1, T, hch), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ctx = ttnn.from_torch(
            torch.randn(1, 1, T, hch * 2), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        enc = ttnn.from_torch(torch.randn(1, 1, 128, 2048), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tp = T // 2
        cos, sin = pipe._rope_tables(tp)
        mask = pipe._sliding_mask(tp)
        ts = ttnn.from_torch(
            torch.tensor([[[[0.7]]]], dtype=torch.float32), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )

        def step():
            return pipe.dit.forward(xt, ctx, ts, ts, cos, sin, enc, sliding_mask=mask)

        try:
            from tracy import signpost
        except ImportError:
            signpost = None

        # Warmup / compile / program-cache region.
        if signpost:
            signpost("caching")
        for _ in range(3):
            ttnn.deallocate(step())
        ttnn.synchronize_device(device)

        # Measured region: one full DiT step. (One step is enough — the device profiler captures
        # every op's kernel duration; averaging is done by post-processing multiple ops if needed.)
        if signpost:
            signpost("performance")
        out = step()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
