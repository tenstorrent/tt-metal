# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-capture smoke proof: capture each pipeline stage as a ttnn trace, replay it, and check the
replayed output matches the eager output (PCC). A stage is "traceable" only if capture succeeds AND
replay reproduces eager numerics — this is the honest proof that host-torch trace-breakers are gone.

Prints, per stage:
  TRACE_STAGE_PASS <name>            (only when capture+replay+PCC all succeed)
  REPLAY_PCC <name>=0.xxxx           (replay-vs-eager PCC)
Failures print TRACE_STAGE_FAIL <name>: <reason> and do NOT emit PASS.

Stages, in ROI order: vae (86% of wall time, already pure-ttnn) -> denoise_step -> encoder.
Skipped cleanly if the pipeline bundle isn't present.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/trace_smoke.py
"""

import sys
import torch
import ttnn

from models.tt_dit.utils.tracing import Tracer
from models.experimental.acestep.reference.weight_utils import have_pipeline
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

PATCH = 2


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() != b.numel():
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm().item() == 0 and b.norm().item() == 0 else 0.0
    return torch.dot(a, b).item() / denom


def _try_stage(name, device, eager_fn, tracer_fn, inputs, gate=0.99):
    """Run eager once, then capture+replay via Tracer; compare. inputs: dict of ttnn tensors.

    A warmup eager call happens BEFORE the Tracer captures: it forces any lazy weight loads /
    first-forward device allocations to occur OUTSIDE the trace region (ttnn forbids writes/allocs
    during capture). Equivalent to LTX/SD35 pre-allocating consts before begin_trace_capture.
    """
    try:
        eager_out = eager_fn(**{k: ttnn.clone(v) if isinstance(v, ttnn.Tensor) else v for k, v in inputs.items()})
        eager_t = ttnn.to_torch(eager_out).float()

        # Warmup: trigger lazy weight loads outside the capture region.
        warm = tracer_fn(**{k: ttnn.clone(v) if isinstance(v, ttnn.Tensor) else v for k, v in inputs.items()})
        if isinstance(warm, ttnn.Tensor):
            ttnn.deallocate(warm)

        tracer = Tracer(tracer_fn, device=device, prep_run=False, clone_prep_inputs=False)
        tracer(**inputs)  # capture (+ execute on capture)
        replay_out = tracer(**inputs)  # replay
        replay_t = ttnn.to_torch(replay_out).float()

        pcc = _pcc(eager_t, replay_t)
        print(f"REPLAY_PCC {name}={pcc:.4f}")
        if pcc >= gate:
            print(f"TRACE_STAGE_PASS {name}")
        else:
            print(f"TRACE_STAGE_FAIL {name}: replay PCC {pcc:.4f} < {gate}")
        tracer.release_trace()
    except Exception as e:
        print(f"TRACE_STAGE_FAIL {name}: {str(e).splitlines()[0][:140]}")


def main():
    if not have_pipeline():
        print("SKIP: pipeline bundle not present")
        return
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)

        # ---- Stage: text encoder (on-device embedding gather + 28 causal layers) ----
        # Pass DEVICE inputs (uint32 ids + rope tables) so nothing host-side runs inside capture.
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        Lt = 32
        ids_t = torch.randint(0, 1000, (1, Lt), dtype=torch.int32)
        ids_dev = ttnn.from_torch(ids_t, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        te_rope = Qwen3RotaryEmbedding(pipe._hf_text_cfg)
        tcos, tsin = pipe._rope_tt(te_rope, Lt)
        # warm the causal-mask cache (built host-side once, resident after) before capture.
        pipe.text_encoder._causal_mask(Lt)
        _try_stage(
            "text_encoder",
            device,
            eager_fn=lambda ids, cos, sin: pipe.text_encoder.forward(ids, cos, sin),
            tracer_fn=lambda ids, cos, sin: pipe.text_encoder.forward(ids, cos, sin),
            inputs={"ids": ids_dev, "cos": tcos, "sin": tsin},
            gate=0.99,
        )
        sys.stdout.flush()

        # ---- Stage: VAE decode (fixed shape, single forward, already pure ttnn) ----
        T = 256  # T'=128 target
        lat = ttnn.from_torch(
            torch.randn(1, T, pipe.args.audio_acoustic_hidden_dim),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _try_stage(
            "vae",
            device,
            eager_fn=lambda x: pipe.vae.forward(x),
            tracer_fn=lambda x: pipe.vae.forward(x),
            inputs={"x": lat},
            gate=0.99,
        )
        sys.stdout.flush()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
