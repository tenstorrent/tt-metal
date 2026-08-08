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

        # ---- Stage: DiT denoise step (the hot loop body; timestep now device-side) ----
        # Everything device-side: xt, context_latents, encoder_hidden_states, rope, and the timestep
        # as a device tensor [1,1,1,1]. Warmup builds lazy weights outside capture.
        Tlat = 256  # T'=128 target
        xt = ttnn.from_torch(
            torch.randn(1, 1, Tlat, pipe.args.audio_acoustic_hidden_dim),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        ctxl = ttnn.from_torch(
            torch.randn(1, 1, Tlat, pipe.args.audio_acoustic_hidden_dim * 2),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        enc = ttnn.from_torch(
            torch.randn(1, 1, 128, 2048),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        tp = Tlat // PATCH
        dcos, dsin = pipe._rope_tables(tp)
        dmask = pipe._sliding_mask(tp)  # None at T'=128 (sliding==full)
        ts_dev = ttnn.from_torch(
            torch.tensor([[[[0.7]]]], dtype=torch.float32),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )
        # zero delta timestep buffer (generate uses same scalar for t and t_r -> diff 0)
        ts_r_dev = ttnn.from_torch(
            torch.tensor([[[[0.0]]]], dtype=torch.float32),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        )

        def _dit_step(hidden, ctx, ts, tsr, cos, sin, ehs):
            return pipe.dit.forward(hidden, ctx, ts, tsr, cos, sin, ehs, sliding_mask=dmask)

        _try_stage(
            "dit_step",
            device,
            eager_fn=_dit_step,
            tracer_fn=_dit_step,
            inputs={"hidden": xt, "ctx": ctxl, "ts": ts_dev, "tsr": ts_r_dev, "cos": dcos, "sin": dsin, "ehs": enc},
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

        # ---- Stage: generate() eager vs traced (the real denoise-loop integration proof) ----
        # Run the whole ODE loop both ways from the SAME noise; assert the clean latents match.
        try:
            Tg = 256
            hch = pipe.args.audio_acoustic_hidden_dim
            torch.manual_seed(0)
            noise = torch.randn(1, 1, Tg, hch)
            ctx = torch.cat([torch.zeros(1, 1, Tg, hch), torch.ones(1, 1, Tg, hch)], dim=-1)

            def _mk(x):
                return ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            enc_hs = _mk(torch.randn(1, 1, 128, 2048))
            eager_t = ttnn.to_torch(pipe.generate(_mk(noise), _mk(ctx), enc_hs, infer_steps=6, use_trace=False)).float()
            traced_t = ttnn.to_torch(pipe.generate(_mk(noise), _mk(ctx), enc_hs, infer_steps=6, use_trace=True)).float()
            pcc = _pcc(eager_t, traced_t)
            print(f"REPLAY_PCC generate_loop={pcc:.4f}")
            if pcc >= 0.99:
                print("TRACE_STAGE_PASS generate_loop")
            else:
                print(f"TRACE_STAGE_FAIL generate_loop: eager-vs-traced PCC {pcc:.4f} < 0.99")
        except Exception as e:
            print(f"TRACE_STAGE_FAIL generate_loop: {str(e).splitlines()[0][:140]}")
        sys.stdout.flush()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
