# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate for Call 1 (text-to-speech) of `voxtral-tts-full`.

Real prompt (HF tokenizer + voice preset) -> the chained TTNN pipeline in `tt/pipeline.py` --
the SAME function `demo/demo_tts.py` calls -> a real 24 kHz waveform, scored against the HF
reference for the same prompt, voice and frame horizon.

  Gate 1  every routed graduated stub is still real ttnn (native probe torch_ops == 0, live file
          identical to its `.last_good_native` snapshot, no runtime fallback, and the objects the
          pipeline routes through are instances of the stub classes)
  Gate 2  all 7 graduated modules are INVOKED inside the real forward path, with counts that
          match the chain's shape (not a coverage sweep -- there is no such helper in the package)
  Gate 3  final waveform PCC >= 0.99 vs the HF golden

`test_e2e_pcc` prints `e2e PCC=<value>` on its own line before the assert, pass or fail.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from models.demos.voxtral_tts_full.tt import pipeline as P
from models.demos.voxtral_tts_full.tt import reference as ref

DEMO_ROOT = pathlib.Path(P.__file__).resolve().parents[1]
STUBS = DEMO_ROOT / "_stubs"
PCC_TARGET = 0.99


# ------------------------------------------------------------------------------------ Gate 1
def test_gate1_routed_stubs_are_native(pipe):
    """Every graduated body the pipeline routes through is the graduated body, unmodified."""
    fallbacks = json.loads((DEMO_ROOT / "_runtime_fallbacks.json").read_text())
    assert fallbacks == {}, f"runtime fallbacks present: {fallbacks}"

    for name in P.GRADUATED_MODULES:
        live = (STUBS / f"{name}.py").read_bytes()
        snapshot = (STUBS / f"{name}.py.last_good_native").read_bytes()
        assert live == snapshot, f"{name}: live stub differs from its graduated snapshot"
        probe = json.loads((STUBS / f"{name}.py.native_probe.json").read_text())
        assert probe["torch_ops"] == 0, f"{name}: native probe shows torch ops {probe}"
        assert probe["ttnn_dispatch"] > 0, f"{name}: native probe shows no ttnn dispatch"

    # ... and the objects actually wired into the chain are those classes.
    from models.demos.voxtral_tts_full._stubs import attention as attention_stub
    from models.demos.voxtral_tts_full._stubs import codec_decoder as codec_stub
    from models.demos.voxtral_tts_full._stubs import decoder_layer as decoder_layer_stub
    from models.demos.voxtral_tts_full._stubs import flow_matching as flow_stub
    from models.demos.voxtral_tts_full._stubs import m_l_p as mlp_stub
    from models.demos.voxtral_tts_full._stubs import r_m_s_norm as rms_stub
    from models.demos.voxtral_tts_full._stubs import tts_backbone as backbone_stub

    layer0, layer1 = pipe.backbone_layers[0], pipe.backbone_layers[1]
    assert isinstance(pipe.backbone, backbone_stub.TtVoxtralTtsBackbone)
    assert isinstance(layer0, P.TtDecomposedLayer)
    assert isinstance(layer0.input_layernorm, rms_stub.TtVoxtralRMSNorm)
    assert isinstance(layer0.post_attention_layernorm, rms_stub.TtVoxtralRMSNorm)
    assert isinstance(layer0.self_attn, attention_stub.TtVoxtralAttention)
    assert isinstance(layer0.mlp, mlp_stub.TtVoxtralMLP)
    assert isinstance(layer1.layer, decoder_layer_stub.TtVoxtralDecoderLayer)
    assert isinstance(pipe.flow, flow_stub.TtVoxtralFlowMatching)
    assert isinstance(pipe.codec, codec_stub.TtVoxtralCodecDecoder)

    # the stack is a plain list of same-typed elements, so a structural walk can find it
    assert all(isinstance(l, P.TtBackboneStackLayer) for l in pipe.backbone_layers)
    assert len(pipe.backbone_layers) == len(pipe.hf.backbone.layers) == 26
    print(f"Gate 1 OK: {len(P.GRADUATED_MODULES)} graduated stubs native and routed")


def test_gate1_stubs_are_native_live(pipe, tt_run):
    """Gate 1, measured rather than remembered.

    The recorded sidecars were taken when the stubs graduated; the bodies they share
    (`tt_backbone.py`, `tt_common.py`) have been tightened since, so nativeness is re-measured
    HERE against the objects the pipeline actually routes through.  `run_native_probe` counts
    torch compute ops via a TorchFunctionMode and ttnn dispatches by wrapping the ops, so it
    cannot be evaded by aliasing, and it refreshes each sidecar as it goes.

    Inputs are staged BEFORE the probe: `ttnn.from_torch` is itself a torch op."""
    from models.common.native_probe import run_native_probe

    from models.demos.voxtral_tts_full import tt_common as tc

    layer0, layer1 = pipe.backbone_layers[0], pipe.backbone_layers[1]
    x = tc.stage(torch.randn(1, 64, P.DIM), pipe.device)
    h = tc.stage(torch.randn(1, 1, P.DIM), pipe.device)
    codes = tt_run["frames"].float()  # the pipeline's OWN emitted frames
    codes_dev = tc.stage(codes, pipe.device)

    cases = {
        "r_m_s_norm": lambda: layer0.input_layernorm(x),
        "attention": lambda: layer0.self_attn(x, bias=True),
        "m_l_p": lambda: layer0.mlp(x),
        "decoder_layer": lambda: layer1.layer(x, bias=True),
        "tts_backbone": lambda: pipe.backbone(x),
        "flow_matching": lambda: pipe.flow(h),
        "codec_decoder": lambda: pipe.codec(codes_dev),
    }
    results = {}
    for name, thunk in cases.items():
        _, probe = run_native_probe(STUBS / f"{name}.py", thunk)
        results[name] = probe
        assert probe["torch_ops"] == 0, (
            f"{name} is no longer native: {probe['torch_ops']} torch ops "
            f"{probe['torch_op_names']}")
        assert probe["ttnn_dispatch"] > 0, f"{name} dispatched no ttnn ops"
    print("Gate 1 (live probe) OK: " + ", ".join(
        f"{k}={v['ttnn_dispatch']}ttnn/{v['torch_ops']}torch" for k, v in results.items()))


# ------------------------------------------------------------------------------------ Gate 2
def test_gate2_all_graduated_modules_invoked(pipe, tt_run, horizon):
    """Every graduated module ran INSIDE the chain, the number of times the chain implies.

    The counts are THIS run's deltas, not the pipeline object's lifetime totals, so no other
    test's calls can stand in for a module the chain failed to invoke."""
    counts = tt_run["invoked"]
    missing = [m for m in P.GRADUATED_MODULES if counts.get(m, 0) < 1]
    assert not missing, f"graduated modules never invoked: {missing} (counts={counts})"

    n_backbone = 1 + tt_run["n_frames"]  # prompt prefill + one per emitted frame
    assert counts["tts_backbone"] == n_backbone
    assert counts["decoder_layer"] == n_backbone, "layer 1 must run once per backbone pass"
    assert counts["attention"] == n_backbone, "layer 0's attention must run once per backbone pass"
    assert counts["m_l_p"] == n_backbone
    assert counts["r_m_s_norm"] == 2 * n_backbone, "layer 0 has two norms"
    assert counts["flow_matching"] == tt_run["n_frames"] + (1 if tt_run["stopped"] else 0)
    assert counts["codec_decoder"] == 1
    print(f"Gate 2 OK: all 7 graduated modules invoked in the forward path -> {counts}")


# ------------------------------------------------------------------------------------ Gate 3
def test_e2e_pcc(tt_run, golden, horizon):
    """The gate: TT waveform vs the HF reference waveform for the same prompt and horizon."""
    tt_wave, ref_wave = tt_run["waveform"], golden["waveform"]
    tt_frames, ref_frames = tt_run["frames"], golden["frames"]

    # stage diagnostics -- a failure should say WHERE, not just that it failed
    n = min(tt_frames.shape[0], ref_frames.shape[0])
    frame_match = bool(torch.equal(tt_frames[:n].long(), ref_frames[:n].long()))
    flips = int((tt_frames[:n].long() != ref_frames[:n].long()).sum())
    hid_pcc = [ref.pcc(golden["hiddens"][:, i], tt_run["hiddens"][:, i])
               for i in range(min(tt_run["hiddens"].shape[1], golden["hiddens"].shape[1]))]
    print(f"frames: tt={tuple(tt_frames.shape)} ref={tuple(ref_frames.shape)} "
          f"exact_match={frame_match} code_flips={flips}")
    print("per-step hidden PCC: " + ", ".join(f"{p:.6f}" for p in hid_pcc))
    print(f"waveform: tt={tuple(tt_wave.shape)} ref={tuple(ref_wave.shape)} "
          f"peak_tt={tt_wave.abs().max():.4f} peak_ref={ref_wave.abs().max():.4f}")

    assert tt_frames.shape == ref_frames.shape, "TT and reference stopped at different lengths"
    assert tt_wave.shape == ref_wave.shape
    assert tt_wave.shape[-1] == tt_frames.shape[0] * P.SAMPLES_PER_FRAME

    achieved_pcc = ref.pcc(ref_wave, tt_wave)
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, (
        f"e2e waveform PCC {achieved_pcc:.6f} < {PCC_TARGET} "
        f"(code flips={flips}, worst per-step hidden PCC={min(hid_pcc):.6f})")
    assert frame_match, f"{flips} audio codes differ from the reference despite PCC {achieved_pcc:.6f}"


# ------------------------------------------------------------------ fully-on-device / trace
def test_host_op_selftest(pipe):
    """AUTHORITATIVE fully-on-device check: zero host aten ops in the model math."""
    verdict = pipe.host_op_selftest(max_frames=2)
    print(f"host_op_selftest: {verdict['reason']}")
    assert verdict["on_device"], verdict["reason"]


def test_trace_capture_selftest(pipe, device):
    """Every stage in PIPELINE_STAGES captures host-free and replays to the same answer."""
    ok = pipe.trace_capture_selftest(device)
    print(f"trace_capture_selftest: {pipe.trace_selftest_results}")
    assert ok, f"stage trace capture failed: {pipe.trace_selftest_results}"


def test_layer_cap_is_not_inert(device, hf_model):
    """`layers` must actually cap the depth built -- the perf tool proves the knob by capping and
    re-measuring, and reports it INERT if the work signal does not move."""
    capped = P.build_pipeline(device, model=hf_model, layers=2, flow_layers=1, vocode_layers=1)
    assert len(capped.backbone_layers) == 2
    assert len(capped.flow_layers) == 1
    assert capped.depths["vocode"] == 1
    assert capped.depths["backbone_total"] == 26, "the full depth must still be reported"
    # every DISTINCT op still runs: embeddings, norms and heads are intact
    out = capped.run_tts(max_frames=1, early_stop=False)
    assert out["waveform"].shape[-1] == P.SAMPLES_PER_FRAME
    print(f"layer cap OK: backbone {len(capped.backbone_layers)}/26 "
          f"flow {len(capped.flow_layers)}/3 vocode {capped.depths['vocode']}/2")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-vv"]))
