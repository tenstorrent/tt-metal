# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 end-to-end gate: real prompt -> chained TTNN pipeline -> real 24 kHz waveform.

    ./python_env/bin/python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_e2e_tts.py -s

Gate 1  every routed graduated stub is still real ttnn (native probes + a live host-op observation)
Gate 2  every one of the 7 graduated modules is INVOKED inside the real forward path
Gate 3  the final waveform's PCC against the HF golden is >= 0.99, over the WHOLE rollout

The pipeline under test is `tt/pipeline.py::run_tts` -- the same function `demo/demo_tts.py`
calls. There is no second copy of the wiring here.

THE ROLLOUT IS 8 FRAMES -- the reference `forward`'s own default safety cap -- and the stop rule
is the model's (`codes[0,0] == config.end_audio_id`), applied identically to both sides.

WHY THE COMPARISON RUNS THE WHOLE WAY, WHICH IS NOT FREE EITHER. This model does not degrade
gracefully, so a port either tracks it exactly or diverges:

  * Block 2 rounds 36 floats onto 21 FSQ levels every frame, and 1-2 of those 36 dimensions per
    frame land within 1e-3 of a rounding boundary (in the scaled 0..20 units).
  * A flipped code is not a small error. Adjacent FSQ codes index unrelated learned rows of the
    audio embedding table (|d|/|r| = 0.335), so one flip moves the fed-back frame embedding ~5%,
    which moves the next hidden state, which flips more codes. The rollout is a feedback system
    and a single early flip is amplified, not averaged away.

So the honest measurement is the whole rollout, and the way to pass it is to leave the hidden
state accurate enough that NO code flips -- which is what `_stubs/attention.py`'s numerics notes
are about. An earlier revision of this port sat at 3.8e-3 relative error on the prefill hidden
state, flipped 36 of 288 acoustic codes over these 8 frames, and its full-rollout PCC was 0.9797
and seed-dependent. Composing the fused reductions out of `ttnn.mean` and carrying matmul
operands as hi/lo bfloat16 pairs took that to 4.9e-4, zero flips, and 0.9999.

`test_report_divergence_curve` prints the per-horizon curve on every run, so a regression shows
up as the frame at which the codes start to flip rather than as one number moving.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import ttnn

from models.demos.voxtral_tts_full.tt.pipeline import (
    GRADUATED_COMPONENTS,
    TRACE_REGION_SIZE,
    build_pipeline,
    load_hf_model,
    pcc,
)
from models.demos.voxtral_tts_full.tests.e2e.reference import golden

_DEMO_ROOT = Path(__file__).resolve().parents[2]
_STUBS = _DEMO_ROOT / "_stubs"

# The reference `forward`'s own default frame cap. The stop rule is the model's
# (`codes[0,0] == config.end_audio_id`), applied identically to both sides, with this as the
# safety cap so a non-terminating run cannot hang the gate.
MAX_FRAMES = 8
# The ODE initial condition is drawn from N(0,1) as real inference does, seeded so the test is
# deterministic and both sides integrate the same trajectory.
SEED = 0
SAMPLES_PER_FRAME = 1920
PCC_GATE = 0.99


@pytest.fixture(scope="module")
def hf():
    return load_hf_model()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def pipe(device, hf):
    return build_pipeline(device, model=hf)


@pytest.fixture(scope="module")
def run(pipe):
    """ONE real run of the chained pipeline, shared by every gate."""
    pipe.reset_counts()
    enc = pipe.encode_inputs(max_frames=MAX_FRAMES, seed=SEED)
    out = pipe.run_tts(enc)
    assert out["waveform"] is not None, "the pipeline emitted [END_AUDIO] on the first frame"
    return {
        "enc": enc,
        "out": out,
        "waveform": ttnn.to_torch(out["waveform"]).float(),
        "frames": ttnn.to_torch(out["frames"]).to(torch.int64),
        "counts": dict(pipe.invocations),
    }


@pytest.fixture(scope="module")
def ref(pipe, hf, run):
    enc = run["enc"]
    out, path = golden(
        lambda: hf, enc["ids"], enc["voice"], enc["max_frames"],
        enc["x0_bank_host"], int(pipe.config.end_audio_id),
    )
    print(f"\n[golden] {path} | n_frames={out['n_frames']}")
    return out


# ------------------------------------------------------------------------------------------
# Gate 1 -- every routed graduated stub is still real ttnn
# ------------------------------------------------------------------------------------------
def test_gate1_stubs_are_native_ttnn():
    """Each routed stub has a graduation snapshot and a native probe reporting zero torch ops."""
    for name in GRADUATED_COMPONENTS:
        assert (_STUBS / f"{name}.py").is_file(), f"{name}: no stub"
        snapshots = [s for s in ("last_good_native", "last_good_sharded") if (_STUBS / f"{name}.py.{s}").is_file()]
        assert snapshots, f"{name}: no graduation snapshot"
        probe = json.loads((_STUBS / f"{name}.py.native_probe.json").read_text())
        assert probe["torch_ops"] == 0, f"{name}: torch fallback -- {probe['torch_op_names']}"
        assert probe["ttnn_dispatch"] > 0, f"{name}: no ttnn dispatch"
        print(f"[gate1] {name:16s} {snapshots[0]:18s} ttnn_dispatch={probe['ttnn_dispatch']:5d} torch_ops=0")


def test_gate1_forward_is_on_device(pipe):
    """The AUTHORITATIVE check: the model math fires ZERO host aten ops.

    Input encoding and the weight build are outside the observed region; the prefix embedding,
    the prefill, every decode step, the frame embedding and the vocoder are inside it.
    """
    verdict = pipe.host_op_selftest(max_frames=2)
    print(f"[gate1] host-op verdict: {verdict['reason']}")
    assert verdict["on_device"], f"host compute in the forward: {verdict['host_ops']}"


# ------------------------------------------------------------------------------------------
# Gate 2 -- every graduated module is INVOKED in the real forward path
# ------------------------------------------------------------------------------------------
def test_gate2_all_graduated_modules_invoked(run, pipe):
    counts = run["counts"]
    n_layers = len(pipe.backbone.layers)
    n_steps = run["out"]["n_frames"]

    for name in GRADUATED_COMPONENTS:
        print(f"[gate2] {name:16s} invoked {counts.get(name, 0):5d}x")
    missing = [n for n in GRADUATED_COMPONENTS if counts.get(n, 0) == 0]
    assert not missing, f"graduated modules never invoked: {missing}"

    # The counts must be the ones the real chain produces -- a coverage sweep would show 1 each.
    assert counts["tts_backbone"] == 1
    assert counts["decoder_layer"] == n_layers
    assert counts["attention"] == n_layers * (1 + n_steps)
    assert counts["m_l_p"] == n_layers * n_steps
    assert counts["r_m_s_norm"] == (2 * n_layers + 1) * n_steps
    assert counts["flow_matching"] == n_steps
    assert counts["codec_decoder"] == 1


def test_gate2_output_actually_depends_on_the_chain(run):
    """The waveform is the vocoder's output over the codes the loop really produced."""
    frames, waveform = run["frames"], run["waveform"]
    assert frames.shape == (run["out"]["n_frames"], 37)
    assert waveform.shape[-1] == frames.shape[0] * SAMPLES_PER_FRAME
    assert torch.isfinite(waveform).all() and waveform.abs().max() > 0


# ------------------------------------------------------------------------------------------
# Port fidelity -- the stable, horizon-independent measurements
# ------------------------------------------------------------------------------------------
def test_prefill_hidden_pcc(run, ref):
    achieved = pcc(ttnn.to_torch(run["out"]["prefill_hidden"]).float(), ref["prefill_hidden"])
    print(f"[fidelity] prefill hidden PCC={achieved:.6f}")
    assert achieved >= PCC_GATE


def test_semantic_codes_exact_over_full_rollout(run, ref):
    """The semantic codebook carries the linguistic content, and it is decided by an argmax --
    robust to the arithmetic floor where the FSQ rounding is not. Exact over the WHOLE rollout."""
    got, want = run["frames"], ref["frames"]
    n = min(got.shape[0], want.shape[0])
    exact = int((got[:n, 0] == want[:n, 0]).sum())
    print(f"[fidelity] semantic codes exact: {exact}/{n} frames  got={got[:n,0].tolist()}")
    assert exact == n, f"semantic codes diverged: {got[:n,0].tolist()} vs {want[:n,0].tolist()}"


def test_report_divergence_curve(run, ref):
    """Printed on every run: where the model's own amplification takes over. Never asserts."""
    got, want = run["frames"], ref["frames"]
    n = min(got.shape[0], want.shape[0])
    flips = [int((got[i, 1:] != want[i, 1:]).sum()) for i in range(n)]
    print(f"[curve] acoustic code flips per frame (of 36): {flips}")
    for k in range(1, n + 1):
        p = pcc(run["waveform"][..., : k * SAMPLES_PER_FRAME], ref["waveform"][..., : k * SAMPLES_PER_FRAME])
        print(f"[curve] horizon {k} frame(s) ({k/12.5:.2f}s): cumulative flips={sum(flips[:k]):3d}  PCC={p:.6f}")
    print(f"[curve] full-rollout ({n} frames) PCC={pcc(run['waveform'], ref['waveform']):.6f}")


# ------------------------------------------------------------------------------------------
# Gate 3 -- final output PCC vs the HF golden
# ------------------------------------------------------------------------------------------
def test_gate3_e2e_waveform_pcc(run, ref):
    assert run["out"]["n_frames"] == ref["n_frames"], (
        f"TT emitted {run['out']['n_frames']} frames, the reference {ref['n_frames']} -- the "
        f"stop rule diverged, so the two are not comparable"
    )
    achieved_pcc = pcc(run["waveform"], ref["waveform"])
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_GATE, f"e2e waveform PCC {achieved_pcc} < {PCC_GATE}"
