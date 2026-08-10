# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC test for the ACE-Step 1.5 DiT (Block 1), S=128 -> 768.

Block boundary (master doc §3.6):

    hidden_states  [1, T, 64]    noisy latents x_t
    context_latents[1, T, 128]   cat([src_latents(64), chunk_masks(64)], -1)
    timestep       scalar t      (timestep_r is NOT an input: it always equals t and its
                                 whole path is folded into the weights)
    encoder_hidden_states [1, enc_L, 2048]
      ->  velocity [1, T, 64]

Reference durations exercised (``duration = 2.56 * k`` gives ``S = 32 * k``):
``S=128`` (10.24 s) and ``S=768`` (61.44 s). ``T = 2 * S``.

Two modes:

*   **default** — random-init reference weights, oracles taken with torch forward hooks on
    the reference model. Self-contained; no dependency on Block 0.
    ``ACE_STEP_DIT_LAYERS`` overrides the layer count for a quick smoke run (the layer-type
    alternation is preserved, so 2 layers still covers one sliding and one global layer).
*   **``ACE_STEP_DIT_GOLDEN=1``** — the **real converted checkpoint**, driven by
    ``golden/dit/s<S>/`` (Block 0's dump: real inputs at seed 1234, all 24
    ``layers.{i}.out``, and ``transformer.out0``). No CPU forward runs at all, so this mode
    is both faster and the higher-fidelity gate. ``ACE_STEP_DIT_LAYERS`` is ignored here.

Either way the printed per-stage PCCs localise divergence to ``proj_in`` /
``condition_embedder`` / a specific layer before the final gate.

Cost warning: the random-init mode builds the fp32 CPU reference — ~6.3 GB of host weights
and a couple of minutes of CPU at S=768 for 24 layers.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    Capture,
    to_device,
    to_host,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_dit import AceStepTransformer1DModel

GOLDEN = str(R.GOLDEN_DIR)
TARGET_PCC = 0.99

SEQ_LENS = (R.SEQ_LEN_BLOCK, R.SEQ_LEN_E2E)  # 128, 768
ENC_LEN = 96
TIMESTEP = 0.9545  # the second turbo timestep at shift=3 -- a non-degenerate value
SEED = 1234


def _num_layers(config: AceStepDiTConfig) -> int:
    override = os.environ.get("ACE_STEP_DIT_LAYERS")
    return int(override) if override else config.num_hidden_layers


def _to_11sc(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, S, C]`` torch -> ``[1, 1, S, C]`` device tensor.  # BATCH-1 ASSUMPTION"""
    assert x.shape[0] == 1, "# BATCH-1 ASSUMPTION"
    return to_device(x.reshape(1, 1, *x.shape[1:]), device)


def _install_hooks(ref_model, num_layers: int) -> dict[str, torch.Tensor]:
    """Record reference sub-module outputs, keyed to match the TTNN capture dict."""
    recorded: dict[str, torch.Tensor] = {}

    def save(key, transform=None):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            recorded[key] = (transform(value) if transform else value).detach().to(torch.float32)

        return hook

    # proj_in_conv / proj_out_conv are NCL in the reference; the TTNN model is NSC.
    ref_model.proj_in_conv.register_forward_hook(save("proj_in", lambda t: t.transpose(1, 2)))
    ref_model.condition_embedder.register_forward_hook(save("condition_embedder"))
    for i in range(num_layers):
        ref_model.layers[i].register_forward_hook(save(f"layers.{i}.out"))
    return recorded


def _golden_model_case(seq_len: int):
    """Real-weight case: everything from ``golden/dit/s<S>`` + the converted checkpoint."""
    goldens = R.DitGoldens(seq_len)
    config = AceStepDiTConfig.from_diffusers_config(goldens.meta["transformer_config"])
    assert goldens.meta["dit_tokens_S"] == seq_len
    oracles = {
        # proj_in_conv is NCL in the reference; the TTNN model is NSC.
        "proj_in": goldens["proj_in_conv.out"].transpose(1, 2),
        "condition_embedder": goldens["condition_embedder.out"],
    }
    for i in range(config.num_hidden_layers):
        key = f"layers.{i}.out"
        if goldens.has(key):
            oracles[key] = goldens[key]
    return {
        "config": config,
        "state_dict": R.real_dit_state_dict(),
        "hidden_states": goldens["kw_hidden_states"],
        "context_latents": goldens["kw_context_latents"],
        "encoder_hidden_states": goldens["kw_encoder_hidden_states"],
        "timestep": goldens["kw_timestep"],
        "expected": goldens["out0"],
        "oracles": oracles,
        "source": f"golden/dit/s{seq_len} (real weights)",
    }


def _random_model_case(seq_len: int):
    """Random-init case: build the reference model and hook its submodules for oracles."""
    torch.manual_seed(SEED)
    base = AceStepDiTConfig()
    config = dataclasses.replace(base, num_hidden_layers=_num_layers(base))
    latent_t = seq_len * config.patch_size

    ref_model = R.reference_model(config, seed=SEED)
    recorded = _install_hooks(ref_model, config.num_hidden_layers)

    hidden_states = torch.randn(1, latent_t, config.audio_acoustic_hidden_dim) * 0.5
    context_latents = torch.randn(1, latent_t, config.in_channels - config.audio_acoustic_hidden_dim) * 0.5
    encoder_hidden_states = torch.randn(1, ENC_LEN, config.cross_attention_input_dim) * 0.5
    timestep = torch.tensor([TIMESTEP])

    with torch.no_grad():
        ref_out = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,  # always equal at inference: this is what makes the r fold valid
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents,
            return_dict=False,
        )[0]

    return {
        "config": config,
        "state_dict": ref_model.state_dict(),
        "hidden_states": hidden_states,
        "context_latents": context_latents,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "expected": ref_out,
        "oracles": dict(recorded),
        "source": f"random-init reference ({config.num_hidden_layers} layers)",
    }


def run_dit_pcc(device, *, seq_len: int = SEQ_LENS[0], verbose: bool = True):
    case = _golden_model_case(seq_len) if R.use_golden() else _random_model_case(seq_len)
    config = case["config"]
    num_layers = config.num_hidden_layers
    hidden_states = case["hidden_states"]
    context_latents = case["context_latents"]
    encoder_hidden_states = case["encoder_hidden_states"]
    timestep = case["timestep"]
    ref_out = case["expected"]
    oracles = case["oracles"]
    latent_t = int(hidden_states.shape[-2])

    # ------------------------------------------------------------------------ TTNN model --
    tt_model = AceStepTransformer1DModel(config, mesh_device=device)
    tt_model.load_torch_state_dict(case["state_dict"])
    tt_model.prepare_rope(seq_len)  # allocate before any trace capture

    hs_tt = _to_11sc(hidden_states, device)
    ctx_tt = _to_11sc(context_latents, device)
    enc_tt = _to_11sc(encoder_hidden_states, device)

    keys = ("proj_in", "condition_embedder", *(f"layers.{i}.out" for i in range(num_layers)), "proj_out")
    capture = Capture(keys=keys)

    cross_kv = tt_model.precompute_cross_kv(enc_tt, capture=capture)
    out_tt = tt_model(hs_tt, ctx_tt, timestep, cross_kv=cross_kv, capture=capture)
    got = to_host(out_tt)

    # ------------------------------------------------------------------------ comparison --
    results: dict[str, float] = {}
    for key in keys:
        if key in capture and key in oracles:
            _, pcc = comp_pcc(oracles[key].to(torch.float32), capture[key], pcc=0.0)
            results[key] = float(pcc)
    _, final_pcc = comp_pcc(ref_out.to(torch.float32), got, pcc=0.0)
    results["FINAL"] = float(final_pcc)

    if verbose:
        print(
            f"\n=== DiT full-model PCC (S={seq_len}, T={latent_t}, "
            f"enc_L={encoder_hidden_states.shape[-2]}, {num_layers} layers, "
            f"target {TARGET_PCC}, {case['source']}) ==="
        )
        for name, pcc in results.items():
            flag = "ok " if pcc >= TARGET_PCC else "FAIL"
            print(f"  [{flag}] {name:24s} pcc={pcc:.6f}")
        print(f"  output {tuple(got.shape)} vs reference {tuple(ref_out.shape)}")

    assert tuple(got.shape[-2:]) == (latent_t, config.audio_acoustic_hidden_dim)

    ttnn.deallocate(out_tt)
    for k, v in cross_kv:
        ttnn.deallocate(k)
        ttnn.deallocate(v)
    for t in (hs_tt, ctx_tt, enc_tt):
        ttnn.deallocate(t)
    tt_model.deallocate_weights()

    failures = {name: pcc for name, pcc in results.items() if pcc < TARGET_PCC}
    return not failures, results, failures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"S{s}" for s in SEQ_LENS])
def test_dit_pcc(device, seq_len):
    passed, _results, failures = run_dit_pcc(device, seq_len=seq_len)
    assert passed, f"DiT model PCC below {TARGET_PCC} at S={seq_len}: {failures}"


if __name__ == "__main__":
    import sys
    import time

    dev = None
    for attempt in range(20):
        try:
            dev = ttnn.open_device(device_id=0, l1_small_size=32768)
            break
        except Exception as err:  # device momentarily busy (shared with other blocks)
            print(f"open_device attempt {attempt} failed ({err}); retrying in 45s")
            time.sleep(45)
    if dev is None:
        print("FAILED could not open device")
        sys.exit(1)
    all_ok = True
    all_fails = {}
    try:
        for s in SEQ_LENS:
            ok, _res, fails = run_dit_pcc(dev, seq_len=s)
            all_ok = all_ok and ok
            if fails:
                all_fails[f"S={s}"] = fails
    finally:
        ttnn.close_device(dev)
    print("PASSED" if all_ok else f"FAILED {all_fails}")
    sys.exit(0 if all_ok else 1)
