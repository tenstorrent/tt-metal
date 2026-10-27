# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the traced KV-cached decode step of the XTTS-v2 GPT core.

Everything here drives TTNNGPTTracedDecoder, which is what a request runs: one-shot prefill into
the cache, then the trace-replayed single-token step.

  * latents     — step through the reference `inputs_embeds` and stack the per-step latents, which
    must match the CPU reference's one-pass latents. Causal attention makes decode step t the same
    quantity as position t of a wide pass, so any drift is the cache or the arithmetic.
  * odd max_seq — BUG-1 regression: an odd-tile cache request used to collapse decode to PCC ~0.63.
  * head        — the sampling head is host-side, so this feeds the device's latents through it and
    checks the greedy choice against the reference's. bf16 and fp32 order near-ties differently, so
    a differing choice only counts when the reference's own margin says it was not a tie.

The reference is computed live in-process (see tests/reference_helpers.py); set XTTS_GOLDEN_DIR to
use stored fixtures instead.

Run:
    pytest -svv models/experimental/xtts_v2/tests/pcc/test_gpt_decode_pcc.py
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head
from models.experimental.xtts_v2.tests.reference_helpers import gpt_generate_reference, gpt_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder

TARGET_PCC = 0.999  # native bf16 decode path
TRACE_REGION = 50_000_000


def _decoder(device, max_seq):
    return TTNNGPTTracedDecoder(device, preprocess_gpt_parameters(device, dtype=ttnn.bfloat16), max_seq=max_seq)


def _step(dec, device, row, pos):
    """One traced decode step, host [1,1,1024] row in -> host latent out."""
    emb = ttnn.from_torch(
        row.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=dec.mesh_mapper
    )
    return ttnn.to_torch(dec.step_device(emb, pos)).to(torch.float32)


def run_decode_pcc(device, max_seq_request=None):
    """max_seq_request exercises the BUG-1 path (odd tile counts); None sizes tightly to S."""
    ref = gpt_reference()
    inputs_embeds, golden = ref["inputs_embeds"], ref["latents"]
    S = inputs_embeds.shape[1]

    dec = _decoder(device, max_seq_request or ((S + 31) // 32) * 32)
    dec.reset_caches()
    dec.capture()
    # A shape _decode_matmul_cfg cannot express falls back to ttnn's heuristic, which stays correct
    # and so passes the gate below while costing throughput. Catch it here.
    missing = [n for n, c in dec._prg.items() if c is None]
    assert not missing, f"decode matmuls fell back to the default program config: {missing}"
    assert dec._prg["c_fc"].fused_activation, "c_fc lost its fused gelu, which costs a second kernel"

    lat = [_step(dec, device, inputs_embeds[:, t : t + 1, :], t) for t in range(S)]
    decoded = torch.cat(lat, dim=1)  # [1,S,1024]
    passed, pcc_msg = comp_pcc(golden, decoded, pcc=TARGET_PCC)
    _, allclose_msg = comp_allclose(golden, decoded)
    print(f"  max_seq={dec.max_seq:4d} steps={S}  pcc: {pcc_msg}  {allclose_msg}")
    return passed, pcc_msg


def _head(latents, heads):
    """Latents -> code scores, the host-side sampling head."""
    return latents @ heads["mel_head_w"].t() + heads["mel_head_b"]


def _decided_flips(ref_logits, logits, ref_codes, picks):
    """Disagreeing steps the reference had actually resolved -> (their margins, the tie tolerance).

    The tolerance is read from how far the two sides' scores sit apart on this run, so a more
    accurate device makes the check stricter on its own."""
    tol = 2 * (logits - ref_logits).abs().flatten().quantile(0.999).item()
    steps = (picks != ref_codes).nonzero().flatten()
    margins = ref_logits[0, steps, ref_codes[steps]] - ref_logits[0, steps, picks[steps]]
    return margins[margins > tol], tol


def run_head_argmax(device):
    """Prefill the reference prompt, replay its per-step inputs, then run the host head."""
    g = gpt_generate_reference()
    heads = load_gen_head()
    prompt, step_inputs = g["prompt_embeds"], g["step_inputs"]
    P, T = prompt.shape[1], step_inputs.shape[1]

    dec = _decoder(device, ((P + T + 31) // 32) * 32)
    dec.reset_caches()
    dec.prefill(prompt)  # one-shot, as a request does
    dec.capture()  # after prefill: it leaves the prompt's K/V intact
    latents = torch.cat([_step(dec, device, step_inputs[:, m : m + 1, :], P + m) for m in range(T)], dim=1)

    logits, ref_logits = _head(latents, heads), _head(g["ref_latents"], heads)
    lat_ok, lat_msg = comp_pcc(g["ref_latents"], latents, pcc=TARGET_PCC)
    ref_codes = g["ref_codes"].flatten()
    picks = logits.argmax(-1).flatten()
    agree = (picks == ref_codes).float().mean().item()
    decided, tol = _decided_flips(ref_logits, logits, ref_codes, picks)
    msg = f"latents {lat_msg}, argmax agreement {agree * 100:.1f}%, {len(decided)} decided flips"
    print(f"  head: {msg} (tie tolerance {tol:.4f}, {len(ref_codes)} steps)")
    return lat_ok and len(decided) == 0, msg


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_decode_pcc(device):
    passed, pcc_msg = run_decode_pcc(device)
    assert passed, f"traced decode PCC below {TARGET_PCC}: {pcc_msg}"


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_decode_pcc_large_odd_max_seq(device):
    """BUG-1 regression: an odd-tile max_seq (736 = 23 tiles) used to collapse decode to PCC ~0.63.
    The decoder rounds up to an even tile count (768), so this must stay above target."""
    passed, pcc_msg = run_decode_pcc(device, max_seq_request=736)
    assert passed, f"BUG-1 regression: odd-tile max_seq decode PCC below {TARGET_PCC}: {pcc_msg}"


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_head_argmax_agreement(device):
    passed, msg = run_head_argmax(device)
    assert passed, f"the head disagreed with the reference's greedy choice: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        dev.enable_program_cache()
        results = [run_decode_pcc(dev), run_decode_pcc(dev, max_seq_request=736), run_head_argmax(dev)]
    finally:
        ttnn.close_device(dev)
    ok = all(r[0] for r in results)
    print(("PASSED " if ok else "FAILED ") + "; ".join(str(r[1]) for r in results))
    sys.exit(0 if ok else 1)
