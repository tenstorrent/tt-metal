# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN ACE-Step 1.5 condition encoder (**Block 2**, 608.37 M params).

Gate: ``encoder_hidden_states [1, enc_L, 2048]`` at **PCC >= 0.998** vs Block 0's fp32 golden.
Per-stage oracle PCCs (``text_projector`` / ``lyric_encoder`` / pooled timbre) are printed first
with ``comp_pcc(ref, got, pcc=0.0)`` so a regression localises immediately.

Two sources of truth, selected automatically:

1. **Goldens** (preferred — the real gate). Block 0's ``reference/dump_goldens.py`` output at
   ``golden/cond/s<S>/<name>.pt``: fp32, seed 1234, real 2 B turbo weights, bit-exactly
   reproducible. 19 tensors per duration; the 8 this test consumes are listed in ``_G``. Weights
   come from :func:`load_cond_state` (the converted diffusers checkpoint).
2. **Random-init reference** (fallback). ``diffusers.AceStepConditionEncoder`` at seed 1234 with
   reduced *depth* and full width — the tt_dit convention for block tests. Also the only way to
   run this test without the 2.4 GB checkpoint on disk.

Measured shapes at every reference duration: ``L_lyr = 32``, ``L_txt = 70``, timbre ``750``,
``enc_L = 103`` (102 where the caption tokenises one token shorter). ``L_lyr`` happens to be
tile-aligned; ``L_txt`` and 750 are not, so both padding paths are exercised by the real case.

Env knobs
---------
``ACE_STEP_COND_DTYPE``    ``bfloat16`` (default) | ``float32`` — TTNN weight/activation dtype.
``ACE_STEP_COND_PAD``      ``logical`` (default) | ``dense_mask`` — sequence padding strategy.
``ACE_STEP_COND_S``        golden duration bucket to use (default: the smallest present, ``32``).
``ACE_STEP_COND_RANDOM=1`` force the random-init path even when goldens exist.
``ACE_STEP_COND_LAYERS``   ``"<lyric>,<timbre>"`` depth for the random-init path (default ``2,2``).
``ACE_STEP_COND_LENS``     ``"<L_lyr>,<L_txt>,<timbre_frames>"`` for the random-init path
                           (default ``300,40,750`` — all three non-tile-aligned, and
                           ``L_lyr > sliding_window`` so the band is actually active).
``ACE_STEP_COND_STATE`` / ``ACE_STEP_PIPELINE``  weight location, see ``load_cond_state``.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tt.ttnn_ace_step_cond import (
    BLOCK1_REQUESTS,
    AceStepCondConfig,
    TTNNAceStepConditionEncoder,
    load_cond_state,
    reference_condition_encoder,
)

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "golden", "cond")
TARGET_PCC = 0.998
SEED = 1234

#: Block 0's golden names (``golden/cond/s<S>/<name>.pt``, 19 tensors per duration). ``kw_*`` are
#: forward kwargs, ``in0``/``out`` are submodule I/O, ``out0``/``out1`` the two return values.
_G = {
    "text_hidden_states": "condition_encoder.kw_text_hidden_states",
    "lyric_hidden_states": "condition_encoder.kw_lyric_hidden_states",
    "timbre_latents": "condition_encoder.kw_refer_audio_acoustic_hidden_states_packed",
    "text_attention_mask": "condition_encoder.kw_text_attention_mask",
    "lyric_attention_mask": "condition_encoder.kw_lyric_attention_mask",
    "refer_audio_order_mask": "condition_encoder.kw_refer_audio_order_mask",
    "text_projector_out": "condition_encoder.text_projector.out",
    "lyric_encoder_out": "condition_encoder.lyric_encoder.out",
    "timbre_pooled": "condition_encoder.timbre_encoder.out0",
    "encoder_hidden_states": "condition_encoder.out0",
    "encoder_attention_mask": "condition_encoder.out1",
}

_STAGES = ("text_projector_out", "lyric_encoder_out", "timbre_pooled", "encoder_hidden_states")


def _golden_dirs():
    """Available ``s<S>`` buckets, smallest S first."""
    if not os.path.isdir(GOLDEN):
        return []
    buckets = [d for d in os.listdir(GOLDEN) if d.startswith("s") and d[1:].isdigit()]
    return sorted(buckets, key=lambda d: int(d[1:]))


def _resolve_bucket():
    buckets = _golden_dirs()
    if not buckets:
        return None
    wanted = os.environ.get("ACE_STEP_COND_S")
    if wanted:
        assert f"s{wanted}" in buckets, f"no golden bucket s{wanted}; have {buckets}"
        return f"s{wanted}"
    return buckets[0]


def _load(bucket, key):
    path = os.path.join(GOLDEN, bucket, f"{_G[key]}.pt")
    return torch.load(path, map_location="cpu", weights_only=True) if os.path.exists(path) else None


def _have_goldens(bucket):
    if bucket is None or os.environ.get("ACE_STEP_COND_RANDOM") == "1":
        return False
    return all(
        _load(bucket, k) is not None for k in ("text_hidden_states", "lyric_hidden_states", "encoder_hidden_states")
    )


# --------------------------------------------------------------------------------------------- #
#                                      case construction                                        #
# --------------------------------------------------------------------------------------------- #


def _case_from_goldens(bucket):
    config = AceStepCondConfig()
    inputs = {
        k: _load(bucket, k)
        for k in (
            "text_hidden_states",
            "lyric_hidden_states",
            "timbre_latents",
            "text_attention_mask",
            "lyric_attention_mask",
            "refer_audio_order_mask",
        )
    }
    for k in ("text_hidden_states", "lyric_hidden_states", "timbre_latents"):
        inputs[k] = inputs[k].float()
    ref = {name: _load(bucket, name).float() for name in _STAGES if _load(bucket, name) is not None}
    ref["encoder_attention_mask"] = _load(bucket, "encoder_attention_mask")
    return inputs, ref, config, load_cond_state(), f"golden/cond/{bucket} (Block 0 dump, real weights)"


def _case_from_random_reference():
    n_lyric, n_timbre = (int(x) for x in os.environ.get("ACE_STEP_COND_LAYERS", "2,2").split(","))
    l_lyr, l_txt, n_frames = (int(x) for x in os.environ.get("ACE_STEP_COND_LENS", "300,40,750").split(","))
    config = AceStepCondConfig(
        num_lyric_encoder_hidden_layers=n_lyric,
        num_timbre_encoder_hidden_layers=n_timbre,
    )
    model = reference_condition_encoder(config, seed=SEED)

    gen = torch.Generator().manual_seed(SEED)
    inputs = {
        "text_hidden_states": torch.randn(1, l_txt, config.text_hidden_dim, generator=gen),
        "lyric_hidden_states": torch.randn(1, l_lyr, config.text_hidden_dim, generator=gen),
        "timbre_latents": torch.randn(1, n_frames, config.timbre_hidden_dim, generator=gen),
        "text_attention_mask": torch.ones(1, l_txt, dtype=torch.bool),
        "lyric_attention_mask": torch.ones(1, l_lyr, dtype=torch.bool),
        "refer_audio_order_mask": torch.arange(1),
    }

    with torch.no_grad():
        text_out = model.text_projector(inputs["text_hidden_states"])
        lyric_out = model.lyric_encoder(
            inputs_embeds=inputs["lyric_hidden_states"], attention_mask=inputs["lyric_attention_mask"]
        )
        timbre_unpack, _ = model.timbre_encoder(inputs["timbre_latents"], inputs["refer_audio_order_mask"])
        enc, enc_mask = model(
            text_hidden_states=inputs["text_hidden_states"],
            text_attention_mask=inputs["text_attention_mask"],
            lyric_hidden_states=inputs["lyric_hidden_states"],
            lyric_attention_mask=inputs["lyric_attention_mask"],
            refer_audio_acoustic_hidden_states_packed=inputs["timbre_latents"],
            refer_audio_order_mask=inputs["refer_audio_order_mask"],
        )
    ref = {
        "text_projector_out": text_out,
        "lyric_encoder_out": lyric_out,
        "timbre_pooled": timbre_unpack,
        "encoder_hidden_states": enc,
        "encoder_attention_mask": enc_mask,
    }
    return inputs, ref, config, model.state_dict(), f"random-init reference (seed {SEED}, {n_lyric}+{n_timbre} layers)"


# --------------------------------------------------------------------------------------------- #
#                                           the run                                             #
# --------------------------------------------------------------------------------------------- #


def run_cond_pcc(device, verbose=True):
    dtype = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[os.environ.get("ACE_STEP_COND_DTYPE", "bfloat16")]
    pad_mode = os.environ.get("ACE_STEP_COND_PAD", "logical")
    bucket = _resolve_bucket()
    inputs, ref, config, state_dict, source = (
        _case_from_goldens(bucket) if _have_goldens(bucket) else _case_from_random_reference()
    )

    l_lyr = inputs["lyric_hidden_states"].shape[1]
    l_txt = inputs["text_hidden_states"].shape[1]
    enc_l = l_lyr + 1 + l_txt  # BATCH-1 ASSUMPTION: n_timbre == 1

    # The B=1 degenerate packing must reproduce the reference's own output exactly, on the host.
    gold_enc = ref["encoder_hidden_states"]
    assert gold_enc.shape[1] == enc_l, (
        f"BATCH-1 ASSUMPTION broken: golden enc_L={gold_enc.shape[1]} but "
        f"L_lyr + 1 + L_txt = {enc_l} — _pack_sequences is no longer the identity"
    )
    if "lyric_encoder_out" in ref and "timbre_pooled" in ref and "text_projector_out" in ref:
        concat = torch.cat([ref["lyric_encoder_out"], ref["timbre_pooled"], ref["text_projector_out"]], dim=1)
        assert torch.equal(concat, gold_enc), (
            "concat(lyric, timbre, text) != golden encoder_hidden_states — the B=1 `_pack_sequences` "
            "identity assumption does not hold for this case"
        )
    assert bool(
        ref["encoder_attention_mask"].reshape(-1).bool().all()
    ), "golden encoder_attention_mask is not all-ones — the B=1 padding='longest' assumption is broken"

    if verbose:
        print(f"\nsource:   {source}")
        print(f"dtype:    {dtype}   pad_mode: {pad_mode}")
        print(f"shapes:   L_lyr={l_lyr}  n_timbre=1  L_txt={l_txt}  ->  enc_L={enc_l}")
        print(f"          timbre input {tuple(inputs['timbre_latents'].shape)}")
        print(
            f"layers:   lyric {config.num_lyric_encoder_hidden_layers}, "
            f"timbre {config.num_timbre_encoder_hidden_layers}"
        )
        for req in BLOCK1_REQUESTS:
            print(f"  block1-request: {req.split(':')[0]}")

    encoder = TTNNAceStepConditionEncoder(config, mesh_device=device, dtype=dtype, pad_mode=pad_mode)
    incompatible = encoder.load_torch_state_dict(state_dict, strict=True)
    if verbose and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"load: missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")

    enc_tt, enc_mask, parts = encoder(
        inputs["text_hidden_states"],
        inputs["lyric_hidden_states"],
        inputs["timbre_latents"],
        text_attention_mask=inputs.get("text_attention_mask"),
        lyric_attention_mask=inputs.get("lyric_attention_mask"),
        refer_audio_order_mask=inputs.get("refer_audio_order_mask"),
        return_parts=True,
    )
    lyric_out_tt, timbre_out_tt, text_out_tt = parts

    def host(x, rows):
        # parts are [1, 1, S_padded, C]; the reference stages are [1, L, C].
        return ttnn.to_torch(x).float().reshape(1, -1, x.shape[-1])[:, :rows, :]

    got = {
        "text_projector_out": host(text_out_tt, l_txt),
        "lyric_encoder_out": host(lyric_out_tt, l_lyr),
        "timbre_pooled": host(timbre_out_tt, 1),
        "encoder_hidden_states": host(enc_tt, enc_l),
    }

    assert enc_mask.shape == (1, enc_l), f"encoder_attention_mask shape {tuple(enc_mask.shape)} != (1, {enc_l})"
    assert bool(enc_mask.all()), "BATCH-1 ASSUMPTION: encoder_attention_mask must be all-ones"

    # per-stage oracle PCCs (pcc=0.0 -> report only, never gate)
    for stage in _STAGES:
        if stage not in ref:
            if verbose:
                print(f"{stage:<24} (no golden — skipped)")
            continue
        r = ref[stage].reshape(got[stage].shape)
        _, msg = comp_pcc(r, got[stage], pcc=0.0)
        if verbose:
            print(f"{stage:<24} {tuple(got[stage].shape)}  pcc: {msg}")

    gold = gold_enc.reshape(got["encoder_hidden_states"].shape)
    passed, msg = comp_pcc(gold, got["encoder_hidden_states"], pcc=TARGET_PCC)
    if verbose:
        print(f"GATE encoder_hidden_states {tuple(gold.shape)}  pcc: {msg}  (target {TARGET_PCC})")
    return passed, msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_cond_pcc(device):
    passed, msg = run_cond_pcc(device)
    assert passed, f"condition encoder PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        ok, msg = run_cond_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
