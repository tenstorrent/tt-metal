# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-block PCC test for the ACE-Step 1.5 DiT (Block 1), S=128 (10.24 s).

Gates the whole 14-step block against the fp32 CPU reference, for **both** layer types:

  * ``sliding`` — an even layer (0, 2, ..., 22), ``sliding_attention``, TTNN window 256;
  * ``global``  — an odd layer (1, 3, ..., 23), ``full_attention``, no mask at all.

Every documented stage is printed with its oracle PCC *before* the final gate, so a
regression names the step it broke at rather than just "the block".

Two modes:

*   **default** — random-init reference weights. The oracle is ``dit_reference.block_stages``,
    which re-runs the reference block's own submodules in the reference order; the test first
    asserts that oracle reproduces ``block(...)`` at PCC 1.0.
*   **``ACE_STEP_DIT_GOLDEN=1``** — the **real converted checkpoint** for this layer, with
    inputs and per-stage oracles from ``golden/dit/s<S>/``. Only Block 0's *detail* layers
    (layer 0, and layer 1 at S in {32, 128}) carry sub-stage tensors; other layers gate on the
    ``kw_hidden_states`` / ``out`` boundary only.

The block-level PCC target (0.998) is looser than the op target (0.999) because the residual
chain accumulates bf16 error across three sub-blocks.

⚠ At S=128 the ``|i-j| <= 128`` band still covers almost the whole sequence, so this test
does **not** meaningfully exercise the window geometry — that is
``test_dit_banded_pcc.py``'s job at S=256.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v15.tests import dit_reference as R
from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import (
    AceStepDiTConfig,
    Capture,
    build_rope_tables,
    to_device,
    to_host,
)
from models.experimental.ace_step_v15.tt.ttnn_ace_step_dit import AceStepTransformerBlock

GOLDEN = str(R.GOLDEN_DIR)
TARGET_PCC = 0.998

SEQ_LEN = R.SEQ_LEN_BLOCK  # 128 -> 10.24 s
ENC_LEN = 96
SEED = 1234

#: Stages compared, in execution order. Anything not listed is still captured by name if the
#: block emits it, but these are the ones that get an assertion.
STAGES = (
    "self_attn_norm_modulated",
    "self_attn.q_pre_norm",
    "self_attn.k_pre_norm",
    "self_attn.v",
    "self_attn.q_normed",
    "self_attn.k_normed",
    "self_attn.q_rope",
    "self_attn.k_rope",
    "self_attn.sdpa",
    "self_attn.out",
    "after_self_attn",
    "cross_attn_norm",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.sdpa",
    "cross_attn.out",
    "after_cross_attn",
    "mlp_norm_modulated",
    "mlp_out",
    "out",
)


def _to_11sc(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[1, S, C]`` torch -> ``[1, 1, S, C]`` device tensor.  # BATCH-1 ASSUMPTION"""
    assert x.shape[0] == 1, "# BATCH-1 ASSUMPTION"
    return to_device(x.reshape(1, 1, *x.shape[1:]), device)


#: ``TT capture key -> golden key`` for the real-weight (``ACE_STEP_DIT_GOLDEN=1``) mode.
#: ``{i}`` is substituted with the layer index. Values needing a layout change carry a
#: transform. Note the reference's ``self_attn_norm.out`` is the *bare* norm, so the oracle for
#: the modulated value is the self-attention module's recorded input instead.
_GOLDEN_STAGE_MAP = {
    "self_attn_norm_modulated": ("layers.{i}.self_attn.kw_hidden_states", None),
    "self_attn.q_normed": ("layers.{i}.self_attn.norm_q.out", "bshd_to_bhsd"),
    "self_attn.k_normed": ("layers.{i}.self_attn.norm_k.out", "bshd_to_bhsd"),
    "self_attn.out": ("layers.{i}.self_attn.out", None),
    "cross_attn_norm": ("layers.{i}.cross_attn.kw_hidden_states", None),
    "cross_attn.k": ("layers.{i}.cross_attn.norm_k.out", "bshd_to_bhsd"),
    "cross_attn.out": ("layers.{i}.cross_attn.out", None),
    "mlp_norm_modulated": ("layers.{i}.mlp.in0", None),
    "mlp_out": ("layers.{i}.mlp.out", None),
    "out": ("layers.{i}.out", None),
}


def _golden_block_case(seq_len: int, sliding: bool):
    """Real-weight case: config, weights, inputs and per-stage oracles from ``golden/dit``."""
    goldens = R.DitGoldens(seq_len)
    config = AceStepDiTConfig.from_diffusers_config(goldens.meta["transformer_config"])
    layer_index = 0 if sliding else 1
    assert config.is_sliding(layer_index) == sliding
    # Cross-check the dump against our own layer-type derivation before trusting it.
    assert goldens.has(f"layers.{layer_index}.kw_attention_mask") == sliding, (
        "golden mask presence disagrees with layer_types: even layers are sliding_attention "
        "(dense mask dumped), odd layers are full_attention (mask is None)"
    )

    # timestep_proj is shared by every layer: timestep_proj_t comes straight from time_embed,
    # and the constant r half from time_embed_r (which the TT model folds into the weights).
    timestep_proj_t = goldens["time_embed.out1"]  # [1, 6, hidden]
    timestep_proj_r = goldens["time_embed_r.out1"][0]  # [6, hidden] constant
    if goldens.has(f"layers.{layer_index}.kw_temb"):
        # Cross-check the split whenever the detailed dump is available for this layer.
        expected = timestep_proj_t + timestep_proj_r.reshape(1, *timestep_proj_r.shape)
        assert torch.allclose(goldens[f"layers.{layer_index}.kw_temb"], expected, atol=1e-5), (
            "layers.*.kw_temb != time_embed.out1 + time_embed_r.out1 -- the dual-timestep "
            "sum is not what the r fold assumes"
        )

    # Only the detail layers (Block 0 dumps layer 0, sometimes 1) carry sub-stage tensors;
    # every layer has the boundary keys kw_hidden_states / out.
    stages = {}
    for tt_key, (golden_key, transform) in _GOLDEN_STAGE_MAP.items():
        key = golden_key.format(i=layer_index)
        if not goldens.has(key):
            continue
        value = goldens[key]
        if transform == "bshd_to_bhsd":
            value = value.permute(0, 2, 1, 3)
        stages[tt_key] = value
    assert "out" in stages, f"golden/dit/s{seq_len} has no layers.{layer_index}.out"

    # The layer sees condition_embedder's output; that value is layer-independent, so fall
    # back to the model-level dump when the per-layer kwarg was not captured.
    encoder_key = f"layers.{layer_index}.cross_attn.kw_encoder_hidden_states"
    encoder_hidden_states = goldens.get(encoder_key)
    if encoder_hidden_states is None:
        encoder_hidden_states = goldens["condition_embedder.out"]

    return {
        "config": config,
        "layer_index": layer_index,
        "state_dict": R.real_dit_state_dict(f"layers.{layer_index}"),
        "timestep_proj_r": timestep_proj_r,
        "timestep_proj_t": timestep_proj_t,
        "hidden_states": goldens[f"layers.{layer_index}.kw_hidden_states"],
        "encoder_hidden_states": encoder_hidden_states,
        "stages": stages,
        "oracle_pcc": 1.0,
        "source": f"golden/dit/s{seq_len} (real weights)",
    }


def _random_block_case(seq_len: int, sliding: bool):
    """Random-init case: build the reference block, run the staged oracle, cross-check it."""
    torch.manual_seed(SEED)
    config = AceStepDiTConfig()
    hidden = config.hidden_size
    layer_index = 0 if sliding else 1
    assert config.is_sliding(layer_index) == sliding, "even layers slide, odd layers are global"

    ref_block = R.reference_block(config, sliding=sliding, seed=SEED)
    # A non-zero time_embed_r constant, so the fold is genuinely under test rather than
    # trivially adding zero.
    ref_time_r = R.reference_timestep_embedding(config, seed=SEED + 7)
    _, timestep_proj_r = ref_time_r(torch.zeros(1))  # [1, 6, hidden]

    x_nsc = torch.randn(1, seq_len, hidden) * 0.5
    enc_nsc = torch.randn(1, ENC_LEN, hidden) * 0.5
    timestep_proj_t = torch.randn(1, 6, hidden) * 0.1
    temb = timestep_proj_t + timestep_proj_r  # what the reference block receives

    rope_ref = R.rope_for(seq_len, config)
    mask = R.sliding_mask(seq_len, config.sliding_window) if sliding else None

    stages = R.block_stages(
        ref_block,
        config,
        hidden_states=x_nsc,
        temb=temb,
        rope=rope_ref,
        attention_mask=mask,
        encoder_hidden_states=enc_nsc,
    )
    direct = ref_block(
        hidden_states=x_nsc,
        position_embeddings=rope_ref,
        temb=temb,
        attention_mask=mask,
        encoder_hidden_states=enc_nsc,
        encoder_attention_mask=None,
    )
    _, oracle_pcc = comp_pcc(direct, stages["out"], pcc=0.0)

    return {
        "config": config,
        "layer_index": layer_index,
        "state_dict": ref_block.state_dict(),
        "timestep_proj_r": timestep_proj_r[0],
        "timestep_proj_t": timestep_proj_t,
        "hidden_states": x_nsc,
        "encoder_hidden_states": enc_nsc,
        "stages": stages,
        "oracle_pcc": float(oracle_pcc),
        "source": "random-init reference",
    }


def run_dit_block_pcc(device, *, seq_len: int = SEQ_LEN, sliding: bool = True, verbose: bool = True):
    case = _golden_block_case(seq_len, sliding) if R.use_golden() else _random_block_case(seq_len, sliding)
    config = case["config"]
    hidden = config.hidden_size
    layer_index = case["layer_index"]
    stages = case["stages"]
    timestep_proj_t = case["timestep_proj_t"]
    oracle_pcc = case["oracle_pcc"]
    assert oracle_pcc > 0.99999, f"staged oracle diverges from the reference forward (pcc={oracle_pcc})"

    # ------------------------------------------------------------------------ TTNN block --
    tt_block = AceStepTransformerBlock(config, layer_index=layer_index, mesh_device=device)
    # Must be set BEFORE loading: it is folded into the modulation constants at load time.
    tt_block.timestep_proj_r_fold = case["timestep_proj_r"]
    tt_block.load_torch_state_dict(case["state_dict"])

    x_nsc = case["hidden_states"]
    enc_nsc = case["encoder_hidden_states"]
    x_tt = _to_11sc(x_nsc, device)
    enc_tt = _to_11sc(enc_nsc, device)
    rope_tt = build_rope_tables(device, seq_len, head_dim=config.head_dim, theta=config.rope_theta, dtype=ttnn.bfloat16)
    modulation = [to_device(timestep_proj_t[:, i].reshape(1, 1, 1, hidden), device) for i in range(6)]
    cross_kv = tt_block.cross_attn.compute_kv(enc_tt)

    capture = Capture(keys=STAGES)
    out_tt = tt_block(
        x_tt,
        modulation_chunks=modulation,
        rope=rope_tt,
        cross_kv=cross_kv,
        capture=capture,
    )

    # ------------------------------------------------------------------------ comparison --
    results: dict[str, float] = {}
    for name in STAGES:
        # Golden mode only has oracles for the subset in _GOLDEN_STAGE_MAP.
        if name not in capture or name not in stages:
            continue
        _, pcc = comp_pcc(stages[name].to(torch.float32), capture[name], pcc=0.0)
        results[name] = float(pcc)

    _, final_pcc = comp_pcc(stages["out"].to(torch.float32), to_host(out_tt), pcc=0.0)
    results["FINAL"] = float(final_pcc)

    if verbose:
        kind = "sliding_attention" if sliding else "full_attention"
        window = tt_block.window
        print(
            f"\n=== DiT block PCC (S={seq_len}, layer {layer_index} / {kind}, window={window}, "
            f"enc_L={enc_nsc.shape[-2]}, {case['source']}) ==="
        )
        print(f"  staged-oracle vs reference forward: pcc={oracle_pcc:.8f}")
        for name, pcc in results.items():
            flag = "ok " if pcc >= TARGET_PCC else "FAIL"
            shape = tuple(stages[name].shape) if name in stages else ()
            print(f"  [{flag}] {name:28s} pcc={pcc:.6f}  ref{shape}")

    ttnn.deallocate(out_tt)
    for t in (*modulation, *cross_kv, x_tt, enc_tt):
        ttnn.deallocate(t)

    failures = {name: pcc for name, pcc in results.items() if pcc < TARGET_PCC}
    return not failures, results, failures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("sliding", [True, False], ids=["sliding_attention", "full_attention"])
def test_dit_block_pcc(device, sliding):
    passed, _results, failures = run_dit_block_pcc(device, sliding=sliding)
    assert passed, f"DiT block PCC below {TARGET_PCC}: {failures}"


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
        for sliding in (True, False):
            ok, _res, fails = run_dit_block_pcc(dev, sliding=sliding)
            all_ok = all_ok and ok
            if fails:
                all_fails[f"sliding={sliding}"] = fails
    finally:
        ttnn.close_device(dev)
    print("PASSED" if all_ok else f"FAILED {all_fails}")
    sys.exit(0 if all_ok else 1)
