# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sweep's candidate matrix, as precision artifacts.

Every candidate is a full :mod:`tt.precision_config` artifact written to
``doc/datatype_sweep/configs/<id>.json`` and then handed to ``build_generator``
through ``precision_config=<path>`` -- the same required-artifact path the
shipped default uses.  Nothing here reaches into the model to poke a constant,
so a candidate that measures well is a candidate that can ship by copying one
file.

Run ``python doc/datatype_sweep/bench/candidates.py`` to (re)write them.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt import model as M  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    DEFAULT_DECODE_CCL_DTYPE,
    DEFAULT_PREFILL_CCL_DTYPE,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import DEFAULT_PRECISION  # noqa: E402

CONFIG_DIR = ROOT / "doc/datatype_sweep/configs"

LoFi = ttnn.MathFidelity.LoFi
HiFi2 = ttnn.MathFidelity.HiFi2

#: The three attention roles, as a ``by_role`` fidelity map helper.
ATTN_ROLES = ("wqkv", "attn_gate", "o_proj")
#: The 13 full-attention (NoPE) layers of this checkpoint's 52, read off the built
#: model's ``capability_report()['layer_kinds']``.  Every fourth layer from 3.
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51)
MLP_ROLES = ("mlp_gate", "mlp_up", "mlp_down")


def _by_role(roles, fidelity):
    return tuple((role, fidelity) for role in roles)


def _head_defaults() -> dict:
    return {
        "lm_head_dtype": M.LM_HEAD_DTYPE,
        "lm_head_fidelity": M.LM_HEAD_FIDELITY,
        "lm_head_fp32_acc": M.LM_HEAD_FP32_ACC,
        "lm_head_output_dtype": M.LM_HEAD_OUTPUT_DTYPE,
        "lm_head_matmul": M.LM_HEAD_MATMUL,
        "lm_head_cores": M.LM_HEAD_CORES,
        "lm_head_in0_block_w": M.LM_HEAD_IN0_BLOCK_W,
    }


def candidates() -> dict[str, dict]:
    """``config_id -> artifact``, in the order the sweep evaluates them."""
    out: dict[str, dict] = {}

    def add(config_id: str, description: str, *, policy=None, ccl=None, head=None, overrides=None):
        head_kwargs = _head_defaults()
        head_kwargs.update(head or {})
        prefill_ccl, decode_ccl = ccl or (DEFAULT_PREFILL_CCL_DTYPE, DEFAULT_DECODE_CCL_DTYPE)
        out[config_id] = pc.config_from_policy(
            config_id=config_id,
            description=description,
            policy=replace(policy or DEFAULT_PRECISION, name=config_id),
            prefill_ccl_dtype=prefill_ccl,
            decode_ccl_dtype=decode_ccl,
            decoder_overrides=overrides,
            **head_kwargs,
        )

    # ---- C00: the carried-forward optimized-full-model policy.
    add(
        "c00-baseline-attn8-mlp4-kv8-lofi",
        "Baseline: the optimized full model's shipped policy. BFP8 attention weights, BFP4 MLP "
        "weights, BFP8 KV cache, BF16 activations/residual, BFP8 prefill CCL payload, BF16 decode "
        "CCL payload, LoFi everywhere, BFP4/LoFi LM head.",
    )

    # ---- C01/C02: BFP4 attention weights, both legal fidelities.
    # The decoder stage measured BFP4 attention weights 3.1 % faster at layer PCC
    # 0.977 and declined them on that PCC; $datatype-sweep requires the decision to
    # be made on full-model top-1/top-5 instead, and requires the LoFi arm.
    add(
        "c01-attn4-mlp4-kv8-lofi",
        "BFP4 attention weights (wqkv, attn_gate, o_proj) at LoFi, everything else baseline. The "
        "BFP4+LoFi candidate for the attention weight group.",
        policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b),
    )
    add(
        "c02-attn4-mlp4-kv8-attn-hifi2",
        "BFP4 attention weights at HiFi2, MLP left at BFP4+LoFi. The fidelity comparison for the "
        "BFP4 attention group.",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            decode_math_fidelity_by_role=_by_role(ATTN_ROLES, HiFi2),
            prefill_math_fidelity_by_role=_by_role(ATTN_ROLES, HiFi2),
        ),
    )

    # ---- C03: BFP8 attention weights at HiFi2, i.e. the same-dtype fidelity
    # comparison the skill requires for the dominant decode projection groups.
    add(
        "c03-attn8-hifi2-mlp4-lofi",
        "Baseline dtypes with the BFP8 attention projections moved to HiFi2. The BFP8+HiFi2 arm of "
        "the BFP8+LoFi-vs-BFP8+HiFi2 comparison; the baseline is the BFP8+LoFi arm.",
        policy=replace(
            DEFAULT_PRECISION,
            decode_math_fidelity_by_role=_by_role(ATTN_ROLES, HiFi2),
            prefill_math_fidelity_by_role=_by_role(ATTN_ROLES, HiFi2),
        ),
    )

    # ---- C04: BFP4 MLP weights at HiFi2, the fidelity comparison for the MLP
    # group the baseline already ships at BFP4+LoFi.
    add(
        "c04-mlp4-hifi2",
        "Baseline dtypes with the BFP4 MLP projections moved to HiFi2. The BFP4+HiFi2 arm for the "
        "MLP weight group; the baseline is its BFP4+LoFi arm.",
        policy=replace(
            DEFAULT_PRECISION,
            decode_math_fidelity_by_role=_by_role(MLP_ROLES, HiFi2),
            prefill_math_fidelity_by_role=_by_role(MLP_ROLES, HiFi2),
        ),
    )

    # ---- C05: BFP4 KV cache.
    add(
        "c05-kv4",
        "Baseline weights and fidelity with the paged KV cache at BFP4. Halves cache bytes per "
        "token against BFP8, so it changes both decode SDPA bandwidth and the context contract.",
        policy=replace(DEFAULT_PRECISION, kv_cache_dtype=ttnn.bfloat4_b),
    )

    # ---- C06: BFP8 decode CCL payload.
    add(
        "c06-decode-ccl-bfp8",
        "Baseline with the decode reduce-scatter/all-gather payload at BFP8 instead of the BF16 "
        "activation dtype. Costs no extra op: the row-parallel matmul is asked for the payload "
        "dtype directly.",
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C07: BFP4 attention weights everywhere but the first and last layer.
    # The skill's restore ladder: if C01 fails full-model accuracy, the first
    # thing to try is keeping the terminal layers at the safer dtype.
    add(
        "c07-attn4-except-first-last",
        "BFP4 attention weights at LoFi on the inner 50 layers, BFP8 on layers 0 and 51. The "
        "layer-exception arm of the BFP4 attention trial.",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            layer_exceptions=(((0, 51), (("attn_weight_dtype", ttnn.bfloat8_b),)),),
        ),
    )

    # ---- C08: the stacked lower-precision candidate.  Only meaningful if its
    # parts pass; recorded either way.
    add(
        "c08-attn4-kv4-cclbfp8",
        "Everything the individual switches offer at once: BFP4 attention weights at LoFi, BFP4 KV "
        "cache, BFP8 decode CCL payload, BFP4 MLP at LoFi.",
        policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b, kv_cache_dtype=ttnn.bfloat4_b),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C09: the KV/CCL pair without the attention weight change, i.e. the
    # stack that survives if BFP4 attention weights fail accuracy.
    add(
        "c09-kv4-cclbfp8",
        "BFP4 KV cache plus a BFP8 decode CCL payload, baseline weights. The stacked candidate that "
        "does not depend on the BFP4 attention trial.",
        policy=replace(DEFAULT_PRECISION, kv_cache_dtype=ttnn.bfloat4_b),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C10: BFP8 LM head, the one *higher*-precision arm.  The LM head is
    # 190 MB/device of BFP4 weight and the largest single matmul in the step, so
    # it is where a precision restore would cost the most; measuring it prices
    # the baseline's choice rather than assuming it.
    # The head's static circular-buffer budget is dtype-scaled: at BFP8 the
    # shipped ``in0_block_w=2`` overflows L1 (1,821,824 B against 1,572,864 B,
    # ``logs/smoketest_first_pass.log``), so the BFP8 arm carries the largest
    # legal block width for its own dtype rather than being rejected on the
    # first TTNN error.  ``per_core_K = 208 / 52 = 4``, so 1, 2 and 4 are the
    # legal widths and 1 is the only one that fits.
    add(
        "c10-lm-head-bfp8",
        "Baseline with the LM head restored to BFP8 weights at LoFi, on the largest in0_block_w "
        "that fits L1 at that dtype. The higher-precision control for the terminal projection.",
        head={"lm_head_dtype": ttnn.bfloat8_b, "lm_head_in0_block_w": 1},
    )

    # ---- C11: the baseline head geometry at the baseline dtype but with the
    # narrower block, so the BFP8 arm's geometry change is attributable.  Without
    # it, C10 confounds "BFP8 weights" with "in0_block_w=1".
    add(
        "c11-lm-head-bfp4-in0bw1",
        "Baseline dtypes with the BFP4 LM head moved to in0_block_w=1. The geometry control for "
        "c10, whose BFP8 head cannot use the shipped in0_block_w=2.",
        head={"lm_head_in0_block_w": 1},
    )

    # ---- C12: BFP8 activations / residual stream.  Expected to be blocked by an
    # exact op contract; run anyway so this stage has first-hand evidence rather
    # than an inherited claim from the decoder stage.
    add(
        "c12-activations-bfp8",
        "Baseline weights with the activation and residual stream at BFP8. The decoder stage "
        "recorded this as blocked by nlp_create_qkv_heads_decode's dtype contract; re-run here so "
        "the blocker is reproduced under this stage's own evidence.",
        policy=replace(DEFAULT_PRECISION, activation_dtype=ttnn.bfloat8_b),
    )

    # ---- C13: BFP4 prefill CCL payload.  Prefill is where the collectives are
    # 33 % of the wall time, so the payload dtype is a TTFT lever even though it
    # is not a decode one.
    add(
        "c13-prefill-ccl-bfp4",
        "Baseline with the prefill reduce-scatter/all-gather payload at BFP4 instead of BFP8. A TTFT "
        "lever rather than a decode one: prefill dispatches 209 collectives.",
        ccl=(ttnn.bfloat4_b, DEFAULT_DECODE_CCL_DTYPE),
    )

    # ---- C14: the combination pass 1 pointed at and the matrix did not contain.
    # The two switches that are both fast *and* accuracy-neutral are BFP4 attention
    # weights (c01) and a BFP8 decode CCL payload (c06); the third fast one, a BFP4
    # KV cache (c05), is the only one that moves top-1 (0.990 -> 0.970) and it is
    # also the only one worth nothing on its own. c08 stacked all three and paid
    # that accuracy for its speed; this is c08 without the cache change.
    add(
        "c14-attn4-cclbfp8-kv8",
        "BFP4 attention weights at LoFi plus a BFP8 decode CCL payload, with the BFP8 KV cache "
        "kept. The two accuracy-neutral switches from the first pass, stacked.",
        policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C15: the same, with the first and last layer restored to BFP8 attention
    # weights.  The layer-exception safety variant of c14, so the ladder has a rung
    # between c14 and the baseline rather than a cliff.
    add(
        "c15-attn4-except-first-last-cclbfp8",
        "c14 with layers 0 and 51 restored to BFP8 attention weights. The layer-exception rung "
        "between the fully-BFP4 attention stack and the baseline.",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            layer_exceptions=(((0, 51), (("attn_weight_dtype", ttnn.bfloat8_b),)),),
        ),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C16: BFP4 decode CCL payload.  c06 showed BFP8 there is worth ~0.5 %
    # and costs no accuracy; BFP4 is the next rung and had not been tried.
    add(
        "c16-decode-ccl-bfp4",
        "Baseline with the decode reduce-scatter/all-gather payload at BFP4. The next rung below "
        "the BFP8 decode payload of c06.",
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat4_b),
    )

    # ---- C17: BFP4 attention weights plus a BFP4 decode CCL payload.
    add(
        "c17-attn4-cclbfp4-kv8",
        "c14 with the decode CCL payload at BFP4 instead of BFP8, KV cache kept at BFP8.",
        policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat4_b),
    )

    # ---- C18: the adapted retry for c13.  A BFP4 prefill CCL payload lands in
    # ``layernorm_pre_all_gather``, which takes BF16/BFP8/FP32 only -- but only
    # because the fractured prefill norm consumes the reduce-scatter partial
    # directly.  With that off the payload is finished by an all-gather before
    # any norm sees it, so the op contract no longer applies.  Whether that trade
    # is worth making is then a measurement rather than a crash.
    add(
        "c18-prefill-ccl-bfp4-unfractured-norm",
        "BFP4 prefill CCL payload with the fractured prefill norm disabled, which is the layout "
        "that makes the payload legal. The adapted retry for c13's op-contract failure.",
        ccl=(ttnn.bfloat4_b, DEFAULT_DECODE_CCL_DTYPE),
        overrides={"prefill_fractured_norm": False},
    )

    # ---- C19: the control for c18.  Disabling the fractured norm is itself a
    # change the previous stage measured as a win, so c18 confounds "BFP4 payload"
    # with "no fractured norm" unless the BFP8 payload is measured in the same
    # layout.
    add(
        "c19-prefill-ccl-bfp8-unfractured-norm",
        "The shipped BFP8 prefill payload with the fractured prefill norm disabled. The layout " "control for c18.",
        overrides={"prefill_fractured_norm": False},
    )

    # ---- C20: the layer-scoped KV cache.  ``logs/layer_ab.log`` shows the BFP4
    # cache's decode win is confined to the 13 full-attention layers -- the 39
    # sliding layers read a bounded window and are flat at 0.4416 ms either way --
    # while the *capacity* saving is per layer and therefore mostly in the 39.
    # So the two effects separate: put BFP4 on the sliding layers only and the
    # cache shrinks by 39/52 of the way to c05 while the layers whose reads the
    # accuracy loss most plausibly comes from keep BFP8.  Legal (the schema takes
    # ``kv_cache_dtype`` as a layer exception) and previously unmeasured.
    add(
        "c20-attn4-cclbfp8-kv4-sliding-only",
        "c14 with the KV cache at BFP4 on the 39 sliding-window layers and BFP8 on the 13 "
        "full-attention layers. Separates the cache's capacity saving (per layer, so mostly in the "
        "39) from its decode win (only in the 13) and from its accuracy cost.",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            kv_cache_dtype=ttnn.bfloat4_b,
            layer_exceptions=((FULL_ATTENTION_LAYERS, (("kv_cache_dtype", ttnn.bfloat8_b),)),),
        ),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    # ---- C21: the LM head's BFP4+HiFi2 arm.  `$datatype-sweep` asks for
    # BFP4+LoFi against BFP4+HiFi2 for every material BFP4 group, and the head is
    # one: 190 MB/device of BFP4 weight and the single largest matmul in the decode
    # step.  The two decoder-group pairs (c01/c02, c00/c04) came back bit-identical,
    # so this is very likely a formality -- but it is the comparison the skill names
    # and the matrix did not contain it.  Round 2 of the stage review.
    add(
        "c21-lm-head-bfp4-hifi2",
        "The selected policy with the BFP4 LM head at HiFi2 instead of LoFi. The BFP4+HiFi2 arm "
        "for the LM-head weight group; c14 is its BFP4+LoFi arm.",
        policy=replace(DEFAULT_PRECISION, attn_weight_dtype=ttnn.bfloat4_b),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
        head={"lm_head_fidelity": HiFi2},
    )

    # ---- C22: the other half of the layer-scoped KV lever.  `c20` puts BFP4 on
    # the 39 sliding layers, which is where the *capacity* is; this puts it on the
    # 13 full-attention layers, which is where `logs/layer_ab.log` measured the
    # whole of the *decode* win (0.5149 -> 0.5027 ms at context 131071) and where
    # a whole-cache read happens. Between them the two candidates price each half
    # of the lever separately.
    add(
        "c22-attn4-cclbfp8-kv4-full-attn-only",
        "c14 with the KV cache at BFP4 on the 13 full-attention layers and BFP8 on the 39 "
        "sliding ones -- the converse of c20, and the half that carries the long-context decode "
        "win rather than the capacity saving.",
        policy=replace(
            DEFAULT_PRECISION,
            attn_weight_dtype=ttnn.bfloat4_b,
            layer_exceptions=((FULL_ATTENTION_LAYERS, (("kv_cache_dtype", ttnn.bfloat4_b),)),),
        ),
        ccl=(DEFAULT_PREFILL_CCL_DTYPE, ttnn.bfloat8_b),
    )

    return out


def write(config_dir: pathlib.Path = CONFIG_DIR) -> list[pathlib.Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for config_id, config in candidates().items():
        path = config_dir / f"{config_id}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        # Round-trip every artifact at write time: a candidate that cannot be
        # loaded back into build kwargs is a typo, not a measurement.
        pc.build_kwargs_from_config(pc.load_precision_config(path))
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write():
        print(path)
