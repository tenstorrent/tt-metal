# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end gates for the FLUX.2-klein-9B text-encoder TTNN pipeline.

Two task heads, one shared chain (`tt/pipeline.py`), three gates:

  Gate 1  every routed graduated stub is still real ttnn, and the bodies that
          graduated SHARDED are still sharded (not quietly replicated).
  Gate 2  every graduated module is INVOKED inside the real forward path, with
          its output feeding downstream compute -- observed, never driven.
  Gate 3  the final task output PCC vs the HF golden is >= 0.95.

Run:  ./python_env/bin/python -m pytest \
        models/demos/flux_2_klein_9b_text_encoder/tests/e2e/test_e2e_pipeline.py -s
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.flux_2_klein_9b.text_encoder.tt import model_ref
from models.demos.flux_2_klein_9b.text_encoder.tt.pipeline import (
    GRADUATED_COMPONENTS,
    PIPELINE_STAGES,
    build_pipeline,
    graduated_invocation_probe,
)

TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))
PCC_TARGET = 0.95

_BRINGUP = Path("models/tt_transformers/demo/flux_2_klein_9b_text_encoder")
_STUBS = _BRINGUP / "_stubs"
_PIPELINE_SRC = Path("models/demos/flux_2_klein_9b_text_encoder/tt/pipeline.py")

# torch host-compute wrappers and HF orchestration that must not appear in the
# TT forward path. Shape/dtype prep (zeros, arange, cat, reshape) is allowed.
_FORBIDDEN = re.compile(
    r"torch\.(matmul|mm|bmm|einsum|softmax|log_softmax|layer_norm|rms_norm|batch_norm|group_norm|"
    r"embedding|embedding_bag|conv\d d?|conv_transpose|scaled_dot_product_attention|relu|gelu|silu|"
    r"tanh|sigmoid|leaky_relu|argmax|topk|multinomial|dropout)\b"
    r"|(?<![\w.])F\.\w+"
    r"|\.generate\("
    r"|\.forward\s*="
)
_COVERAGE_SWEEP = re.compile(
    r"def\s+(coverage_step|coverage_sweep|invoke_all_stubs|_touch_all_graduated|_invoke_every)"
)


def _routed_stub_files():
    return [_STUBS / f"{name}.py" for name in GRADUATED_COMPONENTS]


# ==========================================================================
#  Gate 1 -- static half (no device needed)
# ==========================================================================


def test_gate1_no_torch_fallback_in_routed_bodies():
    """The routed graduated bodies and the chain are pure ttnn."""
    offenders = {}
    for path in set(_routed_stub_files()) | {_PIPELINE_SRC}:
        src = path.read_text()
        hits = [
            f"{i}: {line.strip()}"
            for i, line in enumerate(src.splitlines(), 1)
            if _FORBIDDEN.search(line) and not line.strip().startswith("#")
        ]
        if hits:
            offenders[str(path)] = hits
    assert not offenders, f"forbidden torch-compute / HF-orchestration in the forward path: {offenders}"

    # The pipeline must never reach for a golden helper: those live in model_ref
    # and are only ever called by tests/demos, outside the chain.
    src = _PIPELINE_SRC.read_text()
    assert "hf_reference_" not in src, "the TT chain must not call a reference helper (SV-4)"
    assert not _COVERAGE_SWEEP.search(src), "a coverage sweep does not count for Gate 2"
    for path in _routed_stub_files():
        assert not _COVERAGE_SWEEP.search(path.read_text()), f"coverage sweep in {path}"
    print("[gate1] static: no torch fallback, no HF orchestration, no coverage sweep")


def test_gate1_bringup_reported_no_runtime_fallbacks():
    fallbacks = json.loads((_BRINGUP / "_runtime_fallbacks.json").read_text())
    assert fallbacks == {}, f"bring-up recorded runtime fallbacks: {fallbacks}"


def test_all_graduated_components_are_routed():
    """No graduated module is left out of the routing table (Source-B audit)."""
    graduated = set()
    for path in _STUBS.glob("*.py.last_good_*"):
        graduated.add(path.name.split(".py.last_good_")[0])
    assert graduated, "no graduated snapshots found under _stubs/"
    missing = graduated - set(GRADUATED_COMPONENTS)
    assert not missing, f"graduated but not routed by the pipeline: {sorted(missing)}"
    extra = set(GRADUATED_COMPONENTS) - graduated
    assert not extra, f"routed but not graduated: {sorted(extra)}"
    print(f"[gate2] routing table covers all {len(graduated)} graduated modules: {sorted(graduated)}")


# ==========================================================================
#  Gates 1/2/3 -- on device, both task heads, one shared pipeline
# ==========================================================================


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TP], indirect=True)
def test_e2e_pipeline(mesh_device):
    torch.manual_seed(0)
    pipeline = build_pipeline(mesh_device)
    hf = pipeline.hf_model
    n_devices = pipeline.num_devices

    # ---------------------------------------------------------- Gate 1 (live)
    _assert_still_sharded(pipeline, n_devices)

    input_ids = model_ref.encode_prompt(model_ref.DEFAULT_PROMPT)
    prompt_len = int(input_ids.shape[-1])
    print(f"[e2e] prompt_len={prompt_len} tp={n_devices} layers={pipeline.n_layers} stages={PIPELINE_STAGES}")

    # ===================== Call 2 -- text -> prompt embedding ==============
    with graduated_invocation_probe() as counts_call2:
        prompt_embeds = pipeline.run_prompt_encoding(input_ids=input_ids)
    golden_embeds = model_ref.hf_reference_prompt_encoding(hf, input_ids)
    pcc_call2 = model_ref.pcc(golden_embeds, prompt_embeds)
    print(f"[e2e] call=prompt_encoding shape={tuple(prompt_embeds.shape)} counts={dict(counts_call2)}")

    # ===================== Call 1 -- text -> text ==========================
    # Both sides decode under the SAME model-grounded stop rule: run until the
    # model's own eos, bounded by the same horizon.
    horizon = model_ref.resolve_max_new_tokens(hf, prompt_len)
    with graduated_invocation_probe() as counts_call1:
        tt_gen = pipeline.run_text_generation(input_ids=input_ids, max_new_tokens=horizon)
    golden_gen = model_ref.hf_reference_text_generation(hf, input_ids, max_new_tokens=horizon)

    steps = int(tt_gen["step_logits"].shape[0])
    assert steps > 0, "no decode steps ran"

    # The scored golden is the reference evaluated on the SAME contexts the TT loop
    # actually decoded (prompt + tt_tokens[:i] at step i). That is what makes this a
    # measurement of the pipeline's arithmetic: a free-running greedy pair that has
    # parted at one near-tie is afterwards being asked two different questions, so
    # comparing those logits would report the tie-break, not the port. The TT side is
    # untouched -- it is the same free-running chain, argmax and all, that the demo
    # and the host-op self-test run; only the reference is aligned to it.
    golden_step_logits = model_ref.hf_reference_step_logits(hf, input_ids, tt_gen["token_ids"])
    assert tuple(golden_step_logits.shape) == tuple(
        tt_gen["step_logits"].shape
    ), f"golden {tuple(golden_step_logits.shape)} vs TT {tuple(tt_gen['step_logits'].shape)}"
    pcc_call1 = model_ref.pcc(golden_step_logits, tt_gen["step_logits"])
    per_step = [model_ref.pcc(golden_step_logits[i], tt_gen["step_logits"][i]) for i in range(steps)]

    # Reported in full, not gated on: how often this pipeline's argmax picks the same
    # token the reference picks from the same context, and where the two free-running
    # greedy runs part. Both are token-identity facts (a single near-tie flips them),
    # which is exactly why the PCC above is measured on logits over shared contexts.
    ref_top1 = golden_step_logits.argmax(dim=-1).tolist()
    top1_agree = sum(1 for i, t in enumerate(tt_gen["token_ids"]) if t == ref_top1[i])
    divergence = model_ref.first_divergence(tt_gen["token_ids"], golden_gen["token_ids"])
    free_run_agree = divergence if divergence is not None else min(steps, len(golden_gen["token_ids"]))

    print(f"[e2e] call=text_generation counts={dict(counts_call1)}")
    print(f"[e2e] horizon={horizon} tt_steps={len(tt_gen['token_ids'])} hf_steps={len(golden_gen['token_ids'])}")
    print(f"[e2e] TT text : {tt_gen['text']!r}")
    print(f"[e2e] HF text : {golden_gen['text']!r}")
    print(f"[e2e] TT ids  : {tt_gen['token_ids']}")
    print(f"[e2e] HF ids  : {golden_gen['token_ids']}")
    print(f"[e2e] same-context top-1 agreement: {top1_agree}/{steps}")
    print(f"[e2e] free-running greedy agreement: {free_run_agree}/{steps} (first divergence {divergence})")

    # ---------------------------------------------------------- Gate 2
    merged = {k: counts_call1.get(k, 0) + counts_call2.get(k, 0) for k in GRADUATED_COMPONENTS}
    print(f"[gate2] invocation counts over the real forward: {merged}")
    missing = [k for k, v in merged.items() if v < 1]
    assert not missing, f"graduated modules never invoked in the pipeline: {missing}"
    # Structural, not a flat one-each sweep: the per-layer bodies run once per layer
    # per forward, the stack and the head run once per forward.
    depth = pipeline.n_layers
    for per_layer in ("attention", "decoder_layer", "layer", "m_l_p", "mlp"):
        assert merged[per_layer] >= depth, (
            f"{per_layer} ran {merged[per_layer]}x but the stack has {depth} layers -- "
            "that is not the real forward path"
        )
    assert merged["r_m_s_norm"] >= 2 * depth, "q_norm and k_norm run once each per layer"

    # ---------------------------------------------------------- Gate 3
    # The gate is the WORST number the two task heads produce, and for the decode
    # head the worst SINGLE step -- not an average that a good prefill could carry.
    worst_step = min(per_step)
    achieved_pcc = min(pcc_call1, pcc_call2, worst_step)
    print(f"e2e PCC (call=prompt_encoding) ={pcc_call2}")
    print(f"e2e PCC (call=text_generation)={pcc_call1}")
    print(f"e2e PCC (worst decode step)   ={worst_step}")
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, (
        f"e2e PCC {achieved_pcc} below {PCC_TARGET} "
        f"(prompt_encoding={pcc_call2}, text_generation={pcc_call1}, worst_step={worst_step}, "
        f"first_free_run_divergence={divergence})"
    )


def _assert_still_sharded(pipeline, n_devices):
    """A body that graduated SHARDED must still be sharded, not replicated.

    Two independent facts are checked, because either alone can be faked:
      * the PER-DEVICE shape is the TP-divided one predicted from the config
        (ttnn reports a mesh tensor's local shape, so this is the real slice), and
      * the shards actually DIFFER from each other -- a replicated weight would
        hand every chip identical bytes no matter what its shape said.
    """
    assert n_devices > 1, (
        f"this gate runs the TP={TP} layout the components graduated at; a {n_devices}-device "
        "mesh has no sharding to check, so it is a gate failure, not a reason to stand down"
    )

    cfg = pipeline.config
    hidden = int(cfg.hidden_size)
    head_dim = int(cfg.head_dim)
    inter = int(cfg.intermediate_size)
    n_local_q = int(cfg.num_attention_heads) // n_devices
    n_local_kv = int(cfg.num_key_value_heads) // n_devices
    padded_vocab = pipeline.decoder_head.padded_vocab_size

    layer0 = pipeline.layers[0]
    sharded = {
        # column-parallel: the OUTPUT axis is split
        "attention.wqkv": (layer0.attention.wqkv, (hidden, (n_local_q + 2 * n_local_kv) * head_dim)),
        "m_l_p.w_gate": (layer0.mlp.w_gate, (hidden, inter // n_devices)),
        "m_l_p.w_up": (layer0.mlp.w_up, (hidden, inter // n_devices)),
        "decoder_head.weight": (pipeline.decoder_head.weight, (hidden, padded_vocab // n_devices)),
        # row-parallel: the INPUT axis is split
        "attention.wo": (layer0.attention.wo, (n_local_q * head_dim, hidden)),
        "m_l_p.w_down": (layer0.mlp.w_down, (inter // n_devices, hidden)),
    }
    for label, (tensor, expected) in sharded.items():
        shards = ttnn.get_device_tensors(tensor)
        assert len(shards) == n_devices, f"{label}: {len(shards)} shards on {n_devices} devices"
        per_dev = tuple(int(d) for d in shards[0].shape)
        assert per_dev == expected, f"{label}: per-device shape {per_dev}, expected the TP-divided {expected}"
        a = ttnn.to_torch(shards[0]).float()
        b = ttnn.to_torch(shards[1]).float()
        assert not torch.equal(a, b), f"{label}: chip 0 and chip 1 hold identical bytes -> REPLICATED, not sharded"
        print(f"[gate1] {label}: per-device={per_dev} x{n_devices} shards, chips differ -> really sharded")

    replicated = {
        "token_embed.weights (lookup table)": pipeline.token_embed.weights,
        "attention.q_norm gamma (r_m_s_norm)": layer0.attention.q_norm.gamma,
        "encoder_stack.final_norm_gamma": pipeline.encoder_stack.final_norm_gamma,
    }
    for label, tensor in replicated.items():
        shards = ttnn.get_device_tensors(tensor)
        a = ttnn.to_torch(shards[0]).float()
        b = ttnn.to_torch(shards[1]).float()
        assert torch.equal(a, b), f"{label} must stay replicated"
        print(f"[gate1] {label}: replicated across {len(shards)} chips, per-device={tuple(shards[0].shape)}")

    # The KV cache inherits the head split rather than adding one of its own.
    k_cache = pipeline.kv_caches[0][0]
    kc = tuple(int(d) for d in ttnn.get_device_tensors(k_cache)[0].shape)
    assert kc == (pipeline.batch, n_local_kv, pipeline.kv_capacity, head_dim), kc
    print(f"[gate1] kv cache per-device={kc} (kv heads already the sharded axis)")


# ==========================================================================
#  SV-6 / SV-9 -- one build surface, and a knob that is not inert
# ==========================================================================


def test_demo_and_test_share_one_pipeline():
    from models.demos.flux_2_klein_9b.text_encoder.demo import demo_prompt_encoding, demo_text_generation
    from models.demos.flux_2_klein_9b.text_encoder.tt import pipeline as tt_pipeline

    assert demo_text_generation.build_pipeline is tt_pipeline.build_pipeline
    assert demo_prompt_encoding.build_pipeline is tt_pipeline.build_pipeline
    # and neither demo carries its own copy of the wiring
    for mod in (demo_text_generation, demo_prompt_encoding):
        src = Path(mod.__file__).read_text()
        assert (
            "encoder_stack(" not in src and "decoder_head(" not in src
        ), f"{mod.__name__} re-implements the chain instead of importing it"
    print("[sv6] demo and e2e test call the same build_pipeline and the same run_* methods")
