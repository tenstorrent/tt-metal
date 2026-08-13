# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for `/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

Both task heads the checkpoint registers, through the SAME `tt/pipeline.py` the
demos import, on device:

  Call 2  `causal_lm_logits`  teacher-forced logits for the whole prompt
                              vs `hf(input_ids).logits`
  Call 1  `text_generation`   free-running greedy decode (each step consuming the
                              previous TT step's own on-device argmax token)
                              vs `hf.generate(...)`'s per-step scores

and the three gates over them:

  G1 native   the real forward runs under `models/common/native_probe`, which
              counts what ACTUALLY executes: zero torch compute ops. Plus a
              static scan of `tt/pipeline.py` and the five graduated stubs for
              HF orchestration / torch-compute patterns in the TT path.
  G2 invoked  every graduated module is counted INSIDE the real chain, at the
              per-layer multiplicity the architecture implies (26 layers, so a
              count of 1 for a per-layer body is a failure, not a pass).
  G3 PCC      the final task output of BOTH calls against the HF golden, at the
              same model-grounded length (`eos_token_id` truncates both sides).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.native_probe import run_native_probe
from models.common.utility_functions import comp_pcc
from models.demos.voxtral_tts_backbone._stubs.attention import TtAttention
from models.demos.voxtral_tts_backbone._stubs.decoder_layer import TtDecoderLayer
from models.demos.voxtral_tts_backbone._stubs.m_l_p import TtMLP
from models.demos.voxtral_tts_backbone._stubs.r_m_s_norm import TtRMSNorm
from models.demos.voxtral_tts_backbone._stubs.rotary_embedding import TtRotaryEmbedding
from models.demos.voxtral_tts_backbone.tt.pipeline import (
    DEFAULT_PROMPT,
    DEMO_DIR,
    _hf_reference_generate,
    _hf_reference_logits,
    build_pipeline,
    persist_captured_inputs,
)

PCC_THRESHOLD = 0.95

#: The five graduated components, mapped to the class the pipeline composes.
GRADUATED = {
    "decoder_layer": TtDecoderLayer,
    "attention": TtAttention,
    "m_l_p": TtMLP,
    "r_m_s_norm": TtRMSNorm,
    "rotary_embedding": TtRotaryEmbedding,
}

#: HF orchestration / torch compute that must not appear in the TT path.
FORBIDDEN_IN_TT_PATH = (
    r"\.generate\s*\(",
    r"\.forward\s*=",
    r"torch\.(matmul|mm|bmm|addmm|einsum|softmax|log_softmax|layer_norm|rms_norm|batch_norm|group_norm"
    r"|embedding|embedding_bag|conv[123]d|conv_transpose[123]d|scaled_dot_product_attention|relu|gelu"
    r"|silu|tanh|sigmoid|leaky_relu|argmax|topk|multinomial|dropout)\s*\(",
    r"torch\.nn\.functional\.\w+\s*\(",
    r"(?<![\w.])F\.\w+\s*\(",
    r"(?<![\w.])hf_model\.\w+\s*\(",
    r"self\.hf\.\w+\s*\(",
)

#: Functions that legitimately hold HF calls: the golden helpers and weight load.
REFERENCE_FUNCTIONS = ("load_hf_model", "_hf_reference_logits", "_hf_reference_generate")
REFERENCE_CLASSES = ("_hf_depth",)


def _tt_path_sources():
    yield DEMO_DIR / "tt" / "pipeline.py"
    for name in sorted(GRADUATED):
        yield DEMO_DIR / "_stubs" / f"{name}.py"


def _forbidden_hits(path: Path):
    """Every forbidden pattern found in a function that is part of the TT path.

    The reference helpers are excluded by name — they are where the golden is
    computed, and HF belongs there.
    """
    source = path.read_text()
    tree = ast.parse(source)
    reference_bodies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in REFERENCE_CLASSES:
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    reference_bodies.add(child.name)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in REFERENCE_FUNCTIONS or node.name in reference_bodies:
            continue
        segment = ast.get_source_segment(source, node) or ""
        segment = re.sub(r"#.*", "", segment)
        for pattern in FORBIDDEN_IN_TT_PATH:
            for match in re.findall(pattern, segment):
                hits.append(f"{path.name}::{node.name}(): {pattern} -> {match}")
    return hits


class _Counters:
    """Count invocations by wrapping the graduated CLASSES' `__call__`.

    The counting lives in the TEST: the pipeline contains no touch-only calls and
    no coverage sweep, so these numbers can only come from the real chain.
    """

    def __init__(self):
        self.counts = {name: 0 for name in GRADUATED}
        self._originals = {}

    def __enter__(self):
        for name, klass in GRADUATED.items():
            original = klass.__call__
            self._originals[name] = original

            def make(name=name, original=original):
                def wrapper(inner_self, *args, **kwargs):
                    self.counts[name] += 1
                    return original(inner_self, *args, **kwargs)

                return wrapper

            klass.__call__ = make()
        return self

    def __exit__(self, *exc):
        for name, original in self._originals.items():
            GRADUATED[name].__call__ = original
        self._originals.clear()
        return False


def _expected_counts(depth: int, decode_steps: int) -> dict:
    """What the architecture implies: one rotary table per forward, the whole
    stack per forward, two norms per layer plus the model-level final norm."""
    forwards = 1 + decode_steps
    return {
        "rotary_embedding": forwards,
        "decoder_layer": depth * forwards,
        "attention": depth * forwards,
        "m_l_p": depth * forwards,
        "r_m_s_norm": (2 * depth + 1) * forwards,
    }


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_pipeline(device):
    pipeline = build_pipeline(device)
    staged = pipeline.stage_inputs(DEFAULT_PROMPT)
    prompt_len = staged["prompt_len"]
    # Persist the pipeline-level golden inputs for the zero-arg trace hooks.
    persist_captured_inputs(staged["prompt_tokens"])
    print(
        "\n[e2e] prompt=%r prompt_len=%d seq_len=%d depth=%d capacity=%d stop_token_ids=%s"
        % (DEFAULT_PROMPT, prompt_len, staged["seq_len"], pipeline.depth, pipeline.capacity, pipeline.stop_token_ids),
        flush=True,
    )

    probes = []

    # ---------------------------------------------------------------- Call 2
    with _Counters() as call2:
        prefill_logits, probe = run_native_probe(
            DEMO_DIR / "_captured" / "e2e_pipeline" / "call_causal_lm_logits",
            lambda: pipeline.run_prefill_logits(staged),
        )
    probes.append(("causal_lm_logits", probe))
    tt_logits = ttnn.to_torch(pipeline.unpadded_logits(prefill_logits, prompt_len)).float()
    golden_logits = _hf_reference_logits(pipeline, staged)
    ok_call2, pcc_call2 = comp_pcc(golden_logits, tt_logits, PCC_THRESHOLD)
    tt_next = int(tt_logits[0, -1].argmax())
    hf_next = int(golden_logits[0, -1].argmax())
    tt_top5 = [int(i) for i in tt_logits[0, -1].topk(5).indices]
    hf_top5 = [int(i) for i in golden_logits[0, -1].topk(5).indices]
    print(
        "[e2e] call=causal_lm_logits shape=%s next_token tt=%d hf=%d top5_overlap=%d/5"
        % (tuple(tt_logits.shape), tt_next, hf_next, len(set(tt_top5) & set(hf_top5))),
        flush=True,
    )
    print("[e2e] call=causal_lm_logits tt_top5=%s hf_top5=%s" % (tt_top5, hf_top5), flush=True)

    # ---------------------------------------------------------------- Call 1
    horizon = pipeline.decode_horizon(prompt_len)
    with _Counters() as call1:
        generated, probe = run_native_probe(
            DEMO_DIR / "_captured" / "e2e_pipeline" / "call_text_generation",
            lambda: pipeline.run_generate(staged, max_new_tokens=horizon),
        )
    probes.append(("text_generation", probe))
    tt_token_ids = pipeline.token_ids(generated["tokens"])
    tt_scores = pipeline.stacked_logits(generated["step_logits"])
    hf_token_ids, hf_scores = _hf_reference_generate(pipeline, staged, generated["max_new_tokens"])
    # The stop token truncates BOTH sides to one common, model-grounded length.
    common = pipeline.common_stop_length(tt_token_ids, hf_token_ids)
    ok_call1, pcc_call1 = comp_pcc(hf_scores[:common], tt_scores[:common], PCC_THRESHOLD)
    per_step = [round(float(comp_pcc(hf_scores[i], tt_scores[i], PCC_THRESHOLD)[1]), 5) for i in range(common)]
    divergence = next((i for i in range(common) if tt_token_ids[i] != hf_token_ids[i]), -1)
    decoded_tt = pipeline.tokenizer.decode(tt_token_ids[:common])
    decoded_hf = pipeline.tokenizer.decode(hf_token_ids[:common])
    print(
        "[e2e] call=text_generation horizon=%d compared=%d first_divergence=%d"
        % (generated["max_new_tokens"], common, divergence),
        flush=True,
    )
    print("[e2e] tt tokens=%s" % tt_token_ids[:common], flush=True)
    print("[e2e] hf tokens=%s" % hf_token_ids[:common], flush=True)
    print("[e2e] tt text=%r" % decoded_tt, flush=True)
    print("[e2e] hf text=%r" % decoded_hf, flush=True)
    print("[e2e] per-step corr=%s" % per_step, flush=True)

    # ------------------------------------------------------------ G1 native
    static_hits = []
    for path in _tt_path_sources():
        static_hits.extend(_forbidden_hits(path))
    for name, probe in probes:
        print(
            "[e2e] G1 native_probe call=%s ttnn_dispatch=%d torch_ops=%d %s"
            % (name, probe["ttnn_dispatch"], probe["torch_ops"], probe["torch_op_names"]),
            flush=True,
        )

    # ----------------------------------------------------------- G2 invoked
    expected2 = _expected_counts(pipeline.depth, 0)
    expected1 = _expected_counts(pipeline.depth, generated["max_new_tokens"] - 1)
    print("[e2e] G2 call=causal_lm_logits counts=%s expected=%s" % (call2.counts, expected2), flush=True)
    print("[e2e] G2 call=text_generation counts=%s expected=%s" % (call1.counts, expected1), flush=True)

    # --------------------------------------------------------------- G3 PCC
    print("[e2e] FINAL_PCC call=causal_lm_logits pcc=%s" % pcc_call2, flush=True)
    print("[e2e] FINAL_PCC call=text_generation pcc=%s" % pcc_call1, flush=True)
    print("[causal_lm_logits] e2e PCC=%s" % pcc_call2, flush=True)
    print("[text_generation] e2e PCC=%s" % pcc_call1, flush=True)
    print("e2e PCC=%s" % min(float(pcc_call1), float(pcc_call2)), flush=True)

    assert not static_hits, "G1 static: HF orchestration / torch compute in the TT path: %s" % static_hits
    for name, probe in probes:
        assert probe["torch_ops"] == 0, "G1 native: call=%s executed torch compute ops %s" % (
            name,
            probe["torch_op_names"],
        )
        assert probe["ttnn_dispatch"] > 0, "G1 native: call=%s dispatched no ttnn ops" % name
    assert call2.counts == expected2, "G2: call=causal_lm_logits invocation counts %s != %s" % (
        call2.counts,
        expected2,
    )
    assert call1.counts == expected1, "G2: call=text_generation invocation counts %s != %s" % (
        call1.counts,
        expected1,
    )
    # Call 1's decode is free-running, so a wiring bug shows up as an early
    # token divergence rather than as a silently-passing metric.
    assert tt_token_ids[:common] == hf_token_ids[:common], (
        "behavioral: greedy token sequences diverge at index %d (tt=%s hf=%s)"
        % (divergence, tt_token_ids[:common], hf_token_ids[:common])
    )
    assert decoded_tt == decoded_hf, "behavioral: decoded continuations differ"
    assert tt_next == hf_next, "behavioral: prefill next-token prediction tt=%d != hf=%d" % (tt_next, hf_next)
    assert ok_call2, "G3: causal_lm_logits PCC %s < %s" % (pcc_call2, PCC_THRESHOLD)
    assert ok_call1, "G3: text_generation PCC %s < %s" % (pcc_call1, PCC_THRESHOLD)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_prefill_decode_consistency(device):
    """The KV-cache proof, no HF reference needed: one decode step at position S
    must equal a full re-prefill of S+1 tokens read at its last position.

    This is what catches a cache written at the wrong row or read through the
    wrong window — a bug that a logits-only comparison can hide.
    """
    pipeline = build_pipeline(device)
    tokens = pipeline.tokenizer(DEFAULT_PROMPT, return_tensors="pt")["input_ids"]
    prompt_len = int(tokens.shape[1])

    staged = pipeline.stage_inputs(input_ids=tokens)
    state = pipeline.decode_prefill(staged)
    first_token = int(ttnn.to_torch(state["first_token"]).reshape(-1)[0])
    step_logits, _ = pipeline.decode_step()
    from_cache = ttnn.to_torch(step_logits).float().reshape(-1, pipeline.vocab_size)[-1]

    # The same sequence, one token longer, prefilled from scratch.
    extended = torch.zeros(1, prompt_len + 1, dtype=torch.int64)
    extended[0, :prompt_len] = tokens[0]
    extended[0, prompt_len] = first_token
    fresh = pipeline.stage_inputs(input_ids=extended)
    logits = pipeline.run_prefill_logits(fresh)
    from_prefill = ttnn.to_torch(pipeline.unpadded_logits(logits, prompt_len + 1)).float()[0, -1]

    ok, pcc = comp_pcc(from_prefill, from_cache, 0.99)
    print(
        "\n[e2e] kv-cache consistency: decode@%d vs re-prefill(%d) corr=%s argmax %d/%d"
        % (prompt_len, prompt_len + 1, pcc, int(from_cache.argmax()), int(from_prefill.argmax())),
        flush=True,
    )
    assert int(from_cache.argmax()) == int(from_prefill.argmax()), "kv-cache: decode step picks a different token"
    assert ok, "kv-cache: decode step disagrees with a full re-prefill (corr %s)" % pcc
