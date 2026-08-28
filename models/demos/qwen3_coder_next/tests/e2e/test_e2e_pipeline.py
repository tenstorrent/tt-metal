# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for `Qwen/Qwen3-Coder-Next` -- Call 1, text -> text.

    ./python_env/bin/python -m pytest models/demos/qwen3_coder_next/tests/e2e/test_e2e_pipeline.py -s

The test imports the SAME `tt/pipeline.py` the demo runs, so a pass here is a working demo.  The
`pipeline` fixture lives in `conftest.py`, which is the session's ONLY device opener.

  Gate 1  every routed graduated stub is still native ttnn (no torch compute in the hot path),
          and the composed pipeline really is TENSOR-PARALLEL -- sharded weights plus a
          collective, not a pure-replication placement
  Gate 2  all ten graduated components are INVOKED by the real forward pass, with the counts the
          built topology implies
  Gate 3  the final output PCC vs the HF golden (Source A `model.generate`) is >= 0.95
  plus    the pipeline is fully on device (zero host aten ops in decode) and every declared stage
          really captures as a trace and replays to the eager result
"""
from __future__ import annotations

import os
import re

from models.demos.qwen3_coder_next.tt import mesh as tt_mesh
from models.demos.qwen3_coder_next.tt.pipeline import (
    DEFAULT_PROMPT,
    GRADUATED_ENTRYPOINTS,
    PIPELINE_STAGES,
    InvocationProbe,
    _pcc,
)

PCC_GATE = 0.95
MAX_NEW_TOKENS = int(os.environ.get("TT_QWEN3_MAX_NEW_TOKENS", 8))

# The `_stubs/` bodies these ten names resolve to; Gate 1 scans exactly these files.
STUB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "_stubs")

# Host compute that must not appear in a stub's forward path. Shape/dtype helpers
# (torch.zeros / arange / eye / cat / reshape) are prep, not compute, and are allowed.
FORBIDDEN_TORCH_COMPUTE = re.compile(
    r"\btorch\.(matmul|mm|bmm|einsum|softmax|log_softmax|layer_norm|rms_norm|batch_norm|group_norm"
    r"|embedding|embedding_bag|conv1d|conv2d|conv3d|conv_transpose\w*|scaled_dot_product_attention"
    r"|relu|gelu|silu|tanh|sigmoid|leaky_relu|argmax|topk|multinomial|dropout)\b"
    r"|\bF\.\w+\(|\btorch\.nn\.functional\."
)


def test_gate1_stubs_are_native_ttnn():
    """Gate 1a: every routed graduated body is real ttnn, not a torch fallback."""
    offenders = {}
    for name in GRADUATED_ENTRYPOINTS:
        path = os.path.join(STUB_DIR, f"{name}.py")
        assert os.path.exists(path), f"graduated stub {name} is missing at {path}"
        source = open(path).read()
        body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        hits = FORBIDDEN_TORCH_COMPUTE.findall(body)
        if hits:
            offenders[name] = hits
        assert "ttnn." in body, f"{name} contains no ttnn ops at all"
    assert not offenders, f"torch host compute found in graduated stubs: {offenders}"
    print(f"Gate 1a OK: {len(GRADUATED_ENTRYPOINTS)} graduated stubs are native ttnn")


def test_gate1_placement_is_tensor_parallel(pipeline):
    """Gate 1b: the composed pipeline is really TP-sharded -- weights split AND a collective.

    Measured on the BUILT object against the LOGICAL width each weight would have unsharded, so a
    pipeline that opened a multi-chip TP group and replicated everything fails here even though a
    source scan would pass.  (`ttnn.Tensor.shape` on a mesh tensor already reports the per-chip
    shard, so comparing a shard to its own parent proves nothing -- the config is the reference.)
    """
    tp = pipeline.device.get_num_devices()
    print(f"placement: {getattr(pipeline, 'placement', 'n/a')} (TP group = {tp} chip(s))")
    assert tp > 1, (
        f"the pipeline was built on a {tp}-chip TP group: there is no tensor-parallel axis to "
        f"shard over, so the mandated TP={tt_mesh.TP_DEGREE} placement did not happen"
    )

    cfg = pipeline.config
    layer = pipeline.layers[0]          # a linear_attention layer
    moe = layer.mlp
    attn_layer = next(
        (l for l, t in zip(pipeline.layers, pipeline.layer_types) if t == "full_attention"), None
    )
    e_local = cfg.num_experts // tp

    # (label, device tensor, axis, expected SHARDED width, unsharded width)
    checks = [
        ("lm_head[vocab]", pipeline.lm_head, -1,
         cfg.vocab_size // pipeline.lm_tp, cfg.vocab_size),
        ("experts.w_gate_up[E*I]", moe.experts.w_gate_up, -1,
         2 * e_local * cfg.moe_intermediate_size, 2 * cfg.num_experts * cfg.moe_intermediate_size),
        ("experts.w_down[E*I]", moe.experts.w_down, -2,
         e_local * cfg.moe_intermediate_size, cfg.num_experts * cfg.moe_intermediate_size),
        ("experts.selector[experts]", moe.experts.selector, -1, e_local, cfg.num_experts),
        ("top_k_router.weight[experts]", moe.router.weight, -1, e_local, cfg.num_experts),
        ("m_l_p.w_gate[inter]", moe.shared_expert.w_gate, -1,
         cfg.shared_expert_intermediate_size // tp, cfg.shared_expert_intermediate_size),
        ("m_l_p.w_down[inter]", moe.shared_expert.w_down, -2,
         cfg.shared_expert_intermediate_size // tp, cfg.shared_expert_intermediate_size),
        ("gated_delta_net.w_out[value]", layer.mixer.w_out, -2,
         cfg.linear_num_value_heads * cfg.linear_value_head_dim // tp,
         cfg.linear_num_value_heads * cfg.linear_value_head_dim),
    ]
    if attn_layer is not None:
        checks.append(
            ("attention.wo[heads*hd]", attn_layer.mixer.wo, -2,
             cfg.num_attention_heads * cfg.head_dim // tp, cfg.num_attention_heads * cfg.head_dim)
        )

    for label, tensor, axis, want, whole in checks:
        got = int(tensor.shape[axis])
        assert got == want, f"{label}: per-chip width {got}, expected {want} (unsharded {whole})"
        assert got < whole, f"{label}: width {got} == the unsharded width -- REPLICATED, not sharded"
        print(f"  sharded {label}: {whole} -> {got} per chip (x{tp})")
    print(f"Gate 1b OK: {len(checks)} weights genuinely split across the TP={tp} group")

    # ...and the split has to be put back together somewhere.
    collectives = set()
    for name in GRADUATED_ENTRYPOINTS:
        body = open(os.path.join(STUB_DIR, f"{name}.py")).read()
        for op in ("ttnn.all_reduce", "ttnn.all_gather"):
            if op in body:
                collectives.add(f"{name}:{op.split('.')[-1]}")
    assert collectives, "no graduated stub issues a collective"
    print(f"Gate 1b OK: collectives on the path -> {sorted(collectives)}")


def test_e2e_text_generation(pipeline):
    """Gates 2 and 3 on ONE real run of the chained pipeline."""
    tokenizer = pipeline._tokenizer
    prompt = os.environ.get("TT_QWEN3_PROMPT", DEFAULT_PROMPT)

    with InvocationProbe() as probe:
        result = pipeline.run_text_generation(
            tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, collect_logits=True
        )

    # SOURCE A goldens.  Two of them, because a generative head has two things to be right about:
    #   * `generate()` free-running  -> WHAT the model emits (behavioural proof, printed below)
    #   * logits along a trajectory  -> the FUNCTION the model computes (the Gate 3 metric)
    # Both are capped to the same horizon. Nothing from either ever enters the TT chain.
    golden = pipeline._hf_reference_text_generation(tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    golden_steps = pipeline._hf_score_sequence(result["prompt_ids"], result["tokens"])

    n = min(len(result["tokens"]), len(golden["tokens"]))
    assert n > 0, "the pipeline generated nothing"

    # ---- behavioral proof -------------------------------------------------------------------
    print()
    print(f"PROMPT      : {prompt}")
    print(f"TT   output : {result['text']!r}")
    print(f"HF   golden : {golden['text']!r}")
    print(f"TT   tokens : {result['tokens'][:n]}")
    print(f"HF   tokens : {golden['tokens'][:n]}")
    matched = sum(int(a == b) for a, b in zip(result["tokens"][:n], golden["tokens"][:n]))
    print(f"token agreement (free-running, both greedy): {matched}/{n}")
    free_pcc = _pcc(golden["logits"][:n], result["logits"][:n])
    print(f"free-running per-step logits PCC: {free_pcc:.6f}")
    for step in range(len(result["tokens"])):
        same_hist = _pcc(golden_steps[step], result["logits"][step])
        line = f"  step {step}: PCC={same_hist:.6f}"
        if step < n:
            line += f"  (free-running {_pcc(golden['logits'][step], result['logits'][step]):.6f})"
        print(line)

    # ---- Gate 2: every graduated module actually ran, the expected number of times -----------
    depth = pipeline.depth
    full = sum(1 for t in pipeline.layer_types if t == "full_attention")
    linear = depth - full
    steps = len(result["tokens"])
    expected_per_forward = {
        "rotary_embedding": 1,
        "r_m_s_norm": 2 * depth + 1,  # two per layer + the model's final norm
        "decoder_layer": depth,
        "gated_delta_net": linear,
        "r_m_s_norm_gated": linear,  # the delta net's output norm
        "attention": full,
        "sparse_moe_block": depth,  # mlp_only_layers=[] and decoder_sparse_step=1
        "top_k_router": depth,
        "experts": depth,
        "m_l_p": depth,  # the MoE block's shared expert
    }
    print(f"invocation counts over {steps} decode step(s): {probe.counts}")
    missing = [name for name, count in probe.counts.items() if count == 0]
    assert not missing, f"Gate 2 FAILED -- graduated modules never invoked: {missing}"
    for name, per_forward in expected_per_forward.items():
        assert probe.counts[name] == per_forward * steps, (
            f"Gate 2: {name} ran {probe.counts[name]} times, expected "
            f"{per_forward} x {steps} = {per_forward * steps} for a {depth}-layer build"
        )
    print(f"Gate 2 OK: all {len(probe.counts)} graduated modules invoked inside the real forward pass")

    # ---- Gate 3: parity against the HF golden -------------------------------------------------
    # Per-step next-token logits, TT vs HF, over the SAME token history -- the one the TT chain
    # produced by itself.  When the two free-running sequences agree this is literally the
    # free-running number; when they diverge it still measures the port rather than the divergence.
    achieved_pcc = _pcc(golden_steps, result["logits"])
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_GATE, f"Gate 3 FAILED: e2e PCC {achieved_pcc} < {PCC_GATE}"


def test_host_op_selftest(pipeline):
    """The authoritative fully-on-device check: zero host aten ops in the model math."""
    verdict = pipeline.host_op_selftest()
    print(f"host_op_selftest: {verdict['reason']}")
    assert verdict["on_device"], verdict["reason"]


def test_trace_capture_selftest(pipeline):
    """Every declared stage captures host-free and its traced output matches the eager one.

    This runs on the fixture's device -- the same one the weights were uploaded to -- which is why
    `begin_trace_capture` finds the command queue it expects.
    """
    assert PIPELINE_STAGES == ["prefill", "decode"]
    ok = pipeline.trace_capture_selftest()
    print(f"trace_capture_selftest: {pipeline.trace_results}")
    assert ok, f"trace capture/verify failed: {pipeline.trace_results}"
