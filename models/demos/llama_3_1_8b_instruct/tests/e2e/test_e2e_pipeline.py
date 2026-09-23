# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for Call 1 (text -> text) on `/home/ttuser/benchmark-data/Llama-3.1-8B-Instruct`.

Real input (HF chat template) -> the SHARED chained TTNN pipeline
(`tt.pipeline.run_text_generation`, the very function `demo/demo_text_generation.py`
calls) -> real output (generated text), compared against the HF golden
(`LlamaForCausalLM.generate`, Source A).

  Gate 1 — every routed graduated stub is still real ttnn, and the three TP=4
           bodies still shard (ShardTensorToMesh) and still collect (all_reduce_async).
  Gate 2 — every graduated module is INVOKED inside the real forward path.
  Gate 3 — final-output PCC vs the HF golden >= 0.95.

Run on device:
  ./python_env/bin/python -m pytest models/demos/llama_3_1_8b_instruct/tests/e2e/test_e2e_pipeline.py -s
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3_1_8b_instruct.tt import _invocation
from models.demos.llama_3_1_8b_instruct.tt import pipeline as pl

DEMO_DIR = Path(__file__).resolve().parents[2]
STUBS = DEMO_DIR / "_stubs"
PCC_TARGET = 0.95

MAX_NEW_TOKENS = int(os.environ.get("TT_E2E_N", pl.GATE_MAX_NEW_TOKENS))
LAYERS = int(os.environ["TT_E2E_LAYERS"]) if os.environ.get("TT_E2E_LAYERS") else None
PROMPT = os.environ.get("TT_E2E_PROMPT", pl.DEFAULT_PROMPT)


# ---------------------------------------------------------------------------
# S1 — re-derive the graduated inventory from disk (nothing wasted)
# ---------------------------------------------------------------------------
def graduated_from_disk() -> set[str]:
    import json

    status = json.loads((DEMO_DIR / "bringup_status.json").read_text())
    out = set()
    for comp in status["components"]:
        name = comp["name"]
        if (STUBS / f"{name}.py.last_good_native").exists() or (STUBS / f"{name}.py.last_good_sharded").exists():
            out.add(name)
    return out


def test_graduated_inventory_is_fully_routed():
    on_disk = graduated_from_disk()
    routed = set(pl.GRADUATED_MODULES)
    assert on_disk == routed, f"graduated on disk {sorted(on_disk)} != routed by the pipeline {sorted(routed)}"
    print(f"[S1] graduated modules re-derived from disk and all routed: {sorted(on_disk)}")


# ---------------------------------------------------------------------------
# Gate 1 (static half) — the routed stubs are still real ttnn, still sharded
# ---------------------------------------------------------------------------
TORCH_COMPUTE = re.compile(
    r"\btorch\.(matmul|mm|bmm|einsum|softmax|log_softmax|layer_norm|rms_norm|batch_norm|group_norm|"
    r"embedding|embedding_bag|conv[123]d|conv_transpose\w*|scaled_dot_product_attention|relu|gelu|silu|"
    r"tanh|sigmoid|leaky_relu|argmax|topk|multinomial|dropout)\b|\bF\.\w+\(|torch\.nn\.functional\."
)


def _forward_source(path: Path) -> str:
    """Everything from the first forward-ish def to EOF — the hot path."""
    text = path.read_text()
    idx = text.find("    def forward(")
    if idx < 0:
        idx = text.find("    def __call__(")
    return text[idx:] if idx >= 0 else text


def test_gate1_stubs_are_native_and_sharded():
    # attention / m_l_p own the ShardTensorToMesh weight splits; decoder_layer is the
    # composite that carries the split by COMPOSING those two sharded bodies, so it is
    # checked for that composition rather than for a shard call of its own.
    shards_its_own_weights = {"attention", "m_l_p"}
    for name in pl.GRADUATED_MODULES:
        path = STUBS / f"{name}.py"
        hot = _forward_source(path)
        hits = TORCH_COMPUTE.findall(hot)
        assert not hits, f"Gate 1: {name} has torch compute in its forward: {hits}"
        assert "ttnn." in hot, f"Gate 1: {name} forward runs no ttnn ops"
        src = path.read_text()
        if name in shards_its_own_weights:
            assert "ShardTensorToMesh" in src, f"Gate 1: {name} lost its ShardTensorToMesh (rewritten to replication?)"
            assert "all_reduce_async" in src, f"Gate 1: {name} lost its collective"
    dl = (STUBS / "decoder_layer.py").read_text()
    assert "TtLlamaAttention" in dl and "TtLlamaMLP" in dl, "Gate 1: decoder_layer no longer composes the sharded bodies"
    assert "all_reduce_async" in dl, "Gate 1: decoder_layer lost the MLP collective it owns"
    print(f"[Gate 1/static] all {len(pl.GRADUATED_MODULES)} routed stubs are pure ttnn; TP bodies keep shard+collective")


# ---------------------------------------------------------------------------
# The on-device end-to-end gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_e2e_text_generation(mesh_device):
    n_dev = mesh_device.get_num_devices()
    if n_dev != 4:
        print(f"[e2e] NOTE: running on {n_dev} device(s), not the requested 4-chip mesh")

    tokenizer = pl.load_tokenizer()
    hf_model = pl.load_hf_model()

    input_ids = pl.encode_prompt(tokenizer, PROMPT)
    pl.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(input_ids, pl.GOLDEN_DIR / "input_ids.pt")

    pipe = pl.build_pipeline(mesh_device, model=hf_model, tokenizer=tokenizer, layers=LAYERS)
    print(f"[e2e] pipeline: {pipe.describe()}")

    # ---- Gate 1 (runtime half): the built model really is sharded + collective ----
    tp_degrees = {layer.tp for layer in pipe.layers}
    assert tp_degrees == {n_dev} or n_dev == 1, f"expected every layer at TP={n_dev}, got {tp_degrees}"
    if n_dev > 1:
        assert all(layer.attention._ar_semaphores is not None for layer in pipe.layers)
        assert all(layer._mlp_semaphores is not None for layer in pipe.layers)
        print(f"[Gate 1/runtime] {len(pipe.layers)} layers at TP={n_dev}, 2 all_reduce collectives per layer")

    # ---- Run the REAL pipeline (the same call the demo makes) ----
    _invocation.reset()
    got = pl.run_text_generation(pipe, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, input_ids=input_ids)

    # ---- HF golden: same eos rule, same cap ----
    golden = pl.hf_reference_text_generation(
        hf_model, tokenizer, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, input_ids=input_ids
    )

    # ------------------------------------------------------------------
    # Gate 3 is measured with TWO injection-free numbers, then reported as their min.
    #
    # (a) PREFILL PARITY over every real prompt position: the whole 32-layer stack,
    #     every graduated module, 45 positions of logits vs the HF forward. Nothing
    #     is injected; this is the pipeline's own output at every position.
    # (b) DECODE PARITY over the free-running steps whose PREFIX both sides agree on.
    #     Greedy decoding is chaotic: once the two sides pick differently at a
    #     near-tie, every later step is conditioned on a DIFFERENT context, so
    #     comparing those logits measures the tie-break, not the pipeline. The set of
    #     comparable steps is therefore chosen by the data, not by hand -- and the
    #     divergence step and the golden's own top-2 gap there are printed, so a
    #     genuine error can never hide behind "it was just a tie".
    #
    # No reference tensor or reference token is ever fed back into the TT chain: the
    # TT side free-runs on its own output throughout.
    # ------------------------------------------------------------------
    def _pcc(a, b):
        _, v = comp_pcc(a.float(), b.float(), 0.0)
        return float(v) if not isinstance(v, str) else float(v.split()[-1])

    # (a) prefill parity — TT logits at EVERY prompt position vs the HF forward
    real_len = int(input_ids.shape[-1])
    padded_len = ((real_len + 31) // 32) * 32
    padded = torch.zeros(1, padded_len, dtype=torch.int64)
    padded[0, :real_len] = input_ids[0]
    ids_tt = pipe.ids_to_device(padded)
    tt_prefill = pl._first_shard(mesh_device, pipe.prefill(ids_tt, padded_len, real_len - 1, return_all=True))
    tt_prefill = tt_prefill.reshape(padded_len, -1)[:real_len, : pipe.vocab_size].float()
    with torch.no_grad():
        hf_prefill = hf_model(input_ids).logits[0].float()
    prefill_pcc = _pcc(hf_prefill, tt_prefill)

    # (b) decode parity over the common-prefix steps
    steps = min(len(got["new_ids"]), len(golden["new_ids"]))
    tt_ids, hf_ids = got["new_ids"][:steps], golden["new_ids"][:steps]
    common = 0
    while common < steps and tt_ids[common] == hf_ids[common]:
        common += 1
    comparable = min(common + 1, steps)  # step k shares a context iff tokens 0..k-1 match
    tt_logits = got["step_logits"][:comparable]
    hf_logits = golden["step_logits"][:comparable]
    per_step = [_pcc(hf_logits[i], tt_logits[i]) for i in range(comparable)]
    decode_pcc = _pcc(hf_logits, tt_logits)
    free_running_pcc = _pcc(golden["step_logits"][:steps], got["step_logits"][:steps])
    matched = sum(1 for a, b in zip(tt_ids, hf_ids) if a == b)

    achieved_pcc = min(prefill_pcc, decode_pcc)
    ok = achieved_pcc >= PCC_TARGET

    print("\n================ Call 1: text_generation ================")
    print(f"prompt            : {PROMPT}")
    print(f"TT     output     : {got['text']!r}")
    print(f"HF     golden     : {golden['text']!r}")
    print(f"TT     ids        : {tt_ids}")
    print(f"HF     ids        : {hf_ids}")
    print(f"steps generated   : {steps} (stop-token driven, cap={MAX_NEW_TOKENS} on BOTH sides)")
    print(f"token match       : {matched}/{steps} ({100.0 * matched / max(steps, 1):.1f}%)")
    print(f"prefill parity    : PCC={prefill_pcc:.6f} over {real_len} prompt positions x {pipe.vocab_size} logits")
    print(f"decode parity     : PCC={decode_pcc:.6f} over {comparable} common-prefix step(s); "
          f"per-step min={min(per_step):.6f} mean={sum(per_step) / len(per_step):.6f}")
    print(f"free-running PCC  : {free_running_pcc:.6f} over all {steps} step(s) "
          f"(diagnostic: after a divergence the two sides condition on different text)")
    if common < steps:
        gap = torch.topk(golden["step_logits"][common].float(), 2).values
        print(f"first divergence  : step {common} — TT {tt_ids[common]} vs HF {hf_ids[common]}; "
              f"the GOLDEN's own top-2 logit gap there is {float(gap[0] - gap[1]):.4f} "
              f"(a small gap means greedy argmax is a coin-flip, not an error)")
    else:
        print("first divergence  : none — the TT sequence is identical to the golden")

    # The decode path must actually be under test: if the two sides split immediately
    # there is nothing left to compare and the gate would be vacuous.
    min_decode_steps = min(4, steps)
    assert comparable >= min_decode_steps, (
        f"only {comparable} comparable decode step(s) (need {min_decode_steps}): the TT sequence "
        f"left the golden at step {common}, so the decode path is effectively untested"
    )

    # ---- Gate 2: every graduated module actually ran, inside the real forward ----
    invoked = got["invoked"]
    missing = set(pl.GRADUATED_MODULES) - invoked
    print(f"[Gate 2] graduated modules invoked: {sorted(invoked & set(pl.GRADUATED_MODULES))}")
    assert not missing, f"Gate 2: graduated modules never invoked in the pipeline: {sorted(missing)}"

    # ---- Gate 3 ----
    print(f"e2e PCC={achieved_pcc}")
    assert ok and achieved_pcc >= PCC_TARGET, f"Gate 3: e2e PCC {achieved_pcc} below target {PCC_TARGET}"


# ---------------------------------------------------------------------------
# S6 — the demo and the test really do share ONE pipeline
# ---------------------------------------------------------------------------
def test_demo_and_test_share_one_pipeline():
    from models.demos.llama_3_1_8b_instruct.demo import demo_text_generation as demo

    assert demo.pl is pl
    assert demo.pl.run_text_generation is pl.run_text_generation
    src = Path(demo.__file__).read_text()
    assert "pl.run_text_generation(" in src, "the demo must CALL the shared pipeline, not re-implement it"
    print("[S6] demo/ and tests/e2e/ call the same tt/pipeline.run_text_generation")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-svv"]))
