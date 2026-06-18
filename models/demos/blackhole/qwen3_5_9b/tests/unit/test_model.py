# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end parity test for the Qwen3.5 text model (tt/model.py) vs the HF golden.

tt/model.py assembles embedding -> N hybrid decoder layers -> final RMSNorm -> LM head and drives
prefill + decode. The individual pieces (token mixers, MLP, norms, the whole decoder block) each have
their own unit tests; this test is the INTEGRATION check the others can't give: that the model glues
embedding+stack+norm+head together AND that decode correctly continues from the state prefill left
behind (the attention KV cache and the GDN conv/recurrent buffers) — the prefill->decode hand-off no
per-piece test exercises.

A small Qwen3_5ForCausalLM (the real config truncated to N_LAYERS, so it still spans BOTH layer kinds —
3 linear_attention + 1 full_attention) is built with random weights and used as the golden. Its
state_dict is remapped to the internal key scheme and handed to the TT model, so both compute with
identical weights. We PCC-check (1) prefill's next-token logits and (2) several greedy decode steps
that continue from the prefilled state, on a single device and the (1,4) tensor-parallel mesh.

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_model.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import text_config
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

### Test Parameters & Fixtures ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Truncate to the first 4 layer_types (["linear","linear","linear","full"]), so the tiny model still
# exercises both block kinds while keeping the random-weight ForCausalLM cheap to build.
N_LAYERS = 4
PREFILL_LEN = 128  # prompt length: a tile multiple and == the GDN chunk size (cleanest prefill)
DECODE_STEPS = 4  # greedy tokens stepped after prefill — the path generate() runs
MAX_SEQ = 256  # KV-cache length; must cover PREFILL_LEN + DECODE_STEPS
# Whole-stack RANDOM-weight bound, not the strict 0.99 the per-piece tests use. With random weights the
# residual error compounds across the 4 layers to ~0.94 even though each layer passes 0.99+ on its own
# (attention/gdn/mlp/layer tests hold that strict per-layer bound). This is the integration check for the
# embedding+stack+norm+head glue and the prefill->decode hand-off, so it uses the realistic accumulated
# bound; tightening toward 0.99 only makes sense when validating on REAL weights, not these.
PCC = 0.92

# Single device and the (1,4) TP mesh with the 1D fabric the reduce-scatters/all-gathers ride.
_MESH_PARAMS = [
    ((1, 1), {}),
    ((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
]


def _remap_hf_state_dict(state_dict):
    """HF Qwen3_5ForCausalLM keys -> the internal scheme the TT model + its loaders consume.

    The per-module weight loaders (attention/gdn/mlp/weights.py) all read the RAW transformers
    submodule names (q_proj/in_proj_qkv/conv1d/gate_proj/...), exactly as test_attention/test_gdn/
    test_layer feed them — so layer keys pass through with only the `model.` prefix stripped. Only the
    three top-level weights are renamed to what the framework Embedding / final RMSNorm / LM head
    look up: embed_tokens->tok_embeddings, model.norm->norm, lm_head->output."""
    out = {}
    for key, tensor in state_dict.items():
        if "visual" in key or key.startswith("mtp"):
            continue
        nk = key[len("model.") :] if key.startswith("model.") else key
        if nk == "embed_tokens.weight":
            out["tok_embeddings.weight"] = tensor
        elif key == "lm_head.weight":
            out["output.weight"] = tensor
        else:
            out[nk] = tensor  # norm.weight stays norm.weight; layers.N.* stay raw HF names
    return out


def _build_pair(mesh_device, max_seq_len):
    """Matched (cfg, HF Qwen3_5ForCausalLM golden, TT Qwen35Model) sharing one set of random weights.

    The HF model is the real config truncated to N_LAYERS; its state_dict is remapped to the internal
    key scheme and handed to the TT model so both compute with identical weights. The weight cache
    stays None — caching random weights would corrupt a later real-checkpoint run."""
    cfg = text_config()
    cfg.num_hidden_layers = N_LAYERS
    cfg.layer_types = list(cfg.layer_types[:N_LAYERS])
    hf = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()
    state_dict = _remap_hf_state_dict({k: v.float() for k, v in hf.state_dict().items()})

    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    args.n_layers = N_LAYERS
    args.attention_type_list = cfg.layer_types
    # dummy_weights only nulls the framework Embedding's cache filename (it still uses the real
    # state_dict weight). We pass no cache here on purpose — these ARE random weights, and caching
    # mesh-sharded random weights would corrupt a later real-checkpoint run (and waste GBs of disk).
    args.dummy_weights = True
    tt = Qwen35Model(mesh_device, args, state_dict, tensor_cache_path=None)
    return cfg, hf, tt


# ── Tests ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_model_prefill(mesh_device, reset_seeds, ensure_gc):
    """Prefill next-token logits PCC vs HF, on a single device and the (1,4) TP mesh."""
    cfg, hf, tt = _build_pair(mesh_device, MAX_SEQ)

    input_ids = torch.randint(0, cfg.vocab_size, (1, PREFILL_LEN))
    ref = hf(input_ids, use_cache=False).logits[0, -1].float()  # [vocab] at the last prompt position

    tt.reset_state()
    got = tt.prefill(input_ids)  # [vocab]

    passing, pcc = comp_pcc(ref, got, PCC)
    logger.info(f"model prefill PCC (mesh={tuple(mesh_device.shape)}) = {pcc}")
    assert passing, f"prefill logits PCC too low: {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_model_decode(mesh_device, reset_seeds, ensure_gc):
    """Greedy decode continuation PCC vs HF — the prefill->decode hand-off.

    Prefill seeds both models' state, then we greedily step DECODE_STEPS tokens, feeding each model the
    SAME token (argmax of the shared HF logit) so they stay in lock-step and any drift is the TT
    model's, not a divergent token stream. HF advances a DynamicCache; the TT model advances its own
    KV cache + GDN recurrent state.

    decode() defaults to the on-device argmax token for generation, so we PCC its exact NUMERICS by
    asking for the logits instead (decode(..., return_token=False)) and comparing in lock-step. We then
    separately assert that the default token path — argmax_on_device — matches a host argmax of the same
    logits, i.e. it is lossless. (A raw token-vs-HF check would be flaky here: random weights give
    near-flat logits where bf16 ties flip the argmax.)"""
    cfg, hf, tt = _build_pair(mesh_device, MAX_SEQ)

    input_ids = torch.randint(0, cfg.vocab_size, (1, PREFILL_LEN))
    hf_cache = DynamicCache(config=cfg)

    # Prefill both: HF fills hf_cache, TT fills its internal KV + GDN state.
    ref_logits = hf(input_ids, past_key_values=hf_cache, use_cache=True).logits[0, -1].float()
    tt.reset_state()
    got_logits = tt.prefill(input_ids)
    passing, pcc = comp_pcc(ref_logits, got_logits, PCC)
    logger.info(f"model decode step -1 (prefill) PCC (mesh={tuple(mesh_device.shape)}) = {pcc}")
    assert passing, f"prefill logits PCC too low: {pcc}"

    # Greedy decode continuation: feed the shared argmax token at its absolute position.
    for step in range(DECODE_STEPS):
        tok = int(torch.argmax(ref_logits).item())
        pos = PREFILL_LEN + step
        ref_logits = hf(torch.tensor([[tok]]), past_key_values=hf_cache, use_cache=True).logits[0, -1].float()

        # return_token=False surfaces decode()'s raw logits (one state advance) — the numerics we PCC.
        in_tok = torch.tensor([[tok]], dtype=torch.int32)
        pos_t = torch.tensor([pos], dtype=torch.int32)
        got_logits = tt.decode(in_tok, pos_t, return_token=False)[0]  # [vocab]

        passing, pcc = comp_pcc(ref_logits, got_logits, PCC)
        logger.info(f"model decode step {step} (pos {pos}) PCC (mesh={tuple(mesh_device.shape)}) = {pcc}")
        assert passing, f"decode logits PCC too low (step={step}): {pcc}"

    # decode()'s DEFAULT (return_token=True) is just argmax_on_device of those same logits — assert that
    # argmax is lossless vs a host argmax. This check needs BOTH the device argmax id AND a host argmax
    # of ONE forward's logits, which a single decode() call surfaces only one of, so it reaches into the
    # device logits directly here. One extra decode forward after the PCC loop, where the state advance
    # is harmless.
    pos = PREFILL_LEN + DECODE_STEPS
    in_tok = torch.tensor([[int(torch.argmax(ref_logits).item())]], dtype=torch.int32)
    dev_logits = tt.lm_head(tt.forward(in_tok, mode="decode", positions=torch.tensor([pos], dtype=torch.int32)))
    dev_tok = int(ttnn.to_torch(ttnn.get_device_tensors(tt.argmax_on_device(dev_logits))[0]).item())
    host_tok = int(torch.argmax(tt._logits_to_torch(dev_logits, n_rows=1)[0]).item())
    assert dev_tok == host_tok, f"argmax_on_device not lossless: {dev_tok} != {host_tok}"
