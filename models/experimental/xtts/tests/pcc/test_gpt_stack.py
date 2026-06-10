"""PCC tests for the TTNN XTTS GPT decoder: full 30-layer stack + KV-cache decode.

Self-contained — the reference XTTS-v2 model is loaded inside this file (weights
AND golden come from it), so there are no pre-dumped tensor dependencies.

  * test_gpt_block        — single GPT2 decoder block vs the reference GPT2Block
    module (xtts.gpt.gpt.h[0]) on a correctly-shaped [1, S, 1024] input.
  * test_gpt_stack_prefill — runs a prefill sequence through all 30 TTNN blocks
    (+ ln_f) and compares to the reference GPT2 block stack run on the same input.
  * test_kv_cache_decode  — validates the KV-cache decode invariant: a token
    decoded from the cache (after prefilling the prior tokens) matches the same
    position from a full causal prefill of the reference stack.

Environment (set before running):
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH="${TT_METAL_HOME}"
    export TRACY_NO_INVARIANT_CHECK=1
    export BH_ARCH_YAML=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
    export MESH_DEVICE=P150
    export COQUI_TOS_AGREED=1   # accept XTTS license; first run downloads ~1.9 GB

    pytest models/experimental/xtts/tests/pcc/test_gpt_stack.py
"""

import os
import sys

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tt")))
import gpt2_block as G  # noqa: E402
from model_config import load_reference_state_dict  # noqa: E402

N_LAYERS = 30
HIDDEN = 1024
PCC = 0.99


# --- fixtures ------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def reference():
    """Load the reference XTTS-v2 once: returns (gpt2_blocks, ln_f, state_dict)."""
    model, sd = load_reference_state_dict("cpu")
    return model.gpt.gpt.h, model.gpt.gpt.ln_f, sd


def _to_dev(t, device):
    return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _ref_stack(blocks, x):
    """Reference: causal prefill through every GPT2Block (returns h.29 output)."""
    with torch.no_grad():
        for blk in blocks:
            x = blk(x)[0]
    return x


# --- single GPT2 block vs reference module -------------------------------------
@pytest.mark.parametrize("seq_len", [32, 67, 96])
def test_gpt_block(device, reference, seq_len):
    blocks, _, sd = reference
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1  # GPT2Block input [1, S, 1024]

    with torch.no_grad():
        ref = blocks[0](x)[0]  # reference GPT2Block (causal)

    p = G.load_block_params(sd, device, prefix="gpt.gpt.h.0.")
    got = ttnn.to_torch(G.gpt2_block(_to_dev(x, device), p)).float()[:, :seq_len, :]

    ok, msg = comp_pcc(ref, got, PCC)
    print(f"\nseq={seq_len}  gpt2_block: {msg}")
    assert ok, msg


# --- (b) full 30-layer stack prefill -------------------------------------------
@pytest.mark.parametrize("seq_len", [32, 67])
def test_gpt_stack_prefill(device, reference, seq_len):
    blocks, ln_f, sd = reference
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    with torch.no_grad():
        ref_h29 = _ref_stack(blocks, x)
        ref_lnf = ln_f(ref_h29)

    layers = G.load_stack_params(sd, device, N_LAYERS)
    h29 = G.stack_prefill(_to_dev(x, device), layers)
    out = G.layer_norm(h29, _to_dev(sd["gpt.gpt.ln_f.weight"], device), _to_dev(sd["gpt.gpt.ln_f.bias"], device))

    got_h29 = ttnn.to_torch(h29).float()[:, :seq_len, :]
    got_lnf = ttnn.to_torch(out).float()[:, :seq_len, :]
    ok1, m1 = comp_pcc(ref_h29, got_h29, PCC)
    ok2, m2 = comp_pcc(ref_lnf, got_lnf, PCC)
    print(f"\nseq={seq_len}  stack h.29: {m1}\nseq={seq_len}  stack+ln_f: {m2}")
    assert ok1 and ok2, f"h29={m1} lnf={m2}"


# --- (a) KV-cache decode invariant ---------------------------------------------
@pytest.mark.parametrize("prefill_len", [64])
def test_kv_cache_decode(device, reference, prefill_len):
    blocks, _, sd = reference
    torch.manual_seed(0)
    L = prefill_len + 1
    x = torch.randn(1, L, HIDDEN) * 0.1

    # reference: full causal prefill; target = the last position
    ref = _ref_stack(blocks, x)[:, prefill_len, :]  # [1, 1024]

    layers = G.load_stack_params(sd, device, N_LAYERS)
    caches = G.init_kv_cache(device, N_LAYERS, max_seq=128)

    G.stack_prefill_with_cache(_to_dev(x[:, :prefill_len, :], device), layers, caches)
    dec = G.stack_decode(_to_dev(x[:, prefill_len:L, :], device), layers, caches, cur_pos=prefill_len)
    got = ttnn.to_torch(dec).float().reshape(1, -1)[:, :HIDDEN]

    ok, msg = comp_pcc(ref.reshape(1, -1), got, PCC)
    print(f"\nkv-cache decode @pos {prefill_len}: {msg}")
    assert ok, msg
