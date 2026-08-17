# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Advertised-context evidence: prefill and decode at the full HF-advertised 262144 tokens.

These are the capability-contract tests behind ``doc/context_contract.json``. They are slow
by construction (a 262144-token prefill of one layer) and are marked ``slow`` so the main
suite stays quick; the values they record are what the contract file claims.

The HF golden at this length uses the *tail* references in ``tt/reference.py``: every earlier
token's K/V (full attention) or conv+recurrent state (linear attention) is advanced in
O(seq), and the HF layer then runs normally over the last ``TAIL`` query positions, which
therefore attend to / carry the complete 262144-token context. This is exact for those
positions and avoids the O(seq^2) CPU cost of query positions the test does not compare.

Override the length for a smaller smoke run:

    QWEN36_LONG_CONTEXT=32768 pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_long_context.py
"""

import json
import os
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import (
    ARTIFACT_DIR,
    build_layer_pair,
    compare,
    from_tt,
    read_tt_linear_state,
    record,
    to_tt_decode,
    to_tt_positions,
    to_tt_prefill,
    tt_conv_state_to_hf,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

PCC_BAR = 0.995

#: HF-advertised context (``text_config.max_position_embeddings``). Overridable only to run
#: a cheaper smoke; the recorded contract evidence uses the full value.
LONG_CONTEXT = int(os.environ.get("QWEN36_LONG_CONTEXT", 262144))
#: number of trailing query positions compared against the exact HF tail reference
TAIL = int(os.environ.get("QWEN36_LONG_TAIL", 128))

KINDS = ["linear", "full"]

pytestmark = pytest.mark.slow


def _long_hidden(hf_config, seq_len, seed):
    """Deterministic long input built in slices to keep peak host memory bounded."""
    return ref.synthetic_hidden_states(hf_config, 1, seq_len, seed=seed)


def _record_contract(key, value):
    path = ARTIFACT_DIR.parent / "context_contract.json"
    contract = json.loads(path.read_text())
    contract[key] = max(contract.get(key) or 0, value)
    path.write_text(json.dumps(contract, indent=2) + "\n")


@pytest.fixture(scope="function")
def long_device():
    """Function-scoped on purpose: these are the largest allocations in the stage, so each
    test starts from an empty device DRAM rather than inheriting another test's buffers."""
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield device
    ttnn.close_mesh_device(device)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(21600)
def test_longest_prefill(long_device, kind):
    """Prefill the full advertised context, in one call, at a non-tile-aligned length.

    ``seq_len = LONG_CONTEXT - 1`` is deliberately the largest *non*-aligned length that
    fits, so the longest test also exercises the internal pad/mask/slice path rather than
    only friendly shapes.
    """
    seq_len = LONG_CONTEXT - 1
    pair = build_layer_pair(long_device, kind=kind, max_batch_size=1, supported_context=LONG_CONTEXT)
    pair.tt.reset_state()
    x = _long_hidden(pair.hf_config, seq_len, seed=1)

    tt_x = to_tt_prefill(long_device, x)
    ttnn.synchronize_device(long_device)
    started = time.perf_counter()
    tt_out = pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table)
    ttnn.synchronize_device(long_device)
    elapsed = time.perf_counter() - started
    got = from_tt(tt_out)[:, :, -TAIL:, :].reshape(1, TAIL, pair.cfg.hidden_size)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_out)

    if kind == "full":
        want = ref.hf_prefill_tail(pair.hf, pair.hf_config, x, tail=TAIL)
    else:
        want, hf_cache = ref.hf_linear_prefill_tail(pair.hf, pair.hf_config, x, tail=TAIL)

    result = compare(
        f"longest-prefill[{kind}] seq={seq_len} tail={TAIL}",
        got,
        want,
        seq_len=seq_len,
        tail=TAIL,
        wall_seconds=round(elapsed, 3),
    )
    record(result, "long_context")

    if kind == "linear":
        # also pin the carried state at full length: it is what decode continues from
        taps, recurrent = read_tt_linear_state(pair)
        want_conv, want_recurrent = ref.hf_linear_attention_state(hf_cache, pair.layer_idx)
        got_conv = tt_conv_state_to_hf(pair, [tap[0:1] for tap in taps])
        state_conv = compare("longest-prefill state conv", got_conv[..., 1:], want_conv[..., 1:])
        state_rec = compare("longest-prefill state recurrent", recurrent[0:1], want_recurrent)
        record([state_conv, state_rec], "long_context")
        assert state_conv.pcc >= PCC_BAR, state_conv
        assert state_rec.pcc >= PCC_BAR, state_rec

    _record_contract("largest_prefill_tested", seq_len)
    assert result.pcc >= PCC_BAR, result


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.timeout(21600)
def test_longest_decode_context(long_device, kind):
    """Decode the last position of the advertised context.

    Both kinds build the state the *real* way — by prefilling 262143 tokens through the layer —
    and the HF side advances its own cache/state over the same tokens in O(seq), so this is the
    contract's own path rather than a synthetic cache.

    This case's full-attention decode PCC drove README section 3.8: it was the one number in the
    stage materially below the rest. That investigation ended at ``SDPAProgramConfig::k_chunk_size``
    — the decode SDPA's bf16 accumulation depth — reproduced with no model code at all in
    ``tests/diag_sdpa_decode.py`` and settled on the real layer in
    ``tests/diag_decode_sdpa_onmodel.py``. Shipping the largest legal chunk (512) took this case from
    **0.9986** (the op's own default, ``k_chunk_size=32``) through 0.9997685 (128) to
    **0.9999939**, in line with every other context, while also being 3.8x faster than the op
    default and 1.55x faster than 128.

    Three earlier explanations in this docstring were wrong and are gone: the conditioning of random
    K/V; 0.9986 quoted as if it were the current value; and ``max_cores_per_head_batch`` named as the
    variable, which it is not — the paged op already defaults to 1 core/head, so it never differed
    between the settings being compared (``tt/functional_decoder.py`` ->
    ``_decode_sdpa_program_config``).
    """
    position = LONG_CONTEXT - 1
    pair = build_layer_pair(long_device, kind=kind, max_batch_size=1, supported_context=LONG_CONTEXT)
    pair.tt.reset_state()

    x = _long_hidden(pair.hf_config, position, seed=2)
    tt_x = to_tt_prefill(long_device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=0, page_table=pair.page_table))
    ttnn.deallocate(tt_x)
    if kind == "full":
        hf_cache = ref.hf_fill_full_attention_cache(pair.hf, pair.hf_config, x)
    else:
        hf_cache = ref.hf_linear_attention_chunked(pair.hf, pair.hf_config, x)
    del x

    token = ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=3)
    tt_tok = to_tt_decode(long_device, token.reshape(1, 1, -1))
    tt_pos = to_tt_positions(long_device, torch.tensor([position]))
    ttnn.synchronize_device(long_device)
    started = time.perf_counter()
    tt_out = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=pair.page_table)
    ttnn.synchronize_device(long_device)
    elapsed = time.perf_counter() - started
    got = from_tt(tt_out).reshape(1, 1, pair.cfg.hidden_size)
    for t in (tt_tok, tt_pos, tt_out):
        ttnn.deallocate(t)

    want = ref.hf_decode(pair.hf, pair.hf_config, token, positions=torch.tensor([position]), cache=hf_cache)
    result = compare(
        f"longest-decode[{kind}] pos={position}",
        got,
        want,
        position=position,
        wall_seconds=round(elapsed, 3),
    )
    record(result, "long_context")
    _record_contract("largest_decode_context_tested", position + 1)
    assert result.pcc >= PCC_BAR, result


@pytest.mark.timeout(7200)
def test_longest_decode_context_batched(long_device):
    """The advertised context decoded with **two** slots live, not one.

    Why this exists: every measurement behind the README section 3.8 decode-SDPA choice is batch 1 —
    the op sweep, the on-model comparison and the sibling test above. The shipped config pins both
    ``k_chunk_size`` and ``max_cores_per_head_batch``, and the factory divides the core budget by the
    batch (``sdpa_decode_program_factory.cpp:195-196``), so decoding at batch > 1 is a different
    program. Without this case, that the choice still holds above one slot would be a source-level
    argument rather than a measurement. ``full`` only: the linear recurrence has no cross-slot
    coupling and no SDPA.

    Slot 1 holds the advertised context and is the one compared; slot 0 holds a shorter sequence with
    a different seed and decodes with ``current_pos = -1``, so the two slots hold genuinely different
    state and a slot-indexing error cannot pass unnoticed.
    """
    position = LONG_CONTEXT - 1
    pair = build_layer_pair(long_device, kind="full", max_batch_size=2, supported_context=LONG_CONTEXT)
    pair.tt.reset_state()

    decoy = _long_hidden(pair.hf_config, 4096, seed=7)
    tt_decoy = to_tt_prefill(long_device, decoy)
    ttnn.deallocate(pair.tt.prefill_forward(tt_decoy, user_id=0, page_table=pair.page_table))
    ttnn.deallocate(tt_decoy)
    del decoy

    x = _long_hidden(pair.hf_config, position, seed=2)
    tt_x = to_tt_prefill(long_device, x)
    ttnn.deallocate(pair.tt.prefill_forward(tt_x, user_id=1, page_table=pair.page_table))
    ttnn.deallocate(tt_x)
    hf_cache = ref.hf_fill_full_attention_cache(pair.hf, pair.hf_config, x)
    del x

    token = ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=3)
    both = torch.cat([torch.zeros_like(token), token], dim=0).reshape(2, 1, -1)
    tt_tok = to_tt_decode(long_device, both)
    tt_pos = to_tt_positions(long_device, torch.tensor([-1, position]))
    tt_out = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=pair.page_table)
    got = from_tt(tt_out).reshape(2, 1, pair.cfg.hidden_size)[1:2]
    for t in (tt_tok, tt_pos, tt_out):
        ttnn.deallocate(t)

    want = ref.hf_decode(pair.hf, pair.hf_config, token, positions=torch.tensor([position]), cache=hf_cache)
    result = compare(
        f"batched-longest-decode[full] pos={position} batch=2 slot=1",
        got,
        want,
        position=position,
        batch=2,
        slot=1,
    )
    record(result, "long_context")
    assert result.pcc >= PCC_BAR, result


@pytest.mark.timeout(3600)
def test_max_batch_full_context_capacity(long_device):
    """Byte accounting + a real allocation probe for batch 32 at the advertised context.

    The contract claims no capability reduction, so the paged cache for the advertised
    context at the advertised batch has to actually fit in device DRAM. This allocates it.
    """
    hf = ref.load_hf_text_config()
    batch, block, n_kv, hd = 32, 64, hf.num_key_value_heads, hf.head_dim
    blocks = batch * (LONG_CONTEXT // block)
    per_cache = blocks * n_kv * block * hd * 2  # bf16
    device = long_device
    caches = []
    try:
        for _ in range(2):  # K and V
            caches.append(
                ttnn.from_torch(
                    torch.zeros(blocks, n_kv, block, hd, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
                )
            )
        record(
            [
                {
                    "label": "kv-capacity batch32 full context",
                    "batch": batch,
                    "context": LONG_CONTEXT,
                    "blocks": blocks,
                    "bytes_per_cache": per_cache,
                    "bytes_total": 2 * per_cache,
                    "allocated": True,
                }
            ],
            "long_context",
        )
    finally:
        for cache in caches:
            ttnn.deallocate(cache)
