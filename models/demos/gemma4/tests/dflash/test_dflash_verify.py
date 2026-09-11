# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step 6 validation: the full draft -> verify -> accept loop, one iteration, on real T3K
hardware -- real prefill (as test_dflash_context.py), real drafter forward (as
test_dflash_drafter.py/test_dflash_logits.py), then a REAL verify against the target
model continuing from that SAME prefill's KV cache via Gemma4's existing
ttnn_verify_forward (tt/dflash/verify.py), checked against the torch reference's exact
accept/reject decision (dump_torch_verify_tile32.py).

Uses a 32-token prompt (tile-aligned ctx_len) rather than the original 18-token one --
see tt/dflash/verify.py's module docstring for the known Gemma4 kernel bug this avoids
(the first decode-style KV write after a non-tile-aligned prefill reads leftover
garbage from the unused rows of the last 32-row cache tile; a tile-aligned ctx_len has
no unused rows in that tile, so the bug's precondition never occurs). This is a
real, self-consistent prompt (not padding with throwaway filler) with its own matching
torch reference -- not a workaround baked into the DFlash code itself.

Run: pytest models/demos/gemma4/tests/dflash/test_dflash_verify.py -k 1x8 -s
"""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.context import compute_context
from models.demos.gemma4.tt.dflash.drafter import dflash_drafter_forward, dflash_drafter_update_kv_caches
from models.demos.gemma4.tt.dflash.lm_head import compute_dflash_logits, load_gemma4_lm_head_weight
from models.demos.gemma4.tt.dflash.verify import dflash_verify, greedy_accept_from_posterior, make_verify_buffers
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights
from models.tt_transformers.tt.common import PagedAttentionConfig

TORCH_REF_PATH = (
    "/tmp/claude-1002/-home-user-proj-sdk/4b2381cd-910e-4293-86a0-feecee22c8e9/scratchpad/torch_verify_ref_tile32.pt"
)
MAX_SEQ_LEN = 128


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL", "google/gemma-4-31B-it")


@parametrize_mesh_with_fabric([(1, 8)])
def test_dflash_verify_t3k(mesh_device, device_params, model_path):
    from models.common.utility_functions import comp_pcc

    ref = torch.load(TORCH_REF_PATH, map_location="cpu")
    ctx_len = ref["ctx_len"]
    block_size = ref["block_size"]
    input_ids = ref["input_ids"]
    real_first_token = ref["real_first_token"]
    noise_torch = ref["noise_embedding"]
    cos_torch = ref["cos"]
    sin_torch = ref["sin"]
    layer_configs = [(bool(c[0]), int(c[1]) if c[1] is not None else None) for c in ref["layer_configs"]]
    draft_tokens_ref = ref["draft_tokens_only"]
    posterior_ref = ref["posterior"]
    acceptance_length_ref = ref["acceptance_length"]
    bonus_ref = ref["bonus"]
    committed_ref = ref["committed"]

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)

    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)
    lm_head_weight = load_gemma4_lm_head_weight(mesh_device, mesh_config)

    # ---- real prefill (same as test_dflash_context.py) ----
    page_params = {"page_block_size": 64, "page_max_num_blocks": MAX_SEQ_LEN // 64}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks"]
    )
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )

    padded_len = MAX_SEQ_LEN
    input_ids_padded = torch.nn.functional.pad(input_ids.squeeze(0), (0, padded_len - ctx_len), value=0)
    embeds, _, _, _, _, _ = model.prepare_inputs_prefill(input_ids_padded.unsqueeze(0), page_table=page_table_tt)

    tapped = {}

    def _probe(layer_idx, hidden_states):
        if layer_idx in config.target_layer_ids:
            tapped[layer_idx] = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

    model.layer_probe = _probe
    try:
        prefill_logits = model.ttnn_prefill_forward(
            embeds,
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            get_last_token=-1,
            input_ids_torch=input_ids_padded.unsqueeze(0),
            embeds_torch=None,
        )
        ttnn.deallocate(prefill_logits)
    finally:
        model.layer_probe = None

    tapped_ordered = [tapped[i] for i in config.target_layer_ids]
    context_tt_padded = compute_context(weights, tapped_ordered)
    # Prefill ran the full padded_len (128) bucket; the tapped hidden states -- and so
    # context_tt_padded -- cover all 128 positions, of which only the first ctx_len are
    # the real prompt (the rest is padding-token garbage). Slice down to the real prompt
    # length before this feeds the drafter, or ctx_len inside dflash_attention_forward
    # comes out as 128 instead of 18 (hit exactly this on the first real run: a
    # ttnn.slice past the RoPE table's real length).
    context_tt = ttnn.slice(context_tt_padded, [0, 0, 0, 0], [1, 1, ctx_len, context_tt_padded.shape[-1]])
    ttnn.deallocate(context_tt_padded)
    context_back = ttnn.to_torch(ttnn.get_device_tensors(context_tt)[0]).float().reshape(1, ctx_len, -1)
    passing, pcc = comp_pcc(ref["context"], context_back, pcc=0.97)
    print(f"context PCC: {pcc}")
    assert passing, f"context PCC {pcc} below threshold"

    # ---- real drafter forward (real anchor token + mask block, from the torch reference) ----
    def to_tt(x, dtype=ttnn.bfloat16):
        return ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=replicate)

    noise_tt = to_tt(noise_torch.unsqueeze(0))
    cos_full_tt = to_tt(cos_torch.unsqueeze(0))
    sin_full_tt = to_tt(sin_torch.unsqueeze(0))
    head_dim = config.head_dim
    cos_ctx_tt = ttnn.slice(cos_full_tt, [0, 0, 0, 0], [1, 1, ctx_len, head_dim])
    sin_ctx_tt = ttnn.slice(sin_full_tt, [0, 0, 0, 0], [1, 1, ctx_len, head_dim])
    cos_noise_tt = ttnn.slice(cos_full_tt, [0, 0, ctx_len, 0], [1, 1, ctx_len + block_size, head_dim])
    sin_noise_tt = ttnn.slice(sin_full_tt, [0, 0, ctx_len, 0], [1, 1, ctx_len + block_size, head_dim])

    num_local_heads = config.num_attention_heads // mesh_config.tp
    num_local_kv_heads = config.num_key_value_heads // mesh_config.tp

    # One-shot (k_cache, v_cache) pair per layer, EXACTLY ctx_len-wide -- see
    # test_dflash_drafter.py.
    kv_caches = [
        (
            to_tt(torch.zeros(1, num_local_kv_heads, ctx_len, head_dim)),
            to_tt(torch.zeros(1, num_local_kv_heads, ctx_len, head_dim)),
        )
        for _ in weights.layers
    ]
    dflash_drafter_update_kv_caches(
        context_tt,
        ctx_len,
        weights,
        cos_ctx_tt,
        sin_ctx_tt,
        kv_caches,
        offset=0,
        num_local_heads=num_local_heads,
        num_local_kv_heads=num_local_kv_heads,
        head_dim=head_dim,
        eps=config.rms_norm_eps,
    )

    drafter_out = dflash_drafter_forward(
        kv_caches,
        noise_tt,
        weights,
        cos_noise_tt,
        sin_noise_tt,
        mesh_device,
        mesh_config,
        ccl_manager,
        num_local_heads,
        num_local_kv_heads,
        config.head_dim,
        config.rms_norm_eps,
        layer_configs,
        ctx_len,
    )
    final_out = weights.norm(drafter_out)
    logits = compute_dflash_logits(final_out, lm_head_weight, mesh_device, config.final_logit_softcapping)
    logits = logits.reshape(1, block_size, -1)
    draft_tokens = torch.argmax(logits, dim=-1)[0, 1:].tolist()  # positions 1..block_size-1 are the real drafts
    print(f"draft tokens (ttnn):  {draft_tokens}")
    print(f"draft tokens (torch): {draft_tokens_ref}")

    # ---- real verify: candidates = [real anchor token] + our own drafted tokens ----
    candidate_ids = [real_first_token] + draft_tokens
    verify_buffers = make_verify_buffers(mesh_device, page_table, block_size)
    posterior, _ = dflash_verify(model, mesh_device, tt_kv_cache, verify_buffers, candidate_ids, start_pos=ctx_len)
    acceptance_length, bonus, committed = greedy_accept_from_posterior(candidate_ids, posterior)

    print(f"posterior (ttnn):  {posterior.tolist()}")
    print(f"posterior (torch): {posterior_ref.tolist()}")
    print(f"acceptance_length: ttnn={acceptance_length} torch={acceptance_length_ref}")
    print(f"bonus: ttnn={bonus} torch={bonus_ref}")
    print(f"committed: ttnn={committed} torch={committed_ref}")

    draft_match = sum(a == b for a, b in zip(draft_tokens, draft_tokens_ref)) / len(draft_tokens_ref)
    print(
        f"draft token match rate: {draft_match:.2%} (informational -- Step 3-5 already validate the drafter's "
        f"own PCC/near-exact match separately; occasional bf16 near-tie divergence here, e.g. the 106/'<turn|>' "
        f"vs 107/'\\n' pair, is expected and does not by itself indicate a defect)"
    )

    # posterior positions AFTER the true accept boundary are conditioned on whichever draft
    # tokens preceded them -- once our drafter's own tokens diverge from the reference
    # drafter's tokens (bf16 near-tie noise, see above), the target's downstream posterior
    # legitimately differs too: it is answering a different question ("what follows THIS
    # token") for those positions. Only the prefix up to and including the accept boundary
    # is causally independent of that divergence, since acceptance stops at the first
    # mismatch regardless of what comes after -- that prefix is the one that must match
    # exactly for the accept/commit decision itself to be correct.
    boundary = acceptance_length_ref + 1
    posterior_prefix_match = torch.equal(posterior[:, :boundary], posterior_ref[:, :boundary])
    print(
        f"posterior[:{boundary}] (the accept-relevant prefix): ttnn={posterior[:, :boundary].tolist()} "
        f"torch={posterior_ref[:, :boundary].tolist()}"
    )

    assert posterior_prefix_match, (
        f"posterior diverged within the accept-relevant prefix: ttnn={posterior[:, :boundary].tolist()} "
        f"vs torch={posterior_ref[:, :boundary].tolist()}"
    )
    assert (
        acceptance_length == acceptance_length_ref
    ), f"accept length mismatch: ttnn={acceptance_length} vs torch={acceptance_length_ref}"
    assert committed == committed_ref, f"committed mismatch: ttnn={committed} vs torch={committed_ref}"
    print("[PASSED] real draft -> real verify against the real target -> accept/reject matches the torch reference")
