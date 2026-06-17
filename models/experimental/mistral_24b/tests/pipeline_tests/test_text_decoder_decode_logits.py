# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-step logits PCC test for the Mistral-Small-3.1-24B text decoder (text-only).

Differs from other text-decoder tests:
  * ``test_text_decoder.py`` — decode hidden states only (no ``lm_head``), synthetic activations.
  * ``test_text_decoder_prefill_logits.py`` — prefill last-token logits over long sequences.
  * This test — after a real-token prefill (verified with PCC ≥ 0.99), compares per-step
    **decode logits** (full model, ``norm`` + ``lm_head``) against HF for 32 generation steps.
    Next-step tokens follow the ``test_model.py`` pattern: teacher-force ground-truth prompt
    tokens while they remain, then ``sample_host`` from the logits (temperature=0) so TT and
    HF KV caches stay aligned.

Modeled on ``models/tt_transformers/tests/test_model.py`` decode loop.

PCC thresholds:
  * Prefill last-token logits: ≥ 0.99
  * Per-step decode logits:    ≥ 0.91  (each of the 32 steps)
"""

import bz2
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_blackhole
from models.experimental.mistral_24b.tests.pipeline_tests.test_end2end import (
    fabric_1d_trace_device_params,
    setup_vision_model_args,
)
from models.experimental.mistral_24b.tt.generator import MistralGenerator
from models.experimental.mistral_24b.tt.model import MistralTransformer as Transformer
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, sample_host
from models.tt_transformers.tt.model_config import DecodersPrecision

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)
PROMPT_FILE = os.path.join(TT_METAL_HOME, "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("prefill_len", (128,), ids=["128"])
@pytest.mark.parametrize("generation_length", (32,), ids=["32"])
@pytest.mark.parametrize(
    "max_seq_len",
    (128 * 1024,),
    ids=["max128k"],
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "device_params",
    fabric_1d_trace_device_params(num_command_queues=2),
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "P150x4": (1, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_text_decoder_decode_logits(
    prefill_len, generation_length, max_seq_len, page_params, mesh_device, reset_seeds
):
    """Prefill real tokens (PCC ≥ 0.99), then compare 32 decode-step logits vs HF (``test_model.py`` token policy)."""
    pcc_required_prefill = 0.99
    pcc_required_decode = 0.91
    dtype = ttnn.bfloat8_b
    batch_size = 1

    optimizations = lambda margs: DecodersPrecision.accuracy(margs.n_layers, margs.model_name)
    model_args, instruct = setup_vision_model_args("instruct", max_seq_len, batch_size, mesh_device, optimizations)

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -2) if batch_size > 1 else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    state_dict = model_args.load_state_dict()
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [[layer.attention.layer_past for layer in tt_model.layers]]
    generator = MistralGenerator([tt_model], [model_args], mesh_device, tokenizer=model_args.tokenizer)

    with bz2.open(PROMPT_FILE, "rt", encoding="utf-8") as f:
        prompt = f.read()
    encoded_prompt = model_args.encode_prompt(prompt, instruct=instruct)[: prefill_len + generation_length + 1]
    assert len(encoded_prompt) > prefill_len, "Need tokens beyond prefill_len for teacher-forced decode steps"

    encoded_prompt_tensor = torch.tensor(encoded_prompt)
    tt_prefill_input = encoded_prompt_tensor[:prefill_len].unsqueeze(0)

    # --- TT prefill (populates KV cache) ---
    tt_prefill_logits = generator.prefill_forward_text(
        tt_prefill_input,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[prefill_len],
    )

    # --- HF prefill (populates reference KV cache) ---
    reference_model = model_args.reference_transformer(load_checkpoint=True)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    embd = model_args.reference_embedding()
    weight = state_dict[f"{state_dict_prefix}tok_embeddings.weight"]
    embd.load_state_dict({"emb.weight": weight})

    pt_prefill = embd(encoded_prompt_tensor[:prefill_len]).view(batch_size, prefill_len, -1)
    ref_prefill_logits = reference_model(pt_prefill, start_pos=0)
    ref_last_prefill = ref_prefill_logits[:, -1:, : model_args.vocab_size]

    passing, pcc_message = comp_pcc(ref_last_prefill, tt_prefill_logits[:, :, : model_args.vocab_size], pcc_required_prefill)
    logger.info(f"Prefill last-token logits PCC: {pcc_message}")
    assert passing, f"Prefill last-token logits PCC below {pcc_required_prefill}. {pcc_message}"

    # --- Decode loop (mirrors ``test_model.py``) ---
    # Per-step PCC results are collected and printed as a summary table at the end.
    decode_pcc_results = []  # list of (step, position, pcc_value, passed)

    seqlen = 1
    batch = model_args.max_batch_size
    generation_start_pos = prefill_len

    first_token_idx = prefill_len
    pt_decode_input = embd(encoded_prompt_tensor[first_token_idx : first_token_idx + 1]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(
        mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
    )

    all_tests_pass = True
    for i in range(generation_length):
        logger.info(f"[Decode logits] step {i} at position {current_pos[0].item()}")

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.get_residual_mem_config(Mode.DECODE, None),
        )

        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
        rot_mats_local = (
            tt_model.rope_local_setup.get_rot_mats(current_pos) if hasattr(tt_model, "rope_local_setup") else None
        )

        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=Mode.DECODE,
            page_table=page_table_tt,
        )

        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        )
        ttnn.deallocate(tt_out)

        ref_output = reference_model(pt_decode_input, current_pos[0])

        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        token_idx = prefill_len + i
        if token_idx in range(len(encoded_prompt)):
            tt_decode_input = embd(encoded_prompt_tensor[token_idx : token_idx + 1]).view(batch, seqlen, -1)
            pt_decode_input = tt_decode_input
        else:
            _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
            pt_decode_input = embd(pt_out_tok)
            tt_decode_input = pt_decode_input

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required_decode)
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"Decode logits PCC (pos={generation_start_pos + i}): {pcc_message}")

        # Extract numeric PCC value from message string (format: "PCC: X.XXXXXX")
        import re as _re
        _pcc_match = _re.search(r"PCC:\s*([\d.eE+\-]+)", pcc_message)
        _pcc_val = float(_pcc_match.group(1)) if _pcc_match else float("nan")
        decode_pcc_results.append((i, generation_start_pos + i, _pcc_val, passing))

        if not passing:
            logger.warning(f"Decode logits failed at position {generation_start_pos + i}")
            all_tests_pass = False

    # --- PCC summary table (32 generation steps) ---
    logger.info("")
    logger.info("=== Decode Logits PCC Summary (32 generation steps) ===")
    logger.info(f"{'Step':>4}  {'Position':>8}  {'PCC':>10}  {'Pass?':>6}")
    logger.info("-" * 38)
    for step, pos, pcc_val, passed in decode_pcc_results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"{step:>4}  {pos:>8}  {pcc_val:>10.6f}  {status:>6}")
    logger.info("-" * 38)
    passing_count = sum(1 for _, _, _, p in decode_pcc_results if p)
    logger.info(f"Passed: {passing_count} / {len(decode_pcc_results)}  (threshold: PCC ≥ {pcc_required_decode})")
    logger.info("=======================================================")
    logger.info("")

    assert all_tests_pass, f"Decode logits PCC below {pcc_required_decode} for one or more steps."
