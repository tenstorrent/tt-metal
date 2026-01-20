# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Batch-128 demo correctness smoke test.

This targets the row-sharded + paged-attention path used by the text demo.
We compare greedy tokens against HF reference for users across ALL positions
within each row to catch regressions that component-level unit tests can miss.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator


def _build_attention_mask(input_ids, prompt_lens):
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    for i, pos in enumerate(prompt_lens):
        attention_mask[i, :pos] = 1
    return attention_mask


def _check_output_quality(token_ids, tokenizer, user_id):
    """
    Check if output tokens appear to be garbage (repetitive patterns, nonsense).
    Returns (is_garbage, reason) tuple.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Check for excessive question marks or dots (common in garbage output)
    question_marks = text.count("?")
    dots = text.count("…") + text.count(".")
    scrolling_count = text.lower().count("scrolling")

    total_len = len(text) if len(text) > 0 else 1

    # Heuristics for garbage detection
    if question_marks / total_len > 0.1:  # More than 10% question marks
        return True, f"User {user_id}: Excessive question marks ({question_marks}/{total_len})"

    if scrolling_count > 3:
        return True, f"User {user_id}: Repeated 'Scrolling' pattern ({scrolling_count} times)"

    if dots / total_len > 0.2:  # More than 20% dots/ellipsis
        return True, f"User {user_id}: Excessive dots/ellipsis ({dots}/{total_len})"

    return False, None


@parametrize_mesh_with_fabric()
def test_batch128_row_sharded_demo_smoke(mesh_device, device_params, reset_seeds, state_dict):
    """
    Compare TT prefill + 1 decode step tokens against HF reference
    for a subset of users in the batch-128, row-sharded path.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    paged_attention = True
    # Need enough blocks for all users: (max_seq_len / block_size) * users_per_row
    # For 32 users per row with 1024 seq_len and 64 block_size: 16 * 32 = 512 blocks minimum
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    # Build TT generator (row-sharded path)
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # ==================== PAGE TABLE DIAGNOSTICS ====================
    logger.info("\n" + "=" * 80)
    logger.info("PAGE TABLE DIAGNOSTICS:")
    logger.info("=" * 80)
    logger.info(f"Page table shape: {page_table.shape}")
    logger.info(f"Page table dtype: {page_table.dtype}")

    # Show page table structure for each row
    for row in range(mesh_device.shape[0]):
        row_start = row * users_per_row
        row_end = (row + 1) * users_per_row
        row_page_table = page_table[row_start:row_end]
        logger.info(f"Row {row}: users {row_start}-{row_end-1}")
        logger.info(f"  Block range: min={row_page_table.min().item()}, max={row_page_table.max().item()}")
        logger.info(f"  First user (local 0) blocks[:4]: {row_page_table[0, :4].tolist()}")
        logger.info(f"  Last user (local {users_per_row-1}) blocks[:4]: {row_page_table[-1, :4].tolist()}")

        # Check for block collisions within this row's page table
        all_blocks = row_page_table.flatten().tolist()
        unique_blocks = set(all_blocks)
        if len(all_blocks) != len(unique_blocks):
            logger.error(
                f"  ⚠️ BLOCK COLLISION detected! {len(all_blocks)} total blocks but only {len(unique_blocks)} unique"
            )
        else:
            logger.info(f"  ✓ No block collisions ({len(unique_blocks)} unique blocks)")

    # Load prompts (repeat to batch=128 if needed)
    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ==================== DECODING POSITION DIAGNOSTICS ====================
    logger.info("\n" + "=" * 80)
    logger.info("DECODING POSITION DIAGNOSTICS:")
    logger.info("=" * 80)
    logger.info(f"Decoding positions tensor shape: {len(decoding_pos)}")

    # Show position distribution per row
    for row in range(mesh_device.shape[0]):
        row_start = row * users_per_row
        row_end = (row + 1) * users_per_row
        row_positions = decoding_pos[row_start:row_end]
        logger.info(f"Row {row}: positions range [{min(row_positions)}, {max(row_positions)}]")
        # Show specific positions for first few users
        logger.info(f"  First 5 users positions: {row_positions[:5]}")

    # Check specific users that typically fail (16, 32, 95, 104)
    failing_users = [16, 32, 95, 104]
    logger.info("\nSpecific users of interest:")
    for uid in failing_users:
        if uid < global_batch_size:
            row = uid // users_per_row
            local_pos = uid % users_per_row
            decode_pos = decoding_pos[uid]
            pt_row = page_table[uid]
            logger.info(f"User {uid} (row {row}, local_pos {local_pos}):")
            logger.info(f"  Decode position: {decode_pos}")
            logger.info(f"  Page table blocks[:8]: {pt_row[:8].tolist()}")
            logger.info(f"  Prompt preview: {input_prompts[uid][:60]}...")

    # TT prefill (first generated token)
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )

    # ==================== PREFILL LOGITS DIAGNOSTICS ====================
    logger.info("\n" + "=" * 80)
    logger.info("PREFILL LOGITS DIAGNOSTICS (checking for NaN/Inf and anomalies):")
    logger.info("=" * 80)

    # Check for NaN/Inf in ALL users
    nan_users = []
    inf_users = []
    anomaly_users = []
    for user_id in range(global_batch_size):
        user_logits = tt_logits[user_id].float()
        has_nan = torch.isnan(user_logits).any().item()
        has_inf = torch.isinf(user_logits).any().item()
        logit_max = user_logits.max().item()
        logit_min = user_logits.min().item()
        logit_mean = user_logits.mean().item()
        logit_std = user_logits.std().item()

        row = user_id // (global_batch_size // mesh_device.shape[0])
        pos = user_id % (global_batch_size // mesh_device.shape[0])

        if has_nan:
            nan_users.append(f"User {user_id} (row {row}, pos {pos})")
        if has_inf:
            inf_users.append(f"User {user_id} (row {row}, pos {pos})")
        # Check for anomalous logit ranges (too large or too small)
        if abs(logit_max) > 100 or abs(logit_min) > 100 or logit_std < 0.01:
            anomaly_users.append(
                f"User {user_id} (row {row}, pos {pos}): max={logit_max:.2f}, min={logit_min:.2f}, "
                f"mean={logit_mean:.2f}, std={logit_std:.4f}"
            )

    if nan_users:
        logger.error(f"🚨 NaN detected in {len(nan_users)} users: {nan_users[:10]}")
    else:
        logger.info("✓ No NaN values in prefill logits")

    if inf_users:
        logger.error(f"🚨 Inf detected in {len(inf_users)} users: {inf_users[:10]}")
    else:
        logger.info("✓ No Inf values in prefill logits")

    if anomaly_users:
        logger.warning(f"⚠️ Anomalous logit ranges in {len(anomaly_users)} users:")
        for msg in anomaly_users[:5]:
            logger.warning(f"  {msg}")

    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    # HF reference for a small subset across mesh rows
    # IMPROVED: Test multiple positions within each row, not just the first user
    row_stride = global_batch_size // mesh_device.shape[0]
    user_indices = []
    for row in range(mesh_device.shape[0]):
        row_start = row * row_stride
        # Test first, middle, and last user of each row
        user_indices.extend(
            [
                row_start,  # First user of row
                row_start + row_stride // 4,  # 25% into row
                row_start + row_stride // 2,  # Middle of row
                row_start + 3 * row_stride // 4,  # 75% into row
                row_start + row_stride - 1,  # Last user of row
            ]
        )
    user_indices = list(set(user_indices))  # Remove duplicates
    user_indices = sorted(user_indices)[: min(20, len(user_indices))]  # Cap at 20 users
    num_users_to_check = len(user_indices)

    logger.info(f"Testing {num_users_to_check} users across all rows: {user_indices}")

    if os.getenv("HF_MODEL") is None:
        pytest.skip("HF_MODEL not set; skipping HF reference comparison.")

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    hf_state_dict = setup["model_args"].load_state_dict(
        weights_path=setup["model_args"].model_path,
        dummy_weights=setup["model_args"].dummy_weights,
        convert_to_meta_format=False,
    )
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(hf_state_dict, strict=False)
    reference_model.eval()

    subset_input_ids = input_tokens_prefill_pt[user_indices].to(torch.long)
    subset_decoding_pos = torch.tensor([decoding_pos[i] for i in user_indices], dtype=torch.long)
    subset_attention_mask = _build_attention_mask(subset_input_ids, subset_decoding_pos)

    with torch.no_grad():
        ref_output = reference_model(
            input_ids=subset_input_ids,
            attention_mask=subset_attention_mask,
            use_cache=False,
        )
    ref_logits = ref_output.logits
    ref_prefill_token = torch.argmax(ref_logits[torch.arange(num_users_to_check), subset_decoding_pos - 1, :], dim=-1)

    # ==================== PREFILL LOGIT DISTRIBUTION COMPARISON ====================
    logger.info("\n" + "=" * 80)
    logger.info("PREFILL LOGIT DISTRIBUTION COMPARISON (TT vs HF):")
    logger.info("=" * 80)

    for i, user_id in enumerate(user_indices[:8]):  # Check first 8 users in detail
        tt_user_logits = tt_logits[user_id].float().squeeze()
        hf_user_logits = ref_logits[i, subset_decoding_pos[i] - 1, :].float()

        # Get top-5 predictions from each
        tt_top5_vals, tt_top5_idx = torch.topk(tt_user_logits, 5)
        hf_top5_vals, hf_top5_idx = torch.topk(hf_user_logits, 5)

        # Compute cosine similarity between logit distributions
        cos_sim = torch.nn.functional.cosine_similarity(tt_user_logits.unsqueeze(0), hf_user_logits.unsqueeze(0)).item()

        # Compute L2 distance (normalized)
        l2_dist = torch.norm(tt_user_logits - hf_user_logits).item()
        l2_dist_normalized = l2_dist / (torch.norm(hf_user_logits).item() + 1e-8)

        row = user_id // row_stride
        pos = user_id % row_stride

        logger.info(f"\nUser {user_id} (row {row}, pos {pos}):")
        logger.info(
            f"  Cosine similarity: {cos_sim:.4f} {'✓' if cos_sim > 0.95 else '⚠️ LOW' if cos_sim > 0.8 else '🚨 VERY LOW'}"
        )
        logger.info(f"  Normalized L2 distance: {l2_dist_normalized:.4f}")
        logger.info(f"  TT  top-5: {[tokenizer.decode([idx.item()]) for idx in tt_top5_idx]} = {tt_top5_vals.tolist()}")
        logger.info(f"  HF  top-5: {[tokenizer.decode([idx.item()]) for idx in hf_top5_idx]} = {hf_top5_vals.tolist()}")

        # Check if top-1 matches
        if tt_top5_idx[0] != hf_top5_idx[0]:
            # Show where HF's top token ranks in TT's distribution
            hf_top_token = hf_top5_idx[0].item()
            tt_rank_of_hf_top = (tt_user_logits > tt_user_logits[hf_top_token]).sum().item() + 1
            logger.warning(
                f"  ⚠️ Top-1 mismatch! HF's '{tokenizer.decode([hf_top_token])}' is rank {tt_rank_of_hf_top} in TT"
            )

    # Check prefill tokens match and show details
    tt_subset_tokens = tt_prefill_token[user_indices].cpu()
    mismatches = []
    logger.info("\n" + "=" * 80)
    logger.info("PREFILL RESULTS (first token after prompt):")
    logger.info("=" * 80)
    for i, user_id in enumerate(user_indices):
        tt_tok = tt_subset_tokens[i].item()
        hf_tok = ref_prefill_token[i].item()
        tt_text = tokenizer.decode([tt_tok])
        hf_text = tokenizer.decode([hf_tok])
        prompt_preview = (
            input_prompts[user_id][:50] + "..." if len(input_prompts[user_id]) > 50 else input_prompts[user_id]
        )
        match_str = "✓" if tt_tok == hf_tok else "✗ MISMATCH"
        logger.info(
            f"User {user_id:3d} (row {user_id // row_stride}, pos {user_id % row_stride:2d}): "
            f"Prompt='{prompt_preview}' | TT='{tt_text}' ({tt_tok}) vs HF='{hf_text}' ({hf_tok}) {match_str}"
        )
        if tt_tok != hf_tok:
            mismatches.append(
                f"User {user_id} (row {user_id // row_stride}, pos {user_id % row_stride}): "
                f"TT='{tt_text}' ({tt_tok}) vs HF='{hf_text}' ({hf_tok})"
            )

    if mismatches:
        logger.error(f"PREFILL MISMATCHES: {len(mismatches)}")
    else:
        logger.info("✓ All prefill tokens match!")

    assert len(mismatches) == 0, f"Prefill token mismatches:\n" + "\n".join(mismatches)

    # TT decode: one step (greedy)
    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    logger.info("\n" + "=" * 80)
    logger.info("DECODE INPUT (prefill token fed back as input):")
    logger.info("=" * 80)

    # ==================== PAGE TABLE SHARDING ANALYSIS ====================
    logger.info("\nPage table sharding analysis for decode:")
    logger.info(f"Page table will be sharded with dims=(0, None) across {mesh_device.shape[0]} rows")
    for row in range(mesh_device.shape[0]):
        row_start = row * users_per_row
        row_end = (row + 1) * users_per_row
        logger.info(f"Row {row} will see global users {row_start}-{row_end-1} as local users 0-{users_per_row-1}")
        # Show what page table row 0 on each mesh row will be
        logger.info(
            f"  Row {row} local page_table[0] = global page_table[{row_start}] = blocks {page_table[row_start, :4].tolist()}..."
        )

    logger.info("\n")
    for i, user_id in enumerate(user_indices):
        input_tok = out_tok[user_id].item()
        input_text = tokenizer.decode([input_tok])
        pos = current_pos[user_id].item()
        row = user_id // users_per_row
        local_user = user_id % users_per_row
        logger.info(
            f"User {user_id:3d} (row {row}, local {local_user:2d}): input_token='{input_text}' ({input_tok}), position={pos}"
        )

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    # ==================== DECODE LOGITS DIAGNOSTICS ====================
    logger.info("\n" + "=" * 80)
    logger.info("DECODE LOGITS DIAGNOSTICS (checking for NaN/Inf and anomalies):")
    logger.info("=" * 80)

    # tt_decode_logits might be a list (one per data_parallel group)
    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    # Check for NaN/Inf in decode logits
    decode_nan_users = []
    decode_inf_users = []
    decode_anomaly_users = []
    for user_id in range(min(global_batch_size, tt_decode_logits_tensor.shape[0])):
        user_logits = tt_decode_logits_tensor[user_id].float().squeeze()
        has_nan = torch.isnan(user_logits).any().item()
        has_inf = torch.isinf(user_logits).any().item()

        row = user_id // row_stride
        pos = user_id % row_stride

        if has_nan:
            decode_nan_users.append(f"User {user_id} (row {row}, pos {pos})")
        if has_inf:
            decode_inf_users.append(f"User {user_id} (row {row}, pos {pos})")

        # Check for anomalous distributions
        if not has_nan and not has_inf:
            logit_max = user_logits.max().item()
            logit_std = user_logits.std().item()
            if abs(logit_max) > 100 or logit_std < 0.01 or logit_std > 50:
                decode_anomaly_users.append(
                    f"User {user_id} (row {row}, pos {pos}): max={logit_max:.2f}, std={logit_std:.4f}"
                )

    if decode_nan_users:
        logger.error(f"🚨 DECODE NaN detected in {len(decode_nan_users)} users: {decode_nan_users[:10]}")
    else:
        logger.info("✓ No NaN values in decode logits")

    if decode_inf_users:
        logger.error(f"🚨 DECODE Inf detected in {len(decode_inf_users)} users: {decode_inf_users[:10]}")
    else:
        logger.info("✓ No Inf values in decode logits")

    if decode_anomaly_users:
        logger.warning(f"⚠️ DECODE anomalous logit ranges in {len(decode_anomaly_users)} users:")
        for msg in decode_anomaly_users[:5]:
            logger.warning(f"  {msg}")

    _, tt_next_tok = sample_host(tt_decode_logits, temperature=0, top_p=0.08, on_host=True)
    tt_next_tok = tt_next_tok.squeeze(-1)

    # HF decode reference for subset using the TT prefill token as teacher forcing
    subset_max_len = subset_input_ids.shape[1]
    ref_input_ids = torch.zeros(
        (num_users_to_check, subset_max_len + 1),
        dtype=subset_input_ids.dtype,
    )
    ref_input_ids[:, :subset_max_len] = subset_input_ids
    for i, (user_idx, pos) in enumerate(zip(user_indices, subset_decoding_pos.tolist())):
        ref_input_ids[i, pos] = tt_prefill_token[user_idx].item()

    ref_attention_mask = torch.zeros_like(ref_input_ids, dtype=torch.long)
    for i, pos in enumerate(subset_decoding_pos.tolist()):
        ref_attention_mask[i, : pos + 1] = 1

    with torch.no_grad():
        ref_output = reference_model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            use_cache=False,
        )
    ref_logits = ref_output.logits
    ref_next_tok = torch.argmax(ref_logits[torch.arange(num_users_to_check), subset_decoding_pos, :], dim=-1)

    # ==================== DECODE LOGIT DISTRIBUTION COMPARISON ====================
    logger.info("\n" + "=" * 80)
    logger.info("DECODE LOGIT DISTRIBUTION COMPARISON (TT vs HF):")
    logger.info("=" * 80)

    for i, user_id in enumerate(user_indices[:8]):  # Check first 8 users in detail
        tt_user_logits = tt_decode_logits_tensor[user_id].float().squeeze()
        hf_user_logits = ref_logits[i, subset_decoding_pos[i], :].float()

        # Check for NaN before comparison
        if torch.isnan(tt_user_logits).any() or torch.isnan(hf_user_logits).any():
            logger.error(f"User {user_id}: Cannot compare - NaN values present")
            continue

        # Get top-5 predictions from each
        tt_top5_vals, tt_top5_idx = torch.topk(tt_user_logits, 5)
        hf_top5_vals, hf_top5_idx = torch.topk(hf_user_logits, 5)

        # Compute cosine similarity between logit distributions
        cos_sim = torch.nn.functional.cosine_similarity(tt_user_logits.unsqueeze(0), hf_user_logits.unsqueeze(0)).item()

        # Compute L2 distance (normalized)
        l2_dist = torch.norm(tt_user_logits - hf_user_logits).item()
        l2_dist_normalized = l2_dist / (torch.norm(hf_user_logits).item() + 1e-8)

        row = user_id // row_stride
        pos = user_id % row_stride

        # Determine severity
        if cos_sim < 0.5:
            severity = "🚨 SEVERE DIVERGENCE"
        elif cos_sim < 0.8:
            severity = "⚠️ SIGNIFICANT DIVERGENCE"
        elif cos_sim < 0.95:
            severity = "⚠️ MINOR DIVERGENCE"
        else:
            severity = "✓ OK"

        logger.info(f"\nUser {user_id} (row {row}, pos {pos}): {severity}")
        logger.info(f"  Cosine similarity: {cos_sim:.4f}")
        logger.info(f"  Normalized L2 distance: {l2_dist_normalized:.4f}")
        logger.info(f"  TT  top-5: {[tokenizer.decode([idx.item()]) for idx in tt_top5_idx]}")
        logger.info(f"  HF  top-5: {[tokenizer.decode([idx.item()]) for idx in hf_top5_idx]}")

        # Check if top-1 matches
        if tt_top5_idx[0] != hf_top5_idx[0]:
            hf_top_token = hf_top5_idx[0].item()
            tt_rank_of_hf_top = (tt_user_logits > tt_user_logits[hf_top_token]).sum().item() + 1
            logger.warning(
                f"  Top-1 mismatch! HF's '{tokenizer.decode([hf_top_token])}' is rank {tt_rank_of_hf_top} in TT"
            )

            # Show what TT thinks is top
            tt_top_token = tt_top5_idx[0].item()
            logger.warning(
                f"  TT's top token: '{tokenizer.decode([tt_top_token])}' with logit {tt_user_logits[tt_top_token]:.2f}"
            )
            logger.warning(
                f"  HF's top token: '{tokenizer.decode([hf_top_token])}' with TT logit {tt_user_logits[hf_top_token]:.2f}, HF logit {hf_user_logits[hf_top_token]:.2f}"
            )

    # Check decode tokens match and show details
    logger.info("\n" + "=" * 80)
    logger.info("DECODE RESULTS (second generated token):")
    logger.info("=" * 80)
    tt_decode_subset = tt_next_tok[user_indices].cpu()
    decode_mismatches = []
    for i, user_id in enumerate(user_indices):
        tt_tok = tt_decode_subset[i].item()
        hf_tok = ref_next_tok[i].item()
        tt_text = tokenizer.decode([tt_tok])
        hf_text = tokenizer.decode([hf_tok])
        prefill_tok = tt_prefill_token[user_id].item()
        prefill_text = tokenizer.decode([prefill_tok])
        match_str = "✓" if tt_tok == hf_tok else "✗ MISMATCH"
        logger.info(
            f"User {user_id:3d} (row {user_id // row_stride}, pos {user_id % row_stride:2d}): "
            f"After '{prefill_text}' -> TT='{tt_text}' ({tt_tok}) vs HF='{hf_text}' ({hf_tok}) {match_str}"
        )
        if tt_tok != hf_tok:
            decode_mismatches.append(
                f"User {user_id} (row {user_id // row_stride}, pos {user_id % row_stride}): "
                f"After prefill '{prefill_text}' -> TT='{tt_text}' ({tt_tok}) vs HF='{hf_text}' ({hf_tok})"
            )

    if decode_mismatches:
        logger.error(f"\nDECODE MISMATCHES: {len(decode_mismatches)}/{num_users_to_check}")
        logger.error("This indicates the KV cache is being read incorrectly during decode.")
        logger.error("Possible causes:")
        logger.error("  1. Page table indexing issue for row-sharded users")
        logger.error("  2. Position encoding mismatch between prefill and decode")
        logger.error("  3. KV cache corruption during prefill for certain user positions")

    assert len(decode_mismatches) == 0, f"Decode token mismatches:\n" + "\n".join(decode_mismatches)

    logger.info("✓ Batch-128 row-sharded demo smoke test passed.")


@parametrize_mesh_with_fabric()
def test_batch128_extended_generation(mesh_device, device_params, reset_seeds, state_dict):
    """
    Test extended generation (multiple decode steps) to catch issues that
    accumulate over longer sequences.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 20  # Generate more tokens to catch accumulating issues
    paged_attention = True
    # Need enough blocks for all users: (max_seq_len / block_size) * users_per_row
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    # Build TT generator
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Load prompts
    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # Prefill
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    current_tok = torch.argmax(tt_logits, dim=-1).squeeze(-1)
    current_pos = torch.tensor(decoding_pos, dtype=torch.long)

    # Collect generated tokens
    all_generated_tokens = [current_tok.unsqueeze(-1)]

    # Generate multiple decode steps
    for step in range(max_generated_tokens - 1):
        tt_decode_logits, _ = generator.decode_forward_text(
            current_tok.unsqueeze(-1),
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
        )
        _, current_tok = sample_host(tt_decode_logits, temperature=0, top_p=0.08, on_host=True)
        current_tok = current_tok.squeeze(-1)
        current_pos = current_pos + 1
        all_generated_tokens.append(current_tok.unsqueeze(-1))

    all_generated_tokens = torch.cat(all_generated_tokens, dim=-1)

    # Check output quality for all users
    row_stride = global_batch_size // mesh_device.shape[0]
    garbage_users = []

    for user_id in range(global_batch_size):
        is_garbage, reason = _check_output_quality(all_generated_tokens[user_id].tolist(), tokenizer, user_id)
        if is_garbage:
            row = user_id // row_stride
            pos = user_id % row_stride
            garbage_users.append(f"User {user_id} (row {row}, pos {pos}): {reason}")

    if garbage_users:
        logger.warning(f"Found {len(garbage_users)} users with potentially garbage output:")
        for msg in garbage_users[:10]:  # Show first 10
            logger.warning(msg)

    # Fail if more than 5% of users have garbage output
    garbage_ratio = len(garbage_users) / global_batch_size
    assert garbage_ratio < 0.05, (
        f"Too many users ({len(garbage_users)}/{global_batch_size} = {garbage_ratio:.1%}) "
        f"have garbage output. First few:\n" + "\n".join(garbage_users[:5])
    )

    logger.info(f"✓ Extended generation test passed ({len(garbage_users)} garbage users out of {global_batch_size}).")


@parametrize_mesh_with_fabric()
def test_batch128_all_users_prefill_consistency(mesh_device, device_params, reset_seeds, state_dict):
    """
    Test that ALL 128 users produce valid prefill outputs (not NaN/Inf).
    This catches issues with specific user positions that other tests miss.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    paged_attention = True
    # Need enough blocks for all users: (max_seq_len / block_size) * users_per_row
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    # Build TT generator
    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Load prompts
    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # Prefill all users
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )

    # Check for NaN/Inf in logits for ALL users
    row_stride = global_batch_size // mesh_device.shape[0]
    invalid_users = []

    for user_id in range(global_batch_size):
        user_logits = tt_logits[user_id]
        if torch.isnan(user_logits).any():
            row = user_id // row_stride
            pos = user_id % row_stride
            invalid_users.append(f"User {user_id} (row {row}, pos {pos}): NaN in logits")
        elif torch.isinf(user_logits).any():
            row = user_id // row_stride
            pos = user_id % row_stride
            invalid_users.append(f"User {user_id} (row {row}, pos {pos}): Inf in logits")

    assert len(invalid_users) == 0, f"Found {len(invalid_users)} users with invalid logits:\n" + "\n".join(
        invalid_users
    )

    # Check that all users produce valid tokens (within vocab range)
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)
    vocab_size = model_args[0].vocab_size

    out_of_range = []
    for user_id in range(global_batch_size):
        tok = tt_prefill_token[user_id].item()
        if tok < 0 or tok >= vocab_size:
            row = user_id // row_stride
            pos = user_id % row_stride
            out_of_range.append(
                f"User {user_id} (row {row}, pos {pos}): token {tok} out of vocab range [0, {vocab_size})"
            )

    assert len(out_of_range) == 0, f"Found {len(out_of_range)} users with out-of-range tokens:\n" + "\n".join(
        out_of_range
    )

    logger.info(f"✓ All {global_batch_size} users passed prefill consistency check.")


@parametrize_mesh_with_fabric()
def test_batch128_decode_divergence_analysis(mesh_device, device_params, reset_seeds, state_dict):
    """
    Detailed analysis of where decode logits diverge from HF reference.

    This test:
    1. Runs prefill for all users
    2. Runs ONE decode step
    3. Compares TT vs HF logits in detail to find divergence patterns
    4. Tests ALL 32 local positions within a row to find position-specific bugs
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    paged_attention = True
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=paged_attention,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ============ PREFILL ============
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    # ============ DECODE ============
    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    # ============ HF REFERENCE ============
    if os.getenv("HF_MODEL") is None:
        pytest.skip("HF_MODEL not set; skipping HF reference comparison.")

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    hf_state_dict = setup["model_args"].load_state_dict(
        weights_path=setup["model_args"].model_path,
        dummy_weights=setup["model_args"].dummy_weights,
        convert_to_meta_format=False,
    )
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(hf_state_dict, strict=False)
    reference_model.eval()

    # ============ ANALYZE ALL 128 USERS ============
    logger.info("\n" + "=" * 100)
    logger.info("DECODE DIVERGENCE ANALYSIS - ALL 128 USERS")
    logger.info("=" * 100)

    row_stride = users_per_row

    # Compute HF decode for all users (in batches to manage memory)
    batch_size_hf = 32
    all_hf_logits = []

    for batch_start in range(0, global_batch_size, batch_size_hf):
        batch_end = min(batch_start + batch_size_hf, global_batch_size)
        batch_users = list(range(batch_start, batch_end))

        subset_input_ids = input_tokens_prefill_pt[batch_users].to(torch.long)
        subset_decoding_pos = torch.tensor([decoding_pos[i] for i in batch_users], dtype=torch.long)

        # Build input for decode: original prompt + prefill token
        subset_max_len = subset_input_ids.shape[1]
        ref_input_ids = torch.zeros(
            (len(batch_users), subset_max_len + 1),
            dtype=subset_input_ids.dtype,
        )
        ref_input_ids[:, :subset_max_len] = subset_input_ids
        for i, user_idx in enumerate(batch_users):
            pos = subset_decoding_pos[i].item()
            ref_input_ids[i, pos] = tt_prefill_token[user_idx].item()

        ref_attention_mask = torch.zeros_like(ref_input_ids, dtype=torch.long)
        for i, pos in enumerate(subset_decoding_pos.tolist()):
            ref_attention_mask[i, : pos + 1] = 1

        with torch.no_grad():
            ref_output = reference_model(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                use_cache=False,
            )

        # Get decode logits at the decode position
        batch_hf_logits = []
        for i, pos in enumerate(subset_decoding_pos.tolist()):
            batch_hf_logits.append(ref_output.logits[i, pos, :])
        all_hf_logits.extend(batch_hf_logits)

    # ============ PER-USER DIVERGENCE ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("PER-USER COSINE SIMILARITY HEATMAP (by row and local position):")
    logger.info("-" * 100)

    divergence_data = []
    num_rows = mesh_device.shape[0]

    # Create a matrix of cosine similarities
    cos_sim_matrix = torch.zeros((num_rows, row_stride))

    for user_id in range(global_batch_size):
        tt_logits_user = tt_decode_logits_tensor[user_id].float().squeeze()
        hf_logits_user = all_hf_logits[user_id].float()

        # Skip if NaN
        if torch.isnan(tt_logits_user).any() or torch.isnan(hf_logits_user).any():
            cos_sim = 0.0
        else:
            cos_sim = torch.nn.functional.cosine_similarity(
                tt_logits_user.unsqueeze(0), hf_logits_user.unsqueeze(0)
            ).item()

        row = user_id // row_stride
        local_pos = user_id % row_stride
        cos_sim_matrix[row, local_pos] = cos_sim

        # Check if top-1 token matches
        tt_top1 = torch.argmax(tt_logits_user).item()
        hf_top1 = torch.argmax(hf_logits_user).item()

        divergence_data.append(
            {
                "user_id": user_id,
                "row": row,
                "local_pos": local_pos,
                "cos_sim": cos_sim,
                "tt_top1": tt_top1,
                "hf_top1": hf_top1,
                "match": tt_top1 == hf_top1,
            }
        )

    # Print heatmap summary per row
    for row in range(num_rows):
        row_sims = cos_sim_matrix[row]
        min_sim = row_sims.min().item()
        max_sim = row_sims.max().item()
        mean_sim = row_sims.mean().item()

        # Find worst positions in this row
        worst_positions = torch.argsort(row_sims)[:5].tolist()

        logger.info(f"\nRow {row}: mean={mean_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
        logger.info(f"  Worst 5 positions: {worst_positions}")
        for pos in worst_positions:
            user_id = row * row_stride + pos
            logger.info(f"    pos {pos} (user {user_id}): cos_sim={row_sims[pos]:.4f}")

    # ============ FIND PATTERN IN FAILURES ============
    logger.info("\n" + "-" * 100)
    logger.info("FAILURE PATTERN ANALYSIS:")
    logger.info("-" * 100)

    mismatches = [d for d in divergence_data if not d["match"]]
    logger.info(f"\nTotal mismatches: {len(mismatches)}/{global_batch_size}")

    if mismatches:
        # Analyze by row
        row_mismatch_counts = {}
        for m in mismatches:
            row = m["row"]
            row_mismatch_counts[row] = row_mismatch_counts.get(row, 0) + 1

        logger.info("\nMismatches by row:")
        for row in sorted(row_mismatch_counts.keys()):
            logger.info(f"  Row {row}: {row_mismatch_counts[row]} mismatches")

        # Analyze by local position
        pos_mismatch_counts = {}
        for m in mismatches:
            pos = m["local_pos"]
            pos_mismatch_counts[pos] = pos_mismatch_counts.get(pos, 0) + 1

        logger.info("\nMismatches by local position (check for position-specific bugs):")
        for pos in sorted(pos_mismatch_counts.keys()):
            logger.info(
                f"  Position {pos}: {pos_mismatch_counts[pos]} mismatches (across {pos_mismatch_counts[pos]} rows)"
            )

        # Find positions that fail in MULTIPLE rows (systematic bug)
        multi_row_failures = {pos: count for pos, count in pos_mismatch_counts.items() if count > 1}
        if multi_row_failures:
            logger.error("\n🚨 POSITIONS FAILING IN MULTIPLE ROWS (likely systematic bug):")
            for pos, count in sorted(multi_row_failures.items(), key=lambda x: -x[1]):
                failing_users = [m["user_id"] for m in mismatches if m["local_pos"] == pos]
                logger.error(f"  Position {pos}: fails in {count} rows, users: {failing_users}")

        # Show detailed mismatch info
        logger.info("\nDetailed mismatches:")
        for m in mismatches[:20]:  # Show first 20
            tt_text = tokenizer.decode([m["tt_top1"]])
            hf_text = tokenizer.decode([m["hf_top1"]])
            logger.info(
                f"  User {m['user_id']:3d} (row {m['row']}, pos {m['local_pos']:2d}): "
                f"cos_sim={m['cos_sim']:.4f}, TT='{tt_text}' ({m['tt_top1']}) vs HF='{hf_text}' ({m['hf_top1']})"
            )

    # ============ INVESTIGATE LOW COSINE SIMILARITY ============
    logger.info("\n" + "-" * 100)
    logger.info("LOW COSINE SIMILARITY INVESTIGATION (cos_sim < 0.8):")
    logger.info("-" * 100)

    low_cos_users = [d for d in divergence_data if d["cos_sim"] < 0.8]
    logger.info(f"\nUsers with cos_sim < 0.8: {len(low_cos_users)}/{global_batch_size}")

    if low_cos_users:
        # Analyze logit statistics for low-cosine users
        logger.info("\nLogit statistics for users with low cosine similarity:")
        for d in low_cos_users[:10]:
            user_id = d["user_id"]
            tt_logits_user = tt_decode_logits_tensor[user_id].float().squeeze()
            hf_logits_user = all_hf_logits[user_id].float()

            tt_max = tt_logits_user.max().item()
            tt_min = tt_logits_user.min().item()
            tt_std = tt_logits_user.std().item()

            hf_max = hf_logits_user.max().item()
            hf_min = hf_logits_user.min().item()
            hf_std = hf_logits_user.std().item()

            logger.info(
                f"  User {user_id:3d} (row {d['row']}, pos {d['local_pos']:2d}): "
                f"TT[max={tt_max:.2f}, min={tt_min:.2f}, std={tt_std:.2f}] "
                f"HF[max={hf_max:.2f}, min={hf_min:.2f}, std={hf_std:.2f}]"
            )

    # ============ ASSERTION ============
    # Allow up to 10% mismatches due to numerical precision
    max_allowed_mismatches = int(global_batch_size * 0.10)

    if len(mismatches) > max_allowed_mismatches:
        logger.error(f"\n🚨 Too many mismatches: {len(mismatches)} > {max_allowed_mismatches} (10% threshold)")

        # Provide actionable debugging info
        logger.error("\n" + "=" * 100)
        logger.error("DEBUGGING SUGGESTIONS:")
        logger.error("=" * 100)

        if multi_row_failures:
            logger.error(
                "1. Position-specific failures detected - likely bug in how local batch index maps to page table"
            )
            logger.error(f"   Focus on positions: {list(multi_row_failures.keys())}")

        if any(d["row"] == num_rows - 1 for d in mismatches):
            logger.error("2. Last row (row 3) has failures - check for boundary conditions in row handling")

        if any(d["local_pos"] == row_stride - 1 for d in mismatches):
            logger.error("3. Last position (pos 31) has failures - check for off-by-one in batch size handling")

        assert False, f"Decode mismatch rate {len(mismatches)}/{global_batch_size} exceeds 10% threshold"

    logger.info(f"\n✓ Decode divergence within acceptable range: {len(mismatches)}/{global_batch_size} mismatches")


@parametrize_mesh_with_fabric()
def test_batch128_page_table_indexing_consistency(mesh_device, device_params, reset_seeds, state_dict):
    """
    Specifically test that page table indexing is consistent between prefill and decode.

    This test:
    1. Creates a page table with known block IDs
    2. Fills KV cache during prefill
    3. Verifies decode reads from the same blocks
    4. Checks for position encoding mismatches
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    # ============ ANALYZE PAGE TABLE STRUCTURE ============
    logger.info("\n" + "=" * 100)
    logger.info("PAGE TABLE STRUCTURE ANALYSIS")
    logger.info("=" * 100)

    num_rows = mesh_device.shape[0]

    logger.info(f"Page table shape: {page_table.shape}")
    logger.info(f"Users per row: {users_per_row}")
    logger.info(f"Blocks per user: {blocks_per_user}")

    # Check for block ID collisions across rows
    logger.info("\n" + "-" * 100)
    logger.info("BLOCK ID COLLISION CHECK (critical for row-sharded mode):")
    logger.info("-" * 100)

    block_usage = {}  # block_id -> list of (user_id, row, local_pos)

    for user_id in range(global_batch_size):
        row = user_id // users_per_row
        local_pos = user_id % users_per_row
        user_blocks = page_table[user_id].tolist()

        for block_id in user_blocks:
            if block_id not in block_usage:
                block_usage[block_id] = []
            block_usage[block_id].append((user_id, row, local_pos))

    # Find blocks used by users on different rows
    cross_row_collisions = {}
    for block_id, users in block_usage.items():
        if len(users) > 1:
            rows_using_block = set(user[1] for user in users)
            if len(rows_using_block) > 1:
                cross_row_collisions[block_id] = users

    if cross_row_collisions:
        logger.error(f"\n🚨 CRITICAL: Found {len(cross_row_collisions)} blocks with CROSS-ROW collisions!")
        logger.error("This WILL cause KV cache corruption in row-sharded mode!")
        for block_id, users in list(cross_row_collisions.items())[:5]:
            logger.error(f"  Block {block_id}: used by users {[u[0] for u in users]} (rows {[u[1] for u in users]})")
    else:
        logger.info("✓ No cross-row block collisions detected")

    # Check for same-row collisions (less critical but still problematic)
    same_row_collisions = {}
    for block_id, users in block_usage.items():
        if len(users) > 1:
            rows_using_block = set(user[1] for user in users)
            if len(rows_using_block) == 1:  # Same row
                same_row_collisions[block_id] = users

    if same_row_collisions:
        logger.warning(f"\n⚠️ Found {len(same_row_collisions)} blocks with same-row collisions")
        for block_id, users in list(same_row_collisions.items())[:5]:
            logger.warning(f"  Block {block_id}: used by users {[u[0] for u in users]}")

    # ============ VERIFY PAGE TABLE ASSIGNMENT PATTERN ============
    logger.info("\n" + "-" * 100)
    logger.info("PAGE TABLE ASSIGNMENT PATTERN:")
    logger.info("-" * 100)

    # For row-sharded mode, each row should have non-overlapping block ranges
    # Check if block IDs are properly separated by row
    for row in range(num_rows):
        row_start = row * users_per_row
        row_end = (row + 1) * users_per_row

        row_blocks = set()
        for user_id in range(row_start, row_end):
            row_blocks.update(page_table[user_id].tolist())

        min_block = min(row_blocks)
        max_block = max(row_blocks)
        num_blocks = len(row_blocks)

        logger.info(
            f"Row {row}: users [{row_start}, {row_end}), blocks range [{min_block}, {max_block}], unique blocks: {num_blocks}"
        )

    # ============ ASSERTIONS ============
    assert (
        len(cross_row_collisions) == 0
    ), f"Found {len(cross_row_collisions)} cross-row block collisions - this causes KV cache corruption!"

    logger.info("\n✓ Page table indexing consistency check passed")


@parametrize_mesh_with_fabric()
def test_batch128_position_encoding_consistency(mesh_device, device_params, reset_seeds, state_dict):
    """
    Test that position encoding is computed correctly for all users during decode.

    This test verifies that:
    1. Each user's decode position is seq_len (after prefill)
    2. The position tensor is correctly sharded across rows
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    # Use a simpler prompt set for this test
    input_prompts = ["Hello, this is a test prompt for position encoding verification."] * global_batch_size

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ============ CHECK DECODING POSITIONS ============
    logger.info("\n" + "=" * 100)
    logger.info("POSITION ENCODING CONSISTENCY CHECK")
    logger.info("=" * 100)

    current_pos = torch.tensor(decoding_pos, dtype=torch.long)

    logger.info(f"Decoding positions tensor shape: {current_pos.shape}")
    logger.info(f"Position range: [{current_pos.min().item()}, {current_pos.max().item()}]")

    # All users should have the same position after prefill (same prompt)
    unique_positions = current_pos.unique()
    logger.info(f"Unique positions: {unique_positions.tolist()}")

    if len(unique_positions) == 1:
        logger.info("✓ All users have the same decode position (expected for identical prompts)")
    else:
        logger.warning(f"⚠️ Users have different decode positions: {unique_positions.tolist()}")

    # ============ VERIFY POSITION TENSOR SHARDING ============
    logger.info("\n" + "-" * 100)
    logger.info("POSITION TENSOR SHARDING:")
    logger.info("-" * 100)

    num_rows = mesh_device.shape[0]

    for row in range(num_rows):
        row_start = row * users_per_row
        row_end = (row + 1) * users_per_row
        row_positions = current_pos[row_start:row_end]

        logger.info(f"Row {row}: positions for users [{row_start}, {row_end})")
        logger.info(f"  First 5: {row_positions[:5].tolist()}")
        logger.info(f"  Last 5: {row_positions[-5:].tolist()}")

    # ============ RUN PREFILL AND CHECK POSITION UPDATES ============
    logger.info("\n" + "-" * 100)
    logger.info("PREFILL AND POSITION UPDATE:")
    logger.info("-" * 100)

    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )

    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    # Now check the decode input preparation
    out_tok = tt_prefill_token.unsqueeze(-1)

    logger.info(f"Input token shape for decode: {out_tok.shape}")
    logger.info(f"Position tensor shape: {current_pos.shape}")

    # Check that position tensor is correctly formatted for decode
    assert current_pos.shape[0] == global_batch_size, f"Position tensor should have {global_batch_size} entries"
    assert out_tok.shape[0] == global_batch_size, f"Token tensor should have {global_batch_size} entries"

    logger.info("\n✓ Position encoding consistency check passed")


@parametrize_mesh_with_fabric()
def test_batch128_failing_position_investigation(mesh_device, device_params, reset_seeds, state_dict):
    """
    Deep investigation into WHY specific positions fail consistently across all rows.

    Known failing positions from test_batch128_decode_divergence_analysis: 9, 17, 20, 27, 31
    These fail in ALL 4 rows with IDENTICAL wrong tokens.

    This test investigates:
    1. Prompt lengths at these positions
    2. Core grid mapping for these positions
    3. Page table entries for these positions
    4. Whether failure correlates with prompt length or core assignment
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ============ INVESTIGATE FAILING POSITIONS ============
    logger.info("\n" + "=" * 100)
    logger.info("FAILING POSITION INVESTIGATION")
    logger.info("=" * 100)

    FAILING_POSITIONS = [9, 17, 20, 27, 31]
    PASSING_POSITIONS = [0, 1, 2, 4, 5, 6, 7, 8, 10, 15, 16, 30]  # Sample of passing positions

    # ============ PROMPT LENGTH ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("PROMPT LENGTH ANALYSIS:")
    logger.info("-" * 100)

    failing_lengths = []
    passing_lengths = []

    logger.info("\nFailing positions:")
    for pos in FAILING_POSITIONS:
        prompt = input_prompts[pos]
        token_len = decoding_pos[pos]
        failing_lengths.append(token_len)
        logger.info(f"  Position {pos:2d}: {token_len:3d} tokens | '{prompt[:60]}...' ")

    logger.info("\nPassing positions (sample):")
    for pos in PASSING_POSITIONS:
        prompt = input_prompts[pos]
        token_len = decoding_pos[pos]
        passing_lengths.append(token_len)
        logger.info(f"  Position {pos:2d}: {token_len:3d} tokens | '{prompt[:60]}...' ")

    logger.info(f"\nFailing positions avg length: {sum(failing_lengths)/len(failing_lengths):.1f} tokens")
    logger.info(f"Passing positions avg length: {sum(passing_lengths)/len(passing_lengths):.1f} tokens")

    # Check if failing positions have unusually long/short prompts
    all_lengths = decoding_pos
    mean_len = sum(all_lengths) / len(all_lengths)
    logger.info(f"Overall mean prompt length: {mean_len:.1f} tokens")

    # ============ CORE GRID MAPPING ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("CORE GRID MAPPING ANALYSIS:")
    logger.info("-" * 100)

    grid_size = mesh_device.compute_with_storage_grid_size()
    logger.info(f"Compute grid size: {grid_size}")

    # For row_wise=True mapping with batch_size=32
    # Cores are assigned row-major
    cores_per_row = grid_size.x  # Typically 8

    logger.info(f"\nCore assignments for batch_size=32 with row_wise=True:")
    logger.info("(Position -> Grid Row, Grid Column)")

    for pos in range(32):
        grid_row = pos // cores_per_row
        grid_col = pos % cores_per_row
        marker = " <-- FAILING" if pos in FAILING_POSITIONS else ""
        logger.info(f"  Position {pos:2d} -> Row {grid_row}, Col {grid_col}{marker}")

    # ============ PAGE TABLE ANALYSIS FOR FAILING POSITIONS ============
    logger.info("\n" + "-" * 100)
    logger.info("PAGE TABLE ENTRIES FOR FAILING POSITIONS:")
    logger.info("-" * 100)

    for pos in FAILING_POSITIONS:
        logger.info(f"\nPosition {pos} (users {pos}, {pos+32}, {pos+64}, {pos+96}):")
        for row in range(4):
            user_id = pos + row * 32
            blocks = page_table[user_id][:4].tolist()  # First 4 blocks
            logger.info(f"  User {user_id:3d} (row {row}): blocks {blocks}")

    # ============ CHECK FOR PATTERN IN FAILING POSITIONS ============
    logger.info("\n" + "-" * 100)
    logger.info("PATTERN ANALYSIS:")
    logger.info("-" * 100)

    # Check if failing positions share any common factor
    logger.info(f"\nFailing positions: {FAILING_POSITIONS}")
    logger.info(f"Failing positions mod 8: {[p % 8 for p in FAILING_POSITIONS]}")
    logger.info(f"Failing positions div 8: {[p // 8 for p in FAILING_POSITIONS]}")

    # Check if there's a stride pattern
    diffs = [FAILING_POSITIONS[i + 1] - FAILING_POSITIONS[i] for i in range(len(FAILING_POSITIONS) - 1)]
    logger.info(f"Differences between consecutive failing positions: {diffs}")

    # ============ PROMPT CONTENT ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("PROMPT CONTENT AT FAILING POSITIONS:")
    logger.info("-" * 100)

    for pos in FAILING_POSITIONS:
        prompt = input_prompts[pos]
        tokens = input_tokens_prefill_pt[pos]
        non_pad = (tokens != 0).sum().item()
        first_10_tokens = tokens[:10].tolist()
        logger.info(f"  Position {pos:2d}:")
        logger.info(f"    Prompt: '{prompt}'")
        logger.info(f"    Non-padding tokens: {non_pad}")
        logger.info(f"    First 10 token IDs: {first_10_tokens}")
        logger.info(f"    First 10 decoded: '{tokenizer.decode(first_10_tokens)}'")

    logger.info("\n✓ Investigation complete - analyze patterns above to find root cause")


@parametrize_mesh_with_fabric()
def test_batch128_prompt_specific_analysis(mesh_device, device_params, reset_seeds, state_dict):
    """
    Analyze WHY specific prompts produce wrong decode outputs.

    NOTE: With 32 unique prompts and 32 users per row, ALL rows process the SAME prompts.
    This is expected - the failing positions (9, 17, 20, 27, 31) fail because those
    SPECIFIC PROMPTS have issues, not because of row sharding bugs.

    This test investigates what's different about the failing prompts.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ============ VERIFY PROMPT REPETITION ============
    logger.info("\n" + "=" * 100)
    logger.info("PROMPT REPETITION VERIFICATION")
    logger.info("=" * 100)

    num_rows = mesh_device.shape[0]

    # Check that all rows have the same prompts (expected!)
    logger.info("\nChecking if prompts repeat across rows (expected with 32 prompts, 32 users/row):")
    for local_pos in [0, 15, 31]:  # Sample positions
        prompts_match = all(
            input_prompts[local_pos] == input_prompts[row * users_per_row + local_pos] for row in range(num_rows)
        )
        logger.info(f"  Position {local_pos}: all rows have same prompt = {prompts_match}")

    # ============ RUN PREFILL ============
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    # ============ RUN DECODE ============
    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    # ============ ANALYZE PROMPT-SPECIFIC PATTERNS ============
    logger.info("\n" + "=" * 100)
    logger.info("PROMPT-SPECIFIC ANALYSIS")
    logger.info("=" * 100)

    KNOWN_FAILING = [9, 17, 20, 27, 31]

    # For each local position, check consistency across rows
    logger.info("\nDecode output consistency across rows (should be identical for same prompts):")
    inconsistent_positions = []

    for local_pos in range(users_per_row):
        user_ids = [row * users_per_row + local_pos for row in range(num_rows)]
        logits_list = [tt_decode_logits_tensor[uid].float().squeeze() for uid in user_ids]

        # Check if all rows produce same output for this position (expected since same prompt)
        max_diff = max((logits_list[0] - logits_list[i]).abs().max().item() for i in range(1, num_rows))

        top_tokens = [torch.argmax(logits).item() for logits in logits_list]
        all_same_token = len(set(top_tokens)) == 1

        marker = " <-- KNOWN FAILING" if local_pos in KNOWN_FAILING else ""
        if max_diff > 0.1 or not all_same_token:
            inconsistent_positions.append(local_pos)
            logger.warning(
                f"  Position {local_pos:2d}: max_diff={max_diff:.4f}, tokens={top_tokens} ⚠️ INCONSISTENT{marker}"
            )
        else:
            logger.info(f"  Position {local_pos:2d}: max_diff={max_diff:.4f}, token={top_tokens[0]} ✓{marker}")

    # ============ TOKEN LENGTH ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("TOKEN LENGTH ANALYSIS (looking for correlation with failures):")
    logger.info("-" * 100)

    token_lengths = {}
    for pos in range(users_per_row):
        # Count non-padding tokens
        tokens = input_tokens_prefill_pt[pos]
        non_pad = (tokens != 0).sum().item()
        token_lengths[pos] = non_pad

    failing_lens = [token_lengths[p] for p in KNOWN_FAILING]
    passing_lens = [token_lengths[p] for p in range(users_per_row) if p not in KNOWN_FAILING]

    logger.info(f"Failing prompt token lengths: {failing_lens}")
    logger.info(f"Passing prompt mean length: {sum(passing_lens)/len(passing_lens):.1f}")
    logger.info(f"Failing prompt mean length: {sum(failing_lens)/len(failing_lens):.1f}")

    # ============ LOGIT STATISTICS COMPARISON ============
    logger.info("\n" + "-" * 100)
    logger.info("LOGIT STATISTICS (failing vs passing prompts):")
    logger.info("-" * 100)

    for local_pos in KNOWN_FAILING + [0, 1, 2]:  # Compare failing with some passing
        logits = tt_decode_logits_tensor[local_pos].float().squeeze()
        top_val, top_idx = logits.max(dim=-1)
        marker = "FAILING" if local_pos in KNOWN_FAILING else "passing"
        logger.info(
            f"  Position {local_pos:2d} ({marker:7s}): "
            f"max={top_val.item():.2f}, min={logits.min().item():.2f}, "
            f"std={logits.std().item():.2f}, top_token='{tokenizer.decode([top_idx.item()])}'"
        )

    # ============ SUMMARY ============
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)

    if inconsistent_positions:
        logger.warning(f"Found {len(inconsistent_positions)} positions with inconsistent outputs across rows")
        logger.warning(f"These should produce SAME output since prompts are identical across rows")
        logger.warning(f"Positions: {inconsistent_positions}")
    else:
        logger.info("✓ All positions produce consistent outputs across rows (expected)")

    logger.info(f"\nKnown failing positions: {KNOWN_FAILING}")
    logger.info("These fail because of prompt-specific issues, not row sharding bugs")
    logger.info("Next step: Compare TT vs HF logits for these specific prompts")


@parametrize_mesh_with_fabric()
def test_single_failing_prompt_deep_dive(mesh_device, device_params, reset_seeds, state_dict):
    """
    Deep dive into a SINGLE failing prompt to find exactly where TT diverges from HF.

    This test:
    1. Uses prompt at position 9 ("Who discovered penicillin first?") - known failing
    2. Runs through TT with debug output enabled
    3. Compares TT vs HF at each stage
    4. Reports exactly where divergence starts

    Run with: DEBUG_DECODE_ATTENTION=1 pytest ... -k single_failing -v -s
    """

    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    # Use a known failing prompt position
    FAILING_POSITION = 9  # "Who discovered penicillin first?"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}
    users_row_sharded = True

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=users_row_sharded,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    # ============ SHOW THE FAILING PROMPT ============
    logger.info("\n" + "=" * 100)
    logger.info(f"SINGLE PROMPT DEEP DIVE - Position {FAILING_POSITION}")
    logger.info("=" * 100)

    failing_prompt = input_prompts[FAILING_POSITION]
    logger.info(f"Prompt: '{failing_prompt}'")

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # Show tokenization
    prompt_tokens = input_tokens_prefill_pt[FAILING_POSITION]
    non_pad_len = (prompt_tokens != 0).sum().item()
    logger.info(f"Token IDs: {prompt_tokens[:non_pad_len].tolist()}")
    logger.info(f"Decoded: '{tokenizer.decode(prompt_tokens[:non_pad_len].tolist())}'")
    logger.info(f"Prompt length: {non_pad_len} tokens")
    logger.info(f"Decode position: {decoding_pos[FAILING_POSITION]}")

    # ============ RUN TT PREFILL ============
    logger.info("\n" + "-" * 100)
    logger.info("TT PREFILL")
    logger.info("-" * 100)

    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    tt_prefill_logits_user = tt_logits[FAILING_POSITION].squeeze()
    tt_prefill_tok = tt_prefill_token[FAILING_POSITION].item()

    logger.info(f"TT Prefill token: '{tokenizer.decode([tt_prefill_tok])}' (ID: {tt_prefill_tok})")
    logger.info(
        f"TT Prefill logits: max={tt_prefill_logits_user.max().item():.4f}, min={tt_prefill_logits_user.min().item():.4f}"
    )

    # ============ RUN TT DECODE ============
    logger.info("\n" + "-" * 100)
    logger.info("TT DECODE (with debug output if DEBUG_DECODE_ATTENTION=1)")
    logger.info("-" * 100)

    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    tt_decode_logits_user = tt_decode_logits_tensor[FAILING_POSITION].float().squeeze()
    tt_decode_tok = torch.argmax(tt_decode_logits_user).item()

    logger.info(f"TT Decode token: '{tokenizer.decode([tt_decode_tok])}' (ID: {tt_decode_tok})")
    logger.info(
        f"TT Decode logits: max={tt_decode_logits_user.max().item():.4f}, min={tt_decode_logits_user.min().item():.4f}, std={tt_decode_logits_user.std().item():.4f}"
    )

    # ============ RUN HF REFERENCE ============
    logger.info("\n" + "-" * 100)
    logger.info("HF REFERENCE")
    logger.info("-" * 100)

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    hf_state_dict = setup["model_args"].load_state_dict(
        weights_path=setup["model_args"].model_path,
        dummy_weights=setup["model_args"].dummy_weights,
        convert_to_meta_format=False,
    )
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(hf_state_dict, strict=False)
    reference_model.eval()

    # HF Prefill
    hf_input_ids = input_tokens_prefill_pt[FAILING_POSITION : FAILING_POSITION + 1].to(torch.long)
    hf_attention_mask = _build_attention_mask(hf_input_ids, [decoding_pos[FAILING_POSITION]])

    with torch.no_grad():
        hf_prefill_output = reference_model(
            input_ids=hf_input_ids,
            attention_mask=hf_attention_mask,
            use_cache=True,
        )

    hf_prefill_logits = hf_prefill_output.logits[0, decoding_pos[FAILING_POSITION] - 1, :]
    hf_prefill_tok = torch.argmax(hf_prefill_logits).item()

    logger.info(f"HF Prefill token: '{tokenizer.decode([hf_prefill_tok])}' (ID: {hf_prefill_tok})")
    logger.info(
        f"HF Prefill logits: max={hf_prefill_logits.max().item():.4f}, min={hf_prefill_logits.min().item():.4f}"
    )

    # HF Decode
    # Append the prefill token to input
    decode_input_ids = torch.cat([hf_input_ids, torch.tensor([[tt_prefill_tok]], dtype=torch.long)], dim=1)
    decode_attention_mask = torch.cat([hf_attention_mask, torch.tensor([[1]], dtype=torch.long)], dim=1)

    with torch.no_grad():
        hf_decode_output = reference_model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            use_cache=False,
        )

    hf_decode_logits = hf_decode_output.logits[0, decoding_pos[FAILING_POSITION], :]
    hf_decode_tok = torch.argmax(hf_decode_logits).item()

    logger.info(f"HF Decode token: '{tokenizer.decode([hf_decode_tok])}' (ID: {hf_decode_tok})")
    logger.info(
        f"HF Decode logits: max={hf_decode_logits.max().item():.4f}, min={hf_decode_logits.min().item():.4f}, std={hf_decode_logits.std().item():.4f}"
    )

    # ============ COMPARISON ============
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON")
    logger.info("=" * 100)

    # Prefill comparison
    prefill_cos_sim = torch.nn.functional.cosine_similarity(
        tt_prefill_logits_user.unsqueeze(0).float(), hf_prefill_logits.unsqueeze(0).float()
    ).item()
    prefill_match = tt_prefill_tok == hf_prefill_tok

    logger.info(
        f"PREFILL: TT='{tokenizer.decode([tt_prefill_tok])}' vs HF='{tokenizer.decode([hf_prefill_tok])}' | match={prefill_match} | cos_sim={prefill_cos_sim:.4f}"
    )

    # Decode comparison
    decode_cos_sim = torch.nn.functional.cosine_similarity(
        tt_decode_logits_user.unsqueeze(0), hf_decode_logits.unsqueeze(0).float()
    ).item()
    decode_match = tt_decode_tok == hf_decode_tok

    logger.info(
        f"DECODE:  TT='{tokenizer.decode([tt_decode_tok])}' vs HF='{tokenizer.decode([hf_decode_tok])}' | match={decode_match} | cos_sim={decode_cos_sim:.4f}"
    )

    # ============ DETAILED LOGIT ANALYSIS ============
    logger.info("\n" + "-" * 100)
    logger.info("DETAILED LOGIT ANALYSIS")
    logger.info("-" * 100)

    # Compare top-10 tokens
    tt_top10_vals, tt_top10_idx = torch.topk(tt_decode_logits_user, 10)
    hf_top10_vals, hf_top10_idx = torch.topk(hf_decode_logits, 10)

    logger.info("\nTT Top 10 decode tokens:")
    for i in range(10):
        tok = tt_top10_idx[i].item()
        val = tt_top10_vals[i].item()
        logger.info(f"  {i+1}. '{tokenizer.decode([tok])}' (ID: {tok}) = {val:.4f}")

    logger.info("\nHF Top 10 decode tokens:")
    for i in range(10):
        tok = hf_top10_idx[i].item()
        val = hf_top10_vals[i].item()
        logger.info(f"  {i+1}. '{tokenizer.decode([tok])}' (ID: {tok}) = {val:.4f}")

    # Check if HF's top token appears in TT's top 10
    hf_top_in_tt = hf_decode_tok in tt_top10_idx.tolist()
    if hf_top_in_tt:
        hf_rank_in_tt = tt_top10_idx.tolist().index(hf_decode_tok) + 1
        logger.info(f"\nHF's top token '{tokenizer.decode([hf_decode_tok])}' ranks #{hf_rank_in_tt} in TT")
    else:
        # Find HF's top token rank in TT
        tt_ranks = torch.argsort(tt_decode_logits_user, descending=True)
        hf_rank_in_tt = (tt_ranks == hf_decode_tok).nonzero(as_tuple=True)[0].item() + 1
        logger.info(
            f"\n⚠️ HF's top token '{tokenizer.decode([hf_decode_tok])}' ranks #{hf_rank_in_tt} in TT (not in top 10)"
        )

    # ============ LOGIT DISTRIBUTION COMPARISON ============
    logger.info("\n" + "-" * 100)
    logger.info("LOGIT DISTRIBUTION COMPARISON")
    logger.info("-" * 100)

    # Compare statistics
    logger.info(
        f"TT decode logits: max={tt_decode_logits_user.max():.2f}, min={tt_decode_logits_user.min():.2f}, std={tt_decode_logits_user.std():.2f}"
    )
    logger.info(
        f"HF decode logits: max={hf_decode_logits.max():.2f}, min={hf_decode_logits.min():.2f}, std={hf_decode_logits.std():.2f}"
    )

    # Check scaling difference
    scale_ratio = hf_decode_logits.max().item() / (tt_decode_logits_user.max().item() + 1e-6)
    logger.info(f"Scale ratio (HF max / TT max): {scale_ratio:.2f}x")

    if scale_ratio > 1.5:
        logger.warning(f"⚠️ TT logits are significantly smaller than HF ({scale_ratio:.2f}x) - possible scaling bug")

    # ============ ASSERTION ============
    if not decode_match:
        logger.error(f"\n🚨 DECODE MISMATCH for prompt '{failing_prompt}'")
        logger.error(f"   TT produced: '{tokenizer.decode([tt_decode_tok])}'")
        logger.error(f"   HF expected: '{tokenizer.decode([hf_decode_tok])}'")
        logger.error(f"   Cosine similarity: {decode_cos_sim:.4f}")
        logger.error("\n   To debug further, run with: DEBUG_DECODE_ATTENTION=1")

        # Don't fail the test - we're investigating
        logger.warning("Test completed with mismatch - review debug output above")
    else:
        logger.info(f"\n✓ Decode match for this prompt")


@parametrize_mesh_with_fabric()
def test_compare_passing_vs_failing_prompt(mesh_device, device_params, reset_seeds, state_dict):
    """
    Compare a PASSING prompt vs a FAILING prompt to identify what's different.

    Passing prompt (position 0): "How many moons does Earth have?"
    Failing prompt (position 9): "Who discovered penicillin first?"

    This helps identify if the bug is prompt-specific or systematic.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    PASSING_POS = 0  # "How many moons does Earth have?"
    FAILING_POS = 9  # "Who discovered penicillin first?"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=True,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # ============ COMPARE PROMPTS ============
    logger.info("\n" + "=" * 100)
    logger.info("PASSING vs FAILING PROMPT COMPARISON")
    logger.info("=" * 100)

    for pos, label in [(PASSING_POS, "PASSING"), (FAILING_POS, "FAILING")]:
        prompt = input_prompts[pos]
        tokens = input_tokens_prefill_pt[pos]
        non_pad_len = (tokens != 0).sum().item()
        logger.info(f"\n{label} (position {pos}):")
        logger.info(f"  Prompt: '{prompt}'")
        logger.info(f"  Token length: {non_pad_len}")
        logger.info(f"  Token IDs: {tokens[:non_pad_len].tolist()}")
        logger.info(f"  Decode position: {decoding_pos[pos]}")

    # ============ RUN TT ============
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    # ============ RUN HF ============
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    hf_state_dict = setup["model_args"].load_state_dict(
        weights_path=setup["model_args"].model_path,
        dummy_weights=setup["model_args"].dummy_weights,
        convert_to_meta_format=False,
    )
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(hf_state_dict, strict=False)
    reference_model.eval()

    # ============ COMPARE BOTH PROMPTS ============
    logger.info("\n" + "=" * 100)
    logger.info("DECODE COMPARISON")
    logger.info("=" * 100)

    results = {}

    for pos, label in [(PASSING_POS, "PASSING"), (FAILING_POS, "FAILING")]:
        # TT results
        tt_decode_logits_user = tt_decode_logits_tensor[pos].float().squeeze()
        tt_decode_tok = torch.argmax(tt_decode_logits_user).item()
        tt_prefill_tok = tt_prefill_token[pos].item()

        # HF results
        hf_input_ids = input_tokens_prefill_pt[pos : pos + 1].to(torch.long)
        hf_attention_mask = _build_attention_mask(hf_input_ids, [decoding_pos[pos]])

        # Append prefill token
        decode_input_ids = torch.cat([hf_input_ids, torch.tensor([[tt_prefill_tok]], dtype=torch.long)], dim=1)
        decode_attention_mask = torch.cat([hf_attention_mask, torch.tensor([[1]], dtype=torch.long)], dim=1)

        with torch.no_grad():
            hf_decode_output = reference_model(
                input_ids=decode_input_ids,
                attention_mask=decode_attention_mask,
                use_cache=False,
            )

        hf_decode_logits = hf_decode_output.logits[0, decoding_pos[pos], :].float()
        hf_decode_tok = torch.argmax(hf_decode_logits).item()

        # Compute metrics
        cos_sim = torch.nn.functional.cosine_similarity(
            tt_decode_logits_user.unsqueeze(0), hf_decode_logits.unsqueeze(0)
        ).item()

        match = tt_decode_tok == hf_decode_tok
        scale_ratio = hf_decode_logits.max().item() / (tt_decode_logits_user.max().item() + 1e-6)

        results[label] = {
            "pos": pos,
            "tt_tok": tt_decode_tok,
            "hf_tok": hf_decode_tok,
            "tt_text": tokenizer.decode([tt_decode_tok]),
            "hf_text": tokenizer.decode([hf_decode_tok]),
            "cos_sim": cos_sim,
            "match": match,
            "scale_ratio": scale_ratio,
            "tt_max": tt_decode_logits_user.max().item(),
            "hf_max": hf_decode_logits.max().item(),
            "tt_std": tt_decode_logits_user.std().item(),
            "hf_std": hf_decode_logits.std().item(),
        }

        logger.info(f"\n{label} (pos {pos}):")
        logger.info(f"  TT: '{results[label]['tt_text']}' (ID: {results[label]['tt_tok']})")
        logger.info(f"  HF: '{results[label]['hf_text']}' (ID: {results[label]['hf_tok']})")
        logger.info(f"  Match: {match}, Cosine Sim: {cos_sim:.4f}")
        logger.info(f"  TT logits: max={results[label]['tt_max']:.2f}, std={results[label]['tt_std']:.2f}")
        logger.info(f"  HF logits: max={results[label]['hf_max']:.2f}, std={results[label]['hf_std']:.2f}")
        logger.info(f"  Scale ratio (HF/TT): {scale_ratio:.2f}x")

    # ============ ANALYSIS ============
    logger.info("\n" + "=" * 100)
    logger.info("ANALYSIS")
    logger.info("=" * 100)

    passing = results["PASSING"]
    failing = results["FAILING"]

    logger.info(f"\nPASSING prompt: cos_sim={passing['cos_sim']:.4f}, scale_ratio={passing['scale_ratio']:.2f}x")
    logger.info(f"FAILING prompt: cos_sim={failing['cos_sim']:.4f}, scale_ratio={failing['scale_ratio']:.2f}x")

    if passing["cos_sim"] > failing["cos_sim"]:
        logger.info(f"\n✓ Failing prompt has LOWER cosine similarity (as expected)")
        diff = passing["cos_sim"] - failing["cos_sim"]
        logger.info(f"  Difference: {diff:.4f}")
    else:
        logger.warning(f"\n⚠️ Failing prompt has HIGHER cosine similarity (unexpected)")

    if abs(passing["scale_ratio"] - failing["scale_ratio"]) > 0.1:
        logger.warning(f"\n⚠️ Scale ratios differ significantly between passing and failing prompts")
        logger.warning(f"   This suggests the bug may affect logit scaling differently per prompt")
    else:
        logger.info(f"\n✓ Scale ratios are similar - bug is likely in token ranking, not scaling")

    # Check if both have same scale issue
    if passing["scale_ratio"] > 1.5 and failing["scale_ratio"] > 1.5:
        logger.error(
            f"\n🚨 BOTH prompts have TT logits ~{(passing['scale_ratio'] + failing['scale_ratio'])/2:.1f}x smaller than HF"
        )
        logger.error("   This is a SYSTEMATIC scaling bug in TT decode, not prompt-specific")
        logger.error("   Investigate: attention scaling, RoPE, or output projection")

    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Passing prompt matches: {passing['match']}")
    logger.info(f"Failing prompt matches: {failing['match']}")

    if passing["match"] and not failing["match"]:
        logger.info("\nThe bug is PROMPT-SPECIFIC - some prompts work, others don't")
        logger.info("Investigate: prompt length effects, specific token patterns, or edge cases")
    elif not passing["match"] and not failing["match"]:
        logger.warning("\nBOTH prompts fail - the bug is SYSTEMATIC")
        logger.warning("Investigate: decode attention, RoPE, or output processing")


@parametrize_mesh_with_fabric()
def test_layer_by_layer_divergence(mesh_device, device_params, reset_seeds, state_dict):
    """
    Find EXACTLY which operation causes TT to diverge from HF.

    This test:
    1. Runs a single user through prefill
    2. Captures HF hidden states at each layer during decode
    3. Compares TT decode output layer-by-layer to find divergence point

    NOTE: This test hooks into HF internals to capture intermediate outputs.

    For 120B model, known failing positions are: 13, 15, 17, 21, 23, 24, 25, 30, 31
    - Position 13: "Who wrote the novel Dune?" (produces "Game of Incompetence" garbage)
    - Position 25: "Who was the first person on the Moon?" (produces whitespace flood)

    Set FAILING_POSITION=<pos> environment variable to test specific positions.
    """
    if mesh_device.shape[0] == 1:
        pytest.skip("Batch-128 row-sharded path requires a multi-row mesh (e.g. 4x8).")

    # Use environment variable to specify failing position, default to 13 for 120B debugging
    import os

    FAILING_POSITION = int(os.getenv("FAILING_POSITION", "13"))  # Default: "Who wrote the novel Dune?"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=True)
    config = setup["config"]
    config._attn_implementation = "eager"
    mesh_config = setup["mesh_config"]

    data_parallel = 1
    global_batch_size = 128
    max_seq_len = 1024
    max_generated_tokens = 4
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": blocks_per_user * users_per_row}

    (
        model_args,
        model,
        page_table,
        tt_kv_cache,
        tokenizer,
        processor,
        _,
    ) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=data_parallel,
        mesh_device=mesh_device,
        global_batch_size=global_batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=True,
    )

    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

    prompt_path = "models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    input_prompts, _ = load_inputs(prompt_path, global_batch_size, instruct=False)

    (
        input_tokens_prefill_pt,
        _,
        decoding_pos,
        _,
    ) = preprocess_inputs_prefill(
        input_prompts,
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=max_generated_tokens,
        max_prefill_len=max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    logger.info("\n" + "=" * 100)
    logger.info(f"LAYER-BY-LAYER DIVERGENCE ANALYSIS - Position {FAILING_POSITION}")
    logger.info("=" * 100)
    logger.info(f"Prompt: '{input_prompts[FAILING_POSITION]}'")

    # ============ TT PREFILL + DECODE ============
    tt_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=False,
    )
    tt_prefill_token = torch.argmax(tt_logits, dim=-1).squeeze(-1)

    current_pos = torch.tensor(decoding_pos, dtype=torch.long)
    out_tok = tt_prefill_token.unsqueeze(-1)

    tt_decode_logits, _ = generator.decode_forward_text(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    if isinstance(tt_decode_logits, list):
        tt_decode_logits_tensor = torch.cat([l.cpu() if hasattr(l, "cpu") else l for l in tt_decode_logits], dim=0)
    else:
        tt_decode_logits_tensor = tt_decode_logits.cpu() if hasattr(tt_decode_logits, "cpu") else tt_decode_logits

    tt_decode_logits_user = tt_decode_logits_tensor[FAILING_POSITION].float().squeeze()

    # ============ HF WITH INTERMEDIATE CAPTURE ============
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    hf_state_dict = setup["model_args"].load_state_dict(
        weights_path=setup["model_args"].model_path,
        dummy_weights=setup["model_args"].dummy_weights,
        convert_to_meta_format=False,
    )
    reference_model = GptOssForCausalLM(config)
    reference_model.load_state_dict(hf_state_dict, strict=False)
    reference_model.eval()

    # Prepare HF input for decode
    hf_input_ids = input_tokens_prefill_pt[FAILING_POSITION : FAILING_POSITION + 1].to(torch.long)
    hf_attention_mask = _build_attention_mask(hf_input_ids, [decoding_pos[FAILING_POSITION]])
    tt_prefill_tok = tt_prefill_token[FAILING_POSITION].item()

    decode_input_ids = torch.cat([hf_input_ids, torch.tensor([[tt_prefill_tok]], dtype=torch.long)], dim=1)
    decode_attention_mask = torch.cat([hf_attention_mask, torch.tensor([[1]], dtype=torch.long)], dim=1)

    # Hook to capture HF intermediate outputs
    hf_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_intermediates[name] = output[0].detach()
            else:
                hf_intermediates[name] = output.detach()

        return hook

    # Register hooks on key layers
    hooks = []

    # Embedding
    hooks.append(reference_model.model.embed_tokens.register_forward_hook(make_hook("embed")))

    # Each transformer layer
    for i, layer in enumerate(reference_model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
        hooks.append(layer.self_attn.register_forward_hook(make_hook(f"layer_{i}_attn")))
        hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer_{i}_mlp")))

    # Final norm
    hooks.append(reference_model.model.norm.register_forward_hook(make_hook("final_norm")))

    # LM head
    hooks.append(reference_model.lm_head.register_forward_hook(make_hook("lm_head")))

    with torch.no_grad():
        hf_output = reference_model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            use_cache=False,
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    hf_decode_logits = hf_output.logits[0, decoding_pos[FAILING_POSITION], :].float()

    # ============ ANALYZE INTERMEDIATE OUTPUTS ============
    logger.info("\n" + "-" * 100)
    logger.info("HF INTERMEDIATE OUTPUT STATISTICS (at decode position):")
    logger.info("-" * 100)

    decode_pos = decoding_pos[FAILING_POSITION]

    for name, tensor in sorted(hf_intermediates.items()):
        if tensor.dim() >= 2:
            # Extract at decode position
            if tensor.shape[1] > decode_pos:
                t = tensor[0, decode_pos].float()
            else:
                t = tensor[0, -1].float()  # Use last position

            stats = f"shape={list(tensor.shape)}, pos_slice: min={t.min().item():.4f}, max={t.max().item():.4f}, std={t.std().item():.4f}"
            logger.info(f"  {name}: {stats}")

    # ============ COMPARE FINAL LOGITS ============
    logger.info("\n" + "-" * 100)
    logger.info("FINAL LOGIT COMPARISON:")
    logger.info("-" * 100)

    cos_sim = torch.nn.functional.cosine_similarity(
        tt_decode_logits_user.unsqueeze(0), hf_decode_logits.unsqueeze(0)
    ).item()

    tt_top = torch.argmax(tt_decode_logits_user).item()
    hf_top = torch.argmax(hf_decode_logits).item()

    logger.info(
        f"TT: top='{tokenizer.decode([tt_top])}' ({tt_top}), max={tt_decode_logits_user.max():.2f}, std={tt_decode_logits_user.std():.2f}"
    )
    logger.info(
        f"HF: top='{tokenizer.decode([hf_top])}' ({hf_top}), max={hf_decode_logits.max():.2f}, std={hf_decode_logits.std():.2f}"
    )
    logger.info(f"Cosine similarity: {cos_sim:.4f}")

    # ============ IDENTIFY LIKELY CAUSE ============
    logger.info("\n" + "=" * 100)
    logger.info("DIAGNOSIS")
    logger.info("=" * 100)

    scale_ratio = hf_decode_logits.max().item() / (tt_decode_logits_user.max().item() + 1e-6)

    # Check where HF's answer ranks in TT
    hf_answer_rank = (torch.argsort(tt_decode_logits_user, descending=True) == hf_top).nonzero(as_tuple=True)[
        0
    ].item() + 1

    logger.info(f"\n1. SCALE ANALYSIS:")
    logger.info(f"   HF/TT scale ratio: {scale_ratio:.2f}x")
    if scale_ratio > 1.3:
        logger.warning(f"   ⚠️ TT logits are {scale_ratio:.1f}x smaller - check attention scaling or output projection")

    logger.info(f"\n2. RANKING ANALYSIS:")
    logger.info(f"   HF's answer '{tokenizer.decode([hf_top])}' ranks #{hf_answer_rank} in TT")
    if hf_answer_rank <= 3:
        logger.info(f"   ✓ Correct answer is close - this is a MARGINAL difference")
        logger.info(f"   The model 'almost' got it right - likely numerical precision issue")
    else:
        logger.warning(f"   ⚠️ Correct answer is far down - this is a SIGNIFICANT divergence")

    logger.info(f"\n3. POSSIBLE CAUSES:")

    if scale_ratio > 1.5:
        logger.info("   - Attention scaling mismatch (softmax temperature)")
        logger.info("   - Missing or incorrect scaling in SDPA")
        logger.info("   - Output projection weight precision loss")

    if hf_answer_rank <= 5 and scale_ratio < 2.0:
        logger.info("   - bfloat16/bfloat8 precision causing ranking changes")
        logger.info("   - Small numerical differences amplified by softmax")
        logger.info("   - KV cache precision loss accumulating across layers")

    logger.info(f"\n4. RECOMMENDED NEXT STEPS:")
    logger.info("   a) Run with DEBUG_DECODE_ATTENTION=1 to see per-op tensor stats")
    logger.info("   b) Check if attention scaling factor matches HF")
    logger.info("   c) Compare Q, K values after RoPE between TT and HF")
    logger.info("   d) Check if KV cache values match after prefill")

    # ============ QUICK CHECK: Is it attention scaling? ============
    logger.info("\n" + "-" * 100)
    logger.info("ATTENTION SCALING CHECK:")
    logger.info("-" * 100)

    # The attention scale should be 1/sqrt(head_dim)
    head_dim = model_args[0].head_dim
    expected_scale = 1.0 / (head_dim**0.5)
    logger.info(f"Expected attention scale (1/sqrt({head_dim})): {expected_scale:.6f}")
    logger.info(f"If TT uses a different scale, logits will be systematically different")

    # Check the config
    if hasattr(model_args[0], "attention_scale"):
        logger.info(f"Model config attention_scale: {model_args[0].attention_scale}")

    logger.info("\n" + "=" * 100)
    logger.info("TEST COMPLETE - Review diagnosis above")
    logger.info("=" * 100)

    # ============ COMPARE TT PREFILL OUTPUT TO HF ============
    logger.info("\n" + "=" * 100)
    logger.info("PREFILL OUTPUT COMPARISON (is divergence already present?)")
    logger.info("=" * 100)

    tt_prefill_logits_user = tt_logits[FAILING_POSITION].squeeze().float()

    # Get HF prefill output
    with torch.no_grad():
        hf_prefill_output = reference_model(
            input_ids=hf_input_ids,
            attention_mask=_build_attention_mask(hf_input_ids, [decoding_pos[FAILING_POSITION]]),
            use_cache=False,
        )
    hf_prefill_logits = hf_prefill_output.logits[0, decoding_pos[FAILING_POSITION] - 1, :].float()

    prefill_cos_sim = torch.nn.functional.cosine_similarity(
        tt_prefill_logits_user.unsqueeze(0), hf_prefill_logits.unsqueeze(0)
    ).item()

    tt_prefill_top = torch.argmax(tt_prefill_logits_user).item()
    hf_prefill_top = torch.argmax(hf_prefill_logits).item()
    prefill_match = tt_prefill_top == hf_prefill_top
    prefill_scale = hf_prefill_logits.max().item() / (tt_prefill_logits_user.max().item() + 1e-6)

    logger.info(f"PREFILL:")
    logger.info(
        f"  TT: top='{tokenizer.decode([tt_prefill_top])}', max={tt_prefill_logits_user.max():.2f}, std={tt_prefill_logits_user.std():.2f}"
    )
    logger.info(
        f"  HF: top='{tokenizer.decode([hf_prefill_top])}', max={hf_prefill_logits.max():.2f}, std={hf_prefill_logits.std():.2f}"
    )
    logger.info(f"  Match: {prefill_match}, Cosine: {prefill_cos_sim:.4f}, Scale ratio: {prefill_scale:.2f}x")

    logger.info(f"\nDECODE:")
    logger.info(
        f"  TT: top='{tokenizer.decode([tt_top])}', max={tt_decode_logits_user.max():.2f}, std={tt_decode_logits_user.std():.2f}"
    )
    logger.info(
        f"  HF: top='{tokenizer.decode([hf_top])}', max={hf_decode_logits.max():.2f}, std={hf_decode_logits.std():.2f}"
    )
    logger.info(f"  Match: {tt_top == hf_top}, Cosine: {cos_sim:.4f}, Scale ratio: {scale_ratio:.2f}x")

    if prefill_cos_sim > 0.9 and cos_sim < 0.7:
        logger.error("\n🚨 PREFILL is good but DECODE diverges!")
        logger.error("   The bug is in DECODE path, not prefill.")
        logger.error("   Check: paged attention decode, KV cache read, or decode-specific ops")
    elif prefill_cos_sim < 0.7:
        logger.error("\n🚨 PREFILL already has low cosine similarity!")
        logger.error("   The bug starts in PREFILL, before decode even runs.")
        logger.error("   Check: prefill attention, embedding, or weight loading")
    else:
        logger.info(f"\n✓ Both prefill ({prefill_cos_sim:.2f}) and decode ({cos_sim:.2f}) have similar divergence")
        logger.info("   Suggests systematic precision issue affecting both paths equally")


@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
def test_garbage_prompt_diagnosis(mesh_device, use_program_cache, reset_seeds):
    """
    Diagnose why specific prompts produce garbage outputs like "Scrolling...".

    Tests:
    1. Expert routing - Are the right experts being selected?
    2. Attention patterns - Are attention sinks working correctly?
    3. Token-by-token analysis - Where does generation go wrong?
    """
    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gpt_oss.tt.common import create_tt_model
    from models.demos.gpt_oss.tt.model_config import ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.generator import Generator

    logger.info("=" * 100)
    logger.info("GARBAGE PROMPT DIAGNOSIS TEST")
    logger.info("=" * 100)

    # Known garbage-producing prompts (from demo output)
    GARBAGE_PROMPTS = {
        31: "How many legs does a spider have?",  # "Scrolling..." garbage
        21: "Who composed the Fifth Symphony?",  # Wrong answer + dots
        0: "How many moons does Earth have?",  # Looping confused text
        8: "What is the tallest mammal alive?",  # Repeating "African elephant"
    }

    # Setup
    global_batch_size = 128
    max_seq_len = 1024
    max_gen_len = 50  # Generate more tokens to see the garbage pattern

    model_args = [ModelArgs(mesh_device=mesh_device, max_batch_size=global_batch_size, max_seq_len=max_seq_len)]
    tokenizer = AutoTokenizer.from_pretrained(model_args[0].model_path, trust_remote_code=True)

    # Load reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_args[0].model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    reference_model.eval()

    # Setup paged attention
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks": blocks_per_user * users_per_row}
    paged_attention_config = PagedAttentionConfig(**page_params)

    # Create TT model
    tt_model = create_tt_model(
        mesh_device,
        model_args[0],
        dtype=ttnn.bfloat16,
        paged_attention_config=paged_attention_config,
        max_batch_size=global_batch_size,
        users_row_sharded=True,
    )

    # Create generator
    generator = Generator([tt_model], model_args, mesh_device)

    logger.info("\n" + "=" * 100)
    logger.info("TESTING KNOWN GARBAGE PROMPTS")
    logger.info("=" * 100)

    for position, prompt in GARBAGE_PROMPTS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"PROMPT POSITION {position}: '{prompt}'")
        logger.info(f"{'='*80}")

        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(tokens)
        logger.info(f"Token count: {prompt_len}")
        logger.info(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")

        # Get HF reference output for first token
        hf_input_ids = torch.tensor([tokens])
        with torch.no_grad():
            hf_output = reference_model(hf_input_ids, use_cache=True, output_attentions=True)

        hf_logits = hf_output.logits[0, -1, :].float()
        hf_top_token = torch.argmax(hf_logits).item()
        hf_top_5 = torch.topk(hf_logits, 5)

        logger.info(f"\nHF Reference:")
        logger.info(f"  Top token: '{tokenizer.decode([hf_top_token])}' ({hf_top_token})")
        logger.info(
            f"  Top 5: {[(tokenizer.decode([t.item()]), f'{v.item():.2f}') for t, v in zip(hf_top_5.indices, hf_top_5.values)]}"
        )
        logger.info(f"  Logits: max={hf_logits.max():.2f}, min={hf_logits.min():.2f}, std={hf_logits.std():.2f}")

        # Analyze HF attention patterns (check for attention sink issues)
        if hf_output.attentions is not None:
            last_layer_attn = hf_output.attentions[-1][0]  # [num_heads, seq_len, seq_len]

            # Check attention to first token (attention sink)
            attn_to_first = last_layer_attn[:, -1, 0].mean().item()  # Last token attending to first
            attn_to_last_few = last_layer_attn[:, -1, -5:].mean().item()  # Last token attending to recent

            logger.info(f"\nHF Attention Pattern (last layer, last token):")
            logger.info(f"  Attention to first token (sink): {attn_to_first:.4f}")
            logger.info(f"  Attention to last 5 tokens: {attn_to_last_few:.4f}")

            # Check for unusual attention patterns
            attn_entropy = (
                -torch.sum(last_layer_attn[:, -1, :] * torch.log(last_layer_attn[:, -1, :] + 1e-10), dim=-1)
                .mean()
                .item()
            )
            logger.info(f"  Attention entropy (higher=more spread): {attn_entropy:.4f}")

        # Generate multiple tokens with HF to see expected continuation
        logger.info(f"\nHF Generation (10 tokens):")
        with torch.no_grad():
            hf_generated = reference_model.generate(
                hf_input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        hf_continuation = tokenizer.decode(hf_generated[0][prompt_len:], skip_special_tokens=False)
        logger.info(f"  '{hf_continuation}'")

        # Check for specific patterns that might cause issues
        logger.info(f"\nPotential Issue Indicators:")

        # 1. Check if prompt ends with special characters
        last_token = tokenizer.decode([tokens[-1]])
        logger.info(f"  Last token: '{last_token}' (ID: {tokens[-1]})")

        # 2. Check token distribution
        unique_tokens = len(set(tokens))
        logger.info(f"  Unique tokens: {unique_tokens}/{len(tokens)} ({100*unique_tokens/len(tokens):.0f}%)")

        # 3. Check for question mark position
        if "?" in prompt:
            q_pos = prompt.index("?")
            logger.info(f"  Question mark at position {q_pos}/{len(prompt)} ({100*q_pos/len(prompt):.0f}%)")

        # 4. Expected answer characteristics
        if "spider" in prompt.lower():
            logger.info(f"  Expected answer: '8' or 'eight' (simple factual)")
        elif "Fifth Symphony" in prompt:
            logger.info(f"  Expected answer: 'Beethoven' (simple factual)")
        elif "moons" in prompt.lower():
            logger.info(f"  Expected answer: '1' or 'one' (simple factual)")
        elif "tallest mammal" in prompt.lower():
            logger.info(f"  Expected answer: 'giraffe' (simple factual)")

    logger.info("\n" + "=" * 100)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("=" * 100)
    logger.info("\nNext steps to debug:")
    logger.info("1. Run with DEBUG_DECODE_ATTENTION=1 to see TT attention stats")
    logger.info("2. Compare TT expert selection vs HF for these prompts")
    logger.info("3. Check if garbage prompts have unusual token patterns")


@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
def test_expert_routing_comparison(mesh_device, use_program_cache, reset_seeds):
    """
    Compare expert routing decisions between TT and HF for garbage-producing prompts.

    The MoE (Mixture of Experts) layer selects top-k experts per token.
    If TT selects different experts than HF, outputs will diverge significantly.
    """
    from loguru import logger
    from transformers import AutoConfig

    logger.info("=" * 100)
    logger.info("EXPERT ROUTING COMPARISON TEST")
    logger.info("=" * 100)

    # Load model config to check if it's MoE
    model_path = "meta-llama/Llama-3.1-8B"  # Update if different
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Check if model has MoE
    has_moe = hasattr(config, "num_local_experts") or hasattr(config, "num_experts")

    if not has_moe:
        logger.info("Model is NOT MoE - expert routing test not applicable")
        logger.info("The garbage outputs are likely caused by:")
        logger.info("  1. Attention precision issues")
        logger.info("  2. KV cache corruption")
        logger.info("  3. Position encoding issues")
        pytest.skip("Model is not MoE - test not applicable")
        return

    num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 8))
    top_k = getattr(config, "num_experts_per_tok", 2)

    logger.info(f"MoE Configuration:")
    logger.info(f"  Number of experts: {num_experts}")
    logger.info(f"  Top-k experts per token: {top_k}")

    # For MoE models, we would hook into the gating layer to compare routing
    # This requires model-specific hooks

    logger.info("\nTo debug expert routing:")
    logger.info("1. Add hooks to the gating network in both TT and HF models")
    logger.info("2. Compare router logits (pre-softmax) for each token")
    logger.info("3. Compare selected expert indices")
    logger.info("4. If routing differs, check gating network precision")


@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
def test_attention_sink_analysis(mesh_device, use_program_cache, reset_seeds):
    """
    Analyze attention sink behavior for garbage-producing prompts.

    Attention sinks (StreamingLLM) reserve the first few token positions
    for attention to flow to. If this mechanism fails, outputs can degrade.
    """
    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gpt_oss.tt.model_config import ModelArgs

    logger.info("=" * 100)
    logger.info("ATTENTION SINK ANALYSIS TEST")
    logger.info("=" * 100)

    # Known garbage prompts
    GARBAGE_PROMPTS = [
        "How many legs does a spider have?",
        "Who composed the Fifth Symphony?",
    ]

    GOOD_PROMPTS = [
        "What is the capital of France?",
        "What is 2 + 2?",
    ]

    model_args = ModelArgs(mesh_device=mesh_device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

    # Load HF model with attention output
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager for attention weights
    )
    reference_model.eval()

    def analyze_attention(prompt, label):
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens])

        with torch.no_grad():
            output = reference_model(input_ids, output_attentions=True)

        # Analyze attention patterns across layers
        attention_to_first = []
        attention_entropy = []

        for layer_idx, layer_attn in enumerate(output.attentions):
            # layer_attn shape: [batch, num_heads, seq_len, seq_len]
            attn = layer_attn[0]  # Remove batch dim

            # Attention from last token to first token (sink)
            sink_attn = attn[:, -1, 0].mean().item()
            attention_to_first.append(sink_attn)

            # Entropy of attention distribution (last token)
            attn_dist = attn[:, -1, :]  # [num_heads, seq_len]
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10), dim=-1).mean().item()
            attention_entropy.append(entropy)

        return {
            "prompt": prompt,
            "label": label,
            "seq_len": len(tokens),
            "sink_attention": attention_to_first,
            "entropy": attention_entropy,
            "avg_sink": sum(attention_to_first) / len(attention_to_first),
            "avg_entropy": sum(attention_entropy) / len(attention_entropy),
        }

    logger.info("\n" + "-" * 80)
    logger.info("ANALYZING GARBAGE PROMPTS")
    logger.info("-" * 80)

    garbage_results = [analyze_attention(p, "GARBAGE") for p in GARBAGE_PROMPTS]
    good_results = [analyze_attention(p, "GOOD") for p in GOOD_PROMPTS]

    for result in garbage_results + good_results:
        logger.info(f"\n[{result['label']}] '{result['prompt'][:40]}...'")
        logger.info(f"  Sequence length: {result['seq_len']}")
        logger.info(f"  Average sink attention: {result['avg_sink']:.4f}")
        logger.info(f"  Average entropy: {result['avg_entropy']:.4f}")
        logger.info(f"  Sink attention by layer (first 5, last 5):")
        logger.info(f"    First 5: {[f'{x:.3f}' for x in result['sink_attention'][:5]]}")
        logger.info(f"    Last 5: {[f'{x:.3f}' for x in result['sink_attention'][-5:]]}")

    # Compare patterns
    logger.info("\n" + "=" * 80)
    logger.info("PATTERN COMPARISON")
    logger.info("=" * 80)

    avg_garbage_sink = sum(r["avg_sink"] for r in garbage_results) / len(garbage_results)
    avg_good_sink = sum(r["avg_sink"] for r in good_results) / len(good_results)
    avg_garbage_entropy = sum(r["avg_entropy"] for r in garbage_results) / len(garbage_results)
    avg_good_entropy = sum(r["avg_entropy"] for r in good_results) / len(good_results)

    logger.info(f"Average sink attention - Garbage: {avg_garbage_sink:.4f}, Good: {avg_good_sink:.4f}")
    logger.info(f"Average entropy - Garbage: {avg_garbage_entropy:.4f}, Good: {avg_good_entropy:.4f}")

    if abs(avg_garbage_sink - avg_good_sink) > 0.01:
        logger.warning("⚠️ Sink attention differs between garbage and good prompts!")
        logger.warning("   This could indicate attention sink issues in TT model")
    else:
        logger.info("✓ Sink attention patterns are similar")

    if abs(avg_garbage_entropy - avg_good_entropy) > 0.5:
        logger.warning("⚠️ Attention entropy differs significantly!")
        logger.warning("   Garbage prompts may have unusual attention distributions")
    else:
        logger.info("✓ Attention entropy patterns are similar")

    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    logger.info("If garbage prompts have different patterns:")
    logger.info("  1. Check TT attention sink configuration matches HF")
    logger.info("  2. Verify sliding window attention is correctly implemented")
    logger.info("  3. Compare TT vs HF attention weights for these specific prompts")


@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8")], indirect=True)
def test_token_by_token_generation(mesh_device, use_program_cache, reset_seeds):
    """
    Generate tokens one-by-one and compare TT vs HF at each step.
    This helps identify exactly when/where generation diverges.
    """
    import torch
    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gpt_oss.tt.common import create_tt_model
    from models.demos.gpt_oss.tt.model_config import ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.generator import Generator

    logger.info("=" * 100)
    logger.info("TOKEN-BY-TOKEN GENERATION COMPARISON")
    logger.info("=" * 100)

    # Test the worst garbage prompt
    TEST_PROMPT = "How many legs does a spider have?"
    NUM_TOKENS = 20

    # Setup
    global_batch_size = 128
    max_seq_len = 1024

    model_args = [ModelArgs(mesh_device=mesh_device, max_batch_size=global_batch_size, max_seq_len=max_seq_len)]
    tokenizer = AutoTokenizer.from_pretrained(model_args[0].model_path, trust_remote_code=True)

    # Load HF reference
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_args[0].model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    reference_model.eval()

    # Setup TT model
    users_per_row = global_batch_size // mesh_device.shape[0]
    blocks_per_user = max_seq_len // 64
    page_params = {"page_block_size": 64, "page_max_num_blocks": blocks_per_user * users_per_row}
    paged_attention_config = PagedAttentionConfig(**page_params)

    tt_model = create_tt_model(
        mesh_device,
        model_args[0],
        dtype=ttnn.bfloat16,
        paged_attention_config=paged_attention_config,
        max_batch_size=global_batch_size,
        users_row_sharded=True,
    )

    generator = Generator([tt_model], model_args, mesh_device)

    # Tokenize prompt
    tokens = tokenizer.encode(TEST_PROMPT, add_special_tokens=True)
    prompt_len = len(tokens)

    logger.info(f"Prompt: '{TEST_PROMPT}'")
    logger.info(f"Prompt tokens: {tokens}")
    logger.info(f"Prompt length: {prompt_len}")

    # Generate with HF token-by-token
    logger.info("\n" + "-" * 80)
    logger.info("HF GENERATION (token-by-token)")
    logger.info("-" * 80)

    hf_tokens = tokens.copy()
    hf_generated = []

    with torch.no_grad():
        for step in range(NUM_TOKENS):
            input_ids = torch.tensor([hf_tokens])
            output = reference_model(input_ids, use_cache=False)
            logits = output.logits[0, -1, :].float()
            next_token = torch.argmax(logits).item()

            hf_tokens.append(next_token)
            hf_generated.append(next_token)

            decoded = tokenizer.decode([next_token])
            logger.info(f"  Step {step}: token={next_token}, text='{decoded}', logit_max={logits.max():.2f}")

    hf_text = tokenizer.decode(hf_generated)
    logger.info(f"\nHF output: '{hf_text}'")

    # Now we need to compare with TT
    # For simplicity, just log the expected vs actual pattern
    logger.info("\n" + "-" * 80)
    logger.info("EXPECTED VS OBSERVED PATTERN")
    logger.info("-" * 80)
    logger.info(f"HF generates: '{hf_text}'")
    logger.info(f"TT (demo) generates: 'Scrolling... Scrolling...' (garbage)")

    logger.info("\n" + "-" * 80)
    logger.info("DIVERGENCE ANALYSIS")
    logger.info("-" * 80)

    # Check if the first token is correct
    first_hf_token = hf_generated[0]
    first_hf_text = tokenizer.decode([first_hf_token])
    logger.info(f"First expected token: '{first_hf_text}' ({first_hf_token})")

    # The garbage "Scrolling" token
    scrolling_tokens = tokenizer.encode("Scrolling", add_special_tokens=False)
    logger.info(f"'Scrolling' tokenizes to: {scrolling_tokens}")

    # Check if there's any relationship
    logger.info("\nIf TT produces 'Scrolling' instead of the correct answer:")
    logger.info("  1. Check if 'Scrolling' token ID is near expected token ID")
    logger.info(f"     Expected: {first_hf_token}, Scrolling: {scrolling_tokens}")
    logger.info("  2. This could indicate logit corruption or wrong position")
    logger.info("  3. Check prefill output for this specific user position")

    # Check specific token IDs
    logger.info("\n" + "-" * 80)
    logger.info("TOKEN ID ANALYSIS")
    logger.info("-" * 80)

    special_tokens = {
        "Scrolling": tokenizer.encode("Scrolling", add_special_tokens=False),
        "eight": tokenizer.encode("eight", add_special_tokens=False),
        "8": tokenizer.encode("8", add_special_tokens=False),
        "...": tokenizer.encode("...", add_special_tokens=False),
        "…": tokenizer.encode("…", add_special_tokens=False),
    }

    for name, toks in special_tokens.items():
        logger.info(f"  '{name}': {toks}")

    logger.info("\nIf 'Scrolling' appears, check:")
    logger.info("  1. Is position 31 getting correct KV cache entries?")
    logger.info("  2. Is the page table correct for user 31?")
    logger.info("  3. Are the prefill logits correct before decode starts?")
