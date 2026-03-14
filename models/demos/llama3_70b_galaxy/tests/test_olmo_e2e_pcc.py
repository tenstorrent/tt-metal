# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B E2E PCC Test (1 layer).

Compares TTNN prefill + decode outputs against CPU reference:
1. Prefill PCC: full model forward (embed → layer → norm → lm_head) on real tokens
2. Decode PCC: single decode step after prefill
3. E2E PCC: prefill → multi-step decode, compare token-by-token

Run with:
    export HF_MODEL=~/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think/snapshots/<hash>
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_e2e_pcc.py -xvs
"""

import os
import math
import torch
import pytest
from loguru import logger
import ttnn

from models.demos.llama3_70b_galaxy.tt.llama_common import (
    PagedAttentionConfig,
    precompute_freqs_yarn,
    gather_cos_sin,
)
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device
from models.demos.llama3_70b_galaxy.reference.olmo import Transformer as RefTransformer, OlmoModelArgs
from models.common.utility_functions import comp_pcc

from transformers import GPT2Tokenizer


def load_hf_raw_state_dict(ckpt_dir):
    """Load raw HF state dict without key conversion."""
    import glob
    from safetensors.torch import load_file

    base_path = os.path.expanduser(ckpt_dir)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snap_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snap_dirs:
            base_path = snap_dirs[0]

    safetensor_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f))

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    return state_dict


def build_ref_model(hf_sd, n_layers=1, max_seq_len=256, max_batch_size=1):
    """Build the CPU reference Transformer with only n_layers."""
    args = OlmoModelArgs(n_layers=n_layers, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    model = RefTransformer(args)

    model.tok_embeddings.weight.data = hf_sd["model.embed_tokens.weight"].float()
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        layer = model.layers[i]
        layer.attention.wq.weight.data = hf_sd[f"{prefix}.self_attn.q_proj.weight"].float()
        layer.attention.wk.weight.data = hf_sd[f"{prefix}.self_attn.k_proj.weight"].float()
        layer.attention.wv.weight.data = hf_sd[f"{prefix}.self_attn.v_proj.weight"].float()
        layer.attention.wo.weight.data = hf_sd[f"{prefix}.self_attn.o_proj.weight"].float()
        layer.feed_forward.w1.weight.data = hf_sd[f"{prefix}.mlp.gate_proj.weight"].float()
        layer.feed_forward.w2.weight.data = hf_sd[f"{prefix}.mlp.down_proj.weight"].float()
        layer.feed_forward.w3.weight.data = hf_sd[f"{prefix}.mlp.up_proj.weight"].float()
        # OLMo3 post-sublayer-norm:
        # attention_norm applied AFTER attention = post_attention_layernorm
        # ffn_norm       applied AFTER FFN       = post_feedforward_layernorm
        attn_norm_key = f"{prefix}.post_attention_layernorm.weight"
        ffn_norm_key = f"{prefix}.post_feedforward_layernorm.weight"
        layer.attention_norm.weight.data = hf_sd[attn_norm_key].float()
        layer.ffn_norm.weight.data = hf_sd[ffn_norm_key].float()

        # QK-norm if present
        q_norm_key = f"{prefix}.self_attn.q_norm.weight"
        k_norm_key = f"{prefix}.self_attn.k_norm.weight"
        if q_norm_key in hf_sd:
            import torch.nn as nn

            layer.attention.q_norm_weight = nn.Parameter(hf_sd[q_norm_key].float())
            layer.attention.k_norm_weight = nn.Parameter(hf_sd[k_norm_key].float())

    model.norm.weight.data = hf_sd["model.norm.weight"].float()
    model.output.weight.data = hf_sd["lm_head.weight"].float()
    model.eval()
    return model


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": True,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
class TestOlmoE2EPCC:
    @torch.no_grad()
    def test_prefill_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: embed → 1 layer → norm → lm_head, compare logits vs CPU."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 1
        max_seq_len = 256
        batch_size = 1
        dtype = ttnn.bfloat8_b

        # Load raw HF weights for reference
        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # Build CPU reference
        logger.info("Building CPU reference model (1 layer)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len)

        # Build TTNN model
        logger.info("Building TTNN model (1 layer)...")
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()

        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )

        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            decode_mode_only=False,
        )

        # Prepare input: real tokens
        tokenizer = model_args.tokenizer
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

        prompt = "What is your favorite condiment?"
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        seq_len = len(input_ids)
        padded_len = 128  # match demo
        input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - seq_len)
        tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

        # ===== CPU Reference: prefill =====
        logger.info(f"Running CPU prefill (seq_len={seq_len}, padded={padded_len})...")
        embeddings_ref = ref_model.tok_embeddings(tokens_pt[:, :padded_len].long()).float()
        ref_logits = ref_model.forward(embeddings_ref, start_pos=0, mode="decode")  # "decode" mode returns logits
        ref_logits_last = ref_logits[:, seq_len - 1, :]  # logits at last real token
        ref_token = ref_logits_last.argmax(dim=-1).item()
        logger.info(f"CPU reference token: {ref_token} ({tokenizer.decode([ref_token])})")

        # ===== TTNN: prefill =====
        logger.info("Running TTNN prefill...")
        kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

        # Compute YaRN RoPE
        ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
            dim=model_args.head_dim,
            end=model_args.max_seq_len * 2,
            theta=model_args.rope_theta,
            scaling_factor=model_args.rope_scaling_factor,
            original_max_position_embeddings=model_args.original_max_position_embeddings,
            beta_fast=model_args.yarn_beta_fast,
            beta_slow=model_args.yarn_beta_slow,
            attention_factor=model_args.yarn_attention_factor,
        )
        position_ids = torch.arange(padded_len)
        cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)
        rot_mats_prefill = [
            ttnn.from_torch(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            ttnn.from_torch(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        ]
        tt_model.tt_rot_mats_prefill = rot_mats_prefill

        block_size = paged_attention_config.block_size
        num_prefill_blocks = math.ceil(padded_len / block_size)
        prefill_page_table = torch.ones(32, num_prefill_blocks, dtype=torch.int32) * -1
        prefill_page_table[0, :] = page_table[0, :num_prefill_blocks]

        host_inputs = tt_model.prepare_prefill_inputs_host(tokens_pt, user_id=0, page_table=prefill_page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
        transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)
        tt_out_prefill = tt_model.ttnn_prefill_forward(
            *transformed_inputs,
            kv_cache=kv_cache,
            batch_size=1,
        )

        # Get logits with tt_out_logits_saved for PCC comparison
        # After all_gather the logits may be tile-aligned (100352 for OLMo), so use larger buffer
        logits_buf_size = 100352  # 8 devices * 12544 per device (tile-aligned)
        tt_out_logits_saved = torch.zeros(1, logits_buf_size)
        tt_tok = tt_model.process_output_prefill(
            tt_out_prefill, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_out_logits_saved
        )
        ttnn.synchronize_device(mesh_device)
        tt_token = int(tt_tok[0])
        logger.info(f"TTNN token: {tt_token} ({tokenizer.decode([tt_token])})")

        # Trim logits to vocab_size for comparison
        vocab_size = model_args.vocab_size
        tt_logits = tt_out_logits_saved[:, :vocab_size]
        ref_logits_trimmed = ref_logits[:, seq_len - 1, :vocab_size]

        logger.info(f"TTNN logits shape: {tt_logits.shape}, ref shape: {ref_logits_trimmed.shape}")
        logger.info(f"TTNN logits stats: mean={tt_logits.mean():.4f}, std={tt_logits.std():.4f}")
        logger.info(f"Ref  logits stats: mean={ref_logits_trimmed.mean():.4f}, std={ref_logits_trimmed.std():.4f}")

        # PCC check
        passing, pcc_msg = comp_pcc(ref_logits_trimmed.float(), tt_logits.float(), 0.80)
        logger.info(f"Prefill logits PCC: {pcc_msg}")
        logger.info(f"Token match: CPU={ref_token}, TTNN={tt_token}, match={ref_token == tt_token}")

        assert passing, f"Prefill PCC {pcc_msg} < 0.80"
        logger.info("PREFILL PCC TEST: PASSED")

    @torch.no_grad()
    def test_decode_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step after prefill, compare hidden states vs CPU."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 1
        max_seq_len = 256
        batch_size = 32
        start_pos = 127
        dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ===== CPU Reference decode =====
        logger.info("Building CPU reference model (1 layer)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, 1, 5120)
        embeddings_ref = pt_input.float()

        # Run reference decode at start_pos
        ref_out = ref_model.forward(embeddings_ref, start_pos=start_pos, mode="decode")
        logger.info(f"CPU ref output shape: {ref_out.shape}")
        logger.info(
            f"CPU ref stats: mean={ref_out.mean():.4f}, std={ref_out.std():.4f}, "
            f"nan={torch.isnan(ref_out).any()}, inf={torch.isinf(ref_out).any()}"
        )

        # ===== TTNN decode =====
        logger.info("Building TTNN model (1 layer)...")
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()

        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)

        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            decode_mode_only=True,
        )

        tt_input = model_args.prepare_residual_tensor_decode(
            pt_input.clone(), model_args.model_config["DECODE_RESIDUAL_MEMCFG"]
        )

        current_pos = torch.tensor([start_pos] * batch_size)
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=model_args.cluster_shape),
        )

        rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

        # Call forward() directly (bypassing embd since tt_input is already hidden states)
        # This tests the full decode path: decoder layers → norm → lm_head
        tt_out_list = tt_model.forward(
            tt_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            kv_cache=[layer.attention.layer_past for layer in tt_model.layers],
        )
        ttnn.synchronize_device(mesh_device)

        # Extract user-0 logits using dims=(3, 1): concat 8 row devices along dim 3 for full vocab
        # tt_out_list[0] shape per device: [1, 1, 32, padded_vocab_shard]
        vocab_size = model_args.vocab_size
        tt_out_logits = ttnn.to_torch(
            tt_out_list[0],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
        )
        # Shape after concat: [1, 4, 32, padded_vocab] — take [col0, row_idx=0, batch_pos=0, :vocab_size]
        tt_logits_u0 = tt_out_logits[0, 0, 0, :vocab_size]  # [vocab_size]
        tt_logits = tt_logits_u0.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
        logger.info(f"TTNN output shape: {tt_logits.shape}")
        logger.info(
            f"TTNN stats: mean={tt_logits.mean():.4f}, std={tt_logits.std():.4f}, "
            f"nan={torch.isnan(tt_logits).any()}, inf={torch.isinf(tt_logits).any()}"
        )

        # ===== PCC: compare user 0 logits =====
        # tt_logits: [1, 1, vocab_size] (user 0 only from tt_out_logits_saved)
        # ref_out:   [batch_size, 1, vocab_size] — use user 0
        ref_logits_u0 = ref_out[0:1, :, :vocab_size]  # [1, 1, vocab_size]
        passing, pcc_msg = comp_pcc(ref_logits_u0, tt_logits.float(), 0.80)
        logger.info(f"Decode 1-layer logits PCC (user 0): {pcc_msg}")

        # Token agreement for user 0
        ref_tok0 = ref_logits_u0.argmax(dim=-1).item()
        tt_tok0 = tt_logits.argmax(dim=-1).item()
        logger.info(f"User 0 token match: ref={ref_tok0}, tt={tt_tok0}, match={ref_tok0 == tt_tok0}")

        assert passing, f"Decode PCC {pcc_msg} < 0.80"
        logger.info("DECODE PCC TEST: PASSED")

    @torch.no_grad()
    def test_e2e_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """E2E PCC: prefill + decode with 1 layer, compare token generation vs CPU."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 1
        max_seq_len = 256
        batch_size = 1
        n_decode_tokens = 10
        dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ===== CPU Reference Model =====
        logger.info("Building CPU reference (1 layer)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len)

        # Build model_args early to get tokenizer path
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers

        tokenizer = model_args.tokenizer
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        seq_len = len(input_ids)
        padded_len = 128
        input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - seq_len)
        tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

        # CPU prefill
        logger.info(f"CPU prefill (seq_len={seq_len})...")
        embeddings_ref = ref_model.tok_embeddings(tokens_pt[:, :padded_len]).float()
        ref_prefill_out = ref_model.forward(embeddings_ref, start_pos=0, mode="decode")
        ref_prefill_logits = ref_prefill_out[:, seq_len - 1, :]
        ref_first_token = ref_prefill_logits.argmax(dim=-1).item()
        logger.info(f"CPU first token: {ref_first_token} ({tokenizer.decode([ref_first_token])})")

        # CPU decode loop
        ref_tokens = [ref_first_token]
        for step in range(n_decode_tokens):
            pos = seq_len + step
            tok_emb = ref_model.tok_embeddings(torch.tensor([[ref_tokens[-1]]])).float()
            ref_decode_out = ref_model.forward(tok_emb, start_pos=pos, mode="decode")
            next_tok = ref_decode_out[:, -1, :].argmax(dim=-1).item()
            ref_tokens.append(next_tok)

        ref_text = tokenizer.decode(ref_tokens)
        logger.info(f"CPU generated: {prompt}{ref_text}")

        # ===== TTNN Model =====
        logger.info("Building TTNN model (1 layer)...")
        state_dict = model_args.load_state_dict()

        # Use paged attention (required for prefill) with a proper page table for decode too
        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )

        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            decode_mode_only=False,
        )

        # TTNN prefill
        logger.info("TTNN prefill...")
        kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

        ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
            dim=model_args.head_dim,
            end=model_args.max_seq_len * 2,
            theta=model_args.rope_theta,
            scaling_factor=model_args.rope_scaling_factor,
            original_max_position_embeddings=model_args.original_max_position_embeddings,
            beta_fast=model_args.yarn_beta_fast,
            beta_slow=model_args.yarn_beta_slow,
            attention_factor=model_args.yarn_attention_factor,
        )
        position_ids = torch.arange(padded_len)
        cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)
        rot_mats_prefill = [
            ttnn.from_torch(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            ttnn.from_torch(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        ]
        tt_model.tt_rot_mats_prefill = rot_mats_prefill

        block_size = paged_attention_config.block_size
        num_prefill_blocks = math.ceil(padded_len / block_size)
        prefill_page_table = torch.ones(32, num_prefill_blocks, dtype=torch.int32) * -1
        prefill_page_table[0, :] = page_table[0, :num_prefill_blocks]

        host_inputs = tt_model.prepare_prefill_inputs_host(tokens_pt, user_id=0, page_table=prefill_page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
        transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)
        tt_out_prefill = tt_model.ttnn_prefill_forward(
            *transformed_inputs,
            kv_cache=kv_cache,
            batch_size=1,
        )
        tt_first_tok_result = tt_model.process_output_prefill(tt_out_prefill, last_token_idx=seq_len - 1)

        # Build decode page_table tensor for device (same page mapping, user 0 only)
        decode_page_table_pt = page_table.clone()  # [batch_size_per_device_group, pages_per_user]
        decode_page_table_tt = ttnn.from_torch(
            decode_page_table_pt,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        tt_first_token = int(tt_first_tok_result[0])
        logger.info(f"TTNN first token: {tt_first_token} ({tokenizer.decode([tt_first_token])})")

        # Compare prefill
        prefill_match = ref_first_token == tt_first_token
        logger.info(f"Prefill token match: {prefill_match} (CPU={ref_first_token}, TTNN={tt_first_token})")

        # TTNN decode loop
        logger.info("TTNN decode loop...")
        tt_model.switch_mode("decode")

        vocab_size = model_args.vocab_size
        tt_out_logits_saved = torch.zeros(vocab_size)

        tt_tokens = [tt_first_token]
        current_pos_val = seq_len
        for step in range(n_decode_tokens):
            tok_tensor = torch.full((1, 1, 1, 32), 0, dtype=torch.long)
            tok_tensor[0, 0, 0, 0] = tt_tokens[-1]

            tt_tok_device = ttnn.from_torch(
                tok_tensor,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            current_pos = torch.full((32,), current_pos_val, dtype=torch.long)
            current_pos[0] = current_pos_val
            current_pos_tt = ttnn.from_torch(
                current_pos,
                device=mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=model_args.cluster_shape),
            )

            rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

            tt_model.ttnn_decode_forward(
                tt_tok_device,
                current_pos_tt,
                rot_mat_idxs,
                page_table=decode_page_table_tt,
                kv_cache=kv_cache,
                tt_out_logits_saved=tt_out_logits_saved,
            )
            ttnn.synchronize_device(mesh_device)

            # tt_out_logits_saved is filled with logits for user 0 via dims=(3,1) gather
            next_tok = tt_out_logits_saved.argmax().item()
            if step < 3:
                logger.info(
                    f"  Decode step {step}: logits mean={tt_out_logits_saved.mean():.4f}, "
                    f"std={tt_out_logits_saved.std():.4f}, max={tt_out_logits_saved.max():.4f}, "
                    f"min={tt_out_logits_saved.min():.4f}, argmax={next_tok}, "
                    f"top5={torch.topk(tt_out_logits_saved, 5).indices.tolist()}"
                )
            tt_tokens.append(next_tok)
            current_pos_val += 1

        tt_text = tokenizer.decode(tt_tokens)
        logger.info(f"TTNN generated: {prompt}{tt_text}")

        # ===== Compare E2E =====
        logger.info("=" * 60)
        logger.info("E2E COMPARISON (1 layer)")
        logger.info("=" * 60)
        logger.info(f"CPU:  {prompt}{ref_text}")
        logger.info(f"TTNN: {prompt}{tt_text}")

        total = n_decode_tokens + 1  # +1 for prefill token
        matches = sum(1 for r, t in zip(ref_tokens, tt_tokens) if r == t)
        logger.info(f"Token match: {matches}/{total} ({matches/total*100:.1f}%)")

        for i, (r, t) in enumerate(zip(ref_tokens, tt_tokens)):
            status = "✓" if r == t else "✗"
            r_str = tokenizer.decode([r])
            t_str = tokenizer.decode([t])
            logger.info(f"  Step {i}: CPU={r}({r_str}) TTNN={t}({t_str}) {status}")

        assert matches >= total * 0.5, f"Token match rate too low: {matches}/{total}"
        logger.info("E2E PCC TEST: PASSED")
