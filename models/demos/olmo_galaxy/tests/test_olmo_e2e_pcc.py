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
    pytest models/demos/olmo_galaxy/tests/test_olmo_e2e_pcc.py -xvs
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import GPT2Tokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.olmo_galaxy.reference.olmo import OlmoModelArgs
from models.demos.olmo_galaxy.reference.olmo import Transformer as RefTransformer
from models.demos.olmo_galaxy.tt.llama_common import PagedAttentionConfig, gather_cos_sin, precompute_freqs_yarn
from models.demos.olmo_galaxy.tt.llama_model import TtTransformer
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device


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
    def test_prefill_pcc_64layers(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: full 64-layer model at ISL 128, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, pcc_threshold=0.99)

    @torch.no_grad()
    def test_prefill_pcc_4layers(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at ISL 128, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: embed → 1 layer → norm → lm_head at ISL 128, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, pcc_threshold=0.999)

    # ── ISL 256 (real padded length for isl-128-b1 demo with ChatML overhead) ──────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl256(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 256 ISL (actual padded length for 177-token ChatML prompt), target > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=256, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl256(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 256 ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=256, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl256(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 64-layer model at 256 ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=256, pcc_threshold=0.99)

    # ── ISL 512 ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl512(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 512 ISL, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=512, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl512(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 512 ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=512, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl512(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 64-layer model at 512 ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=512, pcc_threshold=0.99)

    # ── ISL 1k ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl1k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 1k ISL, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=1024, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl1k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 1k ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=1024, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl1k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: full 64-layer model at 1k ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=1024, pcc_threshold=0.99)

    # ── ISL 2k ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl2k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 2k ISL, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=2048, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl2k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 2k ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=2048, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl2k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: full 64-layer model at 2k ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=2048, pcc_threshold=0.99)

    # ── ISL 4k ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl4k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 4k ISL, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=4096, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl4k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 4k ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=4096, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl4k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: full 64-layer model at 4k ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=4096, pcc_threshold=0.99)

    # ── ISL 8k ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_prefill_pcc_1layer_isl8k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 1-layer model at 8k ISL, target hidden PCC > 0.999."""
        self._run_prefill_pcc(mesh_device, n_layers=1, padded_len=8192, pcc_threshold=0.999)

    @torch.no_grad()
    def test_prefill_pcc_4layers_isl8k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: 4-layer model at 8k ISL, target hidden PCC > 0.995."""
        self._run_prefill_pcc(mesh_device, n_layers=4, padded_len=8192, pcc_threshold=0.995)

    @torch.no_grad()
    def test_prefill_pcc_64layers_isl8k(self, mesh_device, reset_seeds, ensure_gc):
        """Prefill PCC: full 64-layer model at 8k ISL, target hidden PCC > 0.99."""
        self._run_prefill_pcc(mesh_device, n_layers=64, padded_len=8192, pcc_threshold=0.99)

    @torch.no_grad()
    def _run_prefill_pcc(self, mesh_device, n_layers, padded_len=128, pcc_threshold=0.90):
        """Shared implementation for prefill PCC tests."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        max_seq_len = max(256, padded_len * 2)
        batch_size = 1
        dtype = ttnn.bfloat8_b

        # Load raw HF weights for reference
        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # Build CPU reference
        logger.info(f"Building CPU reference model ({n_layers} layer(s))...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len)

        # Build TTNN model
        logger.info(f"Building TTNN model ({n_layers} layer(s))...")
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

        # Fill to near-full padded_len so the ring SDPA processes real tokens, not padding.
        # Using only ~500 tokens in a 2048-padded sequence hides sliding-window bugs.
        if padded_len <= 128:
            prompt = "What is your favorite condiment?"
        else:
            base = "The quick brown fox jumps over the lazy dog. "
            repeat_count = (padded_len * 6) // len(base) + 1
            prompt = base * repeat_count
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = input_ids[: padded_len - 5]
        seq_len = len(input_ids)
        input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - seq_len)
        tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

        # ===== CPU Reference: prefill =====
        logger.info(f"Running CPU prefill (seq_len={seq_len}, padded={padded_len})...")
        embeddings_ref = ref_model.tok_embeddings(tokens_pt[:, :padded_len].long()).float()
        ref_hidden = ref_model.forward(embeddings_ref, start_pos=0, mode="prefill")  # hidden states before norm+LM head
        ref_normed = ref_model.norm(ref_hidden)
        ref_logits = ref_model.output(ref_normed)
        ref_logits_last = ref_logits[:, seq_len - 1, :]
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

        # ===== Hidden state PCC (before norm + LM head) =====
        cluster_shape = model_args.cluster_shape
        tt_hidden = ttnn.to_torch(
            tt_out_prefill,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
        ).float()
        if tt_hidden.dim() == 4 and tt_hidden.shape[1] > 1:
            tt_hidden = tt_hidden[:, 0, :, :]
        if tt_hidden.dim() == 4 and tt_hidden.shape[1] == 1:
            tt_hidden = tt_hidden.squeeze(1)

        import torch.nn.functional as F

        ref_hidden_cmp = ref_hidden[:, : tt_hidden.shape[-2], : tt_hidden.shape[-1]]
        passing_hidden, pcc_hidden_msg = comp_pcc(ref_hidden_cmp.float(), tt_hidden.float(), pcc_threshold)
        logger.info(f"Prefill {n_layers}L hidden state PCC (full tensor): {pcc_hidden_msg}")
        logger.info(f"  ref_hidden std={ref_hidden_cmp.std():.6f}, tt_hidden std={tt_hidden.std():.6f}")

        # Per-position PCC at key positions
        logger.info("Per-position hidden state analysis:")
        for pos in list(range(min(seq_len, 8))) + [seq_len - 1] + [padded_len // 2, padded_len - 1]:
            if pos >= tt_hidden.shape[-2]:
                continue
            ref_pos = ref_hidden_cmp[0, pos, :].float()
            tt_pos = tt_hidden[0, pos, :].float()
            _, pos_pcc = comp_pcc(ref_pos.unsqueeze(0), tt_pos.unsqueeze(0), 0.0)
            cos = F.cosine_similarity(ref_pos.unsqueeze(0), tt_pos.unsqueeze(0)).item()
            delta = tt_pos - ref_pos
            is_real = "REAL" if pos < seq_len else "PAD"
            logger.info(
                f"  pos {pos:3d} [{is_real}]: PCC={pos_pcc}, cos={cos:.6f}, "
                f"max_abs={delta.abs().max():.4f}, mean_delta={delta.mean():.6f}, "
                f"ref_std={ref_pos.std():.4f}, tt_std={tt_pos.std():.4f}"
            )

        # PCC of just the real token positions vs just the padding positions
        real_ref = ref_hidden_cmp[0, :seq_len, :].float()
        real_tt = tt_hidden[0, :seq_len, :].float()
        _, pcc_real = comp_pcc(real_ref, real_tt, 0.0)
        pad_ref = ref_hidden_cmp[0, seq_len:, :].float()
        pad_tt = tt_hidden[0, seq_len:, :].float()
        _, pcc_pad = comp_pcc(pad_ref, pad_tt, 0.0)
        logger.info(f"  Real tokens (0:{seq_len}) PCC: {pcc_real}")
        logger.info(f"  Pad  tokens ({seq_len}:{padded_len}) PCC: {pcc_pad}")

        # ===== Logits PCC (after norm + LM head) =====
        logits_buf_size = 100352
        tt_out_logits_saved = torch.zeros(1, logits_buf_size)
        tt_tok = tt_model.process_output_prefill(
            tt_out_prefill, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_out_logits_saved
        )
        ttnn.synchronize_device(mesh_device)
        tt_token = int(tt_tok[0])
        logger.info(f"TTNN token: {tt_token} ({tokenizer.decode([tt_token])})")

        vocab_size = model_args.vocab_size
        tt_logits = tt_out_logits_saved[:, :vocab_size]
        ref_logits_trimmed = ref_logits[:, seq_len - 1, :vocab_size]

        import torch.nn.functional as F

        ref_flat = ref_logits_trimmed.float().flatten()
        tt_flat = tt_logits.float().flatten()
        delta = tt_flat - ref_flat

        passing_logits, pcc_logits_msg = comp_pcc(ref_logits_trimmed.float(), tt_logits.float(), 0.80)
        cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), tt_flat.unsqueeze(0)).item()
        max_abs_err = delta.abs().max().item()
        mean_offset = delta.mean().item()
        median_offset = delta.median().item()

        logger.info(f"Prefill {n_layers}L logits diagnostics:")
        logger.info(f"  PCC:            {pcc_logits_msg}")
        logger.info(f"  Cosine sim:     {cos_sim:.6f}")
        logger.info(f"  Max abs error:  {max_abs_err:.4f}")
        logger.info(f"  Mean offset:    {mean_offset:.6f}")
        logger.info(f"  Median offset:  {median_offset:.6f}")
        logger.info(
            f"  Ref  stats:     mean={ref_flat.mean():.4f}, std={ref_flat.std():.4f}, "
            f"min={ref_flat.min():.4f}, max={ref_flat.max():.4f}"
        )
        logger.info(
            f"  TTNN stats:     mean={tt_flat.mean():.4f}, std={tt_flat.std():.4f}, "
            f"min={tt_flat.min():.4f}, max={tt_flat.max():.4f}"
        )

        # Per-quantile error analysis
        sorted_ref, sort_idx = ref_flat.abs().sort(descending=True)
        for pct_name, n in [("top-100", 100), ("top-1000", 1000), ("top-10000", 10000)]:
            idx = sort_idx[:n]
            ref_slice = ref_flat[idx]
            tt_slice = tt_flat[idx]
            d = tt_slice - ref_slice
            cos_q = F.cosine_similarity(ref_slice.unsqueeze(0), tt_slice.unsqueeze(0)).item()
            logger.info(
                f"  {pct_name:12s}: cos={cos_q:.4f}, mean_delta={d.mean():.4f}, "
                f"max_abs={d.abs().max():.4f}, ref_range=[{ref_slice.min():.2f},{ref_slice.max():.2f}]"
            )

        # Top-K token comparison
        ref_topk = torch.topk(ref_logits_trimmed.squeeze(), 10)
        tt_topk = torch.topk(tt_logits.squeeze(), 10)
        logger.info(
            f"  Ref  top-10 tokens: {ref_topk.indices.tolist()} vals: {[f'{v:.2f}' for v in ref_topk.values.tolist()]}"
        )
        logger.info(
            f"  TTNN top-10 tokens: {tt_topk.indices.tolist()} vals: {[f'{v:.2f}' for v in tt_topk.values.tolist()]}"
        )

        # Also do host-side norm+LM_head on the TTNN hidden state for isolation
        ref_norm_weight = ref_model.norm.weight.float()
        ref_lm_weight = ref_model.output.weight.float()
        tt_hidden_last = tt_hidden[:, seq_len - 1, :].float()  # [1, 5120]
        ref_hidden_last = ref_hidden[:, seq_len - 1, :].float()  # [1, 5120]

        # Host-side norm of TTNN hidden
        tt_h_normed = tt_hidden_last / (tt_hidden_last.pow(2).mean(-1, keepdim=True).sqrt() + 1e-5)
        tt_h_normed = tt_h_normed * ref_norm_weight
        # Host-side LM head of TTNN hidden
        tt_logits_host = tt_h_normed @ ref_lm_weight.T  # [1, vocab_size]

        # Host-side norm of ref hidden (sanity)
        ref_h_normed = ref_hidden_last / (ref_hidden_last.pow(2).mean(-1, keepdim=True).sqrt() + 1e-5)
        ref_h_normed = ref_h_normed * ref_norm_weight
        ref_logits_host = ref_h_normed @ ref_lm_weight.T

        _, pcc_host = comp_pcc(ref_logits_host[:, :vocab_size], tt_logits_host[:, :vocab_size], 0.80)
        cos_host = F.cosine_similarity(
            ref_logits_host[:, :vocab_size].flatten().unsqueeze(0),
            tt_logits_host[:, :vocab_size].flatten().unsqueeze(0),
        ).item()
        logger.info(f"  Host-side logits (TTNN hidden → float32 norm+LM): PCC={pcc_host}, cos={cos_host:.6f}")
        logger.info(
            f"  Host argmax: ref={ref_logits_host[:, :vocab_size].argmax().item()}, "
            f"tt_via_host={tt_logits_host[:, :vocab_size].argmax().item()}"
        )

        logger.info(f"Token match: CPU={ref_token}, TTNN={tt_token}, match={ref_token == tt_token}")

        logger.info("=" * 60)
        logger.info(f"PREFILL {n_layers}L SUMMARY:")
        logger.info(f"  Hidden state PCC: {pcc_hidden_msg}")
        logger.info(f"  Logits PCC:       {pcc_logits_msg}")
        logger.info(f"  Logits cos sim:   {cos_sim:.6f}")
        logger.info(f"  Host logits PCC:  {pcc_host}")
        logger.info("=" * 60)

        assert passing_hidden, f"Prefill {n_layers}L hidden state PCC {pcc_hidden_msg} < {pcc_threshold}"
        logger.info(f"PREFILL {n_layers}L PCC TEST: PASSED")

    @torch.no_grad()
    def test_decode_pcc_64layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step with full 64-layer model, compare logits vs CPU."""
        self._run_decode_pcc(mesh_device, n_layers=64)

    @torch.no_grad()
    def test_decode_pcc_64layers_1k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=1023 (ISL=1k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=1023, max_seq_len=2048)

    @torch.no_grad()
    def test_decode_pcc_64layers_2k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=2047 (ISL=2k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=2047, max_seq_len=4096)

    @torch.no_grad()
    def test_decode_pcc_64layers_4k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=4095 (ISL=4k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=4095, max_seq_len=8192)

    @torch.no_grad()
    def test_decode_pcc_64layers_8k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=8191 (ISL=8k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=8191, max_seq_len=8192)

    @torch.no_grad()
    def test_decode_pcc_64layers_16k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=16383 (ISL=16k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=16383, max_seq_len=16384)

    @torch.no_grad()
    def test_decode_pcc_64layers_32k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=32767 (ISL=32k) with 64-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=32767, max_seq_len=32768)

    @torch.no_grad()
    def test_decode_pcc_64layers_64k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=65535 (ISL=64k) with 64-layer model.

        n_local_kv_heads=1 (8 KV heads / 8 devices_per_group), so KV cache per device at
        max_num_blocks=8192 is 8192*1*64*128*2*64 = 8GB ≤ 12GB DRAM. Safe with batch=32.
        """
        self._run_decode_pcc(mesh_device, n_layers=64, start_pos=65535, max_seq_len=65536)

    @torch.no_grad()
    def test_decode_pcc_32layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step, 32-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=32)

    @torch.no_grad()
    def test_decode_pcc_16layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step, 16-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=16)

    @torch.no_grad()
    def test_decode_pcc_8layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step, 8-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=8)

    @torch.no_grad()
    def test_decode_pcc_4layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step, 4-layer model to track per-depth degradation."""
        self._run_decode_pcc(mesh_device, n_layers=4)

    @torch.no_grad()
    def test_decode_pcc_4layers_1k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=1023 (ISL=1k) with 4-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=4, start_pos=1023, max_seq_len=2048)

    @torch.no_grad()
    def test_decode_pcc_4layers_2k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=2047 (ISL=2k) with 4-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=4, start_pos=2047, max_seq_len=4096)

    @torch.no_grad()
    def test_decode_pcc_4layers_4k(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step at start_pos=4095 (ISL=4k) with 4-layer model."""
        self._run_decode_pcc(mesh_device, n_layers=4, start_pos=4095, max_seq_len=8192)

    @torch.no_grad()
    def test_decode_pcc_2layers(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step, 2-layer model to track per-depth degradation."""
        self._run_decode_pcc(mesh_device, n_layers=2)

    @torch.no_grad()
    def test_decode_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC: 1 decode step after prefill, compare hidden states vs CPU."""
        self._run_decode_pcc(mesh_device, n_layers=1)

    @torch.no_grad()
    def test_decode_pcc_1layer_bf16weights(self, mesh_device, reset_seeds, ensure_gc):
        """Decode PCC, 1 layer, bf16 weights. Used to upper-bound achievable PCC vs default bf8."""
        self._run_decode_pcc(mesh_device, n_layers=1, dtype=ttnn.bfloat16)

    @torch.no_grad()
    def test_decode_pcc_1layer_1k(self, mesh_device, reset_seeds, ensure_gc):
        self._run_decode_pcc(mesh_device, n_layers=1, start_pos=1023, max_seq_len=2048)

    @torch.no_grad()
    def test_decode_pcc_1layer_4k(self, mesh_device, reset_seeds, ensure_gc):
        self._run_decode_pcc(mesh_device, n_layers=1, start_pos=4095, max_seq_len=8192)

    @torch.no_grad()
    def test_decode_pcc_1layer_16k(self, mesh_device, reset_seeds, ensure_gc):
        self._run_decode_pcc(mesh_device, n_layers=1, start_pos=16383, max_seq_len=16384)

    @torch.no_grad()
    def test_decode_pcc_1layer_32k(self, mesh_device, reset_seeds, ensure_gc):
        self._run_decode_pcc(mesh_device, n_layers=1, start_pos=32767, max_seq_len=32768)

    @torch.no_grad()
    def test_decode_per_op_pcc_4layers(self, mesh_device, reset_seeds, ensure_gc):
        """Per-op PCC: capture SDPA, attn_out, h_attn, ff_out, layer_out for each layer."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        # Use 1 layer for faster head-mapping debug; change to 4 for full test.
        # Override via N_LAYERS_PER_OP env var to run at full 64-layer depth.
        n_layers = int(os.environ.get("N_LAYERS_PER_OP", "1"))
        max_seq_len = 256
        batch_size = 32
        start_pos = 127
        dtype = ttnn.bfloat8_b

        hf_sd = load_hf_raw_state_dict(hf_model_path)
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, 1, 5120)
        embeddings_ref = pt_input.float()

        # ── Reference: capture per-layer intermediates by monkey-patching forward ──
        from models.demos.olmo_galaxy.reference.olmo import TransformerBlock

        ref_captures = [{} for _ in range(n_layers)]

        original_forward = TransformerBlock.forward

        from models.demos.olmo_galaxy.reference.olmo import Attention as RefAttention

        original_attn_forward = RefAttention.forward

        def hf_to_meta_qk(t, head_dim=128):
            """Convert Q/K from HF format to Meta/TTNN format (interleave real/imag per head).
            HF: [..., r0,r1,...,r63, i0,i1,...,i63] per head
            Meta: [..., r0,i0,r1,i1,...,r63,i63] per head"""
            orig_shape = t.shape  # [bsz, seqlen, total_dim]
            total_dim = orig_shape[-1]
            n_heads = total_dim // head_dim
            t = t.view(*orig_shape[:-1], n_heads, head_dim)
            reals = t[..., : head_dim // 2]
            imags = t[..., head_dim // 2 :]
            interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=-2)
            return interleaved.view(orig_shape)

        def patched_attn_forward(self_attn, x, start_pos, freqs_cis, mask):
            li = self_attn.layer_id
            import torch.nn.functional as F

            bsz, seqlen, _ = x.shape
            xq = self_attn.wq(x)
            xk = self_attn.wk(x)
            xv = self_attn.wv(x)

            def _global_rms_norm(t, weight):
                t_f = t.float()
                rms = torch.rsqrt(t_f.pow(2).mean(-1, keepdim=True) + 1e-6)
                return (t_f * rms * weight.to(t_f.device)).type_as(t)

            # Store both HF-format and Meta-format (TTNN uses Meta format due to reverse_permute)
            ref_captures[li]["ref_xq_raw"] = hf_to_meta_qk(xq).detach().clone()
            ref_captures[li]["ref_xk_raw"] = hf_to_meta_qk(xk).detach().clone()
            ref_captures[li]["ref_xv_raw"] = xv.detach().clone()  # V is not permuted

            # Apply norm in Meta format to match TTNN norm
            xq_meta = hf_to_meta_qk(xq)
            xk_meta = hf_to_meta_qk(xk)
            q_norm_meta = hf_to_meta_qk(self_attn.q_norm_weight.unsqueeze(0), head_dim=128).squeeze(0)
            k_norm_meta = hf_to_meta_qk(self_attn.k_norm_weight.unsqueeze(0), head_dim=128).squeeze(0)
            xq_n = _global_rms_norm(xq_meta, q_norm_meta)
            xk_n = _global_rms_norm(xk_meta, k_norm_meta)
            ref_captures[li]["xq_normed_std"] = xq_n.std().item()
            ref_captures[li]["xk_normed_std"] = xk_n.std().item()
            ref_captures[li]["xv_std"] = xv.std().item()
            ref_captures[li]["ref_xq_normed"] = xq_n.detach().clone()  # [bsz, 1, 5120]
            ref_captures[li]["ref_xk_normed"] = xk_n.detach().clone()  # [bsz, 1, 1024]
            from models.demos.olmo_galaxy.reference.olmo import apply_rotary_emb, repeat_kv

            xq_h = xq_n.view(bsz, seqlen, self_attn.n_heads, self_attn.head_dim)
            xk_h = xk_n.view(bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim)
            xv_h = xv.view(bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim)
            xq_r, xk_r = apply_rotary_emb(xq_h, xk_h, freqs_cis)
            ref_captures[li]["ref_xq_post_rope"] = xq_r.detach().clone()  # [bsz, 1, n_heads, head_dim]
            ref_captures[li]["ref_xk_post_rope"] = xk_r.detach().clone()  # [bsz, 1, n_kv_heads, head_dim]
            self_attn.cache_k[:bsz, start_pos : start_pos + seqlen] = xk_r
            self_attn.cache_v[:bsz, start_pos : start_pos + seqlen] = xv_h
            keys = self_attn.cache_k[:bsz, : start_pos + seqlen]
            values = self_attn.cache_v[:bsz, : start_pos + seqlen]
            keys = repeat_kv(keys, self_attn.n_rep)
            values = repeat_kv(values, self_attn.n_rep)
            xq_t = xq_r.transpose(1, 2)
            keys_t = keys.transpose(1, 2)
            values_t = values.transpose(1, 2)
            scale = (self_attn.head_dim**-0.5) * self_attn.mscale
            scores = torch.matmul(xq_t, keys_t.transpose(2, 3)) * scale
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq_t)
            sdpa_raw = torch.matmul(attn_weights, values_t)  # [bsz, n_heads, 1, head_dim]
            sdpa_out = sdpa_raw.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            ref_captures[li]["ref_sdpa_out_std"] = sdpa_out.std().item()
            ref_captures[li]["ref_sdpa_weights_max"] = attn_weights.max().item()
            ref_captures[li]["ref_sdpa_out"] = sdpa_out.detach().clone()  # [bsz, 1, n_heads*head_dim]
            wo_out = self_attn.wo(sdpa_out)
            ref_captures[li]["ref_wo_out"] = wo_out.detach().clone()  # [bsz, 1, dim]
            return original_attn_forward(self_attn, x, start_pos, freqs_cis, mask)

        RefAttention.forward = patched_attn_forward

        def patched_forward(self_block, x, start_pos, freqs_cis, mask):
            li = self_block.layer_id
            ref_captures[li]["layer_in"] = x.detach().clone()
            attn_raw = self_block.attention(x, start_pos, freqs_cis, mask)
            ref_captures[li]["attn_out"] = attn_raw.detach().clone()
            attn_normed = self_block.attention_norm(attn_raw)
            ref_captures[li]["attn_normed"] = attn_normed.detach().clone()
            h = x + attn_normed
            ref_captures[li]["h_attn"] = h.detach().clone()
            # Inline the FFN so we can capture w1/w3/ff1ff3/w2 separately.
            ff = self_block.feed_forward
            w1_out = ff.w1(h)
            w3_out = ff.w3(h)
            ref_captures[li]["w1_out"] = w1_out.detach().clone()
            ref_captures[li]["w3_out"] = w3_out.detach().clone()
            ff1ff3 = F.silu(w1_out) * w3_out
            ref_captures[li]["ff1ff3"] = ff1ff3.detach().clone()
            ff_raw = ff.w2(ff1ff3)
            ref_captures[li]["w2_out_pre_ar"] = ff_raw.detach().clone()
            ref_captures[li]["ff_out"] = ff_raw.detach().clone()
            ff_normed = self_block.ffn_norm(ff_raw)
            ref_captures[li]["ff_normed"] = ff_normed.detach().clone()
            out = h + ff_normed
            ref_captures[li]["layer_out"] = out.detach().clone()
            return out

        TransformerBlock.forward = patched_forward
        ref_model.forward(embeddings_ref, start_pos=start_pos, mode="decode")
        TransformerBlock.forward = original_forward
        RefAttention.forward = original_attn_forward

        # ── TTNN model ──
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()

        # 64L overflows DRAM at max_num_blocks=4096; shrink when running deep stacks.
        per_op_blocks = 128 if n_layers >= 32 else 4096
        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=per_op_blocks)
        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            decode_mode_only=True,
        )

        # Enable capture on every layer
        for layer in tt_model.layers:
            layer.capture_intermediates = True
            layer.attention.capture_intermediates = True
            layer.feed_forward.capture_intermediates = True

        page_table = torch.argsort(torch.randperm(paged_attention_config.max_num_blocks)).reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
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
        rot_mats, _ = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

        tt_model.forward(
            tt_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
            kv_cache=[layer.attention.layer_past for layer in tt_model.layers],
        )
        ttnn.synchronize_device(mesh_device)

        # ── Compare per-layer per-op PCC ──
        # TTNN captured tensors have shape [1, 4, 32, 10240] (from ConcatMesh2dToTensor dims=(3,1))
        # Reference tensors have shape [32, 1, 5120] (batch, seq, dim)
        # Extract user-0 hidden state from each: ref[0, 0, :] = [5120], tt[0, 0, 0, :5120] = [5120]
        logger.info("=" * 70)
        logger.info("Per-op PCC breakdown (4 layers, decode step, user 0)")
        logger.info("=" * 70)

        # TTNN captured tensors shape from decoder (ConcatMesh2dToTensor dims=(3,1), cluster=[8,4]):
        #   [1, 4, 32, 10240]
        #   dim 1 = 4  (4 col groups, each holding a 1280-dim slice of model_dim=5120)
        #   dim 2 = 32 (batch, replicated across all devices)
        #   dim 3 = 10240 = 8×1280  (8 row devices all hold the same 1280-dim slice)
        # To reconstruct full [5120] hidden state for user i:
        #   cat([tt[0, j, i, :1280] for j in range(4)])  → [5120]
        #
        # TTNN attn captures (ConcatMesh2dToTensor dims=(1,3), cluster=[8,4]):
        #   [1, 8, 32, 5120]
        #   dim 1 = 8  (8 row devices, all same data → replicated)
        #   dim 3 = 5120 (4 col devices × 1280 = full model dim)
        # User i full state: attn_t[0, 0, i, :]  → [5120]

        def extract_decoder_u0(tt_t):
            """Reconstruct user-0 full hidden state from decoder captured tensor [1,4,32,10240]."""
            n_col = tt_t.shape[1]  # 4 col groups
            slice_dim = tt_t.shape[3] // 8  # 1280 (each row-device replicated 8×, take first 1×)
            return torch.cat([tt_t[0, j, 0, :slice_dim] for j in range(n_col)], dim=0)  # [5120]

        def extract_attn_u0(tt_t):
            """Reconstruct user-0 full hidden state from attention captured tensor [1,8,32,5120]."""
            return tt_t[0, 0, 0, :]  # first row-device (all 8 same), user 0, full [5120]

        keys = ["attn_out", "attn_normed", "h_attn", "ff_out", "ff_normed", "layer_out"]
        for li in range(n_layers):
            tt_layer = tt_model.layers[li]
            logger.info(f"  Layer {li}:")
            for key in keys:
                ref_t = ref_captures[li].get(key)
                tt_t = tt_layer.captured.get(key)
                if ref_t is None or tt_t is None:
                    logger.warning(f"    {key}: MISSING (ref={ref_t is None}, tt={tt_t is None})")
                    continue
                # ref_t shape: [32, 1, 5120] → user 0: [5120]
                ref_u0 = ref_t[0].flatten().float()
                tt_u0 = extract_decoder_u0(tt_t)
                pcc_val, pcc_msg = comp_pcc(ref_u0.unsqueeze(0), tt_u0.unsqueeze(0), 0.0)
                logger.info(
                    f"    {key:12s}: PCC={pcc_msg}  "
                    f"ref_mean={ref_u0.mean():.4f} tt_mean={tt_u0.mean():.4f}  "
                    f"ref_std={ref_u0.std():.4f} tt_std={tt_u0.std():.4f}"
                )
            # ─── Per-block PCC comparison for attention path ───
            attn_caps = tt_layer.attention.captured
            ref_li = ref_captures[li]

            def _pcc_log(label, ref_t, tt_t, ref_label="ref", tt_label="tt"):
                """Compute and log PCC for a pair of tensors."""
                pcc_val, pcc_msg = comp_pcc(ref_t.unsqueeze(0), tt_t.unsqueeze(0), 0.0)
                logger.info(
                    f"    {label:20s}: PCC={pcc_msg}  "
                    f"{ref_label}_std={ref_t.std():.4f} {tt_label}_std={tt_t.std():.4f}  "
                    f"{ref_label}_mean={ref_t.mean():.4f} {tt_label}_mean={tt_t.mean():.4f}"
                )
                return pcc_val

            # ── Block-level PCC: compare each capture with try/except ──
            # Log all shapes first, then compare what we can
            cap_names = [
                "q_post_norm",
                "k_post_norm",
                "q_post_rope",
                "k_post_rope_sliced",
                "k_expanded_post_rope",
                "sdpa_out",
                "wo_input",
                "attn_out_final",
            ]
            for cn in cap_names:
                t = attn_caps.get(cn)
                if t is not None:
                    logger.info(f"    TT {cn:22s}: shape={list(t.shape)}, std={t.std():.4f}, mean={t.mean():.4f}")

            ref_names = [
                "ref_xq_normed",
                "ref_xk_normed",
                "ref_xq_post_rope",
                "ref_xk_post_rope",
                "ref_sdpa_out",
                "ref_wo_out",
            ]
            for rn in ref_names:
                t = ref_li.get(rn)
                if t is not None:
                    logger.info(f"    REF {rn:21s}: shape={list(t.shape)}, std={t.std():.4f}, mean={t.mean():.4f}")

            # Per-device PCC: compare row-0 col-0 device portion only
            # After ConcatMesh2dToTensor(dims=(1,3)): dim1 = 8 rows concat, dim3 = 4 cols concat
            # Row-0 data: dim1 slice [0:per_row_d1]. Col-0 data: dim3 slice [0:per_col_d3].
            def safe_pcc(label, ref_flat, tt_flat):
                min_len = min(len(ref_flat), len(tt_flat))
                if min_len == 0:
                    logger.warning(f"    {label:20s}: empty tensor")
                    return
                ref_f = ref_flat[:min_len].unsqueeze(0)
                tt_f = tt_flat[:min_len].unsqueeze(0)
                try:
                    _, pcc_msg = comp_pcc(ref_f, tt_f, 0.0)
                    logger.info(
                        f"    {label:20s}: PCC={pcc_msg}  len={min_len}  "
                        f"ref_std={ref_f.std():.4f} tt_std={tt_f.std():.4f}"
                    )
                except Exception as e:
                    logger.error(f"    {label:20s}: comp_pcc failed: {e}")

            # ── Extraction helper ──
            # dims=(0,1) concat: axis0(8 rows)→dim0, axis1(4 cols)→dim1
            # Per-device shapes (from ttnn_shape log):
            #   Q: [1, 8, 8, 128] → batch in dim1, heads in dim2
            #   K: [1, 8, 1, 128] → batch in dim1, 1 KV head in dim2
            #   V: [1, 8, 1, 128] → batch in dim1, 1 KV head in dim2
            # After concat(dims=(0,1)):
            #   Q: [8, 32, 8, 128] → dim0=rows, dim1=4cols×8batch, dim2=8 padded heads
            #   K: [8, 32, 1, 128]
            #   V: [8, 32, 1, 128]
            # User 0 (col 0, batch 0): dim1=0. Head h: dim2=h.
            # So: tensor[row, 0, h, :] = user 0, head h, row device r
            n_heads_q_real = 5

            def reconstruct_q_user0(cap_t):
                """Extract user 0's full Q (40 heads × 128) from [8, 32, 8, 128]."""
                pieces = []
                for row in range(8):
                    for h in range(n_heads_q_real):
                        pieces.append(cap_t[row, 0, h, :].flatten().float())
                return torch.cat(pieces, dim=0)  # [5120]

            def reconstruct_kv_user0(cap_t):
                """Extract user 0's full K or V (8 heads × 128) from [8, 32, 1, 128]."""
                pieces = []
                for row in range(8):
                    pieces.append(cap_t[row, 0, 0, :].flatten().float())
                return torch.cat(pieces, dim=0)  # [1024]

            # ── Q pre-norm ──
            q_pre = attn_caps.get("q_pre_norm")
            ref_q_raw = ref_li.get("ref_xq_raw")
            if q_pre is not None and ref_q_raw is not None:
                logger.info(f"    Q_pre_norm shape={list(q_pre.shape)}")
                tt_q_full = reconstruct_q_user0(q_pre)
                ref_q_full = ref_q_raw[0, 0, :5120].float()
                safe_pcc("Q_pre_norm(full 5120)", ref_q_full, tt_q_full)
                for h in range(n_heads_q_real):
                    tt_h = q_pre[0, 0, h, :].flatten().float()
                    ref_h = ref_q_raw[0, 0, h * 128 : (h + 1) * 128].float()
                    try:
                        _, pm = comp_pcc(ref_h.unsqueeze(0), tt_h.unsqueeze(0), 0.0)
                        pcc_v = float(pm)
                    except Exception:
                        pcc_v = float("nan")
                    logger.info(f"      Q row0 head{h}: PCC={pcc_v:.4f}")

            # ── K pre-norm ──
            k_pre = attn_caps.get("k_pre_norm")
            ref_k_raw = ref_li.get("ref_xk_raw")
            if k_pre is not None and ref_k_raw is not None:
                tt_k_full = reconstruct_kv_user0(k_pre)
                ref_k_full = ref_k_raw[0, 0, :1024].float()
                safe_pcc("K_pre_norm(full 1024)", ref_k_full, tt_k_full)

            # ── V ──
            v_cap = attn_caps.get("v_heads")
            ref_v_raw = ref_li.get("ref_xv_raw")
            if v_cap is not None and ref_v_raw is not None:
                tt_v_full = reconstruct_kv_user0(v_cap)
                ref_v_full = ref_v_raw[0, 0, :1024].float()
                safe_pcc("V_heads(full 1024)", ref_v_full, tt_v_full)

            # ── Q post-norm ──
            q_pn = attn_caps.get("q_post_norm")
            ref_qn = ref_li.get("ref_xq_normed")
            if q_pn is not None and ref_qn is not None:
                tt_qn_full = reconstruct_q_user0(q_pn)
                ref_qn_full = ref_qn[0, 0, :5120].float()
                safe_pcc("Q_post_norm(full 5120)", ref_qn_full, tt_qn_full)

            # ── K post-norm ──
            k_pn = attn_caps.get("k_post_norm")
            ref_kn = ref_li.get("ref_xk_normed")
            if k_pn is not None and ref_kn is not None:
                tt_kn_full = reconstruct_kv_user0(k_pn)
                ref_kn_full = ref_kn[0, 0, :1024].float()
                safe_pcc("K_post_norm(full 1024)", ref_kn_full, tt_kn_full)

            import torch.nn.functional as F_nn

            # ── CORRECTED per-head comparisons ──
            # After ConcatMesh2dToTensor(dims=(0,1)) with cluster=(8,4):
            #   Q/K/V captures: [8_rows, 32_batchcols, n_heads, head_dim=128]
            #   row r = row device r (= KV head r for K/V, holds Q heads r*5..(r+1)*5-1 for Q)
            #   col 0 = first column device, batch positions 0-7 → user 0 is at batch index 0
            #   head_dim = 128 FULL (NOT split across col devices for Q/K/V captures)
            # Q post-rope: [8, 32, 8, 128]  (8 rows, 32 batch, 8 padded Q heads, 128 dim)
            q_pr = attn_caps.get("q_post_rope")
            ref_qr = ref_li.get("ref_xq_post_rope")  # [bsz=32, 1, 40, 128]
            if q_pr is not None and ref_qr is not None:
                # Full 5120-dim Q for user 0: cat 8 row devices × 5 real Q heads × 128
                tt_q_full = torch.cat([q_pr[r, 0, h, :] for r in range(8) for h in range(5)], dim=0).float()
                ref_q_full = ref_qr[0, 0, :40, :].reshape(-1).float()
                safe_pcc("Q_post_rope(full 5120)", ref_q_full, tt_q_full)
                # Per-head cosine similarity (row 0 only, heads 0-4)
                for h in range(5):
                    tt_h = q_pr[0, 0, h, :].float()
                    ref_h = ref_qr[0, 0, h, :].float()
                    if tt_h.std() < 1e-6 or ref_h.std() < 1e-6:
                        logger.info(f"      Q head{h}: NEAR-ZERO tt_std={tt_h.std():.4e} ref_std={ref_h.std():.4e}")
                    else:
                        cos_sim = F_nn.cosine_similarity(tt_h.unsqueeze(0), ref_h.unsqueeze(0)).item()
                        max_err = (tt_h - ref_h).abs().max().item()
                        logger.info(f"      Q head{h}: cos_sim={cos_sim:.4f} max_err={max_err:.4f}")

            # K post-rope sliced: [8, 32, 1, 128]  (8 rows, 32 batch, 1 KV head, 128 dim)
            k_ps = attn_caps.get("k_post_rope_sliced")
            ref_kr = ref_li.get("ref_xk_post_rope")  # [bsz=32, 1, 8, 128]
            if k_ps is not None and ref_kr is not None:
                # Full 1024-dim K for user 0: cat 8 row devices × 1 head × 128
                tt_k_full = torch.cat([k_ps[r, 0, 0, :] for r in range(8)], dim=0).float()
                ref_k_full = ref_kr[0, 0, :, :].reshape(-1).float()
                safe_pcc("K_post_rope(full 1024)", ref_k_full, tt_k_full)
                # Per-head cosine similarity
                for r in range(8):
                    tt_k = k_ps[r, 0, 0, :].float()
                    ref_k = ref_kr[0, 0, r, :].float()
                    if tt_k.std() < 1e-6 or ref_k.std() < 1e-6:
                        logger.info(f"      K head{r}: NEAR-ZERO tt_std={tt_k.std():.4e} ref_std={ref_k.std():.4e}")
                    else:
                        cos_sim = F_nn.cosine_similarity(tt_k.unsqueeze(0), ref_k.unsqueeze(0)).item()
                        max_err = (tt_k - ref_k).abs().max().item()
                        logger.info(f"      K head{r}: cos_sim={cos_sim:.4f} max_err={max_err:.4f}")

            # V: [8, 32, 1, 128]  (8 rows, 32 batch, 1 V head, 128 dim)
            v_cap = attn_caps.get("v_from_create_heads")
            ref_v_raw = ref_li.get("ref_xv_raw")  # [bsz=32, 1, 1024]
            if v_cap is not None and ref_v_raw is not None:
                tt_v_full = torch.cat([v_cap[r, 0, 0, :] for r in range(8)], dim=0).float()
                ref_v_full = ref_v_raw[0, 0, :1024].float()
                safe_pcc("V_pre_cache(full 1024)", ref_v_full, tt_v_full)

            # K expanded post-rope
            k_exp = attn_caps.get("k_expanded_post_rope")
            if k_exp is not None:
                logger.info(f"    K_exp_post_rope   : shape={list(k_exp.shape)}, std={k_exp.std():.4f}")

            # ── KV Cache diagnostic: read K at position start_pos (127) from actual paged cache ──
            # This directly answers: did paged_update_cache write K to the right place?
            logger.info(f"  --- KV Cache @ pos {start_pos} diagnostic (layer {li}) ---")
            block_size_cfg = paged_attention_config.block_size  # 64
            page_for_pos = start_pos // block_size_cfg  # 1
            offset_in_block = start_pos % block_size_cfg  # 63
            block_for_u0 = int(page_table[0, page_for_pos])
            logger.info(
                f"  user0 pos={start_pos} -> page_idx={page_for_pos}, "
                f"physical_block={block_for_u0}, offset={offset_in_block}"
            )
            keys_layer_li = tt_model.layers[li].attention.layer_past[0]
            vals_layer_li = tt_model.layers[li].attention.layer_past[1]
            # Slice just the target block from the paged KV cache (avoids reading all 4096 blocks)
            try:
                k_block_slice = ttnn.slice(
                    keys_layer_li,
                    [block_for_u0, 0, 0, 0],
                    [block_for_u0 + 1, 1, block_size_cfg, 128],
                )
                v_block_slice = ttnn.slice(
                    vals_layer_li,
                    [block_for_u0, 0, 0, 0],
                    [block_for_u0 + 1, 1, block_size_cfg, 128],
                )
                k_block_cpu = ttnn.to_torch(
                    k_block_slice,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape
                    ),
                ).float()
                v_block_cpu = ttnn.to_torch(
                    v_block_slice,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape
                    ),
                ).float()
                ttnn.deallocate(k_block_slice)
                ttnn.deallocate(v_block_slice)
                # k_block_cpu shape: [8, 4, 64, 128]  (8 row-devs × 1, 4 col-devs × 1, 64 pos, 128 dim)
                # K at offset_in_block for user 0 (col device 0):
                k_at_pos = k_block_cpu[:, 0, offset_in_block, :]  # [8, 128]
                v_at_pos = v_block_cpu[:, 0, offset_in_block, :]  # [8, 128]
                logger.info(f"  K_cache_at_{start_pos}: max_abs={k_at_pos.abs().max():.4f}  std={k_at_pos.std():.4f}")
                logger.info(f"  V_cache_at_{start_pos}: max_abs={v_at_pos.abs().max():.4f}  std={v_at_pos.std():.4f}")

                # TT stores K in Meta-interleaved layout [r0,i0,r1,i1,...,r63,i63] while ref
                # uses HF split-half [r0,...,r63,i0,...,i63]. Convert TT→HF before compare so
                # cos_sim measures actual numeric agreement, not layout difference.
                def _tt_to_hf_layout(tt_k_interleaved):
                    even = tt_k_interleaved[..., 0::2]  # [r0, r1, ..., r63]
                    odd = tt_k_interleaved[..., 1::2]  # [i0, i1, ..., i63]
                    return torch.cat([even, odd], dim=-1)

                if ref_kr is not None:
                    for r in range(8):
                        tt_k_raw = k_at_pos[r]
                        tt_k = _tt_to_hf_layout(tt_k_raw)
                        ref_k = ref_kr[0, 0, r, :].float()
                        if tt_k.abs().max() < 1e-4:
                            logger.info(f"    K_cache head{r}: <<ZERO>> write FAILED!")
                        else:
                            cos_sim = F_nn.cosine_similarity(tt_k.unsqueeze(0), ref_k.unsqueeze(0)).item()
                            max_err = (tt_k - ref_k).abs().max().item()
                            ratio = tt_k.std().item() / max(ref_k.std().item(), 1e-9)
                            logger.info(
                                f"    K_cache head{r}: cos_sim={cos_sim:.4f} max_err={max_err:.4f} "
                                f"std_ratio={ratio:.4f}  tt_std={tt_k.std():.4f} ref_std={ref_k.std():.4f}"
                            )
                if v_cap is not None and ref_v_raw is not None:
                    for r in range(8):
                        tt_v = v_at_pos[r]
                        ref_v_h = ref_v_raw[0, 0, r * 128 : (r + 1) * 128].float()
                        if tt_v.abs().max() < 1e-4:
                            logger.info(f"    V_cache head{r}: <<ZERO>> write FAILED!")
                        else:
                            cos_sim = F_nn.cosine_similarity(tt_v.unsqueeze(0), ref_v_h.unsqueeze(0)).item()
                            max_err = (tt_v - ref_v_h).abs().max().item()
                            logger.info(f"    V_cache head{r}: cos_sim={cos_sim:.4f} max_err={max_err:.4f}")
            except Exception as e:
                logger.warning(f"  KV cache slice diagnostic failed: {e}")

            # ── SDPA output comparison (corrected: full head_dim=128, not /4) ──
            # sdpa_out per device: [1, 8, 32, 128]  → after concat(0,1): [8, 32, 32, 128]
            #   dim0=8 rows, dim1=32 (8 padded heads × 4 col-devs), dim2=32 batch, dim3=128
            # Wait: per-device sdpa shape is [1, n_heads_padded=8, batch_all=32, head_dim=128]
            # After concat dims=(0,1): [8, 8*4=32, 32, 128] → sdpa_tt[row, head_col, batch, dim]
            # For row device 0, Q heads 0-4 (real), user 0, full head_dim:
            sdpa_tt = attn_caps.get("sdpa_out")
            ref_sdpa_t = ref_li.get("ref_sdpa_out")  # [32, 1, 5120]
            if sdpa_tt is not None and ref_sdpa_t is not None:
                logger.info(
                    f"  SDPA diagnostic: sdpa_tt.shape={list(sdpa_tt.shape)}  ref.shape={list(ref_sdpa_t.shape)}"
                )
                # sdpa_tt[row_dev, user, q_head_padded, head_dim]
                # Correct: user 0 (dim1=0), real Q heads 0-4 per row device (dim2=0-4)
                tt_s_full = torch.cat([sdpa_tt[r, 0, h, :] for r in range(8) for h in range(5)], dim=0).float()
                ref_s_full = ref_sdpa_t[0, 0, :5120].float()
                safe_pcc("SDPA_out(full 5120)", ref_s_full, tt_s_full)
                logger.info(
                    f"    SDPA tt_std={tt_s_full.std():.4f}  ref_std={ref_s_full.std():.4f}  "
                    f"tt_max={tt_s_full.abs().max():.4f}  ref_max={ref_s_full.abs().max():.4f}"
                )
                # Per-head comparison to find which Q head has high/low attention
                logger.info("    Per-head SDPA check (user=0, Q heads 0-4, row=0):")
                for h in range(5):
                    tt_sh = sdpa_tt[0, 0, h, :].float()
                    ref_sh = ref_sdpa_t[0, 0, h * 128 : (h + 1) * 128].float()
                    cos_sh = F_nn.cosine_similarity(tt_sh.unsqueeze(0), ref_sh.unsqueeze(0)).item()
                    ratio_sh = tt_sh.norm() / (ref_sh.norm() + 1e-8)
                    logger.info(
                        f"      sdpa head{h}: cos_sim={cos_sh:.4f}  tt_std={tt_sh.std():.4f}  ref_std={ref_sh.std():.4f}  norm_ratio={ratio_sh:.4f}"
                    )
                # Compare SDPA vs V[pos=127] to check if attention concentrates on the right position
                if v_cap is not None:
                    scale_val = float(tt_model.layers[li].attention.scale)
                    for r in range(min(2, 8)):
                        tt_sdpa_r = sdpa_tt[r, 0, 0, :].float()  # row r, user 0, Q head 0
                        v_r = v_cap[r, 0, 0, :].float()  # V head r for user 0
                        ratio = tt_sdpa_r.norm() / (v_r.norm() + 1e-8)
                        cos_vs_v = F_nn.cosine_similarity(tt_sdpa_r.unsqueeze(0), v_r.unsqueeze(0)).item()
                        logger.info(
                            f"    SDPA row{r} Q-head0 vs V_head{r}: "
                            f"cos_sim={cos_vs_v:.4f}  norm_ratio={ratio:.4f} "
                            f"(expect ~1.0 if attn concentrates on pos {start_pos})"
                        )
                    # Direct Q × K^T score computation to diagnose SDPA uniform weights
                    try:
                        logger.info("  --- Direct Q×K^T score check ---")
                        # Also compute reference Q×K score directly for comparison
                        ref_q_all = ref_li.get("ref_xq_post_rope")  # [32, 1, 40, 128]
                        ref_k_all = ref_li.get("ref_xk_post_rope")  # [32, 1, 8, 128]
                        for r in range(min(3, 8)):
                            q_r = q_pr[r, 0, 0, :].float()  # Q head 0, user 0, row r
                            k_r = k_at_pos[r, :].float()  # K head r at pos 127, user 0
                            manual_score = (q_r @ k_r).item() * scale_val
                            # Reference score: user 0, Q head r*5 (first Q head of row r), KV head r
                            ref_q_h = ref_q_all[0, 0, r * 5, :].float()  # user 0, Q head r*5
                            ref_k_h = ref_k_all[0, 0, r, :].float()  # user 0, KV head r
                            ref_score = (ref_q_h @ ref_k_h).item() * scale_val
                            logger.info(
                                f"    row{r}: TTNN Q@K[127]*scale={manual_score:.4f}  "
                                f"REF Q@K[127]*scale={ref_score:.4f}  "
                                f"Q_norm={q_r.norm():.4f}  K_norm={k_r.norm():.4f}  "
                                f"ref_Q_norm={ref_q_h.norm():.4f}  ref_K_norm={ref_k_h.norm():.4f}"
                            )
                        # Show max-score head across all 5 Q heads and row devices for reference
                        max_ref_score = -1e9
                        max_ref_idx = (-1, -1)
                        for r2 in range(8):
                            k_r2 = k_at_pos[r2, :].float()
                            for h2 in range(5):
                                q_r2 = q_pr[r2, 0, h2, :].float()
                                ref_q_r2 = ref_q_all[0, 0, r2 * 5 + h2, :].float()
                                ref_k_r2 = ref_k_all[0, 0, r2, :].float()
                                ref_sc2 = (ref_q_r2 @ ref_k_r2).item() * scale_val
                                if ref_sc2 > max_ref_score:
                                    max_ref_score = ref_sc2
                                    max_ref_idx = (r2, h2)
                        r_max, h_max = max_ref_idx
                        ttnn_q_max = q_pr[r_max, 0, h_max, :].float()
                        ref_q_max = ref_q_all[0, 0, r_max * 5 + h_max, :].float()
                        ref_k_max = ref_k_all[0, 0, r_max, :].float()
                        k_max = k_at_pos[r_max, :].float()
                        ttnn_score_max = (ttnn_q_max @ k_max).item() * scale_val
                        ref_score_max = (ref_q_max @ ref_k_max).item() * scale_val
                        logger.info(
                            f"  Max-score head: row={r_max} qhead={h_max}  "
                            f"REF_score={ref_score_max:.4f}  TTNN_score={ttnn_score_max:.4f}  "
                            f"TTNN_Q_norm={ttnn_q_max.norm():.4f}  ref_Q_norm={ref_q_max.norm():.4f}"
                        )
                        # Also check K at position 32 (wrong paged_update_cache offset hypothesis)
                        # If paged_update_cache wrote at tile-row 0 (seq 0 within block) instead of
                        # offset_in_block=63, we'd find K at position 32 (=global 96) non-zero
                        logger.info(f"  --- K at offset 32 within block (global {64+32}) ---")
                        k_at_32 = k_block_cpu[:, 0, 32, :]
                        logger.info(
                            f"  K_cache_at_32: max_abs={k_at_32.abs().max():.4f} std={k_at_32.std():.4f} "
                            f"(non-zero → paged_update_cache wrote at wrong tile row!)"
                        )
                        if k_at_32.abs().max() > 1e-3 and ref_kr is not None:
                            logger.info(
                                "  >>> K at position 32 is NON-ZERO: paged_update_cache wrote to wrong position!"
                            )
                            for r in range(3):
                                cos_32 = F_nn.cosine_similarity(
                                    k_at_32[r].float().unsqueeze(0), ref_kr[0, 0, r, :].float().unsqueeze(0)
                                ).item()
                                logger.info(f"    K_cache_at32 head{r}: cos_sim_vs_ref={cos_32:.4f}")
                    except Exception as e2:
                        logger.warning(f"  Direct score/K@32 check failed: {e2}")

            # wo input: [1, 1, batch, 640] per device → after concat(0,1): [8, 4, 32, 640]
            # WO input = SDPA output reshaped: row r, col c, user 0 = 5 real Q heads × 128
            # Full WO input for user 0 = concat 8 row devs × 5 heads × 128 = 5120 dims
            # ref: ref_sdpa_out[0, 0, :] = [5120] (user 0, all Q heads)
            wo_tt = attn_caps.get("wo_input")
            if wo_tt is not None and ref_sdpa_t is not None:
                # wo_input is [8, 4, 32, 640] after concat: [row, col, batch, 5_heads*128]
                # Full user 0 WO input = 8 row devs × 640 = 5120 (concat row devs)
                tt_wo_full = torch.cat([wo_tt[r, 0, 0, :] for r in range(8)], dim=0).float()  # [5120]
                ref_wo_f_full = ref_sdpa_t[0, 0, :].float()  # [5120]
                safe_pcc("wo_input(full 5120)", ref_wo_f_full, tt_wo_full)

            # attn_out_final: shape [8, 4, 32, 1280] = [row, col, batch, col_dim]
            # After line_all_reduce all row devices have same values, concat 4 col devs
            attn_final = attn_caps.get("attn_out_final")
            ref_wo_out = ref_li.get("ref_wo_out")
            if attn_final is not None and ref_wo_out is not None:
                tt_af = torch.cat([attn_final[0, c, 0, :] for c in range(4)], dim=0).float()  # [5120]
                ref_af = ref_wo_out[0].flatten().float()  # [5120]
                safe_pcc("attn_out_final", ref_af, tt_af)
            elif attn_final is not None and "attn_out" in ref_li:
                ref_attn = ref_li["attn_out"][0].flatten().float()
                tt_attn = torch.cat([attn_final[0, c, 0, :] for c in range(4)], dim=0).float()
                safe_pcc("attn_out_final", ref_attn, tt_attn)

            # Summary stats
            ref_xq_std = ref_li.get("xq_normed_std", float("nan"))
            ref_xk_std = ref_li.get("xk_normed_std", float("nan"))
            ref_xv_std = ref_li.get("xv_std", float("nan"))
            ref_sdpa_std = ref_li.get("ref_sdpa_out_std", float("nan"))
            ref_attn_max = ref_li.get("ref_sdpa_weights_max", float("nan"))
            logger.info(
                f"    REF STATS: xq_norm_std={ref_xq_std:.4f}  xk_norm_std={ref_xk_std:.4f}  "
                f"xv_std={ref_xv_std:.4f}  sdpa_out_std={ref_sdpa_std:.4f}  attn_weight_max={ref_attn_max:.4f}"
            )

            # ── Attention-internal element-wise comparison: TT vs ref ──
            def _ratio_block(label, ref_n, tt_n):
                """Print elementwise diff + |TT/ref| median ratio for the magnitude-bias hunt."""
                if ref_n.numel() != tt_n.numel():
                    n = min(ref_n.numel(), tt_n.numel())
                    ref_n = ref_n[:n]
                    tt_n = tt_n[:n]
                diff = (tt_n - ref_n).abs()
                mask = ref_n.abs() > 1e-3
                if mask.any():
                    ratio = tt_n[mask].abs() / ref_n[mask].abs()
                    r_med = ratio.median().item()
                    r_mean = ratio.mean().item()
                else:
                    r_med = r_mean = float("nan")
                logger.info(
                    f"    ATTN {label:18s}: n={ref_n.numel()}  abs_diff max={diff.max():.4e} "
                    f"mean={diff.mean():.4e}  |TT/ref| mean={r_mean:.4f} med={r_med:.4f}  "
                    f"ref_std={ref_n.std():.4f} tt_std={tt_n.std():.4f}"
                )

            sdpa_tt = attn_caps.get("sdpa_out")
            ref_sdpa_t = ref_li.get("ref_sdpa_out")
            if sdpa_tt is not None and ref_sdpa_t is not None:
                # TT shape [8 rows, 32 batch, 8 heads/dev (5 real + 3 pad), 128 head_dim].
                # User 0 full = 8 rows × 5 real heads × 128 = 5120
                tt_sdpa_full = torch.cat([sdpa_tt[r, 0, h, :] for r in range(8) for h in range(5)], dim=0).float()
                ref_sdpa_full = ref_sdpa_t[0, 0, :].float().flatten()
                _ratio_block("sdpa_out", ref_sdpa_full, tt_sdpa_full)

            wo_tt = attn_caps.get("wo_input")
            if wo_tt is not None and ref_sdpa_t is not None:
                tt_wo_full = torch.cat([wo_tt[r, 0, 0, :] for r in range(8)], dim=0).float()
                ref_wo_in_full = ref_sdpa_t[0, 0, :].float().flatten()
                _ratio_block("wo_input", ref_wo_in_full, tt_wo_full)

            attn_final = attn_caps.get("attn_out_final")
            ref_wo_out = ref_li.get("ref_wo_out")
            if attn_final is not None and ref_wo_out is not None:
                tt_af = torch.cat([attn_final[0, c, 0, :] for c in range(4)], dim=0).float()
                ref_af = ref_wo_out[0].flatten().float()
                _ratio_block("attn_out_final", ref_af, tt_af)

            # Pre-AR WO matmul output: sum the 8 row-device partials on host and compare.
            # If host-sum matches ref_wo_out closely, the matmul itself is correct and the
            # bias is in line_all_reduce. If host-sum is biased, the matmul is the source.
            wo_pre_ar = attn_caps.get("wo_matmul_pre_ar")
            if wo_pre_ar is not None and ref_wo_out is not None:
                # wo_pre_ar capture dims=(0,1) → [8 rows, 4 cols, 32 batch, 1280 dim].
                # Sum across 8 rows for each col, then concat 4 cols for full 5120.
                tt_summed = torch.cat(
                    [sum(wo_pre_ar[r, c, 0, :].float() for r in range(8)) for c in range(4)],
                    dim=0,
                ).float()
                ref_af2 = ref_wo_out[0].flatten().float()
                _ratio_block("wo_matmul_host_sum", ref_af2, tt_summed)

            # ── MLP-internal element-wise comparison: TT vs ref ──
            mlp_caps = getattr(tt_layer.feed_forward, "captured", {}) or {}
            for mlp_name in ("w1_out_reduced", "w3_out_reduced", "ff1ff3", "w2_out_pre_ar"):
                tt_t = mlp_caps.get(mlp_name)
                ref_name = {
                    "w1_out_reduced": "w1_out",
                    "w3_out_reduced": "w3_out",
                    "ff1ff3": "ff1ff3",
                    "w2_out_pre_ar": "w2_out_pre_ar",
                }[mlp_name]
                ref_t = ref_li.get(ref_name)
                if tt_t is None or ref_t is None:
                    continue
                # TT shape: [rows, 1, batch, dim3_total] (rows replicated post-RS).
                # Flatten the row-0 slice user-0 batch entry to a 1-D vector of model values.
                tt_flat = tt_t[0, 0, 0, :].float().flatten() if tt_t.dim() == 4 else tt_t.float().flatten()
                # Ref shape: [batch=32, seq=1, intermediate]. User-0 → flatten.
                ref_flat = ref_t[0, 0, :].float().flatten() if ref_t.dim() == 3 else ref_t.float().flatten()
                n = min(ref_flat.numel(), tt_flat.numel())
                ref_n = ref_flat[:n]
                tt_n = tt_flat[:n]
                diff = (tt_n - ref_n).abs()
                # Relative ratio — only at positions where ref is not tiny (avoid division noise)
                mask = ref_n.abs() > 1e-3
                if mask.any():
                    ratio = tt_n[mask].abs() / ref_n[mask].abs()
                    r_med = ratio.median().item()
                    r_mean = ratio.mean().item()
                else:
                    r_med = r_mean = float("nan")
                logger.info(
                    f"    MLP {mlp_name:18s}: n={n}  abs_diff max={diff.max():.4e} "
                    f"mean={diff.mean():.4e}  |TT/ref| mean={r_mean:.4f} med={r_med:.4f}  "
                    f"ref_std={ref_n.std():.4f} tt_std={tt_n.std():.4f}  "
                    f"ref_shape={list(ref_t.shape)} tt_shape={list(tt_t.shape)}"
                )
        logger.info("=" * 70)

    @torch.no_grad()
    def test_decode_per_position_64layers(self, mesh_device, reset_seeds, ensure_gc):
        """Per-position, per-layer element-wise diagnostic for 64-layer decode.

        For each layer, captures the full hidden state and compares element-by-element
        against the CPU reference to find:
        - Which positions accumulate error fastest
        - Whether error is concentrated in specific dim ranges (col-device shards)
        - Which sub-blocks (attn vs MLP vs norm) contribute most error per layer
        - Whether drift is multiplicative (std ratio) or additive
        """
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 64
        max_seq_len = 256
        batch_size = 32
        start_pos = 127
        dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ===== CPU Reference =====
        logger.info(f"Building CPU reference ({n_layers} layers)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, 1, 5120)
        embeddings_ref = pt_input.float()

        # Capture per-layer hidden states from reference
        ref_layer_outputs = []  # list of (layer_idx, [batch, 1, 5120]) tensors

        def make_hook(layer_idx):
            def hook(module, inp, out):
                ref_layer_outputs.append((layer_idx, out.detach().clone()))

            return hook

        hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(ref_model.layers)]
        ref_model.forward(embeddings_ref, start_pos=start_pos, mode="decode")
        for h in hooks:
            h.remove()

        # Also capture sub-block intermediates for all layers
        from models.demos.olmo_galaxy.reference.olmo import TransformerBlock as RefBlock

        ref_subblock_captures = [{} for _ in range(n_layers)]
        original_fwd = RefBlock.forward

        def patched_fwd(self_blk, x, start_pos, freqs_cis, mask):
            li = self_blk.layer_id
            ref_subblock_captures[li]["layer_in"] = x.detach().clone()
            attn_raw = self_blk.attention(x, start_pos, freqs_cis, mask)
            ref_subblock_captures[li]["attn_out"] = attn_raw.detach().clone()
            attn_normed = self_blk.attention_norm(attn_raw)
            ref_subblock_captures[li]["attn_normed"] = attn_normed.detach().clone()
            h_val = x + attn_normed
            ref_subblock_captures[li]["h_attn"] = h_val.detach().clone()
            ff_raw = self_blk.feed_forward(h_val)
            ref_subblock_captures[li]["ff_out"] = ff_raw.detach().clone()
            ff_normed = self_blk.ffn_norm(ff_raw)
            ref_subblock_captures[li]["ff_normed"] = ff_normed.detach().clone()
            out_val = h_val + ff_normed
            ref_subblock_captures[li]["layer_out"] = out_val.detach().clone()
            return out_val

        RefBlock.forward = patched_fwd
        ref_model.forward(embeddings_ref, start_pos=start_pos, mode="decode")
        RefBlock.forward = original_fwd

        # ===== TTNN Model =====
        logger.info(f"Building TTNN model ({n_layers} layers)...")
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

        # Enable capture on every layer
        for layer in tt_model.layers:
            layer.capture_intermediates = True

        page_table = torch.argsort(torch.randperm(paged_attention_config.max_num_blocks)).reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
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
        rot_mats, _ = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

        tt_model.forward(
            tt_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
            kv_cache=[layer.attention.layer_past for layer in tt_model.layers],
        )
        ttnn.synchronize_device(mesh_device)

        # ===== Extraction helpers =====
        # TTNN decoder captures use ConcatMesh2dToTensor(dims=(3,1)):
        #   Shape: [1, 4, 32, 10240]
        #   dim1=4 col groups (each holds 1280 hidden dims), dim2=32 batch, dim3=10240 = 8 row devs × 1280
        #   All 32 batch users are present on every col device (batch is replicated, hidden is sharded).
        #   Row devices are identical after all_reduce, so we use [:slice_dim] from any one row device.
        def extract_hidden_all_users(tt_t, batch_size=32):
            """Reconstruct [batch, 5120] for ALL users from TTNN decode capture."""
            n_col = tt_t.shape[1]  # 4 col devices
            slice_dim = tt_t.shape[3] // 8  # 1280 per col device (one row device's share)
            # Col device j holds hidden dims [j*slice_dim : (j+1)*slice_dim]
            # All 32 users are on each col device (hidden sharded, batch replicated)
            hidden_per_col = [tt_t[0, j, :batch_size, :slice_dim] for j in range(n_col)]
            return torch.cat(hidden_per_col, dim=1).float()  # [32, 5120]

        def extract_hidden_u0(tt_t):
            """Reconstruct [5120] hidden state for user 0 only (backward compat)."""
            return extract_hidden_all_users(tt_t)[0]

        # ===== Per-layer analysis =====
        logger.info("=" * 80)
        logger.info("PER-LAYER ELEMENT-WISE DECODE ANALYSIS (all users, 64 layers)")
        logger.info(
            f"{'Layer':>5}  {'MaxAbsErr':>10}  {'P99Err':>8}  {'MeanAbsErr':>11}  "
            f"{'StdRatio':>9}  {'WorstUser':>9}  "
            f"{'Col0Err':>8}  {'Col1Err':>8}  {'Col2Err':>8}  {'Col3Err':>8}  "
            f"{'SubBlkMax':>10}"
        )
        logger.info("-" * 120)

        # Track which sub-block contributes most error each layer
        def subblock_errors(ref_caps, tt_layer_obj):
            """Return dict of mean abs error per sub-block for all users (mean across users)."""
            errs = {}
            cap_keys = ["attn_out", "attn_normed", "h_attn", "ff_out", "ff_normed", "layer_out"]
            for key in cap_keys:
                ref_t = ref_caps.get(key)
                tt_t = tt_layer_obj.captured.get(key)
                if ref_t is None or tt_t is None:
                    continue
                ref_all = ref_t[:, 0, :].float()  # [32, 5120]
                tt_all = extract_hidden_all_users(tt_t)[:, :5120]  # [32, 5120]
                errs[key] = (ref_all - tt_all).abs().mean().item()
            return errs

        def subblock_errors_detailed(ref_caps, tt_layer_obj):
            """Return dict of {mean_abs_err, ref_std, rel_err} per sub-block for all users."""
            results = {}
            cap_keys = ["attn_out", "attn_normed", "h_attn", "ff_out", "ff_normed", "layer_out"]
            for key in cap_keys:
                ref_t = ref_caps.get(key)
                tt_t = tt_layer_obj.captured.get(key)
                if ref_t is None or tt_t is None:
                    continue
                ref_all = ref_t[:, 0, :].float()  # [32, 5120]
                tt_all = extract_hidden_all_users(tt_t)[:, :5120]  # [32, 5120]
                abs_err = (ref_all - tt_all).abs().mean().item()
                ref_std = ref_all.std().item()
                rel_err = abs_err / (ref_std + 1e-8)
                results[key] = {"abs_err": abs_err, "ref_std": ref_std, "rel_err": rel_err}
            return results

        for li in range(n_layers):
            tt_layer = tt_model.layers[li]
            ref_all_out = ref_layer_outputs[li][1][:, 0, :].float()  # [32, 5120]

            tt_cap = tt_layer.captured.get("layer_out")
            if tt_cap is None:
                logger.warning(f"  Layer {li:2d}: missing TTNN layer_out capture")
                continue

            tt_all = extract_hidden_all_users(tt_cap)[:, :5120]  # [32, 5120]

            abs_err_all = (ref_all_out - tt_all).abs()  # [32, 5120]

            # Aggregate stats (across all users and all dims)
            max_abs_err = abs_err_all.max().item()
            p99_err = abs_err_all.quantile(0.99).item()
            mean_abs_err = abs_err_all.mean().item()

            # Reference and TT std (across all elements)
            std_ratio = tt_all.std().item() / (ref_all_out.std().item() + 1e-8)

            # Per-user mean error: which user is worst?
            per_user_mean = abs_err_all.mean(dim=1)  # [32]
            worst_user = per_user_mean.argmax().item()
            worst_user_err = per_user_mean[worst_user].item()

            # Per col-device shard errors (each shard holds 1280 dims of the 5120 hidden state)
            col_errs = []
            for col in range(4):
                col_errs.append(abs_err_all[:, col * 1280 : (col + 1) * 1280].mean().item())

            # Sub-block breakdown
            sb_errs = subblock_errors(ref_subblock_captures[li], tt_layer)
            sb_max_key = max(sb_errs, key=sb_errs.get) if sb_errs else "N/A"
            sb_max_val = sb_errs.get(sb_max_key, 0.0)

            layer_type = "sliding" if (li % 4) != 3 else "full"
            logger.info(
                f"  {li:3d} [{layer_type:7s}]  "
                f"max={max_abs_err:8.4f}  p99={p99_err:6.4f}  mean={mean_abs_err:8.5f}  "
                f"std_ratio={std_ratio:7.4f}  worst_u={worst_user:2d}({worst_user_err:.4f})  "
                f"cols=[{col_errs[0]:.4f},{col_errs[1]:.4f},{col_errs[2]:.4f},{col_errs[3]:.4f}]  "
                f"worst_subblk={sb_max_key}({sb_max_val:.5f})"
            )

        logger.info("=" * 80)

        # ===== Top-K worst (user, dim) pairs across all 32 users in layer 63 =====
        logger.info("\nTop-20 worst (user, dim) pairs in layer 63 output:")
        last_tt_layer = tt_model.layers[n_layers - 1]
        last_ref_out = ref_layer_outputs[n_layers - 1][1][:, 0, :].float()  # [32, 5120]
        last_tt_cap = last_tt_layer.captured.get("layer_out")
        if last_tt_cap is not None:
            last_tt_all = extract_hidden_all_users(last_tt_cap)[:, :5120]  # [32, 5120]
            abs_err_last = (last_ref_out - last_tt_all).abs()  # [32, 5120]
            flat_err = abs_err_last.reshape(-1)  # [32*5120]
            topk_vals, topk_flat_idx = flat_err.topk(20)
            for rank, (flat_idx, err) in enumerate(zip(topk_flat_idx.tolist(), topk_vals.tolist())):
                user = flat_idx // 5120
                dim = flat_idx % 5120
                col_dev = dim // 1280
                ref_val = last_ref_out[user, dim].item()
                tt_val = last_tt_all[user, dim].item()
                sign = "SIGN_FLIP" if (ref_val * tt_val < 0) else ""
                logger.info(
                    f"  rank {rank+1:2d}: user={user:2d} dim={dim:5d} (col={col_dev})  "
                    f"abs_err={err:.4f}  ref={ref_val:.4f}  tt={tt_val:.4f}  {sign}"
                )

        # ===== Per-user error summary for layer 63 =====
        logger.info("\nPer-user mean abs error at layer 63:")
        if last_tt_cap is not None:
            per_user_mean_l63 = abs_err_last.mean(dim=1)  # [32]
            per_user_max_l63 = abs_err_last.max(dim=1).values  # [32]
            for u in range(batch_size):
                logger.info(f"  user {u:2d}: mean={per_user_mean_l63[u]:.5f}  max={per_user_max_l63[u]:.4f}")
            # Uniformity check: if all users have similar errors, bug is systematic (not batch-index)
            err_cv = per_user_mean_l63.std().item() / (per_user_mean_l63.mean().item() + 1e-8)
            logger.info(f"\n  Cross-user error CoV (low=uniform/systematic, high=user-specific): {err_cv:.4f}")

        # ===== Per-sub-block summary for first and last layer =====
        for li in [0, 15, 31, 47, 63]:
            if li >= n_layers:
                continue
            logger.info(f"\n--- Sub-block detail for layer {li} ---")
            sb = subblock_errors_detailed(ref_subblock_captures[li], tt_model.layers[li])
            for key, d in sorted(sb.items(), key=lambda x: x[1]["abs_err"], reverse=True):
                logger.info(
                    f"  {key:12s}: abs_err={d['abs_err']:.5f}  ref_std={d['ref_std']:.4f}  rel_err={d['rel_err']:.2%}"
                )

        logger.info("\nDECODE 64L PER-POSITION DIAGNOSTIC COMPLETE")

    @torch.no_grad()
    def test_decode_2step_elementwise(self, mesh_device, reset_seeds, ensure_gc):
        """Quick 2-step autoregressive element-wise decode diagnostic.

        Runs exactly 2 decode steps in closed-loop fashion:
          Step 1: random input → 64 layers → compare element-wise
          Step 2: each model feeds its OWN step-1 output as next input → compare element-wise

        Shows error compounding in the closed-loop scenario (same as real inference).
        Only 2 forward passes → fast run, low hang risk.
        """
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 64
        n_steps = 2
        max_seq_len = 256
        batch_size = 32
        start_pos = 127
        dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ===== CPU Reference =====
        logger.info(f"Building CPU reference ({n_layers} layers)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        # ===== TTNN Model =====
        logger.info(f"Building TTNN model ({n_layers} layers)...")
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
        # Only capture the last layer to keep overhead low
        tt_model.layers[-1].capture_intermediates = True

        page_table = torch.argsort(torch.randperm(paged_attention_config.max_num_blocks)).reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )

        def extract_hidden_all_users(tt_t, batch_size=32):
            """Reconstruct [batch, 5120] for ALL users from TTNN decode capture."""
            n_col = tt_t.shape[1]
            slice_dim = tt_t.shape[3] // 8
            return torch.cat([tt_t[0, j, :batch_size, :slice_dim] for j in range(n_col)], dim=1).float()

        def log_elemwise(step, ref_all, tt_all):
            """Log element-wise stats across all users and return abs diff."""
            diff = (ref_all.float() - tt_all.float()).abs()  # [32, 5120]
            max_err = diff.max().item()
            p99_err = diff.quantile(0.99).item()
            mean_err = diff.mean().item()
            ref_std = ref_all.float().std().item()
            std_ratio = tt_all.float().std().item() / (ref_std + 1e-8)
            per_user_mean = diff.mean(dim=1)  # [32]
            worst_u = per_user_mean.argmax().item()
            cov = per_user_mean.std().item() / (per_user_mean.mean().item() + 1e-8)
            logger.info(
                f"  [step {step}] max={max_err:.5f}  p99={p99_err:.5f}  mean={mean_err:.5f}  "
                f"std_ratio={std_ratio:.4f}  worst_user={worst_u}({per_user_mean[worst_u]:.5f})  CoV={cov:.4f}"
            )
            # Top-5 worst (user, dim) pairs
            topk_vals, topk_idx = diff.reshape(-1).topk(5)
            for rank, (flat_i, err) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist())):
                u, d = flat_i // 5120, flat_i % 5120
                rv, tv = ref_all[u, d].item(), tt_all[u, d].item()
                sign = " SIGN_FLIP" if (rv * tv < 0) else ""
                logger.info(f"    top{rank+1}: user={u:2d} dim={d:5d} ref={rv:.4f} tt={tv:.4f} diff={err:.4f}{sign}")
            return diff

        # ===== Initial shared input =====
        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, 1, 5120)
        ref_hidden = pt_input.float()  # reference feeds forward its own output each step
        tt_hidden_cpu = None  # TTNN's last-layer output on CPU (for closed-loop input)

        logger.info("=" * 80)
        logger.info(f"2-STEP CLOSED-LOOP AUTOREGRESSIVE ELEMENTWISE ({n_layers}L, {batch_size} users)")
        logger.info("=" * 80)

        for step in range(n_steps):
            pos = start_pos + step
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP {step + 1}/{n_steps}  (position {pos})")
            logger.info(f"{'='*60}")

            # ----- Reference step -----
            ref_layer_outs_this_step = []

            def _make_hook(li):
                def _hook(module, inp, out):
                    ref_layer_outs_this_step.append((li, out.detach().clone()))

                return _hook

            hooks = [layer.register_forward_hook(_make_hook(i)) for i, layer in enumerate(ref_model.layers)]
            ref_model.forward(ref_hidden, start_pos=pos, mode="decode")
            for h in hooks:
                h.remove()

            ref_out_all = ref_layer_outs_this_step[-1][1][:, 0, :].float()  # [32, 5120]

            # ----- TTNN step -----
            # Step 0: use the same random input as reference.
            # Step 1+: TTNN feeds its own previous output (closed-loop).
            if tt_hidden_cpu is None:
                tt_src = pt_input.clone()  # [32, 1, 5120]
            else:
                tt_src = tt_hidden_cpu.unsqueeze(1)  # [32, 1, 5120]

            tt_input = model_args.prepare_residual_tensor_decode(
                tt_src, model_args.model_config["DECODE_RESIDUAL_MEMCFG"]
            )
            current_pos = torch.tensor([pos] * batch_size)
            current_pos_tensor = ttnn.from_torch(
                current_pos,
                device=mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=model_args.cluster_shape),
            )
            rot_mats, _ = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

            tt_model.forward(
                tt_input,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
                kv_cache=[layer.attention.layer_past for layer in tt_model.layers],
            )
            ttnn.synchronize_device(mesh_device)

            # ----- Extract TTNN last-layer output -----
            last_cap = tt_model.layers[-1].captured.get("layer_out")
            if last_cap is None:
                logger.warning(f"  Step {step + 1}: no layer_out capture available!")
                continue
            tt_out_all = extract_hidden_all_users(last_cap)[:, :5120]  # [32, 5120]

            # ----- Compare -----
            logger.info(f"\nElement-wise comparison (last layer output, all {batch_size} users):")
            log_elemwise(step + 1, ref_out_all, tt_out_all)

            # ----- Prepare inputs for next step (closed-loop) -----
            # Reference feeds its own output; TTNN feeds its own output.
            ref_hidden = ref_layer_outs_this_step[-1][1].detach()  # [32, 1, 5120]
            tt_hidden_cpu = tt_out_all.detach()  # [32, 5120]

            # Clear capture for next step
            tt_model.layers[-1].captured.clear()

        logger.info("\n2-STEP DECODE ELEMENTWISE COMPLETE")

    @torch.no_grad()
    def _run_decode_pcc(self, mesh_device, n_layers, start_pos=127, max_seq_len=256, batch_size=32, dtype=None):
        """Shared implementation for decode PCC tests."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")
        if dtype is None:
            dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ===== CPU Reference decode =====
        logger.info(f"Building CPU reference model ({n_layers} layer(s))...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, 1, 5120)
        embeddings_ref = pt_input.float()

        # Run reference decode at start_pos, capturing per-layer hidden states
        ref_layer_outputs = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                ref_layer_outputs.append((layer_idx, out.detach().clone()))

            return hook

        hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(ref_model.layers)]
        ref_out = ref_model.forward(embeddings_ref, start_pos=start_pos, mode="decode")
        for h in hooks:
            h.remove()

        logger.info(f"CPU ref output shape: {ref_out.shape}")
        logger.info(
            f"CPU ref stats: mean={ref_out.mean():.4f}, std={ref_out.std():.4f}, "
            f"nan={torch.isnan(ref_out).any()}, inf={torch.isinf(ref_out).any()}"
        )
        for layer_idx, h_state in ref_layer_outputs:
            logger.info(
                f"  Ref layer {layer_idx:2d} hidden: mean={h_state.mean():.4f}, "
                f"std={h_state.std():.4f}, max={h_state.abs().max():.4f}, "
                f"nan={torch.isnan(h_state).any()}, inf={torch.isinf(h_state).any()}"
            )

        # ===== TTNN decode =====
        logger.info(f"Building TTNN model ({n_layers} layer(s))...")
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()

        block_size = 64
        # batch_size_per_device_group = max(batch_size // 4, 1) for TG (4 KV-head device groups)
        batch_per_dg = max(batch_size // 4, 1)
        min_blocks = batch_per_dg * math.ceil(max_seq_len / block_size)
        max_num_blocks = max(4096, min_blocks)
        paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)

        tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            decode_mode_only=True,
        )
        print("DEBUG TEST: After TtTransformer constructor", flush=True)

        # Page table setup for paged attention
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        print("DEBUG TEST: Before page_table from_torch", flush=True)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )
        print("DEBUG TEST: After page_table from_torch", flush=True)

        print("DEBUG TEST: Before prepare_residual_tensor_decode", flush=True)
        tt_input = model_args.prepare_residual_tensor_decode(
            pt_input.clone(), model_args.model_config["DECODE_RESIDUAL_MEMCFG"]
        )
        print("DEBUG TEST: After prepare_residual_tensor_decode", flush=True)

        current_pos = torch.tensor([start_pos] * batch_size)
        print("DEBUG TEST: Before current_pos from_torch", flush=True)
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=model_args.cluster_shape),
        )
        print("DEBUG TEST: After current_pos from_torch", flush=True)

        print("DEBUG TEST: Before get_rm_rot_mats", flush=True)
        rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)
        print("DEBUG TEST: After get_rm_rot_mats", flush=True)

        tt_out_list = tt_model.forward(
            tt_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
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
        logger.info(f"Decode {n_layers}-layer logits PCC (user 0): {pcc_msg}")

        # Token agreement for user 0
        ref_tok0 = ref_logits_u0.argmax(dim=-1).item()
        tt_tok0 = tt_logits.argmax(dim=-1).item()
        logger.info(f"User 0 token match: ref={ref_tok0}, tt={tt_tok0}, match={ref_tok0 == tt_tok0}")

        assert passing, f"Decode {n_layers}L PCC {pcc_msg} < 0.80"
        logger.info(f"DECODE {n_layers}L PCC TEST: PASSED")

    @torch.no_grad()
    def test_e2e_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """E2E PCC: prefill + decode with N layers, compare token generation vs CPU.

        Override depth via N_LAYERS_E2E env var to also run at full 64L for cumulative
        prefill-KV-cache drift checks.
        """
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = int(os.environ.get("N_LAYERS_E2E", "1"))
        # PADDED_LEN_E2E: prefill seq_len. Default 128 (ttnn.linear path); set to e.g. 4096
        # to exercise the minimal_matmul prefill path that AIME demos hit.
        padded_len_override = int(os.environ.get("PADDED_LEN_E2E", "128"))
        max_seq_len = max(256, padded_len_override * 2)
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
        padded_len = padded_len_override
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

        # Use paged attention (required for prefill) with a proper page table for decode too.
        # Shrink max_num_blocks for deep stacks to fit DRAM (4096 × 64L OOMs); but ensure
        # enough blocks for the prefill: ceil(padded_len / block_size) * batch_per_dg.
        block_size_cfg_pre = 64
        batch_per_dg_for_blocks = max(model_args.max_batch_size // 4, 1)
        min_blocks_for_prefill = math.ceil(padded_len / block_size_cfg_pre) * batch_per_dg_for_blocks
        _default_blocks = 128 if n_layers >= 32 else 4096
        _max_blocks = max(_default_blocks, min_blocks_for_prefill)
        paged_attention_config = PagedAttentionConfig(block_size=block_size_cfg_pre, max_num_blocks=_max_blocks)
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

        # ── KV cache comparison after prefill ──
        # Ref Attention.cache_k/cache_v are populated for positions 0..padded_len-1.
        # TT paged KV cache is populated for the same positions on user 0.
        # Compare per-position, per-head: cos_sim, std-ratio, max-err.
        # K is in Meta-interleaved layout in TT; convert to HF split-half before compare.
        logger.info("=" * 60)
        logger.info(f"KV cache state after prefill (n_layers={n_layers}, user 0)")
        logger.info("=" * 60)

        def _tt_to_hf_layout(tt_tensor):
            """[r0,i0,r1,i1,...,r63,i63] → [r0,r1,...,r63,i0,i1,...,i63]"""
            even = tt_tensor[..., 0::2]
            odd = tt_tensor[..., 1::2]
            return torch.cat([even, odd], dim=-1)

        # Sample multiple layers to see if cumulative residual drift biases deep-layer K writes
        if n_layers >= 32:
            sample_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
        else:
            sample_layers = [0]

        for li in sample_layers:
            logger.info(f"--- Layer {li} ---")
            ref_attn_l = ref_model.layers[li].attention
            ref_K_full = ref_attn_l.cache_k[0, :padded_len, :, :].float()
            ref_V_full = ref_attn_l.cache_v[0, :padded_len, :, :].float()

            tt_kv_l = tt_model.layers[li].attention.layer_past
            tt_K_paged = tt_kv_l[0]
            tt_V_paged = tt_kv_l[1]

            # Slice only the blocks user 0 uses (saves memory)
            block_size_cfg = paged_attention_config.block_size
            head_dim = ref_K_full.shape[2]  # 128
            sample_positions = list(range(0, min(padded_len, 128), 16))
            if (padded_len - 1) not in sample_positions:
                sample_positions.append(padded_len - 1)

            cos_sims_k = []
            cos_sims_v = []
            for pos in sample_positions:
                page_idx = pos // block_size_cfg
                offset = pos % block_size_cfg
                block_for_u0 = int(page_table[0, page_idx])
                try:
                    k_block = ttnn.slice(
                        tt_K_paged,
                        [block_for_u0, 0, 0, 0],
                        [block_for_u0 + 1, 1, block_size_cfg, head_dim],
                    )
                    v_block = ttnn.slice(
                        tt_V_paged,
                        [block_for_u0, 0, 0, 0],
                        [block_for_u0 + 1, 1, block_size_cfg, head_dim],
                    )
                    k_block_cpu = ttnn.to_torch(
                        k_block,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape
                        ),
                    ).float()
                    v_block_cpu = ttnn.to_torch(
                        v_block,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(0, 1), mesh_shape=model_args.cluster_shape
                        ),
                    ).float()
                    ttnn.deallocate(k_block)
                    ttnn.deallocate(v_block)
                    tt_k_at_pos = k_block_cpu[:, 0, offset, :]
                    tt_v_at_pos = v_block_cpu[:, 0, offset, :]
                    tt_k_hf = _tt_to_hf_layout(tt_k_at_pos)
                    ref_k_at_pos = ref_K_full[pos]
                    ref_v_at_pos = ref_V_full[pos]

                    k_flat_tt = tt_k_hf.flatten()
                    k_flat_ref = ref_k_at_pos.flatten()
                    v_flat_tt = tt_v_at_pos.flatten()
                    v_flat_ref = ref_v_at_pos.flatten()
                    if k_flat_ref.std() < 1e-6:
                        continue
                    k_cos = F.cosine_similarity(k_flat_tt.unsqueeze(0), k_flat_ref.unsqueeze(0)).item()
                    v_cos = F.cosine_similarity(v_flat_tt.unsqueeze(0), v_flat_ref.unsqueeze(0)).item()
                    k_max_err = (k_flat_tt - k_flat_ref).abs().max().item()
                    v_max_err = (v_flat_tt - v_flat_ref).abs().max().item()
                    k_std_ratio = k_flat_tt.std().item() / max(k_flat_ref.std().item(), 1e-9)
                    v_std_ratio = v_flat_tt.std().item() / max(v_flat_ref.std().item(), 1e-9)
                    cos_sims_k.append(k_cos)
                    cos_sims_v.append(v_cos)
                    logger.info(
                        f"  pos={pos:3d}: K cos={k_cos:.6f} std_ratio={k_std_ratio:.4f} max_err={k_max_err:.4f}  "
                        f"V cos={v_cos:.6f} std_ratio={v_std_ratio:.4f} max_err={v_max_err:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"  pos={pos}: KV cache read failed: {e}")
            if cos_sims_k:
                import statistics

                logger.info(
                    f"  L{li} K cos_sim: min={min(cos_sims_k):.6f} "
                    f"median={statistics.median(cos_sims_k):.6f} max={max(cos_sims_k):.6f}"
                )
                logger.info(
                    f"  L{li} V cos_sim: min={min(cos_sims_v):.6f} "
                    f"median={statistics.median(cos_sims_v):.6f} max={max(cos_sims_v):.6f}"
                )
        logger.info("=" * 60)

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

    @torch.no_grad()
    def test_e2e_pcc_64layer_drift(self, mesh_device, reset_seeds, ensure_gc):
        """
        Cumulative-drift PCC: full 64L prefill + N decode tokens, FREE-RUN both sides.
        Each side picks its own argmax token; we still compute logit PCC at each step.
        Free-run captures the self-amplifying error path (slightly-wrong TT token enters
        TT KV cache and skews downstream attention), which teacher forcing would mask.
        """
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 64
        max_seq_len = 256
        n_decode_tokens = int(os.environ.get("N_DECODE_TOKENS", "32"))
        dtype = ttnn.bfloat8_b

        logger.info("Loading HF state dict...")
        hf_sd = load_hf_raw_state_dict(hf_model_path)

        logger.info(f"Building CPU reference ({n_layers} layers)...")
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len)

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

        # ===== CPU prefill + decode (capture logits per step) =====
        logger.info(f"CPU prefill (seq_len={seq_len})...")
        embeddings_ref = ref_model.tok_embeddings(tokens_pt[:, :padded_len]).float()
        ref_prefill_out = ref_model.forward(embeddings_ref, start_pos=0, mode="decode")
        ref_prefill_logits = ref_prefill_out[:, seq_len - 1, :].squeeze(0)  # [vocab]
        ref_first_token = ref_prefill_logits.argmax(dim=-1).item()
        logger.info(f"CPU prefill token: {ref_first_token} ({tokenizer.decode([ref_first_token])})")

        ref_tokens = [ref_first_token]
        ref_logits_per_step = [ref_prefill_logits.clone()]
        for step in range(n_decode_tokens):
            pos = seq_len + step
            tok_emb = ref_model.tok_embeddings(torch.tensor([[ref_tokens[-1]]])).float()
            ref_decode_out = ref_model.forward(tok_emb, start_pos=pos, mode="decode")
            step_logits = ref_decode_out[:, -1, :].squeeze(0)
            ref_tokens.append(step_logits.argmax(dim=-1).item())
            ref_logits_per_step.append(step_logits.clone())
            logger.info(f"CPU decode step {step}: tok={ref_tokens[-1]}")

        # ===== TTNN prefill + teacher-forced decode =====
        logger.info(f"Building TTNN model ({n_layers} layers)...")
        state_dict = model_args.load_state_dict()
        # Small KV cache: only need padded_len + n_decode_tokens positions; 64 layers × 4096 blocks
        # would OOM. block_size=64, 128 blocks → 8192-token KV → ~2 MB/shard × 64 layers × 2 = 256 MB/dev.
        paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=128)
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
        tt_out_prefill = tt_model.ttnn_prefill_forward(*transformed_inputs, kv_cache=kv_cache, batch_size=1)
        tt_first_tok_result = tt_model.process_output_prefill(tt_out_prefill, last_token_idx=seq_len - 1)
        ttnn.synchronize_device(mesh_device)
        tt_first_token = int(tt_first_tok_result[0])
        logger.info(f"TTNN prefill token: {tt_first_token} ({tokenizer.decode([tt_first_token])})")

        decode_page_table_pt = page_table.clone()
        decode_page_table_tt = ttnn.from_torch(
            decode_page_table_pt,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        tt_model.switch_mode("decode")
        vocab_size = model_args.vocab_size
        tt_out_logits_saved = torch.zeros(vocab_size)

        # Free-run decode: TT consumes its own previous argmax (matches real generation).
        tt_logits_per_step = []
        tt_tokens = [tt_first_token]
        current_pos_val = seq_len
        for step in range(n_decode_tokens):
            tok_input = tt_tokens[-1]  # free-run
            tok_tensor = torch.full((1, 1, 1, 32), 0, dtype=torch.long)
            tok_tensor[0, 0, 0, 0] = tok_input
            tt_tok_device = ttnn.from_torch(
                tok_tensor,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            current_pos = torch.full((32,), current_pos_val, dtype=torch.long)
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
            tt_logits_per_step.append(tt_out_logits_saved.clone())
            tt_tokens.append(int(tt_out_logits_saved.argmax().item()))
            current_pos_val += 1

        # ===== Per-step PCC =====
        logger.info("=" * 60)
        logger.info(f"CUMULATIVE DRIFT (64L, teacher-forced, n_decode={n_decode_tokens})")
        logger.info("=" * 60)
        logger.info(
            f"Prefill token: CPU={ref_first_token}, TTNN={tt_first_token}, "
            f"match={ref_first_token == tt_first_token}"
        )

        # ref_logits_per_step has prefill_logits at [0], step-0 at [1], ...
        # tt_logits_per_step has step-0 at [0], step-1 at [1], ...
        # For step k (0..n_decode_tokens-1), compare ref_logits_per_step[k+1] vs tt_logits_per_step[k]
        argmax_matches = 0
        for k in range(n_decode_tokens):
            ref_l = ref_logits_per_step[k + 1].float()
            tt_l = tt_logits_per_step[k].float()
            ref_top = ref_l.argmax().item()
            tt_top = tt_l.argmax().item()
            _, pcc = comp_pcc(ref_l, tt_l, pcc=0.0)
            argmax_matches += int(ref_top == tt_top)
            ref_str = tokenizer.decode([ref_top])
            tt_str = tokenizer.decode([tt_top])
            logger.info(
                f"  step {k:3d}: PCC={pcc:.6f}  ref={ref_top}({ref_str!r})  "
                f"tt={tt_top}({tt_str!r})  {'✓' if ref_top == tt_top else '✗'}"
            )
        logger.info(
            f"Argmax match: {argmax_matches}/{n_decode_tokens} " f"({100.0 * argmax_matches / n_decode_tokens:.1f}%)"
        )
        logger.info("E2E DRIFT TEST: COMPLETE")

    @torch.no_grad()
    def test_aime_compare_to_golden(self, mesh_device, reset_seeds, ensure_gc):
        """Teacher-forced TT decode against pre-captured HF golden values.

        Loads /data/aime_goldens/aime24_p{N}/{meta.json,layer_LL.pt} produced
        by scripts/capture_aime_golden.py. Replays the prompt's prefill +
        decode steps in TTNN; at each decode step feeds ref_tokens[step]
        (NOT TT's argmax) so KV/context stays aligned with the golden.

        Captures all per-(layer, op) tensors via the existing
        _capture_attn / _capture_mlp / decoder layer .captured hooks, applies
        mesh-extraction + Meta→HF layout conversion, and compares each
        (step, layer, op) tensor to the golden — emits a CSV with cos_sim,
        std_ratio, max_err, |TT/ref| median ratio per row.

        Env vars:
          AIME_GOLDEN_DIR   — root containing aime24_p{N}/  (default /data/aime_goldens)
          AIME_PROBLEM_ID   — 1, 2, 3, or 4  (required)
          AIME_N_STEPS      — cap decode steps to compare  (default: golden's full count)
        """
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        problem_id = os.environ.get("AIME_PROBLEM_ID")
        if problem_id is None:
            pytest.skip("AIME_PROBLEM_ID env var not set (1..4)")
        problem_id = int(problem_id)
        golden_root = Path(os.environ.get("AIME_GOLDEN_DIR", "/data/aime_goldens"))
        golden_dir = golden_root / f"aime24_p{problem_id}"
        if not golden_dir.exists():
            pytest.skip(f"golden dir does not exist: {golden_dir}")

        # ---- Load golden meta ------------------------------------------------
        with open(golden_dir / "meta.json") as f:
            meta = json.load(f)
        golden_n_steps = meta["n_decode_tokens"]
        n_decode_tokens = int(os.environ.get("AIME_N_STEPS", golden_n_steps))
        n_decode_tokens = min(n_decode_tokens, golden_n_steps)
        n_layers = meta["n_layers"]
        prompt = meta["prompt"]
        ref_tokens = meta["ref_tokens"]  # [first_token, after-step-0, after-step-1, ...]
        seq_len_meta = meta["seq_len"]
        padded_len = meta.get("padded_prefill", 512)
        op_names_meta = meta.get("op_names", [])

        logger.info("=" * 70)
        logger.info(
            f"AIME compare: problem={problem_id} n_layers={n_layers} "
            f"steps={n_decode_tokens} (golden has {golden_n_steps}), padded_len={padded_len}"
        )
        logger.info("=" * 70)

        # ---- Lazy golden loader ----------------------------------------------
        # Each layer file: {step_idx (-1 for prefill, 0..N-1 for decode): {op_name: tensor[fp16]}}
        _golden_layer_cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

        def _load_layer_golden(layer_idx: int):
            if layer_idx not in _golden_layer_cache:
                path = golden_dir / f"layer_{layer_idx:02d}.pt"
                _golden_layer_cache[layer_idx] = torch.load(path, map_location="cpu", weights_only=True)
            return _golden_layer_cache[layer_idx]

        # ---- Build TTNN model ------------------------------------------------
        max_seq_len = max(meta["max_seq_len"], padded_len * 2)
        batch_size = 1
        dtype = ttnn.bfloat8_b

        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=max_seq_len)
        model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()
        tokenizer = model_args.tokenizer
        if tokenizer is None:
            tokenizer = GPT2Tokenizer.from_pretrained(model_args.TOKENIZER_PATH)

        # Block sizing for paged attention. Need enough blocks to cover
        # padded_len + n_decode_tokens for user 0 plus the per-device-group
        # batch sharding constraint.
        block_size_pa = 64
        batch_per_dg = max(model_args.max_batch_size // 4, 1)
        min_blocks = batch_per_dg * math.ceil((padded_len + n_decode_tokens) / block_size_pa)
        max_num_blocks = max(128, min_blocks)
        paged_attention_config = PagedAttentionConfig(block_size=block_size_pa, max_num_blocks=max_num_blocks)

        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        page_table = torch.argsort(permutation).reshape(
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
        kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

        # Enable per-op capture on every TT layer for the comparison run.
        for layer in tt_model.layers:
            layer.capture_intermediates = True
            layer.attention.capture_intermediates = True
            layer.feed_forward.capture_intermediates = True

        # ---- Prefill ---------------------------------------------------------
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

        # Tokenize prompt to match the golden
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        seq_len = len(input_ids)
        assert seq_len == seq_len_meta, f"prompt re-tokenization mismatch: {seq_len} vs {seq_len_meta}"
        eos = tokenizer.eos_token_id or 50256
        input_ids_padded = input_ids + [eos] * (padded_len - seq_len)
        tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

        num_prefill_blocks = math.ceil(padded_len / block_size_pa)
        prefill_page_table = torch.ones(32, num_prefill_blocks, dtype=torch.int32) * -1
        prefill_page_table[0, :] = page_table[0, :num_prefill_blocks]

        host_inputs = tt_model.prepare_prefill_inputs_host(tokens_pt, user_id=0, page_table=prefill_page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
        transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)
        tt_out_prefill = tt_model.ttnn_prefill_forward(*transformed_inputs, kv_cache=kv_cache, batch_size=1)
        tt_first_tok_result = tt_model.process_output_prefill(tt_out_prefill, last_token_idx=seq_len - 1)
        ttnn.synchronize_device(mesh_device)
        tt_first_token = int(tt_first_tok_result[0])
        logger.info(f"TTNN prefill first token: {tt_first_token} (golden: {ref_tokens[0]})")

        # ---- Decode loop with teacher-forcing -------------------------------
        decode_page_table_tt = ttnn.from_torch(
            page_table.clone(),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_model.switch_mode("decode")
        vocab_size = model_args.vocab_size
        tt_out_logits_saved = torch.zeros(vocab_size)

        # CSV writer
        csv_path = golden_dir / "tt_compare.csv"
        csv_f = open(csv_path, "w")
        csv_f.write("step,layer,op,cos_sim,std_ratio,max_err,ratio_median,n\n")
        first_divergence = None

        # Mesh-extraction helpers (mirror the patterns used in the existing per-op test)
        def _decoder_user0(t: torch.Tensor) -> torch.Tensor:
            """tt_layer.captured tensors land as [1, 4, 32, 10240]; reconstruct user-0 [5120]."""
            n_col = t.shape[1]
            slice_dim = t.shape[3] // 8  # 1280
            return torch.cat([t[0, j, 0, :slice_dim] for j in range(n_col)], dim=0)

        def _attn_user0(t: torch.Tensor) -> torch.Tensor:
            """Attention captures are [8, 4, 32, dim]; user-0 row-0 col-cat."""
            # Many of the attention captures are already concat'd into a usable shape.
            # For sdpa_out: [8 row, 32 batch, 8 q_heads, 128] → cat row × q_heads (5 real)
            # For wo_input: [8, 4, 32, 640] → row 0, cat 4 cols
            # For attn_out_final: [8, 4, 32, 1280] → row 0, cat 4 cols
            if t.dim() != 4:
                return t.flatten()
            shp = list(t.shape)
            if shp[1] == 4 and shp[3] in (640, 1280):
                # row-replicated × col-sharded: take row 0, concat cols
                return torch.cat([t[0, c, 0, :] for c in range(4)], dim=0)
            if shp[1] == 32 and shp[2] == 8 and shp[3] == 128:
                # sdpa_out: [8 row, 32 batch, 8 q_heads, 128] (5 real heads)
                return torch.cat([t[r, 0, h, :] for r in range(8) for h in range(5)], dim=0)
            return t[0, 0, 0, :].flatten()

        def _mlp_user0(t: torch.Tensor) -> torch.Tensor:
            """MLP captures are [8, 1, 32, dim_per_dev*4]; rows replicated, cols sharded."""
            if t.dim() == 4 and t.shape[0] == 8:
                return t[0, 0, 0, :].flatten()
            return t.flatten()

        def _tt_to_hf_layout(t: torch.Tensor) -> torch.Tensor:
            """[r0,i0,r1,i1,...] → [r0,r1,...,r_d/2-1, i0,i1,...,i_d/2-1] per head."""
            even = t[..., 0::2]
            odd = t[..., 1::2]
            return torch.cat([even, odd], dim=-1)

        # Tensors that are in Meta interleaved layout in TT; need conversion to HF
        # layout before comparison (golden was stored in Meta where applicable;
        # see capture_aime_golden.py which stores q/k_post_norm/post_rope in Meta).
        # In this comparison, the TT q_post_rope / k_post_rope already match
        # golden's Meta-format storage; the HF-layout conversion only applies to
        # raw K cache reads, not to the layer-output captures here.

        # The TT layer's `.captured` attribute already holds
        # {layer_in, attn_out, attn_normed, h_attn, ff_out, ff_normed, layer_out}.
        # The attention's `.captured` has q/k_post_norm/rope, sdpa_out, wo_input/out_final.
        # The MLP's `.captured` has w1_out_reduced, w3_out_reduced, ff1ff3, w2_out_pre_ar.
        # Map those to golden op names.
        TT_TO_GOLDEN_OP = {
            # decoder layer captures
            "layer_in": ("decoder", "layer_in"),
            "attn_out": ("decoder", "attn_out"),
            "attn_normed": ("decoder", "attn_normed"),
            "h_attn": ("decoder", "h_attn"),
            "ff_out": ("decoder", "ff_out"),
            "ff_normed": ("decoder", "ff_normed"),
            "layer_out": ("decoder", "layer_out"),
            # attention captures
            "q_post_norm": ("attn", "q_post_norm"),
            "k_post_norm": ("attn", "k_post_norm"),
            "q_post_rope": ("attn", "q_post_rope"),
            "sdpa_out": ("attn", "sdpa_out"),
            "wo_input": ("attn", "sdpa_out"),  # wo_input == sdpa_out
            "attn_out_final": ("attn", "wo_out"),
            # mlp captures
            "w1_out_reduced": ("mlp", "w1_out"),
            "w3_out_reduced": ("mlp", "w3_out"),
            "ff1ff3": ("mlp", "ff1ff3"),
            "w2_out_pre_ar": ("mlp", "w2_out_pre_ar"),
        }

        def _ratios(tt_v: torch.Tensor, ref_v: torch.Tensor):
            n = min(tt_v.numel(), ref_v.numel())
            tt_v = tt_v[:n].float()
            ref_v = ref_v[:n].float()
            cos = F.cosine_similarity(tt_v.unsqueeze(0), ref_v.unsqueeze(0)).item()
            std_ratio = tt_v.std().item() / max(ref_v.std().item(), 1e-9)
            max_err = (tt_v - ref_v).abs().max().item()
            mask = ref_v.abs() > 1e-3
            r_med = (tt_v[mask].abs() / ref_v[mask].abs()).median().item() if mask.any() else float("nan")
            return cos, std_ratio, max_err, r_med, n

        def _capture_step_to_csv(step_idx: int):
            """Pull TT captures for every layer × every op, compare to golden, write rows."""
            nonlocal first_divergence
            for li, layer in enumerate(tt_model.layers):
                lay_caps = getattr(layer, "captured", {}) or {}
                attn_caps = getattr(layer.attention, "captured", {}) or {}
                mlp_caps = getattr(layer.feed_forward, "captured", {}) or {}
                golden_layer = _load_layer_golden(li)
                golden_step = golden_layer.get(step_idx, {})

                # decoder ops
                for tt_name, (kind, golden_name) in TT_TO_GOLDEN_OP.items():
                    if kind == "decoder":
                        t = lay_caps.get(tt_name)
                        extractor = _decoder_user0
                    elif kind == "attn":
                        t = attn_caps.get(tt_name)
                        extractor = _attn_user0
                    elif kind == "mlp":
                        t = mlp_caps.get(tt_name)
                        extractor = _mlp_user0
                    else:
                        continue
                    ref = golden_step.get(golden_name)
                    if t is None or ref is None:
                        continue
                    try:
                        tt_v = extractor(t)
                    except Exception:
                        continue
                    cos, std_ratio, max_err, r_med, n = _ratios(tt_v, ref)
                    csv_f.write(
                        f"{step_idx},{li},{golden_name},{cos:.6f},{std_ratio:.4f}," f"{max_err:.4e},{r_med:.4f},{n}\n"
                    )
                    if first_divergence is None and cos < 0.99:
                        first_divergence = (step_idx, li, golden_name, cos)
                        logger.info(
                            f"FIRST_DIVERGENCE step={step_idx} layer={li} " f"op={golden_name} cos_sim={cos:.4f}"
                        )

                # Clear per-layer captures after use to bound RAM
                lay_caps.clear()
                attn_caps.clear()
                mlp_caps.clear()
            csv_f.flush()

        # We don't have an easy way to capture *prefill* per-op via the existing
        # _capture_attn (it writes only at specific lines); skip prefill capture
        # for the comparison (use existing test_e2e_pcc_1layer KV-after-prefill
        # diagnostic for that). Compare decode-step intermediates only.

        # Argmax-divergence detection: compare TT's predicted next-token to the
        # golden ref_tokens[step+1]. Skip the heavy per-layer capture in the hot
        # path; only fire it inside an explicit step window provided by the
        # AIME_CAPTURE_RANGE env var ("start:end_inclusive", e.g., "50:57").
        # This avoids the chicken-and-egg problem where divergence is detected
        # AFTER the forward pass has run without capture (so .captured dicts
        # are stale or empty).
        first_argmax_divergence_step: int | None = None
        capture_range_env = os.environ.get("AIME_CAPTURE_RANGE", "")
        if capture_range_env:
            cap_start, cap_end = (int(x) for x in capture_range_env.split(":"))
            logger.info(f"Per-layer capture enabled for steps {cap_start}..{cap_end}")
        else:
            cap_start, cap_end = -1, -1

        def _set_capture(enabled: bool):
            for tt_layer in tt_model.layers:
                tt_layer.capture_intermediates = enabled
                tt_layer.attention.capture_intermediates = enabled
                tt_layer.feed_forward.capture_intermediates = enabled
                # Clear any stale captures (e.g., from prefill) so the next
                # forward populates fresh per-step data.
                if hasattr(tt_layer, "captured"):
                    tt_layer.captured.clear()
                if hasattr(tt_layer.attention, "captured"):
                    tt_layer.attention.captured.clear()
                if hasattr(tt_layer.feed_forward, "captured"):
                    tt_layer.feed_forward.captured.clear()

        # Disable capture in the fast-loop body unless we're inside the window.
        _set_capture(False)

        for step in range(n_decode_tokens):
            # Toggle capture into ON when entering the window, OFF when leaving.
            # This lets us run fast outside the window and capture full
            # per-(layer, op) data inside it.
            if step == cap_start:
                _set_capture(True)
            elif step == cap_end + 1:
                _set_capture(False)

            # Teacher-force: feed the golden's chosen token, not TT's argmax,
            # so KV trajectory stays aligned with golden.
            tok_input = ref_tokens[step]
            tok_tensor = torch.full((1, 1, 1, 32), 0, dtype=torch.long)
            tok_tensor[0, 0, 0, 0] = tok_input
            tt_tok_device = ttnn.from_torch(
                tok_tensor,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            current_pos_val = seq_len + step
            current_pos = torch.full((32,), current_pos_val, dtype=torch.long)
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

            # Argmax check. ref_tokens[step] is what we just fed; the *next*
            # golden token is ref_tokens[step+1] — that's what TT's logits
            # should predict.
            tt_pred = int(tt_out_logits_saved.argmax().item())
            if step + 1 < len(ref_tokens):
                ref_next = int(ref_tokens[step + 1])
                if tt_pred != ref_next:
                    if first_argmax_divergence_step is None:
                        first_argmax_divergence_step = step
                        logger.warning(f"FIRST_ARGMAX_DIVERGENCE step={step} " f"tt_pred={tt_pred} ref_next={ref_next}")
                if step % 50 == 0 or first_argmax_divergence_step is not None:
                    logger.info(
                        f"  step {step}: tt_pred={tt_pred} ref_next={ref_next} "
                        f"match={'Y' if tt_pred == ref_next else 'N'}"
                    )

            # Per-layer capture: only inside the explicit window.
            if cap_start <= step <= cap_end:
                _capture_step_to_csv(step)
                if step == cap_end:
                    logger.info(f"  Captured window {cap_start}..{cap_end}; stopping decode.")
                    break

        csv_f.close()
        logger.info(f"AIME compare DONE. csv={csv_path}")
        if first_divergence is not None:
            s, l, op, c = first_divergence
            logger.info(f"FIRST DIVERGENCE: step={s} layer={l} op={op} cos_sim={c:.6f}")
        else:
            logger.info("No (layer, op, step) crossed cos_sim < 0.99 — clean run.")

    @torch.no_grad()
    def test_prefill_per_op_pcc_1layer(self, mesh_device, reset_seeds, ensure_gc):
        """Per-op PCC for prefill: capture each sublayer output and compare with reference."""
        hf_model_path = os.environ.get("HF_MODEL")
        if not hf_model_path:
            pytest.skip("HF_MODEL not set")

        n_layers = 1
        max_seq_len = 256
        batch_size = 1
        dtype = ttnn.bfloat8_b

        hf_sd = load_hf_raw_state_dict(hf_model_path)

        # ── Reference model: run step-by-step through 1 layer ──
        ref_model = build_ref_model(hf_sd, n_layers=n_layers, max_seq_len=max_seq_len, max_batch_size=batch_size)

        # Resolve HF snapshot path
        import glob

        from transformers import GPT2Tokenizer

        base_path = os.path.expanduser(hf_model_path)
        if os.path.exists(os.path.join(base_path, "snapshots")):
            snap_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
            if snap_dirs:
                base_path = snap_dirs[0]
        tokenizer = GPT2Tokenizer.from_pretrained(base_path)

        prompt = "What is your favorite condiment?"
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        seq_len = len(input_ids)
        padded_len = 128
        if tokenizer:
            input_ids_padded = input_ids + [tokenizer.eos_token_id or 50256] * (padded_len - seq_len)
        else:
            input_ids_padded = input_ids + [0] * (padded_len - seq_len)
        tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

        embeddings_ref = ref_model.tok_embeddings(tokens_pt[:, :padded_len].long()).float()

        # Manually run through 1 layer capturing intermediates
        ref_captures = {}
        ref_layer = ref_model.layers[0]
        freqs_cis = ref_model.freqs_cis[:padded_len]
        mask = torch.full((padded_len, padded_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(embeddings_ref)

        x = embeddings_ref.clone()
        ref_captures["input"] = x.clone()

        # OLMo post-sublayer-norm: attention(x) → norm → residual, then FFN(h) → norm → residual
        # Step-by-step attention to capture Q/K intermediates
        from models.demos.olmo_galaxy.reference.olmo import apply_rotary_emb

        attn = ref_layer.attention
        bsz_r, seqlen_r, _ = x.shape
        xq_raw, xk_raw, xv_raw = attn.wq(x), attn.wk(x), attn.wv(x)

        # Capture Q/K/V before norm
        xq_heads_raw = xq_raw.view(bsz_r, seqlen_r, attn.n_heads, attn.head_dim)
        xk_heads_raw = xk_raw.view(bsz_r, seqlen_r, attn.n_kv_heads, attn.head_dim)
        xv_heads = xv_raw.view(bsz_r, seqlen_r, attn.n_kv_heads, attn.head_dim)
        ref_captures["q_before_norm"] = xq_heads_raw.clone()
        ref_captures["k_before_norm"] = xk_heads_raw.clone()
        ref_captures["v_heads"] = xv_heads.clone()

        def _global_rms_norm(t, weight):
            t_f = t.float()
            rms = torch.rsqrt(t_f.pow(2).mean(-1, keepdim=True) + 1e-6)
            return (t_f * rms * weight.to(t_f.device)).type_as(t)

        xq_normed = _global_rms_norm(xq_raw, attn.q_norm_weight)
        xk_normed = _global_rms_norm(xk_raw, attn.k_norm_weight)

        xq_heads = xq_normed.view(bsz_r, seqlen_r, attn.n_heads, attn.head_dim)
        xk_heads = xk_normed.view(bsz_r, seqlen_r, attn.n_kv_heads, attn.head_dim)

        ref_captures["q_after_norm"] = xq_heads.clone()  # [1, seq, 40, 128]
        ref_captures["k_after_norm"] = xk_heads.clone()  # [1, seq, 8, 128]

        xq_rot, xk_rot = apply_rotary_emb(xq_heads, xk_heads, freqs_cis=freqs_cis)
        ref_captures["q_after_rope"] = xq_rot.clone()
        ref_captures["k_after_rope"] = xk_rot.clone()

        attn.cache_k = attn.cache_k.to(xq_rot)
        attn.cache_v = attn.cache_v.to(xq_rot)
        attn.cache_k[:bsz_r, :seqlen_r] = xk_rot
        attn.cache_v[:bsz_r, :seqlen_r] = xv_heads
        keys = attn.cache_k[:bsz_r, :seqlen_r]
        values = attn.cache_v[:bsz_r, :seqlen_r]
        from models.demos.olmo_galaxy.reference.olmo import repeat_kv

        keys = repeat_kv(keys, attn.n_rep)
        values = repeat_kv(values, attn.n_rep)
        xq_t = xq_rot.transpose(1, 2)
        keys_t = keys.transpose(1, 2)
        values_t = values.transpose(1, 2)
        scale = (attn.head_dim**-0.5) * attn.mscale
        scores = torch.matmul(xq_t, keys_t.transpose(2, 3)) * scale
        if mask is not None:
            scores = scores + mask
        scores = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq_t)
        attn_output = torch.matmul(scores, values_t)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz_r, seqlen_r, -1)
        ref_captures["wo_input"] = attn_output.clone()  # [1, seq, 5120]
        attn_raw = attn.wo(attn_output)
        ref_captures["wo_out"] = attn_raw.clone()
        ref_captures["attn_out"] = attn_raw.clone()

        attn_normed = ref_layer.attention_norm(attn_raw)
        ref_captures["attn_normed"] = attn_normed.clone()

        h = x + attn_normed
        ref_captures["h_attn"] = h.clone()

        ff_raw = ref_layer.feed_forward(h)
        ref_captures["ff_out"] = ff_raw.clone()

        ff_normed = ref_layer.ffn_norm(ff_raw)
        ref_captures["ff_normed"] = ff_normed.clone()

        out = h + ff_normed
        ref_captures["layer_out"] = out.clone()

        # Also get logits for end-to-end check
        normed = ref_model.norm(out)
        ref_captures["final_norm"] = normed.clone()
        ref_logits = ref_model.output(normed)
        ref_captures["logits"] = ref_logits.clone()

        logger.info("Reference captures:")
        for k, v in ref_captures.items():
            logger.info(f"  {k}: shape={list(v.shape)}, std={v.std():.6f}, mean={v.mean():.6f}")

        # ── TTNN model ──
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

        # Enable capture on the single layer
        tt_model.layers[0].capture_intermediates = True

        # Prepare RoPE
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

        # Run TTNN prefill
        host_inputs = tt_model.prepare_prefill_inputs_host(tokens_pt, user_id=0, page_table=prefill_page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
        transformed_inputs = tt_model.transform_prefill_inputs_device(*device_inputs)
        tt_out_prefill = tt_model.ttnn_prefill_forward(
            *transformed_inputs,
            kv_cache=[tt_model.layers[0].attention.layer_past],
            batch_size=1,
        )
        ttnn.synchronize_device(mesh_device)

        # Extract TTNN hidden state (before norm + lm_head)
        cluster_shape = model_args.cluster_shape
        tt_hidden = ttnn.to_torch(
            tt_out_prefill,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape),
        ).float()
        logger.info(f"TTNN hidden state raw shape: {list(tt_hidden.shape)}")
        # Take first copy from replicated mesh dim 0
        if tt_hidden.dim() == 4 and tt_hidden.shape[1] > 1:
            tt_hidden = tt_hidden[:, 0, :, :]
        else:
            tt_hidden = tt_hidden.squeeze()
        if tt_hidden.dim() == 2:
            tt_hidden = tt_hidden.unsqueeze(0)
        logger.info(f"TTNN hidden state shape: {list(tt_hidden.shape)}")

        # Compare hidden state (layer_out) PCC
        ref_layer_out = ref_captures["layer_out"]  # [1, 128, 5120]
        if tt_hidden.shape != ref_layer_out.shape:
            logger.warning(f"Shape mismatch: TTNN={list(tt_hidden.shape)}, ref={list(ref_layer_out.shape)}")
            min_seq = min(tt_hidden.shape[-2], ref_layer_out.shape[-2])
            min_dim = min(tt_hidden.shape[-1], ref_layer_out.shape[-1])
            tt_hidden_cmp = tt_hidden[..., :min_seq, :min_dim]
            ref_cmp = ref_layer_out[..., :min_seq, :min_dim]
        else:
            tt_hidden_cmp = tt_hidden
            ref_cmp = ref_layer_out

        passing, pcc_msg = comp_pcc(ref_cmp.float(), tt_hidden_cmp.float(), 0.80)
        logger.info(f"Hidden state (layer_out) PCC: {pcc_msg}")
        logger.info(f"  ref std={ref_cmp.std():.6f}, tt std={tt_hidden_cmp.std():.6f}")

        # Also get logits PCC
        logits_buf_size = 100352
        tt_out_logits_saved = torch.zeros(1, logits_buf_size)
        tt_tok = tt_model.process_output_prefill(
            tt_out_prefill, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_out_logits_saved
        )
        ttnn.synchronize_device(mesh_device)

        vocab_size = model_args.vocab_size
        tt_logits = tt_out_logits_saved[:, :vocab_size]
        ref_logits_trimmed = ref_captures["logits"][:, seq_len - 1, :vocab_size]
        passing_logits, pcc_logits_msg = comp_pcc(ref_logits_trimmed.float(), tt_logits.float(), 0.80)
        logger.info(f"Logits PCC: {pcc_logits_msg}")

        # ── Per-op PCC from captured intermediates ──
        tt_captured = tt_model.layers[0].captured
        logger.info(f"\nCaptured TTNN intermediates: {list(tt_captured.keys())}")

        def _extract_tt(t):
            """Extract first replica from mesh-composed tensor."""
            if t.dim() == 4 and t.shape[1] > 1:
                t = t[:, 0, :, :]
            if t.dim() == 4 and t.shape[1] == 1:
                t = t.squeeze(1)
            return t.float()

        def _per_pos_pcc(name, ref_3d, tt_3d, real_len):
            """Per-position PCC: real vs padding breakdown."""
            ref_3d = ref_3d.float()
            tt_3d = tt_3d.float()
            # Full, real, padding PCC
            _, full_pcc = comp_pcc(ref_3d, tt_3d, 0.0)
            _, real_pcc = comp_pcc(ref_3d[:, :real_len, :], tt_3d[:, :real_len, :], 0.0)
            _, pad_pcc = comp_pcc(ref_3d[:, real_len:, :], tt_3d[:, real_len:, :], 0.0)
            logger.info(f"    {name:15s}: FULL={full_pcc}, REAL(0:{real_len})={real_pcc}, PAD({real_len}:)={pad_pcc}")
            # Per-position for key positions
            for p in [0, 1, seq_len - 1, padded_len // 2, padded_len - 1]:
                if p >= ref_3d.shape[1]:
                    continue
                r = ref_3d[:, p, :]
                t = tt_3d[:, p, :]
                _, pos_pcc = comp_pcc(r, t, 0.0)
                r_std = r.std().item()
                t_std = t.std().item()
                mae = (r - t).abs().max().item()
                tag = "REAL" if p < real_len else "PAD"
                logger.info(
                    f"      pos {p:3d} [{tag}]: PCC={pos_pcc}, max_abs={mae:.4f}, "
                    f"ref_std={r_std:.4f}, tt_std={t_std:.4f}"
                )
            return full_pcc, real_pcc, pad_pcc

        # Helper to compare attention Q/K heads: TTNN has [1, NH_row, seq, HD*4cols]
        # Reference has [1, seq, NH, HD]. We need to align them.
        n_heads_total = model_args.n_heads  # 40
        n_kv_heads_total = model_args.n_kv_heads  # 8
        head_dim = model_args.head_dim  # 128
        n_rows = model_args.cluster_shape[0]  # 8
        n_cols = model_args.cluster_shape[1]  # 4

        results = {}

        # --- Per-device Q comparison (direct per-device extraction) ---
        n_local_q = n_heads_total // n_rows  # 5
        n_local_kv = n_kv_heads_total // n_rows  # 1

        def _gptj_to_neox(tensor):
            """Convert Q/K from GPT-J format [r0,i0,r1,i1,...] to neox format [r0,r1,...,i0,i1,...].

            TTNN stores Q/K in GPT-J format (via reverse_permute on weights), while the
            reference model uses HF/neox format. This converts TTNN captures for comparison.
            Operates on the last dimension (head_dim).
            """
            hd = tensor.shape[-1]
            reshaped = tensor.reshape(*tensor.shape[:-1], hd // 2, 2)
            reals = reshaped[..., 0]
            imags = reshaped[..., 1]
            return torch.cat([reals, imags], dim=-1)

        def _compare_grid(name, ref_bshd, tt_grid, n_local, is_qk=False):
            """Compare tensor captured as per-device grid.
            ref_bshd: [1, seq, total_heads, head_dim]
            tt_grid: dict[(row, col)] → per-device torch tensor [1, n_local, seq, head_dim]
            is_qk: if True, convert TTNN from GPT-J to neox format before comparison
            """
            ref = ref_bshd.float()
            logger.info(f"\n  {name}: ref_shape={list(ref.shape)}, is_qk={is_qk}")
            for r in range(n_rows):
                tt_dev = tt_grid[(r, 0)].float()
                if is_qk:
                    tt_dev = _gptj_to_neox(tt_dev)
                ref_row = ref[:, :, r * n_local : (r + 1) * n_local, :]
                ref_match = ref_row.permute(0, 2, 1, 3)
                _, pcc = comp_pcc(ref_match, tt_dev, 0.0)
                best = {"pcc": float(pcc.split()[-1]) if isinstance(pcc, str) else float(pcc), "row": r}
                for trial_r in range(n_rows):
                    if trial_r == r:
                        continue
                    ref_trial = ref[:, :, trial_r * n_local : (trial_r + 1) * n_local, :].permute(0, 2, 1, 3)
                    _, trial_pcc = comp_pcc(ref_trial, tt_dev, 0.0)
                    trial_val = float(trial_pcc.split()[-1]) if isinstance(trial_pcc, str) else float(trial_pcc)
                    if trial_val > best["pcc"]:
                        best = {"pcc": trial_val, "row": trial_r}
                logger.info(
                    f"      PCC(mesh_row{r} vs ref_row{r})={pcc}, "
                    f"best_match=ref_row{best['row']}(PCC={best['pcc']:.6f})"
                )

        for name, n_local in [
            ("q_before_norm", n_local_q),
            ("q_after_norm", n_local_q),
            ("k_before_norm", n_local_kv),
            ("k_after_norm", n_local_kv),
            ("v_heads", n_local_kv),
        ]:
            if name not in tt_captured or name not in ref_captures:
                logger.warning(f"  {name}: NOT captured")
                continue
            is_qk = name.startswith("q_") or name.startswith("k_")
            _compare_grid(name, ref_captures[name], tt_captured[name], n_local, is_qk=is_qk)

        # --- WO out (from attention capture, stored as per-device grid) ---
        if "wo_out" in tt_captured and "wo_out" in ref_captures:
            wo_grid = tt_captured["wo_out"]
            # wo_out per device: [1, 1, seq, 1280]. After all_reduce on axis=0, all rows identical.
            # Reconstruct by concatenating cols: [1, seq, 1280*4=5120]
            wo_dev = wo_grid[(0, 0)]  # [1, 1, seq, 1280] from row 0, col 0
            # Check cols are different (hidden dim parts) or same
            wo_cols = [wo_grid[(0, c)] for c in range(n_cols)]
            col_diff = (wo_cols[0] - wo_cols[1]).abs().max().item()
            logger.info(f"\n  wo_out: per_dev_shape={list(wo_dev.shape)}, col0-col1_diff={col_diff:.4f}")
            if col_diff < 1e-3:
                # All cols identical (replicated), take col 0 from all rows
                wo_row0 = wo_cols[0].squeeze(1)  # [1, seq, 1280]
                logger.info(f"    wo cols are identical. Shape: {list(wo_row0.shape)}")
                tt_wo = wo_row0
            else:
                # Cols have different hidden dim parts, concatenate
                tt_wo = torch.cat([c.squeeze(1) for c in wo_cols], dim=-1)  # [1, seq, 5120]
            ref_wo = ref_captures["wo_out"].float()
            logger.info(f"    tt_wo shape: {list(tt_wo.shape)}, ref_wo shape: {list(ref_wo.shape)}")
            if tt_wo.shape[-1] == ref_wo.shape[-1]:
                full_pcc, real_pcc, pad_pcc = _per_pos_pcc("wo_out", ref_wo, tt_wo, seq_len)
                results["wo_out"] = {"full": full_pcc, "real": real_pcc, "pad": pad_pcc, "ratio": 1.0}

        # --- Decoder intermediates (all have [1, seq, 5120] shape, stored as tensors) ---
        for name in ["attn_out", "attn_normed", "h_attn", "ff_out", "ff_normed", "layer_out"]:
            if name not in tt_captured or name not in ref_captures:
                logger.warning(f"  {name}: NOT captured")
                continue

            tt_val = tt_captured[name]
            if isinstance(tt_val, dict):
                logger.warning(f"  {name}: is a grid (unexpected for decoder capture)")
                continue
            tt_t = _extract_tt(tt_val)
            ref_t = ref_captures[name].float()

            if tt_t.shape != ref_t.shape:
                min_seq = min(tt_t.shape[-2], ref_t.shape[-2])
                min_dim = min(tt_t.shape[-1], ref_t.shape[-1])
                tt_t = tt_t[..., :min_seq, :min_dim]
                ref_t = ref_t[..., :min_seq, :min_dim]

            full_pcc, real_pcc, pad_pcc = _per_pos_pcc(name, ref_t, tt_t, seq_len)
            norm_ratio = tt_t.std().item() / max(ref_t.std().item(), 1e-10)
            results[name] = {"full": full_pcc, "real": real_pcc, "pad": pad_pcc, "ratio": norm_ratio}

        # Also compare embedding input to verify it's identical
        ref_emb = ref_captures["input"]
        tt_emb_val = tt_model.layers[0].captured.get("input")
        if tt_emb_val is not None and not isinstance(tt_emb_val, dict):
            tt_emb = _extract_tt(tt_emb_val)
            if tt_emb.shape == ref_emb.shape:
                _per_pos_pcc("embedding", ref_emb, tt_emb, seq_len)

        logger.info("\n" + "=" * 70)
        logger.info("PREFILL PER-OP PCC SUMMARY (full / real-only / pad-only)")
        logger.info("=" * 70)
        for name, r in results.items():
            logger.info(f"  {name:15s}: FULL={r['full']}, REAL={r['real']}, PAD={r['pad']}, ratio={r['ratio']:.4f}")
        logger.info(f"  {'hidden_state':15s}: PCC={pcc_msg}")
        logger.info(f"  {'logits':15s}: PCC={pcc_logits_msg}")
        logger.info("=" * 70)
