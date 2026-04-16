# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OLMo3 paged cache operations against CPU golden reference.

1. test_olmo_fused_cache_vs_cpu: Isolated fused cache update + SDPA vs PyTorch CPU
2. test_olmo_1layer_prefill_vs_hf: Full 1-layer prefill on TT device vs HuggingFace CPU
"""

import torch
import pytest
import ttnn
import os
import glob

from models.common.utility_functions import comp_pcc


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Isolated fused cache update + SDPA vs CPU golden
# ─────────────────────────────────────────────────────────────────────────────


def cpu_paged_attention(q, k_cache, v_cache, page_table, current_pos, block_size, scale):
    """CPU reference for paged SDPA decode."""
    batch, n_q_heads, head_dim = q.shape
    outputs = torch.zeros_like(q)
    for b in range(batch):
        pos = current_pos[b].item()
        if pos < 0:
            continue
        seq_len = pos + 1
        k_seq = torch.zeros(seq_len, head_dim, dtype=q.dtype)
        v_seq = torch.zeros(seq_len, head_dim, dtype=q.dtype)
        for t in range(seq_len):
            phys_block = page_table[b, t // block_size].item()
            k_seq[t] = k_cache[phys_block, t % block_size, :]
            v_seq[t] = v_cache[phys_block, t % block_size, :]
        for h in range(n_q_heads):
            scores = (q[b, h, :] @ k_seq.T) * scale
            weights = torch.softmax(scores, dim=-1)
            outputs[b, h, :] = weights @ v_seq
    return outputs


@pytest.mark.parametrize("batch_size", [1, 32], ids=["b1", "b32"])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_olmo_fused_cache_vs_cpu(batch_size, mesh_device):
    """Paged_fused_update_cache + paged SDPA vs CPU golden. OLMo3 shapes."""
    torch.manual_seed(42)

    n_q_heads, n_kv_expanded, head_dim = 5, 8, 128
    block_size, max_num_blocks = 64, 256
    n_cols = mesh_device.shape[1]
    batch_per_device = max(batch_size, 32) // n_cols
    n_active = max(batch_size // n_cols, 1)
    scale = head_dim**-0.5

    page_table = torch.arange(max_num_blocks).reshape(batch_per_device, max_num_blocks // batch_per_device).int()

    prefill_len, decode_pos = 42, 42
    k_cache_cpu = torch.zeros(max_num_blocks, block_size, head_dim)
    v_cache_cpu = torch.zeros(max_num_blocks, block_size, head_dim)
    for b in range(n_active):
        for t in range(prefill_len):
            pb = page_table[b, t // block_size].item()
            k_cache_cpu[pb, t % block_size] = torch.randn(head_dim)
            v_cache_cpu[pb, t % block_size] = torch.randn(head_dim)

    k_new = torch.randn(n_active, head_dim)
    v_new = torch.randn(n_active, head_dim)
    for b in range(n_active):
        pb = page_table[b, decode_pos // block_size].item()
        k_cache_cpu[pb, decode_pos % block_size] = k_new[b]
        v_cache_cpu[pb, decode_pos % block_size] = v_new[b]

    current_pos = torch.full([batch_per_device], -1, dtype=torch.int32)
    current_pos[:n_active] = decode_pos
    q_cpu = torch.randn(batch_per_device, n_q_heads, head_dim)
    cpu_output = cpu_paged_attention(q_cpu, k_cache_cpu, v_cache_cpu, page_table, current_pos, block_size, scale)

    # Device
    k_pre = torch.zeros_like(k_cache_cpu)
    v_pre = torch.zeros_like(v_cache_cpu)
    for b in range(n_active):
        for t in range(prefill_len):
            pb = page_table[b, t // block_size].item()
            k_pre[pb, t % block_size] = k_cache_cpu[pb, t % block_size]
            v_pre[pb, t % block_size] = v_cache_cpu[pb, t % block_size]

    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    k_cache_tt = ttnn.from_torch(
        k_pre.unsqueeze(1).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=rep,
    )
    v_cache_tt = ttnn.from_torch(
        v_pre.unsqueeze(1).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=rep,
    )

    k_pad = torch.zeros(batch_per_device, head_dim)
    k_pad[:n_active] = k_new
    v_pad = torch.zeros(batch_per_device, head_dim)
    v_pad[:n_active] = v_new
    k_exp = k_pad.unsqueeze(0).unsqueeze(2).expand(1, batch_per_device, n_kv_expanded, head_dim).contiguous()
    v_exp = v_pad.unsqueeze(0).unsqueeze(2).expand(1, batch_per_device, n_kv_expanded, head_dim).contiguous()

    k_tt = ttnn.from_torch(
        k_exp.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    v_tt = ttnn.from_torch(
        v_exp.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    n_cores = 8
    k_tt = ttnn.to_memory_config(
        k_tt,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, n_cores - 1))]),
                [n_kv_expanded, head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    v_tt = ttnn.to_memory_config(
        v_tt,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, n_cores - 1))]),
                [n_kv_expanded, head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    cp = ttnn.from_torch(current_pos, dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
    pt = ttnn.from_torch(page_table, dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
    ttnn.experimental.paged_fused_update_cache(k_cache_tt, k_tt, v_cache_tt, v_tt, update_idxs_tensor=cp, page_table=pt)
    ttnn.synchronize_device(mesh_device)

    # Cache PCC
    k_dev = ttnn.to_torch(ttnn.get_device_tensors(k_cache_tt)[0]).float().squeeze(1)
    v_dev = ttnn.to_torch(ttnn.get_device_tensors(v_cache_tt)[0]).float().squeeze(1)
    written = sorted({page_table[b, t // block_size].item() for b in range(n_active) for t in range(decode_pos + 1)})
    k_pass, k_pcc = comp_pcc(
        torch.stack([k_dev[i] for i in written]), torch.stack([k_cache_cpu[i] for i in written]), 0.99
    )
    v_pass, v_pcc = comp_pcc(
        torch.stack([v_dev[i] for i in written]), torch.stack([v_cache_cpu[i] for i in written]), 0.99
    )
    print(f"\n[Fused ops vs CPU] batch={batch_size}: K PCC={k_pcc}, V PCC={v_pcc}")
    assert k_pass and v_pass, f"Cache PCC fail: K={k_pcc}, V={v_pcc}"

    # SDPA PCC
    q_tt = ttnn.from_torch(
        q_cpu.bfloat16().unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    cp2 = ttnn.from_torch(current_pos, dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
    pt2 = ttnn.from_torch(page_table, dtype=ttnn.int32, device=mesh_device, mesh_mapper=rep)
    sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tt, k_cache_tt, v_cache_tt, cur_pos_tensor=cp2, page_table_tensor=pt2, scale=scale
    )
    ttnn.synchronize_device(mesh_device)
    sdpa_dev = ttnn.to_torch(ttnn.get_device_tensors(sdpa)[0]).float().squeeze(0)
    sdpa_pass, sdpa_pcc = comp_pcc(sdpa_dev[:n_active], cpu_output[:n_active], 0.98)
    print(f"  SDPA PCC (active)={sdpa_pcc}")
    assert sdpa_pass, f"SDPA PCC {sdpa_pcc} < 0.98"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Full 1-layer OLMo3 prefill on TT device vs HuggingFace CPU
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 184915840,
            "num_command_queues": 1,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_1layer_prefill_vs_hf(mesh_device):
    """
    1-layer OLMo3 prefill: compare TT device logits vs HuggingFace CPU logits.
    Uses real weights. Checks PCC > 0.99 on prefill output logits.
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
    from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.tt_transformers.tt.common import copy_host_to_device

    torch.manual_seed(42)

    hf_model_path = os.path.expanduser(os.environ.get("HF_MODEL", "~/models/OLMo-3.1-32B-Think"))
    snap_dirs = glob.glob(os.path.join(hf_model_path, "snapshots", "*"))
    config_dir = snap_dirs[0] if snap_dirs else hf_model_path

    # ── Load HF model (1 layer, CPU, real weights) ──────────────────────
    print("\nLoading HF reference model (1 layer)...")
    hf_config = AutoConfig.from_pretrained(config_dir)
    hf_config.num_hidden_layers = 1
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        config=hf_config,
        torch_dtype=torch.float32,
    )
    hf_model.eval()
    tokenizer = hf_model.config.tokenizer_class if hasattr(hf_model.config, "tokenizer_class") else None

    # ── Load TT model (1 layer, device) ─────────────────────────────────
    print("Loading TT model (1 layer)...")
    paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=4096)
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        decode_mode_only=False,
    )
    tt_tokenizer = model_args.tokenizer

    # ── Prepare input tokens ────────────────────────────────────────────
    prompt = "The quick brown fox jumps over the lazy dog"
    encoded = tt_tokenizer.encode(prompt)
    seq_len = len(encoded)
    padded_len = 128
    tokens_padded = encoded + [tt_tokenizer.eos_token_id] * (padded_len - seq_len)
    tokens_tensor = torch.tensor(tokens_padded, dtype=torch.long).unsqueeze(0)
    print(f"Prompt: '{prompt}' → {seq_len} tokens")

    # ── HF reference forward (CPU) ─────────────────────────────────────
    hf_input = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        hf_output = hf_model(hf_input)
    # Logits at last token position: [vocab_size]
    hf_logits = hf_output.logits[0, -1, :].float()
    hf_top5 = torch.topk(hf_logits, 5)
    print(f"HF top-5 tokens: {hf_top5.indices.tolist()}")
    print(f"HF top-5 values: {[f'{v:.2f}' for v in hf_top5.values.tolist()]}")

    # ── TT device prefill ───────────────────────────────────────────────
    # Page table (identity for simplicity)
    page_table = (
        torch.arange(paged_attention_config.max_num_blocks).reshape(1, paged_attention_config.max_num_blocks).int()
    )
    page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 31), value=0)  # pad to 32 rows

    kv_cache = [tt_model.layers[0].attention.layer_past]

    host_inputs = tt_model.prepare_prefill_inputs_host(tokens_tensor, user_id=0, page_table=page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    transformed = tt_model.transform_prefill_inputs_device(*device_inputs)
    tt_out = tt_model.ttnn_prefill_forward(*transformed, kv_cache=kv_cache, batch_size=1)

    # Extract logits at last real token position
    tt_logits_saved = torch.zeros(1, model_args.padded_vocab_size)
    first_tok = tt_model.process_output_prefill(tt_out, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_logits_saved)
    ttnn.synchronize_device(mesh_device)

    tt_logits = tt_logits_saved[0, : model_args.vocab_size].float()
    tt_top5 = torch.topk(tt_logits, 5)
    print(f"TT top-5 tokens: {tt_top5.indices.tolist()}")
    print(f"TT top-5 values: {[f'{v:.2f}' for v in tt_top5.values.tolist()]}")

    # ── Compare PCC ─────────────────────────────────────────────────────
    pcc_pass, pcc_val = comp_pcc(tt_logits, hf_logits, 0.90)
    print(f"Prefill logits PCC (TT vs HF): {pcc_val}")

    # Check top-1 match
    top1_match = tt_top5.indices[0].item() == hf_top5.indices[0].item()
    print(f"Top-1 match: {top1_match} (TT={tt_top5.indices[0].item()}, HF={hf_top5.indices[0].item()})")

    # Top-5 overlap
    tt_set = set(tt_top5.indices.tolist())
    hf_set = set(hf_top5.indices.tolist())
    top5_overlap = len(tt_set & hf_set)
    print(f"Top-5 overlap: {top5_overlap}/5")

    assert pcc_pass, f"Prefill logits PCC {pcc_val} < 0.90"
