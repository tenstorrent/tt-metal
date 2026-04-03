# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation: Sub-device partition + L1 state + GDN fused kernel + full model decode.

Setup:
- SD0 (48 cores): GDN L1 state (HEIGHT_SHARDED)
- SD1 (62 cores): compute (SDPA, matmul, etc.)

Tests:
1. Sub-device creation + state allocation
2. GDN fused kernel with L1 state
3. Full model decode with 6 GDN layers in L1
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def setup_sub_devices(mesh_device):
    """Create SD0 (48 cores for GDN state) + SD1 (62 cores for compute)."""
    gx = mesh_device.compute_with_storage_grid_size().x
    gy = mesh_device.compute_with_storage_grid_size().y
    logger.info(f"Grid: {gx}×{gy} = {gx*gy} cores")

    # SD0: first 48 cores (0,0)-(7,5) for GDN state
    sd0_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    # SD1: remaining 62 cores for everything else
    sd1_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(gx - 1, gy - 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(7, gy - 1)),
        ]
    )
    logger.info(f"SD0: {sd0_cores.num_cores()} cores (GDN state)")
    logger.info(f"SD1: {sd1_cores.num_cores()} cores (compute)")

    sd0 = ttnn.SubDevice([sd0_cores])
    sd1 = ttnn.SubDevice([sd1_cores])
    manager_id = mesh_device.create_sub_device_manager([sd0, sd1], 0)
    mesh_device.load_sub_device_manager(manager_id)

    return sd0_cores, sd1_cores, manager_id


def create_hs_config(ncores, total_height=49152, width=128):
    """Create HEIGHT_SHARDED L1 config for state tensor on SD0 cores."""
    shard_h = total_height // ncores
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cg, [shard_h, width], ttnn.ShardOrientation.ROW_MAJOR),
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_subdevice_l1_state_e2e(mesh_device, reset_seeds, ensure_gc):
    """Full model decode with 6 GDN layers' state in L1 on sub-device 0."""
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 256
    max_gen_tokens = 10
    N_L1_LAYERS = 5  # 5 fits (256KB/core * 5 = 1280KB), 6 OOMs (1427KB bank limit)

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ---- Setup sub-devices BEFORE model creation ----
    logger.info("Setting up sub-devices...")
    sd0_cores, sd1_cores, manager_id = setup_sub_devices(mesh_device)

    # ---- Create HEIGHT_SHARDED config for state ----
    # rec_states: [B*Nv_TP, Dk, Dv] = [384, 128, 128]
    # Flattened height: 384 * (128/32) = 384 * 4 tile rows = 1536 tile rows
    # Total rows (in elements): 384 * 128 = 49152
    hs_cfg = create_hs_config(ncores=48)

    # ---- Create model ----
    logger.info("Creating model...")
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=max_seq_len)
    args = model.args

    gdn_indices = [i for i in range(args.n_layers) if args.layer_types[i] == "linear_attention"]
    logger.info(f"Moving first {N_L1_LAYERS} of {len(gdn_indices)} GDN layers to L1 HEIGHT_SHARDED...")

    # ---- Move first N GDN layers' rec_states to L1 HEIGHT_SHARDED ----
    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        if gdn.rec_states is None:
            gdn.reset_state()
        # DRAM INTERLEAVED → L1 HEIGHT_SHARDED on SD0 cores
        l1_state = ttnn.to_memory_config(gdn.rec_states, hs_cfg)
        ttnn.deallocate(gdn.rec_states)
        gdn.rec_states = l1_state
        logger.info(f"  Layer {idx}: state -> L1 HS (addr=0x{l1_state.buffer_address():x})")

    # ---- Prefill ----
    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' -> {len(prompt_tokens)} tokens")

    logger.info("Prefilling...")
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

    # ---- Decode ----
    logger.info(f"Decoding {max_gen_tokens} tokens...")
    generated_tokens = []
    current_token = prompt_tokens[-1]
    decode_times = []

    for step in range(max_gen_tokens):
        t_step = time.time()
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), len(prompt_tokens) - 1 + step, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
        generated_tokens.append(next_token)
        current_token = next_token
        dt = time.time() - t_step
        decode_times.append(dt)

        if step < 3:
            logger.info(f"  Step {step+1}: '{tokenizer.decode([next_token])}' ({dt*1000:.0f}ms)")

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    if len(decode_times) > 1:
        avg_time = sum(decode_times[1:]) / len(decode_times[1:])
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Avg step time: {avg_time*1000:.1f}ms ({tps:.1f} tok/s/user)")

    # Verify L1 state survived
    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        is_l1 = gdn.rec_states.memory_config().buffer_type == ttnn.BufferType.L1
        assert is_l1, f"Layer {idx} state fell out of L1!"
    logger.info("All L1 states survived decode")

    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"

    # Cleanup
    mesh_device.clear_loaded_sub_device_manager()
    logger.info("PASSED: Sub-device L1 state decode produces correct output")
