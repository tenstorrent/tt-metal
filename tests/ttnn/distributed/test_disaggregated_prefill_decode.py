#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Disaggregated Prefill-Decode Example for Llama 8B on 2x N300 devices.

This demonstrates pipeline parallelism where:
- Process 0 (N300 #0): Runs prefill, generates KV cache, sends to Process 1
- Process 1 (N300 #1): Receives KV cache, continues decode to generate tokens
Each N300 has 2 chips (1x2 mesh topology).

Required files:
- Mesh descriptor: tests/tt_metal/tt_fabric/custom_mesh_descriptors/n300_dual_mesh_graph_descriptor.textproto
- Rank binding: tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml

Run with:
    cd $TT_METAL_HOME
    tt-run --rank-binding tests/ttnn/distributed/disaggregated_pd_rank_binding.yaml \
           python3 tests/ttnn/distributed/test_disaggregated_prefill_decode.py
"""

import time
import torch
import ttnn
from loguru import logger
from transformers import AutoTokenizer

# Model imports
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

DEFAULT_PROMPT = "Explain the concept of machine learning in detail:"
DEFAULT_MAX_NEW_TOKENS = 50


def setup_kv_cache_sockets(device, mesh_shape, num_layers):
    """
    Setup sockets for transferring KV cache between prefill and decode nodes.
    We need 2 sockets per layer (one for K, one for V).
    """
    sender_coord = ttnn.CoreCoord(0, 0)
    recv_coord = ttnn.CoreCoord(0, 0)

    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        socket_connections.append(
            ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sender_coord), ttnn.MeshCoreCoord(coord, recv_coord))
        )

    # Larger buffer for KV cache tensors
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16384)  # 16KB buffer

    socket_config = ttnn.SocketConfig(
        socket_connections,
        socket_mem_config,
        sender_rank=0,  # Prefill node
        receiver_rank=1,  # Decode node
    )

    return socket_config


def create_model_and_cache(device, max_batch_size=1, max_seq_len=8192):
    """
    Create model and get KV cache.

    Note: For non-paged attention, the KV cache is automatically initialized
    inside each attention layer during model construction. We extract it from
    layer.attention.layer_past for each layer.
    """
    model_args = ModelArgs(
        device,
        instruct=True,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )

    state_dict = model_args.load_state_dict()

    model = Transformer(
        args=model_args,
        mesh_device=device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        use_paged_kv_cache=False,  # Use simple KV cache for this example
    )

    # Extract KV cache from each layer's attention module
    # Each layer.attention.layer_past is a list of [K_cache, V_cache] tensors
    kv_cache = [layer.attention.layer_past for layer in model.layers]

    return model, model_args, kv_cache


def run_prefill_node(device, model, model_args, kv_cache, tokens, send_socket, actual_seq_len=None):
    """
    Prefill Node (Process 0):
    1. Run prefill to populate KV cache
    2. Send KV cache to decode node
    3. Send the output logits (for next token)

    Args:
        send_socket: Pre-created MeshSocket for sending data (created before prefill to avoid timeout).
        actual_seq_len: The actual sequence length before padding (for metadata).
                        If None, uses tokens.shape[1] (assumes no padding).
    """
    logger.info("=== PREFILL NODE (Process 0) ===")

    seq_len = tokens.shape[1]
    if actual_seq_len is None:
        actual_seq_len = seq_len
    logger.info(f"Running prefill for {seq_len} tokens (actual: {actual_seq_len})")

    # Prepare inputs
    prefill_input, rot_mats, *_ = model.prepare_inputs_prefill(tokens)

    # Run prefill forward pass with timing
    t_prefill_start = time.perf_counter()
    tt_logits = model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats,
        rot_mats_local=None,
        user_id=0,
        kv_cache=kv_cache,
        get_last_token=(actual_seq_len - 1) // 32 * 32,
    )

    # Get the next token prediction
    logits_cpu = model.process_output_prefill(tt_logits.cpu(), (actual_seq_len - 1) % 32)
    t_prefill_end = time.perf_counter()
    prefill_time_ms = (t_prefill_end - t_prefill_start) * 1000

    # logits_cpu is already a 1D tensor [vocab_size] for the last token
    next_token = torch.argmax(logits_cpu)

    logger.info(f"Prefill complete in {prefill_time_ms:.1f}ms. Next token: {next_token.item()}")

    # Send KV cache layer by layer with timing
    t_transfer_start = time.perf_counter()
    for layer_idx, (k_cache, v_cache) in enumerate(kv_cache):
        logger.info(f"Sending KV cache layer {layer_idx}")
        ttnn.experimental.send_async(k_cache, send_socket)
        ttnn.experimental.send_async(v_cache, send_socket)

    # Also send the sequence length and next token as metadata
    # (In practice, you'd use a separate control channel)
    # Use actual_seq_len for metadata (before padding)
    #
    # NOTE: We use UINT32 dtype to preserve integer precision.
    # BFLOAT16 loses precision for large integers (e.g., token IDs > 32K)
    # which causes the wrong token to be decoded on the receiver side.
    #
    # Use explicit 4D shape [1, 1, 1, 2] to match receiver's expected dimensions.
    # TILE_LAYOUT will pad this to [1, 1, 32, 32] on both sides.
    metadata = torch.tensor([[[[actual_seq_len, next_token.item()]]]], dtype=torch.int32)
    metadata_tt = ttnn.from_torch(
        metadata,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"Sending metadata: seq_len={actual_seq_len}, next_token={next_token.item()}")
    ttnn.experimental.send_async(metadata_tt, send_socket)

    # CRITICAL: Synchronize after all send_async calls to ensure transfers complete.
    # Without this, if the prefill process exits or closes the socket before the
    # async DMA transfers finish, the decode node may receive incomplete/corrupted data:
    #   1. KV cache layers may be partially transferred (truncated tensors)
    #   2. Metadata may not arrive, causing decode node to hang or crash
    #   3. Socket teardown during active transfer can cause fabric-level errors
    ttnn.synchronize_device(device)

    t_transfer_end = time.perf_counter()
    transfer_time_ms = (t_transfer_end - t_transfer_start) * 1000

    logger.info(f"KV cache sent to decode node in {transfer_time_ms:.1f}ms")
    logger.info(f"=== PREFILL TIMING SUMMARY ===")
    logger.info(f"  Prefill compute: {prefill_time_ms:.1f}ms")
    logger.info(f"  KV cache transfer: {transfer_time_ms:.1f}ms")
    logger.info(f"  Total: {prefill_time_ms + transfer_time_ms:.1f}ms")
    return next_token


def run_decode_node(device, model, model_args, kv_cache, recv_socket, tokenizer, max_new_tokens=20):
    """
    Decode Node (Process 1):
    1. Receive KV cache from prefill node
    2. Continue autoregressive decode

    Args:
        recv_socket: Pre-created MeshSocket for receiving data (created before prefill to avoid timeout).
    """
    logger.info("=== DECODE NODE (Process 1) ===")

    # Receive KV cache layer by layer with timing
    t_recv_start = time.perf_counter()
    for layer_idx in range(len(kv_cache)):
        k_cache_recv = ttnn.allocate_tensor_on_device(kv_cache[layer_idx][0].spec, device)
        v_cache_recv = ttnn.allocate_tensor_on_device(kv_cache[layer_idx][1].spec, device)

        ttnn.experimental.recv_async(k_cache_recv, recv_socket)
        ttnn.experimental.recv_async(v_cache_recv, recv_socket)

        # Update KV cache with received data
        kv_cache[layer_idx] = [k_cache_recv, v_cache_recv]
        logger.info(f"Received KV cache layer {layer_idx}")

    # Receive metadata (UINT32 to preserve integer precision for token IDs)
    # Shape must match sender's explicit 4D shape [1, 1, 1, 2], which gets tile-padded to [1, 1, 32, 32]
    metadata_recv = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec([1, 1, 32, 32], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT), device
    )
    ttnn.experimental.recv_async(metadata_recv, recv_socket)

    # CRITICAL: Synchronize after all recv_async calls before using the received data.
    # Race condition: recv_async is non-blocking - it only initiates the DMA transfer.
    # Without synchronization, the decode loop would start reading KV cache tensors while
    # the fabric is still transferring data, causing:
    #   1. Corrupted KV cache values (partial/garbage data)
    #   2. Incorrect attention outputs leading to nonsensical token generation
    #   3. Potential hardware hangs if tensors are deallocated during active transfer
    ttnn.synchronize_device(device)
    # Convert metadata tensor to torch
    # Since we're on a distributed mesh, use mesh composer (tensor is replicated, so take first slice)
    metadata_full = ttnn.to_torch(
        ttnn.from_device(metadata_recv), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1)
    )
    # Take first slice since tensor is replicated across mesh
    metadata = metadata_full[:, :, :, : metadata_full.shape[3] // 2] if metadata_full.shape[3] > 2 else metadata_full

    # Debug: log the metadata tensor shape and first few values
    logger.info(f"Metadata tensor shape: {metadata.shape}")
    logger.info(f"Metadata tensor first values: {metadata[0, 0, 0, :8].tolist()}")

    # Extract metadata from the first tile (data is in the first row of the first tile)
    # The tensor shape is [1, 1, 32, 32] but we only sent 2 values, so they're at [0, 0, 0, 0] and [0, 0, 0, 1]
    seq_len = int(metadata[0, 0, 0, 0].item())
    current_token = int(metadata[0, 0, 0, 1].item())
    t_recv_end = time.perf_counter()
    recv_time_ms = (t_recv_end - t_recv_start) * 1000

    logger.info(
        f"Received metadata: seq_len={seq_len}, current_token={current_token} ({tokenizer.decode([current_token])})"
    )
    logger.info(f"Received KV cache in {recv_time_ms:.1f}ms. Starting decode from position {seq_len}")

    # Decode loop
    generated_tokens = [current_token]
    current_pos = seq_len

    logger.info(
        f"Starting decode loop. Initial token: {current_token} ({tokenizer.decode([current_token])}) at pos={current_pos}"
    )

    t_decode_start = time.perf_counter()
    decode_times = []
    step = 0
    while True:
        t_step_start = time.perf_counter()

        # Prepare decode input (single token)
        token_tensor = torch.tensor([[current_token]], dtype=torch.long)
        pos_tensor = torch.tensor([current_pos], dtype=torch.long)

        # prepare_inputs_decode uses *inputs, so pass arguments positionally
        # Signature: prepare_inputs_decode(tokens, current_pos, page_table=None)
        tt_tokens, tt_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(token_tensor, pos_tensor)

        # Run decode forward
        tt_logits, _ = model.ttnn_decode_forward(
            tt_tokens,
            tt_pos,
            rot_mat_idxs=tt_rot_idxs,
            kv_cache=kv_cache,
        )

        # Get next token
        logits_cpu = model.process_output_decode(tt_logits.cpu(), 1, S=1)
        next_token = torch.argmax(logits_cpu[:, -1], dim=-1).item()

        t_step_end = time.perf_counter()
        step_time_ms = (t_step_end - t_step_start) * 1000
        decode_times.append(step_time_ms)

        logger.info(
            f"Step {step}: Generated token {next_token} ({tokenizer.decode([next_token])}) [{step_time_ms:.1f}ms]"
        )

        generated_tokens.append(next_token)
        current_token = next_token
        current_pos += 1

        step += 1

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            logger.info(f"EOS token detected at step {step}")
            break

        # Check for max tokens limit
        if step >= max_new_tokens:
            logger.info(f"Reached max_new_tokens limit ({max_new_tokens})")
            break

    # Decode timing summary
    t_decode_end = time.perf_counter()
    total_decode_time_ms = (t_decode_end - t_decode_start) * 1000
    num_tokens = len(generated_tokens) - 1  # Exclude the initial token from prefill
    avg_decode_time_ms = sum(decode_times) / len(decode_times) if decode_times else 0
    tokens_per_sec = (num_tokens / total_decode_time_ms * 1000) if total_decode_time_ms > 0 else 0

    # Decode tokens to text
    logger.info(f"All generated tokens: {generated_tokens}")
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")

    logger.info(f"=== DECODE TIMING SUMMARY ===")
    logger.info(f"  KV cache receive: {recv_time_ms:.1f}ms")
    logger.info(f"  Decode steps: {num_tokens} tokens in {total_decode_time_ms:.1f}ms")
    logger.info(f"  Avg time per token: {avg_decode_time_ms:.1f}ms")
    logger.info(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")

    return generated_tokens, generated_text


def run_disaggregated_prefill_decode():
    """Main entry point for disaggregated PD example."""

    # Initialize TT-Fabric for inter-device communication
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # Each process gets one N300 (1x2 mesh, 2 chips per board)
    mesh_shape = ttnn.MeshShape(1, 2)

    # Get the visible device IDs for this rank (set by TT_VISIBLE_DEVICES in rank binding)
    # For N300, setting TT_VISIBLE_DEVICES="0" or "1" exposes both chips (PCIe + remote)
    # So get_device_ids() should return exactly 2 device IDs for the 1x2 mesh
    visible_device_ids = ttnn.get_device_ids()
    if len(visible_device_ids) != 2:
        raise ValueError(
            f"Expected exactly 2 devices for N300 (1x2 mesh), got {len(visible_device_ids)}. "
            f"Visible device IDs: {visible_device_ids}. "
            f"Make sure TT_VISIBLE_DEVICES is set correctly in the rank binding."
        )

    # Use both devices (the 2 chips of the N300 board)
    # Specifying physical_device_ids explicitly bypasses SystemMesh lookup issues
    physical_device_ids = visible_device_ids
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, physical_device_ids=physical_device_ids)

    # Verify distributed context
    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized. Run with tt-run.")

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise ValueError(f"This example requires exactly 2 processes, got {world_size}")

    rank = int(ttnn.distributed_context_get_rank())
    logger.info(f"Process {rank} started on N300 device")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Create model and KV cache on both nodes
    model, model_args, kv_cache = create_model_and_cache(device)

    # Setup sockets for KV cache transfer
    socket_config = setup_kv_cache_sockets(device, mesh_shape, model_args.n_layers)

    # The prompt to process
    prompt = DEFAULT_PROMPT

    logger.info(f"Prompt: '{prompt}'")

    # === CREATE SOCKETS BEFORE HEAVY COMPUTATION ===
    # This ensures both ranks establish the socket connection before prefill starts.
    # Without this, the decode node would timeout waiting while prefill is running.
    logger.info(f"Rank {rank}: Creating socket before computation...")
    if rank == 0:
        socket = ttnn.MeshSocket(device, socket_config)
        logger.info("Prefill node: Send socket created")
    else:
        socket = ttnn.MeshSocket(device, socket_config)
        logger.info("Decode node: Receive socket created")

    # Barrier to ensure both sockets are established
    ttnn.distributed_context_barrier()
    logger.info(f"Rank {rank}: Socket handshake complete, proceeding with computation")

    if rank == 0:
        # === PREFILL NODE ===
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
        logger.info(f"Tokenized: {tokens.tolist()}")

        # Pad tokens to be divisible by 128 (model requirement)
        actual_seq_len = tokens.shape[1]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Round up to nearest multiple of 128
        padded_len = ((actual_seq_len + 127) // 128) * 128

        if padded_len > actual_seq_len:
            padding = torch.full((1, padded_len - actual_seq_len), pad_token_id, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=1)
            logger.info(f"Padded sequence from {actual_seq_len} to {padded_len} tokens")

        next_token = run_prefill_node(
            device, model, model_args, kv_cache, tokens, socket, actual_seq_len=actual_seq_len
        )
        logger.info(f"Prefill node complete. First generated token: {tokenizer.decode([next_token.item()])}")
    else:
        # === DECODE NODE ===
        generated_tokens, generated_text = run_decode_node(
            device, model, model_args, kv_cache, socket, tokenizer, max_new_tokens=DEFAULT_MAX_NEW_TOKENS
        )
        logger.info("Decode node complete.")
        logger.info(f"Full output: {prompt}{generated_text}")

    # Cleanup socket before closing device
    # MeshSocket resources are released when the object is deleted.
    # Explicit deletion ensures socket teardown completes before device close.
    del socket

    # Synchronize before cleanup
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)

    logger.info(f"Process {rank} finished")


if __name__ == "__main__":
    run_disaggregated_prefill_decode()
