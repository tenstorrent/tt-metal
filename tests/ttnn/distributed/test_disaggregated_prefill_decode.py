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

DEFAULT_PROMPT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor ultrices ligula. Phasellus at dui in ligula mollis ultricies. Vestibulum blandit rhoncus risus. Quisque ligula ipsum, euismod atras vulputate scelerisque et dictum.

Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Proin pharetra nonummy pede. Mauris et orci. Aenean nec lorem. In porttitor consequat ut. Pellentesque dapibus hendrerit tortor. Praesent egestas tristique nibh. Sed a libero. Cras varius. Donec vitae orci sed dolor rutrum auctor. Fusce egestas elit eget lorem. Suspendisse nisl elit rhoncus eget elementum ac condimentum eget diam.

Nam pretium turpis et arcu. Duis arcu tortor suscipit eget imperdiet nec imperdiet iaculis ipsum. Sed aliquam ultrices mauris. Integer ante arcu accumsan a consectetuer eget posuere ut mauris. Praesent adipiscing. Phasellus ullamcorper ipsum rutrum nunc. Nunc nonummy metus. Vestibulum volutpat pretium libero. Cras id dui. Aenean ut eros et nisl sagittis vestibulum. Nullam nulla eros ultricies sit amet nonummy id imperdiet feugiat pede.

Sed lectus. Donec mollis hendrerit risus. Phasellus nec sem in justo pellentesque facilisis. Etiam imperdiet imperdiet orci. Nunc nec neque. Phasellus leo dolor tempus non auctor et rutrum non nonummy eget. Praesent quis nisi ut arcu dignissim mattis. Aenean auctor gravida sem. Praesent id massa id nisl venenatis lacinia. Aenean sit amet justo. Morbi ut odio. Cras mi pede malesuada in imperdiet et commodo vulputate justo.

In blandit ultrices enim. Lorem ipsum dolor sit amet consectetuer adipiscing elit. Proin interdum mauris non ligula pellentesque ultrices. Phasellus id sapien in sapien iaculis congue. Vivamus metus arcu adipiscing molestie hendrerit at vulputate vitae nisl. Aenean lectus. Pellentesque eget nunc. Donec quis orci eget orci vehicula condimentum. Curabitur in libero ut massa volutpat convallis. Morbi odio odio elementum eu interdum eu tincidunt in leo.

Maecenas malesuada. Praesent congue erat at massa. Sed cursus turpis vitae tortor. Donec posuere vulputate arcu. Phasellus accumsan cursus velit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae. Sed aliquam nisi quis porttitor congue elit erat euismod orci ac placerat dolor lectus quis orci. Phasellus consectetuer vestibulum elit. Aenean tellus metus bibendum sed posuere ac mattis non nunc.

Vestibulum fringilla pede sit amet augue. In turpis pellentesque purus. Quisque a lectus. Donec consectetuer ligula vulputate sem tristique cursus. Nam nulla. Donec sodales sagittis magna. Sed consequat leo eget bibendum sodales augue velit cursus nunc quis gravida ante nibh vel velit auctor aliquet. Aenean sollicitudin nec sagittis nisi semper in. Duis vel nibh at velit scelerisque suscipit. Curabitur turpis. Vestibulum suscipit nulla quis orci.

Fusce ac felis sit amet ligula pharetra condimentum. Maecenas egestas arcu quis ligula mattis placerat. Duis lobortis massa imperdiet quam. Suspendisse potenti. Pellentesque commodo eros a enim. Vestibulum turpis sem aliquet eget lobortis pellentesque rutrum eu nisl. Proin nec tellus sit amet turpis lacinia placerat imperdiet nulla. Sed augue ipsum egestas nec vestibulum et molestie sit amet dui. Fusce anteam justo in pellentesque mollis pretium sit amet justo.

Nulla metus metus ullamcorper vel tincidunt sed euismod in nibh. Quisque volutpat condimentum velit. Class aptent taciti sociosqu ad litora torquent per conubia nostra per inceptos himenaeos. Nam nec ante. Sed lacinia arcu eu justo posuere ullamcorper. Donec vitae ligula id urna malesuada dictum. Fusce neque. Nunc eleifend consequat lorem. Sed lacinia nulla vitae enim. Pellentesque tincidunt purus vel magna. Integer non enim. Praesent euismod nunc eu purus.

Donec bibendum quam in tellus. Nullam cursus pulvinar lectus. Donec et mi. Nam vulputate metus eu enim. Vestibulum pellentesque felis eu massa. Quisque ullamcorper placerat ipsum. Cras nibh. Morbi vel justo vitae lacus tincidunt ultrices. Lorem ipsum dolor sit amet consectetuer adipiscing elit. In hac habitasse platea dictumst. Integer tempus convallis augue. Etiam facilisis ligula nec velit. Praesent quam turpis feugiat sit amet ultricies cursus.

Proin sodales libero eget ante. Nulla quam. Aenean laoreet. Vestibulum nisi lectus commodo ac facilisis ac ultricies eu pede. Ut orci risus accumsan porttitor cursus quis aliquet eget justo. Sed pretium blandit orci. Ut eu diam at pede suscipit sodales. Aenean lectus elit fermentum non convallis id sagittis at neque. Nullam mauris orci aliquet et faucibus posuere in nunc. Donec blandit feugiat ligula."""
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
    # Record wall-clock time (monotonic nanoseconds) for end-to-end transfer measurement
    t_transfer_start = time.perf_counter()
    t_transfer_start_ns = time.time_ns()  # Wall-clock for cross-process measurement
    for layer_idx, (k_cache, v_cache) in enumerate(kv_cache):
        logger.info(f"Sending KV cache layer {layer_idx}")
        ttnn.experimental.send_async(k_cache, send_socket)
        ttnn.experimental.send_async(v_cache, send_socket)

    # Also send the sequence length, next token, and transfer start timestamp as metadata
    # (In practice, you'd use a separate control channel)
    # Use actual_seq_len for metadata (before padding)
    #
    # NOTE: We use UINT32 dtype to preserve integer precision.
    # BFLOAT16 loses precision for large integers (e.g., token IDs > 32K)
    # which causes the wrong token to be decoded on the receiver side.
    #
    # Use explicit 4D shape [1, 1, 1, 4] to match receiver's expected dimensions.
    # TILE_LAYOUT will pad this to [1, 1, 32, 32] on both sides.
    # We split the 64-bit nanosecond timestamp into two 32-bit integers (high, low).
    t_start_high = int(t_transfer_start_ns >> 32)
    t_start_low = int(t_transfer_start_ns & 0xFFFFFFFF)
    metadata = torch.tensor([[[[actual_seq_len, next_token.item(), t_start_high, t_start_low]]]], dtype=torch.int64)
    metadata_tt = ttnn.from_torch(
        metadata,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(
        f"Sending metadata: seq_len={actual_seq_len}, next_token={next_token.item()}, transfer_start_ns={t_transfer_start_ns}"
    )
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

    # Receive KV cache layer by layer
    for layer_idx in range(len(kv_cache)):
        k_cache_recv = ttnn.allocate_tensor_on_device(kv_cache[layer_idx][0].spec, device)
        v_cache_recv = ttnn.allocate_tensor_on_device(kv_cache[layer_idx][1].spec, device)

        ttnn.experimental.recv_async(k_cache_recv, recv_socket)
        ttnn.experimental.recv_async(v_cache_recv, recv_socket)

        # Update KV cache with received data
        kv_cache[layer_idx] = [k_cache_recv, v_cache_recv]
        logger.info(f"Received KV cache layer {layer_idx}")

    # Receive metadata (UINT32 to preserve integer precision for token IDs)
    # Shape must match sender's explicit 4D shape [1, 1, 1, 4], which gets tile-padded to [1, 1, 32, 32]
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

    # Record wall-clock time immediately after sync completes (all data received)
    t_recv_end_ns = time.time_ns()

    # Convert metadata tensor to torch
    # Since we're on a distributed mesh, use mesh composer (tensor is replicated, so take first slice)
    metadata_full = ttnn.to_torch(
        ttnn.from_device(metadata_recv), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1)
    )
    # Take first slice since tensor is replicated across mesh (4 values per replica)
    metadata = metadata_full[:, :, :, : metadata_full.shape[3] // 2] if metadata_full.shape[3] > 4 else metadata_full

    # Debug: log the metadata tensor shape and first few values
    logger.info(f"Metadata tensor shape: {metadata.shape}")
    logger.info(f"Metadata tensor first values: {metadata[0, 0, 0, :8].tolist()}")

    # Extract metadata from the first tile (data is in the first row of the first tile)
    # The tensor shape is [1, 1, 32, 32] but we only sent 4 values:
    #   [0]: seq_len, [1]: next_token, [2]: timestamp_high, [3]: timestamp_low
    seq_len = int(metadata[0, 0, 0, 0].item())
    current_token = int(metadata[0, 0, 0, 1].item())
    t_start_high = int(metadata[0, 0, 0, 2].item())
    t_start_low = int(metadata[0, 0, 0, 3].item())

    # Reconstruct the 64-bit nanosecond timestamp from prefill node
    # Mask to 32 bits to handle UINT32 -> signed int32 conversion (negative values if high bit set)
    t_transfer_start_ns = ((t_start_high & 0xFFFFFFFF) << 32) | (t_start_low & 0xFFFFFFFF)
    e2e_transfer_time_ms = (t_recv_end_ns - t_transfer_start_ns) / 1_000_000

    logger.info(f"=== END-TO-END KV CACHE TRANSFER TIME: {e2e_transfer_time_ms:.1f}ms ===")

    logger.info(
        f"Received metadata: seq_len={seq_len}, current_token={current_token} ({tokenizer.decode([current_token])})"
    )
    logger.info(f"Starting decode from position {seq_len}")

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
    logger.info(f"  KV cache transfer (end-to-end): {e2e_transfer_time_ms:.1f}ms")
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
