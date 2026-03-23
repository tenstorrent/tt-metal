# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B decode benchmark on P300x2 (4 Blackhole chips).

Measures TPOT (time per output token) with trace capture.
Supports batch=1 through batch=32.

Usage:
    export HF_MODEL=/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd
    python models/demos/qwen35/demo/benchmark_p300x2.py [--batch_size 32] [--decode_steps 10]
"""

import os
import time

import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, copy_host_to_device, create_tt_model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5-27B decode benchmark on P300x2")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (users)")
    parser.add_argument("--decode_steps", type=int, default=10, help="Number of decode steps to measure")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--no_trace", action="store_true", help="Disable trace capture")
    parser.add_argument("--prefetcher", action="store_true", help="Enable DRAM prefetcher")
    parser.add_argument("--bfp4", action="store_true", help="Use bfp4 weights (half DRAM reads)")
    args_cli = parser.parse_args()

    batch_size = args_cli.batch_size

    # Resolve HF model path
    hf_model = os.environ.get(
        "HF_MODEL",
        "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/"
        "b7ca741b86de18df552fd2cc952861e04621a4bd",
    )
    if not os.path.isdir(hf_model):
        raise RuntimeError(f"HF_MODEL path does not exist: {hf_model}")
    os.environ["HF_MODEL"] = hf_model

    # ================================================================
    # 1. Open mesh device with trace region
    # ================================================================
    print(f"Opening mesh device (batch={batch_size}) ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        trace_region_size=100_000_000,
    )
    print(f"Mesh device opened: {mesh_device.get_num_devices()} devices")

    # ================================================================
    # 2. Build model with paged attention for batch>1
    # ================================================================
    if batch_size > 1:
        # KV cache only serves the 16 full_attention layers (not DeltaNet).
        # Cap max_num_blocks to fit DRAM; for decode benchmarks we only use
        # a few positions so 2048 blocks (128K tokens capacity) is sufficient.
        paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=2048)
    else:
        paged_config = None

    tt_model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=None,
        max_seq_len=args_cli.max_seq_len,
        paged_attention_config=paged_config,
        dtype=ttnn.bfloat4_b if args_cli.bfp4 else ttnn.bfloat8_b,
        use_prefetcher=args_cli.prefetcher,
    )
    print(f"Qwen3.5-27B: {tt_model_args.n_layers} layers, batch={batch_size}, " f"devices={tt_model_args.num_devices}")

    B_pad = tt_model_args.tile_padded_batch_rows

    # ================================================================
    # 3. Create page table for paged attention
    # ================================================================
    if paged_config:
        # Each user gets contiguous blocks, capped to fit max_num_blocks
        blocks_per_user = min(
            args_cli.max_seq_len // paged_config.block_size,
            paged_config.max_num_blocks // batch_size,
        )
        page_table = torch.zeros(batch_size, blocks_per_user, dtype=torch.int32)
        for b in range(batch_size):
            for p in range(blocks_per_user):
                page_table[b, p] = b * blocks_per_user + p
    else:
        page_table = None

    # ================================================================
    # 4. Prepare input tensors
    # ================================================================
    tokens = torch.randint(100, 50000, (batch_size,), dtype=torch.int64)
    current_pos = torch.zeros(batch_size, dtype=torch.long)

    # Use model's prepare_decode_inputs_host for proper formatting
    host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)
    tt_tokens, tt_current_pos, tt_rot_idxs, tt_page_table = host_inputs

    # Transfer to device
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    tt_tokens_dev, tt_current_pos_dev, tt_rot_idxs_dev, tt_page_table_dev = device_inputs

    # ================================================================
    # 5. Warmup forward pass
    # ================================================================
    # Initialize prefetcher for decode mode (must be before first forward call)
    if args_cli.prefetcher:
        from models.tt_transformers.tt.common import Mode

        model.switch_mode(Mode.DECODE)
        print("Prefetcher initialized for decode mode")

    print("Warmup (kernel compilation) ...")
    t_warmup = time.time()
    rot_mats = model.rope_setup.get_rot_mats(tt_rot_idxs_dev)
    tt_out = model.ttnn_decode_forward(
        tt_tokens_dev,
        tt_current_pos_dev,
        tt_rot_idxs_dev,
        page_table=tt_page_table_dev,
        kv_cache=tt_kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    print(f"Warmup done in {time.time() - t_warmup:.2f}s")

    # ================================================================
    # 6. Trace capture
    # ================================================================
    if not args_cli.no_trace:
        print("Capturing trace ...")
        # Re-prepare inputs for trace (need fresh buffers)
        current_pos += 1
        host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
        tt_tokens_dev, tt_current_pos_dev, tt_rot_idxs_dev, tt_page_table_dev = device_inputs

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        rot_mats = model.rope_setup.get_rot_mats(tt_rot_idxs_dev)
        tt_out_trace = model.ttnn_decode_forward(
            tt_tokens_dev,
            tt_current_pos_dev,
            tt_rot_idxs_dev,
            page_table=tt_page_table_dev,
            kv_cache=tt_kv_cache,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        print("Trace captured!")

    # ================================================================
    # 7. Decode loop
    # ================================================================
    decode_steps = args_cli.decode_steps
    print(f"Running {decode_steps} decode steps ...")

    times = []
    for step in range(decode_steps):
        current_pos += 1
        host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)

        if not args_cli.no_trace:
            # Update trace input buffers in-place
            copy_host_to_device(host_inputs, device_tensors=device_inputs)
            t0 = time.time()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = time.time() - t0
        else:
            device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
            tt_tokens_dev, tt_current_pos_dev, tt_rot_idxs_dev, tt_page_table_dev = device_inputs
            t0 = time.time()
            rot_mats = model.rope_setup.get_rot_mats(tt_rot_idxs_dev)
            tt_out = model.ttnn_decode_forward(
                tt_tokens_dev,
                tt_current_pos_dev,
                tt_rot_idxs_dev,
                page_table=tt_page_table_dev,
                kv_cache=tt_kv_cache,
            )
            ttnn.synchronize_device(mesh_device)
            dt = time.time() - t0

        times.append(dt)
        tps_user = 1.0 / dt
        tps_total = batch_size / dt
        print(f"  Step {step}: {dt*1000:.1f}ms ({tps_user:.1f} tok/s/user, {tps_total:.0f} total tok/s)", flush=True)

    # ================================================================
    # 8. Report results
    # ================================================================
    steady = times[1:] if len(times) > 1 else times
    avg_tpot = sum(steady) / len(steady)
    tok_per_sec_user = 1.0 / avg_tpot
    tok_per_sec_total = batch_size / avg_tpot

    print("\n" + "=" * 60)
    print(f"Qwen3.5-27B Decode Benchmark Results (P300x2, 4 BH chips)")
    print(f"=" * 60)
    print(f"  Batch size:          {batch_size}")
    print(f"  Decode steps:        {decode_steps}")
    print(f"  Avg TPOT:            {avg_tpot * 1000:.2f} ms")
    print(f"  Tokens/sec/user:     {tok_per_sec_user:.2f}")
    print(f"  Total throughput:    {tok_per_sec_total:.2f} tok/s")
    print(f"  Trace:               {'enabled' if not args_cli.no_trace else 'disabled'}")
    print(f"=" * 60)

    # Cleanup
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("Done.")


if __name__ == "__main__":
    main()
