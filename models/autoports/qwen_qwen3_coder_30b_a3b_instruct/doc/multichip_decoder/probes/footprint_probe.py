# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Allocate the whole 48-layer model's per-die footprint, instead of computing it.

`mesh_plan.md` §9 lists this as the one capacity claim it did not measure: "the
48-layer footprint is computed, not allocated. A load-time probe that actually
allocates 48 layers' worth of per-die weights should be run in implementation
before the contract file is updated." This is that probe, and
`doc/context_contract.json` quotes it rather than the arithmetic.

Three stages, each reporting the allocator's own view of DRAM:

  1. 48 layers of sharded decoder weights (the same layer 0 tensors uploaded 48
     times -- the shapes and dtypes are what matter, not the values).
  2. embed_tokens replicated + lm_head column-parallel, the two tensors the full
     model adds outside the decoder stack.
  3. paged KV cache, walked out towards the advertised 262144 context, one KV
     head per die.

    python footprint_probe.py [max_context]

Prints ``P|`` lines only.
"""
import sys

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

MAX_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
BLOCK = 32

hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)
N_LAYERS = hf.num_hidden_layers
VOCAB = hf.vocab_size

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=32768)


def dram_used_gb():
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return v.total_bytes_allocated_per_bank * v.num_banks / 1e9, v.total_bytes_per_bank * v.num_banks / 1e9


try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    base, total = dram_used_gb()
    print(f"P|DRAM per die: {total:.2f} GB total, {base:.3f} GB already in use", flush=True)

    # Allocate the per-die shapes directly, 48 times, from host tensors built
    # once. Calling upload_multichip_weights in a loop would re-run its float()
    # conversion of a 402M-element expert tensor on every iteration, which is
    # minutes of host work per layer and measures numpy, not DRAM.
    a, m = cfg.local_attention, cfg.local_moe
    from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (
        ATTENTION_WEIGHT_DTYPE,
        EXPERT_WEIGHT_DTYPE,
    )

    H = hf.hidden_size
    per_die = [
        ((1, m.num_experts, H, 2 * m.moe_intermediate_size), EXPERT_WEIGHT_DTYPE),  # gate_up
        ((1, m.num_experts, m.moe_intermediate_size, H), EXPERT_WEIGHT_DTYPE),  # down
        ((1, 1, H, (a.num_attention_heads + 2 * a.num_key_value_heads) * a.head_dim), ATTENTION_WEIGHT_DTYPE),
        ((1, 1, H, (a.num_attention_heads + 2 * a.num_key_value_heads) * a.head_dim), ATTENTION_WEIGHT_DTYPE),
        ((1, 1, a.num_attention_heads * a.head_dim, H), ATTENTION_WEIGHT_DTYPE),  # wo, prefill copy
        ((1, 1, a.num_attention_heads * a.head_dim, H), ATTENTION_WEIGHT_DTYPE),  # wo, decode copy
        ((1, 1, H, hf.num_experts), ttnn.bfloat16),  # router, replicated
        ((1, 1, 1, H), ttnn.bfloat16),
        ((1, 1, 1, H), ttnn.bfloat16),
    ]
    hosts = [(torch.zeros(*shape), dt) for shape, dt in per_die]
    print("P|per-die tensors per layer: " + ", ".join(f"{tuple(t.shape)}" for t, _ in hosts), flush=True)

    held = []
    for layer in range(N_LAYERS):
        held.append(
            [
                ttnn.from_torch(
                    t,
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                for t, dt in hosts
            ]
        )
        if (layer + 1) % 12 == 0:
            used, _ = dram_used_gb()
            print(f"P|after {layer + 1:>2} layers of sharded weights: {used - base:7.3f} GB/die", flush=True)
    weights_gb = dram_used_gb()[0] - base

    embed = ttnn.from_torch(
        torch.zeros(1, 1, VOCAB, hf.hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    lm_head = ttnn.from_torch(
        torch.zeros(1, 1, hf.hidden_size, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    with_embed = dram_used_gb()[0] - base
    print(
        f"P|48 layers = {weights_gb:.3f} GB/die; + embed (replicated) and lm_head "
        f"(column-parallel, per-die N = {VOCAB // 4}) = {with_embed:.3f} GB/die",
        flush=True,
    )

    # Now actually allocate all 48 layers' KV at the advertised context, which
    # is the claim doc/context_contract.json makes and which mesh_plan.md only
    # computed.
    per_layer_token_bytes = cfg.local_attention.num_key_value_heads * cfg.local_attention.head_dim * 2 * 2
    caches = []
    try:
        for layer in range(N_LAYERS):
            caches.append(MC.create_mesh_kv_cache(mesh, cfg, 1, MAX_CTX, block_size=BLOCK))
            if (layer + 1) % 12 == 0:
                used, _ = dram_used_gb()
                print(
                    f"P|+ paged KV for {layer + 1:>2} layers at ctx {MAX_CTX}: {used - base:7.3f} GB/die total",
                    flush=True,
                )
        used, _ = dram_used_gb()
        print(
            f"P|ALL {N_LAYERS} layers' weights + KV at ctx {MAX_CTX}, batch 1: "
            f"{used - base:.3f} GB/die of {total:.2f} GB, {total - (used - base):.3f} GB free",
            flush=True,
        )
    except Exception as exc:
        print(f"P|KV allocation FAILED after {len(caches)} layers: {str(exc)[:160]}", flush=True)

    kv_all = N_LAYERS * MAX_CTX * per_layer_token_bytes
    print(
        f"P|per-die KV arithmetic: {per_layer_token_bytes} B/token/layer, "
        f"{N_LAYERS} layers x {MAX_CTX} tokens = {kv_all / 1e9:.3f} GB/die",
        flush=True,
    )
    print(
        f"P|per-die total at full context, batch 1 = {with_embed + kv_all / 1e9:.3f} GB of {total:.2f} GB", flush=True
    )
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
