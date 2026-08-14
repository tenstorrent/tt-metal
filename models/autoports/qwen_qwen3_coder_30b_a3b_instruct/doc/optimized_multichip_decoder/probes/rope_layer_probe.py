# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why the 3.05x Meta rotary cannot be adopted for decode alone. The measurement.

``rope_probe.py`` showed ``rotary_embedding_llama`` is 3.05x faster than the
shipped HF rotary and **bit-identical** at the shipped shape. Wiring it into the
layer nonetheless dropped ``test_multichip_decode_vs_single_chip`` to PCC 0.876.
This probe finds out why, and the answer is not the rotary:

    fresh KV cache          PCC 0.9999697   <- the rotary is right
    prefill-primed cache    PCC 0.1932974   <- the *cache* is the problem

RoPE is applied before K is written, so the cache inherits the rotary's channel
convention. Prefill is untouched by this lever and writes **HF**-ordered keys;
a Meta-ordered decode Q then scores against them and SDPA's dot products are
meaningless. With a fresh cache there are no HF keys to conflict with, which is
exactly why the op-level probe looked clean. Adopting the llama op for decode
therefore means adopting it for **prefill too**, plus the KV-cache channel
convention -- not a decode-local change. See ``README.md`` limitation 4.

It also runs the real ``attention_decode_optimized`` twice on one mesh -- HF
weights + HF rope, then Meta weights + llama rope -- and prints the shapes and
memory configs the rotary op actually sees for Q and K.

    python rope_layer_probe.py

Prints ``P|`` lines only.
"""
import sys

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tests.reference import layer_state_dict, load_config
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import functional_decoder as F
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import multichip_decoder as MC
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt import optimized_decoder as O
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import convert_layer_weights

CTX = 128
hf = load_config()
tw = convert_layer_weights(layer_state_dict(0), hf)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=90_000_000, l1_small_size=32768)

_apply = F._apply_rope
seen = []


def spy(tag, inner):
    def fn(t, cos, sin, token_index):
        out = inner(t, cos, sin, token_index)
        seen.append(
            f"P|  {tag} in  shape={list(t.shape)} padded={list(t.padded_shape)} "
            f"{'L1' if t.memory_config().buffer_type == ttnn.BufferType.L1 else 'DRAM'} "
            f"{t.memory_config().memory_layout}"
        )
        seen.append(f"P|  {tag} out shape={list(out.shape)} padded={list(out.padded_shape)}")
        return out

    return fn


try:
    cfg = MC.MeshDecoderConfig.from_hf(hf)
    ctx = MC.mesh_context(mesh)
    weights = MC.upload_multichip_weights(tw, mesh, cfg, meta_rope=True)
    cos, sin = F.build_rope_cache(hf, 1024, mesh)
    kv_hf = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)
    kv_mt = MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)

    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, 1, 1, hf.hidden_size) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pos = ttnn.from_torch(
        torch.tensor([CTX - 1], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

    # --- leg A: shipped ------------------------------------------------------
    n = len(seen)
    seen.append("P|HF leg:")
    a = O.attention_decode_optimized(
        x,
        weights.experts,
        cfg.local_attention,
        cos,
        sin,
        kv_hf,
        pos,
        CTX - 1,
        sdpa_program_config=MC._sdpa_program_config(mesh),
        rope=spy("hf ", _apply),
    )
    ref = ttnn.to_torch(a, mesh_composer=comp).float()

    # --- leg B: Meta weights + llama rope ------------------------------------
    seen.append("P|Meta leg:")
    meta_rope = MC._meta_rope(ctx, cos, sin, cfg.local_attention.head_dim)
    b = O.attention_decode_optimized(
        x,
        weights.experts_meta,
        cfg.local_attention,
        cos,
        sin,
        kv_mt,
        pos,
        CTX - 1,
        sdpa_program_config=MC._sdpa_program_config(mesh),
        rope=spy("mt ", meta_rope),
    )
    got = ttnn.to_torch(b, mesh_composer=comp).float()

    for line in seen:
        print(line, flush=True)
    d = (got - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(f"P|attention out: max|diff| {d:.3e}  PCC {pcc:.7f}", flush=True)

    # Is the divergence in the rotary itself, or downstream? Compare the two
    # KV caches: k is written straight out of the rotary, so if the caches
    # disagree after permuting the Meta one back, the rotary is the problem.
    from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.weight_mapping import hf_to_meta_channels

    inv = torch.argsort(hf_to_meta_channels(cfg.local_attention.head_dim))
    kh = ttnn.to_torch(kv_hf.k, mesh_composer=comp).float()
    km = ttnn.to_torch(kv_mt.k, mesh_composer=comp).float()
    kd = (km[..., inv] - kh).abs().max().item()
    print(f"P|k cache (Meta permuted back) vs HF: max|diff| {kd:.3e}  shape {list(kh.shape)}", flush=True)
    vh = ttnn.to_torch(kv_hf.v, mesh_composer=comp).float()
    vm = ttnn.to_torch(kv_mt.v, mesh_composer=comp).float()
    print(f"P|v cache vs v cache: max|diff| {(vm - vh).abs().max().item():.3e}", flush=True)

    # --- the decisive leg: a cache primed by PREFILL --------------------------
    # Prefill is untouched by this lever and therefore writes **HF-ordered** keys.
    # SDPA scores a Meta-ordered Q against every key already in the cache, so the
    # question is whether the two conventions can coexist in one cache. Above,
    # both caches were fresh (only the one new key, everything else zero), which
    # is why the Meta leg looked right. Prime them and re-ask.
    print("P|", flush=True)
    print("P|--- with a cache primed by prefill (the shipped harness order) ---", flush=True)
    sparsity = MC.build_local_sparsity(mesh, cfg.local_moe)
    PROMPT = 32
    torch.manual_seed(1)
    full = torch.randn(1, 1, PROMPT + 1, hf.hidden_size) * 0.02

    def rep4(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    tok = rep4(full[:, :, PROMPT : PROMPT + 1, :])
    pos2 = ttnn.from_torch(
        torch.tensor([PROMPT], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    outs = {}
    for tag, kvc in (
        ("HF", MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)),
        ("Meta", MC.create_mesh_kv_cache(mesh, cfg, 1, 1024, block_size=32)),
    ):
        # Prefill: identical in both legs, HF weights and HF rope, by design.
        MC.decoder_layer_prefill_multichip(rep4(full[:, :, :PROMPT, :]), weights, cfg, ctx, cos, sin, sparsity, kvc)
        ttnn.synchronize_device(mesh)
        w = weights.experts if tag == "HF" else weights.experts_meta
        r = None if tag == "HF" else MC._meta_rope(ctx, cos, sin, cfg.local_attention.head_dim)
        o = O.attention_decode_optimized(
            tok,
            w,
            cfg.local_attention,
            cos,
            sin,
            kvc,
            pos2,
            PROMPT,
            sdpa_program_config=MC._sdpa_program_config(mesh),
            rope=r,
        )
        outs[tag] = ttnn.to_torch(o, mesh_composer=comp).float()
    d2 = (outs["Meta"] - outs["HF"]).abs().max().item()
    p2 = torch.corrcoef(torch.stack([outs["Meta"].flatten(), outs["HF"].flatten()]))[0, 1].item()
    print(f"P|attention out, primed cache: max|diff| {d2:.3e}  PCC {p2:.7f}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
