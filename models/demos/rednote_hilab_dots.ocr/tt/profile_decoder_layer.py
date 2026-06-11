# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtDecoderLayer at the PRODUCTION operating point.

Production context: the fp32 text decoder runs the layer on a REPLICATED
[1, 1, 128, 1536] fp32 residual stream on the 1x4 mesh, real model.layers.0
weights, with host-prepared rope tables and causal mask (persistent device
tensors created once outside the trace, like the production AR loop).

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_decoder_layer.py --traced

Decode operating point (production traced token step, ocr_model
_decode_step_traced — the hot path: 28 layers x every generated token):
bf16 token row [1, 1, 1, 1536] through forward_decode against persistent
bf16 KV caches (capacity 3200), slot ~1500, production weight dtypes
(attn/MLP weights bfloat8_b) — replayed as a metal trace. Run with
``--decode --traced``.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TT_DIR.parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_decoder_layer", _TT_DIR / "decoder_layer.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtDecoderLayer = _mod.TtDecoderLayer

REPO = "rednote-hilab/dots.ocr"
HIDDEN = 1536

LAYER_KEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def _load_weights(prefix, keys):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in keys:
        full = f"{prefix}.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--decode", action="store_true", help="KV-cached decode token step instead of prefill")
    parser.add_argument("--iters", type=int, default=1, help="profiled replay count")
    args = parser.parse_args()

    golden = torch.load(_MODEL_DIR / "reference" / "golden" / "decoder_layer.pt")
    x = golden["input"]
    cos, sin = golden["cos"], golden["sin"]
    _, seq, dim = x.shape
    assert dim == HIDDEN

    sd = _load_weights("model.layers.0", LAYER_KEYS)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=80_000_000,
    )
    try:
        if args.decode:
            # Production decode posture (ocr_model): fp32 attention-path
            # activations, bfloat8_b attn/MLP weight storage, bf16 residual.
            layer = TtDecoderLayer(
                mesh_device,
                sd,
                num_heads=12,
                num_kv_heads=2,
                dtype=ttnn.float32,
                mlp_dtype=ttnn.bfloat8_b,
                attn_weight_dtype=ttnn.bfloat8_b,
                mlp_gate_up_dtype=ttnn.bfloat8_b,
            )
            MAX_SEQ, SLOT = 3200, 1500
            kv_cache = layer.init_kv_cache(MAX_SEQ)
            pos_host = ttnn.from_torch(torch.tensor([SLOT], dtype=torch.int32), dtype=ttnn.int32)
            ttnn.copy_host_to_device_tensor(pos_host, kv_cache["pos"])
            rot_step = layer.prepare_decode_rope(SLOT)
            x_tt = ttnn.from_torch(
                x[:, :1, :].reshape(1, 1, 1, dim).float(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(3):
                out = layer.forward_decode(x_tt, kv_cache, rot_step)
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh_device)
            if args.traced:
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                out = layer.forward_decode(x_tt, kv_cache, rot_step)
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh_device)
                for _ in range(max(1, args.iters)):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh_device)
                ttnn.release_trace(mesh_device, tid)
            else:
                out = layer.forward_decode(x_tt, kv_cache, rot_step)
                ttnn.synchronize_device(mesh_device)
            print("profiled iteration complete (decode, traced=%s, slot=%d)" % (args.traced, SLOT))
            return

        # Production dtype: the whole layer runs fp32 (attention path mandate).
        layer = TtDecoderLayer(mesh_device, sd, num_heads=12, num_kv_heads=2)

        # Persistent inputs: stable addresses for trace replay.
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, seq, dim).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rot_mats = layer.prepare_rope(cos, sin)
        causal_mask = layer.prepare_causal_mask(seq)

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = layer.forward(x_tt, rot_mats, causal_mask)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = layer.forward(x_tt, rot_mats, causal_mask)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replays.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            for _ in range(max(1, args.iters)):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = layer.forward(x_tt, rot_mats, causal_mask)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s, seq=%d)" % (args.traced, seq))
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
