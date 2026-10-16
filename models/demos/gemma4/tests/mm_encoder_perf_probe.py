# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timing of the on-device multimodal encoders on REAL processor outputs.

Times, with device sync, N calls of: the vision tower (31B: SigLIP blocks at the
padded encoder seq len, then the projector; 12B: the encoder-free unified embedder),
``encode_vision`` end to end (incl. the valid-token gather), the device->host copy of
the soft tokens, and (12B) the audio projection on the clip's waveform frames.
Writes a JSON summary to ``$PROBE_OUT`` so the numbers can be set against the
roofline floors (weights / DRAM bandwidth, FLOPs / matmul peak).

    HF_MODEL=google/gemma-4-31B-it PROBE_OUT=/tmp/mm_perf_31b.json MESH_DEVICE=P150x4 \\
    TT_VISIBLE_DEVICES=12,13,14,15 pytest models/demos/gemma4/tests/mm_encoder_perf_probe.py -s
"""
import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.vision_demo import _device_params
from models.demos.gemma4.tt.generator import Gemma4Generator, _vision_encoder_seq_len

IMG = "/home/ttuser/ashai/gemma4/mm_samples/shapes.png"
WAV = "/home/ttuser/ashai/gemma4/mm_samples/journal1.wav"


def _timed(mesh_device, fn, iters, warmup=2):
    for _ in range(warmup):
        out = fn()
        ttnn.synchronize_device(mesh_device)
        if hasattr(out, "deallocate"):
            out.deallocate(True)
    times = []
    for _ in range(iters):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
        if hasattr(out, "deallocate"):
            out.deallocate(True)
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times), "min_ms": min(times), "n": iters}


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_mm_encoder_perf(mesh_device):
    from transformers import AutoProcessor

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    iters = int(os.environ.get("PROBE_ITERS", "10"))
    out_path = os.environ.get("PROBE_OUT", f"/tmp/mm_perf_{model_path.split('/')[-1]}.json")
    processor = AutoProcessor.from_pretrained(model_path)
    msgs = [
        {"role": "user", "content": [{"type": "image", "image": IMG}, {"type": "text", "text": "Describe this image."}]}
    ]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False
    )
    pv = inputs["pixel_values"].to(torch.bfloat16)
    pos = inputs["image_position_ids"].to(torch.long)
    num_patches = int(pos.shape[1])
    result = {
        "model": model_path,
        "mesh": list(mesh_device.shape),
        "num_patches": num_patches,
        "pixel_values": list(pv.shape),
    }
    logger.info(f"image: {num_patches} patches, pixel_values {tuple(pv.shape)}")

    generator, _kv, _tok = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device, model_path=model_path, max_batch_size=1, max_seq_len=4096, multimodal=True
    )
    tower = generator.vision_tower

    if os.environ.get("PROBE_OPSPLIT") == "1":
        # Host-timed op split: every ttnn op the tower uses is wrapped with a
        # device sync before and after, so each call's wall time is (close to) its
        # device time plus one dispatch. Serializes the tower (the total is above
        # the unpatched one) but attributes the time per op type without the
        # device profiler, whose marker post-processing aborts on this build.
        if getattr(tower, "projects_to_text", False):
            call = lambda: tower(pv, pos)[0]
        else:
            seq_len = _vision_encoder_seq_len(num_patches)
            call = lambda: tower(pv, pos, seq_len)[0]
        for _ in range(2):  # compile + warm, unpatched
            o = call()
            ttnn.synchronize_device(mesh_device)
            o.deallocate(True)
        base = _timed(mesh_device, call, 3, warmup=0)
        acc = {}
        targets = {
            "linear": (ttnn, "linear"),
            "matmul": (ttnn, "matmul"),
            "sdpa": (ttnn.transformer, "scaled_dot_product_attention"),
            "all_reduce": (ttnn, "all_reduce"),
            "rms_norm": (ttnn, "rms_norm"),
            "layer_norm": (ttnn, "layer_norm"),
            "rotary_embedding_llama": (ttnn.experimental, "rotary_embedding_llama"),
            "nlp_create_qkv_heads": (ttnn.experimental, "nlp_create_qkv_heads"),
            "nlp_concat_heads": (ttnn.experimental, "nlp_concat_heads"),
            "split": (ttnn, "split"),
            "concat": (ttnn, "concat"),
            "pad": (ttnn, "pad"),
            "reshape": (ttnn, "reshape"),
            "typecast": (ttnn, "typecast"),
            "add": (ttnn, "add"),
            "multiply": (ttnn, "multiply"),
            "to_memory_config": (ttnn, "to_memory_config"),
            "to_layout": (ttnn, "to_layout"),
            "from_torch": (ttnn, "from_torch"),
            "embedding": (ttnn, "embedding"),
            "slice": (ttnn, "slice"),
        }
        originals = {}
        for key, (mod, attr) in targets.items():
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            originals[key] = (mod, attr, fn)

            def _mk(key, fn):
                def _w(*a, **k):
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    out = fn(*a, **k)
                    ttnn.synchronize_device(mesh_device)
                    acc.setdefault(key, [0.0, 0])
                    acc[key][0] += (time.perf_counter() - t0) * 1e3
                    acc[key][1] += 1
                    return out

                return _w

            setattr(mod, attr, _mk(key, fn))
        n_calls = 2
        try:
            for _ in range(n_calls):
                o = call()
                ttnn.synchronize_device(mesh_device)
                o.deallocate(True)
        finally:
            for key, (mod, attr, fn) in originals.items():
                setattr(mod, attr, fn)
        split = {k: {"ms_per_call": v[0] / n_calls, "ops_per_call": v[1] / n_calls} for k, v in acc.items()}
        serialized_total = sum(v["ms_per_call"] for v in split.values())
        result = {
            "model": model_path,
            "num_patches": num_patches,
            "unpatched_tower_ms": base,
            "serialized_total_ms": serialized_total,
            "split": split,
        }
        json.dump(result, open(out_path, "w"), indent=1)
        for k, v in sorted(split.items(), key=lambda x: -x[1]["ms_per_call"]):
            logger.info(f"OPSPLIT {k:24s} {v['ms_per_call']:8.2f} ms/call  {v['ops_per_call']:6.0f} ops/call")
        logger.info("PROBE_RESULT " + json.dumps({"unpatched_tower_ms": base, "serialized_total_ms": serialized_total}))
        return

    if os.environ.get("PROBE_PROFILE") == "1":
        # Device-profiler split of ONE encoder call (TT_METAL_DEVICE_PROFILER=1), the
        # decode-roofline way: graph-capture the op order once, flush the device
        # profiler after every call, then name the last N x ops rows by order
        # (``decode_roofline_report.py --ops-json ... --steps N``).
        from models.demos.gemma4.tests.unit.test_decode_roofline import _device_ops_in_order

        if getattr(tower, "projects_to_text", False):
            call = lambda: tower(pv, pos)[0]
        else:
            seq_len = _vision_encoder_seq_len(num_patches)
            call = lambda: tower(pv, pos, seq_len)[0]

        def _flush():
            try:
                ttnn.ReadDeviceProfiler(mesh_device)
            except Exception:
                pass

        # One encoder call overflows the per-core profiler buffers (27 blocks x ~46
        # device ops): flush after every block so no marker is dropped. The sync
        # inflates wall time only; device durations are unaffected.
        enc = next((v for v in vars(tower).values() if isinstance(getattr(v, "blocks", None), list)), None)
        if enc is not None:
            for blk in enc.blocks:
                orig = blk.forward

                def _wrapped(*a, _orig=orig, **k):
                    out = _orig(*a, **k)
                    ttnn.synchronize_device(mesh_device)
                    _flush()
                    return out

                blk.forward = _wrapped
            logger.info(f"profile mode: flushing the device profiler after each of {len(enc.blocks)} blocks")
        for _ in range(2):  # compile + warm
            o = call()
            ttnn.synchronize_device(mesh_device)
            o.deallocate(True)
        _flush()
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        o = call()
        ttnn.synchronize_device(mesh_device)
        captured = ttnn.graph.end_graph_capture()
        o.deallocate(True)
        ops = _device_ops_in_order(captured)
        ops_json = out_path.replace(".json", "_ops.json")
        json.dump({"ops": ops}, open(ops_json, "w"))
        logger.info(f"captured {len(ops)} graph ops for one encoder call -> {ops_json}")
        _flush()
        for _ in range(iters):
            o = call()
            ttnn.synchronize_device(mesh_device)
            _flush()
            o.deallocate(True)
        _flush()
        json.dump({"model": model_path, "profiled_calls": iters, "ops_json": ops_json}, open(out_path, "w"))
        logger.info("PROBE_RESULT " + json.dumps({"profiled_calls": iters, "ops_json": ops_json}))
        return
    if getattr(tower, "projects_to_text", False):
        result["vision_kind"] = "unified_embedder"
        result["embedder_fwd"] = _timed(mesh_device, lambda: tower(pv, pos)[0], iters)
    else:
        seq_len = _vision_encoder_seq_len(num_patches)
        result["vision_kind"] = "siglip_tower"
        result["encoder_seq_len"] = seq_len
        result["tower_fwd"] = _timed(mesh_device, lambda: tower(pv, pos, seq_len)[0], iters)
        pooled_tt, _mask = tower(pv, pos, seq_len)
        ttnn.synchronize_device(mesh_device)
        result["projector_fwd"] = _timed(mesh_device, lambda: generator.embed_vision(pooled_tt), iters)
    result["encode_vision_e2e"] = _timed(mesh_device, lambda: generator.encode_vision(pv, pos), iters)
    embeds_tt = generator.encode_vision(pv, pos)
    ttnn.synchronize_device(mesh_device)
    result["soft_tokens"] = list(embeds_tt.shape)

    def _to_host():
        t = ttnn.get_device_tensors(embeds_tt)[0] if hasattr(mesh_device, "shape") else embeds_tt
        return ttnn.to_torch(t)

    result["soft_tokens_to_host"] = _timed(mesh_device, _to_host, iters)

    # 12B unified: audio projection on the clip's waveform frames.
    try:
        from safetensors import safe_open

        from models.demos.gemma4.tt.audio.unified_audio_embedder import PROJECTION_KEYS, Gemma4UnifiedAudioEmbedder
        from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

        hf_config = Gemma4ModelArgs.load_hf_config(model_path)
        if getattr(hf_config, "audio_config", None) is not None:
            amsgs = [
                {"role": "user", "content": [{"type": "text", "text": "Transcribe."}, {"type": "audio", "audio": WAV}]}
            ]
            ain = processor.apply_chat_template(
                amsgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            feats, amask = ain["input_features"], ain["input_features_mask"]
            result["audio_frames"] = list(feats.shape)
            snap = Gemma4ModelArgs(model_path).model_base_path if hasattr(Gemma4ModelArgs, "model_base_path") else None
            weight = None
            import glob

            for f in glob.glob(
                os.path.expanduser(
                    "~/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/*/model.safetensors"
                )
            ):
                with safe_open(f, "pt") as sf:
                    if PROJECTION_KEYS[0] in sf.keys():
                        weight = sf.get_tensor(PROJECTION_KEYS[0])
            emb = Gemma4UnifiedAudioEmbedder(
                mesh_device, {PROJECTION_KEYS[0]: weight}, text_hidden_size=int(hf_config.text_config.hidden_size)
            )
            result["audio_embed_fwd"] = _timed(mesh_device, lambda: emb.forward(feats, amask)[0], iters)
    except Exception as e:  # keep the vision numbers even if the audio probe cannot run
        result["audio_error"] = repr(e)
        logger.warning(f"audio probe skipped: {e!r}")

    json.dump(result, open(out_path, "w"), indent=1)
    logger.info("PROBE_RESULT " + json.dumps(result))
