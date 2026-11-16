# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast per-change check for the Gemma 4 26B perf plan (perf/PLAN.md, "Quick check"), ~2-4 min on one QB2.

Builds the model with real layers 0 (sliding) and 5 (full) only and checks:
  1. accuracy: prefill (65 tokens) and traced decode vs the HF layer, from references cached on disk;
  2. batched vs one-user-at-a-time decode at --check-batch users with mixed lengths (per-user PCC, KV caches);
  3. traced decode time per layer at each --batches size.
Compares with --baseline (PCC may drop at most 0.0005, time may grow at most 3%) and exits 1 on failure.

  python models/demos/gemma4_26b_a4b_qb2/perf/tools/quick_check.py --baseline <json> [--save-baseline]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION, Checkpoint

LENGTH = 65  # prefill tokens; the 66th is decoded
LAYERS = (0, 5)
TOKENS = Path(__file__).resolve().parents[2] / "tests" / "decoder_input_tokens.json"


def reference_layer(config, index):
    with torch.device("meta"):
        layer = Gemma4TextDecoderLayer(config, index)
    state = Checkpoint().load(f"model.language_model.layers.{index}.")
    layer.load_state_dict({k: v.float() for k, v in state.items()}, assign=True)
    return layer.eval()


@torch.inference_mode()
def build_refs(path):
    """HF inputs for layers 0 and 5 (layer 5 gets the real output of layers 0-4) and each layer's outputs."""
    fixture = json.loads(TOKENS.read_text())
    assert fixture["revision"] == REVISION
    tokens = torch.tensor(fixture["token_ids"])
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION).text_config
    config._attn_implementation = "eager"
    embedding = Checkpoint().load("model.language_model.embed_tokens.")["weight"]
    hidden = embedding[tokens].float()[None] * torch.tensor(config.hidden_size**0.5, dtype=torch.float32)
    del embedding
    n = LENGTH + 1
    causal = torch.zeros(n, n).masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), -torch.inf)[None, None]
    rotary = Gemma4TextRotaryEmbedding(config)
    inputs = {0: hidden.bfloat16().float()}
    for index in range(5):
        layer = reference_layer(config, index)
        rope = rotary(hidden, torch.arange(n)[None], layer_type=config.layer_types[index])
        hidden = layer(hidden, position_embeddings=rope, attention_mask=causal)
    inputs[5] = hidden.bfloat16().float()
    refs = {"revision": REVISION, "inputs": inputs, "prefill": {}, "decode": {}}
    for index in LAYERS:
        x, layer, kind = inputs[index], reference_layer(config, index), config.layer_types[index]
        cos, sin = rotary(x, torch.arange(128)[None], layer_type=kind)
        cache = DynamicCache()
        refs["prefill"][index] = layer(
            x[:, :LENGTH],
            position_embeddings=(cos[:, :LENGTH], sin[:, :LENGTH]),
            attention_mask=causal[..., :LENGTH, :LENGTH],
            past_key_values=cache,
        )
        refs["decode"][index] = layer(
            x[:, LENGTH:],
            position_embeddings=(cos[:, LENGTH : LENGTH + 1], sin[:, LENGTH : LENGTH + 1]),
            attention_mask=torch.zeros(1, 1, 1, n),
            past_key_values=cache,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refs, path)
    return refs


def host(tensor):
    replicas = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], r) for r in replicas[1:]), "replicas differ"
    return replicas[0]


def per_device(cache):
    """KV cache contents chip by chip (each chip holds its own heads, so replicas legitimately differ)."""
    return [ttnn.to_torch(d).float() for tensor in cache for d in ttnn.get_device_tensors(tensor)]


def pcc(expected, actual):
    return float(comp_pcc(expected.float(), actual.float(), 0.0)[1])


def decode_args(model, hidden, positions, table):
    """Device inputs of a decode call: hidden [1,1,B,H], current_pos uint32 [1,B], cache_pos int32 [B], table [B,P]."""
    return dict(
        hidden=model.upload(hidden),
        current_pos=model.upload(positions[None].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        cache_pos=model.upload(positions.int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        page_table=model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    )


def run_decode(model, number, kind, cache, a):
    return model.layers[number].decode_forward(
        a["hidden"],
        rope_mats=model.rope_decode[kind],
        current_pos=a["current_pos"],
        cache_pos=a["cache_pos"],
        page_table=a["page_table"],
        kv_cache=cache,
    )


def accuracy(model, refs):
    out = {}
    for number, index in enumerate(LAYERS):
        kind, x = model.config.layer_types[index], refs["inputs"][index]
        caches, table = model.allocate_cache(slots=1, context=128)
        cache = caches[number]
        device_table = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        actual = model.layers[number].prefill_forward(
            model.upload(x[:, :LENGTH][None]),
            rope_mats=model.rope_prefill[kind],
            page_table=device_table,
            kv_cache=cache,
        )
        out[f"L{index}_prefill"] = pcc(refs["prefill"][index], host(actual).squeeze(0))
        a = decode_args(model, x[:, LENGTH:][None], torch.tensor([LENGTH]), table)
        run_decode(model, number, kind, cache, a)  # compile
        ttnn.synchronize_device(model.mesh)
        # Re-prefill so the traced step decodes from the same cache state as the reference.
        model.layers[number].prefill_forward(
            model.upload(x[:, :LENGTH][None]),
            rope_mats=model.rope_prefill[kind],
            page_table=device_table,
            kv_cache=cache,
        )
        trace = ttnn.begin_trace_capture(model.mesh, cq_id=0)
        actual = run_decode(model, number, kind, cache, a)
        ttnn.end_trace_capture(model.mesh, trace, cq_id=0)
        try:
            ttnn.execute_trace(model.mesh, trace, cq_id=0, blocking=True)
            out[f"L{index}_decode"] = pcc(refs["decode"][index], host(actual).squeeze(0))
        finally:
            ttnn.release_trace(model.mesh, trace)
    return out


def batched_vs_serial(model, refs, batch):
    """Batched decode of `batch` users with mixed lengths vs each user decoded alone, from identical caches."""
    lengths = [LENGTH - 7 * (s % 4) for s in range(batch)]
    out = {}
    for number, index in enumerate(LAYERS):
        kind, x = model.config.layer_types[index], refs["inputs"][index]

        def prefilled():
            caches, table = model.allocate_cache(slots=batch, context=128)
            device_table = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for slot, length in enumerate(lengths):
                model.layers[number].prefill_forward(
                    model.upload(x[:, :length][None]),
                    rope_mats=model.rope_prefill[kind],
                    page_table=device_table,
                    kv_cache=caches[number],
                    user_id=slot,
                )
            return caches[number], table

        hidden = torch.cat([x[:, length : length + 1] for length in lengths], dim=1)[None]  # [1,1,B,H]
        positions = torch.tensor(lengths)
        cache, table = prefilled()
        batched = host(run_decode(model, number, kind, cache, decode_args(model, hidden, positions, table))).squeeze(0)
        batched_kv = per_device(cache)
        cache, table = prefilled()
        rows = []
        for s in range(batch):
            a = decode_args(model, hidden[:, :, s : s + 1], positions[s : s + 1], table[s : s + 1])
            rows.append(host(run_decode(model, number, kind, cache, a)).squeeze(0))
        serial_kv = per_device(cache)
        serial = torch.cat(rows, dim=-2)
        out[f"L{index}_min_user_pcc"] = min(pcc(serial[..., s, :], batched[..., s, :]) for s in range(batch))
        # Slots at the reference length decode the HF reference token: batched accuracy against HF directly.
        out[f"L{index}_batched_hf_pcc"] = min(
            pcc(refs["decode"][index], batched[..., s : s + 1, :]) for s in range(batch) if lengths[s] == LENGTH
        )
        out[f"L{index}_kv_equal"] = all(torch.equal(a, b) for a, b in zip(batched_kv, serial_kv))
        # Slot 1 inactive (position -1): its cache pages stay bit-identical, the other users still match serial.
        cache, table = prefilled()
        before = per_device(cache)
        idle = positions.clone()
        idle[1] = -1
        partial = host(run_decode(model, number, kind, cache, decode_args(model, hidden, idle, table))).squeeze(0)
        pages = table[1].long()
        out[f"L{index}_inactive_untouched"] = all(
            torch.equal(a[pages], b[pages]) for a, b in zip(before, per_device(cache))
        )
        out[f"L{index}_inactive_others_pcc"] = min(
            pcc(serial[..., s, :], partial[..., s, :]) for s in range(batch) if s != 1
        )
    return out


def timing(model, batches, steps):
    out = {}
    for batch in batches:
        for number, index in enumerate(LAYERS):
            kind = model.config.layer_types[index]
            caches, table = model.allocate_cache(slots=batch, context=256)
            a = decode_args(
                model,
                torch.randn(1, 1, batch, model.config.hidden_size) * 0.1,
                torch.full((batch,), 64),
                table,
            )
            run_decode(model, number, kind, caches[number], a)  # compile
            ttnn.synchronize_device(model.mesh)
            trace = ttnn.begin_trace_capture(model.mesh, cq_id=0)
            run_decode(model, number, kind, caches[number], a)
            ttnn.end_trace_capture(model.mesh, trace, cq_id=0)
            try:
                samples = []
                for _ in range(steps):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(model.mesh, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - t0) * 1e6)
            finally:
                ttnn.release_trace(model.mesh, trace)
            out[f"B{batch}_L{index}_us"] = statistics.median(samples)
        # 25 sliding + 5 full layers; excludes embedding, LM head and sampling.
        out[f"B{batch}_layers_ms"] = (25 * out[f"B{batch}_L0_us"] + 5 * out[f"B{batch}_L5_us"]) / 1000
    return out


def compare(result, baseline):
    failures = []
    for key, value in result.items():
        if key not in baseline or isinstance(value, bool):
            if isinstance(value, bool) and not value:
                failures.append(f"{key} false")
            continue
        base = baseline[key]
        if "pcc" in key or key.endswith(("_prefill", "_decode")):
            if value < base - 0.0005:
                failures.append(f"{key} {value:.5f} < baseline {base:.5f} - 0.0005")
        elif key.endswith("_us") and value > base * 1.03:
            failures.append(f"{key} {value:.0f} us > baseline {base:.0f} us + 3%")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="1,32", help="traced timing batch sizes")
    p.add_argument("--check-batch", default="8", help="users in the batched-vs-serial check, comma list (0 skips it)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--refs", type=Path, default=Path.home() / "gemma26-eval" / "refs" / f"layers_0_5_{REVISION[:8]}.pt")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--save-baseline", action="store_true", help="write this run's numbers to --baseline")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(8)
    refs = torch.load(args.refs) if args.refs.exists() else build_refs(args.refs)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        generator = Gemma4Generator(mesh, max_seq_len=2048, layer_indices=LAYERS)
        model = generator.model
        result = {"precision": model.precision_config["config_id"]}
        result.update(accuracy(model, refs))
        checks = [int(b) for b in args.check_batch.split(",") if int(b) > 1]
        for batch in checks:
            # Keys of the largest batch stay unprefixed (baseline compatibility); every batch is also reported.
            found = batched_vs_serial(model, refs, batch)
            result.update({f"B{batch}_{k}": v for k, v in found.items()})
        for key in {k.split("_", 1)[1] for k in result if k.startswith("B") and "_L" in k and not k.endswith("_us")}:
            values = [result[f"B{b}_{key}"] for b in checks if f"B{b}_{key}" in result]
            if values:
                result[key] = min(values) if not isinstance(values[0], bool) else all(values)
        result.update(timing(model, [int(b) for b in args.batches.split(",")], args.steps))
        generator._release_trace()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    floor = [k for k, v in result.items() if (k.endswith(("_prefill", "_decode")) or "pcc" in k) and v < 0.995]
    failures = [f"{k} {result[k]:.5f} < 0.995" for k in floor]
    if args.baseline and args.baseline.exists() and not args.save_baseline:
        failures += compare(result, json.loads(args.baseline.read_text()))
    if args.save_baseline and args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(result, indent=1))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))
    acc = " ".join(
        f"{k}={v:.5f}" for k, v in result.items() if k.endswith(("_prefill", "_decode", "_pcc")) and k[0] == "L"
    )
    speed = " ".join(f"{k}={v:.2f}" for k, v in result.items() if k.endswith("_layers_ms"))
    kv = " ".join(f"{k}={v}" for k, v in result.items() if k.endswith(("_kv_equal", "_untouched")) and k[0] == "L")
    failures += [f"{k} is False" for k, v in result.items() if v is False]
    print(f"QUICK {'PASS' if not failures else 'FAIL'} {acc} {kv} {speed}", flush=True)
    for failure in failures:
        print(f"QUICK_FAIL {failure}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
