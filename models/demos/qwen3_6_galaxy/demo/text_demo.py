# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B Galaxy AR text-generation demo, runnable under tracy.

Mirrors the llama3_70b_galaxy/demo/text_demo.py pattern: a parametrised
pytest function that loads the model, runs prefill + decode, and emits
``tracy.signpost("start"/"stop")`` markers around the region of interest
so ``python -m tracy -p -v -r -m pytest ...`` produces an op-level CSV
covering exactly that span.

Default invocation (1 prefill layer, 1 decode token):

    python -m tracy -p -v -r -m pytest \
        models/demos/qwen3_6_galaxy/demo/text_demo.py::test_demo_text \
        -k "perf_1L_1T"

The tracy output xlsx lands under ``generated/profiler/reports/<timestamp>/``.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config
from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

_DEFAULT_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


def _load_global_weights(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


def _load_layer_weights(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer_idx}"
    keys = [k for k in weight_map if k.startswith(pfx + ".")]
    files = sorted({weight_map[k] for k in keys})
    raw = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in keys:
            if k in shard:
                raw[k] = shard[k].float()
    return {k[len(pfx) + 1 :]: v for k, v in raw.items()}


@pytest.fixture(scope="module")
def mesh_8x4():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
@pytest.mark.parametrize(
    "num_layers, num_tokens, prompt, mode",
    [
        # Smallest meaningful perf sample: 1 decoder layer, 1 prefill pass
        # + 1 decode step. num_tokens=2 because prefill produces token 1
        # and the decode loop produces 1 additional token (=2 total).
        # Profiled region is bracketed by tracy signposts:
        #   start --> prefill_done --> stop
        # which segments the tracy CSV into prefill-only and decode-only rows.
        # Use under tracy:
        #   python -m tracy -p -v -r -m pytest \
        #     models/demos/qwen3_6_galaxy/demo/text_demo.py::test_demo_text -k perf_1L_1T
        (1, 2, "The capital of France is", "perf_1L_1T"),
        # Full 64-layer model, generate 10 tokens (matches the demo.py CLI default).
        (64, 10, "The capital of France is", "e2e_64L_10T"),
    ],
    ids=["perf_1L_1T", "e2e_64L_10T"],
)
def test_demo_text(mesh_8x4, num_layers, num_tokens, prompt, mode):
    """Parametrised demo: prefill + greedy decode, signpost-wrapped for tracy.

    The two existing modes:
      * ``perf_1L_1T`` — 1 decoder layer, prefill T=32, decode 1 token.
        The minimum meaningful e2e run; the tracy CSV covers only the
        between-``signpost("start")``-and-``signpost("stop")`` region so
        op count is bounded to what fits on one screen.
      * ``e2e_64L_10T`` — full 64-layer model, decode 10 tokens. The
        ``num_tokens=10`` sanity check that the model actually generates
        coherent text on the Galaxy.
    """
    try:
        from tracy import signpost
    except ImportError:  # Allow non-tracy runs (signposts become no-ops).
        signpost = lambda *_args, **_kwargs: None  # noqa: E731

    from transformers import AutoTokenizer

    snapshot_dir = _DEFAULT_SNAPSHOT
    print(f"\n[demo] mode={mode} num_layers={num_layers} num_tokens={num_tokens}")
    print(f"[demo] loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), trust_remote_code=True)

    print(f"[demo] tokenizing prompt: {prompt!r}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids
    print(f"[demo] T_prompt={T_prompt} T_padded={T_padded}")

    print(f"[demo] loading weights for {num_layers} layers ...")
    global_weights = _load_global_weights(snapshot_dir)
    layers_weights = [_load_layer_weights(snapshot_dir, i) for i in range(num_layers)]

    with open(snapshot_dir / "config.json") as f:
        config = Qwen36Config(json.load(f))
    model_args = TtQwen36ModelArgs(mesh_8x4)

    print(f"[demo] building TtQwen36Transformer ({num_layers} layers) ...")
    t0 = time.time()
    model = TtQwen36Transformer(
        mesh_device=mesh_8x4,
        args=model_args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=num_layers,
    )
    print(f"[demo] model built in {time.time()-t0:.1f}s")

    # ----- Warmup: avoids compile-time skewing the profile sample -----
    print("[demo] warmup prefill+decode (not profiled) ...")
    wlogits, wkv, wdn, wcv = model.forward_prefill(input_ids_padded, return_caches=True)
    wnext = int(wlogits[0, T_prompt - 1, : config.vocab_size].argmax().item())
    _ = model.forward_decode(
        torch.tensor([[wnext]], dtype=torch.long),
        current_pos=T_padded,
        kv_caches=wkv,
        dn_states=wdn,
        conv_states=wcv,
    )

    # ----- Profiled prefill -----
    print("[demo] PROFILED prefill ...")
    signpost("start")
    t0 = time.time()
    tt_logits, kv_caches, dn_states, conv_states = model.forward_prefill(input_ids_padded, return_caches=True)
    prefill_dt = time.time() - t0
    signpost("prefill_done")
    next_id = int(tt_logits[0, T_prompt - 1, : config.vocab_size].argmax().item())
    print(f"[demo] prefill: {prefill_dt*1000:.1f} ms  ->  token id={next_id} text={tokenizer.decode([next_id])!r}")

    # ----- Profiled decode (num_tokens - 1 decode steps; the prefill itself produced token 1) -----
    print(f"[demo] PROFILED decode ({num_tokens - 1} steps) ...")
    generated = [next_id]
    current_pos = T_padded
    decode_times = []
    for step in range(1, num_tokens):
        next_id_tensor = torch.tensor([[generated[-1]]], dtype=torch.long)
        t0 = time.time()
        step_logits, kv_caches, dn_states, conv_states = model.forward_decode(
            next_id_tensor,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )
        dt = time.time() - t0
        decode_times.append(dt)
        step_last = step_logits[0, 0, : config.vocab_size]
        assert not torch.isnan(step_last).any(), f"NaN at step {step}"
        assert not torch.isinf(step_last).any(), f"Inf at step {step}"
        next_id = int(step_last.argmax().item())
        generated.append(next_id)
        current_pos += 1
        print(f"[demo]   step {step}: id={next_id} text={tokenizer.decode([next_id])!r} ({dt*1000:.1f} ms)")
    signpost("stop")

    full_text = tokenizer.decode(generated)
    print(f"\n[demo] generated {len(generated)} tokens: {full_text!r}")
    print(f"[demo] full sequence: {(prompt + full_text)!r}")
    print(f"[demo] === perf summary ({mode}) ===")
    print(f"[demo]   num_layers           : {num_layers}")
    print(f"[demo]   prefill T            : {T_padded}")
    print(f"[demo]   prefill latency      : {prefill_dt*1000:.1f} ms")
    if decode_times:
        avg_decode_ms = sum(decode_times) / len(decode_times) * 1000
        print(f"[demo]   decode steps         : {len(decode_times)}")
        print(f"[demo]   decode avg latency   : {avg_decode_ms:.1f} ms/step")
        print(f"[demo]   decode throughput    : {1000.0/avg_decode_ms:.2f} tok/s")
