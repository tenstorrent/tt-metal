# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 batched decode on the Galaxy: every slot of the pool a different prompt, against the vendored CPU reference.

Prefills ``sp * V41_BATCH_USERS`` prompts of real text (different windows of the text, lengths of both parities
around ``V41_BATCH_PROMPT``) with the layer schedule ``V41_BATCH_LAYERS`` on the device (``TtV41Transformer.prefill``,
each joining a slot of ``TtV41BatchDecoder``), then decodes ``V41_BATCH_STEPS`` tokens for all of them in one step
each, teacher-forced on the text's own next tokens. The slots in ``V41_BATCH_REF_SLOTS`` are also run on the CPU
reference (``Transformer.forward``; prefill then the same tokens step by step), and per step report the logits PCC,
the greedy agreement and every block's output PCC. With ``V41_BATCH_TRACE=1`` (default) the step is captured once
after ``V41_BATCH_WARMUP`` eager steps and replayed; replays report the device time. ``V41_BATCH_REF=0`` runs the
device alone (checkpoint Engram schedules), judged by ``book_top1`` over all slots.

Results go to the log as ``V41_BATCH_RESULT {json}`` and to ``V41_BATCH_OUT`` when set. Reported, not gated here: a NaN or a crash fails the test, a low PCC does not.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_decode import _pcc, _reference_forward
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, _rss_gb, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


DEFAULT_LAYERS = "0,2,3,20,21,24"
DEFAULT_REF_SLOTS = "0,1,9,14,22,31"
TRACE_REGION = 512 * 1024 * 1024


def _row_to_host(x: ttnn.Tensor, rank: int, u: int, tp: int, hc: int) -> torch.Tensor:
    """Row ``u`` of the decode tile's streams on mesh row ``rank`` (its TP shards) -> ``[hc, hidden]``."""
    shards = ttnn.get_device_tensors(x)
    parts = [ttnn.to_torch(shards[rank * tp + j])[0, 0, u].float().view(hc, -1) for j in range(tp)]
    return torch.cat(parts, dim=-1)


def _debug_join(decoder, state, layers) -> dict:
    """Slot 0's pool rows against the state's (device 0): max abs difference of every layer's ring and every
    source's compressed rows and index keys."""
    pool, out = decoder.pool, {}
    for layer in layers:
        base = pool.ring_row(layer)
        ring = state.to_host(pool.kv_tensor(layer))[base : base + 128]
        carry = state.to_host(state.window_carry[layer])
        out[f"ring{layer}"] = round(float((ring - carry).abs().max()), 5)
    g = state.geometry
    for src in state.kv:
        written = state.start // decoder.config.compress_ratio(src)
        comp = state.to_host(pool.kv[src])[pool.comp_row(src) : pool.comp_row(src) + written]
        rows = state.to_host(state.kv[src])[g.kv_row_of_compressed(0) : g.kv_row_of_compressed(0) + written]
        out[f"comp{src}"] = round(float((comp - rows).abs().max()), 5)
        keys = state.to_host(pool.index_k[src])[:written]  # [0, 0]: slot 0's tiled rows, untilized on the host
        out[f"keys{src}"] = round(float((keys - state.to_host(state.index_k[src])[:written]).abs().max()), 5)
        if src in pool.index_k_blocks:
            blocks = state.to_host(pool.index_k_blocks[src]).reshape(-1, keys.shape[1])[:written]
            out[f"blocks{src}"] = round(float((blocks - state.to_host(state.index_k[src])[:written]).abs().max()), 5)
        out[f"comp{src}_norm"] = round(float(rows.abs().max()), 3)
    for src, (k0, s0) in pool.carry.items():
        carry = state.compressor_carry.get(src)
        if carry is not None:
            out[f"carry{src}"] = round(float((state.to_host(k0)[0] - state.to_host(carry[0])[0]).abs().max()), 5)
    return out


def _release(state) -> None:
    """Free a prefill state's device tensors (its rows live in the pool now)."""
    for group in (state.kv, state.index_k, state.window_carry):
        for t in group.values():
            ttnn.deallocate(t)
    if state.swa_scratch is not None:
        ttnn.deallocate(state.swa_scratch)
    for carry in state.carry_buffers.values():
        for t in carry:
            ttnn.deallocate(t)


@pytest.mark.timeout(int(os.environ.get("V41_BATCH_TIMEOUT", 4 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_batch_decode(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_BATCH_LAYERS", DEFAULT_LAYERS))
    base = int(os.environ.get("V41_BATCH_PROMPT", 255))
    steps = int(os.environ.get("V41_BATCH_STEPS", 16))
    chunk = int(os.environ.get("V41_BATCH_CHUNK", 1024))
    users = int(os.environ.get("V41_BATCH_USERS", 4))
    use_ref = os.environ.get("V41_BATCH_REF", "1") == "1"
    trace = os.environ.get("V41_BATCH_TRACE", "1") == "1"
    warmup = int(os.environ.get("V41_BATCH_WARMUP", 2))
    sample = os.environ.get("V41_BATCH_SAMPLE", "0") == "1"  # argmax on device (DeviceSampler) instead of logits
    profile = os.environ.get("V41_BATCH_PROFILE", "0") == "1"  # as test_v41_decode: drain the device profiler per block
    ref_slots = (
        [int(s) for s in os.environ.get("V41_BATCH_REF_SLOTS", DEFAULT_REF_SLOTS).split(",") if s] if use_ref else []
    )
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    sp, tp = tuple(mesh_device.shape)
    n_slots = sp * users
    ref_slots = [s for s in ref_slots if s < n_slots]
    # prompt lengths of both parities around the base, each slot its own window of the text
    lengths = [base - (s * 5) % 32 for s in range(n_slots)]
    assert min(lengths) >= C.SLIDING_WINDOW
    stride = base + steps + 8
    text = orc.text_tokens(n_slots * stride)[0]
    prompts = {s: text[s * stride : s * stride + lengths[s]] for s in range(n_slots)}
    books = {s: text[s * stride + lengths[s] : s * stride + lengths[s] + steps + 1] for s in range(n_slots)}
    if os.environ.get("V41_BATCH_SAME_PROMPT", "0") == "1":  # every slot the same request: identical expert routing
        lengths = [lengths[0]] * n_slots
        prompts = {s: prompts[0] for s in range(n_slots)}
        books = {s: books[0] for s in range(n_slots)}
    max_seq = -(-(base + steps + 32) // chunk) * chunk
    result = {
        "layers": layers,
        "n_layers": len(layers),
        "slots": n_slots,
        "users_per_rank": users,
        "lengths": lengths,
        "steps": steps,
        "chunk": chunk,
        "max_seq": max_seq,
        "reference_slots": ref_slots,
        "trace": trace,
        "warmup": warmup,
        "sample_on_device": sample,
        "mesh": list(mesh_device.shape),
        "fabric": str(device_params.get("fabric_config")),
    }
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)

    # --- the reference, slot by slot: prefill logits, then per step the logits and every block's output ---
    reference = {}
    if ref_slots:
        t0 = time.time()
        ref = orc.build_reference(spec)
        result["reference_build_s"] = round(time.time() - t0, 1)
        blocks = {}
        hooks = [
            ref.layers[pos].register_forward_hook(
                lambda mod, args, out, layer=layer: blocks.__setitem__(layer, out[0][0, 0].float())
            )
            for pos, layer in enumerate(layers)
        ]
        try:
            t0 = time.time()
            for s in ref_slots:
                orc._reset_state(ref)
                entry = {"prefill": _reference_forward(ref, prompts[s][None], 0), "steps": []}
                for i in range(steps):
                    blocks.clear()
                    logits = _reference_forward(ref, torch.tensor([[int(books[s][i])]]), lengths[s] + i)
                    entry["steps"].append((logits, {layer: blocks[layer].clone() for layer in layers}))
                reference[s] = entry
            result["reference_s"] = round(time.time() - t0, 1)
        finally:
            for h in hooks:
                h.remove()
        result["rss_after_reference_gb"] = round(_rss_gb(), 1)
        logger.info(f"batch decode: reference of {len(ref_slots)} slots in {result['reference_s']} s")

    # --- the device: build, prefill every slot into the pool ---
    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    decoder = TtV41BatchDecoder(model, root, users_per_rank=users, trace=trace, warmup=warmup, sample_on_device=sample)
    assert not (sample and reference), "device sampling returns ids: no logits to compare with the reference"
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)

    t0 = time.time()
    prefill_argmax = {}
    for s in range(n_slots):
        logits, state = model.prefill(prompts[s], logit_positions=1)
        assert not torch.isnan(logits[-1]).any(), f"NaN in the prefill logits of slot {s}"
        prefill_argmax[s] = int(logits[-1].argmax())
        if s in reference:
            reference[s]["prefill_pcc"] = round(_pcc(logits[-1], reference[s]["prefill"]), 5)
        decoder.join(s, prompts[s], state)
        if s == 0 and os.environ.get("V41_BATCH_DEBUG", "0") == "1":
            result["debug_join"] = _debug_join(decoder, state, layers)
            logger.info(f"batch decode: join check {result['debug_join']}")
        _release(state)
    result["prefill_join_s"] = round(time.time() - t0, 1)
    result["prefill_logits_pcc"] = {str(s): reference[s]["prefill_pcc"] for s in reference}
    result["dram_after_join_bytes"] = _dram_bytes(mesh_device)
    logger.info(f"batch decode: {n_slots} slots joined in {result['prefill_join_s']} s, {result['prefill_logits_pcc']}")

    # --- decode, all slots per step ---
    captured = {}
    if reference:
        decoder.on_block = lambda layer, x: captured.__setitem__(layer, x)
    if profile:
        ttnn.ReadDeviceProfiler(mesh_device)
        previous = decoder.on_block

        def drain(layer, x):
            if previous is not None:
                previous(layer, x)
            ttnn.ReadDeviceProfiler(mesh_device)

        decoder.on_block = drain
    per_step = []
    hits = {s: 0 for s in range(n_slots)}
    for i in range(steps):
        tokens = {s: int(books[s][i]) for s in range(n_slots)}
        captured.clear()
        t0 = time.time()
        if sample and i == 0 and os.environ.get("V41_BATCH_DEBUG", "0") == "1":
            decoder.sampler.debug = {}
        if profile and i == steps - 1:
            signpost("start")
        out = decoder.decode(tokens, {s: (1, 0.0, 0.0) for s in tokens} if sample else None)
        wall = time.time() - t0
        if profile:
            ttnn.ReadDeviceProfiler(mesh_device)
            if i == steps - 1:
                signpost("stop")
        if sample and decoder.sampler.debug:
            dbg = decoder.sampler.debug
            decoder.sampler.debug = None
            for s in (0, 1, 4, 5, 9, 31):
                r, u = divmod(s, users)
                chips = [
                    ttnn.to_torch(ttnn.get_device_tensors(dbg["logits"])[r * tp + j])[0, 0, u].float()
                    for j in range(tp)
                ]
                full = torch.cat(chips)
                vals = ttnn.to_torch(ttnn.get_device_tensors(dbg["values"])[r * tp])[0, 0, u].float()
                idx = ttnn.to_torch(ttnn.get_device_tensors(dbg["indices"])[r * tp])[0, 0, u].to(torch.int64)
                ids_r = ttnn.to_torch(ttnn.get_device_tensors(dbg["ids"])[r * tp])[0, 0, 0].to(torch.int64)
                logger.info(
                    f"sample debug slot {s} (rank {r} row {u}): host argmax {int(full.argmax())} ({full.max():.2f}); "
                    f"candidate max {vals.max():.2f} at {int(idx[vals.argmax()])}; sampled {int(ids_r[u])}; "
                    f"out {out[s]}; per-chip argmax {[int(c.argmax()) + j * (C.VOCAB_SIZE // tp) for j, c in enumerate(chips)]}; "
                    f"cand idx[0:4] {idx[:4].tolist()} idx[32:36] {idx[32:36].tolist()}"
                )
        entry = {"step": i, "wall_s": round(wall, 3), **decoder.last, "slots": {}}
        for s in range(n_slots):
            if sample:
                dev = int(out[s])
            else:
                logits = out[s]
                assert not torch.isnan(logits).any(), f"NaN in the logits of slot {s} at step {i}"
                dev = int(logits.argmax())
            hits[s] += dev == int(books[s][i + 1])
            if s in reference:
                ref_logits, ref_blocks = reference[s]["steps"][i]
                r, u = divmod(s, users)
                entry["slots"][str(s)] = {
                    "dev_argmax": dev,
                    "ref_argmax": int(ref_logits.argmax()),
                    "logits_pcc": round(_pcc(logits, ref_logits), 5),
                    "top5_hit": dev in ref_logits.topk(5).indices.tolist(),
                    "block_pcc": {
                        str(layer): round(
                            _pcc(_row_to_host(captured[layer], r, u, tp, C.HC_MULT), ref_blocks[layer]), 5
                        )
                        for layer in layers
                    },
                }
        per_step.append(entry)
        logger.info(f"batch step {i}: {json.dumps({k: v for k, v in entry.items() if k != 'slots'})} {entry['slots']}")
    result["steps_detail"] = per_step
    replays = [e for e in per_step if e["mode"] == "replay"]
    result["replay_steps"] = len(replays)
    if replays:
        result["trace_segments"] = decoder.trace_segments
        result["replay_wall_min_s"] = min(e["wall_s"] for e in replays)
        result["replay_device_min_s"] = min(e["device_s"] for e in replays)
        for key in ("inputs_s", "stage_s", "sync_s", "issue_s", "read_s"):
            result["replay_" + key] = [e.get(key) for e in replays]
        result["replay_device_only_s"] = decoder.replay_device_time()
    decoder.release()
    result["book_top1_per_slot"] = {str(s): round(hits[s] / steps, 4) for s in range(n_slots)}
    result["book_top1"] = round(sum(hits.values()) / (steps * n_slots), 4)
    result["text"] = {str(s): tokenizer.decode(books[s][1 : steps + 1].tolist()) for s in (0, n_slots - 1)}
    if reference:
        agg = {}
        for s in reference:
            rows = [e["slots"][str(s)] for e in per_step]
            agg[str(s)] = {
                "top1_agreement": round(sum(r["dev_argmax"] == r["ref_argmax"] for r in rows) / steps, 4),
                "top5_agreement": round(sum(r["top5_hit"] for r in rows) / steps, 4),
                "logits_pcc_min": min(r["logits_pcc"] for r in rows),
                "block_pcc_min": {str(layer): min(r["block_pcc"][str(layer)] for r in rows) for layer in layers},
            }
        result["reference_agreement"] = agg
        result["logits_pcc_min"] = min(a["logits_pcc_min"] for a in agg.values())
        result["top1_agreement"] = round(sum(a["top1_agreement"] for a in agg.values()) / len(agg), 4)
        result["top5_agreement"] = round(sum(a["top5_agreement"] for a in agg.values()) / len(agg), 4)
        result["block_pcc_min"] = {
            str(layer): min(a["block_pcc_min"][str(layer)] for a in agg.values()) for layer in layers
        }
    result["dram_after_decode_bytes"] = _dram_bytes(mesh_device)
    result["rss_peak_gb"] = round(_rss_gb(), 1)
    logger.info("V41_BATCH_RESULT " + json.dumps(result))
    out_path = os.environ.get("V41_BATCH_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
