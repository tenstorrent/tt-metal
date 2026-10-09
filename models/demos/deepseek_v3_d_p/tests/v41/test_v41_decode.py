# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 single-token decode on the Galaxy after a real-weight prefill, against the vendored CPU reference.

Prefills ``V41_DECODE_PROMPT`` tokens of real text (odd by default, so the ratio-2 compressor carry is
consumed by
the first step) with the layer schedule ``V41_DECODE_LAYERS`` on the device (``TtV41Transformer.prefill``) and
on
the CPU reference (``Transformer.forward(tokens, 0)``), then decodes ``V41_DECODE_STEPS`` tokens step by step
(``TtV41Decoder.decode`` against ``Transformer.forward(token, start_pos)``), teacher-forced on the reference's
greedy
token. Per step it reports the greedy agreement, the logits PCC and every block's output PCC; after the
prefill the
compressor carry against the reference's ``kv_state`` / ``score_state``. With ``V41_DECODE_TRACE=1`` (default)
the
decoder captures the step per position parity after ``V41_DECODE_WARMUP`` eager steps and replays it: each
step
reports its mode and, for replays, the device time of the replay. With ``V41_DECODE_REF=0`` (schedules the
reference cannot run: checkpoint Engram layers) the device decodes alone, teacher-forced on the text's own
next tokens
(``book_top1``: the depth test's next-token accuracy, through the decode step), or free-running on its greedy
token
with ``V41_DECODE_FREE=1``; both report the text.

With ``V41_DECODE_PROFILE=1`` (under ``python -m tracy -r -p -v -m pytest ...`` with
``TT_METAL_DEVICE_PROFILER=1``) the
device profiler is read after the prefill and after every block of every step, so no step overflows the
device's
marker buffers, and the last step runs between the ``start`` / ``stop`` signposts (its ops are the
measurement).

Results go to the log as ``V41_DECODE_RESULT {json}`` and to ``V41_DECODE_OUT`` when set. Reported, not gated
here: a NaN or a crash fails the test, a low PCC does not.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, _rss_gb, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.decode import TtV41Decoder
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


DEFAULT_LAYERS = "0,2,3,20,21,24"
TRACE_REGION = 512 * 1024 * 1024  # as test_v41_trace.py; two captures of the step


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2 or torch.isnan(a).any() or torch.isnan(b).any():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _reference_forward(ref: v41.Transformer, ids: torch.Tensor, start_pos: int) -> torch.Tensor:
    """``Transformer.forward`` under the bf16 default dtype it was built with (``testing.prefill``) -> fp32
    logits
    ``[vocab]`` of the last position."""
    with v41.set_dtype(torch.bfloat16):
        _, logits, _ = ref.forward(ids, start_pos)
    return logits[0].float()


def _streams_to_host(x: ttnn.Tensor, tp: int, hc: int) -> torch.Tensor:
    """Row 0 of the decode tile's streams ``[1, 1, 32, hc * hidden/tp]`` (mesh row 0's TP shards) -> ``[hc,
    hidden]``."""
    shards = ttnn.get_device_tensors(x)
    parts = [ttnn.to_torch(shards[j])[0, 0, 0].float().view(hc, -1) for j in range(tp)]
    return torch.cat(parts, dim=-1)


@pytest.mark.timeout(int(os.environ.get("V41_DECODE_TIMEOUT", 4 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_decode(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_DECODE_LAYERS", DEFAULT_LAYERS))
    prompt = int(os.environ.get("V41_DECODE_PROMPT", 255))
    steps = int(os.environ.get("V41_DECODE_STEPS", 8))
    chunk = int(os.environ.get("V41_DECODE_CHUNK", 1024))
    use_ref = os.environ.get("V41_DECODE_REF", "1") == "1"
    free_run = os.environ.get("V41_DECODE_FREE", "0") == "1"
    trace = os.environ.get("V41_DECODE_TRACE", "1") == "1"
    warmup = int(os.environ.get("V41_DECODE_WARMUP", 4))
    profile = os.environ.get("V41_DECODE_PROFILE", "0") == "1"
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    max_seq = -(-(prompt + steps + 32) // chunk) * chunk
    result = {
        "layers": layers,
        "n_layers": len(layers),
        "prompt": prompt,
        "steps": steps,
        "chunk": chunk,
        "max_seq": max_seq,
        "reference": use_ref,
        "free_run": free_run,
        "trace": trace,
        "warmup": warmup,
        "profile": profile,
        "mesh": list(mesh_device.shape),
        "fabric": str(device_params.get("fabric_config")),
    }
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    text = orc.text_tokens(prompt + steps + 1)[0]
    tokens, book = text[:prompt], text[prompt:]  # the prompt and the text's own continuation
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    sp, tp = tuple(mesh_device.shape)

    ref = None
    if use_ref:
        t0 = time.time()
        ref = orc.build_reference(spec)
        orc._reset_state(ref)
        result["reference_build_s"] = round(time.time() - t0, 1)
        result["rss_after_reference_gb"] = round(_rss_gb(), 1)
        logger.info(f"decode: reference built in {result['reference_build_s']} s")

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
    decoder = TtV41Decoder(model, root, trace=trace, warmup=warmup)
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)

    # --- prefill on both ---
    t0 = time.time()
    logits, state = model.prefill(tokens, logit_positions=1)
    result["prefill_s"] = round(time.time() - t0, 3)
    dev_logits = logits[-1]
    assert not torch.isnan(dev_logits).any(), "NaN in the device prefill logits"
    if profile:
        ttnn.ReadDeviceProfiler(mesh_device)
    decoder.begin(tokens, state)
    if ref is not None:
        t0 = time.time()
        ref_logits = _reference_forward(ref, tokens[None], 0)
        result["reference_prefill_s"] = round(time.time() - t0, 1)
        result["prefill_logits_pcc"] = round(_pcc(dev_logits, ref_logits), 5)
        result["prefill_argmax"] = [int(dev_logits.argmax()), int(ref_logits.argmax())]
        carry = {}
        for pos, layer in enumerate(layers):
            if state.compressor_carry.get(layer) is not None:
                comp = ref.layers[pos].attn.compressor
                kv, score = (state.to_host(t)[0] for t in state.compressor_carry[layer])
                slot = (prompt - 1) % comp.compress_ratio
                carry[layer] = {
                    "kv": round(_pcc(kv, comp.kv_state[0, slot]), 5),
                    "score": round(_pcc(score, comp.score_state[0, slot]), 5),
                }
        result["prefill_carry_pcc"] = carry
        logger.info(f"decode: prefill logits PCC {result['prefill_logits_pcc']}, carry {carry}")

    # --- decode ---
    captured = {}
    ref_blocks = {}
    hooks = []
    if ref is not None:
        # every block's output to the host, for the per-block PCC: 40 reads per step, only worth it with a
        # reference
        decoder.on_block = lambda layer, x: captured.__setitem__(layer, _streams_to_host(x, tp, C.HC_MULT))
        for pos, layer in enumerate(layers):
            hooks.append(
                ref.layers[pos].register_forward_hook(
                    lambda mod, args, out, layer=layer: ref_blocks.__setitem__(layer, out[0][0, 0].float())
                )
            )
    if profile:
        previous = decoder.on_block

        def drain(layer, x):
            if previous is not None:
                previous(layer, x)
            ttnn.ReadDeviceProfiler(mesh_device)

        decoder.on_block = drain
    token = int((ref_logits if ref is not None else dev_logits).argmax())
    if ref is None and not free_run:
        token = int(book[0])  # teacher forcing on the text
    generated, per_step = [token], []
    try:
        for i in range(steps):
            p = state.start
            captured.clear()
            if profile and i == steps - 1:
                signpost("start")
            t0 = time.time()
            dev_logits = decoder.decode(token, state)
            wall = time.time() - t0
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
                if i == steps - 1:
                    signpost("stop")
            assert not torch.isnan(dev_logits).any(), f"NaN in the device logits at position {p}"
            entry = {"pos": p, "token": token, "dev_argmax": int(dev_logits.argmax()), "wall_s": round(wall, 3)}
            entry.update(decoder.last)
            entry["book_next"] = int(book[i + 1])
            if ref is not None:
                t0 = time.time()
                ref_logits = _reference_forward(ref, torch.tensor([[token]]), p)
                entry["ref_s"] = round(time.time() - t0, 2)
                entry["ref_argmax"] = int(ref_logits.argmax())
                entry["logits_pcc"] = round(_pcc(dev_logits, ref_logits), 5)
                entry["top5_hit"] = entry["dev_argmax"] in ref_logits.topk(5).indices.tolist()
                entry["block_pcc"] = {
                    str(layer): round(_pcc(captured[layer], ref_blocks[layer]), 5) for layer in layers
                }
                token = entry["ref_argmax"]  # teacher forcing on the reference
            else:
                token = entry["dev_argmax"] if free_run else entry["book_next"]
            generated.append(token)
            per_step.append(entry)
            logger.info(f"decode step {i} at {p}: {entry}")
    finally:
        for h in hooks:
            h.remove()
    result["steps_detail"] = per_step
    replays = [e for e in per_step if e["mode"] == "replay"]
    result["replay_steps"] = len(replays)
    if replays:
        result["trace_segments"] = decoder.trace_segments
        result["replay_wall_s"] = [e["wall_s"] for e in replays]
        result["replay_wall_min_s"] = min(e["wall_s"] for e in replays)
        result["replay_device_s"] = [e["device_s"] for e in replays]
        result["replay_device_min_s"] = min(e["device_s"] for e in replays)
        for key in ("inputs_s", "stage_s", "sync_s", "issue_s", "read_s", "history_s"):
            result["replay_" + key] = [e.get(key) for e in replays]
        # the captured graphs alone, replayed back to back without host work: the device's time per token
        result["replay_device_only_s"] = decoder.replay_device_time()
    decoder.release()
    result["fed"] = generated  # the tokens fed in (teacher-forced or the device's own)
    result["text"] = tokenizer.decode([e["dev_argmax"] for e in per_step])  # the device's greedy predictions
    result["book_text"] = tokenizer.decode(book[1 : steps + 1].tolist())
    result["book_top1"] = round(sum(e["dev_argmax"] == e["book_next"] for e in per_step) / len(per_step), 4)
    result["step_wall_s"] = [e["wall_s"] for e in per_step]
    result["step_wall_min_s"] = min(e["wall_s"] for e in per_step)
    if ref is not None:
        result["top1_agreement"] = round(sum(e["dev_argmax"] == e["ref_argmax"] for e in per_step) / len(per_step), 4)
        result["top5_agreement"] = round(sum(e["top5_hit"] for e in per_step) / len(per_step), 4)
        result["logits_pcc_min"] = min(e["logits_pcc"] for e in per_step)
        result["block_pcc_min"] = {str(layer): min(e["block_pcc"][str(layer)] for e in per_step) for layer in layers}
    result["dram_after_decode_bytes"] = _dram_bytes(mesh_device)
    result["rss_peak_gb"] = round(_rss_gb(), 1)
    logger.info("V41_DECODE_RESULT " + json.dumps(result))
    out = os.environ.get("V41_DECODE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
