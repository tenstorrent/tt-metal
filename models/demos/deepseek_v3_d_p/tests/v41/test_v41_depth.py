# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""V4.1 prefill at an arbitrary layer depth on real weights: fit, time and next-token sanity (harness).

The production gate runs the six-layer schedule the CPU oracle can afford. This test takes any layer schedule
(``V41_DEPTH_LAYERS``: a comma list with ranges, ``all`` = every backbone layer; Engram layers 1 and 14 get
the
checkpoint tables through ``engram_checkpoint``), prefills ``V41_DEPTH_SEQ`` tokens of real text in chunks of
``V41_DEPTH_CHUNK`` on the Galaxy, and reports:

* bit-identical repeat (gated, like every V4.1 gate);
* wall time per untraced prefill, tokens -> last-N logits on the host, first run (kernel compile) and repeats;
* trace mode (``V41PrefillTrace``, one scored row as in generation; ``V41_DEPTH_TRACE=0`` skips it): wall time
  per replayed prefill and device-only replay time, the bit-identical check against the untraced logits;
* next-token top-1 / top-5 accuracy and mean log-probability of the real text over the last
``V41_DEPTH_SCORED``
  positions, against the prompt's own next tokens: no reference needed, so it runs where the oracle cannot (a
  working model predicts English far above chance; a broken one does not). Reported, not gated;
* per-chip DRAM in use after the build and after the prefill, and host RSS.

Results go to the log as ``V41_DEPTH_RESULT {json}`` and to ``V41_DEPTH_OUT`` when set.
"""

import json
import os
import resource
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, V41PrefillTrace
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

DEFAULT_LAYERS = "0,2,3,20,21,24"


def parse_layers(spec: str) -> list[int]:
    if spec.strip() == "all":
        return list(range(C.NUM_LAYERS))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def _dram_bytes(mesh_device) -> int | None:
    try:  # as tests/v41/test_v41_expert_dtype.py: per chip (the view is one chip of the mesh)
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return int(view.total_bytes_allocated_per_bank * view.num_banks)
    except Exception as e:  # the API moves; the number is a report, not a gate
        logger.warning(f"DRAM statistics unavailable: {e}")
        return None


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


@pytest.mark.timeout(int(os.environ.get("V41_DEPTH_TIMEOUT", 6 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_depth(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_DEPTH_LAYERS", DEFAULT_LAYERS))
    seq = int(os.environ.get("V41_DEPTH_SEQ", 8192))
    chunk = int(os.environ.get("V41_DEPTH_CHUNK", min(seq, 8192)))
    scored = int(os.environ.get("V41_DEPTH_SCORED", 256))
    repeats = int(os.environ.get("V41_DEPTH_REPEATS", 3))
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    result = {
        "layers": layers,
        "n_layers": len(layers),
        "seq": seq,
        "chunk": chunk,
        "scored": scored,
        "mesh": list(mesh_device.shape),
        "fabric": str(device_params.get("fabric_config")),
    }
    spec = orc.real_spec(tuple(layers), seq, checkpoint=ckpt.root)  # identity for the weight cache only
    root = weight_cache_dir(spec, mesh_device.shape)
    tokens = orc.text_tokens(seq)[0]
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)

    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    result["engram_layers"] = sorted(engram)
    result["engram_build_s"] = round(time.time() - t0, 1)
    result["rss_after_engram_gb"] = round(_rss_gb(), 1)

    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    t0 = time.time()
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=-(-seq // chunk) * chunk,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)
    result["rss_after_build_gb"] = round(_rss_gb(), 1)
    logger.info(f"depth: built {len(layers)} layers in {result['build_s']} s; DRAM {result['dram_after_build_bytes']}")

    times = []
    first = None
    for i in range(repeats):
        t0 = time.time()
        logits, state = model.prefill(tokens, logit_positions=scored)
        times.append(round(time.time() - t0, 3))
        logger.info(f"depth: prefill {i} of {seq} tokens, {len(layers)} layers: {times[-1]} s")
        if first is None:
            first = logits.clone()
        else:
            assert torch.equal(first, logits), f"prefill {i} differs from the first (not bit-identical)"
        del state
    result["prefill_s"] = times
    result["prefill_first_s"] = times[0]
    result["prefill_repeat_s"] = min(times[1:]) if len(times) > 1 else None
    result["dram_after_prefill_bytes"] = _dram_bytes(mesh_device)

    if os.environ.get("V41_DEPTH_TRACE", "1") == "1":
        t0 = time.time()
        trace = V41PrefillTrace(model, tokens, 1)
        result["trace_capture_s"] = round(time.time() - t0, 1)
        result["trace_segments"] = trace.num_segments
        traced_times = []
        for i in range(repeats):
            t0 = time.time()
            traced_logits, _ = trace.run(tokens)
            traced_times.append(round(time.time() - t0, 3))
        # the traced last-row logits equal the untraced ones (the trace test checks every cache; this checks
        # what
        # generation would consume)
        assert torch.equal(traced_logits, first[-1:]), "traced last-row logits differ from the untraced prefill"
        device_times = []
        for i in range(repeats):
            ttnn.synchronize_device(mesh_device)
            t0 = time.time()
            for controller in trace.controllers:
                controller.replay(blocking=False)
            ttnn.synchronize_device(mesh_device)
            device_times.append(round(time.time() - t0, 3))
        trace.release()
        result["traced_prefill_s"] = traced_times
        result["traced_prefill_min_s"] = min(traced_times)
        result["traced_device_only_s"] = device_times
        result["traced_device_only_min_s"] = min(device_times)
        logger.info(f"depth: traced prefill {traced_times} s, device-only replay {device_times} s")

    # positions seq-scored .. seq-1 predict tokens seq-scored+1 .. seq (the last one has no target)
    logp = torch.log_softmax(first.float(), dim=-1)[:-1]
    targets = tokens[seq - scored + 1 :]
    top5 = logp.topk(5, dim=-1).indices
    result["top1_acc"] = round((top5[:, 0] == targets).float().mean().item(), 4)
    result["top5_acc"] = round((top5 == targets[:, None]).any(-1).float().mean().item(), 4)
    result["mean_logprob"] = round(logp.gather(1, targets[:, None]).mean().item(), 4)
    result["ppl"] = round(float(torch.exp(-logp.gather(1, targets[:, None]).mean())), 2)
    result["argmax_tokens"] = tokenizer.decode(top5[:64, 0].tolist())
    result["target_tokens"] = tokenizer.decode(targets[:64].tolist())
    result["rss_peak_gb"] = round(_rss_gb(), 1)
    logger.info("V41_DEPTH_RESULT " + json.dumps(result))
    out = os.environ.get("V41_DEPTH_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
