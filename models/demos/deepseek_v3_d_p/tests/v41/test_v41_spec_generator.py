# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The served speculative round through the generator's contract (``TT_V41_SPEC=K``), driven as
the plugin's model runner drives it, without vLLM: ``initialize_vllm_model`` (the verify decoder with the drafter
attached), the two-phase warm-up (the plain 32-slot capture, then slot 0's rounds at 1 .. K + 1 rows captured), then
one request at a time on a real prompt: its prefill (``prefill_forward``, chunked as the plugin's), then
``V41_SG_STEPS`` ``decode_forward`` calls feeding back the token each returned (vLLM's loop; the generator hands the
round's accepted tokens over one per call, running the device only when its buffer is empty), greedy. Request A
speculates; request B, the same prompt, runs plain on the same decoder (the generator's rings owner cleared: every
step a plain row), and the two texts must be equal. The record keeps the per-call wall, the buffered calls, the
rounds' accepted counts, the DRAM and trace bytes. ``V41_SG_PROMPT`` tokens (default 300; the served short chunk
takes 1,028), ``V41_SG_OUT`` the JSON."""

import json
import os
import time

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table, _trace_bytes
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint


class _Params:  # the plugin's TTSamplingParams as lists per row: argmax (temperature 0) on every row
    def __init__(self, n):
        self.temperature, self.top_k, self.top_p = [0.0] * n, [1] * n, [0.0] * n
        self.repetition_penalty = 1.0


@pytest.mark.timeout(int(os.environ.get("V41_SG_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_SG_TRACE_REGION", 2 * 1024**3))),
    indirect=True,
)
def test_v41_spec_generator(mesh_device, device_params, monkeypatch):
    k = int(os.environ.get("TT_V41_SPEC", "0"))
    assert k > 0, "run with TT_V41_SPEC=K"
    layers = parse_layers(os.environ.get("V41_SG_LAYERS", "all"))
    prompt_len = int(os.environ.get("V41_SG_PROMPT", 300))
    steps = int(os.environ.get("V41_SG_STEPS", 32))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    batch = int(os.environ.get("V41_SG_BATCH", 32))
    max_seq_len = int(os.environ.get("V41_SG_MAX_SEQ", 1053696))
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    result = {
        "k": k,
        "layers": len(layers),
        "prompt": prompt_len,
        "steps": steps,
        "chunk": chunk,
        "batch": batch,
        "max_model_len": max_seq_len,
        "flags": {kk: v for kk, v in os.environ.items() if kk.startswith(("V41_", "TT_V41_"))},
    }
    t0 = time.time()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    t0 = time.time()
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup_bytes"] = _dram_bytes(mesh_device)
    result["trace_after_warmup_bytes"] = _trace_bytes(mesh_device)
    result["captures"] = sorted(model.decoder.captures)
    logger.info(
        f"spec generator: built in {result['build_s']} s, warmed up in {result['warmup_s']} s; captures {result['captures']}"
    )

    text = orc.text_tokens(prompt_len + 8)[0]
    prompt = text[:prompt_len]
    tokenizer = model.tokenizer

    def prefill(block: int) -> int:
        n = int(prompt.numel())
        out = None
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out = model.prefill_forward(
                tokens=prompt.view(1, -1).to(torch.int32),
                prompt_lens=torch.tensor([end]),
                page_table=_page_table([block]),
                start_pos=torch.tensor([start], dtype=torch.int32),
                kv_cache=kv,
                enable_trace=True,
                prompt_total_lens=torch.tensor([n]),
            )
        assert out.shape == (1, 1, C.VOCAB_SIZE)
        return int(out[0, 0].argmax())

    def run(block: int, spec: bool) -> dict:
        t0 = time.time()
        first = prefill(block)
        rec = {
            "prefill_s": round(time.time() - t0, 2),
            "first": first,
            "tokens": [first],
            "calls": [],
            "stats0": dict(model.spec_stats),
        }
        tok = first
        for s in range(steps):
            p = prompt_len + s
            if not spec:
                model.ring_owner = None  # no drafts for any slot: plain rows
                model.decoder.draft_plain = False  # and no drafter in the graph: the plain step
            t1 = time.time()
            out = model.decode_forward(
                tokens=torch.tensor([[tok]], dtype=torch.int32),
                start_pos=torch.tensor([p], dtype=torch.int32),
                page_table=_page_table([block]),
                kv_cache=kv,
                enable_trace=True,
                read_from_device=True,
                sampling_params=_Params(1),
            )
            wall = time.time() - t1
            tok = int(out[0])
            rec["tokens"].append(tok)
            last = model.decoder.last
            rec["calls"].append(
                {
                    "s": s,
                    "wall_s": round(wall, 4),
                    "buffered": model.spec_stats["buffered_steps"] > rec.get("_buffered", 0),
                    "variant": last.get("variant"),
                    "mode": last.get("mode"),
                    "device_s": last.get("device_s"),
                }
            )
            rec["_buffered"] = model.spec_stats["buffered_steps"]
        rec.pop("_buffered")
        rec["stats"] = dict(model.spec_stats)
        rec["text"] = tokenizer.decode(rec["tokens"])
        rec["wall_s_per_token"] = round(sum(c["wall_s"] for c in rec["calls"]) / steps, 4)
        steady = [c for c in rec["calls"] if c["mode"] == "replay" or c["buffered"]]
        rec["steady_calls"] = len(steady)
        rec["steady_s_per_token"] = round(sum(c["wall_s"] for c in steady) / max(1, len(steady)), 4)
        logger.info(
            f"spec generator {'spec' if spec else 'plain'}: {json.dumps({kk: v for kk, v in rec.items() if kk not in ('calls', 'tokens')})}"
        )
        return rec

    result["spec"] = run(7, True)
    result["plain"] = run(9, False)
    result["texts_equal"] = result["spec"]["tokens"] == result["plain"]["tokens"]
    result["dram_after_bytes"] = _dram_bytes(mesh_device)
    logger.info("V41_SPEC_GENERATOR_RESULT " + json.dumps(result))
    out = os.environ.get("V41_SG_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    for trace in model.prefill_traces.values():
        trace.release()
    model.decoder.release()
    assert result["texts_equal"], "the speculative text differs from the plain text"
