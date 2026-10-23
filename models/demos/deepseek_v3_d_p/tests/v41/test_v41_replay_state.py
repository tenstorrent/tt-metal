# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The one-chunk prefill replay's device time and logits after decode steps of 32 rows and of 1 row: the
served path reads the 2,052-token capture at 0.26 s after 1-row decode steps and at
0.32 to 0.36 s after 32-row ones, flat across a wave. Through ``DeepseekV41ForCausalLM`` as the generator
test:
phase A prefills the book prompt (``V41_RS_PROMPT`` tokens, a replay) ``V41_RS_REPEAT`` times; phase B
prefills
``V41_RS_BATCH`` requests and decodes ``V41_RS_STEPS`` steps of all of them; C prefills the book again; D
decodes
``V41_RS_STEPS`` steps of one row; E prefills the book again; F repeats B and C. Each prefill's logits are
compared
bit for bit with phase A's. ``V41_RS_PROFILE=1`` reads the device profiler after every prefill replay (run
under
``python -m tracy`` with ``TT_METAL_DEVICE_PROFILER=1``). Results: ``V41_RS_RESULT {json}``, ``V41_RS_OUT``.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_decode import TRACE_REGION
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint


@pytest.mark.timeout(int(os.environ.get("V41_RS_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_RS_TRACE_REGION", TRACE_REGION))),
    indirect=True,
)
def test_v41_replay_state(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_RS_LAYERS", "all"))
    prompt = int(os.environ.get("V41_RS_PROMPT", 2052))
    steps = int(os.environ.get("V41_RS_STEPS", 127))
    repeat = int(os.environ.get("V41_RS_REPEAT", 3))
    batch = int(os.environ.get("V41_RS_BATCH", 32))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    profile = os.environ.get("V41_RS_PROFILE", "0") == "1"
    monkeypatch.setenv("TT_V41_CHUNK", str(chunk))
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    monkeypatch.setenv("TT_V41_TRACE_PREFILL_LENGTHS", str(prompt))
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    max_seq_len = int(os.environ.get("V41_RS_MAX_SEQ", prompt + steps + 64))
    result = {
        "layers": layers,
        "prompt": prompt,
        "steps": steps,
        "batch": batch,
        "chunk": chunk,
        "max_seq": max_seq_len,
    }

    t0 = time.time()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    result["build_warmup_s"] = round(time.time() - t0, 1)

    need = (batch + 1) * (prompt + steps + 2)
    try:
        text = orc.text_tokens(need)[0]
    except ValueError as e:
        have = int(str(e).split()[0])
        book = orc.text_tokens(have)[0]
        text = book.repeat(-(-need // have))[:need]
    windows = [text[i * (prompt + steps + 2) : (i + 1) * (prompt + steps + 2)] for i in range(batch + 1)]
    book = windows[0]

    def prefill(block: int, tokens: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        t0 = time.time()
        out = model.prefill_forward(
            tokens=tokens[None].to(torch.int32),
            prompt_lens=torch.tensor([int(tokens.numel())]),
            page_table=_page_table([block]),
            start_pos=torch.tensor([0], dtype=torch.int32),
            kv_cache=kv,
            enable_trace=True,
        )
        wall = time.time() - t0
        waits = [t.last.get("wait", 0.0) for t in model.prefill_traces.values() if getattr(t, "last", None)]
        if profile:
            ttnn.ReadDeviceProfiler(mesh_device)
        return out[0, 0].clone(), wall, max(waits) if waits else -1.0

    sample_on_device = DeepseekV41ForCausalLM.SAMPLE_ON_DEVICE

    class _Params:  # the plugin's TTSamplingParams as lists per row: argmax on every row
        def __init__(self, n: int):
            self.temperature, self.top_k, self.top_p = [0.0] * n, [1] * n, [0.0] * n
            self.repetition_penalty = 1.0

    def decode(rows: list[tuple[int, int, int]]) -> list[int]:
        """rows: (first block, token, position) -> the argmax token per row."""
        out = model.decode_forward(
            tokens=torch.tensor([[t] for _, t, _ in rows], dtype=torch.int32),
            start_pos=torch.tensor([p for _, _, p in rows], dtype=torch.int32),
            page_table=_page_table([b for b, _, _ in rows]),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_Params(len(rows)) if sample_on_device else None,
        )
        if sample_on_device:
            return [int(x) for x in out]
        return [int(out[i, 0].argmax()) for i in range(len(rows))]

    phases = {}
    reference = None
    next_block = 100

    def prefill_phase(name: str) -> None:
        nonlocal reference, next_block
        entries = []
        for _ in range(repeat):
            logits, wall, wait = prefill(next_block, book[:prompt])
            next_block += 1
            if reference is None:
                reference = logits
            same = bool(torch.equal(logits, reference))
            diff = float((logits - reference).abs().max())
            entries.append(
                {
                    "wall_s": round(wall, 3),
                    "wait_s": round(wait, 3),
                    "equal": same,
                    "max_abs_diff": diff,
                    "argmax": int(logits.argmax()),
                }
            )
            logger.info(
                f"replay state {name}: prefill wall {wall:.3f} s wait {wait:.3f} s equal {same} max diff {diff:.3g} argmax {int(logits.argmax())}"
            )
        phases[name] = entries

    def wave(name: str) -> None:
        nonlocal next_block
        blocks = []
        waits = []
        for i in range(1, batch + 1):
            logits, wall, wait = prefill(next_block, windows[i][:prompt])
            blocks.append((next_block, int(logits.argmax())))
            waits.append(round(wait, 3))
            next_block += 1
        logger.info(f"replay state {name}: the wave's prefill waits {waits}")
        t0 = time.time()
        for s in range(steps):
            out = decode([(b, t, prompt + s) for b, t in blocks])
            blocks = [(b, out[i]) for i, (b, _) in enumerate(blocks)]
        phases[name] = {"decode_rows": batch, "steps": steps, "decode_s": round(time.time() - t0, 1), "waits": waits}
        logger.info(
            f"replay state {name}: {batch} prefills and {steps} decode steps of {batch} rows in {phases[name]['decode_s']} s"
        )

    def single(name: str) -> None:
        nonlocal next_block
        logits, wall, wait = prefill(next_block, windows[1][:prompt])
        b, t = next_block, int(logits.argmax())
        next_block += 1
        t0 = time.time()
        for s in range(steps):
            out = decode([(b, t, prompt + s)])
            t = out[0]
        phases[name] = {"decode_rows": 1, "steps": steps, "decode_s": round(time.time() - t0, 1)}
        logger.info(f"replay state {name}: 1 prefill and {steps} decode steps of 1 row in {phases[name]['decode_s']} s")

    prefill_phase("A")
    wave("B")
    prefill_phase("C")
    single("D")
    prefill_phase("E")
    wave("F")
    prefill_phase("G")
    single("H")
    prefill_phase("I")
    result["phases"] = phases
    logger.info("V41_RS_RESULT " + json.dumps(result))
    out = os.environ.get("V41_RS_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    for trace in model.prefill_traces.values():
        trace.release()
    model.decoder.release()
