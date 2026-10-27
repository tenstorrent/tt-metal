# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer-30B text demo: data-parallel replicas of a 1 x TP model, paged KV, traced decode.

Env: MESH_DEVICE (optional), HF_MODEL. Select a configuration with ``-k`` on the parametrize ids.
"""

import json
import os
import time

import pytest
from loguru import logger

from models.demos.muse_glimmer_30b.tt.generator import MuseGlimmerGenerator
from models.demos.muse_glimmer_30b.tt.mesh_utils import device_params, dp_submeshes, system_mesh_shape_from_env
from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir
from models.demos.muse_glimmer_30b.tt.tokenizer_utils import decode, encode_chat, load_tokenizer

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "sample_prompts.json")


_MAX_NEW_TOKENS = 128
_MAX_SEQ_LEN = 4096
_TOP_P = {"temperature": 1.0, "top_p": 0.95, "top_k": 32}
_MAX_CONTEXT = 131072
_SAMPLE_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tt_transformers", "demo", "sample_prompts")
# Same per-ISL inputs as the Gemma4 demo: nominal ISL -> tt_transformers sample prompt file.
_ISL_PROMPTS = {
    128: "input_data_questions_prefill_128.json",
    256: "input_data_questions_prefill_256.json",
    1024: "input_data_long_1k.json",
    2048: "input_data_long_2k.json",
    4096: "input_data_long_4k.json",
    8192: "input_data_long_8k.json",
    16384: "input_data_long_16k.json",
    32768: "input_data_long_32k.json",
    65536: "input_data_long_64k.json",
    131072: "input_data_long_256k.json",  # the book alone is ~103k Muse tokens: use the repeated one, clipped to fit
}


def _isl_prompt(tokenizer, isl: int) -> list[int]:
    """Chat-encoded Gemma4 demo input for ``isl``, clipped from the front of the book to fit the context."""
    from models.demos.gemma4.demo.text_demo_v2 import load_inputs

    text = load_inputs(os.path.join(_SAMPLE_PROMPTS, _ISL_PROMPTS[isl]), 1, instruct=True)[0]
    prompt = encode_chat(tokenizer, text)
    excess = len(prompt) - (_MAX_CONTEXT - _MAX_NEW_TOKENS)
    if excess > 0:  # text is ```book```\n\nquestion: drop book tokens, keep the question
        body = tokenizer(text[3:], add_special_tokens=False).input_ids
        prompt = encode_chat(tokenizer, "```" + tokenizer.decode(body[excess + 32 :]))  # 32: retokenization slack
    return prompt


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@(
    pytest.mark.parametrize("mesh_device", [system_mesh_shape_from_env()], indirect=True)
    if system_mesh_shape_from_env() is not None
    else (lambda f: f)
)
@pytest.mark.parametrize(
    "data_parallel, tp, batch_size_per_replica, sampling",
    [
        (1, 4, 1, None),
        (1, 4, 32, None),
        (1, 8, 1, None),
        (1, 8, 32, None),
        (8, 4, 32, None),
        (4, 8, 32, None),
        (1, 4, 32, _TOP_P),
    ],
    ids=[
        "dp1-tp4-b1-greedy",
        "dp1-tp4-b32-greedy",
        "dp1-tp8-b1-greedy",
        "dp1-tp8-b32-greedy",
        "dp8-tp4-b32-greedy",
        "dp4-tp8-b32-greedy",
        "dp1-tp4-b32-topp",
    ],
)
def test_muse_glimmer_demo(mesh_device, data_parallel, tp, batch_size_per_replica, sampling):
    n_dev = mesh_device.get_num_devices()
    if data_parallel * tp > n_dev:
        pytest.skip(f"dp={data_parallel} x tp={tp} needs more devices than the {n_dev} available")

    submeshes = dp_submeshes(mesh_device, data_parallel, tp)

    ckpt = checkpoint_dir()
    models = []
    for i, sm in enumerate(submeshes):
        t0 = time.time()
        args = MuseGlimmerModelArgs.from_checkpoint(
            ckpt, mesh_device=sm, max_batch_size=batch_size_per_replica, max_seq_len=_MAX_SEQ_LEN
        )
        models.append(MuseGlimmerModel(args, sm))
        logger.info(f"Replica {i + 1}/{len(submeshes)} loaded in {time.time() - t0:.1f}s")

    gen = MuseGlimmerGenerator(models, enable_trace=True)
    gb = gen.global_batch_size

    tokenizer = load_tokenizer(ckpt)
    with open(_PROMPTS_FILE) as f:
        texts = json.load(f)
    texts = [texts[i % len(texts)] for i in range(gb)]
    prompts = [encode_chat(tokenizer, t) for t in texts]

    stop_ids = list(models[0].args.eos_token_ids)

    result = gen.generate(prompts, _MAX_NEW_TOKENS, sampling=sampling, stop_token_ids=stop_ids, seed=0)

    for text, toks in zip(texts, result["tokens"]):
        shown = text if len(text) <= 100 else text[:100] + "..."
        print(f"\n=== PROMPT: {shown}\n--- OUTPUT: {decode(tokenizer, toks)}")

    logger.info(
        f"SUMMARY: TTFT {result['ttft_s']:.2f}s | decode {result['tok_s_per_user']:.2f} tok/s/user | "
        f"total {result['throughput_tok_s']:.1f} tok/s | dp={data_parallel} tp={tp} batch={gb}"
    )
    bds = result.get("step_breakdown_s", [])[1:]
    if bds:
        mean = {k: sum(b[k] for b in bds) / len(bds) * 1e3 for k in bds[0]}
        logger.info(
            "STEP BREAKDOWN (mean ms, steady): "
            + " | ".join(f"{k[:-2]} {v:.1f}" for k, v in mean.items())
            + f" | device_sampling={result.get('device_sampling')}"
        )
    gen.release_traces()


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@(
    pytest.mark.parametrize("mesh_device", [system_mesh_shape_from_env()], indirect=True)
    if system_mesh_shape_from_env() is not None
    else (lambda f: f)
)
@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize(
    "isl", list(_ISL_PROMPTS), ids=[f"isl{n // 1024}k" if n >= 1024 else f"isl{n}" for n in _ISL_PROMPTS]
)
def test_muse_glimmer_isl(mesh_device, tp, isl):
    n_dev = mesh_device.get_num_devices()
    if tp > n_dev:
        pytest.skip(f"tp={tp} needs more devices than the {n_dev} available")

    sm = dp_submeshes(mesh_device, 1, tp)[0]
    ckpt = checkpoint_dir()
    tokenizer = load_tokenizer(ckpt)
    prompt = _isl_prompt(tokenizer, isl)
    max_seq_len = min(-(-(len(prompt) + _MAX_NEW_TOKENS) // 256) * 256, _MAX_CONTEXT)
    args = MuseGlimmerModelArgs.from_checkpoint(
        ckpt, mesh_device=sm, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sm)

    gen = MuseGlimmerGenerator([model], enable_trace=True)
    result = gen.generate([prompt], _MAX_NEW_TOKENS, sampling=None, stop_token_ids=list(args.eos_token_ids), seed=0)

    logger.info(f"prompt tokens: {len(prompt)}\n--- OUTPUT: {decode(tokenizer, result['tokens'][0])}")
    logger.info(
        f"SUMMARY: TTFT {result['ttft_s']:.2f}s | decode {result['tok_s_per_user']:.2f} tok/s/user | "
        f"isl={len(prompt)} tp={tp}"
    )
    gen.release_traces()
