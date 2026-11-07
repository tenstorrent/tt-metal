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
_MAX_CONTEXT = 131072
_SAMPLE_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ttt_compat", "demo", "sample_prompts")
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


def _isl_id(n: int) -> str:
    return f"isl{n // 1024}k" if n >= 1024 else f"isl{n}"


_DFLASH_ISLS = [128, 8192, 65536, 131072]  # subset of _ISL_PROMPTS
_DFLASH_BLOCK = 16  # DFlash2 block size (drafter.cfg.block_size): rows the last verify round writes past the output


def _isl_prompt(tokenizer, isl: int, reserve: int = _MAX_NEW_TOKENS) -> list[int]:
    """Chat-encoded Gemma4 demo input for ``isl``, clipped from the front of the book so that prompt + ``reserve``
    tokens fit the context."""
    from models.demos.gemma4.demo.text_demo_v2 import load_inputs

    text = load_inputs(os.path.join(_SAMPLE_PROMPTS, _ISL_PROMPTS[isl]), 1, instruct=True)[0]
    prompt = encode_chat(tokenizer, text)
    excess = len(prompt) - (_MAX_CONTEXT - reserve)
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
        (1, 4, 32, "default"),
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

    if sampling == "default":
        sampling = models[0].args.default_sampling

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


def _dflash_generator(sm, args, model, enable_trace=True):
    """DFlash2 drafter + selector + generator for batch 1 on `model` (user 0 owns the arange page-table row)."""
    import torch

    import ttnn
    from models.demos.muse_glimmer_30b.tt.dflash.drafter import TtDFlash2Drafter
    from models.demos.muse_glimmer_30b.tt.dflash.generator import DFlash2Generator
    from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector

    drafter = TtDFlash2Drafter(sm, args, model.ccl, model.rope)
    selector = HostCandidateSelector.from_checkpoint()
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = ttnn.from_torch(
        pt,
        device=sm,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(sm),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return DFlash2Generator(model, drafter, selector, pt, pt_tt, list(args.eos_token_ids), enable_trace=enable_trace)


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@(
    pytest.mark.parametrize("mesh_device", [system_mesh_shape_from_env()], indirect=True)
    if system_mesh_shape_from_env() is not None
    else (lambda f: f)
)
@pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])
@pytest.mark.parametrize("isl", list(_ISL_PROMPTS), ids=[_isl_id(n) for n in _ISL_PROMPTS])
@pytest.mark.parametrize("sampling", [None, "default"], ids=["greedy", "sample"])
def test_muse_glimmer_isl(mesh_device, tp, isl, sampling):
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
    if sampling == "default":
        sampling = args.default_sampling

    gen = MuseGlimmerGenerator([model], enable_trace=True)
    result = gen.generate([prompt], _MAX_NEW_TOKENS, sampling=sampling, stop_token_ids=list(args.eos_token_ids), seed=0)

    logger.info(f"prompt tokens: {len(prompt)}\n--- OUTPUT: {decode(tokenizer, result['tokens'][0])}")
    logger.info(
        f"SUMMARY: TTFT {result['ttft_s']:.2f}s | decode {result['tok_s_per_user']:.2f} tok/s/user | "
        f"isl={len(prompt)} tp={tp} | sampling={sampling}"
    )
    gen.release_traces()


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@(
    pytest.mark.parametrize("mesh_device", [system_mesh_shape_from_env()], indirect=True)
    if system_mesh_shape_from_env() is not None
    else (lambda f: f)
)
@pytest.mark.parametrize("tp", [4], ids=["tp4"])
@pytest.mark.parametrize("isl", _DFLASH_ISLS, ids=[_isl_id(n) for n in _DFLASH_ISLS])
@pytest.mark.parametrize("sampling", [None, "default"], ids=["greedy", "sample"])
def test_muse_glimmer_dflash(mesh_device, tp, isl, sampling):
    n_dev = mesh_device.get_num_devices()
    if tp > n_dev:
        pytest.skip(f"tp={tp} needs more devices than the {n_dev} available")

    sm = dp_submeshes(mesh_device, 1, tp)[0]
    ckpt = checkpoint_dir()
    tokenizer = load_tokenizer(ckpt)
    prompt = _isl_prompt(tokenizer, isl, reserve=_MAX_NEW_TOKENS + _DFLASH_BLOCK)
    max_seq_len = min(-(-(len(prompt) + _MAX_NEW_TOKENS + _DFLASH_BLOCK) // 256) * 256, _MAX_CONTEXT)
    args = MuseGlimmerModelArgs.from_checkpoint(
        ckpt, mesh_device=sm, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sm)
    if sampling == "default":
        sampling = args.default_sampling
    gen = _dflash_generator(sm, args, model, enable_trace=True)
    tokens, stats = gen.generate(prompt, _MAX_NEW_TOKENS, sampling=sampling, seed=0)

    logger.info(f"prompt tokens: {len(prompt)}\n--- OUTPUT: {decode(tokenizer, tokens)}")
    mm = stats["mean_ms"]
    logger.info(
        f"SUMMARY: TTFT {stats['ttft_s']:.2f}s | decode {stats['decode_tok_s']:.2f} tok/s/user | "
        f"isl={len(prompt)} tp={tp} | dflash2 mean_accept={stats['mean_accept']:.2f} rounds={stats['rounds']} "
        f"round_ms={mm['total']:.1f} (draft {mm['draft']:.1f} / select {mm['select']:.1f} / "
        f"verify {mm['verify']:.1f} / accept {mm['accept']:.1f} / other {mm['other']:.1f}) | sampling={sampling}"
    )
    logger.info(f"dflash2 stats: {stats}")
    gen.close()
    assert len(tokens) >= 2 and stats["mean_accept"] >= (2.0 if stats["sampling"] is None else 1.5)


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@(
    pytest.mark.parametrize("mesh_device", [system_mesh_shape_from_env()], indirect=True)
    if system_mesh_shape_from_env() is not None
    else (lambda f: f)
)
def test_muse_glimmer_repeat(mesh_device):
    """Repeated generate() calls on one TP4 batch-32 generator compile no new program once the prompt-length buckets
    are known (prefill program plans depend on free L1, which the first decode step must not change; new prompt
    lengths must not compile per-length ops), and the same prompts give the same greedy tokens."""
    if mesh_device.get_num_devices() < 4:
        pytest.skip("needs 4 devices")
    sm = dp_submeshes(mesh_device, 1, 4)[0]
    ckpt = checkpoint_dir()
    args = MuseGlimmerModelArgs.from_checkpoint(ckpt, mesh_device=sm, max_batch_size=32, max_seq_len=_MAX_SEQ_LEN)
    gen = MuseGlimmerGenerator([MuseGlimmerModel(args, sm)], enable_trace=True)
    tokenizer = load_tokenizer(ckpt)
    with open(_PROMPTS_FILE) as f:
        texts = json.load(f)
    prompts = [encode_chat(tokenizer, texts[i % len(texts)]) for i in range(gen.global_batch_size)]
    stop_ids = list(args.eos_token_ids)
    first = gen.generate(prompts, 32, stop_token_ids=stop_ids)["tokens"]
    n_programs = sm.num_program_cache_entries()
    # Other prompt lengths in the same 128-row bucket.
    other = [encode_chat(tokenizer, texts[i % len(texts)] + " Be brief.") for i in range(gen.global_batch_size)]
    assert {len(p) for p in other} - {len(p) for p in prompts} and max(map(len, other)) <= 128
    gen.generate(other, 32, stop_token_ids=stop_ids)
    again = gen.generate(prompts, 32, stop_token_ids=stop_ids)["tokens"]
    gen.release_traces()
    assert sm.num_program_cache_entries() == n_programs, "a later generate() compiled new programs"
    assert again == first
