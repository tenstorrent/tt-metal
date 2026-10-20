# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text demo: real prompts through the resident mesh model, TTFT and decode tok/s like the
other tt-metal demos (``models/demos/gpt_oss/demo/text_demo.py``, ``models/tt_transformers/demo``).

    # 8 prompts, 45 layers, rows-6-7 (2,4) block of the Galaxy (bfp4 experts fit a 2x4; bfp8 needs the 8x4)
    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_EXPERT_DTYPE=bfp4 GLM53_MODEL_EXPERT_DOWN_DTYPE=bfp4 \\
      TT_VISIBLE_DEVICES=25,24,29,28,21,20,17,16 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto MESH_DEVICE=P150x8 \\
      pytest models/demos/glm53_flash/demo/text_demo.py -s -k "questions_8 and 4users"
    # the whole Galaxy, 32 users (4 per row) or 256 users (32 per row), bfp8 experts
    GLM53_MESH_SHAPE=8,4 pytest models/demos/glm53_flash/demo/text_demo.py -s -k "prefill_128 and 32users"

Flow per run: tokenizer + chat template (``instruct``) -> one prefill per user (TTFT = prefill + head +
first-token readback, per user) -> one traced decode step replayed ``max_generated_tokens`` times with the
greedy ids fed back (decode tok/s per user = 1 / mean step, aggregate = users x that) -> detokenized text per
user, a metrics table, and ``generated/glm53_flash_demo/<tag>.json``.

Knobs: ``GLM53_MESH_SHAPE`` (rows,cols; default 2,4), ``GLM53_DEMO_LAYERS`` (default 0-44; e.g. 0-5 for a
plumbing check), ``GLM53_MODEL_EXPERT_DTYPE`` / ``GLM53_MODEL_EXPERT_DOWN_DTYPE`` (bfp8 | bfp4),
``GLM53_MODEL_TRACE_MB`` (trace region, default 512), ``GLM53_DEMO_WARMUP`` (decode steps excluded from the
rate, default 2).

Current limitation (sequence-parallel prefill v1): a prompt prefilled into user slot ``u`` lands in slot ``u``
of EVERY row, so a multi-row mesh runs ``users_per_row`` distinct prompts replicated over the rows; the
decode step still processes all ``rows x users_per_row`` slots (the aggregate rate counts them), and the
texts are reported for the distinct prompts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
# tensorbin weight cache shared with tests/test_mesh_model.py ("none" disables)
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
WARMUP = int(os.environ.get("GLM53_DEMO_WARMUP", "2"))
SAMPLE_DIR = Path(__file__).parent / "sample_prompts"
TTT_SAMPLES = Path("models/tt_transformers/demo/sample_prompts")


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_DEMO_LAYERS", "0-44")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def load_prompts(path: Path, n: int) -> list[str]:
    data = json.loads(Path(path).read_text())
    prompts = [d["prompt"] if isinstance(d, dict) else str(d) for d in data]
    assert prompts, path
    return [prompts[i % len(prompts)] for i in range(n)]


def encode(tok, prompt: str, instruct: bool) -> list[int]:
    if instruct and getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]
    return [int(t) for t in tok(prompt).input_ids]


def eos_ids(snapshot: Path, tok) -> set[int]:
    ids = set()
    gc = snapshot / "generation_config.json"
    if gc.exists():
        v = json.loads(gc.read_text()).get("eos_token_id", [])
        ids.update(v if isinstance(v, list) else [v])
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize(
    "input_prompts, users_per_row, max_seq_len, max_generated_tokens, instruct",
    [
        pytest.param(SAMPLE_DIR / "questions_8.json", 4, 1024, 64, True, id="questions_8-4users"),
        pytest.param(
            TTT_SAMPLES / "input_data_questions_prefill_128.json", 4, 1024, 128, True, id="prefill_128-4users"
        ),
        pytest.param(
            TTT_SAMPLES / "input_data_questions_prefill_128.json", 32, 1024, 128, True, id="prefill_128-32users"
        ),
        pytest.param(TTT_SAMPLES / "input_data_long_1k.json", 4, 2048, 64, True, id="long_1k-4users"),
        pytest.param(TTT_SAMPLES / "input_data_long_4k.json", 4, 8192, 64, True, id="long_4k-4users"),
    ],
)
def test_text_demo(mesh_device, input_prompts, users_per_row, max_seq_len, max_generated_tokens, instruct):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    stop = eos_ids(snapshot, tok)
    layers = _layers()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    rows, cols = MESH
    prompts = load_prompts(input_prompts, users_per_row)
    prompt_ids = [encode(tok, p, instruct) for p in prompts]
    isl = [len(p) for p in prompt_ids]
    assert max(isl) + max_generated_tokens <= max_seq_len, (max(isl), max_generated_tokens, max_seq_len)
    logger.info(
        f"mesh {rows}x{cols}, {users_per_row} users/row ({rows * users_per_row} slots), {len(layers)} layers, prompts {isl} tokens"
    )

    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=users_per_row,
        max_seq_len=max_seq_len,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    t_build = time.perf_counter() - t0
    U = model.total_users
    logger.info(
        f"model built in {t_build:.0f} s ({len(layers)} layers resident, experts {EXPERT_DTYPE}/{EXPERT_DOWN_DTYPE})"
    )

    # ---- decode trace first (compile + capture on dummy ids at position 0; the prefills below overwrite the state)
    trace_id, (_, _, ids_tr) = model.capture(torch.zeros(U, dtype=torch.long), [0] * U)
    model.reset_kda_states()

    # ---- prefill: one prompt per user slot (TTFT per user); one untimed warm-up prefill compiles the prefill
    # programs (the timed prefill of user 0 overwrites its slot)
    model.prefill(torch.tensor(prompt_ids[0], dtype=torch.long), user=0)
    ttnn.synchronize_device(mesh_device)
    ttft_ms, first_tokens = [], []
    for u in range(users_per_row):
        ids = torch.tensor(prompt_ids[u], dtype=torch.long)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        _, logits_last, owner_row = model.prefill(ids, user=u)
        first = model.first_token(logits_last, owner_row)
        ttft_ms.append((time.perf_counter() - t0) * 1000)
        first_tokens.append(first)
        logger.info(
            f"user {u}: prefill {isl[u]} tokens, TTFT {ttft_ms[-1]:.0f} ms, first token {first} {tok.decode([first])!r}"
        )

    # ---- decode: every slot of every row runs (rows replicate the prompts, SP prefill v1)
    gen = [[first_tokens[u % users_per_row]] for u in range(U)]
    done = [False] * U
    positions = [isl[u % users_per_row] for u in range(U)]  # the first generated token sits at position isl
    step_ms = []
    for step in range(max_generated_tokens - 1):
        model.load_ids(torch.tensor([g[-1] for g in gen], dtype=torch.long))
        model.set_positions(positions)
        t0 = time.perf_counter()
        model.replay(trace_id)
        nxt = model.ids_to_host(ids_tr)
        step_ms.append((time.perf_counter() - t0) * 1000)
        for u in range(U):
            tid = int(nxt[u])
            if not done[u]:
                gen[u].append(tid)
                if tid in stop:
                    done[u] = True
            positions[u] += 1
        if all(done):
            break
    rate = step_ms[WARMUP:] or step_ms
    mean_ms = sum(rate) / len(rate)
    tok_s_user = 1000.0 / mean_ms
    tok_s_agg = tok_s_user * U

    # ---- report
    texts = []
    for u in range(users_per_row):
        out = gen[u]
        text = tok.decode([t for t in out if t not in stop], skip_special_tokens=True)
        texts.append(text)
        logger.info(f"=== user {u} (ISL {isl[u]}, {len(out)} generated) ===\nPROMPT: {prompts[u]}\nOUTPUT: {text}")
    logger.info(
        f"PERF GLM-5.3-Flash {len(layers)} layers on {rows}x{cols}: ISL {min(isl)}-{max(isl)} tokens, "
        f"{U} user slots ({users_per_row} distinct prompts): TTFT mean {sum(ttft_ms) / len(ttft_ms):.0f} ms "
        f"(min {min(ttft_ms):.0f}, max {max(ttft_ms):.0f}); decode step {mean_ms:.2f} ms -> {tok_s_user:.1f} tok/s/user, "
        f"{tok_s_agg:.0f} tok/s aggregate over {len(rate)} steps ({len(step_ms)} total, {WARMUP} warm-up)"
    )
    out_dir = Path("generated/glm53_flash_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{rows}x{cols}_{users_per_row}upr_{Path(input_prompts).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out_dir / f"{tag}.json").write_text(
        json.dumps(
            {
                "model": "GLM-5.3-Flash",
                "layers": len(layers),
                "mesh": [rows, cols],
                "users_per_row": users_per_row,
                "user_slots": U,
                "expert_dtype": str(EXPERT_DTYPE),
                "isl": isl,
                "max_generated_tokens": max_generated_tokens,
                "build_s": t_build,
                "ttft_ms": ttft_ms,
                "decode_step_ms": step_ms,
                "decode_tok_s_user": tok_s_user,
                "decode_tok_s_aggregate": tok_s_agg,
                "generated_ids": gen[:users_per_row],
                "texts": texts,
            },
            indent=1,
        )
    )
    logger.info(f"metrics written to {out_dir / (tag + '.json')}")
    ttnn.release_trace(mesh_device, trace_id)
    assert all(len(g) > 1 for g in gen[:users_per_row]), "no tokens generated"
