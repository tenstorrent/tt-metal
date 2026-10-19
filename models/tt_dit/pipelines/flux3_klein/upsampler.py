# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein prompt upsampler ("pup") on one Blackhole chip (design record ``docs/M9_DESIGN.md``).

The upsampler is BFL's Apache-2.0 fine-tune of Qwen3.5-9B (``flux-3-image-klein-pup``): a RAW user prompt, with
up to ten reference images and optional layout boxes, becomes the long caption the DiT was trained on. The
reference implementation is the vendored ``reference/flux3_klein/pup.py`` (vLLM, greedy, presence penalty 1.5 for
text-to-image and 0 for edits, stop on ``<|im_end|>`` / ``<|endoftext|>``, at most 3584 new tokens). This module
reproduces that call step by step on the ``models/demos/blackhole/qwen36`` single-device path:

  1. :func:`build_pup_inputs` (host, no device): ``bbox.plan`` picks the system prompt and the locked box rows, the
     references go through the reference's own ``policy_reference`` sizing (LANCZOS to the per-count pixel budget
     on a 32 multiple, JPEG q92 round trip, RGB over white), the chat template is rendered with
     ``add_generation_prompt=True, enable_thinking=False`` by the processor (edits) or the tokenizer (text-to-image)
     and the processor / tokenizer turn it into ``input_ids`` (+ ``pixel_values``, ``image_grid_thw``). The CPU
     oracle (``tests/models/flux3_klein/pup_oracle.py``) consumes the SAME object, so a device / oracle difference
     is a numerics difference, never a prompt one.
  2. :class:`Flux3KleinUpsampler` (device): the qwen36 model built by ``create_tt_model`` with ``enable_mtp=False``
     (the pup ships no ``mtp.*`` tensors) under ``QWEN36_EXACT_PROFILE=1`` (bf16 weights everywhere instead of the
     demo's bfp8 / bfp4 profile; the switch is the additive change of design section 4.2), one paged KV cache sized
     for the largest admitted prompt, the chunked prefill trace, the vision tower for edits and the ``Generator``
     decode trace; the next token is chosen on the HOST by :func:`presence_greedy`, vLLM's rule at temperature 0
     (subtract the presence penalty from the logit of every token already generated, then argmax).
  3. :func:`finish_caption`: decode without the stop id, strip, ``RuntimeError("empty generation")`` when empty,
     ``bbox.reconcile`` with the locked rows (layout mode).

Process model (design decision 1): the pup runs STANDALONE on a ``(1, 1)`` mesh opened WITHOUT fabric (a 1-device
mesh of a multi-chip box must not initialise fabric; no CCL is issued on one chip), owned by the object when
``mesh_device`` is not given. The Klein pipeline holds the whole 4x8 ring and ``set_fabric_config`` is
process-global, so the demo runs the pup first in a subprocess (:class:`Flux3KleinUpsamplerSubprocess`, this
module's CLI) and opens the ring afterwards; co-residency on a submesh of the fabric-enabled parent is the M10 item.

The module imports ``ttnn`` and the qwen36 package lazily inside the device methods, so the host helpers (inputs,
sampler, caption) import without a device build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger
from PIL import Image

from ...reference.flux3_klein import bbox
from ...reference.flux3_klein.pup import (
    I2I_SCAFFOLD,
    MAX_REFERENCES,
    PRESENCE_PENALTY,
    SAMPLING,
    STOP_TOKENS,
    policy_reference,
    reference_size,
)
from .weights_flux3_klein import resolve_weights_dir

# --- constants (reference pup.py, the pup snapshot, vision_demo.py) ---------------------------------------------------------

FLUX3_KLEIN_PUP_STOP_IDS = (248046, 248044)
"""``<|im_end|>``, ``<|endoftext|>``: the pup's ``generation_config.json`` ``eos_token_id``; asserted equal to the
tokenizer's ids of ``STOP_TOKENS`` at load."""
FLUX3_KLEIN_PUP_MAX_NEW_TOKENS = int(SAMPLING["max_tokens"])  # 3584
PUP_MAX_SEQ_LEN = 8192
"""The reference engine's ``max_model_len`` (pup.py ``ENGINE``) and the qwen36 vision demo's context."""
PUP_MAX_PROMPT = PUP_MAX_SEQ_LEN - 1  # 8191
"""The longest prompt the reference engine upsamples. vLLM's only host-side length rule is its input processor's
prompt-length check against ``max_model_len`` (``vllm/v1/engine/processor.py``: ``if prompt_len > max_prompt_len:
raise ValueError(...)``; ``pup.py`` hands the request straight to ``LLM.generate``); the scheduler then finishes a
request with reason ``length`` when ``num_tokens >= max_model_len or num_output_tokens >= max_tokens``
(``vllm/v1/core/sched/utils.py::check_stop``), i.e. a prompt shorter than ``max_model_len`` is generated to
``min(max_tokens, max_model_len - prompt)`` new tokens. The port reproduces that (:func:`new_token_budget`) and
refuses a prompt of ``max_model_len`` tokens or more with ``ValueError`` (``build_pup_inputs``): vLLM's check admits
exactly ``max_model_len`` tokens, but such a prompt has no token budget and ends in ``RuntimeError("empty
generation")`` at ``pup.py:121``, so the port's refusal (the pass-through fallback) is the one-token difference at
the edge, chosen on purpose. Read from a vLLM v1 checkout on the development host (not executed: no vLLM in this
environment); the M9 fixer replaced the earlier ``8192 - 3584 = 4608`` cap, which refused 4609..8191-token prompts
the reference upsamples with a shorter tail. The largest shipped-example prompt is 1914 tokens; the ten-reference
worst cases are 3422 (``i2i``) / 3703 (``i2i_bbox`` with one locked box) tokens, so every admitted example keeps
the full 3584."""
PUP_PREFILL_CHUNK = 2048  # vision_demo.py PREFILL_CHUNK
PUP_BLOCK_SIZE = 64  # vision_demo.py BLOCK_SIZE
PUP_SYSTEM_KEYS = ("t2i", "i2i", "t2i_bbox", "i2i_bbox")
PUP_MESH_SHAPE = (1, 1)
PUP_L1_SMALL_SIZE = 24576  # qwen36 GDN_CONV1D_L1_SMALL_SIZE: the prefill depthwise conv needs it
PUP_TRACE_REGION_SIZE = 256 * 1024 * 1024  # vision_demo.py: prefill chunk trace + decode trace
PUP_DEVICE_PARAMS = {
    "l1_small_size": PUP_L1_SMALL_SIZE,
    "num_command_queues": 2,
    "trace_region_size": PUP_TRACE_REGION_SIZE,
}
"""The single-device parameters of the qwen36 demos: NO ``fabric_config`` (see ``open_pup_mesh``)."""
EXACT_PROFILE_ENV = "QWEN36_EXACT_PROFILE"
PUP_CACHE_SUBDIR = "flux3_klein_pup"
"""``$TT_DIT_CACHE_DIR/<this>`` becomes ``TT_CACHE_PATH`` when the caller has not set one: the qwen36 tensor cache
(18.8 GB bf16 + 0.9 GB vision, written once) lands under ``<it>/P150/tensor_cache_bf16/`` instead of a
``model_cache/`` directory relative to the working directory."""
SUBPROCESS_PASS_THROUGH_EXIT = 4
"""The CLI's exit code for a ``ValueError`` (too many references, an aspect over 200:1, a prompt over the cap): the
caller uses the prompt as typed, as ``generate.py:59-61``."""


def open_pup_mesh():
    """A standalone 1-chip mesh with NO fabric (``tests/models/flux2/device_params.py:39-50``): a 1-device mesh of
    a multi-chip box must not initialise fabric (the router sync waits on an unconfigured chip) and a single-chip
    model issues no CCL, so ``set_fabric_config`` is never called here. ``TT_VISIBLE_DEVICES`` pins the chip."""
    import ttnn

    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*PUP_MESH_SHAPE), **PUP_DEVICE_PARAMS)


def new_token_budget(
    prompt_tokens: int, max_new_tokens: int = FLUX3_KLEIN_PUP_MAX_NEW_TOKENS, max_seq_len: int = PUP_MAX_SEQ_LEN
) -> int:
    """vLLM's generation length under ``max_model_len``: ``min(max_tokens, max_model_len - prompt_len)``. A prompt
    of ``max_seq_len`` tokens or more has no room for a token and raises ``ValueError`` (the refusal of
    :func:`build_pup_inputs`, ``PUP_MAX_PROMPT``)."""
    prompt_tokens, max_new_tokens, max_seq_len = int(prompt_tokens), int(max_new_tokens), int(max_seq_len)
    if prompt_tokens >= max_seq_len:
        raise ValueError(f"prompt of {prompt_tokens} tokens leaves no room for a token under max_seq_len {max_seq_len}")
    return max(0, min(max_new_tokens, max_seq_len - prompt_tokens))


def blocks_for(prompt_tokens: int, max_new_tokens: int, max_seq_len: int) -> int:
    """``vision_demo.py::_blocks_for``: paged-KV blocks for the padded prefill bucket plus the generated tokens,
    a multiple of 8 (the flexible chunked SDPA program factory), at least 64, capped at ``max_seq_len`` blocks."""
    bucket = ((prompt_tokens + PUP_PREFILL_CHUNK - 1) // PUP_PREFILL_CHUNK) * PUP_PREFILL_CHUNK
    needed = bucket + max_new_tokens
    blocks = max(64, (needed + PUP_BLOCK_SIZE - 1) // PUP_BLOCK_SIZE)
    blocks = ((blocks + 7) // 8) * 8
    return min(max_seq_len // PUP_BLOCK_SIZE, blocks)


# --- host: inputs, sampler, caption ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PupInputs:
    """One upsampler request after the reference's host preparation (``pup.py:99-119``)."""

    task: str
    """``"t2i"`` | ``"i2i"``."""
    system_key: str
    """``"t2i"`` | ``"i2i"`` | ``"t2i_bbox"`` | ``"i2i_bbox"`` (``bbox.plan``)."""
    locked: tuple | None
    """The locked box rows on the 0-1000 canvas (``bbox.plan``), or ``None``."""
    text: str
    """The rendered chat template (ends with ``assistant\\n<think>\\n\\n</think>\\n\\n``)."""
    input_ids: torch.Tensor
    """int64 ``[1, T]``."""
    pixel_values: torch.Tensor | None
    """fp32 ``[P, 1536]`` (the processor's patches) or ``None``."""
    image_grid_thw: torch.Tensor | None
    """int64 ``[n, 3]`` or ``None``."""
    num_image_tokens: int
    presence_penalty: float
    """``PRESENCE_PENALTY[task]``: 1.5 (t2i) / 0.0 (i2i)."""

    @property
    def prompt_tokens(self) -> int:
        return int(self.input_ids.shape[1])

    def sha256(self) -> str:
        """Over the token ids and, for edits, the pixel values: the oracle cache key."""
        digest = hashlib.sha256(self.input_ids.contiguous().numpy().tobytes())
        if self.pixel_values is not None:
            digest.update(self.pixel_values.float().contiguous().numpy().tobytes())
            digest.update(self.image_grid_thw.contiguous().numpy().tobytes())
        digest.update(f"|{self.presence_penalty}".encode())
        return digest.hexdigest()


def load_system_prompts(snapshot: Path | str | None) -> dict[str, str]:
    """The four system prompts: the snapshot's ``system_prompts.json`` when it exists, else the vendored
    ``reference/flux3_klein/pup_prompts.json`` (``pup.py:20, 93-94``). Keys asserted."""
    shipped = Path(snapshot) / "system_prompts.json" if snapshot is not None else None
    vendored = Path(__file__).resolve().parents[2] / "reference" / "flux3_klein" / "pup_prompts.json"
    source = shipped if shipped is not None and shipped.is_file() else vendored
    prompts = json.loads(source.read_text())
    if set(prompts) != set(PUP_SYSTEM_KEYS):
        raise ValueError(f"{source} holds the system prompts {sorted(prompts)}; expected {sorted(PUP_SYSTEM_KEYS)}")
    return prompts


def build_pup_inputs(
    prompt: str,
    images: Sequence[Image.Image],
    boxes: Sequence[dict] | None,
    *,
    prompts: dict[str, str],
    tokenizer,
    processor,
) -> PupInputs:
    """``pup.py:99-119`` verbatim on the host: more than ``MAX_REFERENCES`` images -> ``ValueError``; ``bbox.plan``;
    edits: ``policy_reference`` per image, the scaffold, the processor's chat template and ``processor(text=...,
    images=...)``; text-to-image: the tokenizer's chat template and ``tokenizer.encode(text,
    add_special_tokens=False)``. A prompt over ``PUP_MAX_PROMPT`` tokens (``max_model_len - 1``) raises
    ``ValueError``."""
    images = list(images)
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"the upsampler takes at most {MAX_REFERENCES} references")
    task = "i2i" if images else "t2i"
    system_key, user_text, locked = bbox.plan(prompt, list(boxes) if boxes else None, len(images))
    if images:
        references = [policy_reference(image, len(images)) for image in images]
        scaffold = I2I_SCAFFOLD[locked is not None].format(instruction=user_text)
        content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = processor(text=text, images=references, return_tensors="pt")
        input_ids = encoded["input_ids"].to(torch.int64)
        pixel_values = encoded["pixel_values"]
        grid = encoded["image_grid_thw"].to(torch.int64)
        merge = int(processor.image_processor.merge_size) ** 2
        num_image_tokens = int(grid.prod(dim=-1).sum().item() // merge)
    else:
        messages = [{"role": "system", "content": prompts[system_key]}, {"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], dtype=torch.int64)
        pixel_values, grid, num_image_tokens = None, None, 0
    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens > PUP_MAX_PROMPT:
        raise ValueError(
            f"prompt of {prompt_tokens} tokens exceeds the upsampler's {PUP_MAX_PROMPT} "
            f"(max_model_len {PUP_MAX_SEQ_LEN} leaves no room for a generated token)"
        )
    return PupInputs(
        task=task,
        system_key=system_key,
        locked=tuple(locked) if locked else None,
        text=text,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=grid,
        num_image_tokens=num_image_tokens,
        presence_penalty=float(PRESENCE_PENALTY[task]),
    )


def presence_greedy(logits: torch.Tensor, generated: Sequence[int], penalty: float) -> int:
    """vLLM's ``SamplingParams(temperature=0, presence_penalty=p)`` on one logit row: in fp32 (a copy), subtract
    ``p`` from the logit of every token id that already occurs in the OUTPUT (prompt tokens excluded, each id once
    regardless of its count), then argmax. ``top_p`` / ``top_k`` / ``min_p`` never change an argmax."""
    v = logits.detach().reshape(-1).float().clone()
    if penalty > 0 and generated:
        v[torch.tensor(sorted(set(int(t) for t in generated)), dtype=torch.long)] -= float(penalty)
    return int(torch.argmax(v).item())


def finish_caption(generated: Sequence[int], locked, *, tokenizer) -> str:
    """``pup.py:120-123``: the text WITHOUT the stop id (vLLM's ``.text`` excludes it), stripped; empty ->
    ``RuntimeError("empty generation")``; ``bbox.reconcile`` with the locked rows in layout mode."""
    ids = [int(t) for t in generated]
    if ids and ids[-1] in FLUX3_KLEIN_PUP_STOP_IDS:
        ids = ids[:-1]
    caption = tokenizer.decode(ids, skip_special_tokens=True).strip() if ids else ""
    if not caption:
        raise RuntimeError("empty generation")
    return bbox.reconcile(caption, [dict(row) for row in locked]) if locked else caption


def token_f1(a_ids: Sequence[int], b_ids: Sequence[int]) -> float:
    """Bag-of-token-ids F1 (SQuAD style, no normalisation): ``2PR / (P + R)`` with the multiset intersection as the
    common count. 1.0 for identical sequences, 0.0 when nothing is shared."""
    a, b = Counter(int(t) for t in a_ids), Counter(int(t) for t in b_ids)
    common = sum((a & b).values())
    if common == 0:
        return 0.0
    precision = common / sum(b.values())
    recall = common / sum(a.values())
    return 2 * precision * recall / (precision + recall)


def first_divergence(a_ids: Sequence[int], b_ids: Sequence[int]) -> int:
    """The index of the first differing token (``min(len)`` when one is a prefix of the other)."""
    i = 0
    while i < min(len(a_ids), len(b_ids)) and int(a_ids[i]) == int(b_ids[i]):
        i += 1
    return i


AUDIT_BOUNDARY = (1, 2)
"""Positions of the teacher-forced audit excluded from its verdict: the first one and the last two. The record's
caption is ``.strip()``ped text (``pup.py:120``), so its first and last text tokens, and the stop after the last one,
need not be the whitespace-carrying tokens the model emitted; every interior position is the model's own choice."""


def audit_target_ids(caption: str, tokenizer, *, locked=None) -> list[int]:
    """The teacher-forcing target of the divergence audit (design 9.3): ``tokenizer.encode(caption)`` + the first
    stop id. Only a caption the model could have emitted is a valid target: a layout answer (``locked`` rows, or a
    box table detected in the caption) is ``bbox.reconcile`` output (rows re-serialised, ids renamed, uncited rows
    dropped), so it raises ``ValueError`` and such a case has no audit."""
    if locked or bbox.detect(caption) is not None:
        raise ValueError("a layout caption is bbox.reconcile output, not a sequence the model emitted: no audit target")
    return [int(t) for t in tokenizer.encode(caption, add_special_tokens=False)] + [FLUX3_KLEIN_PUP_STOP_IDS[0]]


def divergence_audit(
    device: Sequence[tuple[int, int, float]],
    oracle: Sequence[tuple[int, int, int, float]],
    *,
    tie_gap: float | None,
) -> dict:
    """Design 9.3 on one case: ``device`` = ``Flux3KleinUpsampler.teacher_forced`` rows ``(top, target, gap)``,
    ``oracle`` = ``pup_oracle.teacher_forced_gaps`` rows ``(pos, top, target, gap)`` of the SAME target. Over the
    interior positions (``AUDIT_BOUNDARY`` excluded): ``suspicious`` = the device disagrees with the target where
    the oracle agrees, with both gaps; ``tie_scale`` = the largest oracle gap where the oracle itself disagrees
    (the record's own tie scale, what ``TIE_GAP`` is frozen from); ``errors`` = suspicious positions whose oracle
    gap exceeds ``tie_gap``. ``passed`` is True only with a frozen ``tie_gap`` and no error: with ``tie_gap`` None
    the audit reports and never passes."""
    n = len(device)
    if len(oracle) != n:
        raise ValueError(f"device audit has {n} positions, the oracle {len(oracle)}")
    oracle_rows = {int(pos): (int(top), int(tgt), float(gap)) for pos, top, tgt, gap in oracle}
    interior = range(AUDIT_BOUNDARY[0], max(AUDIT_BOUNDARY[0], n - AUDIT_BOUNDARY[1]))
    suspicious, disagreeing_gaps = [], []
    for pos in interior:
        top, tgt, gap = int(device[pos][0]), int(device[pos][1]), float(device[pos][2])
        o_top, o_tgt, o_gap = oracle_rows[pos]
        if o_tgt != tgt:
            raise ValueError(f"position {pos}: device target {tgt} != oracle target {o_tgt}")
        if o_top != o_tgt:
            disagreeing_gaps.append(o_gap)
        if top != tgt and o_top == tgt:
            suspicious.append((pos, top, tgt, gap, o_gap))
    errors = [s for s in suspicious if tie_gap is not None and s[4] > tie_gap]
    return {
        "positions": n,
        "interior": len(interior),
        "dev_agree": sum(1 for top, tgt, _ in device if int(top) == int(tgt)),
        "orc_agree": sum(1 for top, tgt, _ in oracle_rows.values() if top == tgt),
        "suspicious": suspicious,
        "tie_scale": max(disagreeing_gaps, default=0.0),
        "max_suspicious_oracle_gap": max((s[4] for s in suspicious), default=0.0),
        "errors": errors,
        "tie_gap": tie_gap,
        "passed": tie_gap is not None and not errors,
    }


def split_images(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The processor's packed ``pixel_values [P, 1536]`` cut per image (``grid.prod()`` patch rows each, in
    attachment order) with that image's ``[1, 3]`` grid: the per-image calls of ``Flux3KleinUpsampler``
    (``vision_demo``'s single ``get_image_features`` call over several images concatenates the per-image tower
    outputs along the batch dim, which needs equal token counts; references of different sizes fail there)."""
    parts, offset = [], 0
    for i in range(int(image_grid_thw.shape[0])):
        n = int(image_grid_thw[i].prod().item())
        parts.append((pixel_values[offset : offset + n], image_grid_thw[i : i + 1]))
        offset += n
    if offset != int(pixel_values.shape[0]):
        raise ValueError(f"image_grid_thw covers {offset} patch rows, pixel_values has {int(pixel_values.shape[0])}")
    return parts


def vision_padded_rows(patch_rows: int) -> int:
    """The tower's own per-image padding of the patch count (``vision/model.py``: a multiple of 2048, always at
    least one more block): the compile key of the eager vision programs."""
    return ((int(patch_rows) // 2048) + 1) * 2048


# --- device -----------------------------------------------------------------------------------------------------------------------


@dataclass
class PupTimings:
    load_s: float = 0.0
    """Process start to ready: weights, KV cache (the prefill trace is captured on the first request)."""
    vision_s: float = 0.0
    """The (warm) vision tower call of this request; 0 for text-to-image."""
    prefill_s: float = 0.0
    ttft_s: float = 0.0
    """vision + prefill + the first host argmax (the qwen36 demo's definition)."""
    decode_s: float = 0.0
    decode_tok_s: float = 0.0
    """Steady state: every decode step but the first."""
    prompt_tokens: int = 0
    image_tokens: int = 0
    generated_tokens: int = 0
    total_s: float = 0.0
    finish: str = ""
    """``"stop"`` (a stop id) or ``"length"`` (the new-token budget of :func:`new_token_budget`: ``max_tokens``, or
    the context cap for a long prompt)."""
    trace_captured: bool = False
    """Whether this request captured the prefill trace (the first request of a process, or after a release)."""
    vision_compiled: bool = False
    """Whether this request compiled the vision tower (its first use, or a new padded patch count)."""
    dram: str = ""


def configure_pup_env(snapshot: Path | str, *, exact_profile: bool = True) -> None:
    """The qwen36 package reads process-global environment: ``HF_MODEL`` (the snapshot: ONE pup per process),
    ``QWEN36_MTP=0`` (no drafter: the pup ships no ``mtp.*`` tensors), ``QWEN36_EXACT_PROFILE=1`` and
    ``TT_CACHE_PATH`` (``$TT_DIT_CACHE_DIR/flux3_klein_pup`` unless the caller set it). ``QWEN_SDPA_BF8=1`` (bf8 KV)
    is refused under the exact profile."""
    os.environ["HF_MODEL"] = str(snapshot)
    os.environ.setdefault("QWEN36_MTP", "0")
    if exact_profile:
        if os.environ.get("QWEN_SDPA_BF8", "0") == "1":
            raise ValueError("QWEN_SDPA_BF8=1 (bf8 KV cache) is not the exact profile; unset it")
        os.environ[EXACT_PROFILE_ENV] = "1"
    elif os.environ.get(EXACT_PROFILE_ENV) == "1":
        del os.environ[EXACT_PROFILE_ENV]
    if not os.environ.get("TT_CACHE_PATH") and os.environ.get("TT_DIT_CACHE_DIR"):
        os.environ["TT_CACHE_PATH"] = str(Path(os.environ["TT_DIT_CACHE_DIR"]) / PUP_CACHE_SUBDIR)


def load_pup_model(
    mesh_device, snapshot: Path | str, *, max_seq_len: int = PUP_MAX_SEQ_LEN, exact_profile: bool = True
):
    """``create_tt_model(mesh, 1, max_seq_len, hf_model=<snapshot>, enable_mtp=False)`` under
    :func:`configure_pup_env`; returns ``(args, model, tokenizer, processor)``. The host state dict the loader
    returns is dropped (the device holds the weights; the tensor cache serves the next process)."""
    from transformers import AutoProcessor, AutoTokenizer

    configure_pup_env(snapshot, exact_profile=exact_profile)
    from models.demos.blackhole.qwen36.tt.common import create_tt_model

    args, model, state = create_tt_model(
        mesh_device, max_batch_size=1, max_seq_len=max_seq_len, hf_model=str(snapshot), enable_mtp=False
    )
    del state
    if bool(getattr(args, "exact_profile", False)) != bool(exact_profile):
        raise RuntimeError(
            f"qwen36 built with exact_profile={getattr(args, 'exact_profile', None)}; wanted {exact_profile}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    return args, model, tokenizer, processor


def _dram_per_bank(device) -> str:
    import ttnn

    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    trace = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip; "
        f"TRACE {trace.total_bytes_allocated_per_bank} B per bank"
    )


class Flux3KleinUpsampler:
    """The pup on one chip (module docstring). ``mesh_device`` ``None`` opens :func:`open_pup_mesh` and owns it
    (closed by :meth:`close`); a given 1-device mesh is used as is and left open. Never used by the Klein
    pipeline's recorded-caption gates (they pass finished captions through ``bbox.is_finished``)."""

    def __init__(
        self,
        snapshot: Path | str | None = None,
        *,
        mesh_device=None,
        max_seq_len: int = PUP_MAX_SEQ_LEN,
        use_trace: bool = True,
        exact_profile: bool = True,
    ) -> None:
        import ttnn

        t0 = time.time()
        self.snapshot = Path(snapshot) if snapshot is not None else resolve_weights_dir("pup")
        self.max_seq_len = int(max_seq_len)
        self.use_trace = bool(use_trace)
        self.exact_profile = bool(exact_profile)
        self._owns_device = mesh_device is None
        self._device = open_pup_mesh() if mesh_device is None else mesh_device
        try:
            self._load(ttnn)
        except BaseException:
            # An owned mesh is never leaked by a failed load (a shape / weight / stop-id / KV error): closed here,
            # the error re-raised; a caller's mesh is left as it was.
            if self._owns_device:
                self._owns_device = False
                ttnn.close_mesh_device(self._device)
            raise
        self.last_inputs: PupInputs | None = None
        self.last_ids: list[int] | None = None
        self.last_timings: PupTimings | None = None
        ttnn.synchronize_device(self._device)
        self.load_s = time.time() - t0
        logger.info(
            f"RECORD pup load: {self.snapshot.name} max_seq_len {self.max_seq_len} blocks {self.num_blocks} "
            f"exact_profile {self.exact_profile} trace {self.use_trace} in {self.load_s:.1f}s; {_dram_per_bank(self._device)}"
        )

    def _load(self, ttnn) -> None:
        """The init steps after the mesh exists (model, tokenizer, prompts, stop ids, KV cache)."""
        if tuple(int(s) for s in self._device.shape) != PUP_MESH_SHAPE:
            raise ValueError(f"the upsampler runs on a {PUP_MESH_SHAPE} mesh, got {tuple(self._device.shape)}")
        try:
            self._device.enable_program_cache()
        except Exception:  # already enabled by the owner
            pass
        self.args, self.model, self.tokenizer, self.processor = load_pup_model(
            self._device, self.snapshot, max_seq_len=self.max_seq_len, exact_profile=self.exact_profile
        )
        self.prompts = load_system_prompts(self.snapshot)
        stop_ids = tuple(int(self.tokenizer.convert_tokens_to_ids(t)) for t in STOP_TOKENS)
        if stop_ids != FLUX3_KLEIN_PUP_STOP_IDS:
            raise RuntimeError(f"stop ids {stop_ids} differ from {FLUX3_KLEIN_PUP_STOP_IDS}")
        self.stop_ids = stop_ids
        # 128 blocks of 64 = the whole 8192 context: the KV cache of the longest admitted prompt plus its tail.
        self.num_blocks = blocks_for(PUP_MAX_PROMPT, FLUX3_KLEIN_PUP_MAX_NEW_TOKENS, self.max_seq_len)
        self._kv_shape = [self.num_blocks, int(self.args.n_kv_heads), PUP_BLOCK_SIZE, int(self.args.head_dim)]
        self.page_table = torch.arange(self.num_blocks, dtype=torch.int32).unsqueeze(0)
        self.model.allocate_kv_caches(self._kv_shape, ttnn.bfloat16, batch_size=1)
        self._generator = None
        self._prefill_trace_captured = False
        self._vision_patch_rows: set[int] = set()

    # --- accessors ---------------------------------------------------------------------------------------------------------

    @property
    def device(self):
        return self._device

    @property
    def owns_device(self) -> bool:
        return self._owns_device

    def inputs(self, prompt: str, images: Sequence[Image.Image] = (), boxes: Sequence[dict] | None = None) -> PupInputs:
        return build_pup_inputs(
            prompt, images, boxes, prompts=self.prompts, tokenizer=self.tokenizer, processor=self.processor
        )

    # --- internals -----------------------------------------------------------------------------------------------------------

    def _release_traces(self) -> None:
        """Every trace (prefill chunk, decode) released and the KV cache re-allocated: before an eager compile (a
        new vision patch count) while a trace is parked, which would otherwise clobber it."""
        import ttnn

        if self._generator is not None:
            self._generator.release_persistent_capture()
            self._generator = None
        self.model.free_kv_caches()
        self._prefill_trace_captured = False
        self.model.allocate_kv_caches(self._kv_shape, ttnn.bfloat16, batch_size=1)
        ttnn.synchronize_device(self._device)

    def _image_features(self, inputs: PupInputs):
        """``[num_image_tokens, H]`` on device in attachment order: one ``get_image_features`` call for a single
        reference; for several, one call PER IMAGE (``split_images``) and the per-image rows concatenated on the
        host into one device tensor (the tower's own multi-image path concatenates its per-image outputs along the
        batch dim, which requires equal token counts: two references of different sizes, e.g. 832x608 and 864x576
        -> 494 and 486 tokens, raise ``TT_FATAL shapes_match`` in ``ttnn.concat``; M9 fixer finding 13). The
        model's per-request image grid (M-RoPE) is then reset to the WHOLE grid, as the single call would have."""
        import ttnn

        grids = inputs.image_grid_thw
        if int(grids.shape[0]) == 1:
            return self.model.get_image_features(inputs.pixel_values, grids)
        parts = []
        for pixels, grid in split_images(inputs.pixel_values, grids):
            parts.append(self.model.get_image_features(pixels, grid))
        ttnn.synchronize_device(self._device)
        rows = torch.cat([ttnn.to_torch(part) for part in parts], dim=0)
        dtype, layout = parts[0].dtype, parts[0].layout
        for part in parts:
            ttnn.deallocate(part)
        if int(rows.shape[0]) != inputs.num_image_tokens:
            raise RuntimeError(
                f"vision tower gave {int(rows.shape[0])} rows for {inputs.num_image_tokens} image tokens"
            )
        tokens = ttnn.from_torch(
            rows, dtype=dtype, layout=layout, device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.model._req_image_grid_thw = grids
        self.model._req_video_grid_thw = None
        return tokens

    def _vision_tokens(self, inputs: PupInputs, timings: PupTimings):
        """The vision tower on the request's patches (``vision_demo.py::_compute_vision_tokens``): built lazily on
        the first edit (bf16 under the exact profile), compiled before any trace is parked, warm call timed. The
        compile key is the set of per-image padded patch counts (``vision_padded_rows``)."""
        import ttnn

        if inputs.pixel_values is None:
            return None
        padded = sorted({vision_padded_rows(int(grid.prod().item())) for grid in inputs.image_grid_thw})
        compile_needed = self.model.vision_model is None or any(p not in self._vision_patch_rows for p in padded)
        if compile_needed and (self._prefill_trace_captured or self._generator is not None):
            logger.info(f"releasing the traces before the vision compile of {padded} padded patch rows")
            self._release_traces()
        if self.model.vision_model is None:
            t0 = time.time()
            self.model.init_vision_model(dtype=ttnn.bfloat16 if self.exact_profile else ttnn.bfloat8_b)
            logger.info(f"RECORD pup vision tower built in {time.time() - t0:.1f}s; {_dram_per_bank(self._device)}")
        if compile_needed:
            t0 = time.time()
            warm = self._image_features(inputs)
            ttnn.synchronize_device(self._device)
            ttnn.deallocate(warm)
            logger.info(f"RECORD pup vision compile ({padded} padded rows) {time.time() - t0:.1f}s")
            self._vision_patch_rows.update(padded)
            timings.vision_compiled = True
        t0 = time.time()
        tokens = self._image_features(inputs)
        ttnn.synchronize_device(self._device)
        timings.vision_s = time.time() - t0
        return tokens

    def _prefill(self, inputs: PupInputs, vision_tokens, timings: PupTimings) -> torch.Tensor:
        """The chunked prefill (traced: the captured 2048-chunk replay + the masked bucket tail; eager: paged);
        returns the fp32 logit row of the last prompt token."""
        import ttnn
        from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch

        if self.use_trace and not self._prefill_trace_captured:
            t0 = time.time()
            self.model.capture_prefill_trace_chunked(
                self._device, self.page_table, chunk_size=PUP_PREFILL_CHUNK, warmup_masked_buckets=True
            )
            self._prefill_trace_captured = True
            timings.trace_captured = True
            logger.info(f"RECORD pup prefill trace captured in {time.time() - t0:.1f}s; {_dram_per_bank(self._device)}")
        t0 = time.time()
        logits = prefill_dispatch(
            self.model,
            inputs.input_ids,
            self.page_table,
            torch.tensor([inputs.prompt_tokens]),
            self.use_trace,
            vision_tokens=vision_tokens,
        )
        ttnn.synchronize_device(self._device)
        row = ttnn.to_torch(logits).reshape(-1).float()
        timings.prefill_s = time.time() - t0
        if torch.isnan(row).any():
            raise RuntimeError("NaN in the prefill logits")
        return row

    def _decode_step(self, token: int, position: int) -> torch.Tensor:
        out = self._generator.decode_forward(
            torch.tensor([[token]], dtype=torch.long),
            torch.tensor([position]),
            page_table=self.page_table,
            kv_cache=None,
            enable_trace=self.use_trace,
            read_from_device=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        row = (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
        if torch.isnan(row).any():
            raise RuntimeError("NaN in the decode logits")
        return row

    def _start_decode(self, first_token: int, position: int) -> None:
        """The ``Generator`` (once per process) and, on the traced path, ``prime_decode_trace`` once per request:
        the decode trace is captured / replayed at the real post-prefill position with the GDN recurrent state
        snapshot and restored around it (the capture runs the forward twice; GDN state is not idempotent)."""
        from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace
        from models.tt_transformers.tt.generator import Generator

        if self._generator is None:
            self._generator = Generator([self.model], [self.model.args], self._device)
        if self.use_trace:
            prime_decode_trace(
                self._generator,
                self.model,
                torch.tensor([[first_token]], dtype=torch.long),
                torch.tensor([position]),
                self.page_table,
            )

    # --- the request -----------------------------------------------------------------------------------------------------

    def generate_ids(
        self, inputs: PupInputs, *, max_new_tokens: int = FLUX3_KLEIN_PUP_MAX_NEW_TOKENS
    ) -> tuple[list[int], PupTimings]:
        """Greedy decode with the presence penalty; returns the generated ids WITHOUT the stop id (the stop id ends
        the sequence and is never appended) and the timings. The count is capped by :func:`new_token_budget`
        (``max_new_tokens``, or the context cap ``max_seq_len - prompt`` for a long prompt: vLLM's ``length``
        finish); a prompt of ``max_seq_len`` tokens or more raises ``ValueError``."""
        import ttnn

        max_new_tokens = new_token_budget(inputs.prompt_tokens, max_new_tokens, self.max_seq_len)
        timings = PupTimings(
            load_s=self.load_s, prompt_tokens=inputs.prompt_tokens, image_tokens=inputs.num_image_tokens
        )
        t_total = time.time()
        vision_tokens = self._vision_tokens(inputs, timings)
        logits = self._prefill(inputs, vision_tokens, timings)
        penalty = inputs.presence_penalty
        token = presence_greedy(logits, [], penalty)
        timings.ttft_s = timings.vision_s + timings.prefill_s
        generated: list[int] = []
        decode_times: list[float] = []
        position = inputs.prompt_tokens
        timings.finish = "stop" if token in self.stop_ids else "length"
        if token not in self.stop_ids and max_new_tokens > 0:
            generated.append(token)
            self._start_decode(token, position)
            t_decode = time.time()
            while len(generated) < max_new_tokens:
                t_step = time.time()
                logits = self._decode_step(token, position)
                decode_times.append(time.time() - t_step)
                token = presence_greedy(logits, generated, penalty)
                position += 1
                if token in self.stop_ids:
                    timings.finish = "stop"
                    break
                generated.append(token)
            timings.decode_s = time.time() - t_decode
        if vision_tokens is not None:
            ttnn.deallocate(vision_tokens)
        steady = decode_times[1:] if len(decode_times) > 1 else decode_times
        timings.decode_tok_s = (len(steady) / sum(steady)) if steady and sum(steady) > 0 else 0.0
        timings.generated_tokens = len(generated)
        timings.total_s = time.time() - t_total
        timings.dram = _dram_per_bank(self._device)
        return generated, timings

    def teacher_forced(self, inputs: PupInputs, target_ids: Sequence[int]) -> list[tuple[int, int, float]]:
        """The divergence audit (design 9.3): prefill, then feed ``target_ids`` one by one; per position
        ``(device argmax, target, gap)`` with ``gap = v[argmax] - v[target]`` on the presence-penalised logits of the
        TARGET prefix (so a tie shows as a gap near 0). The greedy continuation is never taken."""
        targets = [int(t) for t in target_ids]
        timings = PupTimings(
            load_s=self.load_s, prompt_tokens=inputs.prompt_tokens, image_tokens=inputs.num_image_tokens
        )
        vision_tokens = self._vision_tokens(inputs, timings)
        logits = self._prefill(inputs, vision_tokens, timings)
        out: list[tuple[int, int, float]] = []
        position = inputs.prompt_tokens
        for i, target in enumerate(targets):
            v = logits.clone()
            prefix = targets[:i]
            if inputs.presence_penalty > 0 and prefix:
                v[torch.tensor(sorted(set(prefix)), dtype=torch.long)] -= inputs.presence_penalty
            top = int(torch.argmax(v).item())
            out.append((top, target, float(v[top] - v[target])))
            if i + 1 < len(targets):
                if i == 0:
                    self._start_decode(target, position)
                logits = self._decode_step(target, position)
                position += 1
        if vision_tokens is not None:
            import ttnn

            ttnn.deallocate(vision_tokens)
        return out

    def __call__(self, prompt: str, images: Sequence[Image.Image] = (), boxes: Sequence[dict] | None = None) -> str:
        inputs = self.inputs(prompt, images, boxes)
        ids, timings = self.generate_ids(inputs)
        caption = finish_caption(ids, inputs.locked, tokenizer=self.tokenizer)
        self.last_inputs, self.last_ids, self.last_timings = inputs, ids, timings
        logger.info(
            f"RECORD pup[{inputs.system_key}] prompt T={inputs.prompt_tokens} image_tokens={inputs.num_image_tokens} "
            f"new={timings.generated_tokens}: load {timings.load_s:.1f}s vision {timings.vision_s:.2f}s prefill "
            f"{timings.prefill_s:.2f}s ttft {timings.ttft_s:.2f}s decode {timings.decode_tok_s:.1f} tok/s "
            f"({timings.decode_s:.1f}s) total {timings.total_s:.1f}s finish {timings.finish} (trace captured "
            f"{timings.trace_captured}, vision compiled {timings.vision_compiled}) {timings.dram}"
        )
        return caption

    def close(self) -> None:
        """Traces, KV cache, the vision tower's and the text model's device tensors are dropped; the mesh is closed
        when this object opened it."""
        import ttnn

        try:
            if self._generator is not None:
                self._generator.release_persistent_capture()
                self._generator = None
            self.model.free_kv_caches()
            self._prefill_trace_captured = False
            ttnn.synchronize_device(self._device)
        finally:
            if self._owns_device:
                ttnn.close_mesh_device(self._device)
                self._owns_device = False


# --- the demo's sequential model -----------------------------------------------------------------------------------------------


def _tree_root() -> Path:
    """The tt-metal root (``models/tt_dit/pipelines/flux3_klein/upsampler.py`` -> four levels up)."""
    return Path(__file__).resolve().parents[4]


def stage_reference(image: Image.Image, path: Path) -> Path:
    """A PIL reference handed to the CLI child as a lossless PNG. The child opens the file and runs the reference's
    ``policy_reference`` on it, so the steps that depend on what only the caller's object carries are applied here
    first: the EXIF orientation (``ImageOps.exif_transpose``, which a PNG save would drop; the child's own
    ``exif_transpose`` is then a no-op on the untagged copy) and the alpha composite over white, then
    ``convert("RGB")``. ``policy_reference(Image.open(path)) == policy_reference(image)`` pixel for pixel (host row
    U0g)."""
    from PIL import ImageOps

    staged = ImageOps.exif_transpose(image)
    if staged.mode in ("RGBA", "LA", "PA") or (staged.mode == "P" and "transparency" in staged.info):
        rgba = staged.convert("RGBA")
        staged = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    staged.convert("RGB").save(path, format="PNG")
    return path


class Flux3KleinUpsamplerSubprocess:
    """``__call__`` runs this module's CLI in a child process (its own 1-chip mesh, closed on exit) and returns the
    caption; a ``ValueError`` in the child (exit ``SUBPROCESS_PASS_THROUGH_EXIT``) is re-raised as ``ValueError`` so
    the pipeline falls back to the prompt as typed, anything else as ``RuntimeError``. The demo's model: the ring
    cannot share a process with the pup's fabric-less mesh. Not a pipeline default.

    ``scratch_dir``: where the per-call directory (staged references, ``boxes.json``, ``caption.json``) is made;
    ``None`` = ``$FLUX3_KLEIN_OUT`` when set, else the ``tempfile`` default (``$TMPDIR``). ``timeout_s``: ``None``
    (the default) waits for the child however long it takes; a value raises ``RuntimeError`` once it has passed and
    LEAVES THE CHILD RUNNING (it is never killed: a killed traced device run wedges the board), with its pid in the
    message."""

    def __init__(
        self,
        *,
        snapshot: Path | str | None = None,
        python: str = sys.executable,
        timeout_s: float | None = None,
        use_trace: bool = True,
        extra_args: Sequence[str] = (),
        scratch_dir: Path | str | None = None,
    ) -> None:
        self.snapshot = Path(snapshot) if snapshot is not None else None
        self.python = python
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.use_trace = bool(use_trace)
        self.extra_args = tuple(extra_args)
        scratch = scratch_dir if scratch_dir is not None else os.environ.get("FLUX3_KLEIN_OUT")
        self.scratch_dir = Path(scratch) if scratch else None
        self.last_record: dict | None = None

    def __call__(self, prompt: str, images: Sequence = (), boxes: Sequence[dict] | None = None) -> str:
        import shutil

        if self.scratch_dir is not None:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="flux3_klein_pup_", dir=self.scratch_dir))
        keep = False
        try:
            command = [self.python, "-m", "models.tt_dit.pipelines.flux3_klein.upsampler", "--prompt", prompt]
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    path = stage_reference(image, tmp_dir / f"reference_{i}.png")
                else:
                    path = Path(image)
                command += ["--image", str(path)]
            if boxes:
                boxes_path = tmp_dir / "boxes.json"
                boxes_path.write_text(json.dumps(list(boxes)))
                command += ["--boxes", str(boxes_path)]
            out_path = tmp_dir / "caption.json"
            command += ["--out", str(out_path)]
            if self.snapshot is not None:
                command += ["--weights", str(self.snapshot)]
            if not self.use_trace:
                command.append("--no-trace")
            command += list(self.extra_args)
            logger.info(f"pup subprocess: {' '.join(command[:6])} ... (scratch {tmp_dir})")
            child = subprocess.Popen(command, cwd=str(_tree_root()))
            try:
                returncode = child.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                keep = True  # the child still writes there
                raise RuntimeError(
                    f"pup subprocess (pid {child.pid}) still running after {self.timeout_s:.0f}s; NOT killed (a "
                    f"killed traced device run wedges the board): wait for it or end its Slurm step; scratch {tmp_dir}"
                ) from None
            if returncode == SUBPROCESS_PASS_THROUGH_EXIT:
                reason = json.loads(out_path.read_text()).get("error") if out_path.is_file() else "see the child log"
                raise ValueError(f"upsampler refused the request: {reason}")
            if returncode != 0 or not out_path.is_file():
                raise RuntimeError(f"pup subprocess failed with exit code {returncode}")
            record = json.loads(out_path.read_text())
            self.last_record = record
            return record["caption"]
        finally:
            if not keep:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# --- CLI ------------------------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX 3 Image Klein prompt upsampler on one Blackhole chip (no fabric).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="the raw prompt (or an upsampler input with boxes)")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help=f"reference image; repeat for several (max {MAX_REFERENCES})",
    )
    parser.add_argument("--boxes", type=Path, help="JSON list of 0-1 boxes ([y_min, x_min, y_max, x_max], id, desc)")
    parser.add_argument(
        "--out", type=Path, help="write {caption, input_prompt, boxes, ids, timings, task, system_key} here"
    )
    parser.add_argument("--weights", type=Path, default=None, help="the pup snapshot (default: the HuggingFace cache)")
    parser.add_argument("--no-trace", action="store_true", help="eager prefill and decode (the paged path)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=FLUX3_KLEIN_PUP_MAX_NEW_TOKENS, help="cap on the generated tokens"
    )
    parser.add_argument(
        "--teacher-force",
        type=Path,
        default=None,
        help="an examples/<set>/<case>.json record: audit the device argmax against the record's caption instead "
        "(text-to-image / edit records only: a layout record's caption is reconcile output, exit 2)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    boxes = json.loads(args.boxes.read_text()) if args.boxes else None
    images = [Image.open(p) for p in args.image]
    snapshot = args.weights if args.weights is not None else resolve_weights_dir("pup")
    record: dict = {"input_prompt": args.prompt, "boxes": boxes, "references": [str(p) for p in args.image]}
    try:
        # The host preparation first: a refused request never opens the device.
        tokenizer_prompts = load_system_prompts(snapshot)
        from transformers import AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        inputs = build_pup_inputs(
            args.prompt, images, boxes, prompts=tokenizer_prompts, tokenizer=tokenizer, processor=processor
        )
    except ValueError as error:
        record["error"] = str(error)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(f"upsampling skipped: {error}", file=sys.stderr, flush=True)
        return SUBPROCESS_PASS_THROUGH_EXIT
    record.update(task=inputs.task, system_key=inputs.system_key, prompt_tokens=inputs.prompt_tokens)
    target_ids = None
    if args.teacher_force is not None:
        try:
            target = json.loads(args.teacher_force.read_text())["prompt"]
            target_ids = audit_target_ids(target, tokenizer, locked=inputs.locked)
        except ValueError as error:
            print(f"no audit: {error}", file=sys.stderr, flush=True)
            return 2
    pup = None
    try:
        # Built inside the try: __init__ closes its own mesh on a failed load, close() covers everything after.
        pup = Flux3KleinUpsampler(snapshot, use_trace=not args.no_trace)
        if target_ids is not None:
            audit = pup.teacher_forced(inputs, target_ids)
            agree = sum(1 for top, tgt, _ in audit if top == tgt)
            record.update(audit=audit, target_ids=target_ids, agreement=agree / max(1, len(audit)))
            print(f"teacher-forced agreement {agree}/{len(audit)}", flush=True)
        else:
            ids, timings = pup.generate_ids(inputs, max_new_tokens=args.max_new_tokens)
            caption = finish_caption(ids, inputs.locked, tokenizer=tokenizer)
            record.update(caption=caption, ids=ids, timings=asdict(timings))
            logger.info(
                f"RECORD pup[{inputs.system_key}] prompt T={inputs.prompt_tokens} image_tokens={inputs.num_image_tokens} "
                f"new={timings.generated_tokens}: load {timings.load_s:.1f}s ttft {timings.ttft_s:.2f}s decode "
                f"{timings.decode_tok_s:.1f} tok/s total {timings.total_s:.1f}s finish {timings.finish} (trace "
                f"captured {timings.trace_captured}) {timings.dram}"
            )
            print(f"\n{caption}", flush=True)
    finally:
        if pup is not None:
            pup.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return 0


__all__ = [
    "FLUX3_KLEIN_PUP_MAX_NEW_TOKENS",
    "FLUX3_KLEIN_PUP_STOP_IDS",
    "I2I_SCAFFOLD",
    "MAX_REFERENCES",
    "PRESENCE_PENALTY",
    "PUP_DEVICE_PARAMS",
    "PUP_MAX_PROMPT",
    "PUP_MAX_SEQ_LEN",
    "PUP_SYSTEM_KEYS",
    "SAMPLING",
    "STOP_TOKENS",
    "SUBPROCESS_PASS_THROUGH_EXIT",
    "Flux3KleinUpsampler",
    "Flux3KleinUpsamplerSubprocess",
    "PupInputs",
    "PupTimings",
    "AUDIT_BOUNDARY",
    "audit_target_ids",
    "blocks_for",
    "divergence_audit",
    "build_pup_inputs",
    "configure_pup_env",
    "finish_caption",
    "first_divergence",
    "load_pup_model",
    "load_system_prompts",
    "new_token_budget",
    "open_pup_mesh",
    "policy_reference",
    "presence_greedy",
    "reference_size",
    "split_images",
    "stage_reference",
    "token_f1",
    "vision_padded_rows",
]


if __name__ == "__main__":
    sys.exit(main())
