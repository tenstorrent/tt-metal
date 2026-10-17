# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""M5: the FLUX 3 Image Klein text encoder (Qwen3-VL-4B, 8 taps) on device vs the vendored torch ``TextEncoder``.

Oracle: ``reference/flux3_klein/text_encoder.py::TextEncoder.encode`` (``Qwen3VLForConditionalGeneration`` in bf16 on
CPU, transformers 5.12.1): ``(L, 20480)`` bf16 = per token ``[hidden_states[4] | hidden_states[8] | ... |
hidden_states[32]]`` with L a multiple of 80 (<= 4096) and ``ids = (0, 0, 0, pos)``. The three golden ``text_cache``
tensors under ``$FLUX3_KLEIN_GOLDENS/text_cache`` are that class's output for ``''`` (80 rows, the CFG unconditional)
and the base / distilled man_dog prompts (320 rows each) and are the exact DiT inputs of the 1024 goldens; synthetic
prompts are compared to a live run of the same class through the oracle cache below. PAD ROWS ARE PART OF THE OUTPUT:
the DiT attends to all L rows, so rows ``[R, L)`` (R = real tokens) are compared too, separately -- a mask bug shows
there first (HF right padding: every query row is computed, keys >= R are excluded; the two plausible wrong masks move
the pad rows by max |d| 34-43 on rows of norm ~350 while the real rows barely move).

Host rows (login box, no device; the checkpoint via ``common.checkpoint_dir``):
  H1 test_tokenize_matches_reference[prompt]  render byte-equal to ``apply_chat_template``, ``tokenize`` torch.equal
                          to the reference recipe, L == padded_text_length(real), positions rule, ids == text_ids(L),
                          frozen ids for ``''`` and the pinned (real, L) of every prompt.
  H2 test_rope_tables_match_hf[prompt]  ``flux3_klein_rope_tables`` torch.equal (bf16) to HF
                          ``Qwen3VLTextRotaryEmbedding`` for every prompt incl. L 4096; ``create_rope_tensors`` is NOT
                          equal at L >= 320 (why the builder exists).
  H3 test_config_taps_and_keys  config values, taps == OUTPUT_LAYERS - 1, 354 kept / 44 dropped layers / 315 dropped
                          vision keys, lm_head absent, 3,618,744,832 kept elements (shard headers), the 11-name per-layer
                          map and the 290 module parameters after q/k/v fusion (encoder built on a stub device).
  H4 test_loader_negatives  unrouted index keys and wrong configs are rejected (``expect_error``). The module-level
                          strict-load negatives (extra ``layers.32`` key, missing ``norm.weight``, a ``visual`` key) need
                          parameters on a mesh and run in D1.
  H5 test_reference_hidden_state_index (slow)  on the real model: ``hidden_states[k]`` == forward-hook output of layer
                          k-1 for k in OUTPUT_LAYERS, the 2D right-padding mask == the explicit 4D ``causal AND key < R``
                          mask, and the vendored ``TextEncoder('')`` torch.equal the golden text_cache tensor (the
                          oracle IS the golden path).

Device rows (parent (4,8) ``ring_params_8k_flux3_klein`` from the root fixture, ``create_submesh`` here; ONE submesh per
parent open, never a second one in the same process -- ``docs/M5_DESIGN.md`` section 4):
  D1 test_fast_random[row]  depth-2 random HF ``Qwen3VLTextModel`` (head_dim 128 passed explicitly: 2560 / 32 = 80 would
                          silently build 80-wide heads), taps (0, 1), ``''`` (L 80) and the DISTILLED man_dog prompt (golden
                          50c3439c, L 320): per-tap pcc >= 0.98 over all rows AND over the pad rows, tap identity (each device
                          tap best matches ITS reference tap), second forward torch.equal, device ``ttnn.concat`` ==
                          ``torch.cat`` of the taps; strict-load negatives first.
  D2 test_real_weights[row]  the checkpoint through ``load_flux3_klein_text_encoder_weights`` (cache MISS / HIT wall,
                          DRAM per bank logged); the three golden prompts vs ``text_cache``, the synthetic prompts
                          (``cat``, exact80) and the NATURAL long prompts (the snapshot's own text_encoder/README.md L 2000,
                          README.md L 3600, LICENSE.md truncated to R = L = 4096) vs the cached CPU oracle: shape / dtype / finite,
                          per-tap pcc >= TAP_PCC over all rows (pad-row per-tap pcc reported), ``assert_quality(pcc >=
                          CONCAT_PCC, relative_rmse <= CONCAT_RMSE)`` over ALL rows, the pad block ``[R, L)`` reported
                          (pcc, rel_rmse, per-tap) and asserted at the pcc floor PAD_BLOCK_PCC only, the device pad-row
                          spread, tap identity (the 8 x 8 cross-tap pcc matrix: adjacent middle taps are near-collinear,
                          pcc 0.9991 at L 320, so a swapped middle pair passes both numeric gates; the best-match check does
                          not), ids == text_ids(L); encode walls first vs warm per L (the D4 timing rows). The repetitive
                          synthetic prompts (``long`` 1760, ``trunc`` 4096) are SENSITIVITY rows: HF itself flips a
                          massive-activation row on them between bf16/sdpa and bf16/eager (see SENSITIVITY_PROMPTS), so
                          only their pre-tipping taps 0-3, pad block, finiteness and determinism are asserted; the rest is
                          logged with the sink rows of oracle and device.
  D3 test_dit_handoff[row]  ``output_sp_axis=0``: the returned tensor read back with ``mesh_axes=[None, 0, None]``
                          equals ``packing.pad_rows(host output)`` bit for bit (zero pad rows, real rows unchanged).

Imports: ``ttnn`` and ``.device_params`` are imported at module level as in the M3 test (``test_transformer_flux3_klein.py``;
ttnn imports without a device on the login box, the host rows run there), a documented deviation from docs/M5_DESIGN.md 6.

ORACLE CACHE: ``FLUX3_KLEIN_TEXT_ORACLE_CACHE`` names a directory (unset -> recompute every time; the rows pass from an
empty directory). Key = sha256 of ``"{blob1}|{blob2}|{transformers}|{torch}|{sha256(prompt)}|{L}|{OUTPUT_LAYERS}|
{sha256(text_encoder.py bytes)}"`` (``blob`` = the HuggingFace blob names of the two text_encoder shards); files
``<digest>.safetensors`` (``tokens`` (L, 20480) bf16, ``ids`` (L, 4)) + ``<digest>.json`` with the fields. A miss logs
``ORACLE MISS <digest>``, computes with the vendored class on CPU (``torch.set_num_threads(16)``; 4-21 s at L 80, 13 s
at 320, 83 s at 1760, 154 s at 4096 on the login box) and writes; a hit logs ``ORACLE HIT <digest>``. Never run two
4096-token oracles at once. Golden prompts never go through the cache (their oracle is the text_cache tensor).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.qwen3vl.loader_flux3_klein import (
    FLUX3_KLEIN_ACTIVATION_LAYERS,
    FLUX3_KLEIN_TEXT_ENCODER_LAYERS,
    FLUX3_KLEIN_TEXT_HEAD_DIM,
    build_flux3_klein_text_encoder,
    check_flux3_klein_text_config,
    flux3_klein_rope_theta,
    flux3_klein_text_config,
    flux3_klein_text_encoder_blob,
    flux3_klein_text_state_keys,
    load_flux3_klein_text_encoder_weights,
    route_flux3_klein_text_keys,
)
from ....encoders.qwen3vl.model_qwen3vl import create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein import packing
from ....pipelines.flux3_klein.prompt_encoder import (
    Flux3KleinPromptEncoder,
    Flux3KleinTextEncoding,
    flux3_klein_rope_tables,
)
from ....reference.flux3_klein import text_encoder as reference_text_encoder
from ....reference.flux3_klein.text_encoder import MAX_LENGTH, OUTPUT_LAYERS, PAD_MULTIPLE
from ....utils import cache, tensor
from ....utils.check import assert_quality
from .common import REFERENCE_DIR, checkpoint_dir, goldens_root
from .common import pcc as pcc_of
from .device_params import ring_params_8k_flux3_klein

TEXT_ORACLE_CACHE_ENV = "FLUX3_KLEIN_TEXT_ORACLE_CACHE"
ORACLE_THREADS = 16
REFERENCE_FILE = REFERENCE_DIR / "text_encoder.py"
HIDDEN = 2560
WIDTH = HIDDEN * len(OUTPUT_LAYERS)  # 20480

# --- gate (PLAN.md M5; never loosened) ------------------------------------------------------------------------------------
# Per 2560-wide tap over ALL L rows (H3's real-weight Qwen3-VL floor, test_qwen3vl.py:104).
TAP_PCC = 0.99
# Per 20480-wide concat over ALL L rows (flux2's prompt-encoder gate).
CONCAT_PCC = 0.9997
CONCAT_RMSE = 0.04
# The pad block [R, L) alone: rows of norm ~350 (vs ~4162 for real rows) that are near-identical to each other, so pcc and
# rel_rmse over the block are dominated by legitimate rounding noise and the concat gate does NOT apply to it. Host emulation
# of the device-form arithmetic inside the HF model (logs/verify5_e2_model.log; distilled golden 50c3439c, L 320, R 283):
# RMSNorm single rounding alone -> pad pcc 0.999559; down_proj TP8 K-split with bf16 partials alone -> 0.996935 / rel_rmse
# 0.0786; all device-form sites together -> 0.997736 / 0.0674, while the whole-row concat stays >= 0.999982 / 0.0085 and the
# real rows >= 0.999992. The plausible mask bugs (causal-only, pad-attends-itself, all-ones, arange pad positions) sit at pad
# pcc 0.33-0.76 / rel_rmse >= 0.70, two orders of magnitude away. So the pad block is asserted at a bug-discriminating pcc
# floor ONLY (no rel_rmse), set from the first measured device run rounded DOWN and never raised above a measured value.
# Measured 2026-10-17 on b03u02 (logs/m5_main_dev2_bh_glx_1x4sub_linear.log, logs/m5_main_dev2_bh_4x8_ring.log): pad block pcc
# >= 0.9997 at L 80 / 320 (both meshes), 0.999167 / 0.999528 at L 2000 (32 pad rows), 0.997347 (TP4) / 0.998405 (TP8 FSDP) at
# L 3600 (43 pad rows); the floor below is the lowest measured value rounded down.
PAD_BLOCK_PCC = 0.997
# Random weights at depth 2 (test_qwen3vl.py:104; measured depth-2 floor on the (1,4) row 0.99980 all rows / 0.99977 pad).
RANDOM_TAP_PCC = 0.98
# SENSITIVITY rows (measured 2026-10-17, b03u02, both meshes; logs/m5_probe_long_1x4.log, logs/m5_probe_cpu_variants*.log):
# the repetitive synthetic prompts sit on a bf16 TIPPING POINT of the model itself, not on a device defect. 'long' (130 x
# "... market with {i} stalls,"): the lone-space token at row 95 grows a massive activation (|h| 6336 at layer 17-19, ch 4,
# the model's attention-sink channel) on device, and HF ITSELF does the same in bf16 with attn_implementation='eager'
# (6752; concat vs the bf16/sdpa oracle 0.9203, taps 4-7 at 0.87-0.91, 520 real rows > 10 % off) while bf16/sdpa and fp32
# HF do not (fp32 vs oracle 0.99883). The device matches the bf16/eager HF signature (0.9116 / 0.8668, taps 0-3 >= 0.99993,
# pad rows 0.9997). The same prompt at half (L 880) and double (L 3600) length fails identically; PLAN.md text cut to the
# same R 1718 / L 1760 passes at 0.99990 / rel_rmse 0.0145. Such a prompt cannot gate a port: two faithful bf16
# implementations disagree on it. The sensitivity rows therefore assert only what is implementation-independent: the taps
# BEFORE the tipping layer (taps 0-3 = layers 3-15) at SENSITIVITY_EARLY_TAP_PCC, the pad block, finiteness, determinism
# and the tap order of taps 0-3; the full numbers are logged (RESULT) and the sink rows (rows whose |tap 4| exceeds
# SINK_ABS) of oracle and device are printed so a flipped tipping point is visible as a changed sink set. The natural long
# prompts (readme_te / readme / license, from the snapshot itself) are the gate rows at those lengths. 'trunc' (700 x
# "sentence number {i} about the weather,", R = L = 4096) has the same sink rows on both sides but drifts in the late taps:
# device 0.998705 (TP4) / 0.998558 (TP8) vs the oracle, inside the band of HF's own faithful variants on the same prompt
# (bf16/eager 0.998273, fp32 0.999054; logs/m5_probe_cpu_variants_trunc.log), so it is recorded, not gated.
SENSITIVITY_PROMPTS = ("long", "trunc")
SENSITIVITY_EARLY_TAP_PCC = 0.9999  # measured 0.99993-1.00000 on taps 0-3 for both repetitive prompts, both meshes
SINK_ABS = 1000.0  # the model's massive activations are ~6e3-1.1e4; ordinary rows stay < 200

# Frozen facts of transformers 5.12.1's rendering of the chat template (a template change moves both sides of H1
# together, so they are pinned here): '' renders to '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n'.
EMPTY_RENDER = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
EMPTY_IDS = [151644, 872, 198, 151645, 198, 151644, 77091, 198]
PAD_ID = 151643
# (real tokens, L) per prompt; the golden prompts are keyed by the text_cache file stem's first 8 characters.
EXPECTED_LENGTHS = {
    "empty": (8, 80),
    "one_word": (9, 80),
    "exact80": (80, 80),
    "golden_50c3439c": (283, 320),
    "golden_d1d3cb25": (278, 320),
    "golden_f65ac131": (8, 80),
    # Natural long prompts = the checkpoint snapshot's own markdown files (deterministic per snapshot, non-repetitive).
    "readme_te": (1968, 2000),  # text_encoder/README.md
    "readme": (3557, 3600),  # README.md
    "license": (4096, 4096),  # LICENSE.md, truncated: the R == L (no pad rows) path
    # Synthetic repetitive prompts: SENSITIVITY rows (see SENSITIVITY_PROMPTS), not gate rows.
    "long": (1718, 1760),
    "trunc": (4096, 4096),
}
PROMPT_NAMES = tuple(EXPECTED_LENGTHS)
SNAPSHOT_TEXT_PROMPTS = {"readme_te": "text_encoder/README.md", "readme": "README.md", "license": "LICENSE.md"}
# D1 / D3 run the short prompt and one 320-token prompt. Golden identity MEASURED from the 1024 goldens' DiT inputs
# (outputs/goldens_cpu_1024/{base,distilled}/man_dog/inputs.safetensors text_tokens): 50c3439c (283 real tokens) is the
# DISTILLED man_dog prompt, d1d3cb25 (278 real) the BASE one, f65ac131 the '' unconditional of both.
DISTILLED_MAN_DOG = "golden_50c3439c"
BASE_MAN_DOG = "golden_d1d3cb25"

# Checkpoint facts (judge, from the safetensors headers): keys and bf16 elements kept.
KEPT_KEYS, DROPPED_LAYER_KEYS, DROPPED_VISION_KEYS = 354, 44, 315
KEPT_ELEMENTS = 3_618_744_832
MODULE_PARAMETERS = 290  # 32 x 9 (q/k/v fused) + embed + norm
HF_LAYER_NAMES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
)
TT_LAYER_NAMES = (
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.qkv_proj.weight",
)


# --- snapshot, goldens, prompts ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot() -> Path:
    found = checkpoint_dir("distilled", required=("text_encoder/config.json", "text_encoder/tokenizer.json"))
    if found is None:
        pytest.skip("FLUX 3 Klein checkpoint with text_encoder/ not found (FLUX3_KLEIN_MODEL_PATH / HuggingFace cache)")
    return found


@pytest.fixture
def text_config(snapshot) -> dict:
    """Function-scoped on purpose: ``transformers.Qwen3VLTextConfig(**cfg)`` mutates the ``rope_scaling`` sub-dict in place."""
    return flux3_klein_text_config(snapshot)


def hf_text_config(text_config: dict, **overrides):
    """A transformers ``Qwen3VLTextConfig`` from a deep copy of ``text_config`` (the constructor mutates its input)."""
    import transformers

    return transformers.Qwen3VLTextConfig(**{**json.loads(json.dumps(text_config)), **overrides})


@pytest.fixture(scope="module")
def host_encoder(snapshot) -> Flux3KleinPromptEncoder:
    """The prompt encoder's host half (tokenizer, template, positions, rope tables); no device, no model."""
    return Flux3KleinPromptEncoder(None, snapshot, None, text_config=flux3_klein_text_config(snapshot))


def golden_text_cache() -> dict[str, tuple[str, Path]]:
    """``{golden_<stem[:8]>: (prompt, safetensors path)}`` of ``$FLUX3_KLEIN_GOLDENS/text_cache``."""
    root = goldens_root()
    if root is None or not (root / "text_cache").is_dir():
        return {}
    out = {}
    for sidecar in sorted((root / "text_cache").glob("*.json")):
        fields = json.loads(sidecar.read_text())
        out[f"golden_{sidecar.stem[:8]}"] = (fields["prompt"], sidecar.with_suffix(".safetensors"))
    return out


def exact80_prompt(host_encoder: Flux3KleinPromptEncoder) -> str:
    """A prompt whose rendering is exactly 80 tokens (no pad rows): single-token words until the count lands."""
    words: list[str] = []
    while True:
        _, mask, _ = host_encoder.tokenize(" ".join(words))
        real = int(mask.sum())
        if real >= PAD_MULTIPLE:
            assert real == PAD_MULTIPLE, f"could not land on exactly {PAD_MULTIPLE} tokens: {real}"
            return " ".join(words)
        words.append("hello")


@pytest.fixture(scope="module")
def prompts(host_encoder, snapshot) -> dict[str, str]:
    """The prompt set; golden prompts only when ``$FLUX3_KLEIN_GOLDENS/text_cache`` is present (rows skip otherwise)."""
    out = {
        "empty": "",
        "one_word": "cat",
        "exact80": exact80_prompt(host_encoder),
        **{name: (snapshot / rel).read_text() for name, rel in SNAPSHOT_TEXT_PROMPTS.items()},
        "long": " ".join(f"a detailed description of a bustling market with {i} stalls," for i in range(130)),
        "trunc": " ".join(f"sentence number {i} about the weather," for i in range(700)),
    }
    for name, (prompt, _) in golden_text_cache().items():
        out[name] = prompt
    return out


def prompt_or_skip(prompts: dict[str, str], name: str) -> str:
    if name not in prompts:
        pytest.skip(f"{name}: $FLUX3_KLEIN_GOLDENS/text_cache has no such golden prompt")
    return prompts[name]


def reference_tokenize(processor, prompt: str):
    """``text_encoder.py:30-38`` re-executed inline (the recipe the prompt encoder must reproduce)."""
    import math

    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    tokenizer = processor.tokenizer
    real = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
    length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
    mask = inputs["attention_mask"]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    return text, real, length, inputs["input_ids"], mask, positions


# --- H1: tokenization ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_tokenize_matches_reference(host_encoder, prompts, name):
    prompt = prompt_or_skip(prompts, name)
    processor = host_encoder._processor  # noqa: SLF001 -- the very object the reference recipe runs on
    assert processor.tokenizer.padding_side == "right"

    text, real, length, ref_ids, ref_mask, ref_positions = reference_tokenize(processor, prompt)
    assert host_encoder.render(prompt) == text
    input_ids, mask, L = host_encoder.tokenize(prompt)
    assert L == length == packing.padded_text_length(real)
    assert input_ids.dtype == torch.int64 and mask.dtype == torch.int64
    assert tuple(input_ids.shape) == tuple(mask.shape) == (1, L)
    assert torch.equal(input_ids, ref_ids) and torch.equal(mask, ref_mask)
    R = int(mask.sum())
    assert R == min(real, length)
    assert torch.equal(mask[0, :R], torch.ones(R, dtype=torch.long)) and not mask[0, R:].any(), "right padding"

    positions = host_encoder.positions(mask)
    assert torch.equal(positions, ref_positions)
    assert torch.equal(positions[0, :R], torch.arange(R)) and (positions[0, R:] == 1).all()
    assert torch.equal(packing.text_ids(L), reference_text_ids(L))

    expected_real, expected_length = EXPECTED_LENGTHS[name]
    assert (real, L) == (expected_real, expected_length), f"{name}: real {real} -> L {L}"
    if L > R:
        assert (input_ids[0, R:] == PAD_ID).all()
    if name == "empty":
        assert text == EMPTY_RENDER
        assert input_ids[0].tolist() == EMPTY_IDS + [PAD_ID] * (80 - len(EMPTY_IDS))
    if name == "trunc":
        assert R == L == MAX_LENGTH, "an over-long prompt truncates to real == L == 4096 (no pad rows)"
    logger.info(f"H1 {name}: real {real} -> L {L}, pad rows {L - R}, render {len(text)} chars")


def reference_text_ids(length: int) -> torch.Tensor:
    """``text_encoder.py:48-49``."""
    positions = torch.arange(length)
    return torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)


# --- H2: rope tables ---------------------------------------------------------------------------------------------------------


def hf_rope_tables(text_config: dict, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HF's own ``Qwen3VLTextRotaryEmbedding`` on the 3-row text positions, bf16 outputs ``(L, head_dim)``."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    rotary = Qwen3VLTextRotaryEmbedding(config=hf_text_config(text_config))
    x = torch.zeros(1, dtype=torch.bfloat16)
    cos, sin = rotary(x, positions.reshape(1, -1).expand(3, -1, -1))
    return cos[0], sin[0]


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_rope_tables_match_hf(host_encoder, text_config, prompts, name):
    prompt = prompt_or_skip(prompts, name)
    _, mask, L = host_encoder.tokenize(prompt)
    positions = host_encoder.positions(mask)
    cos, sin = host_encoder.rope_tables(positions)
    assert cos.shape == sin.shape == (1, 1, L, FLUX3_KLEIN_TEXT_HEAD_DIM) and cos.dtype == sin.dtype == torch.bfloat16

    hf_cos, hf_sin = hf_rope_tables(text_config, positions)
    assert hf_cos.dtype == torch.bfloat16
    assert torch.equal(cos[0, 0], hf_cos) and torch.equal(
        sin[0, 0], hf_sin
    ), f"{name}: rope tables differ from HF at L {L}"

    # The (L,) and (1, L) input forms agree, and the tables depend on the positions only.
    cos2, sin2 = flux3_klein_rope_tables(
        positions[0], head_dim=FLUX3_KLEIN_TEXT_HEAD_DIM, rope_theta=flux3_klein_rope_theta(text_config)
    )
    assert torch.equal(cos, cos2) and torch.equal(sin, sin2)

    # Why this builder exists: create_rope_tensors' `theta ** -x` form is bf16-different from HF at L >= 320.
    other_cos, other_sin = create_rope_tensors(
        1,
        L,
        mask,
        FLUX3_KLEIN_TEXT_HEAD_DIM,
        flux3_klein_rope_theta(text_config),
        text_config["rope_scaling"]["mrope_section"],
        interleaved=True,
    )
    mismatches = int((other_cos.to(torch.bfloat16) != cos).sum() + (other_sin.to(torch.bfloat16) != sin).sum())
    logger.info(f"H2 {name}: L {L} HF-equal; create_rope_tensors bf16 mismatches {mismatches}")
    if L >= 320:
        assert mismatches > 0, "create_rope_tensors became HF-exact; the dedicated builder would then be redundant"


# --- H3: config, taps, keys ------------------------------------------------------------------------------------------------


class _StubDevice:
    """Enough of a MeshDevice for module construction on the host: no tensors exist until a load."""

    shape = (1, 1)

    @staticmethod
    def arch():
        return ttnn.device.Arch.BLACKHOLE


def module_parameter_names(module, prefix: str = "") -> dict[str, tuple[int, ...]]:
    names = {prefix + n: tuple(p.total_shape) for n, p in module.named_parameters()}
    for n, child in module.named_children():
        names.update(module_parameter_names(child, f"{prefix}{n}."))
    return names


def test_config_taps_and_keys(snapshot, text_config):
    assert (text_config["hidden_size"], text_config["intermediate_size"], text_config["num_hidden_layers"]) == (
        HIDDEN,
        9728,
        36,
    )
    assert (text_config["num_attention_heads"], text_config["num_key_value_heads"], text_config["head_dim"]) == (
        32,
        8,
        128,
    )
    assert text_config["rms_norm_eps"] == 1e-6 and flux3_klein_rope_theta(text_config) == 5e6
    assert (
        text_config["rope_scaling"]["mrope_section"] == [24, 20, 20]
        and text_config["rope_scaling"]["mrope_interleaved"] is True
    )
    assert text_config["tie_word_embeddings"] is True and text_config["vocab_size"] == 151936
    assert text_config["hidden_act"] == "silu"

    assert OUTPUT_LAYERS == (4, 8, 12, 16, 20, 24, 28, 32)
    assert FLUX3_KLEIN_ACTIVATION_LAYERS == tuple(k - 1 for k in OUTPUT_LAYERS) == (3, 7, 11, 15, 19, 23, 27, 31)
    assert FLUX3_KLEIN_TEXT_ENCODER_LAYERS == 32 == FLUX3_KLEIN_ACTIVATION_LAYERS[-1] + 1
    assert reference_text_encoder.MAX_LENGTH == packing.FLUX3_KLEIN_MAX_TEXT_LENGTH == 4096
    assert reference_text_encoder.PAD_MULTIPLE == packing.FLUX3_KLEIN_TEXT_PAD_MULTIPLE == 80

    keys = flux3_klein_text_state_keys(snapshot)
    index = json.loads((snapshot / "text_encoder" / "model.safetensors.index.json").read_text())["weight_map"]
    assert (len(keys.kept), len(keys.dropped_layers), len(keys.dropped_vision)) == (
        KEPT_KEYS,
        DROPPED_LAYER_KEYS,
        DROPPED_VISION_KEYS,
    )
    assert len(index) == KEPT_KEYS + DROPPED_LAYER_KEYS + DROPPED_VISION_KEYS
    assert set(keys.kept) | set(keys.dropped_layers) | set(keys.dropped_vision) == set(index)
    assert not any("lm_head" in k for k in index), "lm_head.weight is tied (absent); a separate one would be unrouted"
    assert all(k.startswith("model.visual.") for k in keys.dropped_vision)
    assert {int(k.split(".")[3]) for k in keys.dropped_layers} == {32, 33, 34, 35}
    assert {int(k.split(".")[3]) for k in keys.kept if ".layers." in k} == set(range(32))
    per_layer = sorted({k.split(".", 4)[4] for k in keys.kept if ".layers." in k})
    assert tuple(per_layer) == HF_LAYER_NAMES
    assert "model.language_model.embed_tokens.weight" in keys.kept and "model.language_model.norm.weight" in keys.kept
    shards = {index[k] for k in keys.kept}
    assert len(shards) == 2, "norm.weight and the upper layers live in shard 00002: both shards are read"

    # Element count from the shard headers (no tensor loads).
    from math import prod

    from safetensors import safe_open

    elements = 0
    for shard in sorted(shards):
        with safe_open(str(snapshot / "text_encoder" / shard), framework="pt") as handle:
            for k in keys.kept:
                if index[k] == shard:
                    elements += prod(handle.get_slice(k).get_shape())
    assert elements == KEPT_ELEMENTS, elements
    logger.info(f"H3: {len(keys.kept)} kept keys, {elements:,} bf16 elements = {elements * 2 / 1e9:.4f} GB")

    # The module side: 290 parameters after q/k/v fusion, same element count at TP 1 (no head padding).
    encoder, cfg = build_flux3_klein_text_encoder(
        snapshot, mesh_device=_StubDevice(), parallel_config=None, ccl_manager=None, is_fsdp=False, load_weights=False
    )
    assert cfg == text_config
    names = module_parameter_names(encoder)
    assert len(names) == MODULE_PARAMETERS
    assert tuple(sorted({n.split(".", 2)[2] for n in names if n.startswith("layers.")})) == TT_LAYER_NAMES
    assert {"embed_tokens.weight", "norm.weight"} <= set(names)
    assert sum(prod(s) for s in names.values()) == KEPT_ELEMENTS
    assert names["layers.0.self_attn.qkv_proj.weight"] == (HIDDEN, (32 + 8 + 8) * 128)
    assert encoder._activation_layers == FLUX3_KLEIN_ACTIVATION_LAYERS and len(encoder.layers) == 32  # noqa: SLF001
    assert (
        len(flux3_klein_text_encoder_blob(snapshot)) >= 16
    ), "the cache key takes the first 16 characters of the blob name"


# --- H4: loader negatives -----------------------------------------------------------------------------------------------------


def test_loader_negatives(snapshot, text_config, expect_error):
    index = json.loads((snapshot / "text_encoder" / "model.safetensors.index.json").read_text())["weight_map"]
    routed = route_flux3_klein_text_keys(index)
    assert len(routed.kept) == KEPT_KEYS

    with expect_error(ValueError, "lm_head.weight"):
        route_flux3_klein_text_keys([*index, "lm_head.weight"])
    with expect_error(ValueError, "model.language_model.rotary_emb.inv_freq"):
        route_flux3_klein_text_keys([*index, "model.language_model.rotary_emb.inv_freq"])
    with expect_error(ValueError, "model.foo.weight"):
        route_flux3_klein_text_keys(["model.foo.weight"])
    # Layer-index cross-check against the config depth: nothing beyond layers 32-35 may hide in the dropped group.
    layers = text_config["num_hidden_layers"]
    assert route_flux3_klein_text_keys(index, num_hidden_layers=layers).dropped_layers == routed.dropped_layers
    with expect_error(ValueError, "dropped layer indices"):
        route_flux3_klein_text_keys(
            [*index, "model.language_model.layers.36.input_layernorm.weight"], num_hidden_layers=layers
        )
    with expect_error(ValueError, "dropped layer indices"):
        route_flux3_klein_text_keys(
            [k for k in index if not k.startswith("model.language_model.layers.35.")], num_hidden_layers=layers
        )
    with expect_error(ValueError, "kept layer indices"):
        route_flux3_klein_text_keys(
            [k for k in index if not k.startswith("model.language_model.layers.7.")], num_hidden_layers=layers
        )

    def with_(**changes):
        cfg = json.loads(json.dumps(text_config))
        for k, v in changes.items():
            if k == "mrope_section":
                cfg["rope_scaling"]["mrope_section"] = v
            else:
                cfg[k] = v
        return cfg

    assert check_flux3_klein_text_config(text_config) is text_config
    with expect_error(ValueError, "head_dim 80"):
        check_flux3_klein_text_config(with_(head_dim=80))
    with expect_error(ValueError, "num_hidden_layers 31"):
        check_flux3_klein_text_config(with_(num_hidden_layers=31))
    with expect_error(ValueError, "mrope_section"):
        check_flux3_klein_text_config(with_(mrope_section=[16, 24, 24]))
    with expect_error(ValueError, "tie_word_embeddings"):
        check_flux3_klein_text_config(with_(tie_word_embeddings=False))
    no_theta = with_()
    del no_theta["rope_theta"]
    with expect_error(ValueError, "rope_theta"):
        check_flux3_klein_text_config(no_theta)
    with expect_error(ValueError, "outside the built layers"):
        build_flux3_klein_text_encoder(
            snapshot,
            mesh_device=_StubDevice(),
            parallel_config=None,
            ccl_manager=None,
            load_weights=False,
            num_layers=2,
            activation_layers=(0, 2),
        )
    with expect_error(ValueError, "full 32-layer encoder only"):
        build_flux3_klein_text_encoder(
            snapshot,
            mesh_device=_StubDevice(),
            parallel_config=None,
            ccl_manager=None,
            load_weights=True,
            num_layers=2,
            activation_layers=(0, 1),
        )


# --- oracle cache ------------------------------------------------------------------------------------------------------------


def text_oracle_key(snapshot: Path, prompt: str, length: int) -> tuple[str, dict]:
    import transformers

    fields = {
        "blob1": flux3_klein_text_encoder_blob(snapshot),
        "blob2": loader_blob(snapshot, "model-00002-of-00002.safetensors"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "length": length,
        "output_layers": list(OUTPUT_LAYERS),
        "reference_sha256": hashlib.sha256(REFERENCE_FILE.read_bytes()).hexdigest(),
    }
    key = "|".join(str(fields[k]) for k in fields)
    return hashlib.sha256(key.encode()).hexdigest(), fields


def loader_blob(snapshot: Path, name: str) -> str:
    path = snapshot / "text_encoder" / name
    return Path(os.readlink(path)).name if path.is_symlink() else f"{path.name}:{path.stat().st_size}"


_REFERENCE_ENCODERS: dict[Path, object] = {}


def reference_encoder(snapshot: Path):
    """One vendored ``TextEncoder`` (9 GB bf16 on CPU) per session, built lazily on the first cache miss."""
    if snapshot not in _REFERENCE_ENCODERS:
        torch.set_num_threads(ORACLE_THREADS)
        t0 = time.time()
        _REFERENCE_ENCODERS[snapshot] = reference_text_encoder.TextEncoder(str(snapshot / "text_encoder"), device="cpu")
        logger.info(f"vendored TextEncoder loaded on CPU in {time.time() - t0:.1f}s ({ORACLE_THREADS} threads)")
    return _REFERENCE_ENCODERS[snapshot]


def cached_text_oracle(snapshot: Path, prompt: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(tokens (L, 20480) bf16, ids (L, 4))`` of the vendored encoder through the oracle cache (module docstring)."""
    from safetensors.torch import load_file, save_file

    digest, fields = text_oracle_key(snapshot, prompt, length)
    root = os.environ.get(TEXT_ORACLE_CACHE_ENV)
    root = Path(root) if root else None
    if root is not None and (root / f"{digest}.safetensors").is_file():
        logger.info(f"ORACLE HIT {digest} (L {length}, prompt sha {fields['prompt_sha256'][:12]})")
        loaded = load_file(str(root / f"{digest}.safetensors"))
        return loaded["tokens"], loaded["ids"]
    logger.info(f"ORACLE MISS {digest} (L {length}, prompt sha {fields['prompt_sha256'][:12]})")
    t0 = time.time()
    tokens, ids = reference_encoder(snapshot).encode(prompt)
    tokens, ids = tokens.contiguous(), ids.contiguous()
    logger.info(f"oracle computed in {time.time() - t0:.1f}s: {tuple(tokens.shape)} {tokens.dtype}")
    assert tuple(tokens.shape) == (length, WIDTH) and tokens.dtype == torch.bfloat16
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        save_file({"tokens": tokens, "ids": ids}, str(root / f"{digest}.safetensors"))
        (root / f"{digest}.json").write_text(json.dumps(fields, indent=1))
        logger.info(f"oracle written to {root / digest}.safetensors")
    return tokens, ids


def expected_tokens(snapshot: Path, name: str, prompt: str, length: int) -> tuple[torch.Tensor, str]:
    """The oracle of ``prompt``: the golden text_cache tensor for a golden prompt, else the cached CPU oracle."""
    goldens = golden_text_cache()
    if name in goldens:
        from safetensors.torch import load_file

        golden_prompt, path = goldens[name]
        assert golden_prompt == prompt
        tokens = load_file(str(path))["tokens"]
        return tokens, f"golden text_cache {path.stem[:8]}"
    tokens, ids = cached_text_oracle(snapshot, prompt, length)
    assert torch.equal(ids, packing.text_ids(length))
    return tokens, "CPU oracle"


# --- H5: the reference's own facts (slow) ------------------------------------------------------------------------------------


@pytest.mark.slow
def test_reference_hidden_state_index(snapshot, host_encoder, prompts):
    """On the real model: hidden_states[k] is the output of layer k-1; the 2D right-padding mask equals the explicit 4D
    ``causal AND key < R`` mask (pad queries computed, masked keys excluded); the vendored encoder reproduces the golden.
    """
    from ...encoders.qwen3vl.common import capture_layer_outputs

    encoder = reference_encoder(snapshot)
    model, lm = encoder.model, encoder.model.model.language_model

    # H1 again, against the reference CLASS's own processor object (H1 uses the prompt encoder's): render byte-equal,
    # tokenize torch.equal, positions equal for every prompt of the set.
    assert encoder.processor.tokenizer.padding_side == "right"
    for name in sorted(prompts):
        text, _, length, ref_ids, ref_mask, ref_positions = reference_tokenize(encoder.processor, prompts[name])
        own_ids, own_mask, own_length = host_encoder.tokenize(prompts[name])
        assert (
            host_encoder.render(prompts[name]) == text
        ), f"{name}: render differs from the reference class's processor"
        assert own_length == length and torch.equal(own_ids, ref_ids) and torch.equal(own_mask, ref_mask), name
        assert torch.equal(host_encoder.positions(own_mask), ref_positions), name
    logger.info(
        f"H5: render / tokenize / positions equal to the reference class's processor for {len(prompts)} prompts"
    )

    prompt = prompts["empty"]
    input_ids, mask, L = host_encoder.tokenize(prompt)
    R = int(mask.sum())
    positions = host_encoder.positions(mask)

    taps = tuple(k - 1 for k in OUTPUT_LAYERS)
    with torch.inference_mode(), capture_layer_outputs(lm, taps) as captured:
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
    assert len(out.hidden_states) == 37
    assert torch.equal(out.hidden_states[0], lm.embed_tokens(input_ids))
    for k in OUTPUT_LAYERS:
        assert torch.equal(
            out.hidden_states[k], captured[k - 1]
        ), f"hidden_states[{k}] is not the output of layer {k - 1}"

    # HF right-padding semantics, spelled out as a 4D boolean mask: query i attends key j iff j <= i and j < R.
    i = torch.arange(L)[:, None]
    j = torch.arange(L)[None, :]
    explicit = ((j <= i) & (j < R))[None, None]
    with torch.inference_mode():
        out4 = model(
            input_ids=input_ids,
            attention_mask=explicit,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
    for k in OUTPUT_LAYERS:
        assert torch.equal(out.hidden_states[k], out4.hidden_states[k]), f"2D mask != causal AND key < R at tap {k}"

    tokens, ids = encoder.encode(prompt)
    stacked = (
        torch.stack([out.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        .permute(0, 2, 1, 3)
        .reshape(L, -1)
        .to(torch.bfloat16)
    )
    assert torch.equal(tokens, stacked) and torch.equal(ids, packing.text_ids(L))
    pad = tokens[R:].float()
    logger.info(
        f"H5: '' pad rows identical: {bool((pad == pad[:1]).all())}; pad norm {pad[0].norm():.1f}, real mean norm {tokens[:R].float().norm(dim=1).mean():.1f}"
    )
    golden = golden_text_cache()
    if "golden_f65ac131" in golden:
        from safetensors.torch import load_file

        assert torch.equal(
            load_file(str(golden["golden_f65ac131"][1]))["tokens"], tokens
        ), "the vendored encoder is not the golden path"
    else:
        logger.warning("H5: no golden text_cache for '' under $FLUX3_KLEIN_GOLDENS; oracle == golden not checked")


# --- device: mesh rows ----------------------------------------------------------------------------------------------------------

mesh_rows = pytest.mark.parametrize(
    "mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp, device_params",
    [
        [(4, 8), (1, 4), 1, ttnn.Topology.Linear, 1, False, ring_params_8k_flux3_klein],
        [(4, 8), None, 1, ttnn.Topology.Ring, 2, True, ring_params_8k_flux3_klein],
    ],
    ids=["bh_glx_1x4sub_linear", "bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)


@dataclass
class Mesh:
    device: ttnn.MeshDevice
    tp_axis: int
    ccl: CCLManager
    parallel_config: EncoderParallelConfig
    is_fsdp: bool
    tag: str

    @property
    def tp(self) -> int:
        return self.parallel_config.tensor_parallel.factor


def open_mesh(mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp) -> Mesh:
    device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape)) if submesh_shape else mesh_device
    shape = tuple(device.shape)
    ccl = CCLManager(mesh_device=device, num_links=num_links, topology=topology)
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis))
    tag = f"{shape[0]}x{shape[1]}_tp{shape[tp_axis]}_{'ring' if topology == ttnn.Topology.Ring else 'linear'}{num_links}L{'_fsdp' if is_fsdp else ''}"
    logger.info(f"mesh {tag}: parent {tuple(mesh_device.shape)}")
    return Mesh(device, tp_axis, ccl, parallel_config, is_fsdp, tag)


def dram_per_bank(device) -> str:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip"
    )


def split_taps(rows: torch.Tensor) -> list[torch.Tensor]:
    """``(L, n * H)`` layer-major rows -> the n per-tap ``(L, H)`` blocks."""
    return [rows[:, k * HIDDEN : (k + 1) * HIDDEN] for k in range(rows.shape[1] // HIDDEN)]


def rel_rmse(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """``assert_quality``'s definition: rmse / std(reference)."""
    a, b = reference.double().flatten(), actual.double().flatten()
    return float(((a - b) ** 2).mean().sqrt() / a.std())


def compare_rows(name: str, reference: torch.Tensor, actual: torch.Tensor, R: int) -> dict[str, float]:
    """Log per-tap / concat / pad-block numbers; return them (the caller asserts)."""
    L = reference.shape[0]
    ref_f, act_f = reference.float(), actual.float()
    ref_taps, act_taps = split_taps(ref_f), split_taps(act_f)
    tap_pcc = [pcc_of(g, a) for g, a in zip(ref_taps, act_taps)]
    # Tap identity: device tap i must correlate best with reference tap i. The numeric gates cannot tell a swapped pair of
    # adjacent middle taps apart (cross-tap pcc 0.9991 at L 320 on the goldens), this n x n matrix can.
    cross = [[pcc_of(g, a) for g in ref_taps] for a in act_taps]
    best = [max(range(len(ref_taps)), key=lambda j: row[j]) for row in cross]
    margin = (
        min(row[i] - max(v for j, v in enumerate(row) if j != i) for i, row in enumerate(cross))
        if len(cross) > 1
        else 1.0
    )
    sink_tap = min(
        4, len(ref_taps) - 1
    )  # the massive-activation (attention-sink) rows, visible from tap 4 (layer 19) on
    sinks = [
        sorted(set((t.abs().max(dim=1).values > SINK_ABS).nonzero().flatten().tolist()))
        for t in (ref_taps[sink_tap], act_taps[sink_tap])
    ]
    out = {
        "pcc": pcc_of(ref_f, act_f),
        "rel_rmse": rel_rmse(ref_f, act_f),
        "max_abs": float((ref_f - act_f).abs().max()),
        "tap_pcc": tap_pcc,
        "tap_min": min(tap_pcc),
        "tap_identity": best == list(range(len(ref_taps))),
        "tap_best": best,
        "tap_margin": margin,
        "sinks_ref": sinks[0],
        "sinks_dev": sinks[1],
    }
    line = (
        f"RESULT {name} L={L} R={R}: concat pcc={out['pcc']:.6f} rel_rmse={out['rel_rmse']:.5f} max_abs={out['max_abs']:.3g}; "
        f"per-tap pcc {['%.5f' % p for p in tap_pcc]}; tap identity {out['tap_identity']} best-match {best} margin {margin:.5f}; "
        f"sink rows (|tap {sink_tap}| > {SINK_ABS:g}) ref {sinks[0][:8]} dev {sinks[1][:8]}"
    )
    if R < L:
        pad_tap = [pcc_of(g[R:], a[R:]) for g, a in zip(split_taps(ref_f), split_taps(act_f))]
        out.update(
            pad_pcc=pcc_of(ref_f[R:], act_f[R:]),
            pad_rel_rmse=rel_rmse(ref_f[R:], act_f[R:]),
            pad_tap_min=min(pad_tap),
            pad_spread=float((act_f[R:] - act_f[R : R + 1]).abs().max()),
            real_pcc=pcc_of(ref_f[:R], act_f[:R]),
            real_rel_rmse=rel_rmse(ref_f[:R], act_f[:R]),
        )
        line += (
            f"; PAD rows [{R}, {L}) pcc={out['pad_pcc']:.6f} rel_rmse={out['pad_rel_rmse']:.5f} per-tap {['%.5f' % p for p in pad_tap]}"
            f" device spread={out['pad_spread']:.3g}; REAL rows pcc={out['real_pcc']:.6f} rel_rmse={out['real_rel_rmse']:.5f}"
        )
    logger.info(line)
    return out


def device_taps(encoder, prompt_encoder: Flux3KleinPromptEncoder, device, prompt: str):
    """The device steps of ``encode`` with the taps kept: ``(taps on device, concat on device, (input_ids, mask, L))``."""
    input_ids, mask, L = prompt_encoder.tokenize(prompt)
    cos, sin = prompt_encoder.rope_tables(prompt_encoder.positions(mask))
    tt_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_mask = tensor.from_torch(mask.bool(), device=device)
    taps = encoder.forward(
        tt_ids,
        attention_mask=tt_mask,
        pos_embeds=(tensor.from_torch(cos, device=device), tensor.from_torch(sin, device=device)),
    )
    return taps, ttnn.concat(taps, dim=-1), (input_ids, mask, L)


def timed_encode(prompt_encoder: Flux3KleinPromptEncoder, device, prompt: str) -> tuple[Flux3KleinTextEncoding, float]:
    ttnn.synchronize_device(device)
    t0 = time.time()
    encoding = prompt_encoder.encode(prompt)
    ttnn.synchronize_device(device)
    return encoding, time.time() - t0


# --- D1: random weights, depth 2 -----------------------------------------------------------------------------------------------


@mesh_rows
def test_fast_random(
    mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp, snapshot, text_config, prompts, expect_error
):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from ...encoders.qwen3vl.common import capture_layer_outputs

    mesh = open_mesh(mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp)
    depth, taps = 2, (0, 1)
    torch.manual_seed(0)
    cfg = hf_text_config(text_config, num_hidden_layers=depth)
    lm = Qwen3VLTextModel(cfg).to(torch.bfloat16).eval()
    assert cfg.head_dim == FLUX3_KLEIN_TEXT_HEAD_DIM
    state = lm.state_dict()

    encoder, _ = build_flux3_klein_text_encoder(
        snapshot,
        mesh_device=mesh.device,
        parallel_config=mesh.parallel_config,
        ccl_manager=mesh.ccl,
        is_fsdp=is_fsdp,
        load_weights=False,
        num_layers=depth,
        activation_layers=taps,
    )
    # Strict-load negatives (the module-level half of H4): an extra layer key, a missing norm, a vision key.
    with expect_error(ValueError, "unexpected Torch state keys: layers.32.input_layernorm.weight"):
        encoder.load_torch_state_dict({**state, "layers.32.input_layernorm.weight": state["norm.weight"].clone()})
    with expect_error(ValueError, "missing Torch state keys: norm.weight"):
        encoder.load_torch_state_dict({k: v for k, v in state.items() if k != "norm.weight"})
    with expect_error(ValueError, "unexpected Torch state keys: visual.blocks.0.norm1.weight"):
        encoder.load_torch_state_dict({**state, "visual.blocks.0.norm1.weight": state["norm.weight"].clone()})
    t0 = time.time()
    encoder.load_torch_state_dict(state)
    logger.info(
        f"D1 {mesh.tag}: depth-{depth} random weights loaded in {time.time() - t0:.1f}s; {dram_per_bank(mesh.device)}"
    )

    prompt_encoder = Flux3KleinPromptEncoder(encoder, snapshot, mesh.device, text_config=text_config)
    for name in ("empty", DISTILLED_MAN_DOG):
        prompt = prompt_or_skip(prompts, name)
        input_ids, mask, L = prompt_encoder.tokenize(prompt)
        R = int(mask.sum())
        positions = prompt_encoder.positions(mask)
        with torch.inference_mode(), capture_layer_outputs(lm, taps) as captured:
            lm(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=positions[None].expand(4, -1, -1),
                use_cache=False,
            )
        reference = torch.cat([captured[i][0] for i in taps], dim=-1)  # (L, 2 * H) layer-major

        tt_taps, tt_concat, _ = device_taps(encoder, prompt_encoder, mesh.device, prompt)
        host_taps = [tensor.to_torch(t, mesh_axes=[None, None, None])[0] for t in tt_taps]
        concat = tensor.to_torch(tt_concat, mesh_axes=[None, None, None])[0]
        assert torch.equal(
            concat, torch.cat(host_taps, dim=-1)
        ), "device concat is not the layer-major torch.cat of the taps"
        assert concat.dtype == torch.bfloat16 and tuple(concat.shape) == (L, depth * HIDDEN)

        encoding, wall = timed_encode(prompt_encoder, mesh.device, prompt)
        rows, ids = Flux3KleinPromptEncoder.to_host(encoding)
        assert torch.equal(rows, concat) and torch.equal(ids, packing.text_ids(L))
        assert (
            encoding.length == L
            and encoding.padded_length == L
            and encoding.real_length == R
            and encoding.sp_axis is None
        )
        again, wall2 = timed_encode(prompt_encoder, mesh.device, prompt)
        assert torch.equal(Flux3KleinPromptEncoder.to_host(again)[0], rows), "second forward differs from the first"
        logger.info(f"D1 {mesh.tag} {name}: encode walls {wall:.3f}s / {wall2:.3f}s")

        result = compare_rows(f"D1 {mesh.tag} {name}", reference, rows, R)
        assert torch.isfinite(rows.float()).all()
        assert result["tap_min"] >= RANDOM_TAP_PCC, result
        assert result["tap_identity"], f"tap order: device taps best match reference taps {result['tap_best']}"
        if R < L:
            assert result["pad_tap_min"] >= RANDOM_TAP_PCC, result


# --- D2: real weights vs the goldens and the CPU oracle -------------------------------------------------------------------------


def load_real(mesh: Mesh, snapshot: Path, encoder) -> str:
    """``load_flux3_klein_text_encoder_weights`` with the MISS / HIT state and the wall logged."""
    blob = flux3_klein_text_encoder_blob(snapshot)
    cache_dir = cache.model_cache_dir(
        model_name=f"flux3_klein_text_encoder_{blob[:16]}",
        subfolder="text_encoder_L32",
        parallel_config=mesh.parallel_config,
        mesh_shape=tuple(mesh.device.shape),
        mesh_device=mesh.device,
        is_fsdp=mesh.is_fsdp,
        required=False,
    )
    state = (
        "HIT"
        if cache_dir is not None and (Path(cache_dir) / cache.CACHE_DICT_FILE).is_file()
        else ("MISS" if cache_dir is not None else "NO CACHE DIR")
    )
    t0 = time.time()
    load_flux3_klein_text_encoder_weights(
        encoder, snapshot, parallel_config=mesh.parallel_config, mesh_device=mesh.device, is_fsdp=mesh.is_fsdp
    )
    ttnn.synchronize_device(mesh.device)
    logger.info(
        f"RECORD weights {mesh.tag}: cache {state} ({cache_dir}) loaded in {time.time() - t0:.1f}s; {dram_per_bank(mesh.device)}"
    )
    return state


@mesh_rows
def test_real_weights(
    mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp, snapshot, text_config, prompts
):
    mesh = open_mesh(mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp)
    encoder, _ = build_flux3_klein_text_encoder(
        snapshot,
        mesh_device=mesh.device,
        parallel_config=mesh.parallel_config,
        ccl_manager=mesh.ccl,
        is_fsdp=is_fsdp,
        load_weights=False,
    )
    load_real(mesh, snapshot, encoder)
    prompt_encoder = Flux3KleinPromptEncoder(encoder, snapshot, mesh.device, text_config=text_config)

    assert "empty" in prompts, "the CFG unconditional is mandatory"
    goldens = [n for n in sorted(prompts) if n.startswith("golden_")]
    if not goldens:
        logger.warning(
            "no $FLUX3_KLEIN_GOLDENS/text_cache: the golden rows are skipped, synthetic prompts run against the CPU oracle"
        )
    order = ["empty", "one_word", "exact80", *goldens, *SNAPSHOT_TEXT_PROMPTS, *SENSITIVITY_PROMPTS]
    failures = []
    for name in order:
        prompt = prompts[name]
        _, mask, L = prompt_encoder.tokenize(prompt)
        R = int(mask.sum())
        reference, source = expected_tokens(snapshot, name, prompt, L)
        assert tuple(reference.shape) == (L, WIDTH) and reference.dtype == torch.bfloat16

        encoding, first = timed_encode(prompt_encoder, mesh.device, prompt)
        rows, ids = Flux3KleinPromptEncoder.to_host(encoding)
        ttnn.deallocate(encoding.tokens)
        again, warm = timed_encode(prompt_encoder, mesh.device, prompt)
        rows2, _ = Flux3KleinPromptEncoder.to_host(again)
        ttnn.deallocate(again.tokens)
        logger.info(
            f"TIMING {mesh.tag} {name} L={L}: encode first={first:.3f}s warm={warm:.3f}s; {dram_per_bank(mesh.device)}"
        )
        assert torch.equal(rows, rows2), "second encode differs from the first"

        assert tuple(rows.shape) == (L, WIDTH) and rows.dtype == torch.bfloat16
        assert torch.isfinite(rows.float()).all()
        assert torch.equal(ids, packing.text_ids(L)) and encoding.length == L and encoding.real_length == R
        result = compare_rows(f"D2 {mesh.tag} {name} vs {source}", reference, rows, R)
        try:
            if name in SENSITIVITY_PROMPTS:
                # bf16 tipping-point prompts (see SENSITIVITY_PROMPTS): implementation-independent checks only.
                early = result["tap_pcc"][:4]
                if result["sinks_ref"] != result["sinks_dev"]:
                    logger.warning(
                        f"D2 {mesh.tag} {name}: TIPPING POINT: sink rows oracle {result['sinks_ref']} vs device "
                        f"{result['sinks_dev']}; concat pcc {result['pcc']:.6f} recorded, not gated"
                    )
                else:
                    logger.info(
                        f"D2 {mesh.tag} {name}: sensitivity row, same sink rows; concat pcc {result['pcc']:.6f} recorded"
                    )
                assert (
                    min(early) >= SENSITIVITY_EARLY_TAP_PCC
                ), f"taps 0-3 pcc {['%.5f' % p for p in early]} < {SENSITIVITY_EARLY_TAP_PCC}"
                assert result["tap_best"][:4] == [0, 1, 2, 3], f"tap order of taps 0-3: {result['tap_best'][:4]}"
                if R < L:
                    assert (
                        result["pad_pcc"] >= PAD_BLOCK_PCC
                    ), f"pad block [{R}, {L}) pcc {result['pad_pcc']:.6f} < {PAD_BLOCK_PCC} (rel_rmse {result['pad_rel_rmse']:.5f})"
                continue
            assert result["tap_min"] >= TAP_PCC, f"per-tap pcc {result['tap_min']:.6f} < {TAP_PCC}"
            assert result["tap_identity"], f"tap order: device taps best match reference taps {result['tap_best']}"
            logger.info(f"D2 {mesh.tag} {name}: concat gate")
            assert_quality(reference, rows, pcc=CONCAT_PCC, relative_rmse=CONCAT_RMSE)
            if R < L:
                assert (
                    result["pad_pcc"] >= PAD_BLOCK_PCC
                ), f"pad block [{R}, {L}) pcc {result['pad_pcc']:.6f} < {PAD_BLOCK_PCC} (rel_rmse {result['pad_rel_rmse']:.5f})"
        except Exception as error:  # noqa: BLE001 -- run every prompt, then fail with the whole table
            logger.error(f"D2 {mesh.tag} {name}: FAIL {error}")
            failures.append(f"{name}: {error}")
    assert not failures, "; ".join(failures)


# --- D3: DiT handoff ---------------------------------------------------------------------------------------------------------------


@mesh_rows
def test_dit_handoff(mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp, snapshot, text_config, prompts):
    mesh = open_mesh(mesh_device, submesh_shape, tp_axis, topology, num_links, is_fsdp)
    sp_axis = 0
    sp = tuple(mesh.device.shape)[sp_axis]
    if sp == 1:
        pytest.skip(
            f"mesh {mesh.tag} has no sequence-parallel axis (shape {tuple(mesh.device.shape)}); the handoff row needs the DiT mesh"
        )
    encoder, _ = build_flux3_klein_text_encoder(
        snapshot,
        mesh_device=mesh.device,
        parallel_config=mesh.parallel_config,
        ccl_manager=mesh.ccl,
        is_fsdp=is_fsdp,
        load_weights=False,
    )
    load_real(mesh, snapshot, encoder)
    plain = Flux3KleinPromptEncoder(encoder, snapshot, mesh.device, text_config=text_config)
    for_dit = Flux3KleinPromptEncoder(encoder, snapshot, mesh.device, text_config=text_config, output_sp_axis=sp_axis)

    for name in ("empty", DISTILLED_MAN_DOG):
        prompt = prompt_or_skip(prompts, name)
        rows, ids = plain.encode_host(prompt)
        L = rows.shape[0]
        L_pad = packing.padded_length(L, sp)
        expected, _ = packing.pad_rows(rows, L_pad)

        encoding, wall = timed_encode(for_dit, mesh.device, prompt)
        assert (encoding.length, encoding.padded_length, encoding.sp_axis) == (L, L_pad, sp_axis)
        # A mesh-sharded ttnn tensor reports its PER-DEVICE shard shape (measured 2026-10-17 on the 4x8 mesh: both
        # tensor.from_torch(mesh_axes=[None, 0, None]) and ttnn.mesh_partition(dim=1) report (1, L_pad // sp, W) and
        # read back equal); the global shape is what the [None, sp, None] readback restores.
        assert tuple(encoding.tokens.shape) == (1, L_pad // sp, WIDTH) and encoding.tokens.dtype == ttnn.bfloat16
        got = tensor.to_torch(encoding.tokens, mesh_axes=[None, sp_axis, None])
        assert tuple(got.shape) == (1, L_pad, WIDTH)
        assert torch.equal(
            got[0], expected
        ), f"{name}: the [None, {sp_axis}, None] readback is not pad_rows(host output)"
        assert not got[0, L:].any() and torch.equal(got[0, :L], rows)
        host_rows, host_ids = Flux3KleinPromptEncoder.to_host(encoding)
        assert torch.equal(host_rows, rows) and torch.equal(host_ids, ids) and torch.equal(ids, packing.text_ids(L))
        logger.info(f"D3 {mesh.tag} {name}: L {L} -> L_pad {L_pad} over SP {sp}: exact; encode wall {wall:.3f}s")
        ttnn.deallocate(encoding.tokens)
