# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The FLUX.2 Klein text encoder (Qwen3-4B / Qwen3-8B, taps 9 / 18 / 27) on the mesh vs transformers'
``Qwen3ForCausalLM`` on CPU in bf16, on the same tokens.

Oracle: ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` (``pipeline_flux2_klein.py:208-263``) re-executed on the
snapshot's own ``text_encoder`` / ``tokenizer`` (``common.reference_prompt_embeds``): ``[512, joint]`` bf16 = per
token ``[hidden_states[9] | hidden_states[18] | hidden_states[27]]`` (HF ``hidden_states[k]`` is the output of layer
``k - 1``, no final norm) with ``ids = (0, 0, 0, pos)``. PAD ROWS ARE PART OF THE OUTPUT: the DiT attends to all 512
rows (``transformer_flux2.py:1228``, no text mask), so the pad block ``[R, 512)`` (R = real tokens) is compared too,
separately: a mask bug shows there first (HF right padding: every query row is computed, keys >= R are excluded).

Host rows (login box, no device; the checkpoint through ``common.require_checkpoint_dir``):
  H1 test_tokenize_matches_reference[size, prompt]  the port's ``format_prompts`` / ``tokenize`` byte- and
                          ``torch.equal`` to the reference recipe; right padding with ``<|endoftext|>``; the pinned
                          real-token count of every prompt; ``text_ids(512)`` == ``_prepare_text_ids``.
  H2 test_config_and_sizes[size]  ``text_encoder/config.json`` (hidden, layers, heads, GQA, head_dim 128, tied
                          embeddings) against the ``FLUX2_KLEIN_SIZES`` row and ``Flux2KleinConfig.from_snapshot``;
                          joint = 3 x hidden; the cache name takes 16 characters of the shard blob.
  H3 test_reduced_depth_config[size]  the device encoder is built at depth 27 = max(taps) without the final norm
                          (``FLUX2_KLEIN_TEXT_ENCODER_LAYERS``; the deepest tap, hidden_states[27], is the output of
                          layer 26): ``Qwen3Checkpoint`` config ``num_layers`` 27 / ``final_norm`` False, cache
                          subfolder ``text_encoder_L27`` (the full checkpoint keeps ``text_encoder``), the converted
                          checkpoint keys route to kept layers 0-26 + embedding / dropped layers 27-35 /
                          dropped ``final_norm.weight`` with nothing unrouted, and the depth / final-norm
                          contradictions raise.

Device rows (the (4, 8) parent with the 8 KB ring fabric; the encoder is TP over the DiT layout's TP axis and FSDP
over the other axis, exactly as the pipeline places it):
  D1 test_device_taps[size, layout]  ``build_flux2_klein_text_encoder`` + ``Flux2KleinTextEncoding.encode_host`` on
                          ``empty`` (the CFG unconditional, R 8), ``short`` (one sentence) and ``full512`` (the
                          snapshot's README.md, truncated: R == L == 512, no pad rows) vs the cached CPU oracle:
                          shape / dtype / finite, ids == text_ids(512), second encode torch.equal, per-tap pcc >=
                          TAP_PCC and per-tap rel_rmse logged, tap identity (the 3 x 3 cross-tap pcc matrix: each
                          device tap best matches ITS reference tap), ``assert_quality(pcc >= CONCAT_PCC,
                          relative_rmse <= CONCAT_RMSE)`` over ALL rows, the pad block ``[R, 512)`` at PAD_BLOCK_PCC.
                          The encoder under test is the pipeline's: 27 layers, no final norm.
  D2 test_reduced_depth_equals_full[size, layout]  the 27-layer encoder (the default) against the full 36-layer +
                          final-norm build of the same checkpoint on the same mesh: the ``[512, joint]`` rows of
                          every prompt ``torch.equal`` (layers 0-26 and their FSDP gathers are the same programs on
                          the same weights; the dropped layers and the final norm never touched the taps), both
                          cache entries named, warm encode walls of both logged (the 9-layer saving).
  D3 test_dit_handoff[size, layout]  ``Flux2KleinTextEncoding.encode_device`` (the pipeline's path: the rows stay on
                          the device, ``ttnn.mesh_partition`` ed over the layout's SP axis into the denoiser's
                          ``embed_prompt`` placement) vs ``encode_host`` on every prompt: per-device shard shape
                          ``(1, 512 / SP, joint)`` bf16, every device's shard ``torch.equal`` its slice of the host
                          rows (replicated over the other axis), the ``[None, sp, None]`` readback equal, the
                          ``with_host`` rows equal, the replicated variant (``sp_axis None``) equal, ids the text ids;
                          a sequence length that does not partition in tile rows raises; encode walls logged (the
                          FLUX 3 Klein D3 handoff row).
  D4 test_constants_cache[size, layout]  the per-request constants kept on the device (RoPE tables of ``arange(512)``
                          per sequence length, the causal + padding bias per real length in an LRU, module docstring
                          of ``prompt_encoder``): the host-built bias (``encoders.transformer.attention_bias``)
                          ``torch.equal`` the generic encoder's device-built one (``_prepare_attn_bias`` + pad) for
                          R 12 / 32 / 512 at L 512 and R 12 at L 500; every prompt's rows through the cached path
                          (``encode_host``, miss then hit) ``torch.equal`` the mask path (mask uploaded, bias built on
                          the device, fresh tables); hit / miss accounting, ``clear_constants`` rebuilds, the LRU bound
                          evicts; walls of both paths logged.

Gates are the FLUX 3 Klein text-encoder floors (``tests/models/flux3_klein/test_text_encoder_flux3_klein.py``, the
same Qwen3 text stack at TP4 / TP8 FSDP): applied here unmeasured, to be frozen from the first ring run (rounded
down) and never loosened without a measurement.

DEVICE ROWS: with ``$FLUX2_KLEIN_OUT`` set, D1 saves every device encoding as ``<out>/text_<tag>_<size>_<name>.safetensors``
(``rows`` (512, joint) bf16, ``reference`` the oracle rows, ``mask``) for host-side diagnostics (e.g. the pad block
against HF's own bf16/eager and fp32 variants, ``tmp_verify/text_pad_band.py``).

ORACLE CACHE: ``$FLUX2_KLEIN_ORACLE_CACHE`` (``common.cached_oracle``). Key = the text_encoder shard blobs, the
transformers / torch versions, sha256(prompt), L and the taps; files ``<digest>.safetensors`` (``tokens`` (512, joint)
bf16, ``input_ids``, ``mask``) + ``<digest>.json``. A 4B oracle takes ~10-30 s per prompt on a node CPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.qwen3 import Qwen3Checkpoint, Qwen3Encoder, qwen3_cache_subfolder, route_qwen3_state_keys
from ....encoders.transformer import MASK_VALUE, attention_bias
from ....models.transformers.flux2_klein.config_flux2_klein import FLUX2_KLEIN_SIZES, Flux2KleinConfig
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.flux2_klein import packing
from ....pipelines.flux2_klein.prompt_encoder import (
    FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE,
    FLUX2_KLEIN_TEXT_ENCODER_LAYERS,
    FLUX2_KLEIN_TEXT_OUT_LAYERS,
    FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH,
    FLUX2_KLEIN_TEXT_SUBFOLDER,
    Flux2KleinPromptEncoder,
    Flux2KleinTextEncoding,
    build_flux2_klein_text_encoder,
    format_prompts,
    real_lengths,
    text_encoder_cache_name,
    tokenize,
)
from ....utils import tensor
from ....utils.check import assert_quality
from .common import (
    BF16,
    DEFAULT_LAYOUT,
    LAYOUTS,
    MESH_SHAPE,
    SIZES,
    TEXT_LENGTH,
    TEXT_TAPS,
    cached_oracle,
    pcc,
    reference_prompt_embeds,
    reference_tokenize,
    rel_rmse_of,
    renders_dir,
    require_checkpoint_dir,
    text_encoder_blobs,
)
from .device_params import ring_params_8k_flux2_klein

# --- gate (never loosened) --------------------------------------------------------------------------------------------
# Per tap (one hidden-size block) over ALL 512 rows.
TAP_PCC = 0.99
# Per concat over ALL rows (the flux2 prompt-encoder gate, tests/models/flux2/test_prompt_encoder.py).
CONCAT_PCC = 0.9997
CONCAT_RMSE = 0.04
# The pad block [R, 512) alone: near-identical rows of small norm, so only a bug-discriminating pcc floor applies
# (a wrong mask lands two orders of magnitude lower on the FLUX 3 Klein measurements).
PAD_BLOCK_PCC = 0.997

PAD_ID = 151643  # <|endoftext|>, tokenizer_config.json pad_token
EMPTY_RENDER = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SHORT_PROMPT = "A man and his dog on a hill at sunset, photorealistic, 35mm."
EMPTY_IDS = [151644, 872, 198, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
PROMPT_NAMES = ("empty", "short", "full512")
# Pinned real-token counts (R) per prompt, measured with transformers 5.12.1 on the 4B snapshot's tokenizer (the
# thinking-disabled template appends an empty ``<think>`` block: 12 tokens for ''); full512 truncates to the full
# stream.
EXPECTED_REAL = {"empty": 12, "short": 32, "full512": TEXT_LENGTH}


# --- snapshot and prompts ---------------------------------------------------------------------------------------------


def snapshot_for(size: str) -> Path:
    return require_checkpoint_dir("distilled", size, required=("text_encoder/config.json", "tokenizer"))


def prompts_for(snapshot: Path, tokenizer) -> dict[str, str]:
    """The prompt set: ``full512`` is the snapshot's own README.md (natural text, deterministic per snapshot),
    repeated until its rendering exceeds 512 tokens so the truncation path (R == L) is exercised."""
    text = (snapshot / "README.md").read_text()
    full = text
    while int(reference_tokenize(tokenizer, full)[2].sum()) < TEXT_LENGTH:
        full = f"{full}\n{text}"
    return {"empty": "", "short": SHORT_PROMPT, "full512": full}


@pytest.fixture(scope="module", params=list(SIZES))
def size(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def snapshot(size) -> Path:
    return snapshot_for(size)


@pytest.fixture(scope="module")
def tokenizer(snapshot):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(str(snapshot), subfolder="tokenizer")


@pytest.fixture(scope="module")
def prompts(snapshot, tokenizer) -> dict[str, str]:
    return prompts_for(snapshot, tokenizer)


# --- H1: tokenization -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_tokenize_matches_reference(size, tokenizer, prompts, name) -> None:
    prompt = prompts[name]
    assert tokenizer.padding_side == "right"
    text, ref_ids, ref_mask = reference_tokenize(tokenizer, prompt)
    assert format_prompts(tokenizer, [prompt]) == [text]
    input_ids, mask = tokenize(tokenizer, [prompt], sequence_length=FLUX2_KLEIN_TEXT_SEQUENCE_LENGTH)
    assert input_ids.dtype == torch.int64 and tuple(input_ids.shape) == tuple(mask.shape) == (1, TEXT_LENGTH)
    assert torch.equal(input_ids, ref_ids) and torch.equal(mask, ref_mask)
    R = int(mask.sum())
    assert R == EXPECTED_REAL[name], f"{name}: {R} real tokens"
    assert torch.equal(mask[0, :R], torch.ones(R, dtype=mask.dtype)) and not mask[0, R:].any(), "right padding"
    if R < TEXT_LENGTH:
        assert (input_ids[0, R:] == PAD_ID).all()
    if name == "empty":
        assert text == EMPTY_RENDER
        assert input_ids[0].tolist() == EMPTY_IDS + [PAD_ID] * (TEXT_LENGTH - len(EMPTY_IDS))
    assert torch.equal(packing.text_ids(TEXT_LENGTH), reference_text_ids(TEXT_LENGTH))
    logger.info(f"H1 {size} {name}: R {R} / L {TEXT_LENGTH}, render {len(text)} chars")


def reference_text_ids(length: int) -> torch.Tensor:
    """``_prepare_text_ids`` (``pipeline_flux2_klein.py:266-283``): ``(0, 0, 0, l)``."""
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length))


# --- H2: config -------------------------------------------------------------------------------------------------------


def test_config_and_sizes(size, snapshot) -> None:
    cfg = json.loads((snapshot / "text_encoder" / "config.json").read_text())
    row = FLUX2_KLEIN_SIZES[size]
    assert cfg["hidden_size"] == row.text_hidden and cfg["model_type"] == "qwen3"
    assert cfg["num_hidden_layers"] == 36 and max(FLUX2_KLEIN_TEXT_OUT_LAYERS) <= cfg["num_hidden_layers"]
    assert cfg["head_dim"] == 128 and cfg["num_attention_heads"] * cfg["head_dim"] >= cfg["hidden_size"]
    assert cfg["num_attention_heads"] % cfg["num_key_value_heads"] == 0
    assert cfg["tie_word_embeddings"] is True and cfg["hidden_act"] == "silu"
    assert all(t == "full_attention" for t in cfg.get("layer_types", ["full_attention"]))
    config = Flux2KleinConfig.from_snapshot(snapshot)
    assert config.size == size and config.joint_dim == len(TEXT_TAPS) * row.text_hidden == row.joint_dim
    assert FLUX2_KLEIN_TEXT_OUT_LAYERS == TEXT_TAPS == config.text_taps
    assert text_encoder_cache_name(snapshot).startswith("flux2_klein_text_encoder_")
    assert len(text_encoder_cache_name(snapshot)) == len("flux2_klein_text_encoder_") + 16
    logger.info(f"H2 {size}: hidden {row.text_hidden}, joint {row.joint_dim}, shards {text_encoder_blobs(snapshot)}")


# --- H3: reduced depth ------------------------------------------------------------------------------------------------


def test_reduced_depth_config(size, snapshot, expect_error) -> None:
    cfg = json.loads((snapshot / "text_encoder" / "config.json").read_text())
    depth = cfg["num_hidden_layers"]
    assert FLUX2_KLEIN_TEXT_ENCODER_LAYERS == max(FLUX2_KLEIN_TEXT_OUT_LAYERS) == 27 < depth == 36

    # The pipeline's encoder: depth 27, no final norm, its own cache subfolder; the full checkpoint keeps the plain one.
    encoder = Flux2KleinPromptEncoder(
        checkpoint_name=str(snapshot), device=None, ccl_manager=None, parallel_config=None
    )
    assert encoder.num_layers == FLUX2_KLEIN_TEXT_ENCODER_LAYERS and encoder.final_norm is False
    assert encoder.cache_subfolder == f"{FLUX2_KLEIN_TEXT_SUBFOLDER}_L{FLUX2_KLEIN_TEXT_ENCODER_LAYERS}"
    checkpoint = Qwen3Checkpoint(str(snapshot), subfolder=FLUX2_KLEIN_TEXT_SUBFOLDER)
    assert checkpoint.num_layers == checkpoint.num_hidden_layers == depth and checkpoint.config.final_norm is True
    assert checkpoint.cache_subfolder == FLUX2_KLEIN_TEXT_SUBFOLDER
    full = Flux2KleinPromptEncoder(
        checkpoint_name=str(snapshot),
        device=None,
        ccl_manager=None,
        parallel_config=None,
        num_layers=depth,
        final_norm=True,
    )
    assert full.num_layers == depth and full.cache_subfolder == FLUX2_KLEIN_TEXT_SUBFOLDER
    assert (
        qwen3_cache_subfolder("text_encoder", num_layers=27, num_hidden_layers=36, final_norm=True)
        == "text_encoder_L27_norm"
    )

    # The converted checkpoint keys (the index, no tensors) route with nothing left over.
    index = json.loads((snapshot / "text_encoder" / "model.safetensors.index.json").read_text())["weight_map"]
    converted = Qwen3Encoder.convert_state({k: torch.empty(0) for k in index})
    keys = route_qwen3_state_keys(converted, num_layers=27, final_norm=False, num_hidden_layers=depth)
    per_layer = 11  # q/k/v/o_proj, q/k_norm, gate / linear_in / linear_out, attn_norm, ff_norm
    assert len(keys.kept) == 27 * per_layer + 1 and "token_embedding.weight" in keys.kept
    assert len(keys.dropped_layers) == 9 * per_layer and all(k.startswith("layers.") for k in keys.dropped_layers)
    assert keys.dropped_heads == ("final_norm.weight",)
    assert not any(k.startswith(tuple(f"layers.{i}." for i in range(27, 36))) for k in keys.kept)
    assert "lm_head.weight" not in index, "tied embeddings: a separate lm_head would be routed as final_linear"
    full_keys = route_qwen3_state_keys(converted, num_layers=depth, final_norm=True, num_hidden_layers=depth)
    assert set(full_keys.kept) == set(converted) and not full_keys.dropped_layers and not full_keys.dropped_heads

    with expect_error(ValueError, "outside the routed groups"):
        route_qwen3_state_keys([*converted, "vision.weight"], num_layers=27, final_norm=False)
    with expect_error(ValueError, "kept layer indices"):
        route_qwen3_state_keys(
            [k for k in converted if not k.startswith("layers.3.")], num_layers=27, num_hidden_layers=depth
        )
    with expect_error(ValueError, "deepest tap"):
        Flux2KleinPromptEncoder(
            checkpoint_name=str(snapshot), device=None, ccl_manager=None, parallel_config=None, num_layers=26
        )
    with expect_error(ValueError, "normed state"):
        Flux2KleinPromptEncoder(
            checkpoint_name=str(snapshot), device=None, ccl_manager=None, parallel_config=None, final_norm=True
        )
    with expect_error(ValueError, "num_layers"):
        Qwen3Checkpoint(str(snapshot), subfolder=FLUX2_KLEIN_TEXT_SUBFOLDER, num_layers=depth + 1)
    logger.info(f"H3 {size}: {len(keys.kept)} kept / {len(keys.dropped_layers)} dropped layer keys of {len(index)}")


# --- oracle -----------------------------------------------------------------------------------------------------------


def cached_text_oracle(snapshot: Path, prompt: str) -> torch.Tensor:
    import transformers

    fields = {
        "blobs": ",".join(text_encoder_blobs(snapshot)),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "length": TEXT_LENGTH,
        "taps": list(TEXT_TAPS),
    }

    def compute() -> dict[str, torch.Tensor]:
        tokens, input_ids, mask = reference_prompt_embeds(snapshot, prompt)
        return {"tokens": tokens, "input_ids": input_ids, "mask": mask}

    return cached_oracle(fields, f"text L {TEXT_LENGTH} prompt {fields['prompt_sha256'][:12]}", compute)["tokens"]


# --- device -----------------------------------------------------------------------------------------------------------

mesh_rows = pytest.mark.parametrize(
    "mesh_device, device_params",
    [[MESH_SHAPE, ring_params_8k_flux2_klein]],
    ids=["bh_4x8_ring"],
    indirect=["mesh_device", "device_params"],
)
layout_rows = pytest.mark.parametrize("layout", [DEFAULT_LAYOUT, *(n for n in LAYOUTS if n != DEFAULT_LAYOUT)])


@dataclass
class Mesh:
    device: ttnn.MeshDevice
    tp_axis: int
    ccl: CCLManager
    parallel_config: EncoderParallelConfig
    tag: str


def open_mesh(mesh_device, layout: str) -> Mesh:
    _, tp_axis = LAYOUTS[layout]
    shape = tuple(mesh_device.shape)
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=shape[tp_axis], mesh_axis=tp_axis))
    tag = f"{shape[0]}x{shape[1]}_{layout}_tp{shape[tp_axis]}_fsdp{shape[1 - tp_axis]}_ring"
    logger.info(f"mesh {tag}")
    return Mesh(mesh_device, tp_axis, ccl, parallel_config, tag)


def dram_per_bank(device) -> str:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return (
        f"DRAM {view.total_bytes_allocated_per_bank / 1e6:.1f} MB per bank x {view.num_banks} banks = "
        f"{view.total_bytes_allocated_per_bank * view.num_banks / 1e9:.2f} GB per chip"
    )


def split_taps(rows: torch.Tensor, hidden: int) -> list[torch.Tensor]:
    """``(L, n * H)`` layer-major rows -> the n per-tap ``(L, H)`` blocks."""
    return [rows[:, k * hidden : (k + 1) * hidden] for k in range(rows.shape[1] // hidden)]


def compare_rows(name: str, reference: torch.Tensor, actual: torch.Tensor, R: int, hidden: int) -> dict:
    """Log per-tap / concat / pad-block numbers; return them (the caller asserts)."""
    L = reference.shape[0]
    ref_f, act_f = reference.float(), actual.float()
    ref_taps, act_taps = split_taps(ref_f, hidden), split_taps(act_f, hidden)
    tap_pcc = [pcc(g, a) for g, a in zip(ref_taps, act_taps)]
    tap_rmse = [rel_rmse_of(g, a) for g, a in zip(ref_taps, act_taps)]
    cross = [[pcc(g, a) for g in ref_taps] for a in act_taps]
    best = [max(range(len(ref_taps)), key=lambda j: row[j]) for row in cross]
    out = {
        "pcc": pcc(ref_f, act_f),
        "rel_rmse": rel_rmse_of(ref_f, act_f),
        "max_abs": float((ref_f - act_f).abs().max()),
        "tap_pcc": tap_pcc,
        "tap_rel_rmse": tap_rmse,
        "tap_min": min(tap_pcc),
        "tap_identity": best == list(range(len(ref_taps))),
        "tap_best": best,
    }
    line = (
        f"RESULT {name} L={L} R={R}: concat pcc={out['pcc']:.6f} rel_rmse={out['rel_rmse']:.5f} "
        f"max_abs={out['max_abs']:.3g}; "
        f"per-tap pcc {['%.5f' % p for p in tap_pcc]} rel_rmse {['%.4f' % r for r in tap_rmse]}; tap identity "
        f"{out['tap_identity']} best-match {best}"
    )
    if R < L:
        pad_tap = [pcc(g[R:], a[R:]) for g, a in zip(ref_taps, act_taps)]
        out.update(
            pad_pcc=pcc(ref_f[R:], act_f[R:]),
            pad_rel_rmse=rel_rmse_of(ref_f[R:], act_f[R:]),
            pad_tap_min=min(pad_tap),
            real_pcc=pcc(ref_f[:R], act_f[:R]),
            real_rel_rmse=rel_rmse_of(ref_f[:R], act_f[:R]),
        )
        line += (
            f"; PAD rows [{R}, {L}) pcc={out['pad_pcc']:.6f} rel_rmse={out['pad_rel_rmse']:.5f} per-tap "
            f"{['%.5f' % p for p in pad_tap]}; REAL rows pcc={out['real_pcc']:.6f} rel_rmse={out['real_rel_rmse']:.5f}"
        )
    logger.info(line)
    return out


def timed_encode(encoding: Flux2KleinTextEncoding, device, prompt: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    ttnn.synchronize_device(device)
    t0 = time.time()
    rows, ids = encoding.encode_host(prompt)
    ttnn.synchronize_device(device)
    return rows, ids, time.time() - t0


@mesh_rows
@layout_rows
def test_device_taps(mesh_device, layout: str, size, snapshot, tokenizer, prompts) -> None:
    mesh = open_mesh(mesh_device, layout)
    hidden = FLUX2_KLEIN_SIZES[size].text_hidden
    t0 = time.time()
    encoder = build_flux2_klein_text_encoder(
        snapshot, mesh_device=mesh.device, parallel_config=mesh.parallel_config, ccl_manager=mesh.ccl, is_fsdp=True
    )
    logger.info(f"RECORD weights[{size}] {mesh.tag}: loaded in {time.time() - t0:.1f}s; {dram_per_bank(mesh.device)}")
    assert encoder.hidden_size == hidden and encoder.joint_dim == FLUX2_KLEIN_SIZES[size].joint_dim
    encoding = Flux2KleinTextEncoding(encoder)
    failures = []
    for name in PROMPT_NAMES:
        prompt = prompts[name]
        _, ref_ids, ref_mask = reference_tokenize(tokenizer, prompt)
        R = int(ref_mask.sum())
        reference = cached_text_oracle(snapshot, prompt)
        assert tuple(reference.shape) == (TEXT_LENGTH, encoder.joint_dim) and reference.dtype == BF16

        rows, ids, first = timed_encode(encoding, mesh.device, prompt)
        rows2, _, warm = timed_encode(encoding, mesh.device, prompt)
        logger.info(
            f"TIMING {mesh.tag} {size} {name}: encode first={first:.3f}s warm={warm:.3f}s; {dram_per_bank(mesh.device)}"
        )
        assert torch.equal(rows, rows2), "second encode differs from the first"
        assert tuple(rows.shape) == (TEXT_LENGTH, encoder.joint_dim) and rows.dtype == BF16
        assert torch.isfinite(rows.float()).all()
        assert torch.equal(ids, packing.text_ids(TEXT_LENGTH))

        result = compare_rows(f"D1 {mesh.tag} {size} {name}", reference, rows, R, hidden)
        out_dir = renders_dir()
        if out_dir is not None:
            from safetensors.torch import save_file

            path = out_dir / f"text_{mesh.tag}_{size}_{name}.safetensors"
            save_file(
                {"rows": rows.contiguous(), "reference": reference.contiguous(), "mask": ref_mask[0].contiguous()},
                str(path),
            )
            logger.info(f"DUMP {path}")
        try:
            assert result["tap_min"] >= TAP_PCC, f"per-tap pcc {result['tap_min']:.6f} < {TAP_PCC}"
            assert result["tap_identity"], f"tap order: device taps best match reference taps {result['tap_best']}"
            assert_quality(reference, rows, pcc=CONCAT_PCC, relative_rmse=CONCAT_RMSE)
            if R < TEXT_LENGTH:
                assert (
                    result["pad_pcc"] >= PAD_BLOCK_PCC
                ), f"pad block [{R}, {TEXT_LENGTH}) pcc {result['pad_pcc']:.6f} < {PAD_BLOCK_PCC}"
        except Exception as error:  # noqa: BLE001 -- run every prompt, then fail with the whole table
            logger.error(f"D1 {mesh.tag} {size} {name}: FAIL {error}")
            failures.append(f"{name}: {error}")
    encoder.deallocate_weights()
    ttnn.synchronize_device(mesh.device)
    assert not failures, "; ".join(failures)


@mesh_rows
@layout_rows
def test_reduced_depth_equals_full(mesh_device, layout: str, size, snapshot, prompts) -> None:
    """D2: the 27-layer encoder (default) == the full 36-layer + final-norm encoder on the taps, bit for bit."""
    mesh = open_mesh(mesh_device, layout)
    depth = json.loads((snapshot / "text_encoder" / "config.json").read_text())["num_hidden_layers"]
    builds = {}
    for name, kwargs in (("reduced", {}), ("full", {"num_layers": depth, "final_norm": True})):
        t0 = time.time()
        encoder = build_flux2_klein_text_encoder(
            snapshot,
            mesh_device=mesh.device,
            parallel_config=mesh.parallel_config,
            ccl_manager=mesh.ccl,
            is_fsdp=True,
            **kwargs,
        )
        logger.info(
            f"RECORD weights[{size}] {mesh.tag} {name}: {encoder.num_layers} layers, final norm {encoder.final_norm}, "
            f"cache {encoder.cache_subfolder}, loaded in {time.time() - t0:.1f}s; {dram_per_bank(mesh.device)}"
        )
        builds[name] = encoder
    assert builds["reduced"].num_layers == FLUX2_KLEIN_TEXT_ENCODER_LAYERS and not builds["reduced"].final_norm
    assert builds["reduced"].cache_subfolder == f"{FLUX2_KLEIN_TEXT_SUBFOLDER}_L{FLUX2_KLEIN_TEXT_ENCODER_LAYERS}"
    assert builds["full"].num_layers == depth and builds["full"].cache_subfolder == FLUX2_KLEIN_TEXT_SUBFOLDER
    encodings = {name: Flux2KleinTextEncoding(encoder) for name, encoder in builds.items()}
    failures = []
    for name in PROMPT_NAMES:
        prompt = prompts[name]
        rows = {}
        for build, encoding in encodings.items():
            _, _, first = timed_encode(encoding, mesh.device, prompt)
            rows[build], _, warm = timed_encode(encoding, mesh.device, prompt)
            logger.info(f"TIMING {mesh.tag} {size} {name} {build}: encode first={first:.3f}s warm={warm:.3f}s")
        a, b = rows["reduced"].float(), rows["full"].float()
        diff = int((a != b).sum())
        logger.info(
            f"RESULT D2 {mesh.tag} {size} {name}: reduced vs full equal={diff == 0} differing={diff}/{a.numel()} "
            f"max_abs={float((a - b).abs().max()):.3g} pcc={pcc(a, b):.8f}"
        )
        if diff:
            failures.append(f"{name}: {diff} of {a.numel()} elements differ")
    for encoder in builds.values():
        encoder.deallocate_weights()
    ttnn.synchronize_device(mesh.device)
    assert not failures, "; ".join(failures)


def timed_encode_device(encoding: Flux2KleinTextEncoding, device, prompt: str, *, with_host: bool):
    ttnn.synchronize_device(device)
    t0 = time.time()
    encoded = encoding.encode_device(prompt, with_host=with_host)
    ttnn.synchronize_device(device)
    return encoded, time.time() - t0


@mesh_rows
@layout_rows
def test_dit_handoff(mesh_device, layout: str, size, snapshot, prompts, expect_error) -> None:
    """D3: ``encode_device`` == ``encode_host`` bit for bit, in the DiT's placement (module docstring)."""
    mesh = open_mesh(mesh_device, layout)
    sp_axis, _ = LAYOUTS[layout]
    shape = tuple(mesh.device.shape)
    sp = shape[sp_axis]
    rows_per_device = TEXT_LENGTH // sp
    assert TEXT_LENGTH % (32 * sp) == 0
    encoder = build_flux2_klein_text_encoder(
        snapshot, mesh_device=mesh.device, parallel_config=mesh.parallel_config, ccl_manager=mesh.ccl, is_fsdp=True
    )
    plain = Flux2KleinTextEncoding(encoder)
    for_dit = Flux2KleinTextEncoding(encoder, sp_axis=sp_axis)
    assert plain.sp_axis is None and for_dit.sp_axis == sp_axis
    bad_length = 32 * (sp + 1)  # tile rows, but not a multiple of 32 x SP
    with expect_error(ValueError, "partition"):
        Flux2KleinTextEncoding(
            encoder, sequence_length=bad_length, text_ids=packing.text_ids(bad_length), sp_axis=sp_axis
        )
    with expect_error(ValueError, "not an axis"):
        Flux2KleinTextEncoding(encoder, sp_axis=len(shape))
    failures = []
    for name in PROMPT_NAMES:
        prompt = prompts[name]
        rows, ids, host_wall = timed_encode(plain, mesh.device, prompt)
        # (a) partitioned over the layout's SP axis, with the host copy
        encoded, wall_with_host = timed_encode_device(for_dit, mesh.device, prompt, with_host=True)
        assert (
            encoded.sp_axis == sp_axis
            and torch.equal(encoded.ids, ids)
            and torch.equal(ids, packing.text_ids(TEXT_LENGTH))
        )
        assert encoded.pair == (encoded.tokens, encoded.ids) and encoded.host is not None
        # A mesh-sharded ttnn tensor reports its PER-DEVICE shard shape (the FLUX 3 Klein D3 measurement).
        assert tuple(encoded.tokens.shape) == (1, rows_per_device, encoder.joint_dim), tuple(encoded.tokens.shape)
        assert encoded.tokens.dtype == ttnn.bfloat16 and encoded.tokens.layout == ttnn.TILE_LAYOUT
        host_equal = torch.equal(encoded.rows, rows)
        readback = tensor.to_torch(encoded.tokens, mesh_axes=[None, sp_axis, None])
        assert tuple(readback.shape) == (1, TEXT_LENGTH, encoder.joint_dim)
        readback_equal = torch.equal(readback[0], rows)
        # every device: its SP position's slice, replicated along the other axis
        coords = [tuple(int(v) for v in c) for c in encoded.tokens.tensor_topology().mesh_coords()]
        shards = ttnn.get_device_tensors(encoded.tokens)
        assert len(shards) == len(coords) == shape[0] * shape[1]
        bad_shards = []
        for coord, shard in zip(coords, shards):
            k = coord[sp_axis]
            expected = rows[k * rows_per_device : (k + 1) * rows_per_device]
            if not torch.equal(ttnn.to_torch(shard)[0], expected):
                bad_shards.append(coord)
        ttnn.deallocate(encoded.tokens)
        # (b) partitioned, no host copy (the pipeline's serving request): same device values
        encoded2, wall = timed_encode_device(for_dit, mesh.device, prompt, with_host=False)
        assert encoded2.rows is None and encoded2.host is None
        again_equal = torch.equal(tensor.to_torch(encoded2.tokens, mesh_axes=[None, sp_axis, None])[0], rows)
        ttnn.deallocate(encoded2.tokens)
        # (c) replicated (sp_axis None): the whole rows on every chip
        encoded3, wall_replicated = timed_encode_device(plain, mesh.device, prompt, with_host=False)
        assert encoded3.sp_axis is None
        assert tuple(encoded3.tokens.shape) == (1, TEXT_LENGTH, encoder.joint_dim)
        replicated_equal = torch.equal(ttnn.to_torch(ttnn.get_device_tensors(encoded3.tokens)[0])[0], rows)
        ttnn.deallocate(encoded3.tokens)
        logger.info(
            f"RESULT D3 {mesh.tag} {size} {name}: {TEXT_LENGTH} rows over SP {sp} (axis {sp_axis}) = {rows_per_device} per "
            f"device: host rows equal {host_equal}, readback equal {readback_equal}, bad shards {bad_shards}, "
            f"no-host run equal {again_equal}, replicated equal {replicated_equal}; walls encode_host {host_wall:.3f}s "
            f"encode_device {wall:.3f}s (+host read {wall_with_host:.3f}s, replicated {wall_replicated:.3f}s)"
        )
        if not (host_equal and readback_equal and not bad_shards and again_equal and replicated_equal):
            failures.append(
                f"{name}: host {host_equal} readback {readback_equal} bad shards {bad_shards} again {again_equal} "
                f"replicated {replicated_equal}"
            )
    encoder.deallocate_weights()
    ttnn.synchronize_device(mesh.device)
    assert not failures, "; ".join(failures)


def mask_path_rows(encoder: Flux2KleinPromptEncoder, prompt: str, device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The per-request path WITHOUT the constants cache: the mask uploaded, the bias built on the device by the generic
    encoder from it, the RoPE tables uploaded; ``(rows [512, joint] bf16, mask [1, 512], wall)``."""
    tokens, mask = tokenize(encoder.tokenizer, [prompt], sequence_length=TEXT_LENGTH)
    ttnn.synchronize_device(device)
    t0 = time.time()
    tt_tokens = tensor.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_mask = tensor.from_torch(mask, device=device)
    cos, sin = encoder.rope_tables(torch.arange(TEXT_LENGTH))
    pos_embeds = (tensor.from_torch(cos, device=device), tensor.from_torch(sin, device=device))
    hidden = encoder._encoder.forward(
        tt_tokens,
        mask=tt_mask,
        pos_embeds=pos_embeds,
        skip_final_linear=True,
        output_hidden_states=set(encoder.out_layers),
    )
    tt_embeds = ttnn.concat(list(hidden), dim=-1)
    rows = ttnn.to_torch(ttnn.get_device_tensors(tt_embeds)[0])[0].to(torch.bfloat16)
    ttnn.synchronize_device(device)
    wall = time.time() - t0
    for t in (tt_tokens, tt_mask, *pos_embeds, tt_embeds, *hidden):
        ttnn.deallocate(t)
    return rows, mask, wall


def device_built_bias(encoder: Flux2KleinPromptEncoder, mask: torch.Tensor, device) -> torch.Tensor:
    """What ``TransformerEncoder.forward`` builds from ``mask`` ``[1, L]``: ``_prepare_attn_bias`` + the tile pad."""
    seq_len = mask.shape[1]
    padded = -(-seq_len // 32) * 32
    tt_mask = tensor.from_torch(mask, device=device)
    bias = encoder._encoder._prepare_attn_bias(
        tt_mask, query_length=seq_len, query_pos=0, kv_length=seq_len, device=device
    )
    bias = ttnn.pad(bias, [(0, padded - seq_len), (0, padded - seq_len)], value=MASK_VALUE)
    out = ttnn.to_torch(ttnn.get_device_tensors(bias)[0])
    ttnn.deallocate(tt_mask)
    ttnn.deallocate(bias)
    return out


@mesh_rows
@layout_rows
def test_constants_cache(mesh_device, layout: str, size, snapshot, prompts) -> None:
    """D4: the cached per-request constants give the mask path's rows bit for bit (module docstring)."""
    mesh = open_mesh(mesh_device, layout)
    encoder = build_flux2_klein_text_encoder(
        snapshot, mesh_device=mesh.device, parallel_config=mesh.parallel_config, ccl_manager=mesh.ccl, is_fsdp=True
    )
    encoding = Flux2KleinTextEncoding(encoder)
    assert encoder.constants_info() == {
        "pos_embeds": [],
        "attn_biases": [],
        "bias_cache_size": FLUX2_KLEIN_ENCODER_BIAS_CACHE_SIZE,
        "hits": 0,
        "misses": 0,
    }
    # (a) the host bias is the device-built one, value for value
    bias_equal = {}
    for length, real in ((TEXT_LENGTH, 12), (TEXT_LENGTH, 32), (TEXT_LENGTH, TEXT_LENGTH), (500, 12)):
        mask = (torch.arange(length)[None, :] < real).to(torch.int64)
        assert real_lengths(mask) == (real,)
        host = attention_bias(mask)
        assert host.dtype == torch.bfloat16 and tuple(host.shape) == (
            1,
            1,
            -(-length // 32) * 32,
            -(-length // 32) * 32,
        )
        assert set(host.unique().tolist()) <= {0.0, MASK_VALUE}
        device = device_built_bias(encoder, mask, mesh.device)
        bias_equal[(length, real)] = torch.equal(device.to(torch.bfloat16), host) and device.dtype == torch.bfloat16
    assert real_lengths(torch.tensor([[1, 0, 1, 1]])) is None
    assert all(bias_equal.values()), bias_equal
    # (b) every prompt: cached path (miss, then hit) == mask path
    failures = []
    walls = {}
    for i, name in enumerate(PROMPT_NAMES):
        prompt = prompts[name]
        reference, mask, mask_wall = mask_path_rows(encoder, prompt, mesh.device)
        real = int(mask.sum())
        rows_miss, _, miss_wall = timed_encode(encoding, mesh.device, prompt)
        info = encoder.constants_info()
        assert info["pos_embeds"] == [TEXT_LENGTH] and (1, TEXT_LENGTH, (real,)) in info["attn_biases"]
        assert info["misses"] == i + 1, info
        rows_hit, _, hit_wall = timed_encode(encoding, mesh.device, prompt)
        assert encoder.constants_info()["hits"] == i + 1 and encoder.constants_info()["misses"] == i + 1
        miss_equal, hit_equal = torch.equal(rows_miss, reference), torch.equal(rows_hit, reference)
        if not (miss_equal and hit_equal):
            failures.append((name, miss_equal, hit_equal))
        walls[name] = (mask_wall, miss_wall, hit_wall)
        logger.info(
            f"RESULT D4 {mesh.tag} {size} {name}: R {real}: cached rows == mask path: miss {miss_equal} hit {hit_equal}; "
            f"walls mask path {mask_wall:.3f}s cached miss {miss_wall:.3f}s hit {hit_wall:.3f}s"
        )
    assert not failures, failures
    # (c) clear rebuilds; the LRU bound evicts the oldest length
    encoder.clear_constants()
    assert encoder.constants_info()["pos_embeds"] == [] and encoder.constants_info()["attn_biases"] == []
    rows, _, _ = timed_encode(encoding, mesh.device, prompts["short"])
    assert torch.equal(rows, mask_path_rows(encoder, prompts["short"], mesh.device)[0])
    # the cache holds short (R 32); the bound at 2 then keeps the two most recently USED lengths: empty (R 12) comes in,
    # short is hit (moved to the back), full512 (R 512) comes in and evicts empty
    encoder._bias_cache_size = 2
    for name in PROMPT_NAMES:
        timed_encode(encoding, mesh.device, prompts[name])
    kept = encoder.constants_info()["attn_biases"]
    assert kept == [(1, TEXT_LENGTH, (EXPECTED_REAL["short"],)), (1, TEXT_LENGTH, (EXPECTED_REAL["full512"],))], kept
    rows, _, _ = timed_encode(encoding, mesh.device, prompts["short"])
    assert torch.equal(rows, mask_path_rows(encoder, prompts["short"], mesh.device)[0])
    encoder.deallocate_weights()
    assert encoder.constants_info()["attn_biases"] == [] and encoder.constants_info()["pos_embeds"] == []
