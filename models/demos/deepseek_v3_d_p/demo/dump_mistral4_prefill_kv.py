# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dump Mistral Small 4 prefill KV cache to disk, for handoff to the tt-blaze decode stack.

WHAT THIS IS
------------
The prefill half of a hacky disaggregated prefill(ttnn) -> decode(blaze) pipe-cleaner. It
boots the SAME configuration `serve_mistral4_interactive.py` serves (mesh 8x4, torus_xy,
num_links 2, GPT_DEVICE gate, 36 layers, real weights), prefills ONE prompt, and writes:

    <out_dir>/meta.json                  handoff contract: token ids, first token, dims, basis
    <out_dir>/kv_layer_<L>.safetensors   [prompt_len, 320] bf16, NATURAL position order, L=0..35

then exits so the devices are free for the decode leg. Decode is a separate process on a
separate mesh shape (blaze wants 4x2), so the handoff is a directory, not a socket.

WHY THE READBACK LOOKS LIKE THIS (the landmine)
-----------------------------------------------
There are TWO different KV layouts in this tree and they need different un-rotations:

  * CHUNKED runner (`tt/runners/prefill_kv_validation.py`): sequence chunks are distributed
    BLOCK-CYCLICALLY, so the readback must un-rotate via `blockcyclic_positions`.
  * FULL-WINDOW serve/demo path, which is what this script uses: `is_balanced=False`, so
    `chunk_order is None` and each chip holds a CONTIGUOUS slice of the sequence. Composing
    with `ConcatMesh2dToTensor(dims=(2, 1))` alone yields NATURAL position order -- no
    un-rotation at all.

Applying the block-cyclic inverse here would silently scramble positions and the KV would
still look plausible (right shape, right magnitudes). The recipe below is copied from the
canonical per-layer KVPE PCC in `tests/test_prefill_transformer.py` (the `do_return_kv`
block), which is the only place that compares this exact cache against a reference.

BASIS
-----
The `pe` half (last 64 of the 320) is written in the DEVICE's rotary basis, which is
Meta-INTERLEAVED. `prefill_kv_validation.py` re-interleaves the HF half-split golden to
compare against it, which is what pins the convention. `meta.json` records
`"pe_basis": "meta_interleaved"` so the decode side never has to guess -- and deliberately
does NOT reuse the `kv_post_transform_layer_<L>` golden-trace key, whose declared
convention is HF half-split.

RUN
---
    cd /data/kmabee/tt-metal
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    export LD_LIBRARY_PATH=$PWD/build_Release/lib:$LD_LIBRARY_PATH
    export MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603
    export TT_MISTRAL4_PREFILL_TTNN_CACHE=/data/kmabee/mistral4_caches/ttnn_cache_8x4
    export TT_MISTRAL4_PREFILL_HOST_REF_CACHE=/data/kmabee/mistral4_caches/ref_cache
    export PREFILL_SERVE_SEQ_LEN=1024
    export M4_DUMP_DIR=/data/kmabee/disagg_kv/run1
    export M4_DUMP_PROMPT="What is the capital of France? Answer in one sentence."
    export M4_DUMP_REF_TOKENS=8        # extra greedy tokens recorded as a decode reference
    ./python_env/bin/pytest models/demos/deepseek_v3_d_p/demo/dump_mistral4_prefill_kv.py -k dump -s

Run it DETACHED (a SIGTERM mid-device-run can wedge the board):
    setsid nohup <the pytest line> > dump.log 2>&1 < /dev/null &
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests import fabric_profiles as _fp
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

SERVE_SEQ_LEN = int(os.environ.get("PREFILL_SERVE_SEQ_LEN", 1024))
DUMP_DIR = os.environ.get("M4_DUMP_DIR", "/data/kmabee/disagg_kv/run1")
PROMPT = os.environ.get("M4_DUMP_PROMPT", "What is the capital of France? Answer in one sentence.")
# Extra greedy tokens generated AFTER the KV dump, recorded in meta.json purely as a reference
# continuation for the decode side to be graded against. Cheap (~1 s each traced) and it is the
# only same-weights, same-numerics continuation we can get.
REF_TOKENS = int(os.environ.get("M4_DUMP_REF_TOKENS", 8))

# Handoff format version. Bump on any incompatible change to meta.json / tensor layout.
HANDOFF_VERSION = 1

# Fabric profile selector. Default torus_xy == the profile test_serve uses, so the dumped KV comes
# from the validated configuration. BUT torus_xy is Ring on BOTH axes
# (tt_ccl._FABRIC_PER_AXIS_TOPOLOGY), and `torus_xy_device_params` is documented as the
# "production 8x4 Ring/Ring profile; requires a cabling-certified explicit descriptor". On a galaxy
# that is NOT wrap-cabling certified (the tests' own conftest says "generic CI galaxies are not
# wrap-cabling certified" and SKIPs torus_xy there), the tp-axis Ring reduce_scatter in
# MLA._q_a_latent never completes -- it waits on a wrap link that does not exist. Set
# M4_DUMP_FABRIC=fabric2d for all-Linear collectives, which need no wrap.
_FABRIC_CHOICES = {
    "torus_xy": _fp.torus_xy_device_params,  # (Ring, Ring)  -- needs BOTH wraps
    "torus_x": _fp.torus_x_device_params,  # (Linear, Ring) -- tp still Ring
    "torus_y": _fp.torus_y_device_params,  # (Ring, Linear) -- tp Linear, sp needs Y wrap
    "fabric2d": _fp.fabric2d_device_params,  # (Linear, Linear) -- no wrap assumed anywhere
}
_FABRIC = os.environ.get("M4_DUMP_FABRIC", "torus_xy").strip()
if _FABRIC not in _FABRIC_CHOICES:
    raise ValueError(f"M4_DUMP_FABRIC={_FABRIC!r} must be one of {sorted(_FABRIC_CHOICES)}")


def _encode_prompt(tokenizer, text):
    """Chat-templated prompt ids.

    Mirrors tt-blaze's `_encode_prompt` in `tests/blaze/backed/test_temporal_mistral_e2e.py`
    exactly, so both stacks tokenize identically. The ids are written to meta.json and the
    decode side must feed THOSE, not re-template the string -- a template mismatch would shift
    every position by a token or two and the KV cache would be silently wrong for the prompt
    blaze thinks it has.
    """
    try:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    except Exception as e:  # noqa: BLE001 -- a template-less tokenizer still encodes, raw
        logger.warning(f"[m4-dump] no chat template ({e!r}); encoding the raw prompt")
        templated = text
    ids = tokenizer.encode(templated, add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"prompt {text!r} encoded to zero tokens")
    return ids


def _read_kv_natural(kvpe_cache, mesh_device, num_layers, prompt_len):
    """[num_layers, prompt_len, kvpe] bf16 in NATURAL position order.

    Copied from the canonical KVPE readback in tests/test_prefill_transformer.py's
    `do_return_kv` block. Two things this does and why:
      * `ConcatMesh2dToTensor(dims=(2, 1))` -- mesh dim 0 (the sp=8 rows, each holding a
        contiguous slice of the sequence) concatenates onto tensor dim 2 (seq); mesh dim 1
        (the tp=4 cols, which hold REPLICAS) concatenates onto dim 1.
      * `[:, :1]` -- keep one TP replica and drop the other three.
    No block-cyclic inverse: see this module's docstring.
    """
    composed = ttnn.to_torch(
        kvpe_cache.storage,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    kv = composed[:, :1, :, :]  # [num_layers*num_users, 1, seq_total, kvpe]
    if kv.shape[0] < num_layers:
        raise AssertionError(
            f"KV cache batch dim {kv.shape[0]} < num_layers {num_layers}; "
            "init_kvpe_cache lays caches out user-major (batch = slot*num_layers + layer), so a "
            "smaller batch means this is not the cache this dumper assumes"
        )
    if kv.shape[2] < prompt_len:
        raise AssertionError(f"KV cache holds {kv.shape[2]} positions but the prompt is {prompt_len}")
    # num_users == 1 and slot == 0, so batch index == layer index.
    return kv[:num_layers, 0, :prompt_len, :]


def dump_hook(
    *,
    transformer,
    mesh_device,
    kvpe_cache,
    index_kv_cache,
    tokenizer,
    config,
    isl_total: int,
    sp_factor: int,
    isl_per_chip: int,
    chunk_order,
    padding_side: str,
):
    """Prefill one prompt, dump its KV, record a reference continuation, return (so run_model tears down)."""
    # Right padding is load-bearing exactly as in serve: the LM head reads row actual_isl-1.
    assert padding_side == "right", f"dump requires right padding (LM head reads row actual_isl-1); got {padding_side!r}"
    # The full-window path must not be block-cyclic; if this ever fires, the readback in
    # _read_kv_natural is wrong for the config and would scramble positions silently.
    assert chunk_order is None, (
        f"chunk_order is {chunk_order!r}, i.e. this is a BALANCED/chunked config whose KV is "
        "block-cyclic. _read_kv_natural assumes the contiguous full-window layout -- see the "
        "module docstring before changing it."
    )

    kv_lora = config.kv_lora_rank
    kvpe_dim = kv_lora + config.qk_rope_head_dim
    # NOTE: `config` here is the HF `Mistral4Config`, NOT the prefill runtime config, so it exposes
    # `num_hidden_layers` (HF naming) and has no `num_layers` / `sp_factor` / `chunk_size`.
    # `kv_lora_rank` and `qk_rope_head_dim` DO exist on it.
    num_layers = int(config.num_hidden_layers)

    prompt_ids = _encode_prompt(tokenizer, PROMPT)
    n = len(prompt_ids)
    if n >= isl_total:
        raise ValueError(
            f"prompt is {n} tokens but the window is {isl_total}; raise PREFILL_SERVE_SEQ_LEN "
            f"(multiple of {64 * sp_factor}) or shorten the prompt"
        )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 11
    window = torch.full((1, isl_total), pad_id, dtype=torch.int64)
    window[0, :n] = torch.tensor(prompt_ids, dtype=torch.int64)
    logger.info(f"[m4-dump] prompt {PROMPT!r} -> {n} tokens, window {isl_total}, pad {pad_id}")

    def _upload(host_ids):
        return ttnn.from_torch(
            host_ids.reshape(sp_factor, 1, isl_per_chip),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )

    def _forward(win, actual):
        """Run the block stack over the FULL padded window, then read the token at row ``actual-1``.

        This mirrors serve's *traced* path (``enable_trace`` + its eager tail), NOT serve's eager
        ``_forward_token``. The difference is load-bearing: the blocks must run with
        ``actual_isl=isl_total`` -- the shape the bring-up validated and the only one the traced
        capture ever uses -- because passing a non-tile-aligned ``actual_isl`` (e.g. 552 = 17.25
        tiles, 69/chip) HANGS the first tp-axis CCL in ``MLA._q_a_latent``, before any assert can
        fire. Measured: reproducible hang across 3 galaxy resets and both Ring and all-Linear
        fabrics; ``PREFILL_SERVE_TRACE=1`` is the default, so serve's eager branch that passes a
        bare ``actual_isl`` is essentially never exercised.

        KV correctness is unaffected: attention is causal, so rows 0..actual-1 attend only to real
        tokens whatever ``actual_isl`` says, and we export only positions [0, prompt_len).
        """
        tt_tokens = _upload(win)
        hidden = transformer(
            tt_tokens,
            kvpe_cache,
            actual_isl=isl_total,  # NOT `actual` -- see the docstring
            return_intermediates=False,
            read_profiler=False,
            temperature=0.0,
            index_kv_cache=index_kv_cache,
            stop_after_blocks=True,
        )
        # Token extraction at the TRUE row, exactly as serve does outside its trace.
        h = transformer.norm(hidden)
        _logits, first_token_logits = transformer._lm_head_and_extract(h, actual)
        token_id, _prob, _sweep = transformer._sample(first_token_logits, actual, 0.0)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt_tokens)
        return int(token_id)

    # ---- 1. prefill the prompt ------------------------------------------------------------
    t0 = time.time()
    first_token = _forward(window, n)
    logger.info(f"[m4-dump] prefill({n}) -> first token {first_token} in {time.time() - t0:.1f}s")

    # ---- 2. dump the KV *before* generating anything else ---------------------------------
    # Every extra token re-prefills a LONGER window and overwrites the cache, so the dump has
    # to happen here, while the cache holds exactly the prompt's KV.
    os.makedirs(DUMP_DIR, exist_ok=True)
    kv = _read_kv_natural(kvpe_cache, mesh_device, num_layers, n)
    if kv.shape[-1] != kvpe_dim:
        raise AssertionError(f"KV last dim {kv.shape[-1]} != kv_lora {kv_lora} + rope {config.qk_rope_head_dim}")

    from safetensors.torch import save_file

    norms = []
    for layer in range(num_layers):
        rows = kv[layer].contiguous()  # [n, kvpe]
        save_file(
            {"kv": rows},
            os.path.join(DUMP_DIR, f"kv_layer_{layer}.safetensors"),
            metadata={"layer": str(layer), "prompt_len": str(n), "kvpe_dim": str(kvpe_dim)},
        )
        f = rows.float()
        norms.append([round(f[:, :kv_lora].norm().item(), 4), round(f[:, kv_lora:].norm().item(), 4)])
    logger.info(f"[m4-dump] wrote {num_layers} layers x [{n}, {kvpe_dim}] -> {DUMP_DIR}")
    # Per-layer nope/pe norms are the cheapest smoke test on the decode side: an all-zero or
    # exploding layer says the readback is wrong before any PCC is run.
    logger.info(f"[m4-dump] per-layer [|nope|, |pe|]: {norms}")

    # ---- 3. reference continuation (AFTER the dump; it overwrites the cache) --------------
    ref_tokens, ref_pos = [], n
    stop_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    win = window.clone()
    win[0, ref_pos] = first_token
    ref_tokens.append(first_token)
    ref_pos += 1
    for _ in range(max(0, REF_TOKENS - 1)):
        if ref_pos >= isl_total:
            logger.warning(f"[m4-dump] hit the {isl_total}-token window; stopping the reference continuation")
            break
        tok = _forward(win, ref_pos)
        if tok in stop_ids:
            logger.info(f"[m4-dump] stop token {tok} after {len(ref_tokens)} reference tokens")
            break
        ref_tokens.append(tok)
        win[0, ref_pos] = tok
        ref_pos += 1

    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
    logger.info(f"[m4-dump] reference continuation {ref_tokens} -> {ref_text!r}")

    meta = {
        "handoff_version": HANDOFF_VERSION,
        "producer": "tt-metal deepseek_v3_d_p mistral_small4 prefill (full-window serve path)",
        "model_path": os.environ.get("MISTRAL4_HF_MODEL", ""),
        "prompt": PROMPT,
        "prompt_token_ids": [int(t) for t in prompt_ids],
        "prompt_len": n,
        "first_token": int(first_token),
        "reference_continuation": [int(t) for t in ref_tokens],
        "reference_text": ref_text,
        "num_layers": int(num_layers),
        "kvpe_dim": int(kvpe_dim),
        "kv_lora_rank": int(kv_lora),
        "qk_rope_head_dim": int(config.qk_rope_head_dim),
        # The two facts the decode side cannot recover from the tensors themselves.
        "position_order": "natural",
        "pe_basis": "meta_interleaved",
        "dtype": "bfloat16",
        "tensor_key": "kv",
        "tensor_shape": ["prompt_len", "kvpe_dim"],
        # Provenance, so a mismatched decode config is obvious rather than mysterious.
        "prefill_window": int(isl_total),
        "prefill_sp_factor": int(sp_factor),
        "prefill_mesh_shape": [int(mesh_device.shape[0]), int(mesh_device.shape[1])],
        "prefill_kv_format": str(getattr(kvpe_cache, "format", "unknown")),
        "greedy": True,
    }
    with open(os.path.join(DUMP_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    logger.success(f"[m4-dump] DONE -> {DUMP_DIR} (prompt_len={n}, first_token={first_token})")


@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            _FABRIC_CHOICES[_FABRIC](fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
            2,
            ttnn.Topology.Linear,
            id=f"mesh-8x4-{_FABRIC}",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
# timeout(0) by default (a prefill can legitimately be slow), but M4_DUMP_TIMEOUT=<sec> makes a
# HANG diagnosable: pytest-timeout raises in-process and prints the stack at the hang point,
# where a SIGKILL just leaves a dead process and a board needing a reset.
@pytest.mark.timeout(int(os.environ.get("M4_DUMP_TIMEOUT", "0")))
def test_dump(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Prefill one prompt and dump its KV cache for the tt-blaze decode leg.

    Same parametrization as `serve_mistral4_interactive.py::test_serve` (which in turn copies
    test_mistral4_prefill_transformer's validated `pretrained` row) so the dumped KV comes from
    the configuration the bring-up actually measured.
    """
    from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer import run_model

    sp_factor = 8
    min_multiple = 64 * sp_factor  # MoE masked_bincount needs 64 tokens/chip; 512 is the floor
    if SERVE_SEQ_LEN % min_multiple != 0:
        pytest.fail(
            f"PREFILL_SERVE_SEQ_LEN={SERVE_SEQ_LEN} must be a multiple of {min_multiple} "
            f"(64 tokens/chip for the MoE masked_bincount grid x sp={sp_factor}); 512 is the minimum"
        )

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        False,  # is_balanced -- MUST stay False: the readback assumes the contiguous layout
        SERVE_SEQ_LEN,  # isl_total
        8,  # dispatch_buffer_capacity_factor
        36,  # num_layers
        MistralSmall4Config.NUM_ROUTED_EXPERTS,
        GateComputeMode.GPT_DEVICE,
        num_links,
        topology,
        False,  # pcc_validation
        False,  # determinism_check
        1,  # num_iterations
        "json_prompts",  # input_source -- only builds the throwaway startup window
        True,  # use_pretrained
        False,  # return_kv_cache
        0.0,  # temperature
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
        serve_hook=dump_hook,
    )
