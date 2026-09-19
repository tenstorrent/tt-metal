# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dump Mistral Small 4 prefill KV cache to disk, for handoff to the tt-blaze decode stack.

WHAT THIS IS
------------
The prefill half of a hacky disaggregated prefill(ttnn) -> decode(blaze) pipe-cleaner. It
boots the validated `test_mistral4_prefill_transformer` row (mesh 8x4, num_links 2,
GPT_DEVICE gate, 36 layers, real weights), prefills ONE prompt, and writes:

    <out_dir>/meta.json                  handoff contract: token ids, dims, basis, resume point
    <out_dir>/kv_layer_<L>.safetensors   [prompt_len, 320] bf16, NATURAL position order, L=0..35

then exits so the devices are free for the decode leg. Decode is a separate process on a
separate mesh shape (blaze wants a 2x2 tile per stage), so the handoff is a directory, not a
socket.

NO TOKEN COMES OUT OF PREFILL (changed 2026-09)
-----------------------------------------------
The first revision of this script also sampled a first token and a short greedy reference
continuation, using `transformer._lm_head_and_extract` / `_sample` from the serve branch.
Those methods are GONE: `tt_prefill_transformer.forward` on this branch ends at the block
stack and its docstring now says so outright -- "there is no norm / LM-head / sampling tail:
decode owns the processing ... the populated KV cache is the output".

That is the disaggregated split done properly, and it changes the handoff contract rather
than breaking it. Prefill owns positions [0, prompt_len) of the KV cache and nothing else.
The decode side resumes by feeding the LAST prompt token at position `prompt_len - 1`, so it
recomputes that one position's KV on top of the seeded prefix and its own LM head emits the
first generated token. `meta.json` states that resume point explicitly
(`decode_resume_position`, `decode_feed_token`) so the receiver does not have to infer it.

The consequence for grading: there is no prefill-side reference continuation to compare
against any more. The control is the decode stack self-prefilling the SAME token ids and
generating from its own cache -- see the decode leg's A/B.

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
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests import fabric_profiles as _fp
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode

SERVE_SEQ_LEN = int(os.environ.get("PREFILL_SERVE_SEQ_LEN", 1024))
DUMP_DIR = os.environ.get("M4_DUMP_DIR", "/data/kmabee/disagg_kv/run1")
PROMPT = os.environ.get("M4_DUMP_PROMPT", "What is the capital of France? Answer in one sentence.")
# Handoff format version. Bump on any incompatible change to meta.json / tensor layout.
#   1 -- prefill sampled a first token and a reference continuation.
#   2 -- prefill emits KV only (the LM-head tail no longer exists on this branch); the receiver
#        resumes at `decode_resume_position` by feeding `decode_feed_token`.
HANDOFF_VERSION = 2

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
_FABRIC = os.environ.get("M4_DUMP_FABRIC", "fabric2d").strip()
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
    """Prefill one prompt, dump its KV, return (so run_model tears down and frees the chips)."""
    # Right padding keeps the prompt at rows [0, n) so the export slice is a prefix. Under left
    # padding the real tokens sit at the END of the window and every exported position would be
    # pad -- with entirely plausible shapes and magnitudes.
    assert padding_side == "right", f"dump requires right padding (prompt must occupy rows [0,n)); got {padding_side!r}"
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

    def _forward(win):
        """Run the block stack over the FULL padded window. The populated KV cache is the output.

        ``actual_isl=isl_total`` -- the padded window, NOT the token count -- is load-bearing.
        Passing a non-tile-aligned ``actual_isl`` (e.g. 552 = 17.25 tiles, 69/chip) HANGS the
        first tp-axis CCL in ``MLA._q_a_latent`` before any assert can fire. Measured on the
        first run of this experiment: reproducible across 3 galaxy resets and both Ring and
        all-Linear fabrics, with the inspector log showing the reduce_scatter kernel compiling
        fine and the dispatch simply never completing.

        KV correctness is unaffected: attention is causal, so rows 0..n-1 attend only to real
        tokens whatever ``actual_isl`` says, and only [0, n) is exported.
        """
        tt_tokens = _upload(win)
        transformer(
            tt_tokens,
            kvpe_cache,
            actual_isl=isl_total,  # NOT `n` -- see the docstring
            return_intermediates=False,
            read_profiler=False,
            index_kv_cache=index_kv_cache,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt_tokens)

    # ---- 1. prefill the prompt ------------------------------------------------------------
    t0 = time.time()
    _forward(window)
    logger.info(f"[m4-dump] prefill({n} tokens, window {isl_total}) took {time.time() - t0:.1f}s")

    # ---- 2. dump the KV --------------------------------------------------------------------
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

    # ---- 3. the handoff contract ----------------------------------------------------------
    # No reference continuation: this branch's prefill has no LM head (see the module docstring).
    # What the receiver needs instead is the RESUME POINT, stated rather than inferred.
    #
    # Seed KV rows [0, prompt_len-1) and feed prompt_token_ids[-1] at position prompt_len-1. The
    # decode stack then computes that last prompt position's KV with its OWN kernels, on top of
    # the seeded prefix, and its own LM head emits the first generated token. Seeding a causal
    # prefix is self-consistent: rows 0..k-1 depend only on tokens 0..k-1.
    #
    # Rows [0, prompt_len) are all exported -- the last row is a superset the receiver may use to
    # PCC its own recomputation of that position against prefill's, which is the cheapest
    # end-to-end check that the two stacks agree before any token is graded.
    meta = {
        "handoff_version": HANDOFF_VERSION,
        "producer": "tt-metal deepseek_v3_d_p mistral_small_4 prefill (full-window, KV-only)",
        "model_path": os.environ.get("MISTRAL4_HF_MODEL", ""),
        "prompt": PROMPT,
        "prompt_token_ids": [int(t) for t in prompt_ids],
        "prompt_len": n,
        # How to resume. Explicit because an off-by-one here is silent: decode would attend over
        # a cache that is one position short or one position stale and still emit fluent text.
        "decode_seed_positions": int(n - 1),
        "decode_resume_position": int(n - 1),
        "decode_feed_token": int(prompt_ids[-1]),
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
        "prefill_fabric": _FABRIC,
    }
    with open(os.path.join(DUMP_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    logger.success(
        f"[m4-dump] DONE -> {DUMP_DIR} (prompt_len={n}; decode seeds [0,{n - 1}) and feeds "
        f"token {prompt_ids[-1]} at position {n - 1})"
    )


@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 bring-up targets Blackhole")
@pytest.mark.parametrize("tokenizer", ["right"], indirect=True, ids=["right_pad"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            _FABRIC_CHOICES[_FABRIC](fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE),
            2,
            id=f"mesh-8x4-{_FABRIC}",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
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
    weight_cache_path,
    is_ci_env,
    is_ci_v2_env,
    tokenizer,
    request,
):
    """Prefill one prompt and dump its KV cache for the tt-blaze decode leg.

    Parametrized off `test_mistral4_prefill_transformer`'s validated pretrained row, so the
    dumped KV comes from the configuration the bring-up actually measured.
    """
    from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer import run_model
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    # Per-axis (sp, tp) topology, read off the fabric that was actually opened -- the same line
    # test_mistral4_prefill_transformer uses. A hardcoded scalar would disagree with
    # M4_DUMP_FABRIC.
    topology = per_axis_topology(device_params["fabric_config"])
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
        weight_cache_path,
        is_ci_env,
        is_ci_v2_env,
        tokenizer,
        request,
        serve_hook=dump_hook,
    )
