# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""**Delta 3**: chunk *k*'s queries attending the prefix read back out of the cache. Gate: `G-CHUNK-ATTN`.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaModel.forward` — the whole stack, run
twice with the same weights and the same tokens and two different attention cores.

`G-CHUNK` (P7) proved deltas **1** (the indexed RoPE offset) and **2** (the advancing cache-write
offset) exactly, on one card, by feeding both KV producers the same hidden states. It could not
touch delta 3, which needs the ring path and TP=8, so it was recorded `BLOCKED` with `07_RISKS.md`
R-023 naming P8 as the owner (`BRINGUP_RECIPE.md:1656-1660`). **This file is that gate.**

## The two arms, and the one model that runs both

One `TtPrefillRuntime` is built with `chunk_size=512` and `additional_chunk_sizes=(1024,)`, so a
single set of weights and a single `CCLManager` serve both arms and the only difference between
them is the chunk size each call passes (`DEC-086`):

| arm | chunk_global | chunks | core | why |
|---|---|---|---|---|
| one-shot | 1024 == `max_seq_len` | 1 | `sp_bootstrap` | the ring op enters chunked mode only when Q's per-device length is **less** than K's, and a chunk that fills the cache makes them equal |
| chunked | 512 | 2 | `sp_ring` | chunk 1 attends `[0, 1024)` with only `[512, 1024)` live — the rest comes out of the cache |

Which core ran is **asserted** on both arms, not inferred: Appendix B's final row is "everything
passes but the numbers look too good | you measured the SP bootstrap because
`max_seq_len == chunk_size`", and that is a mis-selection between exactly these two.

## Why the threshold names a depth, and what is gated where

A mutual-PCC claim ("path A == path B") is a **per-op** claim. Applied to a 32-layer accumulated
statistic it measures depth, not the op (`BRINGUP_RECIPE.md:1768-1772`, and the same trap
`G-CHUNK`/`G-MODEL` carry). So:

* **layer 1** — one attention layer has run, so the mutual K PCC there is the per-op claim.
  Threshold **>= 0.999**. Layer 0 is reported too and is a *control by construction*: its K and V
  are produced before any attention, so the two arms must agree essentially exactly there and a
  delta-3 bug cannot show up in it. That is the whole reason a layer-0-only check is worthless here.
* **deep layers** — gated on the per-layer error **step** (<= 4x from layer 3), which is
  depth-invariant.
* **both arms vs the fp32 golden** — `G-CHUNK`'s carried thresholds, K >= 0.99 / V >= 0.98.
* **the accumulated min over 32 layers** — recorded, **not** gated.

Run:
    export TT_CACHE_PATH=$HOME/.cache/llama31_8b_d_p
    export PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s1024
    HF_MODEL=... pytest models/demos/llama31_8b_d_p/tests/unit/test_chunked_attention_ring.py -x -q
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.galaxy_prefill_kv_pcc import (
    GOLDEN_K_THRESHOLD,
    GOLDEN_V_THRESHOLD,
    allocate_engine_cache,
    build_runtime,
    expected_core,
    load_golden,
    meta_head_index,
    plan,
    read_kv_cache,
    run_prefill,
    score,
)
from models.demos.llama31_8b_d_p.tests.test_factory import (
    GALAXY_MESH_SHAPE,
    galaxy_device_params,
    requires_galaxy,
    requires_hf_reference,
    requires_ring_fabric,
)
from models.demos.llama31_8b_d_p.tests.unit.test_attention_chunked_vs_ref import _golden_metadata, requires_golden_trace
from models.demos.llama31_8b_d_p.tests.unit.test_decoder_layer_vs_ref import _meta_head_index
from models.demos.llama31_8b_d_p.tt.attention.prefill import select_attention_core
from models.demos.llama31_8b_d_p.tt.config import derive_head_dim
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs

# `BRINGUP_RECIPE.md:1768-1772`.
PER_OP_PCC_THRESHOLD = 0.999  # at layer 1, and only there
MAX_LAYER_STEP = 4.0
FIRST_GATED_STEP_LAYER = 3  # "from layer 3", 0-based, as `G-CHUNK` uses

ONE_SHOT_CHUNK = 1024  # == max_seq_len -> sp_bootstrap
RING_CHUNK = 512  # 2 chunks -> sp_ring

_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bringup_log", "raw"
)


def test_meta_head_index_does_not_drift_between_the_script_and_the_tests():
    """The one duplicated helper in the package, pinned by equality (`DEC-085`).

    `tests/galaxy_prefill_kv_pcc.py` is a **script** and must not import a pytest module, so it
    carries its own copy of the HF->Meta head-dim permutation. A silent divergence would permute the
    golden one way in `G-MESH-KV` and another in every unit gate, and both would still look
    plausible. Cheap to pin; expensive to discover.
    """
    for head_dim in (64, 128):
        assert torch.equal(meta_head_index(head_dim), _meta_head_index(head_dim)), (
            f"the script's meta_head_index({head_dim}) and the tests' _meta_head_index({head_dim}) "
            f"have diverged; G-MESH-KV and every unit gate would permute the golden differently"
        )


def _steps(curve, n_layers):
    """`{layer: (1 - pcc_l) / (1 - pcc_{l-1})}` — the per-layer error step, `inf` on a zero base."""
    out = {}
    for layer in range(1, n_layers):
        prev = 1.0 - curve[layer - 1]
        out[layer] = float("inf") if prev <= 0 else (1.0 - curve[layer]) / prev
    return out


@requires_galaxy
@requires_ring_fabric
@requires_hf_reference
@requires_golden_trace
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_ring_chunked_attention_matches_one_shot_and_golden(mesh_device, reset_seeds):
    """**`G-CHUNK-ATTN`.** Ring (cache-backed) vs SP bootstrap (one chunk), and both vs the golden.

    * **Input distribution:** the golden trace's own `token_ids` — 1024 real
      Llama-3.1-8B-Instruct tokens — and real checkpoint weights. §2.2.2's "real embedding scale"
      arm, which is the one that predicts model behaviour; there is no synthetic arm here because
      the quantity of interest is an accumulated model-level statistic and a `randn` input would
      dilute the attention error through the residual stream (measured 1.47-1.81x vs 3.83x at
      `G-LAYER`).
    * **Reference dtype policy:** the golden is **fp32** throughout (`DEC-059`), bit-identical to
      `LlamaModel`'s own loop (`G-GOLDEN`, re-run for this 1024-token trace: `max|delta| = 0.0` on
      K, V and the post-norm hidden over all 32 layers). K is permuted **HF -> Meta before** any
      quantiser touches it (recipe §2.2.3a).
    * **Computed noise floor:** for the *mutual* comparison the floor is **1.0 by construction** —
      the two arms are the same weights, the same dtypes and the same tokens, so at layer 1 they
      differ only by the attention core and the expected value is exactness. Against the golden the
      per-layer floors are `G-KV-TP8`'s (layer 0 measured 1.10x on K and 1.02x on V there, on the
      identical producer), and this gate deliberately does **not** recompute them: it gates the
      mutual claim and carries `G-CHUNK`'s absolute thresholds for the golden claim.
    * **Negative control:** `test_ring_prefix_control_collapses_past_layer_zero` below — run chunk 1
      against a cache whose chunk-0 prefix was never written.
    """
    metadata = _golden_metadata()
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    sp, tp = GALAXY_MESH_SHAPE
    if n_tokens != ONE_SHOT_CHUNK:
        # A **skip**, not an assert (`DEC-091`). This gate needs a trace long enough for the chunked
        # arm to have two chunks of >= 512 global (`chunk_local = 128 = the ring q_chunk_size`),
        # i.e. >= 1024 tokens. Asserting would make the per-phase regression FAIL on the 512-token
        # trace P7's gates were recorded against, which is a harness fact masquerading as a defect.
        pytest.skip(
            f"this gate needs a {ONE_SHOT_CHUNK}-token golden trace so the chunked arm has two "
            f"chunks of {RING_CHUNK} with a tile-aligned per-chip chunk; PREFILL_TRACE_DIR has "
            f"{n_tokens}. Regenerate with: scripts/generate_golden_kv_cache.py --tokens "
            f"{ONE_SHOT_CHUNK} --out <dir> --verify-loop"
        )

    args = ModelArgs(mesh_device, max_seq_len=ONE_SHOT_CHUNK)
    hf = args.hf_config
    n_layers = hf["num_hidden_layers"]
    head_dim = derive_head_dim(hf)
    golden = load_golden(os.environ["PREFILL_TRACE_DIR"], n_layers, head_dim, n_tokens)

    state_dict = ModelArgs.load_state_dict(args.model_path)
    # ONE runtime, ONE CCLManager, both chunk sizes (`DEC-086`).
    runtime = build_runtime(
        mesh_device,
        hf,
        state_dict,
        num_layers=n_layers,
        chunk_global=RING_CHUNK,
        total=ONE_SHOT_CHUNK,
        cache_path=args.weight_cache_path(ttnn.bfloat8_b),
    )
    runtime.config.additional_chunk_sizes = (ONE_SHOT_CHUNK,)
    runtime.config.__post_init__()
    runtime.rope_indexed = runtime._build_indexed_rope()
    del state_dict
    assert set(runtime.rope_indexed) == {RING_CHUNK, ONE_SHOT_CHUNK}, (
        f"both arms need an indexed RoPE table; the runtime has {sorted(runtime.rope_indexed)}. A "
        f"table cannot be built per chunk (tt_prefill_runtime.py::_require_supported_chunk_size)."
    )

    arms = {}
    for name, chunk_size in (("one_shot", ONE_SHOT_CHUNK), ("ring", RING_CHUNK)):
        n_chunks, chunk_global, total = plan(n_tokens, chunk_size, chunk_size < n_tokens, sp)
        assert (chunk_global, total) == (chunk_size, ONE_SHOT_CHUNK)
        kv_cache = allocate_engine_cache(mesh_device, num_layers=n_layers, total=total, head_dim=head_dim)
        core = select_attention_core(
            runtime.model.attention_config,
            runtime.mesh_config,
            kv_cache,
            seq_len=chunk_global // sp,
            cached_len=0,
        )
        want = expected_core(chunk_size < n_tokens, total, chunk_global)
        assert core == want, f"the {name} arm selects {core!r}, not the {want!r} core it must measure"
        seconds = run_prefill(
            runtime, kv_cache, token_ids, n_chunks=n_chunks, chunk_global=chunk_global, n_tokens=n_tokens
        )
        read_back = read_kv_cache(
            kv_cache,
            num_layers=n_layers,
            n_kv=hf["num_key_value_heads"],
            head_dim=head_dim,
            chunk_global=chunk_global,
            total=total,
            n_tokens=n_tokens,
            mesh_shape=GALAXY_MESH_SHAPE,
        )
        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)
        arms[name] = {"read_back": read_back, "vs_golden": score(read_back, golden, n_layers), "core": core}
        logger.info(
            f"[G-CHUNK-ATTN] {name} arm: core={core} chunk_global={chunk_global} "
            f"chunk_local={chunk_global // sp} n_chunks={n_chunks} sp={sp} tp={tp} in "
            f"{seconds * 1000:.1f} ms; vs golden min K = {min(arms[name]['vs_golden']['k'].values()):.7f}, "
            f"min V = {min(arms[name]['vs_golden']['v'].values()):.7f}"
        )

    mutual = {"k": {}, "v": {}}
    for layer in range(n_layers):
        ring_k, ring_v = arms["ring"]["read_back"][layer]
        one_k, one_v = arms["one_shot"]["read_back"][layer]
        _, pcc_k = comp_pcc(one_k, ring_k, 0.0)
        _, pcc_v = comp_pcc(one_v, ring_v, 0.0)
        mutual["k"][layer], mutual["v"][layer] = float(pcc_k), float(pcc_v)
        logger.info(
            f"[G-CHUNK-ATTN] L{layer:>2}: ring vs one-shot K={float(pcc_k):.7f} V={float(pcc_v):.7f} "
            f"| vs golden ring K={arms['ring']['vs_golden']['k'][layer]:.7f} "
            f"one-shot K={arms['one_shot']['vs_golden']['k'][layer]:.7f}"
        )

    steps = {name: _steps(mutual[name], n_layers) for name in ("k", "v")}
    gated = {
        name: {layer: step for layer, step in steps[name].items() if layer >= FIRST_GATED_STEP_LAYER}
        for name in ("k", "v")
    }
    logger.info(
        f"[G-CHUNK-ATTN] the depth structure the threshold has to name: "
        f"L0 (no attention has run) K={mutual['k'][0]:.7f}, "
        f"**L1 (one attention layer) K={mutual['k'][1]:.7f}** (threshold {PER_OP_PCC_THRESHOLD}), "
        f"L8 K={mutual['k'][8]:.7f}, min over {n_layers} layers K="
        f"{min(mutual['k'].values()):.7f} at L{min(mutual['k'], key=mutual['k'].get)} — "
        f"RECORDED, NOT GATED: that min is one attention output pushed through 20+ residual "
        f"streams, so holding it to a per-op threshold measures depth, not the op."
    )
    for name in ("k", "v"):
        worst_layer = max(gated[name], key=gated[name].get)
        logger.info(
            f"[G-CHUNK-ATTN] mutual {name.upper()}: worst gated per-layer error step "
            f"L{worst_layer} = {gated[name][worst_layer]:.2f}x (budget {MAX_LAYER_STEP}x); "
            f"min over all layers {min(mutual[name].values()):.7f}"
        )

    os.makedirs(_RAW_DIR, exist_ok=True)
    with open(os.path.join(_RAW_DIR, "G-CHUNK-ATTN_per_layer_pcc.json"), "w") as f:
        json.dump(
            {
                "trace_dir": os.environ["PREFILL_TRACE_DIR"],
                "mesh": list(GALAXY_MESH_SHAPE),
                "sp": sp,
                "tp": tp,
                "n_tokens": n_tokens,
                "n_layers": n_layers,
                "one_shot": {"chunk_global": ONE_SHOT_CHUNK, "core": arms["one_shot"]["core"]},
                "ring": {"chunk_global": RING_CHUNK, "core": arms["ring"]["core"]},
                "mutual": {name: {str(k): v for k, v in sorted(vals.items())} for name, vals in mutual.items()},
                "mutual_step": {name: {str(k): v for k, v in sorted(vals.items())} for name, vals in steps.items()},
                "vs_golden": {
                    arm: {
                        name: {str(k): v for k, v in sorted(vals.items())} for name, vals in data["vs_golden"].items()
                    }
                    for arm, data in arms.items()
                },
            },
            f,
            indent=2,
        )

    # --- assertions ---------------------------------------------------------------------------
    assert mutual["k"][1] >= PER_OP_PCC_THRESHOLD, (
        f"at layer 1 — ONE attention layer, i.e. the per-op claim — the ring core's K differs from "
        f"the one-shot core's at PCC {mutual['k'][1]:.7f} < {PER_OP_PCC_THRESHOLD}. Check the "
        f"block-cyclic Q/cache position mapping and kv_cache_batch_idx (G-SP-RING's controls) "
        f"before suspecting depth."
    )
    assert mutual["v"][1] >= PER_OP_PCC_THRESHOLD, (
        f"at layer 1 the ring core's V differs from the one-shot core's at {mutual['v'][1]:.7f} < "
        f"{PER_OP_PCC_THRESHOLD}. V is never rotated, so only layer 0's attention output can move it."
    )
    for name in ("k", "v"):
        for layer, step in gated[name].items():
            assert step <= MAX_LAYER_STEP, (
                f"mutual {name.upper()}: the per-layer error step at L{layer} is {step:.2f}x "
                f"(budget {MAX_LAYER_STEP}x) — a step is one layer's logic error, not depth "
                f"accumulation"
            )
    for arm, data in arms.items():
        worst_k = min(data["vs_golden"]["k"].values())
        worst_v = min(data["vs_golden"]["v"].values())
        assert worst_k >= GOLDEN_K_THRESHOLD, f"{arm} vs golden: worst layer K {worst_k:.7f} < {GOLDEN_K_THRESHOLD}"
        assert worst_v >= GOLDEN_V_THRESHOLD, f"{arm} vs golden: worst layer V {worst_v:.7f} < {GOLDEN_V_THRESHOLD}"


@requires_galaxy
@requires_ring_fabric
@requires_hf_reference
@requires_golden_trace
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_ring_prefix_control_collapses_past_layer_zero(mesh_device, reset_seeds):
    """**The control.** Serve chunk 1 against a cache whose chunk-0 prefix was never written.

    This breaks the one thing delta 3 *is* — "chunk *k*'s queries attend the prefix **read back out
    of the cache**" — and it has a signature the gate's whole threshold design turns on:

    * **layer 0 must not move.** Its K and V are produced by the projections and RoPE before any
      attention runs, so a delta-3 bug is invisible there. A control that only checked layer 0
      would pass on a completely broken ring read.
    * **layers 1+ must collapse**, because layer 0's attention output is wrong and every later
      layer's input inherits it.

    Only the positions chunk 1 wrote — global `[512, 1024)` — are compared, since the control never
    wrote the others. Six layers, because the control is about the cache read and not about depth.
    """
    metadata = _golden_metadata()
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    sp = GALAXY_MESH_SHAPE[0]
    n_layers = 6

    args = ModelArgs(mesh_device, max_seq_len=ONE_SHOT_CHUNK)
    hf = args.hf_config
    head_dim = derive_head_dim(hf)
    state_dict = ModelArgs.load_state_dict(args.model_path)
    runtime = build_runtime(
        mesh_device,
        hf,
        state_dict,
        num_layers=n_layers,
        chunk_global=RING_CHUNK,
        total=ONE_SHOT_CHUNK,
        cache_path=args.weight_cache_path(ttnn.bfloat8_b),
    )
    del state_dict

    def _serve(chunks):
        kv_cache = allocate_engine_cache(mesh_device, num_layers=n_layers, total=ONE_SHOT_CHUNK, head_dim=head_dim)
        for chunk in chunks:
            start = chunk * RING_CHUNK
            runtime.prefill_chunk(
                runtime.make_chunk_input(token_ids[start : start + RING_CHUNK], RING_CHUNK),
                kv_cache,
                slot_id=0,
                actual_start=start,
                actual_end=min(start + RING_CHUNK, n_tokens),
                chunk_size=RING_CHUNK,
            )
        ttnn.synchronize_device(mesh_device)
        read_back = read_kv_cache(
            kv_cache,
            num_layers=n_layers,
            n_kv=hf["num_key_value_heads"],
            head_dim=head_dim,
            chunk_global=RING_CHUNK,
            total=ONE_SHOT_CHUNK,
            n_tokens=n_tokens,
            mesh_shape=GALAXY_MESH_SHAPE,
        )
        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)
        return read_back

    correct = _serve([0, 1])
    control = _serve([1])  # chunk 1 only: the prefix it attends was never written

    pccs = {}
    for layer in range(n_layers):
        lo = RING_CHUNK
        _, pcc = comp_pcc(correct[layer][0][:, :, lo:, :], control[layer][0][:, :, lo:, :], 0.0)
        pccs[layer] = float(pcc)
        logger.info(
            f"[G-CHUNK-ATTN] control L{layer}: chunk-1 K with an unwritten prefix scores "
            f"PCC {float(pcc):.7f} against the correctly served run"
        )
    logger.info(
        f"[G-CHUNK-ATTN] control signature: L0 = {pccs[0]:.7f} (must NOT move — no attention has "
        f"run), L1 = {pccs[1]:.5f}, worst = {min(pccs.values()):.5f} (threshold "
        f"{PER_OP_PCC_THRESHOLD}). A layer-0-only check would have passed on a completely broken "
        f"ring read, which is why this gate's per-op threshold is at layer 1."
    )
    assert pccs[0] >= PER_OP_PCC_THRESHOLD, (
        f"layer 0's K moved to {pccs[0]:.7f} when only the prefix was blanked. Layer 0's K/V are "
        f"produced before any attention, so either the control perturbed more than the cache read "
        f"or the write offset is entangled with it."
    )
    assert min(pccs[layer] for layer in range(1, n_layers)) < PER_OP_PCC_THRESHOLD, (
        f"blanking the attended prefix left every layer above {PER_OP_PCC_THRESHOLD} "
        f"({ {k: round(v, 5) for k, v in pccs.items()} }) — then this gate is not measuring "
        f"whether chunk 1 reads the prefix out of the cache at all"
    )
