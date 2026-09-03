# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-CHUNK-ATTN` — chunk *k*'s queries attending the prefix **read back out of the cache**.

This is the third of chunked prefill that P7 recorded `BLOCKED` (`R-028`): it needs a
chunk-position-aware attention core, which `tt/attention/dense_sp.py` now provides. Two device paths
run the same 512 tokens through the same 32-layer model on the same `(4,8)` mesh, and differ in
exactly one thing — the attention core:

| path | chunks | `max_seq_len` vs `chunk` | attention core taken |
|---|---|---|---|
| A | 1 x 512 | equal | the SP **bootstrap**: all-gather Q/K/V on the SP axis -> plain causal SDPA over the whole sequence -> reduce-scatter (`DEC-021`) |
| B | 2 x 256 | `512 > 256` | the **ring cache-read** (`dense_sp_attention`) on *both* chunks, chunk 0 included |

Path A never reads the cache; path B reads it on every chunk and, on chunk 1, attends a prefix it can
only obtain from the cache across the SP ring. So the comparison is a device-vs-device test of the
attention core with the weights, the input, the mesh, the dtypes and the RoPE convention all held
fixed — the sharper form the recipe asks for (`BRINGUP_RECIPE.md:850`), and it does not inherit the
golden's own error.

**What the per-layer numbers mean, and which one is the gate.** The comparison is on the KV product,
because that is what `gather_layer` can reconstruct in natural token order across the block-cyclic SP
shards. Read the depth axis carefully, because the three regions mean different things:

* **layer 0** — a function of the embedding alone; no attention has run. A mismatch here would be the
  RoPE offset or the cache write, not the ring. Measured **1.00000 / 1.00000**.
* **layer 1** — the first product that depends on any attention, and therefore the one carrying
  **exactly one** attention layer's worth of ring-vs-bootstrap difference. This is what the recipe's
  ">= 0.999 on the attention OUTPUT" is about, and it is what this file asserts (`DEC-085`).
* **layers 2-31** — the same difference compounded through 2..31 residual streams. Recorded in full,
  and gated by the two instruments that are meaningful at depth: the per-layer error **step**
  (unchanged from `DEC-047`/`DEC-060` at 4.0x from layer 3) and both paths' PCC against the fp32
  golden at `G-CHUNK`'s thresholds. Holding an accumulated quantity to a per-op threshold would be
  measuring depth, not the op — and the accumulation is not free: it is the measured cost of the ring
  op's mandatory `fp32_dest_acc_en=False` (`DEC-084`, `G-SP-RING`).

**Input distribution:** none — the real tokenized prompt from the golden trace, the real checkpoint.
**Reference dtype policy:** for the mutual number, none — both sides are the same device code at the
same dtypes. For the golden numbers, the fp32 trace, which shares none of the device's rounding.
**Negative control:** a third path that runs chunk 1 with `cached_len = 0`, i.e. attending and roping
as though it were the first chunk. It must collapse against the golden.

Mesh: the full `(4,8)` galaxy (`DEC-080`), `Topology.Ring`, `num_links = 2`.

Run::

    export PREFILL_TRACE_DIR=/home/mstojkovic/llama31_8b_golden/p7_s512
    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    pytest models/demos/llama31_8b_d_p/tests/unit/test_sp_attention_chunked.py -x -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.scripts.verify_golden_kv import hf_to_meta_lane_permutation
from models.demos.llama31_8b_d_p.tests.test_factory import (
    TestFactory,
    err_ratio,
    llama_config_dims,
    parametrize_galaxy_submeshes,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
from models.demos.llama31_8b_d_p.tt.rope import build_indexed_rope
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

# `bringup_log/06_GATES.md:29`, the row P7 opened for this gate: ">= 0.999 chunked == one-shot **on
# the attention OUTPUT**". (G-CHUNK-ATTN is NOT in BRINGUP_RECIPE.md Appendix A — P7 created it from
# Appendix F.10's coverage-hole list, so the ledger row is its only definition.) Applied where that
# sentence is true — see `DEC-085`. Layer 1's K/V is the first product
# that depends on any attention at all, so it carries exactly ONE attention layer's worth of
# ring-vs-bootstrap difference. Layer 31's carries thirty-one, amplified through 31 residual streams,
# and holding an accumulated quantity to a per-op threshold measures depth, not the op.
MUTUAL_PCC = 0.999
# The accumulated curve is gated two other ways instead, both pre-existing and neither chosen after
# seeing this measurement: the per-layer error STEP (`DEC-047`/`DEC-060`'s instrument, unchanged at
# 4.0x from layer 3) localises a single-layer bug, and both paths are scored against the fp32 golden
# at `G-CHUNK`'s own thresholds.
MAX_MUTUAL_ERROR_STEP = 4.0
STEP_CHECK_FROM_LAYER = 3
# G-CHUNK's golden thresholds, carried over verbatim (same product, same golden).
GOLDEN_PCC_K = 0.99
GOLDEN_PCC_V = 0.98
NEGATIVE_CONTROL_MAX_PCC = 0.90
MAX_LAYER0_ERR_RATIO = 3.0

SEQ_LEN = 512
CHUNK_ONE_SHOT = 512
CHUNK_CHUNKED = 256

_TRACE_ENV = "PREFILL_TRACE_DIR"


def _trace_dir():
    raw = os.environ.get(_TRACE_ENV)
    if not raw:
        pytest.skip(f"${_TRACE_ENV} is unset; generate a golden with scripts/generate_golden_kv_cache.py")
    path = Path(raw)
    if not (path / "metadata.json").exists():
        pytest.skip(f"${_TRACE_ENV}={path} has no metadata.json")
    return path


def _stub_runtime(mesh_device, hf_config, kv_cache, *, num_layers, chunk_size):
    """A `TtPrefillRuntime` carrying only what `gather_layer` reads (`tests/unit/test_kv_cache_tp8.py`).

    `gather_layer` inverts the block-cyclic SP layout and stacks the per-column KV heads, which is
    exactly the read-back this gate needs and is code the gate should be exercising anyway (`R-029`).
    """
    config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=SEQ_LEN,
        mesh_shape=tuple(mesh_device.shape),
        default_chunk_size=chunk_size,
        num_users=1,
        sequence_parallel=True,
    )
    stub = TtPrefillRuntime.__new__(TtPrefillRuntime)
    stub.config = config
    stub.hf_config = hf_config
    stub.kv_cache = kv_cache
    return stub


@parametrize_galaxy_submeshes([(4, 8)])
@requires_hf_reference
@torch.no_grad()
def test_chunked_ring_attention_equals_one_shot(  # noqa: C901
    mesh_device, device_params, submesh_shape, state_dict, reset_seeds
):
    """`G-CHUNK-ATTN`: the ring cache-read core reproduces the one-shot core, layer by layer."""
    trace_dir = _trace_dir()
    with open(trace_dir / "metadata.json") as handle:
        metadata = json.load(handle)
    token_ids = list(metadata["token_ids"])[:SEQ_LEN]
    assert len(token_ids) == SEQ_LEN
    if not state_dict:
        pytest.skip("no real checkpoint loaded; G-CHUNK-ATTN is a real-weight gate")

    hf_config = llama_hf_config(llama_config_dims())
    n_kv, head_dim = hf_config.num_key_value_heads, hf_config.head_dim
    num_layers = min(int(metadata["num_layers"]), hf_config.num_hidden_layers)

    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    mesh = objs["mesh_device"]
    sp, tp = tuple(mesh.shape)
    assert (sp, tp) == (4, 8) and objs["ccl_manager"].num_links == 2, (
        f"G-CHUNK-ATTN is the target-mesh gate: expected (4,8) with 2 links, got {(sp, tp)} with "
        f"{objs['ccl_manager'].num_links}"
    )

    tt_model = Model(
        mesh,
        hf_config,
        state_dict,
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=SEQ_LEN,
        num_layers=num_layers,
        sequence_parallel=True,
        with_lm_head=False,
    )
    rope = {
        size: build_indexed_rope(mesh, hf_config, max_seq_len=SEQ_LEN, chunk_size=size, sp_axis=0)
        for size in (CHUNK_ONE_SHOT, CHUNK_CHUNKED)
    }

    def _fresh_cache():
        return allocate_kv_cache(
            mesh,
            num_layers=num_layers,
            max_seq_len=SEQ_LEN,
            sp_axis=0,
            num_users=1,
            head_dim=head_dim,
            cache_dtype=ttnn.bfloat8_b,
        )

    def _prefill(cache, *, chunk_size, chunk_index, cached_len):
        lo = chunk_index * chunk_size
        tokens_embd, rot_mats, _ = tt_model.prepare_inputs_prefill(
            torch.tensor(token_ids[lo : lo + chunk_size], dtype=torch.int32), start_pos=lo, build_rope=False
        )
        assert rot_mats is None
        out = tt_model.prefill_forward(
            tokens_embd,
            rot_mats_global=rope[chunk_size],
            kv_cache=cache,
            cached_len=cached_len,
            user_id=0,
            skip_lm_head=True,
            indexed_rope=True,
        )
        out.deallocate(True)

    # --- path A: one shot. max_seq_len == chunk, so the SP bootstrap runs (DEC-021). ---
    cache_one_shot = _fresh_cache()
    _prefill(cache_one_shot, chunk_size=CHUNK_ONE_SHOT, chunk_index=0, cached_len=0)

    # --- path B: two chunks. The ring cache-read runs on BOTH, chunk 0 included. ---
    cache_chunked = _fresh_cache()
    for chunk_index in range(SEQ_LEN // CHUNK_CHUNKED):
        _prefill(
            cache_chunked,
            chunk_size=CHUNK_CHUNKED,
            chunk_index=chunk_index,
            cached_len=chunk_index * CHUNK_CHUNKED,
        )

    # --- negative control: chunk 1 run as though it were chunk 0 (prefix and RoPE both wrong). ---
    cache_control = _fresh_cache()
    _prefill(cache_control, chunk_size=CHUNK_CHUNKED, chunk_index=0, cached_len=0)
    _prefill(cache_control, chunk_size=CHUNK_CHUNKED, chunk_index=1, cached_len=0)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"[G-CHUNK-ATTN] three paths written on {(sp, tp)}: one-shot 1x{CHUNK_ONE_SHOT} (bootstrap), "
        f"chunked {SEQ_LEN // CHUNK_CHUNKED}x{CHUNK_CHUNKED} (ring cache-read), and the "
        f"cached_len=0 control"
    )

    from safetensors import safe_open

    rt_one_shot = _stub_runtime(mesh, hf_config, cache_one_shot, num_layers=num_layers, chunk_size=CHUNK_ONE_SHOT)
    rt_chunked = _stub_runtime(mesh, hf_config, cache_chunked, num_layers=num_layers, chunk_size=CHUNK_CHUNKED)
    rt_control = _stub_runtime(mesh, hf_config, cache_control, num_layers=num_layers, chunk_size=CHUNK_CHUNKED)
    perm = hf_to_meta_lane_permutation(head_dim, head_dim)

    worst = {"mutual_k": 1.0, "mutual_v": 1.0, "golden_k": 1.0, "golden_v": 1.0, "one_shot_k": 1.0}
    mutual_rows = []
    layer0 = {}
    layer1 = {}
    control_worst = 1.0
    ratio_layer0 = None
    for layer_idx in range(num_layers):
        k_one, v_one = rt_one_shot.gather_layer(slot_id=0, layer_idx=layer_idx, n_tokens=SEQ_LEN)
        k_chunk, v_chunk = rt_chunked.gather_layer(slot_id=0, layer_idx=layer_idx, n_tokens=SEQ_LEN)
        k_bug, _ = rt_control.gather_layer(slot_id=0, layer_idx=layer_idx, n_tokens=SEQ_LEN)
        with safe_open(str(trace_dir / "kv_cache" / f"layer_{layer_idx}.safetensors"), framework="pt") as handle:
            golden_k = handle.get_tensor(f"key_cache_layer_{layer_idx}").float()[:, :, :SEQ_LEN, :][..., perm]
            golden_v = handle.get_tensor(f"value_cache_layer_{layer_idx}").float()[:, :, :SEQ_LEN, :]
        assert tuple(k_one.shape) == (1, n_kv, SEQ_LEN, head_dim)

        _, mutual_k = comp_pcc(k_one, k_chunk, 0.0)
        _, mutual_v = comp_pcc(v_one, v_chunk, 0.0)
        _, golden_k_pcc = comp_pcc(golden_k, k_chunk, 0.0)
        _, golden_v_pcc = comp_pcc(golden_v, v_chunk, 0.0)
        _, one_shot_k_pcc = comp_pcc(golden_k, k_one, 0.0)
        _, control_pcc = comp_pcc(golden_k, k_bug, 0.0)
        control_worst = min(control_worst, float(control_pcc))

        mutual_rows.append((layer_idx, float(mutual_k), float(mutual_v)))
        if layer_idx == 1:
            # ONE attention layer's worth of difference: this is the gate's own quantity.
            layer1 = {"mutual_k": float(mutual_k), "mutual_v": float(mutual_v)}
        if layer_idx == 0:
            # Attention-independent: layer 0's K/V comes from the embedding. Recorded apart so the
            # attention claim below rests only on layers that actually ran attention.
            layer0 = {"mutual_k": float(mutual_k), "mutual_v": float(mutual_v)}
            _, floor_k = comp_pcc(golden_k, quantize_like_device(golden_k, ttnn.bfloat8_b), 0.0)
            ratio_layer0 = err_ratio(golden_k_pcc, floor_k)
        else:
            worst["mutual_k"] = min(worst["mutual_k"], float(mutual_k))
            worst["mutual_v"] = min(worst["mutual_v"], float(mutual_v))
        worst["golden_k"] = min(worst["golden_k"], float(golden_k_pcc))
        worst["golden_v"] = min(worst["golden_v"], float(golden_v_pcc))
        worst["one_shot_k"] = min(worst["one_shot_k"], float(one_shot_k_pcc))
        logger.info(
            f"[G-CHUNK-ATTN] L{layer_idx:>2}: ring-vs-bootstrap K={float(mutual_k):.5f} "
            f"V={float(mutual_v):.5f} | vs golden (ring) K={float(golden_k_pcc):.5f} "
            f"V={float(golden_v_pcc):.5f} (bootstrap) K={float(one_shot_k_pcc):.5f} | "
            f"cached_len=0 control K={float(control_pcc):.5f}"
        )

    # The step curve: consecutive ratios of (1 - mutual PCC). A *step* is a single-layer bug; smooth
    # growth is accumulation. Same statistic and same 4.0x ceiling as `G-CHUNK` / `DEC-060`.
    steps_k = [
        (mutual_rows[i][0], (1 - mutual_rows[i][1]) / max(1 - mutual_rows[i - 1][1], 1e-12))
        for i in range(1, len(mutual_rows))
    ]
    checked = [(i, st) for i, st in steps_k if i >= STEP_CHECK_FROM_LAYER]
    max_step, at_layer = max(checked, key=lambda t: t[1])[::-1]

    logger.info(
        f"[G-CHUNK-ATTN] ring-vs-bootstrap divergence is ACCUMULATION, not a step: max consecutive "
        f"error ratio {max_step:.2f}x at L{at_layer} (ceiling {MAX_MUTUAL_ERROR_STEP}x, checked from "
        f"L{STEP_CHECK_FROM_LAYER}); excluded early steps "
        f"{[(i, round(st, 2)) for i, st in steps_k if i < STEP_CHECK_FROM_LAYER]}"
    )
    logger.info(
        f"[G-CHUNK-ATTN] {num_layers} layers on {(sp, tp)}: layer 0 (attention-independent) "
        f"K={layer0['mutual_k']:.5f} V={layer0['mutual_v']:.5f}; layer 1 (ONE attention layer, the "
        f"gate's quantity) K={layer1['mutual_k']:.5f} V={layer1['mutual_v']:.5f} (threshold "
        f"{MUTUAL_PCC}); layers 1-{num_layers - 1} accumulated min K={worst['mutual_k']:.5f} "
        f"min V={worst['mutual_v']:.5f} (recorded, not gated — DEC-085) | vs golden: ring min K={worst['golden_k']:.5f} "
        f"V={worst['golden_v']:.5f}, bootstrap min K={worst['one_shot_k']:.5f} | layer-0 "
        f"err_ratio={ratio_layer0:.2f}x of the bf8_b storage floor | control worst K="
        f"{control_worst:.5f}"
    )

    assert layer0["mutual_k"] >= MUTUAL_PCC and layer0["mutual_v"] >= MUTUAL_PCC, (
        f"[G-CHUNK-ATTN] layer 0 already differs between the paths (K={layer0['mutual_k']:.5f}, "
        f"V={layer0['mutual_v']:.5f}). Layer 0's K/V does not depend on attention at all, so this is "
        f"the indexed RoPE offset or the cache write, not the ring."
    )
    assert layer1["mutual_k"] >= MUTUAL_PCC and layer1["mutual_v"] >= MUTUAL_PCC, (
        f"[G-CHUNK-ATTN] ONE attention layer already disagrees between the ring cache-read and the "
        f"one-shot core: layer 1 K {layer1['mutual_k']:.5f}, V {layer1['mutual_v']:.5f} < "
        f"{MUTUAL_PCC}. Both paths saw the same tokens, weights and mesh; the only difference is that "
        f"path B's queries attend a prefix read back out of the cache. This is the gate's own "
        f"quantity (DEC-085) — a failure here is the ring op or its arguments, not accumulation."
    )
    assert max_step <= MAX_MUTUAL_ERROR_STEP, (
        f"[G-CHUNK-ATTN] one layer multiplies the ring-vs-bootstrap divergence by {max_step:.2f}x at "
        f"L{at_layer} (ceiling {MAX_MUTUAL_ERROR_STEP}x). Smooth growth is the fp32-accumulator cost "
        f"compounding through the residual stream (DEC-084); a step is a per-layer bug — chase the "
        f"layer, not the depth."
    )
    assert worst["golden_k"] >= GOLDEN_PCC_K, f"[G-CHUNK-ATTN] ring K vs golden: {worst['golden_k']:.5f}"
    assert worst["golden_v"] >= GOLDEN_PCC_V, f"[G-CHUNK-ATTN] ring V vs golden: {worst['golden_v']:.5f}"
    assert worst["one_shot_k"] >= GOLDEN_PCC_K, f"[G-CHUNK-ATTN] bootstrap K vs golden: {worst['one_shot_k']:.5f}"
    assert ratio_layer0 <= MAX_LAYER0_ERR_RATIO, (
        f"[G-CHUNK-ATTN] layer 0's K sits {ratio_layer0:.2f}x off the bf8_b storage floor "
        f"(ceiling {MAX_LAYER0_ERR_RATIO}x); layer 0's input is exact, so nothing upstream explains it"
    )
    assert control_worst <= NEGATIVE_CONTROL_MAX_PCC, (
        f"[G-CHUNK-ATTN] NEGATIVE CONTROL FAILED: running chunk 1 with cached_len=0 — wrong prefix "
        f"AND wrong RoPE — still scores K PCC {control_worst:.5f} > {NEGATIVE_CONTROL_MAX_PCC}. This "
        f"gate cannot tell a correct chunk offset from a wrong one."
    )


# =====================================================================================
# G-SP-RING — the ring op ALONE against fp32 torch, and what `fp32_dest_acc_en=False` costs
# =====================================================================================
def _torch_causal_attention(q, k, v, *, q_offset, scale, gqa_group):
    """fp32 reference: Q rows at global positions `q_offset + i` attend K/V rows `0..q_offset+i`.

    `q` is `[1, n_heads, q_len, hd]`, `k`/`v` are `[1, n_kv, total, hd]`. Q head `h` reads KV head
    `h // gqa_group` — the same GQA grouping `ttnn.transformer` applies internally, written out here
    because a reference that repeated K/V the wrong way would agree with a wrong device answer.
    """
    n_heads, q_len = q.shape[1], q.shape[2]
    total = k.shape[2]
    positions = torch.arange(q_len).unsqueeze(1) + q_offset
    mask = (torch.arange(total).unsqueeze(0) <= positions).unsqueeze(0)  # [1, q_len, total]
    out = torch.empty_like(q)
    for head in range(n_heads):
        kv_head = head // gqa_group
        scores = (q[0, head] @ k[0, kv_head].transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask[0], float("-inf"))
        out[0, head] = torch.softmax(scores, dim=-1) @ v[0, kv_head]
    return out


@parametrize_galaxy_submeshes([(4, 8)])
@torch.no_grad()
def test_ring_sdpa_alone_vs_torch_and_the_fp32_accumulator_cost(mesh_device, device_params, submesh_shape, reset_seeds):
    """`G-SP-RING`: `dense_sp_attention` on synthetic tensors, against an fp32 torch reference.

    `G-CHUNK-ATTN` above compares the ring against another *device* path, which is the sharper test
    of the two but says nothing in absolute terms. This one isolates the op: random Q, a cache
    written through `write_kv_chunk`, and a torch reference that repeats the GQA group explicitly.

    It also answers the question `DEC-031` leaves open for this one path. Every other op in the
    package runs `fp32_dest_acc_en=True`, measured worth two to three orders of magnitude on
    matmuls (Appendix E.4); the ring op is documented as requiring `False`
    (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:200`). So the ring runs **twice** here — once
    with the package's ring config (`False`) and once with `True` forced — and the gate records both,
    plus the bf8_b storage floor. That turns "the SP path scores worse" from an unexplained
    regression into a named, measured cost (`DEC-084`).

    **Input distribution:** `randn * 0.5` for Q/K/V, seed 0 — synthetic on purpose, so the only thing
    under test is the op.
    **Reference dtype policy:** fp32 torch on the **bf8_b-quantised** K/V (what the cache actually
    holds) and bf16-quantised Q; that is the noise floor, and the measured PCC is reported against it.
    """
    from models.demos.llama31_8b_d_p.tt.attention.config import ProgramConfig
    from models.demos.llama31_8b_d_p.tt.attention.dense_sp import dense_sp_attention
    from models.demos.llama31_8b_d_p.tt.attention.kv_cache import write_kv_chunk

    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    mesh = objs["mesh_device"]
    sp, tp = tuple(mesh.shape)
    dims = llama_config_dims()
    n_heads, n_kv, head_dim = dims["num_attention_heads"], dims["num_key_value_heads"], dims["head_dim"]
    gqa_group = n_heads // n_kv
    scale = head_dim**-0.5

    total, chunk = 512, 256
    chunk_local = chunk // sp
    generator = torch.Generator().manual_seed(0)

    host_k = torch.randn(1, n_kv, total, head_dim, generator=generator) * 0.5
    host_v = torch.randn(1, n_kv, total, head_dim, generator=generator) * 0.5
    host_q = torch.randn(1, n_heads, chunk, head_dim, generator=generator) * 0.5

    def _shard(host):
        # seq on the SP rows, heads on the TP cols — the layout every module produces.
        return ttnn.from_torch(
            host,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(2, 1)),
        )

    kv_cache = allocate_kv_cache(
        mesh, num_layers=1, max_seq_len=total, sp_axis=0, num_users=1, head_dim=head_dim, cache_dtype=ttnn.bfloat8_b
    )
    for chunk_index in range(total // chunk):
        lo = chunk_index * chunk
        tt_k = _shard(host_k[:, :, lo : lo + chunk])
        tt_v = _shard(host_v[:, :, lo : lo + chunk])
        write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=lo, sp_axis=0)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    ttnn.synchronize_device(mesh)

    program_config = ProgramConfig()
    kv_actual = total - chunk  # the last chunk's Q attends the whole 512-token prefix

    def _run(compute_kernel_config):
        tt_q = _shard(host_q)
        out = dense_sp_attention(
            tt_q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=total,
            n_kv=n_kv,
            cache_global=total,
            head_dim=head_dim,
            mesh_device=mesh,
            ccl_manager=objs["ccl_manager"],
            program_config=program_config,
            scale=scale,
            cluster_axis=0,
            compute_kernel_config=compute_kernel_config,
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            write_chunk=False,
        )
        ttnn.synchronize_device(mesh)
        device_tensors = ttnn.get_device_tensors(out)
        # [1, 4, chunk_local, hd] per chip -> [1, 32, chunk, hd] natural order: col c holds Q heads
        # [4c, 4c+4), row r holds this chunk's rows [r*chunk_local, (r+1)*chunk_local).
        host = torch.zeros(1, n_heads, chunk, head_dim)
        for row in range(sp):
            for col in range(tp):
                shard = ttnn.to_torch(device_tensors[row * tp + col]).float()
                lo_head, lo_row = col * (n_heads // tp), row * chunk_local
                host[0, lo_head : lo_head + n_heads // tp, lo_row : lo_row + chunk_local] = shard[0]
        out.deallocate(True)
        return host

    measured_false = _run(None)  # the package's ring config: fp32_dest_acc_en=False
    forced_true = None
    forced_error = None
    try:
        forced_true = _run(program_config.get_compute_kernel_config(mesh))  # fp32_dest_acc_en=True
    except Exception as exc:  # noqa: BLE001 - the point is to record WHICH failure, if any
        forced_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    # The floor: exactly the values the device holds (bf8_b K/V, bf16 Q), all arithmetic in fp32.
    q_ref = quantize_like_device(host_q, ttnn.bfloat16)
    k_ref = quantize_like_device(host_k, ttnn.bfloat8_b)
    v_ref = quantize_like_device(host_v, ttnn.bfloat8_b)
    reference = _torch_causal_attention(host_q, host_k, host_v, q_offset=kv_actual, scale=scale, gqa_group=gqa_group)
    floor_out = _torch_causal_attention(q_ref, k_ref, v_ref, q_offset=kv_actual, scale=scale, gqa_group=gqa_group)
    _, floor_pcc = comp_pcc(reference, floor_out, 0.0)
    _, pcc_false = comp_pcc(reference, measured_false, 0.0)
    ratio_false = err_ratio(pcc_false, floor_pcc)

    if forced_error is None:
        _, pcc_true = comp_pcc(reference, forced_true, 0.0)
        ratio_true = err_ratio(pcc_true, floor_pcc)
        _, mutual = comp_pcc(measured_false, forced_true, 0.0)
        verdict = (
            f"fp32_dest_acc_en=True RAN: PCC {float(pcc_true):.6f} ({ratio_true:.1f}x floor); "
            f"the two configs agree to {float(mutual):.6f}. Cost of the required False = "
            f"{(1 - float(pcc_false)) / max(1 - float(pcc_true), 1e-12):.2f}x the error"
        )
    else:
        verdict = f"fp32_dest_acc_en=True REFUSED by the op: {forced_error}"

    # Negative control: the same call with `logical_n` cut to the first 256 tokens, i.e. the ring is
    # told the prefix ends where it begins. Legal arguments, plausible output, wrong answer — which
    # is the entire failure mode this op has (there is no shape error to catch it), so a gate that
    # could not detect it would be measuring nothing.
    tt_q_control = _shard(host_q)
    control_out = dense_sp_attention(
        tt_q_control,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        kv_actual=0,
        logical_n=chunk,
        n_kv=n_kv,
        cache_global=total,
        head_dim=head_dim,
        mesh_device=mesh,
        ccl_manager=objs["ccl_manager"],
        program_config=program_config,
        scale=scale,
        cluster_axis=0,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        write_chunk=False,
    )
    ttnn.synchronize_device(mesh)
    control_tensors = ttnn.get_device_tensors(control_out)
    control_host = torch.zeros(1, n_heads, chunk, head_dim)
    for row in range(sp):
        for col in range(tp):
            shard = ttnn.to_torch(control_tensors[row * tp + col]).float()
            lo_head, lo_row = col * (n_heads // tp), row * chunk_local
            control_host[0, lo_head : lo_head + n_heads // tp, lo_row : lo_row + chunk_local] = shard[0]
    control_out.deallocate(True)
    _, control_pcc = comp_pcc(reference, control_host, 0.0)

    logger.info(
        f"[G-SP-RING] negative control: the same call with logical_n={chunk} instead of {total} "
        f"(the ring told the prefix ends where the chunk begins) scores {float(control_pcc):.6f} "
        f"against the same fp32 reference, vs {float(pcc_false):.6f} for the correct call"
    )
    assert float(control_pcc) <= 0.99, (
        f"[G-SP-RING] NEGATIVE CONTROL FAILED: halving logical_n still scores {float(control_pcc):.6f}. "
        f"This gate cannot tell a correct prefix length from a wrong one, so its PASS means nothing."
    )

    logger.info(
        f"[G-SP-RING] ring_joint SDPA alone on {(sp, tp)}: Q {tuple(host_q.shape)} at offset "
        f"{kv_actual}, cache {total} tokens bf8_b, GQA {n_heads}/{n_kv}. "
        f"fp32_dest_acc_en=False (the package's ring config): PCC {float(pcc_false):.6f}; "
        f"bf8_b K/V + bf16 Q noise floor {float(floor_pcc):.6f}; err_ratio {ratio_false:.2f}x. {verdict}"
    )
    # The op is a fused kernel, so Appendix E.5 applies: the floor does not model its interior and a
    # ratio above 1 is expected. The gate is the absolute PCC plus the named A/B above.
    assert float(pcc_false) >= 0.99, (
        f"[G-SP-RING] the ring cache-read scores {float(pcc_false):.6f} against fp32 torch on the "
        f"same values (floor {float(floor_pcc):.6f}). This is the op alone — no model, no weights — "
        f"so a failure here is the call arguments: kv_actual_isl, logical_n, kv_cache_batch_idx, the "
        f"gather-buffer extent, or the block-cyclic Q layout."
    )
