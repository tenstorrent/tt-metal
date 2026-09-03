# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-KV-TP8` — the **model -> cache** path at TP=8, which closes `R-027`'s coverage hole.

`G-KV` (P5.6) and `G-CHUNK` (P7) both ran on `(1,1)` with `nkv = tp = 1`. That is the head count a
chip holds at TP=8, so the layout they pinned is the deployment layout — but it is **not a head count
the model produces on a `(1,1)` mesh**, where attention makes all 8 KV heads locally and
`update_padded_kv_cache` dies with `TT_FATAL: cache and input num-heads dim must match`
(`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
So the *cache primitive* was gated and the *model-to-cache interface* was not: what remained unproven
is the mesh-mapper step that puts KV head `c` on mesh column `c` (`Appendix F.6`, corrected by P7).

This file proves that step at **TP=8 with no sequence parallelism** — deliberately the cheapest
configuration that can produce it, so a mesh-mapper bug cannot arrive tangled up with an SP ring bug
(`R-027`'s recommended first P8 step). Two tests, in increasing cost:

1. `test_kv_head_to_column_mapping_is_the_identity` — no model, no weights, ~seconds. Writes a
   position/head-labelled tensor through the model's own mesh mapper and asserts, **bit-exactly**,
   that column `c` holds head `c`. Its negative control reads the columns rotated by one and requires
   the comparison to fail, so "column `c` holds head `c`" is a claim this test can actually refute.
2. `test_model_produced_kv_at_tp8_matches_golden` — the real 32-layer model on the real checkpoint,
   writing the real cache through `Model.prefill_forward`, read back through the **runtime's own**
   `gather_layer` / `dump_slot_kv` / `kv_cache_pcc_check` (`R-029`: their first execution on device;
   until now their format contract was only asserted from source text) and scored against the fp32
   golden trace.

**Mesh.** `(1,8)` carved as a submesh of the open `(4,8)` galaxy — a top-level `(1,8)` cannot bring
the fabric up on this machine (`DEC-080`), and `Topology.Ring` is used throughout (`DEC-081`).
`sp = 1`, so the block-cyclic sequence layout degenerates to the identity and every device holds the
whole sequence; the only thing distributed here is the head/feature dimension. That is the point.

**Input distribution:** none — the input is the real tokenized prompt from the golden trace and the
weights are the real checkpoint (Appendix E.1's strongest case: nothing to choose).
**Reference dtype policy:** the golden is `transformers`' own fp32 math on the checkpoint's bf16
weights upcast exactly, stored fp32 (`scripts/generate_golden_kv_cache.py`). The device runs bf8_b
weights, bf16 activations and a bf8_b cache, so the reference shares none of the device's rounding.
**Noise floor** (Appendix E.2): the golden K/V through ttnn's own bf8_b quantiser — the cache dtype
is the whole *storage* budget — reported per layer as `err_ratio` and asserted at layer 0, where the
input is the exact embedding and nothing upstream can explain a gap (Appendix E.5 / `DEC-058`).

Run::

    export PREFILL_TRACE_DIR=/home/mstojkovic/llama31_8b_golden/p7_s512
    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_tp8.py -x -q
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
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.llama31_8b_d_p.scripts.verify_golden_kv import compare_device_dump, hf_to_meta_lane_permutation
from models.demos.llama31_8b_d_p.tests.test_factory import (
    TestFactory,
    err_ratio,
    llama_config_dims,
    parametrize_galaxy_submeshes,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
from models.demos.llama31_8b_d_p.tt.rope import build_indexed_rope
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

# G-CHUNK's thresholds, carried over verbatim rather than re-chosen: this gate scores the same
# product (post-RoPE K, raw V, bf8_b cache, real weights, 32 layers) against the same golden, so a
# fresh number here would be unfalsifiable. `06_GATES.md` G-CHUNK measured min K 0.99818 / V 0.99206.
GOLDEN_PCC_K = 0.99
GOLDEN_PCC_V = 0.98
# A wrong head->column map must DESTROY the PCC, not dent it.
NEGATIVE_CONTROL_MAX_PCC = 0.90
# Appendix E.2 ceiling, asserted at layer 0 only (DEC-058's reasoning, unchanged).
MAX_LAYER0_ERR_RATIO = 3.0

SEQ_LEN = 512
CHUNK = 512
# The cache is sized to TWO chunks so there is an unwritten pad tail to assert on, and two users so
# there is a neighbouring slot to assert on. Both are collateral-write checks a PCC cannot see.
MAX_SEQ_LEN = 2 * CHUNK
NUM_USERS = 2

_TRACE_ENV = "PREFILL_TRACE_DIR"


def _trace_dir():
    raw = os.environ.get(_TRACE_ENV)
    if not raw:
        pytest.skip(f"${_TRACE_ENV} is unset; generate a golden with scripts/generate_golden_kv_cache.py")
    path = Path(raw)
    if not (path / "metadata.json").exists():
        pytest.skip(f"${_TRACE_ENV}={path} has no metadata.json")
    return path


def _read_layer_with_col_map(mesh_device, cache_tensor, *, slot, col_map, n_tokens):
    """Read one packed-cache slot back, taking KV head `h` from mesh column `col_map[h]`.

    The identity `col_map` is what `TtPrefillRuntime.gather_layer` assumes (`r * cols + col`, with
    `col` running over the KV heads). A rotated map is the negative control: if the read scores just
    as well with the columns shuffled, the test is not measuring the mapping at all.
    """
    rows, cols = tuple(mesh_device.shape)
    device_tensors = ttnn.get_device_tensors(cache_tensor)
    positions = blockcyclic_positions(rows, CHUNK, MAX_SEQ_LEN)
    assert torch.equal(positions, torch.arange(len(positions))), (
        f"at sp={rows} the block-cyclic layout must be the identity, got a non-trivial permutation; "
        f"this reader would then be wrong (use gather_layer instead)"
    )
    heads = []
    for head in range(len(col_map)):
        rows_cat = torch.cat(
            [ttnn.to_torch(device_tensors[r * cols + col_map[head]])[slot, 0].float() for r in range(rows)],
            dim=0,
        )
        heads.append(rows_cat[:n_tokens])
    return torch.stack(heads, dim=0).unsqueeze(0)


def _stub_runtime(mesh_device, hf_config, kv_cache, *, num_layers):
    """A `TtPrefillRuntime` with only the fields the read-back helpers touch.

    `gather_layer`, `dump_slot_kv` and `kv_cache_pcc_check` need `self.config`, `self.hf_config` and
    `_resolve_kv` — not the 8 B model this test has already built for itself. Constructing the real
    runtime here would build a second copy of every weight on the same 8 devices. Same trick as
    `tests/unit/test_attention_chunked_vs_ref.py`'s refusal test, used here to *exercise* the helpers
    rather than to reach an assert.
    """
    config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=tuple(mesh_device.shape),
        default_chunk_size=CHUNK,
        num_users=NUM_USERS,
    )
    stub = TtPrefillRuntime.__new__(TtPrefillRuntime)
    stub.config = config
    stub.hf_config = hf_config
    stub.kv_cache = kv_cache
    return stub


@parametrize_galaxy_submeshes([(1, 8)])
@torch.no_grad()
def test_kv_head_to_column_mapping_is_the_identity(mesh_device, device_params, submesh_shape, reset_seeds):
    """KV head `c` lands on mesh column `c`, bit-exactly — the one step `G-KV` could not reach.

    Values are integers below 256 so `bfloat16` holds them exactly (Appendix E.6: 257 rounds to 256).
    Lanes `[0, 64)` carry the token position, lanes `[64, 128)` the head index, so a head mix-up and a
    row mix-up show up in different halves of the same tensor.
    """
    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    submesh = objs["mesh_device"]
    rows, cols = tuple(submesh.shape)
    assert (rows, cols) == (1, 8), f"this test is about TP=8 with no SP; got {(rows, cols)}"
    n_kv, head_dim, seq_len = cols, 128, 128
    max_seq_len = seq_len

    position = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
    head = torch.arange(n_kv, dtype=torch.float32).reshape(n_kv, 1, 1)
    half = head_dim // 2
    sent = torch.cat([position.repeat(n_kv, 1, half), head.repeat(1, seq_len, half)], dim=-1)
    assert sent.max().item() < 256, "bfloat16 is exact only up to 256 (Appendix E.6)"

    kv_cache = allocate_kv_cache(
        submesh,
        num_layers=1,
        max_seq_len=max_seq_len,
        sp_axis=0,
        num_users=1,
        head_dim=head_dim,
        cache_dtype=ttnn.bfloat16,
    )
    # The model's own mapper: heads on the TP cols, sequence replicated across the (single) SP row.
    # Per chip this is [1, 1, seq, head_dim] — exactly one KV head, which is all the cache slot holds.
    tt_chunk = ttnn.from_torch(
        sent.reshape(1, n_kv, seq_len, head_dim),
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=(rows, cols), dims=(None, 1)),
    )
    per_chip = tuple(ttnn.get_device_tensors(tt_chunk)[0].shape)
    assert per_chip == (1, 1, seq_len, head_dim), (
        f"the mesh mapper gave each chip {per_chip}, not one KV head; write_kv_chunk would then "
        f"silently write only head 0 (tt/attention/kv_cache.py:181)"
    )
    write_kv_chunk(kv_cache, tt_chunk, tt_chunk, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=0)
    tt_chunk.deallocate(True)
    ttnn.synchronize_device(submesh)

    identity = list(range(n_kv))
    got = _read_layer_with_col_map(submesh, kv_cache.k, slot=0, col_map=identity, n_tokens=seq_len)
    torch.testing.assert_close(got[0], sent, rtol=0.0, atol=0.0)
    logger.info(
        f"[G-KV-TP8] head -> column map is the IDENTITY, bit-exactly (rtol=atol=0): {n_kv} heads x "
        f"{seq_len} positions x head_dim {head_dim} on mesh {(rows, cols)}, one head per chip"
    )

    # Negative control: rotate the column map by one. The discriminator here is **bit-equality**,
    # not PCC: by construction lanes [0, 64) hold the position and are the same in every head, so a
    # head rotation leaves half the tensor untouched and the PCC stays high (measured ~0.9989). That
    # is exactly why this test asserts `rtol=atol=0` above rather than a correlation — and the
    # head-lane block below turns the control into a positive identification of the shift, not just
    # a "differs".
    rotated = [(c + 1) % n_kv for c in identity]
    wrong = _read_layer_with_col_map(submesh, kv_cache.k, slot=0, col_map=rotated, n_tokens=seq_len)
    _, pcc_rotated = comp_pcc(sent, wrong[0], 0.0)
    assert not torch.equal(wrong[0], sent), "rotating the head->column map changed nothing: unreadable test"
    for head in range(n_kv):
        # The head-id lane block of the rotated read must carry head (h+1)%8's id, exactly.
        torch.testing.assert_close(wrong[0][head, :, half:], sent[(head + 1) % n_kv, :, half:], rtol=0.0, atol=0.0)
    logger.info(
        f"[G-KV-TP8] negative control: reading column (c+1)%{n_kv} as head c is NOT bit-equal, and "
        f"its head-id lane block carries head (c+1)%{n_kv} exactly — the map is positively "
        f"identified, not merely 'different'. Its PCC is {float(pcc_rotated):.5f}, which is high "
        f"BY CONSTRUCTION (half the lanes are the head-independent position) and is why this test "
        f"gates on bit-equality instead"
    )


@parametrize_galaxy_submeshes([(1, 8)])
@requires_hf_reference
@torch.no_grad()
def test_model_produced_kv_at_tp8_matches_golden(  # noqa: C901
    mesh_device, device_params, submesh_shape, state_dict, reset_seeds, tmp_path
):
    """The real 32-layer model writes the real cache at TP=8; every layer's K/V matches the golden.

    This is the first time in this bring-up that `Model.prefill_forward` writes a KV cache at all
    (`R-027`: on one card it cannot), and the first time the runtime's read-back helpers run on
    device (`R-029`).
    """
    trace_dir = _trace_dir()
    with open(trace_dir / "metadata.json") as handle:
        metadata = json.load(handle)
    token_ids = list(metadata["token_ids"])[:SEQ_LEN]
    assert len(token_ids) == SEQ_LEN, f"golden has {len(metadata['token_ids'])} tokens, need >= {SEQ_LEN}"
    if not state_dict:
        pytest.skip("no real checkpoint loaded; G-KV-TP8 is a real-weight gate")

    dims = llama_config_dims()
    hf_config = llama_hf_config(dims)
    n_kv, head_dim = hf_config.num_key_value_heads, hf_config.head_dim
    num_layers = min(int(metadata["num_layers"]), hf_config.num_hidden_layers)

    objs = TestFactory.setup_submesh(mesh_device, submesh_shape)
    submesh = objs["mesh_device"]
    rows, cols = tuple(submesh.shape)
    assert cols == n_kv == 8, f"TP must equal num_key_value_heads for the packed cache (R-027); got TP={cols}"

    tt_model = Model(
        submesh,
        hf_config,
        state_dict,
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=MAX_SEQ_LEN,
        num_layers=num_layers,
        with_lm_head=False,
    )
    kv_cache = allocate_kv_cache(
        submesh,
        num_layers=num_layers,
        max_seq_len=MAX_SEQ_LEN,
        sp_axis=0,
        num_users=NUM_USERS,
        head_dim=head_dim,
        cache_dtype=ttnn.bfloat8_b,
    )
    rope_indexed = build_indexed_rope(submesh, hf_config, max_seq_len=MAX_SEQ_LEN, chunk_size=CHUNK, sp_axis=0)

    # One chunk-0 forward with the cache attached. `indexed_rope=True` + `build_rope=False` is the
    # chunked runtime's own combination (DEC-044), so this is the deployment op sequence, not a
    # test-only one.
    tokens_embd, rot_mats, _ = tt_model.prepare_inputs_prefill(
        torch.tensor(token_ids, dtype=torch.int32), start_pos=0, build_rope=False
    )
    assert rot_mats is None
    out = tt_model.prefill_forward(
        tokens_embd,
        rot_mats_global=rope_indexed,
        kv_cache=kv_cache,
        cached_len=0,
        user_id=0,
        skip_lm_head=True,
        indexed_rope=True,
    )
    out.deallocate(True)
    ttnn.synchronize_device(submesh)
    logger.info(
        f"[G-KV-TP8] model wrote {num_layers} layers x {SEQ_LEN} tokens into slot 0 of a "
        f"[{NUM_USERS * num_layers}, 1, {MAX_SEQ_LEN // rows}, {head_dim}] bf8_b cache at TP={cols}"
    )

    # --- written-region-only: the three collateral-write failures a PCC cannot see ---
    def _raw(cache_tensor, slot, col=0):
        return ttnn.to_torch(ttnn.get_device_tensors(cache_tensor)[col])[slot, 0].float()

    for layer_idx in (0, num_layers // 2, num_layers - 1):
        other_slot = 1 * num_layers + layer_idx  # user 1, same layer — never written
        worst = max(_raw(kv_cache.k, other_slot).abs().max().item(), _raw(kv_cache.v, other_slot).abs().max().item())
        assert worst == 0.0, (
            f"writing user 0 changed user 1's slot for layer {layer_idx} (max|v| = {worst}); "
            f"slot = user_id*num_layers + layer_idx addressing is wrong at TP=8"
        )
    tail = _raw(kv_cache.k, 0)[SEQ_LEN:]
    assert tail.abs().max().item() == 0.0, (
        f"the unwritten pad tail [{SEQ_LEN}, {MAX_SEQ_LEN}) is not zero "
        f"(max|v| = {tail.abs().max().item()}); a write ran past kv_actual + chunk"
    )
    logger.info(
        f"[G-KV-TP8] written-region-only: user 1's slots exactly 0 at layers "
        f"(0, {num_layers // 2}, {num_layers - 1}); pad tail [{SEQ_LEN}, {MAX_SEQ_LEN}) exactly 0"
    )

    # --- per-layer PCC vs the fp32 golden, read back through the RUNTIME's helpers (R-029) ---
    from safetensors import safe_open

    runtime = _stub_runtime(submesh, hf_config, kv_cache, num_layers=num_layers)
    perm = hf_to_meta_lane_permutation(head_dim, head_dim)
    identity_cols = list(range(n_kv))
    rotated_cols = [(c + 1) % n_kv for c in identity_cols]

    rows_out = []
    worst = {"k": 1.0, "v": 1.0, "ratio_k": 0.0}
    control_worst = 1.0
    for layer_idx in range(num_layers):
        dev_k, dev_v = runtime.gather_layer(slot_id=0, layer_idx=layer_idx, n_tokens=SEQ_LEN, chunk_size=CHUNK)
        assert tuple(dev_k.shape) == (1, n_kv, SEQ_LEN, head_dim), f"gather_layer gave {tuple(dev_k.shape)}"
        with safe_open(str(trace_dir / "kv_cache" / f"layer_{layer_idx}.safetensors"), framework="pt") as handle:
            golden_k = handle.get_tensor(f"key_cache_layer_{layer_idx}").float()[:, :, :SEQ_LEN, :][..., perm]
            golden_v = handle.get_tensor(f"value_cache_layer_{layer_idx}").float()[:, :, :SEQ_LEN, :]
        _, pcc_k = comp_pcc(golden_k, dev_k, 0.0)
        _, pcc_v = comp_pcc(golden_v, dev_v, 0.0)
        # The bf8_b STORAGE floor: the cache holds K and nothing else, so the dtype is the whole
        # storage budget. bf8_b weights and bf16 activations sit on top, which is why a ratio above 1
        # is expected from layer 1 on and is named rather than blamed on the cache (E.5 / DEC-058).
        _, floor_k = comp_pcc(golden_k, quantize_like_device(golden_k, ttnn.bfloat8_b), 0.0)
        ratio_k = err_ratio(pcc_k, floor_k)
        # Negative control, per layer: the same device bytes read with the columns rotated.
        wrong_k = _read_layer_with_col_map(
            submesh, kv_cache.k, slot=0 * num_layers + layer_idx, col_map=rotated_cols, n_tokens=SEQ_LEN
        )
        _, pcc_control = comp_pcc(golden_k, wrong_k, 0.0)
        control_worst = min(control_worst, float(pcc_control))

        rows_out.append((layer_idx, float(pcc_k), float(pcc_v), ratio_k))
        worst["k"] = min(worst["k"], float(pcc_k))
        worst["v"] = min(worst["v"], float(pcc_v))
        worst["ratio_k"] = max(worst["ratio_k"], ratio_k)
        logger.info(
            f"[G-KV-TP8] L{layer_idx:>2}: K={pcc_k:.5f} V={pcc_v:.5f} | bf8_b K floor={floor_k:.5f} "
            f"err_ratio={ratio_k:.2f}x | rotated-column control K={pcc_control:.5f}"
        )

    logger.info(
        f"[G-KV-TP8] {num_layers} layers at TP={cols}, seq={SEQ_LEN}: min K={worst['k']:.5f} "
        f"(>= {GOLDEN_PCC_K}) min V={worst['v']:.5f} (>= {GOLDEN_PCC_V}) | mean K="
        f"{sum(r[1] for r in rows_out) / len(rows_out):.5f} mean V="
        f"{sum(r[2] for r in rows_out) / len(rows_out):.5f} | layer-0 err_ratio={rows_out[0][3]:.2f}x "
        f"(ceiling {MAX_LAYER0_ERR_RATIO}x) worst err_ratio={worst['ratio_k']:.2f}x | "
        f"rotated-column control worst K={control_worst:.5f} (must be <= {NEGATIVE_CONTROL_MAX_PCC})"
    )

    # --- R-029: dump_slot_kv + compare_device_dump, both on device output for the first time ---
    dump_dir = runtime.dump_slot_kv(tmp_path / "device_dump", slot_id=0, n_tokens=SEQ_LEN, chunk_size=CHUNK)
    lines = []
    dump_ok, _dump_rows, summary = compare_device_dump(
        trace_dir, dump_dir, pcc_k=GOLDEN_PCC_K, pcc_v=GOLDEN_PCC_V, out=lines.append
    )
    for line in lines:
        logger.info(f"[G-KV-TP8] {line}")
    # ... and kv_cache_pcc_check, the helper the galaxy harness actually calls.
    min_pcc = runtime.kv_cache_pcc_check(slot_id=0, n_chunks=1, trace_dir=str(trace_dir), chunk_size=CHUNK)
    logger.info(f"[G-KV-TP8] kv_cache_pcc_check (the harness entry point) returned min PCC {min_pcc:.5f}")

    assert worst["k"] >= GOLDEN_PCC_K, (
        f"[G-KV-TP8] model-produced K at TP=8 vs golden: min {worst['k']:.5f} < {GOLDEN_PCC_K}. "
        f"G-KV proved the cache primitive at nkv=1; this is the model -> cache mesh-mapper step."
    )
    assert worst["v"] >= GOLDEN_PCC_V, f"[G-KV-TP8] model-produced V at TP=8 vs golden: {worst['v']:.5f}"
    assert rows_out[0][3] <= MAX_LAYER0_ERR_RATIO, (
        f"[G-KV-TP8] layer 0's K sits {rows_out[0][3]:.2f}x off the bf8_b storage floor (ceiling "
        f"{MAX_LAYER0_ERR_RATIO}x). Layer 0's input is the exact embedding, so nothing upstream can "
        f"explain it: it is the projection, the RoPE, the cache write or the head->column map."
    )
    assert control_worst <= NEGATIVE_CONTROL_MAX_PCC, (
        f"[G-KV-TP8] NEGATIVE CONTROL FAILED: reading the KV columns rotated by one still scores "
        f"K PCC {control_worst:.5f} > {NEGATIVE_CONTROL_MAX_PCC} against the golden, so this gate "
        f"cannot tell the right head->column map from a wrong one and its PASS means nothing."
    )
    assert dump_ok, (
        f"[G-KV-TP8] compare_device_dump rejected the runtime's own dump: min K {summary.get('min_k')} "
        f"min V {summary.get('min_v')}"
    )
    assert min_pcc >= min(GOLDEN_PCC_K, GOLDEN_PCC_V), f"kv_cache_pcc_check returned {min_pcc}"
