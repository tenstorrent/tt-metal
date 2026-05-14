# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-9 — Decode trace-capture parity test on BH GLX 8x4.

Goal
----

Build a 4-layer hybrid TtTransformer (pattern ``[lin, lin, lin, full]``),
run one eager decode step, then capture and replay the same decode under
``ttnn.begin_trace_capture`` / ``ttnn.execute_trace``.  Acceptance is
``PCC(eager_logits, traced_logits) >= 0.9999`` — trace replay should be
bit-identical to the eager forward.

Status (as of V2-9 attempt, see BRINGUP_LOG.md)
-----------------------------------------------

**SKIPPED — pre-trace blocker chain (decode never wired end-to-end).**

V2-9 bring-up confirmed: qwen3.6 decode in v2 is structurally broken at
multiple boundaries. The v2 tree inherited 70B's generator+model decode
contract (batch-32 packed in T-dim, L1-width-sharded residual via
``DECODE_RESIDUAL_MEMCFG``, ``tt_sharded_distributed_rmsnorm`` for the
norms), but the qwen3.6 attention + DeltaNet blocks are written against
v1's contract (single-user ``[B=batch, 1, T=1, H]``, DRAM-interleaved
residual, ``tt_distributed_rmsnorm`` for the norms — same primitive used
in qwen3.6 prefill, which is verified at PCC > 0.99).

V2-9 attempt explored the gap and surfaced this dependency chain (each
fix unblocks the next failure point):

1. **Pre-decode infrastructure fixes** — APPLIED in this commit, low
   risk, no impact on passing prefill tests:

     - ``TtLlamaAttention.prefetch`` references ``self.wqkv`` which does
       not exist on qwen3.6 (we use ``self.wqkvg``).  Guard with
       ``not self.is_qwen36``.  *Fixed.*
     - ``_NoOpPrefetcherSetup`` was missing ``worker_sub_device_id``,
       which ``TtTransformer.forward(mode='decode')`` reads
       unconditionally for the stall-group set call.  Added the attr;
       ``setup_decode`` keeps it in sync with the worker sub_device.
       *Fixed.*

2. **Layout chain** — REVERTED from this commit; documented here:

     (a) The decoder block currently only takes the ``is_qwen36_prefill``
         gather/scatter branch (``self.is_qwen36 and mode == "prefill"``).
         Decode falls through to the 70B path which passes ``[1,1,32,H/4]``
         (col-sharded) to ``TtLlamaAttention.forward_decode`` →
         ``_forward_decode_qwen36`` expects full-H ``[1,1,32,H=5120]``
         → matmul shape mismatch (1280 vs 5120).
         **Tried fix**: extend ``is_qwen36_path = is_qwen36 and mode in (prefill, decode)``,
         relax the ``skip_mem_cfg`` assert for qwen36 decode, use
         ``DRAM_MEMORY_CONFIG`` everywhere.  Got past matmul to next layer.
     (b) ``mesh_partition`` produces DRAM-interleaved output, but
         ``DistributedNorm.forward(mode='decode')`` uses
         ``tt_sharded_distributed_rmsnorm`` which requires L1-WIDTH-sharded
         input with the exact ``gather_in_mem_cfg`` shape config — DRAM
         input crashes ``LayerNormShardedMultiCoreProgramConfig`` with
         "bad optional access".
         **Tried fix**: ``to_memory_config(x_norm_in, gather_in_mem_cfg)``
         before each norm call in decode mode.  Got past the norm to
         attention.
     (c) ``[1,1,32,H]`` reaches ``_forward_decode_qwen36`` and DeltaNet
         ``forward_decode``.  Both interpret dim-2=32 as T-axis, slice
         to T=1.  BUT the DeltaNet persistent state buffers were sized
         ``[B=max_batch_size=32, ...]`` — slicing to ``[B=1, T=1, H]``
         then matmul-ing against state ``[B=32, ...]`` → bmm shape
         mismatch (a=1 vs b=32).
         **Tried fix**: re-interpret ``[1,1,32,H]`` as ``[B=32, T=1, H]``
         (the 32 IS the batch dim; previous reshape was the bug).  Got
         further: DeltaNet l2_norm runs on ``[32,1,6,128]`` correctly.
     (d) DeltaNet internal recurrent kernel hits
         ``K=68 must be divisible by in0_block_w=5`` — DeltaNet
         program configs in v2 may be inheriting prefill-time tile
         configs that don't match the batch=32 decode shape.  Needs
         dedicated decode-time program configs.
     (e) Post-layer ``self.norm`` at the TtTransformer.forward exit
         (line 1035) also goes through ``tt_sharded_distributed_rmsnorm``
         in decode mode — same DRAM/L1 mismatch as (b), needs the same
         ``to_memory_config`` shim.
     (f) ``self.lm_head`` decode path expects an L1-sharded input
         (the 70B contract).  qwen3.6 decode exit is DRAM ⇒ another
         layout converter needed.
     (g) Once eager decode works, V2-9 trace capture itself can be
         attempted.  No host-write blockers were seen in the qwen3.6
         forward paths during the static review (the v1 PERF.md
         ``to_memory_config`` blocker is in the *70B* branch of
         ``TtLlamaAttention.forward_decode``, NOT in
         ``_forward_decode_qwen36``).

Recommended path for **V2-decode (predecessor to V2-9 trace)**:

  - Rather than incrementally bridging the 70B↔qwen3.6 decode boundaries
    one layout converter at a time, mirror v1's decode contract directly:
    add a ``forward_decode_qwen36`` entry point on ``TtTransformer`` that
    takes ``[B=batch, 1, T=1, H]`` natively, uses DRAM-interleaved
    residual throughout, and calls ``tt_distributed_rmsnorm`` (not the
    L1-sharded variant).  The generator's ``ttnn_decode_forward`` should
    dispatch to this for qwen3.6.  Estimated effort: 1-2 sessions.
  - Once eager decode is PCC-verified vs HF reference, flip the
    ``_DECODE_ENABLED`` flag below to ``True`` — all the trace machinery
    in ``models/demos/qwen3_6_galaxy_v2/tt/generator.py`` is already
    wired (begin_trace_capture / end_trace_capture / execute_trace,
    trace_ids_decode dict, _capture_trace_text, _decode_easy_trace_text,
    release_traces in __del__).  A static review found no
    ``from_torch(device=...)`` / ``to_torch`` host writes in the qwen3.6
    forward path — the trace should capture cleanly.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_trace_parity.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_B = 1
_T_PREFILL = 32
_N_LAYERS = 4
_PATTERN = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
_PCC_PARITY = 0.9999

# Set to True once V2-decode (Path (i)+(ii) above) lands and decode runs
# end-to-end.  Until then the body skips with a clear blocker message; the
# test stays in the suite as a "wake me when decode works" sentinel.
_DECODE_ENABLED = False


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_state_dict_for_layers(snapshot_dir: pathlib.Path, layer_indices: list[int]) -> dict:
    """Load HF weights needed for an N-layer TtTransformer."""
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
    ] + [f"model.language_model.layers.{i}." for i in layer_indices]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict, pattern: list[str], n_layers: int, max_batch_size: int = 32):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    # Decode embedding output is L1-sharded with shard shape (1,1,32,...) — needs
    # batch = 32 to fit the 60-core grid. max_batch_size=1 (default) trips the
    # 'num_shards <= num_cores' assert at the very first embedding op.
    args = TtQwen36ModelArgs(mesh, max_batch_size=max_batch_size)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


@pytest.mark.hardware
def test_qwen36_decode_trace_parity(bh_glx_mesh):
    """Eager-vs-traced 1-step decode parity on 4-layer hybrid TtTransformer.

    Currently SKIPS because V2-decode (the prerequisite) is not yet wired.
    See module docstring for the recommended path.
    """
    if not _DECODE_ENABLED:
        pytest.skip(
            "[V2-9 BLOCKER] qwen3.6 decode is not yet wired in v2 — "
            "TtTransformerBlock.forward only takes the qwen36 gather/scatter "
            "branch for prefill. Decode falls through to the 70B path, which "
            "passes [1, 1, 32, H/4] into _forward_decode_qwen36 / DeltaNet "
            "forward_decode and trips 'assert T == 1' (DeltaNet interprets "
            "the batch-in-T-dim packing as T=32). Fix V2-decode first; this "
            "test is the sentinel that wakes back up once decode runs."
        )

    layer_indices = list(range(_N_LAYERS))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[V2-9 parity] loaded {len(state_dict)} weights for layers {layer_indices}")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS)
    print(f"[V2-9 parity] TT 4-layer hybrid model built (pattern={_PATTERN})")

    # _build_tt_model exits in prefill setup; switch to decode before
    # capture so the per-layer attention.prefetch hook + tt_ccl decode
    # binding are in place.
    model.switch_mode("decode")

    # prepare_decode_inputs_host pads tokens to max_batch_size (=32).
    # rope_setup hard-codes a [32]-shape position tensor.
    B = args.max_batch_size  # 32
    tokens_torch = torch.zeros(B, dtype=torch.long)
    tokens_torch[0] = 1  # arbitrary token id for user 0; others stay 0
    current_pos_torch = torch.zeros(32, dtype=torch.long)

    host_inputs = model.prepare_decode_inputs_host(tokens_torch, current_pos_torch, page_table=None)

    from models.demos.qwen3_6_galaxy_v2.tt.llama_common import copy_host_to_device

    device_inputs_eager = copy_host_to_device(host_inputs, mesh_device=bh_glx_mesh)
    tokens_tt_e, current_pos_tt_e, rope_idxs_tt_e, page_table_tt_e = device_inputs_eager

    eager_logits_tt = model.ttnn_decode_forward(
        tokens_tt_e,
        current_pos_tt_e,
        rope_idxs_tt_e,
        page_table=page_table_tt_e,
        kv_cache=None,
        return_logits=True,
    )

    logits_tt_eager = eager_logits_tt[0] if isinstance(eager_logits_tt, tuple) else eager_logits_tt
    eager_logits_cpu = ttnn.to_torch(
        logits_tt_eager,
        mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    print(f"[V2-9 parity] eager decode logits shape: {eager_logits_cpu.shape}")

    # Trace-captured decode — second device_inputs binding so we can verify
    # the eager and traced runs see the exact same on-device tokens.
    device_inputs_trace = copy_host_to_device(host_inputs, mesh_device=bh_glx_mesh)
    tokens_tt_t, current_pos_tt_t, rope_idxs_tt_t, page_table_tt_t = device_inputs_trace

    trace_id = None
    try:
        # Compile-run inside capture is the standard 70B pattern; we run
        # one warm-up forward to settle any lazy allocs first.
        _ = model.ttnn_decode_forward(
            tokens_tt_t,
            current_pos_tt_t,
            rope_idxs_tt_t,
            page_table=page_table_tt_t,
            kv_cache=None,
            return_logits=True,
        )
        ttnn.synchronize_device(bh_glx_mesh)

        trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
        traced_logits_tt = model.ttnn_decode_forward(
            tokens_tt_t,
            current_pos_tt_t,
            rope_idxs_tt_t,
            page_table=page_table_tt_t,
            kv_cache=None,
            return_logits=True,
        )
        ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(bh_glx_mesh)

        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
    finally:
        if trace_id is not None:
            try:
                ttnn.release_trace(bh_glx_mesh, trace_id)
            except Exception:
                pass

    logits_tt_traced = traced_logits_tt[0] if isinstance(traced_logits_tt, tuple) else traced_logits_tt
    traced_logits_cpu = ttnn.to_torch(
        logits_tt_traced,
        mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    print(f"[V2-9 parity] traced decode logits shape: {traced_logits_cpu.shape}")

    pcc = _pcc(eager_logits_cpu, traced_logits_cpu)
    print(f"[V2-9 parity] eager-vs-traced PCC = {pcc:.6f} (thresh={_PCC_PARITY})")
    assert pcc >= _PCC_PARITY, f"trace replay drift: PCC {pcc:.6f} < {_PCC_PARITY}"
    print(f"[V2-9 parity] PASSED — trace replay matches eager within {1 - _PCC_PARITY:.0e} PCC")
