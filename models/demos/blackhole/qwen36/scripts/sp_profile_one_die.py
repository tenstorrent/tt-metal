# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op profiling driver: one die, T=1024, comms off, plain 1x1 mesh (no fabric).

Tracy cannot profile submesh traces, so this reuses SPPrefill's die-0 free functions
(_sp_layer_mixer, _sp_layer_mlp, _identity_page_table) and two instance methods
(_mk_kv_cache, _tail) via a tiny attribute-only shim, instead of opening a real (1,4)
mesh + sockets. Mirrors SPPrefill's die-0 program: embed -> per layer [attention
forward_prefill_paged | GDN forward_prefill, zero initial state] -> norm + LM head.

Run: env per SP_PREFILL_HANDOFF.md sec 6, then
  python -m tracy -r -p <abs path to this script> --runs 1
to capture a Tracy ops-perf CSV; feed it to analyze_ops_csv.py for the per-op breakdown.
"""
import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
from models.demos.blackhole.qwen36.tt.sp_prefill import SPPrefill, _identity_page_table, _sp_layer_mixer, _sp_layer_mlp

SPAN_LEN = 1024
BLOCK_SIZE = 64


class _HelperCtx:
    """Attribute-only shim so SPPrefill._mk_kv_cache / SPPrefill._tail (unbound methods) can run
    with no real SPPrefill instance (no submesh, no sockets). Both methods only read a handful of
    self.* attributes (checked against sp_prefill.py) and never touch self.subs/self.rbuf/sockets."""

    def __init__(self, num_blocks, nkv, hd, span_len):
        self.num_blocks = num_blocks
        self.nkv = nkv
        self.hd = hd
        self.span_len = span_len


def build_one_die(mesh):
    args, model, _state_dict = create_tt_model(
        mesh, max_batch_size=1, max_seq_len=SPAN_LEN, hf_model=os.environ.get("HF_MODEL"), sequence_parallel=True
    )
    return args, model


def setup_die0(mesh, args, model):
    """Same KV cache / page table / GDN zero-state setup SPPrefill.__init__ does for die 0."""
    num_blocks = SPAN_LEN // BLOCK_SIZE  # single 1024-token span -> 16 blocks, no prefix
    ctx = _HelperCtx(num_blocks=num_blocks, nkv=args.n_local_kv_heads, hd=args.head_dim, span_len=SPAN_LEN)

    for layer in model.layers:
        if layer.is_full_attention:
            k_cache = SPPrefill._mk_kv_cache(ctx, mesh)
            v_cache = SPPrefill._mk_kv_cache(ctx, mesh)
            layer.attention.set_paged_kv_cache(k_cache, v_cache)

    full_page_table = _identity_page_table(mesh, num_blocks)
    own_chunk_page_table = _identity_page_table(mesh, num_blocks, offset=0)

    # Die 0's GDN initial state is a permanent zero tensor (see sp_prefill.py::_alloc_rbuf).
    state_shape = (1, args.gdn_nv_tp, args.gdn_dk, args.gdn_dv)
    conv_shape = (1, args.linear_conv_kernel_dim - 1, args.gdn_qkv_dim_tp)
    zero_state = ttnn.zeros(
        state_shape, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    zero_conv = ttnn.zeros(
        conv_shape, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return ctx, full_page_table, own_chunk_page_table, (zero_state, zero_conv)


def run_prefill(model, args, tokens, full_page_table, own_chunk_page_table, gdn_zero, ctx):
    """Mirrors _run_layer_major's per-die body for d=0, comm=False (no send/recv at all)."""
    mesh = model.device
    tok = ttnn.from_torch(
        tokens.to(torch.int32), dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    cos_t, sin_t = model._rope_tp_cos_sin_torch(0, SPAN_LEN)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    # L1 seeds the whole residual/norm chain: rms_norm and the residual add both inherit
    # input_a's memory config, so DRAM here (the default) would force every norm/add in DRAM.
    x = model.embd(tok, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.reshape(x, (1, 1, SPAN_LEN, x.shape[-1]))

    for layer in model.layers:
        if layer.is_full_attention:
            x, _ = _sp_layer_mixer(
                layer,
                x,
                cos=cos,
                sin=sin,
                d=0,
                span_len=SPAN_LEN,
                page_table=full_page_table,
                chunk_page_table=own_chunk_page_table,
                gdn_in=None,
            )
        else:
            x, _gdn_out = _sp_layer_mixer(
                layer,
                x,
                cos=cos,
                sin=sin,
                d=0,
                span_len=SPAN_LEN,
                page_table=None,
                chunk_page_table=None,
                gdn_in=gdn_zero,
            )
        # Mirrors _run_layer_major: the cross-die send (skipped here, comm=False / single die)
        # happens between the mixer and the MLP -- see _sp_layer_mixer's docstring.
        x = _sp_layer_mlp(layer, x)

    logits = SPPrefill._tail(ctx, model, x, traced=False)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args_cli = ap.parse_args()

    # No ttnn.set_fabric_config call at all: mirrors conftest.py's mesh_device fixture under
    # QWEN36_SP=1 + MESH_DEVICE=P150 (test_factory.py::parametrize_mesh_tp), which omits
    # "fabric_config" from device_params entirely for the (1,1)+SP case, so conftest's
    # set_fabric(fabric_config=None, ...) is a no-op and no fabric is ever configured.
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=GDN_CONV1D_L1_SMALL_SIZE)
    try:
        model_args, model = build_one_die(mesh)
        ctx, full_page_table, own_chunk_page_table, gdn_zero = setup_die0(mesh, model_args, model)

        torch.manual_seed(0)
        tokens = torch.randint(1000, 100000, (1, SPAN_LEN), dtype=torch.long)

        # SPPrefill._tail(traced=False) (called inside run_prefill) already does the host
        # readback (ttnn.to_torch) itself and returns a plain torch.Tensor -- see sp_prefill.py
        # lines 391-421. Do NOT call ttnn.to_torch on its output again.
        logger.info("[run_one_die_prefill] warmup run")
        warm_logits = run_prefill(model, model_args, tokens, full_page_table, own_chunk_page_table, gdn_zero, ctx)
        ttnn.synchronize_device(mesh)
        logger.info(f"[run_one_die_prefill] warmup argmax={int(torch.argmax(warm_logits))}")

        for i in range(args_cli.runs):
            t0 = time.perf_counter()
            logits = run_prefill(model, model_args, tokens, full_page_table, own_chunk_page_table, gdn_zero, ctx)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            argmax = int(torch.argmax(logits))
            print(f"RESULT run={i} wall_ms={(t1 - t0) * 1000:.3f} argmax={argmax}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
