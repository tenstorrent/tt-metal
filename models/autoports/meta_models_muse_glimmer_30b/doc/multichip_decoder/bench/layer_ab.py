# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the multichip decoder, and for the single-chip baseline.

Same protocol as the optimized stage's harness so the two stages' numbers are
comparable: warmed traced decode ms/token (min of ``--rounds``), warmed prefill
ms, and PCC against the same HF reference layer.  The single-chip baseline is a
candidate in the same file (``--candidates single --mesh 1x1``) so the speedup is
measured on this host with this protocol rather than quoted from another log.

    # baseline, one chip
    python .../bench/layer_ab.py --mesh 1x1 --candidates single
    # the shipped multichip layer plus geometry candidates, four chips
    python .../bench/layer_ab.py --mesh 1x4 --candidates tp4,mlp_bw26,sdpa_mc16
    python .../bench/layer_ab.py --list
    # the fabric packet payload, which is a mesh-open argument
    python .../bench/layer_ab.py --mesh 1x4 --candidates tp4 --packet-bytes 4352
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import reference_layer_indices
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_L1_SMALL_SIZE,
    FABRIC_CONFIG,
    FABRIC_PACKET_PAYLOAD_BYTES,
    MULTICHIP_DECODE_MATMUL,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder

PAGE_BLOCK = 64
MAX_SEQ = 16384
BF16, BFP8, BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b


def geometry(**overrides):
    """``MULTICHIP_DECODE_MATMUL`` with per-role ``(cores, in0_block_w)`` overrides.

    ``role__dtype=(cores, in0_block_w)``; the dtype suffix matters because the L1
    circular-buffer budget is dtype-scaled, so a legal value at BFP4 can be
    illegal at BFP8.
    """
    table = dict(MULTICHIP_DECODE_MATMUL)
    for key, value in overrides.items():
        role, dtype = key.rsplit("__", 1)
        table[(role, {"bfp4": BFP4, "bfp8": BFP8, "bf16": BF16}[dtype])] = value
    return table


def mlp_geometry(cores: int, gate_bw: int, down_bw: int):
    return geometry(mlp_gate__bfp4=(cores, gate_bw), mlp_up__bfp4=(cores, gate_bw), mlp_down__bfp4=(cores, down_bw))


#: ``name -> kwargs`` for ``MultichipDecoder.from_state_dict``.  Every candidate
#: changes exactly one decision against ``tp4`` so a regression can be assigned.
#:
#: Legal ``in0_block_w`` per role at 8 boundary cores (must divide the
#: activation's per-core K-tile count): ``wqkv``/``attn_gate``/``mlp_gate``/
#: ``mlp_up`` 26 K-tiles -> {1, 2, 13, 26}; ``o_proj`` 4 -> {1, 2, 4};
#: ``mlp_down`` 20 -> {1, 2, 4, 5, 10, 20}.  MLP core counts have to divide both
#: the padded 160 intermediate tiles and the 208 hidden tiles, i.e. {1, 2, 4, 8,
#: 16}.
def all_cores(cores: int, wqkv_bw: int, oproj_bw: int, gate_bw: int, down_bw: int) -> dict:
    """One core count for every width-sharded decode tensor."""
    return {
        "boundary_cores": cores,
        "decode_matmul": geometry(
            wqkv__bfp8=(cores, wqkv_bw),
            attn_gate__bfp8=(cores, wqkv_bw),
            o_proj__bfp8=(cores, oproj_bw),
            mlp_gate__bfp4=(cores, gate_bw),
            mlp_up__bfp4=(cores, gate_bw),
            mlp_down__bfp4=(cores, down_bw),
        ),
    }


CANDIDATES: dict[str, dict] = {
    #: The shipped layer: 16 cores everywhere, largest legal ``in0_block_w``.
    "tp4": {},
    # -- the grid, one core count for the whole step -------------------------
    "grid8": all_cores(8, 26, 4, 13, 20),
    "grid4": all_cores(4, 26, 8, 26, 20),
    # -- a separate MLP working grid, which is what the single-chip layer needs
    "grid8_mlp16": {
        "boundary_cores": 8,
        "decode_matmul": {
            **all_cores(8, 26, 4, 13, 20)["decode_matmul"],
            ("mlp_gate", BFP4): (16, 13),
            ("mlp_up", BFP4): (16, 13),
            ("mlp_down", BFP4): (16, 10),
        },
    },
    "grid16_mlp8": {"decode_matmul": mlp_geometry(8, 26, 20)},
    # -- in0_block_w per role, at the shipped 16-core grid -------------------
    "qkv_bw1": {"decode_matmul": geometry(wqkv__bfp8=(16, 1), attn_gate__bfp8=(16, 1))},
    "oproj_bw1": {"decode_matmul": geometry(o_proj__bfp8=(16, 1))},
    "gu_bw1": {"decode_matmul": mlp_geometry(16, 1, 10)},
    "down_bw5": {"decode_matmul": mlp_geometry(16, 13, 5)},
    "down_bw2": {"decode_matmul": mlp_geometry(16, 13, 2)},
    # -- decode SDPA: the cores-per-(batch, head) cap is the multichip knob --
    "sdpa_mc16": {"decode_sdpa": (None, None, 0, 0, 16)},
    "sdpa_mc32": {"decode_sdpa": (None, None, 0, 0, 32)},
    "sdpa_mc64": {"decode_sdpa": (None, None, 0, 0, 64)},
    "sdpa_fixed": {"decode_sdpa": (None, None, 32, 64, 64)},
    "sdpa_8x8_mc64": {"decode_sdpa": (8, 8, 0, 0, 64)},
    # -- collective payload dtype -------------------------------------------
    "ccl_bfp8": {"ccl_dtype": BFP8},
    "ccl_bf16": {"prefill_ccl_dtype": None, "decode_ccl_dtype": None},
    # -- the reducer: one fused all-reduce vs an explicit reduce-scatter +
    # all-gather pair (identical bytes on a ring, two dispatches instead of one)
    "ccl_rs_ag": {"ccl_mode": "rs_ag"},
    "ccl_all_reduce": {"ccl_mode": "all_reduce"},
    "ccl_rs_w2": {"ccl_rs_workers": 2},
    # NOTE: identical to ``ccl_rs_ag`` -- ``ccl_mode="rs_ag"`` resolves to the same
    # per-mode pair, and both take the shipped per-payload worker counts.  Kept as
    # a **same-config repeat control**: running it alongside ``ccl_rs_ag`` in one
    # invocation is how the prefill noise floor is measured.  For a genuinely
    # different prefill reducer, use ``ccl_rs_ag_prefill_w1``.
    "ccl_rs_ag_prefill": {"prefill_ccl_mode": "rs_ag"},
    "ccl_rs_ag_prefill_w1": {"prefill_ccl_mode": "rs_ag", "prefill_ccl_rs_workers": 1},
    # -- fold the activations into the matmul instead of the elementwise op --
    "fused_act": {"decode_fused_activation": True},
}


def page_table(mesh, batch, max_seq_len, seed=7):
    blocks = (max_seq_len + PAGE_BLOCK - 1) // PAGE_BLOCK
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch * blocks, generator=gen).reshape(batch, blocks).to(torch.int32)
    return ttnn.from_torch(
        perm,
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_dev(mesh, hidden):
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def pos_tensors(mesh, positions):
    cur = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rope = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cur, rope


def host(tensor: ttnn.Tensor) -> torch.Tensor:
    """Device 0's copy of a replicated tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def build(mesh, name, layer_idx, state_dict, decode_context):
    max_seq = MAX_SEQ if decode_context < MAX_SEQ else 131072
    common = dict(
        hf_config=R.hf_config(),
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=max_seq,
        page_block_size=PAGE_BLOCK,
        prefill_chunk_size=8192,
    )
    if name == "single":
        return OptimizedDecoder.from_state_dict(state_dict, **common)
    return MultichipDecoder.from_state_dict(state_dict, **common, **CANDIDATES[name])


def run_candidate(mesh, name, kind, layer_idx, state_dict, ref_layer, args):
    dec = build(mesh, name, layer_idx, state_dict, args.decode_context)
    pt = page_table(mesh, 1, dec.config.max_seq_len, seed=3)
    result = {"name": name, "kind": kind}

    hidden = R.synthetic_hidden_states(1, args.pcc_seq_len, seed=42)
    ref_out, ref_cache = R.reference_prefill(ref_layer, layer_idx, hidden)
    tt_out = dec.prefill_forward(to_dev(mesh, hidden), page_table=pt, user_id=0)
    result["prefill_pcc"] = pcc(host(tt_out).reshape(1, args.pcc_seq_len, -1), ref_out)
    ttnn.deallocate(tt_out)
    token = R.synthetic_hidden_states(1, 1, seed=100)
    ref_dec = R.reference_decode(
        ref_layer, layer_idx, token, past_key_values=ref_cache, positions=torch.tensor([args.pcc_seq_len])
    )
    cur, rope = pos_tensors(mesh, torch.tensor([args.pcc_seq_len]))
    tt_dec = dec.decode_forward(to_dev(mesh, token), current_pos=cur, page_table=pt, rope_pos_ids=rope)
    result["decode_pcc"] = pcc(host(tt_dec).reshape(1, 1, -1), ref_dec)
    ttnn.deallocate(tt_dec)
    for t in (cur, rope):
        ttnn.deallocate(t)

    if args.prefill_seq:
        tt_hidden = to_dev(mesh, R.synthetic_hidden_states(1, args.prefill_seq, seed=43))
        for _ in range(2):
            ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
        best = float("inf")
        for _ in range(args.rounds):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(2):
                ttnn.deallocate(dec.prefill_forward(tt_hidden, page_table=pt, user_id=0))
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - t0) / 2 * 1e3)
        result["prefill_ms"] = best
        ttnn.deallocate(tt_hidden)
    else:
        result["prefill_ms"] = float("nan")

    batch = args.batch
    tt_token = to_dev(mesh, R.synthetic_hidden_states(1, batch, seed=44).reshape(1, batch, -1))
    cur, rope = pos_tensors(mesh, torch.full((batch,), args.decode_context))
    ttnn.deallocate(dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope))
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    tt_trace_out = dec.decode_forward(tt_token, current_pos=cur, page_table=pt, rope_pos_ids=rope)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(8):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    best = float("inf")
    per_round = []
    for _ in range(args.rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(args.decode_iters):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        per_round.append((time.perf_counter() - t0) / args.decode_iters * 1e3)
        best = min(best, per_round[-1])
    result["decode_ms"] = best
    result["decode_rounds"] = per_round
    ttnn.release_trace(mesh, trace_id)
    for t in (tt_token, cur, rope, pt, tt_trace_out):
        ttnn.deallocate(t)
    del dec
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="tp4")
    ap.add_argument("--kinds", default="sliding,full")
    ap.add_argument("--mesh", default="1x4")
    ap.add_argument("--pcc-seq-len", type=int, default=100)
    ap.add_argument("--prefill-seq", type=int, default=8192)
    ap.add_argument("--decode-context", type=int, default=2048)
    ap.add_argument("--decode-iters", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    # The fabric packet payload is a mesh-open argument, not a decoder argument,
    # so it cannot be a CANDIDATE: measuring it whole-layer needs one process per
    # value.  ``bench/run_review2_chain.sh`` runs the pair.
    ap.add_argument("--packet-bytes", type=int, default=FABRIC_PACKET_PAYLOAD_BYTES)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--real-weights", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, kwargs in CANDIDATES.items():
            print(f"{name:16s} {kwargs}")
        return

    mesh_shape = tuple(int(v) for v in args.mesh.split("x"))
    single = mesh_shape == (1, 1)
    # A 1x1 mesh is opened *without* fabric and *without* an L1_SMALL region:
    # there is no link to configure and no collective to hold a semaphore, so the
    # baseline runs in exactly the regime the single-chip stage measured -- the
    # whole L1 pool, no reservation.
    mesh = open_multichip_mesh(
        mesh_shape,
        trace_region_size=90112 * 12,
        l1_small_size=0 if single else DEFAULT_L1_SMALL_SIZE,
        packet_payload_bytes=args.packet_bytes,
        fabric_config=None if single else FABRIC_CONFIG,
    )
    ttnn.SetDefaultDevice(mesh)
    print(f"MESH {mesh.shape} devices={mesh.get_num_devices()} grid={mesh.compute_with_storage_grid_size()}")
    try:
        idxs = reference_layer_indices(R.hf_config())
        for kind in args.kinds.split(","):
            layer_idx = idxs[kind]
            state_dict = R.real_state_dict(layer_idx) if args.real_weights else R.synthetic_state_dict(layer_idx)
            ref_layer = R.reference_layer(layer_idx, state_dict)
            for name in args.candidates.split(","):
                try:
                    r = run_candidate(mesh, name, kind, layer_idx, state_dict, ref_layer, args)
                    print(
                        f"AB{'[real]' if args.real_weights else '      '} {name:16s} kind={kind:8s} "
                        f"mesh={args.mesh} batch={args.batch} "
                        f"prefill{args.prefill_seq}={r['prefill_ms']:8.2f} ms  "
                        f"traced_decode@{args.decode_context}={r['decode_ms']:7.4f} ms/token  "
                        f"prefill_pcc={r['prefill_pcc']:.6f} decode_pcc={r['decode_pcc']:.6f}  "
                        f"({'/'.join(f'{v:.4f}' for v in r['decode_rounds'])})",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                    print(f"FAILED {name:16s} kind={kind:8s} {msg[:400]}", flush=True)
                    traceback.print_exc()
    finally:
        close_multichip_mesh(mesh, fabric_config=None if single else FABRIC_CONFIG)


if __name__ == "__main__":
    main()
