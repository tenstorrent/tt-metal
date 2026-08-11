#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill / decode / K-cache PCC for every named precision policy, on both weight sources.

The point of running the *same* sweep on synthetic and on real checkpoint weights is that
they disagree for the BFP4 policies and agree for everything else — the `$optimize` skill's
OPT-012 case. Synthetic weights are i.i.d. Gaussian at the checkpoint's real per-tensor
scales, which is adversarial for block-float quantisation; the checkpoint's own blocks are
strongly correlated and quantise far better. Selecting a policy on the synthetic column
alone would reject a policy that is correct for this model.

Writes ``doc/optimized_decoder/pcc/policy_sweep.json``.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_optimized_precision.py

Environment: ``POLICIES`` (comma-separated, default all of them), ``KINDS``, ``SEQ``,
``WEIGHTS`` (``synthetic``, ``real`` or both), ``OUT``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import POLICIES, OptimizedDecoder

MODEL_DIR = Path(__file__).resolve().parents[1]
WEIGHT_STATS = MODEL_DIR / "doc" / "functional_decoder" / "weight_stats.json"
OUT = Path(os.environ.get("OUT", MODEL_DIR / "doc" / "optimized_decoder" / "pcc" / "policy_sweep.json"))
POLICY_NAMES = os.environ.get("POLICIES", ",".join(POLICIES)).split(",")
KINDS = os.environ.get("KINDS", "sliding_rope,full_nope").split(",")
SEQ = int(os.environ.get("SEQ", "512"))
WEIGHT_SOURCES = os.environ.get("WEIGHTS", "synthetic,real").split(",")
BLOCK_SIZE = 64
SYNTHETIC_SEED = 7


def _state_dict(source: str, layer_idx: int):
    if source == "real":
        return R.load_real_layer_state_dict(layer_idx)
    stats = json.loads(WEIGHT_STATS.read_text())
    return R.synthetic_layer_state_dict(stats["layers"][str(layer_idx)]["tensors"], seed=SYNTHETIC_SEED)


def _layer_input(source: str, text_config, layer_idx: int, seq: int, *, offset: int = 0):
    """Real activations for the real-weight sweep, unit-RMS noise for the synthetic one."""
    if source == "real":
        return R.stacked_layer_input(text_config, layer_idx, R.real_token_ids(seq, offset=offset))
    return R.unit_rms_hidden_states((1, seq, text_config.hidden_size), seed=13 + offset)


def main() -> int:
    R.configure_host_threads()
    text_config = R.load_text_config()
    kinds = R.layer_kinds(text_config)
    rows = []

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for source in WEIGHT_SOURCES:
            for kind_id in KINDS:
                kind = kinds[kind_id]
                state_dict = _state_dict(source, kind.layer_idx)
                reference = R.ReferenceDecoderLayer(text_config, kind.layer_idx, state_dict)

                hidden = _layer_input(source, text_config, kind.layer_idx, SEQ)
                ref_out, ref_k, ref_v = reference.prefill(hidden, backend="eager")
                cache_len = SEQ + 2
                k_full = torch.zeros(1, ref_k.shape[1], cache_len, ref_k.shape[3], dtype=ref_k.dtype)
                v_full = torch.zeros_like(k_full)
                k_full[:, :, :SEQ] = ref_k
                v_full[:, :, :SEQ] = ref_v
                hidden_d = _layer_input(source, text_config, kind.layer_idx, 1, offset=SEQ)
                # decode() updates the caches in place, so hand it copies
                ref_decode = reference.decode(hidden_d, k_full.clone(), v_full.clone(), torch.tensor([SEQ]))

                for policy in POLICY_NAMES:
                    decoder = OptimizedDecoder.from_state_dict(
                        state_dict,
                        hf_config=text_config,
                        layer_idx=kind.layer_idx,
                        mesh_device=mesh,
                        precision=policy,
                        block_size=BLOCK_SIZE,
                    )
                    blocks = decoder.blocks_per_seq(cache_len)
                    page_table = U.build_page_table(batch=1, blocks_per_seq=blocks, total_blocks=blocks * 3, seed=909)
                    kv_cache = decoder.allocate_kv_cache(
                        batch_size=1, max_seq_len=blocks * BLOCK_SIZE, num_blocks=blocks * 3
                    )
                    page_table_tt = U.page_table_to_device(page_table, mesh)

                    hidden_tt = U.prefill_input(hidden, mesh)
                    out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
                    prefill_pcc = U.pcc(ref_out, U.prefill_output(out))
                    out.deallocate(True)
                    hidden_tt.deallocate(True)

                    cache_pcc = U.pcc(
                        ref_k, U.read_paged_cache(kv_cache[0], page_table, block_size=BLOCK_SIZE, seq_len=SEQ)
                    )

                    current_pos, rope_idxs = U.position_tensors([SEQ], mesh)
                    out_d = decoder.decode_forward(
                        U.decode_input(hidden_d, mesh),
                        kv_cache=kv_cache,
                        page_table=page_table_tt,
                        current_pos=current_pos,
                        rope_idxs=rope_idxs,
                    )
                    decode_pcc = U.pcc(ref_decode, U.decode_output(out_d))
                    out_d.deallocate(True)

                    row = {
                        "weights": source,
                        "kind": kind_id,
                        "policy": policy,
                        "seq_len": SEQ,
                        "prefill_pcc": prefill_pcc,
                        "decode_pcc": decode_pcc,
                        "k_cache_pcc": cache_pcc,
                        "dtypes": decoder.config_summary()["dtypes"],
                        "fidelity": decoder.config_summary()["fidelity"],
                    }
                    rows.append(row)
                    print(
                        f"POLICY {source:9s} {kind_id:12s} {policy:22s} "
                        f"prefill={prefill_pcc:.6f} decode={decode_pcc:.6f} kcache={cache_pcc:.6f}",
                        flush=True,
                    )
                    for tensor in (kv_cache[0], kv_cache[1], page_table_tt):
                        tensor.deallocate(True)
                    del decoder
    finally:
        ttnn.close_mesh_device(mesh)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seq_len": SEQ, "records": rows}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
