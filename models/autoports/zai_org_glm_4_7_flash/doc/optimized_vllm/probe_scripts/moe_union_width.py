# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How many distinct experts does the routed-expert union actually contain, on
the real model, as a function of live decode rows?

This is the number the compact-vs-union decision turns on and the one the stage
was missing. The single-layer probes (``moe_compact_layer.json``,
``moe_union_vs_compact.json``) drive routing from synthetic ``torch.randn``
activations, and they report compact ``kc = n_experts`` *beating* the union path
at 32 rows -- the opposite of what the whole-model arm measures
(``adapter_decode_floor_kc64.json``). The explanation has to be that synthetic
activations select far more distinct experts per batch than real ones do, but
until this probe that was an assertion, not a measurement.

Runs the real 47-layer model's **eager** (untraced) decode -- the compatibility
path, not the measured serving path -- with the compact routing mask on, and
reads back each MoE layer's union mask per step. Also re-runs the same shapes
with ``torch.randn`` activations injected into one layer, so the synthetic and
real widths are measured the same way.

    python .../probe_scripts/moe_union_width.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tests import utils  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import (  # noqa: E402
    VLLM_PREFILL_BUCKETS,
    VLLM_PREFILL_CHUNK_SIZE,
    GLM47FlashForCausalLM,
)
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / "moe_union_width.json"

MAX_SEQ_LEN = 202752
BLOCK_SIZE = 64
MAX_BATCH = 32
NUM_BLOCKS = 7362
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    widths = {}
    try:
        gen = build_generator(
            MODEL_DIR,
            dev,
            max_batch_size=MAX_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            defer_cache_and_traces=True,
            prefill_chunk_size=VLLM_PREFILL_CHUNK_SIZE,
            prefill_buckets=VLLM_PREFILL_BUCKETS,
            progress=lambda m: None,
        )
        model = GLM47FlashForCausalLM(gen)
        kv = model.allocate_kv_cache(
            kv_cache_shape=(NUM_BLOCKS, 1, BLOCK_SIZE, gen.model.layers[0].kvpe_dim),
            dtype=torch.bfloat16,
            num_layers=len(gen.model.layers),
        )
        gen.reset()
        pt = torch.zeros((MAX_BATCH, BLOCKS_PER_USER), dtype=torch.int32)
        for r in range(MAX_BATCH):
            pt[r] = torch.arange(r * 16, r * 16 + BLOCKS_PER_USER, dtype=torch.int32) % NUM_BLOCKS
        gen.refresh_page_table(pt)

        # Instrument every MoE layer's routing: count the distinct experts the
        # union selects. Runs on the eager path, so the readback is legal.
        seen = []
        original = OptimizedDecoder._routing_weights_decode

        def instrumented(self, x, active_mask=None):
            routing = original(self, x, active_mask)
            host = ttnn.to_torch(routing).float()  # [1,1,B,E]
            union = host.max(dim=2).values  # [1,1,E]
            seen.append(int((union > 0).sum()))
            return routing

        OptimizedDecoder._routing_weights_decode = instrumented
        try:
            for rows in (1, 2, 4, 8, 16, 32):
                toks = [(1000 + 977 * r) % 150000 for r in range(rows)] + [0] * (MAX_BATCH - rows)
                pos = [128 + 13 * r for r in range(rows)] + [-1] * (MAX_BATCH - rows)
                gen.set_decode_tokens(toks)
                gen.set_decode_positions(pos)
                seen.clear()
                logits = gen._decode_logits_device(kv_cache=kv, moe_kc=4, advance_positions=True)
                ttnn.deallocate(logits)
                widths[str(rows)] = {
                    "live_rows": rows,
                    "bound_live_rows_x_top_k": min(gen.model.layers[1].n_experts, rows * gen.model.layers[1].top_k),
                    "moe_layers_sampled": len(seen),
                    "union_width_min": min(seen),
                    "union_width_median": int(statistics.median(seen)),
                    "union_width_max": max(seen),
                    "union_width_mean": round(statistics.fmean(seen), 2),
                }
                print(json.dumps(widths[str(rows)]), flush=True)
        finally:
            OptimizedDecoder._routing_weights_decode = original

        # Same measurement under the synthetic activations the single-layer
        # probes use, on one real MoE layer, so the two are comparable.
        cfg = utils.hf_config()
        idx_layer = utils.LAYER_KINDS["moe"]
        synth_layer = OptimizedDecoder.from_state_dict(
            utils.load_real_layer_state_dict(cfg, idx_layer),
            hf_config=cfg,
            layer_idx=idx_layer,
            mesh_device=dev,
            max_batch_size=MAX_BATCH,
            max_context=1024,
            prefill_chunk_size=1024,
        )
        torch.manual_seed(0)
        xs = ttnn.from_torch(
            torch.randn(1, 1, MAX_BATCH, synth_layer.hidden) * 0.02,
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        r = synth_layer._routing_weights_decode(xs, None)
        host = ttnn.to_torch(r).float()
        synth_width = int((host.max(dim=2).values > 0).sum())
        print("synthetic randn activations, 32 rows, union width:", synth_width, flush=True)

        OUT.write_text(
            json.dumps(
                {
                    "purpose": (
                        "Distinct routed experts in the decode union as a function of live rows, measured on the "
                        "real 47-layer model's eager decode with real activations, plus the same measurement "
                        "under the synthetic torch.randn activations the single-layer probes use. This is the "
                        "number that explains why compact kc = n_experts loses to the union path on the real "
                        "model even though the single-layer probes show the opposite sign."
                    ),
                    "method": (
                        "OptimizedDecoder._routing_weights_decode wrapped on the EAGER (untraced) decode path -- "
                        "the compatibility path, never the measured serving path -- so the union mask can be read "
                        "back per MoE layer. 46 MoE layers sampled per step."
                    ),
                    "n_experts": gen.model.layers[1].n_experts,
                    "top_k": gen.model.layers[1].top_k,
                    "real_activations_by_live_rows": widths,
                    "synthetic_randn_activations_32_rows_union_width": synth_width,
                },
                indent=2,
            )
            + "\n"
        )
        print("WROTE", OUT, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
