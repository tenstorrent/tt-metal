#!/usr/bin/env python3
"""Minimal tt_transformers prefill GO/NO-GO on a mesh with dummy weights.

Decision gate:
  PASS -> base tt_transformers prefill works -> proceed to device validation
  HANG -> investigate the mesh/fabric path before relying on the NPU demo

Run from the repository root (defaults to a 1x2 mesh):
  REPRO_SEQ_LEN=128 timeout 600 python models/demos/llava/tools/prefill_go_no_go.py
Single-chip isolation: REPRO_MESH=1,1
"""

import os
import sys
import time

import torch
from loguru import logger

import ttnn

SEQ_LEN = int(os.getenv("REPRO_SEQ_LEN", "128"))
N_LAYERS = int(os.getenv("REPRO_N_LAYERS", "1"))
MESH = tuple(int(x) for x in os.getenv("REPRO_MESH", "1,2").split(","))
HANG_BUDGET_S = int(os.getenv("REPRO_HANG_BUDGET_S", "180"))


def main() -> int:
    sd = os.environ.get("TT_METAL_SLOW_DISPATCH_MODE", "unset")
    logger.info(f"[repro] slow_dispatch={sd} seq_len={SEQ_LEN} n_layers={N_LAYERS} mesh={MESH}")

    mesh_device = None
    try:
        t0 = time.time()
        # CCL collectives need the fabric; the dispatch-only auto-enable at
        # open is not sufficient for the workload path. Set it before open.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH))
        logger.info(f"[repro] mesh open: {mesh_device.shape} in {time.time() - t0:.1f}s")

        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import ModelArgs

        model_args = ModelArgs(
            mesh_device=mesh_device,
            instruct=False,
            dummy_weights=True,
            max_batch_size=1,
            max_seq_len=1024,
        )
        model_args.n_layers = N_LAYERS
        logger.info(
            f"[repro] ModelArgs: model={model_args.model_name} dim={model_args.dim} "
            f"heads={model_args.n_heads}/{model_args.n_kv_heads} head_dim={model_args.head_dim} "
            f"vocab={model_args.vocab_size} layers={model_args.n_layers} "
            f"moe={getattr(model_args, 'is_mixture_of_experts', False)} "
            f"moe_layer_ids={getattr(model_args, 'moe_layer_ids', None)}"
        )

        dtype = ttnn.bfloat16
        t0 = time.time()
        state_dict = model_args.load_state_dict()
        model = Transformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=None,
            paged_attention_config=PagedAttentionConfig(block_size=32, max_num_blocks=1024),
        )
        logger.info(f"[repro] Transformer built in {time.time() - t0:.1f}s")

        kv_cache = [layer.attention.layer_past for layer in model.layers]
        page_table = torch.arange(1024, dtype=torch.int32).unsqueeze(0)
        tokens = torch.randint(1, model_args.vocab_size, (1, SEQ_LEN))

        t0 = time.time()
        x, rot_global, rot_local, tt_page_table, _, _ = model.prepare_inputs_prefill(tokens, page_table=page_table)
        logger.info(f"[repro] prepare_inputs_prefill done in {time.time() - t0:.2f}s")

        t0 = time.time()
        tt_out = model.ttnn_prefill_forward(
            x,
            rot_global,
            rot_local,
            page_table=tt_page_table,
            kv_cache=kv_cache,
        )
        elapsed = time.time() - t0
        if elapsed > HANG_BUDGET_S:
            logger.error(f"[repro] NO-GO: prefill returned after {elapsed:.1f}s (> {HANG_BUDGET_S}s budget)")
            return 1

        logits = model.process_output_prefill(tt_out.cpu(), last_token_idx=SEQ_LEN - 1)
        logger.info(
            f"[repro] prefill logits: shape={tuple(logits.shape)} dtype={logits.dtype} "
            f"finite={torch.isfinite(logits).all().item()}"
        )
        logger.info(f"[repro] prefill+output in {elapsed:.1f}s")
        logger.info("[repro] GO — prefill works on this mesh")
        return 0
    except Exception:
        logger.exception("[repro] NO-GO: prefill repro crashed")
        return 2
    finally:
        if mesh_device is not None:
            try:
                ttnn.close_mesh_device(mesh_device)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                logger.warning("[repro] close_mesh_device failed (ETH state may be dirty)")


if __name__ == "__main__":
    sys.exit(main())
