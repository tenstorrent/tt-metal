#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE REAL tt_transformers Llama DECODE step of a SINGLE transformer layer under ttnn graph
capture, dump JSON for raw_hazard_analyzer.py. Single Wormhole chip (1x1 mesh / N150).

This is the REAL decode path (NOT the random-op proxy): a models/tt_transformers TransformerBlock
run in Mode.DECODE with paged attention, KV cache, page table, and RoPE rot_mats -- exactly the setup
in models/tt_transformers/tests/test_decoder.py, reduced to n_layers=1, batch=1, one decode() call.

Weights: dummy_weights=True builds a RANDOM-weight 1-layer model from the LOCAL config in
models/tt_transformers/model_params/Llama-3.2-1B-Instruct (no checkpoint download; the device-program
dependency/hazard structure is independent of weight VALUES -- shapes/config are what matter).
"""
import json
import os
import sys

# Fully offline: local config only, never hit the HF hub.
os.environ.setdefault("HF_MODEL", "unsloth/Llama-3.2-1B-Instruct")  # basename -> model_name "Llama-3.2-1B-Instruct"
os.environ.setdefault("MESH_DEVICE", "N150")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from tests.scripts.common import get_updated_device_params  # noqa: E402
from models.tt_transformers.tt.ccl import TT_CCL  # noqa: E402
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig  # noqa: E402
from models.tt_transformers.tt.decoder import TransformerBlock  # noqa: E402
from models.tt_transformers.tt.model_config import ModelArgs  # noqa: E402
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup  # noqa: E402

OUT = "/tmp/tt_transformers_decode_capture.json"
BATCH_SIZE = 1
MAX_SEQ_LEN = 256
PAGE_BLOCK_SIZE = 32
PAGE_MAX_NUM_BLOCKS = 1024


def _set_fabric(fabric_config):
    # Mirror conftest.set_fabric(fabric_config=True) for the decoder test's device_params.
    if fabric_config:
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )


def _reset_fabric(fabric_config):
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    # ---- open a single-chip (1x1) mesh ----
    # The decoder test parametrizes fabric_config=True for generality, but fabric cannot be launched on a
    # 1x1 SUBSET of this multi-device host. For a single-device decode the CCL ops are no-ops, so we open a
    # plain 1x1 mesh with no fabric (dispatch_core_config from get_updated_device_params).
    device_params = {}
    updated = get_updated_device_params(device_params)
    fabric_config = None
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), **updated)

    try:
        dtype = ttnn.bfloat8_b
        mode = Mode.DECODE

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            cache_hf=True,
            dummy_weights=True,  # random weights from LOCAL config -> fully offline
            use_hf_rope=False,
        )
        model_args.n_layers = 1
        logger.info(
            f"ModelArgs: {model_args.model_name} dim={model_args.dim} n_heads={model_args.n_heads} "
            f"n_kv_heads={model_args.n_kv_heads} head_dim={model_args.head_dim}"
        )

        state_dict = model_args.load_state_dict()  # dummy -> random-weight 1-layer state_dict

        # ---- RoPE transformation matrices (decode) ----
        DefaultRopeSetup = HfRotarySetup if model_args.use_hf_rope else RotarySetup
        rope_setup = DefaultRopeSetup(
            mesh_device,
            model_args.max_batch_size,
            model_args.head_dim,
            model_args.max_seq_len,
            model_args.rope_theta,
            model_args.rope_scaling,
            model_args.use_qk_fused,
        )
        if model_args.rope_theta_local is not None:
            rope_setup_local = RotarySetup(
                mesh_device,
                model_args.max_batch_size,
                model_args.head_dim,
                model_args.max_seq_len,
                model_args.rope_theta_local,
                None,
            )
        else:
            rope_setup_local = None
        transformation_mats = rope_setup.get_both_trans_mats()

        # ---- paged-attention page table ----
        paged_attention_config = PagedAttentionConfig(
            block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=PAGE_MAX_NUM_BLOCKS,
        )
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        # ---- build the TT TransformerBlock (single layer, layer 0) ----
        tt_ccl = TT_CCL(mesh_device)
        tt_model = TransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            layer_num=0,
            weight_cache_path=model_args.weight_cache_path(dtype),
            transformation_mats=transformation_mats,
            paged_attention_config=paged_attention_config,
        )

        seqlen = 1
        generation_start_pos = 0
        current_pos = torch.tensor([generation_start_pos for _ in range(BATCH_SIZE)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        # ---- decode-mode sharded input activation ----
        pt_decode_input = (torch.rand(BATCH_SIZE, seqlen, model_args.dim) * 2) - 1
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input,
            model_args.get_residual_mem_config(mode, None),
        )
        rot_mats = rope_setup.get_rot_mats(current_pos)
        rot_mats_local = None if rope_setup_local is None else rope_setup_local.get_rot_mats(current_pos)

        # Warm up program cache OUTSIDE capture so the graph reflects steady-state decode dispatch
        # (first invocation compiles/allocs; we want the second, cached invocation's op graph).
        logger.info("Warm-up decode invocation (outside capture)...")
        _ = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=mode,
            page_table=page_table_tt,
        )
        ttnn.synchronize_device(mesh_device)

        # Rebuild a fresh input tensor (the previous one may have been consumed/deallocated).
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input,
            model_args.get_residual_mem_config(mode, None),
        )

        logger.info("Capturing ONE decode step...")
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=mode,
            page_table=page_table_tt,
        )
        ttnn.synchronize_device(mesh_device)
        captured = ttnn.graph.end_graph_capture()

        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}")
        ttnn.deallocate(tt_out)
    finally:
        ttnn.close_mesh_device(mesh_device)
        _reset_fabric(fabric_config)


if __name__ == "__main__":
    sys.exit(main())
