# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Step 1e — REAL-weights prefill -> first token, at TP=8 on (1,8).

Loads the dequantized MiniMax-M2.7 bf16 checkpoint (HF_MODEL), builds the full
Model (experts bfp4 by default), embeds a real prompt, runs one ttnn_prefill_forward,
and decodes the argmax first token + top-5. PREFILL ONLY — this is the first token,
not a full answer (decode lives in tt-blaze).

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/data/vmelnykov/MiniMax-M2
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_1x8_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m2/tests/galaxy_first_token.py --prompt "The capital of France is"
"""

import argparse
import gc
import sys

import torch

import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument(
        "--dump-logits", type=str, default=None, help="path to save last-pos logits (.npy) for PCC vs HF oracle"
    )
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    from models.demos.minimax_m2.config import MeshConfig, ModeConfig
    from models.demos.minimax_m2.tt.ccl import CCLManager
    from models.demos.minimax_m2.tt.model import Model
    from models.demos.minimax_m2.tt.model_config import ModelArgs
    from models.demos.minimax_m2.utils.general_utils import get_default_num_links

    shape = (args.rows, args.cols)
    print(f"[first-token] mesh={shape} TP={args.cols} prompt={args.prompt!r}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
    print(f"[first-token] mesh opened: {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)

    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

        print("[first-token] loading + converting real weights (host ~426GB, slow first run) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)

        mesh_config = MeshConfig(shape, decode=ModeConfig(tp=shape[1], ep=shape[0]))
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        model = Model(
            mesh_device=mesh,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=cache,
            create_kv_cache=True,
            max_local_batch_size=1,
        )
        del state_dict
        gc.collect()
        print("[first-token] model built; weights on device", flush=True)

        # Tokenize (chat template if available, else plain) and pad seq to a 32-multiple.
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}], add_generation_prompt=True, tokenize=True
            )
        except Exception:
            ids = tok(args.prompt)["input_ids"]
        real_len = len(ids)
        seq = ((real_len + 31) // 32) * 32
        padded = ids + [tok.pad_token_id or 0] * (seq - real_len)
        print(f"[first-token] prompt_tokens={real_len} padded_seq={seq}", flush=True)

        tokens = ttnn.from_torch(
            torch.tensor(padded, dtype=torch.int32).reshape(1, 1, 1, seq),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tokens.deallocate(True)
        emb = ttnn.unsqueeze_to_4D(emb)

        last = ((real_len - 1) // 32) * 32
        logits = model.ttnn_prefill_forward(emb, get_last_token=last)
        print("[first-token] forward complete", flush=True)

        full = ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=mesh.shape, dims=(-2, -1))
        ).float()
        row = full.reshape(-1, full.shape[-1])[real_len - 1 - last][: hf_config.vocab_size]
        if args.dump_logits:
            import numpy as np

            np.save(args.dump_logits, row.numpy())
            print(f"[first-token] dumped last-pos logits -> {args.dump_logits}", flush=True)
        top = torch.topk(row, 5)
        tok_id = int(top.indices[0])
        print(f"[first-token] FIRST TOKEN id={tok_id} -> {tok.decode([tok_id])!r}", flush=True)
        print("[first-token] top-5:", [(int(i), tok.decode([int(i)])) for i in top.indices], flush=True)
        print("[first-token] PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
