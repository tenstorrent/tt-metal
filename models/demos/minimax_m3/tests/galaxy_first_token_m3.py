# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
M3 REAL-WEIGHTS first token on the galaxy: full 60-layer model, TP=4 + EP=32 + DP (one prompt per
mesh row), prefill a short prompt, print the argmax first token per prompt.

Layout: (8,4) = TP=4 (4 cols, forced by 4 KV heads) + DP=8 (8 rows, one prompt each) + EP=32 (128
experts / 4 per device). Full-GQA (S<2048, no MSA yet). Uses the Step A real-weights load path
(text_config unwrap + bf16 safetensors, language_model. strip, partial-rotary swizzle); weights
mmap from disk so host RAM stays low; experts quantize to bf4/bf8 on device (+ disk weight cache).

GOLDEN: TBD — for now we print + decode the first token (coherence). Add an expected-token assert
for the oracle prompt once we settle the golden (known/published token vs HF-on-a-big-box).

Run (after the 869GB download completes):
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m3/tests/galaxy_first_token_m3.py
"""

import sys

import torch

import ttnn

ORACLE = "The capital of France is"
PROMPTS = [
    ORACLE,
    "Once upon a time",
    "The meaning of life is",
    "Water boils at",
    "The opposite of hot is",
    "Two plus two equals",
    "The sun rises in the",
    "Roses are red, violets are",
]


def main():
    from models.demos.minimax_m3.config import MeshConfig, ModeConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    rows, cols = 8, 4  # TP=4 (cols), DP=8 (rows), EP=32
    print(f"[ft-m3] mesh=({rows},{cols}) TP={cols} DP={rows} EP=32; oracle={ORACLE!r}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    print(f"[ft-m3] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL -> /data/vmelnykov/MiniMax-M3-ref
        hf_config = model_args.hf_config  # Step A: unwrapped text_config
        tok = model_args.tokenizer

        # tokenize one prompt per row, pad to a common 32-multiple
        ids_list = [
            tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
            for p in PROMPTS[:rows]
        ]
        real_lens = [len(ids) for ids in ids_list]
        seq = ((max(real_lens) + 31) // 32) * 32
        toks = torch.zeros(rows, seq, dtype=torch.int32)
        for r, ids in enumerate(ids_list):
            toks[r, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        print(f"[ft-m3] real_lens={real_lens} padded_seq={seq}", flush=True)

        print("[ft-m3] loading real bf16 weights (mmap) + EP placement (slow first run, builds cache) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=cols, ep=rows))
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        model = Model(
            mesh_device=mesh,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=cache,
            max_local_batch_size=1,
            users_row_sharded=True,
            use_ep_moe=True,
            ep_seq_len_per_chip=seq,
        )
        del state_dict
        print("[ft-m3] model built (60L, TP=4 + EP=32); running prefill ...", flush=True)

        host_out = model.prepare_inputs_prefill(toks, page_table=None, batched_prefill=True)
        logits = model.ttnn_prefill_forward(
            host_out[0],
            rot_mats_global=host_out[1],
            rot_mats_local=host_out[2],
            page_table=host_out[3],
            kv_cache=None,
            batch_size=1,
            get_last_token=-1,
        )
        ttnn.synchronize_device(mesh)
        print("[ft-m3] prefill complete", flush=True)

        dts = ttnn.get_device_tensors(logits)
        V = hf_config.vocab_size
        for r in range(rows):
            row = torch.cat([ttnn.to_torch(dts[r * cols + c]) for c in range(cols)], dim=-1).float()
            row = row.reshape(-1, row.shape[-1])
            vec = row[real_lens[r] - 1][:V]
            tid = int(vec.argmax())
            tag = "   <-- ORACLE" if r == 0 else ""
            print(f"[ft-m3] prompt{r} {PROMPTS[r]!r} -> first token id={tid} {tok.decode([tid])!r}{tag}", flush=True)
        print("[ft-m3] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
