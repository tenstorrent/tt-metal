# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
M3 REAL-WEIGHTS multi-token generation (coherence check): full 60-layer model on (8,4) =
TP=4 + EP=32 + DP=8, greedy-generate a few tokens per prompt and DECODE to text.

Prefill-only model (no KV-cache decode), so generation = repeated prefill: prefill -> argmax the
last real token -> append -> re-prefill. Model is built ONCE (loads the bf4/bf8 weight cache) and
the forward loops NUM_GEN times. Causal masking ignores the trailing pad, so the growing sequence
is handled by re-prefilling with the appended tokens.

EXPERT_DTYPE env: "bf4" (default; reuses the existing cache, fast) or "bf8" (higher precision, but
needs a fresh ~2h cache rebuild). Coherence check — eyeball the decoded text (no HF golden here).

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/data/vmelnykov/MiniMax-M3-ref
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  export EXPERT_DTYPE=bf4          # or bf8
  python3 models/demos/minimax_m3/tests/galaxy_generate_m3.py
"""

import os
import sys

import torch

import ttnn

NUM_GEN = 6
PROMPTS = [
    "The capital of France is",
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

    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    rows, cols = 8, 4
    print(
        f"[gen-m3] mesh=({rows},{cols}) TP={cols} DP={rows} EP=32 | expert_dtype={expert_dtype} | gen={NUM_GEN}",
        flush=True,
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        tok = model_args.tokenizer
        V = hf_config.vocab_size

        ids_list = [
            tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
            for p in PROMPTS[:rows]
        ]
        real_lens = [len(ids) for ids in ids_list]
        seq = ((max(real_lens) + NUM_GEN + 31) // 32) * 32
        toks = torch.zeros(rows, seq, dtype=torch.int32)
        for r, ids in enumerate(ids_list):
            toks[r, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        print(f"[gen-m3] real_lens={real_lens} padded_seq={seq}", flush=True)

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
            expert_weight_dtype=expert_dtype,
        )
        del state_dict
        print("[gen-m3] model built; generating ...", flush=True)

        gen_tokens = [[] for _ in range(rows)]
        for g in range(NUM_GEN):
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
            dts = ttnn.get_device_tensors(logits)
            for r in range(rows):
                row = torch.cat([ttnn.to_torch(dts[r * cols + c]) for c in range(cols)], dim=-1).float()
                row = row.reshape(-1, row.shape[-1])
                pred_pos = real_lens[r] + g - 1
                nxt = int(row[pred_pos][:V].argmax())
                gen_tokens[r].append(nxt)
                toks[r, real_lens[r] + g] = nxt
            print(f"[gen-m3] step {g+1}/{NUM_GEN} done (tokens: {[gt[-1] for gt in gen_tokens]})", flush=True)

        print("\n[gen-m3] ===== GENERATED (greedy, coherence check) =====", flush=True)
        for r in range(rows):
            text = tok.decode(gen_tokens[r])
            print(f"[gen-m3] {PROMPTS[r]!r}\n          -> ids={gen_tokens[r]}\n          -> {text!r}\n", flush=True)
        print("[gen-m3] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
