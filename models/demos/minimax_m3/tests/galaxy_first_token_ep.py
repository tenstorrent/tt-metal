# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
EP=32 END-TO-END: real-weights prefill of 4 prompts at once on (4,8), verify prompt 0.

Config B: DP attention (each mesh ROW = one prompt, full attention, TP=8 cols) + EP=32
shared MoE (256 experts spread 8/chip across 32; dispatch routes all 4 prompts' tokens).
Prompt 0 = the oracle prompt ("The capital of France is"); we verify its first token
matches the saved HF oracle (argmax 758 'The'). Prompts 1-3 just sanity (finite + decoded).

Run:
  cd /data/vmelnykov/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/data/vmelnykov/MiniMax-M2
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8_mesh_graph_descriptor.textproto
  python3 models/demos/minimax_m3/tests/galaxy_first_token_ep.py
"""

import sys

import torch

import ttnn


def main():
    from transformers import AutoConfig, AutoTokenizer

    from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
    from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import open_mesh_device
    from models.demos.minimax_m3.config import MeshConfig, ModeConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    ORACLE = "The capital of France is"
    prompts = [ORACLE, "Once upon a time", "The meaning of life is", "Water boils at"]
    shape = (4, 8)
    print(f"[ft-ep] mesh={shape} DP=4 (4 prompts) + EP=32. prompt0={ORACLE!r}", flush=True)

    mesh = open_mesh_device(shape, MiniMaxM27Config)
    print(f"[ft-ep] mesh opened {tuple(mesh.shape)}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)

        # tokenize 4 prompts (chat template), pad all to a common 32-multiple
        ids_list, real_lens = [], []
        for p in prompts:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
            ids_list.append(ids)
            real_lens.append(len(ids))
        seq = ((max(real_lens) + 31) // 32) * 32
        toks = torch.zeros(4, seq, dtype=torch.int32)
        for r, ids in enumerate(ids_list):
            toks[r, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        print(f"[ft-ep] real_lens={real_lens} padded_seq={seq}", flush=True)

        print("[ft-ep] loading real weights + EP placement (slow first run) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        mesh_config = MeshConfig(shape, decode=ModeConfig(tp=8, ep=4))
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        model = Model(
            mesh_device=mesh,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=cache,
            create_kv_cache=False,
            max_local_batch_size=1,
            users_row_sharded=True,
            use_ep_moe=True,
            ep_seq_len_per_chip=seq,
        )
        del state_dict
        print("[ft-ep] model built (DP-attn + EP=32); running prefill ...", flush=True)

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
        print("[ft-ep] prefill complete", flush=True)

        dts = ttnn.get_device_tensors(logits)
        nc = shape[1]
        V = hf_config.vocab_size
        for r in range(shape[0]):
            row = torch.cat([ttnn.to_torch(dts[r * nc + c]) for c in range(nc)], dim=-1).float()
            row = row.reshape(-1, row.shape[-1])
            vec = row[real_lens[r] - 1][:V]
            tid = int(vec.argmax())
            tag = "  <-- ORACLE prompt" if r == 0 else ""
            print(f"[ft-ep] prompt{r} first token id={tid} -> {tok.decode([tid])!r}{tag}", flush=True)
            if r == 0:
                ok = tid == 758
                print(
                    f"[ft-ep] ORACLE CHECK: got {tid} ({tok.decode([tid])!r}), expected 758 ('The') -> {'MATCH' if ok else 'MISMATCH'}",
                    flush=True,
                )
        print("[ft-ep] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
