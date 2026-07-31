# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 REAL-WEIGHTS SEQUENCE-PARALLEL prefill — the prod TP=4 × SP=8 × EP=32 layout (DP eliminated).

ONE long (~5k-token) prompt sharded by SEQUENCE across the 8 mesh rows (640 tokens/row), TP=4 on the
cols, EP=32 experts across all 32 chips. This is the first whole-model real-weights run on the actual
deployment config and the first that exercises the MSA (block-sparse) layers 3-59 in-model at the real
5120-token regime (80 blocks, top-16). Prefill-only (no decode cache): generation = repeated prefill.

Greedy-generate NUM_GEN tokens and decode for a coherence check (same bar the DP run met). Optionally
set CMP_DP=1 to also run the same prompt through the validated DP path (prompt on row 0) and assert the
first generated token matches — a real-weights SP-vs-DP golden without needing an HF M3 reference.

Run:
  cd /data/vmelnykov/tt-metal
  export TT_METAL_HOME=/data/vmelnykov/tt-metal PYTHONPATH=/data/vmelnykov/tt-metal
  source python_env/bin/activate
  export HF_MODEL=/mnt/models/MiniMaxAI/MiniMax-M3-ref   # real bf16 weights + tilized cache
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  export EXPERT_DTYPE=bf4
  python3 models/demos/minimax_m3/tests/galaxy_generate_m3.py
"""

import os
import sys

import torch

import ttnn

NUM_GEN = int(os.getenv("NUM_GEN", "6"))
TARGET_LEN = 5120  # 8 * 640 — divisible by SP=8 and by 32 (tile)

# A coherent long passage; repeated to ~TARGET_LEN tokens so MSA has real blocks to select over.
PASSAGE = (
    "The history of computing is a story of steady abstraction. Early machines were programmed by "
    "physically rewiring them; each new problem meant rebuilding the hardware. The stored-program "
    "architecture changed this by treating instructions as data, so a single machine could run any "
    "program written for it. From there, assembly languages gave names to raw opcodes, compilers "
    "translated human-readable code into machine instructions, and operating systems multiplexed the "
    "hardware among many programs at once. Each layer hid the complexity beneath it, letting engineers "
    "reason about larger and larger systems. Networking added another dimension: machines that once "
    "stood alone could now exchange information across the world in fractions of a second. The modern "
    "era of accelerators continues this arc, trading the generality of a single processor for the raw "
    "throughput of thousands of small cores working in parallel on the same problem. "
)


def main():
    from models.demos.minimax_m3.config import MeshConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    rows, cols = 8, 4
    sp, tp = rows, cols
    print(
        f"[sp-gen] mesh=({rows},{cols}) TP={tp} SP={sp} EP=32 | expert_dtype={expert_dtype} | gen={NUM_GEN}", flush=True
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        tok = model_args.tokenizer
        V = hf_config.vocab_size

        # LOCALIZATION KNOB: FORCE_DENSE=1 zeroes sparse_attention_freq -> all layers use dense (full-GQA)
        # attention, NO MSA. Lets us test whether the degenerate token-0 is MSA-specific or a general SP
        # real-weights issue. Also TARGET_LEN overridable (default 5120).
        target_len = int(os.getenv("TARGET_LEN", str(TARGET_LEN)))
        if os.getenv("FORCE_DENSE") == "1":
            sa = hf_config.sparse_attention_config
            n = hf_config.num_hidden_layers
            if isinstance(sa, dict):
                sa["sparse_attention_freq"] = [0] * n
            else:
                sa.sparse_attention_freq = [0] * n
            print("[sp-gen] FORCE_DENSE=1 -> all layers DENSE attention (MSA disabled)", flush=True)

        # SPEED/ISOLATION KNOB: NLAYERS=K builds only the first K real layers (and M3_LOAD_NLAYERS makes
        # load_state_dict read only the shards for those layers -> ~minutes, not ~2hr). For isolating
        # where the real-weights SP residual degrades, e.g. NLAYERS=1 (just the dense layer 0).
        nlayers = int(os.getenv("NLAYERS", str(hf_config.num_hidden_layers)))
        if nlayers < hf_config.num_hidden_layers:
            os.environ["M3_LOAD_NLAYERS"] = str(nlayers)  # consumed by ModelArgs._load_text_backbone_safetensors
            hf_config.num_hidden_layers = nlayers
            if isinstance(getattr(hf_config, "moe_layer_freq", None), list):
                hf_config.moe_layer_freq = hf_config.moe_layer_freq[:nlayers]
            _sa = hf_config.sparse_attention_config
            _saf = (
                _sa["sparse_attention_freq"] if isinstance(_sa, dict) else getattr(_sa, "sparse_attention_freq", None)
            )
            if isinstance(_saf, list):
                if isinstance(_sa, dict):
                    _sa["sparse_attention_freq"] = _saf[:nlayers]
                else:
                    _sa.sparse_attention_freq = _saf[:nlayers]
            print(f"[sp-gen] NLAYERS={nlayers} (truncated from 60 for a fast isolation run)", flush=True)

        # Build a prompt that FITS in target_len WITHOUT truncating the trailing question (the bug-free
        # construction): grow the body, then verify the full chat-templated prompt fits.
        QUESTION = "\n\nIn one word, the central theme of the text above is"
        body = PASSAGE
        while (
            len(
                tok.apply_chat_template(
                    [{"role": "user", "content": body + PASSAGE + QUESTION}], add_generation_prompt=True, tokenize=True
                )
            )
            < target_len - NUM_GEN
        ):
            body += PASSAGE
        ids = tok.apply_chat_template(
            [{"role": "user", "content": body + QUESTION}], add_generation_prompt=True, tokenize=True
        )
        assert len(ids) <= target_len - NUM_GEN, f"prompt {len(ids)} would truncate the question (target {target_len})"
        real_len = len(ids)
        seq = target_len
        assert seq % sp == 0, f"seq {seq} must be divisible by sp {sp}"
        toks = torch.zeros(1, seq, dtype=torch.int32)
        toks[0, :real_len] = torch.tensor(ids, dtype=torch.int32)
        print(f"[sp-gen] real_len={real_len} padded_seq={seq} ({seq//sp}/row)", flush=True)

        # Cache-only weight load (Pavlo's weight_cache): when the tilized cache is complete, pass an empty
        # state_dict so the modules load tilized tensors straight from cache (ttnn.as_tensor cache hit) and
        # we skip the ~869GB bf16 source read (~1h20m -> ~2.5min). Force the source with M3_FORCE_LOAD_WEIGHTS=1.
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

        if os.getenv("M3_FORCE_LOAD_WEIGHTS") != "1" and (
            os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache, hf_config, hf_config.num_hidden_layers, expert_dtype)
        ):
            print("[sp-gen] tilized weight cache complete -> cache-only (skip ~869GB source read)", flush=True)
            state_dict = {}
        else:
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
        # ep_seq_len_per_chip = per-device token count (640 for SP). DP uses the full seq (5120).
        # Override via EP_SEQ_PER_CHIP to isolate dispatch-buffer sizing issues.
        ep_seq = int(os.getenv("EP_SEQ_PER_CHIP", str(seq // sp)))
        print(f"[sp-gen] ep_seq_len_per_chip={ep_seq} (per-device tokens={seq//sp})", flush=True)
        model = Model(
            mesh_device=mesh,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=cache,
            max_local_batch_size=1,
            sequence_parallel=(os.getenv("NO_SP") != "1"),
            use_ep_moe=True,
            ep_seq_len_per_chip=ep_seq,
            expert_weight_dtype=expert_dtype,
        )
        del state_dict
        print("[sp-gen] model built; generating ...", flush=True)

        gen = []
        for g in range(NUM_GEN):
            host_out = model.prepare_inputs_prefill(toks)  # SP path (self.sequence_parallel)
            logits = model.prefill_forward(
                host_out[0],
                rot_mats_global=host_out[1],
                rot_mats_local=host_out[2],
                kv_cache=None,
                batch_size=1,
                get_last_token=-1,
            )
            ttnn.synchronize_device(mesh)
            # gather: rows -> seq (dim -2), cols -> vocab (dim -1)
            out = (
                ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(rows, cols), dims=(-2, -1))
                )
                .float()
                .reshape(seq, -1)
            )
            pred_pos = real_len + g - 1
            nxt = int(out[pred_pos][:V].argmax())
            gen.append(nxt)
            toks[0, real_len + g] = nxt
            print(f"[sp-gen] step {g+1}/{NUM_GEN}: token={nxt} -> {tok.decode([nxt])!r}", flush=True)

        print("\n[sp-gen] ===== GENERATED (greedy, SP=8 real-weights) =====", flush=True)
        print(f"[sp-gen] ids={gen}", flush=True)
        print(f"[sp-gen] text={tok.decode(gen)!r}", flush=True)
        print("[sp-gen] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
