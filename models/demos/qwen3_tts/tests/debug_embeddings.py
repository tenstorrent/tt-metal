# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug script: compare TT embedding construction vs HF reference."""

import os
import sys
import torch
import numpy as np

os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + os.path.dirname(__file__) + "/../../../.."

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEXT = "こんにちは"
LANGUAGE = "japanese"


def build_hf_embeddings():
    """Build input embeddings by loading just the Talker's weights directly."""
    import json
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    # Load config
    config_path = hf_hub_download(MODEL_PATH, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)

    tc = raw_config["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load relevant weights from safetensors
    print("Loading weights from safetensors...")
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(MODEL_PATH)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])

    # Load all talker embedding/projection weights
    needed_keys = {
        "talker.model.text_embedding.weight",
        "talker.model.codec_embedding.weight",
        "talker.text_projection.linear_fc1.weight",
        "talker.text_projection.linear_fc1.bias",
        "talker.text_projection.linear_fc2.weight",
        "talker.text_projection.linear_fc2.bias",
    }
    weights = {}
    for shard in shard_files:
        fpath = hf_hub_download(MODEL_PATH, shard)
        shard_dict = load_file(fpath)
        for k in list(shard_dict.keys()):
            if k in needed_keys:
                weights[k] = shard_dict[k]
                print(f"  Loaded {k}: {shard_dict[k].shape}")
        if len(weights) == len(needed_keys):
            break

    # Build embedding functions
    text_embed_weight = weights["talker.model.text_embedding.weight"]  # [151936, text_hidden_size]
    codec_embed_weight = weights["talker.model.codec_embedding.weight"]  # [3072, dim]
    tp_fc1_w = weights["talker.text_projection.linear_fc1.weight"]
    tp_fc1_b = weights["talker.text_projection.linear_fc1.bias"]
    tp_fc2_w = weights["talker.text_projection.linear_fc2.weight"]
    tp_fc2_b = weights["talker.text_projection.linear_fc2.bias"]

    print(f"text_embed_weight: {text_embed_weight.shape}")
    print(f"codec_embed_weight: {codec_embed_weight.shape}")
    print(f"text_proj fc1: {tp_fc1_w.shape}, fc2: {tp_fc2_w.shape}")

    def text_embed_fn(ids):
        return torch.nn.functional.embedding(ids, text_embed_weight)

    def codec_embed_fn(ids):
        return torch.nn.functional.embedding(ids, codec_embed_weight)

    def text_proj_fn(x):
        h = torch.nn.functional.linear(x, tp_fc1_w, tp_fc1_b)
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, tp_fc2_w, tp_fc2_b)
        return h

    # Tokenize
    formatted = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([tokenizer.encode(formatted)])
    print(f"\nToken IDs: {input_ids[0].tolist()}")
    print(f"First 3 (role): {input_ids[0, :3].tolist()}")
    print(f"Middle (text): {input_ids[0, 3:-5].tolist()}")
    print(f"Last 5 (suffix): {input_ids[0, -5:].tolist()}")

    config_tts_bos = raw_config.get("tts_bos_token_id", 151672)
    config_tts_eos = raw_config.get("tts_eos_token_id", 151673)
    config_tts_pad = raw_config.get("tts_pad_token_id", 151671)

    # Special TTS tokens
    tts_special_ids = torch.tensor([[config_tts_bos, config_tts_eos, config_tts_pad]])
    tts_bos_embed, tts_eos_embed, tts_pad_embed = text_proj_fn(
        text_embed_fn(tts_special_ids)
    ).chunk(3, dim=1)

    print(f"\ntts_bos_embed: shape={tts_bos_embed.shape}, norm={tts_bos_embed.float().norm():.4f}")
    print(f"tts_eos_embed: shape={tts_eos_embed.shape}, norm={tts_eos_embed.float().norm():.4f}")
    print(f"tts_pad_embed: shape={tts_pad_embed.shape}, norm={tts_pad_embed.float().norm():.4f}")

    # Codec tag
    language_id = tc["codec_language_id"][LANGUAGE.lower()]
    codec_tag_ids = [[tc["codec_think_id"], tc["codec_think_bos_id"], language_id, tc["codec_think_eos_id"]]]
    codec_tag = codec_embed_fn(torch.tensor(codec_tag_ids))
    codec_suffix_ids = [[tc["codec_pad_id"], tc["codec_bos_id"]]]
    codec_suffix = codec_embed_fn(torch.tensor(codec_suffix_ids))
    codec_prefill = torch.cat([codec_tag, codec_suffix], dim=1)

    print(f"\ncodec_tag tokens: {codec_tag_ids}")
    print(f"codec_tag: shape={codec_tag.shape}, norm={codec_tag.float().norm():.4f}")
    print(f"codec_suffix: shape={codec_suffix.shape}, norm={codec_suffix.float().norm():.4f}")

    # Role prefix
    role_proj = text_proj_fn(text_embed_fn(input_ids[:, :3]))
    print(f"\nrole_proj: shape={role_proj.shape}, norm={role_proj.float().norm():.4f}")
    print(f"role_proj[0,0,:5]: {role_proj[0,0,:5].float().tolist()}")

    # Tag overlay (tts_pad*4 + tts_bos) + codec_prefill[:-1]
    n_pad = codec_prefill.shape[1] - 2  # 4
    text_side_tag = torch.cat([
        tts_pad_embed.expand(-1, n_pad, -1),
        tts_bos_embed,
    ], dim=1)
    part_tag = text_side_tag + codec_prefill[:, :-1, :]
    print(f"\npart_tag: shape={part_tag.shape}, norm={part_tag.float().norm():.4f}")
    print(f"part_tag[0,0,:5]: {part_tag[0,0,:5].float().tolist()}")

    # Non-streaming: text content + tts_eos paired with codec_pad
    text_content = input_ids[:, 3:-5]
    print(f"\ntext_content tokens: {text_content[0].tolist()}")
    text_content_proj = text_proj_fn(text_embed_fn(text_content))
    text_with_eos = torch.cat([text_content_proj, tts_eos_embed], dim=1)
    codec_pad_emb = codec_embed_fn(torch.tensor([[tc["codec_pad_id"]]]))
    codec_pads = codec_pad_emb.expand(-1, text_with_eos.shape[1], -1)
    part_text = text_with_eos + codec_pads
    print(f"part_text: shape={part_text.shape}, norm={part_text.float().norm():.4f}")

    # Final: tts_pad + codec_bos
    part_final = tts_pad_embed + codec_prefill[:, -1:, :]
    print(f"part_final: shape={part_final.shape}, norm={part_final.float().norm():.4f}")

    # Concatenate
    full_embed = torch.cat([role_proj, part_tag, part_text, part_final], dim=1)
    print(f"\nFull HF embedding: shape={full_embed.shape}")
    for i in range(full_embed.shape[1]):
        print(f"  pos {i}: norm={full_embed[0,i].float().norm():.4f}, first5={full_embed[0,i,:5].float().tolist()}")

    return full_embed, 0


def build_tt_embeddings():
    """Build input embeddings using TT pipeline's _build_input_embeds."""
    import ttnn
    from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
    from models.demos.qwen3_tts.tt.talker import TalkerTransformer

    os.environ["HF_MODEL"] = MODEL_PATH

    device_ids = ttnn.get_device_ids()
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        mesh_device.enable_program_cache()
    except AttributeError:
        ttnn.enable_program_cache(mesh_device)

    talker_args = TalkerModelArgs(
        mesh_device=mesh_device, max_batch_size=1, max_seq_len=768, use_hf_rope=True,
    )
    state_dict = talker_args.load_state_dict()
    weight_cache_path = talker_args.weight_cache_path(ttnn.bfloat16)

    talker = TalkerTransformer(
        args=talker_args, dtype=ttnn.bfloat16, mesh_device=mesh_device,
        state_dict=state_dict, weight_cache_path=weight_cache_path,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build a minimal generator-like object to call _build_input_embeds
    from models.demos.qwen3_tts.tt.generator import TTSGenerator
    gen = TTSGenerator.__new__(TTSGenerator)
    gen.talker = talker
    gen.talker_args = talker_args
    gen.mesh_device = mesh_device
    gen.tokenizer = tokenizer

    input_embeds, trailing, tts_pad = gen._build_input_embeds(TEXT, LANGUAGE)
    print(f"\nTT input_embeds: shape={input_embeds.shape}")
    real_len = input_embeds.shape[1]
    norms = input_embeds.squeeze(0).norm(dim=-1)
    for i in range(min(real_len, 20)):
        print(f"  pos {i}: norm={norms[i]:.4f}, first5={input_embeds[0,i,:5].tolist()}")

    ttnn.close_mesh_device(mesh_device)
    return input_embeds


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hf":
        hf_embed, greedy = build_hf_embeddings()
        torch.save(hf_embed, "/home/yito/ttwork/tt-metal/demo_ja_output/hf_embed.pt")
        print(f"\nSaved HF embedding to demo_ja_output/hf_embed.pt")
    elif len(sys.argv) > 1 and sys.argv[1] == "tt":
        tt_embed = build_tt_embeddings()
        torch.save(tt_embed, "/home/yito/ttwork/tt-metal/demo_ja_output/tt_embed.pt")
        print(f"\nSaved TT embedding to demo_ja_output/tt_embed.pt")
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        hf = torch.load("/home/yito/ttwork/tt-metal/demo_ja_output/hf_embed.pt")
        tt = torch.load("/home/yito/ttwork/tt-metal/demo_ja_output/tt_embed.pt")
        print(f"HF shape: {hf.shape}, TT shape: {tt.shape}")
        min_len = min(hf.shape[1], tt.shape[1])
        hf_f = hf[:, :min_len, :].float()
        tt_f = tt[:, :min_len, :].float()
        for i in range(min_len):
            cos_sim = torch.nn.functional.cosine_similarity(hf_f[0, i:i+1], tt_f[0, i:i+1]).item()
            diff = (hf_f[0, i] - tt_f[0, i]).abs().max().item()
            print(f"  pos {i}: cos_sim={cos_sim:.6f}, max_diff={diff:.4f}")
    else:
        print("Usage: python debug_embeddings.py [hf|tt|compare]")
