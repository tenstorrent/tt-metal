# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Strict Gemma4 decode trace harness with device-side feedback.

This is a bringup/perf harness for batch=1 decode.  It keeps the sampled token,
next token embedding, RoPE position, and KV-cache position on device across
TTNN trace replays.  It intentionally prints machine-readable evidence for
RESULTS.md rather than returning generated text.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.gemma4.demo.text_demo import _load_tokenizer
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.model_config import DEFAULT_GEMMA4_MODEL
from models.tt_transformers.tt.common import PagedAttentionConfig


def _set_fabric_1d():
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    except TypeError:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            None,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )


def _mesh_values(tt_tensor, is_mesh, limit=8):
    tensors = ttnn.get_device_tensors(tt_tensor) if is_mesh else [tt_tensor]
    return [ttnn.to_torch(t).flatten().detach().cpu().tolist()[:limit] for t in tensors]


def run(args):
    print("RUN_LABEL gemma4_strict_device_feedback", flush=True)
    print("MODEL", args.model_path, flush=True)
    print("HF_REVISION", args.hf_revision, flush=True)
    print("TT_METAL_CACHE", os.environ.get("TT_METAL_CACHE"), flush=True)
    print("TT_CACHE_PATH", os.environ.get("TT_CACHE_PATH"), flush=True)
    print("LOCAL_REPO", REPO_ROOT, flush=True)
    print("BUILT_REPO_CWD", os.getcwd(), flush=True)
    print(
        "GOAL strict device-side token/position feedback under TTNN trace replay",
        flush=True,
    )

    _set_fabric_1d()
    wall_start = time.perf_counter()
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(args.mesh_rows, args.mesh_cols),
        trace_region_size=args.trace_region_size,
    )
    try:
        is_mesh = hasattr(mesh, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh) if is_mesh else None

        page_cfg = PagedAttentionConfig(
            block_size=args.page_block_size,
            max_num_blocks=args.max_seq_len // args.page_block_size,
        )
        page_table = torch.arange(page_cfg.max_num_blocks, dtype=torch.int32).reshape(1, page_cfg.max_num_blocks)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )

        tokenizer = _load_tokenizer(args.model_path)
        model_t0 = time.perf_counter()
        model_args, model, tt_kv_cache, state_dict = create_tt_model(
            mesh_device=mesh,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            model_path=args.model_path,
            create_kv_cache=True,
            paged_attention_config=page_cfg,
        )
        print(
            "MODEL_READY",
            f"{time.perf_counter() - model_t0:.2f}s",
            "layers",
            len(model.layers),
            "sampling",
            model.sampling is not None,
            "force_argmax",
            getattr(model.sampling.tt_sampling, "force_argmax_sampling", None) if model.sampling else None,
            "max_batch",
            getattr(model.sampling.tt_sampling, "max_batch_size", None) if model.sampling else None,
            flush=True,
        )
        if model.sampling is None:
            raise RuntimeError("Strict feedback harness requires on-device sampling")
        if getattr(model_args, "hidden_size_per_layer_input", 0):
            raise RuntimeError("Strict feedback harness does not yet handle per-layer input embeddings")

        if not args.base_completion and getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        else:
            input_ids = tokenizer.encode(args.prompt, return_tensors="pt").squeeze(0)
        prompt_len = input_ids.shape[0]
        if prompt_len <= 128:
            padded_len = 128
        elif prompt_len <= 1024:
            padded_len = 1024
        else:
            padded_len = 2 ** (prompt_len - 1).bit_length()
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        print("PROMPT", repr(args.prompt), "prompt_len", prompt_len, "padded_len", padded_len, flush=True)

        prefill_t0 = time.perf_counter()
        tokens_tt = ttnn.from_torch(
            input_ids_padded.unsqueeze(0).to(torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)

        embed_weight = state_dict.get(
            "model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight")
        )
        embeds_torch = F.embedding(input_ids_padded.unsqueeze(0).long(), embed_weight) * model.embed_scale
        get_last_token = ((prompt_len - 1) // 32) * 32
        logits = model.ttnn_prefill_forward(
            embeds,
            page_table=page_table_tt,
            kv_cache=tt_kv_cache,
            get_last_token=get_last_token,
            input_ids_torch=input_ids_padded.unsqueeze(0),
            embeds_torch=embeds_torch.float(),
        )
        logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]) if is_mesh else ttnn.to_torch(logits)
        logits.deallocate(True)
        pos_in_tile = (prompt_len - 1) - get_last_token
        first_token = int(logits_cpu[0, 0, pos_in_tile, :].argmax().item())
        prefill_s = time.perf_counter() - prefill_t0
        print(
            "PREFILL_DONE",
            f"{prefill_s:.3f}s",
            "first_token",
            first_token,
            repr(tokenizer.decode([first_token])),
            flush=True,
        )

        token_in = ttnn.from_torch(
            torch.tensor([first_token], dtype=torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        token_out = ttnn.from_torch(
            torch.zeros((32,), dtype=torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        pos_rope = ttnn.from_torch(
            torch.tensor([prompt_len], dtype=torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        pos_cache = ttnn.from_torch(
            torch.tensor([prompt_len], dtype=torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=replicate,
        )
        print(
            "INITIAL_TOKEN_IN",
            _mesh_values(token_in, is_mesh),
            "POS_ROPE",
            _mesh_values(pos_rope, is_mesh),
            "POS_CACHE",
            _mesh_values(pos_cache, is_mesh),
            flush=True,
        )

        def device_step():
            x = model.embed_tokens(token_in)
            x = ttnn.reshape(x, (1, 1, 1, model_args.hidden_size))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            token_index = None if model.rope_caches_2d else 0
            logits = model(
                hidden_states=x,
                position_idx=pos_rope,
                page_table=page_table_tt,
                kv_caches=tt_kv_cache,
                is_decode=True,
                token_index=token_index,
                position_idx_cache=pos_cache,
            )
            if logits.shape[2] < 32:
                logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - logits.shape[2]), (0, 0)], value=0.0)
            sampled = model.sampling.sample(logits, enable_trace=False, tt_out_tok=token_out)
            tok32 = sampled[0] if isinstance(sampled, tuple) else sampled
            tok32_3d = ttnn.reshape(tok32, (1, 1, 32))
            tok0 = ttnn.slice(tok32_3d, (0, 0, 0), (1, 1, 1))
            tok0 = ttnn.reshape(tok0, (1,))
            ttnn.copy(tok0, token_in)
            next_rope = ttnn.plus_one(pos_rope)
            ttnn.copy(next_rope, pos_rope)
            next_cache = ttnn.plus_one(pos_cache)
            ttnn.copy(next_cache, pos_cache)
            return tok32

        print("COMPILE_STRICT_DEVICE_FEEDBACK_START", flush=True)
        compile_t0 = time.perf_counter()
        compile_out = device_step()
        ttnn.synchronize_device(mesh)
        print(
            "COMPILE_STRICT_DEVICE_FEEDBACK_DONE",
            f"{time.perf_counter() - compile_t0:.3f}s",
            "out_shape",
            compile_out.shape,
            "token_in",
            _mesh_values(token_in, is_mesh),
            "token_out_first",
            _mesh_values(token_out, is_mesh),
            "pos_rope",
            _mesh_values(pos_rope, is_mesh),
            "pos_cache",
            _mesh_values(pos_cache, is_mesh),
            flush=True,
        )

        print("CAPTURE_STRICT_DEVICE_FEEDBACK_TRACE_START", flush=True)
        trace_t0 = time.perf_counter()
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_out = device_step()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        print(
            "CAPTURE_STRICT_DEVICE_FEEDBACK_TRACE_DONE",
            f"{time.perf_counter() - trace_t0:.3f}s",
            "trace_id",
            trace_id,
            "trace_out_shape",
            trace_out.shape,
            "token_in",
            _mesh_values(token_in, is_mesh),
            "pos_rope",
            _mesh_values(pos_rope, is_mesh),
            "pos_cache",
            _mesh_values(pos_cache, is_mesh),
            flush=True,
        )

        print("REPLAY_STRICT_DEVICE_FEEDBACK_START", "tokens", args.max_new_tokens, flush=True)
        replay_ms = []
        replay_total_t0 = time.perf_counter()
        report_indices = {0, 1, 2, 3, 15, 31, 63, args.max_new_tokens - 1}
        for i in range(args.max_new_tokens):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            dt_ms = (time.perf_counter() - t0) * 1000
            replay_ms.append(dt_ms)
            if i in report_indices:
                print("REPLAY_TOKEN", i + 1, f"{dt_ms:.3f}ms", flush=True)

        replay_total_s = time.perf_counter() - replay_total_t0
        final_token_in = _mesh_values(token_in, is_mesh)
        final_token_out = _mesh_values(token_out, is_mesh)
        final_pos_rope = _mesh_values(pos_rope, is_mesh)
        final_pos_cache = _mesh_values(pos_cache, is_mesh)
        ttnn.release_trace(mesh, trace_id)

        expected_pos = prompt_len + 1 + args.max_new_tokens
        print("FINAL_TOKEN_IN", final_token_in, flush=True)
        print("FINAL_TOKEN_OUT_FIRST", final_token_out, flush=True)
        print("FINAL_POS_ROPE", final_pos_rope, flush=True)
        print("FINAL_POS_CACHE", final_pos_cache, flush=True)
        if final_pos_rope[0][0] != expected_pos or final_pos_cache[0][0] != expected_pos:
            raise RuntimeError(
                f"position mismatch expected {expected_pos}, got rope={final_pos_rope[0][0]}, cache={final_pos_cache[0][0]}"
            )

        avg_ms = sum(replay_ms) / len(replay_ms)
        print("STRICT_REPLAY_TOTAL_SECONDS", f"{replay_total_s:.3f}", flush=True)
        print("STRICT_REPLAY_AVG_MS", f"{avg_ms:.3f}", flush=True)
        print("STRICT_REPLAY_TOK_S_USER", f"{1000.0 / avg_ms:.3f}", flush=True)
        print("STRICT_REPLAY_1ST_MS", f"{replay_ms[0]:.3f}", flush=True)
        print("STRICT_REPLAY_LAST_MS", f"{replay_ms[-1]:.3f}", flush=True)
        print("STRICT_TTFT_MS", f"{prefill_s * 1000:.3f}", flush=True)
        print("STRICT_DEVICE_FEEDBACK_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    print("RUN_WALL_SECONDS", f"{time.perf_counter() - wall_start:.2f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_GEMMA4_MODEL)
    parser.add_argument("--hf-revision", default="unknown")
    parser.add_argument("--prompt", default="Explain in two sentences why paged attention helps LLM serving.")
    parser.add_argument("--base-completion", action="store_true", help="Bypass the tokenizer chat template")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--mesh-rows", type=int, default=1)
    parser.add_argument("--mesh-cols", type=int, default=8)
    parser.add_argument("--trace-region-size", type=int, default=50_000_000)
    parser.add_argument("--page-block-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
