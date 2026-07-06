# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 CHUNKED-prefill first-token test (the cache-read path).

Builds a chat-templated prompt that spans >=2 chunks of CHUNK_SIZE, prefills it chunk-by-chunk via
TtPrefillRuntime (so chunk>=1 exercises the dense ring_joint cache-read + MSA cache-read), then argmaxes
the LAST REAL token's logits -> the first generated token. This is the chunked analogue of
galaxy_generate_m3_sp.py (which is kv_cache=None / no chunking).

Env: HF_MODEL, CHUNK_SIZE (default 5120), TARGET_LEN (default 10240 -> 2 chunks), EXPERT_DTYPE=bf8,
M3_FORCE_LOAD_WEIGHTS / M3_WEIGHTS_FROM_CACHE. DEBUG_LAYERS=1 for per-layer residual.
"""
import os
import sys
import time

import torch

import ttnn

PASSAGE = (
    "The history of computing is a story of relentless abstraction. Early machines were programmed by "
    "physically rewiring them; every new task meant rebuilding the hardware. Stored-program computers "
    "changed that: instructions lived in the same memory as data, so a machine could be repurposed by "
    "loading a new program. Assembly gave names to opcodes, compilers translated human-readable code "
    "into machine instructions, and operating systems multiplexed the hardware among many programs. Each "
    "layer hid the complexity beneath it, letting engineers reason about ever larger systems. Networking "
    "let once-isolated machines exchange information across the world in fractions of a second, and the "
    "modern era of accelerators trades the generality of a single processor for the throughput of "
    "thousands of small cores working in parallel on the same problem. "
)


def main():
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    chunk_size = int(os.getenv("CHUNK_SIZE", "5120"))
    target_len = int(os.getenv("TARGET_LEN", "10240"))
    assert target_len % chunk_size == 0, f"TARGET_LEN {target_len} must be a multiple of CHUNK_SIZE {chunk_size}"
    n_chunks = target_len // chunk_size
    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf8") == "bf8" else ttnn.bfloat4_b
    rows, cols = 8, 4

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    print(f"[chunk-ft] mesh={tuple(mesh.shape)} chunk={chunk_size} target={target_len} n_chunks={n_chunks}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        tok = model_args.tokenizer
        V = hf_config.vocab_size
        num_layers = hf_config.num_hidden_layers

        # Build a chat-templated prompt that fills > (n_chunks-1)*chunk_size so the last chunk has real tokens
        # and at least one cache-read chunk runs. Grow the body, then template.
        QUESTION = "\n\nIn one word, the central theme of the text above is"
        target_real = (n_chunks - 1) * chunk_size + chunk_size // 2  # land mid-last-chunk
        body = PASSAGE
        while (
            len(
                tok.apply_chat_template(
                    [{"role": "user", "content": body + QUESTION}], add_generation_prompt=True, tokenize=True
                )
            )
            < target_real
        ):
            body += PASSAGE
        ids = tok.apply_chat_template(
            [{"role": "user", "content": body + QUESTION}], add_generation_prompt=True, tokenize=True
        )
        ids = ids[: target_len - 1]  # ensure it fits with room
        n_real = len(ids)
        padded = list(ids) + [0] * (target_len - n_real)
        print(
            f"[chunk-ft] n_real={n_real} padded={target_len} ({target_len // rows}/row), last real token @ pos {n_real - 1}",
            flush=True,
        )

        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        cache_only = os.getenv("M3_FORCE_LOAD_WEIGHTS") != "1" and (
            os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
            or weight_cache_is_complete(cache, hf_config, num_layers, expert_dtype)
        )
        if cache_only:
            print("[chunk-ft] cache-only weight load", flush=True)
            state_dict = {}
        else:
            print("[chunk-ft] loading bf16 source (slow)", flush=True)
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=target_len,
            mesh_shape=(rows, cols),
            chunk_size=chunk_size,
            num_users=1,
            weight_cache_path=cache,
            expert_weight_dtype=expert_dtype,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        print(f"[chunk-ft] compiling ({num_layers}L) ...", flush=True)
        runtime.compile()
        print("[chunk-ft] compile() OK", flush=True)

        t0 = time.perf_counter()
        last_logits = None
        last_chunk_start = (n_chunks - 1) * chunk_size
        for c in range(n_chunks):
            a = c * chunk_size
            is_last = c == n_chunks - 1
            print(f"[chunk-ft] chunk {c} start (cache_read={a > 0}) ...", flush=True)
            inp = runtime.make_chunk_input(padded[a : a + chunk_size])
            out = runtime.prefill(
                inp,
                slot_id=0,
                actual_start=a,
                actual_end=min(a + chunk_size, n_real),
                skip_lm_head=not is_last,
                get_last_token=-1,
            )
            print(f"[chunk-ft] chunk {c} done", flush=True)
            if is_last:
                last_logits = out
        ttnn.synchronize_device(mesh)
        print(f"[chunk-ft] chunked prefill done in {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)

        # gather logits: rows -> seq (dim -2), cols -> vocab (dim -1); pick the last real token row
        full = (
            ttnn.to_torch(
                last_logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(rows, cols), dims=(-2, -1))
            )
            .float()
            .reshape(chunk_size, -1)
        )
        pos_in_chunk = (n_real - 1) - last_chunk_start
        row = full[pos_in_chunk][:V]
        finite = bool(torch.isfinite(row).all())
        nxt = int(row.argmax())
        top5 = torch.topk(torch.nan_to_num(row), 5)
        print(
            f"[chunk-ft] last-real-token logits: finite={finite} max={row.max():.3f} argmax={nxt} -> {tok.decode([nxt])!r}",
            flush=True,
        )
        print(
            f"[chunk-ft] top5 ids={top5.indices.tolist()} vals={[round(v,2) for v in top5.values.tolist()]}", flush=True
        )
        print(f"[chunk-ft] FIRST TOKEN (chunked, {n_chunks} chunks) = {nxt} {tok.decode([nxt])!r}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
