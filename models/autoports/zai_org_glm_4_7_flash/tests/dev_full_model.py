# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model probe for GLM-4.7-Flash (debug loop, not stage evidence).

One real layer of each kind (HF layers 0 dense + 1 moe), the real embedding,
final norm, LM head, real paged-cache/page-table shapes and the real terminal
sampling path. Runs in ~1 minute, so wrapper/trace/cache/LM-head/sampling bugs
get localized here instead of in a 3-minute 47-layer load.

    python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py smoke
    python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py pcc
    python models/autoports/zai_org_glm_4_7_flash/tests/dev_full_model.py trace
"""

import argparse
import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator
from models.autoports.zai_org_glm_4_7_flash.tt.model import GLM47FlashModel, ShardedCheckpoint, resolve_checkpoint_dir

PROBE_LAYERS = [0, 1]
PROBE_SEQ_LEN = 4096
TRACE_REGION = 350_000_000


def open_device():
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=TRACE_REGION)


def build(dev, *, layers=PROBE_LAYERS, seq=PROBE_SEQ_LEN, batch=1, **kw):
    if layers == "all":
        layers = None
    t0 = time.perf_counter()
    model = GLM47FlashModel.from_pretrained(
        dev,
        max_batch_size=batch,
        max_seq_len=seq,
        layer_indices=layers,
        progress=lambda m: print(f"  [{time.perf_counter() - t0:6.1f}s] {m}", flush=True),
        **kw,
    )
    print(f"model built in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def torch_reference(cfg, snapshot, layer_indices, token_ids):
    """fp32 host reference of the reduced model: embed -> layers -> norm -> lm_head."""
    ckpt = ShardedCheckpoint(snapshot)
    try:
        embed = ckpt.get("model.embed_tokens.weight")
        norm_w = ckpt.get("model.norm.weight")
        head = ckpt.get("lm_head.weight")
        x = embed[torch.as_tensor(token_ids, dtype=torch.long)].unsqueeze(0)  # [1, S, H]
        for idx in layer_indices:
            sd = ckpt.layer_state_dict(idx)
            layer = utils.build_hf_layer(cfg, idx, sd)
            x = utils.hf_forward(cfg, layer, x)
            del layer, sd
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w
        return (x @ head.T)[0]  # [S, V]
    finally:
        ckpt.close()


def cmd_smoke(args):
    dev = open_device()
    try:
        model = build(dev)
        cache = model.allocate_kv_cache()
        pt = model.page_table_to_device(model.default_page_table())
        ids = list(range(100, 100 + 37))  # deliberately non-aligned
        t0 = time.perf_counter()
        logits = model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=len(ids))
        print(f"prefill(37) -> {tuple(logits.shape)} in {time.perf_counter() - t0:.2f}s", flush=True)
        assert logits.shape == (1, 1, model.vocab_size), logits.shape
        allp = model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=len(ids), return_all_logits=True)
        print(f"prefill all-logits -> {tuple(allp.shape)}", flush=True)
        assert allp.shape == (1, 37, model.vocab_size), allp.shape
        assert torch.allclose(allp[:, -1], logits[:, 0], atol=0), "last-row logits differ between modes"

        gen = GLM47FlashGenerator(model)
        gen.bind_decode_state(kv_cache=cache, page_table=model.default_page_table())
        t0 = time.perf_counter()
        gen.capture_decode_trace(kv_cache=cache)
        print(f"trace captured in {time.perf_counter() - t0:.1f}s", flush=True)
        gen._kv_cache = cache
        gen._page_table_torch = model.default_page_table()
        out = gen.generate(ids, 8, enable_trace=True)
        print("generated:", out, flush=True)
        print("counters:", gen.counters, flush=True)
        gen.teardown()
        print("SMOKE_OK")
    finally:
        ttnn.close_mesh_device(dev)


def cmd_pcc(args):
    """Reduced-model prefill + teacher-forced decode against a torch fp32 reference."""
    cfg = utils.hf_config()
    snapshot = resolve_checkpoint_dir()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    text = (
        "The capital city of France is Paris, which sits on the river Seine. "
        "Machine learning models are trained on large corpora of text so that they can "
        "predict the next token in a sequence. In this example we simply continue prose."
    )
    ids = tok.encode(text, add_special_tokens=True)[: args.seq]
    print(f"prompt tokens: {len(ids)}", flush=True)
    print("building torch reference...", flush=True)
    ref = torch_reference(cfg, snapshot, PROBE_LAYERS, ids)
    dev = open_device()
    try:
        model = build(dev, lm_head_dtype=getattr(ttnn, args.lm_head_dtype))
        cache = model.allocate_kv_cache()
        pt_torch = model.default_page_table()
        pt = model.page_table_to_device(pt_torch)
        got = model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=len(ids), return_all_logits=True)[0]
        print(f"prefill logits PCC = {utils.pcc(ref, got):.6f}", flush=True)
        ref_top = ref.topk(5, dim=-1).indices
        got_top1 = got.argmax(-1)
        top1 = (ref_top[:, 0] == got_top1).float().mean().item()
        top5 = (ref_top == got_top1[:, None]).any(-1).float().mean().item()
        print(f"prefill top-1 {top1:.4f}  top-5 {top5:.4f}", flush=True)

        gen = GLM47FlashGenerator(model)
        gen._kv_cache = cache
        gen._page_table_torch = pt_torch
        gen.bind_decode_state(kv_cache=cache, page_table=pt_torch)
        gen.capture_decode_trace(kv_cache=cache)
        gen.reset()
        n_steps = 8
        p_len = len(ids) - n_steps
        forced = ids[p_len:]
        preds = gen.generate(ids[:p_len], n_steps, enable_trace=True, next_input=lambda step, pred: forced[step])
        ref_top_dec = ref[p_len - 1 : p_len - 1 + n_steps].topk(5, dim=-1).indices
        d1 = sum(int(p == int(r[0])) for p, r in zip(preds, ref_top_dec))
        d5 = sum(int(p in r.tolist()) for p, r in zip(preds, ref_top_dec))
        print("tt  teacher-forced preds:", preds, flush=True)
        print("ref teacher-forced preds:", ref_top_dec[:, 0].tolist(), flush=True)
        print(f"decode top-1 {d1}/{n_steps}  top-5 {d5}/{n_steps}", flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


def random_token(i):
    g = torch.Generator().manual_seed(1234 + i)
    return int(torch.randint(0, 100000, (1,), generator=g).item())


def cmd_prefill(args):
    """Per-layer prefill wall clock at several prompt lengths."""
    dev = open_device()
    try:
        model = build(dev, layers=("all" if args.layers == "all" else PROBE_LAYERS), seq=args.seq_cap)
        cache = model.allocate_kv_cache()
        pt = model.page_table_to_device(model.default_page_table())
        n_layers = len(model.layers)
        for seq in [int(v) for v in args.prefill_lens.split(",")]:
            ids = list(range(1000, 1000 + seq))
            model.prefill_forward(ids, kv_cache=cache, page_table=pt, seq_len=seq)  # warm
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            hidden, _ = model.run_layer_stack_prefill(ids, kv_cache=cache, page_table=pt, seq_len=seq)
            ttnn.synchronize_device(dev)
            t_stack = time.perf_counter() - t0
            t0 = time.perf_counter()
            model._logits_host_rows(hidden, seq, seq - 1, seq, 320)
            ttnn.synchronize_device(dev)
            t_head = time.perf_counter() - t0
            ttnn.deallocate(hidden)
            print(
                f"S={seq:6d}  stack {t_stack * 1e3:9.1f} ms ({t_stack / n_layers * 1e3:7.2f} ms/layer)"
                f"  head {t_head * 1e3:7.2f} ms   full-model est {(t_stack / n_layers * 46 + t_stack / n_layers + t_head) * 1e3:9.1f} ms",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(dev)


def cmd_capacity(args):
    """Full 47-layer build: DRAM accounting at the advertised context, then a
    short prefill + traced decode to prove the whole stack runs in that budget."""
    dev = open_device()
    try:
        model = build(dev, layers="all", seq=args.seq_cap)
        wb = model.weight_bytes()
        gib = 2**30
        for key, val in wb.items():
            print(f"  weights[{key:10s}] = {val / gib:8.3f} GiB", flush=True)
        cache_bytes = model.kv_cache_bytes()
        print(
            f"  kv cache @ {model.max_seq_len} x batch {model.max_batch_size} = {cache_bytes / gib:.3f} GiB", flush=True
        )
        print(f"  weights + cache = {(wb['total'] + cache_bytes) / gib:.3f} GiB of 31.5 GiB allocatable", flush=True)
        t0 = time.perf_counter()
        cache = model.allocate_kv_cache()
        print(f"  cache allocated in {time.perf_counter() - t0:.1f}s", flush=True)
        gen = GLM47FlashGenerator(model)
        gen._kv_cache = cache
        gen._page_table_torch = model.default_page_table()
        gen.bind_decode_state(kv_cache=cache, page_table=gen._page_table_torch)
        t0 = time.perf_counter()
        gen.capture_decode_trace(kv_cache=cache)
        print(f"  decode + sampling traces captured in {time.perf_counter() - t0:.1f}s", flush=True)
        gen.reset()
        ids = list(range(500, 500 + args.seq))
        t0 = time.perf_counter()
        gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)
        print(f"  cold generate (compiles prefill programs): {time.perf_counter() - t0:.2f}s", flush=True)
        gen.reset_counters()
        t0 = time.perf_counter()
        preds, timing = gen.generate(ids, 128, enable_trace=True, stop_on_eos=False, return_timing=True)
        print(f"  warmed generate(prompt={len(ids)}, 128) in {time.perf_counter() - t0:.2f}s", flush=True)
        ms = timing["decode_s"] / max(timing["decode_tokens"], 1) * 1e3
        print(
            f"  TTFT {timing['ttft_s'] * 1e3:.1f} ms   token-out decode {ms:.3f} ms/token ({1000 / ms:.2f} t/s/u)",
            flush=True,
        )
        print("  preds[:16]:", preds[:16], flush=True)
        print("  counters:", gen.counters, flush=True)

        n = 32

        def bench(label, fn):
            fn()
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / n
            print(f"  {label:38s} {dt * 1e3:8.3f} ms/token ({1 / dt:8.2f} t/s/u)", flush=True)

        bench("model trace only", gen.replay_decode_trace)
        bench("model + sampling traces", gen.decode_step_traced)
        bench("model + sampling + token readback", lambda: (gen.decode_step_traced(), gen.read_decode_tokens(1)))
        gen.teardown()
        print("CAPACITY_OK")
    finally:
        ttnn.close_mesh_device(dev)


def cmd_trace(args):
    dev = open_device()
    try:
        model = build(dev, layers=("all" if args.layers == "all" else PROBE_LAYERS), seq=args.seq_cap)
        cache = model.allocate_kv_cache()
        gen = GLM47FlashGenerator(model)
        gen._kv_cache = cache
        gen._page_table_torch = model.default_page_table()
        gen.bind_decode_state(kv_cache=cache, page_table=gen._page_table_torch)
        gen.capture_decode_trace(kv_cache=cache)
        gen.reset()
        ids = list(range(200, 200 + 64))
        first = gen._prefill_and_sample_first(ids)
        gen.set_decode_positions([len(ids)])
        gen.reset_counters()
        n = 32

        def bench(label, fn):
            fn()  # warm
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / n
            print(f"{label:38s} {dt * 1e3:7.3f} ms/token ({1 / dt:8.2f} t/s/u)", flush=True)
            return dt

        bench("model trace only", gen.replay_decode_trace)
        bench("model + sampling traces", gen.decode_step_traced)

        def full():
            gen.decode_step_traced()
            gen.read_decode_tokens(1)

        bench("model + sampling + token readback", full)
        toks = []
        for _ in range(n):
            gen.decode_step_traced()
            toks.append(gen.read_decode_tokens(1)[0])
        print(f"first={first} tokens={toks}", flush=True)
        print("counters:", gen.counters, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "pcc", "trace", "capacity", "prefill"])
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lm-head-dtype", default="bfloat8_b")
    ap.add_argument("--layers", default="probe")
    ap.add_argument("--seq-cap", type=int, default=202752)
    ap.add_argument("--prefill-lens", default="128,512,2048")
    args = ap.parse_args()
    {
        "smoke": cmd_smoke,
        "pcc": cmd_pcc,
        "trace": cmd_trace,
        "capacity": cmd_capacity,
        "prefill": cmd_prefill,
    }[
        args.cmd
    ](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
