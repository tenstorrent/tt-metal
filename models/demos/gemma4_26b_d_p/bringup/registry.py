# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bring-up registry: blocks, layers, phases, op gaps. Single source of truth for the dashboard.

Status vocabulary (per block, per mesh for device results):
  todo          nothing yet
  reference     torch reference written
  ref_tested    torch reference matches HF / chunk-invariant
  ttnn_reuse    TTNN impl reuses an existing module/op as-is
  ttnn_compose  TTNN impl composed from existing primitive ops (candidate for op generation)
  missing_op    needs an op that does not exist / lacks a feature (op-generation target)
  ttnn_pass     TTNN impl passes PCC on device
  ttnn_fail     TTNN impl fails PCC on device

Export for the dashboard DB:  python -m models.demos.gemma4_26b_d_p.bringup.registry --out <dir>
Device test results are merged from ``results/*.json`` (written by tests via ``record_result``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MESHES = ["2x2", "1x4", "4x1"]  # (SP rows) x (TP cols) on the BH QuietBox

PHASES = [
    (1, "Analyze HF reference", "done", "HF modeling_gemma4 + config audited; 30L = 5x(5 sliding + 1 full); MoE 128e top-8 + dense MLP in parallel."),
    (2, "Break into blocks", "done", "26 blocks, tagged on the fx graph."),
    (3, "Group blocks into layers", "done", "Embedding / attention (sliding, full) / FFN (dense + MoE) / decoder layer x2 / output head."),
    (4, "Torch reference + fx graph", "done", "Functional reference, chunked-prefill driver, fx trace at per-chip chunk shapes."),
    (5, "Correctness tests (CPU)", "done", "vs HF 6L S=1152 PCC 0.99936 (bf16); chunk-invariance exact in fp32."),
    (6, "TTNN blocks (reuse)", "done", "All attention + FFN blocks on device, 2x2 / 1x4 / 4x1."),
    (7, "Mark / compose missing ops", "in_progress", "5 op changes landed (GeluTanh, ring sliding SP1/2, RoPE DEST fix, offset_cumsum + dispatch/combine on extent-1 axis); router + SP1 sliding composed."),
    (8, "Stitch blocks into layers", "in_progress", "Attention + FFN sub-layers pass on all meshes; decoder layer + model driver written."),
    (9, "Test layers (+ full model)", "in_progress", "30L on 2x2/1x4/4x1: top-1 matches, logits PCC 0.994-0.996; hidden drifts to ~0.89 mid-depth. Teacher-forced: every single layer >= 0.9997, so it is accumulation, not a bad layer."),
    (10, "tt-d-gen integration", "in_progress", "Adapter + runtime + 20-config KV chunk table + manifests; runtime contract passes on all meshes; real prefill_runner served H2D chunks with layer acks (Gate 1 plumbing)."),
]

# Layer groups (dashboard "layers"). A decoder layer = attention sub-layer + FFN sub-layer.
LAYERS = [
    {"id": "embedding", "name": "Embedding", "blocks": ["embed"]},
    {
        "id": "attention_sliding",
        "name": "Sliding attention (25 layers)",
        "blocks": ["input_norm", "qkv_proj", "qk_norm", "v_norm", "rope", "kv_cache_write", "sdpa", "o_proj", "post_attn_norm", "attn_residual"],
        "variant": "sliding",
    },
    {
        "id": "attention_full",
        "name": "Full attention (5 layers)",
        "blocks": ["input_norm", "qkv_proj", "qk_norm", "v_norm", "rope", "kv_cache_write", "sdpa", "o_proj", "post_attn_norm", "attn_residual"],
        "variant": "full",
    },
    {
        "id": "ffn",
        "name": "FFN: dense MLP + MoE (30 layers)",
        "blocks": ["pre_ff_norm", "dense_mlp", "post_ff_norm_1", "router", "pre_ff_norm_2", "moe_experts", "post_ff_norm_2", "ffn_out_norm", "ffn_residual", "layer_scalar"],
    },
    {"id": "output", "name": "Output head", "blocks": ["final_norm", "lm_head"]},
    {"id": "model_L6", "name": "Model, first 6 layers (5 sliding + 1 full), chunked prefill", "blocks": ["embed", "final_norm", "lm_head"]},
    {"id": "model_L30", "name": "Full model, 30 layers, chunked prefill", "blocks": ["embed", "final_norm", "lm_head"]},
    {"id": "runtime_contract", "name": "tt-d-gen runtime contract (device-order tokens, pad tail, acks, KV readback)", "blocks": ["kv_cache_write"]},
]


def B(id, name, layer, torch_ref, ttnn_plan, status, source="", ops=(), gap=None, parallel="", variants=None):
    return {
        "id": id,
        "name": name,
        "layer": layer,
        "torch_ref": torch_ref,
        "ttnn_plan": ttnn_plan,
        "status": status,
        "source": source,
        "ops": list(ops),
        "gap": gap,
        "parallel": parallel,
        "variants": variants or {},
        "device": {},
    }


BLOCKS = [
    B("embed", "Scaled embedding", "embedding", "ScaledEmbedding: embed(ids) * sqrt(2816)",
      "TtParallelEmbedding (vocab-sharded over TP) + mul(sqrt(d)); tokens block-cyclic over SP", "ref_tested",
      "deepseek_v3_d_p/tt/tt_parallel_embedding.py (edit: scale, config)", ["ttnn.embedding", "ttnn.mul", "all_gather/reduce_scatter (TP)"],
      parallel="SP: token chunk slice per row; TP: vocab-parallel"),
    B("input_norm", "input_layernorm", "attention", "RMSNorm(w), fp32 stats, no (1+w)",
      "TtDistributedRmsNorm (pre/post all-gather stats over TP)", "ref_tested",
      "deepseek_v3_d_p/tt/tt_distributed_rms_norm.py", ["rms_norm_pre_all_gather", "all_gather", "rms_norm_post_all_gather"],
      parallel="hidden sharded on TP"),
    B("qkv_proj", "Q/K/V projections", "attention", "q_proj [4096|8192], k_proj, v_proj (sliding only; full: V = raw K)",
      "all_gather hidden (TP) -> fused column-parallel wqkv per chip; full layers: no V weight, V taken from K before k_norm",
      "ref_tested", "gpt_oss_d_p/tt/attention/weights.py + gemma4/tt/attention/weights.py (K=V, GQA head assignment)",
      ["all_gather_async", "ttnn.linear", "nlp_create_qkv_heads"],
      parallel="TP over heads: sliding 16q/8kv, full 16q/2kv (kv replicated when TP>2)",
      variants={"sliding": "q 16x256, k/v 8x256", "full": "q 16x512, k 2x512, V = K_raw (k_eq_v)"}),
    B("qk_norm", "q_norm / k_norm (per head)", "attention", "RMSNorm(head_dim, w) per head",
      "rms_norm on [1,1,H*S,D] view", "ref_tested", "gemma4/tt/attention/operations.py:apply_per_head_norm", ["ttnn.rms_norm"]),
    B("v_norm", "v_norm (no scale)", "attention", "RMSNorm(head_dim, with_scale=False)",
      "rms_norm without weight", "ref_tested", "gemma4/tt/attention/operations.py", ["ttnn.rms_norm"]),
    B("rope", "RoPE (block-cyclic positions)", "attention", "rotate-half; sliding theta 1e4 full-dim; full theta 1e6 proportional (64 of 256 pairs)",
      "rotary_embedding_indexed with per-chip block-cyclic position ids; partial rope encoded in cos/sin tables (cos=1,sin=0)",
      "ref_tested", "ttnn.experimental.deepseek_prefill.rotary_embedding_indexed (verify D=512)", ["rotary_embedding_indexed"],
      variants={"sliding": "D=256, theta=1e4", "full": "D=512, theta=1e6, 25% rotated"}),
    B("kv_cache_write", "KV cache write (block-cyclic)", "attention", "cat(prefix, chunk) on seq",
      "update_padded_kv_cache into per-head K and V caches, block-cyclic ND-sharded over SP", "ref_tested",
      "gpt_oss_d_p/tt/attention/kv_cache.py", ["update_padded_kv_cache"],
      parallel="cache seq sharded block-cyclic on SP, heads on TP"),
    B("sdpa", "SDPA (ring, chunked)", "attention", "GQA softmax(QK^T * 1.0 + mask) V; sliding window 1024 / causal",
      "ring_joint_scaled_dot_product_attention, cache-backed, over SP ring; sliding: halo gather + sliding_window_size=1024; scale=1.0, no sinks",
      "ref_tested", "gpt_oss_d_p/tt/attention/dense_sp.py", ["ring_joint_scaled_dot_product_attention"],
      gap="SP=1 sliding composed from primitives (no neighbour halo in the ring op); SP>1 needs chunk_local >= window (chunk >= 1024*SP); SP-axis CCL topology must be Ring where the fabric wraps",
      variants={"sliding": "D=256, window 1024, GQA 2", "full": "D=512, causal, GQA 8"}),
    B("o_proj", "o_proj", "attention", "Linear(4096|8192 -> 2816)",
      "row-parallel matmul + reduce_scatter (TP)", "ref_tested", "gpt_oss_d_p/tt/attention/prefill.py", ["nlp_concat_heads", "ttnn.linear", "reduce_scatter"]),
    B("post_attn_norm", "post_attention_layernorm", "attention", "RMSNorm(w)", "TtDistributedRmsNorm", "ref_tested",
      "deepseek_v3_d_p/tt/tt_distributed_rms_norm.py", ["rms_norm_pre/post_all_gather"]),
    B("attn_residual", "residual add", "attention", "x + attn", "ttnn.add", "ref_tested", "", ["ttnn.add"]),
    B("pre_ff_norm", "pre_feedforward_layernorm", "ffn", "RMSNorm(w)", "TtDistributedRmsNorm", "ref_tested",
      "deepseek_v3_d_p/tt/tt_distributed_rms_norm.py", ["rms_norm_pre/post_all_gather"]),
    B("dense_mlp", "Dense MLP (GELU-tanh)", "ffn", "down(gelu_tanh(gate x) * up x), I=2112",
      "TtFfn/TtSharedExpert with GELU-tanh activation (column/row parallel over TP)", "ref_tested",
      "deepseek_v3_d_p/tt/tt_ffn.py (edit: activation SiLU -> gelu_tanh)", ["ttnn.linear", "ttnn.gelu", "ttnn.mul", "reduce_scatter"]),
    B("post_ff_norm_1", "post_feedforward_layernorm_1", "ffn", "RMSNorm(w)", "TtDistributedRmsNorm", "ref_tested"),
    B("router", "Router (softmax -> top8 -> renorm -> per-expert scale)", "ffn",
      "rmsnorm_noscale(residual) * scale * H^-0.5 -> proj -> softmax(128) -> topk(8) -> w/sum(w) -> * per_expert_scale[idx]",
      "new TtGemma4Router composed from primitives, output fed to TtMoERoutingSetup (DS)", "ref_tested",
      "gemma4/tt/router.py (math) + deepseek_v3_d_p/tt/moe routing setup", ["rms_norm", "mul", "linear", "softmax", "topk", "sum", "div", "embedding(gather)"],
      gap="compose-only: fused router op would be an op-generation target"),
    B("pre_ff_norm_2", "pre_feedforward_layernorm_2", "ffn", "RMSNorm(w)", "TtDistributedRmsNorm", "ref_tested"),
    B("moe_experts", "MoE experts (EP dispatch/combine)", "ffn", "128 experts, gate_up [E,1408,2816] (gate rows first), down [E,2816,704]; gelu_tanh(gate)*up",
      "TtDispatchModule -> unified_routed_expert_ffn -> TtCombineModule -> TtReduceModule; EP over all chips (32 experts/chip)",
      "ttnn_reuse", "deepseek_v3_d_p/tt/moe/*", ["dispatch", "unified_routed_expert_ffn(GeluTanh)", "combine", "post_combine_reduce"],
      gap=None,
      parallel="EP=4 (all chips); dispatch within column"),
    B("post_ff_norm_2", "post_feedforward_layernorm_2", "ffn", "RMSNorm(w)", "TtDistributedRmsNorm", "ref_tested"),
    B("ffn_out_norm", "m1 + m2 -> post_feedforward_layernorm", "ffn", "RMSNorm(w)(m1 + m2)", "ttnn.add + TtDistributedRmsNorm", "ref_tested"),
    B("ffn_residual", "residual add", "ffn", "r + ffn", "ttnn.add", "ref_tested", "", ["ttnn.add"]),
    B("layer_scalar", "layer_scalar", "ffn", "out * layer_scalar (per-layer constant)", "ttnn.mul (host scalar)", "ref_tested", "", ["ttnn.mul"]),
    B("final_norm", "final norm", "output", "RMSNorm(w)", "TtDistributedRmsNorm on last-token rows", "ref_tested"),
    B("lm_head", "LM head + softcap(30)", "output", "tied embed^T; tanh(logits/30)*30",
      "TtLMHead (vocab-parallel) + mul/tanh/mul softcap", "ref_tested", "deepseek_v3_d_p/tt/tt_lm_head.py (edit: softcap, config)",
      ["ttnn.linear", "ttnn.tanh", "ttnn.mul", "all_gather"]),
]

OPS = [
    {
        "id": "unified_routed_expert_ffn.gelu_tanh",
        "title": "GELU-tanh activation in unified_routed_expert_ffn",
        "kind": "extend_op",
        "status": "done",
        "blocks": ["moe_experts"],
        "detail": "Added RoutedExpertActivation::GeluTanh (gelu_tanh_tile on the gate, then * up). Single-chip PCC at 2816->704, real weights: bfp8 0.9993, bfp4 0.981 (128/1k/4k tokens). SiLU/SwiGLU-OAI regression tests pass.",
    },
    {
        "id": "ring_sdpa.d512",
        "title": "Ring SDPA at head_dim 512 (full layers)",
        "kind": "verify_op",
        "status": "done",
        "blocks": ["sdpa"],
        "detail": "Verified as-is: D=512, 16Q/2KV, scale 1.0, causal, cache-backed over SP on 2x2/1x4/4x1; PCC 0.9978-0.9981 (2x4k, 3x4k, 2x8k).",
    },
    {
        "id": "ring_sdpa.sliding_sp2",
        "title": "Ring SDPA sliding window on SP=2",
        "kind": "extend_op",
        "status": "done",
        "blocks": ["sdpa"],
        "detail": "Host allowlist was SP4/SP8 only; widened to SP1/2/4/8. SP2 (2x2) sliding D=256 window 1024 passes, PCC 0.9992. Earlier hang was a Linear CCL topology on a torus-Y fabric: the halo is cyclic and needs Ring.",
    },
    {
        "id": "ring_sdpa.sliding_sp1",
        "title": "Sliding attention on SP=1 (1x4)",
        "kind": "compose",
        "status": "in_progress",
        "blocks": ["sdpa"],
        "detail": "Ring op sliding needs a neighbour halo (program-factory assert). Composed: cache tail slice + concat + zero-prepended Q + local windowed SDPA + slice; PCC 0.9993. Costs window/chunk extra Q rows: op-generation target (tail-aware sliding SDPA).",
    },
    {
        "id": "rope_indexed.d512_partial",
        "title": "rotary_embedding_indexed at D=512 with partial rotation",
        "kind": "verify_op",
        "status": "done",
        "blocks": ["rope"],
        "detail": "Works via tables (cos=1, sin=0 on 192 of 256 pairs) after the DEST-blocking fix; q/k rope PCC 0.99995. A partial-aware op would skip 75% of the work (perf target).",
    },
    {
        "id": "rope.dest_blocking",
        "title": "RoPE kernel overflowed DEST at head_dim 512",
        "kind": "fix_op",
        "status": "done",
        "blocks": ["rope"],
        "detail": "rotary_embedding_llama compute kernel (shared by rotary_embedding_indexed) kept the whole head row in DEST: silent corruption at D=512 (PCC 0.598); llama op hid it behind head_dim<=256 / <=128-fp32 asserts. Now processes DEST-sized blocks; interleaved asserts lifted (sharded kernel keeps them). rope D=64..512 x bf16/fp32 dest: PCC >= 0.99999; deepseek indexed-rope tests pass.",
    },
    {
        "id": "gemma4_router",
        "title": "Fused Gemma-4 router",
        "kind": "compose",
        "status": "todo",
        "blocks": ["router"],
        "detail": "Composed from rms_norm/mul/linear/softmax/topk/sum/div/gather. Good op-generation target (softmax over 128 then top-8 renorm + per-expert scale).",
    },
    {
        "id": "kv_table.gemma4",
        "title": "KV migration table for 2 cache geometries",
        "kind": "new_code",
        "status": "todo",
        "blocks": ["kv_cache_write"],
        "detail": "Sliding layers 8x256 KV heads, full layers 2x512; per-head K and V configs (k_eq_v still stores V = v_norm(K_raw)).",
    },
]


def record_result(block: str, mesh: str, pcc: float, passed: bool, note: str = "", layer_variant: str | None = None):
    """Called from device tests: persists one result for the dashboard sync."""
    RESULTS_DIR.mkdir(exist_ok=True)
    key = f"{block}{'.' + layer_variant if layer_variant else ''}__{mesh}"
    rec = {
        "block": block,
        "variant": layer_variant,
        "mesh": mesh,
        "pcc": pcc,
        "passed": passed,
        "note": note,
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / f"{key}.json").write_text(json.dumps(rec))


def _merged_blocks():
    blocks = {b["id"]: json.loads(json.dumps(b)) for b in BLOCKS}
    for f in sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []:
        r = json.loads(f.read_text())
        b = blocks.get(r["block"])
        if b is None:
            continue
        key = r["mesh"] + (f":{r['variant']}" if r.get("variant") else "")
        b["device"][key] = {k: r[k] for k in ("pcc", "passed", "note", "at")}
        results = list(b["device"].values())
        if results:
            b["status"] = "ttnn_pass" if all(x["passed"] for x in results) else "ttnn_fail"
    return list(blocks.values())


def _merged_layers():
    layers = json.loads(json.dumps(LAYERS))
    by_id = {l["id"]: l for l in layers}
    for l in layers:
        l["device"] = {}
    for f in sorted(RESULTS_DIR.glob("layer:*.json")) if RESULTS_DIR.exists() else []:
        r = json.loads(f.read_text())
        l = by_id.get(r["block"].split(":", 1)[1])
        if l is not None:
            l["device"][r["mesh"]] = {k: r[k] for k in ("pcc", "passed", "note", "at")}
    return layers


def export(out_dir: Path, fx_json: Path | None):
    """Write one JSON file per DB document + manifest.json listing (collection, doc_id, file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def put(coll, doc_id, data):
        p = out_dir / f"{coll}__{doc_id}.json"
        p.write_text(json.dumps(data))
        manifest.append({"collection": coll, "doc_id": doc_id, "file_path": str(p)})

    now = _dt.datetime.now().isoformat(timespec="seconds")
    put("meta", "state", {
        "phases": [{"n": n, "name": nm, "status": st, "note": note} for n, nm, st, note in PHASES],
        "layers": _merged_layers(),
        "meshes": MESHES,
        "updated": now,
    })
    for b in _merged_blocks():
        put("blocks", b["id"], b)
    for o in OPS:
        put("ops", o["id"].replace(".", "-"), o)
    if fx_json and fx_json.exists():
        fx = json.loads(fx_json.read_text())
        for name, g in fx["graphs"].items():
            put("graphs", name, {"name": name, "chunk": fx["chunk"], "prefix": fx["prefix"], "nodes": g["nodes"], "edges": g["edges"]})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fx", default=None, help="fx_graph.py JSON output")
    a = ap.parse_args()
    m = export(Path(a.out), Path(a.fx) if a.fx else None)
    print(len(m), "docs")
