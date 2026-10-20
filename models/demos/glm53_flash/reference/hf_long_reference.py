# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF (CPU, transformers) greedy reference for the long-context needle prompts of
``tests/test_mesh_long_context.py``, so the mesh run's answer can be checked token by token.

The prompt is built by the test's own ``build_needle_ids`` (same seeds, same text, same chat template), so the
token ids are identical to the device run's; the ids and their sha256 are saved next to the result.

The whole ``Glm5NextTextModel`` is held in host RAM EXCEPT the routed experts: 42 MoE layers x 288 experts are
304 B of the model's 313 B parameters (~610 GB in bf16, more than the host has), so each MoE layer's
``Glm5NextTextExperts`` is replaced by ``LazyExperts``, the same eager per-expert loop as the HF module with the
expert weights read from the safetensors shards and dequantized (``checkpoint.dequantize_weight_tensor``, the
arithmetic of ``load_layer_raw`` / ``build_layer``) on demand for the experts a forward actually hits. A decode
step touches 8 experts per layer; a prefill chunk touches all of them (~8 s of dequantization per MoE layer).
Everything else (attention, KDA, indexer, mHC, router, shared expert, norms, embedding, head) is the unmodified HF
module with the weights ``from_pretrained`` would give it (bf16, strict-fp32 groups kept).

Attention runs through HF's ``sdpa`` path (the CPU flash kernel with the DSA top-k bool mask, memory-blocked),
which is what ``from_pretrained`` selects; ``--attn eager`` reproduces the layer dumps' eager path for short
prompts (it materializes the full score matrix). The indexer's pool scoring is evaluated in query blocks
(``--index-qblock``) so a 16K-query chunk against a 64K cache stays at a few GB instead of ~100 GB; the top-k is
unchanged (``--selftest`` checks it against the stock forward).

    python models/demos/glm53_flash/reference/hf_long_reference.py --n 4096 16384
    python models/demos/glm53_flash/reference/hf_long_reference.py --n 65536 --chunk 16384
    python models/demos/glm53_flash/reference/hf_long_reference.py --validate-g16   # 13-token prompt vs FULLMODEL_G16

Writes ``bringup/references/glm53_flash/NEEDLE_<N>/{reference.json, prompt_ids.json, last_logits.pt}``:
prompt ids, first token, the greedy ids/text (``--gen`` tokens, stopping at an eos like the test), the top-20 of
the prefill's last-token logits, wall times and whether the needle was retrieved.
``compare_to_tt.py`` lines the result up with the mesh test's JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import (  # noqa: E402
    EMBED_KEY,
    FINAL_NORM_KEY,
    FP8_BLOCK,
    LAYER_PREFIX,
    LM_HEAD_KEY,
    default_snapshot_dir,
    dequantize_weight_tensor,
    load_layer_raw,
    load_tensors,
    shard_headers,
)
from hf_layer_reference import cast_like_from_pretrained, convert_to_hf_module_keys, load_config  # noqa: E402

GEN_DEFAULT = 16
DEPTH_DEFAULT = 0.5
_EXPERT_RE = re.compile(r"^mlp\.experts\.\d+\.")


# --------------------------------------------------------------------------------------------
# Prompt (the test's builder, unchanged)
# --------------------------------------------------------------------------------------------
class _ListTokenizer:
    """``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding`` in transformers 5 (``return_dict``
    defaults to True); the builder wants the id list. Forwarding with ``return_dict=False`` gives the same ids."""

    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, *args, **kwargs):
        kwargs["return_dict"] = False
        return self._tok.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def needle_prompt(tok, n_tokens: int, depth: float) -> tuple[list[int], str, dict]:
    """Token ids + text of the test's needle prompt and the needle constants (imported, not copied: the mesh test
    is the single source of the prompt; ``ttnn`` must be importable for that module)."""
    from models.demos.glm53_flash.tests import test_mesh_long_context as T

    ids, text = T.build_needle_ids(_ListTokenizer(tok), n_tokens, depth)
    ids = [int(t) for t in ids]
    return ids, text, {"needle": T.NEEDLE, "question": T.QUESTION, "needle_key": T.NEEDLE_KEY}


def eos_ids(snapshot: Path, tok) -> set[int]:
    """The test's stop set: generation_config eos ids + the tokenizer's."""
    ids = set()
    gc = snapshot / "generation_config.json"
    if gc.exists():
        v = json.loads(gc.read_text()).get("eos_token_id", [])
        ids.update(v if isinstance(v, list) else [v])
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


# --------------------------------------------------------------------------------------------
# Experts streamed from the checkpoint
# --------------------------------------------------------------------------------------------
class ExpertStore:
    """Per-expert bf16 weights of one MoE layer, dequantized from the fp8 shards on request.

    ``gate_up(e)`` is ``cat([gate_proj, up_proj], 0)`` -> [2*moe_intermediate, hidden], ``down(e)`` ->
    [hidden, moe_intermediate]: the tensors ``convert_to_hf_module_keys`` stacks into ``gate_up_proj[e]`` /
    ``down_proj[e]``, produced by the same ``dequantize_weight_tensor`` call, so they are bitwise what
    ``build_layer`` holds. Shard handles are shared and kept open (the reads go through the page cache)."""

    _handles: dict[str, object] = {}

    def __init__(self, snapshot: Path, headers: dict, layer_idx: int):
        self.snapshot = Path(snapshot)
        self.headers = headers
        self.prefix = f"{LAYER_PREFIX}{layer_idx}.mlp.experts."
        self.bytes_read = 0
        self.seconds = 0.0

    def _tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        fname = self.headers[key][0]
        h = self._handles.get(fname)
        if h is None:
            h = self._handles[fname] = safe_open(str(self.snapshot / fname), framework="pt", device="cpu")
        t = h.get_tensor(key)
        self.bytes_read += t.numel() * t.element_size()
        return t

    def _weight(self, name: str) -> torch.Tensor:
        t0 = time.perf_counter()
        key = f"{self.prefix}{name}.weight"
        t = self._tensor(key)
        if t.dtype == torch.float8_e4m3fn:
            inv = self._tensor(key.replace(".weight", ".weight_scale_inv"))
            t = dequantize_weight_tensor(t, inv, FP8_BLOCK, dtype=torch.bfloat16)
        else:
            t = t.to(torch.bfloat16)
        self.seconds += time.perf_counter() - t0
        return t

    def gate_up(self, e: int) -> torch.Tensor:
        return torch.cat([self._weight(f"{e}.gate_proj"), self._weight(f"{e}.up_proj")], dim=0)

    def down(self, e: int) -> torch.Tensor:
        return self._weight(f"{e}.down_proj")


class LazyExperts(nn.Module):
    """``Glm5NextTextExperts`` (its eager ``forward`` verbatim) with the weights fetched per hit expert."""

    def __init__(self, config, store: ExpertStore):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.swiglu_limit = config.swiglu_limit
        self.store = store

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            e = int(expert_idx)
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.store.gate_up(e)))
            current = F.linear(current, self.store.down(e)) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


# --------------------------------------------------------------------------------------------
# Indexer scoring in query blocks (same top-k, bounded memory)
# --------------------------------------------------------------------------------------------
def install_blocked_indexer(qblock: int):
    """Replace ``Glm5NextTextIndexer.forward`` by a copy that evaluates the [B, S, H, P] pool scores ``qblock``
    queries at a time (the stock forward holds the fp32 score tensor and two temporaries for the whole chunk)."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, attention_mask, past_key_values):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]
        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)
        kv_len = seq_len
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()
        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(valid_keys=valid_keys, q_length=seq_len, current_length=current_length)
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
        pool_keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)  # [B, 1, D, P]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        blocks = []
        for s in range(0, seq_len, qblock):
            scores = torch.matmul(q[:, s : s + qblock].float(), pool_keys_t)
            scores = F.relu(scores * self.softmax_scale)
            blocks.append(torch.matmul(weights[:, s : s + qblock].unsqueeze(-2), scores).squeeze(-2))
            del scores
        index_scores = torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]
        del blocks
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1))
        valid_candidates = pool_visible & pool_valid[:, None]
        index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
            output_width += self.index_kpool - 1
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)

    if not hasattr(Glm5NextTextIndexer, "_stock_forward"):
        Glm5NextTextIndexer._stock_forward = Glm5NextTextIndexer.forward
    Glm5NextTextIndexer.forward = forward


def selftest_blocked_indexer(text_config, qblock: int = 64) -> bool:
    """Blocked vs stock indexer forward on a random layer: same top-k set per query (prefill + a cached step)."""
    from transformers import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    torch.manual_seed(0)
    idx = Glm5NextTextIndexer(text_config, 3)
    with torch.no_grad():
        for p in idx.parameters():
            p.normal_(0, 0.05)
    idx = idx.to(torch.bfloat16).eval()
    install_blocked_indexer(qblock)
    ok = True
    for S, step in ((300, 1), (257, 5)):
        hs = torch.randn(1, S, text_config.hidden_size).to(torch.bfloat16)
        qr = torch.randn(1, S, text_config.q_lora_rank).to(torch.bfloat16)
        hs2 = torch.randn(1, step, text_config.hidden_size).to(torch.bfloat16)
        qr2 = torch.randn(1, step, text_config.q_lora_rank).to(torch.bfloat16)
        outs = []
        for fwd in (Glm5NextTextIndexer._stock_forward, Glm5NextTextIndexer.forward):
            cache = DynamicCache(config=text_config)
            cache.layers[3].update(torch.zeros(1, 1, S, 8), torch.zeros(1, 1, S, 8), cache_kwargs=None)
            a = fwd(idx, hs, qr, torch.ones(1, S, dtype=torch.bool), cache)
            cache.layers[3].update(torch.zeros(1, 1, step, 8), torch.zeros(1, 1, step, 8), cache_kwargs=None)
            b = fwd(idx, hs2, qr2, torch.ones(1, step, dtype=torch.bool), cache)
            outs.append((a, b))
        for (a0, b0), (a1, b1) in [outs]:
            same = all(
                torch.equal(x.sort(-1).values, y.sort(-1).values) for x, y in ((a0, a1), (b0, b1))
            )  # same selected set per query (order inside the top-k is not part of the contract)
            print(f"selftest blocked indexer S={S} step={step}: {'same top-k' if same else 'DIFFERENT'}", flush=True)
            ok &= same
    return ok


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------
def build_model(snapshot: Path, text_config, attn_impl: str, headers=None, *, verbose=True):
    """``Glm5NextTextModel`` on the host with every non-expert weight resident (bf16 / strict fp32 as
    ``from_pretrained``) and ``LazyExperts`` in the MoE layers. Returns (model, lm_head weight bf16)."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel, Glm5NextTextMoE

    headers = headers or shard_headers(snapshot)
    text_config._attn_implementation = attn_impl
    t0 = time.perf_counter()
    with torch.device("meta"):
        model = Glm5NextTextModel(text_config)
    for li, layer in enumerate(model.layers):
        if isinstance(layer.mlp, Glm5NextTextMoE):
            layer.mlp.experts = LazyExperts(text_config, ExpertStore(snapshot, headers, li))
    model = model.to_empty(device="cpu")
    sd = {
        "embed_tokens.weight": load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY],
        "norm.weight": load_tensors(snapshot, [FINAL_NORM_KEY], headers)[FINAL_NORM_KEY],
    }
    for li in range(text_config.num_hidden_layers):
        raw = load_layer_raw(snapshot, li, headers, name_filter=lambda n: not _EXPERT_RE.match(n))
        for k, v in convert_to_hf_module_keys(raw).items():
            sd[f"layers.{li}.{k}"] = v
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    cast_like_from_pretrained(model)
    lm_w = load_tensors(snapshot, [LM_HEAD_KEY], headers)[LM_HEAD_KEY].to(torch.bfloat16)
    n_res = sum(p.numel() for p in model.parameters()) + lm_w.numel()
    if verbose:
        print(
            f"model built in {time.perf_counter() - t0:.0f} s: {n_res / 1e9:.2f} B resident params "
            f"(+ {sum(1 for l in model.layers if isinstance(l.mlp, Glm5NextTextMoE))} MoE layers streamed), "
            f"attn={model.config._attn_implementation}, rss {_rss_gb():.1f} GB",
            flush=True,
        )
    return model, lm_w


def _rss_gb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return float("nan")


class _LayerTimer:
    """Per-decoder-layer wall time of one forward (``--layer-timing``)."""

    def __init__(self, model):
        self.t = {}
        self.handles = []
        for i, layer in enumerate(model.layers):
            self.handles.append(
                layer.register_forward_pre_hook(lambda m, a, i=i: self.t.__setitem__(i, time.perf_counter()))
            )
            self.handles.append(
                layer.register_forward_hook(
                    lambda m, a, o, i=i: print(
                        f"    layer {i:2d}: {time.perf_counter() - self.t[i]:6.1f} s", flush=True
                    )
                )
            )


@torch.inference_mode()
def generate(model, lm_w, ids: list[int], gen: int, chunk: int, stop: set[int], *, log=print) -> dict:
    """Chunked prefill of ``ids`` through the HF cache, then greedy decode of up to ``gen`` tokens (the first from
    the prefill's last-token logits), stopping after an eos like the mesh test. Returns ids, logits top-20, times."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    S = len(ids)
    x = torch.tensor([ids], dtype=torch.long)
    t_pre = time.perf_counter()
    chunks = []
    out = None
    for s0 in range(0, S, chunk):
        e = min(S, s0 + chunk)
        t0 = time.perf_counter()
        out = model(input_ids=x[:, s0:e], past_key_values=cache, use_cache=True)
        chunks.append([s0, e, round(time.perf_counter() - t0, 1)])
        log(
            f"prefill chunk [{s0}, {e}): {chunks[-1][2]:.1f} s ({(e - s0) / chunks[-1][2]:.0f} tok/s), rss {_rss_gb():.1f} GB"
        )
    logits = F.linear(out.last_hidden_state[:, -1], lm_w)[0].float()  # bf16 head, as the model's
    prefill_s = time.perf_counter() - t_pre
    top = logits.topk(20)
    first = int(logits.argmax())
    log(
        f"prefill {S} tokens: {prefill_s:.1f} s ({S / prefill_s:.0f} tok/s); first token {first}; top-5 {top.indices[:5].tolist()}"
    )
    gen_ids = [first]
    steps = [
        {
            "position": S - 1,
            "token": first,
            "top5_ids": top.indices[:5].tolist(),
            "top5_logits": [round(float(v), 4) for v in top.values[:5]],
            "s": round(prefill_s, 1),
        }
    ]
    step_s = []
    while len(gen_ids) < gen and gen_ids[-1] not in stop:
        t0 = time.perf_counter()
        out = model(input_ids=torch.tensor([[gen_ids[-1]]], dtype=torch.long), past_key_values=cache, use_cache=True)
        lg = F.linear(out.last_hidden_state[:, -1], lm_w)[0].float()
        tid = int(lg.argmax())
        step_s.append(time.perf_counter() - t0)
        t5 = lg.topk(5)
        steps.append(
            {
                "position": S + len(gen_ids) - 1,
                "token": tid,
                "top5_ids": t5.indices.tolist(),
                "top5_logits": [round(float(v), 4) for v in t5.values],
                "s": round(step_s[-1], 1),
            }
        )
        gen_ids.append(tid)
        log(
            f"decode step {len(gen_ids) - 1} (pos {S + len(gen_ids) - 2}): token {tid} in {step_s[-1]:.1f} s; top-5 {t5.indices.tolist()}"
        )
    return {
        "isl": S,
        "first_token": first,
        "generated": gen_ids,
        "steps": steps,
        "last_logits_top20_ids": top.indices.tolist(),
        "last_logits_top20_values": [float(v) for v in top.values],
        "last_logits": logits,
        "prefill_s": round(prefill_s, 1),
        "prefill_chunks": chunks,
        "decode_step_s": [round(s, 2) for s in step_s],
        "decode_s": round(sum(step_s), 1),
    }


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_needle(
    model, lm_w, tok, snapshot: Path, n: int, depth: float, gen: int, chunk: int, out_root: Path, meta: dict
) -> dict:
    import transformers

    out_dir = out_root / f"NEEDLE_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "run.log", "a")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t_all = time.perf_counter()
    ids, text, consts = needle_prompt(tok, n, depth)
    ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    (out_dir / "prompt_ids.json").write_text(
        json.dumps(
            {
                "n_requested": n,
                "depth": depth,
                "isl": len(ids),
                "sha256": ids_sha,
                "ids": ids,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    )
    log(f"NEEDLE_{n}: {len(ids)} prompt tokens (sha256 {ids_sha[:16]}), depth {depth}, chunk {chunk}, gen {gen}")
    stop = eos_ids(snapshot, tok)
    res = generate(model, lm_w, ids, gen, chunk, stop, log=log)
    torch.save(res.pop("last_logits"), out_dir / "last_logits.pt")
    out_text = tok.decode([t for t in res["generated"] if t not in stop], skip_special_tokens=True)
    found = consts["needle_key"] in out_text
    wall = time.perf_counter() - t_all
    record = {
        "n_requested": n,
        "isl": len(ids),
        "depth": depth,
        "prompt_sha256": ids_sha,
        "gen": gen,
        "chunk": chunk,
        "first_token": res["first_token"],
        "first_token_text": tok.decode([res["first_token"]]),
        "generated": res["generated"],
        "text": out_text,
        "text_raw": tok.decode(res["generated"]),
        "needle_found": found,
        "needle_key": consts["needle_key"],
        "stopped_at_eos": res["generated"][-1] in stop,
        "last_logits_top20_ids": res["last_logits_top20_ids"],
        "last_logits_top20_values": res["last_logits_top20_values"],
        "steps": res["steps"],
        "prefill_s": res["prefill_s"],
        "prefill_chunks": res["prefill_chunks"],
        "decode_step_s": res["decode_step_s"],
        "decode_s": res["decode_s"],
        "wall_s": round(wall, 1),
        "config": {
            **meta,
            "attn_implementation": model.config._attn_implementation,
            "experts": "eager loop, streamed from the fp8 shards (LazyExperts)",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
    }
    (out_dir / "reference.json").write_text(json.dumps(record, indent=1))
    log(
        f"=== NEEDLE_{n}: isl {len(ids)} first {res['first_token']} gen {res['generated']}\n"
        f"text {out_text!r}\nneedle {'FOUND' if found else 'MISSING'}; prefill {res['prefill_s']:.1f} s, "
        f"decode {res['decode_s']:.1f} s ({len(res['decode_step_s'])} steps), wall {wall:.1f} s -> {out_dir}"
    )
    log_file.close()
    return record


def validate_g16(model, lm_w, tok, snapshot: Path, g16: Path, chunk: int) -> bool:
    """Re-run the FULLMODEL_G16 prompt (13 tokens, 16 greedy tokens with per-step top-5) through this model."""
    ref = json.loads(g16.read_text())
    ids = ref["prompt_ids"]
    res = generate(model, lm_w, ids, len(ref["generated"]), chunk, set())
    ok = res["generated"] == ref["generated"]
    print(f"G16 tokens ref {ref['generated']}\n           got {res['generated']} -> {'MATCH' if ok else 'MISMATCH'}")
    for r, s in zip(ref["records"], res["steps"]):
        d = max(abs(a - b) for a, b in zip(r["top5_logits"], s["top5_logits"]))
        print(
            f"  step {r['step']:2d}: top5 ids {'same' if r['top5_ids'] == s['top5_ids'] else 'DIFF ' + str(s['top5_ids'])}; max |dlogit| {d:.3f}"
        )
    print(f"G16 text: {tok.decode(res['generated'])!r}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="*", default=[4096], help="requested prompt token counts (the test's N)")
    ap.add_argument("--depth", type=float, default=DEPTH_DEFAULT)
    ap.add_argument("--gen", type=int, default=GEN_DEFAULT, help="greedy tokens incl. the first")
    ap.add_argument("--chunk", type=int, default=16384, help="prefill chunk (tokens per forward through the cache)")
    ap.add_argument("--index-qblock", type=int, default=2048, help="indexer scoring query block; 0 = stock forward")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("bringup/references/glm53_flash"))
    ap.add_argument("--layer-timing", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="blocked-indexer check only (no model)")
    ap.add_argument(
        "--validate-g16",
        nargs="?",
        const="bringup/references/glm53_flash/FULLMODEL_G16/generate.json",
        default=None,
        metavar="JSON",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    snapshot = args.snapshot or default_snapshot_dir()
    cfg, text = load_config(snapshot)
    if args.selftest:
        sys.exit(0 if selftest_blocked_indexer(text) else 1)
    if args.index_qblock > 0:
        install_blocked_indexer(args.index_qblock)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snapshot))
    print(
        f"snapshot {snapshot}\nthreads {torch.get_num_threads()}, attn {args.attn}, chunk {args.chunk}, index qblock {args.index_qblock}",
        flush=True,
    )
    model, lm_w = build_model(snapshot, text, args.attn)
    if args.layer_timing:
        _LayerTimer(model)
    meta = {
        "snapshot": str(snapshot),
        "index_qblock": args.index_qblock,
        "git": _git_sha(Path(__file__).resolve().parent),
    }
    if args.validate_g16:
        sys.exit(0 if validate_g16(model, lm_w, tok, snapshot, Path(args.validate_g16), args.chunk) else 1)
    for n in args.n:
        run_needle(model, lm_w, tok, snapshot, n, args.depth, args.gen, args.chunk, args.out, meta)


if __name__ == "__main__":
    main()
