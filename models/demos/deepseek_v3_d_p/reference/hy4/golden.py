# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4 golden-trace layout, key-frame transforms and the comparison metrics.

Layout (the row-sharded "vLLM structured trace" the prefill producer and the chunked tests already
read -- ``prefill_kv_validation._load_golden_kv_post`` / ``_load_golden_index_k``)::

    <trace>/metadata.json                         token_ids, n_prompt_tokens, frames, provenance
    <trace>/kv_cache/layer_{L}/rows_{a:08d}_{b:08d}.safetensors
            kv_post_transform_layer_{L}  [rows, 576]   all 78 layers
    <trace>/dsa/indexer_k_layer_{L}/rows_*.safetensors
            indexer_k_layer_{L}          [rows, 128]   full-indexer layers only (global layer index L)
    <trace>/dsa/dsa_topk_indices_layer_{L}/rows_*.safetensors
            dsa_topk_indices_layer_{L}   [rows, 2048]  int32, full-indexer layers only, -1 = no key
    <trace>/final/logits_last.safetensors          next-token logits at the last prompt position

Frames (what the producer compares against):
  * KVPE golden is the HF frame: ``[kv_a_layernorm(latent) (512) | k_pe (64), half-split RoPE]``. The
    producer re-bases the pe half itself (``cache_half_pccs(pe_interleave=True)``), exactly as for GLM-5.3.
  * Index-key golden is written in the DEVICE frame by default: the producer compares it directly against
    the de-Hadamarded device key. Hy4 ropes the LAST 64 of the 128 index dims, half-split; the device
    indexer (``TtIndexer``) ropes the FIRST 64, interleaved. A port that permutes the indexer weights'
    output columns (``wk`` rows, ``k_norm`` weight/bias, ``wq_b`` per-head rows) by
    ``index_k_device_perm()`` produces exactly ``index_k_hf_to_device(hf_key)``: the permutation commutes
    with the LayerNorm, maps half-split RoPE pairs onto interleaved ones at the same frequency, and cancels
    in every q.k score (so top-k is unchanged). ``--index-k-frame hf`` writes the HF frame instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.hy4.config import Hy4RefConfig

ROWS_PER_SHARD = 5120  # the producer/vLLM chunk; any value works, the readers parse the row ranges
TRACE_FORMAT = "hy4_golden_v1"


# ------------------------------------------------------------------------------------------ frames
def half_split_to_interleaved_perm(d: int) -> torch.Tensor:
    """``out = x[..., perm]`` turns a half-split RoPE vector (pairs ``(i, i+d/2)``) into an interleaved
    one (pairs ``(2i, 2i+1)``) at the same frequencies. Same convention as ``utils.test_utils.interleave_pe``
    and ``tt.mla.rope.interleaved_perm_matrix``."""
    h = d // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=-1).reshape(d)


def index_k_device_perm(index_head_dim: int, rope_dim: int) -> torch.Tensor:
    """Column permutation HF index key -> device frame: ``[nope | rope(half-split)]`` ->
    ``[rope(interleaved) | nope]``. Apply the SAME permutation to the output rows of ``wk``, to
    ``k_norm.weight``/``bias`` and to each head's rows of ``wq_b`` to make the device emit this frame."""
    nope = index_head_dim - rope_dim
    rope = nope + half_split_to_interleaved_perm(rope_dim)
    return torch.cat([rope, torch.arange(nope)])


def index_k_hf_to_device(k: torch.Tensor, rope_dim: int) -> torch.Tensor:
    return k[..., index_k_device_perm(k.shape[-1], rope_dim)]


def index_k_device_to_hf(k: torch.Tensor, rope_dim: int) -> torch.Tensor:
    return k[..., torch.argsort(index_k_device_perm(k.shape[-1], rope_dim))]


# -------------------------------------------------------------------------------------- the writer
def _shard_name(a: int, b: int) -> str:
    return f"rows_{a:08d}_{b:08d}.safetensors"


def _write_rows(root: Path, key: str, t: torch.Tensor, rows_per_shard: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for a in range(0, t.shape[0], rows_per_shard):
        b = min(a + rows_per_shard, t.shape[0])
        save_file({key: t[a:b].contiguous()}, str(root / _shard_name(a, b)))


class GoldenTraceWriter:
    """Sink for ``Hy4PrefillReference.run`` that writes a trace dir the producer reads.

    ``metadata.json`` is written up front with ``complete: false`` and the list of layers written so far,
    and rewritten after every layer, so a killed run leaves a trace that reads as partial (the producer
    then fails on the missing layers rather than scoring a subset)."""

    def __init__(
        self,
        out_dir,
        cfg: Hy4RefConfig,
        token_ids,
        *,
        rows_per_shard: int = ROWS_PER_SHARD,
        kv_dtype: torch.dtype = torch.float32,
        index_k_frame: str = "device",
        save_topk: bool = True,
        extra_metadata: dict | None = None,
    ):
        if index_k_frame not in ("device", "hf"):
            raise ValueError(f"index_k_frame must be 'device' or 'hf', got {index_k_frame!r}")
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.rows_per_shard = rows_per_shard
        self.kv_dtype = kv_dtype
        self.index_k_frame = index_k_frame
        self.save_topk = save_topk
        ids = [int(t) for t in torch.as_tensor(token_ids).flatten().tolist()]
        self.meta = {
            "format": TRACE_FORMAT,
            "model": "hy4_preview",
            "model_type": "hy_v4",
            "token_ids": ids,
            "n_prompt_tokens": len(ids),
            "num_layers": cfg.num_hidden_layers,
            "full_indexer_layers": cfg.full_indexer_layers,
            "index_topk": cfg.index_topk,
            "kv_dtype": str(kv_dtype).replace("torch.", ""),
            "frames": {
                "kvpe": {
                    "layout": f"[kv_a_layernorm(latent) {cfg.kv_lora_rank} | k_pe {cfg.qk_rope_head_dim}]",
                    "pe_rope": "half_split",
                },
                "index_k": {
                    "frame": index_k_frame,
                    "layout": (
                        f"[rope {cfg.qk_rope_head_dim} interleaved | nope {cfg.index_head_dim - cfg.qk_rope_head_dim}]"
                        if index_k_frame == "device"
                        else f"[nope {cfg.index_head_dim - cfg.qk_rope_head_dim} | rope {cfg.qk_rope_head_dim} half_split]"
                    ),
                    "hadamard": False,
                },
                "topk": {"order": "descending score", "invalid": -1, "positions": "absolute"},
            },
            "layers_written": [],
            "complete": False,
            **(extra_metadata or {}),
        }
        self._flush_meta()

    def _flush_meta(self) -> None:
        tmp = self.out / "metadata.json.tmp"
        tmp.write_text(json.dumps(self.meta))
        tmp.replace(self.out / "metadata.json")

    def on_layer(self, layer: int, *, kvpe, index_k=None, topk=None, streams=None) -> None:
        cfg = self.cfg
        assert kvpe.shape[-1] == cfg.kvpe_dim, f"layer {layer}: KVPE width {kvpe.shape[-1]} != {cfg.kvpe_dim}"
        key = f"kv_post_transform_layer_{layer}"
        _write_rows(self.out / "kv_cache" / f"layer_{layer}", key, kvpe.to(self.kv_dtype), self.rows_per_shard)
        if index_k is not None:
            k = index_k_hf_to_device(index_k, cfg.qk_rope_head_dim) if self.index_k_frame == "device" else index_k
            key = f"indexer_k_layer_{layer}"
            _write_rows(self.out / "dsa" / key, key, k.to(self.kv_dtype), self.rows_per_shard)
        if topk is not None and self.save_topk:
            key = f"dsa_topk_indices_layer_{layer}"
            _write_rows(self.out / "dsa" / key, key, topk.to(torch.int32), self.rows_per_shard)
        self.meta["layers_written"].append(layer)
        self._flush_meta()

    def on_final(self, *, logits: torch.Tensor, position: int) -> None:
        (self.out / "final").mkdir(exist_ok=True)
        save_file(
            {"logits_last": logits.to(torch.float32).contiguous()}, str(self.out / "final" / "logits_last.safetensors")
        )
        self.meta["next_token_id"] = int(logits.argmax().item())
        self.meta["next_token_position"] = int(position)

    def finalize(self, **extra) -> Path:
        written = sorted(self.meta["layers_written"])
        self.meta["layers_written"] = written
        self.meta["complete"] = written == list(range(self.cfg.num_hidden_layers))
        self.meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.meta.update(extra)
        self._flush_meta()
        return self.out


# ------------------------------------------------------------------------------------------ reading
def load_rows(trace_dir, group: str, key: str, start: int = 0, length: int | None = None) -> torch.Tensor:
    """Rows ``[start, start+length)`` of one row-sharded stream (``group`` = ``kv_cache/layer_L`` or
    ``dsa/<key>``), in its stored dtype."""
    root = Path(trace_dir) / group
    shards = sorted(root.glob("rows_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise FileNotFoundError(f"no shards under {root}")
    end = None if length is None else start + length
    parts = []
    for s in shards:
        a, b = (int(x) for x in s.stem.split("_")[1:3])
        if b <= start or (end is not None and a >= end):
            continue
        with safe_open(str(s), framework="pt") as f:
            t = f.get_tensor(key)
        parts.append(t[max(start - a, 0) : (t.shape[0] if end is None else min(end - a, t.shape[0]))])
    out = torch.cat(parts, dim=0)
    if length is not None and out.shape[0] != length:
        raise FileNotFoundError(f"{root} covers only {out.shape[0]} of rows [{start},{end})")
    return out


def read_metadata(trace_dir) -> dict:
    return json.loads((Path(trace_dir) / "metadata.json").read_text())


# ------------------------------------------------------------------------------------------ metrics
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over all elements, in float64 (1.0 for two identical constant tensors)."""
    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else 0.0


def kvpe_half_pccs(golden: torch.Tensor, other: torch.Tensor, kv_lora_rank: int, other_pe_interleaved: bool):
    """(nope PCC, pe PCC) of a KVPE block vs the HF-frame golden; re-frames the golden's pe when ``other``
    is a device dump (interleaved pe), the same convention ``cache_half_pccs`` applies in the producer."""
    g_pe = golden[:, kv_lora_rank:]
    if other_pe_interleaved:
        g_pe = g_pe[:, half_split_to_interleaved_perm(g_pe.shape[-1])]
    return pcc(golden[:, :kv_lora_rank], other[:, :kv_lora_rank]), pcc(g_pe, other[:, kv_lora_rank:])


def topk_agreement(ref: torch.Tensor, other: torch.Tensor, row_block: int = 4096) -> torch.Tensor:
    """Per query row, the fraction of ``ref``'s valid (>= 0) selected positions that ``other`` also selected.
    Rows where ``ref`` selects nothing count as 1.0. Order inside a row does not matter; the narrower
    side is padded with -1 (HF returns ``min(topk, S)`` columns, the golden always ``topk``)."""
    assert ref.shape[0] == other.shape[0], f"{tuple(ref.shape)} vs {tuple(other.shape)}"
    width = max(ref.shape[1], other.shape[1])
    ref, other = (F.pad(t, (0, width - t.shape[1]), value=-1) for t in (ref, other))
    n = ref.shape[0]
    out = torch.empty(n, dtype=torch.float64)
    span = int(max(ref.max().item(), other.max().item())) + 2  # row stride keeping rows disjoint
    for a in range(0, n, row_block):
        b = min(a + row_block, n)
        r, o = ref[a:b].long(), other[a:b].long()
        off = torch.arange(b - a).unsqueeze(1) * span
        r_valid = r >= 0
        hit = torch.isin(r + off, torch.where(o >= 0, o + off, torch.full_like(o, -1))) & r_valid
        cnt = r_valid.sum(1)
        out[a:b] = torch.where(
            cnt > 0, hit.sum(1).double() / cnt.clamp(min=1), torch.ones_like(cnt, dtype=torch.float64)
        )
    return out
