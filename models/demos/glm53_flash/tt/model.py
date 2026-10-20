# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash full model on ONE chip by weight streaming (design Phase 3, teacher-forced check).

``GLM53Model`` chains the stage-1 ``FunctionalDecoder`` through all 45 layers plus the embedding
(``tt/embedding.py``) and the head (``tt/model_head.py``). One chip holds one layer's weights
(~7.7 GB bfp8 for a MoE layer) next to the embedding table (1.27 GB bf16), the LM head (0.65 GB
bfp8) and the autoregressive state of EVERY layer, so a pass over the stack streams the weights:

    for each layer i:  build (checkpoint -> device) -> run -> release (deallocate every weight)

The per-layer ``FunctionalDecoder`` objects live for the whole model: their state (KDA user slots,
DSA paged latent cache + pooled-key index, ``next_start``) persists across the passes, only the
``TtDecoderLayer`` (weights, folded constants) is rebuilt per pass (``release_weights`` /
``attach_layer``). A build reads the layer's raw tensors (``reference/checkpoint.load_layer_raw``)
and converts them; with ``weight_cache_root`` the routed experts (95 % of the bytes) and the KDA
prefill weights come from tensorbins (``ttnn.load_tensor``, ~2.6 s for a MoE layer against ~50 s
of host bfp8 conversion) and the expert tensors are not even dequantized.

Residual contract between the pieces: packed mHC streams ``[1, 1, T, 4*4096]`` fp32 (the
embedding repeats the token row 4x, every layer maps streams to streams, the head averages them).
Decode runs the ``max_batch`` user slots of every layer (B rows); ``generate`` is greedy with one
user. Nothing here is traced or batched across layers: it is the functional reference path of the
full model, not a serving path.
"""

from __future__ import annotations

import gc
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.decoder_layer import TtDecoderLayer
from models.demos.glm53_flash.tt.dsa import SparseDecodeMeta, decode_position_tensors
from models.demos.glm53_flash.tt.embedding import TtEmbedding, padded_rows
from models.demos.glm53_flash.tt.functional_decoder import DecodePositions, FunctionalDecoder
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.moe import TtMoE

_EXPERT_RE = re.compile(r"^mlp\.experts\.\d+\.")


@dataclass
class LayerTiming:
    """Per-layer build / run / release wall times of every pass (seconds)."""

    kind: str
    build_s: list[float] = field(default_factory=list)
    run_s: list[tuple[str, float]] = field(default_factory=list)  # (phase, seconds)
    release_s: list[float] = field(default_factory=list)
    cache_hits: list[bool] = field(default_factory=list)
    dram_built_gib: list[float] = field(default_factory=list)  # allocated right after the build (the pass peak)
    released_buffers: list[int] = field(default_factory=list)


def dram_view(device) -> tuple[float, float]:
    """(allocated GiB, free GiB) of the device DRAM from the allocator (sum over the banks)."""
    mv = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    gib = 2**30
    return mv.num_banks * mv.total_bytes_allocated_per_bank / gib, mv.num_banks * mv.total_bytes_free_per_bank / gib


class GLM53Model:
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        snapshot: Path,
        headers=None,
        *,
        layers: list[int] | None = None,
        max_batch: int = 1,
        max_seq_len: int = 256,
        weight_cache_root: Path | str | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
    ):
        """``layers``: the layer subset to chain (all 45 by default; a prefix like ``0..4`` is only
        meaningful for mechanics checks against the per-layer dumps). ``weight_cache_root``: directory
        with one ``layer<i>`` subdirectory per layer (tensorbins written on the first build)."""
        self.device = device
        self.cfg = cfg
        self.snapshot = Path(snapshot)
        self.headers = headers or shard_headers(self.snapshot)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.weight_cache_root = Path(weight_cache_root) if weight_cache_root else None
        self.weight_dtype = weight_dtype
        self.expert_dtype = expert_dtype
        self.expert_down_dtype = expert_down_dtype
        self.mesh = GLMMesh.from_device(device)
        t0 = time.perf_counter()
        self.embedding = TtEmbedding.from_checkpoint(device, cfg, self.snapshot, self.headers)
        self.head = TtModelHead.from_checkpoint(device, cfg, self.snapshot, self.headers)
        self.embed_head_build_s = time.perf_counter() - t0
        self.decoders: dict[int, FunctionalDecoder] = {}  # state owners, created on the first pass
        self.timing: dict[int, LayerTiming] = {
            i: LayerTiming(f"{'KDA' if cfg.is_kda(i) else 'DSA'}+{'MoE' if cfg.is_moe(i) else 'dense'}")
            for i in self.layers
        }
        self.passes = 0

    @classmethod
    def from_snapshot(cls, device, snapshot: Path | str | None = None, **kw) -> "GLM53Model":
        snapshot = Path(snapshot) if snapshot else default_snapshot_dir()
        cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
        return cls(device, cfg, snapshot, **kw)

    # ------------------------------------------------------------- weight streaming
    def layer_cache_dir(self, i: int) -> Path | None:
        return self.weight_cache_root / f"layer{i}" if self.weight_cache_root else None

    def _expert_cache_complete(self, i: int) -> bool:
        return self.cfg.is_moe(i) and TtMoE.expert_cache_complete(
            self.layer_cache_dir(i), self.mesh, self.expert_dtype, self.expert_down_dtype
        )

    def build_layer(self, i: int) -> FunctionalDecoder:
        """Load layer ``i``'s weights onto the device: the first time a ``FunctionalDecoder`` (state
        allocated), afterwards a fresh ``TtDecoderLayer`` attached to the persistent decoder."""
        t0 = time.perf_counter()
        hit = self._expert_cache_complete(i)
        name_filter = (lambda n: not _EXPERT_RE.match(n)) if hit else None  # skip dequantizing cached experts
        raw = load_layer_raw(self.snapshot, i, self.headers, name_filter=name_filter)
        layer = TtDecoderLayer(
            self.device,
            self.cfg,
            i,
            raw,
            weight_cache_path=self.layer_cache_dir(i),
            weight_dtype=self.weight_dtype,
            expert_dtype=self.expert_dtype,
            expert_down_dtype=self.expert_down_dtype,
            mesh=self.mesh,
        )
        del raw
        dec = self.decoders.get(i)
        if dec is None:
            dec = FunctionalDecoder(layer, max_batch=self.max_batch, max_seq_len=self.max_seq_len)
            self.decoders[i] = dec
        else:
            dec.attach_layer(layer)
        ttnn.synchronize_device(self.device)
        tm = self.timing[i]
        tm.build_s.append(time.perf_counter() - t0)
        tm.cache_hits.append(hit)
        tm.dram_built_gib.append(dram_view(self.device)[0])
        return dec

    def release_layer(self, i: int) -> int:
        t0 = time.perf_counter()
        n = self.decoders[i].release_weights()
        gc.collect()
        tm = self.timing[i]
        tm.release_s.append(time.perf_counter() - t0)
        tm.released_buffers.append(n)
        return n

    def _run_layers(self, x: ttnn.Tensor, step, phase: str, on_layer=None) -> ttnn.Tensor:
        """Stream the layers over ``x`` (packed streams): build -> ``step(dec, x)`` -> release.
        ``on_layer(i, out)`` sees every layer's output (device tensor, read-only) before it is consumed."""
        for i in self.layers:
            dec = self.build_layer(i)
            t0 = time.perf_counter()
            y = step(dec, x)
            ttnn.synchronize_device(self.device)
            self.timing[i].run_s.append((phase, time.perf_counter() - t0))
            if on_layer is not None:
                on_layer(i, y)
            if y is not x:
                ttnn.deallocate(x)
            x = y
            self.release_layer(i)
        self.passes += 1
        return x

    # ----------------------------------------------------------------- forwards
    def prefill(
        self,
        token_ids: torch.Tensor,
        *,
        user: int = 0,
        start: int = 0,
        all_rows: bool = False,
        on_layer=None,
    ) -> ttnn.Tensor:
        """One user's prompt (``token_ids`` any-shape ints, ``S`` tokens) -> logits ``[1, 1, 1, vocab]``
        bf16 of the last token, or ``[1, 1, T_pad, vocab]`` for every row with ``all_rows`` (rows past
        ``S`` are padding). ``start`` > 0 continues a chunked prompt (``FunctionalDecoder`` contract)."""
        ids = token_ids.reshape(-1)
        S = int(ids.numel())
        x = self.embedding(self.embedding.tokens_to_device(ids, pad_to=padded_rows(S)))
        x = self._run_layers(
            x, lambda dec, h: dec.prefill_forward(h, user=user, seq_len=S, start=start), "prefill", on_layer
        )
        logits = self.head(x) if all_rows else self.head.last_token_logits(x, S)
        ttnn.deallocate(x)
        return logits

    def decode_positions(self, positions: list[int]) -> DecodePositions:
        if len(positions) != self.max_batch:
            raise ValueError(f"{len(positions)} positions for {self.max_batch} user slots")
        pos, pool_ids, ring_slots = decode_position_tensors(self.device, positions, self.cfg.index_kpool)
        sparse = SparseDecodeMeta.from_positions(self.device, positions, self.cfg)
        return DecodePositions(pos, pool_ids, ring_slots, list(positions), sparse=sparse)

    def decode(self, token_ids: list[int], positions: list[int], *, on_layer=None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One token per user slot at ``positions`` -> (logits ``[1, 1, B, vocab]`` bf16, greedy ids
        ``[1, 1, B, 1]`` uint32 from the device argmax)."""
        B = self.max_batch
        if len(token_ids) != B:
            raise ValueError(f"{len(token_ids)} tokens for {B} user slots")
        dp = self.decode_positions(positions)
        x = self.embedding(self.embedding.tokens_to_device(torch.tensor(token_ids, dtype=torch.int32), pad_to=B))
        last = self.cfg.num_hidden_layers - 1
        x = self._run_layers(
            x,
            lambda dec, h: dec.decode_forward(
                h, positions=dp, collapse=dec.layer_idx == last and dec.supports_collapse
            ),
            "decode",
            on_layer,
        )
        logits, ids = self.head.forward_greedy(x)  # a collapsed [1,1,B,hidden] passes straight to the norm
        ttnn.deallocate(x)
        for t in (dp.positions, dp.pool_ids, dp.ring_slots):
            ttnn.deallocate(t)
        return logits, ids

    # ---------------------------------------------------------------- generation
    @staticmethod
    def host_logits(logits: ttnn.Tensor) -> torch.Tensor:
        """Device logits ``[1, 1, rows, vocab]`` -> host fp32 ``[rows, vocab]``."""
        return ttnn.to_torch(logits).float()[0, 0]

    def generate(
        self, prompt_ids: torch.Tensor, n: int, *, user: int = 0, on_token=None
    ) -> tuple[list[int], list[dict]]:
        """Greedy generation of ``n`` tokens for one user (``max_batch == 1``): a prefill pass, then
        ``n - 1`` decode passes (each streams all layers). The next token is the host argmax of the
        device logits (first index on ties, like torch); the device greedy id is recorded next to it.
        Returns the ids and per-step records (position, top-5 ids / logits, pass seconds)."""
        if self.max_batch != 1:
            raise ValueError("generate is a single-user loop (max_batch == 1)")
        ids = prompt_ids.reshape(-1)
        S = int(ids.numel())
        if S + n > self.max_seq_len:
            raise ValueError(f"prompt {S} + {n} tokens exceed max_seq_len {self.max_seq_len}")
        out: list[int] = []
        records: list[dict] = []

        def take(logits_host: torch.Tensor, position: int, seconds: float, device_id: int | None):
            top = logits_host.topk(5)
            tok = int(logits_host.argmax())
            records.append(
                {
                    "step": len(out),
                    "position": position,
                    "token": tok,
                    "device_greedy": device_id,
                    "top5_ids": top.indices.tolist(),
                    "top5_logits": [round(float(v), 4) for v in top.values],
                    "pass_s": round(seconds, 1),
                }
            )
            out.append(tok)
            if on_token is not None:
                on_token(records[-1])

        t0 = time.perf_counter()
        logits = self.prefill(ids, user=user)
        take(self.host_logits(logits)[0], S - 1, time.perf_counter() - t0, None)
        ttnn.deallocate(logits)
        while len(out) < n:
            t0 = time.perf_counter()
            pos = S + len(out) - 1  # the last generated token is fed at this position
            logits, dev_ids = self.decode([out[-1]], [pos])
            dev_id = int(ttnn.to_torch(dev_ids).view(torch.int32).reshape(-1)[0])
            take(self.host_logits(logits)[0], pos, time.perf_counter() - t0, dev_id)
            ttnn.deallocate(logits)
            ttnn.deallocate(dev_ids)
        return out, records

    # ------------------------------------------------------------------- reports
    def timing_table(self) -> str:
        """One line per layer: kind, builds (cache hits), mean build / run (per phase) / release seconds."""
        lines = ["layer kind      builds hits  build_s  prefill_s decode_s release_s dram_peak_gib"]
        for i in self.layers:
            t = self.timing[i]
            pre = [s for p, s in t.run_s if p == "prefill"]
            dec = [s for p, s in t.run_s if p == "decode"]
            mean = lambda v: f"{sum(v) / len(v):8.2f}" if v else "       -"
            lines.append(
                f"{i:5d} {t.kind:9s} {len(t.build_s):6d} {sum(t.cache_hits):4d} {mean(t.build_s)} {mean(pre):9s} "
                f"{mean(dec)} {mean(t.release_s)} {max(t.dram_built_gib) if t.dram_built_gib else 0:13.2f}"
            )
        return "\n".join(lines)

    def cache_size_bytes(self) -> int:
        if not self.weight_cache_root or not self.weight_cache_root.exists():
            return 0
        return sum(f.stat().st_size for f in self.weight_cache_root.rglob("*.tensorbin"))
