"""Music3Generator: caption + lyrics -> song on Tenstorrent.

Phase A split: global LLM (tt_transformers, traced decode) + depth decoder (TTNN, traced step) + DiT (TTNN, traced
per window length) on the chip; frame embedding, sampling, scheduler math, condition encoder and vocoder on the host
(torch). The generation recipe and the torch.Generator consumption order are the reference's (reference/pipeline.py),
so the same seed drives the same sampling decisions given the same logits.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

import ttnn
from models.autoports.minimaxai_minimax_music3.config import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    DIT_NUM_STEPS,
    FRAME_RATE,
    MAX_AUDIO_FRAMES,
    NUM_CODEBOOKS,
    SEMANTIC_VOCAB_SIZE,
    SLICED_VOCAB,
    Music3Config,
    resolve_snapshot,
)
from models.autoports.minimaxai_minimax_music3.reference.pipeline import Music3Reference
from models.autoports.minimaxai_minimax_music3.reference.prompt import Music3Tokenizer, build_text_ids
from models.autoports.minimaxai_minimax_music3.reference.sampling import (
    guided_c0_logits_sliced,
    guided_depth_logits,
    sample_top_k,
)
from models.autoports.minimaxai_minimax_music3.tt import weights as W


@dataclass
class GenStats:
    prompt_len: int = 0
    frames: int = 0
    ended: bool = False
    prefill_s: float = 0.0
    semantic_s: float = 0.0
    llm_s: float = 0.0
    depth_s: float = 0.0
    denoise_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    audio_s: float = 0.0
    chunks: int = 0

    @property
    def rtf(self):
        return self.total_s / self.audio_s if self.audio_s else float("inf")

    @property
    def ms_per_frame_llm(self):
        return 1000.0 * self.llm_s / max(self.frames, 1)

    @property
    def ms_per_frame_depth(self):
        return 1000.0 * self.depth_s / max(self.frames, 1)

    def as_dict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(rtf=self.rtf, ms_per_frame_llm=self.ms_per_frame_llm, ms_per_frame_depth=self.ms_per_frame_depth)
        return d


class CPUDepthDecoder:
    """Torch fallback with the TTDepthDecoder.frame() API (bf16 on CPU)."""

    def __init__(self, snapshot, cfg, full_embed: torch.Tensor, dtype=torch.bfloat16):
        from models.autoports.minimaxai_minimax_music3.reference.depth_decoder import Music3DepthDecoder

        self.m = Music3DepthDecoder.load(snapshot, dtype=dtype, cfg=cfg)
        self.full_embed = full_embed.to(dtype)
        self.dtype = dtype

    @torch.inference_mode()
    def frame(self, last_hidden, semantic_code: int, choose, forced=None, record=None):
        d = self.m
        lh = last_hidden.to(self.dtype)
        seq = [
            d.projection(lh).unsqueeze(1),
            d.projection(self.full_embed[semantic_code + AUDIO_CODE_OFFSET]).view(1, 1, -1).expand(2, 1, -1),
        ]
        codes, parts = [int(semantic_code)], []
        for index in range(1, NUM_CODEBOOKS):
            h = d(torch.cat(seq, dim=1))[:, -1]
            parts.append(h[:1].float())
            logits = d.audio_heads[index - 1](h).float()
            if record is not None:
                record.setdefault("depth_logits", []).append(logits.clone())
                record.setdefault("depth_hidden", []).append(h[:1].float().clone())
            code = int(forced[index - 1]) if forced is not None else int(choose(guided_depth_logits(logits), index))
            codes.append(code)
            if index < NUM_CODEBOOKS - 1:
                seq.append(
                    d.projection(d.audio_embeddings.weight[code + (index - 1) * AUDIO_VOCAB_SIZE])
                    .view(1, 1, -1)
                    .expand(2, 1, -1)
                )
        return torch.tensor(codes, dtype=torch.int64), torch.cat(parts, dim=0)

    def warmup(self):
        pass

    def release(self):
        pass


class Music3Generator:
    def __init__(
        self,
        mesh_device,
        snapshot: Optional[str] = None,
        *,
        max_seq_len: int = 16384,
        llm_dtype=ttnn.bfloat8_b,
        depth_dtype=ttnn.bfloat16,
        dit_dtype=ttnn.bfloat16,
        load_llm: bool = True,
        load_depth: bool = True,
        load_dit: bool = True,
        load_vocoder: bool = True,
        depth_device: str = "tt",
        dit_device: str = "tt",
        use_trace: bool = True,
        n_layers: Optional[int] = None,
        log: Callable[[str], None] = print,
    ):
        self.log, self.mesh, self.use_trace = log, mesh_device, use_trace
        self.snapshot = (
            Path(snapshot)
            if snapshot
            else W.resolve_snapshot()
            if hasattr(W, "resolve_snapshot")
            else resolve_snapshot()
        )
        self.cfg = Music3Config.from_snapshot(self.snapshot)
        self.tok = Music3Tokenizer(self.snapshot)
        self.max_seq_len = max_seq_len
        t0 = time.time()
        tables = W.load_lm_tables(self.snapshot)
        self.full_embed = tables["model.embed_tokens.weight"].to(torch.bfloat16)
        del tables
        self.model = self.gen = self.args = None
        if load_llm:
            self._build_llm(llm_dtype, n_layers)
        self.depth = None
        if load_depth:
            if depth_device == "tt":
                from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import TTDepthDecoder

                self.depth = TTDepthDecoder(
                    mesh_device,
                    W.load_depth_state(self.snapshot),
                    self.cfg.depth,
                    weights_dtype=depth_dtype,
                    full_embed=self.full_embed,
                    use_trace=use_trace,
                    log=log,
                )
            else:
                self.depth = CPUDepthDecoder(self.snapshot, self.cfg.depth, self.full_embed)
        self.depth_device = depth_device
        # condition encoder + vocoder (+ the scheduler loop) come from the torch reference; the DiT forward is swapped
        self.ref = Music3Reference(
            self.snapshot,
            dtype=torch.float32,
            load_llm=False,
            load_dit=(load_dit and dit_device == "cpu"),
            load_vocoder=load_vocoder,
            log=log,
        )
        self.vocoder_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
            os.environ.get("MUSIC3_VOCODER_DTYPE", "bf16")
        ]
        if load_vocoder and self.vocoder_dtype != torch.float32:
            self.ref.vocoder = self.ref.vocoder.to(self.vocoder_dtype)
        self.dit = None
        if load_dit and dit_device == "tt":
            from models.autoports.minimaxai_minimax_music3.tt.dit import TTDiT

            self.dit = TTDiT(
                mesh_device,
                W.load_dit_state(self.snapshot),
                self.cfg.dit,
                weights_dtype=dit_dtype,
                use_trace=use_trace,
                log=log,
            )
        elif load_dit:
            self.dit = self.ref.dit
        self.dit_device = dit_device
        self.impl = {
            "llm": f"ttnn/tt_transformers ({str(llm_dtype).split('.')[-1]})" if load_llm else "-",
            "depth": f"ttnn ({str(depth_dtype).split('.')[-1]})"
            if load_depth and depth_device == "tt"
            else ("torch-cpu" if load_depth else "-"),
            "dit": f"ttnn ({str(dit_dtype).split('.')[-1]})"
            if load_dit and dit_device == "tt"
            else ("torch-cpu" if load_dit else "-"),
            "cond_encoder": "torch-cpu",
            "vocoder": f"torch-cpu ({os.environ.get('MUSIC3_VOCODER_DTYPE', 'bf16')})" if load_vocoder else "-",
        }
        log(f"Music3Generator ready in {time.time() - t0:.1f}s: {self.impl}")

    # ------------------------------------------------------------------ LLM
    def _build_llm(self, dtype, n_layers):
        from models.autoports.minimaxai_minimax_music3.tt.backbone import Music3Backbone
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model_config import ModelArgs

        view = W.ensure_sliced_view(self.snapshot, log=self.log)
        os.environ["HF_MODEL"] = str(view)
        os.environ.setdefault("TT_CACHE_PATH", str(W.cache_root() / "tt_cache"))
        Path(os.environ["TT_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.args = ModelArgs(self.mesh, instruct=False, max_batch_size=2, max_seq_len=self.max_seq_len, cache_hf=False)
        if n_layers:
            self.args.n_layers = int(n_layers)
        # the two rows are prefilled one at a time (batched prefill would call prepare_inputs_prefill with batch_size=2)
        self.args.disable_batched_prefill = True
        self.log(
            f"ModelArgs: {self.args.model_name} device={self.args.device_name} vocab={self.args.vocab_size}/{self.args.padded_vocab_size} "
            f"layers={self.args.n_layers} max_seq_len={self.args.max_seq_len} prefill_chunk={getattr(self.args, 'max_prefill_chunk_size', '?')}"
        )
        assert self.args.vocab_size >= SLICED_VOCAB
        sd = self.args.load_state_dict()
        self.log(f"LLM state dict loaded ({len(sd)} tensors) in {time.time() - t0:.1f}s; moving to device ...")
        t1 = time.time()
        self.model = Music3Backbone(
            self.args, dtype, self.mesh, sd, self.args.weight_cache_path(dtype), full_embed=self.full_embed
        )
        del sd
        self.gen = Generator([self.model], [self.args], self.mesh)
        self.log(f"LLM on device in {time.time() - t1:.1f}s")

    def _prefill(self, text_ids: torch.Tensor):
        S = text_ids.shape[1]
        out = self.gen.prefill_forward_text(
            text_ids, page_table=None, kv_cache=None, prompt_lens=torch.tensor([S, S]), enable_trace=False
        )
        logits = out.reshape(2, -1)[:, :SLICED_VOCAB].float()
        hidden = torch.stack([self.model.read_prefill_hidden(u, S - 1) for u in range(2)])
        self.model.free_prefill_hidden()
        return logits, hidden

    def _decode(self, residual: torch.Tensor, pos: int):
        self.model.set_decode_residual(residual)
        out = self.gen.decode_forward(
            torch.zeros(2, 1, dtype=torch.int64),
            torch.tensor([pos, pos], dtype=torch.int64),
            page_table=None,
            kv_cache=None,
            enable_trace=self.use_trace,
            read_from_device=True,
        )
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.reshape(2, -1)[:, :SLICED_VOCAB].float()
        hidden = self.model.read_decode_hidden(2)
        return logits, hidden

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [8] -> [D] bf16: (E[c0+offset] + sum_i E_depth[c_i + (i-1)*1024]) * 8^-0.5 (float accumulation)."""
        e = self.full_embed[int(codes[0]) + AUDIO_CODE_OFFSET].float()
        table = (
            self.depth.audio_embeddings
            if hasattr(self.depth, "audio_embeddings")
            else self.depth.m.audio_embeddings.weight
        )
        idx = codes[1:].to(torch.int64) + torch.arange(NUM_CODEBOOKS - 1) * AUDIO_VOCAB_SIZE
        e = e + table[idx].float().sum(dim=0)
        return (e * NUM_CODEBOOKS**-0.5).to(torch.bfloat16)

    # ------------------------------------------------------------------ autoregressive stage
    @torch.inference_mode()
    def semantic_generation(
        self,
        text_ids: torch.Tensor,
        max_frames: int,
        generator,
        *,
        forced_codes: Optional[torch.Tensor] = None,
        record: bool = False,
        skip_depth: bool = False,
        stats: Optional[GenStats] = None,
        on_frame=None,
    ) -> dict:
        stats = stats or GenStats()
        max_frames = min(int(max_frames), MAX_AUDIO_FRAMES)
        S = text_ids.shape[1]
        stats.prompt_len = S
        t0 = time.time()
        logits, last_hidden = self._prefill(text_ids)
        stats.prefill_s = time.time() - t0
        pos = S
        frame_hiddens, codes_all, ended = [], [], False
        dumps: Dict[str, list] = {"hidden_all": [], "c0_logits": [], "depth_logits": [], "depth_hidden": []}
        n_steps = max_frames + 1 if forced_codes is None else forced_codes.shape[0]
        t_sem = time.time()
        for frame_index in range(n_steps):
            if record:
                dumps["hidden_all"].append(last_hidden.clone())
                dumps["c0_logits"].append(logits.clone())
            if forced_codes is None:
                idx = int(sample_top_k(guided_c0_logits_sliced(logits), generator))
                if idx == SEMANTIC_VOCAB_SIZE:  # <|audio_end|>
                    ended = True
                    break
                semantic_code, forced = idx, None
            else:
                row = forced_codes[frame_index]
                semantic_code, forced = int(row[0]), row[1:].tolist()
            if skip_depth:
                codes = torch.tensor([semantic_code, *forced], dtype=torch.int64)
                depth_hidden = None
            else:
                td = time.time()
                rec = {} if record else None
                codes, depth_hidden = self.depth.frame(
                    last_hidden, semantic_code, lambda g, i: int(sample_top_k(g, generator)), forced=forced, record=rec
                )
                stats.depth_s += time.time() - td
                if record:
                    dumps["depth_logits"].append(torch.stack(rec["depth_logits"]))
                    dumps["depth_hidden"].append(torch.cat(rec["depth_hidden"]))
            codes_all.append(codes)
            if frame_index > 0:
                if depth_hidden is not None:
                    frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden.reshape(1, -1)), dim=-1))
                stats.frames = frame_index
                if on_frame:
                    on_frame(frame_index - 1, codes)
                if forced_codes is None and frame_index >= max_frames:
                    break
            if forced_codes is not None and frame_index == n_steps - 1:
                break
            tl = time.time()
            logits, last_hidden = self._decode(self.embed_audio_frame(codes).unsqueeze(0).expand(2, -1), pos)
            stats.llm_s += time.time() - tl
            pos += 1
        stats.semantic_s = time.time() - t_sem
        stats.ended = ended
        res = {
            "codes_all": torch.stack(codes_all) if codes_all else torch.zeros(0, NUM_CODEBOOKS, dtype=torch.int64),
            "ended": ended,
            "prompt_len": S,
        }
        if frame_hiddens:
            res["frame_hiddens"] = torch.stack(frame_hiddens, dim=1)
        if record:
            res.update({k: torch.stack(v) for k, v in dumps.items() if v})
        return res

    # ------------------------------------------------------------------ flow matching + vocoder
    def dit_forward(self, latents, timestep, condition):
        if self.dit_device == "tt":
            return self.dit.forward(latents, timestep, condition)
        return self.dit(latents, timestep, condition)

    @torch.inference_mode()
    def denoise(self, frame_hiddens: torch.Tensor, generator, num_steps: int = DIT_NUM_STEPS, record_steps=()):
        return self.ref.denoise(
            frame_hiddens, generator, num_steps=num_steps, record_steps=record_steps, dit_forward=self.dit_forward
        )

    @torch.inference_mode()
    def decode(self, latent_chunks: List[torch.Tensor]) -> torch.Tensor:
        return self.ref.decode(latent_chunks)

    # ------------------------------------------------------------------ end to end
    @torch.inference_mode()
    def generate(
        self,
        caption: str,
        lyrics: str,
        *,
        audio_duration: float = 60.0,
        seed: int = 0,
        num_steps: int = DIT_NUM_STEPS,
        max_frames: Optional[int] = None,
        on_frame=None,
        stages=("semantic", "denoise", "decode"),
    ) -> dict:
        stats = GenStats()
        generator = torch.Generator("cpu").manual_seed(int(seed))
        text_ids = build_text_ids(self.tok, caption, lyrics)
        assert (
            text_ids.shape[1] + (max_frames or int(audio_duration * FRAME_RATE)) + 2 <= self.max_seq_len
        ), "prompt + frames exceed max_seq_len"
        mf = max_frames if max_frames is not None else min(int(audio_duration * FRAME_RATE), MAX_AUDIO_FRAMES)
        t0 = time.time()
        sem = self.semantic_generation(text_ids, mf, generator, stats=stats, on_frame=on_frame)
        out = {**sem, "text_ids": text_ids}
        if "denoise" in stages and "frame_hiddens" in sem:
            t1 = time.time()
            den = self.denoise(sem["frame_hiddens"], generator, num_steps=num_steps)
            stats.denoise_s = time.time() - t1
            stats.chunks = len(den["latent_chunks"])
            out.update(den)
            if "decode" in stages:
                t2 = time.time()
                out["audio"] = self.decode(den["latent_chunks"])
                stats.decode_s = time.time() - t2
                stats.audio_s = out["audio"].shape[-1] / self.cfg.vocoder.sampling_rate
                out["sample_rate"] = self.cfg.vocoder.sampling_rate
        stats.total_s = time.time() - t0
        out["stats"] = stats
        self.log(
            f"generate: {stats.frames} frames, {stats.audio_s:.1f}s audio in {stats.total_s:.1f}s (RTF {stats.rtf:.2f}; prefill {stats.prefill_s:.1f}s, "
            f"llm {stats.ms_per_frame_llm:.1f} ms/f, depth {stats.ms_per_frame_depth:.1f} ms/f, denoise {stats.denoise_s:.1f}s, vocoder {stats.decode_s:.1f}s)"
        )
        return out

    def release(self):
        for m in (self.depth, self.dit):
            if m is not None and hasattr(m, "release"):
                m.release()
