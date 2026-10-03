"""S2Generator: text (+ reference audio codes) -> codebook frames -> waveform, on Tenstorrent.

Phase A: slow tower on TT via tt_transformers (prefill untraced, decode traced), frame embedding and
sampling on host, fast codebook decoder in torch (CPU), codec decoder on CPU. Later phases swap the fast
decoder and codec for TTNN implementations behind the same interfaces.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

import ttnn
from models.autoports.fishaudio_s2_pro.config import SAMPLES_PER_FRAME, S2Config
from models.autoports.fishaudio_s2_pro.tt import weights as W
from models.autoports.fishaudio_s2_pro.tt.fast_decoder_torch import TorchFastDecoder
from models.autoports.fishaudio_s2_pro.tt.prompt import S2Tokenizer, build_prompt
from models.autoports.fishaudio_s2_pro.tt.sampling import S2Sampler
from models.autoports.fishaudio_s2_pro.tt.slow_model import S2SlowTransformer
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import ModelArgs


@dataclass
class GenStats:
    prompt_len: int = 0
    frames: int = 0
    stopped_on_im_end: bool = False
    prefill_s: float = 0.0
    decode_s: float = 0.0
    slow_s: float = 0.0
    fast_s: float = 0.0
    tokens: List[int] = field(default_factory=list)

    @property
    def frames_per_s(self):
        return self.frames / self.decode_s if self.decode_s else 0.0

    @property
    def audio_s(self):
        return self.frames * SAMPLES_PER_FRAME / 44100.0

    @property
    def rtf_lm(self):
        return (self.prefill_s + self.decode_s) / self.audio_s if self.frames else float("inf")


class S2Generator:
    def __init__(
        self,
        mesh_device,
        snapshot: Optional[str] = None,
        *,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat8_b,
        fast_dtype=torch.bfloat16,
        cache_dir: Optional[str] = None,
        log: Callable[[str], None] = print,
    ):
        self.log = log
        snapshot = Path(snapshot) if snapshot else W.resolve_snapshot()
        self.snapshot = snapshot
        self.cfg = S2Config.from_snapshot(snapshot)
        self.tok = S2Tokenizer(snapshot)
        cache_dir = Path(cache_dir or os.environ.get("TT_DIT_CACHE_DIR") or Path.home() / ".cache" / "fish_s2_pro")
        cache_dir.mkdir(parents=True, exist_ok=True)
        view = W.ensure_hf_view(snapshot, cache_dir / "hf_view" / "s2-pro")
        os.environ["HF_MODEL"] = str(view)
        os.environ.setdefault("TT_CACHE_PATH", str(cache_dir / "tt_cache"))
        Path(os.environ["TT_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        log("Loading weights ...")
        sd = W.load_fish_state_dict(snapshot)
        slow_sd = W.slow_tower_state_dict(sd, self.cfg)
        fast_sd = W.fast_tower_state_dict(sd, self.cfg, split_qkv=False)
        cb_table = W.codebook_table(sd, self.cfg)
        log(f"weights loaded in {time.time() - t0:.1f}s ({len(sd)} tensors)")

        self.mesh_device = mesh_device
        self.args = ModelArgs(mesh_device, instruct=False, max_batch_size=1, max_seq_len=max_seq_len)
        self.args.n_layers = self.cfg.slow.n_layer
        log(
            f"ModelArgs: device_name={self.args.device_name} num_devices={self.args.num_devices} max_seq_len={self.args.max_seq_len} "
            f"max_prefill_chunk_size={getattr(self.args, 'max_prefill_chunk_size', '?')}"
        )
        t0 = time.time()
        self.model = S2SlowTransformer(
            self.args,
            dtype,
            mesh_device,
            slow_sd,
            self.args.weight_cache_path(dtype),
            codebook_table=cb_table,
            s2cfg=self.cfg,
        )
        log(f"slow tower on device in {time.time() - t0:.1f}s")
        self.gen = Generator([self.model], [self.args], mesh_device)
        fast_device = os.environ.get("FISH_S2_FAST_DEVICE", "cpu")
        if fast_device == "tt":
            from models.autoports.fishaudio_s2_pro.tt.fast_decoder import TTFastDecoder

            t0 = time.time()
            self.fast = TTFastDecoder(
                mesh_device,
                W.fast_tower_state_dict(sd, self.cfg, split_qkv=True),
                self.cfg,
                weights_dtype=dtype,
                log=log,
            )
            log(f"fast decoder on device (TP{mesh_device.get_num_devices()}) in {time.time() - t0:.1f}s")
        else:
            self.fast = TorchFastDecoder(fast_sd, self.cfg, dtype=fast_dtype)
        self.fast_device = fast_device
        del sd
        self.vocab = self.cfg.slow.vocab_size

    # ------------------------------------------------------------------ helpers
    def _slow_logits_prefill(self, tokens_row: torch.Tensor) -> torch.Tensor:
        T = tokens_row.shape[0]
        out = self.gen.prefill_forward_text(
            tokens_row.view(1, T), page_table=None, kv_cache=None, prompt_lens=torch.tensor([T]), enable_trace=False
        )
        return out.reshape(-1)[: self.vocab].float()

    def _hidden_row(self, row: int, mode: str) -> torch.Tensor:
        h = self.model.read_last_hidden(mode)
        return h[row % h.shape[0]]

    def _slow_logits_decode(self, tok: int, pos: int) -> torch.Tensor:
        out = self.gen.decode_forward(
            torch.tensor([[tok]], dtype=torch.int64),
            torch.tensor([pos], dtype=torch.int64),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        logits = out[0] if isinstance(out, tuple) else out
        return logits.reshape(-1)[: self.vocab].float()

    # ------------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        ref_codes: Optional[Sequence[torch.Tensor]] = None,
        ref_texts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.8,
        greedy: bool = False,
        seed: Optional[int] = None,
        on_frame: Optional[Callable[[int, List[int]], None]] = None,
    ):
        """Returns (codes (num_codebooks, T) int64, GenStats). on_frame(frame_index, frame) is called per frame."""
        cfg = self.cfg
        prompt = build_prompt(self.tok, text, ref_codes, ref_texts, cfg.num_codebooks)
        T = prompt.shape[1]
        max_new_tokens = min(max_new_tokens, self.args.max_seq_len - T - 1)
        assert max_new_tokens > 0, f"prompt of {T} tokens leaves no room under max_seq_len={self.args.max_seq_len}"
        sampler = S2Sampler(self.vocab, cfg.semantic_begin_id, cfg.semantic_end_id, cfg.im_end_id, seed=seed)
        stats = GenStats(prompt_len=T)

        def choose_fast(logits, _i):
            return sampler.sample_fast(logits, temperature, top_p, greedy)

        t0 = time.time()
        self.model.set_frame_codes(prompt[1:], offset=0)
        logits = self._slow_logits_prefill(prompt[0])
        hidden = self._hidden_row((T - 1) % 32, "prefill")
        stats.prefill_s = time.time() - t0

        frames: List[List[int]] = []
        pos = T
        t_dec = time.time()
        for step in range(max_new_tokens):
            tok = sampler.sample_slow(logits, temperature, top_p, greedy)
            stats.tokens.append(tok)
            if tok == cfg.im_end_id:
                stats.stopped_on_im_end = True
                break
            c0 = min(max(tok - cfg.semantic_begin_id, 0), cfg.codebook_size - 1)
            tf = time.time()
            rest = self.fast.frame(hidden, c0, choose_fast)
            stats.fast_s += time.time() - tf
            frame = [tok, c0, *rest]
            frames.append(frame)
            if on_frame:
                on_frame(len(frames) - 1, frame)
            ts = time.time()
            self.model.set_frame_codes(torch.tensor(frame[1:], dtype=torch.int64).view(-1, 1), offset=pos)
            logits = self._slow_logits_decode(tok, pos)
            hidden = self._hidden_row(0, "decode")
            stats.slow_s += time.time() - ts
            pos += 1
        stats.decode_s = time.time() - t_dec
        stats.frames = len(frames)
        codes = (
            torch.tensor(frames, dtype=torch.int64).T[1:]
            if frames
            else torch.zeros(cfg.num_codebooks, 0, dtype=torch.int64)
        )
        return codes, stats

    @torch.inference_mode()
    def teacher_forced(self, frames: torch.Tensor, prompt_len: int):
        """Feed golden frames (1+num_codebooks, prompt_len+G) and return per-step slow argmax + fast argmax.
        Mirrors the CPU baseline's teacher_forced() so agreement can be measured against the same goldens."""
        cfg = self.cfg
        prompt, gen = frames[:, :prompt_len], frames[:, prompt_len:]
        sampler = S2Sampler(self.vocab, cfg.semantic_begin_id, cfg.semantic_end_id, cfg.im_end_id)
        self.model.set_frame_codes(prompt[1:], offset=0)
        logits = self._slow_logits_prefill(prompt[0])
        hidden_tile = self.model.read_last_hidden("prefill").clone()
        hidden = hidden_tile[(prompt_len - 1) % hidden_tile.shape[0]]
        slow_argmax, slow_top5, fast_argmax, slow_logits_first, hidden_first = [], [], [], [], []
        pos = prompt_len
        for g in range(gen.shape[1]):
            biased = logits + sampler.bias
            slow_argmax.append(int(biased.argmax()))
            slow_top5.append(torch.topk(biased, 5).indices.tolist())
            if g < 8:
                slow_logits_first.append(logits.clone())
                hidden_first.append(hidden.clone())
            frame = gen[:, g].tolist()
            if frame[0] != cfg.im_end_id:
                rec: List[torch.Tensor] = []
                self.fast.frame(hidden, frame[1], lambda lg, i, f=frame: f[i + 1], record=rec)
                fast_argmax.append([int(r.argmax()) for r in rec])
            else:
                fast_argmax.append([0] * (cfg.num_codebooks - 1))
            self.model.set_frame_codes(torch.tensor(frame[1:], dtype=torch.int64).view(-1, 1), offset=pos)
            logits = self._slow_logits_decode(frame[0], pos)
            hidden = self._hidden_row(0, "decode")
            pos += 1
        return {
            "slow_argmax": torch.tensor(slow_argmax),
            "slow_top5": torch.tensor(slow_top5),
            "fast_argmax": torch.tensor(fast_argmax),
            "slow_logits_first": torch.stack(slow_logits_first),
            "hidden_first": torch.stack(hidden_first),
            "hidden_tile_first": hidden_tile,
        }
