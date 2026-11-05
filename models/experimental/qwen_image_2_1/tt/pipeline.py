# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Qwen-Image-2.1 text-to-image on one Blackhole p150.

prompt -> (host tokenizer) -> Qwen3-VL-8B text stack (device) -> DiT prefix KV (device, once per prompt)
       -> N Euler steps of the DiT (device; one metal trace per step shape) -> VAE decode (device) -> PIL.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import ttnn

from ..common import rope as rope_mod
from ..common import schedule
from ..common import text as text_mod
from ..common.config import DROP_IDX
from ..common.config import VAE as VAE_CFG
from ..common.config import snapshot_dir
from ..common.device import DEFAULT_TRACE_REGION
from ..common.weights import text_encoder_ckpt, transformer_ckpt, vae_ckpt
from .dit import DeviceCond, DiTPrecision, QwenImageDiT, RopeTables
from .text_encoder import Qwen3VLTextEncoder, TEPrecision

LATENT_C = 64


def to_pil(img: torch.Tensor):
    """[1, 4, H, W] float in [-1, 1] (RGBA) -> (PIL RGB composited over white, PIL RGBA); mirrors
    diffusers' VaeImageProcessor.postprocess (x / 2 + 0.5, clamp, uint8)."""
    from PIL import Image

    x = (img[0].float() / 2 + 0.5).clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    rgba = Image.fromarray(arr, mode="RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    return white, rgba


PROMPT_SLOTS = int(os.environ.get("QWEN_PROMPT_SLOTS", "4"))  # resident text-to-image prompts (K/V slots)
# Resident loop traces: a 40-step t2i trace takes ~235 MB of the trace region (editing traces more), so keep at most
# region / 256 MB of them and release the least recently used one before capturing another
MAX_TRACES = max(1, int(os.environ.get("QWEN_MAX_TRACES", str(DEFAULT_TRACE_REGION // (256 << 20)))))
SLOT_TOKENS = int(
    os.environ.get("QWEN_SLOT_TOKENS", "256")
)  # text tokens a slot holds; longer prompts use the legacy path


@dataclass
class PromptState:
    text_len: int  # prefix length P (text tokens + condition-image latent tokens)
    kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]]
    rt_img: RopeTables
    text_embeds: torch.Tensor  # [T, 4096] bf16 host (encoder output after dropping the system tokens)
    key_mask: Optional[ttnn.Tensor] = None  # legacy long-prefix step path (condition images, additive mask)
    target_hw: Tuple[int, int] = (64, 64)  # target latent grid (h, w)
    segments: Optional[list] = None
    long_prefix: bool = False  # mask-free long-prefix step path: kv hold round_up(P, 32) rows, P = text_len
    slot: int = -1  # >= 0: kv / rt_img live in a pre-allocated prompt slot (see QwenImage21Pipeline._slots)


@dataclass
class Timings:
    encode_s: float = 0.0
    prefix_s: float = 0.0
    denoise_s: float = 0.0
    decode_s: float = 0.0
    total_s: float = 0.0
    steps: int = 0
    traced: bool = False

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class QwenImage21Pipeline:
    def __init__(
        self,
        dev,
        *,
        dit_prec: Optional[DiTPrecision] = None,
        te_prec: Optional[TEPrecision] = None,
        load_text_encoder: bool = True,
        load_vae: bool = True,
        load_editing: bool = True,
        use_trace: bool = True,
        device_loop: bool = True,
        height: int = 1024,
        width: int = 1024,
        root: Optional[str] = None,
    ):
        self.dev = dev
        self.root = root or snapshot_dir()
        self.use_trace = use_trace
        self.device_loop = device_loop
        self.height, self.width = height, width
        self.height_default, self.width_default = height, width
        self.lat_h, self.lat_w = height // VAE_CFG.scale_factor_spatial, width // VAE_CFG.scale_factor_spatial
        self.lat_h_default, self.lat_w_default = self.lat_h, self.lat_w
        self.n_img = self.lat_h * self.lat_w
        t0 = time.time()
        self.tf_ckpt = transformer_ckpt(self.root)
        self.dit = QwenImageDiT(dev, self.tf_ckpt, dit_prec)
        self.load_dit_s = time.time() - t0
        # Prompt slots: persistent K/V and image-RoPE buffers for up to PROMPT_SLOTS text-to-image prompts, allocated
        # NOW so that no persistent buffer is ever allocated after a metal trace exists (buffers allocated later land
        # in a trace's transient range and get overwritten by its replay). A new prompt is copied INTO a slot with
        # ttnn.copy, so the slot's loop trace stays valid for repeated prompts and only a slot reassignment
        # (different prompt length) releases that slot's traces. Needs the plain (streaming) t2i attention, which
        # slices [image ; text] K/V to the logical length per step (the joint op would attend the slot's padding).
        self._slots: list = []
        if PROMPT_SLOTS > 0 and self.dit.prec.t2i_sdpa == "plain":
            zeros = lambda shape: ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            c = self.dit.cfg
            for _ in range(PROMPT_SLOTS):
                self._slots.append(
                    {
                        "kv": [
                            (zeros((1, c.heads, SLOT_TOKENS, c.head_dim)), zeros((1, c.heads, SLOT_TOKENS, c.head_dim)))
                            for _ in range(c.num_layers)
                        ],
                        "cos": zeros((1, 1, self.n_img, c.head_dim)),
                        "sin": zeros((1, 1, self.n_img, c.head_dim)),
                        "key": None,
                        "tick": 0,
                    }
                )
            self._tick = 0
        self.te = None
        if load_text_encoder:
            t0 = time.time()
            self.te = Qwen3VLTextEncoder(dev, text_encoder_ckpt(self.root), te_prec)
            self.load_te_s = time.time() - t0
        self.output_resolution = height
        self._proc = None
        self._tick = 0
        self._cond_table = {}
        self._vae_cfg_json = None
        self.te_vl = None
        self.vae_enc = None
        if load_text_encoder and load_editing:
            from .text_encoder_vl import Qwen3VLEncoderVL
            from .vae_encoder import QwenImageVAEEncoder

            t0 = time.time()
            self.te_vl = Qwen3VLEncoderVL(dev, text_encoder_ckpt(self.root), te=self.te)
            self.vae_enc = QwenImageVAEEncoder(dev, vae_ckpt(self.root))
            self.vae_enc.warmup(height, width)
            self.load_edit_s = time.time() - t0
        self.vae = None
        if load_vae:
            from .vae import QwenImageVAEDecoder

            t0 = time.time()
            self.vae = QwenImageVAEDecoder(dev, vae_ckpt(self.root))
            # Materialize every persistent VAE device buffer (per-resolution prepared conv weights) NOW, before
            # any metal trace is captured: buffers allocated after a capture can land inside that trace's
            # transient address range and get overwritten by the next replay (this produced all-white images).
            dummy = torch.zeros(1, LATENT_C, 1, self.lat_h, self.lat_w, dtype=torch.bfloat16)
            self.vae.decode(dummy)
            ttnn.synchronize_device(dev)
            self.load_vae_s = time.time() - t0
        self._trace: Dict[Tuple[int, int], dict] = {}
        self._cond_dev: Optional[DeviceCond] = None
        self._lat_dev: Optional[ttnn.Tensor] = None
        self._prompt_cache: Dict[str | tuple, PromptState] = {}

    # ------------------------------------------------------------------ text
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """prompt -> [T, 4096] bf16 encoder hidden states (system tokens dropped)."""
        ids = text_mod.tokenize_prompt(prompt, self.root)
        assert self.te is not None, "text encoder not loaded (pass text_embeds directly)"
        hidden = self.te.encode(ids)  # [L, 4096]
        return hidden[DROP_IDX:].contiguous()

    def prepare_prompt(self, text_embeds: torch.Tensor, cache_key=None) -> PromptState:
        """Text-only prefix (t2i). With prompt slots (and T <= SLOT_TOKENS) the prefix runs over SLOT_TOKENS rows
        (text positions 0.., real tokens first) and its K/V and the image RoPE tables are copied into a slot."""
        T = text_embeds.shape[0]
        segments = [("text", T), ("image", self.lat_h, self.lat_w)]
        cos, sin = rope_mod.dit_cos_sin(T, self.lat_h, self.lat_w)
        cond0 = DeviceCond.from_host(self.dev, schedule.StepConditioning.make(self.dit.time_cond, 0.0))
        if self._slots and T <= SLOT_TOKENS and (self.lat_h, self.lat_w) == (self.lat_h_default, self.lat_w_default):
            idx = self._acquire_slot(cache_key)
            slot = self._slots[idx]
            cos_s, sin_s = rope_mod.dit_cos_sin(SLOT_TOKENS, self.lat_h, self.lat_w)
            rt_text = self.dit.rope_tables(cos_s[:SLOT_TOKENS], sin_s[:SLOT_TOKENS])  # text positions 0..SLOT_TOKENS-1
            padded = torch.zeros(SLOT_TOKENS, text_embeds.shape[1], dtype=torch.bfloat16)
            padded[:T] = text_embeds.to(torch.bfloat16)
            kv, _ = self.dit.prefix_kv(padded, rt_text, cond0)
            for (k, v), (ks, vs) in zip(kv, slot["kv"]):
                ttnn.copy(k, ks)
                ttnn.copy(v, vs)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            rt_img = self.dit.rope_tables(cos[T:], sin[T:])
            ttnn.copy(rt_img.cos, slot["cos"])
            ttnn.copy(rt_img.sin, slot["sin"])
            ttnn.deallocate(rt_img.cos)
            ttnn.deallocate(rt_img.sin)
            ttnn.deallocate(rt_text.cos)
            ttnn.deallocate(rt_text.sin)
            ttnn.synchronize_device(self.dev)
            slot["key"] = cache_key
            return PromptState(
                T,
                slot["kv"],
                RopeTables(slot["cos"], slot["sin"], self.dit.trans_mat),
                text_embeds,
                key_mask=None,
                target_hw=(self.lat_h, self.lat_w),
                segments=segments,
                slot=idx,
            )
        rt_text = self.dit.rope_tables(cos[:T], sin[:T])
        rt_img = self.dit.rope_tables(cos[T:], sin[T:])
        kv, _ = self.dit.prefix_kv(text_embeds, rt_text, cond0)
        ttnn.synchronize_device(self.dev)
        return PromptState(
            T, kv, rt_img, text_embeds, key_mask=None, target_hw=(self.lat_h, self.lat_w), segments=segments
        )

    def _acquire_slot(self, cache_key) -> int:
        """Least recently used slot; evicts the prompt it held (its cache entry and the traces bound to it)."""
        free = [i for i, s in enumerate(self._slots) if s["key"] is None]
        idx = free[0] if free else min(range(len(self._slots)), key=lambda i: self._slots[i]["tick"])
        slot = self._slots[idx]
        if slot["key"] is not None:
            self._prompt_cache.pop(slot["key"], None)
            for key in [k for k in self._trace if len(k) > 1 and k[1] == idx]:
                ttnn.release_trace(self.dev, self._trace[key]["tid"])
                del self._trace[key]
        self._tick += 1
        slot["tick"] = self._tick
        slot["key"] = cache_key
        return idx

    def is_resident(self, cache_key) -> bool:
        """True when the prompt's state is cached (a request for it needs no prefix pass or trace capture)."""
        return cache_key in self._prompt_cache

    @staticmethod
    def cache_key_for(prompt: str, images=None):
        if not images:
            return prompt
        return (
            prompt,
            tuple(i.size for i in images),
            tuple(hashlib.sha1(i.tobytes()).hexdigest() for i in images),
        )

    # ------------------------------------------------------------------ condition images (editing)
    def _processor(self):
        if self._proc is None:
            from transformers import AutoProcessor

            self._proc = AutoProcessor.from_pretrained(os.path.join(self.root, "processor"))
        return self._proc

    def encode_prompt_with_images(self, prompt: str, images: list):
        """prompt + PIL condition images -> (prompt_embeds [L-14, 4096] bf16, slot mask [L-14] bool,
        cond latents packed [sum(h*w), 64] bf16 (normalized), cond latent shapes [(1,h,w)...], target (h, w))."""
        from ..common import images as im

        assert self.te_vl is not None, "vision-conditioned text encoder not loaded"
        assert self.vae_enc is not None, "VAE encoder not loaded"
        vision_pils, vae_tensors, sizes = [], [], []
        for img in images:
            white, vae_t, (w, h) = im.prepare_condition_image(img, self.output_resolution)
            vision_pils.append(white)
            vae_tensors.append(vae_t)
            sizes.append((w, h))
        text = im.edit_prompt_text(prompt, len(images))
        proc = self._processor()
        mi = proc(text=[text], images=vision_pils, padding=True, padding_side="left", return_tensors="pt")
        hidden = self.te_vl.encode(mi.input_ids, mi.pixel_values, mi.image_grid_thw)  # [L, 4096]
        embeds = hidden[DROP_IDX:].contiguous()
        slot_mask = im.image_pad_mask_from_ids(mi.input_ids, DROP_IDX)
        mean = torch.tensor(self._vae_mean(), dtype=torch.float32).view(1, LATENT_C, 1, 1)
        std = torch.tensor(self._vae_std(), dtype=torch.float32).view(1, LATENT_C, 1, 1)
        lat_list, shapes = [], []
        for vae_t, (w, h) in zip(vae_tensors, sizes):
            mode = self.vae_enc.encode(vae_t)  # [1, 64, h/16, w/16] float
            mode = mode.reshape(1, LATENT_C, mode.shape[-2], mode.shape[-1]).float()
            normalized = (mode - mean) / std
            lh, lw = normalized.shape[-2], normalized.shape[-1]
            lat_list.append(normalized.reshape(LATENT_C, lh * lw).t())  # packed [h*w, 64]
            shapes.append((1, lh, lw))
        cond_lat = torch.cat(lat_list, 0).to(torch.bfloat16)
        tw, th = im.calculate_dimensions(self.output_resolution * self.output_resolution, sizes[-1][0] / sizes[-1][1])
        return (
            embeds,
            slot_mask,
            cond_lat,
            shapes,
            (th // VAE_CFG.scale_factor_spatial, tw // VAE_CFG.scale_factor_spatial),
        )

    def prepare_prompt_with_images(
        self, embeds: torch.Tensor, slot_mask: torch.Tensor, cond_lat: torch.Tensor, cond_shapes: list, target_hw
    ) -> PromptState:
        th, tw = target_hw
        full_mask = torch.cat([slot_mask.bool(), torch.ones(th * tw // 4, dtype=torch.bool)])
        img_shapes = list(cond_shapes) + [(1, th, tw)]
        segments = rope_mod.segments_from_image_pad_mask(full_mask, img_shapes)
        cos, sin = rope_mod.joint_cos_sin(segments)
        n_target = th * tw
        P = cos.shape[0] - n_target
        rt_prefix = self.dit.rope_tables(cos[:P], sin[:P])
        rt_img = self.dit.rope_tables(cos[P:], sin[P:])
        text_rows = embeds[~slot_mask.bool()]
        cond0 = DeviceCond.from_host(self.dev, schedule.StepConditioning.make(self.dit.time_cond, 0.0))
        kv, P2 = self.dit.prefix_kv_segments(segments[:-1], text_rows, cond_lat, rt_prefix, cond0)
        assert P2 == P
        P_pad = kv[0][0].shape[-2]
        key_mask = None
        long_prefix = os.environ.get("QWEN_EDIT_MASKFREE", "1") == "1"
        if long_prefix:
            # trim the 256-padded prefix K/V to round_up(P, 32) rows once per prompt; the step concatenates them
            # with the image K/V at the logical length and runs SDPA without a mask
            P32 = (P + 31) // 32 * 32
            if P32 != P_pad:
                trimmed = []
                for k, v in kv:
                    kt = ttnn.slice(
                        k, [0, 0, 0, 0], [1, k.shape[1], P32, k.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    vt = ttnn.slice(
                        v, [0, 0, 0, 0], [1, v.shape[1], P32, v.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    trimmed.append((kt, vt))
                kv = trimmed
        else:
            key_mask = ttnn.from_torch(
                self.dit.step_key_mask(n_target, P, P_pad),
                dtype=ttnn.bfloat8_b if os.environ.get("QWEN_EDIT_MASK_DTYPE", "bfp8") == "bfp8" else ttnn.bfloat16,
                # 0 / -1e9 are exact in bfp8; halves the per-layer mask read
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.deallocate(rt_prefix.cos)
        ttnn.deallocate(rt_prefix.sin)
        ttnn.synchronize_device(self.dev)
        return PromptState(
            P, kv, rt_img, embeds, key_mask=key_mask, target_hw=(th, tw), segments=segments, long_prefix=long_prefix
        )

    # ------------------------------------------------------------------ latents
    def initial_latents(self, seed: int, hw=None) -> torch.Tensor:
        """Same noise as diffusers: torch.randn on a CPU generator in bf16, packed to [1, HW, C]."""
        h, w = hw or (self.lat_h, self.lat_w)
        gen = torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn((1, 1, LATENT_C, h, w), generator=gen, dtype=torch.bfloat16)
        return noise.view(1, LATENT_C, h * w).transpose(1, 2).contiguous()

    def _set_target(self, hw):
        """Switch the working latent grid (editing can produce non-square outputs)."""
        h, w = hw
        if (h, w) != (self.lat_h, self.lat_w):
            self.release_traces()
            self.lat_h, self.lat_w = h, w
            self.n_img = h * w
            self.height, self.width = h * VAE_CFG.scale_factor_spatial, w * VAE_CFG.scale_factor_spatial
            if self._lat_dev is not None:
                ttnn.deallocate(self._lat_dev)
                self._lat_dev = None
            if self.vae is not None:
                self.vae.decode(torch.zeros(1, LATENT_C, 1, h, w, dtype=torch.bfloat16))
                ttnn.synchronize_device(self.dev)

    def schedule(self, num_steps: int):
        sig = schedule.make_sigmas(num_steps, self.n_img)
        return sig, schedule.sigmas_to_timesteps(sig)

    # ------------------------------------------------------------------ denoise
    def _ensure_step_buffers(self):
        if self._lat_dev is None:
            self._lat_dev = ttnn.from_torch(
                torch.zeros(1, 1, self.n_img, LATENT_C, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._cond_dev is None:
            self._cond_dev = DeviceCond.from_host(self.dev, schedule.StepConditioning.make(self.dit.time_cond, 1.0))

    def _write_step_inputs(self, latents_bf16: torch.Tensor, sc: schedule.StepConditioning):
        host = ttnn.from_torch(
            latents_bf16.reshape(1, 1, self.n_img, LATENT_C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self._lat_dev)
        self._cond_dev.update(sc)

    def _get_trace(self, ps: PromptState):
        key = (ps.text_len, self.n_img)
        if key in self._trace:
            return self._trace[key]
        # warm-up (compiles programs) then capture
        out = self.dit.step(
            self._lat_dev,
            self._cond_dev,
            ps.kv,
            ps.rt_img,
            ps.text_len,
            key_mask=ps.key_mask,
            long_prefix=ps.long_prefix,
        )
        ttnn.deallocate(out)
        ttnn.synchronize_device(self.dev)
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        out = self.dit.step(
            self._lat_dev,
            self._cond_dev,
            ps.kv,
            ps.rt_img,
            ps.text_len,
            key_mask=ps.key_mask,
            long_prefix=ps.long_prefix,
        )
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        ttnn.synchronize_device(self.dev)
        self._trace[key] = {"tid": tid, "out": out, "kv": ps.kv}
        return self._trace[key]

    def release_traces(self):
        for tr in self._trace.values():
            ttnn.release_trace(self.dev, tr["tid"])
        self._trace.clear()

    def denoise(self, ps: PromptState, latents: torch.Tensor, num_steps: int, progress=None) -> torch.Tensor:
        """latents [1, HW, C] bf16 -> denoised latents (same shape, bf16). Euler steps on host in fp32."""
        sig, ts = self.schedule(num_steps)
        self._ensure_step_buffers()
        tr = self._get_trace(ps) if self.use_trace else None
        if tr is not None and tr["kv"] is not ps.kv:
            # a different prompt of the same length: the trace hard-codes the KV buffers, re-capture
            self.release_traces()
            tr = self._get_trace(ps)
        lat = latents.to(torch.bfloat16)
        for i in range(num_steps):
            sc = schedule.StepConditioning.make(self.dit.time_cond, float(ts[i]) / 1000.0)
            self._write_step_inputs(lat, sc)
            if tr is not None:
                ttnn.execute_trace(self.dev, tr["tid"], cq_id=0, blocking=False)
                v = ttnn.to_torch(tr["out"])
            else:
                out = self.dit.step(
                    self._lat_dev,
                    self._cond_dev,
                    ps.kv,
                    ps.rt_img,
                    ps.text_len,
                    key_mask=ps.key_mask,
                    long_prefix=ps.long_prefix,
                )
                v = ttnn.to_torch(out)
                ttnn.deallocate(out)
            v = v.reshape(1, self.n_img, LATENT_C).float()
            dt = float(sig[i + 1] - sig[i])
            lat = (lat.float() + dt * v).to(torch.bfloat16)
            if progress is not None:
                progress(i, num_steps)
        return lat

    # ------------------------------------------------------------------ whole loop on device
    def _step_conds_device(self, num_steps: int) -> List[DeviceCond]:
        """Per-step modulation rows resident on device (depend only on the schedule)."""
        key = ("conds", num_steps, self.n_img)
        cache = self._cond_table
        if key not in cache:
            sig, ts = self.schedule(num_steps)
            cache[key] = [
                DeviceCond.from_host(self.dev, schedule.StepConditioning.make(self.dit.time_cond, float(t) / 1000.0))
                for t in ts
            ]
        return cache[key]

    def _loop_body(
        self, ps: PromptState, lat_in: ttnn.Tensor, conds: List[DeviceCond], sig: np.ndarray, num_steps: int
    ) -> ttnn.Tensor:
        """num_steps Euler steps on device. lat_in [1,1,HW,C] bf16 -> new bf16 tensor. Latents are rounded
        to bf16 after every step like the reference scheduler (fp32 math, bf16 storage)."""
        lat = lat_in
        for i in range(num_steps):
            v = self.dit.step(
                lat, conds[i], ps.kv, ps.rt_img, ps.text_len, key_mask=ps.key_mask, long_prefix=ps.long_prefix
            )
            if len(v.shape) != 4:
                v = ttnn.reshape(v, [1, 1, self.n_img, LATENT_C])
            dt = float(sig[i + 1] - sig[i])
            v32 = ttnn.typecast(v, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v)
            l32 = ttnn.typecast(lat, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if lat is not lat_in:
                ttnn.deallocate(lat)
            upd = ttnn.multiply(v32, dt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v32)
            n32 = ttnn.add(l32, upd, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(l32)
            ttnn.deallocate(upd)
            lat = ttnn.typecast(n32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(n32)
        return lat

    def denoise_on_device(self, ps: PromptState, latents: torch.Tensor, num_steps: int) -> torch.Tensor:
        """All steps in ONE metal trace: a single execute_trace per image, one upload and one readback."""
        sig, _ = self.schedule(num_steps)
        if ("conds", num_steps, self.n_img) not in self._cond_table:
            self.release_traces()  # the new per-step modulation rows are persistent: never allocate them under a live trace
        conds = self._step_conds_device(num_steps)
        self._ensure_step_buffers()
        key = ("loop", ps.slot, ps.text_len, self.n_img, num_steps, ps.key_mask is not None, ps.long_prefix)
        tr = self._trace.get(key)
        if tr is not None and tr["kv"] is not ps.kv:
            ttnn.release_trace(self.dev, tr["tid"])
            tr = None
            del self._trace[key]
        host = ttnn.from_torch(
            latents.to(torch.bfloat16).reshape(1, 1, self.n_img, LATENT_C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self._lat_dev)
        if tr is None:
            while len(self._trace) >= MAX_TRACES:  # LRU eviction keeps the trace region within budget
                lru = min(self._trace, key=lambda k: self._trace[k].get("tick", 0))
                ttnn.release_trace(self.dev, self._trace[lru]["tid"])
                del self._trace[lru]
            # compile pass (eager, 1 step is enough to build every program) then capture the whole loop
            warm = self._loop_body(ps, self._lat_dev, conds, sig, 1)
            ttnn.deallocate(warm)
            ttnn.synchronize_device(self.dev)
            ttnn.copy_host_to_device_tensor(host, self._lat_dev)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = self._loop_body(ps, self._lat_dev, conds, sig, num_steps)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            ttnn.synchronize_device(self.dev)
            tr = {"tid": tid, "out": out, "kv": ps.kv}
            self._trace[key] = tr
        self._tick += 1
        tr["tick"] = self._tick
        ttnn.execute_trace(self.dev, tr["tid"], cq_id=0, blocking=False)
        lat = ttnn.to_torch(tr["out"]).reshape(1, self.n_img, LATENT_C)
        return lat

    # ------------------------------------------------------------------ decode
    def unpack_and_denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, HW, C] -> [1, C, 1, H, W] * std + mean (the VAE input)."""
        lat = latents.transpose(1, 2).reshape(1, LATENT_C, 1, self.lat_h, self.lat_w)
        mean = torch.tensor(self._vae_mean(), dtype=torch.float32).view(1, LATENT_C, 1, 1, 1)
        std = torch.tensor(self._vae_std(), dtype=torch.float32).view(1, LATENT_C, 1, 1, 1)
        return (lat.float() * std + mean).to(torch.bfloat16)

    def _vae_cfg(self):
        import json
        import os

        if self._vae_cfg_json is None:
            self._vae_cfg_json = json.load(open(os.path.join(self.root, "vae", "config.json")))
        return self._vae_cfg_json

    def _vae_mean(self):
        return self._vae_cfg()["latents_mean"]

    def _vae_std(self):
        return self._vae_cfg()["latents_std"]

    def decode(self, latents: torch.Tensor):
        """[1, HW, C] latents -> (PIL RGB over white, PIL RGBA).

        The VAE runs eagerly. Its prepared convolution buffers must exist before the denoising
        trace is captured, so shape changes warm the decoder before capturing a new trace."""
        assert self.vae is not None, "VAE not loaded"
        x = self.unpack_and_denormalize(latents)
        img = self.vae.decode(x)  # [1, 4, H, W] float in [-1, 1]
        return to_pil(img)

    # ------------------------------------------------------------------ all together
    def generate(
        self, prompt: str, *, seed: int = 42, num_steps: int = 40, progress=None, images: Optional[list] = None
    ):
        """images: optional list of PIL condition images (editing; up to the device memory: ~2.2 GB of prefix K/V per
        1024x1024 condition image). The output size follows the last condition image's aspect ratio."""
        tm = Timings(steps=num_steps, traced=self.use_trace)
        t_all = time.time()
        t0 = time.time()
        images = images or None
        cache_key = self.cache_key_for(prompt, images)  # content-hashed images: hits across server requests
        ps = self._prompt_cache.get(cache_key)
        if ps is not None and ps.slot >= 0:
            self._tick += 1
            self._slots[ps.slot]["tick"] = self._tick
        slot_eligible = (
            images is None
            and bool(self._slots)
            and text_mod.tokenize_prompt(prompt, self.root).shape[-1] - DROP_IDX <= SLOT_TOKENS
            if ps is None
            else False
        )
        if ps is None and not slot_eligible:
            # legacy single-entry path (editing prompts, prompts longer than a slot): release every trace and the
            # previous heavy prompt's buffers first, so the new K/V are allocated before the new capture (never
            # allocate persistent buffers after a trace that is still going to be replayed); slot-backed prompts
            # keep their K/V (pre-allocated) and re-capture their loop trace lazily
            self.release_traces()
            for key, old_ps in list(self._prompt_cache.items()):
                if old_ps.slot >= 0:
                    continue
                for k, v in old_ps.kv:
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                ttnn.deallocate(old_ps.rt_img.cos)
                ttnn.deallocate(old_ps.rt_img.sin)
                if old_ps.key_mask is not None:
                    ttnn.deallocate(old_ps.key_mask)
                del self._prompt_cache[key]
        if ps is None:
            if images is None:
                self._set_target(
                    (
                        self.height_default // VAE_CFG.scale_factor_spatial,
                        self.width_default // VAE_CFG.scale_factor_spatial,
                    )
                )
                emb = self.encode_prompt(prompt)
                tm.encode_s = time.time() - t0
                t0 = time.time()
                ps = self.prepare_prompt(emb, cache_key)
            else:
                emb, slot_mask, cond_lat, cond_shapes, target_hw = self.encode_prompt_with_images(prompt, images)
                tm.encode_s = time.time() - t0
                t0 = time.time()
                self._set_target(target_hw)
                ps = self.prepare_prompt_with_images(emb, slot_mask, cond_lat, cond_shapes, target_hw)
            tm.prefix_s = time.time() - t0
            self._prompt_cache[cache_key] = ps
        self._set_target(ps.target_hw)
        t0 = time.time()
        lat0 = self.initial_latents(seed, ps.target_hw)
        if self.use_trace and self.device_loop:
            lat = self.denoise_on_device(ps, lat0, num_steps)
        else:
            lat = self.denoise(ps, lat0, num_steps, progress=progress)
        tm.denoise_s = time.time() - t0
        t0 = time.time()
        rgb, rgba = self.decode(lat)
        tm.decode_s = time.time() - t0
        tm.total_s = time.time() - t_all
        return rgb, rgba, lat, tm
