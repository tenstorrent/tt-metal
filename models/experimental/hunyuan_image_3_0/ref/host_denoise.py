# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Streaming fp32 host PyTorch denoise (patch_embed -> MoE backbone -> final_layer)."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import scatter_distill_step_embeds
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    c = json.load(open(model_dir / "config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        HEADS=c["num_attention_heads"],
        KV=c.get("num_key_value_heads", c["num_attention_heads"]),
        HD=c.get("attention_head_dim", c["hidden_size"] // c["num_attention_heads"]),
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        INTER=first(c["moe_intermediate_size"]),
        SHARED=first(c["num_shared_expert"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _prepare_host_embeds(
    cond,
    t_scalar: float,
    hidden: int,
    timestep_emb=None,
    guidance_scalar: float | None = None,
    guidance_emb=None,
    timestep_r_emb=None,
    use_meanflow: bool = False,
    scheduler=None,
) -> torch.Tensor:
    """Assemble [B, S, H] hidden inputs for one denoise step."""
    del hidden
    if (host := cond.get("base_embeds_host")) is not None:
        t_r = None
        if use_meanflow and scheduler is not None:
            t_r = float(scheduler.get_timestep_r(t_scalar))
        return scatter_distill_step_embeds(
            host,
            t_scalar=float(t_scalar),
            gen_timestep_scatter_index=cond.get("gen_timestep_scatter_index"),
            timestep_emb=timestep_emb,
            guidance_scalar=guidance_scalar,
            guidance_scatter_index=cond.get("guidance_scatter_index"),
            guidance_emb=guidance_emb,
            t_r_scalar=t_r,
            gen_timestep_r_scatter_index=cond.get("gen_timestep_r_scatter_index"),
            timestep_r_emb=timestep_r_emb,
        )

    img_slice = cond["gen_slice"]
    pre = cond["text_pre"]
    post = cond.get("text_post")
    B, _, H = pre.shape
    post_len = post.shape[1] if post is not None else 0
    seq_len = img_slice.stop + post_len
    out = torch.zeros(B, seq_len, H, dtype=pre.dtype, device=pre.device)
    out[:, : img_slice.start, :] = pre
    if post is not None:
        out[:, img_slice.stop :, :] = post
    return out


class HostDenoiseRunner:
    """Streaming fp32 reference denoise stack (patch_embed + N layers + final_layer).

    Backbone layers are loaded one at a time per forward to avoid holding all N
    layers resident (~5 GB/layer fp32 -> OOM around layer 20 on typical hosts).
    """

    def __init__(
        self,
        weights,
        model_dir: Path,
        *,
        num_layers: int,
        down_sd,
        up_sd,
        model_cfg: dict | None = None,
    ):
        self.weights = weights
        self.model_dir = Path(model_dir)
        self.num_layers = num_layers
        self.down_sd = down_sd
        self.up_sd = up_sd
        self.c = _model_cfg(self.model_dir)
        if model_cfg is not None:
            self.c.update(model_cfg)
        self.latent_ch, self.hid, self.hsz = _pe_dims(down_sd)
        self.H = self.c["H"]
        self._ref_down = None
        self._ref_up = None
        self._te1 = None
        self._te2 = None
        self._te1_sd = weights.load_prefix("time_embed")
        self._te2_sd = weights.load_prefix("time_embed_2")
        self._stream_note_printed = False

    def _patch_modules(self):
        if self._ref_down is None:
            rd = RefDown(1, self.latent_ch, self.hsz, self.hid, self.hsz).eval()
            ru = RefUp(1, self.hsz, self.hsz, self.hid, self.latent_ch, out_norm=True).eval()
            rd.load_state_dict({k: v.float() for k, v in self.down_sd.items()}, strict=True)
            ru.load_state_dict({k: v.float() for k, v in self.up_sd.items()}, strict=True)
            self._ref_down, self._ref_up = rd, ru
        return self._ref_down, self._ref_up

    def _time_embeds(self, tvec):
        if self._te1 is None:
            self._te1 = RefTimeEmbed(self.H).eval()
            self._te1.load_state_dict({k: v.float() for k, v in self._te1_sd.items()}, strict=True)
            self._te2 = RefTimeEmbed(self.H).eval()
            self._te2.load_state_dict({k: v.float() for k, v in self._te2_sd.items()}, strict=True)
        with torch.no_grad():
            return self._te1(tvec), self._te2(tvec)

    def _run_backbone(self, h, mask_add, cos, sin):
        return self._run_backbone_multi([(h, mask_add, cos, sin)])[0]

    def _run_backbone_multi(self, streams):
        verbose = os.environ.get("HY_VERBOSE", "1") != "0"
        if verbose and not self._stream_note_printed:
            n = len(streams)
            note = f" x{n} streams (CFG)" if n > 1 else ""
            print(
                f"[host_denoise] streaming {self.num_layers} ref layers{note} "
                f"(one layer in RAM at a time; slow but avoids OOM) ...",
                flush=True,
            )
            self._stream_note_printed = True

        hs = [s[0] for s in streams]
        c = self.c
        for i in range(self.num_layers):
            sd = self.weights.load_prefix(f"model.layers.{i}")
            layer = RefLayer(
                hidden_size=c["H"],
                num_attention_heads=c["HEADS"],
                num_key_value_heads=c["KV"],
                attention_head_dim=c["HD"],
                num_experts=c["E"],
                moe_topk=c["K"],
                moe_intermediate_size=c["INTER"],
                num_shared_expert=c["SHARED"],
                use_mixed_mlp_moe=c["MIXED"],
                norm_topk_prob=c["NORM"],
                use_qk_norm=c["QKN"],
                rms_norm_eps=c["EPS"],
                layer_idx=i,
            )
            layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            layer.eval()
            for j, (_, mask_add, cos, sin) in enumerate(streams):
                hs[j] = layer(hs[j], attention_mask=mask_add, custom_pos_emb=(cos, sin))
            del layer, sd
            if (i + 1) % 8 == 0:
                gc.collect()
        return hs

    def predict_pair(
        self,
        latent,
        t_scalar: float,
        cond,
        uncond,
        img_slice,
        *,
        timestep_emb=None,
        guidance_scalar: float | None = None,
        guidance_emb=None,
        timestep_r_emb=None,
        use_meanflow: bool = False,
        scheduler=None,
    ):
        """Cond + uncond velocity preds sharing one streamed backbone pass (CFG path)."""
        rd, ru = self._patch_modules()
        B = latent.shape[0]
        tvec = torch.tensor([float(t_scalar)] * B)
        t_emb1, t_emb2 = self._time_embeds(tvec)
        with torch.no_grad():
            img_tokens, th, tw = rd(latent, t_emb1)
            h_c = _prepare_host_embeds(
                cond,
                t_scalar,
                self.H,
                timestep_emb=timestep_emb,
                guidance_scalar=guidance_scalar,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                use_meanflow=use_meanflow,
                scheduler=scheduler,
            )
            h_u = _prepare_host_embeds(
                uncond,
                t_scalar,
                self.H,
                timestep_emb=timestep_emb,
                guidance_scalar=guidance_scalar,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                use_meanflow=use_meanflow,
                scheduler=scheduler,
            )
            h_c[:, img_slice, :] = img_tokens
            h_u[:, img_slice, :] = img_tokens

            seq_c = cond["attention_mask"].shape[-1]
            seq_u = uncond["attention_mask"].shape[-1]
            cos_c, sin_c = build_batch_2d_rope(seq_c, self.c["HD"], image_infos=cond["image_infos"])
            cos_u, sin_u = build_batch_2d_rope(seq_u, self.c["HD"], image_infos=uncond["image_infos"])
            h_c, h_u = self._run_backbone_multi(
                [
                    (h_c, cond["attention_mask"], cos_c, sin_c),
                    (h_u, uncond["attention_mask"], cos_u, sin_u),
                ]
            )
            pred_c = ru(h_c[:, img_slice, :], t_emb2, th, tw)
            pred_u = ru(h_u[:, img_slice, :], t_emb2, th, tw)
        return pred_c, pred_u

    def predict(
        self,
        latent,
        t_scalar: float,
        cond,
        img_slice,
        image_infos,
        mask_add,
        *,
        timestep_emb=None,
        guidance_scalar: float | None = None,
        guidance_emb=None,
        timestep_r_emb=None,
        cfg_distilled: bool = False,
        use_meanflow: bool = False,
        scheduler=None,
    ):
        del cfg_distilled
        rd, ru = self._patch_modules()
        B = latent.shape[0]
        tvec = torch.tensor([float(t_scalar)] * B)
        t_emb1, t_emb2 = self._time_embeds(tvec)
        seq_len = mask_add.shape[-1]
        cos, sin = build_batch_2d_rope(seq_len, self.c["HD"], image_infos=image_infos)
        with torch.no_grad():
            img_tokens, th, tw = rd(latent, t_emb1)
            h = _prepare_host_embeds(
                cond,
                t_scalar,
                self.H,
                timestep_emb=timestep_emb,
                guidance_scalar=guidance_scalar,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                use_meanflow=use_meanflow,
                scheduler=scheduler,
            )
            h[:, img_slice, :] = img_tokens
            h = self._run_backbone(h, mask_add, cos, sin)
            return ru(h[:, img_slice, :], t_emb2, th, tw)


def denoise_loop_host(
    runner: HostDenoiseRunner,
    *,
    init_latent,
    cond,
    uncond=None,
    img_slice,
    steps: int,
    guidance_scale: float = 1.0,
    timestep_emb=None,
    guidance_emb=None,
    timestep_r_emb=None,
    cfg_distilled: bool = False,
    use_meanflow: bool = False,
):
    """Multi-step host denoise (I2I or T2I cond dicts). TT scheduler sigmas for parity."""
    sched = HunyuanTtScheduler(None)
    sched.set_timesteps(steps)
    sigmas = sched.sigmas
    timesteps = sched.timesteps
    image_infos = cond["image_infos"]
    do_cfg = uncond is not None and guidance_scale != 1.0 and not cfg_distilled
    distill_guidance = 1000.0 * guidance_scale if cfg_distilled else None
    verbose = os.environ.get("HY_VERBOSE", "1") != "0"

    lat = init_latent.clone()
    for step_i, t in enumerate(timesteps):
        if verbose:
            note = " +CFG" if do_cfg else ""
            print(f"[host_denoise] step {step_i + 1}/{len(timesteps)} t={float(t):.0f}{note} ...", flush=True)
        if do_cfg:
            pred, pred_u = runner.predict_pair(
                lat,
                float(t),
                cond,
                uncond,
                img_slice,
                timestep_emb=timestep_emb,
                guidance_scalar=distill_guidance,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                use_meanflow=use_meanflow,
                scheduler=sched,
            )
            pred = pred_u + guidance_scale * (pred - pred_u)
        else:
            pred = runner.predict(
                lat,
                float(t),
                cond,
                img_slice,
                image_infos,
                cond["attention_mask"],
                timestep_emb=timestep_emb,
                guidance_scalar=distill_guidance,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                cfg_distilled=cfg_distilled,
                use_meanflow=use_meanflow,
                scheduler=sched,
            )
        lat = lat + float(sigmas[step_i + 1] - sigmas[step_i]) * pred
    return lat
