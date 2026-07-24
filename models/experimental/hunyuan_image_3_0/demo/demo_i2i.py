# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 Instruct image-to-image on Tenstorrent.
#
# Full upstream flow (``bot_task=think_recaption``):
#   cond image(s) + prompt
#     → on-device VAE encode + SigLIP2 vision once (cached on host)
#     → on-device AR think/recaption (``run_recaption_on_device``)
#     → reuse same resident backbone for denoise (skip 2nd VAE/ViT + backbone reload)
#     → ``cot_text`` host-injected into I2I denoise bundle via cond cache
#     → denoise_loop + CFG → VAE decode → PNG
#
# Direct edit (``bot_task=image``) skips the AR recaption stage.
#
# Disable backbone keepalive / cond cache reuse: HY_KEEP_BACKBONE=0
#   (falls back to encode → free → AR → free → encode → rebuild denoise backbone).

# Run (Instruct-Distil, 8-step meanflow, no CFG):
#   HY_DISTIL=1 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo_i2i.py \
#     --distil --prompt "make the sky sunset orange" --cond /path/to/image.png
#
# Multi-image:
#   --cond img1.png img2.png
#
# Auto output resolution (matches cond aspect via ratio token):
#   --image-size auto --infer-align-image-size
#
# Host recaption fallback (requires NVIDIA CUDA): HY_RECAPTION_DEVICE=0
# On TT-only systems, host recaption auto-falls back to the on-device path.
#
# Device-logits AR sampling is on by default (same as demo.py):
#   HY_DEVICE_SAMPLING=1 (default) — D2H logits + host topk/shortlist
#   HY_TOPK=32 / HY_TOP_K=32 (or HY_TTNN_SAMPLING_OP=1) — pure ttnn.topk + ttnn.sampling
#   think→recaption stage-force stays on the device path (no host-sampler fallback)
#
# Trace / 2CQ ON (default): HY_TRACE=1 — recaption KV trace; denoise execute_trace when steps > 8.
# VAE decode + cond VAE/ViT encode trace default OFF: HY_VAE_DECODE_TRACE=1 HY_COND_ENCODE_TRACE=1
# Trace / 2CQ OFF: HY_TRACE=0 — eager 1CQ everywhere.
#
# Debug DiT vs VAE decoder: HY_DIT_HOST=1 runs denoise on host PyTorch, TT VAE decode unchanged.
#
# All on-device stages share one pipeline mesh (open once, release between stages, close at end).

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_vae_model_dir() -> None:
    """``MODEL_DIR`` is resolved at import time; set before hunyuan ref imports."""
    if os.environ.get("HUNYUAN_MODEL_DIR"):
        return
    for env_key in ("HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR", "HUNYUAN_INSTRUCT_MODEL_DIR"):
        if p := os.environ.get(env_key):
            os.environ["HUNYUAN_MODEL_DIR"] = p
            return
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for repo in (
        "tencent/HunyuanImage-3.0-Instruct-Distil",
        "tencent/HunyuanImage-3.0-Instruct",
        "tencent/HunyuanImage-3.0",
    ):
        snaps = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if snap.is_dir() and (snap / "model.safetensors.index.json").is_file():
                os.environ["HUNYUAN_MODEL_DIR"] = str(snap)
                return


_bootstrap_vae_model_dir()

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_timestep_embedder,
)
from dataclasses import replace

from models.experimental.hunyuan_image_3_0.ref.recaption import (
    default_recaption_sampling_config,
    extract_recaption_written_prompt,
    extract_think_prose,
    is_meager_recaption_cot,
    run_recaption,
    system_prompt_for_bot_task,
)
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    HunyuanTokenizer,
    build_i2i_cfg_conds,
    enrich_bundle_attention,
    prepare_i2i_inputs,
    prepare_recaption_inputs,
)
from models.experimental.hunyuan_image_3_0.ref.weights import (
    ensure_instruct_distil_weights,
    ensure_instruct_weights,
    load_tensors,
)
from models.experimental.hunyuan_image_3_0.ref.model_config import (
    IMAGE_BASE_SIZE,
    NUM_HIDDEN_LAYERS,
    VAE_SCALING_FACTOR,
    VIT_CONFIG,
    load_config,
    transformer_cfg,
)
from models.experimental.hunyuan_image_3_0.tt.attention.mask import (
    build_attention_mask_tt,
    build_attention_mask_tt_sp_sharded,
)
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel, default_bf16_layers
from models.experimental.hunyuan_image_3_0.tt.pipeline import (
    HunyuanTtDenoiseStep,
    decode_latent,
    denoise_loop,
    upload_denoise_cond_mesh,
)
from models.experimental.hunyuan_image_3_0.tt.vision.i2i_bundle import (
    apply_cond_encode_cache,
    build_cond_encode_cache_tt,
    load_tt_cond_patch_embed,
    load_tt_cond_timestep_embedders,
    load_tt_vision_stack,
    prepare_i2i_denoise_bundle_tt,
    prepare_recaption_ar_bundle_tt,
)
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.experimental.hunyuan_image_3_0.tt.trace_config import (
    invalidate_cond_encode_traces,
    open_pipeline_mesh,
    print_trace_policy,
    release_pipeline_traces,
    release_stage_resources,
)
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte

from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

STEPS_ENV = os.environ.get("HY_STEPS")
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", str(NUM_HIDDEN_LAYERS)))
RECAPTION_LAYERS = int(os.environ.get("HY_RECAPTION_LAYERS", str(NUM_LAYERS)))
RECAPTION_KV = os.environ.get("HY_RECAPTION_KV", "1") != "0"
# Keep resident backbone across AR → denoise by caching cond VAE/ViT tokens on host.
KEEP_BACKBONE = os.environ.get("HY_KEEP_BACKBONE", "1") != "0"
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "2.5"))
SEED = int(os.environ.get("HY_SEED", "42"))
VIT_LAYERS = int(os.environ.get("HY_VIT_LAYERS", str(VIT_CONFIG["num_hidden_layers"])))
MAX_NEW_TOKENS = int(os.environ.get("HY_MAX_NEW_TOKENS", "512"))
RECAPTION_ON_DEVICE = os.environ.get("HY_RECAPTION_DEVICE", "1") != "0"
DIT_ON_HOST = os.environ.get("HY_DIT_HOST", "0") == "1"
USE_DISTIL = os.environ.get("HY_DISTIL", "0") == "1"
SCALING = VAE_SCALING_FACTOR

# Persist pre-tilized mesh weights across runs (see README § Weight cache).
if not os.environ.get("TT_DIT_CACHE_DIR"):
    os.environ["TT_DIT_CACHE_DIR"] = str(Path.home() / ".cache" / "tt-dit")

# --- lightweight per-stage timing (mirrors demo.py) -------------------------
_TIMINGS = []
_AR_METRICS = []  # (label, ttft_seconds, tps, end_of_cot_seconds) per AR decode stage
_DENOISE_METRICS = []  # (label, ms_per_step, num_steps, total_seconds)
_VAE_METRICS = []  # (label, seconds)


def _record_ar_metrics(label, recap_result):
    if recap_result.ttft is not None:
        _AR_METRICS.append((label, recap_result.ttft, recap_result.tps, recap_result.total_seconds))


def _record_denoise_metrics(label, total_seconds, num_steps):
    ms_per_step = (total_seconds / num_steps * 1000) if num_steps else None
    _DENOISE_METRICS.append((label, ms_per_step, num_steps, total_seconds))


def _record_vae_metrics(label, total_seconds):
    _VAE_METRICS.append((label, total_seconds))


def _print_perf_summary(images: int, e2e_seconds: float):
    if _AR_METRICS:
        print("\n-------------------- AR text (think/recaption) --------------------", flush=True)
        for label, ttft, tps, cot_time in _AR_METRICS:
            tps_str = f"{tps:.2f} tok/s" if tps is not None else "n/a"
            cot_str = f"{cot_time:.2f}s" if cot_time is not None else "n/a"
            print(f"[perf] {label:20s} ttft={ttft:.2f}s  tps={tps_str}  end_of_cot={cot_str}", flush=True)

    if _DENOISE_METRICS:
        print("\n-------------------- Diffusion / flow denoise --------------------", flush=True)
        for label, ms_per_step, num_steps, total_seconds in _DENOISE_METRICS:
            ms_str = f"{ms_per_step:.1f} ms/step" if ms_per_step is not None else "n/a"
            print(
                f"[perf] {label:20s} {ms_str}  ({num_steps} steps)  total={total_seconds:.2f}s",
                flush=True,
            )

    if _VAE_METRICS:
        print("\n-------------------- VAE decode --------------------", flush=True)
        for label, total_seconds in _VAE_METRICS:
            print(f"[perf] {label:20s} decode_latency={total_seconds:.2f}s", flush=True)

    print("\n-------------------- End-to-end --------------------", flush=True)
    s_per_image = e2e_seconds / images if images else e2e_seconds
    images_per_hour = (images / e2e_seconds * 3600) if e2e_seconds > 0 else float("inf")
    print(f"[perf] prompt -> final image: {s_per_image:.2f} s/image  ({images_per_hour:.2f} images/hour)", flush=True)
    print("---------------------------------------------------------------------", flush=True)


def _mark(name, since):
    dt = time.time() - since
    _TIMINGS.append((name, dt))
    print(f"[timing] {name}: {dt:.1f}s", flush=True)
    return time.time()


def _print_timing_summary(total):
    print("\n==================== I2I TIMING SUMMARY ====================", flush=True)
    acc = 0.0
    for name, dt in _TIMINGS:
        acc += dt
        print(f"[timing] {name:34s} {dt:8.1f}s  ({100 * dt / total:5.1f}%)", flush=True)
    other = total - acc
    if abs(other) > 0.5:
        print(f"[timing] {'(unaccounted)':34s} {other:8.1f}s  ({100 * other / total:5.1f}%)", flush=True)
    print(f"[timing] {'TOTAL':34s} {total:8.1f}s", flush=True)
    print("============================================================", flush=True)


def _resolve_model_dir(distil: bool) -> Path:
    if distil:
        return ensure_instruct_distil_weights()
    return ensure_instruct_weights()


def _setup_upstream_path(model_dir: Path) -> None:
    upstream = os.environ.get("HUNYUAN_UPSTREAM", str(model_dir))
    if upstream not in sys.path:
        sys.path.insert(0, upstream)


def _default_steps(model_dir: Path, *, distil: bool) -> int:
    gen_cfg = model_dir / "generation_config.json"
    if gen_cfg.is_file():
        return int(json.load(open(gen_cfg)).get("diff_infer_steps", 8 if distil else 50))
    return 8 if distil else 50


def _model_flags(model_dir: Path) -> tuple[bool, bool]:
    cfg = load_config(model_dir)
    return bool(cfg.get("cfg_distilled", False)), bool(cfg.get("use_meanflow", False))


class _WeightLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._wmap = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
        self._open: dict = {}

    def load(self, key):
        shard = self._wmap[key]
        f = self._open.get(shard) or self._open.setdefault(shard, safe_open(self.model_dir / shard, framework="pt"))
        return f.get_tensor(key)

    def load_prefix(self, prefix):
        return {k[len(prefix) + 1 :]: self.load(k) for k in self._wmap if k.startswith(prefix + ".")}


def _use_tt_recaption() -> bool:
    """Run AR recaption on Tenstorrent unless host CUDA recaption was explicitly requested."""
    if RECAPTION_ON_DEVICE:
        return True
    if not torch.cuda.is_available():
        print(
            "[demo_i2i] HY_RECAPTION_DEVICE=0 needs CUDA for host recaption; "
            "no NVIDIA GPU found — using Tenstorrent device instead",
            flush=True,
        )
        return True
    return False


def _recaption_sampling_config() -> SamplingConfig:
    """Match demo.py: HF generation_config defaults with optional HY_* overrides."""
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import resolve_hy_top_k

    cfg = replace(default_recaption_sampling_config(), max_new_tokens=MAX_NEW_TOKENS)
    if os.environ.get("HY_DO_SAMPLE") is not None:
        cfg = replace(cfg, do_sample=os.environ.get("HY_DO_SAMPLE", "1") != "0")
    if os.environ.get("HY_TEMPERATURE"):
        cfg = replace(cfg, temperature=float(os.environ["HY_TEMPERATURE"]))
    top_k = resolve_hy_top_k()
    if top_k is not None:
        cfg = replace(cfg, top_k=top_k)
    if os.environ.get("HY_TOP_P"):
        cfg = replace(cfg, top_p=float(os.environ["HY_TOP_P"]))
    if os.environ.get("HY_REP_PENALTY"):
        cfg = replace(cfg, repetition_penalty=float(os.environ["HY_REP_PENALTY"]))
    return cfg


def _norm_prompt_words(text: str) -> list[str]:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).split()


def _print_recaption_summary(tok, cot_text: str, *, user_prompt: str, image_size) -> str:
    """Print raw cot_text plus the extracted rewritten prose. Returns written prompt."""
    sp = tok.special
    recaption_open = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    recaption_close = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    think_open = tok.tokenizer.convert_ids_to_tokens(sp.think_token_id)
    think_close = tok.tokenizer.convert_ids_to_tokens(sp.end_think_token_id)
    written = extract_recaption_written_prompt(cot_text, recaption_open=recaption_open, recaption_close=recaption_close)
    think = extract_think_prose(
        cot_text,
        think_open=think_open,
        think_close=think_close,
        recaption_open=recaption_open,
    )
    meager = is_meager_recaption_cot(cot_text, recaption_open=recaption_open, recaption_close=recaption_close)
    echo = _norm_prompt_words(written) == _norm_prompt_words(user_prompt)
    print(f"[demo_i2i] recaption raw cot_text:\n{cot_text}", flush=True)
    if think:
        print(f"[demo_i2i] recaption think:\n{think}", flush=True)
    if written:
        print(f"[demo_i2i] recaption written prompt:\n{written}", flush=True)
    else:
        print(
            "[demo_i2i] recaption written prompt: (none — only quad/structure tags, no prose)",
            flush=True,
        )
    print(f"[demo_i2i] original user prompt: {user_prompt!r}", flush=True)
    print(f"[demo_i2i] resolved image_size={image_size}", flush=True)
    if meager:
        print("[demo_i2i] warning: recaption has no usable rewritten prose for image gen", flush=True)
    elif echo:
        print("[demo_i2i] warning: recaption written prompt is an echo of the user prompt", flush=True)
    return written or ""


def _recaption_is_echo_or_meager(tok, cot_text: str, user_prompt: str) -> bool:
    sp = tok.special
    recaption_open = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    recaption_close = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    if is_meager_recaption_cot(cot_text, recaption_open=recaption_open, recaption_close=recaption_close):
        return True
    written = extract_recaption_written_prompt(cot_text, recaption_open=recaption_open, recaption_close=recaption_close)
    return _norm_prompt_words(written) == _norm_prompt_words(user_prompt)


def _run_recaption_on_device_maybe_retry_host_sample(
    *,
    backbone,
    lm_head,
    mesh_device,
    recap_bundle,
    tok,
    bot_task: str,
    proc,
    wte,
    prompt: str,
    config: SamplingConfig,
    generator,
    replicate_to_mesh,
    image_size,
):
    """Device AR recaption (same contract as demo.py).

    Device-logits sampling is on by default; ``HY_TOPK=32`` selects ttnn sampling.
    Empty/echo cot is left for ``_finalize_recaption_cot``. Set
    ``HY_RECAPTION_HOST_RETRY=1`` to allow a host-multinomial retry.
    """
    from models.experimental.hunyuan_image_3_0.tt.device_sampling import device_sampling_enabled

    recap_result = run_recaption_on_device(
        backbone,
        lm_head,
        mesh_device,
        recap_bundle,
        tok,
        bot_task,
        proc,
        wte_weight=wte,
        image_size=image_size,
        config=config,
        generator=generator,
        replicate_to_mesh=replicate_to_mesh,
    )
    cot = recap_result.cot_text[0]
    if not _recaption_is_echo_or_meager(tok, cot, prompt):
        return recap_result

    if device_sampling_enabled() and os.environ.get("HY_RECAPTION_HOST_RETRY", "0") != "1":
        print(
            "[demo_i2i] device-sampling recaption looks empty/echo "
            f"(got {cot!r}); keeping device result "
            "(set HY_RECAPTION_HOST_RETRY=1 to allow host multinomial retry)",
            flush=True,
        )
        return recap_result

    print(
        "[demo_i2i] device-sampling recaption looks empty/echo; "
        "retrying AR with host sampling on device logits (HY_DEVICE_SAMPLING=0) ...",
        flush=True,
    )
    prev = os.environ.get("HY_DEVICE_SAMPLING")
    os.environ["HY_DEVICE_SAMPLING"] = "0"
    try:
        recap_result = run_recaption_on_device(
            backbone,
            lm_head,
            mesh_device,
            recap_bundle,
            tok,
            bot_task,
            proc,
            wte_weight=wte,
            image_size=image_size,
            config=config,
            generator=generator,
            replicate_to_mesh=replicate_to_mesh,
        )
    finally:
        if prev is None:
            os.environ.pop("HY_DEVICE_SAMPLING", None)
        else:
            os.environ["HY_DEVICE_SAMPLING"] = prev

    cot2 = recap_result.cot_text[0]
    if _recaption_is_echo_or_meager(tok, cot2, prompt):
        print(f"[demo_i2i] host-sampling retry still empty/echo (got {cot2!r})", flush=True)
    else:
        print("[demo_i2i] host-sampling retry produced a rewritten recaption", flush=True)
    return recap_result


def _finalize_recaption_cot(tok, prompt: str, cot_text: str, image_size):
    """Choose cot for denoise injection.

    Empty/echo AR output must NOT be wrapped as ``<recaption>{full user prompt}`` —
    the user prompt is already in the chat template, so re-injecting it doubles
    seq_len (e.g. ~17k → ~34k) and exceeds ``max_position_embeddings``.
    """
    if not _recaption_is_echo_or_meager(tok, cot_text, prompt):
        return cot_text, image_size
    print(
        "[demo_i2i] device recaption empty/echo; skipping cot injection for denoise "
        "(user prompt already in the template — re-wrapping would double seq_len)",
        flush=True,
    )
    return None, image_size


def _fit_cot_for_max_seq(
    tok,
    prompt: str,
    cond_images,
    *,
    image_size,
    system_prompt: str | None,
    cot_text: str | None,
    max_seq: int,
) -> str | None:
    """Truncate or drop ``cot_text`` so ``prepare_i2i_inputs`` stays ≤ ``max_seq``."""
    if not cot_text:
        return None
    trial = prepare_i2i_inputs(
        tok,
        prompt,
        cond_images,
        image_size=image_size,
        sequence_template="instruct",
        system_prompt=system_prompt,
        cot_text=cot_text,
    )
    if trial.seq_len <= max_seq:
        return cot_text

    base = prepare_i2i_inputs(
        tok,
        prompt,
        cond_images,
        image_size=image_size,
        sequence_template="instruct",
        system_prompt=system_prompt,
        cot_text=None,
    )
    budget = int(max_seq) - int(base.seq_len)
    if budget <= 0:
        print(
            f"[demo_i2i] base seq_len={base.seq_len} already ≥ max_seq={max_seq}; " "dropping cot for denoise",
            flush=True,
        )
        return None

    cot_ids = tok.encode(cot_text)
    if len(cot_ids) <= budget:
        # Length came from template interaction; drop cot rather than guess.
        print(
            f"[demo_i2i] cot seq_len={trial.seq_len} > max_seq={max_seq}; dropping cot",
            flush=True,
        )
        return None

    clipped = tok.decode(cot_ids[:budget], skip_special_tokens=False)
    print(
        f"[demo_i2i] truncating cot {len(cot_ids)} → {budget} tokens "
        f"(trial seq_len={trial.seq_len} > max_seq={max_seq})",
        flush=True,
    )
    return clipped


def _cfg(model_dir: Path):
    return transformer_cfg(load_config(model_dir))


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _load_i2i_cond_stack(mesh_device, weights: _WeightLoader, wte, *, H: int, LATENT: int, HID: int, HSZ: int):
    """Upload I2I cond modules (wte, patch/time embed, ViT). VAE encoder is lazy per resolution."""
    print("[demo_i2i] loading on-device cond stack (wte, patch/time embed, ViT) ...", flush=True)
    wte_tt = HunyuanTtWte(
        mesh_device,
        wte,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cond_patch_embed = load_tt_cond_patch_embed(
        mesh_device,
        weights.load_prefix("patch_embed"),
        in_channels=LATENT,
        hidden_channels=HID,
        out_channels=HSZ,
    )
    cond_time_embed, cond_timestep_emb = load_tt_cond_timestep_embedders(
        mesh_device,
        hidden_size=H,
        time_embed_sd=weights.load_prefix("time_embed"),
        timestep_emb_sd=weights.load_prefix("timestep_emb"),
    )
    vision_tt, aligner_tt = load_tt_vision_stack(
        mesh_device,
        weights.load_prefix("vision_model"),
        weights.load_prefix("vision_aligner"),
        num_layers=VIT_LAYERS,
    )
    return wte_tt, cond_patch_embed, cond_time_embed, cond_timestep_emb, vision_tt, aligner_tt


def _parse_image_size(raw: str) -> str | int:
    if raw == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError:
        return raw


def _load_cond_images(proc, paths: list[str], *, infer_align: bool) -> tuple[list, list[Image.Image]]:
    random_crop = "resize" if infer_align else "center"
    cond_list = []
    pil_list = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        pil_list.append(img)
        cond, _ = proc.get_image_with_size(img, return_type="vae_vit", random_crop=random_crop)
        cond_list.append(cond)
    return cond_list, pil_list


def _build_backbone(
    mesh_device,
    ccl,
    c,
    weights: _WeightLoader,
    *,
    num_layers: int,
    apply_final_norm: bool,
    sp_factor: int = 2,
    model_cache_name: str = "hunyuan-image-3.0-instruct",
):
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in weights.load_prefix(f"model.layers.{i}").items()}
    bf16_layers = (
        {int(s) for s in os.environ["HY_BF16_LAYERS"].split(",") if s.strip() != ""}
        if os.environ.get("HY_BF16_LAYERS")
        else default_bf16_layers(num_layers)
    )
    # Denoise feeds pre-embedded tokens (base_embeds / text_pre); only AR
    # recaption needs on-device wte + ln_f. Skipping wte here frees ~1GB DRAM
    # for the [S,S] attention mask at I2I seq lengths.
    norm_sd = {"model.ln_f.weight": weights.load("model.ln_f.weight")} if apply_final_norm else None
    embed_sd = {"model.wte.weight": weights.load("model.wte.weight")} if apply_final_norm else None
    return HunyuanTtModel(
        mesh_device,
        num_layers=num_layers,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
        rms_norm_eps=c["EPS"],
        stream_experts=False,
        layer_loader=layer_loader,
        embed_state_dict=embed_sd,
        norm_state_dict=norm_sd,
        apply_final_norm=apply_final_norm,
        weight_dtype=ttnn.bfloat8_b,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=sp_factor,
        bf16_layers=bf16_layers,
        model_cache_name=model_cache_name,
    )


def _run_recaption_host(model_dir, weights, prompt, bot_task, system_prompt, pil_images, image_size):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Host recaption requires an NVIDIA GPU (CUDA). "
            "On Tenstorrent-only systems omit HY_RECAPTION_DEVICE=0 or set HY_RECAPTION_DEVICE=1."
        )
    from transformers import AutoModelForCausalLM

    kwargs = dict(
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
        moe_drop_tokens=True,
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model.load_tokenizer(str(model_dir))
    image_arg = pil_images[0] if len(pil_images) == 1 else pil_images
    result = run_recaption(
        model,
        prompt,
        bot_task=bot_task,
        system_prompt=system_prompt,
        image=image_arg,
        image_size=image_size,
        seed=SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=1,
    )
    return result.cot_text[0], result.image_size


def main():
    t_start = time.time()
    t = t_start
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 Instruct I2I on Tenstorrent")
    parser.add_argument("--prompt", default=os.environ.get("HY_PROMPT", "make the sky more dramatic"))
    parser.add_argument(
        "--cond",
        nargs="+",
        default=[p for p in os.environ.get("HY_COND", "").split(",") if p],
        help="One or more conditional RGB image paths",
    )
    parser.add_argument(
        "--bot-task",
        default=os.environ.get("HY_BOT_TASK", "think_recaption"),
        choices=("image", "recaption", "think_recaption"),
    )
    parser.add_argument("--image-size", default=os.environ.get("HY_IMAGE_SIZE", str(IMAGE_BASE_SIZE)))
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        default=os.environ.get("HY_INFER_ALIGN", "0") == "1",
    )
    parser.add_argument(
        "--distil",
        action="store_true",
        default=USE_DISTIL,
        help="Use HunyuanImage-3.0-Instruct-Distil (8-step meanflow, cfg_distilled)",
    )
    parser.add_argument("--out", default=os.environ.get("HY_OUT", str(ROOT / "hy_i2i.png")))
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.distil)
    _setup_upstream_path(model_dir)
    # TT VAE encoder/ref loaders read HUNYUAN_MODEL_DIR (base repo). Instruct checkpoints
    # ship the same vae.* weights; default to the resolved instruct dir when unset.
    if not os.environ.get("HUNYUAN_MODEL_DIR"):
        os.environ["HUNYUAN_MODEL_DIR"] = str(model_dir)
    if not (model_dir / "model.safetensors.index.json").is_file():
        label = "Instruct-Distil" if args.distil else "Instruct"
        repo = "tencent/HunyuanImage-3.0-Instruct-Distil" if args.distil else "tencent/HunyuanImage-3.0-Instruct"
        raise SystemExit(
            f"{label} weights not found under {model_dir}\n"
            f"Download with: hf download {repo}\n"
            f"Or set HUNYUAN_INSTRUCT_MODEL_DIR / HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR"
        )

    cfg_distilled, use_meanflow = _model_flags(model_dir)
    if args.distil and not cfg_distilled:
        print(f"[demo_i2i] warning: --distil but config.json has cfg_distilled=false ({model_dir})")

    steps = int(STEPS_ENV) if STEPS_ENV is not None else _default_steps(model_dir, distil=args.distil)
    weights = _WeightLoader(model_dir)

    if not args.cond:
        raise SystemExit("Provide --cond /path/to/image.png or set HY_COND")

    image_size = _parse_image_size(args.image_size)
    cot_text = None

    variant = "Instruct-Distil" if args.distil or cfg_distilled else "Instruct"
    model_cache_name = "hunyuan-image-3.0-instruct-distil" if args.distil else "hunyuan-image-3.0-instruct"
    print(
        f"[demo_i2i] model={model_dir}  variant={variant}  bot_task={args.bot_task}  "
        f"image_size={image_size}  steps={steps}  cond={args.cond!r}"
    )
    print_trace_policy(prefix="[demo_i2i]", denoise_steps=steps)

    c = _cfg(model_dir)
    H = c["H"]
    down_sd, up_sd = weights.load_prefix("patch_embed"), weights.load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    proc = HunyuanImage3ImageProcessor(json.load(open(model_dir / "config.json")))
    cond_list, pil_list = _load_cond_images(proc, args.cond, infer_align=args.infer_align_image_size)
    cond_for_bundle = cond_list if len(cond_list) > 1 else cond_list[0]

    tok = HunyuanTokenizer.from_model_dir(model_dir, sequence_template="instruct")
    wte = load_tensors(model_dir, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(SEED)

    timestep_emb_distil = load_timestep_embedder("timestep_emb", model_dir, hidden_size=H)
    t = _mark("1_setup_weights", t)

    recap_system = None
    if args.bot_task != "image":
        sp_key, sp_sub = system_prompt_for_bot_task(args.bot_task)
        recap_system = get_system_prompt(sp_key, sp_sub)
        if not _use_tt_recaption():
            print(f"[demo_i2i] recaption stage (bot_task={args.bot_task}) on host CUDA ...")
            cot_text, image_size = _run_recaption_host(
                model_dir, weights, args.prompt, args.bot_task, recap_system, pil_list, image_size
            )
            print(f"[demo_i2i] cot_text:\n{cot_text}\n[demo_i2i] resolved image_size={image_size}")
            t = _mark("2_recaption_ar", t)

    denoise_system = get_system_prompt("en_unified", "image")
    guidance_emb = load_timestep_embedder("guidance_emb", model_dir, hidden_size=H) if cfg_distilled else None
    timestep_r_emb = load_timestep_embedder("timestep_r_emb", model_dir, hidden_size=H) if use_meanflow else None

    use_tt_recaption_mesh = args.bot_task != "image" and _use_tt_recaption()
    # Keep one resident backbone when layer counts match (skip 2nd VAE/ViT + 140s reload).
    keep_backbone = KEEP_BACKBONE and (not use_tt_recaption_mesh or RECAPTION_LAYERS == NUM_LAYERS)
    if KEEP_BACKBONE and use_tt_recaption_mesh and RECAPTION_LAYERS != NUM_LAYERS:
        print(
            f"[demo_i2i] HY_KEEP_BACKBONE ignored: RECAPTION_LAYERS={RECAPTION_LAYERS} " f"!= NUM_LAYERS={NUM_LAYERS}",
            flush=True,
        )
    shared_sp = 1 if RECAPTION_KV else 2
    if keep_backbone:
        print(
            f"[demo_i2i] HY_KEEP_BACKBONE=1: cache cond embeds once, reuse backbone " f"(shared sp={shared_sp})",
            flush=True,
        )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[demo_i2i] opening pipeline mesh (2x2) ...", flush=True)
    print("[demo_i2i] mesh policy: SINGLE open/close for all on-device stages", flush=True)
    mesh_device = open_pipeline_mesh(ttnn.MeshShape(2, 2), l1_small_size=32768)
    invalidate_cond_encode_traces(mesh_device)
    backbone = None
    latent = None
    cond_cache = None
    try:
        mesh_device.enable_program_cache()
        ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

        def rep(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        # 0) optional on-device AR recaption.
        if use_tt_recaption_mesh:
            print(f"[demo_i2i] recaption stage (bot_task={args.bot_task}) ...", flush=True)
            (
                wte_tt,
                cond_patch_embed,
                cond_time_embed,
                cond_timestep_emb,
                vision_tt,
                aligner_tt,
            ) = _load_i2i_cond_stack(mesh_device, weights, wte, H=H, LATENT=LATENT, HID=HID, HSZ=HSZ)

            if keep_backbone:
                print(
                    "[demo_i2i] encoding cond VAE/ViT once → host cache "
                    "(backbone stays resident through denoise) ...",
                    flush=True,
                )
                cond_cache = build_cond_encode_cache_tt(
                    mesh_device,
                    cond_for_bundle,
                    model_dir=model_dir,
                    vision=vision_tt,
                    aligner=aligner_tt,
                    cond_patch_embed=cond_patch_embed,
                    cond_time_embed=cond_time_embed,
                    cond_timestep_emb=cond_timestep_emb,
                    seed=SEED,
                    cond_encode_trace=False,
                )
                del wte_tt, cond_patch_embed, cond_time_embed, cond_timestep_emb, vision_tt, aligner_tt
                release_stage_resources(mesh_device)
                invalidate_cond_encode_traces(mesh_device)

                print("[demo_i2i] building recaption AR bundle from cond cache (host) ...", flush=True)
                recap_bundle = prepare_recaption_inputs(
                    tok,
                    args.prompt,
                    cond_images=cond_for_bundle,
                    bot_task=args.bot_task.split("_")[0] if args.bot_task == "think_recaption" else args.bot_task,
                    system_prompt=recap_system,
                    sequence_template="instruct",
                )
                recap_bundle = apply_cond_encode_cache(recap_bundle, wte, cond_cache)
                recap_bundle.bot_task = args.bot_task
                recap_bundle = enrich_bundle_attention(recap_bundle, proc)
            else:
                print("[demo_i2i] building recaption AR bundle (wte → VAE encode → ViT inject) ...", flush=True)
                recap_bundle = prepare_recaption_ar_bundle_tt(
                    mesh_device,
                    tok,
                    args.prompt,
                    proc,
                    wte_tt,
                    cond_images=cond_for_bundle,
                    bot_task=args.bot_task,
                    system_prompt=recap_system,
                    sequence_template="instruct",
                    model_dir=model_dir,
                    vision=vision_tt,
                    aligner=aligner_tt,
                    cond_patch_embed=cond_patch_embed,
                    cond_time_embed=cond_time_embed,
                    cond_timestep_emb=cond_timestep_emb,
                    seed=SEED,
                )
                del wte_tt, cond_patch_embed, cond_time_embed, cond_timestep_emb, vision_tt, aligner_tt
                release_stage_resources(mesh_device)
                invalidate_cond_encode_traces(mesh_device)

            prefix_len = int(recap_bundle.input_ids.shape[1])
            recap_sp = shared_sp if keep_backbone else (1 if RECAPTION_KV else 2)
            print(
                f"[demo_i2i] recaption prefix_len={prefix_len} layers={RECAPTION_LAYERS} "
                f"max_new_tokens={MAX_NEW_TOKENS} kv_cache={RECAPTION_KV} sp={recap_sp} "
                f"keep_backbone={keep_backbone}",
                flush=True,
            )
            t0 = time.time()
            print(
                f"[demo_i2i] loading recaption backbone ({RECAPTION_LAYERS} layers, sp={recap_sp}) ...",
                flush=True,
            )
            backbone = _build_backbone(
                mesh_device,
                ccl,
                c,
                weights,
                num_layers=RECAPTION_LAYERS,
                apply_final_norm=True,
                sp_factor=recap_sp,
                model_cache_name=model_cache_name,
            )
            print(f"[demo_i2i] recaption backbone ready ({time.time() - t0:.0f}s)", flush=True)

            print("[demo_i2i] loading LM head ...", flush=True)
            lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": weights.load("lm_head.weight")})
            recap_config = _recaption_sampling_config()
            from models.experimental.hunyuan_image_3_0.tt.device_sampling import (
                device_sampling_enabled,
                ttnn_sampling_op_enabled,
            )

            print(
                f"[demo_i2i] recaption sampling: device_logits={device_sampling_enabled()} "
                f"ttnn_op={ttnn_sampling_op_enabled()} "
                f"temp={recap_config.temperature} top_k={recap_config.top_k} top_p={recap_config.top_p}",
                flush=True,
            )
            recap_result = _run_recaption_on_device_maybe_retry_host_sample(
                backbone=backbone,
                lm_head=lm_head,
                mesh_device=mesh_device,
                recap_bundle=recap_bundle,
                tok=tok,
                bot_task=args.bot_task,
                proc=proc,
                wte=wte,
                prompt=args.prompt,
                config=recap_config,
                generator=gen,
                replicate_to_mesh=rep,
                image_size=image_size,
            )
            del lm_head, recap_bundle
            _record_ar_metrics("recaption", recap_result)
            raw_cot = recap_result.cot_text[0]
            _print_recaption_summary(tok, raw_cot, user_prompt=args.prompt, image_size=recap_result.image_size)
            cot_text, image_size = _finalize_recaption_cot(tok, args.prompt, raw_cot, recap_result.image_size)

            if keep_backbone:
                # Denoise path does not apply ln_f; keep MoE stack resident.
                backbone.apply_final_norm = False
                # AR KV / traces must not linger — denoise needs DRAM for [S,S] masks.
                release_stage_resources(mesh_device)
                print(
                    "[demo_i2i] keeping resident backbone for denoise "
                    "(cond embeds reused from host cache; skip VAE/ViT reload) ...",
                    flush=True,
                )
            else:
                print(
                    "[demo_i2i] releasing recaption backbone before denoise cond encode "
                    "(VAE/ViT needs DRAM headroom on shared mesh) ...",
                    flush=True,
                )
                del backbone
                backbone = None
                release_stage_resources(mesh_device)
                invalidate_cond_encode_traces(mesh_device)
            t = _mark("2_recaption_ar", t)

        # Clamp / drop cot so denoise seq_len stays within max_position_embeddings.
        # Echo fallback used to re-wrap the full user prompt and double seq_len.
        if cot_text:
            cot_text = _fit_cot_for_max_seq(
                tok,
                args.prompt,
                cond_for_bundle,
                image_size=image_size,
                system_prompt=denoise_system,
                cot_text=cot_text if not _recaption_is_echo_or_meager(tok, cot_text, args.prompt) else None,
                max_seq=int(c["MAX_SEQ"]),
            )
            if cot_text is None and args.bot_task != "image":
                print(
                    "[demo_i2i] denoise will use user prompt only (no cot injection)",
                    flush=True,
                )

        # 1) I2I denoise bundle — from cond cache (keepalive) or second TT encode.
        if keep_backbone and cond_cache is not None:
            print("[demo_i2i] building I2I denoise bundle from cond cache (host) ...", flush=True)
            bundle = prepare_i2i_inputs(
                tok,
                args.prompt,
                cond_for_bundle,
                image_size=image_size,
                sequence_template="instruct",
                system_prompt=denoise_system,
                cot_text=cot_text,
            )
            bundle = apply_cond_encode_cache(bundle, wte, cond_cache)
            bundle = enrich_bundle_attention(bundle, proc)
        else:
            if backbone is not None:
                print(
                    "[demo_i2i] releasing backbone before denoise cond encode (VAE/ViT needs DRAM) ...",
                    flush=True,
                )
                del backbone
                backbone = None
                release_stage_resources(mesh_device)
            (
                wte_tt,
                cond_patch_embed,
                cond_time_embed,
                cond_timestep_emb,
                vision_tt,
                aligner_tt,
            ) = _load_i2i_cond_stack(mesh_device, weights, wte, H=H, LATENT=LATENT, HID=HID, HSZ=HSZ)
            print("[demo_i2i] building I2I denoise bundle (wte → VAE encode → ViT inject) ...", flush=True)
            bundle = prepare_i2i_denoise_bundle_tt(
                mesh_device,
                tok,
                args.prompt,
                cond_for_bundle,
                proc,
                wte_tt,
                model_dir=model_dir,
                vision=vision_tt,
                aligner=aligner_tt,
                cond_patch_embed=cond_patch_embed,
                cond_time_embed=cond_time_embed,
                cond_timestep_emb=cond_timestep_emb,
                image_size=image_size,
                sequence_template="instruct",
                system_prompt=denoise_system,
                cot_text=cot_text,
                seed=SEED,
            )
            del wte_tt, cond_patch_embed, cond_time_embed, cond_timestep_emb, vision_tt, aligner_tt
            release_stage_resources(mesh_device)

        cond_row, uncond_row = build_i2i_cfg_conds(bundle, wte, proc)
        img_slice = cond_row["gen_slice"]
        grid = cond_row["gen_hw"]
        seq_len = bundle.seq_len
        print(
            f"[demo_i2i] seq_len={seq_len} gen_span={img_slice} grid={grid}  "
            f"(max_position_embeddings={c['MAX_SEQ']})",
            flush=True,
        )
        if seq_len > c["MAX_SEQ"]:
            # Last resort: rebuild without cot (user prompt already carries the edit text).
            print(
                f"[demo_i2i] seq_len {seq_len} > max_seq {c['MAX_SEQ']}; " "rebuilding denoise bundle without cot ...",
                flush=True,
            )
            if keep_backbone and cond_cache is not None:
                bundle = prepare_i2i_inputs(
                    tok,
                    args.prompt,
                    cond_for_bundle,
                    image_size=image_size,
                    sequence_template="instruct",
                    system_prompt=denoise_system,
                    cot_text=None,
                )
                bundle = apply_cond_encode_cache(bundle, wte, cond_cache)
                bundle = enrich_bundle_attention(bundle, proc)
            else:
                raise RuntimeError(
                    f"seq_len {seq_len} exceeds max_position_embeddings {c['MAX_SEQ']} "
                    "(trim HY_PROMPT / HY_MAX_NEW_TOKENS)"
                )
            cond_row, uncond_row = build_i2i_cfg_conds(bundle, wte, proc)
            img_slice = cond_row["gen_slice"]
            grid = cond_row["gen_hw"]
            seq_len = bundle.seq_len
            print(
                f"[demo_i2i] rebuilt seq_len={seq_len} gen_span={img_slice} grid={grid}",
                flush=True,
            )
        assert (
            seq_len <= c["MAX_SEQ"]
        ), f"seq_len {seq_len} exceeds max_position_embeddings {c['MAX_SEQ']} (trim HY_PROMPT)"
        t = _mark("3_build_i2i_bundle", t)

        torch.manual_seed(SEED + 1)
        init_latent = torch.randn(1, LATENT, grid[0], grid[1])
        mode = "distil" if cfg_distilled else "CFG"

        if DIT_ON_HOST:
            from models.experimental.hunyuan_image_3_0.ref.host_denoise import HostDenoiseRunner, denoise_loop_host

            print(
                f"[demo_i2i] denoising {steps} steps on host PyTorch "
                f"(HY_DIT_HOST=1, {mode}, guidance={GUIDANCE}, seq_len={seq_len}) ...",
                flush=True,
            )
            t0 = time.time()
            runner = HostDenoiseRunner(
                weights,
                model_dir,
                num_layers=NUM_LAYERS,
                down_sd=down_sd,
                up_sd=up_sd,
            )
            latent = denoise_loop_host(
                runner,
                init_latent=init_latent,
                cond=cond_row,
                uncond=uncond_row,
                img_slice=img_slice,
                steps=steps,
                guidance_scale=GUIDANCE,
                timestep_emb=timestep_emb_distil,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                cfg_distilled=cfg_distilled,
                use_meanflow=use_meanflow,
            )
            print(f"[demo_i2i] host denoise done ({time.time() - t0:.0f}s), latent {tuple(latent.shape)}")
            _record_denoise_metrics("denoise(host)", time.time() - t0, steps)
            t = _mark("5_denoise_loop", t)
        else:
            # Attention masks are [B,1,S,S] bf16 (~2·S² bytes). At long I2I seq_len a
            # full replicated mask cannot sit next to a resident sp=1 backbone
            # (DRAM already ~full). Prefer SP query-sharded upload (same as demo.py)
            # and, when needed, drop keepalive to rebuild denoise with sp=2.
            def _span_key(spans):
                out = []
                for s in spans or []:
                    if isinstance(s, slice):
                        out.append((s.start, s.stop))
                    else:
                        out.append((int(s[0]), int(s[1])))
                return tuple(out)

            denoise_sp = int(getattr(backbone, "sp_factor", shared_sp if keep_backbone else 2))
            full_mask_bytes = int(seq_len) * int(seq_len) * 2  # bf16
            # ~400 MiB: beyond this, full [S,S] next to resident sp=1 MoE OOMs.
            _FULL_MASK_OOM_BYTES = 400 << 20
            if backbone is not None and denoise_sp <= 1 and full_mask_bytes > _FULL_MASK_OOM_BYTES:
                print(
                    f"[demo_i2i] seq_len={seq_len} full mask≈{full_mask_bytes / (1 << 30):.2f} GiB; "
                    "releasing sp=1 backbone to rebuild with sp=2 + SP-sharded mask ...",
                    flush=True,
                )
                del backbone
                backbone = None
                release_stage_resources(mesh_device)
                denoise_sp = 2

            use_sharded_mask = os.environ.get("HY_SHARDED_MASK", "1") == "1" and denoise_sp > 1
            print(
                f"[demo_i2i] building cond/uncond attention masks "
                f"({'SP-sharded' if use_sharded_mask else 'full'} bf16, sp={denoise_sp}) ...",
                flush=True,
            )
            cond_slices = bundle.full_attn_slices[0] if bundle.full_attn_slices else None
            bsz = int(cond_row.get("batch", 1))
            if use_sharded_mask:
                mask_tt = build_attention_mask_tt_sp_sharded(
                    mesh_device,
                    seq_len,
                    image_slices=cond_slices,
                    bsz=bsz,
                    sp_factor=denoise_sp,
                    dtype=ttnn.bfloat16,
                )
            else:
                mask_tt = build_attention_mask_tt(
                    mesh_device,
                    seq_len,
                    image_slices=cond_slices,
                    bsz=bsz,
                    dtype=ttnn.bfloat16,
                )
            cond_tt = upload_denoise_cond_mesh(
                cond_row,
                seq_len=seq_len,
                mesh_device=mesh_device,
                replicate_fn=rep,
                attention_mask=mask_tt,
            )
            uncond_tt = None
            if uncond_row is not None:
                uncond_slices = bundle.full_attn_slices[1] if len(bundle.full_attn_slices) > 1 else None
                if uncond_slices is not None and _span_key(uncond_slices) == _span_key(cond_slices):
                    uncond_tt = upload_denoise_cond_mesh(
                        uncond_row,
                        seq_len=seq_len,
                        mesh_device=mesh_device,
                        replicate_fn=rep,
                        attention_mask=cond_tt["attention_mask"],
                    )
                    print("[demo_i2i] CFG uncond shares cond attention_mask", flush=True)
                else:
                    if use_sharded_mask:
                        uncond_mask = build_attention_mask_tt_sp_sharded(
                            mesh_device,
                            seq_len,
                            image_slices=uncond_slices,
                            bsz=int(uncond_row.get("batch", 1)),
                            sp_factor=denoise_sp,
                            dtype=ttnn.bfloat16,
                        )
                    else:
                        uncond_mask = build_attention_mask_tt(
                            mesh_device,
                            seq_len,
                            image_slices=uncond_slices,
                            bsz=int(uncond_row.get("batch", 1)),
                            dtype=ttnn.bfloat16,
                        )
                    uncond_tt = upload_denoise_cond_mesh(
                        uncond_row,
                        seq_len=seq_len,
                        mesh_device=mesh_device,
                        replicate_fn=rep,
                        attention_mask=uncond_mask,
                    )

            print("[demo_i2i] building patch_embed + final_layer ...", flush=True)
            patch_embed = HunyuanTtUNetDown(
                mesh_device,
                {f"patch_embed.{k}": v for k, v in down_sd.items()},
                in_channels=LATENT,
                hidden_channels=HID,
                out_channels=HSZ,
            )
            final_layer = HunyuanTtUNetUp(
                mesh_device,
                {f"final_layer.{k}": v for k, v in up_sd.items()},
                in_channels=HSZ,
                hidden_channels=HID,
                out_channels=LATENT,
            )
            if backbone is None:
                print(
                    f"[demo_i2i] building denoise backbone ({NUM_LAYERS} layers, sp={denoise_sp}) ...",
                    flush=True,
                )
                t0 = time.time()
                backbone = _build_backbone(
                    mesh_device,
                    ccl,
                    c,
                    weights,
                    num_layers=NUM_LAYERS,
                    apply_final_norm=False,
                    sp_factor=denoise_sp,
                    model_cache_name=model_cache_name,
                )
                print(f"[demo_i2i] denoise backbone ready ({time.time() - t0:.0f}s)", flush=True)
            else:
                print(
                    f"[demo_i2i] reusing resident backbone for denoise "
                    f"(layers={NUM_LAYERS}, sp={getattr(backbone, 'sp_factor', '?')}, ln_f=off)",
                    flush=True,
                )
            print("[demo_i2i] building timestep embedders ...", flush=True)
            te1 = HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"time_embed.{k}": v for k, v in weights.load_prefix("time_embed").items()},
                "time_embed",
            )
            te2 = HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"time_embed_2.{k}": v for k, v in weights.load_prefix("time_embed_2").items()},
                "time_embed_2",
            )
            # Gen-image timestep scatter on device (no host TimestepEmbedder / embed H2D).
            timestep_emb_tt = HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"timestep_emb.{k}": v for k, v in weights.load_prefix("timestep_emb").items()},
                "timestep_emb",
            )
            guidance_emb_tt = None
            if cfg_distilled:
                guidance_emb_tt = HunyuanTtTimestepEmbedder(
                    mesh_device,
                    H,
                    {f"guidance_emb.{k}": v for k, v in weights.load_prefix("guidance_emb").items()},
                    "guidance_emb",
                )
            timestep_r_emb_tt = None
            if use_meanflow:
                timestep_r_emb_tt = HunyuanTtTimestepEmbedder(
                    mesh_device,
                    H,
                    {f"timestep_r_emb.{k}": v for k, v in weights.load_prefix("timestep_r_emb").items()},
                    "timestep_r_emb",
                )
            step = HunyuanTtDenoiseStep(
                mesh_device,
                patch_embed=patch_embed,
                backbone=backbone,
                final_layer=final_layer,
                img_slice=img_slice,
                grid_hw=grid,
                seq_len=seq_len,
            )

            sched = HunyuanTtScheduler(mesh_device)
            sched.set_timesteps(steps)
            t = _mark("4_build_denoise_mesh_backbone", t)
            print(
                f"[demo_i2i] denoising {steps} steps ({mode}, guidance={GUIDANCE}, seq_len={seq_len}) ...",
                flush=True,
            )
            latent = denoise_loop(
                step,
                sched,
                init_latent,
                time_embed=te1,
                time_embed_2=te2,
                cond=cond_tt,
                uncond=uncond_tt,
                guidance_scale=GUIDANCE,
                timestep_emb=timestep_emb_tt,
                guidance_emb=guidance_emb_tt,
                timestep_r_emb=timestep_r_emb_tt,
                cfg_distilled=cfg_distilled,
                use_meanflow=use_meanflow,
                mesh_device=mesh_device,
            )
            print(f"[demo_i2i] denoised latent {tuple(latent.shape)}")
            del (
                patch_embed,
                final_layer,
                backbone,
                te1,
                te2,
                timestep_emb_tt,
                guidance_emb_tt,
                timestep_r_emb_tt,
                step,
                sched,
                cond_tt,
                uncond_tt,
                bundle,
            )
            release_stage_resources(mesh_device)
            denoise_seconds = time.time() - t
            t = _mark("5_denoise_loop", t)
            _record_denoise_metrics("denoise", denoise_seconds, steps)

        print("[demo_i2i] VAE decode (TTNN spatial) ...")
        vae_t0 = time.time()
        img = decode_latent(
            mesh_device,
            latent,
            scaling_factor=SCALING,
            grid_hw=grid,
            ccl_manager=ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )
        _record_vae_metrics("vae_decode", time.time() - vae_t0)
        t = _mark("6_vae_decode", t)
    finally:
        release_pipeline_traces(mesh_device)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(args.out)
    print(f"[demo_i2i] saved -> {args.out}")
    t = _mark("7_save_png", t)
    e2e_seconds = time.time() - t_start
    _print_timing_summary(e2e_seconds)
    _print_perf_summary(images=1, e2e_seconds=e2e_seconds)


if __name__ == "__main__":
    main()
