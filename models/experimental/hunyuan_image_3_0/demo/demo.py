# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 text-to-image on Tenstorrent.
#
#   prompt --[ optional AR recaption: text-sampling loop on the resident backbone +
#              LM head -> <recaption>...</recaption> cot_text ]
#          --HunyuanTokenizer (cot_text injected as the assistant turn)--> input_ids
#          --HunyuanTtWte--> text embeddings (cond / uncond rows for CFG)
#   noise latent --[ patch_embed -> RESIDENT bf8 sharded backbone -> final_layer ]
#                  x N scheduler steps with CFG (denoise_loop on the 2x2 sp0tp1 mesh)
#          --> denoised latent
#          --VAE decode (TTNN, on device, 2x2 H/W-spatial-parallel)--> RGB image
#
# Optional AR recaption (HY_RECAPTION=1): rewrites the prompt via the text-sampling loop
# before image gen. By default (HY_RECAPTION_INSTRUCT=1) recaption uses the
# HunyuanImage-3-Instruct checkpoint; set HY_RECAPTION_INSTRUCT=0 for base-checkpoint AR.
# When RECAPTION_LAYERS == NUM_LAYERS, one backbone upload is shared for AR + denoise.
#
# Denoise + VAE share one 2x2 mesh. When recaption runs on device and
# RECAPTION_LAYERS == NUM_LAYERS, the backbone is uploaded once and reused for AR + DiT.
#
# Run:
#   HY_STEPS=50 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"
#
# Host PyTorch DiT (debug TT DiT bugs; slow ~20-30 min for 32 layers):
#   HY_TORCH_BACKBONE=1 HY_STEPS=50 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"
#
# Optional recaption (off by default; uses Instruct weights like demo_i2i.py):
#   HY_RECAPTION=1 HY_BOT_TASK=think_recaption HY_STEPS=50 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"
#
# Trace / 2CQ ON (default): HY_TRACE=1 — denoise + recaption AR trace (VAE decode/encode off by default).
# Optional: HY_VAE_DECODE_TRACE=1  HY_COND_ENCODE_TRACE=1
# Trace / 2CQ OFF (eager 1CQ): HY_TRACE=0 HY_STEPS=50 ... demo.py "a photo of a cat"

import os, sys, json, time
from dataclasses import replace
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = str(Path(__file__).resolve().parents[4])  # tt-metal repo root (robust to checkout location)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights, ensure_instruct_weights, load_tensors

WEIGHTS = ensure_base_weights()
HUNYUAN = str(WEIGHTS)
if HUNYUAN not in sys.path:
    sys.path.insert(0, HUNYUAN)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
    prepare_gen_image_inputs,
    prepare_recaption_ar_bundle,
)
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    default_recaption_sampling_config,
    extract_recaption_written_prompt,
    extract_think_prose,
    is_meager_recaption_cot,
    prompt_fallback_recaption_cot,
    system_prompt_for_bot_task,
)
from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt
from models.experimental.hunyuan_image_3_0.tt.attention.mask import (
    build_attention_mask_tt,
    build_attention_mask_tt_sp_sharded,
)
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel, default_bf16_layers
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.trace_config import (
    open_pipeline_mesh,
    print_trace_policy,
    release_stage_resources,
)
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte, BackboneWteAdapter

# Prompt precedence: HY_PROMPT_FILE (read whole file) > argv[1] > HY_PROMPT > default.
# HY_PROMPT_FILE is handy for max-text-dimension tests (e.g. /tmp/long_prompt.txt).
_PROMPT_FILE = os.environ.get("HY_PROMPT_FILE")
if _PROMPT_FILE:
    PROMPT = Path(_PROMPT_FILE).read_text()
    print(f"[demo] prompt loaded from HY_PROMPT_FILE={_PROMPT_FILE} ({len(PROMPT)} chars)", flush=True)
else:
    PROMPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HY_PROMPT", "a photo of a cat, studio lighting")
STEPS = int(os.environ.get("HY_STEPS", "50"))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))
SEED = int(os.environ.get("HY_SEED", "0"))
# Image side in pixels; grid = IMAGE_SIZE / 16 (vae_downsample_factor), and the
# backbone seq_len = text + grid^2. Raise this to scale seq_len toward the config's
# max_position_embeddings (the run asserts the cap). Default 1024 -> 64x64 grid.
# When set, overrides the recaption-resolved size. The VAE is rebuilt for the grid.
IMAGE_SIZE = int(os.environ["HY_IMAGE_SIZE"]) if os.environ.get("HY_IMAGE_SIZE") else None
SCALING = 0.562679178327931
_DEMO_DIR = Path(__file__).resolve().parent
_PKG_DIR = _DEMO_DIR.parent
OUT_PNG = os.environ.get("HY_OUT", str(_PKG_DIR / "output.png"))
SAVE_LATENT = os.environ.get("HY_SAVE_LATENT")  # optional .pt path to dump denoised latent

# AR recaption (text-sampling loop) — off by default (upstream base bot_task=image).
RECAPTION = os.environ.get("HY_RECAPTION", "0") != "0"
BOT_TASK = os.environ.get("HY_BOT_TASK", "recaption")  # recaption | think | think_recaption
RECAPTION_LAYERS = int(os.environ.get("HY_RECAPTION_LAYERS", str(NUM_LAYERS)))
RECAPTION_KV = os.environ.get("HY_RECAPTION_KV", "1") != "0"
# 1 = on-device AR (default). 0 = host CUDA recaption (gold logits; needs NVIDIA GPU).
RECAPTION_ON_DEVICE = os.environ.get("HY_RECAPTION_DEVICE", "1") != "0"
# 1 = Instruct checkpoint + instruct template for AR (default; matches demo_i2i.py).
RECAPTION_INSTRUCT = os.environ.get("HY_RECAPTION_INSTRUCT", "1") != "0"
MAX_NEW_TOKENS = int(os.environ.get("HY_MAX_NEW_TOKENS", "512"))
# Sampling knobs — defaults match demo_i2i / generation_config.json (do_sample=True).
_SAMPLE_DEFAULTS = default_recaption_sampling_config()

# Debug: run denoise on host PyTorch reference instead of TT DiT.
TORCH_DENOISE = os.environ.get("HY_TORCH_BACKBONE", os.environ.get("HY_DIT_HOST", "0")) == "1"

_INDEX = WEIGHTS / "model.safetensors.index.json"
# Persist pre-tilized mesh weights across runs (see README § Weight cache).
if not os.environ.get("TT_DIT_CACHE_DIR"):
    os.environ["TT_DIT_CACHE_DIR"] = str(Path.home() / ".cache" / "tt-dit")
if not _INDEX.is_file():
    raise SystemExit(
        f"Base HunyuanImage-3 weights not found at {_INDEX}\n"
        f"Download with: hf download tencent/HunyuanImage-3.0\n"
        f"Or set HUNYUAN_MODEL_DIR to a checkpoint dir containing model.safetensors.index.json"
    )
_WMAP = json.load(open(_INDEX))["weight_map"]
_OPEN = {}

# --- lightweight per-stage timing -------------------------------------------
_TIMINGS = []


def _mark(name, since):
    """Record elapsed since `since`, print it, and return a fresh timestamp."""
    dt = time.time() - since
    _TIMINGS.append((name, dt))
    print(f"[timing] {name}: {dt:.1f}s", flush=True)
    return time.time()


def _print_timing_summary(total):
    print("\n==================== T2I TIMING SUMMARY ====================", flush=True)
    acc = 0.0
    for name, dt in _TIMINGS:
        acc += dt
        print(f"[timing] {name:34s} {dt:8.1f}s  ({100 * dt / total:5.1f}%)", flush=True)
    other = total - acc
    if abs(other) > 0.5:
        print(f"[timing] {'(unaccounted)':34s} {other:8.1f}s  ({100 * other / total:5.1f}%)", flush=True)
    print(f"[timing] {'TOTAL':34s} {total:8.1f}s", flush=True)
    print("============================================================", flush=True)


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(WEIGHTS / shard, framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg_from(model_dir: Path):
    c = json.load(open(model_dir / "config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        HEADS=c["num_attention_heads"],
        KV=c.get("num_key_value_heads", c["num_attention_heads"]),
        HD=c.get("attention_head_dim", c["hidden_size"] // c["num_attention_heads"]),
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
        MAX_SEQ=int(c["max_position_embeddings"]),
    )


def _cfg():
    return _cfg_from(WEIGHTS)


class _InstructWeightLoader:
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


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _parse_bf16_layers(num_layers: int) -> set[int]:
    if os.environ.get("HY_BF16_LAYERS"):
        return {int(s) for s in os.environ["HY_BF16_LAYERS"].split(",") if s.strip() != ""}
    return default_bf16_layers(num_layers)


def _build_backbone(
    mesh_device,
    ccl,
    c,
    *,
    num_layers,
    apply_final_norm,
    layer_loader,
    model_cache_name,
    embed_sd=None,
    norm_sd=None,
    sp_factor: int = 2,
    label: str = "resident",
):
    """Resident bf8 backbone on the 2x2 mesh (4-way expert shard, first/last 3 layers bf16).

    ``apply_final_norm=True`` + ``embed_sd``/``norm_sd`` builds the LM-backbone variant
    (device wte embed + ln_f) the AR recaption stage needs; the denoise path passes
    ``apply_final_norm=False`` and feeds pre-embedded hidden states instead.
    """
    bf16_layers = _parse_bf16_layers(num_layers)
    print(
        f"[demo] building {label} backbone ({num_layers} layers, sp={sp_factor}, " f"cache={model_cache_name}) ...",
        flush=True,
    )
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


def _base_layer_loader(i: int):
    return {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}


def _recaption_sampling_config() -> SamplingConfig:
    """Match demo_i2i: HF generation_config defaults with optional HY_* overrides."""
    cfg = replace(_SAMPLE_DEFAULTS, max_new_tokens=MAX_NEW_TOKENS)
    if os.environ.get("HY_DO_SAMPLE") is not None:
        cfg = replace(cfg, do_sample=os.environ.get("HY_DO_SAMPLE", "1") != "0")
    if os.environ.get("HY_TEMPERATURE"):
        cfg = replace(cfg, temperature=float(os.environ["HY_TEMPERATURE"]))
    if os.environ.get("HY_TOP_K"):
        cfg = replace(cfg, top_k=int(os.environ["HY_TOP_K"]))
    if os.environ.get("HY_TOP_P"):
        cfg = replace(cfg, top_p=float(os.environ["HY_TOP_P"]))
    if os.environ.get("HY_REP_PENALTY"):
        cfg = replace(cfg, repetition_penalty=float(os.environ["HY_REP_PENALTY"]))
    return cfg


def _use_tt_recaption() -> bool:
    if RECAPTION_ON_DEVICE:
        return True
    if not torch.cuda.is_available():
        print(
            "[demo] HY_RECAPTION_DEVICE=0 needs CUDA for host recaption; "
            "no NVIDIA GPU found — using Tenstorrent device instead",
            flush=True,
        )
        return True
    return False


def _run_recaption_host(prompt: str, *, image_size: str | int = 1024):
    """Gold recaption on CUDA via upstream ``generate`` (accurate AR logits)."""
    if not torch.cuda.is_available():
        raise RuntimeError("Host recaption requires CUDA (set HY_RECAPTION_DEVICE=1 for TT-only)")
    from transformers import AutoModelForCausalLM

    from models.experimental.hunyuan_image_3_0.ref.recaption import run_recaption

    sp_key, sp_sub = system_prompt_for_bot_task(BOT_TASK)
    recap_system = get_system_prompt(sp_key, sp_sub)
    print("[demo] host recaption on CUDA ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS),
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
        moe_drop_tokens=True,
    )
    model.load_tokenizer(str(WEIGHTS))
    result = run_recaption(
        model,
        prompt,
        bot_task=BOT_TASK,
        system_prompt=recap_system,
        image_size=image_size,
        seed=SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=1,
        sequence_template="pretrain",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[demo] host recaption done ({time.time() - t0:.0f}s)", flush=True)
    return result.cot_text[0], result.image_size


def _print_recaption_summary(tok, cot_text: str, *, user_prompt: str, image_size) -> None:
    """Print raw cot_text plus the extracted rewritten prose (what image gen actually uses)."""
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
    print(f"[demo] recaption raw cot_text:\n{cot_text}", flush=True)
    if think:
        print(f"[demo] recaption think:\n{think}", flush=True)
    if written:
        print(f"[demo] recaption written prompt:\n{written}", flush=True)
    else:
        print(
            "[demo] recaption written prompt: (none — only quad/structure tags, no prose)",
            flush=True,
        )
    print(f"[demo] original user prompt: {user_prompt!r}", flush=True)
    print(f"[demo] resolved image_size={image_size}", flush=True)
    if meager:
        print("[demo] warning: recaption has no usable rewritten prose for image gen", flush=True)


def _finalize_recaption_cot(tok, prompt: str, cot_text: str, image_size):
    """Retry host or fall back to the user prompt when device AR has no prose."""
    sp = tok.special
    recaption_open = tok.tokenizer.convert_ids_to_tokens(sp.recaption_token_id)
    recaption_close = tok.tokenizer.convert_ids_to_tokens(sp.end_recaption_token_id)
    if not is_meager_recaption_cot(
        cot_text,
        recaption_open=recaption_open,
        recaption_close=recaption_close,
    ):
        return cot_text, image_size
    print(
        "[demo] device recaption produced no rewritten prose " f"(got {cot_text!r}); recovering ...",
        flush=True,
    )
    if torch.cuda.is_available():
        return _run_recaption_host(prompt, image_size=image_size)
    cot_text = prompt_fallback_recaption_cot(tok, prompt)
    print(f"[demo] using prompt fallback cot_text: {cot_text!r}", flush=True)
    return cot_text, image_size


def _instruct_layer_loader(weights: _InstructWeightLoader, i: int):
    return {f"model.layers.{i}.{k}": v for k, v in weights.load_prefix(f"model.layers.{i}").items()}


def _run_recaption_instruct_on_mesh(
    mesh_device,
    ccl,
    c,
    weights: _InstructWeightLoader,
    tok,
    proc,
    wte,
    prompt: str,
    generator,
    *,
    backbone=None,
):
    """Instruct AR recaption on the shared pipeline mesh. Returns (cot_text, image_size, backbone)."""
    sp_key, sp_sub = system_prompt_for_bot_task(BOT_TASK)
    recap_system = get_system_prompt(sp_key, sp_sub)
    config = _recaption_sampling_config()
    print(
        f"[demo] instruct recaption (bot_task={BOT_TASK}) "
        f"layers={RECAPTION_LAYERS} max_new_tokens={config.max_new_tokens} kv_cache={RECAPTION_KV} "
        f"sample(do_sample={config.do_sample}, temp={config.temperature})",
        flush=True,
    )
    recap_bundle = prepare_recaption_ar_bundle(
        tok,
        prompt,
        proc,
        wte,
        bot_task=BOT_TASK,
        system_prompt=recap_system,
        sequence_template="instruct",
        generator=generator,
    )
    prefix_len = int(recap_bundle.input_ids.shape[1])
    print(f"[demo] instruct recaption prefix_len={prefix_len}", flush=True)

    def rep(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    recap_sp = 1 if RECAPTION_KV else 2
    if backbone is None:
        t0 = time.time()
        print(
            f"[demo] loading instruct recaption backbone ({RECAPTION_LAYERS} layers, sp={recap_sp}) ...",
            flush=True,
        )
        backbone = _build_backbone(
            mesh_device,
            ccl,
            c,
            num_layers=RECAPTION_LAYERS,
            apply_final_norm=True,
            layer_loader=lambda i: _instruct_layer_loader(weights, i),
            model_cache_name="hunyuan-image-3.0-instruct",
            embed_sd={"model.wte.weight": wte},
            norm_sd={"model.ln_f.weight": weights.load("model.ln_f.weight")},
            sp_factor=recap_sp,
            label="instruct recaption",
        )
        print(f"[demo] instruct recaption backbone ready ({time.time() - t0:.0f}s)", flush=True)
    else:
        print("[demo] reusing pre-built instruct backbone for recaption; loading LM head ...", flush=True)
    lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": weights.load("lm_head.weight")})
    recap_result = run_recaption_on_device(
        backbone,
        lm_head,
        mesh_device,
        recap_bundle,
        tok,
        BOT_TASK,
        proc,
        wte_weight=wte,
        image_size=1024,
        config=config,
        generator=generator,
        replicate_to_mesh=rep,
    )
    del lm_head
    cot_text, image_size = _finalize_recaption_cot(tok, prompt, recap_result.cot_text[0], recap_result.image_size)
    return cot_text, image_size, backbone


def _run_recaption(mesh_device, ccl, c, tok, proc, wte, prompt, generator, *, backbone=None):
    """AR recaption on the shared pipeline mesh. Returns (cot_text, image_size, backbone).

    When ``backbone`` is pre-built (shared with denoise), it is returned for reuse.
    The LM head is always freed before return.
    """
    sp_key, sp_sub = system_prompt_for_bot_task(BOT_TASK)
    recap_system = get_system_prompt(sp_key, sp_sub)
    recap_bundle = prepare_recaption_ar_bundle(
        tok,
        prompt,
        proc,
        wte,
        bot_task=BOT_TASK,
        system_prompt=recap_system,
        sequence_template="pretrain",
        generator=generator,
    )
    prefix_len = int(recap_bundle.input_ids.shape[1])
    config = _recaption_sampling_config()
    print(
        f"[demo] recaption stage (bot_task={BOT_TASK}, system={sp_key}/{sp_sub}) "
        f"prefix_len={prefix_len} layers={RECAPTION_LAYERS} max_new_tokens={config.max_new_tokens} "
        f"kv_cache={RECAPTION_KV} "
        f"sample(do_sample={config.do_sample}, temp={config.temperature}, "
        f"top_k={config.top_k}, top_p={config.top_p}, rep={config.repetition_penalty})",
        flush=True,
    )

    def rep(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    recap_sp = 1 if RECAPTION_KV else 2
    if backbone is None:
        t0 = time.time()
        print(
            f"[demo] loading recaption backbone ({RECAPTION_LAYERS} layers, sp={recap_sp}) ...",
            flush=True,
        )
        backbone = _build_backbone(
            mesh_device,
            ccl,
            c,
            num_layers=RECAPTION_LAYERS,
            apply_final_norm=True,
            layer_loader=_base_layer_loader,
            model_cache_name="hunyuan-image-3.0",
            embed_sd={"model.wte.weight": wte},
            norm_sd={"model.ln_f.weight": _load("model.ln_f.weight")},
            sp_factor=recap_sp,
            label="recaption",
        )
        print(f"[demo] recaption backbone ready ({time.time() - t0:.0f}s); loading LM head ...", flush=True)
    else:
        print("[demo] reusing pre-built backbone for recaption; loading LM head ...", flush=True)
    lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": _load("lm_head.weight")})
    recap_result = run_recaption_on_device(
        backbone,
        lm_head,
        mesh_device,
        recap_bundle,
        tok,
        BOT_TASK,
        proc,
        wte_weight=wte,
        image_size=1024,
        config=config,
        generator=generator,
        replicate_to_mesh=rep,
    )
    del lm_head
    cot_text, image_size = _finalize_recaption_cot(tok, prompt, recap_result.cot_text[0], recap_result.image_size)
    return cot_text, image_size, backbone


def main():
    t_start = time.time()
    t = t_start
    print(
        f"[demo] prompt={PROMPT!r}  steps={STEPS}  layers={NUM_LAYERS}  guidance={GUIDANCE}  "
        f"recaption={RECAPTION}  recaption_instruct={RECAPTION_INSTRUCT}  "
        f"bot_task={BOT_TASK}  torch_dit={TORCH_DENOISE}"
    )
    print_trace_policy(prefix="[demo]", denoise_steps=STEPS)
    c = _cfg()
    H = c["H"]
    down_sd, up_sd = _load_prefix("patch_embed"), _load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    tok = HunyuanTokenizer.from_pretrained()
    wte = _load("model.wte.weight").float()
    proc = HunyuanImage3ImageProcessor(json.load(open(WEIGHTS / "config.json")))
    generator = torch.Generator().manual_seed(SEED)
    t = _mark("1_setup_weights_tokenizer", t)

    cot_text, image_size = None, 1024
    use_tt_recaption_mesh = RECAPTION and _use_tt_recaption() and not TORCH_DENOISE
    can_share_backbone = use_tt_recaption_mesh and RECAPTION_LAYERS == NUM_LAYERS

    if RECAPTION and not _use_tt_recaption():
        cot_text, image_size = _run_recaption_host(PROMPT)
        _print_recaption_summary(tok, cot_text, user_prompt=PROMPT, image_size=image_size)
        t = _mark("2_recaption_ar", t)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("[demo] opening pipeline mesh (2x2) ...", flush=True)
    print("[demo] mesh policy: SINGLE open/close for all on-device stages", flush=True)
    mesh_device = open_pipeline_mesh(ttnn.MeshShape(2, 2), l1_small_size=32768)
    backbone = None
    latent = None
    instruct_weights = None
    recap_tok = tok
    recap_wte = wte
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

        # 0) optional on-device AR recaption on the shared mesh (single backbone when layers match).
        if use_tt_recaption_mesh:
            recap_sp = 1 if RECAPTION_KV else 2
            recap_proc = proc
            recap_wte = wte
            if RECAPTION_INSTRUCT:
                instruct_dir = ensure_instruct_weights()
                instruct_weights = _InstructWeightLoader(instruct_dir)
                c = _cfg_from(instruct_dir)
                H = c["H"]
                recap_tok = HunyuanTokenizer.from_model_dir(instruct_dir, sequence_template="instruct")
                recap_proc = HunyuanImage3ImageProcessor(json.load(open(instruct_dir / "config.json")))
                recap_wte = load_tensors(instruct_dir, ["model.wte.weight"])["model.wte.weight"]
                down_sd = instruct_weights.load_prefix("patch_embed")
                up_sd = instruct_weights.load_prefix("final_layer")
                LATENT, HID, HSZ = _pe_dims(down_sd)
                if can_share_backbone:
                    print(
                        "[demo] instruct recaption + denoise: one backbone upload "
                        f"({NUM_LAYERS} layers, sp={recap_sp})",
                        flush=True,
                    )
            if can_share_backbone:
                t0 = time.time()
                if RECAPTION_INSTRUCT:
                    backbone = _build_backbone(
                        mesh_device,
                        ccl,
                        c,
                        num_layers=NUM_LAYERS,
                        apply_final_norm=True,
                        layer_loader=lambda i: _instruct_layer_loader(instruct_weights, i),
                        model_cache_name="hunyuan-image-3.0-instruct",
                        embed_sd={"model.wte.weight": recap_wte},
                        norm_sd={"model.ln_f.weight": instruct_weights.load("model.ln_f.weight")},
                        sp_factor=recap_sp,
                        label="shared instruct",
                    )
                else:
                    print(
                        f"[demo] loading shared backbone for recaption + denoise "
                        f"({NUM_LAYERS} layers, sp={recap_sp}) ...",
                        flush=True,
                    )
                    backbone = _build_backbone(
                        mesh_device,
                        ccl,
                        c,
                        num_layers=NUM_LAYERS,
                        apply_final_norm=True,
                        layer_loader=_base_layer_loader,
                        model_cache_name="hunyuan-image-3.0",
                        embed_sd={"model.wte.weight": wte},
                        norm_sd={"model.ln_f.weight": _load("model.ln_f.weight")},
                        sp_factor=recap_sp,
                        label="shared",
                    )
                print(f"[demo] shared backbone ready ({time.time() - t0:.0f}s)", flush=True)
            if RECAPTION_INSTRUCT:
                cot_text, image_size, backbone = _run_recaption_instruct_on_mesh(
                    mesh_device,
                    ccl,
                    c,
                    instruct_weights,
                    recap_tok,
                    recap_proc,
                    recap_wte,
                    PROMPT,
                    generator,
                    backbone=backbone,
                )
            else:
                cot_text, image_size, backbone = _run_recaption(
                    mesh_device,
                    ccl,
                    c,
                    recap_tok,
                    recap_proc,
                    recap_wte,
                    PROMPT,
                    generator,
                    backbone=backbone,
                )
            _print_recaption_summary(recap_tok, cot_text, user_prompt=PROMPT, image_size=image_size)
            if can_share_backbone:
                backbone.apply_final_norm = False
                print("[demo] reusing shared backbone for denoise (no second weight upload)", flush=True)
            else:
                del backbone
                backbone = None
                release_stage_resources(mesh_device)
            t = _mark("2_recaption_ar", t)

        if IMAGE_SIZE is not None:  # explicit HY_IMAGE_SIZE overrides the resolved size
            image_size = IMAGE_SIZE
            print(f"[demo] HY_IMAGE_SIZE override -> image_size={image_size}")

        # 1) tokenize -> input_ids (cond row 0, uncond row 1) + contiguous image span.
        bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=image_size, cot_text=cot_text)
        ids = bundle.input_ids  # [2, S]
        S = bundle.seq_len
        span = bundle.rope_image_info[0][0][0]
        grid = bundle.rope_image_info[0][0][1]  # (64, 64)
        print(f"[demo] seq_len={S} image_span={span} grid={grid}  (max_position_embeddings={c['MAX_SEQ']})")
        assert (
            S <= c["MAX_SEQ"]
        ), f"seq_len {S} (image_size {image_size}) exceeds max_position_embeddings {c['MAX_SEQ']}"

        torch.manual_seed(SEED)
        init_latent = torch.randn(1, LATENT, grid[0], grid[1])

        if TORCH_DENOISE:
            from models.experimental.hunyuan_image_3_0.ref.host_denoise import HostDenoiseRunner, denoise_loop_host
            from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import bundle_to_denoise_cond

            class _WeightLoader:
                model_dir = WEIGHTS

                @staticmethod
                def load_prefix(prefix):
                    return _load_prefix(prefix)

            cond = bundle_to_denoise_cond(bundle, wte, proc, row=0)
            uncond = bundle_to_denoise_cond(bundle, wte, proc, row=1)
            img_slice = cond["gen_slice"]
            print(
                f"[demo] denoising {STEPS} steps on host PyTorch "
                f"(HY_TORCH_BACKBONE=1, CFG={GUIDANCE}, seq_len={S}) ...",
                flush=True,
            )
            t0 = time.time()
            runner = HostDenoiseRunner(
                _WeightLoader(),
                WEIGHTS,
                num_layers=NUM_LAYERS,
                down_sd=down_sd,
                up_sd=up_sd,
                model_cfg=c,
            )
            latent = denoise_loop_host(
                runner,
                init_latent=init_latent,
                cond=cond,
                uncond=uncond,
                img_slice=img_slice,
                steps=STEPS,
                guidance_scale=GUIDANCE,
            )
            print(f"[demo] host denoise done ({time.time() - t0:.0f}s), latent {tuple(latent.shape)}")
            t = _mark("5_denoise_loop", t)
        else:
            if backbone is not None and getattr(backbone, "embed_weight", None) is not None:
                print("[demo] reusing backbone embed_weight for denoise wte (no duplicate upload)", flush=True)
                wte_tt = BackboneWteAdapter(backbone)
            else:
                denoise_wte = recap_wte if instruct_weights is not None else wte
                wte_tt = HunyuanTtWte(
                    mesh_device,
                    denoise_wte,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            emb = wte_tt.embedding_torch(ids)  # [2, S, H]

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
                if instruct_weights is not None:
                    backbone = _build_backbone(
                        mesh_device,
                        ccl,
                        c,
                        num_layers=NUM_LAYERS,
                        apply_final_norm=False,
                        layer_loader=lambda i: _instruct_layer_loader(instruct_weights, i),
                        model_cache_name="hunyuan-image-3.0-instruct",
                        label="denoise",
                    )
                else:
                    backbone = _build_backbone(
                        mesh_device,
                        ccl,
                        c,
                        num_layers=NUM_LAYERS,
                        apply_final_norm=False,
                        layer_loader=_base_layer_loader,
                        model_cache_name="hunyuan-image-3.0",
                        label="denoise",
                    )
            load_prefix = instruct_weights.load_prefix if instruct_weights is not None else _load_prefix
            te1 = HunyuanTtTimestepEmbedder(
                mesh_device, H, {f"time_embed.{k}": v for k, v in load_prefix("time_embed").items()}, "time_embed"
            )
            te2 = HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"time_embed_2.{k}": v for k, v in load_prefix("time_embed_2").items()},
                "time_embed_2",
            )
            step = HunyuanTtDenoiseStep(
                mesh_device,
                patch_embed=patch_embed,
                backbone=backbone,
                final_layer=final_layer,
                img_slice=span,
                grid_hw=grid,
                seq_len=S,
            )

            if os.environ.get("HY_SHARDED_MASK", "1") == "1":
                mask_tt = build_attention_mask_tt_sp_sharded(
                    mesh_device,
                    S,
                    image_slices=[span],
                    bsz=1,
                    sp_factor=2,
                    dtype=ttnn.bfloat16,
                )
                print("[demo] attention mask: SP query-sharded upload path", flush=True)
            else:
                mask_tt = build_attention_mask_tt(mesh_device, S, image_slices=[span], bsz=1, dtype=ttnn.bfloat16)
            image_infos = [[(span, grid)]]

            def cond_dict(row):
                return dict(
                    text_pre=rep(emb[row : row + 1, : span.start, :]),
                    text_post=rep(emb[row : row + 1, span.stop :, :]),
                    image_infos=image_infos,
                    attention_mask=mask_tt,
                    batch=1,
                )

            cond, uncond = cond_dict(0), cond_dict(1)

            sched = HunyuanTtScheduler(mesh_device)
            sched.set_timesteps(STEPS)
            t = _mark("4_build_denoise_mesh_backbone", t)
            print(f"[demo] denoising {STEPS} steps (CFG={GUIDANCE}) on resident backbone ...")
            latent = denoise_loop(
                step,
                sched,
                init_latent,
                time_embed=te1,
                time_embed_2=te2,
                cond=cond,
                uncond=uncond,
                guidance_scale=GUIDANCE,
                mesh_device=mesh_device,
            )
            print(f"[demo] denoised latent {tuple(latent.shape)}  (finite={bool(torch.isfinite(latent).all())})")
            del (
                wte_tt,
                patch_embed,
                final_layer,
                backbone,
                te1,
                te2,
                step,
                sched,
                mask_tt,
                cond,
                uncond,
                emb,
            )
            release_stage_resources(mesh_device)
            t = _mark("5_denoise_loop", t)

        if SAVE_LATENT:
            payload = {
                "latent": latent.cpu().float(),
                "prompt": PROMPT,
                "seed": SEED,
                "steps": STEPS,
                "num_layers": NUM_LAYERS,
                "guidance": GUIDANCE,
                "image_size": image_size,
                "grid": grid,
                "scaling_factor": SCALING,
            }
            save_path = Path(SAVE_LATENT)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, save_path)
            print(f"[demo] saved latent -> {save_path.resolve()}", flush=True)

        print("[demo] VAE decode (TTNN, on device, 2x2 H/W-spatial-parallel) ...")
        img = decode_latent(
            mesh_device,
            latent,
            scaling_factor=SCALING,
            grid_hw=grid,
            ccl_manager=ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )
        t = _mark("6_vae_decode", t)
        img = img[0]  # [3, 1024, 1024]
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    from PIL import Image

    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    out_path = Path(OUT_PNG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)
    print(f"[demo] saved image -> {out_path}")
    t = _mark("7_save_png", t)
    _print_timing_summary(time.time() - t_start)


if __name__ == "__main__":
    main()
