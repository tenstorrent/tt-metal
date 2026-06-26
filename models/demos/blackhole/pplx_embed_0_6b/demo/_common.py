# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the pplx-embed-v1-0.6B perf entry points.

pplx-embed-v1-0.6B is a bidirectional Qwen3-0.6B derivative from Perplexity AI
with identical core dimensions (1024 hidden, 28 layers, 16/8 heads) but two
key differences from Qwen3-Embedding-0.6B:

  1. **Bidirectional attention** (is_causal=False) — handled by
     PplxBidirectionalAttention in tt/attention.py.
  2. **Mean-token pooling** — instead of last-token extraction, all non-padding
     token hidden states are averaged.

The same Qwen3-style precision knobs apply (identical architecture).
"""

import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.pplx_embed_0_6b.tt.attention import PplxBidirectionalAttention
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, copy_host_to_device, get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
    determine_device_name,
)

try:
    from tracy import signpost as _tracy_signpost
except ModuleNotFoundError:

    def _tracy_signpost(*_args, **_kwargs):
        pass


MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"
BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# ModelArgs subclass for pplx-embed (trust_remote_code + AutoModel)
# ---------------------------------------------------------------------------


class PplxModelArgs(ModelArgs):
    """ModelArgs variant that enables ``trust_remote_code`` for the custom
    ``bidirectional_pplx_qwen3`` HuggingFace model type and loads weights
    directly from safetensors to avoid the custom modeling.py requiring
    a newer transformers version.
    """

    def _set_hf_params(self, checkpoint_dir):
        self.trust_remote_code_hf = True
        return super()._set_hf_params(checkpoint_dir)

    def get_max_prefill_chunk_size(self):
        """Same as Qwen3-Embedding-0.6B (identical architecture)."""
        chunk_sizes = {
            "N150": 4,
            "N300": 64,
            "T3K": 128,
            "TG": 128,
            "P150": 128,
            "P300": 128,
            "P150x4": 128,
            "P150x8": 128,
        }
        div1024 = chunk_sizes.get(self.device_name, 4)
        return div1024 * 1024

    def get_trace_prefill_supported_seq_lens(self):
        """ISL sweep range capped at ``max_seq_len`` to avoid warmup assertion failures."""
        all_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
        return [s for s in all_lens if s <= self.max_seq_len]

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        """Cap warmup sequence lengths at max_seq_len."""
        return [s for s in to_warmup_seq_lens if s <= self.max_seq_len]

    def load_state_dict(self):
        """Load weights from safetensors, bypassing AutoModel/AutoModelForCausalLM.

        pplx-embed's custom HF modeling.py imports ``TransformersKwargs``
        which may not exist in the installed transformers.  Since the model
        weights are standard Qwen3 format, we load them directly from the
        safetensors file and add the ``model.`` prefix that the downstream
        ``standardize_hf_keys`` / ``convert_hf_to_meta`` pipeline expects.
        """
        if self.dummy_weights:
            return super().load_state_dict()

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        logger.info(f"Loading pplx-embed weights from {self.CKPT_DIR} via safetensors...")
        safetensor_path = hf_hub_download(
            self.CKPT_DIR,
            "model.safetensors",
            local_files_only=os.getenv("CI") == "true",
        )
        raw_sd = load_file(safetensor_path)

        # Add ``model.`` prefix so the state dict matches what
        # ``AutoModelForCausalLM.from_pretrained().state_dict()`` would
        # produce. Also create ``lm_head.weight`` from tied embeddings.
        state_dict = {f"model.{k}": v for k, v in raw_sd.items()}
        if "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        from models.tt_transformers.tt.load_checkpoints import (
            convert_hf_to_meta,
            convert_hf_to_meta_no_qkv_permute,
            standardize_hf_keys,
        )

        self.fuse_qkv = any("qkv" in k for k in state_dict)
        self.fuse_mlp = any("gate_up" in k for k in state_dict)
        state_dict = standardize_hf_keys(state_dict)
        if self.use_hf_rope:
            state_dict = convert_hf_to_meta_no_qkv_permute(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)
        else:
            state_dict = convert_hf_to_meta(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in range(self.n_layers, self.full_model_n_layers)]
        for k in keys_dict:
            if any(r in k for r in remv):
                state_dict.pop(k)

        return state_dict


# ---------------------------------------------------------------------------
# Environment / optimizations
# ---------------------------------------------------------------------------


def apply_recommended_env(batched_l1: bool) -> None:
    """Set the recommended optimization env vars (same Qwen3 architecture).

    Aggressive precision: for embedding workloads accuracy is measured by
    cosine similarity — even BFP4 + LOFI everywhere maintains >0.99 cos-sim
    on Qwen3-0.6B derivatives, so we push every knob.
    """
    os.environ.setdefault("HF_MODEL", MODEL_NAME)
    # Weight precision — all BFP4
    os.environ.setdefault("QWEN_QKV_BFP4", "1")
    os.environ.setdefault("QWEN_WO_BFP4", "1")
    os.environ.setdefault("QWEN_FF2_BFP4", "1")
    # Activation precision
    os.environ.setdefault("QWEN_FF13_OUT_BFP8", "1")
    os.environ.setdefault("QWEN_FFNORM_IN_BFP8", "1")
    os.environ.setdefault("QWEN_RESIDUAL_BFP8", "1")
    # Architecture-level TM optimizations
    os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
    os.environ.setdefault("QWEN_ROPE_PREFILL_L1", "1")
    # RoPE is a cos/sin rotation (operands in [-1,1]); the rotary_embedding_llama
    # op defaults to HiFi4, but LoFi is accuracy-neutral (STS-B 0.8487 vs 0.8481)
    # and saves ~0.4ms on bs1/ISL512. See tt/attention.py:_mllama_rope_prefill.
    os.environ.setdefault("QWEN_ROPE_FIDELITY", "lofi")
    os.environ.setdefault("QWEN_LN_BLOCK_SHARDED", "1")
    # Embedding-specific: skip KV cache fill (no decode)
    os.environ.setdefault("TT_SKIP_KV_CACHE_FILL", "1")
    # SDPA bigger chunks for bs=1 — more work per SDPA launch
    os.environ.setdefault("QWEN_SDPA_BIG_CHUNK_BS1", "1")
    if batched_l1:
        os.environ.setdefault("TT_BATCHED_L1_PREFILL", "1")


# ---------------------------------------------------------------------------
# Per-workload optimization configs
# ---------------------------------------------------------------------------

# Maps (batch_size, seq_len) to optimized settings.  The dp32 multiprocess
# script and individual demo files both look up from here so every
# combination is tuned in exactly one place.
#
# "batched_l1"  – whether the total activation (bs * seq * hidden * 2 B)
#                 fits in P150 L1 (~8 MiB).  When True we set
#                 TT_BATCHED_L1_PREFILL=1.
# "dram_grid"   – when activations spill to DRAM, the per-core L1 budget
#                 is freed and we can widen the MinimalMatmul grid from
#                 (8,8)=64 to (8,10)=80 cores via QWEN_MM_BIG_GRID_BH=1.
# "intermediate_l1" – list of per-op prefill intermediates to force into L1
#                 even though the persistent residual stays in DRAM.  This is
#                 the BGE-M3 bs32 memory strategy: the residual (bs*seq*dim*2 B)
#                 is too big for L1 at bs>8, but the short-lived matmul outputs
#                 inside a layer (produced, consumed, deallocated) can still
#                 ride in L1 — saving the DRAM round-trip on the hottest
#                 tensors.  Each entry sets TT_PREFILL_<OP>_L1=1.  Only the
#                 outputs that fit alongside the matmul static circular buffers
#                 are listed per shape (the big FF13/QKV tensors clash with the
#                 matmul CBs at bs=32 but fit at bs=16).  Measured on P150:
#                   bs16/isl512: +6.8%  (80.9k -> 86.4k tok/s)
#                   bs32/isl512: +2.2%  (83.4k -> 85.2k tok/s)

_INTERM_L1_BS16 = ["SDPA", "FF2", "CONCAT", "FF13", "QKV"]
_INTERM_L1_BS32 = ["SDPA", "FF2", "CONCAT"]

WORKLOAD_CONFIGS = {
    # L1-backed activations (total activation < 10 MB fits in P150 L1)
    (1, 512): {"batched_l1": False, "dram_grid": True},  # 1 MB
    (1, 1024): {"batched_l1": False, "dram_grid": True},  # 2 MB
    (1, 2048): {"batched_l1": False, "dram_grid": True},  # 4 MB
    (8, 512): {"batched_l1": True, "dram_grid": False},  # 8 MB
    # DRAM-backed residual (> 10 MB) — wider matmul grid + per-op L1 intermediates
    (16, 512): {"batched_l1": False, "dram_grid": True, "intermediate_l1": _INTERM_L1_BS16},  # 16 MB
    (8, 1024): {"batched_l1": False, "dram_grid": True},  # 16 MB
    (8, 2048): {"batched_l1": False, "dram_grid": True},  # 32 MB
    (32, 512): {"batched_l1": False, "dram_grid": True, "intermediate_l1": _INTERM_L1_BS32},  # 32 MB
    (32, 1024): {"batched_l1": False, "dram_grid": True},  # 64 MB
    (32, 2048): {"batched_l1": False, "dram_grid": True},  # 128 MB
}


def apply_workload_env(batch_size: int, seq_len: int) -> None:
    """Apply optimized env vars for a specific (batch_size, seq_len) workload.

    Looks up ``WORKLOAD_CONFIGS`` for the exact pair.  Falls back to a
    heuristic for unseen combinations: L1-backed if total activation fits
    in ~8 MiB, DRAM + big matmul grid otherwise.
    """
    cfg = WORKLOAD_CONFIGS.get((batch_size, seq_len))
    if cfg is None:
        activation_bytes = batch_size * seq_len * 1024 * 2
        fits_l1 = activation_bytes <= 8 * 1024 * 1024
        cfg = {"batched_l1": batch_size > 1 and fits_l1, "dram_grid": not fits_l1}

    apply_recommended_env(batched_l1=cfg["batched_l1"])
    if cfg["dram_grid"]:
        os.environ.setdefault("QWEN_MM_BIG_GRID_BH", "1")
    if cfg.get("minimal_mm_bs1"):
        os.environ.setdefault("QWEN_MINIMAL_MM_BS1", "1")
    # Per-op L1 placement for short-lived prefill intermediates (BGE-M3-style):
    # residual stays in DRAM but the hot matmul outputs that fit alongside the
    # static CBs are kept in L1. No-op unless this shape lists any.
    for op in cfg.get("intermediate_l1", []):
        os.environ.setdefault(f"TT_PREFILL_{op}_L1", "1")


def pplx_optimizations(model_args):
    """Aggressive performance precision for pplx-embed embedding workloads.

    All weights BFP4, all matmuls LOFI, SDPA LOFI (safe for embedding
    models — cosine similarity is robust to low-precision attention).
    """
    base = DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    promote_ff2 = os.getenv("QWEN_FF2_BFP4", "0") == "1"
    promote_qkv = os.getenv("QWEN_QKV_BFP4", "0") == "1"
    promote_wo = os.getenv("QWEN_WO_BFP4", "0") == "1"

    seen = set()
    for decoder_id in range(model_args.n_layers):
        opt = base.decoder_optimizations[decoder_id]
        if id(opt) in seen:
            continue
        seen.add(id(opt))
        tp = opt._opt_settings["TensorPrecision"]
        of = opt._opt_settings["OpFidelity"]
        if promote_ff2:
            tp[TensorGroup.FF2] = PrecisionSetting.BFP4
            of[OpGroup.LI_FF2] = MathFidelitySetting.LOFI
        if promote_qkv:
            tp[TensorGroup.WQKV] = PrecisionSetting.BFP4
            of[OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_QKV_DECODE] = MathFidelitySetting.LOFI
        if promote_wo:
            tp[TensorGroup.WO] = PrecisionSetting.BFP4
            of[OpGroup.LI_O_PREFILL] = MathFidelitySetting.LOFI
            of[OpGroup.LI_O_DECODE] = MathFidelitySetting.LOFI
        # SDPA LOFI — safe for embedding (cosine similarity insensitive)
        of[OpGroup.SDPA_PREFILL] = MathFidelitySetting.LOFI
        of[OpGroup.SDPA_DECODE] = MathFidelitySetting.LOFI
        # MLP LOFI across the board (FF1/FF3 already LOFI from performance base)
        of[OpGroup.LI_FF1_FF3] = MathFidelitySetting.LOFI
    base._update_full_name()
    return base


# ---------------------------------------------------------------------------
# Model build / inputs
# ---------------------------------------------------------------------------


def _page_params_for(batch_size: int, seq_len: int) -> dict:
    if batch_size == 1:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 512}
    if batch_size == 8:
        return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": 1024}
    page_max = max(512, math.ceil(seq_len / BLOCK_SIZE) * batch_size * 2)
    return {"page_block_size": BLOCK_SIZE, "page_max_num_blocks": page_max}


def build_single_device_model(mesh_device, batch_size: int, seq_len: int):
    """Build one pplx-embed model instance + Generator + page table."""
    page_params = _page_params_for(batch_size, seq_len)
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    padded_seq_len = get_padded_prefill_len(seq_len)

    model_args = PplxModelArgs(
        mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=pplx_optimizations,
        max_seq_len=padded_seq_len,
        prefetcher=None,
    )

    state_dict = model_args.load_state_dict()

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        attention_class=PplxBidirectionalAttention,
    )

    kv_caches = [[layer.attention.layer_past for layer in model.layers]]
    generator = Generator(
        [model],
        [model_args],
        mesh_device,
        tokenizer=model_args.tokenizer,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)

    return generator, model_args, kv_caches, page_table


def generate_synthetic_inputs(tokenizer, batch_size: int, seq_len: int):
    """Random tokens of exactly ``seq_len``."""
    vocab_size = tokenizer.vocab_size
    high = min(vocab_size, 50000)
    input_ids = torch.randint(100, high, (batch_size, seq_len), dtype=torch.long)
    prompt_lens = [seq_len] * batch_size
    return input_ids, prompt_lens


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_perf(
    mesh_device,
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    *,
    emit_signposts: bool,
    is_ci_env: bool = False,
    full_pipeline: bool = False,
):
    """Build, compile, and benchmark pplx-embed-v1-0.6B."""
    profiler = BenchmarkProfiler()
    profiler.start("run")
    tt_device_name = determine_device_name(mesh_device)

    logger.info(f"Building pplx-embed-v1-0.6B: bs={batch_size}, seq_len={seq_len}, device={tt_device_name}")

    profiler.start("build_model")
    generator, model_args, kv_caches, page_table = build_single_device_model(
        mesh_device, batch_size=batch_size, seq_len=seq_len
    )
    profiler.end("build_model")
    logger.info(f"Built in {profiler.get_duration('build_model'):.1f}s")

    input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, batch_size, seq_len)
    total_input_tokens = sum(prompt_lens)

    logger.info("Compiling (first prefill captures hardware trace + runs warmup)...")
    profiler.start("compile_prefill")
    _ = generator.prefill_forward_text(
        input_ids,
        page_table=page_table,
        kv_cache=kv_caches,
        prompt_lens=prompt_lens,
        enable_trace=True,
        return_hidden_states=True,
        warmup_prefill=True,
    )
    profiler.end("compile_prefill")
    logger.info(f"Compile prefill: {profiler.get_duration('compile_prefill'):.2f}s")

    # Locate the captured trace for direct replay — bypasses the Generator's
    # per-iteration Python overhead (page table reset, prefill_forward_text
    # loop, process_hidden_states_after_prefill_trace, D2H copy). This gives
    # us pure device-execution + sync latency.
    trace_key = f"{seq_len}_0_{batch_size}"
    trace_id = generator.trace_id_prefill.get(trace_key)
    use_direct_trace = (trace_id is not None) and not full_pipeline

    # --- Optimized full-pipeline path ---
    # Capture an *extended* trace that includes post-processing ops (slice +
    # norm + to_layout) so they execute as part of trace replay rather than
    # as individually dispatched ops.  Pre-compute host inputs once to skip
    # Generator Python overhead in the hot loop.
    ext_trace_id = None
    ext_trace_output = None
    ext_device_inputs = None
    ext_host_inputs = None
    if full_pipeline and trace_id is not None:
        model = generator.model[0]
        last_token_idx = seq_len - 1
        is_batched = batch_size > 1
        padded_batch = model_args.max_batch_size

        # Release all Generator traces — we'll capture a new extended trace
        # that includes post-processing ops.  Releasing all avoids "unsafe
        # allocation" warnings from warmup traces at shorter seq_lens.
        for key, tid in list(generator.trace_id_prefill.items()):
            if tid is not None:
                ttnn.release_trace(mesh_device, tid)
                generator.trace_id_prefill[key] = None

        # Prepare inputs matching what the Generator would produce
        if is_batched:
            prefill_ids = torch.zeros(padded_batch, seq_len, dtype=torch.long)
            padded_last_token_idx = [0] * padded_batch
            for slot in range(batch_size):
                prefill_ids[slot] = input_ids[slot]
                padded_last_token_idx[slot] = last_token_idx
            ext_last_token_idx = padded_last_token_idx
            get_last_token = (last_token_idx // 32) * 32
            prefill_kwargs = {"page_table": page_table, "batch_size": batch_size, "user_id": 0}
        else:
            prefill_ids = input_ids
            ext_last_token_idx = last_token_idx
            get_last_token = (last_token_idx // 32) * 32
            prefill_kwargs = {"page_table": page_table[0:1]}

        host_inputs_full = model.prepare_prefill_inputs_trace(prefill_ids, **prefill_kwargs)
        rot_mats_global = host_inputs_full[1]
        rot_mats_local = host_inputs_full[2]
        ext_host_inputs = (host_inputs_full[0], host_inputs_full[3], host_inputs_full[4])

        fwd_kwargs = dict(
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            kv_cache=kv_caches[0],
        )
        if is_batched:
            fwd_kwargs["batch_size"] = batch_size
            fwd_kwargs["user_id"] = 0

        def _forward_and_postprocess(dinputs):
            transformed = model.transform_and_embed_prefill_inputs_device(*dinputs)
            tt_out = model.ttnn_prefill_forward(
                x=transformed[0],
                page_table=transformed[1],
                chunk_page_table=transformed[2],
                **fwd_kwargs,
            )
            if is_batched:
                return model.process_hidden_states_after_prefill_trace_batched(tt_out, get_last_token)
            return model.process_hidden_states_after_prefill_trace(tt_out, last_token_idx)

        # Warm-run to ensure all post-processing kernels are compiled
        device_inputs = copy_host_to_device(ext_host_inputs, mesh_device=mesh_device)
        _ = _forward_and_postprocess(device_inputs)
        ttnn.synchronize_device(mesh_device)

        # Capture extended trace: forward + post-processing in one replay unit
        device_inputs = copy_host_to_device(ext_host_inputs, mesh_device=mesh_device)
        ext_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        ext_trace_output = _forward_and_postprocess(device_inputs)
        ttnn.end_trace_capture(mesh_device, ext_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ext_device_inputs = device_inputs
        logger.info("Captured extended trace (forward + post-processing)")

    if use_direct_trace:
        logger.info(f"Running {num_iterations} iterations via direct trace replay (key={trace_key})...")
    elif ext_trace_id is not None:
        logger.info(f"Running {num_iterations} iterations via extended trace (forward + post-proc)...")
    else:
        logger.info(f"Running {num_iterations} iterations via generator (no direct trace)...")

    dim = model_args.dim

    iteration_times = []
    last_iter_idx = num_iterations - 1
    for i in range(num_iterations):
        sig = emit_signposts and i == last_iter_idx

        profiler.start(f"inference_prefill_{i}")
        if sig:
            _tracy_signpost("start")
        try:
            if use_direct_trace:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
            elif ext_trace_id is not None:
                copy_host_to_device(ext_host_inputs, device_tensors=ext_device_inputs, mesh_device=mesh_device)
                ttnn.execute_trace(mesh_device, ext_trace_id, cq_id=0, blocking=False)
                hidden_host = ext_trace_output.cpu(blocking=False)
                ttnn.synchronize_device(mesh_device)
                _ = ttnn.to_torch(ttnn.get_device_tensors(hidden_host)[0])
            else:
                generator.prev_page_table = None
                generator.prefill_forward_text(
                    input_ids,
                    page_table=page_table,
                    kv_cache=kv_caches,
                    prompt_lens=prompt_lens,
                    enable_trace=True,
                    return_hidden_states=True,
                    warmup_prefill=False,
                )
                ttnn.synchronize_device(mesh_device)
        finally:
            if sig:
                _tracy_signpost("stop")
        profiler.end(f"inference_prefill_{i}")

        t = profiler.get_duration(f"inference_prefill_{i}")
        iteration_times.append(t)
        logger.info(f"  Iteration {i}: {t * 1000:.1f}ms")

    avg_t = sum(iteration_times) / len(iteration_times)
    best_t = min(iteration_times)
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "avg_prefill_time": avg_t,
        "best_prefill_time": best_t,
        "embeddings/s_avg": batch_size / avg_t,
        "embeddings/s_best": batch_size / best_t,
        "prefill_t/s_avg": total_input_tokens / avg_t,
        "prefill_t/s_best": total_input_tokens / best_t,
        "build_model_time": profiler.get_duration("build_model"),
        "batch_size": batch_size,
        "input_seq_len": seq_len,
        "total_input_tokens": total_input_tokens,
    }

    mode_label = "full pipeline" if full_pipeline else "direct trace"
    time_label = "full pipeline time" if full_pipeline else "prefill time"

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  pplx-embed-v1-0.6B Performance  ({tt_device_name})")
    logger.info("=" * 60)
    logger.info(f"  Batch size:           {batch_size}")
    logger.info(f"  Input seq length:     {seq_len}")
    logger.info(f"  Total input tokens:   {total_input_tokens}")
    logger.info(f"  Iterations:           {num_iterations}")
    logger.info(f"  Mode:                 {mode_label}")
    logger.info("-" * 60)
    logger.info(f"  Model build time:     {measurements['build_model_time']:.1f}s")
    logger.info(f"  Compile (1st run):    {measurements['compile_prefill']:.2f}s")
    logger.info("-" * 60)
    logger.info(f"  Avg {time_label}:     {avg_t * 1000:.1f}ms")
    logger.info(f"  Best {time_label}:    {best_t * 1000:.1f}ms")
    logger.info(f"  Avg embeddings/s:     {measurements['embeddings/s_avg']:.1f}")
    logger.info(f"  Best embeddings/s:    {measurements['embeddings/s_best']:.1f}")
    logger.info(f"  Avg tokens/s:         {measurements['prefill_t/s_avg']:.0f}")
    logger.info(f"  Best tokens/s:        {measurements['prefill_t/s_best']:.0f}")
    logger.info("=" * 60)

    profiler.end("run")

    if is_ci_env:
        benchmark_data = create_benchmark_data(profiler, measurements, {}, {})
        benchmark_data.save_partial_run_json(
            profiler,
            run_type=f"{tt_device_name}-demo",
            ml_model_name="pplx-embed-v1-0.6B",
            ml_model_type="embedding",
            num_layers=model_args.n_layers,
            batch_size=batch_size,
            config_params={"data_parallel": 1, "tensor_parallel": 1},
            input_sequence_length=seq_len,
            output_sequence_length=0,
        )

    return measurements


# ---------------------------------------------------------------------------
# Standalone (no-pytest) entry point
# ---------------------------------------------------------------------------


def standalone_main(
    batch_size: int, seq_len: int, iterations: int, device_id: int = 0, full_pipeline: bool = False
) -> None:
    """`python <entry_file>` path — opens its own device, no pytest fixture."""
    apply_workload_env(batch_size, seq_len)

    logger.info(f"Opening device {device_id}...")
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=200_000_000,
        num_command_queues=1,
    )
    try:
        t0 = time.perf_counter()
        run_perf(
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            num_iterations=iterations,
            emit_signposts=False,
            full_pipeline=full_pipeline,
        )
        logger.info(f"Total wall time: {time.perf_counter() - t0:.1f}s")
    finally:
        ttnn.close_device(device)
