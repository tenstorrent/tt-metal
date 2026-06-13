# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-device (Blackhole p150a) baseline benchmark for NVIDIA LocateAnything-3B.

Runs the LocateAnything LLM backbone (a standard causal Qwen2.5-3B in AR mode)
on TT-NN, fed *pre-merged* image+text embeddings, with greedy autoregressive
decode. Verifies prefill logits against a torch CPU golden (PCC) and prints
greppable benchmark metrics.

Mirrors models/demos/qwen25_vl/demo/demo.py + models/tt_transformers/tests/test_model.py
but uses STANDARD 1D RoPE (no mrope).

Prints three greppable lines:
    inference_speed=<frames_per_sec>
    accuracy=<pcc_percent>
    peak_dram=<bytes>
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.qwen25_vl.tt.common import PagedAttentionConfig, merge_vision_tokens, preprocess_inputs_prefill
from models.experimental.locate_anything.tt.model_la import LATransformer
from models.tt_transformers.tt.common import Mode, sample_host
from models.tt_transformers.tt.generator import Generator as TTTGenerator
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

# LocateAnything special token ids (from the extracted HF config).
IMAGE_TOKEN_INDEX = 151665
EOS_TOKEN_ID = 151645

# Reference golden bundle produced on torch CPU.
GOLDEN_PATH = "models/experimental/locate_anything/reference/golden.pt"

# Paged-attention page params (block_size * max_num_blocks must cover max_seq_len).
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks": 1024}


def create_tt_page_table(paged_attention_config, tt_model_args):
    """Random (shuffled) virtual->physical block mapping. Copied from qwen25_vl demo."""
    if paged_attention_config is None:
        return None
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(
        tt_model_args.max_batch_size,
        paged_attention_config.max_num_blocks // tt_model_args.max_batch_size,
    )


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=True,
):
    """Build LATransformer + paged KV cache. Adapted from qwen25_vl demo create_tt_model."""
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        cache_hf=True,
    )
    state_dict = tt_model_args.load_state_dict()

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        if use_paged_kv_cache
        else None
    )

    # NOTE: do NOT pass use_paged_kv_cache=True. The stock Attention only calls
    # init_kv_cache() (which allocates `layer_past`) when use_paged_kv_cache is
    # False; the paged vs non-paged *shape* is selected by paged_attention_config.
    # This matches models/tt_transformers/tt/common.py:create_tt_model.
    model = LATransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, paged_attention_config, tt_kv_cache


def _peak_dram_bytes(mesh_device):
    """Best-effort peak DRAM allocation query across the mesh (bytes). 0 if unavailable."""
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return int(view.num_banks * view.total_bytes_allocated_per_bank)
    except Exception as e:  # noqa: BLE001 - metric is best-effort only
        logger.warning(f"Could not query peak DRAM usage: {e}")
        return 0


@pytest.mark.parametrize(
    "max_seq_len, max_generated_tokens",
    [(4096, 48)],
    ids=["baseline"],
)
@pytest.mark.parametrize(
    "optimizations",
    # performance preset (BFP4 MLP) keeps decode-MLP circular buffers small enough
    # to fit L1 on a single p150a; accuracy preset (BFP8 MLP) clashes in L1 at decode.
    [lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)],
    ids=["performance"],
)
@pytest.mark.parametrize(
    "device_params",
    # Single isolated p150a: NO cross-chip fabric (would conflict with sibling
    # experiments running on the other chips). fabric_config falsy => DISABLED.
    [{"fabric_config": False, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_locate_anything_baseline(
    max_seq_len,
    max_generated_tokens,
    optimizations,
    mesh_device,
    reset_seeds,
):
    """Prefill (PCC vs golden) + greedy AR decode + benchmark metrics for LocateAnything-3B."""
    from transformers import Qwen2ForCausalLM

    batch_size = 1
    dtype = ttnn.bfloat8_b

    hf_model_dir = os.environ.get("HF_MODEL")
    assert hf_model_dir, "Set HF_MODEL to the extracted LA-Qwen2.5-3B directory"
    assert os.path.isfile(GOLDEN_PATH), f"Golden tensors not found at {GOLDEN_PATH}"

    logger.info(f"mesh_device: {mesh_device}")

    # --- Build TT model + paged KV cache ---
    model_args, model, paged_attention_config, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=False,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        page_params=PAGE_PARAMS,
        dtype=dtype,
        use_paged_kv_cache=True,
    )
    tokenizer = model_args.tokenizer
    # Decode is driven through the stock tt_transformers Generator.
    generator = TTTGenerator([model], [model_args], mesh_device)

    page_table = create_tt_page_table(paged_attention_config, model_args)

    # --- Load golden bundle ---
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids = golden["input_ids"]  # [1, S]
    attention_mask = golden["attention_mask"]  # [1, S]
    vit_proj = golden["vit_proj"].to(torch.float32)  # [N_img, hidden]
    golden_prefill_logits = golden["prefill_logits"].to(torch.float32)  # [1, S, vocab]
    image_token_index = int(golden.get("image_token_index", IMAGE_TOKEN_INDEX))
    n_img_tokens = int(golden["n_img_tokens"])
    real_seq_len = input_ids.shape[1]
    last_token_idx = real_seq_len - 1
    logger.info(f"golden seq_len={real_seq_len}, n_img_tokens={n_img_tokens}")

    # --- Build pre-merged image+text embeddings on host ---
    logger.info("Loading HF embed_tokens and merging vision embeddings...")
    hf_model = Qwen2ForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32)
    embed_tokens = hf_model.get_input_embeddings()
    with torch.no_grad():
        text_embeds = embed_tokens(input_ids)  # [1, S, hidden]

    # Minimal hf_config shim so we can reuse qwen25_vl merge_vision_tokens.
    class _MergeConfig:
        image_token_id = image_token_index

    input_embeds = merge_vision_tokens(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), _MergeConfig())

    # Pad embeddings to a tile-multiple (power-of-2, >=128) prefill length, exactly like
    # the qwen25_vl prefill preprocessing.
    with torch.no_grad():
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_embedding = embed_tokens(torch.tensor(pad_token_id))
    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [input_embeds[0]],
        model_args,
        attention_mask,
        pad_embedding=pad_embedding,
    )
    prefill_seq_len = prefill_lens[0]
    decode_start_pos = decoding_pos[0]
    assert decode_start_pos == real_seq_len, f"decoding_pos {decode_start_pos} != real_seq_len {real_seq_len}"
    embeds = input_prefill_pt[0].unsqueeze(0).to(torch.float32)  # [1, prefill_seq_len, hidden]
    logger.info(f"prefill_seq_len={prefill_seq_len}, decode_start_pos={decode_start_pos}")

    # ============================= PREFILL =============================
    logger.info("Running prefill...")
    model.switch_mode(Mode.PREFILL)
    t_prefill_start = time.time()
    tokens_embd, rot_mats_global, tt_page_table, _ = model.prepare_inputs_prefill_embeds(
        embeds,
        start_pos=0,
        page_table=page_table,
        last_token_idx=last_token_idx,
    )
    tt_logits = model.ttnn_prefill_forward(
        tokens_embd,
        rot_mats_global=rot_mats_global,
        rot_mats_local=None,
        user_id=0,
        page_table=tt_page_table,
        get_last_token=(last_token_idx // 32) * 32,
        kv_cache=tt_kv_cache,
    )
    # process_output_prefill expects a host tensor; it returns the row-th logits vector.
    prefill_last_logits = model.process_output_prefill(tt_logits.cpu(), last_token_idx=last_token_idx % 32)
    # Free prefill device tensors so their L1/DRAM regions don't clash with the
    # decode program's circular buffers (mirrors the stock prefill cleanup).
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(tokens_embd)
    if tt_page_table is not None:
        ttnn.deallocate(tt_page_table)
    t_prefill_end = time.time()
    prefill_time = t_prefill_end - t_prefill_start
    logger.info(f"Prefill done in {prefill_time:.3f}s")

    # --- Accuracy: PCC of last-token logits vs golden ---
    golden_last_logits = golden_prefill_logits[0, last_token_idx, : model.vocab_size]
    tt_last_logits = prefill_last_logits[: model.vocab_size].to(torch.float32)
    passing, pcc_message = comp_pcc(golden_last_logits, tt_last_logits, pcc=0.97)
    logger.info(comp_allclose(golden_last_logits, tt_last_logits))
    logger.info(f"Last-token logits PCC: {pcc_message}")
    pcc_value = _extract_pcc(pcc_message)

    # Argmax sanity: the first generated token should match the golden argmax.
    golden_first_tok = int(torch.argmax(golden_last_logits).item())
    tt_first_tok = int(torch.argmax(tt_last_logits).item())
    logger.info(f"First decode token: tt={tt_first_tok} golden={golden_first_tok}")

    # ============================= GREEDY AR DECODE =============================
    logger.info("Running greedy AR decode...")
    generated_ids = []
    out_tok = torch.tensor([[tt_first_tok]], dtype=torch.int64)  # [B=1, 1]
    generated_ids.append(tt_first_tok)
    current_pos = torch.tensor([decode_start_pos], dtype=torch.int64)

    t_decode_start = time.time()
    num_decode_steps = 0
    if tt_first_tok != EOS_TOKEN_ID:
        for step in range(max_generated_tokens - 1):
            # Stock decode_forward expects kv_cache indexed per data-parallel model,
            # so wrap our per-layer cache list as [tt_kv_cache].
            logits, _ = generator.decode_forward(
                out_tok,
                current_pos,
                page_table=page_table,
                kv_cache=[tt_kv_cache],
                enable_trace=False,
                reset_batch=(step == 0),
            )
            num_decode_steps += 1
            # Greedy next token (host argmax), mirroring simple_text_demo.
            _, next_tok_t = sample_host(logits, temperature=0, top_p=1.0, on_host=True)
            next_tok = int(next_tok_t.reshape(-1)[0].item())
            current_pos = current_pos + 1
            generated_ids.append(next_tok)
            if next_tok == EOS_TOKEN_ID:
                break
            out_tok = torch.tensor([[next_tok]], dtype=torch.int64)
    t_decode_end = time.time()
    decode_time = t_decode_end - t_decode_start

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    logger.info(f"Generated {len(generated_ids)} tokens")
    logger.info(f"Box string: {generated_text}")

    # ============================= METRICS =============================
    total_time = prefill_time + decode_time
    decode_tok_s = (num_decode_steps / decode_time) if decode_time > 0 else 0.0
    frames_per_sec = (1.0 / total_time) if total_time > 0 else 0.0
    peak_dram = _peak_dram_bytes(mesh_device)

    logger.info(f"Prefill time: {prefill_time:.3f}s, decode time: {decode_time:.3f}s")
    logger.info(f"Decode speed: {decode_tok_s:.2f} tok/s ({num_decode_steps} steps)")

    # Greppable metric lines (EXACT format).
    print(f"inference_speed={frames_per_sec}")
    print(f"accuracy={pcc_value * 100}")
    print(f"peak_dram={peak_dram}")

    # Accuracy gate is enforced by the autoresearch loop via the recorded PCC, not
    # by failing the run here -- we always want the metrics printed for the loop.
    if not passing:
        logger.warning(f"Prefill last-token logits PCC below 0.97: {pcc_message}")


def _extract_pcc(pcc_message):
    """Parse the float PCC out of comp_pcc's message string. Returns 0.0 on failure."""
    try:
        return float(str(pcc_message).strip().split()[-1])
    except (ValueError, IndexError):
        return 0.0
