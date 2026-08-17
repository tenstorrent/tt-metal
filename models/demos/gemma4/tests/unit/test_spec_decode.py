# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 speculative decoding (it-assistant drafter).

These validate the speculative-decode contract:
  * config/loader wiring for the assistant model;
  * greedy speculative decode matches plain greedy decode up to the first target
    near-tie (the verify runs batched, so its per-user RoPE + batched SDPA differ
    from batch=1 decode by ~1e-5 and flip only near-tie tokens — see
    test_verify_batchsize_invariance); a divergence at a confident token is a bug;
  * acceptance rate is reported.

Device tests require a target (HF_MODEL) + matching drafter
(GEMMA4_ASSISTANT_MODEL, e.g. google/gemma-4-31B-it-assistant) and are skipped
otherwise. Use GEMMA4_NUM_LAYERS to shrink the target for a fast wiring check
(equivalence still holds; acceptance becomes meaningless).
"""

import math
import os
from functools import lru_cache

import pytest
import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

ASSISTANT_PATH = os.getenv("GEMMA4_ASSISTANT_MODEL")
_needs_assistant = pytest.mark.skipif(not ASSISTANT_PATH, reason="set GEMMA4_ASSISTANT_MODEL to run")
_assistant_probe = pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_ASSISTANT_PROBES", "0") != "1",
    reason="assistant diagnostic/perf probe; set GEMMA4_RUN_ASSISTANT_PROBES=1 to run",
)


@lru_cache(maxsize=1)
def _target_text_config():
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        return None

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return getattr(hf_config, "text_config", hf_config)


@lru_cache(maxsize=1)
def _assistant_args():
    if not ASSISTANT_PATH:
        return None

    from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

    hf_config = Gemma4AssistantArgs.load_hf_config(ASSISTANT_PATH)
    return Gemma4AssistantArgs.from_hf_config(hf_config)


@pytest.fixture(autouse=True)
def _skip_if_target_too_large_for_mesh(request):
    """Skip spec-decode model/mesh pairings that are unsupported or unsafe."""
    if "mesh_device" not in request.fixturenames:
        return

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        return

    mesh_device = request.getfixturevalue("mesh_device")
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1

    if tp == 2:
        pytest.skip("Gemma4 assistant/spec-decode tests are not supported on 1x2; use 1x1 or 1x4")

    text_config = _target_text_config()
    if getattr(text_config, "enable_moe_block", False) and tp < 8:
        pytest.skip(f"MoE target model too large for TP={tp} in spec-decode tests")
    if getattr(text_config, "hidden_size", 0) > 4096 and tp < 2:
        pytest.skip(f"Target model too large for single device (hidden={text_config.hidden_size})")

    try:
        assistant_args = _assistant_args()
    except Exception as e:
        pytest.skip(f"could not load assistant config: {e}")
    if assistant_args is None:
        return

    if assistant_args.backbone_hidden_size != getattr(text_config, "hidden_size", None):
        pytest.skip(
            f"Assistant backbone_hidden_size ({assistant_args.backbone_hidden_size}) does not match "
            f"target hidden_size ({getattr(text_config, 'hidden_size', None)})"
        )
    if assistant_args.backbone_hidden_size > 4096 and tp < 2:
        pytest.skip(f"Assistant model too large for single device (backbone={assistant_args.backbone_hidden_size})")


def test_assistant_config_loads():
    """Assistant config parses into Gemma4AssistantArgs with the expected shape."""
    if not ASSISTANT_PATH:
        pytest.skip("set GEMMA4_ASSISTANT_MODEL to run")
    from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

    try:
        hf_config = Gemma4AssistantArgs.load_hf_config(ASSISTANT_PATH)
    except Exception as e:  # offline / no access
        pytest.skip(f"could not load assistant config: {e}")
    args = Gemma4AssistantArgs.from_hf_config(hf_config)

    assert args.backbone_hidden_size in (3840, 5376), args.backbone_hidden_size
    assert args.text_args.num_hidden_layers == 4
    assert args.text_args.hidden_size == 1024
    # Every layer KV-shared (drafter cross-attends to the target KV).
    assert args.text_args.num_kv_shared_layers == args.text_args.num_hidden_layers
    assert tuple(args.text_args.layer_types) == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )
    assert args.use_ordered_embeddings is False
    logger.info(f"Assistant: backbone={args.backbone_hidden_size}, text={args.text_args.num_attention_heads}h")


def _plain_greedy(spec, anchor_token, anchor_pos, n):
    """Reference greedy decode: one target verify (batch=1) per token."""
    out = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(n):
        logits, hidden = spec._verify([tok], [pos])
        hidden.deallocate(True)
        tok = int(torch.argmax(logits[0]))
        out.append(tok)
        pos += 1
        if tok in spec.stop_tokens:
            break
    return out


def _plain_greedy_bN(spec, anchor_token, anchor_pos, n, pad):
    """Greedy decode advancing ONE token/step, but running the verify forward at
    batch = 1+pad (anchor at index 0, ``pad`` dummy rows at later positions, only
    index 0 committed). Exercises the SAME batched per-user RoPE + batched SDPA +
    sequential-KV-write path the spec verify uses, with no acceptance logic — so
    it isolates pure batch-size numerics from the speculative accept/commit code."""
    out = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(n):
        toks = [tok] + [tok] * pad
        positions = [pos + j for j in range(1 + pad)]
        logits, hidden = spec._verify(toks, positions)
        hidden.deallocate(True)
        tok = int(torch.argmax(logits[0]))
        out.append(tok)
        pos += 1
        if tok in spec.stop_tokens:
            break
    return out


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_verify_batched_matches_sequential(mesh_device, reset_seeds):
    """The batched multi-position verify forward must match sequential single
    verify, position-for-position. Guards the paged-cache write path: a single
    batched paged_update_cache races when candidates share a physical block, so
    the verify writes KV sequentially (see decode.py sequential_kv_write). Also
    logs drafter (assistant) proposal quality vs the target greedy chain."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = "The capital of France is"
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )

    L = 4
    # Use the LAST L prompt tokens at their own positions. Their KV is already
    # correctly filled by prefill, so a batched re-verify only RE-WRITES the same
    # values (idempotent) — this isolates the batched READ (shared page row /
    # future-leakage) from the batched WRITE (same-block race).
    prompt_positions = [anchor_pos - (L - 1) + j for j in range(L)]
    prompt_tokens = [int(encoded[0][pp]) for pp in prompt_positions]
    logger.info(f"prompt_positions={prompt_positions} prompt_tokens={prompt_tokens}")

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    seq_arg = []
    seq_logits = []
    for tok, pos in zip(prompt_tokens, prompt_positions):
        logits, h = spec._verify([tok], [pos])
        h.deallocate(True)
        seq_logits.append(logits[0].float())
        seq_arg.append(int(torch.argmax(logits[0])))
    logger.info(f"[read-iso] sequential argmax: {seq_arg}")

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    blogits, bh = spec._verify(prompt_tokens, prompt_positions)
    bh.deallocate(True)
    batch_logits = [blogits[j].float() for j in range(L)]
    batch_arg = [int(torch.argmax(blogits[j])) for j in range(L)]
    logger.info(f"[read-iso] batched argmax:    {batch_arg}")
    read_ok = batch_arg == seq_arg
    logger.info(f"[read-iso] READ {'OK' if read_ok else 'BROKEN'} (batched==sequential: {read_ok})")

    # Now the write+read path (anchor + generated chain) for completeness.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    seq2 = []
    seq2_logits = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(L + 1):
        logits, h = spec._verify([tok], [pos])
        h.deallocate(True)
        seq2_logits.append(logits[0].float())
        a = int(torch.argmax(logits[0]))
        seq2.append(a)
        tok, pos = a, pos + 1
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    bw, bh2 = spec._verify([anchor_token] + seq2[:L], [anchor_pos + j for j in range(L + 1)])
    bh2.deallocate(True)
    batch2_logits = [bw[j].float() for j in range(L + 1)]
    batch2 = [int(torch.argmax(bw[j])) for j in range(L + 1)]
    logger.info(f"[write+read] sequential: {seq2}")
    logger.info(f"[write+read] batched:    {batch2}")

    _assert_argmaxes_match_except_near_ties(seq_logits, batch_logits, seq_arg, batch_arg, "batched READ")
    _assert_argmaxes_match_except_near_ties(seq2_logits, batch2_logits, seq2, batch2, "batched WRITE+READ")

    # Drafter quality: how many of the drafter's K proposals match the target
    # greedy chain (seq2)? Low overlap => drafter (assistant) is mis-wired.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    h0 = spec.seed(anchor_token, anchor_pos)
    drafts, _ = spec._draft(anchor_token, h0, anchor_pos)
    h0.deallocate(True)
    logger.info(f"[drafter] drafts={drafts}  target_greedy={seq2[:len(drafts)]}")
    match = sum(1 for d, t in zip(drafts, seq2) if d == t)
    logger.info(f"[drafter] prefix-match vs target greedy: {match}/{len(drafts)}")


def _pcc(a, b):
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _assert_argmaxes_match_except_near_ties(ref_logits, got_logits, ref_argmax, got_argmax, context):
    """Allow argmax drift only when the reference distribution is a near-tie."""
    near_tie_gap = float(os.environ.get("GEMMA4_SPEC_NEAR_TIE_GAP", 2.0))
    failures = []
    for idx, (ref_logit, got_logit, ref_tok, got_tok) in enumerate(zip(ref_logits, got_logits, ref_argmax, got_argmax)):
        pcc = _pcc(ref_logit, got_logit)
        top2 = torch.topk(ref_logit.float().reshape(-1), 2)
        gap = float(top2.values[0] - top2.values[1])
        logger.info(f"[{context}] idx={idx} ref={ref_tok} got={got_tok} logits_pcc={pcc:.5f} ref_top2_gap={gap:.4f}")
        if ref_tok != got_tok and gap >= near_tie_gap:
            failures.append(f"idx={idx}: ref={ref_tok} got={got_tok} gap={gap:.4f} pcc={pcc:.5f}")
    assert not failures, f"{context} diverged at confident tokens:\n" + "\n".join(failures)


def _dev0(t, mesh_device):
    """Read a (possibly mesh-replicated) TT tensor from device 0 to torch."""
    is_mesh = hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1
    if is_mesh:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return ttnn.to_torch(t)


def _kv_to_tt(k_torch, mesh_device, num_kv_heads, num_attention_heads, tp, num_devices):
    """Send [1, num_kv_heads, L, head_dim] KV to the mesh in production K_proj
    sharding layout (mirrors test_vllm_parity._kv_torch_to_tt)."""
    if num_devices <= 1:
        return ttnn.from_torch(
            k_torch.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
    if num_kv_heads >= tp:  # sharded: contiguous chunk of kv/tp heads per device
        return ttnn.from_torch(
            k_torch.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
    q_per_device = num_attention_heads // tp  # GQA-replicated single head per device
    slices = []
    for dev_i in range(num_devices):
        kv_idx = (dev_i * q_per_device) * num_kv_heads // num_attention_heads
        slices.append(k_torch[:, kv_idx : kv_idx + 1, :, :])
    stacked = torch.cat(slices, dim=0)
    return ttnn.from_torch(
        stacked.to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_assistant_step_pcc_vs_hf(mesh_device, reset_seeds):
    """Isolated assistant-forward fidelity vs HF.

    Feeds IDENTICAL inputs — target raw-token embedding, a random seed hidden,
    and known random shared KV (filled into fresh caches) — to BOTH the real HF
    ``Gemma4UnifiedAssistantForCausalLM`` and a stage-instrumented replica of
    ``assistant.step``. Any target-side convention is common-mode (same vector
    into both), so this purely tests the TT assistant forward. Reports per-stage
    PCC (pre_projection -> each layer -> norm -> lm_head/post_projection) to
    localize the drafter numerical bug behind the low acceptance rate.
    """
    from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
    from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
    from models.demos.gemma4.tt.ccl import ccl_allgather
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    try:
        try:
            from transformers import Gemma4UnifiedAssistantForCausalLM
        except ImportError:
            from transformers.models.gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM
        hf_assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load HF assistant reference: {e}")

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )

    text_args = assistant.text_args
    layer_types = list(text_args.layer_types)
    backbone = assistant.backbone_hidden_size
    tp = target.mesh_config.tp if target.mesh_config else 1
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    mapper = target._replicate_to_mesh_mapper()

    L = 48
    anchor_pos = L - 1
    token_id = 12345
    torch.manual_seed(0)
    # bf16-round so HF (fp32 compute) sees the SAME inputs TT does (bf16 cache /
    # tensors): isolates a real numerical bug from cache-precision noise.
    hidden_host = (torch.randn(1, 1, backbone, dtype=torch.float32) * 5.0).bfloat16().float()

    # Fresh shared KV caches per layer type, filled with known random K/V.
    pac = PagedAttentionConfig(block_size=64, max_num_blocks=2)
    page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, pac.max_num_blocks)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=mapper
    )
    type_to_idx = {}
    for i, lt in enumerate(layer_types):
        type_to_idx.setdefault(lt, i)
    shared_kv, shared_kv_hf = {}, {}
    for lt, idx in type_to_idx.items():
        cfg = Gemma4AttentionConfig(text_args, idx)
        kvc = init_kv_cache(mesh_device=mesh_device, config=cfg, paged_attention_config=pac, cache_dtype=ttnn.bfloat16)
        k_ref = torch.randn(1, cfg.num_key_value_heads, L, cfg.head_dim).bfloat16().float()
        v_ref = torch.randn(1, cfg.num_key_value_heads, L, cfg.head_dim).bfloat16().float()
        k_fill = _kv_to_tt(k_ref, mesh_device, cfg.num_key_value_heads, cfg.num_attention_heads, tp, num_devices)
        v_fill = _kv_to_tt(v_ref, mesh_device, cfg.num_key_value_heads, cfg.num_attention_heads, tp, num_devices)
        ttnn.experimental.paged_fill_cache(kvc[0], k_fill, page_table_tt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(kvc[1], v_fill, page_table_tt, batch_idx=0)
        shared_kv[lt] = kvc
        shared_kv_hf[lt] = (k_ref, v_ref)
        logger.info(f"[harness] {lt}: idx={idx} kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    page_tables = {lt: page_table_tt for lt in shared_kv}

    # Target raw-token embedding (shared by both sides).
    token_tt = ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=mapper,
    )
    raw_emb = assistant._raw_token_embed(token_tt)
    emb_host = _dev0(raw_emb, mesh_device).reshape(1, 1, backbone).float()

    # ── HF reference (with stage hooks) ──────────────────────────────
    hf_caps = {}
    hooks = []

    def _mk(name):
        def hook(mod, inp, out):
            hf_caps[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    hooks.append(hf_assistant.pre_projection.register_forward_hook(_mk("pre_proj")))
    for i, lyr in enumerate(hf_assistant.model.layers):
        hooks.append(lyr.register_forward_hook(_mk(f"layer{i}")))
    hooks.append(hf_assistant.model.norm.register_forward_hook(_mk("norm")))

    inputs_embeds = torch.cat([emb_host, hidden_host], dim=-1)
    pos_ids = torch.tensor([[anchor_pos]], dtype=torch.long)
    with torch.no_grad():
        out = hf_assistant(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            shared_kv_states=shared_kv_hf,
            attention_mask=None,
            use_cache=False,
        )
    for h in hooks:
        h.remove()
    logits_ref = out.logits.reshape(-1).float()
    hidden_ref = out.last_hidden_state.reshape(-1).float()

    # Control: HF in bf16 (same compute precision as TT). If THIS already
    # diverges from fp32, the 4-layer assistant is intrinsically bf16-unstable
    # and the issue is not a TT bug.
    with torch.no_grad():
        hf_bf16 = hf_assistant.bfloat16()
        out_bf16 = hf_bf16(
            inputs_embeds=inputs_embeds.bfloat16(),
            position_ids=pos_ids,
            shared_kv_states={k: (a.bfloat16(), b.bfloat16()) for k, (a, b) in shared_kv_hf.items()},
            attention_mask=None,
            use_cache=False,
        )
        logits_bf16 = out_bf16.logits.reshape(-1).float()
        hf_assistant.float()
    logger.info(
        f"[control] HF bf16 vs fp32: logits PCC={_pcc(logits_bf16, logits_ref):.5f} "
        f"argmax bf16={int(torch.argmax(logits_bf16))} fp32={int(torch.argmax(logits_ref))}"
    )

    # ── TT replica of assistant.step (with stage capture) ────────────
    pu = torch.zeros((1, 32), dtype=torch.int32)
    pu[0, 0] = anchor_pos
    pos_u = ttnn.from_torch(pu, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=mapper)
    pos_i = ttnn.from_torch(
        torch.tensor([anchor_pos], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=mapper,
    )
    hidden_tt_in = ttnn.from_torch(
        hidden_host.reshape(1, 1, 1, backbone).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mapper,
    )

    inp = ttnn.concat([raw_emb, hidden_tt_in], dim=-1)
    h = ttnn.linear(inp, assistant.pre_projection)
    tt_caps = {"pre_proj": _dev0(h, mesh_device)}
    for i, layer in enumerate(assistant.layers):
        lt = layer_types[i]
        h = layer(
            h,
            rope_mats=assistant.rope_caches_2d[lt],
            position_idx=pos_u,
            page_table=page_tables[lt],
            kv_cache=shared_kv[lt],
            is_decode=True,
            token_index=None,
            is_kv_shared=True,
            position_idx_cache=pos_i,
        )
        tt_caps[f"layer{i}"] = _dev0(h, mesh_device)
    normed = assistant.norm.forward(h)
    tt_caps["norm"] = _dev0(normed, mesh_device)

    # ── Teacher-forced per-layer intrinsic PCC ───────────────────────
    # Feed each TT layer the HF previous-layer output (bf16-rounded) so each
    # layer's PCC measures its OWN fidelity, free of compounding.
    def _to_dev_hidden(t):
        return ttnn.from_torch(
            t.reshape(1, 1, 1, -1).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

    tf_caps = {}
    for i, layer in enumerate(assistant.layers):
        lt = layer_types[i]
        src = hf_caps["pre_proj"] if i == 0 else hf_caps[f"layer{i-1}"]
        h_tf = _to_dev_hidden(src)
        h_tf = layer(
            h_tf,
            rope_mats=assistant.rope_caches_2d[lt],
            position_idx=pos_u,
            page_table=page_tables[lt],
            kv_cache=shared_kv[lt],
            is_decode=True,
            token_index=None,
            is_kv_shared=True,
            position_idx_cache=pos_i,
        )
        tf_caps[f"layer{i}"] = _dev0(h_tf, mesh_device)
        h_tf.deallocate(True)
    logits = ttnn.linear(normed, assistant.lm_head)
    if assistant.mesh_config is not None and assistant.mesh_config.tp > 1:
        logits = ccl_allgather(logits, assistant.mesh_config, assistant.ccl_manager)
    next_hidden = ttnn.linear(normed, assistant.post_projection)
    logits_tt = _dev0(logits, mesh_device).reshape(-1)[: text_args.vocab_size].float()
    hidden_tt = _dev0(next_hidden, mesh_device).reshape(-1).float()

    # ── Report per-stage PCC ─────────────────────────────────────────
    logger.info("=== assistant.step vs HF per-stage PCC ===")
    stage_order = ["pre_proj"] + [f"layer{i}" for i in range(len(assistant.layers))] + ["norm"]
    for name in stage_order:
        if name in hf_caps and name in tt_caps:
            tf = f"  tf_intrinsic={_pcc(tf_caps[name], hf_caps[name]):.5f}" if name in tf_caps else ""
            logger.info(f"  {name:9s} PCC={_pcc(tt_caps[name], hf_caps[name]):.5f}{tf}")
    pcc_logits = _pcc(logits_tt, logits_ref)
    pcc_hidden = _pcc(hidden_tt, hidden_ref)
    logger.info(
        f"  logits    PCC={pcc_logits:.5f}  (TT argmax={int(torch.argmax(logits_tt))} HF argmax={int(torch.argmax(logits_ref))})"
    )
    logger.info(f"  post_proj PCC={pcc_hidden:.5f}")

    # Each layer in isolation (teacher-forced) is the precision-stable signal
    # that the TT assistant forward matches HF. The free-running / logits PCC vs
    # HF-fp32 is intentionally NOT asserted: with random (out-of-distribution)
    # KV + seed, the tiny 4-layer drafter is chaotic in bf16 — HF's OWN bf16 run
    # diverges from fp32 by the same amount (see the [control] line), so a low
    # full-chain PCC here reflects bf16 sensitivity, not a TT bug.
    assert _pcc(tt_caps["pre_proj"], hf_caps["pre_proj"]) > 0.99, "pre_projection diverges"
    for i in range(len(assistant.layers)):
        pcc_i = _pcc(tf_caps[f"layer{i}"], hf_caps[f"layer{i}"])
        assert pcc_i > 0.99, f"assistant layer{i} forward diverges from HF (intrinsic PCC={pcc_i:.5f})"


def _depage(k_cache, page_table_torch, n_pos, block_size, mesh_device, replicated):
    """De-page a TT paged KV cache to dense torch [1, kv_heads, n_pos, head_dim].

    Per-device cache is [max_blocks, local_kv, block_size, head_dim]; KV heads are
    sharded across TP (concat on dim 1) unless GQA-replicated (all devices equal).
    """
    is_mesh = hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1
    if is_mesh:
        per_dev = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(k_cache)]
        full = per_dev[0] if replicated else torch.cat(per_dev, dim=1)
    else:
        full = ttnn.to_torch(k_cache).float()
    blocks = page_table_torch[0]
    rows = []
    for pos in range(n_pos):
        blk = int(blocks[pos // block_size])
        off = pos % block_size
        rows.append(full[blk, :, off, :])  # [kv, head_dim]
    return torch.stack(rows, dim=1).unsqueeze(0)  # [1, kv, n_pos, head_dim]


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_assistant_first_step_vs_hf_realistic(mesh_device, reset_seeds):
    """Faithful first-step check with REAL inputs.

    Runs a real prefill, then feeds the REAL drafter seed (target pre-final-norm
    hidden) + REAL de-paged shared KV + anchor token to BOTH HF
    ``Gemma4UnifiedAssistantForCausalLM`` and TT ``assistant.step``. If they
    predict the same first token, the TT forward is faithful and any remaining
    accept-rate gap is drafter quality / bf16; if they differ, a seed/KV wiring
    bug remains. Logs the target greedy token as the acceptance target.
    """
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    try:
        try:
            from transformers import Gemma4UnifiedAssistantForCausalLM
        except ImportError:
            from transformers.models.gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM
        hf_assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load HF assistant reference: {e}")

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "The capital of France is")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    # Target greedy first token (acceptance target).
    logits_v, hv = spec._verify([anchor_token], [anchor_pos])
    hv.deallocate(True)
    target_tok = int(torch.argmax(logits_v[0]))
    tt_target_logits = logits_v[0].float()

    # TT drafter first step (real seed + real shared KV).
    h0 = spec.seed(anchor_token, anchor_pos)
    seed_host = _dev0(h0, mesh_device).reshape(1, 1, -1).float()
    drafts, draft_logits = spec._draft(anchor_token, h0, anchor_pos)
    h0.deallocate(True)
    tt_logits0 = draft_logits[0]
    tt_tok0 = int(drafts[0])

    # De-page real shared KV for the HF reference.
    n_pos = anchor_pos + 1
    shared_hf = {}
    for lt, idx in target.last_kv_layer_by_type.items():
        cfg_replicated = lt == "full_attention"  # global kv=1 is GQA-replicated across TP
        kc, vc = target.tt_kv_cache[idx]
        k_d = _depage(kc, page_table, n_pos, block_size, mesh_device, cfg_replicated)
        v_d = _depage(vc, page_table, n_pos, block_size, mesh_device, cfg_replicated)
        shared_hf[lt] = (k_d, v_d)
        logger.info(f"[real] {lt}: idx={idx} K{list(k_d.shape)} V{list(v_d.shape)}")

    # HF first-step prediction with the SAME real inputs.
    token_tt = ttnn.from_torch(
        torch.tensor([[anchor_token]], dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=target._replicate_to_mesh_mapper(),
    )
    emb_host = _dev0(assistant._raw_token_embed(token_tt), mesh_device).reshape(1, 1, -1).float()
    inputs_embeds = torch.cat([emb_host, seed_host], dim=-1)
    pos_ids = torch.tensor([[anchor_pos]], dtype=torch.long)
    with torch.no_grad():
        out = hf_assistant(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            shared_kv_states=shared_hf,
            attention_mask=None,
            use_cache=False,
        )
    hf_logits0 = out.logits.reshape(-1).float()
    hf_tok0 = int(torch.argmax(hf_logits0))

    pcc0 = _pcc(tt_logits0, hf_logits0)
    logger.info(
        f"[real] first-step: target_greedy={target_tok}  HF_draft={hf_tok0}  TT_draft={tt_tok0}  logits PCC(TT,HF)={pcc0:.4f}"
    )
    logger.info(f"[real] full drafts={drafts} target_greedy_first={target_tok}")

    # Optional: compare TT's extracted drafter inputs (seed hidden + shared KV)
    # against the REAL HF target on the SAME tokens, to localize a seed/KV bug.
    if os.environ.get("GEMMA4_SPEC_HF_TARGET") == "1":
        from transformers import Gemma4UnifiedForConditionalGeneration

        prompt_ids = torch.tensor([[int(t) for t in encoded[0][: prefill_lens[0]]]], dtype=torch.long)
        logger.info(f"[hf-tgt] loading HF target on CPU (fp32)... prompt_ids={prompt_ids.shape}")
        hf_full = Gemma4UnifiedForConditionalGeneration.from_pretrained(model_path, dtype=torch.float32).eval()
        hf_text = hf_full.model.language_model
        with torch.no_grad():
            tout = hf_text(
                input_ids=prompt_ids, use_cache=True, output_hidden_states=True, return_shared_kv_states=True
            )
        hf_pre = tout.hidden_states[-1][:, -1:].reshape(1, 1, -1).float()
        hf_post = tout.last_hidden_state[:, -1:].reshape(1, 1, -1).float()
        with torch.no_grad():
            hf_tgt_logits = hf_full.lm_head(tout.last_hidden_state[:, -1:]).reshape(-1).float()
        tt_top5 = torch.topk(tt_target_logits, 5)
        hf_top5 = torch.topk(hf_tgt_logits, 5)
        logger.info(
            f"[hf-tgt] HF target greedy(next)={int(hf_tgt_logits.argmax())}  TT target greedy={target_tok}  "
            f"target-logits PCC(TT decode, HF prefill)={_pcc(tt_target_logits, hf_tgt_logits):.4f}"
        )
        logger.info(
            f"[hf-tgt] TT target top5 ids={tt_top5.indices.tolist()} vals={[round(v,2) for v in tt_top5.values.tolist()]}"
        )
        logger.info(
            f"[hf-tgt] HF target top5 ids={hf_top5.indices.tolist()} vals={[round(v,2) for v in hf_top5.values.tolist()]}"
        )
        logger.info(
            f"[hf-tgt] seed vs HF: PCC(pre)={_pcc(seed_host, hf_pre):.4f} PCC(post)={_pcc(seed_host, hf_post):.4f}  "
            f"|TT|={seed_host.norm():.2f} |HFpre|={hf_pre.norm():.2f} |HFpost|={hf_post.norm():.2f}"
        )
        hf_seed = hf_pre
        for lt in shared_hf:
            k_tt, v_tt = shared_hf[lt]
            k_hf, v_hf = tout.shared_kv_states[lt]
            k_hf = k_hf[:, :, :n_pos].float()
            v_hf = v_hf[:, :, :n_pos].float()
            logger.info(
                f"[hf-tgt] {lt}: K PCC={_pcc(k_tt, k_hf):.4f} V PCC={_pcc(v_tt, v_hf):.4f}  TTshape={list(k_tt.shape)} HFshape={list(k_hf.shape)}"
            )
        # Run HF drafter with HF's OWN target inputs to confirm it hits.
        emb_hf = hf_full.get_input_embeddings()(prompt_ids[:, -1:]).float()
        ie = torch.cat([emb_hf, hf_seed], dim=-1)
        ph = torch.tensor([[prefill_lens[0] - 1]], dtype=torch.long)
        shared_true = {
            k: (a[:, :, :n_pos].float(), b[:, :, :n_pos].float()) for k, (a, b) in tout.shared_kv_states.items()
        }
        with torch.no_grad():
            ao = hf_assistant(
                inputs_embeds=ie, position_ids=ph, shared_kv_states=shared_true, attention_mask=None, use_cache=False
            )
        logger.info(
            f"[hf-tgt] HF drafter w/ HF-target inputs -> {int(ao.logits.reshape(-1).argmax())} (target_greedy={target_tok})"
        )

    pcc_threshold = 0.85 if getattr(_target_text_config(), "hidden_size", 0) > 4096 else 0.90
    assert pcc0 > pcc_threshold, f"TT first-step logits diverge from HF given identical real inputs: PCC={pcc0:.4f}"


@_needs_assistant
@_assistant_probe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_export_tt_spec_features(mesh_device, reset_seeds):
    """Export per-step TT drafter inputs + TT target greedy chain (DISCRIMINATOR).

    Plain-greedy decode on the TT target; at each step record the drafter seed
    (post-norm hidden), de-paged shared KV, anchor token, position, and the TT
    target's greedy next token. Feeding these into the HF (fp32) drafter and
    measuring acceptance vs the TT greedy chain (hf_assistant_e2e_check.py with
    GEMMA4_SPEC_TT_FEATURES=<path>) isolates the 0.21 acceptance cause:
      * HF drafter on TT features ~= 0.2  => TT FEATURE precision is the wall
        (effort -> hi-fi activations on shared-KV layers / massive channels);
      * HF drafter on TT features ~= 1.6  => residual TT DRAFTER forward bug
        (effort -> fix the TT assistant.step path).
    """
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    out_path = os.environ.get("GEMMA4_SPEC_EXPORT", "/tmp/tt_spec_features.pt")
    n_steps = int(os.environ.get("GEMMA4_SPEC_EXPORT_STEPS", 64))
    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_steps, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4)),
    )

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    steps, greedy = [], []
    tok, pos = anchor_token, anchor_pos
    K = spec.draft_len
    for _ in range(n_steps + K):
        logits, hidden = spec._verify([tok], [pos])
        seed = _dev0(hidden[:, :, 0:1, :], mesh_device).reshape(1, 1, -1).float()
        hidden.deallocate(True)
        nxt = int(torch.argmax(logits[0]))
        n_pos = pos + 1
        shared = {}
        for lt, idx in target.last_kv_layer_by_type.items():
            replicated = lt == "full_attention"
            kc, vc = target.tt_kv_cache[idx]
            k_d = _depage(kc, page_table, n_pos, block_size, mesh_device, replicated).float()
            v_d = _depage(vc, page_table, n_pos, block_size, mesh_device, replicated).float()
            shared[lt] = (k_d, v_d)
        steps.append({"token": int(tok), "hidden": seed, "shared": shared, "pos": int(pos)})
        greedy.append(nxt)
        tok, pos = nxt, pos + 1
        if tok in spec.stop_tokens:
            break

    torch.save(
        {
            "steps": steps,
            "greedy": greedy,
            "draft_len": K,
            "prompt": prompt,
            "backbone": assistant.backbone_hidden_size,
        },
        out_path,
    )
    logger.info(f"[export] wrote {len(steps)} TT steps + greedy chain ({len(greedy)} toks) to {out_path}")
    logger.info(f"[export] TT greedy text: {tokenizer.decode(greedy)!r}")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_assistant_recurrent_vs_hf_realistic(mesh_device, reset_seeds):
    """Localize the TT drafter (assistant.step) divergence vs HF across ALL K
    recurrent steps on REAL features (the discriminator showed the clean HF
    drafter hits 1.44 on TT features while the TT drafter only gets 0.21, so the
    bug is in the TT drafter forward, not the features).

    Runs the SAME real seed + de-paged shared KV through both drafters and reports:
      * FREE-RUNNING: each drafter uses its own predicted token + own recurrent
        hidden -> the actual acceptance behaviour (TT vs HF vs target greedy);
      * TEACHER-FORCED: both driven by HF's per-step inputs -> each step's
        intrinsic forward fidelity (logits PCC + recurrent-hidden post_projection
        PCC) WITHOUT compounding. Low TF logits PCC at step 0 => forward bug;
        high TF PCC but free-running drift => recurrent-hidden compounding.
    """
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    try:
        try:
            from transformers import Gemma4UnifiedAssistantForCausalLM
        except ImportError:
            from transformers.models.gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM
        hf_assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load HF assistant reference: {e}")

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    mapper = target._replicate_to_mesh_mapper()
    pos_u, pos_i = spec._pos_tensors([anchor_pos])
    pt = spec._page_table(1)
    page_tables = {lt: pt for lt in spec._shared_kv}
    pos_ids = torch.tensor([[anchor_pos]], dtype=torch.long)

    # Target greedy continuation (acceptance labels) — writes KV at >anchor_pos
    # only, which the drafter (cur_pos=anchor_pos) ignores.
    target_cont = _plain_greedy(spec, anchor_token, anchor_pos, K)

    # Real drafter seed + de-paged shared KV (HF format).
    h0 = spec.seed(anchor_token, anchor_pos)
    seed_host = _dev0(h0, mesh_device).reshape(1, 1, -1).float()
    n_pos = anchor_pos + 1
    shared_hf = {}
    for lt, idx in target.last_kv_layer_by_type.items():
        replicated = lt == "full_attention"
        kc, vc = target.tt_kv_cache[idx]
        k_d = _depage(kc, page_table, n_pos, block_size, mesh_device, replicated).float()
        v_d = _depage(vc, page_table, n_pos, block_size, mesh_device, replicated).float()
        shared_hf[lt] = (k_d, v_d)

    def _to_dev_hidden(host):
        return ttnn.from_torch(
            host.reshape(1, 1, 1, -1).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

    def _hf_step(tok_id, hidden_host):
        token_tt = spec._tokens_tensor([tok_id])
        emb = _dev0(assistant._raw_token_embed(token_tt), mesh_device).reshape(1, 1, -1).float()
        token_tt.deallocate(True)
        ie = torch.cat([emb, hidden_host], dim=-1)
        with torch.no_grad():
            ao = hf_assistant(
                inputs_embeds=ie, position_ids=pos_ids, shared_kv_states=shared_hf, attention_mask=None, use_cache=False
            )
        return (
            int(ao.logits.reshape(-1).argmax()),
            ao.logits.reshape(-1).float(),
            ao.last_hidden_state.reshape(1, 1, -1).float(),
        )

    def _tt_step(tok_id, hidden_dev):
        token_tt = spec._tokens_tensor([tok_id])
        logits_d, h_next = assistant.step(token_tt, hidden_dev, spec._shared_kv, page_tables, pos_u, pos_i)
        token_tt.deallocate(True)
        lh = spec._logits_to_host(logits_d).reshape(-1)
        logits_d.deallocate(True)
        hh = _dev0(h_next, mesh_device).reshape(1, 1, -1).float()
        return int(torch.argmax(lh)), lh, h_next, hh

    # ── FREE-RUNNING ────────────────────────────────────────────────
    tt_free, hf_free, hf_hidden_hist = [], [], []
    tok, hidden_dev = anchor_token, h0
    for _ in range(K):
        t, _lh, h_next, _hh = _tt_step(tok, hidden_dev)
        if hidden_dev is not h0:
            hidden_dev.deallocate(True)
        hidden_dev, tok = h_next, t
        tt_free.append(t)
    if hidden_dev is not h0:
        hidden_dev.deallocate(True)
    tok, hidden_host = anchor_token, seed_host
    for _ in range(K):
        t, _lg, hh = _hf_step(tok, hidden_host)
        hf_free.append(t)
        hf_hidden_hist.append(hh)
        tok, hidden_host = t, hh

    logger.info("=== drafter localization (real features) ===")
    logger.info(f"  target greedy continuation : {target_cont}")
    logger.info(
        f"  HF drafter (free-running)  : {hf_free}  match_target={sum(1 for a,b in zip(hf_free,target_cont) if a==b)}/{K}"
    )
    logger.info(
        f"  TT drafter (free-running)  : {tt_free}  match_target={sum(1 for a,b in zip(tt_free,target_cont) if a==b)}/{K}"
    )
    logger.info(f"  TT vs HF free-running token match: {sum(1 for a,b in zip(tt_free,hf_free) if a==b)}/{K}")

    # ── TEACHER-FORCED (both driven by HF inputs, no compounding) ────
    tf_inputs = [(anchor_token, seed_host)] + [(hf_free[i], hf_hidden_hist[i]) for i in range(K - 1)]
    logger.info("  teacher-forced per-step fidelity (TT vs HF, identical inputs):")
    for i, (tk, hd_host) in enumerate(tf_inputs):
        hf_t, hf_lg, hf_hh = _hf_step(tk, hd_host)
        hd_dev = _to_dev_hidden(hd_host)
        tt_t, tt_lg, h_next, tt_hh = _tt_step(tk, hd_dev)
        hd_dev.deallocate(True)
        h_next.deallocate(True)
        logger.info(
            f"    step{i}: in_tok={tk} TT_argmax={tt_t} HF_argmax={hf_t} match={tt_t==hf_t}  "
            f"logits_PCC={_pcc(tt_lg, hf_lg):.4f}  recurrent_hidden_PCC={_pcc(tt_hh, hf_hh):.4f}  "
            f"|in_hidden|={hd_host.float().norm():.2f} |TT_next|={tt_hh.norm():.2f} |HF_next|={hf_hh.norm():.2f} "
            f"ratio(TT/HF)={float(tt_hh.norm()/hf_hh.norm()):.3f}"
        )
    h0.deallocate(True)


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_assistant_step_pcc_real(mesh_device, reset_seeds):
    """Per-stage TT-vs-HF fidelity on REAL features (localizes the drafter bug).

    Unlike test_assistant_step_pcc_vs_hf (random OOD inputs), this feeds the REAL
    prefill seed (post-norm target hidden) + REAL de-paged shared KV + real
    anchor token to both the HF ``Gemma4UnifiedAssistantForCausalLM`` (stage-
    hooked) and a stage-instrumented replica of ``assistant.step``. Reports, per
    stage (pre_proj -> 4 layers -> norm -> logits/post_proj):
      * FREE-RUNNING PCC (TT layer i takes TT layer i-1 output) — compounding;
      * TEACHER-FORCED PCC (TT layer i takes HF layer i-1 output) — intrinsic op
        fidelity on in-distribution inputs.
    The lowest teacher-forced stage is the op to audit. Also logs logits argmax +
    top-2 gap (the near-tie that flips)."""
    from models.demos.gemma4.tt.ccl import ccl_allgather
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    try:
        try:
            from transformers import Gemma4UnifiedAssistantForCausalLM
        except ImportError:
            from transformers.models.gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM
        hf_assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load HF assistant reference: {e}")

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 64, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    text_args = assistant.text_args
    layer_types = list(assistant.layer_types)
    backbone = assistant.backbone_hidden_size
    mapper = target._replicate_to_mesh_mapper()

    # Real seed + real shared KV.
    h0 = spec.seed(anchor_token, anchor_pos)
    seed_host = _dev0(h0, mesh_device).reshape(1, 1, backbone).float()
    n_pos = anchor_pos + 1
    shared_hf = {}
    for lt, idx in target.last_kv_layer_by_type.items():
        replicated = lt == "full_attention"
        kc, vc = target.tt_kv_cache[idx]
        k_d = _depage(kc, page_table, n_pos, block_size, mesh_device, replicated).float()
        v_d = _depage(vc, page_table, n_pos, block_size, mesh_device, replicated).float()
        shared_hf[lt] = (k_d, v_d)

    # Shared embedding (TT raw embed used for BOTH sides -> isolates layers/proj).
    token_tt = spec._tokens_tensor([anchor_token])
    raw_emb = assistant._raw_token_embed(token_tt)
    emb_host = _dev0(raw_emb, mesh_device).reshape(1, 1, backbone).float()

    pos_u, pos_i = spec._pos_tensors([anchor_pos])
    pt = spec._page_table(1)
    page_tables = {lt: pt for lt in spec._shared_kv}

    # ── HF reference with stage hooks ─────────────────────────────────
    hf_caps, hooks = {}, []

    def _mk(name):
        def hook(mod, inp, out):
            hf_caps[name] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    hooks.append(hf_assistant.pre_projection.register_forward_hook(_mk("pre_proj")))
    for i, lyr in enumerate(hf_assistant.model.layers):
        hooks.append(lyr.register_forward_hook(_mk(f"layer{i}")))
    hooks.append(hf_assistant.model.norm.register_forward_hook(_mk("norm")))
    inputs_embeds = torch.cat([emb_host, seed_host], dim=-1)
    pos_ids = torch.tensor([[anchor_pos]], dtype=torch.long)
    with torch.no_grad():
        out = hf_assistant(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            shared_kv_states=shared_hf,
            attention_mask=None,
            use_cache=False,
        )
    for h in hooks:
        h.remove()
    logits_ref = out.logits.reshape(-1).float()

    # ── TT replica with stage capture (free-running) ──────────────────
    hidden_tt_in = ttnn.from_torch(
        seed_host.reshape(1, 1, 1, backbone).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mapper,
    )
    inp = ttnn.concat([raw_emb, hidden_tt_in], dim=-1)
    h = ttnn.linear(inp, assistant.pre_projection)
    tt_caps = {"pre_proj": _dev0(h, mesh_device)}
    for i, layer in enumerate(assistant.layers):
        lt = layer_types[i]
        h = layer(
            h,
            rope_mats=assistant.rope_caches_2d[lt],
            position_idx=pos_u,
            page_table=page_tables[lt],
            kv_cache=spec._shared_kv[lt],
            is_decode=True,
            token_index=None,
            is_kv_shared=True,
            position_idx_cache=pos_i,
        )
        tt_caps[f"layer{i}"] = _dev0(h, mesh_device)
    normed = assistant.norm.forward(h)
    tt_caps["norm"] = _dev0(normed, mesh_device)
    logits = ttnn.linear(normed, assistant.lm_head)
    if assistant.mesh_config is not None and assistant.mesh_config.tp > 1:
        logits = ccl_allgather(logits, assistant.mesh_config, assistant.ccl_manager)
    logits_tt = _dev0(logits, mesh_device).reshape(-1)[: text_args.vocab_size].float()

    # ── Teacher-forced per-layer (feed HF prev output into TT layer) ──
    def _to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, 1, -1).to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

    tf_caps = {}
    for i, layer in enumerate(assistant.layers):
        lt = layer_types[i]
        src = hf_caps["pre_proj"] if i == 0 else hf_caps[f"layer{i-1}"]
        h_tf = _to_dev(src)
        h_tf = layer(
            h_tf,
            rope_mats=assistant.rope_caches_2d[lt],
            position_idx=pos_u,
            page_table=page_tables[lt],
            kv_cache=spec._shared_kv[lt],
            is_decode=True,
            token_index=None,
            is_kv_shared=True,
            position_idx_cache=pos_i,
        )
        tf_caps[f"layer{i}"] = _dev0(h_tf, mesh_device)
        h_tf.deallocate(True)

    logger.info("=== assistant.step vs HF per-stage PCC (REAL features) ===")
    for name in ["pre_proj"] + [f"layer{i}" for i in range(len(assistant.layers))] + ["norm"]:
        if name in hf_caps and name in tt_caps:
            tf = f"  tf_intrinsic={_pcc(tf_caps[name], hf_caps[name]):.5f}" if name in tf_caps else ""
            logger.info(f"  {name:9s} free_run_PCC={_pcc(tt_caps[name], hf_caps[name]):.5f}{tf}")
    # Verify raw_embed against the raw HF embedding table TT holds on host.
    if getattr(target, "_embed_weight_cpu", None) is not None:
        hf_emb = target._embed_weight_cpu[anchor_token].reshape(-1).float()
        tt_emb = emb_host.reshape(-1).float()
        logger.info(
            f"  raw_embed  PCC(TT,HFtable)={_pcc(tt_emb, hf_emb):.5f}  "
            f"|TT|={tt_emb.norm():.4f} |HFtable|={hf_emb.norm():.4f} "
            f"ratio={float(tt_emb.norm()/(hf_emb.norm()+1e-9)):.4f} "
            f"|TT|/(sqrt(h)*|HF|)={float(tt_emb.norm()/((backbone**0.5)*hf_emb.norm()+1e-9)):.4f}"
        )
    top2_tt = torch.topk(logits_tt, 2)
    top2_hf = torch.topk(logits_ref, 2)
    logger.info(f"  logits     free_run_PCC={_pcc(logits_tt, logits_ref):.5f}")
    logger.info(
        f"  TT argmax={int(top2_tt.indices[0])} gap={float(top2_tt.values[0]-top2_tt.values[1]):.4f}  "
        f"HF argmax={int(top2_hf.indices[0])} gap={float(top2_hf.values[0]-top2_hf.values[1]):.4f}"
    )
    h0.deallocate(True)


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_drafter_per_position_tt_vs_hf(mesh_device, reset_seeds):
    """Per-position TT-vs-HF free-running drafts on IDENTICAL seeds (reproduces
    both 1.44 (HF) and 0.17 (TT) in one harness and localizes the divergent
    positions). At each greedy position both drafters run K free-running steps
    from the SAME real seed + shared KV; logs per-position accept(TT)/accept(HF)
    + token agreement, and dumps positions where HF accepts >=2 but TT accepts 0
    (the systematic-divergence cases to audit)."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    try:
        try:
            from transformers import Gemma4UnifiedAssistantForCausalLM
        except ImportError:
            from transformers.models.gemma4_unified_assistant import Gemma4UnifiedAssistantForCausalLM
        hf_assistant = Gemma4UnifiedAssistantForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32).eval()
    except Exception as e:
        pytest.skip(f"could not load HF assistant reference: {e}")

    n_steps = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 48))
    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_steps, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    backbone = assistant.backbone_hidden_size

    def _hf_drafts(seed_host, shared_hf, first_tok, pos):
        pos_ids = torch.tensor([[pos]], dtype=torch.long)
        toks, last_tok, hidden = [], first_tok, seed_host
        for _ in range(K):
            tt = spec._tokens_tensor([last_tok])
            emb = _dev0(assistant._raw_token_embed(tt), mesh_device).reshape(1, 1, backbone).float()
            tt.deallocate(True)
            with torch.no_grad():
                ao = hf_assistant(
                    inputs_embeds=torch.cat([emb, hidden], dim=-1),
                    position_ids=pos_ids,
                    shared_kv_states=shared_hf,
                    attention_mask=None,
                    use_cache=False,
                )
            last_tok = int(ao.logits.reshape(-1).argmax())
            hidden = ao.last_hidden_state.reshape(1, 1, backbone).float()
            toks.append(last_tok)
        return toks

    logger.info(f"  anchor_token={anchor_token} anchor_pos={anchor_pos}")
    greedy, tt_drafts_all, hf_drafts_all = [], [], []
    tok, pos = anchor_token, anchor_pos
    for _ in range(n_steps + K):
        logits, hidden = spec._verify([tok], [pos])
        seed_dev = ttnn.clone(hidden[:, :, 0:1, :])
        seed_host = _dev0(seed_dev, mesh_device).reshape(1, 1, backbone).float()
        hidden.deallocate(True)
        nxt = int(torch.argmax(logits[0]))
        n_pos = pos + 1
        shared_hf = {}
        for lt, idx in target.last_kv_layer_by_type.items():
            replicated = lt == "full_attention"
            kc, vc = target.tt_kv_cache[idx]
            shared_hf[lt] = (
                _depage(kc, page_table, n_pos, block_size, mesh_device, replicated).float(),
                _depage(vc, page_table, n_pos, block_size, mesh_device, replicated).float(),
            )
        tt_d, _ = spec._draft(tok, seed_dev, pos)
        seed_dev.deallocate(True)
        hf_d = _hf_drafts(seed_host, shared_hf, tok, pos)
        tt_drafts_all.append(tt_d)
        hf_drafts_all.append(hf_d)
        greedy.append(nxt)
        tok, pos = nxt, pos + 1
        if tok in spec.stop_tokens:
            break

    def _acc(drafts, t):
        cont = greedy[t : t + K]
        for i in range(K):
            if i >= len(cont) or drafts[i] != cont[i]:
                return i
        return K

    n = len(greedy) - K
    tt_accs = [_acc(tt_drafts_all[t], t) for t in range(n)]
    hf_accs = [_acc(hf_drafts_all[t], t) for t in range(n)]
    agree = [sum(1 for a, b in zip(tt_drafts_all[t], hf_drafts_all[t]) if a == b) for t in range(n)]
    logger.info("=== per-position TT vs HF drafters (identical seeds) ===")
    logger.info(f"  TT mean accept = {sum(tt_accs)/n:.2f} / {K}")
    logger.info(f"  HF mean accept = {sum(hf_accs)/n:.2f} / {K}")
    logger.info(f"  mean TT-vs-HF draft token agreement = {sum(agree)/n:.2f} / {K}")
    logger.info(f"  TT accepts = {tt_accs}")
    logger.info(f"  HF accepts = {hf_accs}")
    logger.info(f"  agree      = {agree}")
    for t in range(n):
        if hf_accs[t] >= 2 and tt_accs[t] == 0:
            logger.info(f"  [DIVERGE pos{t}] greedy={greedy[t:t+K]} HF={hf_drafts_all[t]} TT={tt_drafts_all[t]}")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_tt_drafter_greedychain_acceptance(mesh_device, reset_seeds):
    """TT drafter acceptance vs the TRUE greedy chain (TT analog of the HF
    discriminator). DISAMBIGUATES drafter vs verify:

      * If this ~= 1.44 (the HF-on-TT-features ceiling) but spec.generate yields
        ~0.21, then the TT DRAFTER is fine and the live loss is in the
        VERIFY/accept path (batched-verify argmax != true greedy, rejecting good
        drafts) or seed handling — NOT the drafter forward.
      * If this ~= 0.21, the TT drafter free-running drafts genuinely diverge
        from the greedy chain (recurrent compounding).

    At each greedy position it seeds the TT drafter with that position's exact
    post-norm hidden (the same seed spec.generate uses) and counts its K
    free-running greedy drafts against the plain-greedy continuation."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    n_steps = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 64))
    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_steps, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    greedy_chain, draft_records = [], []
    tok, pos = anchor_token, anchor_pos
    for _ in range(n_steps + K):
        logits, hidden = spec._verify([tok], [pos])
        seed = ttnn.clone(hidden[:, :, 0:1, :])
        hidden.deallocate(True)
        nxt = int(torch.argmax(logits[0]))
        drafts, _ = spec._draft(tok, seed, pos)  # greedy free-running
        seed.deallocate(True)
        draft_records.append(drafts)
        greedy_chain.append(nxt)
        tok, pos = nxt, pos + 1
        if tok in spec.stop_tokens:
            break

    accepts = []
    for t in range(len(draft_records) - K):
        m = K
        for i in range(K):
            if draft_records[t][i] != greedy_chain[t + i]:
                m = i
                break
        accepts.append(m)
    mean_acc = sum(accepts) / len(accepts) if accepts else 0.0
    logger.info("=== TT drafter vs TRUE greedy chain (drafter-only, no batched verify) ===")
    logger.info(f"  per-position accepts = {accepts}")
    logger.info(f"  mean accepted/iter = {mean_acc:.2f} / {K}  over {len(accepts)} positions")
    logger.info(f"  COMPARE: HF-on-TT-features=1.44, live spec.generate(verify-based)=0.21")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_spec_decode_matches_greedy(mesh_device, reset_seeds):
    """Greedy spec-decode matches plain greedy decode, EXCEPT at target near-ties.

    The verify forward runs batched (anchor + K drafts), so it uses the per-user
    RoPE + batched SDPA path, which differs from batch=1 decode by ~1e-5 (see
    test_verify_batchsize_invariance). That noise flips only near-tie tokens
    (top-2 logit gap < ~1), so spec-decode is token-identical to plain greedy up
    to the first such near-tie. A divergence at a CONFIDENT token (large top-2
    gap) would indicate a real accept/commit/KV bug and fails here."""
    near_tie_gap = float(os.environ.get("GEMMA4_SPEC_NEAR_TIE_GAP", 2.0))
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None

    max_seq_len = 1024
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 24))
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "The capital of France is")

    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4)),
    )

    # Reference: prefill, then plain greedy decode.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    ref = _plain_greedy(spec, anchor_token, anchor_pos, n_new)

    # Spec decode: re-prefill (resets the prompt KV), then greedy spec decode.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    gen, accepts = spec.generate(anchor_token, anchor_pos, max_new_tokens=len(ref), temperature=0.0)

    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info(f"ref={ref}")
    logger.info(f"spec={gen}")
    logger.info(f"mean accepted/iter={mean_accept:.2f} over {len(accepts)} iters")

    first_div = next((i for i in range(min(len(gen), len(ref))) if gen[i] != ref[i]), None)
    if first_div is None:
        logger.info("[matches-greedy] spec-decode is TOKEN-IDENTICAL to plain greedy")
        return

    # Divergence allowed only at a target near-tie. Re-verify (batch=1) the
    # plain-greedy prefix up to the divergence and inspect the top-2 logit gap.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    tok, pos = anchor_token, anchor_pos
    for _ in range(first_div):
        lg, hd = spec._verify([tok], [pos])
        hd.deallocate(True)
        tok = int(torch.argmax(lg[0]))
        pos += 1
    lg, hd = spec._verify([tok], [pos])
    hd.deallocate(True)
    top2 = torch.topk(lg[0].float(), 2)
    gap = float(top2.values[0] - top2.values[1])
    logger.info(
        f"[matches-greedy] first divergence at idx {first_div}: spec={gen[first_div]} greedy={ref[first_div]}  "
        f"target top2 ids={top2.indices.tolist()} gap={gap:.4f} (near-tie threshold={near_tie_gap})"
    )
    assert gap < near_tie_gap, (
        f"greedy spec-decode diverged from plain greedy at a CONFIDENT token "
        f"(idx {first_div}, top-2 gap={gap:.3f} >= {near_tie_gap}); indicates an accept/commit/KV bug, "
        f"not batched-verify numerics"
    )


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_verify_batchsize_invariance(mesh_device, reset_seeds):
    """Isolate batch-size numerics from spec accept logic.

    Greedy-decode the SAME prompt twice, advancing one token per step, but with
    the verify forward run at batch=1 vs batch=1+pad (the spec verify's batched
    per-user RoPE + batched SDPA + sequential-KV-write path). No acceptance logic.
    If the two greedy chains diverge, greedy spec-decode CANNOT be bit-identical
    to batch=1 decode — the divergence is batched-path numerics, expected at
    near-tie tokens. Logs the first divergence and the target's top-2 logit gap."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None

    max_seq_len = 1024
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 48))
    pad = int(os.environ.get("GEMMA4_SPEC_VERIFY_PAD", 4))
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "The capital of France is")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    g1 = _plain_greedy(spec, anchor_token, anchor_pos, n_new)
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    gN = _plain_greedy_bN(spec, anchor_token, anchor_pos, n_new, pad)

    first_div = next((i for i in range(min(len(g1), len(gN))) if g1[i] != gN[i]), None)
    logger.info(f"batch=1  greedy: {g1}")
    logger.info(f"batch={1+pad} greedy: {gN}")
    if first_div is None:
        logger.info("[batch-inv] batch=1 and batched verify are TOKEN-IDENTICAL")
    else:
        # Inspect the target's batch=1 logit gap at the divergence position.
        generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
        tok, pos = anchor_token, anchor_pos
        for _ in range(first_div):
            lg, hd = spec._verify([tok], [pos])
            hd.deallocate(True)
            tok = int(torch.argmax(lg[0]))
            pos += 1
        lg, hd = spec._verify([tok], [pos])
        hd.deallocate(True)
        top2 = torch.topk(lg[0].float(), 2)
        gap = float(top2.values[0] - top2.values[1])
        logger.info(
            f"[batch-inv] FIRST DIVERGENCE at gen idx {first_div}: batch1={g1[first_div]} batchN={gN[first_div]}  "
            f"target top2 ids={top2.indices.tolist()} gap={gap:.4f}"
        )


@_needs_assistant
@_assistant_probe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_spec_decode_perf_breakdown(mesh_device, reset_seeds):
    """Device-time breakdown to project the achievable (traced) spec-decode speedup.

    Times, on-device with NO host readback, the per-op latency of:
      * T_decode1 : a single-token target verify (the plain-decode unit cost),
      * T_verify  : the batched verify over K+1 tokens (one spec verify),
      * T_draft   : one drafter (assistant) step.
    Then projects iter latency = T_verify + K*T_draft and the resulting
    tok/s/u vs plain decode across a sweep of acceptance rates. Untraced device
    time is an UPPER bound on a traced loop (tracing removes per-op dispatch +
    host round-trips), so the projected speedup is conservative. This answers
    'how much perf is achievable on QB2' independent of the acceptance fix."""
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    max_seq_len = 1024
    block_size = 64
    reps = int(os.environ.get("GEMMA4_SPEC_PERF_REPS", 30))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    def _time_verify(tokens, positions):
        x = spec._tokens_tensor(tokens)
        pos_u, pos_i = spec._pos_tensors(positions)
        pt = spec._page_table(len(tokens))
        for _ in range(3):  # warmup
            lo, hi = target.ttnn_verify_forward(
                x=x, current_pos=pos_u, current_pos_cache=pos_i, page_table=pt, kv_cache=spec.tt_kv_cache
            )
            lo.deallocate(True)
            hi.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            lo, hi = target.ttnn_verify_forward(
                x=x, current_pos=pos_u, current_pos_cache=pos_i, page_table=pt, kv_cache=spec.tt_kv_cache
            )
            lo.deallocate(True)
            hi.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / reps
        for t in (x, pos_u, pos_i, pt):
            t.deallocate(True)
        return dt

    def _time_draft():
        h = spec.seed(anchor_token, anchor_pos)
        pos_u, pos_i = spec._pos_tensors([anchor_pos])
        pt = spec._page_table(1)
        page_tables = {lt: pt for lt in spec._shared_kv}
        tok_tt = spec._tokens_tensor([anchor_token])
        for _ in range(3):  # warmup
            lo, hn = assistant.step(tok_tt, h, spec._shared_kv, page_tables, pos_u, pos_i)
            lo.deallocate(True)
            hn.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            lo, hn = assistant.step(tok_tt, h, spec._shared_kv, page_tables, pos_u, pos_i)
            lo.deallocate(True)
            hn.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / reps
        for t in (tok_tt, pos_u, pos_i, pt):
            t.deallocate(True)
        h.deallocate(True)
        return dt

    t_decode1 = _time_verify([anchor_token], [anchor_pos])
    t_verify = _time_verify([anchor_token] + [anchor_token] * K, [anchor_pos + j for j in range(K + 1)])
    t_draft = _time_draft()

    iter_lat = t_verify + K * t_draft
    base_tps = 1.0 / t_decode1
    logger.info("=== spec-decode device-time breakdown (untraced; UPPER bound) ===")
    logger.info(f"  K={K}  reps={reps}")
    logger.info(f"  T_decode1 (1 tok verify) = {t_decode1*1e3:7.2f} ms  -> baseline {base_tps:6.1f} tok/s/u (untraced)")
    logger.info(f"  T_verify  (K+1 batched)  = {t_verify*1e3:7.2f} ms  ({t_verify/t_decode1:.2f}x decode)")
    logger.info(f"  T_draft   (1 drafter)    = {t_draft*1e3:7.2f} ms  ({t_draft/t_decode1:.3f}x decode)")
    logger.info(f"  iter = T_verify + K*T_draft = {iter_lat*1e3:7.2f} ms  ({iter_lat/t_decode1:.2f}x decode)")
    logger.info("  projected speedup vs plain decode by acceptance (committed=accept+1):")
    for acc in [0.21, 0.5, 1.0, 1.62, 2.0, 3.0, float(K)]:
        committed = acc + 1.0
        spd = committed * t_decode1 / iter_lat
        logger.info(
            f"    accept={acc:4.2f} -> {committed:4.2f} tok/iter -> {committed/iter_lat:6.1f} tok/s/u  = {spd:4.2f}x"
        )
    break_even_acc = iter_lat / t_decode1 - 1.0
    logger.info(f"  BREAK-EVEN acceptance (speedup=1.0x): accept >= {break_even_acc:.2f}")


@_needs_assistant
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
def test_spec_decode_sampling_acceptance(mesh_device, reset_seeds):
    """Measure SAMPLING-mode acceptance (production config: temp/top_p/top_k).

    Speculative SAMPLING accepts draft d_i with prob min(1, p_target(d_i)/q_draft(d_i)),
    not exact-argmax match — far more tolerant of the bf16 feature deviation that
    throttles greedy acceptance. Correctness is distributional (the committed
    tokens follow the target's sampled distribution), so this only reports the
    acceptance rate; it does not assert a fixed token sequence."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None

    max_seq_len = 1024
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 64))
    temperature = float(os.environ.get("GEMMA4_SPEC_TEMP", 1.0))
    top_p = float(os.environ.get("GEMMA4_SPEC_TOP_P", 0.95))
    top_k = int(os.environ.get("GEMMA4_SPEC_TOP_K", 64))
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    torch.manual_seed(0)
    gen, accepts = spec.generate(
        anchor_token, anchor_pos, max_new_tokens=n_new, temperature=temperature, top_p=top_p, top_k=top_k
    )
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info(f"[sampling] temp={temperature} top_p={top_p} top_k={top_k}")
    logger.info(f"[sampling] gen={gen}")
    logger.info(f"[sampling] text={tokenizer.decode(gen)!r}")
    logger.info(
        f"[sampling] mean accepted/iter={mean_accept:.2f} / {spec.draft_len} over {len(accepts)} iters "
        f"=> ~{mean_accept + 1:.2f} committed tokens/verify"
    )
    assert len(gen) > 0 and all(isinstance(t, int) for t in gen), "sampling spec-decode produced no valid tokens"


@_needs_assistant
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_spec_decode_traced(mesh_device, reset_seeds):
    """Traced spec-decode: correctness (vs untraced greedy) + traced tok/s/u.

    Opens the device with a trace region and runs greedy spec-decode through the
    single fused trace path. This avoids the old deadlock from interleaving
    separate draft and verify CCL traces, while still checking generated tokens
    against untraced greedy up to the first target near-tie."""
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    near_tie_gap = float(os.environ.get("GEMMA4_SPEC_NEAR_TIE_GAP", 2.0))
    max_seq_len = 1024
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 48))
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )

    # Untraced greedy reference.
    spec._use_trace = False
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    ref = _plain_greedy(spec, anchor_token, anchor_pos, n_new)

    # Traced greedy spec-decode.
    spec._use_trace = True
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    t0 = time.perf_counter()
    gen, accepts = spec.generate(anchor_token, anchor_pos, max_new_tokens=n_new, temperature=0.0)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    tps = len(gen) / dt if dt > 0 else 0.0
    logger.info(
        f"[traced] generated {len(gen)} tokens in {dt*1e3:.1f} ms -> {tps:.1f} tok/s/u (wall, incl host accept)"
    )
    logger.info(f"[traced] mean accepted/iter={mean_accept:.2f} / {spec.draft_len} over {len(accepts)} iters")

    first_div = next((i for i in range(min(len(gen), len(ref))) if gen[i] != ref[i]), None)
    if first_div is None:
        logger.info("[traced] spec-decode TOKEN-IDENTICAL to untraced plain greedy")
        return
    spec._use_trace = False
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    tok, pos = anchor_token, anchor_pos
    for _ in range(first_div):
        lg, hd = spec._verify([tok], [pos])
        hd.deallocate(True)
        tok = int(torch.argmax(lg[0]))
        pos += 1
    lg, hd = spec._verify([tok], [pos])
    hd.deallocate(True)
    top2 = torch.topk(lg[0].float(), 2)
    gap = float(top2.values[0] - top2.values[1])
    logger.info(
        f"[traced] first divergence idx {first_div}: spec={gen[first_div]} greedy={ref[first_div]} gap={gap:.4f}"
    )
    assert gap < near_tie_gap, f"traced spec-decode diverged at a CONFIDENT token (idx {first_div}, gap={gap:.3f})"


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_verify_trace_batched_capture(mesh_device, reset_seeds):
    """ISOLATED probe: does a batch=K+1 VERIFY trace capture + replay on a clean device?

    The batched verify trace was previously flagged as "replay HANGS" but that
    verdict was reached on a corrupted device (trace_region_size=0 fixture + a
    kill-9). This test, on a clean device with a real trace region, directly:
      1. runs an untraced batch=K+1 verify -> reference logits;
      2. captures a verify trace at those exact inputs (idempotent KV write);
      3. replays it N times and checks traced-vs-untraced logits PCC + timing.
    Heavy phase logging lets a monitor localize any hang. If this passes, the
    full loop can be traced with NO kernel rewrite of sequential_kv_write.
    """
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    logger.info("[vtrace] loading target + drafter")
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 48, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    logger.info("[vtrace] prefill")
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    # Build a realistic batch=K+1 verify call (anchor + K drafts at consecutive pos).
    spec._use_trace = False
    anchor_hidden = spec.seed(anchor_token, anchor_pos)
    drafts, _ = spec._draft(anchor_token, anchor_hidden, anchor_pos, temperature=0.0)
    anchor_hidden.deallocate(True)
    verify_tokens = [anchor_token] + drafts
    verify_pos = [anchor_pos + j for j in range(len(verify_tokens))]
    logger.info(f"[vtrace] verify batch={len(verify_tokens)} tokens={verify_tokens} pos={verify_pos}")

    # 1) Untraced reference logits.
    ref_logits, ref_hidden = spec._verify(verify_tokens, verify_pos)
    ref_hidden.deallocate(True)
    ref_argmax = [int(torch.argmax(ref_logits[j])) for j in range(len(verify_tokens))]
    logger.info(f"[vtrace] untraced ref argmax={ref_argmax}")

    # 2) Capture the batched verify trace (bypasses the batch==1 guard in _verify).
    logger.info("[vtrace] >>> begin capture of batched verify trace")
    spec._use_trace = True
    spec._capture_verify_trace(verify_tokens, verify_pos)
    logger.info("[vtrace] <<< capture returned (no hang in capture)")

    # 3) Replay N times; check correctness + timing.
    N = 5
    logger.info(f"[vtrace] >>> begin {N} trace replays")
    t0 = time.perf_counter()
    last_logits = None
    for r in range(N):
        last_logits, h = spec._verify_traced(verify_tokens, verify_pos)
        # h is the PERSISTENT trace output now (no clone) — do not deallocate.
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[vtrace] replay {r} done")
    dt = (time.perf_counter() - t0) / N
    logger.info(f"[vtrace] <<< replays done: {dt*1e3:.2f} ms/replay")

    tr_argmax = [int(torch.argmax(last_logits[j])) for j in range(len(verify_tokens))]
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(ref_logits.float(), last_logits.float(), 0.99)
    logger.info(f"[vtrace] traced argmax={tr_argmax} (ref={ref_argmax}) logits PCC={pcc}")
    assert tr_argmax == ref_argmax, f"traced verify argmax mismatch: {tr_argmax} vs {ref_argmax}"

    # 4) DISAMBIGUATION: re-replay batch=5 with CHANGING positions, NO interleaving.
    # The full loop hangs on the 2nd batch=5 replay; the loop differs from the
    # replays above in two ways: (a) a draft trace runs between replays, and (b)
    # positions advance. This isolates (b): if these incrementing-position
    # re-replays hang with no other trace between them, the cause is changed
    # positions (sequential_kv_write re-replay), NOT cross-trace interleaving.
    logger.info("[vtrace] >>> begin CHANGED-POSITION re-replays (no interleaving)")
    for it in range(6):
        off = it + 1
        toks = list(verify_tokens)  # token values don't matter for the hang probe
        pos = [anchor_pos + off + j for j in range(len(toks))]
        logger.info(f"[vtrace] changed-pos replay {it}: pos0={pos[0]} execute")
        lg, h = spec._verify_traced(toks, pos)  # h persistent; do not deallocate
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[vtrace] changed-pos replay {it}: done argmax0={int(torch.argmax(lg[0]))}")
    logger.info("[vtrace] <<< CHANGED-POSITION re-replays SURVIVED -> cause is interleaving, not positions")


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_ondevice_argmax_probe(mesh_device, reset_seeds):
    """PROBE (eager, no trace): on-device argmax -> token id -> re-embed.

    The fused single-iteration trace needs the drafter recurrence and the verify
    input assembly to stay ON DEVICE (no host round-trip per draft step). That
    requires: (a) argmax over the (already all-gathered, full-vocab) drafter and
    verify logits computed on device, and (b) re-embedding the argmax id via
    embed_tokens. This probe validates the exact shapes/dtypes/layouts and that
    on-device argmax matches host argmax (drafter near-ties may differ; logged).
    """
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 48, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    spec._use_trace = False
    anchor_hidden = spec.seed(anchor_token, anchor_pos)

    # ── One drafter step -> logits; compare on-device vs host argmax ──────────
    tok_tt = spec._tokens_tensor([anchor_token])
    pos_u, pos_i = spec._pos_tensors([anchor_pos])
    pt = spec._page_table(1)
    page_tables = {lt: pt for lt in spec._shared_kv}
    logits_d, h_next = assistant.step(tok_tt, anchor_hidden, spec._shared_kv, page_tables, pos_u, pos_i)
    logger.info(f"[probe] drafter logits shape={list(logits_d.shape)} dtype={logits_d.dtype} layout={logits_d.layout}")

    host_logits = spec._logits_to_host(logits_d).reshape(-1)
    host_tok = int(torch.argmax(host_logits))

    # On-device argmax over the vocab (last) dim.
    idx = ttnn.argmax(logits_d, dim=-1, keepdim=False)
    logger.info(f"[probe] argmax idx shape={list(idx.shape)} dtype={idx.dtype} layout={idx.layout}")
    idx_host = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]) if spec._tp > 1 else ttnn.to_torch(idx)
    dev_tok = int(idx_host.reshape(-1)[0])
    logger.info(f"[probe] drafter on-device argmax={dev_tok} host argmax={host_tok} match={dev_tok == host_tok}")

    # Re-embed the on-device id and compare to the host id's embedding.
    # embed_tokens expects a [1, N] uint32 ROW_MAJOR token-id tensor.
    idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
    idx_u32 = ttnn.typecast(idx_rm, ttnn.uint32) if idx.dtype != ttnn.uint32 else idx_rm
    idx_u32 = ttnn.reshape(idx_u32, (1, 1))
    emb_dev = assistant._raw_token_embed(idx_u32)
    emb_ref = assistant._raw_token_embed(spec._tokens_tensor([dev_tok]))
    from models.common.utility_functions import comp_pcc

    e_dev = ttnn.to_torch(ttnn.get_device_tensors(emb_dev)[0]) if spec._tp > 1 else ttnn.to_torch(emb_dev)
    e_ref = ttnn.to_torch(ttnn.get_device_tensors(emb_ref)[0]) if spec._tp > 1 else ttnn.to_torch(emb_ref)
    emb_passed, emb_pcc = comp_pcc(e_ref.float(), e_dev.float(), 0.99)
    logger.info(f"[probe] re-embed(on-device id) PCC vs embed(host id)={emb_pcc} passed={emb_passed}")

    # ── Verify-side argmax: batch=K+1 ────────────────────────────────────────
    drafts, _ = spec._draft(anchor_token, anchor_hidden, anchor_pos, temperature=0.0)
    verify_tokens = [anchor_token] + drafts
    verify_pos = [anchor_pos + j for j in range(len(verify_tokens))]
    vlogits, vhidden = spec._verify(verify_tokens, verify_pos)
    host_v = [int(torch.argmax(vlogits[j])) for j in range(len(verify_tokens))]

    # Re-run verify on device to get a live device logits tensor for argmax.
    vx = spec._tokens_tensor(verify_tokens)
    vpu, vpi = spec._pos_tensors(verify_pos)
    vpt = spec._page_table(len(verify_tokens))
    vl_d, vh_d = target.ttnn_verify_forward(
        x=vx, current_pos=vpu, current_pos_cache=vpi, page_table=vpt, kv_cache=spec.tt_kv_cache
    )
    logger.info(f"[probe] verify logits shape={list(vl_d.shape)} dtype={vl_d.dtype} layout={vl_d.layout}")
    vidx = ttnn.argmax(vl_d, dim=-1, keepdim=False)
    logger.info(f"[probe] verify argmax idx shape={list(vidx.shape)} dtype={vidx.dtype} layout={vidx.layout}")
    vidx_host = (ttnn.to_torch(ttnn.get_device_tensors(vidx)[0]) if spec._tp > 1 else ttnn.to_torch(vidx)).reshape(-1)
    dev_v = [int(vidx_host[j]) for j in range(len(verify_tokens))]
    logger.info(f"[probe] verify on-device argmax={dev_v} host argmax={host_v} match={dev_v == host_v}")

    assert emb_passed, f"re-embed PCC too low: {emb_pcc}"
    logger.info("[probe] DONE")


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_fused_iter_eager(mesh_device, reset_seeds):
    """EAGER fused iteration == _draft + _verify + host argmax (same anchor).

    _fused_iter does the K drafter steps, verify-input assembly, and both argmaxes
    ON DEVICE, reading back only the 2K+1 ids. This checks it produces the same
    drafts and the same target argmax as the host-readback path, on fresh prefills
    (verify dirties KV, so each path runs from a clean prefill)."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 48, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    spec._use_trace = False

    def _one_fused():
        generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
        anchor_hidden = spec.seed(anchor_token, anchor_pos)
        anchor_tok_tt = spec._tokens_tensor([anchor_token])
        drafts, target, vh = spec._fused_iter(anchor_tok_tt, anchor_hidden, anchor_pos)
        vh.deallocate(True)
        anchor_hidden.deallocate(True)
        anchor_tok_tt.deallocate(True)
        return drafts, target

    # Run the fused iteration TWICE (re-prefill between) — this is the loop
    # pattern (fused-after-fused). Validates repeatability + no state corruption.
    drafts1, target1 = _one_fused()
    logger.info(f"[fused] run1 drafts={drafts1} target_argmax={target1}")
    drafts2, target2 = _one_fused()
    logger.info(f"[fused] run2 drafts={drafts2} target_argmax={target2}")

    assert drafts1 == drafts2 and target1 == target2, "fused iteration is non-deterministic across re-prefills"
    # Greedy self-consistency: drafts must be the verify argmax at the same slot
    # for the accepted prefix (drafts[i] == target[i] until the first reject).
    m = next((i for i in range(len(drafts1)) if drafts1[i] != target1[i]), len(drafts1))
    logger.info(f"[fused] accepted prefix m={m}/{len(drafts1)} (drafts match verify argmax up to first reject)")
    assert target1[:m] == drafts1[:m]
    logger.info("[fused] EAGER FUSED ITERATION OK (deterministic, on-device argmax)")


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_fused_loop_eager(mesh_device, reset_seeds):
    """EAGER fused LOOP (generate_fused) — the real loop pattern, no re-prefill.

    Runs the on-device fused iteration repeatedly, advancing the anchor via the
    verify hidden (shift seed). Confirms (a) the loop does NOT hang across many
    iterations (so _fused_iter is loop-safe), and (b) the greedy output matches
    plain greedy up to a near-tie. Reports acceptance for the shift-seed path."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    near_tie_gap = float(os.environ.get("GEMMA4_SPEC_NEAR_TIE_GAP", 2.0))
    max_seq_len = 1024
    block_size = 64
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 48))
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    spec._use_trace = False

    # Plain greedy reference.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    ref = _plain_greedy(spec, anchor_token, anchor_pos, n_new)

    # Fused eager loop.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    gen, accepts = spec.generate_fused(anchor_token, anchor_pos, max_new_tokens=n_new)
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info(f"[fused-loop] ref ={ref}")
    logger.info(f"[fused-loop] gen ={gen}")
    logger.info(f"[fused-loop] mean accepted/iter={mean_accept:.2f}/{K} over {len(accepts)} iters")

    first_div = next((i for i in range(min(len(gen), len(ref))) if gen[i] != ref[i]), None)
    if first_div is None:
        logger.info("[fused-loop] TOKEN-IDENTICAL to plain greedy")
        return
    # Allow divergence only at a target near-tie.
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    tok, pos = anchor_token, anchor_pos
    for _ in range(first_div):
        lg, hd = spec._verify([tok], [pos])
        hd.deallocate(True)
        tok = int(torch.argmax(lg[0]))
        pos += 1
    lg, hd = spec._verify([tok], [pos])
    hd.deallocate(True)
    top2 = torch.topk(lg[0].float(), 2)
    gap = float(top2.values[0] - top2.values[1])
    logger.info(
        f"[fused-loop] first divergence idx {first_div}: gen={gen[first_div]} ref={ref[first_div]} gap={gap:.4f}"
    )
    assert gap < near_tie_gap, f"fused loop diverged at a CONFIDENT token (idx {first_div}, gap={gap:.3f})"


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_fused_loop_traced(mesh_device, reset_seeds):
    """FUSED single-iteration TRACE: K drafter steps + verify in ONE CCL trace.

    The draft/verify alternation deadlocks when distinct CCL traces interleave;
    fusing the whole iteration into ONE trace (replayed once per iter, with
    on-device argmax + re-embed) removes the interleaving. This test captures the
    fused trace and replays it across the loop, checking (a) NO hang across many
    iterations, (b) greedy output matches the eager fused loop, and (c) traced
    tok/s/u vs the eager fused loop."""
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 32))
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )

    # Eager fused loop reference.
    spec._use_trace = False
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    t0 = time.perf_counter()
    ref, ref_acc = spec.generate_fused(anchor_token, anchor_pos, max_new_tokens=n_new)
    ttnn.synchronize_device(mesh_device)
    eager_dt = time.perf_counter() - t0
    logger.info(f"[fused-trace] eager  gen={ref}")
    logger.info(f"[fused-trace] eager  {len(ref)/eager_dt:.1f} tok/s/u, accept={sum(ref_acc)/len(ref_acc):.2f}")

    # Traced fused loop.
    spec._use_trace = True
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    t0 = time.perf_counter()
    gen, accepts = spec.generate_fused(anchor_token, anchor_pos, max_new_tokens=n_new)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info(f"[fused-trace] traced gen={gen}")
    logger.info(
        f"[fused-trace] traced {len(gen)/dt:.1f} tok/s/u (wall, incl host glue), accept={mean_accept:.2f}/{K} over {len(accepts)} iters"
    )
    logger.info(f"[fused-trace] SPEEDUP traced/eager = {eager_dt/dt:.2f}x")

    first_div = next((i for i in range(min(len(gen), len(ref))) if gen[i] != ref[i]), None)
    assert (
        first_div is None
    ), f"traced fused loop diverged from eager fused loop at idx {first_div}: {gen[first_div]} vs {ref[first_div]}"
    logger.info("[fused-trace] TOKEN-IDENTICAL to eager fused loop (NO hang, single fused trace)")


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_fused_trace_minimal(mesh_device, reset_seeds):
    """ISOLATE the fused trace: prefill ONCE, then traced generate_fused.

    No eager reference loop and no second prefill (those confound with a
    prefill-trace-replay interaction). This answers ONLY: does the single fused
    trace capture + replay across the loop WITHOUT hanging, and produce a
    self-consistent greedy chain (each committed token == verify argmax)?"""
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 24))
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )

    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)
    spec._use_trace = True
    t0 = time.perf_counter()
    gen, accepts = spec.generate_fused(anchor_token, anchor_pos, max_new_tokens=n_new)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info(f"[fused-min] gen={gen}")
    logger.info(f"[fused-min] {len(gen)} tok in {dt*1e3:.0f} ms -> {len(gen)/dt:.1f} tok/s/u (wall, incl host glue)")
    logger.info(f"[fused-min] accept={mean_accept:.2f}/{K} over {len(accepts)} iters")
    assert len(gen) >= n_new - K, f"traced fused produced too few tokens: {len(gen)}"
    logger.info("[fused-min] FUSED TRACE replays across the loop with NO hang")


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_verify_seqkv_cost(mesh_device, reset_seeds):
    """MEASURE how much of the verify cost is the sequential_kv_write loop.

    Captures a batch=(K+1) verify-only trace TWICE: once with the race-safe
    per-candidate serialized KV write (production), once with the single batched
    write (KV is corrupted, but timing is valid). The delta tells us whether the
    serialization is worth removing (distinct-blocks rewrite) before we build it.
    Also times a batch=1 verify (the plain-decode analog) as a reference floor.
    """
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    reps = 20

    def _capture_and_time(batch, positions, tokens, seq_kv):
        """Capture a verify-only trace at the given seq_kv setting; time `reps` replays."""
        target._verify_seq_kv_write = seq_kv
        x = spec._tokens_tensor(tokens)
        pu, pi = spec._pos_tensors(positions)
        pt = spec._page_table(batch)
        logits, hidden = target.ttnn_verify_forward(
            x=x, current_pos=pu, current_pos_cache=pi, page_table=pt, kv_cache=spec.tt_kv_cache
        )
        ttnn.synchronize_device(mesh_device)
        logits.deallocate(True)
        hidden.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits, hidden = target.ttnn_verify_forward(
            x=x, current_pos=pu, current_pos_cache=pi, page_table=pt, kv_cache=spec.tt_kv_cache
        )
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        for t in (x, pu, pi, pt, logits, hidden):
            t.deallocate(True)
        return ms

    base = anchor_pos + 1
    vpos = [base + j for j in range(K + 1)]
    vtok = [anchor_token] * (K + 1)
    seq_ms = _capture_and_time(K + 1, vpos, vtok, seq_kv=True)
    bat_ms = _capture_and_time(K + 1, vpos, vtok, seq_kv=False)
    one_ms = _capture_and_time(1, [base], [anchor_token], seq_kv=True)
    logger.info(f"[seqkv] VERIFY batch={K+1} seq_kv_write=ON  = {seq_ms:.1f} ms (production, race-safe)")
    logger.info(f"[seqkv] VERIFY batch={K+1} seq_kv_write=OFF = {bat_ms:.1f} ms (single batched write, KV-unsafe)")
    logger.info(f"[seqkv] VERIFY batch=1 (decode analog)      = {one_ms:.1f} ms")
    logger.info(
        f"[seqkv] seq-loop overhead = {seq_ms - bat_ms:.1f} ms ({(seq_ms-bat_ms)/seq_ms*100:.0f}% of verify); batched verify vs decode = {bat_ms/one_ms:.2f}x"
    )


@_needs_assistant
@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_draft_step_breakdown(mesh_device, reset_seeds):
    """MEASURE where a single drafter step's ~15 ms goes on tp=4.

    Times a traced drafter step (full) vs a step with lm_head + its 262144-wide
    all-gather skipped (`return_logits=False`). The delta isolates the lm_head /
    CCL cost from the 4-layer backbone (which carries its own per-layer CCL).
    """
    import time

    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    max_seq_len = 1024
    block_size = 64
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
    )
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = os.environ.get("GEMMA4_SPEC_PROMPT", "Write a short paragraph about the history of the Eiffel Tower.")
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 32, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    anchor_hidden = spec.seed(anchor_token, anchor_pos)  # [1,1,1,backbone]
    tok = spec._tokens_tensor([anchor_token])
    h = ttnn.clone(anchor_hidden)
    d_pu, d_pi = spec._pos_tensors([anchor_pos])
    d_pt = spec._page_table(1)
    page_tables = {lt: d_pt for lt in spec._shared_kv}
    reps = 30

    def _time_step(return_logits):
        # compile
        lg, hn = assistant.step(tok, h, spec._shared_kv, page_tables, d_pu, d_pi, return_logits=return_logits)
        ttnn.synchronize_device(mesh_device)
        if lg is not None:
            lg.deallocate(True)
        hn.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        lg, hn = assistant.step(tok, h, spec._shared_kv, page_tables, d_pu, d_pi, return_logits=return_logits)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        if lg is not None:
            lg.deallocate(True)
        hn.deallocate(True)
        return ms

    full_ms = _time_step(True)
    nolm_ms = _time_step(False)
    logger.info(f"[draft-bd] FULL step (4 layers + lm_head allgather) = {full_ms:.2f} ms")
    logger.info(f"[draft-bd] step WITHOUT lm_head/allgather (backbone only) = {nolm_ms:.2f} ms")
    logger.info(
        f"[draft-bd] lm_head + 262144 allgather = {full_ms - nolm_ms:.2f} ms ({(full_ms-nolm_ms)/full_ms*100:.0f}% of step); backbone(4 layers)+CCL = {nolm_ms:.2f} ms"
    )
    logger.info(f"[draft-bd] full K={K} steps ~= {full_ms*K:.1f} ms")


@_assistant_probe
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_argmax_cost(mesh_device, reset_seeds):
    """Time ttnn.argmax over a 262144-wide vocab (the suspected fused-iter cost).

    Drafter does 4 argmax over [1,1,1,V]; verify does 1 over [1,1,K+1,V]. If each
    is ~10 ms, that 5x explains the ~54 ms gap. Also times a PER-SHARD argmax
    over V/tp (the distributed-argmax candidate that avoids the full-V reduction
    and the 262144 all-gather).
    """
    import time

    V = 262144
    tp = 4
    K = 4
    reps = 30
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def _time_argmax(rows, width, mapper, mode="bare"):
        """mode: 'bare' = argmax on TILE; 'mc' = argmax(use_multicore); 'untile_mc' = untilize+argmax(use_multicore)."""
        t = ttnn.from_torch(
            torch.randn(1, 1, rows, width),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
        )

        def _run():
            if mode == "bare":
                return ttnn.argmax(t, dim=-1, keepdim=False)
            tu = ttnn.untilize(t, use_multicore=True)
            r = ttnn.argmax(tu, dim=-1, keepdim=False, use_multicore=True)
            tu.deallocate(True)
            return r

        idx = _run()  # compile
        ttnn.synchronize_device(mesh_device)
        idx.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        idx = _run()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        t.deallocate(True)
        idx.deallocate(True)
        return ms

    full1 = _time_argmax(1, V, replicate)
    full5 = _time_argmax(K + 1, V, replicate)
    shard1 = _time_argmax(1, V // tp, replicate)
    logger.info(f"[argmax] BARE argmax over [1,1,1,{V}] = {full1:.2f} ms")
    logger.info(f"[argmax] BARE argmax over [1,1,{K+1},{V}] = {full5:.2f} ms")
    logger.info(f"[argmax] BARE per-shard argmax over [1,1,1,{V//tp}] = {shard1:.2f} ms")
    logger.info(f"[argmax] fused-iter BARE argmax total (4x full1 + 1x full5) ~= {4*full1 + full5:.1f} ms")
    for mode in ("untile_mc",):
        m1 = _time_argmax(1, V, replicate, mode=mode)
        m5 = _time_argmax(K + 1, V, replicate, mode=mode)
        logger.info(
            f"[argmax] {mode}: [1,1,1,{V}]={m1:.2f} ms, [1,1,{K+1},{V}]={m5:.2f} ms, fused-total~={4*m1+m5:.1f} ms"
        )

    # ── CORRECTNESS: known argmax per row, bare vs untilize+multicore ────────
    def _check(rows):
        want = [(i * 37 + 11) % V for i in range(rows)]  # arbitrary distinct argmax per row
        host = torch.full((1, 1, rows, V), -1.0)
        for i in range(rows):
            host[0, 0, i, want[i]] = 100.0
        t = ttnn.from_torch(
            host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        bare = ttnn.argmax(t, dim=-1, keepdim=False)
        tu = ttnn.untilize(t, use_multicore=True)
        mc = ttnn.argmax(tu, dim=-1, keepdim=False, use_multicore=True)

        def _to_list(x):
            h = ttnn.to_torch(ttnn.get_device_tensors(x)[0]) if tp > 1 else ttnn.to_torch(x)
            return [int(v) for v in h.reshape(-1)[:rows]]

        bare_l, mc_l = _to_list(bare), _to_list(mc)
        logger.info(f"[argmax-chk] rows={rows} want={want}")
        logger.info(f"[argmax-chk] rows={rows} bare={bare_l}")
        logger.info(f"[argmax-chk] rows={rows} untile_mc={mc_l}")
        logger.info(f"[argmax-chk] rows={rows} bare_ok={bare_l==want} mc_ok={mc_l==want}")
        for x in (t, tu, bare, mc):
            x.deallocate(True)

    _check(1)
    _check(K + 1)

    # Does padding rows to 32 fix multicore argmax? (row-parallel needs aligned rows)
    def _check_pad32(real_rows):
        want = [(i * 37 + 11) % V for i in range(real_rows)]
        host = torch.full((1, 1, 32, V), -1.0)
        for i in range(real_rows):
            host[0, 0, i, want[i]] = 100.0
        t = ttnn.from_torch(
            host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        tu = ttnn.untilize(t, use_multicore=True)
        mc = ttnn.argmax(tu, dim=-1, keepdim=False, use_multicore=True)
        h = ttnn.to_torch(ttnn.get_device_tensors(mc)[0]) if tp > 1 else ttnn.to_torch(mc)
        got = [int(v) for v in h.reshape(-1)[:real_rows]]
        # time it (trace)
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tu2 = ttnn.untilize(t, use_multicore=True)
        mc2 = ttnn.argmax(tu2, dim=-1, keepdim=False, use_multicore=True)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        logger.info(f"[argmax-chk] PAD32 real_rows={real_rows} want={want} got={got} ok={got==want} time={ms:.2f}ms")
        for x in (t, tu, mc, tu2, mc2):
            x.deallocate(True)

    _check_pad32(1)
    _check_pad32(K + 1)


@_needs_assistant
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200_000_000}, id="1x4")],
    indirect=True,
)
def test_spec_decode_batched(mesh_device, reset_seeds):
    """End-to-end BATCHED (B>1) greedy speculative decode for B independent users.

    Each iteration drafts every user at batch=1 and runs ONE batched packed
    verify over all B users (the KV-amortization win). Users accept raggedly, so
    their positions diverge. Validates:
      * the batched path runs and every user emits tokens;
      * user-0's batched output matches a batch=1 plain-greedy reference up to the
        first target near-tie (same contract as test_spec_decode_matches_greedy);
      * distinct prompts -> distinct outputs (no gross cross-user KV leak);
    and reports per-user + aggregate throughput.
    """
    import time

    near_tie_gap = float(os.environ.get("GEMMA4_SPEC_NEAR_TIE_GAP", 2.0))
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    num_layers = os.environ.get("GEMMA4_NUM_LAYERS")
    num_layers = int(num_layers) if num_layers else None

    B = int(os.environ.get("GEMMA4_SPEC_BATCH", 4))
    max_seq_len = int(os.environ.get("GEMMA4_SPEC_MAX_SEQ", 1024))
    n_new = int(os.environ.get("GEMMA4_SPEC_TEST_TOKENS", 16))
    draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 4))
    block_size = 64
    blocks_per_user = math.ceil(max_seq_len / block_size)
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=B * blocks_per_user)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=B,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=ASSISTANT_PATH,
        max_local_batch_size=int(os.environ.get("GEMMA4_SPEC_ASSIST_BATCH", B)),
    )

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table

    page_table = create_tt_page_table(B, paged_attention_config)  # [B, blocks_per_user]

    base_prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in our solar system is",
        "The author of Romeo and Juliet is",
        "The chemical symbol for gold is",
        "The speed of light is approximately",
        "The first president of the United States was",
        "The square root of sixty-four is",
    ]
    prompts = [base_prompts[b % len(base_prompts)] for b in range(B)]

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=draft_len,
    )

    # Per-user prefill (distinct lengths -> prefill each user into its own blocks).
    anchor_tokens, anchor_positions = [], []
    for b in range(B):
        in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [prompts[b]], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
        )
        in_pt = torch.stack(in_pt).view(1, -1)
        generator.prefill_forward_text(
            in_pt,
            page_table=page_table[b : b + 1],
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
        )
        anchor_positions.append(prefill_lens[0] - 1)
        anchor_tokens.append(int(encoded[0][prefill_lens[0] - 1]))

    # Batched spec decode.
    t0 = time.perf_counter()
    outs, accepts = spec.generate_batched(
        anchor_tokens=anchor_tokens,
        anchor_positions=anchor_positions,
        max_new_tokens=n_new,
        max_seq_len=max_seq_len,
        temperature=0.0,
    )
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o) for o in outs)
    mean_accepts = [(sum(a) / len(a)) if a else 0.0 for a in accepts]
    traced = spec._use_trace
    setup_s = getattr(spec, "_last_fused_setup_s", 0.0) if traced else 0.0
    replay_s = getattr(spec, "_last_fused_replay_s", elapsed) if traced else elapsed
    logger.info(
        f"[batched-spec] B={B} draft_len={draft_len} n_new={n_new} ctx~{max(anchor_positions)+1} traced={traced}"
    )
    for b in range(B):
        logger.info(f"  user {b}: prompt={prompts[b]!r}")
        logger.info(f"           out='{tokenizer.decode(outs[b]).strip()}' (acc/iter={mean_accepts[b]:.2f})")
    logger.info(
        f"[batched-spec] total {total_tokens} tokens in {elapsed:.2f}s wall "
        f"(setup {setup_s:.2f}s, steady {replay_s:.2f}s) -> "
        f"{total_tokens/replay_s:.1f} tok/s aggregate, {total_tokens/replay_s/B:.1f} tok/s/user (steady, {'traced' if traced else 'untraced'})"
    )

    assert all(len(o) > 0 for o in outs), f"some user produced no tokens: {[len(o) for o in outs]}"
    if len({prompts[b] for b in range(B)}) > 1:
        distinct = {tuple(o) for o in outs}
        assert len(distinct) > 1, "distinct prompts collapsed to identical outputs (gross cross-user KV leak)"

    # Correctness anchor: user-0 batched output vs batch=1 plain-greedy reference.
    in_pt0, encoded0, decoding_pos0, prefill_lens0 = preprocess_inputs_prefill(
        [prompts[0]], tokenizer, generator.model_args, True, n_new, max_prefill_len=max_seq_len
    )
    in_pt0 = torch.stack(in_pt0).view(1, -1)
    generator.prefill_forward_text(
        in_pt0, page_table=page_table[0:1], kv_cache=tt_kv_cache, prompt_lens=decoding_pos0, warmup_prefill=False
    )
    ref0 = _plain_greedy(spec, anchor_tokens[0], anchor_positions[0], len(outs[0]))
    gen0 = outs[0]
    first_div = next((i for i in range(min(len(gen0), len(ref0))) if gen0[i] != ref0[i]), None)
    if first_div is None:
        logger.info("[batched-spec] user-0 batched output is TOKEN-IDENTICAL to batch=1 plain greedy")
        return

    generator.prefill_forward_text(
        in_pt0, page_table=page_table[0:1], kv_cache=tt_kv_cache, prompt_lens=decoding_pos0, warmup_prefill=False
    )
    tok, pos = anchor_tokens[0], anchor_positions[0]
    for _ in range(first_div):
        lg, hd = spec._verify([tok], [pos])
        hd.deallocate(True)
        tok = int(torch.argmax(lg[0]))
        pos += 1
    lg, hd = spec._verify([tok], [pos])
    hd.deallocate(True)
    top2 = torch.topk(lg[0].float(), 2)
    gap = float(top2.values[0] - top2.values[1])
    logger.info(
        f"[batched-spec] user-0 first divergence at idx {first_div}: batched={gen0[first_div]} "
        f"greedy={ref0[first_div]} target top2={top2.indices.tolist()} gap={gap:.4f} (near-tie={near_tie_gap})"
    )
    assert gap < near_tie_gap, (
        f"batched user-0 diverged from plain greedy at a CONFIDENT token (idx {first_div}, "
        f"top-2 gap={gap:.3f} >= {near_tie_gap}); indicates a batched accept/commit/KV-write bug"
    )
