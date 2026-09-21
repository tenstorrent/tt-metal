"""Acceptance probe: does the DFlash2 drafter predict OUR target's real greedy continuation from the
LIVE taps (eager masked-prefill taps + eager seed taps), and does the production drafter agree with
the oracle? Isolates (a) tap/semantics problems (oracle vs target) from (b) KV-path problems
(production vs oracle).

Steps (prompt T=130, same as test_spec_lossless):
  ref        : plain greedy 12 tokens -> ref[0]=token@T ... ref[11]=token@T+11
  prompt taps: prefill_for_spec with model._dflash_tap -> (1,256,25600) bucket rows, first T real
  oracle A   : DFlash2Draft.propose(taps[:T], anchor=ref[0], C=T)      vs ref[1:8]
  seed taps  : verify_forward([ref[0]], T) -> row 0 = position T; ctx = taps[:T] ++ seed
  oracle B   : propose(ctx, anchor=ref[1], C=T+1)                       vs ref[2:9]   (== loop iter 0)
  production : fill(prompt bucket taps) + extend(seed, T, 1) + draft(ref[1], T+1) vs ref[2:9], vs oracle B

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/dflash2_accept_probe.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NUM_BLOCKS, PROMPT_LEN, _reference_greedy
from models.demos.blackhole.qwen36.tt.dflash2 import DFlash2Draft, taps_to_host
from models.demos.blackhole.qwen36.tt.dflash2_decode import _load_embed_host, drafter_weights_dir, get_drafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import get_block_size


def _prefix(drafts, truth):
    m = 0
    for d, t in zip(drafts, truth):
        if d != t:
            break
        m += 1
    return m


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_accept_probe(mesh_device):
    if not _MULTI:
        pytest.skip("TP only")
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = _get_prompt(PROMPT_LEN, tokenizer)[0].tolist()
    T = len(prompt_ids)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.use_fused_recurrent_decode = True

    ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, 20, use_decode_step=True)
    logger.info(f"[probe] ref tokens {ref}  text={tokenizer.decode(ref)!r}")
    logger.info(f"[probe] ref top-2 gaps {[f'{g:.2f}' for g in gaps]}")

    # ---- live prompt taps (eager masked prefill) ----
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    model._dflash_tap = True
    from models.demos.blackhole.qwen36.tt.dflash2 import load_config

    _cfg = load_config(drafter_weights_dir())
    model._dflash_tap_layers = tuple(_cfg["taps"])
    os.environ["QWEN36_DFLASH_TP"] = "0"  # replicated form: host-tap API
    drafter = get_drafter(model)
    drafter.alloc_kv(pt, get_block_size(model._paged_kv_caches))
    chunks = []

    def on_chunk(hidden, chunk_start, valid_len):
        taps = model.take_dflash_eager_taps()
        assert taps is not None
        host = taps_to_host(model.mesh_device, taps)  # (1, bucket, 25600)
        for t in taps:
            ttnn.deallocate(t)
        chunks.append((chunk_start, host[:, :valid_len]))
        drafter.fill_context(host, chunk_start)  # production: per prefill chunk

    prompt = torch.tensor([prompt_ids], dtype=torch.int32)
    logits_dev = model.prefill_for_spec(prompt, pt, T, on_chunk)
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    first = int(lt.reshape(-1)[: model.vocab_size].float().argmax())
    assert first == ref[0], f"prefill first {first} != ref {ref[0]}"
    chunks.sort(key=lambda c: c[0])
    assert chunks[0][0] == 0 and sum(c[1].shape[1] for c in chunks) == T, [(c[0], c[1].shape[1]) for c in chunks]
    taps = torch.cat([c[1] for c in chunks], dim=1)  # (1,T,25600)
    logger.info(f"[probe] prompt T={T} in {len(chunks)} chunk(s) {[(c[0], c[1].shape[1]) for c in chunks]}")
    for li, lo in enumerate(range(0, 25600, 5120)):
        seg = taps[0, :, lo : lo + 5120]
        logger.info(
            f"[probe] tap L{_cfg['taps'][li]}: mean row-norm {seg.norm(dim=-1).mean():.2f}  "
            f"row0 {seg[0].norm():.2f} rowT-1 {seg[-1].norm():.2f}  absmean {seg.abs().mean():.4f}"
        )

    embed = _load_embed_host(model.args.CKPT_DIR)
    import json

    from safetensors import safe_open

    _shard = json.load(open(f"{model.args.CKPT_DIR}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    with safe_open(f"{model.args.CKPT_DIR}/{_shard}", framework="pt") as f:
        lm = f.get_tensor("lm_head.weight").float()
    oracle = DFlash2Draft(model.mesh_device, drafter_weights_dir(), embed, lm)
    BLOCK = oracle.block
    K = BLOCK - 1
    model._dflash_tap_layers = tuple(oracle.taps)

    dA = oracle.propose(taps, ref[0], T)
    logger.info(f"[probe] oracle A (ctx 0..T-1, anchor=ref[0]@T):  drafts {dA}")
    logger.info(
        f"[probe]                                   truth  {ref[1:1 + K]}  prefix={_prefix(dA, ref[1:1 + K])} match={sum(a == b for a, b in zip(dA, ref[1:1 + K]))}/{K}"
    )

    # ---- seed (position T) taps ----
    clog, chid = model.verify_forward([first], T, pt, gdn_recurrent=True)
    ttnn.deallocate(chid)
    pending = int(clog[0].argmax())
    assert pending == ref[1], f"seed pending {pending} != ref[1] {ref[1]}"
    seed_taps = model.take_dflash_eager_taps()
    assert seed_taps is not None
    seed_host = taps_to_host(model.mesh_device, seed_taps, rows=BLOCK)  # (1,8,25600) row 0 = pos T
    for t in seed_taps:
        ttnn.deallocate(t)
    ctx = torch.cat([taps, seed_host[:, :1]], dim=1)  # (1,T+1,25600)
    dB = oracle.propose(ctx, pending, T + 1)
    logger.info(f"[probe] oracle B (ctx 0..T, anchor=pending@T+1): drafts {dB}")
    logger.info(
        f"[probe]                                   truth  {ref[2:2 + K]}  prefix={_prefix(dB, ref[2:2 + K])} match={sum(a == b for a, b in zip(dB, ref[2:2 + K]))}/{K}"
    )

    # ---- production drafter (context filled per chunk above), same inputs ----
    drafter.extend_context(seed_host, T, 1)
    dP = drafter.draft(pending, T + 1)
    logger.info(f"[probe] production (fill+extend+draft):          drafts {dP}")
    logger.info(
        f"[probe]                                   truth  {ref[2:2 + K]}  prefix={_prefix(dP, ref[2:2 + K])} match_truth={sum(a == b for a, b in zip(dP, ref[2:2 + K]))}/{K} match_oracleB={sum(a == b for a, b in zip(dP, dB))}/{K}"
    )
    drafter.free_kv()
    model._dflash_tap = False
    model.free_kv_caches()
    logger.info(
        f"[probe] decoded: oracleB={tokenizer.decode(dB)!r} prod={tokenizer.decode(dP)!r} truth={tokenizer.decode(ref[2:2 + K])!r}"
    )
