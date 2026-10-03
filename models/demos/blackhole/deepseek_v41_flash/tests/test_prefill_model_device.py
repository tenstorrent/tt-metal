# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2 + M3: all layers resident, DEVICE PREFILL of 16 prompts (tokens -> embedding -> layers + Engram -> first-token logits), then the
existing traced DECODE loop continues from the state the prefill left (teacher-forced tokens of the reference dump).

Env: DSV41_S (prompt length, default 128), DSV41_LAYERS (default 0-39), DSV41_STEPS (decode steps, default 0 = prefill only),
DSV41_CHAIN_PCC=1 (per-layer hidden PCC vs the dump), DSV41_PREFILL_DIR (default /mnt/tt-data/ssinghal/dsv4-prefill-s{S}).
The decode state of every layer is ZEROED after the build (which seeds it from the dump) so only the prefill can have produced it.
"""

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, pad_len
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import DSV41PrefillModel, T, from_chunks
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

S = int(os.environ.get("DSV41_S", "128"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")
STEPS = int(os.environ.get("DSV41_STEPS", "0"))
U = int(os.environ.get("DSV41_U", "4"))  # users per mesh row (batch = 4 U)
CHUNK = int(
    os.environ.get("DSV41_CHUNK", "0")
)  # prompt tokens per chunk (multiple of 128), 0 = whole prompt in one chunk
ZERO = (
    os.environ.get("DSV41_ZERO_STATE") == "1"
)  # no reference dump of this S: random tokens, state/cutoff from the S=128 dump (throughput / consistency runs)
_d = f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}" + ("" if U == 4 else f"b{4 * U}")
DIR = os.environ.get("DSV41_PREFILL_DIR", _d)
BASE = "/mnt/tt-data/ssinghal/dsv4-prefill-s128"  # state seed / gate cutoff source for ZERO runs
MAX_ROPE = max(256, S + 64)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 700_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_model(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    log = lambda m: print(m, flush=True)
    if ZERO:
        g = torch.Generator().manual_seed(0)
        toks = {
            "prefill_tokens": torch.randint(1000, 100000, (4 * U, S), generator=g),
            "decode_tokens": torch.zeros(4 * U, 0, dtype=torch.long),
        }
        fin = None
    else:
        toks = torch.load(os.path.join(DIR, "tokens.pt"))
        fin = torch.load(os.path.join(DIR, "final.pt")) if os.path.exists(os.path.join(DIR, "final.pt")) else None
    prompt = toks["prefill_tokens"]  # [4U, S]
    assert prompt.shape == (4 * U, S)
    chain = DSV41DecodeChain(md, users_per_row=U, max_comp=256, log=log)
    B = chain.B
    sh = _Shards()

    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L, True, MAX_ROPE)) if L in layer_ids else None
    for L in layer_ids[:2]:
        submit(L)
    pls, built, groups, first_moe = [], [], {}, None
    t0 = time.time()
    for L in layer_ids:
        if ZERO:  # decode state is zeroed below anyway: any [B, ...]-shaped seed works
            ref = torch.load(os.path.join(BASE, f"layer_{L}.pt"), mmap=True)
            st0 = {k: v[: 4 * U] for k, v in ref["state"].items()}
            meta = {"state": st0, "S": 1, "gate_cutoff": ref["gate_cutoff"]}
        else:
            ref = torch.load(os.path.join(DIR, f"layer_{L}.pt"), mmap=True)
            meta = {"state": ref["state"], "S": S, "gate_cutoff": ref["gate_cutoff"]}
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, meta, w)
        del w, ref
        # decode state: seeded from the dump by build_layer -> zero it, the prefill must write it
        ttnn.copy(ttnn.zeros_like(attn.cache), attn.cache)
        if getattr(attn, "prev_cs", None) is not None:
            ttnn.copy(ttnn.zeros_like(attn.prev_cs), attn.prev_cs)
        w_sink = sh.get(f"layers.{L}.attn.attn_sink").float()
        attn.prefill = DSV41PrefillAttention(attn, w_sink)
        pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_moe is None else first_moe.decode.buffers)
        first_moe = first_moe or pmoe
        pls.append((L, DSV41PrefillLayer(layer, attn.prefill, pmoe, T=T)))
        key = getattr(attn, "ratio", 0)
        if STEPS and key not in groups:
            groups[key] = DSV41StepState(attn)
        built.append((L, layer, key))
        gc.collect()
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")

    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = (
        HostEngramRows(tuple(engram_ids), max_batch_size=B, max_seq_len=max(256, S + max(STEPS, 1) + 32))
        if engram_ids
        else None
    )
    if host_rows is not None and os.environ.get("DSV41_ENGRAM_RAM", "0") == "1":
        t1 = time.time()
        host_rows.load_ram()
        log(f"Engram tables in RAM {time.time() - t1:.0f}s")
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=U)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    model = DSV41PrefillModel(md, pls, embedding, head, dev_engram, host_rows, users_per_row=U)

    Sp = pad_len(S)
    chain_pcc = os.environ.get("DSV41_CHAIN_PCC") == "1"
    rd = lambda t: torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(rows)])

    def hook(lid, xs, pres):
        ref = torch.load(os.path.join(DIR, f"layer_{lid}.pt"), mmap=True)["prefill"]
        got = from_chunks([rd(x) for x in xs], rows, U, Sp, S, (4, 5120))
        log(
            f"CHAIN layer {lid:2d}: hidden PCC {R.pcc(got, ref['h_out'].float()):.5f}  last-token {R.pcc(got[:, -1], ref['h_out'][:, -1].float()):.5f}"
        )

    res = []
    for rep in range(int(os.environ.get("DSV41_REPS", "2"))):
        hashes = None
        if (
            host_rows is not None and rep == 1
        ):  # the host hash history must see the prompt once: re-hash is idempotent for the same tokens
            pass
        t1 = time.perf_counter()
        logits = model.run_eager(prompt, chunk=CHUNK, hook=hook if (chain_pcc and rep == 0) else None)
        ttft = time.perf_counter() - t1
        res.append(ttft)
        log(
            f"PREFILL run {rep} chunk {CHUNK or 'whole'}: TTFT {ttft:.2f} s for {B} users x {S} tokens ({B * S / ttft:.0f} tok/s)  breakdown {{{', '.join(f'{k}: {v:.2f}' for k, v in model.timing.items())}}}"
        )
    if os.environ.get("DSV41_SAVE_LOGITS"):
        torch.save(logits, os.environ["DSV41_SAVE_LOGITS"])
    if fin is not None and "prefill_logits" in fin:
        p = R.pcc(logits, fin["prefill_logits"])
        am = logits.argmax(-1)
        log(f"FIRST TOKEN: logits PCC {p:.5f}, argmax match {int((am == fin['prefill_argmax']).sum())}/{B}")
    else:
        log(f"first tokens {logits.argmax(-1).tolist()}")
    if os.environ.get("DSV41_TRACE") == "1":
        model.capture_trace(S)
        log("prefill trace captured")
        for rep in range(3):
            t1 = time.perf_counter()
            lg_t = model.run_traced(prompt)
            ttft = time.perf_counter() - t1
            log(
                f"PREFILL TRACED run {rep}: TTFT {ttft:.3f} s for {B} users x {S} tokens ({B * S / ttft:.0f} tok/s) breakdown {{{', '.join(f'{k}: {v:.3f}' for k, v in model.timing.items())}}}  "
                f"logits PCC vs eager {R.pcc(lg_t, logits):.5f}, argmax match {int((lg_t.argmax(-1) == logits.argmax(-1)).sum())}/{B}"
            )
    if STEPS == 0:
        return

    # ---- M3: decode continues from the prefill state ----
    dec_tok = toks["decode_tokens"]
    N = min(dec_tok.shape[1], STEPS)
    dec = DSV41Decoder(md, built, embedding, head, dev_engram, step_states=groups)
    dec.enable_sampling(chain.mesh_config, chain.ccl)
    pos_at = lambda i: torch.full((B,), S + i)
    hashes_dec = [host_rows.hashes(dec_tok[:, i : i + 1], S + i) for i in range(N)] if host_rows is not None else []
    rows_at = lambda i: host_rows.rows_all(hashes_dec[i], engram_ids) if engram_ids else {}
    set_in = lambda i, r: dec.set_packed_inputs(dec_tok[:, i], r, pos_at(i))
    snaps = dec.snapshot_states()
    set_in(0, rows_at(0))
    dec.forward()
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    dl = dec.forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    dec.restore_states(snaps)
    for i in range(N):
        t1 = time.perf_counter()
        set_in(i, rows_at(i))
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        am = head.combine(dec.sampled)[:B]
        wall = (time.perf_counter() - t1) * 1e3
        if fin is not None and "logits_steps" in fin:
            got = head.gather_logits(dl)[:B]
            log(
                f"DECODE STEP {i} pos {S + i}: logits PCC {R.pcc(got, fin['logits_steps'][i]):.5f}  tokens match {int((am == fin['argmax_steps'][i]).sum())}/{B}  wall {wall:.1f} ms"
            )
        else:
            log(f"DECODE STEP {i}: tokens {am.tolist()} wall {wall:.1f} ms")
