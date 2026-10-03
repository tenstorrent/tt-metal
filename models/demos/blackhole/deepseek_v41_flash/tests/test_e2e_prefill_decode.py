# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""END-TO-END device PREFILL -> paged hand-off -> traced closed-loop DECODE, with a numeric check of the hand-off against the CPU reference dumps
(reference/ref_prefill_dump.py). Everything goes through the demo's ``create_tt_model`` / ``Generator`` / ``Model`` (tt/dsv41_model.py).

Checks (env):
  DSV41_LAYERS (default 0-39), DSV41_S (default 128 -> dump /mnt/tt-data/ssinghal/dsv4-prefill-s{S}[b{B}]), DSV41_U (users per row, default 4),
  DSV41_LENS (optional comma list of per-user prompt lengths <= S: RAGGED prompts = prefixes of the dump prompts), DSV41_CHUNK (prefill chunk),
  DSV41_STEPS (teacher-forced decode steps compared with the reference, default 3), DSV41_CLOSED (closed-loop steps, default 16).
Prints: per-layer state PCC (window ring, compressed latents, compressor state), first-token logits PCC + argmax agreement, teacher-forced decode
logits PCC / argmax agreement, closed-loop ms/token, TTFT."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

S = int(os.environ.get("DSV41_S", "128"))
U = int(os.environ.get("DSV41_U", "4"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")
STEPS = int(os.environ.get("DSV41_STEPS", "3"))
CLOSED = int(os.environ.get("DSV41_CLOSED", "16"))
CHUNK = int(os.environ.get("DSV41_CHUNK", "0")) or None
LENS = os.environ.get("DSV41_LENS")
DIR = os.environ.get("DSV41_PREFILL_DIR", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}" + ("" if U == 4 else f"b{4 * U}"))
TRACE = os.environ.get("DSV41_TRACE", "1") == "1"
PTRACE = os.environ.get("DSV41_PREFILL_TRACE", "1") == "1"


def pool_rows(model, r):
    return ttnn.to_torch(ttnn.get_device_tensors(model.pool.pool)[r * model.cols]).reshape(-1, 512).float()


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
def test_e2e_prefill_decode(mesh_device):
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(DIR, "tokens.pt"))
    prompt, dec_tok = toks["prefill_tokens"], toks["decode_tokens"]
    B = prompt.shape[0]
    assert B == 4 * U and prompt.shape[1] == S
    lens = torch.tensor([int(x) for x in LENS.split(",")]) if LENS else torch.full((B,), S)
    assert lens.shape[0] == B and int(lens.max()) <= S
    pp = default_page_params(S + STEPS + CLOSED + 64, U)
    args, model, pool, _ = create_tt_model(
        md,
        B,
        S + STEPS + CLOSED + 64,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    gen = Generator(model, args, md)
    full = layer_ids[0] == 0 and layer_ids[-1] == 39
    fin = torch.load(os.path.join(DIR, "final.pt")) if full and os.path.exists(os.path.join(DIR, "final.pt")) else None

    # ---- prefill (twice: compile + measured) ------------------------------------------------------------------------------------------
    for rep in range(2):
        t = time.perf_counter()
        logits = gen.prefill_forward_text(
            prompt, prompt_lens=lens, chunk=CHUNK, return_logits=True, enable_trace=PTRACE
        )
        ttft = time.perf_counter() - t
        log(
            f"PREFILL run {rep}: {ttft:.2f} s for {B} users, lens {lens.tolist()} -> {int(lens.sum()) / ttft:.0f} tok/s; {model.timing}"
        )
    first = logits.argmax(-1)
    log(f"first tokens {first.tolist()}")

    # ---- state vs the reference dumps ---------------------------------------------------------------------------------------------------
    U_ = model.U
    pools = [pool_rows(model, r) for r in range(model.rows)]
    worst = {"ring": 1.0, "comp": 1.0}
    for L in layer_ids:
        ref = torch.load(os.path.join(DIR, f"layer_{L}.pt"), mmap=True)
        st = ref["state"]
        attn = model.attns[L]
        rp, cp = [], []
        for bb in range(B):
            r, u = divmod(bb, U_)
            n = int(lens[bb])
            m = min(n, 128)
            base = pool.ring_base(attn.ring_slot) + u * pool.ring_rows
            got = pools[r][base : base + 128]
            exp = st["window"][bb]
            slots = [(p % 128) for p in range(max(0, n - 128), n)]
            if n == S:
                rp.append(R.pcc(got[slots], exp[slots].float()))
            else:  # prefix of the dump prompt: positions p < n sit at slot p of the dump window iff S <= 128
                if S <= 128:
                    rp.append(R.pcc(got[slots], exp[slots].float()))
            if getattr(attn, "ratio", 0) and attn.source is None:
                nent = n // attn.ratio
                if nent:
                    rows = pool.phys_rows(bb, L, torch.arange(nent))
                    cp.append(R.pcc(pools[r][rows], st["comp"][bb, :nent].float()))
        g0 = pools[0][pool.ring_base(attn.ring_slot) : pool.ring_base(attn.ring_slot) + 128]
        log(
            f"RINGDBG layer {L} slot {attn.ring_slot} base {pool.ring_base(attn.ring_slot)} total_rows {pool.total_rows}: user0 nan {int(g0.isnan().sum())} absmax {float(g0.abs().nan_to_num().max()):.3f} nonzero rows {int((g0.abs().sum(-1) > 0).sum())}"
        )
        msg = f"STATE layer {L:2d} ratio {getattr(attn, 'ratio', 0)}: ring PCC min {min(rp):.4f} mean {sum(rp) / len(rp):.4f}"
        if cp:
            msg += f" | latents PCC min {min(cp):.4f} mean {sum(cp) / len(cp):.4f}"
        if getattr(attn, "prev_cs", None) is not None:
            # expected state: [wkv x | wgate x] of the LAST prompt token of every user (x = the dump's attention input of that layer), fp32 on the host
            from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer

            cw = load_layer(L, with_moe=False)["compressor"]
            devs = ttnn.get_device_tensors(attn.prev_cs)
            got = torch.cat([ttnn.to_torch(devs[r * model.cols]).reshape(U_, -1) for r in range(model.rows)]).float()
            xin = ref["prefill"]["attn_in"]
            ok = []
            for bb in range(B):
                n = int(lens[bb])
                x = xin[bb, n - 1].float()
                exp = torch.cat([cw["wkv"].float() @ x, cw["wgate"].float() @ x])
                ok.append(R.pcc(got[bb], exp))
            msg += f" | prev_cs PCC min {min(ok):.4f} mean {sum(ok) / len(ok):.4f}"
        log(msg)
        if rp:
            worst["ring"] = min(worst["ring"], min(rp))
        if cp:
            worst["comp"] = min(worst["comp"], min(cp))
    log(f"STATE worst: {worst}")

    # ---- first-token logits vs the reference -------------------------------------------------------------------------------------------
    if fin is not None and bool((lens == S).all()):
        log(
            f"FIRST TOKEN: logits PCC {R.pcc(logits, fin['prefill_logits']):.5f}, argmax match {int((first == fin['prefill_argmax']).sum())}/{B}"
        )
    elif (
        full
    ):  # ragged: reference logits of position n-1 from the dump's last-layer hidden states through the host head
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead

        hh = HostHead()
        ref39 = torch.load(os.path.join(DIR, "layer_39.pt"), mmap=True)["prefill"]
        pcs, am = [], 0
        for bb in range(B):
            n = int(lens[bb])
            lg = hh(ref39["h_out"][bb : bb + 1, n - 1 : n].float(), ref39["pm_out"][bb : bb + 1, n - 1 : n].float())
            pcs.append(R.pcc(logits[bb : bb + 1], lg))
            am += int(lg.argmax(-1).item() == int(first[bb]))
        log(f"FIRST TOKEN (ragged): logits PCC min {min(pcs):.4f} mean {sum(pcs) / B:.4f}, argmax match {am}/{B}")

    # ---- decode: teacher forced against the reference, then closed loop ---------------------------------------------------------
    if STEPS and fin is not None and bool((lens == S).all()) and "logits_steps" in fin:
        for i in range(min(STEPS, dec_tok.shape[1])):
            tk = dec_tok[:, i]
            pos = lens + i
            nxt = model.decode_forward(tk, pos, enable_trace=TRACE, reload_inputs=True)
            lg = model.read_logits()
            log(
                f"TEACHER-FORCED step {i} pos {S + i}: logits PCC {R.pcc(lg, fin['logits_steps'][i]):.5f}, "
                f"argmax match {int((lg.argmax(-1) == fin['argmax_steps'][i]).sum())}/{B}"
            )
        # restore the state of the prompt for the closed loop below: re-prefill
        gen.prefill_forward_text(prompt, prompt_lens=lens, chunk=CHUNK, return_logits=True)
    if CLOSED:
        tok, pos = first.clone(), lens.clone()
        outs, times = [tok.clone()], []
        for i in range(CLOSED):
            t = time.perf_counter()
            tok = model.decode_forward(tok, pos, enable_trace=TRACE, reload_inputs=(i == 0))
            times.append(time.perf_counter() - t)
            pos = pos + 1
            outs.append(tok.clone())
        steady = times[2:] or times
        ms = 1e3 * sum(steady) / len(steady)
        log(
            f"CLOSED LOOP {CLOSED} steps: {ms:.1f} ms/token -> {1e3 / ms:.2f} tok/s/user ({1e3 * B / ms:.0f} tok/s total), first step {1e3 * times[0]:.0f} ms; host breakdown {model.timing}"
        )
        T_ = torch.stack(outs)
        for u in range(min(B, 4)):
            log(f"user {u} tokens: {T_[:, u].tolist()}")
