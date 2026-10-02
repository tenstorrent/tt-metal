# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: bounded sliding x lanes WITHOUT chunk boundaries.

Single-chunk 8K prompts (chunk 8192), bounded rings on, 2 users per lane.
Clean needles here => the ladder corruption lives in the cross-chunk bounded
machinery (per-request tail stash the lane loop neither keys nor commits).
Garbage here => bounded ring fill/wrap is broken under lanes even without
boundaries.
"""

import os
import time

import torch

from ...tests.test_factory import parametrize_mesh_with_fabric

SLOTS_PER_LANE = 32
GEN_TOKENS = 16


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_bounded_probe(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"
    os.environ["GEMMA4_GEN_PREFILL_CHUNK"] = "8192"

    from models.common.sampling.sampling_params import SamplingParams
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    ISL = int(os.environ.get("G4_PROBE_ISL", "8192"))
    BOUNDED = os.environ.get("G4_PROBE_BOUNDED", "1") == "1"
    bpu = ISL // 64
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + 2 * bpu)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=65536,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=BOUNDED,
    )
    mesh_cfg = generator.model[0].mesh_config
    lanes = mesh_cfg.lanes
    B_g = lanes * SLOTS_PER_LANE

    filler_text = " ".join(f"Entry {i}: shipment {i} reached dock {i % 40} on day {i % 28}." for i in range(2600))
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    while len(filler_ids) < 70000:
        filler_ids = filler_ids + filler_ids

    def build_prompt(plen, code):
        pre = tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False)
        q = tokenizer.encode(
            "\nWhat is the secret access code? Answer with just the number.<end_of_turn>\n<start_of_turn>model\n",
            add_special_tokens=False,
        )
        needle = tokenizer.encode(f" The secret access code is {code}. ", add_special_tokens=False)
        room = plen - len(pre) - len(q) - len(needle)
        a = room // 3
        ids = pre + filler_ids[:a] + needle + filler_ids[a:room] + q
        return torch.tensor(ids[:plen], dtype=torch.long)

    positions, cur_tok, codes = {}, {}, {}
    plen = ISL - 64
    for s in range(2):
        toks4 = torch.zeros(lanes, ISL, dtype=torch.long)
        tables4 = torch.zeros(lanes, 1, bpu, dtype=torch.int32)
        plens = []
        for lane in range(lanes):
            code = 5100 + (lane * 2 + s) * 7
            codes[(lane, s)] = str(code)
            t = build_prompt(plen, code)
            toks4[lane, : t.shape[-1]] = t
            tables4[lane, 0] = torch.arange(1 + s * bpu, 1 + (s + 1) * bpu, dtype=torch.int32)
            plens.append(int(t.shape[-1]))
        t0 = time.perf_counter()
        logits4 = generator.prefill_forward_lanes(toks4, tables4, tt_kv_cache, plens)
        print(f"[probe] round {s} prefill {time.perf_counter()-t0:.1f}s")
        for lane in range(lanes):
            positions[(lane, s)] = plens[lane]
            cur_tok[(lane, s)] = int(torch.argmax(logits4[lane]).item())

    outputs = {k: [] for k in codes}
    sp = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    for _ in range(GEN_TOKENS):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, bpu, dtype=torch.int32)
        for (lane, s), tok in cur_tok.items():
            row = lane * SLOTS_PER_LANE + s
            tokens_g[row] = tok
            pos_g[row] = positions[(lane, s)]
            pt_g[row] = torch.arange(1 + s * bpu, 1 + (s + 1) * bpu, dtype=torch.int32)
        ret = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            read_from_device=True,
            sampling_params=sp,
        )
        out0 = ret[0] if isinstance(ret, (list, tuple)) else ret
        nt = out0.reshape(-1)[:B_g].to(torch.long)
        for lane, s in cur_tok:
            row = lane * SLOTS_PER_LANE + s
            outputs[(lane, s)].append(int(nt[row].item()))
            cur_tok[(lane, s)] = int(nt[row].item())
            positions[(lane, s)] += 1

    bad = 0
    for (lane, s), toks in sorted(outputs.items()):
        text = tokenizer.decode(toks, skip_special_tokens=True)
        ok = codes[(lane, s)] in text
        bad += 0 if ok else 1
        print(f"[probe] lane {lane} slot {s} needle {codes[(lane,s)]} -> {text[:44]!r} {'OK' if ok else 'MISS'}")
    assert bad == 0, f"{bad}/8 needles missing (bounded x lanes, single-chunk)"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_chunk_vs_serial(mesh_device, reset_seeds, request):
    """2-chunk bisect: my lane loop vs the serial wrapper on identical tokens."""
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"
    os.environ["GEMMA4_GEN_PREFILL_CHUNK"] = "8192"

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    ISL = 16384
    bpu = ISL // 64
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + 2 * bpu)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=32,
        max_seq_len=32768,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=False,
    )
    lanes = generator.model[0].mesh_config.lanes
    vocab = generator.model[0].vocab_size

    filler_text = " ".join(f"Entry {i}: shipment {i} reached dock {i % 40} on day {i % 28}." for i in range(1300))
    fids = tokenizer.encode(filler_text, add_special_tokens=False)
    while len(fids) < 40000:
        fids = fids + fids
    pre = tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False)
    q = tokenizer.encode(
        "\nWhat is the secret access code? Answer with just the number.<end_of_turn>\n<start_of_turn>model\n",
        add_special_tokens=False,
    )
    needle = tokenizer.encode(" The secret access code is 6217. ", add_special_tokens=False)
    plen = ISL - 64
    room = plen - len(pre) - len(q) - len(needle)
    ids = pre + fids[: room // 3] + needle + fids[room // 3 : room] + q
    t = torch.tensor(ids[:plen], dtype=torch.long)

    # Serial wrapper reference (slot 0 -> lane 0, blocks 1..bpu)
    tok1 = torch.zeros(1, ISL, dtype=torch.long)
    tok1[0, :plen] = t
    blocks0 = torch.arange(1, 1 + bpu, dtype=torch.int32)
    out = generator.prefill_forward_text(
        tok1,
        page_table=blocks0.unsqueeze(0),
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor([plen]),
        empty_slots=[0],
        enable_trace=False,
        sampling_params=None,
        warmup_prefill=False,
    )
    lg_serial = (out[0] if isinstance(out, (list, tuple)) else out).reshape(-1)[:vocab].float()

    # Lane loop: SAME tokens on every lane, per-lane slot-1 blocks
    toks4 = torch.zeros(lanes, ISL, dtype=torch.long)
    tables4 = torch.zeros(lanes, 1, bpu, dtype=torch.int32)
    for lane in range(lanes):
        toks4[lane, :plen] = t
        tables4[lane, 0] = torch.arange(1 + bpu, 1 + 2 * bpu, dtype=torch.int32)
    lg_lane = generator.prefill_forward_lanes(toks4, tables4, tt_kv_cache, [plen] * lanes)

    a = lg_serial - lg_serial.mean()
    for lane in range(lanes):
        b = lg_lane[lane] - lg_lane[lane].mean()
        pcc = float((a @ b) / (a.norm() * b.norm() + 1e-9))
        print(
            f"[bisect] lane {lane}: pcc vs serial = {pcc:.5f} "
            f"(argmax serial={int(lg_serial.argmax())} lane={int(lg_lane[lane].argmax())})"
        )

    # ── Decode phase: serial-prefilled user (slot 0) + lane users (slot 1) ──
    B_g = lanes * 32
    live = {(0, 0): (int(lg_serial.argmax()), blocks0)}
    for lane in range(lanes):
        live[(lane, 1)] = (int(lg_lane[lane].argmax()), torch.arange(1 + bpu, 1 + 2 * bpu, dtype=torch.int32))
    pos = {k: plen for k in live}
    outs = {k: [] for k in live}
    for _ in range(16):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, bpu, dtype=torch.int32)
        for (lane, sl), (tok, blocks) in live.items():
            row = lane * 32 + sl
            tokens_g[row] = tok
            pos_g[row] = pos[(lane, sl)]
            pt_g[row] = blocks
        _tr = os.environ.get("G4_BISECT_TRACE", "0") == "1"
        ret = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=_tr,
            read_from_device=True,
            sampling_params=None,
        )
        out0 = ret[0] if isinstance(ret, (list, tuple)) else ret
        lgd = out0.reshape(B_g, -1)
        for lane, sl in list(live):
            row = lane * 32 + sl
            nxt = int(torch.argmax(lgd[row]).item())
            outs[(lane, sl)].append(nxt)
            tok, blocks = live[(lane, sl)]
            live[(lane, sl)] = (nxt, blocks)
            pos[(lane, sl)] += 1
    for (lane, sl), toks in sorted(outs.items()):
        text = tokenizer.decode(toks, skip_special_tokens=True)
        kind = "SERIAL" if sl == 0 else "LANE"
        print(f"[bisect-decode] {kind} lane {lane} slot {sl}: {text[:44]!r} {'OK' if '6217' in text else 'MISS'}")
    assert all(
        float(
            ((lg_serial - lg_serial.mean()) @ (lg_lane[l] - lg_lane[l].mean()))
            / ((lg_serial - lg_serial.mean()).norm() * (lg_lane[l] - lg_lane[l].mean()).norm() + 1e-9)
        )
        > 0.99
        for l in range(lanes)
    ), "lane chunk loop diverges from serial wrapper"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_decode_crossing(mesh_device, reset_seeds, request):
    """Decode across position 8192 from a known-good 8K prefill."""
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"
    os.environ["GEMMA4_GEN_PREFILL_CHUNK"] = "8192"

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    bpu = 160  # 10240 tokens: 8K prompt + 2K generation
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + bpu)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=32,
        max_seq_len=16384,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=False,
    )
    lanes = generator.model[0].mesh_config.lanes
    B_g = lanes * 32

    filler_text = " ".join(f"Entry {i}: shipment {i} reached dock {i % 40} on day {i % 28}." for i in range(700))
    fids = tokenizer.encode(filler_text, add_special_tokens=False)
    pre = tokenizer.encode("<start_of_turn>user\n", add_special_tokens=False)
    q = tokenizer.encode(
        "\nPlease count upward from 100, one number per line, like 100 101 102, continuing for a long time."
        "<end_of_turn>\n<start_of_turn>model\n",
        add_special_tokens=False,
    )
    plen = 8192 - 64
    room = plen - len(pre) - len(q)
    ids = pre + fids[:room] + q
    t = torch.tensor(ids[:plen], dtype=torch.long)

    tok1 = torch.zeros(1, 8192, dtype=torch.long)
    tok1[0, :plen] = t
    blocks0 = torch.arange(1, 1 + bpu, dtype=torch.int32)
    out = generator.prefill_forward_text(
        tok1,
        page_table=blocks0.unsqueeze(0),
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor([plen]),
        empty_slots=[0],
        enable_trace=False,
        sampling_params=None,
        warmup_prefill=False,
    )
    cur = int((out[0] if isinstance(out, (list, tuple)) else out).reshape(-1).argmax())

    pos = plen  # 8128
    toks = []
    marks = {}
    for step in range(200):
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, bpu, dtype=torch.int32)
        tokens_g[0] = cur
        pos_g[0] = pos
        pt_g[0] = blocks0
        ret = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=False,
            read_from_device=True,
            sampling_params=None,
        )
        out0 = ret[0] if isinstance(ret, (list, tuple)) else ret
        cur = int(torch.argmax(out0.reshape(B_g, -1)[0]).item())
        toks.append(cur)
        if pos in (8189, 8190, 8191, 8192, 8193):
            marks[pos] = len(toks)
        pos += 1
    text = tokenizer.decode(toks, skip_special_tokens=True)
    cross = marks.get(8191, 60)
    print(f"[crossing] BEFORE 8192 (pos 8128..8191): {tokenizer.decode(toks[:cross])!r}")
    print(f"[crossing] AFTER  8192 (pos 8192..):     {tokenizer.decode(toks[cross:cross+80])!r}")
    degraded_before = "1" not in tokenizer.decode(toks[:cross])
    print(f"[crossing] verdict: before={'BAD' if degraded_before else 'ok'}")


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_kv_block_audit(mesh_device, reset_seeds, request):
    """After a 16K (2-chunk) lane prefill, which physical blocks hold K data?"""
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"
    os.environ["GEMMA4_GEN_PREFILL_CHUNK"] = "8192"

    import ttnn as _ttnn
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    ISL = 16384
    bpu = ISL // 64
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + bpu)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=32,
        max_seq_len=32768,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=False,
    )
    model = generator.model[0]
    lanes = model.mesh_config.lanes
    lt = model.hf_config.layer_types
    g_idx = next(i for i, t in enumerate(lt) if t == "full_attention")
    s_idx = next(i for i, t in enumerate(lt) if t == "sliding_attention")

    fids = tokenizer.encode(" ".join(f"Entry {i}: dock {i%40}." for i in range(4000)), add_special_tokens=False)
    plen = ISL - 64
    t = torch.tensor((fids * 4)[:plen], dtype=torch.long)
    toks4 = torch.zeros(lanes, ISL, dtype=torch.long)
    tables4 = torch.zeros(lanes, 1, bpu, dtype=torch.int32)
    for lane in range(lanes):
        toks4[lane, :plen] = t
        tables4[lane, 0] = torch.arange(1, 1 + bpu, dtype=torch.int32)
    generator.prefill_forward_lanes(toks4, tables4, tt_kv_cache, [plen] * lanes)

    for name, idx in (("GLOBAL", g_idx), ("SLIDING", s_idx)):
        kv = model.tt_kv_cache[idx]
        k = kv[0] if isinstance(kv, (list, tuple)) else kv
        shard0 = _ttnn.to_torch(_ttnn.get_device_tensors(k)[0]).float()  # lane 0, tp row 0
        # [pool_blocks, heads, block, dim]
        norms = shard0.reshape(shard0.shape[0], -1).norm(dim=-1)
        nz = (norms > 1e-3).nonzero().reshape(-1).tolist()
        span = f"{nz[0]}..{nz[-1]} ({len(nz)} blocks)" if nz else "NONE"
        halves = (
            int((norms[1 : 1 + bpu // 2] > 1e-3).sum()),
            int((norms[1 + bpu // 2 : 1 + bpu] > 1e-3).sum()),
        )
        print(
            f"[kvaudit] layer {idx} {name}: pool {shard0.shape[0]} blocks; nonzero {span}; "
            f"chunk0-half {halves[0]}/{bpu//2}, chunk1-half {halves[1]}/{bpu//2}"
        )
