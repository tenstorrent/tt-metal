# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capacity-ladder reference on metal (galaxy one-instance lanes).

Rungs: 12x256K / 24x128K / 48x64K / 96x32K — every rung ~3.15M KV tokens
(4 pools x 786K). Prefill runs the lane-parallel CHUNKED loop (4 users per
round, one per lane), so a rung's last-user TTFT is rounds x one chunked
prefill instead of users x. Decode runs the traced lane-major frame with the
rung's users live, device-sampled. Needle oracles (one per lane per rung)
catch any cross-user KV mixing at every rung.

    GEMMA4_GALAXY_FRACTURE=1 GEMMA4_GALAXY_LANES=1 HF_MODEL=google/gemma-4-31B-it \
    pytest models/demos/gemma4/tests/unit/test_lanes_ladder.py -k 8x4 -s
"""

import os
import time

import torch

from ...tests.test_factory import parametrize_mesh_with_fabric

SLOTS_PER_LANE = int(os.environ.get("G4_LADDER_LOCAL_BATCH", "32"))
GEN_TOKENS = 24
POOL_BLOCKS_PER_LANE = 12288  # 786,432 tokens at block 64 — serving parity
import json as _json

RUNGS = _json.loads(os.environ.get("G4_LADDER_RUNGS", "[[32768,24],[65536,12],[131072,6],[253952,3]]"))


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_lanes_ladder(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_GALAXY_LANES"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "0"
    os.environ.setdefault("GEMMA4_GEN_PREFILL_CHUNK", "8192")

    from models.common.sampling.sampling_params import SamplingParams
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + POOL_BLOCKS_PER_LANE)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=262144,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=True,  # rings for the 50 sliding layers; full pools only for the ~5 unique globals
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False)
    lanes = mesh_cfg.lanes
    B_g = lanes * SLOTS_PER_LANE

    filler_text = " ".join(f"Entry {i}: shipment {i} reached dock {i % 40} on day {i % 28}." for i in range(2600))
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    while len(filler_ids) < 260000:
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

    results = []
    for isl, per_lane in RUNGS:
        users = lanes * per_lane
        bpu = isl // 64
        assert per_lane * bpu <= POOL_BLOCKS_PER_LANE
        assert per_lane <= SLOTS_PER_LANE
        plen = isl - 64  # pads back to isl (power-of-2 grid), margin for decode
        codes = {}
        positions, cur_tok = {}, {}

        # ── Prefill: per_lane rounds of 4 (one user per lane) ──────────────
        round_walls = []
        t_rung = time.perf_counter()
        for s in range(per_lane):
            toks4 = torch.zeros(lanes, isl, dtype=torch.long)  # chunk-grid width; real lengths in plens
            tables4 = torch.zeros(lanes, 1, bpu, dtype=torch.int32)
            plens = []
            for lane in range(lanes):
                code = 7000 + isl // 1024 + (lane * per_lane + s) * 3
                codes[(lane, s)] = str(code)
                t = build_prompt(plen, code)
                toks4[lane, : t.shape[-1]] = t
                tables4[lane, 0] = torch.arange(1 + s * bpu, 1 + (s + 1) * bpu, dtype=torch.int32)
                plens.append(int(t.shape[-1]))
            t0 = time.perf_counter()
            if os.environ.get("G4_LADDER_SERIAL_PREFILL", "0") == "1":
                # Discriminator mode: the proven serial wrapper per user (4x
                # slower) — separates lane-loop-specific defects from shared
                # depth machinery under lanes.
                rows_l = []
                for lane in range(lanes):
                    tok1 = toks4[lane : lane + 1]
                    out1 = generator.prefill_forward_text(
                        tok1,
                        page_table=tables4[lane],
                        kv_cache=tt_kv_cache,
                        prompt_lens=torch.tensor([plens[lane]]),
                        empty_slots=[lane * SLOTS_PER_LANE + s],
                        enable_trace=False,
                        sampling_params=None,
                        warmup_prefill=False,
                    )
                    lg1 = out1[0] if isinstance(out1, (list, tuple)) else out1
                    rows_l.append(lg1.reshape(-1)[: generator.model[0].vocab_size].float())
                logits4 = torch.stack(rows_l)
            else:
                logits4 = generator.prefill_forward_lanes(toks4, tables4, tt_kv_cache, plens, slot_ids=s)
            round_walls.append(time.perf_counter() - t0)
            for lane in range(lanes):
                positions[(lane, s)] = plens[lane]
                cur_tok[(lane, s)] = int(torch.argmax(logits4[lane]).item())
        prefill_total = time.perf_counter() - t_rung
        print(f"[rung {isl//1024}K] prefill argmax r0: {[cur_tok[(lane, 0)] for lane in range(lanes)]}")
        med_round = sorted(round_walls)[len(round_walls) // 2]
        print(
            f"[rung {isl//1024}K x {users}] prefill: round med {med_round:.1f}s x {per_lane} rounds "
            f"= {prefill_total:.1f}s total (last-user TTFT); 4-burst TTFT = {round_walls[0]:.1f}s"
        )

        # ── Decode: all rung users live, traced + device-sampled ───────────
        outputs = {k: [] for k in codes}
        # Decode-side per-layer tables: sliding layers read/write their ring
        # pools by slot-local ids; global layers use the rung's block tables.
        # Rung-invariant width: growing per-layer tables across rungs would
        # realloc the persistent buffers and orphan the captured decode trace.
        g_dec = torch.zeros(B_g, 3968, dtype=torch.int32)
        for lane, sl in codes:
            row = lane * SLOTS_PER_LANE + sl
            g_dec[row, :bpu] = torch.arange(1 + sl * bpu, 1 + (sl + 1) * bpu, dtype=torch.int32)
        per_layer_dec, ring_dec = [], {}
        for layer in generator.model[0].layers:
            cfg = getattr(getattr(layer, "self_attn", None), "config", None)
            m = getattr(cfg, "cache_position_modulo", None) if cfg is not None else None
            if not m:
                per_layer_dec.append(g_dec)
                continue
            rb = int(m) // 64
            if rb not in ring_dec:
                rows = torch.zeros(B_g, rb, dtype=torch.int32)
                for r in range(B_g):
                    sl_r = r % SLOTS_PER_LANE
                    rows[r] = torch.arange(sl_r * rb, (sl_r + 1) * rb, dtype=torch.int32)
                ring_dec[rb] = rows
            per_layer_dec.append(ring_dec[rb])
        generator.model[0]._active_page_tables_per_layer = per_layer_dec
        # New rung = new KV geometry: drop the previous rung's decode traces so
        # the first decode step recaptures against the new bindings (mirrors
        # the vLLM wrapper's reset after a page-table buffer grow); a replayed
        # stale trace decodes garbage on every rung after the first.
        from collections import defaultdict as _dd

        import ttnn as _ttnn

        def _release_all(obj):
            if obj is None:
                return
            if isinstance(obj, _ttnn.Tensor):
                try:
                    obj.deallocate(True)
                except Exception:
                    pass
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    _release_all(x)
            elif isinstance(obj, dict):
                for x in obj.values():
                    _release_all(x)

        # Release the previous rung's Metal traces AND their persistent input
        # tensors — clearing the python dicts alone leaks the device buffers
        # (~hundreds of MB per rung; rung 3 OOM'd a 704 MB prefill buffer).
        for _tid in list(getattr(generator, "trace_ids_decode", {}).values()):
            if _tid is not None:
                try:
                    _ttnn.release_trace(mesh_device, _tid)
                except Exception:
                    pass
        for _key in ("trace_inputs_decode", "trace_output_decode"):
            _release_all(dict(getattr(generator, _key, {}) or {}))
        generator.trace_ids_decode = _dd(lambda: None)
        generator.trace_inputs_decode = _dd(lambda: None)
        generator.trace_output_decode = _dd(lambda: None)
        generator._prev_decode_batch = None
        for _m in generator.model:
            _s = getattr(_m, "sampling", None)
            if _s is not None and hasattr(_s, "reset_trace"):
                _s.reset_trace()
        sp = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        step_times = []
        for _step in range(GEN_TOKENS):
            tokens_g = torch.zeros(B_g, dtype=torch.long)
            pos_g = torch.full((B_g,), -1, dtype=torch.long)
            # Fixed table width across rungs: the decode trace captures the
            # page-table input shape once (key = sampling x batch), so every
            # rung must present the same width; unused columns are null block.
            pt_g = torch.zeros(B_g, 3968, dtype=torch.int32)
            for (lane, s), tok in cur_tok.items():
                row = lane * SLOTS_PER_LANE + s
                tokens_g[row] = tok
                pos_g[row] = positions[(lane, s)]
                pt_g[row, :bpu] = torch.arange(1 + s * bpu, 1 + (s + 1) * bpu, dtype=torch.int32)
            t0 = time.perf_counter()
            ret = generator.decode_forward(
                tokens_g.unsqueeze(-1),
                pos_g,
                page_table=pt_g,
                kv_cache=tt_kv_cache,
                enable_trace=os.environ.get("G4_LADDER_EAGER_DECODE", "0") != "1",
                read_from_device=True,
                sampling_params=sp,
            )
            out0 = ret[0] if isinstance(ret, (list, tuple)) else ret
            next_tok = out0.reshape(-1)[:B_g].to(torch.long)
            step_times.append(time.perf_counter() - t0)
            for lane, s in cur_tok:
                row = lane * SLOTS_PER_LANE + s
                nxt = int(next_tok[row].item())
                outputs[(lane, s)].append(nxt)
                cur_tok[(lane, s)] = nxt
                positions[(lane, s)] += 1
        steady = sorted(step_times[3:])[len(step_times[3:]) // 2]
        agg = users / steady
        print(
            f"[rung {isl//1024}K x {users}] decode: {steady*1e3:.1f} ms/step -> "
            f"{agg:.0f} tok/s aggregate, {agg/users:.1f} tok/s/user"
        )

        # ── Oracle: round-0 needles (one per lane) ─────────────────────────
        bad = []
        for lane in range(lanes):
            text = tokenizer.decode(outputs[(lane, 0)], skip_special_tokens=True)
            if codes[(lane, 0)] not in text:
                bad.append((lane, codes[(lane, 0)], text[:60]))
            print(f"[rung {isl//1024}K] lane {lane} needle {codes[(lane,0)]} -> {text[:40]!r}")
        assert not bad, f"rung {isl}: needles missing: {bad}"
        results.append((isl, users, round_walls[0], prefill_total, steady * 1e3, agg))

    print("\n=== LADDER (metal, lanes, chunk " f"{os.environ.get('GEMMA4_GEN_PREFILL_CHUNK')}): ===")
    for isl, users, burst, total, step_ms, agg in results:
        print(
            f"  {isl//1024:3d}K x {users:3d}: 4-burst TTFT {burst:6.1f}s | last-user {total:6.1f}s | "
            f"decode {step_ms:5.1f} ms/step = {agg:4.0f} tok/s agg"
        )
