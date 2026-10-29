# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The per-segment join and the decode tokens after a packed prefill (``TT_V41_PACK`` with ``TT_V41_PACK_LAYOUTS``), through ``DeepseekV41ForCausalLM`` as the generator and short-chunk tests.

For each ``k`` of ``V41_PJ_KS`` (``2,4``): ``k`` distinct ``V41_PJ_PROMPT``-token (1,028) windows of the book are
prefilled one by one through ``prefill_forward`` (the single path on the 5,120-row chunk, eagerly)
and joined into decode slots, then the same ``k`` prompts as one packed call (``k x 1028`` listed) whose rows join
from their segments of the packed state, at the same pool step. Compared per prompt: the prefill logits returned
per row, the slot's pool rows after the join (the compressed KV and index-K rows of every source, the window rings
of every layer, the candidate source's block pages; scaled-FP8 rows decoded) and ``V41_PJ_STEPS`` (8) free-running
greedy decode steps of all ``2k`` slots in one step each, then of each slot alone (the c1 variant under the fused decode flags: the live-user index reads the slot's position, which must be the segment's length). Segment 0's prefill
state is bit-identical to the single run's, so its logits, pool rows and tokens are asserted
equal; the later segments' are reported (PCC, argmax, the first differing step; the TP all-reduce's row-dependent
summation order) and asserted under ``V41_PJ_STRICT=1``. Then the ownership case: a second pack of
``kmax`` other prompts before any decode forces the first pack's pending segments to join (one allocation shared
by its segments), and the decode after it. ``V41_PJ_TRACE=1`` (default 0: the packs eager) captures the layouts
at warm-up (``V41PackTrace``, the segment joins per slot) and checks a replayed pack against the eager pack of the
same prompts bit for bit (logits and pool rows), alternating layouts and a single prompt between them, with the
replays' device time. ``V41_PJ_LONG=1`` adds the ``2 x 2052`` layout. ``V41_PJ_MAX_SEQ`` (16,384) is the context,
``V41_PJ_BATCH`` (16) the slots. Results: ``V41_PJ_RESULT {json}``, ``V41_PJ_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_decode import TRACE_REGION
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table, _trace_bytes
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.pool import BLOCK_ROWS, RING, V41DecodePool
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, reconstruct_scaled_fp8_kv_cache


def _down(mesh_device, t) -> torch.Tensor:
    """Every chip's copy of ``t`` as one host tensor [chips, rows, W]."""
    if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
        every = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    else:
        every = torch.stack([ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)])
    return every.reshape(every.shape[0], -1, every.shape[-1])


def _slot_rows(mesh_device, pool: V41DecodePool, slot: int, length: int) -> dict:
    """What a prompt of ``length`` tokens holds in pool slot ``slot`` on its rank's first chip: per source the
    compressed rows (decoded values) and every consumer layer's ring, the index keys (and the candidate source's
    block pages)."""
    cfg = pool.config
    rank, u = divmod(slot, pool.users)
    chip = rank * pool.tp
    out = {}
    for src, t in pool.kv.items():
        every = _down(mesh_device, t)[chip]
        base = u * pool.slots[src].rows
        if src >= 0:
            n = length // cfg.compress_ratio(src)
            rows = every[base + pool.comp_row(src) : base + pool.comp_row(src) + n]
            if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
                rows = reconstruct_scaled_fp8_kv_cache(rows.contiguous(), pool.kv_geometry).float()
            out[f"kv[{src}] compressed"] = rows
        for layer in pool.slots[src].layers:
            rows = every[base + pool.ring_row(layer) : base + pool.ring_row(layer) + RING]
            if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
                rows = reconstruct_scaled_fp8_kv_cache(rows.contiguous(), pool.kv_geometry).float()
            out[f"ring[{layer}]"] = rows
    for src, t in pool.index_k.items():
        n = length // cfg.compress_ratio(src)
        cap = pool.slots[src].cap + 32
        every = _down(mesh_device, t)[chip]  # [users * (cap + 32), idim]
        out[f"index_k[{src}]"] = every[u * cap : u * cap + n]
    for src, t in pool.index_k_blocks.items():
        n = -(-(length // cfg.compress_ratio(src)) // BLOCK_ROWS)
        cap = pool.slots[src].cap + 32
        every = _down(mesh_device, t)[chip]  # [users * (cap + 32) / BLOCK_ROWS, BLOCK_ROWS * idim]
        out[f"index_k_blocks[{src}]"] = every[u * cap // BLOCK_ROWS : u * cap // BLOCK_ROWS + n]
    return out


def _compare(name: str, want: dict, got: dict) -> tuple[list[str], dict]:
    """Bit-for-bit comparison: the differing keys described, and per key a record."""
    lines, record = [], {}
    for k, a in want.items():
        b = got.get(k)
        if b is None or a.shape != b.shape:
            lines.append(f"{name}: {k} {'missing' if b is None else f'shape {tuple(a.shape)} vs {tuple(b.shape)}'}")
            record[k] = "missing" if b is None else "shape"
            continue
        if torch.equal(a, b):
            record[k] = "equal"
            continue
        af, bf = a.float(), b.float()
        bad = a != b
        entry = {
            "differ": int(bad.sum()),
            "of": a.numel(),
            "max_abs_diff": round(float((af - bf).abs().max()), 6),
            "max_abs": round(float(af.abs().max()), 4),
        }
        if a.dim() >= 2:
            rows = bad.reshape(-1, a.shape[-1]).any(-1).nonzero().flatten()
            entry["rows"] = f"{int(rows.min())}..{int(rows.max())} of {a.reshape(-1, a.shape[-1]).shape[0]}"
        if af.numel() > 1 and af.std() > 0 and bf.std() > 0:
            entry["pcc"] = round(float(torch.corrcoef(torch.stack([af.flatten(), bf.flatten()]))[0, 1]), 6)
        record[k] = entry
        lines.append(f"{name}: {k} {entry}")
    return lines, record


def _logits_record(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    top = lambda t, k: set(t.topk(k).indices.tolist())
    return {
        "equal": bool(torch.equal(a, b)),
        "pcc": round(float(torch.corrcoef(torch.stack([a, b]))[0, 1]), 6),
        "max_abs_diff": round(float((a - b).abs().max()), 4),
        "max_abs": round(float(a.abs().max()), 4),
        "argmax_equal": int(a.argmax()) == int(b.argmax()),
        "top5_overlap": len(top(a, 5) & top(b, 5)),
    }


def _tokens_record(a: list[int], b: list[int]) -> dict:
    first = next((s for s, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return {"equal": a == b, "first_diff_step": first, "equal_steps": sum(x == y for x, y in zip(a, b))}


def _trace_region() -> int:
    return TRACE_REGION if int(os.environ.get("V41_PJ_MAX_SEQ", 16384)) <= 16384 else 2 * 1024**3


@pytest.mark.timeout(int(os.environ.get("V41_PJ_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_PJ_TRACE_REGION", _trace_region()))),
    indirect=True,
)
def test_v41_pack_join(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_PJ_LAYERS", "all"))
    prompt = int(os.environ.get("V41_PJ_PROMPT", 1028))
    long = int(os.environ.get("V41_PJ_LONG_PROMPT", 2052))
    with_long = os.environ.get("V41_PJ_LONG", "0") == "1"
    steps = int(os.environ.get("V41_PJ_STEPS", 8))
    batch = int(os.environ.get("V41_PJ_BATCH", 16))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    ks = sorted({int(k) for k in os.environ.get("V41_PJ_KS", "2,4").split(",")}, reverse=True)
    strict = os.environ.get("V41_PJ_STRICT", "0") == "1"
    traced = os.environ.get("V41_PJ_TRACE", "0") == "1"
    kmax = max(ks)
    layouts = ",".join(f"{k}x{prompt}" for k in ks) + (f",2x{long}" if with_long else "")
    monkeypatch.setenv("TT_V41_CHUNK", str(chunk))
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    # the single prompts eager on the full chunk (reference); ``V41_PJ_TRACE_LENGTHS`` lists lengths to capture
    # as the served catalog does (with TT_V41_SHORT_CHUNK=2048 the references then replay on the short chunk, which
    # differs from the full one: segment 0 is reported, not asserted, against it)
    monkeypatch.setenv("TT_V41_TRACE_PREFILL_LENGTHS", os.environ.get("V41_PJ_TRACE_LENGTHS", ""))
    monkeypatch.setenv("TT_V41_PREFILL_GENERIC", "1")
    monkeypatch.setenv("TT_V41_PREFILL_SLOTS", os.environ.get("TT_V41_PREFILL_SLOTS", "1"))
    monkeypatch.setenv("TT_V41_SHORT_CHUNK", os.environ.get("TT_V41_SHORT_CHUNK", "0"))
    monkeypatch.setenv("TT_V41_PACK", "1")
    monkeypatch.setenv("TT_V41_PACK_LAYOUTS", layouts)
    monkeypatch.setenv("TT_V41_PACK_TRACE", "1" if traced else "0")
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    assert batch >= 3 * kmax, (batch, kmax)
    max_seq_len = int(os.environ.get("V41_PJ_MAX_SEQ", 16384))
    result = {
        "layers": layers,
        "prompt": prompt,
        "ks": ks,
        "layouts": layouts,
        "steps": steps,
        "batch": batch,
        "chunk": chunk,
        "max_seq": max_seq_len,
        "traced": traced,
        "decode_flags": {k: v for k, v in os.environ.items() if k.startswith("V41_DECODE_")},
    }

    t0 = time.time()
    result["dram_before_build"] = _dram_bytes(mesh_device)
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build"] = _dram_bytes(mesh_device)
    assert model.pack_state is not None and len(model.pack_layouts) == len(ks) + int(with_long), model.pack_layouts
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    t0 = time.time()
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup"] = _dram_bytes(mesh_device)
    result["trace_after_warmup"] = _trace_bytes(mesh_device)
    result["pack_traces"] = [[n for _, n in layout] for layout in model.pack_traces]
    result["join_traces"] = len(model.decoder.join_traces)
    result["segment_join_traces"] = sum(1 for k in model.decoder.join_traces if len(k) > 3)
    assert (len(model.pack_traces) > 0) == traced, (traced, model.pack_traces)
    logger.info(
        f"pack-join: built in {result['build_s']} s, warm-up {result['warmup_s']} s; DRAM "
        f"{result['dram_after_warmup'] / 2**30:.2f} GiB, trace {(result['trace_after_warmup'] or 0) / 2**20:.0f} MB, "
        f"{result['join_traces']} join captures ({result['segment_join_traces']} segment)"
    )
    pool = model.decoder.pool
    sample_on_device = DeepseekV41ForCausalLM.SAMPLE_ON_DEVICE

    text = orc.text_tokens(2 * kmax * prompt + 2 * long + steps + 2)[0]
    windows = [text[g * prompt : (g + 1) * prompt].clone() for g in range(2 * kmax)]
    long_windows = [text[2 * kmax * prompt + g * long : 2 * kmax * prompt + (g + 1) * long].clone() for g in range(2)]
    next_key = [1000]

    def new_keys(n: int) -> list[int]:
        keys = list(range(next_key[0], next_key[0] + n))
        next_key[0] += n
        return keys

    def prefill(rows: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        """One ``prefill_forward`` of whole prompts -> the per-row logits [rows, vocab]."""
        width = max(int(t.numel()) for _, t in rows)
        tokens = torch.zeros(len(rows), width, dtype=torch.int32)
        for i, (_, t) in enumerate(rows):
            tokens[i, : t.numel()] = t
        out = model.prefill_forward(
            tokens=tokens,
            prompt_lens=torch.tensor([int(t.numel()) for _, t in rows]),
            page_table=_page_table([b for b, _ in rows]),
            start_pos=torch.tensor([0] * len(rows), dtype=torch.int32),
            kv_cache=kv,
            enable_trace=True,
        )
        assert not torch.isnan(out).any()
        return out[:, 0].clone()

    class _Params:
        def __init__(self, n: int):
            self.temperature, self.top_k, self.top_p = [0.0] * n, [1] * n, [0.0] * n
            self.repetition_penalty = 1.0

    def decode(rows: list[tuple[int, int, int]]) -> list[int]:
        out = model.decode_forward(
            tokens=torch.tensor([[t] for _, t, _ in rows], dtype=torch.int32),
            start_pos=torch.tensor([p for _, _, p in rows], dtype=torch.int32),
            page_table=_page_table([b for b, _, _ in rows]),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_Params(len(rows)) if sample_on_device else None,
        )
        if sample_on_device:
            return [int(x) for x in out]
        return [int(out[i, 0].argmax()) for i in range(len(rows))]

    def join(keys: list[int]) -> list[int]:
        """Join the pending requests ``keys`` now (before any step: the same pool step as the others) -> slots
        (a request the generator joined already, to free its state for the next, keeps its slot)."""
        for key in keys:
            if key in model.pending:
                model._join_pending(key)
            assert key in model.slots, key
        return [model.slots[key] for key in keys]

    def free_run(keys: list[int], first: list[int], lengths: list[int], n: int) -> list[list[int]]:
        """``n`` greedy steps of every key together, each fed its own previous token -> per key its tokens."""
        fed = list(first)
        out = [[] for _ in keys]
        for s in range(n):
            got = decode([(key, fed[i], lengths[i] + s) for i, key in enumerate(keys)])
            for i in range(len(keys)):
                out[i].append(got[i])
                fed[i] = got[i]
        return out

    failures: list[str] = []
    reported: list[str] = []
    checks = result["checks"] = {}
    pack_logits_by_k: dict[int, torch.Tensor] = {}

    def judge(name: str, lines: list[str], must: bool):
        for line in lines:
            (failures if must or strict else reported).append(line)
            (logger.error if must or strict else logger.warning)(line)

    # --- per k: the references, the pack, the joins, the pool rows, the batched and the c1 decode tokens ---
    for k in ks:
        tag = f"k{k}"
        prompts = windows[:k]
        lengths = [prompt] * k
        # the references: each prompt alone (eager, the 5,120-row chunk), joined
        ref_keys = new_keys(k)
        ref_logits = [prefill([(key, p)])[0] for key, p in zip(ref_keys, prompts)]
        assert model.prefill_last.get("pack") is None, model.prefill_last  # the single path
        ref_full = model.prefill_last["chunk"] == chunk  # the reference on the full chunk: segment 0 bit-identical
        checks[f"{tag} reference"] = dict(model.prefill_last)
        ref_slots = join(ref_keys)
        ref_rows = [_slot_rows(mesh_device, pool, slot, prompt) for slot in ref_slots]
        # the pack: the same prompts in one call
        pack_keys = new_keys(k)
        t0 = time.time()
        pack_logits = prefill(list(zip(pack_keys, prompts)))
        checks[f"{tag} pack_call"] = {**model.pack_last, "wall_s": round(time.time() - t0, 3)}
        pack_logits_by_k[k] = pack_logits
        # the same prompt packed under another layout (the kmax pack's first k segments hold these prompts): the
        # single path is deterministic and segment 0 equals it, so segment 0 of every pack is one tensor; the other
        # segments are reported (their TP all-reduce order differs with the layout's total)
        for other, logits_o in pack_logits_by_k.items():
            if other == k:
                continue
            for g in range(min(k, other)):
                rec = _logits_record(logits_o[g], pack_logits[g])
                checks[f"{tag} vs k{other} logits[{g}]"] = rec
                if not rec["equal"]:
                    judge(f"{tag} vs k{other} logits[{g}]", [f"{tag}: segment {g} vs the k{other} pack's {rec}"], must=g == 0)
        assert model.pack_last.get("rows") == list(range(k)), model.pack_last
        assert model.prefill_last.get("mode") == ("replay" if traced else "eager"), model.prefill_last
        assert all(model.pending[key][0] is model.pack_state for key in pack_keys)
        assert [model.pending[key][2]["segment"] for key in pack_keys] == [(g * prompt, prompt, g) for g in range(k)], [
            model.pending[key][2] for key in pack_keys
        ]
        pack_slots = join(pack_keys)
        assert len(set(pack_slots + ref_slots)) == 2 * k, (pack_slots, ref_slots)
        for g, key in enumerate(pack_keys):
            assert model.decoder.position(pack_slots[g]) == prompt, (g, model.decoder.position(pack_slots[g]))
        pack_rows = [_slot_rows(mesh_device, pool, slot, prompt) for slot in pack_slots]
        for g in range(k):
            rec = _logits_record(ref_logits[g], pack_logits[g])
            checks[f"{tag} logits[{g}]"] = rec
            if not rec["equal"]:
                judge(f"{tag} logits[{g}]", [f"{tag}: segment {g} prefill logits {rec}"], must=g == 0 and ref_full)
            lines, record = _compare(f"{tag} rows[{g}]", ref_rows[g], pack_rows[g])
            checks[f"{tag} rows[{g}]"] = record
            judge(f"{tag} rows[{g}]", lines, must=g == 0 and ref_full)
        # the decode tokens: all 2k slots step together, free-running from each prompt's own argmax
        keys = ref_keys + pack_keys
        first = [int(x.argmax()) for x in ref_logits] + [int(x.argmax()) for x in pack_logits]
        tokens = free_run(keys, first, lengths * 2, steps)
        for g in range(k):
            rec = _tokens_record(tokens[g], tokens[k + g])
            checks[f"{tag} tokens[{g}]"] = {**rec, "ref": tokens[g], "pack": tokens[k + g]}
            if not rec["equal"]:
                judge(f"{tag} tokens[{g}]", [f"{tag}: segment {g} decode tokens {rec}"], must=g == 0 and ref_full)
        # the c1 variant: fresh prefills, each slot alone for a few steps (one live row: the c1 capture and the
        # live-user index under FD's flags, bound by the slot's position)
        c1_steps = min(steps, 4)
        ref_keys2, pack_keys2 = new_keys(k), new_keys(k)
        ref_logits2 = [prefill([(key, p)])[0] for key, p in zip(ref_keys2, prompts)]
        # the references are stable: the single path of the same prompt after the decode replays equals its run
        # before them (a padding config built lazily after the decode capture drifted here)
        for g in range(k):
            rec = _logits_record(ref_logits[g], ref_logits2[g])
            checks[f"{tag} reference_stable[{g}]"] = rec
            if not rec["equal"]:
                judge(f"{tag} reference_stable[{g}]", [f"{tag}: reference {g} after the decode replays {rec}"], must=True)
        join(ref_keys2)
        pack_logits2 = prefill(list(zip(pack_keys2, prompts)))
        join(pack_keys2)
        for g in range(k):
            a = free_run([ref_keys2[g]], [int(ref_logits2[g].argmax())], [prompt], c1_steps)[0]
            b = free_run([pack_keys2[g]], [int(pack_logits2[g].argmax())], [prompt], c1_steps)[0]
            rec = _tokens_record(a, b)
            checks[f"{tag} c1_tokens[{g}]"] = {**rec, "ref": a, "pack": b, "variant": model.decoder.last.get("variant")}
            if not rec["equal"]:
                judge(f"{tag} c1_tokens[{g}]", [f"{tag}: segment {g} c1 decode tokens {rec}"], must=g == 0 and ref_full)
        # the pack is deterministic: the second pack's logits equal the first's
        for g in range(k):
            if not torch.equal(pack_logits[g], pack_logits2[g]):
                failures.append(f"{tag}: segment {g} logits differ between two packs of the same prompts")

    # --- ownership: a second pack before any decode joins the first pack's pending segments ---
    tag = f"k{kmax} ownership"
    keys_a, keys_b = new_keys(kmax), new_keys(kmax)
    logits_a = prefill(list(zip(keys_a, windows[:kmax])))
    assert all(key in model.pending for key in keys_a)
    pool_step = pool.step
    logits_b = prefill(list(zip(keys_b, windows[kmax : 2 * kmax])))
    assert not any(key in model.pending for key in keys_a), "the first pack's segments did not join"
    assert all(model.pending[key][0] is model.pack_state for key in keys_b)
    assert pool.step == pool_step
    slots_a = [model.slots[key] for key in keys_a]
    assert all(model.decoder.position(slot) == prompt for slot in slots_a), [model.decoder.position(s) for s in slots_a]
    slots_b = join(keys_b)
    assert len(set(slots_a + slots_b)) == 2 * kmax, (slots_a, slots_b)
    for g in range(kmax):
        if not torch.equal(logits_a[g], pack_logits_by_k[kmax][g]):
            failures.append(f"{tag}: segment {g} logits differ from the earlier pack of the same prompts")
    tokens = free_run(
        keys_a + keys_b, [int(x.argmax()) for x in torch.cat([logits_a, logits_b])], [prompt] * 2 * kmax, steps
    )
    for g in range(kmax):
        rec = _tokens_record(checks[f"k{kmax} tokens[{g}]"]["pack"], tokens[g])
        checks[f"{tag} tokens[{g}]"] = {**rec, "earlier": checks[f"k{kmax} tokens[{g}]"]["pack"], "now": tokens[g]}
        if not rec["equal"]:
            reported.append(f"{tag}: segment {g} joined by the next pack decodes {rec} (other live rows)")
    checks[f"{tag} second_pack_tokens"] = tokens[kmax:]

    # --- captures: a replayed pack against the eager pack of the same prompts, alternating layouts ---
    if traced:
        tag = "replay"
        order = [kmax] + [k for k in ks if k != kmax] + ([2] if with_long else []) + [kmax]
        replays: list[dict] = []
        for turn, k in enumerate(order):
            is_long = with_long and turn == len(order) - 2
            prompts = long_windows if is_long else windows[kmax : kmax + k]
            n = long if is_long else prompt
            keys_r = new_keys(k)
            logits_r = prefill(list(zip(keys_r, prompts)))
            assert model.prefill_last.get("mode") == "replay", model.prefill_last
            replays.append(
                {"k": k, "n": n, **{a: b for a, b in model.pack_traces[model.pack_last["layout"]].last.items()}}
            )
            slots_r = join(keys_r)
            rows_r = [_slot_rows(mesh_device, pool, slot, n) for slot in slots_r]
            # the same prompts eagerly (the captures set aside), at the same pool step
            saved, model.pack_traces = model.pack_traces, {}
            try:
                keys_e = new_keys(k)
                logits_e = prefill(list(zip(keys_e, prompts)))
                assert model.prefill_last.get("mode") == "eager", model.prefill_last
                slots_e = join(keys_e)
            finally:
                model.pack_traces = saved
            for g in range(k):
                rows_e = _slot_rows(mesh_device, pool, slots_e[g], n)
                lines, record = _compare(f"{tag} {turn} k{k} rows[{g}]", rows_e, rows_r[g])
                if not torch.equal(logits_e[g], logits_r[g]):
                    lines.append(
                        f"{tag} {turn} k{k}: segment {g} replayed logits {_logits_record(logits_e[g], logits_r[g])}"
                    )
                checks[f"{tag} {turn} k{k} [{g}]"] = {
                    "rows": record,
                    "logits_equal": bool(torch.equal(logits_e[g], logits_r[g])),
                }
                judge(f"{tag} {turn}", lines, must=True)
            # a single prompt between the packs (the single path, its own state)
            key_s = new_keys(1)[0]
            prefill([(key_s, windows[0])])
            assert model.prefill_last.get("pack") is None and model.prefill_last["mode"] == "eager", model.prefill_last
            join([key_s])
            # one step of everything joined so far keeps the pool moving between the turns
            decode([(key_s, int(windows[0][-1]), prompt)])
        result["replays"] = replays
        result["trace_after_requests"] = _trace_bytes(mesh_device)

    result["dram_after_requests"] = _dram_bytes(mesh_device)
    result["failures"], result["reported"] = failures, reported
    logger.info("V41_PJ_RESULT " + json.dumps(result))
    out = os.environ.get("V41_PJ_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    for trace in model.pack_traces.values():
        trace.release()
    model.decoder.release()
    assert not failures, "\n".join(failures)
