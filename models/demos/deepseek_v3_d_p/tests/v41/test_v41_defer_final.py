# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deferred final chunks through ``DeepseekV41ForCausalLM`` as the plugin drives it (``TT_V41_DEFER_FINAL=1``): a cohort of prompts of mixed length run (a) serially as today (every chunk of
a prompt, then the next prompt; the joins at the first decode step) and (b) bulk / park / restore / final / decode
(every prompt's chunks but the last, in admission order, which parks each one when the next takes the state; then
the finals in the same order, each restoring its prefix from its slot; then decode), in one process, twice (a second
cohort of other prompts reuses the slots of the first through the generator's LRU). Compared per prompt between (a)
and (b): the first token (the argmax of the final chunk's logits), ``V41_DF_STEPS`` (8) greedy decode tokens, and the
slot's pool rows after the decode steps (the compressed KV and index-K rows of every source, every layer's ring, the
candidate source's block pages, on the slot's rank's first chip). Timed per final: the restore (``last_restore``),
the final call's wall, the first decode step's wall (it holds the finals' joins) against the later steps' median.
``V41_DF_PROMPTS`` (``1028,8196,12292,40964,131076,131076``: 1, 2, 3, 8, 26 and 26 chunks), ``V41_DF_LAYERS``
(``all``), ``V41_DF_BATCH`` (8 slots: one a rank, so the second cohort reuses slots), ``V41_DF_MAX_SEQ``
(1,053,696, the served context). Results: ``V41_DF_OUT``."""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table, _trace_bytes
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.pool import BLOCK_ROWS, RING, V41DecodePool
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

FP8 = MlaKvCacheFormat.SCALED_FP8.storage_dtype  # the packed KV rows' dtype (FP8_E4M3)


def _chip_rows(mesh_device, t, a: int, b: int, chip: int) -> torch.Tensor:
    """Rows ``[a, b)`` of ``t`` on ``chip`` as host bytes / values (FP8 bytes through the composer)."""
    shape = list(t.shape)
    whole = a == 0 and b == shape[2] and shape[0] == 1  # the slice would be the tensor itself 
    if whole:
        piece = t
    elif len(shape) == 4 and shape[0] > 1:  # [users, 1, rows, W]: the caller passes absolute rows of user 0's frame
        u, a2 = divmod(a, shape[2])
        piece = ttnn.slice(t, [u, 0, a2, 0], [u + 1, 1, a2 + (b - a), shape[3]])
    else:
        piece = ttnn.slice(t, [0, 0, a, 0], [1, 1, b, shape[3]])
    if t.dtype == FP8:
        host = ttnn.to_torch(piece, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[chip]
    else:
        host = ttnn.to_torch(ttnn.get_device_tensors(piece)[chip])
    if not whole:
        ttnn.deallocate(piece)
    return host.reshape(-1, shape[3]).contiguous()


def _slot_rows(mesh_device, pool: V41DecodePool, slot: int, length: int) -> dict:
    """A prompt of ``length`` tokens in pool slot ``slot`` on its rank's first chip (the bytes as stored)."""
    cfg = pool.config
    rank, u = divmod(slot, pool.users)
    chip = rank * pool.tp
    out = {}
    for src, t in pool.kv.items():
        base = u * pool.slots[src].rows
        if src >= 0:
            n = length // cfg.compress_ratio(src)
            first = base + pool.comp_row(src)
            out[f"kv[{src}]"] = _chip_rows(mesh_device, t, first, first + n, chip)
        for layer in pool.slots[src].layers:
            first = base + pool.ring_row(layer)
            out[f"ring[{layer}]"] = _chip_rows(mesh_device, t, first, first + RING, chip)
    for src, t in pool.index_k.items():
        n = length // cfg.compress_ratio(src)
        cap = pool.slots[src].cap + 32
        out[f"index_k[{src}]"] = _chip_rows(mesh_device, t, u * cap, u * cap + n, chip)
    for src, t in pool.index_k_blocks.items():
        n = -(-(length // cfg.compress_ratio(src)) // BLOCK_ROWS)
        cap = pool.slots[src].cap + 32
        first = u * cap // BLOCK_ROWS
        out[f"index_k_blocks[{src}]"] = _chip_rows(mesh_device, t, first, first + n, chip)
    return out


def _compare(want: dict, got: dict) -> list[str]:
    bad = []
    for k, a in want.items():
        b = got[k]
        if a.shape != b.shape:
            bad.append(f"{k}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        elif not torch.equal(a, b):
            d = a != b
            rows = d.any(-1).nonzero().flatten()
            bad.append(
                f"{k}: {int(d.sum())} of {a.numel()} differ, rows {int(rows.min())}..{int(rows.max())} of {a.shape[0]}"
            )
    return bad


def _trace_region() -> int:
    return 2 * 1024**3


@pytest.mark.timeout(int(os.environ.get("V41_DF_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_DF_TRACE_REGION", _trace_region()))),
    indirect=True,
)
def test_v41_defer_final(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_DF_LAYERS", "all"))
    lengths = [int(x) for x in os.environ.get("V41_DF_PROMPTS", "1028,8196,12292,40964,131076,131076").split(",")]
    steps = int(os.environ.get("V41_DF_STEPS", 8))
    batch = int(os.environ.get("V41_DF_BATCH", 8))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    max_seq_len = int(os.environ.get("V41_DF_MAX_SEQ", 1053696))
    monkeypatch.setenv("TT_V41_CHUNK", str(chunk))
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    monkeypatch.setenv("TT_V41_DEFER_FINAL", "1")
    monkeypatch.setenv("TT_V41_PREFILL_GENERIC", "1")
    monkeypatch.setenv("TT_V41_PREFILL_SLOTS", os.environ.get("TT_V41_PREFILL_SLOTS", "1"))
    monkeypatch.setenv("TT_V41_SHORT_CHUNK", os.environ.get("TT_V41_SHORT_CHUNK", "0"))
    traced = os.environ.get("V41_DF_TRACE", "1") == "1"  # 0: no capture at all 
    if "TT_V41_TRACE_PREFILL_LENGTHS" not in os.environ:
        monkeypatch.setenv("TT_V41_TRACE_PREFILL_LENGTHS", "1028,8196,131076" if traced else "")
        monkeypatch.setenv("TT_V41_TRACE_PREFILL_SLACK", "0")
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    sp = tuple(mesh_device.shape)[0]
    assert batch % sp == 0 and len(lengths) <= batch, (batch, sp, lengths)
    result = {
        "layers": layers,
        "lengths": lengths,
        "steps": steps,
        "batch": batch,
        "chunk": chunk,
        "max_seq": max_seq_len,
        "decode_flags": {k: v for k, v in os.environ.items() if k.startswith("V41_DECODE_")},
        "cohorts": [],
    }

    t0 = time.time()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    result["build_s"] = round(time.time() - t0, 1)
    assert model.decoder.pool.parked, "TT_V41_DEFER_FINAL=1: the parking buffers"
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    t0 = time.time()
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    if traced:
        model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
        model.warmup_model_decode(
            enable_trace=True, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
        )
    result["traced"] = traced
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup"] = _dram_bytes(mesh_device)
    result["trace_after_warmup"] = _trace_bytes(mesh_device)
    result["prefill_traces"] = len(model.prefill_traces)
    logger.info(
        f"defer-final: built in {result['build_s']} s, warm-up {result['warmup_s']} s;"
        f" DRAM {result['dram_after_warmup'] / 2**30:.2f} GiB"
    )

    need = 2 * (sum(lengths) + len(lengths) * (steps + 64))
    try:
        text = orc.text_tokens(need)[0]
    except ValueError as e:
        have = int(str(e).split()[0])
        book = orc.text_tokens(have)[0]
        text = book.repeat(-(-need // have))[:need]

    sample_on_device = DeepseekV41ForCausalLM.SAMPLE_ON_DEVICE

    class _Params:
        def __init__(self, n):
            self.temperature, self.top_k, self.top_p = [0.0] * n, [1] * n, [0.0] * n
            self.repetition_penalty = 1.0

    def prefill_call(block: int, tokens: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, float]:
        t0 = time.time()
        out = model.prefill_forward(
            tokens=tokens.view(1, -1).to(torch.int32),
            prompt_lens=torch.tensor([end]),
            page_table=_page_table([block]),
            start_pos=torch.tensor([start], dtype=torch.int32),
            kv_cache=kv,
            enable_trace=True,
            prompt_total_lens=torch.tensor([int(tokens.numel())]),
        )
        return out[0, 0], time.time() - t0

    def decode(rows: list[tuple[int, int, int]]) -> tuple[list[int], float]:
        t0 = time.time()
        out = model.decode_forward(
            tokens=torch.tensor([[t] for _, t, _ in rows], dtype=torch.int32),
            start_pos=torch.tensor([p for _, _, p in rows], dtype=torch.int32),
            page_table=_page_table([b for b, _, _ in rows]),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_Params(len(rows)) if sample_on_device else None,
        )
        wall = time.time() - t0
        if sample_on_device:
            return [int(x) for x in out], wall
        return [int(out[i, 0].argmax()) for i in range(len(rows))], wall

    def run_order(order: str, prompts: list[tuple[int, torch.Tensor]]) -> dict:
        """``prompts``: (block, tokens). Returns per block the first token, the decode tokens, the slot's rows."""
        rec = {"order": order, "finals": {}, "steps": [], "slots": {}}
        first: dict[int, int] = {}
        t_all = time.time()
        if order == "serial":
            for block, tokens in prompts:
                n = int(tokens.numel())
                for start in range(0, n, chunk):
                    logits, wall = prefill_call(block, tokens, start, min(start + chunk, n))
                    if start + chunk >= n:
                        first[block] = int(logits.argmax())
                        rec["finals"][block] = {"wall_s": round(wall, 3), "mode": model.prefill_last["mode"]}
        else:
            for block, tokens in prompts:  # the bulk chunks, in admission order
                n = int(tokens.numel())
                prefix = (n - 1) // chunk * chunk
                for start in range(0, prefix, chunk):
                    prefill_call(block, tokens, start, start + chunk)
            for block, tokens in prompts:  # the finals, FIFO
                n = int(tokens.numel())
                prefix = (n - 1) // chunk * chunk
                logits, wall = prefill_call(block, tokens, prefix, n)
                first[block] = int(logits.argmax())
                rec["finals"][block] = {
                    "wall_s": round(wall, 3),
                    "mode": model.prefill_last["mode"],
                    "restore": dict(model.decoder.last_restore) if prefix > 0 else None,
                }
                model.decoder.last_restore = {}
        rec["prefill_phase_s"] = round(time.time() - t_all, 2)
        fed = dict(first)
        tokens_out: dict[int, list[int]] = {b: [] for b, _ in prompts}
        for s in range(steps):
            rows = [(b, fed[b], int(t.numel()) + s) for b, t in prompts]
            ids, wall = decode(rows)
            rec["steps"].append(round(wall, 4))
            for (b, _, _), tok in zip(rows, ids, strict=True):
                tokens_out[b].append(tok)
                fed[b] = tok
        rec["first"] = first
        rec["tokens"] = tokens_out
        rec["slot_of"] = {b: model.slots[b] for b, _ in prompts}
        for b, t in prompts:
            rec["slots"][b] = _slot_rows(mesh_device, model.decoder.pool, model.slots[b], int(t.numel()) + steps)
        return rec

    failures = []
    offset = 0
    for cohort in range(2):
        prompts = []
        for i, n in enumerate(lengths):
            block = 100 * (cohort + 1) + i + 1
            prompts.append((block, text[offset : offset + n]))
            offset += n + steps + 64
        serial = run_order("serial", prompts)
        deferred = run_order("defer", prompts)
        rec = {"cohort": cohort, "blocks": [b for b, _ in prompts], "lengths": lengths, "compare": {}}
        for name, r in (("serial", serial), ("defer", deferred)):
            rec[name] = {
                "finals": {str(b): v for b, v in r["finals"].items()},
                "steps_s": r["steps"],
                "step_median_s": round(statistics.median(r["steps"][1:]) if len(r["steps"]) > 1 else r["steps"][0], 4),
                "first_step_s": r["steps"][0],
                "prefill_phase_s": r["prefill_phase_s"],
                "slot_of": {str(b): s for b, s in r["slot_of"].items()},
                "first": {str(b): t for b, t in r["first"].items()},
                "tokens": {str(b): t for b, t in r["tokens"].items()},
            }
        for b, t in prompts:
            cmp = {
                "first_equal": serial["first"][b] == deferred["first"][b],
                "tokens_equal": serial["tokens"][b] == deferred["tokens"][b],
                "slot_mismatches": _compare(serial["slots"][b], deferred["slots"][b]),
                "text_serial": model.tokenizer.decode([serial["first"][b]] + serial["tokens"][b]),
                "text_defer": model.tokenizer.decode([deferred["first"][b]] + deferred["tokens"][b]),
            }
            rec["compare"][str(b)] = cmp
            if not (cmp["first_equal"] and cmp["tokens_equal"] and not cmp["slot_mismatches"]):
                failures.append(f"cohort {cohort} block {b} ({int(t.numel())} tokens): {cmp}")
            logger.info(
                f"defer-final cohort {cohort} block {b} ({int(t.numel())} tokens): first equal {cmp['first_equal']},"
                f" tokens equal {cmp['tokens_equal']}, slot mismatches {len(cmp['slot_mismatches'])}:"
                f" {cmp['text_defer']!r}"
            )
        result["cohorts"].append(rec)
        del serial, deferred

    result["failures"] = failures
    result["dram_end"] = _dram_bytes(mesh_device)
    out = os.environ.get("V41_DF_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    logger.info("defer-final result: " + json.dumps({k: v for k, v in result.items() if k != "cohorts"}))
    for trace in model.prefill_traces.values():
        trace.release()
    model.decoder.release()
    assert not failures, failures
