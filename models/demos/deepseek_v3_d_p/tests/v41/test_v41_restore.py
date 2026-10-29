# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The restore of a parked prefix from its decode slot into a prefill state (``TT_V41_DEFER_FINAL``):
``V41DecodePool.restore_ops`` through ``TtV41BatchDecoder.restore``.

For each prompt length of ``V41_RS_PROMPTS`` (8196, 12292, 131076 tokens: 2, 3 and 26 chunks) the prompt's
bulk
chunks (every chunk but the last) are prefilled eagerly through one generic state at the production pool
width, the
state's prefix tensors are snapshotted on one chip of every SP rank (the compressed KV rows and both index-K
forms of
every source, the 40 window carries), the state is joined into a decode slot and parked, the state is
overwritten with
random bytes (and its carries zeroed), one decode step of another slot runs (``V41_RS_STEP=1``: the step
writes every
slot's ring, which is why the carries are parked), the prefix is restored and timed, every restored tensor is
compared
bit for bit with the snapshot, then the prompt's final chunk runs on the restored state and its logits are
compared bit
for bit with the serial path's (the same prompt prefilled in one go on the reset state). Then the synthetic 1M
fill
(``V41_RS_FILL=1``): the state's prefix tensors filled with random bytes to ``V41_RS_FILL_TOKENS`` (1,044,480:
204
chunks), joined, parked, overwritten, restored and compared, the restore timed and the allocator's peak
sampled after
every gather. ``V41_RS_LAYERS`` (``0,1,2,3,20,21,24``: both KV sources and the index sources)
suffice for the bytes; the timing is layout-bound. ``V41_RS_MAX_SEQ`` (1,054,720) is the pool and the state,
``V41_RS_PIECE`` (65,536) the restore's piece. Results: ``V41_RS_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

FP8 = MlaKvCacheFormat.SCALED_FP8.storage_dtype  # the packed KV rows' dtype (FP8_E4M3, not uint8)

TRACE_REGION = 2 * 1024**3
PIECE_ROWS = 65536  # rows per host comparison piece


def _rows(mesh_device, t, a: int, b: int, chips: list[int]) -> torch.Tensor:
    """Rows ``[a, b)`` of ``t`` (``[1, 1, H, W]``, row-major uint8 / bf16 or tiled bf16) on ``chips`` -> host
    ``[len(chips), b - a, W]`` (logical values; FP8 bytes come through the mesh composer)."""
    w = t.shape[3]
    # a slice over the whole extent is the tensor itself at the C++ level, whatever the Python wrapper says:
    # never slice, never deallocate, in that case
    whole = a == 0 and b == t.shape[2]
    piece = t if whole else ttnn.slice(t, [0, 0, a, 0], [1, 1, b, w])
    if t.dtype == FP8:
        host = ttnn.to_torch(piece, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))  # [chips, 1, n, W]
        host = host[chips].reshape(len(chips), b - a, w)
    else:
        every = ttnn.get_device_tensors(piece)
        host = torch.stack([ttnn.to_torch(every[c])[0, 0] for c in chips])
    if not whole:
        ttnn.deallocate(piece)
    return host.contiguous()


def _snapshot(mesh_device, state: V41PrefillState, length: int, chips: list[int]) -> dict:
    """The prefix tensors of ``state`` for ``length`` tokens: per source the compressed KV rows and both
    index-K
    forms in pieces, and every layer's carry."""
    out = {}
    g = state.geometry
    for src in state.kv:
        n = length // state.config.compress_ratio(src)
        for a in range(0, n, PIECE_ROWS):
            b = min(a + PIECE_ROWS, n)
            base = g.kv_row_of_compressed(0)
            out[f"kv[{src}] {a}:{b}"] = _rows(mesh_device, state.kv[src], base + a, base + b, chips)
            out[f"index_k[{src}] {a}:{b}"] = _rows(mesh_device, state.index_k[src], a, b, chips)
            if src in state.index_k_tiled:
                out[f"index_k_tiled[{src}] {a}:{b}"] = _rows(mesh_device, state.index_k_tiled[src], a, b, chips)
    for layer, t in state.window_carry.items():
        out[f"carry[{layer}]"] = _rows(mesh_device, t, 0, WINDOW_SLOT, chips)
    return out


def _compare(want: dict, got: dict) -> tuple[list[str], int]:
    bad, equal = [], 0
    for k, a in want.items():
        b = got[k]
        if torch.equal(a, b):
            equal += 1
            continue
        diff = a != b
        rows = diff.reshape(diff.shape[0], -1, a.shape[-1]).any(-1)
        bad.append(
            f"{k}: {int(diff.sum())} of {a.numel()} elements differ, rows {int(rows.any(0).sum())} of {a.shape[1]},"
            f" chips {rows.any(-1).any(-1).nonzero().flatten().tolist()}"
        )
    return bad, equal


def _random_like(mesh_device, t, rows: int, g, host=None) -> torch.Tensor:
    """Overwrite rows ``[0, rows)`` of ``t`` with random bytes of its dtype and layout (the dirtying before a
    restore), in blocks of 4,096 rows; ``host`` reuses a block (the index-K twin takes the row-major cache's
    values, as the prefill writes them). Returns the host block."""
    w = t.shape[3]
    block_rows = 4096
    if host is None:
        if t.dtype == FP8:  # random floats packed as the format's bytes: enough to differ from the real rows
            host = torch.randn(1, 1, block_rows, w, generator=g) * 0.5
        else:
            host = (torch.randn(1, 1, block_rows, w, generator=g) * 0.5).to(torch.bfloat16)
    dtype = FP8 if t.dtype == FP8 else ttnn.bfloat16
    block = ttnn.from_torch(
        host, device=mesh_device, dtype=dtype, layout=t.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    for first in range(0, rows, block_rows):
        n = min(block_rows, rows - first)
        piece = block if n == block_rows else ttnn.slice(block, [0, 0, 0, 0], [1, 1, n, w])
        ttnn.experimental.slice_write(piece, t, [0, 0, first, 0], [1, 1, first + n, w], [1, 1, 1, 1])
    ttnn.deallocate(block)
    return host


def _dirty(mesh_device, model, state: V41PrefillState, length: int, g, carries: bool) -> None:
    for src in state.kv:
        n = length // state.config.compress_ratio(src)
        _random_like(mesh_device, state.kv[src], state.geometry.kv_row_of_compressed(0) + n, g)
        host = _random_like(mesh_device, state.index_k[src], n, g)
        if src in state.index_k_tiled:
            _random_like(mesh_device, state.index_k_tiled[src], n, g, host=host)
    if carries:
        for t in state.window_carry.values():
            _random_like(mesh_device, t, WINDOW_SLOT, g)
    else:
        model.reset_state(state)


@pytest.mark.timeout(int(os.environ.get("V41_RS_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_restore(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_RS_LAYERS", "0,1,2,3,20,21,24"))
    max_seq = int(os.environ.get("V41_RS_MAX_SEQ", 1054720))
    chunk = int(os.environ.get("V41_RS_CHUNK", 5120))
    prompts = [int(x) for x in os.environ.get("V41_RS_PROMPTS", "8196,12292,131076").split(",") if x]
    slots = [int(x) for x in os.environ.get("V41_RS_SLOTS", "0,13,31,22").split(",") if x]
    piece = int(os.environ.get("V41_RS_PIECE", 65536))
    fill = os.environ.get("V41_RS_FILL", "1") == "1"
    fill_tokens = int(os.environ.get("V41_RS_FILL_TOKENS", 1044480))
    step = os.environ.get("V41_RS_STEP", "1") == "1"
    repeat = int(os.environ.get("V41_RS_REPEAT", 2))
    out_path = os.environ.get("V41_RS_OUT")
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    sp, tp = tuple(mesh_device.shape)
    chips = [r * tp for r in range(sp)]  # one chip of every SP rank
    assert max_seq % chunk == 0 and fill_tokens % chunk == 0 and fill_tokens + chunk <= max_seq
    result = {
        "layers": layers,
        "max_seq": max_seq,
        "chunk": chunk,
        "prompts": prompts,
        "slots": slots,
        "piece": piece,
        "mesh": [sp, tp],
        "cases": [],
    }
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    text = orc.text_tokens(max(prompts) + 64)[0]

    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    decoder = TtV41BatchDecoder(
        model, root, users_per_rank=4, trace=False, warmup=0, sample_on_device=True, max_seq_len=max_seq, parking=True
    )
    state = model.new_state(generic=True)
    assert state.index_k_tiled, "TT_V41_PREFILL_MASK_ONCE=1: the tiled index-K twin is part of what is restored"
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build"] = _dram_bytes(mesh_device)
    result["parking_slots"] = len(decoder.pool.parked)
    logger.info(
        f"restore: built {len(layers)} layers in {result['build_s']} s;"
        f" DRAM {result['dram_after_build'] / 2**30:.2f} GiB"
    )

    stepper = None
    if step:
        # a short prompt in slot 1 (rank 0): its decode step writes every slot's ring at step % 128
        short = text[:255]
        _, st = model.prefill(short, logit_positions=1)
        decoder.join(1, short, st)
        for group in (st.kv, st.index_k, st.window_carry):
            for t in group.values():
                ttnn.deallocate(t)
        stepper = 1

    def sync() -> None:
        ttnn.synchronize_device(mesh_device)

    g = torch.Generator().manual_seed(148)
    failures = []

    def restore_case(name: str, slot: int, length: int, tokens: torch.Tensor | None, carries_random: bool) -> dict:
        case = {"name": name, "slot": slot, "length": length, "rank": slot // 4}
        sync()
        t0 = time.time()
        before = _snapshot(mesh_device, state, length, chips)
        case["snapshot_s"] = round(time.time() - t0, 1)
        case["snapshot_bytes"] = sum(v.numel() * v.element_size() for v in before.values()) // len(chips)
        t0 = time.time()
        decoder.join(slot, tokens if tokens is not None else torch.zeros(length, dtype=torch.int64), state)
        decoder.park(slot, state)
        sync()
        case["join_park_s"] = round(time.time() - t0, 3)
        _dirty(mesh_device, model, state, length, g, carries_random)
        sync()
        dirty = _snapshot(mesh_device, state, length, chips)
        bad, _ = _compare(before, dirty)
        case["dirtied_keys"] = len(bad)
        if len(bad) != len(before):
            logger.warning(f"{name}: the dirtying left {len(before) - len(bad)} of {len(before)} keys unchanged")
        del dirty
        if stepper is not None:
            t0 = time.time()
            try:
                decoder.decode({stepper: 1}, {stepper: (1, 1.0, 0.0)})
                sync()
                case["step_s"] = round(time.time() - t0, 3)
            except Exception as e:  # the step is a witness (the rings overwritten), not the gate
                logger.warning(f"{name}: the decode step of slot {stepper} failed: {e}")
                case["step_error"] = str(e)[:300]
        dram0 = _dram_bytes(mesh_device)
        peak = [dram0]

        def probe(tag: str) -> None:
            peak.append(_dram_bytes(mesh_device))

        # the restore twice: the first pays the programs' creation and the kernels' JIT at these shapes, the
        # second is the warm cost (idempotent: the same bytes again)
        case["restore_runs"] = []
        for rep in range(repeat):
            sync()
            t0 = time.time()
            decoder.restore(slot, state, length, piece=piece, probe=probe if rep == 0 else None)
            t1 = time.time()
            sync()
            t2 = time.time()
            case["restore_runs"].append({"issue_s": round(t1 - t0, 4), "wall_s": round(t2 - t0, 4)})
        case["restore_issue_s"] = case["restore_runs"][-1]["issue_s"]
        case["restore_wall_s"] = case["restore_runs"][-1]["wall_s"]
        case["restore"] = decoder.last_restore
        case["dram_before_restore"] = dram0
        case["dram_peak_restore"] = max(peak)
        case["dram_after_restore"] = _dram_bytes(mesh_device)
        assert state.start == length and all(c is None for c in state.compressor_carry.values())
        t0 = time.time()
        after = _snapshot(mesh_device, state, length, chips)
        bad, equal = _compare(before, after)
        case["compare_s"] = round(time.time() - t0, 1)
        case["keys"] = len(before)
        case["equal"] = equal
        case["mismatches"] = bad
        if bad:
            failures.append(f"{name}: {len(bad)} of {len(before)} keys differ: {bad[:5]}")
        logger.info(
            f"restore {name}: slot {slot} ({length} tokens): restore {case['restore_wall_s'] * 1e3:.1f} ms wall "
            f"(issue {case['restore_issue_s'] * 1e3:.1f}; runs {case['restore_runs']}), {decoder.last_restore}; "
            f"{equal}/{len(before)} keys equal; DRAM peak +{(max(peak) - dram0) / 2**20:.0f} MB"
        )
        return case

    for n, slot in zip(prompts, slots, strict=False):
        prefix = (n - 1) // chunk * chunk
        assert prefix > 0, n
        tokens = text[:n]
        model.reset_state(state)
        t0 = time.time()
        model.prefill_into(tokens, state, start=0, end=prefix)
        sync()
        case = restore_case(f"prompt {n}", slot, prefix, tokens[:prefix], carries_random=False)
        case["bulk_s"] = round(time.time() - t0, 1)
        # the final chunk on the restored state against the serial path's
        t0 = time.time()
        restored = model.prefill_into(tokens, state, start=prefix, end=n)
        sync()
        case["final_s"] = round(time.time() - t0, 2)
        model.reset_state(state)
        serial = model.prefill_into(tokens, state, start=0, end=n)
        sync()
        case["final_logits_equal"] = bool(torch.equal(restored, serial))
        case["final_argmax"] = [int(restored.argmax()), int(serial.argmax())]
        case["final_max_abs_diff"] = float((restored.float() - serial.float()).abs().max())
        if not case["final_logits_equal"]:
            failures.append(f"prompt {n}: the final's logits differ after the restore ({case['final_max_abs_diff']})")
        logger.info(
            f"restore prompt {n}: final after restore equal {case['final_logits_equal']} argmax {case['final_argmax']}"
        )
        result["cases"].append(case)

    if fill:
        slot = slots[len(prompts)] if len(slots) > len(prompts) else 22
        model.reset_state(state)
        t0 = time.time()
        _dirty(mesh_device, model, state, fill_tokens, g, carries=True)
        state.start = fill_tokens
        sync()
        case = restore_case(f"fill {fill_tokens}", slot, fill_tokens, None, carries_random=True)
        case["fill_s"] = round(time.time() - t0, 1)
        result["cases"].append(case)

    result["failures"] = failures
    result["dram_end"] = _dram_bytes(mesh_device)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
    logger.info("restore result: " + json.dumps({k: v for k, v in result.items() if k != "cases"}))
    for case in result["cases"]:
        logger.info("restore case: " + json.dumps({k: v for k, v in case.items() if k != "mismatches"}))
    assert not failures, failures
