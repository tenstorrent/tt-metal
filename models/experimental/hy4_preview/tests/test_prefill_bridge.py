# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The hand-over of a prefill's caches to decode, on the 8x4 Galaxy (tt/prefill_bridge.py,
tenstorrent/tt-metal#65893).

The prefill stack (models/demos/deepseek_v3_d_p, layers 0-5, checkpoint weights) prefills a golden
prompt in process; the bridge copies its caches into the decode cache (tt/kv_cache.py); and the
rows decode then reads are compared with HF's own cache rows of that prompt
(reference/hf_harness.py record_decode_golden), mapped into the device frame with tt/frames.py:

* path (c), through the host: per layer and kind (latent k_pass, latent k_rope, index key) the
  PCC, and row by row that position p holds HF's row p and not a neighbour's, another tile's,
  another block's or another round's (prefill_bridge.row_report). The same check is shown to
  fail on these very rows moved by one block. Read two ways: the cache's host read-back, and the
  decode step's own read (cache.locate + cache.gather on a list of positions).
* path (b), on the device: the same rows against path (c). Index keys bit for bit; latents when
  the prefill cache is BF16_RM. With the packed scaled-FP8 cache the latents cannot be converted
  on the device (the bridge says which layers are left), which test_reads_of_the_packed_rows
  documents op by op.
* what each path costs: wall time, and DRAM per chip for the prefill weights, the prefill caches
  and the second copy; test_device_conversion_at_128k times path (b) and path (c) on whole
  stripes of 4,096 rows per chip.

Needs the prefill weight cache ($PREFILL_TTNN_CACHE, default prefill_bridge.DEFAULT_WEIGHT_CACHE)
and the goldens. Run one case per job: every case builds the prefill runtime.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.indexer import full_layers
from models.experimental.hy4_preview.tests.common import GALAXY_MESH
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT, KvCacheConfig, TtHy4KvCache

LAYERS = hf_harness.DEFAULT_NUM_LAYERS
TYPES = tuple(full_layers(LAYERS))
FULL = tuple(n for n, kind in enumerate(TYPES) if kind == "full")
USERS = 2
USER = 1  # the prompt goes into the second slot: slot 0 keeps the stack's warm-up rows
MAX_SEQ = 8192
KV_LORA, ROPE = 512, 64

# Prefill rows against HF (BF16), layers 0-5. The prefill stack's own in-process test measures
# latent >= 0.9982, k_rope >= 0.9998 and index keys >= 0.9986 against its fp32 reference with the
# scaled-FP8 cache and BFP4 routed experts; these floors leave room for HF's bf16. Measured here:
# k_pass 0.99818 (scaled FP8) and 0.99923 (BF16_RM) at the lowest, k_rope 0.99986, index keys 0.99902.
PCC_LATENT = {"scaled_fp8": 0.997, "bf16_rm": 0.998}
PCC_ROPE = 0.999
PCC_INDEX = 0.998
# A single row: every row has to be recognisably its own (row_report counts the misplaced ones);
# the floor only excludes a row that is noise. The lowest single rows measured: k_pass 0.982,
# k_rope 0.9990, index key 0.9932.
PCC_ROW = 0.9
# The same rows through path (b) and path (c) in a BFP8 decode cache: both pack one bf16 row.
PCC_B_VS_C = 0.9999

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), GALAXY_MESH)], indirect=True, ids=["8x4-torus"]
)

# id -> (prompt tokens, the prefill stack's KVPE format, decode cache dtype)
CASES = {
    "P1022-fp8-bfp8": (1022, "scaled_fp8", "bfp8"),
    "P5118-fp8-bfp8": (5118, "scaled_fp8", "bfp8"),
    "P5118-bf16rm-bfp8": (5118, "bf16_rm", "bfp8"),
    "P1022-bf16rm-bf16": (1022, "bf16_rm", "bf16"),
}


def _golden(prompt_len):
    golden = hf_harness.find_decode_golden(LAYERS, prompt_len)
    if golden is None:
        pytest.skip(f"no decode golden for a {prompt_len}-token prompt under {hf_harness.GOLDEN_ROOT}")
    assert tuple(golden["meta"]["indexer_types"]) == TYPES
    return golden


def _decode_cache(mesh_device, dtype, max_users=USERS, max_context=MAX_SEQ):
    cfg = KvCacheConfig(
        num_layers=LAYERS, indexer_layers=FULL, max_users=max_users, max_context=max_context, dtype=dtype
    )
    return TtHy4KvCache(mesh_device, cfg)


def _hf_rows(golden, layer, n):
    """HF's cache rows of positions [0, n) in the device frame: (latent [n, 576], index key [n, 128] | None)."""
    cached = golden["cache"][layer]
    latent = frames.tail_to_device(cached["latent"][:n].float(), ROPE)
    key = frames.index_to_device(cached["index_key"][:n], ROPE) if "index_key" in cached else None
    return latent, key


def _check_rows(label, ref, got, threshold, expect_ok=True, placed=True):
    """PCC, every row's own PCC and, with ``placed``, that no row matches a displaced reference row
    at least as well as its own. ``placed`` is for whole rows: half a latent row is not enough to
    tell a position from another one that holds the same token in a similar context (k_pass
    carries no position; at layer 1 two such rows of HF correlate at 0.9998, above the 0.9992 a
    prefill row reaches against its own), k_rope and the index key's rope half are what do."""
    report = pb.row_report(ref, got)
    logger.info(f"{label}: {pb.format_report(report)}")
    ok = report["min_row"] >= PCC_ROW and report["pcc"] >= threshold and (report["misplaced"] == 0 or not placed)
    if expect_ok:
        assert ok, f"{label}: {pb.format_report(report)} (thresholds: pcc {threshold}, row {PCC_ROW})"
    return ok, report


def _decode_read(mesh_device, cache, layer, user, positions):
    """The rows a decode step reads for a list of positions: cache.locate + cache.gather on every
    chip, each row taken from the one chip that owns it. [K, 576] fp32."""
    k = positions.numel()
    picks = ttnn.from_torch(
        positions.float().reshape(1, 1, 1, k), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.float32
    )
    rows, mine = cache.locate(picks, user=user)
    gathered = cache.gather(layer, rows, LATENT)
    out = torch.zeros(k, cache.cfg.latent_dim)
    owners = torch.zeros(k)
    for g, m in zip(ttnn.get_device_tensors(gathered), ttnn.get_device_tensors(mine)):
        m = ttnn.to_torch(m).float().reshape(k)
        out += ttnn.to_torch(g).float().reshape(k, -1) * m[:, None]
        owners += m
    assert torch.equal(owners, torch.ones(k)), "a position is not owned by exactly one chip"
    return out


@galaxy
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.timeout(0)
def test_handover(mesh_device, case):
    prompt_len, kv_format, cache_dtype = CASES[case]
    golden = _golden(prompt_len)
    ids = golden["input_ids"][:prompt_len]
    label = f"handover {case}"
    numbers = {"dram_open": pb.dram(mesh_device)}

    bridge = pb.PrefillBridge(mesh_device, num_layers=LAYERS, num_users=USERS, max_seq_len=MAX_SEQ, kv_format=kv_format)
    try:
        bridge.build()
        numbers["dram_prefill_built"] = pb.dram(mesh_device)
        n = bridge.prefill(ids, user=USER)
        assert n == prompt_len and bridge.indexer_types == TYPES
        numbers["l1_small_after_prefill"] = pb.l1_small(mesh_device)

        # ---------------- (c) through the host
        before = pb.dram(mesh_device)
        cache_c = _decode_cache(mesh_device, cache_dtype)
        numbers["decode_cache_gb_per_chip"] = pb.dram(mesh_device)["allocated_gb"] - before["allocated_gb"]
        host = bridge.read_host()
        took_c = bridge.fill_from_host(cache_c, USER, host)
        g = torch.Generator().manual_seed(5)
        for layer in range(LAYERS):
            ref_latent, ref_key = _hf_rows(golden, layer, n)
            got = cache_c.to_torch(layer, USER, n, LATENT)
            what = f"{label} (c) L{layer}"
            k_pass = (ref_latent[:, :KV_LORA], got[:, :KV_LORA])
            _check_rows(f"{what} latent k_pass", *k_pass, PCC_LATENT[kv_format], placed=False)
            _check_rows(f"{what} latent k_rope", ref_latent[:, KV_LORA:], got[:, KV_LORA:], PCC_ROPE)
            _check_rows(f"{what} latent row", ref_latent, got, PCC_LATENT[kv_format])
            # what the bridge read from the prefill storage, before the decode cache repacked it
            unpacked = host.latent(USER, layer, n)
            logger.info(
                f"{what} latent, decode cache ({cache_dtype}) against the prefill rows it was filled from: "
                f"pcc={pb.row_report(unpacked, got)['pcc']:.6f} | max|Δ|={(unpacked - got).abs().max().item():.2e}"
            )
            # the control: these rows one block away from where they belong must not pass
            moved = torch.roll(got, 128, dims=0)
            ok, report = _check_rows(f"{what} latent row, moved one block (control)", ref_latent, moved, 0.0, False)
            assert not ok and report["misplaced"] > 0, f"{what}: rows moved by one block pass the row check"
            # the decode step's own read of a spread of positions
            positions = torch.randperm(n, generator=g)[: min(n, 2048)]
            seen = _decode_read(mesh_device, cache_c, layer, USER, positions)
            assert torch.equal(seen, got[positions]), f"{what}: locate + gather read other rows than the read-back"
            _check_rows(
                f"{what} latent as locate + gather reads it", ref_latent[positions], seen, PCC_LATENT[kv_format]
            )
            if layer in FULL:
                got_key = cache_c.to_torch(layer, USER, n, INDEX_KEY)
                _check_rows(f"{what} index key", ref_key, got_key, PCC_INDEX)
                held = host.index_key(USER, layer, n)
                if cache_dtype == "bfp8":
                    assert torch.equal(held, got_key), f"{what}: the index keys changed on their way through the host"
                moved = torch.roll(got_key, 128, dims=0)
                ok, report = _check_rows(f"{what} index key, moved one block (control)", ref_key, moved, 0.0, False)
                assert not ok and report["misplaced"] > 0, f"{what}: index keys moved by one block pass the row check"
        # (c) leaves the rows above the prompt zero, and the other user's stripes untouched
        tail = cache_c.to_torch(0, USER, MAX_SEQ, LATENT)[n:]
        assert torch.count_nonzero(tail) == 0, f"{label}: (c) wrote rows above the prompt"
        assert torch.count_nonzero(cache_c.to_torch(0, 0, MAX_SEQ, LATENT)) == 0, f"{label}: (c) wrote another user"
        logger.info(
            f"MEASURED {label} path (c): read both prefill tensors to the host {took_c['read_host']:.2f} s "
            f"({USERS} users x {LAYERS} layers x {MAX_SEQ} positions), write {took_c['layers']} layers x "
            f"{n} tokens into the decode cache {took_c['write']:.2f} s"
        )

        # ---------------- (b) on the device, against (c)
        cache_b = _decode_cache(mesh_device, cache_dtype)
        first_b = bridge.fill_on_device(cache_b, USER)  # compiles its programs
        took_b = bridge.fill_on_device(cache_b, USER)
        for layer in FULL:
            key_b = cache_b.to_torch(layer, USER, n, INDEX_KEY)
            key_c = cache_c.to_torch(layer, USER, n, INDEX_KEY)
            same = torch.equal(key_b, key_c)
            logger.info(
                f"{label} (b) L{layer} index key against (c): {'bit-equal' if same else 'NOT equal'}, "
                f"max|Δ|={(key_b - key_c).abs().max().item():.2e}"
            )
            assert same, f"{label} L{layer}: the index keys of path (b) differ from path (c)"
        if bridge.device_latents():
            assert took_b["latent_layers_left"] == []
            for layer in range(LAYERS):
                lat_b = cache_b.to_torch(layer, USER, n, LATENT)
                lat_c = cache_c.to_torch(layer, USER, n, LATENT)
                ref_latent, _ = _hf_rows(golden, layer, n)
                what = f"{label} (b) L{layer}"
                same = torch.equal(lat_b, lat_c)
                _check_rows(f"{what} latent against (c)", lat_c, lat_b, PCC_B_VS_C)
                logger.info(
                    f"{what} latent against (c): {'bit-equal' if same else 'not bit-equal'}, "
                    f"max|Δ|={(lat_b - lat_c).abs().max().item():.2e}"
                )
                _check_rows(f"{what} latent row against HF", ref_latent, lat_b, PCC_LATENT[kv_format])
        else:
            assert took_b["latent_layers_left"] == list(range(LAYERS)), took_b
            assert torch.count_nonzero(cache_b.to_torch(0, USER, n, LATENT)) == 0
            logger.info(
                f"{label} (b): latents of layers {took_b['latent_layers_left']} are left for the host path "
                "(no op unpacks a scaled-FP8 row on the device)"
            )
        logger.info(
            f"MEASURED {label} path (b): index keys of {len(FULL)} layers {took_b['index_keys'] * 1e3:.1f} ms, "
            f"latents of {0 if took_b['latent_layers_left'] else LAYERS} layers {took_b['latents'] * 1e3:.1f} ms "
            f"({took_b['rows_per_chip']} rows per chip); the first call, compiling: index keys "
            f"{first_b['index_keys'] * 1e3:.1f} ms, latents {first_b['latents'] * 1e3:.1f} ms"
        )

        # ---------------- memory
        numbers["dram_with_both_copies"] = pb.dram(mesh_device)
        cache_b.deallocate()
        cache_c.deallocate()
        bridge.release_model()
        numbers["dram_prefill_model_released"] = pb.dram(mesh_device)
        bridge.release_caches()
        numbers["dram_prefill_caches_released"] = pb.dram(mesh_device)
        numbers["seconds"] = {k: v for k, v in bridge.seconds.items() if k in ("build_runtime", "compile", "prefill")}
        numbers["seconds"]["release_model"] = bridge.seconds.get("release_model")
        gb = {k: round(v["allocated_gb"], 4) for k, v in numbers.items() if k.startswith("dram_")}
        logger.info(f"MEASURED {label} DRAM allocated per chip, GB: {gb}")
        logger.info(
            f"MEASURED {label} decode cache ({cache_dtype}, {USERS} users x {MAX_SEQ} positions x {LAYERS} layers): "
            f"{numbers['decode_cache_gb_per_chip'] * 1e3:.1f} MB per chip; L1_SMALL after the prefill "
            f"{numbers['l1_small_after_prefill']}; seconds {numbers['seconds']}"
        )
    finally:
        bridge.close()


@galaxy
@pytest.mark.parametrize("kv_format", ["scaled_fp8", "bf16_rm"])
@pytest.mark.timeout(0)
def test_reads_of_the_prefill_rows(mesh_device, kv_format):
    """What ttnn can do today with the prefill storage where it is: the measurements behind the
    choice of hand-over. Every op is tried on one (user, layer) slot of a real prefill and its
    result is logged; an op that refuses is recorded as refusing, not failed. Asserted: the slot
    slice returns the bytes the host read-back holds (the one device read every path relies on)."""
    prompt_len = 1022
    golden = _golden(prompt_len)
    bridge = pb.PrefillBridge(mesh_device, num_layers=LAYERS, num_users=USERS, max_seq_len=MAX_SEQ, kv_format=kv_format)
    results = {}

    def attempt(name, fn):
        t0 = time.perf_counter()
        try:
            value = fn()
            ttnn.synchronize_device(mesh_device)
            results[name] = f"ok ({(time.perf_counter() - t0) * 1e3:.1f} ms first call)"
            return value
        except Exception as error:  # the point of the test is to find out which ops refuse
            results[name] = f"REFUSED: {str(error).strip().splitlines()[0][:160]}"
            return None

    try:
        bridge.build()
        n = bridge.prefill(golden["input_ids"][:prompt_len], user=USER)
        host = bridge.read_host()
        storage = bridge.kv_caches.kvpe.storage
        layer = 1
        slot = pb.kvpe_slot(USER, layer, LAYERS)
        rows = bridge.rows_per_chip
        chips = ttnn.get_device_tensors

        # 1. one slot out of the ND-sharded storage
        sl = attempt("slice of one KVPE slot", lambda: bridge._slot_rows(storage, slot, rows))
        assert sl is not None, results
        back = bridge.read_slabs(sl)[:, 0]  # [32, rows, width]
        same = torch.equal(back.contiguous().view(torch.uint8), host.kvpe[:, slot].contiguous().view(torch.uint8))
        results["slice of one KVPE slot"] += f"; bytes equal to the host read-back: {same}"
        assert same, "the slot slice does not return the stored rows"

        if kv_format == "scaled_fp8":
            width = storage.shape[-1]
            fp8, scales, _ = bridge.unpack_fields(host.kvpe[:, slot])
            results["scales"] = (
                f"host: {scales.unique().numel()} distinct powers of two between {scales.min().item():.3e} "
                f"and {scales.max().item():.3e}; on the device they are 16 fp8-typed bytes of every row and no "
                "op reads them as fp32"
            )
            # 2. the latent bytes as numbers
            lat = attempt("slice of the 512 latent bytes", lambda: ttnn.slice(sl, [0, 0, 0, 0], [1, 1, rows, KV_LORA]))
            if lat is not None:
                name = "typecast fp8_e4m3 -> bfloat16 of the latent bytes"
                cast = attempt(name, lambda: ttnn.typecast(lat, ttnn.bfloat16))
                if cast is not None:
                    got = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in chips(cast)])
                    want = fp8.float()
                    off = got != want
                    results[name] += (
                        f"; equal to the FP8 values (before the scales): {not bool(off.any())}; "
                        f"{off.float().mean().item():.2%} of the values differ, by up to "
                        f"{(got - want).abs().max().item():.3g} on values up to {want.abs().max().item():.3g}"
                    )
            attempt("to_layout(TILE) of a packed slot", lambda: ttnn.to_layout(sl, ttnn.TILE_LAYOUT))
            attempt("typecast fp8_e4m3 -> bfloat16 of a packed slot", lambda: ttnn.typecast(sl, ttnn.bfloat16))
            idx = ttnn.from_torch(
                torch.arange(64, dtype=torch.int32).reshape(1, 64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            attempt(
                "ttnn.embedding (row gather) on a packed slot",
                lambda: ttnn.embedding(idx, ttnn.reshape(sl, [1, 1, rows, width]), layout=ttnn.TILE_LAYOUT),
            )
        else:
            # the BF16_RM slot is a row-major bf16 table: ttnn.embedding can gather from it directly
            k = 2048
            g = torch.Generator().manual_seed(3)
            want = torch.randint(0, rows, (k,), generator=g)
            idx = ttnn.from_torch(
                want.to(torch.int32).reshape(1, k), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
            out = attempt(
                "ttnn.embedding (row gather) on a BF16_RM slot slice",
                lambda: ttnn.embedding(idx, sl, layout=ttnn.TILE_LAYOUT),
            )
            if out is not None:
                got = torch.stack([ttnn.to_torch(t).float().reshape(k, -1) for t in chips(out)])
                equal = torch.equal(got, host.kvpe[:, slot].float()[:, want])
                t0 = time.perf_counter()
                for _ in range(8):
                    ttnn.embedding(idx, sl, layout=ttnn.TILE_LAYOUT)
                ttnn.synchronize_device(mesh_device)
                results["ttnn.embedding (row gather) on a BF16_RM slot slice"] += (
                    f"; rows equal to the stored rows: {equal}; warm {(time.perf_counter() - t0) / 8 * 1e3:.2f} ms per call "
                    f"for {k} rows of a {rows}-row table"
                )
            # ... and on the storage itself, without the slot copy
            users_layers = storage.shape[0]
            table = attempt(
                "reshape of the whole ND-sharded storage to one table",
                lambda: ttnn.reshape(storage, [1, 1, users_layers * rows, storage.shape[-1]]),
            )
            if table is not None:
                idx2 = ttnn.from_torch(
                    (slot * rows + want).to(torch.int32).reshape(1, k),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                )
                out = attempt(
                    "ttnn.embedding on the storage in place",
                    lambda: ttnn.embedding(idx2, table, layout=ttnn.TILE_LAYOUT),
                )
                if out is not None:
                    got = torch.stack([ttnn.to_torch(t).float().reshape(k, -1) for t in chips(out)])
                    equal = torch.equal(got, host.kvpe[:, slot].float()[:, want])
                    results["ttnn.embedding on the storage in place"] += f"; rows equal to the stored rows: {equal}"
        # the index keys are decode's format already
        index = bridge.kv_caches.index
        islot = pb.index_slot(USER, layer, TYPES)
        ik = attempt("slice of one index-key slot", lambda: bridge._slot_rows(index, islot, rows))
        if ik is not None:
            back = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in chips(ik)])
            results[
                "slice of one index-key slot"
            ] += f"; equal to the host read-back: {torch.equal(back, host.index[:, islot].float())}"
        for name, result in results.items():
            logger.info(f"MEASURED prefill storage ({kv_format}), {name}: {result}")
        assert n == prompt_len
    finally:
        bridge.close()


@galaxy
@pytest.mark.timeout(0)
def test_device_conversion_at_128k(mesh_device):
    """Path (b) at a size where the copy is what is timed, not the dispatch of a few ops: the whole
    stripes of one user in a 131,072-position cache (4,096 rows per chip per layer), out of a
    BF16_RM prefill cache into a BFP8 decode cache. Every row is converted, filled or not (the
    prompt is 5,118 tokens; above it the slot holds the stack's warm-up rows and zeros), and every
    row of every stripe is compared with the prefill storage read back on the host. The same
    stripes are then written through the host, for the time of path (c) at this size."""
    max_seq, prompt_len = 131072, 5118
    golden = _golden(prompt_len)
    bridge = pb.PrefillBridge(mesh_device, num_layers=LAYERS, num_users=1, max_seq_len=max_seq, kv_format="bf16_rm")
    try:
        bridge.build()
        with_prefill = pb.dram(mesh_device)["allocated_gb"]
        bridge.prefill(golden["input_ids"][:prompt_len], user=0)
        cache = _decode_cache(mesh_device, "bfp8", max_users=1, max_context=max_seq)
        second_copy = pb.dram(mesh_device)["allocated_gb"] - with_prefill
        bridge.lengths[0] = max_seq  # hand over every row of the stripes
        first = bridge.fill_on_device(cache, 0)
        took = bridge.fill_on_device(cache, 0)
        assert took["rows_per_chip"] == max_seq // 32 and took["latent_layers_left"] == []

        # The same again with a read of every chip queued behind it, in case a synchronize returns
        # before the copies have finished; the read alone is timed first and taken off.
        def read_every_chip():
            for shard in ttnn.get_device_tensors(cache.chip_id):
                ttnn.to_torch(shard)

        read_every_chip()
        t0 = time.perf_counter()
        read_every_chip()
        idle_read = time.perf_counter() - t0
        t0 = time.perf_counter()
        bridge.fill_on_device(cache, 0)
        read_every_chip()
        with_read = time.perf_counter() - t0 - idle_read

        host = bridge.read_host()
        for layer in range(LAYERS):
            stored = host.kvpe[:, pb.kvpe_slot(0, layer, LAYERS)].float()  # [32, rows, 576]
            got = cache.stripes_to_torch(layer, LATENT, user=0)
            score = torch.corrcoef(torch.stack([stored.flatten().double(), got.flatten().double()]))[0, 1].item()
            logger.info(
                f"128K (b) L{layer} latent stripes against the prefill storage: pcc={score:.6f} | "
                f"max|Δ|={(stored - got).abs().max().item():.2e}"
            )
            assert score >= PCC_B_VS_C, f"L{layer}: converted latent stripes pcc={score:.6f}"
            if layer in FULL:
                keys = cache.stripes_to_torch(layer, INDEX_KEY, user=0)
                same = torch.equal(keys, host.index[:, pb.index_slot(0, layer, TYPES)].float())
                logger.info(f"128K (b) L{layer} index-key stripes against the prefill storage: bit-equal {same}")
                assert same, f"L{layer}: converted index-key stripes differ from the prefill storage"

        cache_c = _decode_cache(mesh_device, "bfp8", max_users=1, max_context=max_seq)
        took_c = bridge.fill_from_host(cache_c, 0, host)
        rows = max_seq // 32
        per_layer_ms = 1e3 * took["latents"] / LAYERS
        per_key_ms = 1e3 * took["index_keys"] / len(FULL)
        # one user at 1M: 32,768 rows per chip, 78 latent stripes and 21 index-key stripes
        scale = 32768 / rows
        full_model_s = (78 * per_layer_ms + 21 * per_key_ms) * scale / 1e3
        logger.info(
            f"MEASURED 128K path (b), {rows} rows per chip: latents {per_layer_ms:.2f} ms per layer, index keys "
            f"{per_key_ms:.2f} ms per layer, {1e3 * (took['latents'] + took['index_keys']):.2f} ms for the {LAYERS} "
            f"layers, {1e3 * with_read:.2f} ms when a read of every chip is queued behind them (the read alone, "
            f"{1e3 * idle_read:.2f} ms, taken off); first call, compiling: {1e3 * first['latents']:.0f} ms and "
            f"{1e3 * first['index_keys']:.0f} ms in all; scaled by rows to one user at 1M and 78 layers: "
            f"{full_model_s:.2f} s. Path (c) for the same stripes: read both prefill tensors {took_c['read_host']:.1f} s, "
            f"write {took_c['write']:.1f} s for {LAYERS} layers; scaled the same way: "
            f"{(took_c['read_host'] + took_c['write']) * scale * 78 / LAYERS / 60:.1f} min"
        )
        logger.info(
            f"MEASURED 128K second copy: the BFP8 decode cache of 1 user x {max_seq} positions x {LAYERS} layers takes "
            f"{second_copy * 1e3:.1f} MB per chip (arithmetic {cache.cfg.bytes_per_chip() / 1e6:.1f} MB); the BF16_RM "
            f"prefill storage it is copied from {rows * (LAYERS * 1152 + len(FULL) * 136) / 1e6:.1f} MB per chip"
        )
    finally:
        bridge.close()
