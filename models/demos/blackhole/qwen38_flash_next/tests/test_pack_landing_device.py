# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3d-3 device tests: landing a multi-user pack's users directly into decode lanes (``ttnn/pack_landing.py``).

The adapter is built in batch mode (B=8, block 64) with ``QWEN38_PREFILL_PACK=1``: the pack state, one eager pack per bucket and
the landing programs are allocated and compiled in the chain's warm hook, the pack traces are captured after the lane trace.

``land_<S>``  (A) every landed family bitwise: lane ``l`` read back after ``admit_pack`` against a host (torch) re-implementation of
              the landing transform applied to the read-back pack sources (all four devices, all 48 layers), every other lane (and the
              unlanded KV pool rows of the lanes' page window) unchanged, the position rows and page rows, and no new program.
              Lengths cover ``P % 4 = 0..3``, ``P % 32`` in {0, 1, 31}, ``P < 4``, ``P < 9``, ``P = S``, slots 0 and G - 1, lanes 0 and 7.
``sem``       (B) lane state after the single path (chain prefill + forced step + import) against the pack landing, same prompts:
              per-family relative error, the structure (zero / nonzero rows and slots) and the page-table rows.
``decode``    (C) greedy 32 tokens of 20 packed prompts (all buckets) against the single path's; pass-key retrieval.
``leak``      (C) lane state and 32 decoded tokens of user A in pack (A, B, ..) against pack (A, B', ..): bitwise equal.
``timing``    (D) landing ms per user, pack total per bucket, the single path's TTFT.
``dram_*``    (E) free DRAM and pool blocks with the knob off / on at B = 8 and B = 32.
``mixed_single_after_pack`` (P3d-4) a single-path prefill right after a pack (S = 128 / 256 / 512 / 1024) against the same prefill
              without it: the lane state a decode reads, the KV rows [0, P) and the first-token logits, bitwise (the dead rows of the
              single path's compressed cache are history dependent in any order and are reported, not gated).
``mixed_pack_after_single`` (P3d-4) a pack right after a single prefill against the same pack alone: logits, whole lane state, KV rows.
``pack_bucket_256_against_128`` (P3d-4) 8 users with P <= 128 in an S = 256 pack against an S = 128 pack: phase costs and logits.

Run:  pytest .../test_pack_landing_device.py -v -s -k land_s128   (one id per process)
Env:  as ``test_prefill_pack_device`` (QWEN38_HARDWARE_PROFILE, TT_METAL_TRACE_ALLOC_TRACKING=1, MODEL_WEIGHTS_DIR, QWEN38_CACHE_ROOT,
      TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0).  ``P3D3_OUT`` names a directory for the JSON results.
"""

import contextlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import (
    initialize,
    plugin_num_blocks,
    plugin_warmups,
)
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import WARM_TOKEN_ID, lane_page_count
from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

OUT = os.environ.get("P3D3_OUT")
LANES = 8
BLOCK = 64
BUCKETS = {"s128": 128, "s256": 256, "s512": 512, "s1024": 1024}
PAGE_WINDOW = (
    160  # pages 1 .. 159 hold every lane of the exactness test (the KV pool is read back only over this window)
)
FAMILIES = ("recurrent", "conv", "compressed", "staging", "ring", "ple")
VOCAB = TOKENIZER_SIZE
# P3d-2 profile.json medians: K/V ~0.06-0.08, GDN ~0.04-0.05 (growing with depth); anything past this is flagged
FLAG_REL_ERROR = 0.3
PLACEMENT_BUG_REL_ERROR = 0.9
# pack contents of the exactness test per bucket (a list of packs, each a list of lengths, one per slot): P % 4 = 0..3,
# P % 32 in {0, 1, 31}, P < 4, P < 9, P = S, the first and the last slot (the last slot of the 16-slot bucket is a full segment)
PACKS = {
    128: [[1, 2, 3, 4, 5, 8, 9, 31, 32, 33, 63, 64, 65, 97, 127, 128]],
    256: [[1, 4, 31, 32, 33, 126, 255, 256], [2, 5, 63, 64, 65, 129, 200, 254]],
    512: [[2, 287, 321, 512], [3, 31, 33, 511], [9, 128, 257, 100]],
    1024: [[1, 1024], [6, 1022], [31, 1001], [33, 999], [8, 513]],
}


def save(name: str, payload) -> None:
    if OUT:
        (Path(OUT) / name).write_text(json.dumps(payload, indent=1, default=str))


def free_dram(mesh) -> int:
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.num_banks) * int(view.total_bytes_free_per_bank)


@contextlib.contextmanager
def misses_allowed(mesh):
    mesh.set_program_cache_misses_allowed(True)
    try:
        yield
    finally:
        mesh.set_program_cache_misses_allowed(False)


def bits(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.contiguous()
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


# ----------------------------------------------------------------------------------------------------------------- harness


class Landing:
    """The batch adapter with the pack on (or off), its decoder, and the host helpers every case uses."""

    def __init__(self, mesh_device, *, lanes: int = LANES, pack: bool = True, buckets=pack_module.PACK_BUCKETS) -> None:
        self.mesh = mesh_device
        self.checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
        self.lanes = lanes
        if pack:
            os.environ["QWEN38_PREFILL_PACK"] = "1"
            os.environ["QWEN38_PACK_BUCKETS"] = ",".join(str(b) for b in buckets)
        else:
            os.environ.pop("QWEN38_PREFILL_PACK", None)
            os.environ.pop("QWEN38_PACK_BUCKETS", None)
        mesh_device.enable_program_cache()
        started = time.time()
        self.free_before_open = free_dram(mesh_device)
        self.model = initialize(mesh_device, self.checkpoint, lanes)
        self.free_after_chain = free_dram(mesh_device)
        self.blocks = plugin_num_blocks(lanes, BLOCK)
        self.model.allocate_kv_cache((self.blocks, 1, BLOCK, 128), ttnn.bfloat16, 12)
        self.free_after_open = free_dram(mesh_device)
        self.open_s = time.time() - started
        self.decoder = self.model.decoder
        plugin_warmups(self.model, lanes, BLOCK, self.blocks)
        self.chain = self.model.chain
        self.tm = self.chain.built_target.model
        self.layers = self.tm.layers
        self.template = None
        self.rng = random.Random(1234)

    # -- pages / lanes ---------------------------------------------------------------------------------------------

    def window_pages(self, count: int, taken: set) -> list[int]:
        free = [p for p in range(1, PAGE_WINDOW) if p not in taken]
        self.rng.shuffle(free)
        pages = free[:count]
        taken.update(pages)
        return pages

    def release_all(self) -> None:
        for lane in range(self.lanes):
            if self.decoder.positions[lane] is not None or self.decoder.page_rows[lane]:
                self.decoder.release_lane_external(lane)
        self.decoder.sync_page_table()

    # -- prompts ---------------------------------------------------------------------------------------------------

    def random_prompt(self, length: int, seed: int) -> list[int]:
        rng = random.Random(seed)
        return [rng.randrange(1000, 100000) for _ in range(length)]

    def keyed_prompts(self, lengths):
        """Chat prompts with a pass key and needle question of EXACT lengths (>= 120), as the contract test builds them."""

        template = Qwen38OfficialChatTemplate(self.checkpoint)
        config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
        out = []
        for n in lengths:
            ids = build_prompt_ids(
                template.tokenizer,
                lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
                config,
                text,
                n,
            )[0]
            assert len(ids) == n, (len(ids), n)
            out.append(ids)
        return template.tokenizer, config, out

    # -- the two admissions ----------------------------------------------------------------------------------------

    def single_admit(self, lane: int, ids: list[int], pages: list[int] | None = None) -> int:
        """Single path: chain prefill + forced step of the last token, the importer into ``lane``; returns t0."""

        d = self.decoder
        out = self.model._prefill_one(ids, greedy_token=True)
        if pages is None:
            pages = d.allocator.allocate(lane_page_count(len(ids), d.decode_steps, d.paging.page_size))
        d.import_lane(self.model, lane, len(ids), self.model.slot.ple_context, pages)
        return int(out.reshape(-1)[0])

    def pack_admit(self, lanes, prompts, rows, users=None, pages=None) -> torch.Tensor:
        return self.decoder.admit_pack(lanes, prompts, rows, pages=pages, users=users)

    def free_lane(self, lane: int, external: bool = True) -> None:
        self.decoder.release_lane_external(lane)

    def decode(self, first: dict[int, int], n: int = 32) -> dict[int, list[int]]:
        """Greedy decode of the landed lanes in ``first`` (lane -> t0), ``n`` tokens each (t0 included)."""

        d = self.decoder
        outs = {lane: [t] for lane, t in first.items()}
        for _ in range(n - 1):
            idle = [lane for lane in range(self.lanes) if lane not in outs]
            d.sync_page_table()
            d.park_idle(idle)
            tokens = [outs[lane][-1] if lane in outs else WARM_TOKEN_ID for lane in range(self.lanes)]
            result = d.step(tokens)
            for lane in outs:
                outs[lane].append(int(result[lane]))
        return outs

    # -- reading the lane state ------------------------------------------------------------------------------------

    def lane_bits(self) -> dict:
        """Every lane family tensor as int bit patterns per device: ``{family: [[per device tensor]]}``."""

        layout = self.decoder.pager_l.layout
        out = {}
        for name in FAMILIES:
            out[name] = [
                [bits(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(ttnn.from_device(tensor))]
                for tensor in getattr(layout, name)
            ]
        return out

    def pool_bits(self, rows: int) -> list[list[torch.Tensor]]:
        """The first ``rows`` rows of every QSA layer's KV pool, per device."""

        out = []
        for index, layer in enumerate(self.layers):
            attention = self.decoder.ls.layers[index].attention
            if not hasattr(attention, "packed_kv_cache"):
                continue
            with corruptible_allocation_scope(self.mesh):  # a test-side transient: consumed before any replay
                sliced = ttnn.slice(
                    attention.packed_kv_cache, (0, 0, 0, 0), (1, 1, rows, 512), memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                out.append([bits(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(sliced)])
                ttnn.deallocate(sliced)
        return out

    def qsa_indices(self) -> list[int]:
        return [i for i, layer in enumerate(self.decoder.pack_state.layers) if layer is None]

    def gdn_indices(self) -> list[int]:
        return [i for i, layer in enumerate(self.decoder.pack_state.layers) if layer is not None]

    def pack_sources(self) -> dict:
        """The pack's landing sources of the last replay, per device, as bit patterns."""

        ps = self.decoder.pack_state
        ttnn.synchronize_device(self.mesh)

        def devices(tensor):
            return [bits(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(tensor)]

        src = {"states": {}, "conv": {}, "compressed": {}, "kv": {}, "kept_kv": {}, "kept_raw": {}}
        for i in self.gdn_indices():
            src["states"][i] = [devices(t) for t in ps.layers[i].states]
            src["conv"][i] = devices(ps.layers[i].conv_rows)
        for i in self.qsa_indices():
            generic = self.chain.state.layers[i].attention
            kept = ps.chunk_state.layers[i].attention
            src["compressed"][i] = devices(generic.compressed_index_cache)
            src["kv"][i] = devices(generic.packed_kv_cache)
            src["kept_kv"][i] = devices(kept.kept_kv)
            src["kept_raw"][i] = devices(kept.kept_raw)
        src["ple"] = [devices(t) for t in ps.ple_snapshots]
        return src


@pytest.fixture
def landing_factory(mesh_device):
    built = []

    def make(**kwargs):
        h = Landing(mesh_device, **kwargs)
        built.append(h)
        return h

    yield make
    for h in built:
        h.model.release_persistent_capture()


# ----------------------------------------------------------------------------------------------- A: the exact transform


def expected_lane_state(h: Landing, before: dict, src: dict, rows: int, wave: list) -> dict:
    """The lane tensors after the landing, computed on the host: ``before`` with the landed lanes' content replaced by the
    transform of the read-back sources (``wave`` = ``[(lane, user, length)]``).  Written from the documented layout
    (``ttnn/pack.py``), not from the select images."""

    bucket = pack_module.pack_bucket(rows)
    expected = {name: [[t.clone() for t in per_device] for per_device in tensors] for name, tensors in before.items()}
    gdn, qsa = h.gdn_indices(), h.qsa_indices()
    for lane, user, length in wave:
        launch, head0 = pack_module.gdn_state_slot(bucket, user)
        base = user * rows
        for g, layer in enumerate(gdn):
            for d in range(4):
                expected["recurrent"][g][d][lane] = src["states"][layer][launch][d][head0 : head0 + 12]
                # ring slot j <- the conv-input row of token P - 4 + j (zero before the start): conv_rows rows
                # 32 u + 0 .. 3 hold the rows P - 3, P - 2, P - 1, P - 4
                conv = src["conv"][layer][d].reshape(-1, src["conv"][layer][d].shape[-1])
                for slot in range(4):
                    token = length - 4 + slot
                    row = conv[32 * user + {1: 0, 2: 1, 3: 2, 0: 3}[slot]]
                    if token < 0:
                        assert not row.any(), "a conv row before the sequence start must be zero"
                    expected["conv"][4 * g + slot][d][0, 0, lane] = row
        for q, layer in enumerate(qsa):
            for d in range(4):
                comp = src["compressed"][layer][d].reshape(-1, 128)
                tgt = expected["compressed"][q][d]  # [B, 1, R, 128]
                blocks = rows // 4
                tgt[lane, 0] = 0  # the whole lane cache: rows past the user's segment are written as zero too
                complete = length // 4
                tgt[lane, 0, :complete] = comp[user * blocks : user * blocks + complete]
                kept_kv = src["kept_kv"][layer][d].reshape(-1, src["kept_kv"][layer][d].shape[-1])
                tile = (length - 1) // 32
                n = (length - 1) % 32 + 1
                st = expected["staging"][q][d]  # [1, B, 32, 512]
                st[0, lane] = 0
                st[0, lane, :n] = kept_kv[base + 32 * tile : base + 32 * tile + n]
                raw = src["kept_raw"][layer][d].reshape(-1, src["kept_raw"][layer][d].shape[-1])
                block = (length - 1) // 4
                m = (length - 1) % 4 + 1
                rg = expected["ring"][q][d]
                rg[0, lane] = 0
                rg[0, lane, :m] = raw[base + 4 * block : base + 4 * block + m]
        first = pack_module.ple_history_first_row(length)
        for d in range(4):
            window = src["ple"][user][d].reshape(-1, 4, src["ple"][user][d].shape[-1])  # [137, 4, 640]
            for k in range(9):
                expected["ple"][k][d][0, lane] = window[first + k]
    return expected


def compare_lane_state(after: dict, expected: dict, wave: list) -> dict:
    """Mismatching (family, index, device) with the lanes that differ, split into the landed and the other lanes."""

    lanes = {lane for lane, _u, _n in wave}
    bad = {"landed": [], "other": []}
    axis = {"recurrent": 0, "conv": 2, "compressed": 0, "staging": 1, "ring": 1, "ple": 1}
    total = 0
    for name in FAMILIES:
        for index, (ta, te) in enumerate(zip(after[name], expected[name])):
            for d, (a, e) in enumerate(zip(ta, te)):
                total += 1
                if torch.equal(a, e):
                    continue
                ax = axis[name]
                differing = {u for u in range(a.shape[ax]) if not torch.equal(a.select(ax, u), e.select(ax, u))}
                if differing & lanes:
                    bad["landed"].append((name, index, d, sorted(differing & lanes)))
                if differing - lanes:
                    bad["other"].append((name, index, d, sorted(differing - lanes)))
    bad["compared"] = total
    return bad


def check_kv_pages(h: Landing, before_pool, after_pool, src: dict, rows: int, wave: list) -> dict:
    """Lane pages hold the user's dense KV rows ``[0, round_up(P, 32))``; every other pool row of the window is unchanged."""

    written = set()
    bad = []
    d_ = h.decoder
    for q, layer in enumerate(h.qsa_indices()):
        for dev in range(4):
            pool_after = after_pool[q][dev].reshape(-1, 512)
            pool_before = before_pool[q][dev].reshape(-1, 512)
            dense = src["kv"][layer][dev].reshape(-1, 512)
            touched = set()
            for lane, user, length in wave:
                pages = d_.page_rows[lane]
                for t in range(-(-length // 32) * 32):
                    row = pages[t >> 6] * BLOCK + (t & (BLOCK - 1))
                    touched.add(row)
                    if not torch.equal(pool_after[row], dense[user * rows + t]):
                        bad.append(("kv_row", q, dev, lane, t))
                        break
            untouched = torch.tensor([r for r in range(pool_after.shape[0]) if r not in touched])
            if not torch.equal(pool_after[untouched], pool_before[untouched]):
                bad.append(("kv_other_rows", q, dev))
            written |= touched
    return {"bad": bad, "rows_written_per_layer_device": len(written)}


def run_exact_case(h: Landing, rows: int) -> dict:
    d = h.decoder
    pc = lambda: resident_decode.program_cache_count(h.mesh)  # noqa: E731
    results = {"rows": rows, "waves": []}
    for pack_index, lengths in enumerate(PACKS[rows]):
        users = len(lengths)
        prompts = [h.random_prompt(n, seed=100 * rows + 10 * pack_index + i) for i, n in enumerate(lengths)]
        other = [h.random_prompt(n, seed=777 + 100 * rows + 10 * pack_index + i) for i, n in enumerate(lengths)]
        slots = list(range(users))
        waves = [slots[i : i + LANES] for i in range(0, len(slots), LANES)]
        for w, wave_users in enumerate(waves):
            h.release_all()
            # stale state: a pack of other prompts (or this pack's own) fills the lanes, then they are freed again
            filler = other if w % 2 == 0 else prompts
            taken: set = set()
            stale_pages = [h.window_pages(lane_page_count(len(filler[u]), 32, BLOCK), taken) for u in wave_users]
            h.pack_admit(list(range(len(wave_users))), filler, rows, users=wave_users, pages=stale_pages)
            h.release_all()
            # the wave lands in reverse lane order (slot 0 -> lane len - 1; with 8 users: slot 0 -> lane 7, last -> lane 0)
            lanes = [LANES - 1 - i for i in range(len(wave_users))]
            taken = set()
            pages = [h.window_pages(lane_page_count(lengths[u], 32, BLOCK), taken) for u in wave_users]
            with misses_allowed(h.mesh):
                before = h.lane_bits()
                before_pool = h.pool_bits(PAGE_WINDOW * BLOCK)
            entries_before = pc()
            logits = h.pack_admit(lanes, prompts, rows, users=wave_users, pages=pages)
            entries_after = pc()
            wave = [(lane, u, lengths[u]) for lane, u in zip(lanes, wave_users)]
            with misses_allowed(h.mesh):
                after = h.lane_bits()
                after_pool = h.pool_bits(PAGE_WINDOW * BLOCK)
                src = h.pack_sources()
            expected = expected_lane_state(h, before, src, rows, wave)
            bad = compare_lane_state(after, expected, wave)
            kv = check_kv_pages(h, before_pool, after_pool, src, rows, wave)
            position_bad = [
                (lane, d.positions[lane], int(d.ls.position.positions[lane]))
                for lane, u, n in wave
                if d.positions[lane] != n or int(d.ls.position.positions[lane]) != n
            ]
            table = ttnn.to_torch(ttnn.get_device_tensors(d.ls.page_table)[0])[0, 0]
            table_bad = [
                lane
                for lane, u, n in wave
                if [int(v) for v in table[lane, : len(d.page_rows[lane])]] != d.page_rows[lane]
            ]
            results["waves"].append(
                {
                    "pack": pack_index,
                    "users": wave_users,
                    "lanes": lanes,
                    "lengths": [lengths[u] for u in wave_users],
                    "tensors_compared": bad["compared"],
                    "landed_mismatches": bad["landed"][:8],
                    "other_lane_mismatches": bad["other"][:8],
                    "n_landed_mismatches": len(bad["landed"]),
                    "n_other_mismatches": len(bad["other"]),
                    "kv_bad": kv["bad"][:8],
                    "positions_bad": position_bad,
                    "page_table_bad": table_bad,
                    "new_programs": entries_after - entries_before,
                    "logits_finite": bool(torch.isfinite(logits[:, :VOCAB]).all()),
                    "last_pack": dict(d.last_pack),
                }
            )
            logger.info(f"LAND pack {pack_index} wave {w}: {json.dumps(results['waves'][-1], default=str)}")
    return results


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("bucket", [pytest.param(name, id=f"land_{name}") for name in BUCKETS])
@pytest.mark.timeout(5400)
def test_landing_exact(mesh_device, landing_factory, bucket):
    h = landing_factory()
    results = run_exact_case(h, BUCKETS[bucket])
    save(f"land_{bucket}.json", results)
    for wave in results["waves"]:
        assert not wave["landed_mismatches"], wave["landed_mismatches"]
        assert not wave["other_lane_mismatches"], wave["other_lane_mismatches"]
        assert not wave["kv_bad"], wave["kv_bad"]
        assert not wave["positions_bad"] and not wave["page_table_bad"]
        assert wave["new_programs"] == 0, wave["new_programs"]
        assert wave["logits_finite"]


# ------------------------------------------------------------------------------------------- B: against the single path


def as_float(tensor_bits: torch.Tensor) -> torch.Tensor:
    return (
        tensor_bits.view(torch.float32) if tensor_bits.dtype == torch.int32 else tensor_bits.view(torch.bfloat16)
    ).to(torch.float64)


def rel_error(a_bits: torch.Tensor, b_bits: torch.Tensor) -> float:
    """``||a - b|| / ||b||`` of two bit-pattern tensors (0 when both are all zero, inf when only ``b`` is)."""

    a, b = as_float(a_bits), as_float(b_bits)
    norm = float(b.norm())
    diff = float((a - b).norm())
    if norm == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / norm


def nonzero_rows(t: torch.Tensor) -> list[bool]:
    if t.numel() == 0:
        return []
    flat = as_float(t).reshape(t.shape[0], -1) if t.dim() > 1 else as_float(t).reshape(-1, 1)
    return [bool(r.any()) for r in flat]


def lane_slice(name: str, tensor: torch.Tensor, lane: int) -> torch.Tensor:
    axis = {"recurrent": 0, "conv": 2, "compressed": 0, "staging": 1, "ring": 1, "ple": 1}[name]
    return tensor.select(axis, lane)


def family_report(single: dict, packed: dict, lane: int, length: int, rows: int) -> dict:
    """Per family: the relative errors (single path vs pack landing) of the live part of lane ``lane``, the structure
    comparison and the dead part (what lies past the prompt in the single path)."""

    out = {}
    for name in FAMILIES:
        errors, structure_bad, dead_single, dead_pack = [], 0, 0, 0
        for index, (ts, tp) in enumerate(zip(single[name], packed[name])):
            for d in range(4):
                a, b = lane_slice(name, ts[d], lane), lane_slice(name, tp[d], lane)  # single, pack
                if name == "compressed":
                    live = length // 4
                    a_live, b_live = a[0, :live], b[0, :live]
                    dead_single += sum(nonzero_rows(a[0, live : rows // 4]))
                    dead_pack += sum(nonzero_rows(b[0, live : rows // 4]))
                elif name == "staging":  # [1, 32, 512]
                    live = (length - 1) % 32 + 1
                    a_live, b_live = a[0, :live], b[0, :live]
                    dead_single += sum(nonzero_rows(a[0, live:]))
                    dead_pack += sum(nonzero_rows(b[0, live:]))
                elif name == "ring":  # [1, 32, 128]
                    live = (length - 1) % 4 + 1
                    a_live, b_live = a[0, :live], b[0, :live]
                    dead_single += sum(nonzero_rows(a[0, live:]))
                    dead_pack += sum(nonzero_rows(b[0, live:]))
                elif name == "conv":
                    a_live, b_live = a.reshape(1, -1), b.reshape(1, -1)
                else:
                    a_live, b_live = a, b
                if a_live.numel():
                    errors.append(rel_error(b_live, a_live))
                    if name in ("compressed", "staging", "ring", "conv", "ple"):
                        structure_bad += sum(x != y for x, y in zip(nonzero_rows(a_live), nonzero_rows(b_live)))
                    else:  # recurrent [12, 128, 128]: whether each head's state is zero
                        heads = lambda t: [bool(as_float(h_).any()) for h_ in t]  # noqa: E731
                        structure_bad += sum(x != y for x, y in zip(heads(a_live), heads(b_live)))
        out[name] = {
            "median": statistics.median(errors),
            "max": max(errors),
            "n": len(errors),
            "structure_mismatches": structure_bad,
            "dead_nonzero_rows_single": dead_single,
            "dead_nonzero_rows_pack": dead_pack,
        }
    return out


def conv_slot_report(single: dict, packed: dict, lane: int, length: int) -> dict:
    """The ring slots one by one (slot 0 is dead data: the next token overwrites it)."""

    out = {}
    for slot in range(4):
        errors, structure = [], 0
        for g in range(len(single["conv"]) // 4):
            for d in range(4):
                a = single["conv"][4 * g + slot][d][0, 0, lane]
                b = packed["conv"][4 * g + slot][d][0, 0, lane]
                errors.append(rel_error(b, a))
                structure += int(bool(as_float(a).any()) != bool(as_float(b).any()))
        out[slot] = {"median": statistics.median(errors), "max": max(errors), "structure_mismatches": structure}
    return out


def kv_report(h: Landing, pool_single, pool_packed, lane_pages: dict, lengths: dict) -> dict:
    errors, structure = [], 0
    for q in range(len(pool_single)):
        for dev in range(4):
            a, b = pool_single[q][dev].reshape(-1, 512), pool_packed[q][dev].reshape(-1, 512)
            for lane, pages in lane_pages.items():
                rows = [pages[t >> 6] * BLOCK + (t & (BLOCK - 1)) for t in range(lengths[lane])]
                ra, rb = a[rows], b[rows]
                errors.append(rel_error(rb, ra))
                structure += sum(x != y for x, y in zip(nonzero_rows(ra), nonzero_rows(rb)))
    return {
        "median": statistics.median(errors),
        "max": max(errors),
        "n": len(errors),
        "structure_mismatches": structure,
    }


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_landing_semantics_against_single_path(mesh_device, landing_factory):
    from models.demos.blackhole.qwen38_flash_next.tests.test_prefill_pack_device import Prompts
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module

    h = landing_factory()
    d = h.decoder
    prompts = Prompts(h.checkpoint)
    lengths = [20, 33, 47, 64, 90, 127, 128, 150, 200, 256, 300, 400, 511, 512, 600, 700, 800, 900, 1000, 1024, 75, 180]
    items = [(n, prompts.make(n, seed=40 + i)[:n]) for i, n in enumerate(lengths)]
    groups: dict[int, list] = {}
    for n, ids in items:
        groups.setdefault(pack_module.bucket_for_length(n), []).append(ids)
    report = {"prompts": [], "waves": []}
    for rows, group in sorted(groups.items()):
        g = pack_module.pack_bucket(rows).users
        for first in range(0, len(group), min(g, LANES)):
            wave = group[first : first + min(g, LANES)]
            h.release_all()
            lanes = list(range(len(wave)))
            taken: set = set()
            lane_pages = {
                lane: h.window_pages(lane_page_count(len(ids), 32, BLOCK), taken) for lane, ids in zip(lanes, wave)
            }
            logits = h.pack_admit(lanes, wave, rows, pages=[lane_pages[lane] for lane in lanes])
            packed_t0 = [int(logits[i, :VOCAB].argmax()) for i in range(len(wave))]
            with misses_allowed(h.mesh):
                packed = h.lane_bits()
                pool_packed = h.pool_bits(PAGE_WINDOW * BLOCK)
            ctx_pack = [d.contexts[lane] for lane in lanes]
            pos_pack = [int(d.ls.position.positions[lane]) for lane in lanes]
            table_pack = ttnn.to_torch(ttnn.get_device_tensors(d.ls.page_table)[0])[0, 0].clone()
            h.release_all()
            single_t0 = []
            for lane, ids in zip(lanes, wave):
                single_t0.append(h.single_admit(lane, ids, pages=lane_pages[lane]))
            with misses_allowed(h.mesh):
                single = h.lane_bits()
                pool_single = h.pool_bits(PAGE_WINDOW * BLOCK)
            ctx_single = [d.contexts[lane] for lane in lanes]
            pos_single = [int(d.ls.position.positions[lane]) for lane in lanes]
            table_single = ttnn.to_torch(ttnn.get_device_tensors(d.ls.page_table)[0])[0, 0].clone()
            # the single path's own noise floor: the same prompts under the legacy prefill plan (32 / 128-row chunks and slabs,
            # no padded pass), read back the same way
            h.release_all()
            os.environ[driver_module.PLAN_ENV] = "legacy"
            try:
                for lane, ids in zip(lanes, wave):
                    h.single_admit(lane, ids, pages=lane_pages[lane])
            finally:
                os.environ.pop(driver_module.PLAN_ENV, None)
            with misses_allowed(h.mesh):
                legacy = h.lane_bits()
                pool_legacy = h.pool_bits(PAGE_WINDOW * BLOCK)
            kv_floor = kv_report(
                h, pool_single, pool_legacy, lane_pages, {lane: len(i) for lane, i in zip(lanes, wave)}
            )
            kv = kv_report(h, pool_single, pool_packed, lane_pages, {lane: len(ids) for lane, ids in zip(lanes, wave)})
            for i, (lane, ids) in enumerate(zip(lanes, wave)):
                fam = family_report(single, packed, lane, len(ids), rows)
                floor = family_report(single, legacy, lane, len(ids), rows)
                row = {
                    "floor_cost_vs_legacy": {name: floor[name]["median"] for name in FAMILIES}
                    | {f"{name}_max": floor[name]["max"] for name in FAMILIES},
                    "length": len(ids),
                    "bucket": rows,
                    "lane": lane,
                    "families": fam,
                    "conv_slots": conv_slot_report(single, packed, lane, len(ids)),
                    "t0_single": single_t0[i],
                    "t0_pack": packed_t0[i],
                    "context_equal": ctx_pack[i] == ctx_single[i],
                    "position_equal": pos_pack[i] == pos_single[i] == len(ids),
                    "page_table_equal": bool(torch.equal(table_pack[lane], table_single[lane])),
                }
                report["prompts"].append(row)
                logger.info(f"SEM {json.dumps(row, default=str)}")
            report["waves"].append({"bucket": rows, "lengths": [len(i) for i in wave], "kv": kv, "kv_floor": kv_floor})
            logger.info(f"SEM kv wave {rows}: {kv}")
    flagged = []
    summary = {}
    for name in FAMILIES:
        meds = [p["families"][name]["median"] for p in report["prompts"]]
        maxes = [p["families"][name]["max"] for p in report["prompts"]]
        summary[name] = {
            "median_of_medians": statistics.median(meds),
            "max": max(maxes),
            "structure_mismatches": sum(p["families"][name]["structure_mismatches"] for p in report["prompts"]),
            "dead_nonzero_rows_single": sum(p["families"][name]["dead_nonzero_rows_single"] for p in report["prompts"]),
            "dead_nonzero_rows_pack": sum(p["families"][name]["dead_nonzero_rows_pack"] for p in report["prompts"]),
        }
        if max(maxes) > FLAG_REL_ERROR:
            flagged.append(name)
    summary["floor_cost_vs_legacy"] = {
        name: {
            "median_of_medians": statistics.median(p["floor_cost_vs_legacy"][name] for p in report["prompts"]),
            "max": max(p["floor_cost_vs_legacy"][f"{name}_max"] for p in report["prompts"]),
        }
        for name in FAMILIES
    }
    summary["floor_cost_vs_legacy"]["kv"] = {
        "median_of_medians": statistics.median(w["kv_floor"]["median"] for w in report["waves"]),
        "max": max(w["kv_floor"]["max"] for w in report["waves"]),
    }
    summary["kv"] = {
        "median_of_medians": statistics.median(w["kv"]["median"] for w in report["waves"]),
        "max": max(w["kv"]["max"] for w in report["waves"]),
        "structure_mismatches": sum(w["kv"]["structure_mismatches"] for w in report["waves"]),
    }
    if summary["kv"]["max"] > FLAG_REL_ERROR:
        flagged.append("kv")
    summary["conv_slots"] = {
        slot: {
            "median": statistics.median(p["conv_slots"][slot]["median"] for p in report["prompts"]),
            "max": max(p["conv_slots"][slot]["max"] for p in report["prompts"]),
            "structure_mismatches": sum(p["conv_slots"][slot]["structure_mismatches"] for p in report["prompts"]),
        }
        for slot in range(4)
    }
    summary["flagged_above_0.3"] = flagged
    summary["t0_top1_agree"] = sum(p["t0_single"] == p["t0_pack"] for p in report["prompts"])
    summary["prompts"] = len(report["prompts"])
    summary["contexts_equal"] = all(p["context_equal"] for p in report["prompts"])
    summary["positions_equal"] = all(p["position_equal"] for p in report["prompts"])
    summary["page_tables_equal"] = all(p["page_table_equal"] for p in report["prompts"])
    report["summary"] = summary
    logger.info(f"RESULT sem: {json.dumps(summary, default=str)}")
    save("sem.json", report)
    # a placement bug shows as a relative error near 1 or more; the 0.3 flags are reported, not asserted
    for name in (*FAMILIES, "kv"):
        assert summary[name]["max"] < PLACEMENT_BUG_REL_ERROR, (name, summary[name])
    assert summary["contexts_equal"] and summary["positions_equal"] and summary["page_tables_equal"]


# ------------------------------------------------------------------------------------------------ C: decode after landing

DECODE_PLAN = {
    128: [120, 121, 122, 123, 124, 125, 127, 128],
    256: [130, 180, 230, 256, 200, 140],
    512: [300, 400, 512, 450],
    1024: [640, 1024],
}


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_decode_after_landing_against_single_path(mesh_device, landing_factory):
    h = landing_factory()
    d = h.decoder
    lengths = [n for rows in DECODE_PLAN for n in DECODE_PLAN[rows]]
    tokenizer, config, ids_all = h.keyed_prompts(lengths)
    by_len = dict(zip(lengths, ids_all))
    rows_out = []
    for rows, plan in DECODE_PLAN.items():
        ids = [by_len[n] for n in plan]
        lanes = list(range(len(ids)))
        # the single path: B=1 chain prefill + forced step + import, then 32 greedy tokens
        single_t0 = {lane: h.single_admit(lane, p) for lane, p in zip(lanes, ids)}
        single = h.decode(single_t0, 32)
        for lane in lanes:
            d.release_lane(lane)
        # the pack: its own first token, then 32 greedy tokens
        logits = h.pack_admit(lanes, ids, rows)
        own_t0 = {lane: int(logits[i, :VOCAB].argmax()) for i, lane in enumerate(lanes)}
        own = h.decode(own_t0, 32)
        for lane in lanes:
            d.release_lane(lane)
        # the pack again, teacher-forced with the single path's first token: the lane state alone
        h.pack_admit(lanes, ids, rows)
        forced = h.decode(single_t0, 32)
        for lane in lanes:
            d.release_lane(lane)
        for i, (lane, p) in enumerate(zip(lanes, ids)):
            text_own = tokenizer.decode(own[lane])
            text_single = tokenizer.decode(single[lane])
            row = {
                "bucket": rows,
                "length": len(p),
                "t0_equal": own[lane][0] == single[lane][0],
                "own_agree": sum(x == y for x, y in zip(own[lane], single[lane])),
                "own_first_divergence": first_divergence(own[lane], single[lane]),
                "forced_agree": sum(x == y for x, y in zip(forced[lane], single[lane])),
                "forced_first_divergence": first_divergence(forced[lane], single[lane]),
                "pass_key_single": config["pass_key"] in text_single,
                "pass_key_pack_own": config["pass_key"] in text_own,
                "pass_key_pack_forced": config["pass_key"] in tokenizer.decode(forced[lane]),
                "single": single[lane],
                "pack_own": own[lane],
                "pack_forced": forced[lane],
            }
            rows_out.append(row)
            logger.info(
                f"DECODE {json.dumps({k: v for k, v in row.items() if k not in ('single', 'pack_own', 'pack_forced')})}"
            )
    summary = {
        "prompts": len(rows_out),
        "t0_equal": sum(r["t0_equal"] for r in rows_out),
        "own_exact_32": sum(r["own_agree"] == 32 for r in rows_out),
        "forced_exact_32": sum(r["forced_agree"] == 32 for r in rows_out),
        "own_agree_tokens_total": sum(r["own_agree"] for r in rows_out),
        "forced_agree_tokens_total": sum(r["forced_agree"] for r in rows_out),
        "pass_key_single": sum(r["pass_key_single"] for r in rows_out),
        "pass_key_pack_own": sum(r["pass_key_pack_own"] for r in rows_out),
        "pass_key_pack_forced": sum(r["pass_key_pack_forced"] for r in rows_out),
    }
    logger.info(f"RESULT decode: {json.dumps(summary)}")
    save("decode.json", {"rows": rows_out, "summary": summary})
    # the pass key must be retrieved after pack landing wherever the single path retrieves it
    for r in rows_out:
        assert r["pass_key_pack_own"] or not r["pass_key_single"], r


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("bucket", [pytest.param(name, id=f"leak_{name}") for name in ("s128", "s512")])
@pytest.mark.timeout(5400)
def test_lane_level_leak(mesh_device, landing_factory, bucket):
    """User A landed from pack (A, B, ..) vs (A, B', ..): bit-identical lane state, KV pages and 32 decoded tokens."""

    rows = BUCKETS[bucket]
    h = landing_factory()
    d = h.decoder
    users = pack_module.pack_bucket(rows).users
    slot, lane = 3, 5
    _tok, _cfg, (a_ids,) = h.keyed_prompts([rows])
    pages = h.window_pages(lane_page_count(rows, 32, BLOCK), set())

    def pack(seed: int, other_lengths: list[int]):
        others = [h.random_prompt(n, seed=seed + i) for i, n in enumerate(other_lengths)]
        prompts = others[:slot] + [a_ids] + others[slot:]
        return prompts

    rng = random.Random(5)
    lengths1 = [rng.randint(1, rows) for _ in range(users - 1)]
    lengths2 = [rng.randint(1, rows) for _ in range(users - 1)]
    runs = []
    for seed, lengths in ((11, lengths1), (999, lengths2), (11, lengths1)):
        h.release_all()
        prompts = pack(seed, lengths)
        logits = h.pack_admit([lane], prompts, rows, users=[slot], pages=[pages])
        with misses_allowed(h.mesh):
            state = h.lane_bits()
            pool = h.pool_bits(PAGE_WINDOW * BLOCK)
        t0 = int(logits[0, :VOCAB].argmax())
        tokens = h.decode({lane: t0}, 32)[lane]
        runs.append({"logits": logits[0].clone(), "state": state, "pool": pool, "tokens": tokens})
    results = {"rows": rows, "pairs": []}
    for label, left, right in (("B_vs_Bprime", runs[0], runs[1]), ("repeat", runs[0], runs[2])):
        bad = []
        for name in FAMILIES:
            axis = {"recurrent": 0, "conv": 2, "compressed": 0, "staging": 1, "ring": 1, "ple": 1}[name]
            for index, (tl, tr) in enumerate(zip(left["state"][name], right["state"][name])):
                for dev in range(4):
                    # the decode advanced the lane: compare the state read BEFORE the decode (taken right after the landing)
                    if not torch.equal(tl[dev].select(axis, lane), tr[dev].select(axis, lane)):
                        bad.append((name, index, dev))
        kv_bad = []
        length = len(a_ids)
        for q in range(len(left["pool"])):
            for dev in range(4):
                a = left["pool"][q][dev].reshape(-1, 512)
                b = right["pool"][q][dev].reshape(-1, 512)
                rows_a = [pages[t >> 6] * BLOCK + (t & (BLOCK - 1)) for t in range(length)]
                if not torch.equal(a[rows_a], b[rows_a]):
                    kv_bad.append((q, dev))
        pair = {
            "label": label,
            "lane_state_mismatches": len(bad),
            "first": bad[:5],
            "kv_mismatches": len(kv_bad),
            "logits_equal": bool(torch.equal(left["logits"].view(torch.int32), right["logits"].view(torch.int32))),
            "tokens_equal": left["tokens"] == right["tokens"],
        }
        results["pairs"].append(pair)
        logger.info(f"LEAK {json.dumps(pair, default=str)}")
    save(f"lane_leak_{bucket}.json", results)
    for pair in results["pairs"]:
        assert pair["lane_state_mismatches"] == 0 and pair["kv_mismatches"] == 0
        assert pair["logits_equal"] and pair["tokens_equal"]


# ---------------------------------------------------------------------------------------------------- D: timing, E: DRAM


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_landing_timing(mesh_device, landing_factory):
    h = landing_factory(lanes=16)
    d = h.decoder
    results = {"open_s": h.open_s, "buckets": {}, "single": {}}
    for rows in pack_module.PACK_BUCKETS:
        users = pack_module.pack_bucket(rows).users
        prompts = [h.random_prompt(rows, seed=rows + i) for i in range(users)]
        samples = []
        for _rep in range(3):
            h.release_all()
            taken: set = set()
            pages = [h.window_pages(lane_page_count(rows, 32, BLOCK), taken) for _ in range(users)]
            started = time.perf_counter()
            h.pack_admit(list(range(users)), prompts, rows, pages=pages)
            samples.append({**d.last_pack, "wall_ms": (time.perf_counter() - started) * 1e3})
        med = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        results["buckets"][rows] = {"users": users, "median": med, "samples": samples}
        logger.info(f"TIMING pack S={rows}: {json.dumps(med)}")
    h.release_all()
    for n in (32, 128, 1024):
        ids = h.random_prompt(n, seed=n)
        times = []
        for _rep in range(3):
            started = time.perf_counter()
            h.single_admit(0, ids)
            times.append((time.perf_counter() - started) * 1e3)
            d.release_lane(0)
        results["single"][n] = {"median_ms": statistics.median(times), "samples": times}
        logger.info(f"TIMING single P={n}: {results['single'][n]}")
    save("timing.json", results)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("config", [pytest.param(c, id=f"dram_{c}") for c in ("b8_off", "b8_on", "b32_off", "b32_on")])
@pytest.mark.timeout(3600)
def test_landing_dram(mesh_device, landing_factory, config):
    lanes = 8 if config.startswith("b8") else 32
    pack = config.endswith("_on")
    h = landing_factory(lanes=lanes, pack=pack)
    d = h.decoder
    free_after_warmups = free_dram(mesh_device)
    result = {
        "config": config,
        "free_before_open": h.free_before_open,
        "free_after_chain": h.free_after_chain,
        "free_after_lane_capture": h.free_after_open,
        "free_after_warmups": free_after_warmups,
        "pool_blocks": h.blocks,
        "pool_tokens": h.blocks * BLOCK,
        "pack_dram": d.pack_dram,
        "admission_estimate": None,
    }
    if pack:
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import pack_dram_bytes

        result["admission_estimate"] = pack_dram_bytes(d.pack_buckets)
    logger.info(f"RESULT dram: {json.dumps(result, default=str)}")
    save(f"dram_{config}.json", result)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_decode_noise_floor_of_the_single_path(mesh_device, landing_factory):
    """The calibration of ``decode``: the same 20 prompts through the single path under two prefill plans (``cost``, the
    served default, and ``legacy``: 32 / 128-row chunks and slabs without a padded pass) and 32 greedy tokens each.  The
    plans differ at the numerics-class level (P3d-2: different matmul shapes), so their token agreement is the floor any
    other prefill form is measured against."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module

    h = landing_factory(pack=False)
    d = h.decoder
    lengths = [n for rows in DECODE_PLAN for n in DECODE_PLAN[rows]]
    tokenizer, config, ids_all = h.keyed_prompts(lengths)
    by_len = dict(zip(lengths, ids_all))
    rows_out = []
    for rows, plan in DECODE_PLAN.items():
        ids = [by_len[n] for n in plan]
        lanes = list(range(len(ids)))
        runs = {}
        for name in ("cost", "legacy"):
            os.environ[driver_module.PLAN_ENV] = name
            try:
                first = {lane: h.single_admit(lane, p) for lane, p in zip(lanes, ids)}
            finally:
                os.environ.pop(driver_module.PLAN_ENV, None)
            runs[name] = h.decode(first, 32)
            for lane in lanes:
                d.release_lane(lane)
        for lane, p in zip(lanes, ids):
            a, b = runs["cost"][lane], runs["legacy"][lane]
            rows_out.append(
                {
                    "bucket": rows,
                    "length": len(p),
                    "agree": sum(x == y for x, y in zip(a, b)),
                    "first_divergence": first_divergence(a, b),
                    "pass_key_cost": config["pass_key"] in tokenizer.decode(a),
                    "pass_key_legacy": config["pass_key"] in tokenizer.decode(b),
                }
            )
            logger.info(f"FLOOR {json.dumps(rows_out[-1])}")
    summary = {
        "prompts": len(rows_out),
        "exact_32": sum(r["agree"] == 32 for r in rows_out),
        "agree_tokens_total": sum(r["agree"] for r in rows_out),
        "pass_key_cost": sum(r["pass_key_cost"] for r in rows_out),
        "pass_key_legacy": sum(r["pass_key_legacy"] for r in rows_out),
    }
    logger.info(f"RESULT floor: {json.dumps(summary)}")
    save("floor.json", {"rows": rows_out, "summary": summary})


# ------------------------------------------------------------------------------- P3d-4: mixed order, bucket 256 against 128

MIXED_SINGLES = (40, 130, 700, 1500)  # a short chunked prompt, one long chunk, a padded slab pass, slab + padded slab
MIXED_PACKS = (  # (rows per user, lengths): full 16-user S=128, 8-user S=256, 4-user S=512, 2-user S=1024
    (128, [random.Random(1).randint(1, 128) for _ in range(16)]),
    (256, [random.Random(2).randint(1, 256) for _ in range(8)]),
    (512, [random.Random(3).randint(1, 512) for _ in range(4)]),
    (1024, [1024, 513]),
)
SINGLE_LANE = 15
SINGLE_PAGE_WINDOW = range(120, 159)  # the single request's pages; the packs' lanes use pages below 120


def lanes_state_bits(h: Landing, lanes: list[int]) -> list[dict]:
    """The slice of every family tensor of each lane in ``lanes``, per device (one read of the whole state)."""

    axis = {"recurrent": 0, "conv": 2, "compressed": 0, "staging": 1, "ring": 1, "ple": 1}
    state = h.lane_bits()
    return [
        {
            name: [[t.select(axis[name], lane).clone() for t in per_device] for per_device in state[name]]
            for name in FAMILIES
        }
        for lane in lanes
    ]


def pool_page_bits(h: Landing, pages: list[int]) -> list:
    """The pool rows of ``pages`` (every row of every page) per QSA layer and device."""

    top = (max(pages) + 1) * BLOCK
    full = h.pool_bits(top)
    rows = torch.tensor([p * BLOCK + r for p in pages for r in range(BLOCK)])
    return [[dev.reshape(-1, 512)[rows].clone() for dev in layer] for layer in full]


def single_with_logits(h: Landing, lane: int, ids: list[int], pages: list[int]) -> torch.Tensor:
    """The adapter's single path (chain prefill, import into ``lane``) keeping the first-token logits (masked, fp32)."""

    d = h.decoder
    out = h.model._prefill_one(ids, greedy_token=False)
    d.import_lane(h.model, lane, len(ids), h.model.slot.ple_context, pages)
    h.model.slot.committed = []
    h.model.slot.ple_context = None
    return out.reshape(-1).clone()


def single_snapshot(h: Landing, ids: list[int]) -> dict:
    h.release_all()
    pages = [p for p in SINGLE_PAGE_WINDOW][: lane_page_count(len(ids), 32, BLOCK)]
    logits = single_with_logits(h, SINGLE_LANE, ids, pages)
    with misses_allowed(h.mesh):
        (state,) = lanes_state_bits(h, [SINGLE_LANE])
        pool = pool_page_bits(h, pages)
    h.decoder.release_lane_external(SINGLE_LANE)
    return {"logits": logits, "state": state, "pool": pool, "pages": pages}


def live_view(name: str, tensor: torch.Tensor, length: int) -> torch.Tensor:
    """The part of one lane's tensor a decode can read: the QSA compressed rows < P // 4, the staging tile rows up to the row
    of token P - 1 and the raw-key ring rows of the open block up to it; every other family whole.  The rest is dead data
    (the single path's import copies whatever pad-derived rows the chain's dense caches held; the landing writes zero).
    """

    if name == "compressed":
        return tensor[:, : length // 4]
    if name == "staging":
        return tensor[:, : (length - 1) % 32 + 1]
    if name == "ring":
        return tensor[:, : (length - 1) % 4 + 1]
    return tensor


def diff_snapshots(a: dict, b: dict, length: int, pool_rows: int | None = None) -> dict:
    """Bitwise differences of two single-path snapshots of the same prompt: ``live_*`` is the gate (what a decode reads: the
    live rows of every family, the KV rows [0, P), the logits); ``full_*`` also counts the dead rows (history dependent:
    a single after a single already differs there)."""

    def mismatches(live: bool):
        out = []
        for name in FAMILIES:
            for i, (ta, tb) in enumerate(zip(a["state"][name], b["state"][name])):
                for dev in range(4):
                    x, y = ta[dev], tb[dev]
                    if live:
                        x, y = live_view(name, x, length), live_view(name, y, length)
                    if not torch.equal(x, y):
                        out.append((name, i, dev))
        return out

    def pool_mismatches(live: bool):
        out = []
        keep = length if pool_rows is None else pool_rows
        for q, (la, lb) in enumerate(zip(a["pool"], b["pool"])):
            for dev in range(4):
                x, y = la[dev], lb[dev]
                if live:
                    x, y = x[:keep], y[:keep]
                if not torch.equal(x, y):
                    out.append((q, dev))
        return out

    live_state, full_state = mismatches(True), mismatches(False)
    return {
        "live_state_mismatches": len(live_state),
        "first_live_state": live_state[:5],
        "full_state_mismatches": len(full_state),
        "full_state_families": sorted({m[0] for m in full_state}),
        "first_full_state": full_state[:5],
        "live_pool_mismatches": len(pool_mismatches(True)),
        "full_pool_mismatches": len(pool_mismatches(False)),
        "logits_equal": bool(torch.equal(a["logits"].view(torch.int32), b["logits"].view(torch.int32))),
    }


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_mixed_single_after_pack(mesh_device, landing_factory):
    """A single-path prefill right after a pack: lane state (every family, 4 devices), the lane's KV pages and the
    first-token logits are bitwise equal to the same prefill without the pack before it."""

    h = landing_factory(lanes=16)
    singles = {n: h.random_prompt(n, seed=4000 + n) for n in MIXED_SINGLES}
    base = {n: single_snapshot(h, ids) for n, ids in singles.items()}
    repeat = {n: single_snapshot(h, ids) for n, ids in singles.items()}
    results = {"repeat": {n: diff_snapshots(base[n], repeat[n], n) for n in singles}, "after_pack": []}
    logger.info(f"MIXED repeat (single after single): {json.dumps(results['repeat'], default=str)}")
    for rows, lengths in MIXED_PACKS:
        for n, ids in singles.items():
            h.release_all()
            prompts = [h.random_prompt(m, seed=900 + rows + i) for i, m in enumerate(lengths)]
            taken = set(SINGLE_PAGE_WINDOW)
            pages = [h.window_pages(lane_page_count(m, 32, BLOCK), taken) for m in lengths]
            h.pack_admit(list(range(len(lengths))), prompts, rows, pages=pages)
            h.release_all()
            after = single_snapshot(h, ids)
            diff = {"pack_rows": rows, "pack_users": len(lengths), "single": n, **diff_snapshots(base[n], after, n)}
            results["after_pack"].append(diff)
            logger.info(f"MIXED single after pack: {json.dumps(diff, default=str)}")
    save("mixed_single_after_pack.json", results)
    for n, diff in results["repeat"].items():
        assert diff["live_state_mismatches"] == 0 and diff["live_pool_mismatches"] == 0 and diff["logits_equal"], (
            n,
            diff,
        )
    for diff in results["after_pack"]:
        assert diff["live_state_mismatches"] == 0 and diff["live_pool_mismatches"] == 0 and diff["logits_equal"], diff


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_mixed_pack_after_single(mesh_device, landing_factory):
    """A pack right after a single-path prefill equals the same pack alone: first-token logits, lane state of every landed
    lane, and the landed lanes' KV pages, bitwise."""

    h = landing_factory(lanes=16)
    singles = {n: h.random_prompt(n, seed=4000 + n) for n in MIXED_SINGLES}
    results = {"cases": []}

    pages_of: dict = {}

    def run_pack(rows, lengths):
        h.release_all()
        prompts = [h.random_prompt(m, seed=900 + rows + i) for i, m in enumerate(lengths)]
        if rows not in pages_of:  # the same pages in every run of a pack
            taken = set(SINGLE_PAGE_WINDOW)
            pages_of[rows] = [h.window_pages(lane_page_count(m, 32, BLOCK), taken) for m in lengths]
        pages = pages_of[rows]
        lanes = list(range(len(lengths)))
        logits = h.pack_admit(lanes, prompts, rows, pages=pages)
        with misses_allowed(h.mesh):
            states = lanes_state_bits(h, lanes)
            pools = [pool_page_bits(h, row) for row in pages]
        h.release_all()
        return [{"logits": logits[u].clone(), "state": states[u], "pool": pools[u]} for u in range(len(lengths))]

    for rows, lengths in MIXED_PACKS:
        alone = run_pack(rows, lengths)
        for n, ids in singles.items():
            h.release_all()
            single_with_logits(h, SINGLE_LANE, ids, list(SINGLE_PAGE_WINDOW)[: lane_page_count(n, 32, BLOCK)])
            h.release_all()
            after = run_pack(rows, lengths)
            diffs = [
                diff_snapshots(alone[u], after[u], lengths[u], pool_rows=-(-lengths[u] // 32) * 32)
                for u in range(len(lengths))
            ]
            bad = [
                u
                for u, df in enumerate(diffs)
                if df["full_state_mismatches"] or df["live_pool_mismatches"] or not df["logits_equal"]
            ]
            case = {
                "pack_rows": rows,
                "pack_users": len(lengths),
                "single_before": n,
                "bad_users": bad,
                "detail": {u: {k: v for k, v in diffs[u].items() if "first" in k or "families" in k} for u in bad},
            }
            results["cases"].append(case)
            logger.info(f"MIXED pack after single: {json.dumps(case)}")
    save("mixed_pack_after_single.json", results)
    for case in results["cases"]:
        assert not case["bad_users"], case


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_pack_bucket_256_against_128(mesh_device, landing_factory):
    """8 users with P <= 128: one S=256 pack against one S=128 pack (8 of 16 slots): cost per phase, and whether the first-token
    logits agree (the P3d-2 claim: the same prompt in other buckets gives the same logits)."""

    h = landing_factory(lanes=8)
    d = h.decoder
    prompts = [h.random_prompt(n, seed=50 + i) for i, n in enumerate([100, 128, 31, 64, 90, 12, 117, 128])]
    results = {}
    logits = {}
    for rows in (128, 256):
        samples = []
        for _rep in range(5):
            h.release_all()
            taken: set = set()
            pages = [h.window_pages(lane_page_count(len(p), 32, BLOCK), taken) for p in prompts]
            started = time.perf_counter()
            out = h.pack_admit(list(range(8)), prompts, rows, pages=pages)
            samples.append({**d.last_pack, "wall_ms": (time.perf_counter() - started) * 1e3})
            logits[rows] = out.clone()
        med = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        results[rows] = {"median": med, "samples": samples}
        logger.info(f"S{rows} x 8 users: {json.dumps(med)}")
    results["logits_bitwise_equal"] = bool(torch.equal(logits[128].view(torch.int32), logits[256].view(torch.int32)))
    save("bucket256_vs_128.json", results)
