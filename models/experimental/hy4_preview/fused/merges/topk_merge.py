# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The cross-chip half of the index selection (step X2 of the decode layer): from every chip's local top-2048 of a
user to the global top-2048, with only counts crossing chips. Piece 6 of fused/README.md.

Input, per chip and user: the survivor of the index-key scan (fused/indexer_scan/CONTRACT.md): 2048 words
[bf16 score | u16 stamp] in rank order. Output, per chip and user: ``keep``, the number of the chip's entries that
are in the global top-2048, as the cut word ``1 << 31 | keep`` the scan's cut takes. The chip's winners are its
ranks 0 .. keep - 1.

The global order and why the union is exactly the top-k
-------------------------------------------------------
Entries are ordered by (key descending, chip ascending, rank ascending): key is the 16-bit order key of the BF16
score, chip the chip's slot in the merge (row-major over the mesh), rank the entry's place in its chip's list.
Entries a query does not see (keys below KEY_REAL_MIN: NEG rows and -inf padding) are not entries. The order is
total (chip and rank identify an entry), so "the first min(k, N) of the N entries" is one well-defined set S. On
one chip the order is (key descending, rank ascending), and a chip's list has its keys in non-increasing rank
order, so the global order restricted to a chip IS its rank order; S restricted to chip c is therefore a prefix of
c's list, of length keep_c, and sum_c keep_c = |S| = min(k, N). ``golden`` builds S by sorting; the kernel finds
the same keep_c from counts:

    T        the key of the last entry of S: the largest key t with #(key >= t) >= k
    a_c      chip c's entries with key > T (all of them are in S)
    e_c      chip c's entries with key == T; S takes left = k - sum_c a_c of them in chip order
    keep_c = a_c + min(e_c, max(0, left - sum_{c' < c} e_c'))

Why two rounds of counts (the designs compared, 32 chips, 8 users)
------------------------------------------------------------------
(a) gather every chip's 2048 scores on every chip: a chip sends 32 KB and receives 1 MB, then merges 32 lists of
    2048 per user; one fabric pass, but the megabyte and the 65,536-entry merge per user are far outside 171 us.
(c) a tree merge of sorted lists keeping 2048 per node: 5 levels, each a fabric pass of 4 to 8 KB per user and a
    2 x 2048 merge on the SFPU, then a pass back with the cut: 6 passes in series.
(b) counts at thresholds. A bisection on the key is 16 passes; counts at 255 thresholds at once is a 256-ary
    search: 2 passes resolve the 16 key bits, always (expected = worst case = 2 rounds). Because every chip
    receives every chip's vector (an all-gather, not a sum), the second pass also gives every chip the a_c and
    e_c of all chips, so the ties need no third pass. A chip sends 2 x 4.2 KB and receives 2 x 131 KB.
(b) is built. A round's vector of a user is (2^8 + 1) uint16: vector[i] = #(key >= t_i); the first round's
thresholds are the 256 values of the key's high byte, the second round's the 256 keys of the bucket the first
round found.

What can start before the scan has finished: nothing of round 0's arithmetic (it needs the final local list), but
the sender core's fabric connections open at program start, and round 0 of a user starts when THAT user's root
scan core has its survivor, not when the chip's last user has. Round 1 depends on round 0's landing, but while
round 0 is on the fabric every user core already counts the round 1 vector of the bucket it expects the cut in
(its own entry of rank k / chips); when round 0 lands on that bucket, round 1 leaves at once (FLAG_RECOUNT
marks the cores that had to count again). The row reader of the attention needs ``keep``; its queries, masks
and the cut's stamp decode of ranks below a_c known after round 0 (entries above the bucket are certain
winners) can run while round 1 is on the fabric. Replacing round 0 by counts the scan cores take while they
score, at thresholds around the previous step's cut (the score centre makes that cut the origin), would
remove a fabric pass from the path after the scan; that needs the scan's compute kernel and is not built.

As a piece of a layer program
-----------------------------
cores       ``users`` user cores in a logical row (the collective's landing cores) and one sender core that
            carries nothing else on its BRISC. In the layer program the user cores are the users' root scan
            cores (the survivor is in their L1) or cores those push the 8 KB survivor to.
kernel      kernels/topk_merge.hpp ``hy4_topk::merge`` is a function for a data-movement kernel; the standalone
            kernel kernels/topk_merge.cpp wraps it with a DRAM read of the survivor and a DRAM write of the cut
            page (for ``indexer_scan.Cut`` as a program of its own).
semaphores  none of its own; the collective's ``Fragment`` takes one program semaphore (the sender's ready word)
            on the sender and user cores and two per fabric direction on the sender core. The standalone
            program's own hand-off (cut words to user 0's core) uses a persistent L1 counter.
CBs         on the user cores: the survivor (2 x 4096 B, standalone only), two vectors of scratch (2 x 528 B),
            one 64-byte page (standalone only). None on the sender core.
L1          ``MeshSite``: landing 2 parities x 32 members x 528 B = 33.8 KB per user core, a ctrl tile, 4.2 KB of
            staging on the sender core, plus one state tile per user core (standalone). L1 buffers take their
            bytes on every core of the chip.
sender ops  2 x users (one per user and round): 16 of the collective sender's 16 with 8 users.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.merges.mesh_site import MeshSite

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
LIST = 2048
BANKS = 8
KEY_REAL_MIN = 0x0F80
FLAG_ALL, FLAG_TIMEOUT, FLAG_RECOUNT = 1, 2, 4
CB_SURVIVOR, CB_PAYLOAD, CB_PAGE = 0, 1, 2
SENDER_CORE = (0, 0)
USER_CORE = (1, 0)  # user u on (1 + u, 0)
NEG_BITS, NINF_BITS = 0xF180, 0xFF80
# state tile words (kernels/topk_merge.cpp)
ST_CUT, ST_KEY, ST_ABOVE, ST_BEFORE, ST_STAMPS, ST_START, ST_LOADED, ST_END = 0, 1, 2, 3, 4, 10, 11, 12
STAMPS = ("counts0", "handed0", "landed0", "counts1", "handed1", "landed1")


# ---------------------------------------------------------------------------------------------
# Host arithmetic: the order, the golden, and the kernel's protocol in torch
# ---------------------------------------------------------------------------------------------
def order_key(bits: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns (int64) -> 16-bit keys in the order of the values."""
    return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits | 0x8000)


def bf16_bits(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF


def rank_order(words: torch.Tensor) -> torch.Tensor:
    """Fused words [..., n] (int64) sorted as the scan leaves them: as FP32 numbers, best first (by score; equal
    scores by stamp, the larger first for a positive score, the smaller for a negative one)."""
    as_key = torch.where((words & 0x80000000) != 0, 0xFFFFFFFF - words, words | 0x80000000)
    return torch.gather(words, -1, torch.argsort(as_key, dim=-1, descending=True, stable=True))


def keys_of(words: torch.Tensor) -> torch.Tensor:
    return order_key((words >> 16) & 0xFFFF)


def golden(words: torch.Tensor, topk: int = LIST) -> dict:
    """words: int64 [chips, users, 2048] in rank order. The global top-k by (key desc, chip asc, rank asc) over
    the visible entries. Returns keep [chips, users], the cut key per user (0 when everything is kept), and
    ``members`` [users]: a bool [chips, 2048] mask of the selected entries."""
    chips, users, n = words.shape
    keys = keys_of(words)
    keep = torch.zeros(chips, users, dtype=torch.int64)
    cut = torch.zeros(users, dtype=torch.int64)
    for u in range(users):
        k = keys[:, u].reshape(-1)  # chip-major, rank-minor: a stable sort keeps (chip, rank) among equal keys
        order = torch.argsort(k, descending=True, stable=True)
        order = order[k[order] >= KEY_REAL_MIN][:topk]
        keep[:, u] = torch.bincount(order // n, minlength=chips)
        if order.numel() == topk:
            cut[u] = k[order[-1]]
    return {"keep": keep, "key": cut}


def count_ge(keys: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """keys [..., n] non-increasing, thresholds [m] -> [..., m]: visible entries with key >= threshold."""
    t = torch.clamp(thresholds, min=KEY_REAL_MIN)
    ascending = keys.flip(-1).contiguous()
    shape = keys.shape[:-1] + (t.numel(),)
    return keys.shape[-1] - torch.searchsorted(ascending, t.expand(shape).contiguous(), right=False)


def simulate(words: torch.Tensor, topk: int = LIST, bits=(8, 8)) -> dict:
    """The kernels' rounds (kernels/topk_merge.hpp merge, topk_hub.cpp) on the host, for any split of the 16 key
    bits over the rounds: what every chip must end with."""
    chips, users, _ = words.shape
    keys = keys_of(words)
    keep = torch.zeros(chips, users, dtype=torch.int64)
    key, flags = torch.zeros(users, dtype=torch.int64), torch.zeros(users, dtype=torch.int64)
    above_all, before_all = torch.zeros(users, dtype=torch.int64), torch.zeros(chips, users, dtype=torch.int64)
    for u in range(users):
        base, shift = 0, 16
        for r, b in enumerate(bits):
            shift -= b
            n = 1 << b
            v = count_ge(keys[:, u], base + (torch.arange(n + 1) << shift))  # [chips, n + 1]
            total = v.sum(0)
            if r == 0 and int(total[0]) < topk:
                keep[:, u], flags[u] = v[:, 0], FLAG_ALL
                break
            i = int((total[:n] >= topk).nonzero().max())
            base += i << shift
        else:
            a, e = v[:, i + 1], v[:, i] - v[:, i + 1]
            left = topk - int(a.sum())
            before = torch.cumsum(e, 0) - e
            keep[:, u] = a + torch.minimum(e, torch.clamp(left - before, min=0))
            key[u], above_all[u], before_all[:, u] = base, a.sum(), before
    return {"keep": keep, "key": key, "flags": flags, "above": above_all, "before": before_all}


def count_vector_reference(keys, base: int, shift: int, n: int, lo: int, hi: int) -> list:
    """kernels/topk_merge.hpp count_vector / count_ge statement for statement, on a python list of keys."""

    def gallop(t, lo, hi):
        step = 1
        while lo < hi:
            probe = min(lo + step - 1, hi - 1)
            if keys[probe] >= t:
                lo, step = probe + 1, step << 1
            else:
                hi = probe
                break
        while lo < hi:
            mid = (lo + hi) >> 1
            if keys[mid] >= t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    out = [0] * (n + 1)
    r = lo
    out[n] = lo
    i = n - 1
    while i >= 1:
        if r >= hi:
            out[i] = r
            i -= 1
            continue
        bucket = (keys[r] - base) >> shift
        while i > bucket:
            out[i] = r
            i -= 1
        if i == 0:
            break
        r = gallop(base + (i << shift), r, hi)
        out[i] = r
        i -= 1
    out[0] = hi
    return out


# ---------------------------------------------------------------------------------------------
# Survivor tensors (the scan's output layout) from host lists
# ---------------------------------------------------------------------------------------------
def tile_word(row, col):
    return ((((row >> 4) << 1) | (col >> 4)) << 8) | ((row & 15) << 4) | (col & 15)


def rank_word(rank):
    return ((rank & 127) << 4) | (rank >> 7)


def survivor_tiles(words: torch.Tensor) -> torch.Tensor:
    """Rank-ordered words [..., 2048] (int64) -> the survivor's two uint32 tiles [..., 2, 32, 32] as int32, laid
    out as the scan's root core writes them (entry of rank q at raw word rank_word(q) of the two tiles)."""
    raw = torch.zeros(words.shape, dtype=torch.int64)
    raw[..., rank_word(torch.arange(LIST))] = words
    r = torch.arange(TILE)[:, None].expand(TILE, TILE)
    c = torch.arange(TILE)[None, :].expand(TILE, TILE)
    logical = raw.reshape(*words.shape[:-1], 2, TILE * TILE)[..., tile_word(r, c).reshape(-1)]
    logical = torch.where(logical >= 1 << 31, logical - (1 << 32), logical)
    return logical.to(torch.int32).reshape(*words.shape[:-1], 2, TILE, TILE)


def stamp_row(stamp: torch.Tensor, chunks_per_bank: int = 2) -> torch.Tensor:
    """Stripe row of a stamp (fused/indexer_scan Scan.stamp_row; 2 chunks per bank at 32,768 rows per chip)."""
    chunk, col, tile, row = stamp >> 11, (stamp >> 6) & 31, (stamp >> 5) & 1, stamp & 31
    bank, local = chunk // chunks_per_bank, chunk % chunks_per_bank
    return (((local * 2 + tile) * TILE + col) * BANKS + bank) * TILE + row


# ---------------------------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------------------------
class TopkMerge:
    """The merge on ``coords`` (member chips in slot order; default every chip, row-major) for ``users`` users.

    ``survivors``: a uint32 tile tensor laid out as ``indexer_scan.Scan.out`` ([users * 8, 2, 32, 32] per chip in
    interleaved DRAM, the survivor of user u in tiles (u * 8, 0..1)); made here when not given (``set_survivors``
    fills it). ``cuts``: the cut page tensor of ``indexer_scan.Cut`` (uint32 [1, 1, 1, 16] in DRAM), likewise.
    """

    def __init__(
        self,
        mesh_device,
        coords=None,
        users: int = 8,
        topk: int = LIST,
        bits=(8, 8),
        sender_core=SENDER_CORE,
        user_core=USER_CORE,
        survivors=None,
        cuts=None,
        wait_limit: int = cl.DEFAULT_WAIT_LIMIT,
        timing: bool = False,
    ):
        rows, cols = tuple(mesh_device.shape)
        self.mesh_device, self.users, self.topk, self.bits = mesh_device, users, topk, tuple(bits)
        self.all_coords = [(r, c) for r in range(rows) for c in range(cols)]
        self.coords = list(self.all_coords) if coords is None else [tuple(c) for c in coords]
        self.device_index = {coord: i for i, coord in enumerate(self.all_coords)}
        self.sender_core, self.user_core = tuple(sender_core), tuple(user_core)
        self.user_cores = cl.row_cores(self.user_core, users)
        self.wait_limit, self.timing = wait_limit, timing
        assert sum(self.bits) == 16 and 1 <= topk <= LIST and 2 * users <= cl.MAX_OPS
        self.part_bytes = -(-2 * ((1 << max(self.bits)) + 1) // 16) * 16
        self.site = MeshSite(
            mesh_device,
            self.coords,
            self.part_bytes,
            parts=users,
            landing_core=self.user_core,
            sender_core=self.sender_core,
            name="topk",
        )
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.survivors = survivors
        if survivors is None:
            self.survivors = ttnn.from_torch(
                torch.zeros(users * BANKS, 2, TILE, TILE, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        self.cuts = cuts
        if cuts is None:
            self.cuts = ttnn.from_torch(
                torch.zeros(1, 1, 1, 16, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        self.state = cl.tile_column(mesh_device, self.user_core, 1, ttnn.uint32, cores=users)
        self._state_zero = ttnn.from_torch(
            torch.zeros(1, 1, users * TILE, TILE),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        self._descriptors = {}

    # inputs -------------------------------------------------------------------------------------
    def set_survivors(self, words: torch.Tensor):
        """words: int64 [len(coords), users, 2048], every list in rank order. A host write (not traceable)."""
        assert tuple(words.shape) == (len(self.coords), self.users, LIST), tuple(words.shape)
        chips = self.mesh_device.get_num_devices()
        tiles = torch.zeros(chips, self.users, BANKS, 2, TILE, TILE, dtype=torch.int32)
        for slot, coord in enumerate(self.coords):
            tiles[self.device_index[coord], :, 0] = survivor_tiles(words[slot])
        host = ttnn.from_torch(
            tiles.reshape(chips * self.users * BANKS, 2, TILE, TILE),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        ttnn.synchronize_device(self.mesh_device)
        ttnn.copy_host_to_device_tensor(host, self.survivors)

    def reset(self):
        """Zero the persistent state (the site's counters and rounds, the mailbox counter): after a run that
        stopped inside a round or reported a timeout."""
        self.site.reset()
        ttnn.copy_host_to_device_tensor(self._state_zero, self.state)
        ttnn.synchronize_device(self.mesh_device)

    # program ------------------------------------------------------------------------------------
    def _cbs(self, cores):
        def cb(index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=page)
                ],
            )

        return [cb(CB_SURVIVOR, 4096, 2), cb(CB_PAYLOAD, self.part_bytes, 2), cb(CB_PAGE, 64, 1)]

    def append_to(self, desc, frag: cl.Fragment, coord):
        """Append the merge to ``desc``, the program of chip ``coord``, whose collective fragment is ``frag``
        (made with the user cores among its landing cores; the caller calls ``frag.finish()`` after every piece
        has added its uses). Takes 2 x users of the sender's ops."""
        mesh = self.mesh_device
        device = self.device_index[tuple(coord)]
        on_chip = lambda t: ttnn.get_device_tensors(t)[device]
        cores = cl.row_range(self.user_core, self.users)
        desc.cbs = list(desc.cbs) + self._cbs(cores)
        uses = [frag.add(self.site), frag.add(self.site)]
        collector = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*self.user_cores[0]))
        rt = ttnn.RuntimeArgs()
        for user, (x, y) in enumerate(self.user_cores):
            rt[x][y] = [
                user,
                self.coords.index(tuple(coord)),
                on_chip(self.survivors).buffer_address(),
                user * BANKS * 2,
                on_chip(self.cuts).buffer_address(),
                collector.x,
                collector.y,
                on_chip(self.state).buffer_address(),
            ] + [a for use in uses for a in use.core_args[(x, y)]]
        desc.kernels = list(desc.kernels) + [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "topk_merge.cpp"),
                core_ranges=cores,
                compile_time_args=[self.users, self.topk, *self.bits, CB_SURVIVOR, CB_PAYLOAD, CB_PAGE],
                defines=frag.defines,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
                compiler_include_paths=[str(KERNEL_DIR), str(cl.KERNEL_DIR)],
            )
        ]
        return uses

    def descriptor(self, nop: bool = False):
        """One program per member chip. ``nop``: the same cores, buffers and semaphore with kernels that return
        at once (the launch floor of the program's shape)."""
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
            frag = cl.Fragment(
                self.mesh_device, coord, desc, self.sender_core, self.user_cores, self.wait_limit, timing=self.timing
            )
            if nop:
                cores = cl.row_range(self.user_core, self.users)
                desc.cbs = self._cbs(cores)
                desc.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=str(KERNEL_DIR / "topk_merge.cpp"),
                        core_ranges=core_range,
                        compile_time_args=[self.users, self.topk, *self.bits, CB_SURVIVOR, CB_PAYLOAD, CB_PAGE],
                        defines=frag.defines + [("HY4_MERGE_NOP", "1")],
                        config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                        compiler_include_paths=[str(KERNEL_DIR), str(cl.KERNEL_DIR)],
                    )
                    for core_range, processor, noc in (
                        (cores, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
                        (cl.core_set([self.sender_core]), ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                    )
                ]
            else:
                self.append_to(desc, frag, coord)
                frag.finish()
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def io_tensors(self):
        return [self.survivors, self.site.landing, self.site.ctrl, self.site.staging, self.state, self.cuts]

    def compile(self, nop: bool = False):
        ttnn.experimental.prepare_generic_op(self.io_tensors(), self.descriptor(nop))

    def __call__(self, nop: bool = False):
        if nop not in self._descriptors:
            self._descriptors[nop] = self.descriptor(nop)
        return ttnn.generic_op(self.io_tensors(), self._descriptors[nop])

    # results ------------------------------------------------------------------------------------
    def read(self) -> dict:
        """What the last run left on the member chips: keep [members, users], key / flags / above [members,
        users] (every member's own view; they must agree across members), before, the cut page words
        [members, 16] as written to DRAM, and ``stamps`` [members, users, 9] (start, loaded, the six of the
        merge, end; zeros unless built with timing)."""
        ttnn.synchronize_device(self.mesh_device)
        state, cuts = ttnn.get_device_tensors(self.state), ttnn.get_device_tensors(self.cuts)
        out = {k: [] for k in ("keep", "key", "flags", "above", "before", "cuts", "stamps", "cut_word")}
        for coord in self.coords:
            d = self.device_index[coord]
            words = ttnn.to_torch(state[d]).reshape(self.users, TILE, TILE).to(torch.int64) & 0xFFFFFFFF
            row0 = words[:, 0]  # words 0..15 of each core's tile: row 0, columns 0..15
            out["cut_word"].append(row0[:, ST_CUT])
            out["keep"].append(row0[:, ST_CUT] & 0xFFFF)
            out["key"].append(row0[:, ST_KEY] & 0xFFFF)
            out["flags"].append(row0[:, ST_KEY] >> 16)
            out["above"].append(row0[:, ST_ABOVE])
            out["before"].append(row0[:, ST_BEFORE])
            order = [ST_START, ST_LOADED] + [ST_STAMPS + i for i in range(6)] + [ST_END]
            out["stamps"].append(row0[:, order])
            out["cuts"].append(ttnn.to_torch(cuts[d]).reshape(16).to(torch.int64) & 0xFFFFFFFF)
        return {k: torch.stack(v) for k, v in out.items()}

    def bytes_moved(self) -> dict:
        """Fabric bytes per chip of the whole merge (two rounds)."""
        return {k: 2 * v if k != "routes" else v for k, v in self.site.bytes_per_use().items()}

    def scan_view(self, core_of=None):
        """What ``indexer_scan.Cut`` reads of a Scan, for this merge's survivors and user cores: the cut program
        then runs on the user cores and decodes the stamps of the survivors given to ``set_survivors``."""
        from models.experimental.hy4_preview.fused.indexer_scan.indexer_scan import ScanConfig

        return SimpleNamespace(
            cfg=ScanConfig(users=self.users, mode="tree"),
            mesh_device=self.mesh_device,
            core_of=core_of or (lambda user, bank: self.user_cores[user]),
            coords=self.coords,
            out=self.survivors,
        )

    def release(self):
        for tensor in (self.state,):
            ttnn.deallocate(tensor)
        self.site.release()


class TopkMergeHub(TopkMerge):
    """The same merge with ONE fabric payload per chip and round instead of one per user (what the per-user form
    measures on 8x4 is its packets: 8 sender ops x 4 multicast branches per round and chip, 248 deliveries per
    round on every chip, about 40 us per round; see tests/test_topk_merge.py).

    cores       one hub core (the site's only landing core: kernels/topk_hub.cpp), ``users`` user cores next to
                it (kernels/topk_user.cpp; they hold the survivors and count), one sender core.
    a round     every user core writes its vector into its part of the sender's staging buffer and counts on the
                hub; the hub raises the op's ready bit, waits for the landing and decides for all users: a
                16-byte record per user core (the bucket, or at the end the result), counted on that core.
    rounds      ``bits``: (8, 8) is two rounds of 257 counts per user (a 4.2 KB payload, 270 KB of landing);
                (6, 5, 5) three rounds of 65 (1.2 KB, 74 KB of landing); (4, 4, 4, 4) four of 17.
    sender ops  one per round.
    waits       user core: the hub's record count (persistent L1 word, bounded); hub: the users' count, the
                site's arrival counter (bounded). No program semaphore of its own.
    """

    def __init__(
        self,
        mesh_device,
        coords=None,
        users: int = 8,
        topk: int = LIST,
        bits=(8, 8),
        sender_core=SENDER_CORE,
        hub_core=USER_CORE,
        survivors=None,
        cuts=None,
        wait_limit: int = cl.DEFAULT_WAIT_LIMIT,
        timing: bool = False,
    ):
        rows, cols = tuple(mesh_device.shape)
        self.mesh_device, self.users, self.topk, self.bits = mesh_device, users, topk, tuple(bits)
        self.all_coords = [(r, c) for r in range(rows) for c in range(cols)]
        self.coords = list(self.all_coords) if coords is None else [tuple(c) for c in coords]
        self.device_index = {coord: i for i, coord in enumerate(self.all_coords)}
        self.sender_core, self.hub_core = tuple(sender_core), tuple(hub_core)
        self.user_core = (self.hub_core[0] + 1, self.hub_core[1])
        self.user_cores = cl.row_cores(self.user_core, users)
        self.wait_limit, self.timing = wait_limit, timing
        assert sum(self.bits) == 16 and 1 <= len(self.bits) <= 4 and 1 <= topk <= LIST and users <= 8
        self.vector_bytes = -(-2 * ((1 << max(self.bits)) + 2) // 16) * 16
        self.part_bytes = users * self.vector_bytes
        self.site = MeshSite(
            mesh_device,
            self.coords,
            self.part_bytes,
            parts=1,
            landing_core=self.hub_core,
            sender_core=self.sender_core,
            name="topk_hub",
        )
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.survivors = survivors
        if survivors is None:
            self.survivors = ttnn.from_torch(
                torch.zeros(users * BANKS, 2, TILE, TILE, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        self.cuts = cuts
        if cuts is None:
            self.cuts = ttnn.from_torch(
                torch.zeros(1, 1, 1, 16, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=dram,
            )
        self.state = cl.tile_column(mesh_device, self.hub_core, 1, ttnn.uint32, cores=1 + users)  # hub, then users
        self._state_zero = ttnn.from_torch(
            torch.zeros(1, 1, (1 + users) * TILE, TILE),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.synchronize_device(mesh_device)
        self._descriptors = {}

    def _kernel(self, source, cores, ct, rt, defines):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cores,
            compile_time_args=ct,
            defines=defines,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
            compiler_include_paths=[str(KERNEL_DIR), str(cl.KERNEL_DIR)],
        )

    def append_to(self, desc, frag: cl.Fragment, coord, nop: bool = False):
        """As TopkMerge.append_to; ``frag``'s landing cores include the hub core. Takes len(bits) sender ops."""
        mesh, users = self.mesh_device, self.users
        device = self.device_index[tuple(coord)]
        on_chip = lambda t: ttnn.get_device_tensors(t)[device]
        noc = lambda core: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        xy = lambda core: noc(core).x | (noc(core).y << 16)
        user_range, hub_range = cl.row_range(self.user_core, users), cl.core_set([self.hub_core])

        def cb(cores, index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=page)
                ],
            )

        desc.cbs = list(desc.cbs) + [
            cb(user_range, CB_SURVIVOR, 4096, 2),
            cb(user_range, CB_PAYLOAD, self.vector_bytes, 2),
            cb(hub_range, 0, 128, 1),
        ]
        uses = [] if nop else [frag.add(self.site) for _ in self.bits]
        defines = frag.defines + ([("HY4_MERGE_NOP", "1")] if nop else [])
        state = on_chip(self.state).buffer_address()
        bits4 = list(self.bits) + [0] * (4 - len(self.bits))
        user_rt, hub_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for user, (x, y) in enumerate(self.user_cores):
            user_rt[x][y] = [
                user,
                on_chip(self.survivors).buffer_address(),
                user * BANKS * 2,
                state,
                xy(self.hub_core),
                xy(self.sender_core),
                self.site.staging_address + user * self.vector_bytes,
                self.vector_bytes,
            ]
        hx, hy = self.hub_core
        hub_rt[hx][hy] = (
            [self.coords.index(tuple(coord)), state, on_chip(self.cuts).buffer_address(), self.vector_bytes]
            + [xy(core) for core in self.user_cores]
            + [0] * (8 - users)
            + [a for use in uses for a in use.core_args[(hx, hy)]]
        )
        desc.kernels = list(desc.kernels) + [
            self._kernel(
                "topk_user.cpp",
                user_range,
                [len(self.bits), *bits4, self.topk, len(self.coords), CB_SURVIVOR, CB_PAYLOAD],
                user_rt,
                defines,
            ),
            self._kernel("topk_hub.cpp", hub_range, [len(self.bits), *bits4, self.topk, users, 0], hub_rt, defines),
        ]
        return uses

    def descriptor(self, nop: bool = False):
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
            frag = cl.Fragment(
                self.mesh_device, coord, desc, self.sender_core, [self.hub_core], self.wait_limit, timing=self.timing
            )
            self.append_to(desc, frag, coord, nop)
            if nop:
                desc.kernels = list(desc.kernels) + [
                    ttnn.KernelDescriptor(
                        kernel_source=str(KERNEL_DIR / "topk_merge.cpp"),
                        core_ranges=cl.core_set([self.sender_core]),
                        compile_time_args=[self.users, self.topk, 8, 8, CB_SURVIVOR, CB_PAYLOAD, CB_PAGE],
                        defines=frag.defines + [("HY4_MERGE_NOP", "1")],
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                        ),
                        compiler_include_paths=[str(KERNEL_DIR), str(cl.KERNEL_DIR)],
                    )
                ]
            else:
                frag.finish()
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return mesh_desc

    def read(self) -> dict:
        """As TopkMerge.read. ``stamps`` [members, users, 9]: a user core's start, survivor loaded, vector of
        round 0 / 1 / later ready, -, result, -, - ; ``recounts`` [members, users]; ``hub_stamps`` [members, 8]:
        start, then per round (0, 1) ready raised / landed / decided, end."""
        ttnn.synchronize_device(self.mesh_device)
        state, cuts = ttnn.get_device_tensors(self.state), ttnn.get_device_tensors(self.cuts)
        out = {
            k: []
            for k in ("keep", "key", "flags", "above", "before", "cuts", "stamps", "cut_word", "hub_stamps", "recounts")
        }
        for coord in self.coords:
            d = self.device_index[coord]
            words = ttnn.to_torch(state[d]).reshape(1 + self.users, TILE, TILE).to(torch.int64) & 0xFFFFFFFF
            hub, row0 = words[0, 0], words[1:, 0]
            out["cut_word"].append(row0[:, ST_CUT])
            out["keep"].append(row0[:, ST_CUT] & 0xFFFF)
            out["key"].append(row0[:, ST_KEY] & 0xFFFF)
            out["flags"].append(row0[:, ST_KEY] >> 16)
            out["above"].append(row0[:, ST_ABOVE])
            out["before"].append(row0[:, ST_BEFORE])
            out["stamps"].append(torch.cat([row0[:, 8:14], row0[:, 13:14], row0[:, 13:14], row0[:, 13:14]], dim=1))
            out["recounts"].append(row0[:, 14])
            out["hub_stamps"].append(hub[8:16])
            out["cuts"].append(ttnn.to_torch(cuts[d]).reshape(16).to(torch.int64) & 0xFFFFFFFF)
        return {k: torch.stack(v) for k, v in out.items()}

    def bytes_moved(self) -> dict:
        per_use = self.site.bytes_per_use()
        return {k: len(self.bits) * v if k != "routes" else v for k, v in per_use.items()}


def last_reaching_reference(total, n: int, want: int, hint: int):
    """kernels/topk_hub.cpp Column::last_reaching statement for statement: ``total`` is a callable i -> count,
    non-increasing, total(0) >= want > total(n). Returns (the last i in [0, n) with total(i) >= want, the number
    of totals taken)."""
    calls = [0]

    def t(i):
        calls[0] += 1
        return total(i)

    lo, hi, step = 0, n, 1
    hint = min(hint, n - 1)
    if hint == 0 or t(hint) >= want:
        lo = hint
        while lo + step < n:
            if t(lo + step) >= want:
                lo += step
                step <<= 1
            else:
                hi = lo + step
                break
    else:
        hi = hint
        while hi > step:
            if t(hi - step) < want:
                hi -= step
                step <<= 1
            else:
                lo = hi - step
                break
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if t(mid) >= want:
            lo = mid
        else:
            hi = mid
    return lo, calls[0]
