# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lanes server's device side (``--lanes B``): the MTP lane chain beside the single-lane traced chain, and the
device protocol :class:`~models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler.Qwen38LaneScheduler`
drives from its one device thread.

The single-lane chain stays resident: it is the prefill engine (its chunk traces, the forced last step, the generic
and alignment generic states -- ``Qwen38ChatSession``'s own prefill section) and the referee of the startup pin.  The
lanes take the rest: ``B`` lanes verify ``R = k + 1`` rows each in one tile (``ttnn/mtp_lanes.py``), every request a
lane whose state was imported from the single-lane state after its prompt's prefill (the stage-2 admission path),
with the three lane traces (``verify_first``, ``commit``, ``draft``) replayed by the lane chain's pass loop.

Allocation order (the tracker's rule: every buffer a trace bakes an address around exists before any capture): the
pagers, their host slots and the lane verify / draft states are allocated in the traced chain's WARM HOOK (after its
warm pass, before its captures), the lane import programs compile there (the generic state as it stands, imported at
four consecutive positions into every lane: every GDN ring phase and QSA raw-history residue variant per lane index)
and the lane bodies run two eager passes (their programs); the chain then captures its own traces; the three lane
traces are captured after them, the allocation tracker verifies every trace, and the program cache must not have
grown.  From then on the program-cache miss guard stays closed: an admission's prefill, eviction and import are
replays and warmed eager programs, so a program the warm-up missed raises instead of compiling under live traces.
An admission is a sequence of device segments (the chunk groups of the prefill, one event per 128 prompt rows; the
forced last step; the eviction; the import), and between them the scheduler may run the decoding lanes' passes
(:meth:`Qwen38LanesSession.admit`'s ``between``): the lane traces and the chunk traces share the one command queue
in order, the lane states and the traced chain's state are disjoint buffers, and every eager move (the seed, the
hand-off, the eviction, the import) follows a device synchronize or a blocking readback, as it did before.

Greedy only (the lane body resolves by argmax; the sampled lanes form is a later wave).  The verify-rows fold
(``gdn_rows_scan``) serves the lanes through its lanes form (its prefix states are allocated with the lane verify state,
its dispatch is the registry's), so the referee chain and the lanes run the served default set, bitwise alike.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from loguru import logger
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker, acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import CHUNK_PREFILL_MIN_ROWS
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
    Qwen38LaneAdmitted,
    Qwen38LaneTicket,
    lane_geometry,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes, mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import LONG_CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import (
    Qwen38LaneHostPool,
    Qwen38LaneLayout,
    Qwen38TTNNLanePager,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import PLE_CHECKPOINT_LAYER

# The residue variants an import compiles: the GDN ring phase and the QSA raw-history slot rows are ``position % 4``.
IMPORT_RESIDUES = gdn_module.CONV_KERNEL_SIZE
# The eager passes before the captures: the bootstrap (host-written rows) and one split step (commit, PLE rows,
# verify, draft), as the stage-3 tool ran them; every program of the three bodies compiles here.
EAGER_PASSES = 2
WARM_TOKEN_ID = 1  # any exact vocabulary id for the warm passes' rows
# The admission prefill's event cadence in prompt rows: one event (and one ``between`` call, the scheduler's yield
# point) per 128-row chunk, or per four 32-row chunks, so the decoding lanes' passes can run between chunk groups.
ADMISSION_EVENT_ROWS = LONG_CHUNK_ROWS
# The admission's segments as ``between`` names them: an image prompt's images through the tower, the chunk groups
# of the prefill, the forced last step and the row read, the eviction into the host slot; the import into the lane
# ends the admission.
ADMISSION_SEGMENTS = ("tower", "chunks", "prefilled", "evicted")


# The narrow widths in the order the free-DRAM fallback drops them, and the floor below which it does.
DROP_ORDER = (4, 2)
FREE_DRAM_FLOOR_BYTES_PER_BANK = mtp_lanes.FREE_DRAM_FLOOR_BYTES_PER_BANK
mtp_lane_buckets_from_env = mtp_lanes.mtp_lane_buckets_from_env


class Qwen38LanesSessionError(RuntimeError):
    pass


def _dram_view(mesh) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        "num_banks": int(view.num_banks),
        "free_bytes_per_bank": int(view.total_bytes_free_per_bank),
        "largest_contiguous_bytes_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
        "allocated_bytes_per_bank": int(view.total_bytes_allocated_per_bank),
    }


@dataclass
class Qwen38LanesSession:
    """The lane chain beside the traced chain; see the module docstring."""

    lanes: int
    drafts: int
    marker: Callable[[str], Any]
    admission: dict[str, Any] = field(default_factory=dict)  # lanes_capacity_admission's record (the server fills it)
    clock_ns: Callable[[], int] = time.perf_counter_ns
    # The resident tower's request path (the server's ``vision_prompt_for``: prompt ids, rotary positions, decoded
    # images -> the splice's prompt object and the request's record), set once the tower is resident; None on a
    # text-only process, where an image ticket never reaches the driver (the handler refuses it).
    tower: Callable[..., tuple[Any, dict[str, Any]]] | None = None
    rows: int = field(init=False)
    chain: Any = None
    session: Any = None
    model: Any = None
    mesh: Any = None
    backbone_pager: Any = None
    alignment_pager: Any = None
    backbone_pool: Any = None
    alignment_pool: Any = None
    verify: Any = None
    draft: Any = None
    traces: Any = None
    output: Any = None
    lane_chain: Any = None
    eager_next_tokens: Any = None
    # The lane widths (``QWEN38_MTP_LANE_BUCKETS``; ``(lanes,)`` = today's single set, no views) and the narrow widths'
    # trace sets / views by width; ``verify`` / ``draft`` / ``traces`` / ``output`` above stay the full width's.
    environ: Mapping[str, str] | None = None
    buckets: tuple[int, ...] = ()
    width_sets: dict[int, Any] = field(default_factory=dict)
    dropped_widths: list[int] = field(default_factory=list)
    dram: dict[str, dict[str, int]] = field(default_factory=dict)
    capture_ms: dict[str, float] = field(default_factory=dict)
    warmup_ms: dict[str, float] = field(default_factory=dict)
    program_cache_entries: dict[str, int] = field(default_factory=dict)
    closed: bool = False

    def __post_init__(self) -> None:
        self.lanes, self.rows = lane_geometry(self.lanes, self.drafts)

    # -- the warm hook: allocation and compile before any capture ------------------------------

    def prepare(self, chain) -> None:
        """The traced chain's warm hook (after its warm pass, before its captures, misses allowed): the pagers and
        host slots, the lane verify and draft states, the warm imports at every residue into every lane, the eager
        lane passes.  Nothing of the chain's own state is changed."""

        if chain.mtp is None:
            raise Qwen38LanesSessionError("the lanes need an MTP chain (--mtp)")
        if chain.mtp.drafts != self.drafts:
            raise Qwen38LanesSessionError(f"the lanes draft {self.drafts}, the chain {chain.mtp.drafts}")
        if chain.misses_forbidden:
            raise Qwen38LanesSessionError("the lanes prepare before the chain's captures (misses allowed)")
        self.chain = chain
        self.mesh = mesh = chain.mesh
        self.model = model = chain.built_target.model
        contract = model.mesh_contract
        context = model.allocated_context
        self.marker("before-lanes-prepare")
        ttnn.synchronize_device(mesh)
        self.dram["before_prepare"] = _dram_view(mesh)
        started = self.clock_ns()
        backbone_layout = model.generic_lane_layout(chain.state)
        alignment_generic = chain.mtp.alignment.generic_state.attention
        alignment_layout = Qwen38LaneLayout(
            lanes=1,
            allocated_context=context,
            kv=(alignment_generic.packed_kv_cache,),
            compressed=(alignment_generic.compressed_index_cache,),
            staging=(alignment_generic.kv_staging,),
            ring=(alignment_generic.raw_key_ring,),
            recurrent=(),
            conv=(),
            ple=(),
        )
        for name, layout in (("backbone", backbone_layout), ("alignment", alignment_layout)):
            unallocated = [i for i, t in enumerate(layout.tensors()) if not t.is_allocated()]
            if unallocated:
                raise Qwen38LanesSessionError(f"{name} layout tensors {unallocated} are not allocated")
        self.backbone_pager = Qwen38TTNNLanePager(mesh, contract, backbone_layout)
        self.alignment_pager = Qwen38TTNNLanePager(mesh, contract, alignment_layout)
        self.backbone_pool = Qwen38LaneHostPool(mesh, backbone_layout, 1)
        self.alignment_pool = Qwen38LaneHostPool(mesh, alignment_layout, 1)
        ttnn.synchronize_device(mesh)
        self.dram["after_pagers"] = _dram_view(mesh)
        self.verify = mtp_lanes.allocate_lane_verify_state(
            model,
            lanes=self.lanes,
            drafts=self.drafts,
            mtp_components=chain.mtp.components,
            positions=[0] * self.lanes,
        )
        self.draft = mtp_lanes.allocate_lane_draft_state(model, self.verify)
        self.buckets = mtp_lane_buckets_from_env(self.lanes, self.drafts, self.environ)
        self.width_sets = {}
        for width in self.buckets[:-1]:  # the narrow widths: views of the full state, per-width constants and MoE
            view = mtp_lanes.allocate_lane_width_view(model, self.verify, width, parent_draft=self.draft)
            self.width_sets[width] = mtp_lanes.Qwen38TTNNMTPLaneWidthSet(
                width, view.verify, view.draft, None, view=view  # traces after the captures
            )
            ttnn.synchronize_device(mesh)
            self.dram[f"after_view_w{width}"] = _dram_view(mesh)
        ttnn.synchronize_device(mesh)
        self.dram["after_states"] = _dram_view(mesh)
        self.warmup_ms["allocate"] = (self.clock_ns() - started) / 1e6
        # -- the warm imports: the generic state as the warm pass left it, at four consecutive positions into every lane
        started = self.clock_ns()
        self.backbone_pager.evict(0, self.backbone_pool.slots[0])
        self.alignment_pager.evict(0, self.alignment_pool.slots[0])
        base = int(chain.state.position.read())
        gdn_layers = sum(1 for layer in model.layers if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN))
        for shift in range(IMPORT_RESIDUES):
            position = base + shift
            for lane in range(self.lanes):
                mtp_lanes.import_lane_state(
                    model,
                    self.verify,
                    lane,
                    position=position,
                    ple_context=None,
                    gdn_phases=[position % gdn_module.CONV_KERNEL_SIZE] * gdn_layers,
                    backbone_slot=self.backbone_pool.slots[0],
                    backbone_pager=self.backbone_pager,
                    alignment_slot=self.alignment_pool.slots[0],
                    alignment_pager=self.alignment_pager,
                )
        ttnn.synchronize_device(mesh)
        self.warmup_ms["imports"] = (self.clock_ns() - started) / 1e6
        # -- the eager passes: the bootstrap on host-written rows, then one split step.  Narrow widths first (their device
        # bodies on the views, every host write and mirror on the parent), the full width LAST: the chain's next tokens
        # and pending commit belong to the full set when the captures end.
        started = self.clock_ns()
        for width in self.buckets[:-1]:
            self._eager_passes_at(self.width_sets[width])
        mtp_lanes.write_lane_accepted(model, self.verify, [-1] * self.lanes)
        mtp_lanes.write_lane_active(model, self.verify, [1] * self.lanes, commit=[1] * self.lanes)
        tokens = [[WARM_TOKEN_ID] + [mtp_lanes.BOOTSTRAP_DRAFT_TOKEN] * self.drafts for _ in range(self.lanes)]
        next_tokens = None
        for index in range(EAGER_PASSES):
            if index == 0:
                mtp_lanes.write_lane_verify_inputs(model, self.verify, tokens)
            else:
                mtp_lanes.forward_commit_lanes(model, self.verify)
                mtp_lanes.write_lane_ple_rows(model, self.verify, tokens)
            eager_output = mtp_lanes.forward_verify_lanes(model, self.verify, catch_up=False, land_accept=True)
            mtp_lanes.forward_draft_lanes(model, self.verify, self.draft, eager_output)
            ttnn.synchronize_device(mesh)
            readback, next_tokens = mtp_lanes.read_lane_pass_row(self.verify, self.draft)
            eager_output.release_tensors()
            mtp_lanes.commit_lane_verify_host(model, self.verify, [a + 1 for a in readback.accepted])
            tokens = [list(block) for block in next_tokens]
        device_positions = self.verify.positions.read()
        if device_positions != self.verify.positions.positions:
            raise Qwen38LanesSessionError(
                f"device lane positions {device_positions} vs the host mirror {self.verify.positions.positions} after the eager passes"
            )
        self.eager_next_tokens = next_tokens
        ttnn.synchronize_device(mesh)
        self.warmup_ms["eager_passes"] = (self.clock_ns() - started) / 1e6
        # The host-written and device-read buffers of the lane pass loop, acknowledged to the allocation tracker as the
        # chain acknowledges its own before the miss guard closes.
        verify = self.verify
        extra = []
        for width_set in self.width_sets.values():  # the per-width aliases and own tensors the host writes or reads
            extra += [
                width_set.verify.ple_rows.embedding_rows,
                width_set.verify.accepted_lanes,
                width_set.verify.active_lanes,
                width_set.verify.commit_lanes,
                width_set.verify.alignment.residual,
                width_set.draft.pass_row,
            ]
        for tensor in (
            verify.token_row,
            verify.draft_lanes,
            verify.ple_rows.embedding_rows,
            verify.accepted_lanes,
            verify.accepted_row,
            verify.active_lanes,
            verify.active_row,
            verify.inactive_row,
            verify.commit_lanes,
            verify.positions.row,
            self.draft.pass_row,
            *extra,
        ):
            acknowledge_corruptible(tensor)
        for pager in (self.backbone_pager, self.alignment_pager):
            for tensor in (*pager.packs.values(), *pager.kv_stagings):
                acknowledge_corruptible(tensor)
        self.program_cache_entries["after_prepare"] = resident_decode.program_cache_count(mesh)
        self.dram["after_prepare"] = _dram_view(mesh)
        self.marker("after-lanes-prepare")

    def _eager_passes_at(self, width_set) -> None:
        """The two eager passes (bootstrap, one split step) of a narrow width: the device bodies on its view and draft view,
        the host writes and the mirror on the parent with the lanes past the width inactive (their counts and commits 0).
        """

        model, parent, width = self.model, self.verify, width_set.width
        view, draft = width_set.verify, width_set.draft
        pad = [0] * (self.lanes - width)
        mtp_lanes.write_lane_accepted(model, parent, [-1] * self.lanes)
        mtp_lanes.write_lane_active(model, parent, [1] * width + pad, commit=[1] * width + pad)
        tokens = [[WARM_TOKEN_ID] + [mtp_lanes.BOOTSTRAP_DRAFT_TOKEN] * self.drafts for _ in range(self.lanes)]
        for index in range(EAGER_PASSES):
            if index == 0:
                mtp_lanes.write_lane_verify_inputs(model, parent, tokens)
            else:
                mtp_lanes.forward_commit_lanes(model, view)
                mtp_lanes.write_lane_ple_rows(model, parent, tokens)
            output = mtp_lanes.forward_verify_lanes(model, view, catch_up=False, land_accept=True)
            mtp_lanes.forward_draft_lanes(model, view, draft, output)
            ttnn.synchronize_device(self.mesh)
            readback, next_tokens = mtp_lanes.read_lane_pass_row(view, draft)
            output.release_tensors()
            mtp_lanes.commit_lane_verify_host(model, parent, [a + 1 for a in readback.accepted] + pad)
            tokens = [list(block) for block in next_tokens] + [list(block) for block in tokens[width:]]

    # -- the captures after the chain's --------------------------------------------------------

    def capture(self, chain, session) -> None:
        """After the traced chain opened (its traces captured, the miss guard closed): the three lane traces, the
        tracker over every trace, the lane chain with every lane parked."""

        if chain is not self.chain or self.verify is None:
            raise Qwen38LanesSessionError("capture follows prepare on the same chain")
        if not chain.misses_forbidden:
            raise Qwen38LanesSessionError("the lane captures run under the chain's closed miss guard")
        self.session = session
        mesh, model = self.mesh, self.model
        self.marker("before-lanes-captures")
        ttnn.synchronize_device(mesh)
        before = _dram_view(mesh)
        self.dram["before_captures"] = before
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"lanes {label}")  # noqa: E731
        started = self.clock_ns()
        verify_id, output = mtp_lanes.capture_verify_lanes(
            model, self.verify, catch_up=False, land_accept=True, guard=guard
        )
        self.capture_ms["verify_first"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        commit_id = mtp_lanes.capture_commit_lanes(model, self.verify, guard=guard)
        self.capture_ms["commit"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        draft_id = mtp_lanes.capture_draft_lanes(model, self.verify, self.draft, output, guard=guard)
        self.capture_ms["draft"] = (self.clock_ns() - started) / 1e6
        self.traces = mtp_lanes.Qwen38TTNNMTPLaneTraces(verify_first=verify_id, commit=commit_id, draft=draft_id)
        self.output = output
        acknowledge_corruptible(output.readback)
        ttnn.synchronize_device(mesh)
        self.dram["after_captures_w%d" % self.lanes] = _dram_view(mesh)
        # the narrow widths, descending, each its own verify_first / commit / draft
        for width in reversed(self.buckets[:-1]):
            self._capture_width(self.width_sets[width], guard)
        trace_ids = [*chain.trace_ids(), verify_id, commit_id, draft_id]
        for width_set in self.width_sets.values():
            trace_ids += [width_set.traces.verify_first, width_set.traces.commit, width_set.traces.draft]
        for trace_id in trace_ids:
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
        after = _dram_view(mesh)
        self.dram["after_captures"] = after
        entries = resident_decode.program_cache_count(mesh)
        self.program_cache_entries["after_captures"] = entries
        if entries != chain.program_cache_entries:
            raise Qwen38LanesSessionError(
                f"program cache {entries} entries after the lane captures vs {chain.program_cache_entries} at the chain's"
            )
        self.lane_chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
            model,
            self.verify,
            self.draft,
            self.traces,
            replay=chain._replay,
            enqueue=chain._enqueue,
            next_tokens=self.eager_next_tokens,
            width_sets=self.width_sets,
        )
        self.lane_chain.set_active([0] * self.lanes)  # every lane parked until an admission
        ttnn.synchronize_device(mesh)
        self.enforce_free_dram()
        self.marker("after-lanes-captures")

    def _capture_width(self, width_set, guard) -> None:
        mesh, model, width = self.mesh, self.model, width_set.width
        view, draft = width_set.verify, width_set.draft
        started = self.clock_ns()
        verify_id, output = mtp_lanes.capture_verify_lanes(model, view, catch_up=False, land_accept=True, guard=guard)
        self.capture_ms[f"w{width}_verify_first"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        commit_id = mtp_lanes.capture_commit_lanes(model, view, guard=guard)
        self.capture_ms[f"w{width}_commit"] = (self.clock_ns() - started) / 1e6
        started = self.clock_ns()
        draft_id = mtp_lanes.capture_draft_lanes(model, view, draft, output, guard=guard)
        self.capture_ms[f"w{width}_draft"] = (self.clock_ns() - started) / 1e6
        width_set.traces = mtp_lanes.Qwen38TTNNMTPLaneTraces(verify_first=verify_id, commit=commit_id, draft=draft_id)
        width_set.output = output
        acknowledge_corruptible(output.readback)
        ttnn.synchronize_device(mesh)
        self.dram[f"after_captures_w{width}"] = _dram_view(mesh)

    def free_dram_bytes_per_bank(self) -> int:
        return int(_dram_view(self.mesh)["free_bytes_per_bank"])

    def enforce_free_dram(self, floor: int = FREE_DRAM_FLOOR_BYTES_PER_BANK) -> list[int]:
        """After the captures: log the free DRAM per bank; below ``floor`` drop the 4-lane width, then the 2-lane width
        (their traces and views released, a warning logged), re-measuring after each.  The full width always stays.
        Returns the widths dropped."""

        free = self.free_dram_bytes_per_bank()
        logger.info(f"mtp lanes: free DRAM after the captures {free / 2**20:.1f} MiB/bank, widths {self.buckets}")
        dropped: list[int] = []
        for width in DROP_ORDER:
            if free >= floor:
                break
            if width not in self.width_sets:
                continue
            logger.warning(
                f"mtp lanes: {free / 2**20:.1f} MiB/bank free after the captures is below {floor / 2**20:.0f}: "
                f"dropping the {width}-lane width"
            )
            self.drop_width(width)
            dropped.append(width)
            free = self.free_dram_bytes_per_bank()
        if dropped:
            logger.warning(
                f"mtp lanes: widths {dropped} dropped, serving widths {self.buckets}, {free / 2**20:.1f} MiB/bank free"
            )
        return dropped

    def drop_width(self, width: int) -> None:
        """Release a narrow width's traces, output and views (the chain forgets it first)."""

        if self.lane_chain is not None:
            width_set = self.lane_chain.drop_width(width)
        else:
            width_set = self.width_sets[width]
        self.width_sets.pop(width, None)
        self._release_width_set(width_set)
        self.buckets = tuple(w for w in self.buckets if w != width)
        self.dropped_widths.append(width)

    def _release_width_set(self, width_set) -> None:
        ttnn.synchronize_device(self.mesh)
        if width_set.traces is not None:
            for trace_id in (width_set.traces.verify_first, width_set.traces.commit, width_set.traces.draft):
                ttnn.release_trace(self.mesh, trace_id)
            width_set.traces = None
        if width_set.output is not None and width_set.output.active:
            width_set.output.release_tensors()
        width_set.output = None
        if width_set.view is not None:
            mtp_lanes.release_lane_width_view(self.model, width_set.view)
            width_set.view = None

    def warm_widths(self) -> list[int]:
        """Replay every width's verify_first / commit / draft once, so no first replay happens while serving: the passes
        run on the lanes that are active now (the warm requests) with the width forced, full width first then down to the
        smallest the active lanes fit, and once more at that width (each width's commit follows the pass at it).  The
        caller releases the warm requests afterwards.  Returns the widths run."""

        chain = self.lane_chain
        if chain is None:
            raise Qwen38LanesSessionError("the lanes are not captured")
        floor = chain.width_for()
        widths = [width for width in reversed(chain.buckets) if width >= floor]
        for width in [*widths, widths[-1]]:
            chain.step(force_width=width)
        return widths

    def growth_bytes_per_bank(self) -> dict[str, int]:
        """The MEASURED DRAM growth of the lanes per bank: the states (pagers + lane states, the hook's allocations)
        and the three traces."""

        states = (
            self.dram["after_prepare"]["allocated_bytes_per_bank"]
            - self.dram["before_prepare"]["allocated_bytes_per_bank"]
        )
        traces = (
            self.dram["after_captures"]["allocated_bytes_per_bank"]
            - self.dram["before_captures"]["allocated_bytes_per_bank"]
        )
        return {"states": states, "traces": traces}

    def growth_by_width(self) -> dict[str, int]:
        """The MEASURED DRAM growth per bank of each width's three traces (``traces_w<W>``), from the allocation views
        taken after each width's captures (the full width first, then the narrow ones descending)."""

        if "before_captures" not in self.dram:
            return {}
        out: dict[str, int] = {}
        last = self.dram["before_captures"]["allocated_bytes_per_bank"]
        for width in sorted(self.width_sets_captured(), reverse=True):
            now = self.dram[f"after_captures_w{width}"]["allocated_bytes_per_bank"]
            out[f"traces_w{width}"] = now - last
            last = now
        return out

    def width_sets_captured(self) -> list[int]:
        return [int(key.rsplit("w", 1)[1]) for key in self.dram if key.startswith("after_captures_w")]

    def summary(self) -> dict[str, Any]:
        return {
            "lanes": self.lanes,
            "drafts": self.drafts,
            "rows": self.rows,
            "moe_rows": None if self.verify is None else self.verify.moe_rows,
            "traces": 0 if self.traces is None else 3,
            "buckets": list(self.buckets),
            "dropped_widths": list(self.dropped_widths),
            "dram_growth_bytes_per_bank_by_width": self.growth_by_width(),
            "capture_ms": dict(self.capture_ms),
            "warmup_ms": dict(self.warmup_ms),
            "program_cache_entries": dict(self.program_cache_entries),
            "dram": {name: dict(view) for name, view in self.dram.items()},
            "dram_growth_bytes_per_bank": None if "after_captures" not in self.dram else self.growth_bytes_per_bank(),
            "admission": dict(self.admission),
        }

    # -- the device protocol (the driver thread) ------------------------------------------------

    def admit(
        self, lane: int, ticket: Qwen38LaneTicket, between: Callable[[str, int, int], float] | None = None
    ) -> Qwen38LaneAdmitted:
        """The request's prompt prefilled on the traced chain (the session's own prefill section: the reset, the
        chunked prefill of all but the last token when the prompt admits it, the last token forced, the model's next
        token read from the row), the generic and alignment states evicted into the host slots, the image imported
        into ``lane``.  An image prompt (``ticket.images``) runs its images through the resident tower first (the
        ``"tower"`` segment) and prefills through the chunk driver with its rotary positions and feature rows, as
        the single stream does; the lane takes the prompt's rotary shift at the import (the decode shift the
        single stream sets at its hand-off), 0 for text.

        ``between`` is the scheduler's yield point between the admission's device segments: called after the
        tower (``"tower"``), after every 128 prompt rows of the chunked prefill (``"chunks"``, with the rows consumed
        and the rows of the prefill), after the forced last step and its row read (``"prefilled"``) and after the
        eviction (``"evicted"``), each time with the device idle and every chunk input for the next segment prepared
        but not uploaded; it returns the seconds it spent (the decoding lanes' passes), which the segment records
        below exclude.  None: the admission runs whole (the single-stream event cadence)."""

        session, chain, model = self.session, self.chain, self.model
        if session is None or self.lane_chain is None:
            raise Qwen38LanesSessionError("the lanes are not captured")
        ids = [int(token) for token in ticket.prompt_ids]
        if not ids:
            raise Qwen38LanesSessionError("an admission needs a prompt")
        spent = 0.0  # seconds ``between`` took inside the prefill segment

        def between_chunks(done: int, total: int) -> None:
            nonlocal spent
            spent += float(between("chunks", done, total))

        vision = vision_record = None
        tower_seconds = 0.0
        if ticket.images:
            # The tower on this (device) thread, before the prefill: the feature rows and the splice's prompt object
            # (validated against the prompt); a text-only process never sees an image ticket.
            if self.tower is None:
                raise Qwen38LanesSessionError("an image prompt reached the lanes without a resident tower")
            tower_started = self.clock_ns()
            vision, vision_record = self.tower(ids, ticket.vision_positions, ticket.images)
            tower_seconds = (self.clock_ns() - tower_started) / 1e9
            if between is not None:
                between("tower", len(ids), len(ids))
        started = self.clock_ns()
        session.reset()
        suffix = list(ids)
        chunked = None
        if vision is not None and (
            session.prefill_mode != "chunked" or session.chunk_prefill_rows(len(suffix) - 1) < CHUNK_PREFILL_MIN_ROWS
        ):
            raise Qwen38LanesSessionError("an image prompt needs the chunked prefill (its pads never take 1-row steps)")
        if session.prefill_mode == "chunked" and session.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
            chunked = session._prefill_chunked(
                suffix[:-1],
                suffix[-1],
                None,
                vision,
                between_chunks=None if between is None else between_chunks,
                event_rows=None if between is None else ADMISSION_EVENT_ROWS,
            )
            if chunked.stopped is not None:
                raise Qwen38LanesSessionError(f"the chunked prefill stopped: {chunked.stopped!r}")
            suffix = suffix[-1:]
        with chain.loop_guard():
            finish = session._prefill(suffix, None, None)
        if finish is not None:
            raise Qwen38LanesSessionError(f"the prefill ended early: {finish!r}")
        pending = int(chain.read_token_row())
        position = int(chain.position())
        if position != len(ids):
            raise Qwen38LanesSessionError(f"device position {position} after prefilling {len(ids)} tokens")
        ple_context = session.ple_context
        prefilled = self.clock_ns()
        if between is not None:
            between("prefilled", len(ids), len(ids))
        evict_started = self.clock_ns()
        self.backbone_pager.evict(0, self.backbone_pool.slots[0])
        self.alignment_pager.evict(0, self.alignment_pool.slots[0])
        evicted = self.clock_ns()
        if between is not None:
            between("evicted", len(ids), len(ids))
        import_started = self.clock_ns()
        gdn_layers = sum(1 for layer in model.layers if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN))
        # The lane's rotary shift: the prompt's (the chunk driver's hand-off set the same on the traced chain; the
        # prompt ends in text after its last image, so the shift is within the position's block start), 0 for text.
        rope_shift = 0 if vision is None else min(vision.positions.shift_at(position), position & ~3)
        mtp_lanes.import_lane_state(
            model,
            self.verify,
            lane,
            position=position,
            ple_context=ple_context,
            gdn_phases=[position % gdn_module.CONV_KERNEL_SIZE] * gdn_layers,
            backbone_slot=self.backbone_pool.slots[0],
            backbone_pager=self.backbone_pager,
            alignment_slot=self.alignment_pool.slots[0],
            alignment_pager=self.alignment_pager,
            rope_shift=rope_shift,
        )
        ttnn.synchronize_device(self.mesh)
        imported = self.clock_ns()
        timing = None if chunked is None else chunked.timing
        seconds = {
            "prefill": (prefilled - started) / 1e9 - spent,
            "evict": (evicted - evict_started) / 1e9,
            "import": (imported - import_started) / 1e9,
        }
        if vision is not None:
            seconds = {"tower": tower_seconds, **seconds}
        return Qwen38LaneAdmitted(
            pending=pending,
            position=position,
            ple_context=ple_context,
            seconds=seconds,
            vision=vision_record,
            prefill={
                "mode": "teacher_forced" if chunked is None else "chunked",
                "tokens": len(ids),
                "forced_tokens": len(ids) if chunked is None else timing.alignment_steps + 1,
                "chunks": 0 if timing is None else timing.chunks,
                "long_chunks": 0 if timing is None else timing.long_chunks,
                "slabs": 0 if timing is None else timing.slabs,
            },
        )

    def write_counts(self, counts: Sequence[int]) -> None:
        mtp_lanes.write_lane_accepted(self.model, self.verify, list(counts))

    def override_commit(self, lane: int, record, committed_rows: int) -> None:
        """Lane ``lane`` commits ``committed_rows`` of the pass ``record`` instead of the device's count (the forced
        ``</think>``): the position mirror and the device row rewound to the pass's start plus the rows kept, the
        checkpoint layer's n-gram context that of the rows kept.  The count itself goes with the boundary's
        ``write_counts``."""

        verify = self.verify
        if not 1 <= committed_rows <= verify.rows:
            raise ValueError(f"committed rows must be in [1, {verify.rows}], got {committed_rows!r}")
        mirror = list(verify.positions.positions)
        mirror[lane] = int(record.positions[lane]) + committed_rows
        verify.positions.write(mirror)
        ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
        contexts = list(ple_state.token_contexts)
        contexts[lane] = verify.pass_contexts[lane][committed_rows]
        ple_state.token_contexts = tuple(contexts)
        ple_state.validate()

    def set_blocks(self, blocks: Sequence[Sequence[int] | None]) -> None:
        current = [list(block) for block in self.lane_chain.next_tokens]
        for lane, block in enumerate(blocks):
            if block is not None:
                current[lane] = [int(token) for token in block]
        self.lane_chain.set_next_tokens(current)

    def set_held(self, held: Sequence[int] | None) -> None:
        """The lanes that hold a request (the lane chain's width pick covers them even when they are parked)."""

        self.lane_chain.set_held(held)

    def set_active(self, mask: Sequence[int]) -> None:
        flags = [int(flag) for flag in mask]
        if flags != list(self.lane_chain.active):
            self.lane_chain.set_active(flags)

    def step(self):
        return self.lane_chain.step()

    def room(self, position: int) -> bool:
        return mtp_v2.verify_pass_fits(int(position), self.model.allocated_context)

    # -- release ---------------------------------------------------------------------------------

    def close(self) -> None:
        """The lane traces, the output, the draft and verify states, the pagers; before the chain's own close."""

        if self.closed or self.mesh is None:
            return
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        for width_set in list(self.width_sets.values()):  # the narrow widths first, before the parent they alias
            self._release_width_set(width_set)
        self.width_sets = {}
        if self.traces is not None:
            for trace_id in (self.traces.verify_first, self.traces.commit, self.traces.draft):
                ttnn.release_trace(mesh, trace_id)
            self.traces = None
        if self.output is not None and self.output.active:
            self.output.release_tensors()
        self.output = None
        if self.draft is not None:
            mtp_lanes.release_lane_draft_state(self.model, self.verify, self.draft)
            self.draft = None
        if self.verify is not None:
            mtp_lanes.release_lane_verify_state(self.model, self.verify)
            self.verify = None
        for pager in (self.backbone_pager, self.alignment_pager):
            if pager is not None:
                pager.release()
        self.backbone_pager = self.alignment_pager = None
        ttnn.synchronize_device(mesh)
        self.closed = True
